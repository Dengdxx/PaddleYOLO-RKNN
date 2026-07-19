#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO26 知识蒸馏训练脚本

策略：特征蒸馏 + 响应蒸馏（Feature-based + Response-based KD）
  - teacher 置信度加权的 FPN 空间注意力蒸馏（支持不同 scale 通道数）
  - 分类 logits 的温度缩放 KL 散度（Response KD）
  - 分割任务额外蒸馏 mask coefficient、Proto 和语义辅助输出
  - 遵循官方 E2ELoss 的 one2many/one2one 衰减加权机制
  - 不对框回归做 KD（DFL 分布不适合直接蒸馏，feature KD 已隐式覆盖）

实现原理：
  在 DetectionTrainer._do_train() 中，loss 计算流程为：
    self.model(batch) → model.loss(batch) → model.criterion(preds, batch)
  本脚本通过 on_pretrain_routine_end 回调替换 model.criterion，
  将 KD loss 注入训练循环，无需改动 trainer 及 loss.py 其他逻辑。

  官方参照：
    - E2ELoss: one2many(0.8→0.1) + one2one(0.2→0.9) 衰减加权
    - v8DetectionLoss: TAL 分配后的 box/cls/dfl 三项 loss
    - Detect.forward_head: preds["feats"] 即 FPN 三尺度特征图

用法示例：
    python distill.py \\
        --teacher weights/yolo26/yolo26s.pdparams \\
        --student weights/yolo26/yolo26n.pdparams \\
        --data data/your.yaml \\
        --epochs 50 \\
        --kd-weight 1.0 --feat-weight 0.5 \\
        --temperature 4.0 \\
        --imgsz 640 --batch 8 \\
        --output runs/detect/distill_n_from_s

注意：
  - teacher 和 student 必须具有相同 nc（类别数）
  - teacher 通常比 student 大（如 s→n 或 m→n）
  - worker 数建议 4，显存不足时降低 batch
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import paddle

# ── 在 import paddle 之前抑制已知的无害警告 ──────────────────────────
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", message="No ccache found")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export.input_shape import normalize_static_imgsz


# ──────────────────────────────────────────────────────────────────────
# KD 损失函数
# ──────────────────────────────────────────────────────────────────────


def _prediction_branches(preds: dict) -> list[dict]:
    """返回模型中可用于蒸馏的预测分支，兼容 E2E 与普通检测头。"""
    branches = [preds[key] for key in ("one2one", "one2many") if key in preds]
    return branches or [preds]


def _logit_kd_loss(s_preds: dict, t_preds: dict, temperature: float = 4.0) -> paddle.Tensor:
    """
    分类 logits 二元响应蒸馏。

    对 one2one 和 one2many 头的 scores 做温度缩放 Bernoulli KL。
    one2one 直接对齐最终推理头，one2many 则继续约束训练辅助头。

    参照官方 E2ELoss 设计：one2one 是训练后期的主导头。

    参数:
        s_preds: student 训练模式预测 dict
        t_preds: teacher eval 模式预测 dict
        temperature: KD 温度

    返回:
        归一化的标量 KD loss
    """
    import paddle
    import paddle.nn.functional as F

    T = temperature
    kd = paddle.zeros([1])

    n_matched = 0
    for s_branch, t_branch in zip(_prediction_branches(s_preds), _prediction_branches(t_preds)):
        if "scores" not in s_branch or "scores" not in t_branch:
            continue
        s_scores = s_branch["scores"]  # [B, NC, NA]
        t_scores = t_branch["scores"]  # [B, NC, NA]

        # YOLO 各类别是独立 sigmoid，不能按互斥类别做 softmax KL。
        # mean 同时对 batch、anchor 和 class 归一化，避免 KD 随输入尺寸放大。
        t_prob = F.sigmoid(t_scores.detach() / T)
        cross_entropy = F.binary_cross_entropy_with_logits(s_scores / T, t_prob, reduction="mean")
        teacher_entropy = F.binary_cross_entropy_with_logits(t_scores.detach() / T, t_prob, reduction="mean")
        cls_kd = (cross_entropy - teacher_entropy) * (T**2)
        kd = kd + cls_kd
        n_matched += 1

    return kd / max(n_matched, 1)


def _teacher_score_weights(t_preds: dict, t_feats: list) -> list[paddle.Tensor]:
    """按特征层拆分 teacher 置信度，生成 Ultralytics 风格的空间权重。"""
    import paddle

    branch_scores = [branch["scores"] for branch in _prediction_branches(t_preds) if "scores" in branch]
    if not branch_scores:
        return [paddle.ones([feat.shape[0], 1, feat.shape[2], feat.shape[3]]) for feat in t_feats]
    scores = paddle.stack(branch_scores, axis=0).mean(axis=0).detach().sigmoid().max(axis=1, keepdim=True)
    sizes = [int(feat.shape[2] * feat.shape[3]) for feat in t_feats]
    parts = paddle.split(scores, sizes, axis=-1)
    return [part.reshape([part.shape[0], 1, feat.shape[2], feat.shape[3]]) for part, feat in zip(parts, t_feats)]


def _feature_kd_loss(s_feats: list, t_feats: list, score_weights: list[paddle.Tensor]) -> paddle.Tensor:
    """
    FPN 特征图 Channel-Wise 归一化 L2 蒸馏（CWD 简化版）。

    对每个 FPN 层级沿通道聚合平方响应，形成与通道数无关的空间注意力图。
    因此 n/s/m 等不同 scale 可直接蒸馏，不再因通道数不同而整层跳过。
    空间误差由 teacher 的 O2O/O2M 分类置信度加权，与最新版 Ultralytics
    score-weighted feature distillation 的目标一致。

    参数:
        s_feats: student FPN 特征列表 [P3, P4, P5]
        t_feats: teacher FPN 特征列表 [P3, P4, P5]

    返回:
        scalar feature KD loss
    """
    import paddle
    import paddle.nn.functional as F

    feat_loss = paddle.zeros([1])
    n_matched = 0

    for sf, tf, weight in zip(s_feats, t_feats, score_weights):
        # Spatial 尺寸可能不同（不同 batch rect padding），对齐
        if sf.shape[2:] != tf.shape[2:]:
            tf = F.interpolate(tf, size=sf.shape[2:], mode="bilinear", align_corners=False)

        sf_attention = sf.square().mean(axis=1, keepdim=True)
        tf_attention = tf.detach().square().mean(axis=1, keepdim=True)
        sf_attention = F.normalize(sf_attention.flatten(1), p=2, axis=1).reshape(sf_attention.shape)
        tf_attention = F.normalize(tf_attention.flatten(1), p=2, axis=1).reshape(tf_attention.shape)
        weight = weight.astype(sf_attention.dtype)
        feat_loss = feat_loss + ((sf_attention - tf_attention).square() * weight).sum() / (weight.sum() + 1e-9)
        n_matched += 1

    if n_matched > 0:
        feat_loss = feat_loss / n_matched
    return feat_loss


def _segmentation_kd_loss(
    s_preds: dict,
    t_preds: dict,
    temperature: float = 4.0,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """计算 mask coefficient 与 Proto/语义辅助头蒸馏损失。"""
    import paddle
    import paddle.nn.functional as F

    coefficient_loss = paddle.zeros([1])
    n_coeff = 0
    for s_branch, t_branch in zip(_prediction_branches(s_preds), _prediction_branches(t_preds)):
        if "mask_coefficient" not in s_branch or "mask_coefficient" not in t_branch:
            continue
        s_coeff = s_branch["mask_coefficient"]
        t_coeff = t_branch["mask_coefficient"].detach()
        confidence = t_branch["scores"].detach().sigmoid().max(axis=1, keepdim=True)
        raw = F.smooth_l1_loss(s_coeff, t_coeff, reduction="none")
        coefficient_loss += (raw * confidence).sum() / (confidence.sum() * s_coeff.shape[1] + 1e-9)
        n_coeff += 1
    coefficient_loss /= max(n_coeff, 1)

    s_branch = _prediction_branches(s_preds)[-1]
    t_branch = _prediction_branches(t_preds)[-1]
    if "proto" not in s_branch or "proto" not in t_branch:
        return coefficient_loss, paddle.zeros([1])
    s_proto, t_proto = s_branch["proto"], t_branch["proto"]
    s_semseg = t_semseg = None
    if isinstance(s_proto, tuple):
        s_proto, s_semseg = s_proto
    if isinstance(t_proto, tuple):
        t_proto, t_semseg = t_proto
    if tuple(s_proto.shape[-2:]) != tuple(t_proto.shape[-2:]):
        t_proto = F.interpolate(t_proto, size=s_proto.shape[-2:], mode="bilinear", align_corners=False)
    proto_loss = F.mse_loss(
        F.normalize(s_proto.flatten(2), p=2, axis=2),
        F.normalize(t_proto.detach().flatten(2), p=2, axis=2),
        reduction="mean",
    )
    if s_semseg is not None and t_semseg is not None:
        if tuple(s_semseg.shape[-2:]) != tuple(t_semseg.shape[-2:]):
            t_semseg = F.interpolate(t_semseg, size=s_semseg.shape[-2:], mode="bilinear", align_corners=False)
        teacher_prob = F.sigmoid(t_semseg.detach() / temperature)
        cross_entropy = F.binary_cross_entropy_with_logits(s_semseg / temperature, teacher_prob)
        teacher_entropy = F.binary_cross_entropy_with_logits(t_semseg.detach() / temperature, teacher_prob)
        proto_loss += (cross_entropy - teacher_entropy) * (temperature**2)
    return coefficient_loss, proto_loss


# ──────────────────────────────────────────────────────────────────────
# 蒸馏 Trainer
# ──────────────────────────────────────────────────────────────────────


class DistillDetectionTrainer:
    """
    包装 DetectionTrainer，注入知识蒸馏 criterion。

    蒸馏策略对照官方 YOLO26 实现：
      - 使用 E2ELoss 的 one2many/one2one 衰减权重
      - KD loss 包括 logit KD (KL-div) + feature KD (CWD-L2)
      - 通过 callback 在 on_pretrain_routine_end 注入

    用法：
        trainer = DistillDetectionTrainer(
            teacher_weights="weights/yolo26/yolo26s.pdparams",
            kd_weight=1.0,
            feat_weight=0.5,
            temperature=4.0,
            overrides={...},
        )
        trainer.train()
    """

    def __init__(
        self,
        teacher_weights: str,
        kd_weight: float = 1.0,
        feat_weight: float = 0.5,
        mask_weight: float = 0.25,
        proto_weight: float = 0.25,
        temperature: float = 4.0,
        overrides: dict = None,
    ):
        """初始化蒸馏训练器。

        参数:
            teacher_weights: Teacher 模型权重文件路径
            kd_weight: Logit KD 损失权重（KL 散度）
            feat_weight: 特征图 KD 损失权重（CWD-L2）
            mask_weight: mask coefficient KD 损失权重
            proto_weight: Proto 与语义辅助头 KD 损失权重
            temperature: KD 温度参数，越大 soft label 越平滑
            overrides: 传递给 DetectionTrainer 的配置覆盖
        """
        self.teacher_weights = teacher_weights
        self.kd_weight = kd_weight
        self.feat_weight = feat_weight
        self.mask_weight = mask_weight
        self.proto_weight = proto_weight
        self.temperature = temperature
        self.overrides = overrides or {}
        self._teacher = None

        from ddyolo26.models.yolo.detect import DetectionTrainer
        from ddyolo26.models.yolo.segment import SegmentationTrainer
        from ddyolo26.nn.tasks import guess_model_task, paddle_safe_load

        student = self.overrides.get("model", "")
        task = self.overrides.get("task")
        if not task and str(student).endswith((".pdparams", "_paddle.pt")):
            checkpoint, _ = paddle_safe_load(student)
            task = guess_model_task(checkpoint.get("yaml", checkpoint.get("train_args", {})))
        task = task or guess_model_task(student)
        self.task = task
        trainer_cls = SegmentationTrainer if task == "segment" else DetectionTrainer
        self._trainer = trainer_cls(overrides=self.overrides)
        self._trainer.add_callback("on_pretrain_routine_end", self._inject_distill)

    def _inject_distill(self, trainer):
        """在 trainer.model 加载完成后注入 KD criterion。"""
        import paddle
        from ddyolo26.utils.runtime import unwrap_model
        from ddyolo26.utils import LOGGER

        # ── 加载 teacher 模型 ─────────────────────────────────────────
        LOGGER.info(f"[KD] 加载 teacher 模型: {self.teacher_weights}")
        ckpt = paddle.load(str(self.teacher_weights))

        from ddyolo26.nn.tasks import DetectionModel, SegmentationModel, yaml_model_load
        import re as _re

        yaml_cfg = ckpt.get("yaml") if isinstance(ckpt, dict) else None
        # 从权重文件名推断 model scale（n/s/m/l/x）
        teacher_stem = Path(self.teacher_weights).stem.lower()
        scale_match = _re.search(r"yolo26([nsmlx])", teacher_stem)
        teacher_scale = scale_match.group(1) if scale_match else None

        # 如果是 yaml 路径, 加载并注入 scale
        if isinstance(yaml_cfg, str):
            yaml_dict = yaml_model_load(yaml_cfg)
            if teacher_scale and "scales" in yaml_dict:
                yaml_dict["scale"] = teacher_scale
            yaml_cfg = yaml_dict
        elif isinstance(yaml_cfg, dict) and teacher_scale:
            yaml_cfg.setdefault("scale", teacher_scale)

        teacher_cls = SegmentationModel if self.task == "segment" else DetectionModel
        teacher = teacher_cls(
            cfg=yaml_cfg or "ddyolo26/cfg/models/26/yolo26.yaml",
            nc=trainer.data["nc"],
            ch=trainer.data.get("channels", 3),
            verbose=False,
        )
        model_sd = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        if isinstance(model_sd, dict):
            # 过滤掉 shape 不匹配的键（如 teacher nc 不同导致 cls head 维度不同）
            teacher_sd = teacher.state_dict()
            compatible_sd = {
                k: v.astype(teacher_sd[k].dtype) if v.dtype != teacher_sd[k].dtype else v
                for k, v in model_sd.items()
                if k in teacher_sd and v.shape == teacher_sd[k].shape
            }
            skipped = set(model_sd) - set(compatible_sd)
            # 抑制 Paddle set_state_dict 对缺失键的 UserWarning
            import warnings as _w

            with _w.catch_warnings():
                _w.filterwarnings("ignore", message="Skip loading for")
                teacher.set_state_dict(compatible_sd)
            if skipped:
                LOGGER.info(f"[KD] Teacher 跳过 {len(skipped)} 个 shape 不匹配的权重（nc 不同导致）")

        teacher.eval()
        for p in teacher.parameters():
            p.stop_gradient = True
        teacher = teacher.to(trainer.device)
        self._teacher = teacher
        LOGGER.info(
            f"[KD] Teacher 已冻结。logit_kd={self.kd_weight}, feat_kd={self.feat_weight}, "
            f"mask_kd={self.mask_weight}, proto_kd={self.proto_weight}, T={self.temperature}"
        )

        # ── 注入 KD criterion ─────────────────────────────────────────
        student_model = unwrap_model(trainer.model)

        if getattr(student_model, "criterion", None) is None:
            student_model.criterion = student_model.init_criterion()

        base_criterion = student_model.criterion
        kd_w = self.kd_weight
        feat_w = self.feat_weight
        mask_w = self.mask_weight
        proto_w = self.proto_weight
        temperature = self.temperature
        teacher_ref = teacher

        # 检查 base_criterion 是否有 update 方法（E2ELoss 有）
        has_update = hasattr(base_criterion, "update")

        def make_distill_criterion(owner):
            """为指定模型构建带蒸馏的损失函数。"""

            def distill_criterion(preds, batch):
                """合并任务损失与 KD 损失，验证时仅补零占位。"""
                task_loss, task_items = base_criterion(preds, batch)
                if not owner.training:
                    zero = paddle.zeros([1], dtype=task_loss.dtype)
                    return paddle.concat([task_loss, zero]), paddle.concat([task_items, zero])

                with paddle.no_grad():
                    t_output = teacher_ref(batch["img"])
                    if isinstance(t_output, tuple):
                        t_preds = t_output[1]
                    elif isinstance(t_output, dict):
                        t_preds = t_output
                    else:
                        zero = paddle.zeros([1], dtype=task_loss.dtype)
                        return paddle.concat([task_loss, zero]), paddle.concat([task_items, zero])

                s_preds = preds[1] if isinstance(preds, tuple) else preds
                logit_loss = _logit_kd_loss(s_preds, t_preds, temperature) if kd_w > 0 else paddle.zeros([1])
                feat_loss = paddle.zeros([1])
                if feat_w > 0:
                    s_branch = _prediction_branches(s_preds)[-1]
                    t_branch = _prediction_branches(t_preds)[-1]
                    s_feats = s_branch.get("feats", [])
                    t_feats = t_branch.get("feats", [])
                    if s_feats and t_feats:
                        score_weights = _teacher_score_weights(t_preds, t_feats)
                        feat_loss = _feature_kd_loss(s_feats, t_feats, score_weights)

                mask_loss = proto_loss = paddle.zeros([1])
                if self.task == "segment" and (mask_w > 0 or proto_w > 0):
                    mask_loss, proto_loss = _segmentation_kd_loss(s_preds, t_preds, temperature)

                kd_total = kd_w * logit_loss + feat_w * feat_loss + mask_w * mask_loss + proto_w * proto_loss
                combined_loss = paddle.concat([task_loss, kd_total.reshape([1])])
                combined_items = paddle.concat([task_items, kd_total.detach().reshape([1])])
                return combined_loss, combined_items

            if has_update:
                distill_criterion.update = base_criterion.update
            return distill_criterion

        student_model.criterion = make_distill_criterion(student_model)
        # 同步注入到 EMA 模型（validator 使用 EMA），保持 loss 维度一致
        if hasattr(trainer, "ema") and trainer.ema is not None:
            ema_model = unwrap_model(trainer.ema.ema)
            ema_model.criterion = make_distill_criterion(ema_model)
        # 添加 kd_loss 到训练进度显示
        if hasattr(trainer, "loss_names"):
            trainer.loss_names = (*trainer.loss_names, "kd_loss")
        LOGGER.info("[KD] Distillation criterion 已注入。")

    def train(self):
        """启动训练。"""
        self._trainer.train()

    @property
    def trainer(self):
        """获取内部的 DetectionTrainer 实例。"""
        return self._trainer


# ──────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────


def parse_args():
    """解析蒸馏训练的命令行参数（teacher/student 路径、KD 权重、温度等）。"""
    parser = argparse.ArgumentParser(
        description="YOLO26 知识蒸馏训练脚本（特征蒸馏 + 响应蒸馏）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="Teacher 模型权重路径（应比 student 大，例如 weights/yolo26/yolo26s.pdparams）",
    )
    parser.add_argument(
        "--student",
        type=str,
        required=True,
        help="Student 模型权重路径（待压缩的小模型，如 yolo26n.pdparams）",
    )
    parser.add_argument("--resume", type=str, default=None, help="从指定的蒸馏 checkpoint 严格续训")
    parser.add_argument("--data", type=str, default="data/your.yaml", help="数据集配置")
    parser.add_argument("--epochs", type=int, default=50, help="训练 epoch 数")
    parser.add_argument("--imgsz", nargs="+", default=["640"], metavar="SIZE", help="图像尺寸：SIZE、HxW 或 H W")
    parser.add_argument("--batch", type=int, default=8, help="Batch 大小")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--device", type=str, default="0", help="训练设备")
    parser.add_argument(
        "--overlap-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="分割标签是否采用 overlap mask；未指定时继承 student checkpoint",
    )
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率（蒸馏用更小 lr）")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终 lr 衰减因子")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic 增强概率")
    parser.add_argument("--patience", type=int, default=100, help="早停等待 epoch 数")
    parser.add_argument(
        "--kd-weight",
        type=float,
        default=1.0,
        help="Logit KD 权重（cls scores 上的 KL-div，默认 1.0）",
    )
    parser.add_argument(
        "--feat-weight",
        type=float,
        default=0.5,
        help="Feature KD 权重（置信度加权空间注意力，默认 0.5）",
    )
    parser.add_argument(
        "--mask-weight",
        type=float,
        default=0.25,
        help="分割 mask coefficient KD 权重（检测任务自动忽略，默认 0.25）",
    )
    parser.add_argument(
        "--proto-weight",
        type=float,
        default=0.25,
        help="分割 Proto/语义辅助头 KD 权重（检测任务自动忽略，默认 0.25）",
    )
    parser.add_argument(
        "--temperature",
        "-T",
        type=float,
        default=4.0,
        help="KD 温度参数，越大 soft label 越平滑",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="优化器（SGD / Adam / AdamW）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/detect/distill",
        help="输出目录",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=5,
        help="最后 N epoch 关闭 Mosaic 增强",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parsed_imgsz = normalize_static_imgsz(args.imgsz)
    if parsed_imgsz.height != parsed_imgsz.width:
        raise ValueError("蒸馏属于训练路线，当前仅支持方形 imgsz")
    args.imgsz = parsed_imgsz.height

    print("=" * 65)
    print("  YOLO26 知识蒸馏训练（Feature + Response KD）")
    print(f"  Teacher:      {args.teacher}")
    print(f"  Student:      {args.student}")
    print(f"  数据集:       {args.data}")
    print(f"  Epoch 数:     {args.epochs}")
    print(f"  Logit KD:     {args.kd_weight}")
    print(f"  Feature KD:   {args.feat_weight}")
    print(f"  Mask KD:      {args.mask_weight}")
    print(f"  Proto KD:     {args.proto_weight}")
    print(f"  温度:         {args.temperature}")
    print(f"  输出:         {args.output}")
    print("=" * 65)

    output_path = Path(args.output).resolve()
    project = str(output_path.parent)
    name = output_path.name

    overrides = dict(
        model=str(args.resume or args.student),
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        lr0=args.lr0,
        lrf=args.lrf,
        mosaic=args.mosaic,
        patience=args.patience,
        optimizer=args.optimizer,
        close_mosaic=args.close_mosaic,
        project=project,
        name=name,
        exist_ok=True,
    )
    if args.resume:
        overrides["resume"] = str(args.resume)
    if args.overlap_mask is not None:
        overrides["overlap_mask"] = args.overlap_mask

    trainer = DistillDetectionTrainer(
        teacher_weights=args.teacher,
        kd_weight=args.kd_weight,
        feat_weight=args.feat_weight,
        mask_weight=args.mask_weight,
        proto_weight=args.proto_weight,
        temperature=args.temperature,
        overrides=overrides,
    )
    trainer.train()

    print(f"\n知识蒸馏完成！")
    print(f"最佳权重: {args.output}/weights/best.pdparams")
    print("评估命令:")
    print(f'  python -c "')
    print(f"  from ddyolo26 import YOLO")
    print(f"  m = YOLO('{args.output}/weights/best.pdparams')")
    print(f"  r = m.val(data='{args.data}', split='val')")
    print(f"  print(r.box.map50, r.box.map)")
    print(f'  "')
