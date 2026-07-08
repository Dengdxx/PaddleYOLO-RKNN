# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO26 知识蒸馏训练脚本

策略：特征蒸馏 + 响应蒸馏（Feature-based + Response-based KD）
  - FPN 特征图上的 Channel-Wise L2 蒸馏（CWD，对密集预测效果更好）
  - 分类 logits 的温度缩放 KL 散度（Response KD）
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


# ──────────────────────────────────────────────────────────────────────
# KD 损失函数
# ──────────────────────────────────────────────────────────────────────


def _logit_kd_loss(s_preds: dict, t_preds: dict, temperature: float = 4.0) -> paddle.Tensor:
    """
    分类 logits KL 散度蒸馏。

    仅对 one2one 头的 scores 做温度缩放 KL 散度。
    one2one 头是最终推理使用的头，蒸馏其 logits 对部署精度提升最直接。

    参照官方 E2ELoss 设计：one2one 是训练后期的主导头。

    参数:
        s_preds: student 训练模式预测 dict
        t_preds: teacher eval 模式预测 dict
        temperature: KD 温度

    返回:
        scalar KD loss
    """
    import paddle
    import paddle.nn.functional as F

    T = temperature
    kd = paddle.zeros([1])

    for head_key in ("one2one", "one2many"):
        if head_key not in s_preds or head_key not in t_preds:
            continue
        s_scores = s_preds[head_key]["scores"]  # [B, NC, NA]
        t_scores = t_preds[head_key]["scores"]  # [B, NC, NA]

        # KL(teacher || student)，温度缩放
        s_log_prob = F.log_softmax(s_scores / T, axis=1)
        t_prob = F.softmax(t_scores / T, axis=1)
        cls_kd = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T**2)
        kd = kd + cls_kd

    return kd


def _feature_kd_loss(s_feats: list, t_feats: list) -> paddle.Tensor:
    """
    FPN 特征图 Channel-Wise 归一化 L2 蒸馏（CWD 简化版）。

    对每个 FPN 层级的特征图做 channel 维 L2 归一化后计算 MSE，
    这样不同 channel 数的 teacher/student 也能通过 1x1 适配层对齐。

    但由于 YOLO26 同系列模型 FPN 通道数随 scale 变化，
    此处仅在 s/t 通道数一致时直接匹配，否则跳过该层。

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

    for sf, tf in zip(s_feats, t_feats):
        if sf.shape[1] != tf.shape[1]:
            # 通道数不同（teacher/student 不同 scale），跳过
            continue
        # Spatial 尺寸可能不同（不同 batch rect padding），对齐
        if sf.shape[2:] != tf.shape[2:]:
            tf = F.interpolate(tf, size=sf.shape[2:], mode="bilinear", align_corners=False)

        # Channel-wise L2 归一化
        sf_norm = F.normalize(sf, p=2, axis=1)  # [B, C, H, W]
        tf_norm = F.normalize(tf.detach(), p=2, axis=1)
        feat_loss = feat_loss + F.mse_loss(sf_norm, tf_norm, reduction="mean")
        n_matched += 1

    if n_matched > 0:
        feat_loss = feat_loss / n_matched
    return feat_loss


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
        temperature: float = 4.0,
        overrides: dict = None,
    ):
        """初始化蒸馏训练器。

        参数:
            teacher_weights: Teacher 模型权重文件路径
            kd_weight: Logit KD 损失权重（KL 散度）
            feat_weight: 特征图 KD 损失权重（CWD-L2）
            temperature: KD 温度参数，越大 soft label 越平滑
            overrides: 传递给 DetectionTrainer 的配置覆盖
        """
        self.teacher_weights = teacher_weights
        self.kd_weight = kd_weight
        self.feat_weight = feat_weight
        self.temperature = temperature
        self.overrides = overrides or {}
        self._teacher = None

        from ddyolo26.models.yolo.detect import DetectionTrainer

        self._trainer = DetectionTrainer(overrides=self.overrides)
        self._trainer.add_callback("on_pretrain_routine_end", self._inject_distill)

    def _inject_distill(self, trainer):
        """在 trainer.model 加载完成后注入 KD criterion。"""
        import paddle
        from ddyolo26.utils.runtime import unwrap_model
        from ddyolo26.utils import LOGGER

        # ── 加载 teacher 模型 ─────────────────────────────────────────
        LOGGER.info(f"[KD] 加载 teacher 模型: {self.teacher_weights}")
        ckpt = paddle.load(str(self.teacher_weights))

        from ddyolo26.nn.tasks import DetectionModel, yaml_model_load
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

        teacher = DetectionModel(
            cfg=yaml_cfg or "ddyolo26/cfg/models/26/yolo26.yaml",
            nc=trainer.data["nc"],
            ch=trainer.data.get("channels", 3),
            verbose=False,
        )
        model_sd = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        if isinstance(model_sd, dict):
            # 过滤掉 shape 不匹配的键（如 teacher nc 不同导致 cls head 维度不同）
            teacher_sd = teacher.state_dict()
            compatible_sd = {k: v for k, v in model_sd.items() if k in teacher_sd and v.shape == teacher_sd[k].shape}
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
        LOGGER.info(f"[KD] Teacher 已冻结。logit_kd={self.kd_weight}, feat_kd={self.feat_weight}, T={self.temperature}")

        # ── 注入 KD criterion ─────────────────────────────────────────
        student_model = unwrap_model(trainer.model)

        if getattr(student_model, "criterion", None) is None:
            student_model.criterion = student_model.init_criterion()

        base_criterion = student_model.criterion
        kd_w = self.kd_weight
        feat_w = self.feat_weight
        temperature = self.temperature
        teacher_ref = teacher

        # 检查 base_criterion 是否有 update 方法（E2ELoss 有）
        has_update = hasattr(base_criterion, "update")

        def distill_criterion(preds, batch):
            """
            task_loss + logit_kd + feature_kd。
            始终返回 (loss[4], items[4])，验证时 kd_loss=0。
            """
            # 1. 标准检测 loss（E2ELoss / v8DetectionLoss）
            task_loss, task_items = base_criterion(preds, batch)

            # 验证阶段（teacher 不参与，kd_loss 补零保持维度一致）
            if not student_model.training:
                zero = paddle.zeros([1], dtype=task_loss.dtype)
                return paddle.concat([task_loss, zero]), paddle.concat([task_items, zero])

            # 2. teacher 前向
            with paddle.no_grad():
                t_output = teacher_ref(batch["img"])
                if isinstance(t_output, tuple):
                    t_preds = t_output[1]
                elif isinstance(t_output, dict):
                    t_preds = t_output
                else:
                    zero = paddle.zeros([1], dtype=task_loss.dtype)
                    return paddle.concat([task_loss, zero]), paddle.concat([task_items, zero])

            # 3. 解析 student preds（处理 tuple 格式）
            s_preds = preds[1] if isinstance(preds, tuple) else preds

            # 4. Logit KD（classification scores 上的 KL-div）
            logit_loss = _logit_kd_loss(s_preds, t_preds, temperature) if kd_w > 0 else paddle.zeros([1])

            # 5. Feature KD（FPN feature maps 上的 CWD-L2）
            feat_loss = paddle.zeros([1])
            if feat_w > 0:
                # 提取 student 和 teacher 的 FPN 特征
                for head_key in ("one2many",):  # 只在 one2many 上做 feat KD（有梯度）
                    s_feats = s_preds.get(head_key, {}).get("feats", [])
                    t_feats = t_preds.get(head_key, {}).get("feats", [])
                    if s_feats and t_feats:
                        feat_loss = feat_loss + _feature_kd_loss(s_feats, t_feats)

            # 6. 合并
            kd_total = kd_w * logit_loss + feat_w * feat_loss
            combined_loss = paddle.concat([task_loss, kd_total.reshape([1])])
            combined_items = paddle.concat([task_items, kd_total.detach().reshape([1])])
            return combined_loss, combined_items

        # 保留 E2ELoss.update() 方法
        if has_update:
            distill_criterion.update = base_criterion.update

        student_model.criterion = distill_criterion
        # 同步注入到 EMA 模型（validator 使用 EMA），保持 loss 维度一致
        if hasattr(trainer, "ema") and trainer.ema is not None:
            ema_model = unwrap_model(trainer.ema.ema)
            ema_model.criterion = distill_criterion
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
    parser.add_argument("--data", type=str, default="data/your.yaml", help="数据集配置")
    parser.add_argument("--epochs", type=int, default=50, help="训练 epoch 数")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--batch", type=int, default=8, help="Batch 大小")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--device", type=str, default="0", help="训练设备")
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率（蒸馏用更小 lr）")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终 lr 衰减因子")
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
        help="Feature KD 权重（FPN feats 上的 CWD-L2，默认 0.5）",
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

    print("=" * 65)
    print("  YOLO26 知识蒸馏训练（Feature + Response KD）")
    print(f"  Teacher:      {args.teacher}")
    print(f"  Student:      {args.student}")
    print(f"  数据集:       {args.data}")
    print(f"  Epoch 数:     {args.epochs}")
    print(f"  Logit KD:     {args.kd_weight}")
    print(f"  Feature KD:   {args.feat_weight}")
    print(f"  温度:         {args.temperature}")
    print(f"  输出:         {args.output}")
    print("=" * 65)

    output_path = Path(args.output)
    project = str(output_path.parent)
    name = output_path.name

    overrides = dict(
        model=str(args.student),
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        close_mosaic=args.close_mosaic,
        project=project,
        name=name,
        exist_ok=True,
    )

    trainer = DistillDetectionTrainer(
        teacher_weights=args.teacher,
        kd_weight=args.kd_weight,
        feat_weight=args.feat_weight,
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
