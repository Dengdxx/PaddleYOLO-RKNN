#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""!
@file qat_finetune_paddle.py
@brief Paddle 检测模型量化感知训练（QAT）微调脚本。
@details
在已训练的 Paddle 检测模型上执行量化感知微调，使模型学习适应 INT8 量化误差。

核心思路：利用 register_forward_post_hook 在 Conv2D 层输出上注入 INT8 量化噪声，
模拟 RKNN 板端 w8a8 量化行为。这种方法完全兼容 ddyolo26 训练流程，
无需修改模型结构或使用 PaddlePaddle 官方 QAT API（后者不兼容自定义模块）。

量化噪声模拟：
- 每个 Conv2D 的输出被 fake-quantize 到 INT8 范围 [0, 255]
- 使用 EMA 统计量追踪每层的动态范围（min/max）
- 反向传播中用 STE（Straight-Through Estimator）实现梯度穿透

用法：
    # 从已训练的 best.pdparams 开始 QAT 微调（推荐 20-50 epoch）
    conda run -n pdrk python3 quant/qat_finetune_paddle.py \
        --weights runs/detect/train/weights/best.pdparams \
        --data coco.yaml \
        --epochs 30 --lr0 5e-5 --batch 16

    # 快速验证（3 epoch）
    conda run -n pdrk python3 quant/qat_finetune_paddle.py \
        --weights runs/detect/train/weights/best.pdparams \
        --data coco.yaml \
        --epochs 3 --batch 8
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import paddle
import paddle.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export.input_shape import format_static_imgsz, normalize_static_imgsz, static_imgsz_hw


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="Paddle 检测模型 QAT 微调")
    p.add_argument("--weights", required=True, help="预训练 .pdparams 模型路径")
    p.add_argument("--data", required=True, help="数据集 YAML 路径")
    p.add_argument("--epochs", type=int, default=30, help="QAT 微调 epoch 数")
    p.add_argument("--lr0", type=float, default=5e-5, help="初始学习率（应比原始训练小 100x）")
    p.add_argument("--batch", type=int, default=16, help="批大小")
    p.add_argument("--imgsz", nargs="+", default=["640"], metavar="SIZE", help="输入尺寸：SIZE、HxW 或 H W")
    p.add_argument("--project", default="runs/detect", help="输出项目目录")
    p.add_argument("--name", default="qat", help="输出实验名称")
    p.add_argument("--device", default="0", help="GPU 设备")
    p.add_argument("--ema-momentum", type=float, default=0.01, help="EMA 动态范围追踪的动量（越小越平滑）")
    return p.parse_args()


class FakeQuantHook:
    """在 Conv2D 输出上模拟 INT8 量化的前向 hook。

    使用 EMA 统计量追踪每层的 min/max 动态范围，
    在前向传播中用手动 quantize-dequantize + STE 模拟量化。
    反向传播时梯度通过 STE（Straight-Through Estimator）穿透。
    """

    def __init__(self, momentum=0.01):
        """初始化 FakeQuantHook。

        参数:
            momentum: EMA 更新动量。
        """
        self.momentum = momentum
        self.hooks = []
        self.ema_min = defaultdict(lambda: None)
        self.ema_max = defaultdict(lambda: None)
        self.enabled = True

    def _hook_fn(self, module, input, output, name=""):
        """Conv2D 前向 hook：对输出执行 fake quantize。

        使用 STE（Straight-Through Estimator）：
        前向：output_dq = dequant(quant(output))
        反向：grad 直接穿透，等价于 output + (output_dq - output).detach()

        参数:
            module: Conv2D 模块。
            input: 输入张量。
            output: Conv2D 原始输出。
            name: 层名称（用于追踪统计量）。

        返回:
            fake-quantized 后的输出。
        """
        if not self.enabled or not module.training:
            return output

        # 计算当前 batch 的动态范围
        cur_min = output.detach().min().item()
        cur_max = output.detach().max().item()

        # EMA 更新
        if self.ema_min[name] is None:
            self.ema_min[name] = cur_min
            self.ema_max[name] = cur_max
        else:
            m = self.momentum
            self.ema_min[name] = (1 - m) * self.ema_min[name] + m * cur_min
            self.ema_max[name] = (1 - m) * self.ema_max[name] + m * cur_max

        ema_min = self.ema_min[name]
        ema_max = self.ema_max[name]

        # 避免零范围
        if ema_max - ema_min < 1e-6:
            return output

        # 计算 scale 和 zero_point（UINT8，与 RKNN w8a8 匹配）
        scale = (ema_max - ema_min) / 255.0
        zero_point = int(round(-ema_min / scale))
        zero_point = max(0, min(255, zero_point))

        # STE fake quantize：前向量化+反量化，反向直接穿透
        # 量化：round((x - ema_min) / scale)，并 clamp 到 [0, 255]
        # 反量化：x_q * scale + ema_min
        x_q = paddle.round((output - ema_min) / scale).clip(0, 255)
        x_dq = x_q * scale + ema_min
        # STE: 前向用 x_dq，反向梯度穿透到 output
        return output + (x_dq - output).detach()

    def attach(self, model):
        """在模型的所有 Conv2D 层上注册 hook。

        参数:
            model: PaddlePaddle 模型。

        返回:
            注册的 hook 数量。
        """
        n = 0
        for layer_name, mod in model.named_sublayers():
            if isinstance(mod, nn.Conv2D):
                hook = mod.register_forward_post_hook(lambda m, i, o, ln=layer_name: self._hook_fn(m, i, o, ln))
                self.hooks.append(hook)
                n += 1
        return n

    def remove(self):
        """移除所有已注册的 hook。"""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def main():
    """主流程：加载模型 -> 注册 FakeQuant hook -> QAT 微调 -> 保存。"""
    args = parse_args()
    parsed_imgsz = normalize_static_imgsz(args.imgsz)
    input_h, input_w = static_imgsz_hw(parsed_imgsz)
    if input_h != input_w:
        raise ValueError("QAT 属于训练路线，当前仅支持方形 imgsz；静态矩形用于 predict/export")
    args.imgsz = input_h
    shape_tag = format_static_imgsz(parsed_imgsz)

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    try:
        from ddyolo26 import YOLO
    except ImportError:
        print("错误：需要 ddyolo26 包。请在 pdrk conda 环境中运行。", file=sys.stderr)
        sys.exit(1)

    print(f"[QAT-Paddle] 加载预训练模型: {args.weights}")
    yolo = YOLO(args.weights)

    # 创建 FakeQuant hook 管理器
    fq_hook = FakeQuantHook(momentum=args.ema_momentum)

    # 用 ddyolo26 回调机制在训练开始时注册 hook
    def on_train_start(trainer):
        """训练开始时注册 FakeQuant hook。"""
        model = trainer.model
        # 解开 DDP 包装
        if hasattr(model, "_layers"):
            model = model._layers
        n = fq_hook.attach(model)
        print(f"[QAT-Paddle] 已在 {n} 个 Conv2D 层注册 FakeQuantize hook")

    def on_train_end(trainer):
        """训练结束时移除 hook。"""
        fq_hook.remove()
        print("[QAT-Paddle] 已移除所有 FakeQuantize hook")

    # 注册回调
    yolo.add_callback("on_train_start", on_train_start)
    yolo.add_callback("on_train_end", on_train_end)

    # 开始 QAT 微调
    print(f"[QAT-Paddle] 开始微调: epochs={args.epochs}, lr0={args.lr0}, batch={args.batch}, imgsz={args.imgsz}")

    yolo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=0.1,
        warmup_epochs=1,
        cos_lr=True,
        resume=False,
        project=args.project,
        name=args.name,
        device=args.device,
        exist_ok=True,
    )

    out_dir = f"{args.project}/{args.name}"
    print(f"\n[QAT-Paddle] 微调完成！")
    print(f"[QAT-Paddle] 最佳权重: {out_dir}/weights/best.pdparams")
    print(f"\n[QAT-Paddle] 后续导出步骤：")
    print(f"  # 1. 导出 ONNX")
    print(
        f'  conda run -n pdrk python3 -c "from ddyolo26 import YOLO; '
        f"YOLO('{out_dir}/weights/best.pdparams').export(format='onnx', "
        f'imgsz=[{input_h}, {input_w}], simplify=True)"'
    )
    print(f"  # 2. 转换 RKNN INT8（检测主线请走 pre_dist / pre_dfl 导出链路）")
    print(f"  python export/export_det_rknn_i8.py --weights <exported.onnx> --data {args.data}")
    print(f"  # 3. 板端评估")
    print(f"  python -m tools.eval --backend rknn --model <rknn> --data {args.data}")


if __name__ == "__main__":
    main()
