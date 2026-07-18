#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""导出 YOLO26-seg 训练辅助头的 one2many raw ONNX。

该脚本仅用于训练辅助头诊断，产物不得进入 RKNN 部署链。
YOLO26 部署必须使用 one-to-one `seg_pre_dist` 四输出。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ddyolo26 import YOLO
from export.input_shape import StaticInputShape, format_static_imgsz, normalize_static_imgsz


def export_one2many(weights: str, imgsz: StaticInputShape) -> str:
    """导出 one2many 检测头的 raw ONNX。

    参数:
        weights: paddle 权重路径 (.pdparams)
        imgsz: 输入图像尺寸

    返回:
        导出的 ONNX 文件路径
    """
    model = YOLO(weights)
    core = getattr(model, "model", model)

    # 启用 one2many 头 + raw 输出（无后处理尾图）
    if hasattr(core, "set_head_attr"):
        core.set_head_attr(export_use_one2many=True, export_raw_one2one=True)
    else:
        raise RuntimeError("模型头部无 set_head_attr 方法")

    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
    print(f"[INFO] one2many ONNX 已导出: {onnx_path}")

    src = Path(onnx_path)
    stem = src.stem.replace("_paddle", f"_o2m_{format_static_imgsz(imgsz)}")
    dst = src.with_name(stem + src.suffix)
    src.rename(dst)
    print(f"[INFO] 重命名为: {dst}")
    return str(dst)


def main():
    """入口函数。"""
    p = argparse.ArgumentParser(description="导出 YOLO26-seg one2many raw ONNX 中间产物")
    p.add_argument("--weights", required=True, help="Paddle 权重路径")
    p.add_argument("--imgsz", nargs="+", default=["640"], metavar="SIZE", help="输入尺寸：SIZE、HxW 或 H W")
    args = p.parse_args()
    imgsz = normalize_static_imgsz(args.imgsz)
    print(f"\n{'=' * 60}")
    print(f"  导出 one2many ONNX: imgsz={format_static_imgsz(imgsz)}")
    print(f"{'=' * 60}")
    export_one2many(args.weights, imgsz)


if __name__ == "__main__":
    main()
