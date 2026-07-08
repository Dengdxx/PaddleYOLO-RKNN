#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""导出 YOLO26-seg 系列 raw ONNX（用于 RKNN INT8 量化）。

支持两种导出模式：

  one2many (默认):
    输出 (y, proto)，y 为合并的 [B, 4+nc+nm, 8400] 张量。
    适合在 PC 端做精度验证或走 ONNX 图手术再量化。

  rknn (官方 13 输出格式):
    输出 13 个 NCHW 张量，完全绕过 DFL/坐标解码/Sigmoid 拼接：
      对 i=0,1,2（stride=8,16,32）：
        box_i      [B, 4*reg_max, H_i, W_i]
        cls_i      [B, nc, H_i, W_i]         (sigmoid 已应用)
        cls_sum_i  [B, 1, H_i, W_i]          (fast CPU filter)
        mask_i     [B, nm, H_i, W_i]
      proto        [B, nm, H/4, W/4]
    与 rknn_model_zoo 官方 yolov8_seg 输出格式完全一致，
    令 RKNN Toolkit2 每尺度独立量化并充分利用三核拆分。

用法:
    # 默认 one2many（原有行为）
    python export_one2many_onnx.py \\
        --weights weights/yolo26seg/yolo26n-seg.pdparams \\
        --imgsz 640 480 384

    # RKNN 官方 13 输出格式
    python export_one2many_onnx.py \\
        --weights weights/yolo26seg/yolo26n-seg.pdparams \\
        --imgsz 640 480 384 \\
        --mode rknn
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ddyolo26 import YOLO


def export_one2many(weights: str, imgsz: int) -> str:
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
    stem = src.stem.replace("_paddle", f"_o2m_{imgsz}")
    dst = src.with_name(stem + src.suffix)
    src.rename(dst)
    print(f"[INFO] 重命名为: {dst}")
    return str(dst)


def export_rknn(weights: str, imgsz: int) -> str:
    """导出 RKNN 官方 13 输出格式 ONNX（模型级 per-scale，无图手术）。

    输出 13 个 NCHW Tensor（3 尺度 × 4 输出 + proto），
    与 rknn_model_zoo yolov8_seg 接口一致。

    参数:
        weights: paddle 权重路径 (.pdparams)
        imgsz: 输入图像尺寸

    返回:
        导出的 ONNX 文件路径
    """
    model = YOLO(weights)
    core = getattr(model, "model", model)

    if not hasattr(core, "set_head_attr"):
        raise RuntimeError("模型头部无 set_head_attr 方法")

    # 启用 RKNN 专用支路（13 输出，per-scale sigmoid+cls_sum）
    core.set_head_attr(
        export_use_one2many=True,  # 使用 one2many(cv2/cv3/cv4) 而非 one2one
        export_rknn=True,  # 激活 _forward_export_rknn() 分支
        export_raw_one2one=False,  # 避免走 raw 分支
    )

    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
    print(f"[INFO] RKNN ONNX 已导出: {onnx_path}")

    src = Path(onnx_path)
    stem = src.stem.replace("_paddle", f"_rknn_{imgsz}")
    dst = src.with_name(stem + src.suffix)
    src.rename(dst)
    print(f"[INFO] 重命名为: {dst}")
    return str(dst)


def main():
    """入口函数。"""
    p = argparse.ArgumentParser(description="导出 YOLO26-seg raw ONNX（one2many 或 RKNN 13 输出格式）")
    p.add_argument("--weights", required=True, help="Paddle 权重路径")
    p.add_argument("--imgsz", nargs="+", type=int, default=[640, 480, 384], help="输入尺寸列表")
    p.add_argument(
        "--mode",
        default="one2many",
        choices=["one2many", "rknn"],
        help="导出模式：one2many=原有合并输出; rknn=官方 13 输出",
    )
    args = p.parse_args()

    export_fn = export_rknn if args.mode == "rknn" else export_one2many

    for sz in args.imgsz:
        print(f"\n{'=' * 60}")
        print(f"  导出 {args.mode} ONNX: imgsz={sz}")
        print(f"{'=' * 60}")
        export_fn(args.weights, sz)


if __name__ == "__main__":
    main()
