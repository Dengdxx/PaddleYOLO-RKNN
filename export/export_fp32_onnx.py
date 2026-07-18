#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file export_fp32_onnx.py
@brief 导出单个模型的 FP32 ONNX。
@details
该脚本用于 `scripts/export_all_models.py` 的跨 conda 环境导出：
- `.pdparams` → `ddyolo26.YOLO` 导出：
  - YOLO26（`end2end=True`，one2one 头）：走 `_postprocess_export` 默认路径，
    输出 `[B, 300, 6]` xyxy，truly NMS-free（`e2e` 格式）；
  - YOLOv8（`end2end=False`）：设置 `export_raw_one2one=True` 绕过
    `_postprocess_export`，输出 `[B, nc+4, N]` raw（需 NMS）。

口径对应关系：
  YOLO26 `.pdparams` → `e2e` 格式，文件名含 `_e2e_`
  YOLOv8 `.pdparams` → `raw` 格式，文件名含 `_raw_`
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export.input_shape import StaticInputShape, normalize_static_imgsz


def parse_args() -> argparse.Namespace:
    """!
    @brief 解析命令行参数。
    @return 命令行参数对象。
    """
    p = argparse.ArgumentParser(description="导出单个模型的 FP32 ONNX")
    p.add_argument("--weights", required=True, help="输入 Paddle 权重路径（.pdparams）")
    p.add_argument("--data", required=True, help="数据集 YAML 或分类目录，用于生成权威类别名清单")
    p.add_argument("--imgsz", nargs="+", default=["640"], metavar="SIZE", help="导出输入尺寸：SIZE、HxW 或 H W")
    p.add_argument("--output", required=True, help="输出 ONNX 路径")
    return p.parse_args()


def export_fp32_onnx(weights_path: str, imgsz: StaticInputShape) -> str:
    """!
    @brief 根据权重类型导出 FP32 ONNX。
    @param weights_path 输入权重路径。
    @param imgsz 导出尺寸。
    @return 导出的 ONNX 路径。
    @throw ValueError 当输入不是 Paddle 权重时抛出。
    """
    p = Path(weights_path)
    suffix = p.suffix.lower()
    name_lower = p.stem.lower()
    is_paddle_pt = name_lower.endswith("_paddle") or name_lower.endswith("paddle")

    if suffix == ".pdparams" or (suffix == ".pt" and is_paddle_pt):
        from ddyolo26 import YOLO  # noqa: PLC0415

        yolo = YOLO(str(p))
        # 仅对 YOLOv8（end2end=False）设置 export_raw_one2one=True，绕过 cfg 白名单，
        # 避免 _postprocess_export 的整数运算在 ONNX opset 高版本中失效。
        # YOLO26（end2end=True）继续走 _postprocess_export 默认路径，
        # 输出 [B, 300, 6] xyxy（truly NMS-free），无需此绕过。
        is_end2end = any(getattr(m, "end2end", False) for m in yolo.model.modules() if hasattr(m, "end2end"))
        if not is_end2end and hasattr(yolo.model, "set_head_attr"):
            yolo.model.set_head_attr(export_raw_one2one=True)
        onnx_path = yolo.export(format="onnx", imgsz=imgsz, simplify=True)
    elif suffix == ".pt":
        raise ValueError(f"普通 .pt 权重不被 Paddle-only 导出支持: {weights_path}. 请使用 .pdparams。")
    else:
        raise ValueError(f"FP32 ONNX 导出不支持该 Paddle 权重后缀: {weights_path}")

    if isinstance(onnx_path, str):
        return onnx_path
    return str(p.with_suffix(".onnx"))


def main() -> int:
    """!
    @brief 脚本主入口。
    @return 成功返回 `0`。
    """
    args = parse_args()
    imgsz = normalize_static_imgsz(args.imgsz)
    produced = Path(export_fp32_onnx(args.weights, imgsz)).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if produced != output:
        if output.exists():
            output.unlink()
        shutil.move(str(produced), str(output))
    from export.model_manifest import infer_native_output_route, write_model_manifest

    output_route = infer_native_output_route(output)
    manifest_path = write_model_manifest(output, output, output_route, imgsz, data_yaml=args.data)
    size_mb = os.path.getsize(output) / 1024 / 1024
    print(f"[EXPORT-FP32-ONNX] 完成: {output} ({size_mb:.1f} MB)")
    print(f"[EXPORT-FP32-ONNX] 清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
