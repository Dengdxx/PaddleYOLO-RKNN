#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file export_seg_onnx_i8.py
@brief Seg 模型一步式导出 INT8 ONNX。
@details
该脚本用于 `scripts/export_all_models.py` 的跨 conda 环境导出：
1. 根据输入权重自动导出或复用 seg ONNX；
2. 裁剪到 `seg_pre_dist` 四输出或 `seg_pre_dfl` 五输出主线；
3. 使用 ONNX Runtime 静态量化生成 INT8 ONNX。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """!
    @brief 解析命令行参数。
    @return 命令行参数对象。
    """
    p = argparse.ArgumentParser(description="一步导出 seg INT8 ONNX（seg_pre_dist / seg_pre_dfl）")
    p.add_argument("--weights", required=True, help="输入权重路径（.pdparams / onnx）")
    p.add_argument("--data", required=True, help="校准数据集 YAML")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    p.add_argument("--output", default="", help="输出 INT8 ONNX 路径；默认与输入模型同目录")
    p.add_argument("--prepared-output", default="", help="额外保存量化前的 route FP32 ONNX；默认与输入模型同目录")
    p.add_argument("--skip-quant", action="store_true", help="只导出 route FP32 ONNX，不执行 ORT INT8 量化")
    p.add_argument("--calib-batches", type=int, default=20, help="校准 batch 数量")
    return p.parse_args()


def _route_tag(route: str) -> str:
    """把内部 route 名转换为文件名片段。"""
    return "seg_predfl" if route == "seg_pre_dfl" else "seg_predist"


def _base_stem(weights_path: str) -> str:
    """从输入模型路径提取基础 stem，避免重复拼接 route/precision 片段。"""
    stem = Path(weights_path).stem
    if stem.endswith("_paddle"):
        stem = stem[:-7]
    if "_fp32_" in stem:
        stem = stem.split("_fp32_", 1)[0]
    return stem.split("_paddle_", 1)[0]


def default_prepared_output_path(weights_path: str, route: str, imgsz: int) -> Path:
    """生成 route FP32 ONNX 默认路径，默认写到输入模型同目录。"""
    p = Path(weights_path).resolve()
    return p.parent / f"{_base_stem(weights_path)}_paddle_{_route_tag(route)}_fp32_{imgsz}.onnx"


def default_int8_output_path(weights_path: str, route: str, imgsz: int) -> Path:
    """生成 INT8 ONNX 默认路径，默认写到输入模型同目录。"""
    p = Path(weights_path).resolve()
    return p.parent / f"{_base_stem(weights_path)}_paddle_{_route_tag(route)}_int8_{imgsz}.onnx"


def main() -> int:
    """!
    @brief 脚本主入口。
    @return 成功返回 `0`。
    """
    args = parse_args()

    from export.seg_onnx_routes import prepare_seg_onnx_i8_input
    from quant.quantize import auto_export_onnx

    weights_path = str(Path(args.weights).resolve())
    route, prepared_path, cleanup = prepare_seg_onnx_i8_input(weights_path, args.imgsz, auto_export_onnx)
    try:
        prepared_output = Path(args.prepared_output).resolve() if args.prepared_output else None
        if prepared_output is None and args.skip_quant:
            prepared_output = default_prepared_output_path(weights_path, route, args.imgsz)
        if prepared_output is not None:
            prepared_output.parent.mkdir(parents=True, exist_ok=True)
            if Path(prepared_path).resolve() != prepared_output.resolve():
                shutil.copy2(prepared_path, prepared_output)
            size_mb = os.path.getsize(prepared_output) / 1024 / 1024
            print(f"[EXPORT-SEG-FP32-ROUTE-ONNX] 完成: {prepared_output} ({size_mb:.1f} MB)")
        if args.skip_quant:
            return 0

        import onnx as _onnx
        import numpy as _np
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from onnxruntime.quantization import preprocess as quant_preprocess

        from quant.quantize import build_calib_loader

        output = (
            Path(args.output).resolve() if args.output else default_int8_output_path(weights_path, route, args.imgsz)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        loader, n_batches = build_calib_loader(
            str(Path(args.data).resolve()), args.imgsz, batch=1, n_batches=args.calib_batches
        )

        m = _onnx.load(prepared_path)
        in_name = m.graph.input[0].name
        in_dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
        in_h, in_w = in_dims[2], in_dims[3]

        cached: list = []
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            imgs = batch["img"].astype("float32") / 255.0
            if imgs.shape[2] != in_h or imgs.shape[3] != in_w:
                import cv2

                resized = []
                for j in range(imgs.shape[0]):
                    img = imgs[j].transpose(1, 2, 0)
                    img = cv2.resize(img, (in_w, in_h))
                    resized.append(img.transpose(2, 0, 1))
                imgs = _np.stack(resized)
            cached.append(imgs)

        class _DR(CalibrationDataReader):
            """!
            @brief 向 ORT 量化器提供校准样本。
            """

            def __init__(self, samples):
                self._iter = iter(samples)

            def get_next(self):
                try:
                    s = next(self._iter)
                except StopIteration:
                    return None
                return {in_name: s}

        with tempfile.TemporaryDirectory() as td:
            pre_path = Path(td) / "pre.onnx"
            quant_preprocess.quant_pre_process(prepared_path, str(pre_path))
            quantize_static(
                str(pre_path),
                str(output),
                _DR(cached),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
            )

        size_mb = os.path.getsize(output) / 1024 / 1024
        print(f"[EXPORT-SEG-INT8-ONNX] 完成: {output} ({size_mb:.1f} MB)")
        return 0
    finally:
        for p in cleanup:
            if not p:
                continue
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    raise SystemExit(main())
