#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file export_predist_fp32_onnx.py
@brief 将检测权重裁剪为 pre_dist / pre_dfl 主线，导出不经量化的 FP32 ONNX 基线。
@details
供受控实验使用：与 pre_dist INT8 ONNX 对比时两者使用相同链路，
只有精度一个变量，消除了 e2e vs pre_dist 路线差异的干扰。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export.det_onnx_routes import cleanup_temp_paths, prepare_det_onnx_i8_input
from export.input_shape import StaticInputShape, format_static_imgsz, normalize_static_imgsz
from quant.quantize import auto_export_onnx


def parse_args() -> argparse.Namespace:
    """!
    @brief 解析命令行参数。
    @return 命令行参数对象。
    """
    p = argparse.ArgumentParser(description="导出 pre_dist / pre_dfl FP32 ONNX 基线（不经量化）")
    p.add_argument("--weights", required=True, help="输入权重路径（.pdparams / onnx）")
    p.add_argument("--data", required=True, help="数据集 YAML，用于生成权威类别名清单")
    p.add_argument("--output", default=None, help="输出 .onnx 路径；默认与权重同目录")
    p.add_argument("--imgsz", nargs="+", default=["640"], metavar="SIZE", help="导出尺寸：SIZE、HxW 或 H W")
    return p.parse_args()


def default_output_path(weights_path: str, route: str, imgsz: StaticInputShape) -> str:
    """!
    @brief 生成默认 FP32 ONNX 输出路径。
    @param weights_path 用户输入权重路径。
    @param route 推断出的检测主线路由，`pre_dist` 或 `pre_dfl`。
    @return 默认输出 `.onnx` 绝对路径（与权重同目录）。
    """
    p = Path(weights_path)
    # 路由名去掉下划线以对齐命名规范：pre_dist → predist
    route_slug = route.replace("_", "")
    return str(p.parent / f"{p.stem}_{route_slug}_fp32_{format_static_imgsz(imgsz)}.onnx")


def main() -> int:
    """!
    @brief 脚本主入口。
    @return 成功返回 `0`。
    """
    args = parse_args()
    imgsz = normalize_static_imgsz(args.imgsz)

    route, prepared_onnx_path, cleanup_paths = prepare_det_onnx_i8_input(
        str(args.weights),
        imgsz,
        auto_export_onnx,
    )

    try:
        output_path = args.output or default_output_path(str(args.weights), route, imgsz)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"[DET-FP32-ROUTE] route={route}")
        print(f"[DET-FP32-ROUTE] source_onnx={prepared_onnx_path}")
        print(f"[DET-FP32-ROUTE] output={output_path}")

        shutil.copy2(prepared_onnx_path, output_path)
        from export.model_manifest import write_model_manifest

        public_route = "predfl" if route == "pre_dfl" else "predist"
        manifest_path = write_model_manifest(output_path, output_path, public_route, imgsz, data_yaml=args.data)
        print(f"[DET-FP32-ROUTE] 完成: {output_path}")
        print(f"[DET-FP32-ROUTE] 清单: {manifest_path}")
        return 0
    finally:
        cleanup_temp_paths(cleanup_paths)


if __name__ == "__main__":
    raise SystemExit(main())
