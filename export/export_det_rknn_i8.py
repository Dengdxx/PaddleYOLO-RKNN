#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file export_det_rknn_i8.py
@brief 检测模型 RKNN 一步式导出入口。
@details
用户可以直接传入 Paddle 权重或 ONNX，脚本内部会自动执行：
1. 导出普通 ONNX；
2. 识别并裁剪到 `pre_dist / pre_dfl`；
3. 生成 RKNN `fp16 / int8` 模型。

整个流程只允许检测主线 `pre_dist / pre_dfl` 两种输出契约。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export.det_onnx_routes import cleanup_temp_paths, prepare_det_onnx_i8_input
from export.export_rknn import _save_calib_dataset, prepare_onnx

from quant.quantize import auto_export_onnx


def parse_args() -> argparse.Namespace:
    """!
    @brief 解析命令行参数。
    @return 命令行参数对象。
    """
    p = argparse.ArgumentParser(description="一步导出 det RKNN（内部仅走 pre_dist / pre_dfl）")
    p.add_argument("--weights", required=True, help="输入权重路径（.pdparams / onnx）")
    p.add_argument("--data", default=None, help="校准数据集 YAML（仅 int8 模式需要）")
    p.add_argument("--output", default=None, help="输出 .rknn 路径")
    p.add_argument("--mode", default="int8", choices=["fp16", "int8"], help="导出模式")
    p.add_argument("--target", default="rk3588", choices=["rk3588", "rk3588s", "rk3576", "rk3562"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--calib-images", type=int, default=50)
    p.add_argument("--calib-offset", type=int, default=0, help="校准图片起始偏移，用于避免与评测集重叠")
    p.add_argument("--algorithm", default="auto", choices=["auto", "normal", "mmse", "kl_divergence"])
    p.add_argument("--optimization-level", type=int, default=3, choices=[1, 2, 3])
    p.add_argument(
        "--auto-hybrid",
        action="store_true",
        help="启用 RKNN auto_hybrid（敏感层自动 FP16，其余 INT8），与 mmse 配合可大幅缩短量化时间",
    )
    p.add_argument("--no-fix", action="store_true", help="跳过 Paddle ONNX NPU 兼容性修复")
    return p.parse_args()


def default_algorithm(route: str, requested: str) -> str:
    """!
    @brief 为不同检测主线选择默认量化算法。
    @param route 主线路由名，`pre_dist` 或 `pre_dfl`。
    @param requested 用户显式指定的算法；若不是 `auto` 则直接返回。
    @return 实际使用的量化算法名称。
    """
    if requested != "auto":
        return requested
    return "mmse" if route == "pre_dist" else "normal"


def default_output_path(weights_path: str, route: str, mode: str, imgsz: int) -> str:
    """!
    @brief 生成默认 RKNN 输出路径。
    @param weights_path 用户输入权重路径。
    @param route 推断出的检测主线路由。
    @param mode 导出模式，`fp16` 或 `int8`。
    @return 默认输出 `.rknn` 绝对路径，位于输入模型同目录。
    """
    p = Path(weights_path)
    stem = p.stem
    if stem.endswith("_paddle"):
        stem = stem[:-7]
    if "_fp32_" in stem:
        stem = stem.split("_fp32_", 1)[0]
    route_tag = "predfl" if route == "pre_dfl" else "predist"
    base = stem.split("_paddle_", 1)[0]
    precision = "fp16" if mode == "fp16" else "int8"
    return str((p.parent / f"{base}_paddle_{route_tag}_{precision}_{imgsz}.rknn").resolve())


def build_rknn(
    onnx_path: str,
    output_path: str,
    mode: str,
    data_yaml: str | None,
    imgsz: int,
    target: str,
    calib_images: int,
    algorithm: str,
    optimization_level: int,
    auto_hybrid: bool = False,
    calib_offset: int = 0,
) -> str:
    """!
    @brief 将主线 ONNX 编译为 RKNN。
    @param onnx_path 已准备好的 `pre_dist / pre_dfl` ONNX 路径。
    @param output_path 输出 `.rknn` 路径。
    @param mode 导出模式，`fp16` 或 `int8`。
    @param data_yaml 校准数据集 YAML；仅 `int8` 模式需要。
    @param imgsz 校准输入尺寸。
    @param target 目标平台。
    @param calib_images 校准图像数量。
    @param algorithm 量化算法。
    @param optimization_level RKNN 编译优化级别。
    @param auto_hybrid 是否启用 RKNN 自动混合量化。
    @param calib_offset 校准图片起始偏移，用于避免与评测集重叠。
    @return 成功导出的 `.rknn` 路径。
    """
    from tools.eval.backend_utils import patch_onnx_strip_doc_string_for_protobuf7

    patch_onnx_strip_doc_string_for_protobuf7()
    from rknn.api import RKNN

    cleanup_paths: list[str] = []
    rknn = RKNN(verbose=True)

    try:
        config_kwargs = {
            "target_platform": target,
            "mean_values": [[0, 0, 0]],
            "std_values": [[255, 255, 255]],
            "optimization_level": optimization_level,
        }
        list_path = None
        if mode == "int8":
            if not data_yaml:
                raise ValueError("--data 在 int8 模式下为必选")
            list_path = _save_calib_dataset(data_yaml, imgsz, calib_images, offset=calib_offset)
            cleanup_paths.append(str(Path(list_path).parent))
            config_kwargs.update(
                quantized_dtype="w8a8",
                quantized_algorithm=algorithm,
            )
        rknn.config(**config_kwargs)

        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            raise RuntimeError(f"load_onnx 失败: {ret}")
        ret = rknn.build(
            do_quantization=(mode == "int8"),
            dataset=list_path,
            auto_hybrid=auto_hybrid if mode == "int8" else False,
        )
        if ret != 0:
            raise RuntimeError(f"build 失败: {ret}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            raise RuntimeError(f"export_rknn 失败: {ret}")
        return output_path
    finally:
        rknn.release()
        cleanup_temp_paths(cleanup_paths)


def main() -> int:
    """!
    @brief 脚本主入口。
    @return 成功返回 `0`。
    """
    args = parse_args()

    route, prepared_onnx_path, cleanup_paths = prepare_det_onnx_i8_input(
        str(args.weights),
        args.imgsz,
        auto_export_onnx,
    )

    try:
        algorithm = default_algorithm(route, args.algorithm)
        fixed_onnx = prepare_onnx(prepared_onnx_path, fix_paddle=not args.no_fix)
        if fixed_onnx != prepared_onnx_path:
            cleanup_paths.append(fixed_onnx)

        output_path = args.output or default_output_path(str(args.weights), route, args.mode, args.imgsz)
        tag = "DET-RKNN-FP16" if args.mode == "fp16" else "DET-RKNN-I8"
        print(f"[{tag}] route={route} mode={args.mode} algorithm={algorithm}")
        print(f"[{tag}] onnx={fixed_onnx}")
        print(f"[{tag}] output={output_path}")

        build_rknn(
            fixed_onnx,
            output_path,
            args.mode,
            args.data,
            args.imgsz,
            args.target,
            args.calib_images,
            algorithm,
            args.optimization_level,
            auto_hybrid=args.auto_hybrid,
            calib_offset=args.calib_offset,
        )
        print(f"[{tag}] 完成: {output_path}")
        return 0
    finally:
        cleanup_temp_paths(cleanup_paths)


if __name__ == "__main__":
    raise SystemExit(main())
