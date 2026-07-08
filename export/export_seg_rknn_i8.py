#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file export_seg_rknn_i8.py
@brief Seg 模型 RKNN INT8 一步式导出入口。
@details
用户可以直接传入 Paddle 权重或 ONNX，脚本内部会自动执行：
1. 导出普通 seg ONNX；
2. 识别并裁剪到 `seg_pre_dist` 四输出或 `seg_pre_dfl` 五输出主线；
3. 生成 RKNN INT8 模型（RGB 校准）。

两条主线分别对应：
- `seg_pre_dist`：YOLO26-Seg，CPU 端做 dist2bbox + mask 重建；
- `seg_pre_dfl`：YOLOv8-Seg，CPU 端额外执行 DFL softmax-expectation。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cleanup_paths(paths: list[str]) -> None:
    """!
    @brief 删除一步式导出过程中的临时文件和目录。
    @param paths 需要清理的路径列表。
    """
    import os
    import shutil

    for path in paths:
        if not path:
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)


def parse_args() -> argparse.Namespace:
    """!
    @brief 解析命令行参数。
    @return 命令行参数对象。
    """
    p = argparse.ArgumentParser(description="一步导出 seg RKNN INT8（seg_pre_dist / seg_pre_dfl）")
    p.add_argument("--weights", required=True, help="输入权重路径（.pdparams / onnx）")
    p.add_argument("--data", required=True, help="校准数据集 YAML")
    p.add_argument("--output", default=None, help="输出 .rknn 路径")
    p.add_argument("--target", default="rk3588", choices=["rk3588", "rk3588s", "rk3576", "rk3562"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--calib-images", type=int, default=50)
    p.add_argument("--calib-offset", type=int, default=0, help="校准图片起始偏移，用于避免与评测集重叠")
    p.add_argument("--algorithm", default="auto", choices=["auto", "normal", "mmse", "kl_divergence"])
    p.add_argument("--optimization-level", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--no-fix", action="store_true", help="跳过 Paddle ONNX NPU 兼容性修复")
    return p.parse_args()


def default_algorithm(route: str, requested: str) -> str:
    """!
    @brief 为不同分割主线选择默认量化算法。
    @param route 主线路由名，`seg_pre_dist` 或 `seg_pre_dfl`。
    @param requested 用户显式指定的算法；若不是 `auto` 则直接返回。
    @return 实际使用的量化算法名称。
    """
    if requested != "auto":
        return requested
    return "normal"


def default_output_path(weights_path: str, route: str, imgsz: int) -> str:
    """!
    @brief 生成默认 RKNN 输出路径。
    @param weights_path 用户输入权重路径。
    @param route 推断出的 seg 主线，`seg_pre_dist` 或 `seg_pre_dfl`。
    @return 默认输出 `.rknn` 绝对路径，位于输入模型同目录。
    """
    p = Path(weights_path)
    stem = p.stem
    if stem.endswith("_paddle"):
        stem = stem[:-7]
    if "_fp32_" in stem:
        stem = stem.split("_fp32_", 1)[0]
    route_tag = "seg_predfl" if route == "seg_pre_dfl" else "seg_predist"
    if route == "seg_pre_dist":
        for prefix in ("yolo26n_seg_", "yolo26n-seg_", "yolo26_seg_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
                break
    base = stem.split("_paddle_", 1)[0]
    return str((p.parent / f"{base}_paddle_{route_tag}_int8_{imgsz}.rknn").resolve())


def build_rknn_int8(
    onnx_path: str,
    output_path: str,
    data_yaml: str,
    imgsz: int,
    target: str,
    calib_images: int,
    algorithm: str,
    optimization_level: int,
    calib_offset: int = 0,
) -> str:
    """!
    @brief 将已准备好的 Seg ONNX 编译为 RKNN INT8。
    @param onnx_path 已准备好的 `seg_pre_dist` 或 `seg_pre_dfl` ONNX 路径。
    @param output_path 输出 `.rknn` 路径。
    @param data_yaml 校准数据集 YAML。
    @param imgsz 校准输入尺寸。
    @param target 目标平台。
    @param calib_images 校准图像数量。
    @param algorithm 量化算法。
    @param optimization_level RKNN 编译优化级别。
    @param calib_offset 校准图片起始偏移，用于避免与评测集重叠。
    @return 成功导出的 `.rknn` 路径。
    @throw RuntimeError 当 RKNN load/build/export 任一阶段失败时抛出。
    """
    # 必须在导入 rknn.api 之前修补 protobuf 7.x 不兼容。
    from tools.eval.backend_utils import patch_onnx_strip_doc_string_for_protobuf7

    patch_onnx_strip_doc_string_for_protobuf7()

    from rknn.api import RKNN
    from export.export_rknn import _save_calib_dataset

    list_path = _save_calib_dataset(data_yaml, imgsz, calib_images, offset=calib_offset)
    cleanup_paths = [str(Path(list_path).parent)]
    rknn = RKNN(verbose=True)

    try:
        rknn.config(
            target_platform=target,
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            optimization_level=optimization_level,
            quantized_dtype="w8a8",
            quantized_algorithm=algorithm,
        )

        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            raise RuntimeError(f"load_onnx 失败: {ret}")
        ret = rknn.build(do_quantization=True, dataset=list_path)
        if ret != 0:
            raise RuntimeError(f"build 失败: {ret}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            raise RuntimeError(f"export_rknn 失败: {ret}")
        return output_path
    finally:
        rknn.release()
        _cleanup_paths(cleanup_paths)


def main() -> int:
    """!
    @brief 脚本主入口。
    @return 成功返回 `0`。
    """
    args = parse_args()

    from export.seg_onnx_routes import prepare_seg_onnx_i8_input
    from export.export_rknn import prepare_onnx
    from quant.quantize import auto_export_onnx

    route, prepared_onnx_path, cleanup_paths = prepare_seg_onnx_i8_input(
        str(args.weights),
        args.imgsz,
        auto_export_onnx,
    )

    try:
        algorithm = default_algorithm(route, args.algorithm)
        fixed_onnx = prepare_onnx(prepared_onnx_path, fix_paddle=not args.no_fix)
        if fixed_onnx != prepared_onnx_path:
            cleanup_paths.append(fixed_onnx)
        output_path = args.output or default_output_path(str(args.weights), route, args.imgsz)
        print(f"[SEG-RKNN-I8] route={route} algorithm={algorithm}")
        print(f"[SEG-RKNN-I8] onnx={fixed_onnx}")
        print(f"[SEG-RKNN-I8] output={output_path}")

        build_rknn_int8(
            fixed_onnx,
            output_path,
            args.data,
            args.imgsz,
            args.target,
            args.calib_images,
            algorithm,
            args.optimization_level,
            calib_offset=args.calib_offset,
        )
        size_mb = Path(output_path).stat().st_size / 1024 / 1024
        print(f"[SEG-RKNN-I8] 完成: {output_path} ({size_mb:.1f} MB)")
        return 0
    finally:
        _cleanup_paths(cleanup_paths)


if __name__ == "__main__":
    raise SystemExit(main())
