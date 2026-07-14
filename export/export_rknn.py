#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""通用 RKNN 导出入口。

当前该入口仅保留：
1. `.pdparams` -> ONNX 自动导出
2. Paddle ONNX NPU 兼容性修复
3. FP16 RKNN 编译

检测侧 INT8 主线已经迁移到专用的 `pre_dist` / `pre_dfl` 导出链路，因此不再在
这个通用入口里维护旧的 INT8 分支。
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", "No ccache found")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")

from export.input_shape import StaticInputShape, format_static_imgsz, normalize_static_imgsz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出 YOLO 到 RKNN（通用 FP16 入口）")
    p.add_argument("--weights", required=True, help="ONNX 或 .pdparams 模型路径")
    p.add_argument("--data", required=True, help="数据集 YAML 或分类目录，用于生成权威类别名清单")
    p.add_argument("--output", default=None, help="输出 .rknn 路径（默认：自动）")
    p.add_argument(
        "--target",
        default="rk3588",
        choices=["rk3588", "rk3588s", "rk3576", "rk3562"],
        help="目标 NPU 平台（默认：rk3588）",
    )
    p.add_argument("--imgsz", nargs="+", default=["640"], metavar="SIZE", help="输入图像尺寸：SIZE、HxW 或 H W")
    p.add_argument(
        "--quantize",
        default="fp16",
        choices=["fp16"],
        help="量化方式：通用入口仅支持 fp16；检测 INT8 请使用专用 pre_dist / pre_dfl 导出链路",
    )
    p.add_argument("--compress-weight", action="store_true", help="启用权重压缩")
    p.add_argument("--model-pruning", action="store_true", help="启用编译时模型剪枝")
    p.add_argument("--optimization-level", type=int, default=3, choices=[1, 2, 3], help="RKNN 编译优化级别")
    p.add_argument("--no-fix", action="store_true", help="跳过 Paddle ONNX NPU 兼容性修复")
    p.add_argument(
        "--e2e-mode",
        default="postprocess",
        choices=["postprocess", "one2one_raw"],
        help="end2end 导出模式：postprocess 保持模型尾部，one2one_raw 导出 one2one 原始单输出",
    )
    return p.parse_args()


def default_output_path(weights_path: str, imgsz: StaticInputShape) -> str:
    """生成默认 FP16 RKNN 输出路径，默认写到输入模型同目录。"""
    p = Path(weights_path)
    stem = p.stem[:-7] if p.stem.endswith("_paddle") else p.stem
    if "_fp32_" in stem:
        stem = stem.split("_fp32_", 1)[0]
    base = stem.split("_paddle_", 1)[0]
    return str((p.parent / f"{base}_paddle_fp16_{format_static_imgsz(imgsz)}.rknn").resolve())


def configure_rknn_e2e_export_mode(model, e2e_mode: str) -> bool:
    if e2e_mode != "one2one_raw":
        return False

    core_model = getattr(model, "model", None)
    if core_model is None or not getattr(core_model, "end2end", False):
        return False

    if hasattr(core_model, "set_head_attr"):
        core_model.set_head_attr(export_raw_one2one=True)
        return True
    return False


def ensure_onnx(weights_path: str, imgsz: StaticInputShape, e2e_mode: str = "postprocess") -> str:
    p = Path(weights_path)
    if p.suffix == ".onnx":
        return str(p)

    print(f"[INFO] 正在自动导出 {p.name} 为 ONNX...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ddyolo26 import YOLO

    model = YOLO(str(p))
    if configure_rknn_e2e_export_mode(model, e2e_mode):
        print("[INFO] RKNN e2e 导出模式: one2one_raw")
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
    print(f"[INFO] ONNX 已导出到 {onnx_path}")
    return onnx_path


def _is_paddle_onnx(onnx_model) -> bool:
    from collections import Counter

    ops = Counter(n.op_type for n in onnx_model.graph.node)
    return ops.get("Expand", 0) >= 2 and ops.get("Floor", 0) >= 1


def prepare_onnx(onnx_path: str, fix_paddle: bool = True) -> str:
    """在进入 RKNN 编译前验证 FP32 图、修正 Paddle ONNX 并转换 opset。"""
    import onnx

    model = onnx.load(onnx_path)
    qdq_ops = {"QuantizeLinear", "DequantizeLinear"}
    present_qdq = sorted({node.op_type for node in model.graph.node} & qdq_ops)
    if present_qdq:
        raise ValueError("RKNN 编译入口只接受 FP32 ONNX，检测到 ORT QDQ 节点: " + ", ".join(present_qdq))
    modified = False

    if fix_paddle and _is_paddle_onnx(model):
        print("[INFO] 检测到 Paddle 导出的 ONNX — 正在应用 NPU 兼容性修复...")
        sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
        from fix_paddle_onnx import fix_paddle_onnx

        model = fix_paddle_onnx(model, verbose=False)
        modified = True
        print("[INFO] Paddle ONNX 修复已应用（Expand→Tile, Floor+Cast→Div+Mod）")

    current_opset = next(
        (item.version for item in model.opset_import if item.domain in ("", "ai.onnx")),
        0,
    )
    if current_opset > 19:
        print(f"[INFO] 正在使用 ONNX version converter 转换 opset {current_opset} → 19")
        try:
            model = onnx.version_converter.convert_version(model, 19)
        except Exception as exc:
            raise ValueError(f"无法可靠转换 ONNX opset {current_opset} → 19；请从源模型以 opset=13 重新导出") from exc
        modified = True

    onnx.checker.check_model(model)

    if modified:
        out_path = str(Path(onnx_path).with_stem(Path(onnx_path).stem + "_rknn_ready"))
        onnx.save(model, out_path)
        return out_path

    return onnx_path


@contextmanager
def isolated_rknn_workspace():
    """在临时工作目录中运行 Toolkit，避免 `check*.onnx` 污染仓库。"""
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="rknn_toolkit_") as workdir:
        os.chdir(workdir)
        try:
            yield Path(workdir)
        finally:
            os.chdir(original)


def collect_calib_images(data_yaml: str, count: int, offset: int = 0) -> list[str]:
    """! 按 RKNN 校准规则解析并等距抽取图片。

    @param data_yaml 数据集 YAML 路径。
    @param count 校准图片数量。
    @param offset 起始偏移量，跳过前 N 张图片，用于避免与评测集重叠。
    @return 按最终校准顺序排列的绝对图片路径。
    """
    import yaml

    if count <= 0:
        raise ValueError("--calib-images 必须大于 0")

    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    yaml_path = Path(data_yaml).resolve()
    data_root = yaml_path.parent
    if "path" in data:
        p = Path(data["path"])
        data_root = p if p.is_absolute() else (yaml_path.parent / p)
    data_root = data_root.resolve()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _resolve_split_entry(entry: str) -> Path:
        p = Path(entry)
        if p.is_absolute():
            return p.resolve()
        candidate = (data_root / p).resolve()
        if candidate.exists():
            return candidate
        if entry.startswith("../"):
            return (data_root / entry[3:]).resolve()
        return candidate

    def _collect_images(entry: str) -> list[str]:
        resolved = _resolve_split_entry(entry)
        if resolved.is_file() and resolved.suffix.lower() == ".txt":
            images: list[str] = []
            with open(resolved, encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    p = Path(line)
                    if p.is_absolute():
                        images.append(str(p.resolve()))
                        continue
                    txt_relative = (resolved.parent / p).resolve()
                    if txt_relative.exists():
                        images.append(str(txt_relative))
                    else:
                        images.append(str((data_root / p).resolve()))
            return images
        if resolved.is_dir():
            return [str(p.resolve()) for p in sorted(resolved.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
        if resolved.is_file() and resolved.suffix.lower() in exts:
            return [str(resolved.resolve())]
        return []

    images: list[str] = []
    for split in ["val", "valid", "train"]:
        split_rel = data.get(split)
        if not split_rel:
            continue
        entries = split_rel if isinstance(split_rel, list) else [split_rel]
        for entry in entries:
            images.extend(_collect_images(str(entry)))
        images = list(dict.fromkeys(images))
        if images:
            break
    if not images:
        raise FileNotFoundError(f"未在 {data_yaml} 中解析到可用校准图片")

    candidates = images[offset:]
    if not candidates:
        raise ValueError(f"校准偏移 {offset} 超出数据集范围（共 {len(images)} 张）")
    sample_count = min(count, len(candidates))
    if sample_count <= 0:
        raise ValueError("--calib-images 必须大于 0")
    if sample_count == 1:
        images = [candidates[0]]
    else:
        indices = [round(i * (len(candidates) - 1) / (sample_count - 1)) for i in range(sample_count)]
        images = [candidates[i] for i in indices]
    print(f"[RKNN-Calib] 代表性等距抽样={len(images)}/{len(candidates)}，offset={offset}")

    return images


def _save_calib_dataset(data_yaml: str, imgsz, count: int, offset: int = 0) -> str:
    """! 收集校准图片并落盘为固定尺寸 RGB letterbox 副本。

    @param data_yaml 数据集 YAML 路径。
    @param imgsz 校准输入尺寸，按 `(height, width)` 语义解析。
    @param count 校准图片数量。
    @param offset 起始偏移量，跳过前 N 张图片，用于避免与评测集重叠。
    @return dataset.txt 路径。
    @note YOLO26 模型权重训练时使用 RGB（Ultralytics 默认 BGR2RGB），所以
    RKNN INT8 校准必须喂 RGB；否则 quantize scale 与权重通道错位。详见
    `tools.eval.backend_utils.make_rgb_calib_dataset` 注释。
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tools.eval.backend_utils import make_rgb_calib_dataset

    images = collect_calib_images(data_yaml, count, offset)
    return make_rgb_calib_dataset(images, imgsz, prefix="rknn_calib_")


def call_rknn_build(rknn, **kwargs):
    """仅使用当前已安装 toolkit 支持的关键字参数调用 RKNN.build。"""
    signature = inspect.signature(rknn.build)
    supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return rknn.build(**supported)


def build_fp16_rknn(
    onnx_path: str,
    output_path: str,
    target: str,
    optimization_level: int = 3,
    compress_weight: bool = False,
    model_pruning: bool = False,
) -> None:
    """使用 RKNN-toolkit2 将指定 ONNX 构建并导出为 FP16 RKNN。"""
    from rknn.api import RKNN

    onnx_path = str(Path(onnx_path).resolve())
    output_path = str(Path(output_path).resolve())
    rknn = RKNN(verbose=True)
    try:
        with isolated_rknn_workspace():
            rknn.config(
                target_platform=target,
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                optimization_level=optimization_level,
                quantized_dtype="asymmetric_quantized-8",
            )
            ret = rknn.load_onnx(model=onnx_path)
            if ret != 0:
                raise RuntimeError(f"load_onnx 失败: {ret}")
            ret = call_rknn_build(
                rknn,
                do_quantization=False,
                compress_weight=compress_weight,
                model_pruning=model_pruning,
            )
            if ret != 0:
                raise RuntimeError(f"build 失败: {ret}")
            ret = rknn.export_rknn(str(Path(output_path).resolve()))
            if ret != 0:
                raise RuntimeError(f"export_rknn 失败: {ret}")
    finally:
        rknn.release()


def main() -> int:
    """保留通用 FP16 RKNN 导出入口，并拒绝已拆分到专用脚本的 INT8 路线。"""
    args = parse_args()
    imgsz = normalize_static_imgsz(args.imgsz)

    if args.quantize != "fp16":
        raise SystemExit(
            "export_rknn.py 不再提供检测 INT8 通用导出入口。"
            "YOLO26 检测请使用 pre_dist 专用导出链路；DFL 模型请使用 pre_dfl 专用导出链路。"
        )

    onnx_path = ensure_onnx(args.weights, imgsz, args.e2e_mode)
    import onnx

    model = onnx.load(onnx_path)
    output_ranks = [len(output.type.tensor_type.shape.dim) for output in model.graph.output]
    if 4 in output_ranks:
        raise SystemExit(
            "通用 FP16 入口不支持分割模型；请使用 export_seg_rknn_i8.py，生成 YOLO26 四输出或 YOLOv8 五输出部署产物。"
        )
    prepared_onnx = prepare_onnx(onnx_path, fix_paddle=not args.no_fix)

    if args.output:
        out_path = args.output
    else:
        out_path = default_output_path(args.weights, imgsz)

    print(f"[INFO] 开始导出 FP16 RKNN: {prepared_onnx} -> {out_path}")
    build_fp16_rknn(
        prepared_onnx,
        out_path,
        target=args.target,
        optimization_level=args.optimization_level,
        compress_weight=args.compress_weight,
        model_pruning=args.model_pruning,
    )
    from export.model_manifest import infer_native_output_route, write_model_manifest

    output_route = infer_native_output_route(prepared_onnx)
    manifest_path = write_model_manifest(out_path, prepared_onnx, output_route, imgsz, data_yaml=args.data)
    print(f"[INFO] RKNN 已导出: {out_path}")
    print(f"[INFO] 模型清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
