#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""!
@file export_all_models.py
@brief 批量为 Paddle 权重导出 ONNX (FP32 / INT8) 与 RKNN (INT8)。

@details
本脚本把 ONNX / RKNN 产物写入 `--out` 指定目录；未指定时写到输入权重同目录。
统一命名规范为：

    <stem>_<framework>_<route>_<precision>_<imgsz>.<ext>

  - framework:  paddle
    - route:      predist | predfl | seg_predist | seg_predfl
  - fp32_route:  e2e（YOLO26 predist/seg_predist）| raw（YOLOv8 predfl/seg_predfl）
  - precision:  fp32 | int8
  - ext:        onnx | rknn

输入必须是 Paddle 权重（`.pdparams`）。导出过程：

  1. 用 ddyolo26 导出 FP32 ONNX（YOLO26 → e2e，YOLOv8 → raw）。
  2. 用 quant.quantize / 自带 ORT 静态量化 生成 INT8 ONNX（detection 主线 = predist/predfl，
      seg 主线 = seg_predist/seg_predfl）。
  3. 用 export_det_rknn_i8 / export_seg_rknn_i8 生成 RKNN INT8。

所有产物平铺写入目标目录，原 Paddle 权重不动。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(f"[EXPORT-ALL] {msg}", flush=True)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def resolve_python(path: str | None) -> str:
    """!
    @brief 解析用户指定的 Python 可执行文件路径。
    @param path 用户传入路径；为空时回退到当前解释器。
    @return 可执行文件绝对路径字符串。
    """
    return str(Path(path).resolve()) if path else sys.executable


def resolve_task_and_route(
    task_arg: str,
    route_arg: str,
) -> tuple[str, str]:
    """!
    @brief 使用显式参数解析 task/route，不读取权重做框架探测。
    @param task_arg 用户显式指定的 task，可为空。
    @param route_arg 用户显式指定的 route，可为空。
    @return `(task, route)`。
    """
    valid_routes = {
        "detect": {"predist", "predfl"},
        "segment": {"seg_predist", "seg_predfl"},
    }
    if task_arg and route_arg:
        if route_arg not in valid_routes[task_arg]:
            raise ValueError(f"task/route 组合无效: task={task_arg}, route={route_arg}")
        return task_arg, route_arg
    if route_arg:
        return ("segment", route_arg) if route_arg.startswith("seg_") else ("detect", route_arg)
    raise ValueError("Paddle-only 导出必须显式指定 --task/--route，至少需要 --route。")


# ─────────────────────────────────────────────────────────────────────────────
# 步骤
# ─────────────────────────────────────────────────────────────────────────────


def step_fp32_onnx(
    weights_path: Path,
    framework: str,
    imgsz: int,
    out_dir: Path,
    base_stem: str,
    python_exe: str,
    route: str = "",
) -> Path:
    """!
    @brief 导出 FP32 ONNX，重命名到统一规范。

    @param weights_path 源 Paddle 权重路径。
    @param framework 输出文件名中的框架标签，当前固定为 'paddle'。
    @param imgsz 导出输入分辨率。
    @param out_dir 输出目录。
    @param base_stem 模型基础文件名 stem。
    @param python_exe 执行导出脚本的 Python 解释器路径。
    @param route INT8 主线 route（predist/predfl/seg_predist/seg_predfl），
                 用于判断文件名中的 FP32 route 标记：
                 - predist / seg_predist → `e2e`（YOLO26 one2one，truly NMS-free）
                 - predfl / seg_predfl  → `raw`（YOLOv8 one2many，需 NMS）
                 - 空串时回退默认行为（`e2e`，兼容）
    @return 导出的 FP32 ONNX 文件路径。
    """
    fp32_label = fp32_label_for_route(route)
    target = out_dir / f"{base_stem}_{framework}_{fp32_label}_fp32_{imgsz}.onnx"
    if target.exists():
        log(f"FP32 ONNX 已存在，跳过: {target.name}")
        return target
    cmd = [
        python_exe,
        str(ROOT / "export" / "export_fp32_onnx.py"),
        "--weights",
        str(weights_path),
        "--imgsz",
        str(imgsz),
        "--output",
        str(target),
    ]
    run(cmd, cwd=ROOT)
    log(f"FP32 ONNX → {target.name}")
    return target


def fp32_label_for_route(route: str) -> str:
    """!
    @brief 根据 INT8 主线 route 推导普通 FP32 ONNX 文件名标签。
    """
    return "e2e" if "predist" in route else ("raw" if "predfl" in route else "e2e")


def step_int8_onnx_det(
    src_weights: Path,
    framework: str,
    imgsz: int,
    data_yaml: Path,
    out_dir: Path,
    base_stem: str,
    route: str,
    python_exe: str,
    calib_images: int,
) -> Path:
    """!
    @brief detection 模型走 quant.quantize.py 生成 INT8 ONNX（pre_dist / pre_dfl）。
    """
    target = out_dir / f"{base_stem}_{framework}_{route}_int8_{imgsz}.onnx"
    if target.exists():
        log(f"INT8 ONNX 已存在，跳过: {target.name}")
        return target
    cmd = [
        python_exe,
        str(ROOT / "quant" / "quantize.py"),
        "--mode",
        "onnx",
        "--weights",
        str(src_weights),
        "--data",
        str(data_yaml),
        "--imgsz",
        str(imgsz),
        "--output",
        str(target),
        "--calib-batches",
        str(calib_images),
    ]
    run(cmd, cwd=ROOT)
    if not target.exists():
        raise RuntimeError(f"INT8 ONNX 缺失: {target}")
    return target


def step_int8_onnx_seg(
    src_weights: Path,
    framework: str,
    imgsz: int,
    data_yaml: Path,
    out_dir: Path,
    base_stem: str,
    route: str,
    python_exe: str,
    calib_images: int,
) -> Path:
    """!
    @brief segmentation 模型生成 INT8 ONNX (seg_pre_dist/seg_pre_dfl 4 输出 + ORT QDQ 静态量化)。
    """
    target = out_dir / f"{base_stem}_{framework}_{route}_int8_{imgsz}.onnx"
    if target.exists():
        log(f"INT8 SEG ONNX 已存在，跳过: {target.name}")
        return target
    cmd = [
        python_exe,
        str(ROOT / "export" / "export_seg_onnx_i8.py"),
        "--weights",
        str(src_weights),
        "--data",
        str(data_yaml),
        "--imgsz",
        str(imgsz),
        "--output",
        str(target),
        "--calib-batches",
        str(calib_images),
    ]
    run(cmd, cwd=ROOT)
    log(f"INT8 SEG ONNX → {target.name}")
    return target


def step_seg_fp32_route_onnx(
    src_weights: Path,
    framework: str,
    imgsz: int,
    data_yaml: Path,
    out_dir: Path,
    base_stem: str,
    route: str,
    python_exe: str,
) -> Path:
    """!
    @brief 为 Seg RKNN 编译准备量化前的 route FP32 ONNX。
    """
    target = out_dir / f"{base_stem}_{framework}_{route}_fp32_{imgsz}.onnx"
    if target.exists():
        log(f"FP32 SEG route ONNX 已存在，跳过: {target.name}")
        return target
    cmd = [
        python_exe,
        str(ROOT / "export" / "export_seg_onnx_i8.py"),
        "--weights",
        str(src_weights),
        "--data",
        str(data_yaml),
        "--imgsz",
        str(imgsz),
        "--prepared-output",
        str(target),
        "--skip-quant",
    ]
    run(cmd, cwd=ROOT)
    if not target.exists():
        raise RuntimeError(f"FP32 SEG route ONNX 缺失: {target}")
    log(f"FP32 SEG route ONNX → {target.name}")
    return target


def step_fp32predist_onnx_det(
    weights_path: Path,
    framework: str,
    imgsz: int,
    out_dir: Path,
    base_stem: str,
    route: str,
    python_exe: str,
) -> Path:
    """!
    @brief 导出 pre_dist/pre_dfl FP32 ONNX，用于与 INT8 做受控对比。
    @param weights_path 输入 Paddle 权重路径（pdparams / onnx）。
    @param framework 输出文件名中的框架标签，当前固定为 'paddle'。
    @param imgsz 导出 ONNX 时的输入尺寸。
    @param out_dir 输出目录。
    @param base_stem 输出文件名前缀（通常为权重文件 stem）。
    @param route 检测主线路由名，`predist` 或 `predfl`。
    @param python_exe 调用时使用的 Python 可执行文件路径。
    @return 输出 ONNX 文件路径。
    """
    target = out_dir / f"{base_stem}_{framework}_{route}_fp32_{imgsz}.onnx"
    if target.exists():
        log(f"{route} FP32 ONNX 已存在，跳过: {target.name}")
        return target
    cmd = [
        python_exe,
        str(ROOT / "export" / "export_predist_fp32_onnx.py"),
        "--weights",
        str(weights_path),
        "--imgsz",
        str(imgsz),
        "--output",
        str(target),
    ]
    run(cmd, cwd=ROOT)
    log(f"{route} FP32 ONNX → {target.name}")
    return target


def step_rknn_int8(
    src_weights: Path,
    framework: str,
    imgsz: int,
    data_yaml: Path,
    out_dir: Path,
    base_stem: str,
    route: str,
    task: str,
    python_exe: str,
    calib_images: int,
) -> Path:
    """!
    @brief 调用 export_det_rknn_i8 / export_seg_rknn_i8 生成 RKNN INT8。
    """
    target = out_dir / f"{base_stem}_{framework}_{route}_int8_{imgsz}.rknn"
    if target.exists():
        log(f"RKNN INT8 已存在，跳过: {target.name}")
        return target
    if task == "segment":
        cmd = [
            python_exe,
            str(ROOT / "export" / "export_seg_rknn_i8.py"),
            "--weights",
            str(src_weights),
            "--data",
            str(data_yaml),
            "--imgsz",
            str(imgsz),
            "--output",
            str(target),
            "--calib-images",
            str(calib_images),
        ]
    else:
        cmd = [
            python_exe,
            str(ROOT / "export" / "export_det_rknn_i8.py"),
            "--weights",
            str(src_weights),
            "--data",
            str(data_yaml),
            "--imgsz",
            str(imgsz),
            "--mode",
            "int8",
            "--output",
            str(target),
            "--calib-images",
            str(calib_images),
        ]
    run(cmd, cwd=ROOT)
    if not target.exists():
        raise RuntimeError(f"RKNN 缺失: {target}")
    return target


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────


def base_stem_for_weights(weights_path: Path) -> str:
    """返回输出文件 stem，并规范化兼容路径里的 `_paddle` 后缀。"""
    stem = weights_path.stem
    return stem[:-7] if stem.endswith("_paddle") else stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量导出 Paddle YOLO 模型的 ONNX/RKNN")
    p.add_argument("--weights", required=True, help="输入 Paddle 权重路径（.pdparams）")
    p.add_argument("--out", default="", help="输出目录；默认与输入权重同目录")
    p.add_argument("--data", required=True, help="校准/评测 data.yaml")
    p.add_argument("--python-paddle", default="", help="paddle/ddyolo26 路线使用的 Python 可执行文件")
    p.add_argument("--python-rknn", default="", help="RKNN 编译路线使用的 Python 可执行文件")
    p.add_argument("--task", default="", choices=["detect", "segment"], help="显式指定任务类型；不填则由 route 推断")
    p.add_argument(
        "--route",
        required=True,
        choices=["predist", "predfl", "seg_predist", "seg_predfl"],
        help="显式指定导出 route",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--calib-images",
        type=int,
        default=50,
        help="ONNX/RKNN INT8 共用的校准规模；RKNN 解释为图片数，ONNX 解释为校准 batch 数",
    )
    p.add_argument(
        "--steps",
        default="onnx_paddle,int8onnx_paddle,rknn_paddle",
        help="逗号分隔，控制要执行的步骤子集",
    )
    return p.parse_args()


def main() -> int:
    """按任务和 route 编排 Paddle 权重到 ONNX/INT8 ONNX/RKNN 的批量导出流程。"""
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    out_dir = Path(args.out).resolve() if args.out else weights_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = Path(args.data).resolve()
    imgsz = args.imgsz
    base_stem = base_stem_for_weights(weights_path)
    paddle_python = resolve_python(args.python_paddle)
    rknn_python = resolve_python(args.python_rknn)

    if weights_path.suffix == ".pt" and not weights_path.stem.endswith("_paddle"):
        raise ValueError(f"普通 .pt 权重不支持: {weights_path}. 请使用 Paddle 权重。")
    task, route = resolve_task_and_route(args.task, args.route)
    log(f"模型 {weights_path.name}  task={task}  route={route}  imgsz={imgsz}")

    steps = set(s.strip() for s in args.steps.split(",") if s.strip())
    unsupported_steps = {s for s in steps if s.endswith("_pt") or s == "paddle"}
    if unsupported_steps:
        raise ValueError("Paddle-only 导出不支持这些步骤: " + ", ".join(sorted(unsupported_steps)))

    if "onnx_paddle" in steps:
        step_fp32_onnx(weights_path, "paddle", imgsz, out_dir, base_stem, paddle_python, route)

    if task == "detect":
        if "int8onnx_paddle" in steps:
            step_int8_onnx_det(
                weights_path,
                "paddle",
                imgsz,
                data_yaml,
                out_dir,
                base_stem,
                route,
                paddle_python,
                args.calib_images,
            )
        if "fp32predist_paddle" in steps:
            step_fp32predist_onnx_det(weights_path, "paddle", imgsz, out_dir, base_stem, route, paddle_python)
    else:
        if "int8onnx_paddle" in steps:
            step_int8_onnx_seg(
                weights_path,
                "paddle",
                imgsz,
                data_yaml,
                out_dir,
                base_stem,
                route,
                paddle_python,
                args.calib_images,
            )

    if "rknn_paddle" in steps:
        if task == "segment":
            rknn_paddle_src = step_seg_fp32_route_onnx(
                weights_path, "paddle", imgsz, data_yaml, out_dir, base_stem, route, paddle_python
            )
        else:
            rknn_paddle_src = step_fp32predist_onnx_det(
                weights_path, "paddle", imgsz, out_dir, base_stem, route, paddle_python
            )
        step_rknn_int8(
            rknn_paddle_src,
            "paddle",
            imgsz,
            data_yaml,
            out_dir,
            base_stem,
            route,
            task,
            rknn_python,
            args.calib_images,
        )

    log(f"DONE  {weights_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
