#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""!@file export_coco_baselines.py
@brief 导出标准 COCO baseline 所需的 ONNX / RKNN 产物。

默认目标：
- yolo26n
- yolov8n
- yolo26n-seg
- yolov8n-seg

说明：
- 检测模型的 RKNN 导出不复用 `export_all_models.py` 默认策略，而是显式指定量化算法：
  - yolo26n(pre_dist): mmse + auto_hybrid
  - yolov8n(pre_dfl): normal
- 分割模型 `yolo26n-seg` / `yolov8n-seg` 使用 `normal` 算法导出 RKNN INT8。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_BASELINES_ROOT = ROOT / "artifacts" / "coco_baselines"
DEFAULT_DATA = ROOT / "ddyolo26" / "cfg" / "datasets" / "coco-val2017-only.yaml"
EXPORT_ALL_PY = ROOT / "scripts" / "export_all_models.py"
EXPORT_DET_RKNN_PY = ROOT / "export" / "export_det_rknn_i8.py"
EXPORT_SEG_RKNN_PY = ROOT / "export" / "export_seg_rknn_i8.py"

MODEL_CONFIGS = {
    "yolo26n": {
        "weights_name": "yolo26n",
        "route": "predist",
        "task": "detect",
        "rknn_algorithm": "mmse",
        "auto_hybrid": True,
    },
    "yolov8n": {
        "weights_name": "yolov8n",
        "route": "predfl",
        "task": "detect",
        "rknn_algorithm": "normal",
        "auto_hybrid": False,
    },
    "yolo26n-seg": {
        "weights_name": "yolo26n-seg",
        "route": "seg_predist",
        "task": "segment",
        "rknn_algorithm": "normal",
        "auto_hybrid": False,
    },
    "yolov8n-seg": {
        "weights_name": "yolov8n-seg",
        "route": "seg_predfl",
        "task": "segment",
        "rknn_algorithm": "normal",
        "auto_hybrid": False,
    },
}


def log(msg: str) -> None:
    print(f"[export-coco] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出 COCO baseline ONNX/RKNN 产物")
    p.add_argument("--root", default=str(DEFAULT_BASELINES_ROOT), help="baseline 根目录")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="COCO data.yaml")
    p.add_argument("--only", default="", help="仅处理某个模型目录")
    p.add_argument("--python-paddle", default="", help="paddle/ddyolo26 路线使用的 Python 可执行文件")
    p.add_argument("--python-rknn", default="", help="RKNN Toolkit 路线使用的 Python 可执行文件")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--calib-images", type=int, default=50)
    p.add_argument("--skip-rknn", action="store_true", help="仅导出 ONNX，跳过 RKNN")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def file_sha256(path: Path) -> str:
    """计算输入文件 SHA-256，作为导出缓存来源指纹。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline_sha256() -> str:
    """计算 COCO 入口及共享导出流水线的完整代码指纹。"""
    from scripts.export_all_models import pipeline_sha256 as shared_pipeline_sha256

    digest = hashlib.sha256()
    digest.update(file_sha256(Path(__file__)).encode("ascii"))
    digest.update(shared_pipeline_sha256().encode("ascii"))
    return digest.hexdigest()


def calibration_sha256(data_yaml: Path, calib_images: int, calib_offset: int) -> str:
    """哈希实际选中的 RKNN 校准图片及数据集描述。"""
    from export.export_rknn import collect_calib_images

    digest = hashlib.sha256()
    digest.update(file_sha256(data_yaml).encode("ascii"))
    for image_path in collect_calib_images(str(data_yaml), calib_images, offset=calib_offset):
        path = Path(image_path)
        digest.update(str(path).encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def rknn_environment_identity(python_exe: str) -> dict[str, str]:
    """读取 RKNN 解释器和 Toolkit 版本，避免跨环境误复用缓存。"""
    code = (
        "import importlib.metadata,json,platform; "
        "\ntry:\n v=importlib.metadata.version('rknn-toolkit2')\nexcept importlib.metadata.PackageNotFoundError:\n v='missing'\n"
        "print(json.dumps({'python':platform.python_version(),'rknn_toolkit2':v}))"
    )
    result = subprocess.run(
        [python_exe, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def rknn_provenance(
    weights: Path,
    data_yaml: Path,
    imgsz: int,
    calib_images: int,
    calib_offset: int,
    algorithm: str,
    route: str,
    python_exe: str,
    auto_hybrid: bool | None = None,
) -> dict:
    """构造覆盖模型、校准输入、环境和完整代码的 RKNN 缓存指纹。"""
    provenance = {
        "schema": 2,
        "weights_sha256": file_sha256(weights),
        "calibration_sha256": calibration_sha256(data_yaml, calib_images, calib_offset),
        "imgsz": imgsz,
        "calib_images": calib_images,
        "calib_offset": calib_offset,
        "algorithm": algorithm,
        "route": route,
        "python": str(Path(python_exe).resolve()),
        "environment": rknn_environment_identity(python_exe),
        "pipeline_sha256": pipeline_sha256(),
    }
    if auto_hybrid is not None:
        provenance["auto_hybrid"] = auto_hybrid
    return provenance


def provenance_matches(output: Path, inputs: dict) -> bool:
    """仅在产物及其来源参数指纹完全一致时复用缓存。"""
    metadata = output.with_suffix(output.suffix + ".json")
    if not output.exists() or not metadata.exists():
        return False
    try:
        return json.loads(metadata.read_text(encoding="utf-8")) == inputs
    except (OSError, ValueError):
        return False


def write_provenance(output: Path, inputs: dict) -> None:
    """原子写入产物来源参数指纹。"""
    metadata = output.with_suffix(output.suffix + ".json")
    temporary = metadata.with_suffix(metadata.suffix + ".tmp")
    temporary.write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata)


def model_dir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "_eval").mkdir(parents=True, exist_ok=True)
    return path


def ensure_weights(root: Path, name: str, weights_name: str) -> Path:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ddyolo26.utils.downloads import paddle_weight_group

    out_dir = model_dir(root, name)
    paddle_path = out_dir / f"{weights_name}.pdparams"
    legacy_paddle_path = out_dir / f"{weights_name}_paddle.pt"
    shared_group = paddle_weight_group(weights_name)
    shared_paddle_path = ROOT / "weights" / shared_group / f"{weights_name}.pdparams"
    shared_legacy_path = ROOT / "weights" / shared_group / f"{weights_name}_paddle.pt"
    if paddle_path.exists():
        return paddle_path
    if legacy_paddle_path.exists():
        return legacy_paddle_path
    if shared_paddle_path.exists():
        return shared_paddle_path
    if shared_legacy_path.exists():
        return shared_legacy_path
    raise RuntimeError(f"缺少 {weights_name} 的本地 Paddle 权重；期望位置为 {paddle_path} 或 {shared_paddle_path}")


def export_onnx_bundle(
    weights_path: Path,
    out_dir: Path,
    data_yaml: Path,
    imgsz: int,
    paddle_python: str,
    task: str,
    route: str,
    calib_images: int,
) -> None:
    """调用批量导出脚本，一次生成该 baseline 需要的 ONNX 产物。"""
    run(
        [
            paddle_python,
            str(EXPORT_ALL_PY),
            "--weights",
            str(weights_path),
            "--out",
            str(out_dir),
            "--data",
            str(data_yaml),
            "--imgsz",
            str(imgsz),
            "--task",
            task,
            "--route",
            route,
            "--calib-images",
            str(calib_images),
            "--python-paddle",
            paddle_python,
            "--steps",
            "onnx_paddle,int8onnx_paddle,fp32predist_paddle" if task == "detect" else "onnx_paddle,int8onnx_paddle",
        ]
    )


def export_det_rknn(
    weights: Path,
    out_path: Path,
    data_yaml: Path,
    imgsz: int,
    calib_images: int,
    algorithm: str,
    auto_hybrid: bool,
    python_exe: str,
    route: str,
    calib_offset: int = 0,
) -> None:
    """调用检测 RKNN INT8 专用导出脚本；仅复用来源指纹一致的产物。"""
    provenance = rknn_provenance(
        weights,
        data_yaml,
        imgsz,
        calib_images,
        calib_offset,
        algorithm,
        route,
        python_exe,
        auto_hybrid=auto_hybrid,
    )
    if provenance_matches(out_path, provenance):
        log(f"复用来源一致的 RKNN: {out_path.name}")
        return
    out_path.unlink(missing_ok=True)
    cmd = [
        python_exe,
        str(EXPORT_DET_RKNN_PY),
        "--weights",
        str(weights),
        "--route",
        str(provenance["route"]),
        "--data",
        str(data_yaml),
        "--imgsz",
        str(imgsz),
        "--calib-images",
        str(calib_images),
        "--calib-offset",
        str(calib_offset),
        "--algorithm",
        algorithm,
        "--output",
        str(out_path),
    ]
    if auto_hybrid:
        cmd.append("--auto-hybrid")
    run(cmd)
    write_provenance(out_path, provenance)


def export_seg_rknn(
    weights: Path,
    out_path: Path,
    data_yaml: Path,
    imgsz: int,
    calib_images: int,
    algorithm: str,
    python_exe: str,
    route: str,
    calib_offset: int = 0,
) -> None:
    """调用分割 RKNN INT8 专用导出脚本；仅复用来源指纹一致的产物。"""
    provenance = rknn_provenance(
        weights,
        data_yaml,
        imgsz,
        calib_images,
        calib_offset,
        algorithm,
        route,
        python_exe,
    )
    if provenance_matches(out_path, provenance):
        log(f"复用来源一致的 RKNN: {out_path.name}")
        return
    out_path.unlink(missing_ok=True)
    run(
        [
            python_exe,
            str(EXPORT_SEG_RKNN_PY),
            "--weights",
            str(weights),
            "--route",
            route,
            "--data",
            str(data_yaml),
            "--imgsz",
            str(imgsz),
            "--calib-images",
            str(calib_images),
            "--calib-offset",
            str(calib_offset),
            "--algorithm",
            algorithm,
            "--output",
            str(out_path),
        ]
    )
    write_provenance(out_path, provenance)


def main() -> int:
    """准备指定 COCO baseline 模型的 ONNX 与 RKNN 产物。"""
    args = parse_args()
    root = Path(args.root).resolve()
    data_yaml = Path(args.data).resolve()
    paddle_python = str(Path(args.python_paddle).resolve()) if args.python_paddle else sys.executable
    rknn_python = str(Path(args.python_rknn).resolve()) if args.python_rknn else sys.executable
    if not data_yaml.exists():
        raise SystemExit(f"未找到 data.yaml: {data_yaml}")
    root.mkdir(parents=True, exist_ok=True)

    names = [args.only] if args.only else list(MODEL_CONFIGS)
    for name in names:
        if name not in MODEL_CONFIGS:
            raise SystemExit(f"不支持的模型: {name}")
        cfg = MODEL_CONFIGS[name]
        out_dir = model_dir(root, name)
        paddle_path = ensure_weights(root, name, cfg["weights_name"])

        export_onnx_bundle(
            paddle_path,
            out_dir,
            data_yaml,
            args.imgsz,
            paddle_python,
            cfg["task"],
            cfg["route"],
            args.calib_images,
        )
        if args.skip_rknn:
            continue

        if cfg["task"] == "detect":
            export_det_rknn(
                paddle_path,
                out_dir / f"{cfg['weights_name']}_paddle_{cfg['route']}_int8_{args.imgsz}.rknn",
                data_yaml,
                args.imgsz,
                args.calib_images,
                cfg["rknn_algorithm"],
                bool(cfg["auto_hybrid"]),
                rknn_python,
                cfg["route"],
                calib_offset=1000,
            )
        else:
            export_seg_rknn(
                paddle_path,
                out_dir / f"{cfg['weights_name']}_paddle_{cfg['route']}_int8_{args.imgsz}.rknn",
                data_yaml,
                args.imgsz,
                args.calib_images,
                cfg["rknn_algorithm"],
                rknn_python,
                cfg["route"],
                calib_offset=1000,
            )

    log("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
