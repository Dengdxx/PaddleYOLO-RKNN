#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""!@file prepare_coco_baselines.py
@brief 准备 COCO val-only 数据目录与 COCO 基线产物目录。

功能：
1. 准备 `PaddleYOLO-RKNN/artifacts/coco/` 目录，必要时下载并整理：
   - `images/val2017/`
   - `labels/val2017/`（segment labels，可供 detect/segment 量化校准）
   - `annotations/instances_val2017.json`
2. 准备 `PaddleYOLO-RKNN/artifacts/coco_baselines/<model>/` 目录。
3. 输出可直接使用的 data.yaml 路径。

脚本只写入 `artifacts/coco/` 与 `artifacts/coco_baselines/`。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
COCO_ROOT = ARTIFACTS / "coco"
BASELINES_ROOT = ARTIFACTS / "coco_baselines"
DATA_YAML = ROOT / "ddyolo26" / "cfg" / "datasets" / "coco-val2017-only.yaml"
TARGET_MODELS = ["yolo26n", "yolov8n", "yolo26n-seg", "yolov8n-seg"]

VAL_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
ANNO_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
LABELS_SEG_ZIP = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip"


def log(msg: str) -> None:
    print(f"[prepare-coco] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"复用已下载文件: {dest}")
        return
    ensure_dir(dest.parent)
    log(f"下载 {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def extract_member(zip_path: Path, member: str, dest: Path) -> None:
    if dest.exists():
        log(f"复用已解压文件: {dest}")
        return
    ensure_dir(dest.parent)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as src, open(dest, "wb") as f:
            shutil.copyfileobj(src, f)
    log(f"已解压 {member} -> {dest}")


def extract_tree(zip_path: Path, prefix: str, dest_root: Path, *, strip_prefix: str | None = None) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(prefix) and not m.endswith("/")]
        if not members:
            raise RuntimeError(f"zip 中未找到前缀 {prefix}: {zip_path}")
        for member in members:
            rel = member[len(strip_prefix) :] if strip_prefix else member
            out = dest_root / rel
            if out.exists():
                continue
            ensure_dir(out.parent)
            with zf.open(member) as src, open(out, "wb") as f:
                shutil.copyfileobj(src, f)
    log(f"已解压树 {prefix} -> {dest_root}")


def prepare_dataset(download_enabled: bool) -> None:
    """准备 COCO val2017 图片、YOLO labels 与官方 annotations。"""
    ensure_dir(ARTIFACTS)
    ensure_dir(COCO_ROOT)
    ensure_dir(COCO_ROOT / "images")
    ensure_dir(COCO_ROOT / "labels")
    ensure_dir(COCO_ROOT / "annotations")

    needed = [
        COCO_ROOT / "images" / "val2017",
        COCO_ROOT / "labels" / "val2017",
        COCO_ROOT / "annotations" / "instances_val2017.json",
    ]
    if all(p.exists() for p in needed):
        log("COCO val-only 数据已就绪")
        return

    if not download_enabled:
        missing = [str(p) for p in needed if not p.exists()]
        raise RuntimeError("缺少 COCO 数据，且指定了 --no-download: " + ", ".join(missing))

    cache_dir = ARTIFACTS / "_downloads"
    ensure_dir(cache_dir)
    val_zip = cache_dir / "val2017.zip"
    anno_zip = cache_dir / "annotations_trainval2017.zip"
    labels_zip = cache_dir / "coco2017labels-segments.zip"

    download(VAL_ZIP, val_zip)
    download(ANNO_ZIP, anno_zip)
    download(LABELS_SEG_ZIP, labels_zip)

    # val images: zip 根即 val2017/
    if not (COCO_ROOT / "images" / "val2017").exists():
        with zipfile.ZipFile(val_zip) as zf:
            zf.extractall(COCO_ROOT / "images")
        log(f"已解压 val2017 -> {COCO_ROOT / 'images'}")

    # labels zip 内通常为 coco/labels/**
    if not (COCO_ROOT / "labels" / "val2017").exists():
        extract_tree(labels_zip, "coco/labels/val2017/", COCO_ROOT, strip_prefix="coco/")

    # 官方 annotations zip 内为 annotations/instances_val2017.json
    extract_member(
        anno_zip,
        "annotations/instances_val2017.json",
        COCO_ROOT / "annotations" / "instances_val2017.json",
    )


def prepare_model_dirs() -> None:
    ensure_dir(BASELINES_ROOT)
    for name in TARGET_MODELS:
        d = BASELINES_ROOT / name
        ensure_dir(d)
        ensure_dir(d / "_eval")
    log(f"已准备基线产物目录: {BASELINES_ROOT}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="准备 COCO val-only 数据与基线产物目录")
    p.add_argument("--no-download", action="store_true", help="若数据缺失则直接报错，不自动下载")
    return p.parse_args()


def main() -> int:
    """准备 COCO baseline 所需的数据目录与模型产物目录。"""
    args = parse_args()
    prepare_dataset(download_enabled=not args.no_download)
    prepare_model_dirs()
    log(f"data.yaml: {DATA_YAML}")
    log("可用路径：")
    log(f"  data.yaml: {DATA_YAML}")
    log(f"  产物目录: {BASELINES_ROOT}/<model>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
