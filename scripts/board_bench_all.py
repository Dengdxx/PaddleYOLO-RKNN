#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""! 在板卡上跑 bench_rknn_perf 对全部 RKNN 模型测纯 NPU + 端到端。

用法（板卡上执行）::

    python3 scripts/board_bench_all.py \\
        --bench ./bench/bench_rknn_perf \\
        --models-root artifacts/coco_baselines \\
        --summary-out _bench_summary.json

也可通过环境变量提供默认值：

    BENCH_BIN              bench_rknn_perf 可执行路径
    BENCH_MODELS_ROOT      模型产物目录根
    BENCH_SUMMARY          summary 输出 JSON 路径
    BENCH_STATUS           写入每个 bench JSON 的状态，如 official
    BENCH_FREQ_PROFILE     写入每个 bench JSON 的频率口径，如 cpu_npu_ddr_max
    BENCH_SRAM             RKNN SRAM 策略：off / private / shared
    BENCH_SCORE_SUM        分割五输出预筛：on / off
    BENCH_MASK_VERIFY      mask 像素计数/hash 校验：on / off
    BENCH_IOU_THR          NMS IoU 阈值
    BENCH_MASK_CLASS_IDS   生成 mask 的类别：all 或逗号分隔 ID
    BENCH_MASK_OUTPUT_SIZE mask 输出坐标尺寸：WxH
    BENCH_INPUT            固定的 HWC RGB uint8 原始帧（正式 E2E 必填）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WARMUP, RUNS = 10, 200


def infer_postproc_from_path(path: str | Path) -> str:
    """!
    @brief 从 RKNN 文件名推断 bench 后处理路线。
    @param path RKNN 模型路径。
    @return `predist` / `predfl` / `seg_predist` / `seg_predfl`。
    @throw ValueError 当文件名不包含已知 route 时抛出。
    """
    name = Path(path).name
    if "seg_predfl" in name:
        return "seg_predfl"
    if "seg_predist" in name:
        return "seg_predist"
    if "predfl" in name:
        return "predfl"
    if "predist" in name:
        return "predist"
    raise ValueError(f"无法从文件名推断后处理路线: {name}")


def discover_rknn_jobs(models_root: str | Path) -> list[tuple[str, str]]:
    """!
    @brief 自动发现 RKNN 模型并推断后处理路线。

    @details
    `models_root` 既可指向总目录（`<root>/<model>/*.rknn`），也可直接
    指向单个模型目录（`<model>/*.rknn`）。

    @param models_root 模型目录根。
    @return `(rknn_path, postproc)` 列表。
    """
    root = Path(models_root)
    jobs: list[tuple[str, str]] = []
    candidates = sorted(
        {
            *root.glob("*.rknn"),
            *root.glob("*/*.rknn"),
            *root.glob("*/_eval/*.rknn"),
        }
    )
    for path in candidates:
        try:
            postproc = infer_postproc_from_path(path)
        except ValueError as exc:
            print(f"[跳过] {exc}")
            continue
        jobs.append((str(path), postproc))
    return jobs


def run(cmd: list[str]) -> str:
    """!@brief 执行子进程；非零退出时抛出 SystemExit。"""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"命令执行失败: {' '.join(cmd)}")
    return r.stdout


def bench_output_dir_for_model(path: str | Path) -> Path:
    """!
    @brief 返回单个 RKNN 模型对应的 bench JSON 输出目录。

    @details
    COCO baseline 的 RKNN 产物通常位于 `<models-root>/<model>/*.rknn`，
    而汇总脚本扫描 `<model>/_eval/*.bench_core*.json`。若模型本身已在
    `_eval/` 下，则保持原目录；否则统一写入兄弟 `_eval/`。

    @param path RKNN 模型路径。
    @return bench JSON 输出目录。
    """
    parent = Path(path).parent
    return parent if parent.name == "_eval" else parent / "_eval"


def normalize_bench_meta(meta: dict) -> dict:
    """!
    @brief 兼容不同版本 bench_rknn_perf 的 JSON 字段名。
    """
    if "npu_pure_ms" not in meta:
        if "npu_pure_avg_ms" in meta:
            meta["npu_pure_ms"] = meta["npu_pure_avg_ms"]
        elif "npu_wall_avg_ms" in meta:
            meta.setdefault("io_wall_avg_ms", meta["npu_wall_avg_ms"])
    return meta


def parse_args() -> argparse.Namespace:
    """!@brief 解析 CLI；所有路径参数支持环境变量回退。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--bench",
        default=os.environ.get("BENCH_BIN", "./bench/bench_rknn_perf"),
        help="bench_rknn_perf 可执行路径 (env: BENCH_BIN)",
    )
    ap.add_argument(
        "--models-root",
        default=os.environ.get("BENCH_MODELS_ROOT", str(Path.cwd() / "artifacts" / "coco_baselines")),
        help="模型产物目录根 (env: BENCH_MODELS_ROOT)",
    )
    ap.add_argument(
        "--summary-out",
        default=os.environ.get(
            "BENCH_SUMMARY",
            str(Path.cwd() / "_bench_summary.json"),
        ),
        help="汇总 JSON 输出路径 (env: BENCH_SUMMARY)",
    )
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument(
        "--bench-status",
        default=os.environ.get("BENCH_STATUS", ""),
        help="写入 bench JSON 的状态标记，如 official (env: BENCH_STATUS)",
    )
    ap.add_argument(
        "--frequency-profile",
        default=os.environ.get("BENCH_FREQ_PROFILE", ""),
        help="写入 bench JSON 的频率口径，如 cpu_npu_ddr_max (env: BENCH_FREQ_PROFILE)",
    )
    ap.add_argument(
        "--sram",
        choices=("off", "private", "shared"),
        default=os.environ.get("BENCH_SRAM", "off"),
        help="传给 bench_rknn_perf 的 RKNN SRAM 策略 (env: BENCH_SRAM)",
    )
    ap.add_argument(
        "--score-sum",
        choices=("on", "off"),
        default=os.environ.get("BENCH_SCORE_SUM", "on"),
        help="分割五输出 score_sum 预筛策略 (env: BENCH_SCORE_SUM)",
    )
    ap.add_argument(
        "--input",
        default=os.environ.get("BENCH_INPUT", ""),
        help="固定的 HWC RGB uint8 原始帧，用于可复现 E2E 后处理 (env: BENCH_INPUT)",
    )
    ap.add_argument(
        "--mask-verify",
        choices=("on", "off"),
        default=os.environ.get("BENCH_MASK_VERIFY", "off"),
        help="mask 像素计数/hash 校验，不计入正式 E2E (env: BENCH_MASK_VERIFY)",
    )
    ap.add_argument(
        "--iou-thr",
        type=float,
        default=float(os.environ.get("BENCH_IOU_THR", "0.45")),
        help="NMS IoU 阈值 (env: BENCH_IOU_THR, 默认 0.45)",
    )
    ap.add_argument(
        "--mask-class-ids",
        default=os.environ.get("BENCH_MASK_CLASS_IDS", "0"),
        help="生成 mask 的类别：all 或逗号分隔 ID (env: BENCH_MASK_CLASS_IDS)",
    )
    ap.add_argument(
        "--mask-output-size",
        default=os.environ.get("BENCH_MASK_OUTPUT_SIZE", "640x480"),
        help="mask 输出坐标尺寸 WxH (env: BENCH_MASK_OUTPUT_SIZE)",
    )
    return ap.parse_args()


def main() -> None:
    """遍历产物目录中的 RKNN 文件，调用板端 benchmark 并汇总 NPU/E2E 延迟。"""
    args = parse_args()
    bench = os.path.abspath(args.bench)
    models_root = os.path.abspath(args.models_root)
    if not os.path.exists(bench):
        raise SystemExit(f"未找到 bench 可执行文件: {bench}")
    if not os.path.isdir(models_root):
        raise SystemExit(f"未找到模型产物根目录: {models_root}")
    input_path = os.path.abspath(args.input) if args.input else ""
    if input_path and not os.path.isfile(input_path):
        raise SystemExit(f"未找到 bench 输入帧: {input_path}")
    if args.bench_status == "official" and not input_path:
        raise SystemExit("official bench 必须通过 --input 或 BENCH_INPUT 指定真实输入帧")

    rows: list[dict] = []
    for path, pp in discover_rknn_jobs(models_root):
        stem = os.path.splitext(os.path.basename(path))[0]
        parent = Path(path).parent
        model_dir = parent.parent.name if parent.name == "_eval" else parent.name
        for core in ("0", "all"):
            bench_dir = bench_output_dir_for_model(path)
            bench_dir.mkdir(parents=True, exist_ok=True)
            jp = str(bench_dir / f"{stem}.bench_core{core}.json")
            cmd = [
                bench,
                "--model",
                path,
                "--warmup",
                str(args.warmup),
                "--runs",
                str(args.runs),
                "--core",
                core,
                "--postproc",
                pp,
                "--sram",
                args.sram,
                "--score-sum",
                args.score_sum,
                "--mask-verify",
                args.mask_verify,
                "--iou-thr",
                str(args.iou_thr),
                "--mask-class-ids",
                args.mask_class_ids,
                "--mask-output-size",
                args.mask_output_size,
                "--json",
                jp,
            ]
            if input_path:
                cmd.extend(["--input", input_path])
            print(
                f"[运行] {stem} core={core} postproc={pp} sram={args.sram} "
                f"score_sum={args.score_sum} iou={args.iou_thr} "
                f"mask_classes={args.mask_class_ids} "
                f"mask_size={args.mask_output_size}"
            )
            run(cmd)
            with open(jp) as f:
                meta = json.load(f)
            meta = normalize_bench_meta(meta)
            if args.bench_status:
                meta["bench_status"] = args.bench_status
            if args.frequency_profile:
                meta["frequency_profile"] = args.frequency_profile
            meta.update({"stem": stem, "model_dir": model_dir})
            if input_path:
                meta["input_frame"] = input_path
            with open(jp, "w") as f:
                json.dump(meta, f, indent=2)
            rows.append(meta)

    print("\n=== 汇总 ===")
    print(f"{'model':<55}{'core':<6}{'npu_ms':>9}{'pp_ms':>9}{'sync_KiB':>10}{'e2e_ms':>9}{'fps':>8}{'outcome':>28}")
    for r in rows:
        fps = 1000.0 / r["e2e_avg_ms"]
        npu_ms = f"{r['npu_pure_ms']:.3f}" if "npu_pure_ms" in r else "n/a"
        sync_kib = float(r.get("native_sync_bytes", 0.0)) / 1024.0
        outcome = str(r.get("fetch_outcome", "-"))
        print(
            f"{r['stem']:<55}{r['core']:<6}{npu_ms:>9}"
            f"{r['postproc_ms']:>9.3f}{sync_kib:>10.1f}"
            f"{r['e2e_avg_ms']:>9.3f}{fps:>8.1f}{outcome:>28}"
        )

    summary_path = os.path.abspath(args.summary_out)
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n保存 {summary_path}")


if __name__ == "__main__":
    main()
