#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!@file eval_coco_baselines.py
@brief 评测 COCO baseline 目录下的 ONNX 模型，并可选测量 ONNX INT8 纯 CPU 推理速度。

默认行为：
1. 遍历 `artifacts/coco_baselines/<model>/` 下的 `.onnx` 文件
2. 对 `*_e2e_fp32_640.onnx`、`*_raw_fp32_640.onnx`、route FP32 与 INT8 ONNX 调用
   `tools/eval/cli.py` 跑 COCO 官方评测
3. 对 `*_int8_640.onnx` 调用 `bench_onnx_cpu.py` 测主机 CPU 推理速度
4. 将单模型结果写入 `<model>/_eval/*.json`，另写一份排查用运行汇总 JSON；正式表格直接读取 `_eval/`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES_ROOT = ROOT / "artifacts" / "coco_baselines"
DEFAULT_DATA = ROOT / "ddyolo26" / "cfg" / "datasets" / "coco-val2017-only.yaml"
DEFAULT_SUMMARY = DEFAULT_BASELINES_ROOT / "_host_eval_summary.json"
EVAL_PY = ROOT / "tools" / "eval" / "cli.py"
BENCH_PY = ROOT / "scripts" / "bench_onnx_cpu.py"
COCO_VAL_DIR = ROOT / "artifacts" / "coco" / "images" / "val2017"


def log(msg: str) -> None:
    print(f"[eval-coco] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="评测 COCO baseline ONNX 模型并汇总结果")
    p.add_argument("--root", default=str(DEFAULT_BASELINES_ROOT), help="COCO baseline 根目录")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="COCO data.yaml")
    p.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="排查用运行汇总 JSON 输出路径")
    p.add_argument("--only", default="", help="仅评测某个模型目录，如 yolo26n")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-images", type=int, default=0, help="限制评测图片数，0 表示全量")
    p.add_argument("--bench-runs", type=int, default=100, help="CPU benchmark 正式轮数")
    p.add_argument("--bench-warmup", type=int, default=10, help="CPU benchmark 预热轮数")
    p.add_argument("--bench-threads", type=int, default=0, help="ORT 线程数，0 表示默认")
    p.add_argument("--skip-bench", action="store_true", help="跳过 ONNX INT8 CPU benchmark")
    return p.parse_args()


def select_bench_image() -> str:
    imgs = sorted(COCO_VAL_DIR.glob("*.jpg"))
    return str(imgs[0]) if imgs else ""


def run_and_capture(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)


def parse_eval_payload(eval_json: Path) -> dict:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    if len(payload) != 1:
        raise RuntimeError(f"eval payload 键数量异常: {list(payload)}")
    return next(iter(payload.values()))


def portable_path(path: Path) -> str:
    """!
    @brief 生成可入库的相对路径，避免把本机绝对路径写入汇总 JSON。
    @param path 需要记录的文件路径。
    @return 位于 YOLO26 根目录内时返回相对路径，否则仅返回文件名。
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def evaluate_onnx(model_path: Path, data_yaml: Path, imgsz: int, max_images: int, out_json: Path) -> dict:
    """调用统一 eval 后端评估 ONNX，并读取输出 JSON 中的唯一模型结果。"""
    cmd = [
        sys.executable,
        str(EVAL_PY),
        "--backend",
        "onnx",
        "--model",
        str(model_path),
        "--data",
        str(data_yaml),
        "--imgsz",
        str(imgsz),
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--output",
        str(out_json),
    ]
    if max_images > 0:
        cmd.extend(["--max-images", str(max_images)])
    r = run_and_capture(cmd, ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"eval 失败 rc={r.returncode}\nSTDOUT:\n{r.stdout[-4000:]}\nSTDERR:\n{r.stderr[-4000:]}")
    return parse_eval_payload(out_json)


def bench_onnx(
    model_path: Path,
    imgsz: int,
    runs: int,
    warmup: int,
    threads: int,
    image: str,
    out_json: Path,
) -> dict:
    """调用 CPU benchmark 脚本测量 ONNX 推理耗时，并读取其 JSON 结果。"""
    cmd = [
        sys.executable,
        str(BENCH_PY),
        "--model",
        str(model_path),
        "--imgsz",
        str(imgsz),
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--threads",
        str(threads),
        "--json",
        str(out_json),
    ]
    if image:
        cmd.extend(["--image", image])
    r = run_and_capture(cmd, ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"bench 失败 rc={r.returncode}\nSTDOUT:\n{r.stdout[-4000:]}\nSTDERR:\n{r.stderr[-4000:]}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def discover_model_dirs(root: Path, only: str) -> list[Path]:
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith("_")]
    if only:
        dirs = [p for p in dirs if p.name == only]
    return dirs


def discover_onnx_files(model_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in model_dir.glob("*.onnx")
        if ("_e2e_fp32_" in p.name)
        or ("_raw_fp32_" in p.name)
        or ("_predist_fp32_" in p.name)
        or ("_predfl_fp32_" in p.name)
        or ("_int8_" in p.name)
    )


def main() -> int:
    """评估 COCO baseline 目录下的 ONNX 模型，并可选对 INT8 ONNX 跑 CPU benchmark。"""
    args = parse_args()
    root = Path(args.root).resolve()
    data_yaml = Path(args.data).resolve()
    summary_path = Path(args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise SystemExit(f"未找到 baseline 根目录: {root}")
    if not data_yaml.exists():
        raise SystemExit(f"未找到 data.yaml: {data_yaml}")

    bench_image = select_bench_image()
    summary: dict[str, list[dict]] = {}

    for model_dir in discover_model_dirs(root, args.only):
        log(f"=== {model_dir.name} ===")
        eval_dir = model_dir / "_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for model_path in discover_onnx_files(model_dir):
            row: dict = {
                "model_dir": model_dir.name,
                "file": model_path.name,
                "path": portable_path(model_path),
            }
            eval_json = eval_dir / f"{model_path.stem}.eval.json"
            try:
                metrics = evaluate_onnx(model_path, data_yaml, args.imgsz, args.max_images, eval_json)
                row["metrics"] = metrics
                row["eval_json"] = portable_path(eval_json)
            except Exception as e:
                row["error"] = str(e)
                rows.append(row)
                log(f"评测失败: {model_path.name}: {e}")
                continue

            if (not args.skip_bench) and ("_int8_" in model_path.name):
                bench_json = eval_dir / f"{model_path.stem}.cpu.json"
                try:
                    bench = bench_onnx(
                        model_path,
                        args.imgsz,
                        args.bench_runs,
                        args.bench_warmup,
                        args.bench_threads,
                        bench_image,
                        bench_json,
                    )
                    bench["model"] = portable_path(model_path)
                    row["cpu_bench"] = bench
                    row["cpu_bench_json"] = portable_path(bench_json)
                except Exception as e:
                    row["cpu_bench_error"] = str(e)
                    log(f"CPU benchmark 失败: {model_path.name}: {e}")

            rows.append(row)
        summary[model_dir.name] = rows

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"汇总已写入: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
