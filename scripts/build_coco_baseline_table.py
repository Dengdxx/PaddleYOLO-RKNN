#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file build_coco_baseline_table.py
@brief 从 artifacts/coco_baselines/{model}/_eval/ 汇聚评测结果，生成汇总表与 Markdown。

@details
扫描规则（无需 _host_eval_summary.json 中间文件）：
- `_eval/*.eval.json`       → host ONNX 结果（由 eval_coco_baselines.py 或 tools.eval 直接写入）
- `_eval/*.rknn.eval.json`  → 板端 RKNN 结果（由板端 tools.eval 写入后 scp 回 host）
- `_eval/*.bench_core*.json` → 板端 RKNN 纯 NPU / 端到端测速结果
- `_eval/*.cpu.json`        → host CPU benchmark（由 bench_onnx_cpu.py 写入）
  仅 backend=onnx 的模型会关联 cpu bench。

tools.eval 输出格式（单 key dict）：
    { "{stem}.{ext}": { "mAP50": ..., "mAP50-95": ..., "P": ..., "R": ..., ... } }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES_ROOT = ROOT / "artifacts" / "coco_baselines"
OUT_JSON = BASELINES_ROOT / "_host_eval_table.json"
OUT_MD = BASELINES_ROOT / "_host_eval_table.md"
UNVERIFIED_RKNN_BENCH_STATUS = "RKNN测速环境未确认"
RKNN_BENCH_PROFILE = "RKNN 2.3.2 + CPU/NPU/DDR max"


# ─────────────────────────────────────────────────────────────────────────────
# 解析工具
# ─────────────────────────────────────────────────────────────────────────────


def parse_stem(stem: str) -> tuple[str, str, str]:
    """!
    @brief 从文件名 stem 中解析 framework / route / precision。

    @details
    命名规范：`<model>_<fw>_<route>_<prec>_<imgsz>`，例如：
    - `yolo26n_pt_predist_int8_640`         → (pt, predist, int8)
    - `yolo26n_paddle_e2e_fp32_640`         → (paddle, e2e, fp32)
    - `yolov8n_paddle_raw_fp32_640`         → (paddle, raw, fp32)
    - `yolo26n-seg_pt_seg_predist_int8_640` → (pt, seg_predist, int8)

    @param stem 不含扩展名的文件名。
    @return (framework, route, precision)，解析失败时返回 '?'。
    """
    fw = route = prec = "?"
    toks = stem.split("_")
    for i, t in enumerate(toks):
        if t in ("pt", "paddle"):
            fw = t
            rest = toks[i + 1 : -1]  # 去掉末尾的 imgsz 数字
            if rest and rest[-1] in ("fp32", "int8"):
                prec = rest[-1]
                route = "_".join(rest[:-1])
            break
    return fw, route, prec


def fmt(x: object, p: int = 4) -> str:
    """!
    @brief 将数值格式化为固定小数位字符串；非数值返回 "—"。
    @param x 待格式化值。
    @param p 小数位数。
    @return 格式化后的字符串。
    """
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


def fmt_ms(x: object, p: int = 1) -> str:
    """!
    @brief 将毫秒耗时格式化为固定小数位字符串；非数值返回 "—"。
    @param x 待格式化值。
    @param p 小数位数。
    @return 格式化后的字符串。
    """
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


# ─────────────────────────────────────────────────────────────────────────────
# 数据收集
# ─────────────────────────────────────────────────────────────────────────────


def _load_cpu_bench(eval_dir: Path, stem: str) -> dict:
    """!
    @brief 加载 stem 对应的 CPU benchmark JSON（若存在）。
    @param eval_dir _eval/ 目录路径。
    @param stem 不含扩展名的文件名 stem。
    @return bench dict 或空 dict。
    """
    cpu_json = eval_dir / f"{stem}.cpu.json"
    if cpu_json.exists():
        try:
            return json.loads(cpu_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _is_profiled_rknn_bench(meta: dict) -> bool:
    """!
    @brief 判断 bench JSON 是否声明了 RKNN 2.3.2 + 锁频环境。

    @details
    旧 JSON 没有环境元数据，不能确认测速口径；可通过 `bench_status=official`
    或 `environment_valid=true` 明确声明。

    @param meta 单个 bench JSON 的内容。
    @return 若测速口径可确认则返回 True。
    """
    if meta.get("bench_status") == "official" or meta.get("environment_valid") is True:
        return True
    runtime = str(meta.get("rknn_api_version") or meta.get("rknn_runtime_version") or "")
    freq = str(meta.get("frequency_profile") or meta.get("freq_profile") or "").lower()
    return "2.3.2" in runtime and freq in {"max", "locked-max", "cpu_npu_ddr_max"}


def _load_rknn_bench(eval_dir: Path, stem: str) -> dict:
    """!
    @brief 加载 stem 对应的板端 RKNN bench JSON（若存在）。
    @param eval_dir _eval/ 目录路径。
    @param stem 不含扩展名的文件名 stem。
    @return 标准化后的 bench 字段 dict。
    """
    result = {
        "rknn_core0_npu_ms": None,
        "rknn_core0_pp_ms": None,
        "rknn_core0_e2e_ms": None,
        "rknn_coreall_npu_ms": None,
        "rknn_coreall_pp_ms": None,
        "rknn_coreall_e2e_ms": None,
        "rknn_bench_status": None,
    }
    loaded: list[dict] = []
    for core, prefix in (("0", "rknn_core0"), ("all", "rknn_coreall")):
        path = eval_dir / f"{stem}.bench_core{core}.json"
        if not path.exists():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        loaded.append(meta)
        result[f"{prefix}_npu_ms"] = meta.get("npu_pure_ms")
        result[f"{prefix}_pp_ms"] = meta.get("postproc_ms")
        result[f"{prefix}_e2e_ms"] = meta.get("e2e_avg_ms")

    if loaded:
        explicit = next(
            (m.get("bench_status") or m.get("status") for m in loaded if m.get("bench_status") or m.get("status")), None
        )
        if explicit == "official":
            result["rknn_bench_status"] = RKNN_BENCH_PROFILE
        elif explicit:
            result["rknn_bench_status"] = explicit
        elif all(_is_profiled_rknn_bench(m) for m in loaded):
            result["rknn_bench_status"] = RKNN_BENCH_PROFILE
        else:
            result["rknn_bench_status"] = UNVERIFIED_RKNN_BENCH_STATUS
    return result


def _extract_metrics(m: dict, is_seg: bool) -> dict:
    """!
    @brief 从 tools.eval 输出 dict 中提取关键指标。
    @param m tools.eval 输出的 metrics dict。
    @param is_seg 是否为分割模型。
    @return 标准化后的指标 dict。
    """
    if is_seg:
        return {
            "P": m.get("BoxP") or m.get("P"),
            "R": m.get("BoxR") or m.get("R"),
            "mAP50": m.get("BoxmAP50") or m.get("mAP50"),
            "mAP5095": m.get("BoxmAP50-95") or m.get("mAP50-95"),
            "maskmAP50": m.get("MaskmAP50"),
            "maskmAP5095": m.get("MaskmAP50-95"),
        }
    return {
        "P": m.get("P"),
        "R": m.get("R"),
        "mAP50": m.get("mAP50"),
        "mAP5095": m.get("mAP50-95"),
        "maskmAP50": None,
        "maskmAP5095": None,
    }


def collect_from_model_dir(model_dir: Path) -> list[dict]:
    """!
    @brief 扫描单个模型目录的 _eval/ 子目录，收集所有评测结果行。
    @param model_dir 模型目录路径，如 artifacts/coco_baselines/yolo26n。
    @return 该模型下所有可解析结果的行列表。
    """
    eval_dir = model_dir / "_eval"
    if not eval_dir.is_dir():
        return []

    model_name = model_dir.name
    is_seg = "seg" in model_name
    rows: list[dict] = []
    seen: set[str] = set()  # 按 key（含后缀）去重，防止同结果写入两个文件时重复

    # *.rknn.eval.json 是板端 RKNN 结果；*.eval.json（不含 .rknn.）是 host ONNX 结果。
    # glob("*.eval.json") 会同时命中两者，需过滤。
    onnx_jsons = [p for p in eval_dir.glob("*.eval.json") if not p.name.endswith(".rknn.eval.json")]
    rknn_jsons = list(eval_dir.glob("*.rknn.eval.json"))
    for ej in sorted(onnx_jsons + rknn_jsons):
        try:
            payload = json.loads(ej.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[警告] 无法解析 {ej.name}: {e}", file=sys.stderr)
            continue

        for key, m in payload.items():
            if "error" in m:
                continue
            if key in seen:
                continue
            seen.add(key)

            # 从 key 推断 backend 与 stem
            if key.endswith(".rknn"):
                backend = "rknn"
                stem = key[:-5]
            elif key.endswith(".onnx"):
                backend = "onnx"
                stem = key[:-5]
            else:
                # 回退：从文件名判断
                backend = "rknn" if ".rknn." in ej.name else "onnx"
                stem = key.rsplit(".", 1)[0]

            fw, route, prec = parse_stem(stem)

            cpu = _load_cpu_bench(eval_dir, stem) if backend == "onnx" else {}
            rknn_bench = _load_rknn_bench(eval_dir, stem) if backend == "rknn" else {}
            metrics = _extract_metrics(m, is_seg)

            rows.append(
                {
                    "model": model_name,
                    "framework": fw,
                    "route": route,
                    "precision": prec,
                    "backend": backend,
                    "file": key,
                    **metrics,
                    "cpu_total_ms": cpu.get("total_ms"),
                    "cpu_fps": cpu.get("fps_total"),
                    "n_images": m.get("n_images"),
                    "time_s": m.get("time_s"),
                    **rknn_bench,
                }
            )

    return rows


def collect(baselines_root: Path) -> list[dict]:
    """!
    @brief 遍历 baselines_root 下所有模型目录，汇聚结果并排序。
    @param baselines_root coco_baselines 根目录。
    @return 全量结果行列表，已按 model / precision / backend / framework 排序。
    """
    rows: list[dict] = []
    for d in sorted(baselines_root.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            rows.extend(collect_from_model_dir(d))

    rows.sort(
        key=lambda r: (
            r["model"],
            0 if r["precision"] == "fp32" else 1,
            0 if r["backend"] == "onnx" else 1,
            r["framework"],
        )
    )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 输出
# ─────────────────────────────────────────────────────────────────────────────


def print_table(rows: list[dict]) -> None:
    """!
    @brief 在终端打印对齐的汇总表格。
    @param rows collect() 返回的行列表。
    """
    hdr = (
        f"{'model':20s} {'fw':6s} {'route':14s} {'prec':4s} {'be':5s} "
        f"{'mAP50':>7s} {'mAP5095':>8s} {'mskmAP50':>9s} "
        f"{'npu0':>7s} {'e2e0':>7s} {'npuall':>7s} "
        f"{'e2eall':>7s} {'imgs':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['model']:20s} {r['framework']:6s} {r['route']:14s} "
            f"{r['precision']:4s} {r['backend']:5s} "
            f"{fmt(r['mAP50']):>7s} {fmt(r['mAP5095']):>8s} "
            f"{fmt(r['maskmAP50']):>9s} "
            f"{fmt_ms(r.get('rknn_core0_npu_ms')):>7s} "
            f"{fmt_ms(r.get('rknn_core0_e2e_ms')):>7s} "
            f"{fmt_ms(r.get('rknn_coreall_npu_ms')):>7s} "
            f"{fmt_ms(r.get('rknn_coreall_e2e_ms')):>7s} "
            f"{str(r['n_images'] or '—'):>5}"
        )


def to_markdown(rows: list[dict]) -> str:
    """!
    @brief 按模型分组生成 Markdown 表格（适合写入 docs）。
    @param rows collect() 返回的行列表。
    @return Markdown 字符串。
    """
    out: list[str] = []
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    for model, mrows in by_model.items():
        is_seg = "seg" in model
        out.append(f"### `{model}`")
        out.append("")
        if is_seg:
            out.append(
                "| framework | 链路 | 精度 | 后端 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | RKNN core0 NPU (ms) | RKNN core0 E2E (ms) | RKNN coreall NPU (ms) | RKNN coreall E2E (ms) | RKNN测速状态 |"
            )
            out.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        else:
            out.append(
                "| framework | 链路 | 精度 | 后端 | mAP50 | mAP50-95 | RKNN core0 NPU (ms) | RKNN core0 E2E (ms) | RKNN coreall NPU (ms) | RKNN coreall E2E (ms) | RKNN测速状态 |"
            )
            out.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")

        for r in mrows:
            npu0 = fmt_ms(r.get("rknn_core0_npu_ms"))
            e2e0 = fmt_ms(r.get("rknn_core0_e2e_ms"))
            npuall = fmt_ms(r.get("rknn_coreall_npu_ms"))
            e2eall = fmt_ms(r.get("rknn_coreall_e2e_ms"))
            bench_status = r.get("rknn_bench_status") or "—"
            if is_seg:
                out.append(
                    f"| {r['framework']} | {r['route']} | {r['precision']} | {r['backend']} | "
                    f"{fmt(r['mAP50'])} | {fmt(r['mAP5095'])} | "
                    f"{fmt(r['maskmAP50'])} | {fmt(r['maskmAP5095'])} | "
                    f"{npu0} | {e2e0} | {npuall} | {e2eall} | {bench_status} |"
                )
            else:
                out.append(
                    f"| {r['framework']} | {r['route']} | {r['precision']} | {r['backend']} | "
                    f"{fmt(r['mAP50'])} | {fmt(r['mAP5095'])} | "
                    f"{npu0} | {e2e0} | {npuall} | {e2eall} | {bench_status} |"
                )
        out.append("")

    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    """!
    @brief 解析命令行参数。
    @return argparse.Namespace 对象。
    """
    import argparse

    p = argparse.ArgumentParser(description="汇聚 COCO baseline 评测结果并输出表格")
    p.add_argument("--root", default=str(BASELINES_ROOT), help="coco_baselines 根目录")
    p.add_argument("--markdown", action="store_true", help="同时输出 Markdown 文件")
    p.add_argument("--out-json", default=str(OUT_JSON), help="JSON 输出路径")
    p.add_argument("--out-md", default=str(OUT_MD), help="Markdown 输出路径")
    return p.parse_args()


def main() -> int:
    """!
    @brief 脚本主入口。
    @return 成功返回 0。
    """
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[错误] 目录不存在: {root}", file=sys.stderr)
        return 1

    rows = collect(root)
    if not rows:
        print(
            "[警告] 未找到任何评测结果。请先运行 eval_coco_baselines.py 或板端 python -m tools.eval。", file=sys.stderr
        )
        return 1

    print_table(rows)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    if args.markdown:
        md = to_markdown(rows)
        out_md = Path(args.out_md)
        out_md.write_text(md, encoding="utf-8")
        print(f"Markdown → {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
