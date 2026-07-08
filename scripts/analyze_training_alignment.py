#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""分析两份训练结果 CSV，并输出学习率与指标对齐报告。"""

import argparse
import csv
import json
from pathlib import Path


def read_results_csv(path: Path) -> list[dict]:
    """读取 YOLO results.csv。"""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def to_float(row: dict, key: str) -> float:
    """安全读取浮点字段。"""
    value = row.get(key, 0)
    return float(value) if value not in (None, "") else 0.0


def best_metric_row(rows: list[dict], key: str) -> dict:
    """返回指定指标最优的 epoch 行。"""
    return max(rows, key=lambda row: to_float(row, key))


def main():
    """解析两份 results.csv，比较 LR 曲线、最终指标与最佳指标是否对齐。"""
    parser = argparse.ArgumentParser(description="分析训练对齐结果")
    parser.add_argument("lhs", help="左侧 results.csv")
    parser.add_argument("rhs", help="右侧 results.csv")
    parser.add_argument("--lhs-name", default="lhs", help="左侧标签")
    parser.add_argument("--rhs-name", default="rhs", help="右侧标签")
    parser.add_argument("--metric-threshold", type=float, default=0.05, help="mAP50 差值阈值")
    parser.add_argument("--lr-threshold", type=float, default=1e-6, help="LR 差值阈值")
    parser.add_argument("--out", default="", help="可选 JSON 输出路径")
    args = parser.parse_args()

    lhs_rows = read_results_csv(Path(args.lhs))
    rhs_rows = read_results_csv(Path(args.rhs))
    n = min(len(lhs_rows), len(rhs_rows))
    if n == 0:
        raise SystemExit("results.csv 为空，无法分析")

    print("=" * 72)
    print("训练对齐分析")
    print("=" * 72)
    print(f"左侧: {args.lhs_name} -> {args.lhs}")
    print(f"右侧: {args.rhs_name} -> {args.rhs}")
    print(f"对比 epoch 数: {n}")

    lr_rows = []
    max_lr_diff = 0.0
    print("\n--- LR 调度 ---")
    print(f"{'epoch':>5} | {args.lhs_name:>10} | {args.rhs_name:>10} | {'差值':>10}")
    print("-" * 48)
    for i in range(n):
        epoch = int(lhs_rows[i].get("epoch", i + 1))
        lhs_lr = to_float(lhs_rows[i], "lr/pg0")
        rhs_lr = to_float(rhs_rows[i], "lr/pg0")
        diff = abs(lhs_lr - rhs_lr)
        max_lr_diff = max(max_lr_diff, diff)
        lr_rows.append({"epoch": epoch, args.lhs_name: lhs_lr, args.rhs_name: rhs_lr, "diff": diff})
        print(f"{epoch:5d} | {lhs_lr:10.8f} | {rhs_lr:10.8f} | {diff:10.2e}")

    final_lhs = lhs_rows[n - 1]
    final_rhs = rhs_rows[n - 1]
    box_lhs = to_float(final_lhs, "metrics/mAP50(B)")
    box_rhs = to_float(final_rhs, "metrics/mAP50(B)")
    mask_lhs = to_float(final_lhs, "metrics/mAP50(M)")
    mask_rhs = to_float(final_rhs, "metrics/mAP50(M)")
    best_box_lhs_row = best_metric_row(lhs_rows[:n], "metrics/mAP50(B)")
    best_box_rhs_row = best_metric_row(rhs_rows[:n], "metrics/mAP50(B)")
    best_mask_lhs_row = best_metric_row(lhs_rows[:n], "metrics/mAP50(M)")
    best_mask_rhs_row = best_metric_row(rhs_rows[:n], "metrics/mAP50(M)")
    best_box_lhs = to_float(best_box_lhs_row, "metrics/mAP50(B)")
    best_box_rhs = to_float(best_box_rhs_row, "metrics/mAP50(B)")
    best_mask_lhs = to_float(best_mask_lhs_row, "metrics/mAP50(M)")
    best_mask_rhs = to_float(best_mask_rhs_row, "metrics/mAP50(M)")
    box_diff = abs(box_lhs - box_rhs)
    mask_diff = abs(mask_lhs - mask_rhs)
    best_box_diff = abs(best_box_lhs - best_box_rhs)
    best_mask_diff = abs(best_mask_lhs - best_mask_rhs)

    lr_pass = max_lr_diff <= args.lr_threshold
    box_pass = box_diff <= args.metric_threshold
    mask_pass = mask_diff <= args.metric_threshold
    all_pass = lr_pass and box_pass and mask_pass

    print("\n--- 最终指标 ---")
    print(f"box  mAP50: {args.lhs_name}={box_lhs:.4f}, {args.rhs_name}={box_rhs:.4f}, diff={box_diff:.4f}")
    print(f"mask mAP50: {args.lhs_name}={mask_lhs:.4f}, {args.rhs_name}={mask_rhs:.4f}, diff={mask_diff:.4f}")
    print("\n--- 最优指标 ---")
    print(
        f"best box  mAP50: {args.lhs_name}={best_box_lhs:.4f} @ epoch {best_box_lhs_row.get('epoch')}, "
        f"{args.rhs_name}={best_box_rhs:.4f} @ epoch {best_box_rhs_row.get('epoch')}, diff={best_box_diff:.4f}"
    )
    print(
        f"best mask mAP50: {args.lhs_name}={best_mask_lhs:.4f} @ epoch {best_mask_lhs_row.get('epoch')}, "
        f"{args.rhs_name}={best_mask_rhs:.4f} @ epoch {best_mask_rhs_row.get('epoch')}, diff={best_mask_diff:.4f}"
    )
    print("\n--- 结论 ---")
    print(f"LR 对齐:   {'通过' if lr_pass else '失败'} (max_diff={max_lr_diff:.2e})")
    print(f"Box 对齐:  {'通过' if box_pass else '失败'} (diff={box_diff:.4f})")
    print(f"Mask 对齐: {'通过' if mask_pass else '失败'} (diff={mask_diff:.4f})")
    print(f"总体:      {'通过' if all_pass else '失败'}")

    report = {
        "lhs": args.lhs,
        "rhs": args.rhs,
        "lhs_name": args.lhs_name,
        "rhs_name": args.rhs_name,
        "epochs": n,
        "lr_alignment": {
            "threshold": args.lr_threshold,
            "max_diff": max_lr_diff,
            "passed": lr_pass,
            "per_epoch": lr_rows,
        },
        "final_metrics": {
            "box_mAP50": {args.lhs_name: box_lhs, args.rhs_name: box_rhs, "diff": box_diff, "passed": box_pass},
            "mask_mAP50": {args.lhs_name: mask_lhs, args.rhs_name: mask_rhs, "diff": mask_diff, "passed": mask_pass},
        },
        "best_metrics": {
            "box_mAP50": {
                args.lhs_name: best_box_lhs,
                f"{args.lhs_name}_epoch": int(best_box_lhs_row.get("epoch", 0)),
                args.rhs_name: best_box_rhs,
                f"{args.rhs_name}_epoch": int(best_box_rhs_row.get("epoch", 0)),
                "diff": best_box_diff,
            },
            "mask_mAP50": {
                args.lhs_name: best_mask_lhs,
                f"{args.lhs_name}_epoch": int(best_mask_lhs_row.get("epoch", 0)),
                args.rhs_name: best_mask_rhs,
                f"{args.rhs_name}_epoch": int(best_mask_rhs_row.get("epoch", 0)),
                "diff": best_mask_diff,
            },
        },
        "passed": all_pass,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存至: {out_path}")


if __name__ == "__main__":
    main()
