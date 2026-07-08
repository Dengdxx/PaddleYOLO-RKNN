#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""比较两份 ONNX 模型在同一输入下的输出数值差异。"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_input(imgsz: int, seed: int) -> np.ndarray:
    """生成确定性输入张量。"""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, (1, 3, imgsz, imgsz)).astype(np.float32)


def compute_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """计算两组输出的误差指标。"""
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    diff = np.abs(a64 - b64)
    af = a64.ravel()
    bf = b64.ravel()
    cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    return {
        "shape": list(a.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine_similarity": cos,
        "lhs_range": [float(a.min()), float(a.max())],
        "rhs_range": [float(b.min()), float(b.max())],
    }


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个 xyxy box 的 IoU。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compare_detection_outputs(lhs: np.ndarray, rhs: np.ndarray, top_k: int = 50) -> dict:
    """对 e2e detection 输出做排序/匹配比较。

    默认假设格式为 [B, N, 6+nm]，其中第 5 列是 confidence。
    """
    lhs_order = np.argsort(-lhs[0, :, 4])
    rhs_order = np.argsort(-rhs[0, :, 4])
    lhs_sorted = lhs[:, lhs_order, :]
    rhs_sorted = rhs[:, rhs_order, :]

    lhs_topk = lhs_sorted[:, :top_k, :]
    rhs_topk = rhs_sorted[:, :top_k, :]
    sorted_metrics = compute_diff(lhs_topk, rhs_topk)

    matched_lhs = []
    matched_rhs = []
    used_rhs = set()
    lhs_boxes = lhs_topk[0, :, :4]
    rhs_boxes = rhs_topk[0, :, :4]
    for i in range(top_k):
        best_iou = 0.0
        best_j = -1
        for j in range(top_k):
            if j in used_rhs:
                continue
            iou = box_iou(lhs_boxes[i], rhs_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= 0.5 and best_j >= 0:
            used_rhs.add(best_j)
            matched_lhs.append(lhs_topk[0, i])
            matched_rhs.append(rhs_topk[0, best_j])

    if matched_lhs:
        matched_metrics = compute_diff(np.stack(matched_lhs)[None, ...], np.stack(matched_rhs)[None, ...])
        matched_metrics["num_matched"] = len(matched_lhs)
    else:
        matched_metrics = {"error": "未匹配到 detections", "num_matched": 0}

    return {
        "topk_sorted": sorted_metrics,
        "iou_matched": matched_metrics,
    }


def run_model(path: Path, x: np.ndarray) -> list[np.ndarray]:
    """运行单个 ONNX 模型并返回全部输出。"""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return [out.astype(np.float32) for out in sess.run(None, {input_name: x})]


def main():
    """运行两份 ONNX 模型并输出逐输出张量的数值差异报告。"""
    parser = argparse.ArgumentParser(description="比较两份 ONNX 模型输出")
    parser.add_argument("lhs", help="左侧 ONNX 路径")
    parser.add_argument("rhs", help="右侧 ONNX 路径")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out", default="", help="可选 JSON 输出路径")
    args = parser.parse_args()

    lhs = Path(args.lhs)
    rhs = Path(args.rhs)
    x = make_input(args.imgsz, args.seed)

    lhs_out = run_model(lhs, x)
    rhs_out = run_model(rhs, x)

    print(f"左侧: {lhs}")
    print(f"右侧: {rhs}")
    print(f"输入: shape={x.shape}, range=[{x.min():.6f}, {x.max():.6f}]")

    report = {
        "lhs": str(lhs),
        "rhs": str(rhs),
        "imgsz": args.imgsz,
        "seed": args.seed,
        "results": {},
    }

    for i, (a, b) in enumerate(zip(lhs_out, rhs_out)):
        print(f"\n--- output{i} ---")
        print(f"  左侧 shape: {a.shape}")
        print(f"  右侧 shape: {b.shape}")
        if a.shape != b.shape:
            report["results"][f"output{i}"] = {"error": f"shape 不匹配: {a.shape} vs {b.shape}"}
            print("  shape 不匹配")
            continue
        metrics = compute_diff(a, b)
        report["results"][f"output{i}"] = metrics
        print(f"  max_abs_diff:      {metrics['max_abs_diff']:.8e}")
        print(f"  mean_abs_diff:     {metrics['mean_abs_diff']:.8e}")
        print(f"  cosine_similarity: {metrics['cosine_similarity']:.10f}")
        print(f"  左侧 range:        [{metrics['lhs_range'][0]:.6f}, {metrics['lhs_range'][1]:.6f}]")
        print(f"  右侧 range:        [{metrics['rhs_range'][0]:.6f}, {metrics['rhs_range'][1]:.6f}]")

        if a.ndim == 3 and a.shape[0] == 1 and a.shape[1] <= 300 and a.shape[2] >= 6:
            detection_metrics = compare_detection_outputs(a, b)
            report["results"][f"output{i}"]["detection_metrics"] = detection_metrics
            sorted_metrics = detection_metrics["topk_sorted"]
            print("  topk 排序后对比:")
            print(f"    max_abs_diff:      {sorted_metrics['max_abs_diff']:.8e}")
            print(f"    mean_abs_diff:     {sorted_metrics['mean_abs_diff']:.8e}")
            print(f"    cosine_similarity: {sorted_metrics['cosine_similarity']:.10f}")
            matched_metrics = detection_metrics["iou_matched"]
            if "error" not in matched_metrics:
                print("  IoU 匹配后对比:")
                print(f"    匹配数量:          {matched_metrics['num_matched']}")
                print(f"    max_abs_diff:      {matched_metrics['max_abs_diff']:.8e}")
                print(f"    mean_abs_diff:     {matched_metrics['mean_abs_diff']:.8e}")
                print(f"    cosine_similarity: {matched_metrics['cosine_similarity']:.10f}")
            else:
                print(f"  IoU 匹配后对比: {matched_metrics['error']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存至: {out_path}")


if __name__ == "__main__":
    main()
