#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!@file bench_onnx_cpu.py
@brief 使用 ONNX Runtime CPUExecutionProvider 测量 ONNX 模型的纯 CPU 推理速度。

默认统计：
- preprocess_ms：HWC uint8 -> NCHW float32 / 255 预处理
- inference_ms：ORT session.run() 推理
- total_ms：预处理 + 推理
- fps_total：按 total_ms 计算的 FPS

说明：
- 该脚本不包含 decode / NMS / tools.eval 后处理，仅测主机 CPU 侧 ONNX 推理链路。
- 对 QDQ INT8 ONNX，这可以较稳定地反映 ORT CPU 推理吞吐。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def portable_model_name(model_path: str) -> str:
    """!
    @brief 生成可入库的模型标识，避免把本机绝对路径写入 JSON。
    @param model_path 用户传入的模型路径。
    @return 若模型位于 YOLO26 根目录下，返回相对路径；否则返回文件名。
    """
    resolved = Path(model_path).resolve()
    try:
        return resolved.relative_to(_ROOT).as_posix()
    except ValueError:
        return resolved.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在纯 CPU 上测试 ONNX 性能")
    p.add_argument("--model", required=True, help="ONNX 模型路径")
    p.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    p.add_argument("--image", default="", help="可选真实图像路径；默认使用随机 uint8 图像")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--threads", type=int, default=0, help="ORT intra/inter op 线程数；0=默认")
    p.add_argument("--json", default="", help="可选 JSON 输出路径")
    return p.parse_args()


def make_session(model_path: str, threads: int) -> object:
    import onnxruntime as ort  # noqa: PLC0415

    opts = ort.SessionOptions()
    if threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = threads
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


def load_image(image_path: str, imgsz: int) -> np.ndarray:
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"读取图像失败: {image_path}")
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        return img
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(imgsz, imgsz, 3), dtype=np.uint8)


def main() -> int:
    """测量单个 ONNX 模型在 CPUExecutionProvider 上的预处理与推理耗时。"""
    args = parse_args()
    import numpy as np  # noqa: PLC0415

    from tools.eval.backend_utils import prepare_onnx_input  # noqa: PLC0415

    sess = make_session(args.model, args.threads)
    input_name = sess.get_inputs()[0].name
    img = load_image(args.image, args.imgsz)

    preprocess_samples: list[float] = []
    inference_samples: list[float] = []
    total_samples: list[float] = []

    total_iters = args.warmup + args.runs
    for i in range(total_iters):
        t0 = time.perf_counter()
        x = prepare_onnx_input(img)
        t1 = time.perf_counter()
        sess.run(None, {input_name: x})
        t2 = time.perf_counter()
        if i >= args.warmup:
            preprocess_samples.append((t1 - t0) * 1000.0)
            inference_samples.append((t2 - t1) * 1000.0)
            total_samples.append((t2 - t0) * 1000.0)

    result = {
        "model": portable_model_name(args.model),
        "runs": args.runs,
        "warmup": args.warmup,
        "threads": args.threads,
        "preprocess_ms": round(float(np.mean(preprocess_samples)), 4),
        "inference_ms": round(float(np.mean(inference_samples)), 4),
        "total_ms": round(float(np.mean(total_samples)), 4),
        "fps_total": round(1000.0 / float(np.mean(total_samples)), 3),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
