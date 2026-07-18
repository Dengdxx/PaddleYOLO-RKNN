# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""!
@file video_detect.py
@brief 对视频文件或摄像头执行 YOLO 目标检测并输出可视化结果。
@details
该脚本是用户最直接可见的推理入口之一，支持：
- ONNX 模型：适合 PC/ONNX Runtime 路径；
- Paddle 权重：适合直接复用训练产物；
- 视频文件与摄像头源：适合离线验证与实时演示。

这个入口也承担“训练权重 → 实际视觉效果”的最后一跳验证，常用于检查
输出解码、类别名映射和部署前可视化效果。

视频目标检测推理脚本：输入原始视频，输出带检测框视频

支持两种模型格式：
  - ONNX:   python video_detect.py --weights model.onnx --source video.mp4
  - Paddle: python video_detect.py --weights model.pdparams --source video.mp4

用法示例：
    # ONNX 模型推理
    python video_detect.py \
        --weights runs/detect/exp/weights/best.onnx \
        --source input_video.mp4 \
        --data data.yaml \
        --conf 0.25 --imgsz 640

    # 任意文件名的分割 ONNX 显式指定任务
    python video_detect.py \
        --weights model.onnx --task segment \
        --source input_video.mp4 --data data.yaml --imgsz 480x640

    # Paddle 训练权重推理
    python video_detect.py \
        --weights runs/detect/exp/weights/best.pdparams \
        --source input_video.mp4 \
        --data data.yaml \
        --conf 0.25

    # 摄像头实时检测
    python video_detect.py --weights best.onnx --source 0 --data data.yaml --show

    # 指定输出路径
    python video_detect.py --weights best.onnx --source video.mp4 --output result.mp4
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# ── 在 import paddle 之前抑制已知的无害警告 ──────────────────────────
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", message="No ccache found")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export.input_shape import normalize_static_imgsz

import cv2
import numpy as np


def parse_args():
    """解析视频检测命令行参数（模型路径、视频源、阈值等）。"""
    parser = argparse.ArgumentParser(
        description="YOLO 视频目标检测：输入视频/摄像头，输出带检测框视频",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径（.onnx / .pdparams）")
    parser.add_argument("--source", type=str, required=True, help="输入视频路径，或 0 表示摄像头")
    parser.add_argument(
        "--task",
        choices=["detect", "segment"],
        default=None,
        help="模型任务；文件名无法体现任务时显式指定",
    )
    parser.add_argument("--data", type=str, default=None, help="数据集 yaml（用于读取类别名，如 data.yaml）")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径（默认: <source>_det.mp4）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值（默认 0.7）")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        default=["640"],
        metavar="SIZE",
        help="推理图像尺寸：SIZE、HxW 或 H W（默认 640）",
    )
    parser.add_argument("--device", type=str, default=None, help="设备：cpu / 0 / cuda:0（默认自动选择）")
    parser.add_argument("--show", action="store_true", help="实时显示检测结果窗口")
    parser.add_argument("--no-save", action="store_true", help="不保存输出视频（仅在 --show 时有用）")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="只检测指定类别 ID，如 --classes 0 2")
    parser.add_argument("--line-width", type=int, default=2, help="检测框线宽（默认 2）")
    return parser.parse_args()


def main():
    """主函数：加载模型 → 打开视频/摄像头 → 逐帧检测 → 保存/显示结果。"""
    args = parse_args()
    args.imgsz = normalize_static_imgsz(args.imgsz)
    from ddyolo26 import YOLO

    # 加载模型
    print(f"加载模型: {args.weights}")
    model = YOLO(args.weights, task=args.task)

    # 确定输入源
    source = args.source
    is_webcam = source.isdigit()
    if is_webcam:
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"错误：无法打开视频源 '{args.source}'")
        sys.exit(1)

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    print(f"视频: {w}×{h} @ {fps:.1f}fps" + (f"  共 {total} 帧" if total else ""))

    # 设置输出
    writer = None
    save = not args.no_save
    if save:
        if args.output:
            out_path = args.output
        elif is_webcam:
            out_path = "webcam_det.mp4"
        else:
            p = Path(args.source)
            out_path = str(p.parent / f"{p.stem}_det.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"输出: {out_path}")

    # 构建预测参数
    predict_kwargs = dict(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=args.classes,
        verbose=False,
    )
    if args.data is not None:
        # 将 data 交给 AutoBackend 解析，兼容 Paddle 模型和 ONNX 等外部模型；
        # 外部模型的 `model.model` 可能只是路径字符串，不能直接写入 `.names`。
        predict_kwargs["data"] = args.data
    if args.device is not None:
        predict_kwargs["device"] = args.device

    # 逐帧推理
    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # YOLO 推理
            results = model.predict(frame, stream=False, **predict_kwargs)
            r = results[0]

            # 绘制检测框
            annotated = r.plot(line_width=args.line_width)

            # 保存
            if writer is not None:
                writer.write(annotated)

            # 显示
            if args.show:
                cv2.imshow("YOLO Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n用户中断")
                    break

            # 进度
            if frame_idx % 50 == 0 or frame_idx == 1:
                elapsed = time.time() - t_start
                speed = frame_idx / elapsed if elapsed > 0 else 0
                n_det = len(r.boxes) if r.boxes is not None else 0
                progress = f"  ({frame_idx}/{total})" if total else ""
                print(f"帧 {frame_idx}{progress}  检测: {n_det} 个目标  速度: {speed:.1f} fps")

    except KeyboardInterrupt:
        print("\n用户中断")

    # 清理
    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"\n完成！共处理 {frame_idx} 帧，耗时 {elapsed:.1f}s，平均 {avg_fps:.1f} fps")
    if writer is not None:
        import os

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"输出已保存: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
