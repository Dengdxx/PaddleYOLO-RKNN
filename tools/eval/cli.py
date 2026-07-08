#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""!
@file tools/eval/cli.py
@brief YOLO 通用评估入口，支持 ONNX 与 RKNN 后端。
@details
该脚本统一封装了 YOLO 在 PC 与板端的评估路径，可处理主要输出语义：
- `e2e`：NMS-free end2end 单输出；
- `one2one_raw`：保留 one2one 原始头输出，不做传统 NMS；
- `pre_dist` / `pre_dfl` / `seg_pre_dist` / `seg_pre_dfl`：为 RKNN 部署保留原始回归/分类输出，
    由 CPU 侧完成后处理。

这是本仓库 Paddle 模型与 RKNN 导出链路之间的主要评估入口。

计算逐类别及整体的 Precision、Recall、mAP50、mAP50-95 指标。
数据集通过标准 YOLO data.yaml 文件指定（包含 val 路径、nc 类别数、names 类别名）。
自动检测 COCO 数据集结构并使用 pycocotools 官方评估；
对于自定义 YOLO 格式数据集，使用内置的 101 点插值 AP 计算。

支持五种模型输出格式：
  - e2e:   单输出 [1,300,6]，模型内部已完成 TopK + 置信度筛选
  - one2one_raw: 单输出 [1,4+nc,N]，保留 one2one 原始输出
  - pre_dist: 双输出 [1,4,N] + [1,nc,N]，CPU 做 dist2bbox + sigmoid + NMS-free/NMS
  - pre_dfl: 双输出 [1,4*reg_max,N] + [1,nc,N]，CPU 做 DFL + dist2bbox + sigmoid + NMS
  - seg_e2e: 双输出 [1,300,6+nm] + [1,nm,H,W]
  - seg_yolov8_raw: 双输出 [1,4+nc+nm,N] + [1,nm,H,W]
  - seg_pre_dist: 四输出 [1,4,N] + [1,nc,N] + [1,nm,N] + [1,nm,H,W]
  - seg_pre_dfl: 四输出 legacy 或五输出 transposed + score_sum

用法:
    # ONNX FP32 评估自定义数据集
    python -m tools.eval --backend onnx --model runs/detect/exp/weights/best_paddle_predfl_fp32_640.onnx \
        --data data/your.yaml --imgsz 640

    # RKNN INT8 板端评估
    python -m tools.eval --backend rknn --model runs/detect/exp/weights/best_paddle_predfl_int8_640.rknn \
        --data coco.yaml --imgsz 640

    # COCO val2017 评估（自动检测 COCO 目录结构）
    python -m tools.eval --backend onnx --model weights/yolov8/yolov8n_paddle_predfl_fp32_640.onnx \
        --data coco.yaml --imgsz 640

    # 批量评估目录下所有 RKNN 模型
    python -m tools.eval --backend rknn --model "*.rknn" --data data/your.yaml

    # 快速子集测试
    python -m tools.eval --backend onnx --model model.onnx --data data/your.yaml --max-images 50
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .backend_utils import prepare_onnx_input, prepare_rknn_input
except ImportError:  # pragma: no cover - 支持直接执行 `python tools/eval/cli.py`
    from backend_utils import prepare_onnx_input, prepare_rknn_input

# ── COCO 类别 ID 映射（仅数据集为 COCO 时使用）─────────────
COCO_IDS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]

SEGMENT_OUTPUT_FORMATS = ("seg_e2e", "seg_yolov8_topk", "seg_yolov8_raw", "seg_pre_dist", "seg_pre_dfl")


def _parse_bool_arg(value):
    """解析命令行布尔值。"""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"期望布尔值，实际得到: {value}")


def parse_args():
    """解析命令行参数。

    支持的参数包括：推理后端（onnx/rknn/rknn_sim）、模型路径（支持 glob 通配符）、
    数据集 YAML 配置、输入尺寸（默认从文件名自动推断）、置信度阈值、NMS IoU 阈值、
    最大检测数、评估图片上限（用于快速测试）、NPU 核心绑定、输出路径等。
    """
    p = argparse.ArgumentParser(prog="python -m tools.eval", description="在任意 YOLO 格式数据集上评估 YOLO")
    p.add_argument("--backend", required=True, choices=["onnx", "rknn", "rknn_sim"], help="推理后端")
    p.add_argument("--model", required=True, help="模型路径或 glob 通配符（如 '*.rknn'）")
    p.add_argument("--data", required=True, help="数据集 YAML 路径（含 val、nc、names）")
    p.add_argument(
        "--imgsz", default=None, help="输入图像尺寸，支持 640、640x640、480,640（默认：从模型文件名自动推断）"
    )
    p.add_argument("--conf", type=float, default=0.001, help="置信度阈值（默认 0.001，用于 mAP 评估）")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值（默认 0.7，与基准验证口径对齐）")
    p.add_argument("--max-det", type=int, default=300, help="每张图片最大检测数（默认 300）")
    p.add_argument("--max-images", type=int, default=None, help="限制评估图片数（用于快速测试）")
    p.add_argument(
        "--core",
        default="core0",
        choices=["auto", "core0", "core01", "core012"],
        help="RKNN NPU 核心掩码（默认 core0）",
    )
    p.add_argument("--output", default=None, help="输出 JSON 路径（默认自动命名）")
    p.add_argument("--verbose", action="store_true", help="打印逐类别指标")
    p.add_argument(
        "--predist-postprocess",
        default="nmsfree_exact",
        choices=["nms", "nmsfree_simple", "nmsfree_exact"],
        help="pre_dist / seg_pre_dist 路径的 CPU 后处理模式：\n"
        "  nms            - 现有基线：decode + sigmoid + NMS\n"
        "  nmsfree_simple - 每个 anchor 仅保留最高类别，不做 NMS\n"
        "  nmsfree_exact  - 复刻 YOLO26 _postprocess_export 的 top-k/gather 语义",
    )
    p.add_argument(
        "--mask-eval",
        default="fast",
        choices=["native", "fast"],
        help="分割 mask 评估口径（默认 fast，对齐 ultralytics model.val() 打印指标）：\n"
        "  fast   - coeff@proto → crop（proto 分辨率），\n"
        "           等价于 ultralytics 默认 model.val() 打印口径；\n"
        "  native - coeff@proto → 上采样到输入分辨率 → crop，\n"
        "           等价于 ultralytics process_mask_native / save_json=True 或 save_txt=True",
    )
    p.add_argument(
        "--overlap-mask",
        type=_parse_bool_arg,
        default=True,
        help="分割 GT mask 口径：true 生成互斥 overlap mask；false 保留独立实例 mask，"
        "用于对齐 overlap_mask=False 训练。默认 true。",
    )
    return p.parse_args()


# ── 数据集加载 ────────────────────────────────────────────────────


def load_data_yaml(yaml_path):
    """加载 YOLO data.yaml 数据集配置文件。

    优先使用 PyYAML 库解析；如果环境中未安装 PyYAML，
    则回退到内置的简易解析器 _parse_simple_yaml()。

    参数:
        yaml_path: data.yaml 文件路径。

    返回:
        dict: 包含 val、nc、names 等键的配置字典。
    """
    try:
        import yaml
    except ImportError:
        # 简易 YAML 解析器（处理简单 data.yaml 文件）
        return _parse_simple_yaml(yaml_path)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data


def _parse_simple_yaml(yaml_path):
    """简易 YAML 解析器（无 PyYAML 依赖时的回退方案）。

    仅支持简单的 key: value 格式，能解析 nc（整数）、names（列表）、
    train/val/test（字符串路径）等常见字段。不支持嵌套结构。

    参数:
        yaml_path: data.yaml 文件路径。

    返回:
        dict: 解析后的配置字典。
    """
    data = {}
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key == "nc":
                data[key] = int(val)
            elif key == "names":
                # 解析 `['a', 'b', 'c']` 这种简化列表写法
                val = val.strip("[]")
                data[key] = [n.strip().strip("'\"") for n in val.split(",")]
            elif key in ("path", "train", "val", "test"):
                data[key] = val
    return data


def resolve_dataset_paths(data, yaml_path):
    """从 data.yaml 配置中解析验证集的图片和标签目录路径。

    处理逻辑：
    1. 将 val 字段的相对路径基于 yaml 所在目录解析为绝对路径
    2. 根据 YOLO 惯例将 images 目录替换为 labels 目录
    3. 自动检测是否为 COCO 数据集（nc=80 且路径含 val2017/coco）
    4. 路径不存在时尝试常见的 YOLO 数据集目录布局作为回退

    参数:
        data: load_data_yaml() 返回的配置字典。
        yaml_path: data.yaml 文件路径（用于解析相对路径）。

    返回:
        (img_dir, label_dir, nc, names, is_coco) 五元组。
    """
    yaml_dir = Path(yaml_path).resolve().parent
    raw_names = data.get("names")
    if isinstance(raw_names, dict):
        names = [raw_names[i] for i in sorted(raw_names.keys())]
    elif isinstance(raw_names, list):
        names = raw_names
    else:
        names = None
    nc = data.get("nc")
    if nc is None and names is not None:
        nc = len(names)
    if names is None:
        names = [f"class_{i}" for i in range(nc)]

    # 检测是否为 COCO 格式数据集
    is_coco = False
    if nc == 80 and "val" in data:
        val_path = data["val"]
        # COCO 数据集通常在路径中包含 val2017 或有 annotations 目录
        if "val2017" in str(val_path) or "coco" in str(val_path).lower():
            is_coco = True

    val_path = data.get("val", "")
    ds_path = data.get("path", "")

    # 解析优先级：
    # 1. path + val（path 为数据集根，val 为相对子目录）
    # 2. val 相对于 yaml 所在目录
    # 3. val 本身为绝对路径
    if ds_path and not os.path.isabs(ds_path):
        ds_path = str(yaml_dir / ds_path)

    if ds_path and os.path.isdir(ds_path):
        img_dir = os.path.join(ds_path, val_path)
    elif not os.path.isabs(val_path):
        img_dir = str(yaml_dir / val_path)
    else:
        img_dir = val_path

    img_dir = os.path.normpath(img_dir)

    # 回退：路径不存在时尝试常见的 YOLO 数据集目录布局
    if not os.path.isdir(img_dir):
        alt = str(yaml_dir / val_path.lstrip("./").lstrip("../"))
        alt = os.path.normpath(alt)
        if os.path.isdir(alt):
            img_dir = alt

    # YOLO 惯例：labels 目录是将 images 目录的 'images' 替换为 'labels'
    label_dir = img_dir.replace("/images", "/labels").replace("\\images", "\\labels")

    if not os.path.isdir(img_dir):
        print(f"错误：未找到验证集图片目录: {img_dir}")
        sys.exit(1)

    return img_dir, label_dir, nc, names, is_coco


def _warn_label_issue(label_path, line_no, reason):
    """打印可定位的标签破损 warning。"""
    print(f"警告：跳过损坏的标注行 {label_path}:{line_no}: {reason}")


def _parse_yolo_label_line(line, label_path, line_no, nc=None):
    """解析并校验单行 YOLO 标签，破损时返回 ``None``。"""
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) < 5:
        _warn_label_issue(label_path, line_no, "至少需要 5 列")
        return None

    try:
        cls_value = float(parts[0])
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    except ValueError as exc:
        _warn_label_issue(label_path, line_no, f"存在非数值内容 ({exc})")
        return None

    if not np.isfinite(cls_value):
        _warn_label_issue(label_path, line_no, "class id 不是有限数值")
        return None
    cls_id = int(cls_value)
    if abs(cls_value - cls_id) > 1e-9:
        _warn_label_issue(label_path, line_no, f"class id 不是整数: {cls_value}")
        return None
    if nc is not None and not (0 <= cls_id < int(nc)):
        _warn_label_issue(label_path, line_no, f"class id {cls_id} 超出 [0, {int(nc)}) 范围")
        return None
    if not np.all(np.isfinite(coords)):
        _warn_label_issue(label_path, line_no, "坐标包含 NaN 或 Inf")
        return None
    if coords.size != 4 and (coords.size < 6 or coords.size % 2 != 0):
        _warn_label_issue(label_path, line_no, f"坐标数量无效: {coords.size}")
        return None
    if coords.size == 4 and (coords[2] <= 0 or coords[3] <= 0):
        _warn_label_issue(label_path, line_no, "bbox 宽高必须为正")
        return None

    if np.any(coords < -1e-6) or np.any(coords > 1.0 + 1e-6):
        _warn_label_issue(label_path, line_no, "坐标超出 [0, 1]，将进行裁剪")
        coords = np.clip(coords, 0.0, 1.0)

    return cls_id, coords


def load_yolo_labels(label_dir, img_files, nc=None):
    """加载 YOLO 格式的标注文件。

    自动兼容两种标注格式：
    - bbox 格式（5 个值）：class_id cx cy w h
    - polygon 格式（>5 个值）：class_id x1 y1 x2 y2 ... xN yN
      对 polygon 自动计算最小外接 bbox。

    参数:
        label_dir: 标注文件所在目录。
        img_files: 图片文件名列表（用于匹配同名 .txt）。

    返回:
        dict: {图片名(无扩展名): [(class_id, cx, cy, w, h), ...]}。
    """
    labels = {}
    for img_path in img_files:
        stem = Path(img_path).stem
        label_path = os.path.join(label_dir, stem + ".txt")
        boxes = []
        if os.path.isfile(label_path):
            with open(label_path) as f:
                for line_no, line in enumerate(f, start=1):
                    parsed = _parse_yolo_label_line(line, label_path, line_no, nc=nc)
                    if parsed is None:
                        continue
                    cls_id, coords_arr = parsed
                    coords = coords_arr.tolist()
                    if len(coords) == 4:
                        # bbox 格式：`cx cy w h`
                        cx, cy, w, h = coords
                    else:
                        # polygon 格式：`x1 y1 x2 y2 ... xN yN`，这里转成最小外接 bbox
                        xs = coords[0::2]
                        ys = coords[1::2]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        cx = (x_min + x_max) / 2
                        cy = (y_min + y_max) / 2
                        w = x_max - x_min
                        h = y_max - y_min
                    boxes.append((cls_id, cx, cy, w, h))
        labels[stem] = boxes
    return labels


def yolo_to_xyxy(boxes, img_w, img_h):
    """将 YOLO 归一化框坐标转换为像素绝对坐标（xyxy 格式）。

    参数:
        boxes: [(class_id, cx, cy, w, h), ...] 归一化坐标列表。
        img_w: 原图宽度（像素）。
        img_h: 原图高度（像素）。

    返回:
        np.array [N, 5] = (class_id, x1, y1, x2, y2)，像素坐标。
    """
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)
    result = []
    for cls_id, cx, cy, w, h in boxes:
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        result.append([cls_id, x1, y1, x2, y2])
    return np.array(result, dtype=np.float32)


def load_yolo_ground_truth(label_path, img_w, img_h):
    """加载单张图片的 YOLO GT，并在 polygon 标注存在时保留 mask 信息。

    同时兼容两种标签行：
    - bbox:    class_id cx cy w h
    - polygon: class_id x1 y1 x2 y2 ... xN yN

    返回:
        tuple:
            gt_boxes: [N, 4]，所有 GT 的 xyxy 像素框
            gt_classes: [N]，所有 GT 的类别 id
            gt_masks: [M, H, W]，仅 polygon GT 的二值掩码
            gt_mask_classes: [M]，与 gt_masks 对齐的类别 id
    """
    import cv2

    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    empty_classes = np.zeros((0,), dtype=np.int64)
    empty_masks = np.zeros((0, img_h, img_w), dtype=np.uint8)

    if not os.path.isfile(label_path):
        return empty_boxes, empty_classes, empty_masks, empty_classes.copy()

    gt_boxes = []
    gt_classes = []
    gt_masks = []
    gt_mask_classes = []

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)

            if coords.size == 4:
                cx, cy, w, h = coords.tolist()
                x1 = np.clip((cx - w / 2) * img_w, 0, max(img_w - 1, 0))
                y1 = np.clip((cy - h / 2) * img_h, 0, max(img_h - 1, 0))
                x2 = np.clip((cx + w / 2) * img_w, 0, max(img_w - 1, 0))
                y2 = np.clip((cy + h / 2) * img_h, 0, max(img_h - 1, 0))
                gt_boxes.append([x1, y1, x2, y2])
                gt_classes.append(cls_id)
                continue

            if coords.size < 6 or coords.size % 2 != 0:
                continue

            pts = coords.reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
            pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)

            pts_px = np.empty_like(pts)
            pts_px[:, 0] = np.clip(pts[:, 0] * img_w, 0, max(img_w - 1, 0))
            pts_px[:, 1] = np.clip(pts[:, 1] * img_h, 0, max(img_h - 1, 0))

            x1 = float(pts_px[:, 0].min())
            y1 = float(pts_px[:, 1].min())
            x2 = float(pts_px[:, 0].max())
            y2 = float(pts_px[:, 1].max())
            gt_boxes.append([x1, y1, x2, y2])
            gt_classes.append(cls_id)

            pts_i32 = pts_px.astype(np.int32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_i32], 1)
            if mask.any():
                gt_masks.append(mask)
                gt_mask_classes.append(cls_id)

    gt_boxes_arr = np.array(gt_boxes, dtype=np.float32) if gt_boxes else empty_boxes
    gt_classes_arr = np.array(gt_classes, dtype=np.int64) if gt_classes else empty_classes
    gt_masks_arr = np.stack(gt_masks, axis=0).astype(np.uint8) if gt_masks else empty_masks
    gt_mask_classes_arr = np.array(gt_mask_classes, dtype=np.int64) if gt_mask_classes else empty_classes.copy()
    return gt_boxes_arr, gt_classes_arr, gt_masks_arr, gt_mask_classes_arr


def _polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """将 polygon 坐标栅格化为下采样二值 mask。

    参数:
        imgsz: 原始 mask 尺寸，格式为 (height, width)。
        polygons: polygon 点集列表，每个元素为扁平坐标数组。
        color: 写入 mask 的像素值。
        downsample_ratio: 输出 mask 相对 `imgsz` 的下采样倍率。

    返回:
        np.ndarray: 下采样后的二值 mask。
    """
    import cv2

    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    if polygons.size:
        polygons = polygons.reshape((polygons.shape[0], -1, 2))
        cv2.fillPoly(mask, polygons, color=color)
    out_h, out_w = imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio
    return cv2.resize(mask, (out_w, out_h))


def _resample_segments(segments, n=1000):
    """将 polygon 轮廓重采样到固定点数。

    参数:
        segments: polygon 点集列表，每个元素形状为 [N, 2]。
        n: 目标采样点数。

    返回:
        list[np.ndarray]: 重采样后的 polygon 点集列表。
    """
    result = []
    for segment in segments:
        if len(segment) == n:
            result.append(segment.astype(np.float32))
            continue
        s = np.concatenate((segment, segment[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        if len(s) < n:
            x = np.insert(x, np.searchsorted(x, xp), xp)
        result.append(np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T)
    return result


def _polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """按面积排序生成互斥实例 mask。

    参数:
        imgsz: 原始 mask 尺寸，格式为 (height, width)。
        segments: polygon 点集列表，每个元素形状为 [N, 2]。
        downsample_ratio: 输出 mask 相对 `imgsz` 的下采样倍率。

    返回:
        tuple[np.ndarray, np.ndarray]: 互斥实例索引 mask，以及面积降序的原始实例索引。
    """
    dtype = np.int32 if len(segments) > 255 else np.uint8
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=dtype,
    )
    areas = []
    instance_masks = []
    for segment in segments:
        mask = _polygon2mask(
            imgsz,
            [segment.reshape(-1)],
            color=1,
            downsample_ratio=downsample_ratio,
        ).astype(dtype)
        instance_masks.append(mask)
        areas.append(mask.sum())
    sorted_idx = np.argsort(-np.asarray(areas))
    instance_masks = np.asarray(instance_masks, dtype=dtype)[sorted_idx]
    for i, mask in enumerate(instance_masks):
        masks = masks + mask * (i + 1)
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, sorted_idx


def load_yolo_segment_ground_truth_eval(
    label_path, img_w, img_h, imgsz, scale, pad_w, pad_h, mask_downsample_ratio=4, overlap_mask=True, nc=None
):
    """按基准验证口径加载分割 GT。

    box 使用 letterbox 输入空间；mask 将 polygon 映射到 letterbox 空间，
    重采样到固定点数，再按 `overlap_mask=True` 生成下采样 mask。
    """
    input_h, input_w = parse_input_shape(imgsz)
    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    empty_classes = np.zeros((0,), dtype=np.int64)
    mask_h = input_h // mask_downsample_ratio
    mask_w = input_w // mask_downsample_ratio
    empty_masks = np.zeros((0, mask_h, mask_w), dtype=np.uint8)

    if not os.path.isfile(label_path):
        return empty_boxes, empty_classes, empty_masks, empty_classes.copy()

    gt_boxes = []
    gt_classes = []
    segments_norm = []
    segment_classes = []

    with open(label_path) as f:
        for line_no, line in enumerate(f, start=1):
            parsed = _parse_yolo_label_line(line, label_path, line_no, nc=nc)
            if parsed is None:
                continue
            cls_id, coords = parsed

            if coords.size == 4:
                cx, cy, w, h = coords.tolist()
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
            else:
                pts = coords.reshape(-1, 2)
                pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
                pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
                xs = pts[:, 0] * img_w
                ys = pts[:, 1] * img_h
                x1, x2 = float(xs.min()), float(xs.max())
                y1, y2 = float(ys.min()), float(ys.max())

                segments_norm.append(pts.astype(np.float32))
                segment_classes.append(cls_id)

            box_lb = _letterbox_boxes(
                np.array([[x1, y1, x2, y2]], dtype=np.float32),
                scale,
                pad_w,
                pad_h,
                imgsz,
            )[0]
            gt_boxes.append(box_lb)
            gt_classes.append(cls_id)

    gt_masks = []
    gt_mask_classes = []
    if segments_norm:
        segment_resamples = 1000
        max_len = max(len(s) for s in segments_norm)
        segment_resamples = max_len + 1 if segment_resamples < max_len else segment_resamples
        segments_norm = _resample_segments(segments_norm, n=segment_resamples)
        segments_lb = []
        for segment in segments_norm:
            segment_lb = np.empty_like(segment, dtype=np.float32)
            segment_lb[:, 0] = segment[:, 0] * img_w * scale + pad_w
            segment_lb[:, 1] = segment[:, 1] * img_h * scale + pad_h
            segments_lb.append(segment_lb)

        if overlap_mask:
            overlap, sorted_idx = _polygons2masks_overlap(
                (input_h, input_w),
                segments_lb,
                downsample_ratio=mask_downsample_ratio,
            )
            for rank, src_idx in enumerate(sorted_idx):
                mask = (overlap == (rank + 1)).astype(np.uint8)
                gt_masks.append(mask)
                gt_mask_classes.append(segment_classes[int(src_idx)])
        else:
            for cls_id, segment in zip(segment_classes, segments_lb):
                mask = _polygon2mask(
                    (input_h, input_w),
                    [segment.reshape(-1)],
                    color=1,
                    downsample_ratio=mask_downsample_ratio,
                )
                gt_masks.append(mask.astype(np.uint8))
                gt_mask_classes.append(cls_id)

    gt_boxes_arr = np.stack(gt_boxes, axis=0).astype(np.float32) if gt_boxes else empty_boxes
    gt_classes_arr = np.array(gt_classes, dtype=np.int64) if gt_classes else empty_classes
    gt_masks_arr = np.stack(gt_masks, axis=0).astype(np.uint8) if gt_masks else empty_masks
    gt_mask_classes_arr = np.array(gt_mask_classes, dtype=np.int64) if gt_mask_classes else empty_classes.copy()
    return gt_boxes_arr, gt_classes_arr, gt_masks_arr, gt_mask_classes_arr


# ── 图片工具函数 ────────────────────────────────────────────────────


def infer_input_size(model_path):
    """从模型文件名推断输入尺寸。例如 '_384.rknn' -> 384，无匹配则默认 640。"""
    m = re.search(r"_(\d{3,4})\.(rknn|onnx)$", os.path.basename(model_path))
    return int(m.group(1)) if m else 640


def parse_input_shape(value):
    """解析输入尺寸为 (height, width)。"""
    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"输入尺寸 tuple 长度应为 2，实际 {value}")
        h, w = value
        return int(h), int(w)
    if isinstance(value, int):
        return value, value

    text = str(value).strip().lower()
    if not text:
        return None
    for sep in ("x", ","):
        if sep in text:
            parts = [p.strip() for p in text.split(sep)]
            if len(parts) != 2:
                raise ValueError(f"输入尺寸格式错误: {value}")
            h, w = (int(parts[0]), int(parts[1]))
            if h <= 0 or w <= 0:
                raise ValueError(f"输入尺寸必须为正数: {value}")
            return h, w
    size = int(text)
    if size <= 0:
        raise ValueError(f"输入尺寸必须为正数: {value}")
    return size, size


def format_input_shape(input_shape):
    """将 (height, width) 格式化为短字符串。"""
    h, w = parse_input_shape(input_shape)
    return str(h) if h == w else f"{h}x{w}"


def letterbox(img, new_shape=640):
    """等比缩放 + 灰色填充至目标尺寸（letterbox 预处理）。

    保持原图宽高比，将图片缩放到目标画布上，空白区域填充 114。

    参数:
        img: 输入图片（HWC, uint8）。
        new_shape: 目标尺寸，支持 int 或 (height, width)。

    返回:
        (padded_img, scale, (pad_w, pad_h))：填充后图片、缩放比例、填充偏移。
    """
    import cv2

    new_h, new_w = parse_input_shape(new_shape)
    h, w = img.shape[:2]
    scale = min(new_h / h, new_w / w)
    scale = min(scale, 1.0)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh)) if (w, h) != (nw, nh) else img
    canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    pad_w = (new_w - nw) / 2
    pad_h = (new_h - nh) / 2
    top = round(pad_h - 0.1)
    left = round(pad_w - 0.1)
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)


def sigmoid(x):
    """数值稳定的 sigmoid 函数，避免 x 过大/过小时的 exp 溢出。"""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# ── 解码与 NMS ───────────────────────────────────────────────────────


def _infer_box_encoding(det_tensor, conf_thresh=0.001):
    """从 [1, K, C] 输出推断 box 编码（xyxy 还是 cxcywh）。

    通过检测有置信度（score > conf_thresh）的行中是否多数满足 x2 <= x1
    来判断是否为 cxcywh。conf_thresh 排除零填充槽位的干扰。
    无有效检测时默认返回 'xyxy'（YOLO26 e2e 语义）。
    """
    dets = det_tensor[0] if det_tensor.ndim == 3 else det_tensor
    if dets.shape[0] == 0 or dets.shape[1] < 5:
        return "xyxy"
    valid = np.isfinite(dets).all(axis=1) & (dets[:, 4] > conf_thresh)
    dets = dets[valid]
    if len(dets) == 0:
        return "xyxy"
    box = dets[:, :4]
    bad = ((box[:, 2] <= box[:, 0]) | (box[:, 3] <= box[:, 1])).sum()
    return "cxcywh" if bad > len(box) * 0.5 else "xyxy"


def detect_output_format(outputs):
    """自动检测模型输出格式。

    判断规则：
    - 单输出且最后一维=6 -> `e2e`（YOLO26 one2one xyxy，真正 NMS-free）
    - 单输出且形状为 [1,4+nc,N] -> `one2one_raw`
    - 双输出 [1,4*reg_max,N]+[1,nc,N]，其中 4*reg_max>4 -> `pre_dfl`
    - 双输出 [1,4,N]+[1,nc,N] -> `pre_dist`
    - 双输出 [1,K,6+nm]+[1,nm,H,W]，box 为 xyxy -> `seg_e2e`（YOLO26-seg one2one）
    - 双输出 [1,K,6+nm]+[1,nm,H,W]，box 为 cxcywh -> `seg_yolov8_topk`（YOLOv8 Paddle TopK）
    - 双输出 [1,4+nc+nm,N]+[1,nm,H,W] -> `seg_yolov8_raw`
    - 四输出 [1,4,N]+[1,nc,N]+[1,nm,N]+[1,nm,H,W] -> `seg_pre_dist`
    - 四输出 legacy 或五输出 transposed + score_sum -> `seg_pre_dfl`
    - 其他 -> `unknown`
    """
    n = len(outputs)
    if n == 1:
        o = outputs[0]
        if o.ndim == 3 and o.shape[-1] == 6:
            return "e2e"
        if o.ndim == 3 and o.shape[1] > 4:
            # 区分 YOLO26 one2one_raw（已 TopK，N≈300，xyxy）与 YOLOv8 raw
            # （未 NMS，N=8400 等多 anchor，cxcywh+sigmoid，仍需 NMS）。
            n_anchors = o.shape[2]
            if n_anchors >= 1000:
                return "yolov8_raw"
            return "one2one_raw"
    elif n == 2:
        if outputs[0].ndim == 3 and outputs[1].ndim == 4:
            c0 = outputs[0].shape[1]
            nm = outputs[1].shape[1]
            if outputs[0].shape[-1] == 6 + nm:
                # 区分 YOLO26-seg（xyxy，one2one 无 NMS）与 YOLOv8 Paddle TopK（cxcywh，仍需 NMS）
                box_enc = _infer_box_encoding(outputs[0])
                return "seg_yolov8_topk" if box_enc == "cxcywh" else "seg_e2e"
            if c0 > 4 + nm and outputs[0].shape[2] >= 1000:
                return "seg_yolov8_raw"
        if outputs[0].ndim == 3 and outputs[1].ndim == 3:
            b0 = outputs[0].shape[1]
            if b0 == 4:
                return "pre_dist"
            # pre_dfl：`output[0] = [1, 4*reg_max, N]`，其中 `reg_max` 通常为 16
            if b0 % 4 == 0 and b0 > 4:
                return "pre_dfl"
    elif n == 4:
        # seg_pre_dist: 4 输出 [1,4,N] + [1,nc,N] + [1,nm,N] + [1,nm,H,W]
        if outputs[0].ndim == 3 and outputs[1].ndim == 3 and outputs[3].ndim == 4:
            c0 = outputs[0].shape[1]
            if c0 == 4:
                return "seg_pre_dist"
            if c0 > 4 and c0 % 4 == 0:
                return "seg_pre_dfl"
    elif n == 5:
        # seg_pre_dfl: 5 输出，转置 [1,N,4*reg_max] 或 legacy [1,4*reg_max,N] + score_sum
        if outputs[0].ndim == 3 and outputs[1].ndim == 3 and outputs[3].ndim == 4:
            c1 = outputs[0].shape[1]
            c2 = outputs[0].shape[2]
            if c2 > 4 and c2 % 4 == 0 and c1 > c2:
                return "seg_pre_dfl"
            if c1 > 4 and c1 % 4 == 0 and c2 > c1:
                return "seg_pre_dfl"
    return "unknown"


def decode_e2e(output, conf_thresh, max_det, iou_thresh=0.7):
    """解码 e2e 格式输出 [1, 300, 6] -> [N, 6] = x1, y1, x2, y2, score, class_id。

    适用于两类常见 TopK 输出：
    - YOLO26 one2one 头：xyxy 编码，模型侧已完成去重，此处仅过滤与截断；
    - YOLOv8 Paddle TopK：cxcywh 编码，仍可能存在重叠候选，需要 CPU 侧 per-class NMS。
    """
    dets = output[0]
    if dets.ndim == 3:
        dets = dets[0]  # [300, 6]
    valid = np.isfinite(dets).all(axis=1)
    dets = dets[valid]
    box_encoding = _infer_box_encoding(dets, conf_thresh)
    mask = dets[:, 4] > conf_thresh
    dets = dets[mask]
    if len(dets) == 0:
        return dets
    if box_encoding == "cxcywh":
        boxes = dets[:, :4]
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        scores = dets[:, 4]
        classes = dets[:, 5]
        keep_all = []
        for c in np.unique(classes):
            sel = np.where(classes == c)[0]
            keep = nms(boxes_xyxy[sel], scores[sel], iou_threshold=iou_thresh)
            keep_all.extend(sel[keep].tolist())
        if not keep_all:
            return np.zeros((0, 6), dtype=np.float32)
        dets = np.column_stack([boxes_xyxy[keep_all], scores[keep_all], classes[keep_all]]).astype(np.float32)
        idx = np.argsort(-dets[:, 4])[:max_det]
        return dets[idx]
    if len(dets) > max_det:
        idx = np.argsort(-dets[:, 4])[:max_det]
        dets = dets[idx]
    return dets


def decode_one2one_raw(output, conf_thresh, max_det):
    """解码 one2one 原始单输出 [1,4+nc,N] -> [M, 6]。

    与双输出 logits 路径不同，这一路输出的分类分数已经过 sigmoid，且保持 one2one
    头的原始锚点分布。为尽量贴近 e2e 语义，这里只做：
    1. 每个锚点取最大类别分数
    2. 置信度阈值过滤
    3. 按分数排序截断到 max_det
    不做传统 NMS。
    """
    pred = output[0]
    if pred.ndim == 3:
        pred = pred[0]  # [4+nc, N]

    boxes = pred[:4].T
    scores = pred[4:]
    max_scores = np.max(scores, axis=0)
    class_ids = np.argmax(scores, axis=0).astype(np.float32)

    mask = max_scores > conf_thresh
    boxes_f = boxes[mask]
    scores_f = max_scores[mask]
    cls_f = class_ids[mask]

    if len(scores_f) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    dets = np.column_stack([boxes_f, scores_f, cls_f]).astype(np.float32)
    if len(dets) > max_det:
        idx = np.argsort(-dets[:, 4])[:max_det]
        dets = dets[idx]
    return dets


def decode_yolov8_raw(output, conf_thresh, max_det, iou_thresh):
    """解码 YOLOv8 经典 e2e 输出 [1, 4+nc, N] -> [M, 6]。

    与 YOLO26 的 `one2one_raw` 不同：
      - 输出锚点数 N 远大于 300（典型 8400），未做 TopK；
      - box 为 cxcywh，需转 xyxy；
      - 类别分数已 sigmoid；
      - 必须做经典 NMS。
    """
    pred = output[0]
    if pred.ndim == 3:
        pred = pred[0]  # [4+nc, N]

    boxes_cxcywh = pred[:4].T  # [N, 4]
    scores = pred[4:]
    max_scores = np.max(scores, axis=0)
    class_ids = np.argmax(scores, axis=0).astype(np.float32)

    mask = max_scores > conf_thresh
    boxes_f = boxes_cxcywh[mask]
    scores_f = max_scores[mask]
    cls_f = class_ids[mask]

    if len(scores_f) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 把 `cxcywh` 转为 `xyxy`
    cx, cy, w, h = boxes_f[:, 0], boxes_f[:, 1], boxes_f[:, 2], boxes_f[:, 3]
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # 按类别分别执行 NMS
    keep_all = []
    for c in np.unique(cls_f):
        sel = np.where(cls_f == c)[0]
        keep = nms(boxes_xyxy[sel], scores_f[sel], iou_threshold=iou_thresh)
        keep_all.extend(sel[keep].tolist())
    if not keep_all:
        return np.zeros((0, 6), dtype=np.float32)
    keep_all = np.array(keep_all, dtype=int)
    dets = np.column_stack([boxes_xyxy[keep_all], scores_f[keep_all], cls_f[keep_all]]).astype(np.float32)
    if len(dets) > max_det:
        idx = np.argsort(-dets[:, 4])[:max_det]
        dets = dets[idx]
    return dets


def decode_seg_yolov8_raw(outputs, conf_thresh, max_det, iou_thresh):
    """解码 YOLOv8-Seg 常规 FP32 ONNX 双输出，返回 bbox + mask coeff + proto。

    输入输出契约（seg_yolov8_raw）：
      outputs[0]: pred  [1, 4+nc+nm, N] — cxcywh boxes + sigmoid class scores + mask coeff
      outputs[1]: proto [1, nm, H, W]   — proto 特征图

    与 `seg_pre_dfl` 不同，这一路的 DFL 与 box 解码已经在图内完成，
    因此 CPU 侧只需做置信度筛选、per-class NMS 和 mask coeff gather。
    """
    proto = outputs[1][0] if outputs[1].ndim == 4 else outputs[1]
    nm = proto.shape[0]
    pred = outputs[0]
    if pred.ndim == 2:
        pred = pred[None, ...]
    nc = pred.shape[1] - 4 - nm
    if nc <= 0:
        raise ValueError(f"seg_yolov8_raw 输出通道应为 4+nc+nm，实际 pred={pred.shape}, proto={proto.shape}")
    return _segment_multilabel_nms(pred, proto, nc, conf_thresh, max_det, iou_thresh)


def _empty_seg_result(proto):
    """构造空分割解码结果。"""
    nm = proto.shape[0]
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.float32),
        "coeffs": np.zeros((0, nm), dtype=np.float32),
        "proto": proto,
    }


def _xyxy_to_xywh(boxes):
    """xyxy -> cxcywh。"""
    xywh = np.empty_like(boxes, dtype=np.float32)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh


def _xywh_to_xyxy(boxes):
    """cxcywh -> xyxy."""
    xyxy = np.empty_like(boxes, dtype=np.float32)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return xyxy


def _segment_multilabel_nms(prediction, proto, nc, conf_thresh, max_det, iou_thresh):
    """执行 class-aware multi-label NMS，并保留 mask coeff 行对齐。"""
    pred = np.asarray(prediction, dtype=np.float32)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if pred.shape[0] != 1:
        raise ValueError(f"当前评估仅支持 batch=1，实际 prediction={pred.shape}")

    extra = pred.shape[1] - nc - 4
    if extra < 0:
        raise ValueError(f"prediction 通道数不足：shape={pred.shape}, nc={nc}")

    cls_scores = pred[0, 4 : 4 + nc, :]
    candidate_mask = cls_scores.max(axis=0) > conf_thresh
    if not candidate_mask.any():
        return _empty_seg_result(proto)

    x = pred[0].T[candidate_mask]
    boxes = _xywh_to_xyxy(x[:, :4])
    cls = x[:, 4 : 4 + nc]
    coeffs = x[:, 4 + nc : 4 + nc + extra]
    anchor_idx, class_idx = np.nonzero(cls > conf_thresh)
    if anchor_idx.size == 0:
        return _empty_seg_result(proto)

    dets_np = np.column_stack(
        [
            boxes[anchor_idx],
            cls[anchor_idx, class_idx],
            class_idx.astype(np.float32),
            coeffs[anchor_idx],
        ]
    ).astype(np.float32)

    max_nms = 30000
    if dets_np.shape[0] > max_nms:
        order = np.argsort(-dets_np[:, 4])[:max_nms]
        dets_np = dets_np[order]

    max_wh = 7680.0
    boxes_for_nms = dets_np[:, :4] + dets_np[:, 5:6] * max_wh
    keep = nms(boxes_for_nms, dets_np[:, 4], iou_threshold=iou_thresh)
    keep = keep[:max_det]
    if len(keep) == 0:
        return _empty_seg_result(proto)
    dets_np = dets_np[keep]

    return {
        "boxes": dets_np[:, :4],
        "scores": dets_np[:, 4],
        "classes": dets_np[:, 5],
        "coeffs": dets_np[:, 6:],
        "proto": proto,
    }


def decode_seg_e2e(outputs, conf_thresh, max_det, iou_thresh=0.7):
    """解码分割 e2e 双输出，返回 bbox + mask coeff + proto。

    输入输出契约（seg_e2e）：
      outputs[0]: dets  [1, K, 6+nm] — box（xyxy）、score、class、mask coeff
      outputs[1]: proto [1, nm, H, W]

    仅适用于 YOLO26-seg one2one 头输出（xyxy 编码，无需 NMS）。
    YOLOv8 Paddle TopK 的 cxcywh 输出请使用 `decode_seg_yolov8_topk()`。
    """
    dets = outputs[0]
    if dets.ndim == 3:
        dets = dets[0]
    proto = outputs[1][0] if outputs[1].ndim == 4 else outputs[1]
    nm = proto.shape[0]

    if dets.shape[1] < 6:
        raise ValueError(f"seg_e2e dets 最后一维应至少为 6，实际 {dets.shape}")

    valid = np.isfinite(dets).all(axis=1)
    dets = dets[valid]
    dets = dets[dets[:, 4] > conf_thresh]

    empty = {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.float32),
        "coeffs": np.zeros((0, nm), dtype=np.float32),
        "proto": proto,
    }
    if len(dets) == 0:
        return empty

    box = dets[:, :4]
    scores = dets[:, 4]
    cls = dets[:, 5]
    coeffs = dets[:, 6 : 6 + nm]

    if len(scores) > max_det:
        order = np.argsort(-scores)[:max_det]
        box = box[order]
        scores = scores[order]
        cls = cls[order]
        coeffs = coeffs[order]

    return {
        "boxes": box.astype(np.float32),
        "scores": scores.astype(np.float32),
        "classes": cls.astype(np.float32),
        "coeffs": coeffs.astype(np.float32),
        "proto": proto,
    }


def decode_seg_yolov8_topk(outputs, conf_thresh, max_det, iou_thresh=0.7):
    """解码 YOLOv8 Paddle TopK 分割双输出，返回 bbox + mask coeff + proto。

    输入输出契约（seg_yolov8_topk）：
      outputs[0]: dets  [1, K, 6+nm] — box（cxcywh）、score、class、mask coeff
      outputs[1]: proto [1, nm, H, W]

    适用于 YOLOv8-seg Paddle 旧导出：_postprocess_export 在模型内部执行了 TopK
    选出 K 个候选，但未做 IoU NMS，box 编码为 cxcywh。此处补做 per-class NMS。
    """
    dets = outputs[0]
    if dets.ndim == 3:
        dets = dets[0]
    proto = outputs[1][0] if outputs[1].ndim == 4 else outputs[1]
    nm = proto.shape[0]

    if dets.shape[1] < 6:
        raise ValueError(f"seg_yolov8_topk dets 最后一维应至少为 6，实际 {dets.shape}")

    valid = np.isfinite(dets).all(axis=1)
    dets = dets[valid]
    dets = dets[dets[:, 4] > conf_thresh]

    empty = {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.float32),
        "coeffs": np.zeros((0, nm), dtype=np.float32),
        "proto": proto,
    }
    if len(dets) == 0:
        return empty

    box_cxcywh = dets[:, :4]
    scores = dets[:, 4]
    cls = dets[:, 5]
    coeffs = dets[:, 6 : 6 + nm]

    # 把 `cxcywh` 转为 `xyxy`
    cx, cy, w, h = box_cxcywh[:, 0], box_cxcywh[:, 1], box_cxcywh[:, 2], box_cxcywh[:, 3]
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # 模型内 TopK 未做 IoU NMS，补做 per-class NMS
    keep_all = []
    for c_ in np.unique(cls):
        sel = np.where(cls == c_)[0]
        keep = nms(boxes_xyxy[sel], scores[sel], iou_threshold=iou_thresh)
        keep_all.extend(sel[keep].tolist())
    if not keep_all:
        return empty
    keep_arr = np.array(keep_all, dtype=np.int64)
    boxes_xyxy = boxes_xyxy[keep_arr]
    scores = scores[keep_arr]
    cls = cls[keep_arr]
    coeffs = coeffs[keep_arr]

    if len(scores) > max_det:
        order = np.argsort(-scores)[:max_det]
        boxes_xyxy = boxes_xyxy[order]
        scores = scores[order]
        cls = cls[order]
        coeffs = coeffs[order]

    return {
        "boxes": boxes_xyxy.astype(np.float32),
        "scores": scores.astype(np.float32),
        "classes": cls.astype(np.float32),
        "coeffs": coeffs.astype(np.float32),
        "proto": proto,
    }


def decode_outputs(outputs, conf_thresh, max_det, iou_thresh=0.7, imgsz=640, predist_postprocess="nms"):
    """自动检测输出格式并解码。

    根据 outputs 的数量和形状自动判断格式，调用对应的解码函数。
    seg 模型仅提取 det 部分（忽略 mask/proto），用于 mAP 评估。

    返回:
        [N, 6] = x1, y1, x2, y2, score, class_id。
    """
    fmt = detect_output_format(outputs)

    if fmt == "e2e":
        return decode_e2e(outputs, conf_thresh, max_det, iou_thresh)
    elif fmt == "one2one_raw":
        return decode_one2one_raw(outputs, conf_thresh, max_det)
    elif fmt == "yolov8_raw":
        return decode_yolov8_raw(outputs, conf_thresh, max_det, iou_thresh)
    elif fmt == "seg_e2e":
        result = decode_seg_e2e(outputs, conf_thresh, max_det, iou_thresh)
        boxes = result["boxes"]
        if boxes.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.column_stack([boxes, result["scores"], result["classes"]]).astype(np.float32)
    elif fmt == "seg_yolov8_topk":
        result = decode_seg_yolov8_topk(outputs, conf_thresh, max_det, iou_thresh)
        boxes = result["boxes"]
        if boxes.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.column_stack([boxes, result["scores"], result["classes"]]).astype(np.float32)
    elif fmt == "seg_yolov8_raw":
        result = decode_seg_yolov8_raw(outputs, conf_thresh, max_det, iou_thresh)
        boxes = result["boxes"]
        if boxes.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.column_stack([boxes, result["scores"], result["classes"]]).astype(np.float32)
    elif fmt == "pre_dist":
        return decode_predist(
            outputs, conf_thresh, max_det, iou_thresh, imgsz=imgsz, predist_postprocess=predist_postprocess
        )
    elif fmt == "pre_dfl":
        return decode_predfl(outputs, conf_thresh, max_det, iou_thresh, imgsz=imgsz)
    elif fmt == "seg_pre_dist":
        # `seg_pre_dist` 主线中，det 部分直接复用 `pre_dist` 路径（前两个输出）。
        result = decode_seg_predist(
            outputs, conf_thresh, max_det, iou_thresh, imgsz=imgsz, predist_postprocess=predist_postprocess
        )
        boxes = result["boxes"]
        if boxes.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.column_stack([boxes, result["scores"], result["classes"]]).astype(np.float32)
    elif fmt == "seg_pre_dfl":
        result = decode_seg_predfl(outputs, conf_thresh, max_det, iou_thresh, imgsz=imgsz)
        boxes = result["boxes"]
        if boxes.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.column_stack([boxes, result["scores"], result["classes"]]).astype(np.float32)
    else:
        raise ValueError(
            "未知输出格式: "
            f"{[o.shape for o in outputs]}，当前仅支持 e2e / one2one_raw / "
            "yolov8_raw / seg_e2e / seg_yolov8_topk / seg_yolov8_raw / "
            "pre_dist / pre_dfl / seg_pre_dist / seg_pre_dfl"
        )


def _make_anchor_grid(imgsz=640, strides=(8, 16, 32)):
    """生成 YOLO anchor grid，与 ONNX 图中的 Constant_15/16 一致。

    Constant_15[0, 0, :] = anchor_x（快速变化），Constant_15[0, 1, :] = anchor_y（慢速变化）。

    返回:
        anchors: [N, 2] float32，anchors[:, 0]=anchor_x，anchors[:, 1]=anchor_y
        stride_vals: [N] float32
    """
    input_h, input_w = parse_input_shape(imgsz)
    anchor_list = []
    stride_list = []
    for s in strides:
        h = input_h // s
        w = input_w // s
        xs = np.arange(w, dtype=np.float32) + 0.5
        ys = np.arange(h, dtype=np.float32) + 0.5
        # `np.meshgrid(xs, ys)` 默认使用 `indexing='xy'`：
        #   `grid_x[i,j] = xs[j]`（cx 变化更快），`grid_y[i,j] = ys[i]`（cy 变化更慢）
        grid_x, grid_y = np.meshgrid(xs, ys)
        cx = grid_x.ravel()
        cy = grid_y.ravel()
        anchor_list.append(np.stack([cx, cy], axis=1))
        stride_list.extend([s] * (h * w))
    anchors = np.concatenate(anchor_list, axis=0).astype(np.float32)
    strides_arr = np.array(stride_list, dtype=np.float32)
    return anchors, strides_arr


def decode_predfl(outputs, conf_thresh, max_det, iou_thresh=0.7, imgsz=640, strides=(8, 16, 32), reg_max=16):
    """解码 pre_dfl 格式输出 [1,4*reg_max,N]+[1,nc,N] -> [M,6]=x1,y1,x2,y2,score,cls。

    pre_dfl 格式：RKNN 截断点在 DFL Softmax 之前，NPU 输出原始 DFL logits，
    CPU 端在 float32 精度下完成 DFL 解码 + dist2bbox，避免 int8 量化破坏 Softmax 精度。

    DFL 解码流程（与 ONNX 图一致）：
    1. [1, 4*reg_max, N] → reshape [1, 4, reg_max, N]
    2. Transpose perm=[0,2,1,3] → [1, reg_max, 4, N]
    3. Softmax axis=1（在 reg_max 轴）
    4. DFL conv（加权求和，weights=[0,...,reg_max-1]）→ [1, 4, N]（ltrb）
    5. dist2bbox：anchor grid + strides → cxcywh → xyxy

    参数:
        outputs: 2 个 numpy 数组 [1,4*reg_max,N] 和 [1,nc,N]
        imgsz: 模型输入尺寸
        strides: 检测头步长，默认 (8, 16, 32)
        reg_max: DFL 分组数，默认 16
    """
    raw_dfl = outputs[0]  # [1, 4*reg_max, N]
    logits = outputs[1][0]  # [nc, N]

    b, c, n = raw_dfl.shape
    assert c == 4 * reg_max, f"pre_dfl output[0] 期望 {4 * reg_max} 通道，实际 {c}"

    # 执行 DFL 解码
    x = raw_dfl.reshape(b, 4, reg_max, n)  # [1, 4, reg_max, N]
    x = x.transpose(0, 2, 1, 3)  # [1, reg_max, 4, N]
    x_max = x.max(axis=1, keepdims=True)
    x = np.exp(x - x_max)
    x = x / x.sum(axis=1, keepdims=True)  # 在 reg_max 维度上做 softmax
    bins = np.arange(reg_max, dtype=np.float32)
    ltrb = (x * bins[None, :, None, None]).sum(axis=1)  # [1, 4, N]

    # 使用 dist2bbox 和 stride 还原检测框
    anchors, stride_vals = _make_anchor_grid(imgsz, strides)
    l, t, r, b_ = ltrb[0]  # 四个方向分量的长度均为 [N]
    ax, ay = anchors[:, 0], anchors[:, 1]
    x1 = (ax - l) * stride_vals
    y1 = (ay - t) * stride_vals
    x2 = (ax + r) * stride_vals
    y2 = (ay + b_) * stride_vals
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = x2 - x1
    bh = y2 - y1
    boxes_xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)  # [N, 4]

    # 分类得分
    logits = np.clip(logits, -88, 88)
    scores = sigmoid(logits)
    max_scores = np.max(scores, axis=0)
    class_ids = np.argmax(scores, axis=0)

    mask = max_scores > conf_thresh
    boxes_f = boxes_xyxy[mask]
    scores_f = max_scores[mask]
    cls_f = class_ids[mask].astype(np.float32)

    if len(scores_f) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if len(scores_f) > max_det:
        idx = np.argsort(-scores_f)[:max_det]
        boxes_f, scores_f, cls_f = boxes_f[idx], scores_f[idx], cls_f[idx]

    all_dets = np.column_stack([boxes_f, scores_f, cls_f])

    # 按类别 NMS
    keep_all = []
    for c_ in np.unique(cls_f):
        c_mask = cls_f == c_
        c_idx = np.where(c_mask)[0]
        kept = nms(boxes_f[c_mask], scores_f[c_mask], iou_thresh)
        keep_all.extend(c_idx[kept].tolist())

    all_dets = all_dets[keep_all]
    if len(all_dets) > max_det:
        all_dets = all_dets[np.argsort(-all_dets[:, 4])[:max_det]]
    return all_dets


def _decode_predist_boxes(raw_ltrb, imgsz=640, strides=(8, 16, 32), indices=None):
    """将 pre_dist 的 raw ltrb 解码为 xyxy 像素坐标。"""
    anchors, stride_vals = _make_anchor_grid(imgsz, strides)
    anchors_f16 = anchors.astype(np.float16)
    strides_f16 = stride_vals.astype(np.float16)
    ltrb = raw_ltrb[0].astype(np.float16)

    if indices is None:
        sel = slice(None)
    else:
        sel = np.asarray(indices, dtype=np.int64)

    x1 = (anchors_f16[sel, 0] - ltrb[0, sel]) * strides_f16[sel]
    y1 = (anchors_f16[sel, 1] - ltrb[1, sel]) * strides_f16[sel]
    x2 = (anchors_f16[sel, 0] + ltrb[2, sel]) * strides_f16[sel]
    y2 = (anchors_f16[sel, 1] + ltrb[3, sel]) * strides_f16[sel]
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def _select_predist_simple(scores, conf_thresh, max_det):
    """每个 anchor 仅保留最高类别，不做 NMS。"""
    max_scores = np.max(scores, axis=0)
    class_ids = np.argmax(scores, axis=0).astype(np.float32)

    keep = np.where(max_scores > conf_thresh)[0]
    if keep.size == 0:
        return keep, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    order = np.argsort(-max_scores[keep], kind="stable")
    keep = keep[order]
    if keep.size > max_det:
        keep = keep[:max_det]
    return keep, class_ids[keep], max_scores[keep]


def _select_predist_exact(scores, conf_thresh, max_det):
    """复刻 YOLO26 _postprocess_export() 的 top-k/gather 语义。"""
    nc, n = scores.shape
    k = min(max_det, n)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    score_max = scores.max(axis=0)  # [N]
    ori_index = np.argsort(-score_max, kind="stable")[:k]  # [k]

    gathered_scores = scores[:, ori_index].T  # [k, nc]
    flat_scores = gathered_scores.reshape(-1)  # [k*nc]
    flat_index = np.argsort(-flat_scores, kind="stable")[:k]

    scores_topk = flat_scores[flat_index]
    anchor_idx = flat_index // nc
    class_idx = (flat_index % nc).astype(np.float32)
    final_idx = ori_index[anchor_idx]

    mask = scores_topk > conf_thresh
    return final_idx[mask], class_idx[mask], scores_topk[mask]


def decode_predist(
    outputs, conf_thresh, max_det, iou_thresh=0.7, imgsz=640, strides=(8, 16, 32), predist_postprocess="nms"
):
    """解码 pre_dist 格式输出 [1,4,N]+[1,nc,N] -> [M,6]=x1,y1,x2,y2,score,cls。

    pre_dist 格式：RKNN 截断点在 dist2bbox+strides 之前，NPU 输出原始 ltrb
    偏移量（步幅空间），CPU 端用 float16 完成 dist2bbox + strides 乘法。

    适用于 reg_max=1 的 YOLO26 模型（无 DFL Softmax 步骤）。pre_dist 在步幅空间
    保留 raw ltrb（0~10 步幅单位）并在 CPU 侧解码，INT8 分辨率约 0.04 步幅单位
    ≈ 0.32px（P3），更适合当前 RKNN 检测主线。

    解码流程（FP16）：
      x1 = (anchor_x - ltrb[l]) * stride
      y1 = (anchor_y - ltrb[t]) * stride
      x2 = (anchor_x + ltrb[r]) * stride
      y2 = (anchor_y + ltrb[b]) * stride
    """
    raw_ltrb = outputs[0]  # [1, 4, N] float32（RKNN 自动反量化）
    logits = outputs[1][0]  # [nc, N]   float32

    logits_f32 = np.clip(logits.astype(np.float32), -88, 88)
    scores = sigmoid(logits_f32)

    if predist_postprocess == "nmsfree_simple":
        keep_idx, cls_f, scores_f = _select_predist_simple(scores, conf_thresh, max_det)
        if keep_idx.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        boxes_f = _decode_predist_boxes(raw_ltrb, imgsz, strides, keep_idx)
        return np.column_stack([boxes_f, scores_f, cls_f]).astype(np.float32)

    if predist_postprocess == "nmsfree_exact":
        keep_idx, cls_f, scores_f = _select_predist_exact(scores, conf_thresh, max_det)
        if keep_idx.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        boxes_f = _decode_predist_boxes(raw_ltrb, imgsz, strides, keep_idx)
        return np.column_stack([boxes_f, scores_f, cls_f]).astype(np.float32)

    if predist_postprocess != "nms":
        raise ValueError(f"未知 predist_postprocess: {predist_postprocess}")

    boxes_xyxy = _decode_predist_boxes(raw_ltrb, imgsz, strides)
    max_scores = np.max(scores, axis=0)
    class_ids = np.argmax(scores, axis=0)

    mask = max_scores > conf_thresh
    boxes_f = boxes_xyxy[mask]
    scores_f = max_scores[mask]
    cls_f = class_ids[mask].astype(np.float32)

    if len(scores_f) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if len(scores_f) > max_det:
        idx = np.argsort(-scores_f)[:max_det]
        boxes_f, scores_f, cls_f = boxes_f[idx], scores_f[idx], cls_f[idx]

    all_dets = np.column_stack([boxes_f, scores_f, cls_f])
    keep_all = []
    for c_ in np.unique(cls_f):
        c_mask = cls_f == c_
        c_idx = np.where(c_mask)[0]
        kept = nms(boxes_f[c_mask], scores_f[c_mask], iou_thresh)
        keep_all.extend(c_idx[kept].tolist())

    all_dets = all_dets[keep_all]
    if len(all_dets) > max_det:
        all_dets = all_dets[np.argsort(-all_dets[:, 4])[:max_det]]
    return all_dets


def decode_seg_predist(
    outputs, conf_thresh, max_det, iou_thresh=0.7, imgsz=640, strides=(8, 16, 32), predist_postprocess="nms"
):
    """解码 seg_pre_dist 格式四输出，返回 bbox + mask coeff + proto。

    输入输出契约（seg_pre_dist）：
      outputs[0]: raw_ltrb  [1, 4, N]     — 步幅空间原始 ltrb
      outputs[1]: cls_logits [1, nc, N]   — 未 sigmoid 的分类 logits
      outputs[2]: mask_coeff [1, nm, N]   — mask 系数
      outputs[3]: proto      [1, nm, H, W] — proto 特征图

    后处理流程：
      1. 对 cls_logits 做 sigmoid
      2. 按 predist_postprocess 选出候选 anchor 及其 keep_idx
      3. _decode_predist_boxes() 解码 bbox
      4. mask_coeff[:, keep_idx] 做 gather
      5. 返回 dict

    返回:
        dict 包含:
          boxes:   [M, 4] xyxy 像素坐标
          scores:  [M]    置信度
          classes: [M]    类别 id (float32)
          coeffs:  [M, nm] mask 系数
          proto:   [nm, H, W] proto 特征图
    """
    raw_ltrb = outputs[0]  # [1, 4, N]
    logits = outputs[1][0]  # [nc, N]
    mask_coeff_full = outputs[2][0]  # [nm, N]
    proto = outputs[3][0] if outputs[3].ndim == 4 else outputs[3]  # [nm, H, W]

    logits_f32 = np.clip(logits.astype(np.float32), -88, 88)
    scores = sigmoid(logits_f32)

    empty = {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.float32),
        "coeffs": np.zeros((0, mask_coeff_full.shape[0]), dtype=np.float32),
        "proto": proto,
    }

    if predist_postprocess == "nmsfree_simple":
        keep_idx, cls_f, scores_f = _select_predist_simple(scores, conf_thresh, max_det)
        if keep_idx.size == 0:
            return empty
        boxes_f = _decode_predist_boxes(raw_ltrb, imgsz, strides, keep_idx)
        coeffs_f = mask_coeff_full[:, keep_idx].T  # [M, nm]
        return {
            "boxes": boxes_f,
            "scores": scores_f,
            "classes": cls_f,
            "coeffs": coeffs_f.astype(np.float32),
            "proto": proto,
        }

    if predist_postprocess == "nmsfree_exact":
        keep_idx, cls_f, scores_f = _select_predist_exact(scores, conf_thresh, max_det)
        if keep_idx.size == 0:
            return empty
        boxes_f = _decode_predist_boxes(raw_ltrb, imgsz, strides, keep_idx)
        coeffs_f = mask_coeff_full[:, keep_idx].T  # [M, nm]
        return {
            "boxes": boxes_f,
            "scores": scores_f,
            "classes": cls_f,
            "coeffs": coeffs_f.astype(np.float32),
            "proto": proto,
        }

    if predist_postprocess != "nms":
        raise ValueError(f"未知 predist_postprocess: {predist_postprocess}")

    boxes_xyxy = _decode_predist_boxes(raw_ltrb, imgsz, strides)
    pred = np.concatenate(
        [
            _xyxy_to_xywh(boxes_xyxy).T,
            scores.astype(np.float32),
            mask_coeff_full.astype(np.float32),
        ],
        axis=0,
    )[None, ...]
    return _segment_multilabel_nms(pred, proto, scores.shape[0], conf_thresh, max_det, iou_thresh)


def decode_seg_predfl(outputs, conf_thresh, max_det, iou_thresh=0.7, imgsz=640, strides=(8, 16, 32), reg_max=16):
    """解码 `seg_pre_dfl` 输出（4 或 5 输出），返回 bbox + mask coeff + proto。

    输入输出契约（seg_pre_dfl）：
      outputs[0]: raw_dfl            [1, 4*reg_max, N]（legacy）或 [1, N, 4*reg_max]（transposed）
      outputs[1]: cls_logits         [1, nc, N]
      outputs[2]: mask_coeff         [1, nm, N]
      outputs[3]: proto              [1, nm, H, W]
      outputs[4]: score_sum          [1, 1, N]（可选，5 输出时存在）

    后处理流程：
      1. 对 raw_dfl 做 softmax-expectation 得到 ltrb；
      2. 按 anchor grid + stride 解码 xyxy；
      3. 对 cls_logits 做 sigmoid + per-class NMS；
      4. 按最终 keep_idx gather mask_coeff。
    """
    raw_dfl = outputs[0]
    logits = outputs[1][0]
    mask_coeff_full = outputs[2][0]
    proto = outputs[3][0] if outputs[3].ndim == 4 else outputs[3]

    # 自动检测布局：转置 [1, N, 4*reg_max] 的 dim2 是 4*reg_max（>4 且整除 4，
    # 且 N > 4*reg_max）；legacy [1, 4*reg_max, N] 不满足 dim1 > dim2。
    if raw_dfl.ndim == 3 and raw_dfl.shape[2] > 4 and raw_dfl.shape[2] % 4 == 0 and raw_dfl.shape[1] > raw_dfl.shape[2]:
        raw_dfl = raw_dfl.transpose(0, 2, 1)

    b, c, n = raw_dfl.shape
    if c % 4 != 0 or c <= 4:
        raise ValueError(f"seg_pre_dfl output[0] 通道应为 4*reg_max 且 >4，实际 {c}")
    reg_max = c // 4 if reg_max * 4 != c else reg_max

    empty = {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.float32),
        "coeffs": np.zeros((0, mask_coeff_full.shape[0]), dtype=np.float32),
        "proto": proto,
    }

    x = raw_dfl.reshape(b, 4, reg_max, n)
    x = x.transpose(0, 2, 1, 3)
    x_max = x.max(axis=1, keepdims=True)
    x = np.exp(x - x_max)
    x = x / x.sum(axis=1, keepdims=True)
    bins = np.arange(reg_max, dtype=np.float32)
    ltrb = (x * bins[None, :, None, None]).sum(axis=1)

    anchors, stride_vals = _make_anchor_grid(imgsz, strides)
    l, t, r, b_ = ltrb[0]
    ax, ay = anchors[:, 0], anchors[:, 1]
    x1 = (ax - l) * stride_vals
    y1 = (ay - t) * stride_vals
    x2 = (ax + r) * stride_vals
    y2 = (ay + b_) * stride_vals
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    logits = np.clip(logits.astype(np.float32), -88, 88)
    scores = sigmoid(logits)
    pred = np.concatenate(
        [
            _xyxy_to_xywh(boxes_xyxy).T,
            scores.astype(np.float32),
            mask_coeff_full.astype(np.float32),
        ],
        axis=0,
    )[None, ...]
    return _segment_multilabel_nms(pred, proto, scores.shape[0], conf_thresh, max_det, iou_thresh)


def nms(boxes_xyxy, scores, iou_threshold=0.7):
    """经典贪心 NMS（非极大值抑制）。

    按得分降序逐个保留框，抑制与已保留框 IoU 超过阈值的低分框。

    参数:
        boxes_xyxy: [N, 4] xyxy 格式的检测框。
        scores: [N] 对应的置信度得分。
        iou_threshold: IoU 抑制阈值，默认 0.7。

    返回:
        np.array: 保留的索引数组。
    """
    if len(scores) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=int)


def scale_boxes(dets, scale, pad_w, pad_h):
    """将检测框从 letterbox 空间还原到原图像素坐标。

    反向 letterbox 变换：先减去填充偏移，再除以缩放比例。

    参数:
        dets: [N, 6+] 检测结果，前 4 列为 x1,y1,x2,y2。
        scale: letterbox 缩放比例。
        pad_w, pad_h: letterbox 填充偏移量。

    返回:
        还原后的检测结果副本。
    """
    if len(dets) == 0:
        return dets
    dets = dets.copy()
    dets[:, 0] = (dets[:, 0] - pad_w) / scale
    dets[:, 1] = (dets[:, 1] - pad_h) / scale
    dets[:, 2] = (dets[:, 2] - pad_w) / scale
    dets[:, 3] = (dets[:, 3] - pad_h) / scale
    return dets


def _assemble_seg_masks_to_original(seg_result, imgsz, orig_h, orig_w, scale, pad_w, pad_h):
    """将 seg_pre_dist / seg_pre_dfl 的 coeff/proto 组装为原图尺寸二值掩码。

    参数:
        seg_result: decode_seg_predist() 或 decode_seg_predfl() 返回的结果字典。
        imgsz: 模型输入边长。
        orig_h: 原图高度。
        orig_w: 原图宽度。
        scale: letterbox 缩放比例。
        pad_w: letterbox 左右 padding。
        pad_h: letterbox 上下 padding。

    返回:
        np.ndarray: [N, orig_h, orig_w]，uint8 二值掩码。
    """
    import cv2

    input_h, input_w = parse_input_shape(imgsz)
    coeffs = seg_result["coeffs"]
    proto = seg_result["proto"]
    if coeffs.shape[0] == 0:
        return np.zeros((0, orig_h, orig_w), dtype=np.uint8)

    nm, proto_h, proto_w = proto.shape
    masks_raw = coeffs @ proto.reshape(nm, -1)  # [N, proto_h * proto_w]
    masks_raw = sigmoid(masks_raw).reshape(-1, proto_h, proto_w)

    masks_lb = np.zeros((masks_raw.shape[0], input_h, input_w), dtype=np.float32)
    for i, mask_i in enumerate(masks_raw):
        masks_lb[i] = cv2.resize(mask_i, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    boxes = seg_result["boxes"]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(input_w, int(x2) + 1)
        y2 = min(input_h, int(y2) + 1)
        cropped = np.zeros_like(masks_lb[i])
        if x2 > x1 and y2 > y1:
            cropped[y1:y2, x1:x2] = masks_lb[i][y1:y2, x1:x2]
        masks_lb[i] = cropped

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    masks_crop = masks_lb[:, pad_h : pad_h + new_h, pad_w : pad_w + new_w]
    masks_orig = np.zeros((masks_crop.shape[0], orig_h, orig_w), dtype=np.uint8)
    for i, mask_i in enumerate(masks_crop):
        resized = cv2.resize(mask_i, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        masks_orig[i] = (resized > 0.5).astype(np.uint8)
    return masks_orig


def _assemble_seg_masks_for_eval(seg_result, imgsz, upsample=True):
    """按指定口径组装分割 mask。

    参数:
        seg_result: decode 结果，含 coeffs/proto/boxes。
        imgsz: 输入分辨率（用于 native 口径的上采样目标尺寸）。
        upsample: True → native 口径（640×640）；False → fast 口径（proto 分辨率）。
    """
    coeffs = seg_result["coeffs"]
    proto = seg_result["proto"]
    boxes = seg_result["boxes"]
    input_shape = parse_input_shape(imgsz)
    if coeffs.shape[0] == 0:
        mh, mw = proto.shape[1], proto.shape[2]
        out_h, out_w = input_shape if upsample else (mh, mw)
        return np.zeros((0, out_h, out_w), dtype=np.uint8)

    return _process_mask_for_eval(proto, coeffs, boxes, input_shape, upsample=upsample)


def _process_mask_for_eval(proto, coeffs, boxes, shape, upsample=True):
    """组装分割 mask，支持双口径。

    参数:
        proto: mask proto，形状 (c, mh, mw)。
        coeffs: mask 系数，形状 (n, c)。
        boxes: 检测框，xyxy，letterbox 坐标系，形状 (n, 4)。
        shape: 输入分辨率 (ih, iw)，letterbox 空间。
        upsample: 口径选择：

            - ``True``（**native**）：等价于 ultralytics ``process_mask_native``
              / ``model.val(save_json=True)``。coeff @ proto → 双线性上采样到
              ``shape``（640×640）→ crop → > 0。GT mask 在 640×640 空间比较。
            - ``False``（**fast**）：等价于 ultralytics 默认 ``model.val()`` 打印口径。
              coeff @ proto → crop（在 proto 分辨率 160×160，boxes 同比缩放）→ > 0。
              GT mask 在 160×160 空间比较。
    """
    import cv2 as _cv2

    c, mh, mw = proto.shape
    ih, iw = shape

    masks = coeffs.astype(np.float32) @ proto.astype(np.float32).reshape(c, -1)
    masks = masks.reshape(-1, mh, mw)

    if upsample:
        # native 口径：先上采样到输入分辨率，`boxes` 无需缩放（已位于 letterbox 坐标系）。
        n = masks.shape[0]
        masks_up = np.empty((n, ih, iw), dtype=np.float32)
        for i in range(n):
            masks_up[i] = _cv2.resize(masks[i], (iw, ih), interpolation=_cv2.INTER_LINEAR)
        masks_up = _crop_mask_for_eval(masks_up, boxes.astype(np.float32))
        return (masks_up > 0.0).astype(np.uint8)
    else:
        # fast 口径：先把 `boxes` 缩放到 proto 分辨率，再直接在 proto 空间裁剪，不做上采样。
        width_ratio = mw / iw
        height_ratio = mh / ih
        scaled_boxes = boxes.astype(np.float32) * np.array(
            [width_ratio, height_ratio, width_ratio, height_ratio], dtype=np.float32
        )
        masks = _crop_mask_for_eval(masks, scaled_boxes)
        return (masks > 0.0).astype(np.uint8)


def _crop_mask_for_eval(masks, boxes):
    """按检测框裁剪 mask（适用于任意分辨率）。

    采用与官方 GPU 验证路径一致的浮点边界比较语义：像素中心索引需满足
    ``x >= x1 && x < x2 && y >= y1 && y < y2``，等价于切片边界使用
    ``ceil()``。仅拷贝框内区域，避免分配大型 bool 张量。
    """
    n, h, w = masks.shape
    out = np.zeros_like(masks)
    b = np.ceil(boxes).astype(np.int32)
    for i in range(n):
        x1, y1, x2, y2 = b[i]
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w)
        y2 = min(int(y2), h)
        if x2 > x1 and y2 > y1:
            out[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    return out


def _resize_gt_masks_for_eval(masks, target_shape):
    """按官方验证流程调整 GT mask 分辨率。

    参数:
        masks: GT mask，形状为 ``(N, H, W)``。
        target_shape: 目标尺寸 ``(height, width)``。

    返回:
        np.ndarray: ``uint8`` 二值 mask。若尺寸不同，使用双线性插值并按
        ``> 0.5`` 阈值化，匹配官方 ``F.interpolate(..., align_corners=False)``
        后的处理方式。
    """
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if masks.shape[1:] == (target_h, target_w):
        return masks.astype(np.uint8, copy=False)

    import cv2 as _cv2

    resized = np.zeros((masks.shape[0], target_h, target_w), dtype=np.uint8)
    for i, mask_i in enumerate(masks):
        mask_f = _cv2.resize(
            mask_i.astype(np.float32),
            (target_w, target_h),
            interpolation=_cv2.INTER_LINEAR,
        )
        resized[i] = (mask_f > 0.5).astype(np.uint8)
    return resized


def _letterbox_boxes(boxes, scale, pad_w, pad_h, imgsz):
    """将原图 xyxy 框映射到 letterbox 输入空间。"""
    if len(boxes) == 0:
        return boxes.astype(np.float32)
    input_h, input_w = parse_input_shape(imgsz)
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = out[:, [0, 2]] * scale + pad_w
    out[:, [1, 3]] = out[:, [1, 3]] * scale + pad_h
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, input_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, input_h)
    return out


def _coco_rle_to_string(counts):
    """将 RLE 计数列表压缩为 COCO 兼容字符串。"""
    encoded = []
    for i, value in enumerate(counts):
        x = int(value)
        if i > 2:
            x -= int(counts[i - 2])
        while True:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            c += 48
            encoded.append(chr(c))
            if not more:
                break
    return "".join(encoded)


def _encode_binary_masks_to_coco_rles(masks):
    """将多张二值掩码编码为 COCO RLE。"""
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.size == 0:
        return []

    # COCO RLE 需要按列优先展开，等价于先转置，再按 C-order 扁平化。
    pixels = masks.transpose(0, 2, 1).reshape(masks.shape[0], -1)
    transitions = pixels[:, 1:] != pixels[:, :-1]
    row_idx, col_idx = np.where(transitions)
    col_idx = col_idx + 1

    counts_all = []
    for i in range(pixels.shape[0]):
        positions = col_idx[row_idx == i]
        if positions.size:
            count = np.diff(positions).astype(np.int64).tolist()
            count.insert(0, int(positions[0]))
            count.append(int(len(pixels[i]) - positions[-1]))
        else:
            count = [int(len(pixels[i]))]
        if int(pixels[i][0]) == 1:
            count = [0, *count]
        counts_all.append(count)

    h, w = map(int, masks.shape[1:3])
    return [{"size": [h, w], "counts": _coco_rle_to_string(c)} for c in counts_all]


def _limit_coco_result_to_topk(result, topk=100):
    """按 score 降序保留 COCO 官方 maxDets 使用的 top-k 预测。"""
    scores = np.asarray(result.get("scores", []))
    n_pred = int(scores.shape[0]) if scores.ndim > 0 else 0
    if n_pred <= topk:
        return result

    keep = np.argsort(-scores)[:topk]
    limited = {}
    for key, value in result.items():
        arr = np.asarray(value) if isinstance(value, np.ndarray) else value
        if isinstance(arr, np.ndarray) and arr.ndim > 0 and arr.shape[0] == n_pred:
            limited[key] = arr[keep]
        else:
            limited[key] = value
    return limited


def _limit_coco_dets_to_topk(dets, topk=100):
    """按 score 降序保留 COCO 官方 maxDets 使用的 top-k bbox 预测。"""
    if len(dets) <= topk:
        return dets
    keep = np.argsort(-dets[:, 4])[:topk]
    return dets[keep]


def _summarize_coco_eval(coco_eval, prefix=""):
    """从 COCOeval 对象提取 P/R/mAP 指标，可选添加前缀。"""
    stats = coco_eval.stats
    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]

    p_at_50 = precision[0, :, :, 0, 2]
    p_at_50 = p_at_50[p_at_50 > -1]
    mean_p = float(np.mean(p_at_50)) if len(p_at_50) > 0 else 0.0

    r_at_50 = recall[0, :, 0, 2]
    r_at_50 = r_at_50[r_at_50 > -1]
    mean_r = float(np.mean(r_at_50)) if len(r_at_50) > 0 else 0.0

    key = lambda name: f"{prefix}{name}" if prefix else name
    return {
        key("P"): round(mean_p, 4),
        key("R"): round(mean_r, 4),
        key("mAP50"): round(float(stats[1]), 4),
        key("mAP50-95"): round(float(stats[0]), 4),
        key("AP_small"): round(float(stats[3]), 4),
        key("AP_medium"): round(float(stats[4]), 4),
        key("AP_large"): round(float(stats[5]), 4),
        key("AR100"): round(float(stats[8]), 4),
    }


# ── 推理后端 ───────────────────────────────────────────────────────────


class OnnxBackend:
    """ONNX Runtime 推理后端（用于 PC 端评估）。

    使用 CPU ExecutionProvider 进行推理，输入图像自动完成
    uint8->0-1 float32 归一化 + HWC->CHW 转换。
    """

    def __init__(self, model_path, imgsz):
        """初始化 ONNX 会话。

        参数:
            model_path: .onnx 模型文件路径。
            imgsz: 输入图像尺寸（正方形边长）。
        """
        import onnxruntime as ort

        self.sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.imgsz = imgsz

    def infer(self, img_hwc_uint8):
        """执行单张图片推理，返回模型原始输出列表。"""
        x = prepare_onnx_input(img_hwc_uint8)
        return self.sess.run(None, {self.input_name: x})

    def release(self):
        """释放 ONNX 会话资源（无操作）。"""
        pass


class RknnBackend:
    """RKNN Lite 推理后端（用于 RK3588 板端评估）。

    支持 NPU 核心绑定（core0/core01/core012），输入 uint8 HWC 图像，
    归一化由 RKNN 运行时根据量化配置自动处理。
    """

    def __init__(self, model_path, imgsz, core_mask_name="core0"):
        """加载 RKNN 模型并初始化运行时。

        参数:
            model_path: .rknn 模型文件路径。
            imgsz: 输入图像尺寸。
            core_mask_name: NPU 核心绑定模式 (auto/core0/core01/core012)。
        """
        from rknnlite.api import RKNNLite

        core_masks = {
            "auto": RKNNLite.NPU_CORE_AUTO,
            "core0": RKNNLite.NPU_CORE_0,
            "core01": RKNNLite.NPU_CORE_0_1,
            "core012": RKNNLite.NPU_CORE_0_1_2,
        }
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        assert ret == 0, f"load_rknn 失败: {ret}"
        ret = self.rknn.init_runtime(core_mask=core_masks[core_mask_name])
        assert ret == 0, f"init_runtime 失败: {ret}"
        self.imgsz = imgsz

    def infer(self, img_hwc_uint8):
        """执行单张图片推理，输入 uint8 HWC 图像。"""
        x = prepare_rknn_input(img_hwc_uint8)
        return self.rknn.inference(inputs=[x], data_format="nhwc")

    def release(self):
        """释放 RKNN 运行时资源。"""
        self.rknn.release()


class RknnSimBackend:
    """RKNN 模拟器推理后端（用于 PC 端无硬件的模拟测试）。

    使用 RKNN-Toolkit2 的模拟器模式，不需要实际 NPU 硬件，
    适合在开发机上验证模型输出正确性。
    """

    def __init__(self, model_path, imgsz):
        """加载 RKNN 模型并初始化模拟器运行时。

        参数:
            model_path: .rknn 模型文件路径。
            imgsz: 输入图像尺寸。
        """
        from rknn.api import RKNN

        self.rknn = RKNN()
        ret = self.rknn.load_rknn(model_path)
        assert ret == 0, f"load_rknn 失败: {ret}"
        ret = self.rknn.init_runtime(target=None)
        assert ret == 0, f"init_runtime（模拟器）失败: {ret}"
        self.imgsz = imgsz

    def infer(self, img_hwc_uint8):
        """执行单张图片模拟推理。"""
        x = prepare_rknn_input(img_hwc_uint8)
        return self.rknn.inference(inputs=[x], data_format="nhwc")

    def release(self):
        """释放 RKNN 模拟器资源。"""
        self.rknn.release()


# ── mAP 计算 ───────────────────────────────────────────────────


def compute_iou_matrix(pred_boxes, gt_boxes):
    """计算预测框/真实框 IoU 矩阵。

    参数:
        pred_boxes: [M, 4] 预测框 (xyxy 格式)。
        gt_boxes: [N, 4] 真实框 (xyxy 格式)。

    返回:
        [M, N] IoU 矩阵，元素 (i,j) 表示第 i 个预测框与第 j 个真实框的 IoU。
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    pred = pred_boxes.astype(np.float32)
    gt = gt_boxes.astype(np.float32)
    lt = np.maximum(pred[:, None, :2], gt[None, :, :2])
    rb = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_pred = np.prod(pred[:, 2:] - pred[:, :2], axis=1)
    area_gt = np.prod(gt[:, 2:] - gt[:, :2], axis=1)
    return (inter / (area_pred[:, None] + area_gt[None, :] - inter + 1e-7)).astype(np.float32)


def compute_mask_iou_matrix(pred_masks, gt_masks):
    """计算预测 mask/真实 mask IoU 矩阵。"""
    if pred_masks.shape[0] == 0 or gt_masks.shape[0] == 0:
        return np.zeros((pred_masks.shape[0], gt_masks.shape[0]), dtype=np.float32)

    pred = pred_masks.reshape(pred_masks.shape[0], -1).astype(np.float32)
    gt = gt_masks.reshape(gt_masks.shape[0], -1).astype(np.float32)
    intersection = np.clip(pred @ gt.T, a_min=0.0, a_max=None)
    union = pred.sum(axis=1)[:, None] + gt.sum(axis=1)[None, :] - intersection
    return (intersection / (union + 1e-7)).astype(np.float32)


def _match_predictions_from_iou(iou_mat, pred_scores, pred_classes, gt_classes, iou_thresholds):
    """按基准验证口径匹配预测。"""
    n_pred = len(pred_scores)
    n_thresh = len(iou_thresholds)
    correct = np.zeros((n_pred, n_thresh), dtype=bool)

    if n_pred == 0 or len(gt_classes) == 0 or iou_mat.size == 0:
        return correct

    # 使用 LxD 矩阵：L 为 GT，D 为 detections。
    iou = iou_mat.T
    correct_class = gt_classes[:, None] == pred_classes
    iou = iou * correct_class
    for ti, iou_thresh in enumerate(iou_thresholds):
        matches = np.nonzero(iou >= iou_thresh)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), ti] = True

    return correct


def match_predictions(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_thresholds):
    """将预测结果与真实标注在多个 IoU 阈值下进行匹配。

    按得分降序贪心匹配：对每个预测框，找同类别中 IoU 最高的未匹配真实框，
    若超过阈值则记为 TP。每个真实框只能被匹配一次。

    参数:
        pred_boxes: [M, 4] 预测框。
        pred_scores: [M] 置信度。
        pred_classes: [M] 预测类别 ID。
        gt_boxes: [N, 4] 真实框。
        gt_classes: [N] 真实类别 ID。
        iou_thresholds: [T] IoU 阈值数组（如 0.5:0.05:0.95）。

    返回:
        correct [M, T] 布尔矩阵，True 表示预测 i 在阈值 t 下为 TP。
    """
    if len(pred_scores) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_scores), len(iou_thresholds)), dtype=bool)

    iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
    return _match_predictions_from_iou(iou_mat, pred_scores, pred_classes, gt_classes, iou_thresholds)


def compute_ap(recall, precision):
    """使用 101 点插值计算单类别 AP。

    先将 precision 转为单调递减，再在 [0, 1] 等距 101 点上插值求均值。

    参数:
        recall: 单调递增的召回率数组。
        precision: 对应的精确率数组。

    返回:
        tuple[float, np.ndarray, np.ndarray]: AP 值、precision 包络、补哨兵后的 recall。
    """
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))

    # 将 precision 转为单调递减
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 101 点插值
    x = np.linspace(0, 1, 101)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    ap = trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def smooth(y, f=0.05):
    """盒滤波平滑函数。"""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def ap_per_class(tp, conf, pred_cls, target_cls, nc, eps=1e-16):
    """计算逐类别 AP/P/R/F1。

    返回每个有 GT 类别在 max-F1 点的 P/R/F1，以及 10 个 IoU 阈值上的 AP。
    """
    order = np.argsort(-conf)
    tp = tp[order]
    conf = conf[order]
    pred_cls = pred_cls[order]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    x = np.linspace(0, 1, 1000)
    n_unique = unique_classes.shape[0]

    ap = np.zeros((n_unique, tp.shape[1]), dtype=np.float64)
    p_curve = np.zeros((n_unique, 1000), dtype=np.float64)
    r_curve = np.zeros((n_unique, 1000), dtype=np.float64)

    for ci, c in enumerate(unique_classes):
        cls_mask = pred_cls == c
        n_l = nt[ci]
        n_p = int(cls_mask.sum())
        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[cls_mask].astype(np.float64)).cumsum(0)
        tpc = tp[cls_mask].astype(np.float64).cumsum(0)

        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[cls_mask], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc + eps)
        p_curve[ci] = np.interp(-x, -conf[cls_mask], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            ap[ci, j], _, _ = compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    i = smooth(f1_curve.mean(0), 0.1).argmax() if n_unique else 0

    p = p_curve[:, i] if n_unique else np.zeros(0, dtype=np.float64)
    r = r_curve[:, i] if n_unique else np.zeros(0, dtype=np.float64)
    f1 = f1_curve[:, i] if n_unique else np.zeros(0, dtype=np.float64)
    return p, r, f1, ap, unique_classes.astype(int)


def compute_metrics(all_stats, nc):
    """从累积的匹配统计计算逐类别和整体 mAP 指标。

    参数:
        all_stats: [(correct[M,T], scores[M], classes[M], n_gt_per_class[nc]), ...]
        nc: 类别数。

    返回:
        dict: 包含 P, R, mAP50, mAP50-95 以及逐类别明细的指标字典。
    """
    # 拼接所有预测结果
    if not all_stats:
        return _empty_metrics(nc)

    tp_list = [s[0] for s in all_stats]
    score_list = [s[1] for s in all_stats]
    cls_list = [s[2] for s in all_stats]

    tp = np.concatenate(tp_list, axis=0) if tp_list else np.zeros((0, 10), dtype=bool)
    scores = np.concatenate(score_list) if score_list else np.array([])
    pred_cls = np.concatenate(cls_list) if cls_list else np.array([])

    # 统计每个类别的真实标注总数
    n_gt = np.zeros(nc, dtype=int)
    for s in all_stats:
        n_gt += s[3]

    ap_per_class_vals = np.zeros((nc, 10), dtype=np.float64)  # [nc, T]
    p_per_class = np.zeros(nc, dtype=np.float64)
    r_per_class = np.zeros(nc, dtype=np.float64)
    n_det_per_class = np.zeros(nc, dtype=int)

    for c in range(nc):
        n_det_per_class[c] = int((pred_cls == c).sum())

    # 只对有真实标注的类别计算均值
    valid_cls = n_gt > 0
    if not valid_cls.any():
        return _empty_metrics(nc)

    target_cls = np.repeat(np.arange(nc, dtype=np.int32), n_gt)
    if len(tp) > 0 and len(target_cls) > 0:
        p, r, _f1, ap, ap_class_index = ap_per_class(
            tp.astype(bool),
            scores.astype(np.float64),
            pred_cls.astype(np.float64),
            target_cls.astype(np.float64),
            nc,
        )
        p_per_class[ap_class_index] = p
        r_per_class[ap_class_index] = r
        ap_per_class_vals[ap_class_index] = ap

    ap50 = ap_per_class_vals[:, 0]  # 每个类别的 AP@0.5
    ap50_95 = ap_per_class_vals.mean(axis=1)  # 每个类别的 AP@0.5:0.95

    return {
        "P": float(p_per_class[valid_cls].mean()),
        "R": float(r_per_class[valid_cls].mean()),
        "mAP50": float(ap50[valid_cls].mean()),
        "mAP50-95": float(ap50_95[valid_cls].mean()),
        "ap50_per_class": ap50.tolist(),
        "ap50_95_per_class": ap50_95.tolist(),
        "p_per_class": p_per_class.tolist(),
        "r_per_class": r_per_class.tolist(),
        "n_gt_per_class": n_gt.tolist(),
        "n_det_per_class": n_det_per_class.tolist(),
    }


def _empty_metrics(nc):
    """返回全零的指标字典（无检测结果或无真实标注时的回退值）。"""
    return {
        "P": 0.0,
        "R": 0.0,
        "mAP50": 0.0,
        "mAP50-95": 0.0,
        "ap50_per_class": [0.0] * nc,
        "ap50_95_per_class": [0.0] * nc,
        "p_per_class": [0.0] * nc,
        "r_per_class": [0.0] * nc,
        "n_gt_per_class": [0] * nc,
        "n_det_per_class": [0] * nc,
    }


def _print_per_class_table(
    title,
    metrics,
    names,
    nc,
    n_total,
    total_dets,
    p_key="P",
    r_key="R",
    map50_key="mAP50",
    map5095_key="mAP50-95",
    n_gt_key="n_gt_per_class",
    n_det_key="n_det_per_class",
    p_pc_key="p_per_class",
    r_pc_key="r_per_class",
    ap50_pc_key="ap50_per_class",
    ap5095_pc_key="ap50_95_per_class",
):
    """打印逐类别指标表，可复用于 detect / segment(box) / segment(mask)。"""
    print(f"\n  {title}")
    print(f"  {'类别':<20} {'图片':>7} {'GT':>7} {'检测':>7} {'P':>7} {'R':>7} {'mAP50':>7} {'mAP50-95':>9}")
    print(f"  {'-' * 77}")

    n_gt_per_class = metrics[n_gt_key]
    n_det_per_class = metrics[n_det_key]
    p_per_class = metrics[p_pc_key]
    r_per_class = metrics[r_pc_key]
    ap50_per_class = metrics[ap50_pc_key]
    ap50_95_per_class = metrics[ap5095_pc_key]

    for c in range(nc):
        name = names[c] if c < len(names) else f"class_{c}"
        print(
            f"  {name:<20} {n_total:>7} {n_gt_per_class[c]:>7} "
            f"{n_det_per_class[c]:>7} {p_per_class[c]:>7.4f} "
            f"{r_per_class[c]:>7.4f} {ap50_per_class[c]:>7.4f} "
            f"{ap50_95_per_class[c]:>9.4f}"
        )

    print(f"  {'-' * 77}")
    print(
        f"  {'all':<20} {n_total:>7} {sum(n_gt_per_class):>7} {total_dets:>7} "
        f"{metrics[p_key]:>7.4f} {metrics[r_key]:>7.4f} "
        f"{metrics[map50_key]:>7.4f} {metrics[map5095_key]:>9.4f}"
    )


# ── COCO 评估（pycocotools 可用时）─────────────────────────────────


def evaluate_coco(
    backend,
    model_path,
    coco_dir,
    imgsz,
    conf,
    iou_thresh,
    max_det,
    max_images=None,
    predist_postprocess="nms",
    mask_eval="fast",
):
    """使用 pycocotools 官方评估流程评估 COCO 数据集（nc=80）。

    流程：
    1. 加载 COCO 注释文件（instances_val2017.json）
    2. 遍历所有验证集图片，letterbox 预处理后推理
    3. 解码检测结果并转换回原图坐标
    4. 将内部类别索引映射为 COCO 80 类 category_id
    5. 使用 COCOeval 计算官方指标（AP, AP50, AP75, AP_small/medium/large, AR）

    参数:
        backend: 推理后端实例（OnnxBackend/RknnBackend/RknnSimBackend）。
        model_path: 模型文件路径。
        coco_dir: COCO 数据集根目录（包含 val2017/ 和 annotations/）。
        imgsz: 输入图像尺寸。
        conf: 置信度阈值（mAP 评估通常用 0.001）。
        iou_thresh: NMS IoU 阈值。
        max_det: 每张图片最大检测数。
        max_images: 限制评估图片数（用于快速测试）。

    返回:
        dict: detect 返回 bbox 指标；segment 返回 bbox + segm 双套指标。
    """
    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_path = os.path.join(coco_dir, "annotations", "instances_val2017.json")
    img_dir = os.path.join(coco_dir, "val2017")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(coco_dir, "images", "val2017")
    assert os.path.isfile(ann_path), f"未找到标注文件: {ann_path}"
    assert os.path.isdir(img_dir), f"未找到图片目录: {img_dir}"

    coco_gt = COCO(ann_path)
    img_ids = sorted(coco_gt.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]

    results = []
    t_start = time.time()
    n_total = len(img_ids)
    output_format = "unknown"
    is_segment = False

    for i, img_id in enumerate(img_ids):
        info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        padded, scale, (pad_w, pad_h) = letterbox(img, imgsz)
        outputs = backend.infer(padded)

        if output_format == "unknown":
            output_format = detect_output_format(outputs)
            is_segment = output_format in SEGMENT_OUTPUT_FORMATS
            print(f"  输出格式: {output_format}, outputs: {[o.shape for o in outputs]}")

        if output_format in SEGMENT_OUTPUT_FORMATS:
            if output_format == "seg_e2e":
                seg_result = decode_seg_e2e(outputs, conf, max_det, iou_thresh)
            elif output_format == "seg_yolov8_topk":
                seg_result = decode_seg_yolov8_topk(outputs, conf, max_det, iou_thresh)
            elif output_format == "seg_yolov8_raw":
                seg_result = decode_seg_yolov8_raw(outputs, conf, max_det, iou_thresh)
            elif output_format == "seg_pre_dist":
                seg_result = decode_seg_predist(
                    outputs,
                    conf,
                    max_det,
                    iou_thresh,
                    imgsz=imgsz,
                    predist_postprocess=predist_postprocess,
                )
            else:
                seg_result = decode_seg_predfl(outputs, conf, max_det, iou_thresh, imgsz=imgsz)
            seg_result = _limit_coco_result_to_topk(seg_result, min(max_det, 100))
            seg_masks = _assemble_seg_masks_to_original(seg_result, imgsz, orig_h, orig_w, scale, pad_w, pad_h)
            dets = (
                np.column_stack([seg_result["boxes"], seg_result["scores"], seg_result["classes"]]).astype(np.float32)
                if seg_result["boxes"].shape[0]
                else np.zeros((0, 6), dtype=np.float32)
            )
            dets = scale_boxes(dets, scale, pad_w, pad_h)
            mask_rles = _encode_binary_masks_to_coco_rles(seg_masks)
            for det, rle in zip(dets, mask_rles):
                x1, y1, x2, y2, score, cls_id = det
                cls_idx = int(cls_id)
                if cls_idx >= len(COCO_IDS):
                    continue
                coco_cat = COCO_IDS[cls_idx]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_w, x2)
                y2 = min(orig_h, y2)
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": coco_cat,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                        "segmentation": rle,
                    }
                )
        else:
            dets = decode_outputs(
                outputs, conf, max_det, iou_thresh, imgsz=imgsz, predist_postprocess=predist_postprocess
            )
            dets = _limit_coco_dets_to_topk(dets, min(max_det, 100))
            dets = scale_boxes(dets, scale, pad_w, pad_h)

            for det in dets:
                x1, y1, x2, y2, score, cls_id = det
                cls_idx = int(cls_id)
                if cls_idx >= len(COCO_IDS):
                    continue
                coco_cat = COCO_IDS[cls_idx]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_w, x2)
                y2 = min(orig_h, y2)
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": coco_cat,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )

        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            print(f"  [{i + 1}/{n_total}] {fps:.1f} img/s，目前 {len(results)} 个 dets")

    elapsed = time.time() - t_start
    print(f"  推理完成: {n_total} 张图片, {len(results)} 个检测结果, {elapsed:.1f}s ({n_total / elapsed:.1f} 张/秒)")

    if not results:
        print("  警告：未检测到任何目标！")
        empty = {
            "task": "segment" if is_segment else "detect",
            "P": 0,
            "R": 0,
            "mAP50": 0,
            "mAP50-95": 0,
            "n_images": n_total,
            "n_dets": 0,
            "time_s": round(elapsed, 1),
            "output_format": output_format,
        }
        if is_segment:
            empty.update(
                {
                    "BoxP": 0,
                    "BoxR": 0,
                    "BoxmAP50": 0,
                    "BoxmAP50-95": 0,
                    "MaskP": 0,
                    "MaskR": 0,
                    "MaskmAP50": 0,
                    "MaskmAP50-95": 0,
                }
            )
        return empty

    coco_dt = coco_gt.loadRes(results)
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval_bbox.params.imgIds = img_ids
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()

    bbox_metrics = _summarize_coco_eval(coco_eval_bbox)

    if not is_segment:
        return {
            "task": "detect",
            **bbox_metrics,
            "n_images": n_total,
            "n_dets": len(results),
            "time_s": round(elapsed, 1),
            "output_format": output_format,
        }

    coco_eval_mask = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval_mask.params.imgIds = img_ids
    coco_eval_mask.evaluate()
    coco_eval_mask.accumulate()
    coco_eval_mask.summarize()

    return {
        "task": "segment",
        **bbox_metrics,
        **_summarize_coco_eval(coco_eval_bbox, prefix="Box"),
        **_summarize_coco_eval(coco_eval_mask, prefix="Mask"),
        "n_images": n_total,
        "n_dets": len(results),
        "time_s": round(elapsed, 1),
        "output_format": output_format,
    }


# ── 通用 YOLO 评估 ───────────────────────────────────────────────────


def evaluate_yolo(
    backend,
    model_path,
    img_dir,
    label_dir,
    nc,
    names,
    imgsz,
    conf,
    iou_thresh,
    max_det,
    max_images=None,
    verbose=False,
    predist_postprocess="nms",
    mask_eval="fast",
    overlap_mask=True,
):
    """在任意 YOLO 格式数据集上运行评估。

    流程：
    1. 扫描图片目录，加载对应的 YOLO .txt 标注
    2. 遍历所有图片：letterbox 预处理 -> 推理 -> 解码 -> 坐标还原
    3. 将预测与真实标注在多 IoU 阈值下匹配
    4. 累积所有图片的统计后计算逐类别 mAP
    5. 可选打印逐类别明细表格

    参数:
        backend: 推理后端实例。
        model_path: 模型文件路径。
        img_dir: 验证集图片目录。
        label_dir: 验证集标注目录。
        nc: 类别数。
        names: 类别名称列表。
        imgsz: 输入图像尺寸。
        conf: 置信度阈值。
        iou_thresh: NMS IoU 阈值。
        max_det: 每张图片最大检测数。
        max_images: 限制评估图片数。
        verbose: 是否打印逐类别明细。

    返回:
        dict: 包含 mAP50-95, mAP50, P, R 及逐类别明细的指标字典。
    """
    import cv2

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    img_files = sorted(p for p in os.listdir(img_dir) if Path(p).suffix.lower() in IMG_EXTS)
    if max_images:
        img_files = img_files[:max_images]

    n_total = len(img_files)
    if n_total == 0:
        print(f"  错误：在 {img_dir} 中未找到图片")
        return _empty_metrics(nc)

    # 预加载 bbox GT（detect 路径使用；segment 路径逐图保留 polygon/mask）
    labels = load_yolo_labels(label_dir, img_files, nc=nc)

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    all_box_stats = []
    all_mask_stats = []
    total_dets = 0
    t_start = time.time()
    output_format = "unknown"
    is_segment = False

    for i, img_name in enumerate(img_files):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        padded, scale, (pad_w, pad_h) = letterbox(img, imgsz)
        outputs = backend.infer(padded)

        if output_format == "unknown":
            output_format = detect_output_format(outputs)
            is_segment = output_format in SEGMENT_OUTPUT_FORMATS
            print(f"  输出格式: {output_format}, outputs: {[o.shape for o in outputs]}")

        stem = Path(img_name).stem
        label_path = os.path.join(label_dir, stem + ".txt")

        if is_segment:
            if output_format == "seg_e2e":
                seg_result = decode_seg_e2e(outputs, conf, max_det, iou_thresh)
            elif output_format == "seg_yolov8_topk":
                seg_result = decode_seg_yolov8_topk(outputs, conf, max_det, iou_thresh)
            elif output_format == "seg_yolov8_raw":
                seg_result = decode_seg_yolov8_raw(outputs, conf, max_det, iou_thresh)
            elif output_format == "seg_pre_dist":
                seg_result = decode_seg_predist(
                    outputs,
                    conf,
                    max_det,
                    iou_thresh,
                    imgsz=imgsz,
                    predist_postprocess=predist_postprocess,
                )
            else:
                seg_result = decode_seg_predfl(outputs, conf, max_det, iou_thresh, imgsz=imgsz)
            pred_masks = _assemble_seg_masks_for_eval(seg_result, imgsz, upsample=(mask_eval == "native"))

            if seg_result["boxes"].shape[0] > 0:
                dets = np.column_stack([seg_result["boxes"], seg_result["scores"], seg_result["classes"]]).astype(
                    np.float32
                )
                pred_boxes = dets[:, :4]
                pred_scores = dets[:, 4]
                pred_classes = dets[:, 5].astype(int)
            else:
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_scores = np.array([], dtype=np.float32)
                pred_classes = np.array([], dtype=int)

            input_h, input_w = parse_input_shape(imgsz)
            proto_h, proto_w = seg_result["proto"].shape[1:]
            ratio_h = input_h // proto_h
            ratio_w = input_w // proto_w
            if ratio_h != ratio_w or proto_h * ratio_h != input_h or proto_w * ratio_w != input_w:
                raise ValueError(f"proto 与输入分辨率比例不一致: input={imgsz}, proto={(proto_h, proto_w)}")
            mask_ratio = ratio_h
            gt_boxes, gt_classes, gt_masks, gt_mask_classes = load_yolo_segment_ground_truth_eval(
                label_path,
                orig_w,
                orig_h,
                imgsz,
                scale,
                pad_w,
                pad_h,
                mask_downsample_ratio=mask_ratio,
                overlap_mask=overlap_mask,
                nc=nc,
            )
            gt_masks = _resize_gt_masks_for_eval(gt_masks, pred_masks.shape[1:])
        else:
            dets = decode_outputs(
                outputs, conf, max_det, iou_thresh, imgsz=imgsz, predist_postprocess=predist_postprocess
            )
            if len(dets) > 0:
                pred_boxes = dets[:, :4]
                pred_scores = dets[:, 4]
                pred_classes = dets[:, 5].astype(int)
            else:
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_scores = np.array([], dtype=np.float32)
                pred_classes = np.array([], dtype=int)

            gt_raw = labels.get(stem, [])
            gt_xyxy = yolo_to_xyxy(gt_raw, orig_w, orig_h)  # [N_gt, 5]
            if len(gt_xyxy) > 0:
                gt_boxes = _letterbox_boxes(gt_xyxy[:, 1:5], scale, pad_w, pad_h, imgsz)
                gt_classes = gt_xyxy[:, 0].astype(int)
            else:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)
                gt_classes = np.array([], dtype=int)

        # bbox 匹配
        correct_box = match_predictions(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_thresholds)

        n_box_gt_per_class = np.zeros(nc, dtype=int)
        for c in gt_classes:
            if 0 <= c < nc:
                n_box_gt_per_class[c] += 1

        all_box_stats.append((correct_box, pred_scores, pred_classes, n_box_gt_per_class))

        if is_segment:
            correct_mask = np.zeros((len(pred_scores), len(iou_thresholds)), dtype=bool)
            if len(pred_scores) > 0 and len(gt_masks) > 0:
                mask_iou_mat = compute_mask_iou_matrix(pred_masks, gt_masks)
                correct_mask = _match_predictions_from_iou(
                    mask_iou_mat,
                    pred_scores,
                    pred_classes,
                    gt_mask_classes,
                    iou_thresholds,
                )

            n_mask_gt_per_class = np.zeros(nc, dtype=int)
            for c in gt_mask_classes:
                if 0 <= c < nc:
                    n_mask_gt_per_class[c] += 1

            all_mask_stats.append((correct_mask, pred_scores, pred_classes, n_mask_gt_per_class))

        total_dets += len(pred_scores)

        if (i + 1) % 200 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            print(f"  [{i + 1}/{n_total}] {fps:.1f} img/s，目前 {total_dets} 个 dets")

    elapsed = time.time() - t_start
    print(f"  推理完成: {n_total} 张图片, {total_dets} 个检测结果, {elapsed:.1f}s ({n_total / elapsed:.1f} 张/秒)")

    if is_segment:
        box_metrics = compute_metrics(all_box_stats, nc)
        mask_metrics = compute_metrics(all_mask_stats, nc)
        metrics = {
            "task": "segment",
            **box_metrics,
            "BoxP": box_metrics["P"],
            "BoxR": box_metrics["R"],
            "BoxmAP50": box_metrics["mAP50"],
            "BoxmAP50-95": box_metrics["mAP50-95"],
            "box_ap50_per_class": box_metrics["ap50_per_class"],
            "box_ap50_95_per_class": box_metrics["ap50_95_per_class"],
            "box_p_per_class": box_metrics["p_per_class"],
            "box_r_per_class": box_metrics["r_per_class"],
            "box_n_gt_per_class": box_metrics["n_gt_per_class"],
            "box_n_det_per_class": box_metrics["n_det_per_class"],
            "MaskP": mask_metrics["P"],
            "MaskR": mask_metrics["R"],
            "MaskmAP50": mask_metrics["mAP50"],
            "MaskmAP50-95": mask_metrics["mAP50-95"],
            "mask_ap50_per_class": mask_metrics["ap50_per_class"],
            "mask_ap50_95_per_class": mask_metrics["ap50_95_per_class"],
            "mask_p_per_class": mask_metrics["p_per_class"],
            "mask_r_per_class": mask_metrics["r_per_class"],
            "mask_n_gt_per_class": mask_metrics["n_gt_per_class"],
            "mask_n_det_per_class": mask_metrics["n_det_per_class"],
        }
    else:
        metrics = compute_metrics(all_box_stats, nc)
        metrics["task"] = "detect"

    metrics["n_images"] = n_total
    metrics["n_dets"] = total_dets
    metrics["time_s"] = round(elapsed, 1)
    metrics["output_format"] = output_format

    # 打印逐类别结果
    if verbose or nc <= 20:
        if is_segment:
            _print_per_class_table(
                "Box 逐类精度",
                metrics,
                names,
                nc,
                n_total,
                total_dets,
                p_key="BoxP",
                r_key="BoxR",
                map50_key="BoxmAP50",
                map5095_key="BoxmAP50-95",
                n_gt_key="box_n_gt_per_class",
                n_det_key="box_n_det_per_class",
                p_pc_key="box_p_per_class",
                r_pc_key="box_r_per_class",
                ap50_pc_key="box_ap50_per_class",
                ap5095_pc_key="box_ap50_95_per_class",
            )
            _print_per_class_table(
                "Mask 逐类精度",
                metrics,
                names,
                nc,
                n_total,
                total_dets,
                p_key="MaskP",
                r_key="MaskR",
                map50_key="MaskmAP50",
                map5095_key="MaskmAP50-95",
                n_gt_key="mask_n_gt_per_class",
                n_det_key="mask_n_det_per_class",
                p_pc_key="mask_p_per_class",
                r_pc_key="mask_r_per_class",
                ap50_pc_key="mask_ap50_per_class",
                ap5095_pc_key="mask_ap50_95_per_class",
            )
        else:
            _print_per_class_table(
                "逐类精度",
                metrics,
                names,
                nc,
                n_total,
                total_dets,
            )

    return metrics


# ── 主函数 ───────────────────────────────────────────────────────────────


def make_backend(backend_name, model_path, imgsz, core):
    """根据后端名称创建对应的推理后端实例（工厂函数）。"""
    if backend_name == "onnx":
        return OnnxBackend(model_path, imgsz)
    elif backend_name == "rknn_sim":
        return RknnSimBackend(model_path, imgsz)
    else:
        return RknnBackend(model_path, imgsz, core)


def is_coco_dataset(data, data_yaml_path):
    """检测数据集是否为 COCO 格式（存在 annotations/instances_val2017.json）。

    检查 val 路径的父目录或祖父目录下是否有 COCO 注释文件。

    参数:
        data: load_data_yaml() 返回的配置字典。
        data_yaml_path: data.yaml 文件路径。

    返回:
        (bool, str|None): (is_coco, coco_root_dir)。
    """
    yaml_dir = Path(data_yaml_path).resolve().parent

    def _abspath_no_resolve(path: Path) -> Path:
        """返回绝对路径但不跟随末级 symlink，避免丢失临时数据集根目录。"""
        return Path(os.path.abspath(os.fspath(path)))

    val_path = Path(data.get("val", ""))
    ds_path = data.get("path", "")

    val_candidates: list[Path] = []
    if ds_path:
        ds_root = Path(ds_path)
        if not ds_root.is_absolute():
            ds_root = yaml_dir / ds_root
        val_candidates.append(_abspath_no_resolve(ds_root / val_path))
    if val_path.is_absolute():
        val_candidates.append(_abspath_no_resolve(val_path))
    else:
        val_candidates.append(_abspath_no_resolve(yaml_dir / val_path))

    seen: set[Path] = set()
    for val_abs in val_candidates:
        coco_root_candidates = [val_abs]
        if val_abs.name == "val2017":
            coco_root_candidates.append(val_abs.parent)
            if val_abs.parent.name == "images":
                coco_root_candidates.append(val_abs.parent.parent)

        for coco_root in coco_root_candidates:
            coco_root = _abspath_no_resolve(coco_root)
            if coco_root in seen:
                continue
            seen.add(coco_root)
            ann = coco_root / "annotations" / "instances_val2017.json"
            img_dir = coco_root / "val2017"
            img_dir_nested = coco_root / "images" / "val2017"
            if ann.is_file() and (img_dir.is_dir() or img_dir_nested.is_dir()):
                return True, str(coco_root)

    return False, None


def main():
    """主函数：解析参数 -> 加载数据集 -> 遍历模型评估 -> 汇总输出。

    支持 glob 通配符批量评估多个模型，自动检测 COCO 数据集
    并切换到 pycocotools 官方评估流程。结果同时打印到终端
    并保存为 JSON 文件。
    """
    args = parse_args()
    data_yaml = os.path.expanduser(args.data)
    data = load_data_yaml(data_yaml)

    img_dir, label_dir, nc, names, _ = resolve_dataset_paths(data, data_yaml)

    # 检测数据集是否为 COCO 格式（存在 pycocotools 兼容的标注文件）
    use_coco_eval, coco_dir = is_coco_dataset(data, data_yaml)
    if use_coco_eval and nc == 80:
        try:
            from pycocotools.coco import COCO  # noqa: F401

            print(f"检测到 COCO 数据集：使用 pycocotools 进行评估")
        except ImportError:
            print("pycocotools 不可用，使用通用 YOLO 评估流程")
            use_coco_eval = False
    else:
        use_coco_eval = False

    # 解析模型路径（支持 glob 通配符）
    if "*" in args.model or "?" in args.model:
        model_paths = sorted(glob.glob(args.model))
    else:
        model_paths = [args.model]

    if not model_paths:
        print(f"未找到匹配的模型: {args.model}")
        sys.exit(1)

    print(f"数据集: {data_yaml} (nc={nc}, names={names})")
    print(f"验证集图片: {img_dir}")
    if not use_coco_eval:
        print(f"验证集标注: {label_dir}")
        if not os.path.isdir(label_dir):
            print(f"警告：未找到标注目录: {label_dir}")

    all_results = {}
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        imgsz = parse_input_shape(args.imgsz or infer_input_size(model_path))
        imgsz_text = format_input_shape(imgsz)
        print(f"\n{'=' * 70}")
        print(f"模型: {model_name}  (imgsz={imgsz_text})")
        print(f"{'=' * 70}")

        print(f"  predist_postprocess={args.predist_postprocess} (仅 pre_dist / seg_pre_dist 生效)")

        try:
            engine = make_backend(args.backend, model_path, imgsz, args.core)

            if use_coco_eval:
                metrics = evaluate_coco(
                    engine,
                    model_path,
                    coco_dir,
                    imgsz,
                    args.conf,
                    args.iou,
                    args.max_det,
                    args.max_images,
                    predist_postprocess=args.predist_postprocess,
                    mask_eval=args.mask_eval,
                )
            else:
                metrics = evaluate_yolo(
                    engine,
                    model_path,
                    img_dir,
                    label_dir,
                    nc,
                    names,
                    imgsz,
                    args.conf,
                    args.iou,
                    args.max_det,
                    args.max_images,
                    args.verbose,
                    predist_postprocess=args.predist_postprocess,
                    mask_eval=args.mask_eval,
                    overlap_mask=args.overlap_mask,
                )

            engine.release()
        except Exception as e:
            print(f"  错误: {e}")
            import traceback

            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}
            continue

        metrics["imgsz"] = imgsz_text
        if metrics.get("output_format") in ("pre_dist", "seg_pre_dist"):
            metrics["predist_postprocess"] = args.predist_postprocess
        # `mask_eval` 仅对自定义数据集评估有效；若走 COCO pycocotools 路径，
        # mask 总会被映射回原图尺寸，因此该字段不参与口径区分。
        if metrics.get("task") == "segment" and not use_coco_eval:
            metrics["mask_eval"] = args.mask_eval
            metrics["overlap_mask"] = args.overlap_mask
        all_results[model_name] = metrics

        print(
            f"\n  结果: mAP50-95={metrics['mAP50-95']:.4f}, "
            f"mAP50={metrics['mAP50']:.4f}, "
            f"P={metrics['P']:.4f}, R={metrics['R']:.4f}"
        )
        if metrics.get("task") == "segment":
            print(
                f"  Mask: mAP50-95={metrics['MaskmAP50-95']:.4f}, "
                f"mAP50={metrics['MaskmAP50']:.4f}, "
                f"P={metrics['MaskP']:.4f}, R={metrics['MaskR']:.4f}"
            )

    # 打印汇总表
    print(f"\n\n{'=' * 90}")
    print("汇总")
    print(f"{'=' * 90}")
    header = f"{'模型':<45} {'尺寸':>5} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}"
    print(header)
    print("-" * len(header))
    for name, m in all_results.items():
        if "error" in m:
            print(f"{name:<45} {'错误':>5} — {m['error']}")
        else:
            print(
                f"{name:<45} {str(m['imgsz']):>5} {m['P']:>6.4f} "
                f"{m['R']:>6.4f} {m['mAP50']:>7.4f} "
                f"{m['mAP50-95']:>9.4f}"
            )

    # 保存结果
    out_path = args.output or f"eval_results_{args.backend}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
