# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO 模型类聚合：导出所有任务的预测/训练/验证器。
@details
将 detect/ 和 segment/ 目录下的任务组件
汇聚到 YOLO 模型类中，通过任务派发逻辑支持 `task='detect'`/`task='segment'` 路由。
"""

from __future__ import annotations
from .detect import DetectionPredictor, DetectionTrainer, DetectionValidator
from .segment import SegmentationPredictor, SegmentationTrainer, SegmentationValidator
import paddle
from .model import YOLO

__all__ = (
    "YOLO",
    "DetectionTrainer",
    "DetectionValidator",
    "DetectionPredictor",
    "SegmentationTrainer",
    "SegmentationValidator",
    "SegmentationPredictor",
)
