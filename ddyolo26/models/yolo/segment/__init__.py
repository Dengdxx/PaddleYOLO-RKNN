# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO 分割任务组件聚合。
@details
导出 SegmentationPredictor、SegmentationTrainer、SegmentationValidator，
供 YOLO 入口类 task_map 路由使用。
"""

from .predict import SegmentationPredictor
from .train import SegmentationTrainer
from .val import SegmentationValidator

__all__ = "SegmentationPredictor", "SegmentationTrainer", "SegmentationValidator"
