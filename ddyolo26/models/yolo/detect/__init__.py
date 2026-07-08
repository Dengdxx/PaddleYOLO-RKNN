# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 检测任务模块公开接口。
@details
导出 `DetectionPredictor`、`DetectionTrainer`、`DetectionValidator` 三个类，
对应 YOLO 推理/训练/验证三个模式。
"""

from __future__ import annotations
from .predict import DetectionPredictor
import paddle
from .train import DetectionTrainer
from .val import DetectionValidator

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator"
