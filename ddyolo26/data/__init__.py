# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 数据模块公开接口。
@details
导出 BaseDataset、YOLODataset、DataLoader 构建函数等，
供 DetectionTrainer.build_dataset() 调用。
"""

from __future__ import annotations
from .base import BaseDataset
import paddle
from .build import build_dataloader, build_grounding, build_yolo_dataset, load_inference_source
from .dataset import GroundingDataset, YOLOConcatDataset, YOLODataset, YOLOMultiModalDataset

__all__ = (
    "BaseDataset",
    "GroundingDataset",
    "YOLOConcatDataset",
    "YOLODataset",
    "YOLOMultiModalDataset",
    "build_dataloader",
    "build_grounding",
    "build_yolo_dataset",
    "load_inference_source",
)
