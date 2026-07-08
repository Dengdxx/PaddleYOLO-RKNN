# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 模型包公开接口：导出顶层 YOLO 类。
@details
该模块将 `ddyolo26.models.yolo.YOLO` 作为顶层接口导出，
用户通过 `from ddyolo26 import YOLO` 即可访问完整功能。
"""

from __future__ import annotations
from .yolo import YOLO
import paddle

__all__ = ("YOLO",)
