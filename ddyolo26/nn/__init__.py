# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 神经网络高层模块公开接口。
@details
导出 tasks.py 中的模型类和 autobackend.py 的推理后端，
供 engine/model.py 和 models/ 层调用。
"""

from __future__ import annotations
import paddle
from .tasks import (
    BaseModel,
    DetectionModel,
    guess_model_scale,
    guess_model_task,
    load_checkpoint,
    parse_model,
    paddle_safe_load,
    yaml_model_load,
)

__all__ = (
    "BaseModel",
    "DetectionModel",
    "guess_model_scale",
    "guess_model_task",
    "load_checkpoint",
    "paddle_safe_load",
    "parse_model",
    "yaml_model_load",
)
