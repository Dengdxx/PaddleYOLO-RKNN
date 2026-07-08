# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 回调系统公开接口。
@details
导出 `default_callbacks`、`add_integration_callbacks` 等函数，
供 BaseTrainer 初始化时注册回调链。
"""

from __future__ import annotations
import paddle
from .base import add_integration_callbacks, default_callbacks, get_default_callbacks

__all__ = ("add_integration_callbacks", "default_callbacks", "get_default_callbacks")
