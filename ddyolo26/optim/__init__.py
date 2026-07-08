# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 优化器模块公开接口。
@details
导出 MuSGD 优化器类，供训练器通过 `optimizer='MuSGD'` 或 `optimizer='auto'` 选项调用。
"""

from __future__ import annotations
from .muon import Muon, MuSGD
import paddle

__all__ = ["MuSGD", "Muon"]
