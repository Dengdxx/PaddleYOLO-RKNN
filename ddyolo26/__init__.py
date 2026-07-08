# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 包根级公开接口：导出 YOLO 类和版本号。
@details
本仓库是 Ultralytics YOLO26 的 PaddlePaddle 移植版，
专注于目标检测与实例分割两条任务主线，剔除姿态/OBB/分类等冗余逻辑。

顶层导出：
- `YOLO`：统一用户 API，支持 train/val/predict/export
- `__version__`：当前版本号

YOLO26 三大架构特性（ddyolo26 均已实现）：
1. 端到端 NMS-free（one-to-one 匈牙利匹配头，推理无后处理）
2. 移除 DFL（reg_max=1，Detect Head 直接回归坐标）
3. MuSGD 优化器（SGD + Muon 正交化更新混合策略）
"""

from __future__ import annotations
import os
import warnings

# ── 在 import paddle 之前抑制已知的无害警告 ──────────────────────────
os.environ.setdefault("GLOG_minloglevel", "2")  # 屏蔽 Paddle C++ glog INFO/WARNING
warnings.filterwarnings("ignore", message="No ccache found")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")

import importlib
from typing import TYPE_CHECKING

from ddyolo26.version import __version__

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"
from ddyolo26.utils import ASSETS, SETTINGS
from ddyolo26.utils.checks import check_yolo as checks

settings = SETTINGS
MODELS = ("YOLO",)
__all__ = "__version__", "ASSETS", *MODELS, "checks", "download", "settings"
if TYPE_CHECKING:
    from ddyolo26.models import YOLO


def __getattr__(name: str):
    """首次访问模型类时再懒加载。"""
    if name in MODELS:
        return getattr(importlib.import_module("ddyolo26.models"), name)
    raise AttributeError(f"module {__name__} 没有属性 {name}")


def download(*args, **kwargs):
    """懒加载普通资源下载入口，避免影响 `python -m ddyolo26.utils.downloads`。"""
    from ddyolo26.utils.downloads import download as _download

    return _download(*args, **kwargs)


def __dir__():
    """扩展 dir() 结果，让 IDE 能补全懒加载模型名。"""
    return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
    print(__version__)
