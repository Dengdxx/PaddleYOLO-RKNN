#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file image_shape.py
@brief 为训练包保留静态尺寸工具的兼容导入路径。
"""

from image_shape import ImageShape, ImageShapeLike, format_imgsz, parse_imgsz, validate_imgsz_stride

__all__ = ["ImageShape", "ImageShapeLike", "format_imgsz", "parse_imgsz", "validate_imgsz_stride"]
