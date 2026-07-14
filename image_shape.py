#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file image_shape.py
@brief 不依赖训练框架的静态图像尺寸解析与校验工具。
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from typing import NamedTuple


class ImageShape(NamedTuple):
    """!
    @brief 按高度、宽度顺序保存的图像尺寸。
    """

    height: int
    width: int


ImageShapeLike = int | str | Sequence[int]

_RECTANGULAR_SHAPE_PATTERN = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")


def parse_imgsz(value: ImageShapeLike) -> ImageShape:
    """!
    @brief 将外部尺寸表达解析为统一的 `(height, width)`。
    @param value 单一边长、`HxW` 字符串，或包含一至两个整数的序列。
    @return 解析后的图像尺寸；单一边长会扩展为正方形。
    @throws TypeError 输入类型不受支持时抛出。
    @throws ValueError 维度数量不正确或尺寸非正整数时抛出。
    """
    parsed_value = value
    if isinstance(value, str):
        match = _RECTANGULAR_SHAPE_PATTERN.fullmatch(value)
        if match:
            parsed_value = (int(match.group(1)), int(match.group(2)))
        elif value.strip().isdecimal():
            parsed_value = int(value.strip())
        else:
            try:
                parsed_value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"imgsz={value!r} 无法解析；请使用 640、480x640 或 [480, 640]") from exc

    if isinstance(parsed_value, bool):
        raise TypeError("imgsz 不能为布尔值")
    if isinstance(parsed_value, int):
        dimensions = (parsed_value, parsed_value)
    elif isinstance(parsed_value, Sequence) and not isinstance(parsed_value, (str, bytes)):
        dimensions = tuple(parsed_value)
        if len(dimensions) == 1:
            dimensions = (dimensions[0], dimensions[0])
        elif len(dimensions) != 2:
            raise ValueError(f"imgsz 只接受一个边长或 H W 两个值，实际为 {list(dimensions)}")
    else:
        raise TypeError(
            f"imgsz={parsed_value!r} 类型无效: {type(parsed_value).__name__}；请使用整数、HxW 字符串或整数序列"
        )

    if any(isinstance(dimension, bool) or not isinstance(dimension, int) for dimension in dimensions):
        raise TypeError(f"imgsz 的高度和宽度必须为整数，实际为 {dimensions}")
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(f"imgsz 的高度和宽度必须大于 0，实际为 {dimensions}")
    return ImageShape(height=dimensions[0], width=dimensions[1])


def validate_imgsz_stride(shape: ImageShapeLike, stride: int) -> ImageShape:
    """!
    @brief 校验图像高度和宽度均与模型最大步幅对齐。
    @param shape 待校验的图像尺寸。
    @param stride 模型的最大特征图步幅。
    @return 解析且校验通过的图像尺寸。
    @throws ValueError 步幅非正整数或任一维度未对齐时抛出。
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError(f"stride 必须为正整数，实际为 {stride!r}")
    parsed = parse_imgsz(shape)
    if parsed.height % stride != 0 or parsed.width % stride != 0:
        raise ValueError(f"imgsz={format_imgsz(parsed)} 的高度和宽度必须均为最大步幅 {stride} 的整数倍")
    return parsed


def format_imgsz(shape: ImageShapeLike) -> str:
    """!
    @brief 将图像尺寸格式化为稳定的 `H` 或 `HxW` 字符串。
    @param shape 待格式化的图像尺寸。
    @return 正方形返回单一边长，矩形返回 `heightxwidth`。
    """
    parsed = parse_imgsz(shape)
    return str(parsed.height) if parsed.height == parsed.width else f"{parsed.height}x{parsed.width}"
