#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file input_shape.py
@brief 导出链静态输入尺寸规范化工具。
"""

from __future__ import annotations

from collections.abc import Sequence

from image_shape import ImageShape, format_imgsz, parse_imgsz, validate_imgsz_stride

StaticInputShape = ImageShape


def normalize_static_imgsz(value: int | str | Sequence[int | str], stride: int = 32) -> StaticInputShape:
    """!
    @brief 将命令行输入尺寸规范化为正方形边长或 `(H, W)`。
    @param value 单个边长，或按高度、宽度排列的一至两个整数。
    @param stride 模型最大步幅；两个维度必须均为其整数倍。
    @return 正方形返回单个整数，矩形返回 `(H, W)`。
    @throw ValueError 当尺寸数量、数值或步幅对齐不合法时抛出。
    """
    normalized_value: int | str | Sequence[int]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if len(items) == 1:
            normalized_value = items[0]
        elif len(items) == 2:
            try:
                normalized_value = [int(item) for item in items]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"--imgsz H W 必须为整数，实际收到 {items}") from exc
        else:
            raise ValueError(f"--imgsz 只接受一个边长或 H W 两个值，实际收到 {items}")
    else:
        normalized_value = value
    return validate_imgsz_stride(normalized_value, stride)


def static_imgsz_hw(value: StaticInputShape) -> tuple[int, int]:
    """!
    @brief 将规范化输入尺寸展开为 `(H, W)`。
    @param value 正方形边长或 `(H, W)`。
    @return 高度与宽度二元组。
    """
    parsed = parse_imgsz(value)
    return parsed.height, parsed.width


def format_static_imgsz(value: StaticInputShape) -> str:
    """!
    @brief 将静态输入尺寸格式化为稳定的文件名片段。
    @param value 正方形边长或 `(H, W)`。
    @return 正方形返回 `640`，矩形返回 `480x640`。
    """
    return format_imgsz(value)
