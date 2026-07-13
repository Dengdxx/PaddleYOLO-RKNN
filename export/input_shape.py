#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file input_shape.py
@brief 导出链静态输入尺寸规范化工具。
"""

from __future__ import annotations

from collections.abc import Sequence

StaticInputShape = int | tuple[int, int]


def normalize_static_imgsz(value: int | Sequence[int], stride: int = 32) -> StaticInputShape:
    """!
    @brief 将命令行输入尺寸规范化为正方形边长或 `(H, W)`。
    @param value 单个边长，或按高度、宽度排列的一至两个整数。
    @param stride 模型最大步幅；两个维度必须均为其整数倍。
    @return 正方形返回单个整数，矩形返回 `(H, W)`。
    @throw ValueError 当尺寸数量、数值或步幅对齐不合法时抛出。
    """
    values = [value] if isinstance(value, int) else list(value)
    if len(values) not in (1, 2):
        raise ValueError(f"--imgsz 只接受一个边长或 H W 两个值，实际收到 {values}")
    if stride <= 0:
        raise ValueError(f"stride 必须大于 0，实际为 {stride}")

    sizes = [int(size) for size in values]
    if any(size <= 0 for size in sizes):
        raise ValueError(f"--imgsz 必须为正整数，实际收到 {sizes}")
    if any(size % stride != 0 for size in sizes):
        raise ValueError(f"--imgsz 的每个维度必须是最大步幅 {stride} 的整数倍，实际收到 {sizes}")
    if len(sizes) == 1:
        return sizes[0]
    return sizes[0], sizes[1]


def static_imgsz_hw(value: StaticInputShape) -> tuple[int, int]:
    """!
    @brief 将规范化输入尺寸展开为 `(H, W)`。
    @param value 正方形边长或 `(H, W)`。
    @return 高度与宽度二元组。
    """
    if isinstance(value, int):
        return value, value
    return value


def format_static_imgsz(value: StaticInputShape) -> str:
    """!
    @brief 将静态输入尺寸格式化为稳定的文件名片段。
    @param value 正方形边长或 `(H, W)`。
    @return 正方形返回 `640`，矩形返回 `480x640`。
    """
    height, width = static_imgsz_hw(value)
    return str(height) if height == width else f"{height}x{width}"
