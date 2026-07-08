# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 网络模块辅助工具函数。
@details
提供：
- `make_divisible()`：使通道数对齐到指定除数
- `bias_init_with_prob()`：检测头偏置初始化（基于先验概率）
- `linear_init()`：线性层权重初始化
"""

from __future__ import annotations
import sys
import paddle

import copy
import math

import numpy as np

from ddyolo26.paddle_utils import *

__all__ = "inverse_sigmoid", "multi_scale_deformable_attn_paddle"


def _get_clones(module, n):
    """基于给定模块创建克隆模块列表。

    参数:
        module (nn.Module): 待克隆的模块。
        n (int): 需要创建的克隆数量。

    返回:
        (nn.ModuleList): 包含 n 个输入模块克隆的 ModuleList。

    示例:
        >>> import paddle.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    """
    return paddle.nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """根据给定先验概率初始化 conv/fc 偏置值。

    该函数通过 inverse sigmoid（logit）根据先验概率计算偏置初始化值。目标检测模型中常用它
    将分类层初始化为指定的正样本预测概率。

    参数:
        prior_prob (float, optional): 偏置初始化使用的先验概率。

    返回:
        (float): 根据先验概率计算得到的偏置初始化值。

    示例:
        >>> bias = bias_init_with_prob(0.01)
        >>> print(f"Bias initialization value: {bias:.4f}")
        Bias initialization value: -4.5951
    """
    return float(-np.log((1 - prior_prob) / prior_prob))


def linear_init(module):
    """初始化线性模块的权重和偏置。

    该函数根据输出维度计算均匀分布边界并初始化线性模块权重；如果模块带有偏置，也会一并初始化。

    参数:
        module (nn.Module): 待初始化的线性模块。

    示例:
        >>> import paddle.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> linear_init(linear)
    """
    bound = 1 / math.sqrt(module.weight.shape[0])
    paddle.nn.init.uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        paddle.nn.init.uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-05):
    """计算张量的 inverse sigmoid。

    该函数对张量应用 sigmoid 的反函数，常用于神经网络中的注意力机制和坐标变换。

    参数:
        x (paddle.Tensor): 输入张量，取值范围为 [0, 1]。
        eps (float, optional): 防止数值不稳定的小 epsilon。

    返回:
        (paddle.Tensor): 应用 inverse sigmoid 后的张量。

    示例:
        >>> x = paddle.to_tensor([0.2, 0.5, 0.8])
        >>> inverse_sigmoid(x)
        tensor([-1.3863,  0.0000,  1.3863])
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return paddle.log(x1 / x2)


def multi_scale_deformable_attn_paddle(
    value: paddle.Tensor,
    value_spatial_shapes: paddle.Tensor,
    sampling_locations: paddle.Tensor,
    attention_weights: paddle.Tensor,
) -> paddle.Tensor:
    """实现多尺度可变形注意力。

    该函数在多个特征图尺度上执行可变形注意力，使模型可以通过学习到的偏移关注不同空间位置。

    参数:
        value (paddle.Tensor): value 张量，形状为 (bs, num_keys, num_heads, embed_dims)。
        value_spatial_shapes (paddle.Tensor): value 张量的空间形状，形状为 (num_levels, 2)。
        sampling_locations (paddle.Tensor): 采样位置，形状为 (bs, num_queries, num_heads, num_levels, num_points, 2)。
        attention_weights (paddle.Tensor): 注意力权重，形状为 (bs, num_queries, num_heads, num_levels, num_points)。

    返回:
        (paddle.Tensor): 输出张量，形状为 (bs, num_queries, num_heads * embed_dims)。

    参考:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([(H_ * W_) for H_, W_ in value_spatial_shapes], axis=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = paddle.nn.functional.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (paddle.stack(sampling_value_list, axis=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# 向后兼容别名
multi_scale_deformable_attn_pytorch = multi_scale_deformable_attn_paddle
