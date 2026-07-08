# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 Transformer 模块：RT-DETR / 注意力机制组件。
@details
包含用于检测颈部和 RT-DETR 解码器的 Transformer 组件：
- `TransformerLayer` / `TransformerBlock`：原始 YOLO Transformer 层
- `DeformableTransformerDecoderLayer/Decoder`：可变形注意力解码器（RT-DETR）
- `AIFI`：具有固定余弦编码的注意力层
- `MLP`：多层感知机
"""

from __future__ import annotations
import sys
import paddle


from ddyolo26.paddle_utils import *

"""Transformer 模块。"""

import math

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_paddle

__all__ = (
    "AIFI",
    "MLP",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "LayerNorm2d",
    "MLPBlock",
    "MSDeformAttn",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
)


class TransformerEncoderLayer(paddle.nn.Module):
    """Transformer 编码器的单层结构。

    该类实现带多头注意力和前馈网络的标准 Transformer 编码器层，同时支持前置归一化和后置归一化配置。

    属性:
        ma (nn.MultiheadAttention): 多头注意力模块。
        fc1 (nn.Linear): 前馈网络中的第一层线性层。
        fc2 (nn.Linear): 前馈网络中的第二层线性层。
        norm1 (nn.LayerNorm): 注意力之后的层归一化。
        norm2 (nn.LayerNorm): 前馈网络之后的层归一化。
        dropout (nn.Dropout): 前馈网络中的 dropout 层。
        dropout1 (nn.Dropout): 注意力之后的 dropout 层。
        dropout2 (nn.Dropout): 前馈网络之后的 dropout 层。
        act (nn.Module): 激活函数。
        normalize_before (bool): 是否在注意力和前馈网络之前执行归一化。
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: paddle.nn.Module = paddle.nn.GELU(),
        normalize_before: bool = False,
    ):
        """使用指定参数初始化 TransformerEncoderLayer。

        参数:
            c1 (int): 输入维度。
            cm (int): 前馈网络隐藏维度。
            num_heads (int): 注意力头数量。
            dropout (float): dropout 概率。
            act (nn.Module): 激活函数。
            normalize_before (bool): 是否在注意力和前馈网络之前执行归一化。
        """
        super().__init__()
        self.ma = paddle.compat.nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = paddle.compat.nn.Linear(c1, cm)
        self.fc2 = paddle.compat.nn.Linear(cm, c1)
        self.norm1 = paddle.nn.LayerNorm(c1)
        self.norm2 = paddle.nn.LayerNorm(c1)
        self.dropout = paddle.nn.Dropout(dropout)
        self.dropout1 = paddle.nn.Dropout(dropout)
        self.dropout2 = paddle.nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: paddle.Tensor, pos: (paddle.Tensor | None) = None) -> paddle.Tensor:
        """如果提供位置编码，则将其加到张量上。"""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: paddle.Tensor,
        src_mask: (paddle.Tensor | None) = None,
        src_key_padding_mask: (paddle.Tensor | None) = None,
        pos: (paddle.Tensor | None) = None,
    ) -> paddle.Tensor:
        """执行后置归一化的前向传播。

        参数:
            src (paddle.Tensor): 输入张量。
            src_mask (paddle.Tensor, optional): src 序列 mask。
            src_key_padding_mask (paddle.Tensor, optional): 每个 batch 中 src key 的 mask。
            pos (paddle.Tensor, optional): 位置编码。

        返回:
            (paddle.Tensor): 经过注意力和前馈网络后的输出张量。
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: paddle.Tensor,
        src_mask: (paddle.Tensor | None) = None,
        src_key_padding_mask: (paddle.Tensor | None) = None,
        pos: (paddle.Tensor | None) = None,
    ) -> paddle.Tensor:
        """执行前置归一化的前向传播。

        参数:
            src (paddle.Tensor): 输入张量。
            src_mask (paddle.Tensor, optional): src 序列 mask。
            src_key_padding_mask (paddle.Tensor, optional): 每个 batch 中 src key 的 mask。
            pos (paddle.Tensor, optional): 位置编码。

        返回:
            (paddle.Tensor): 经过注意力和前馈网络后的输出张量。
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: paddle.Tensor,
        src_mask: (paddle.Tensor | None) = None,
        src_key_padding_mask: (paddle.Tensor | None) = None,
        pos: (paddle.Tensor | None) = None,
    ) -> paddle.Tensor:
        """将输入前向传播通过编码器模块。

        参数:
            src (paddle.Tensor): 输入张量。
            src_mask (paddle.Tensor, optional): src 序列 mask。
            src_key_padding_mask (paddle.Tensor, optional): 每个 batch 中 src key 的 mask。
            pos (paddle.Tensor, optional): 位置编码。

        返回:
            (paddle.Tensor): 经过 Transformer 编码器层后的输出张量。
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """带位置编码的 2D 数据 AIFI Transformer 层。

    该类扩展 TransformerEncoderLayer，通过加入 2D sine-cosine 位置编码并正确处理空间维度，
    使其可用于 2D 特征图。
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        act: paddle.nn.Module = paddle.nn.GELU(),
        normalize_before: bool = False,
    ):
        """使用指定参数初始化 AIFI 实例。

        参数:
            c1 (int): 输入维度。
            cm (int): 前馈网络隐藏维度。
            num_heads (int): 注意力头数量。
            dropout (float): dropout 概率。
            act (nn.Module): 激活函数。
            normalize_before (bool): 是否在注意力和前馈网络之前执行归一化。
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """AIFI Transformer 层的前向传播。

        参数:
            x (paddle.Tensor): 输入张量，形状为 [B, C, H, W]。

        返回:
            (paddle.Tensor): 输出张量，形状为 [B, C, H, W]。
        """
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        x = super().forward(
            x.flatten(2).permute(0, 2, 1),
            pos=pos_embed.to(device=x.device, dtype=x.dtype),
        )
        return x.permute(0, 2, 1).reshape([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> paddle.Tensor:
        """构建 2D sine-cosine 位置编码。

        参数:
            w (int): 特征图宽度。
            h (int): 特征图高度。
            embed_dim (int): 嵌入维度。
            temperature (float): sine/cosine 函数的温度系数。

        返回:
            (paddle.Tensor): 位置编码，形状为 [1, h*w, embed_dim]。
        """
        assert embed_dim % 4 == 0, "2D sin-cos 位置编码要求 embed_dim 可被 4 整除"
        grid_w = paddle.arange(w, dtype=paddle.float32)
        grid_h = paddle.arange(h, dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1.0 / temperature**omega
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return paddle.cat(
            [
                paddle.sin(out_w),
                paddle.cos(out_w),
                paddle.sin(out_h),
                paddle.cos(out_h),
            ],
            1,
        )[None]


class TransformerLayer(paddle.nn.Module):
    """Transformer 层 https://arxiv.org/abs/2010.11929（为提升性能移除了 LayerNorm 层）。"""

    def __init__(self, c: int, num_heads: int):
        """使用线性变换和多头注意力初始化自注意力机制。

        参数:
            c (int): 输入和输出通道维度。
            num_heads (int): 注意力头数量。
        """
        super().__init__()
        self.q = paddle.compat.nn.Linear(c, c, bias=False)
        self.k = paddle.compat.nn.Linear(c, c, bias=False)
        self.v = paddle.compat.nn.Linear(c, c, bias=False)
        self.ma = paddle.compat.nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = paddle.compat.nn.Linear(c, c, bias=False)
        self.fc2 = paddle.compat.nn.Linear(c, c, bias=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """对输入 x 应用 Transformer 块并返回输出。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 经过 Transformer 层后的输出张量。
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(paddle.nn.Module):
    """基于 https://arxiv.org/abs/2010.11929 的视觉 Transformer 块。

    该类实现完整 Transformer 块，包含可选的通道调整卷积层、可学习位置编码和多个 Transformer 层。

    属性:
        conv (Conv, optional): 当输入和输出通道不同时使用的卷积层。
        linear (nn.Linear): 可学习位置编码。
        tr (nn.Sequential): Transformer 层的顺序容器。
        c2 (int): 输出通道维度。
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """初始化带位置编码、指定头数和层数的 Transformer 模块。

        参数:
            c1 (int): 输入通道维度。
            c2 (int): 输出通道维度。
            num_heads (int): 注意力头数量。
            num_layers (int): Transformer 层数。
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = paddle.compat.nn.Linear(c2, c2)
        self.tr = paddle.nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """将输入前向传播通过 Transformer 块。

        参数:
            x (paddle.Tensor): 输入张量，形状为 [b, c1, h, w]。

        返回:
            (paddle.Tensor): 输出张量，形状为 [b, c2, h, w]。
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, h, w = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, h, w)


class MLPBlock(paddle.nn.Module):
    """多层感知机中的单个块。"""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=paddle.nn.GELU):
        """使用指定嵌入维度、MLP 维度和激活函数初始化 MLPBlock。

        参数:
            embedding_dim (int): 输入和输出维度。
            mlp_dim (int): 隐藏维度。
            act (type): 激活函数类。
        """
        super().__init__()
        self.lin1 = paddle.compat.nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = paddle.compat.nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """MLPBlock 前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 经过 MLP 块后的输出张量。
        """
        return self.lin2(self.act(self.lin1(x)))


class MLP(paddle.nn.Module):
    """简单多层感知机（也称 FFN）。

    该类实现可配置 MLP，包含多层线性层、激活函数以及可选的 sigmoid 输出激活。

    属性:
        num_layers (int): MLP 层数。
        layers (nn.ModuleList): 线性层列表。
        sigmoid (bool): 是否对输出应用 sigmoid。
        act (nn.Module): 激活函数。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act=paddle.nn.ReLU,
        sigmoid: bool = False,
        residual: bool = False,
        out_norm: paddle.nn.Module = None,
    ):
        """使用指定输入、隐藏、输出维度和层数初始化 MLP。

        参数:
            input_dim (int): 输入维度。
            hidden_dim (int): 隐藏维度。
            output_dim (int): 输出维度。
            num_layers (int): 层数。
            act (type): 激活函数类。
            sigmoid (bool): 是否对输出应用 sigmoid。
            residual (bool): 是否使用残差连接。
            out_norm (nn.Module, optional): 输出归一化层。
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = paddle.nn.ModuleList(
            paddle.compat.nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim])
        )
        self.sigmoid = sigmoid
        self.act = act()
        if residual and input_dim != output_dim:
            raise ValueError("仅当 input_dim == output_dim 时才支持 residual")
        self.residual = residual
        assert isinstance(out_norm, paddle.nn.Module) or out_norm is None
        self.out_norm = out_norm or paddle.nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """完整 MLP 的前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 经过 MLP 后的输出张量。
        """
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", paddle.nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        if getattr(self, "residual", False):
            x = x + orig_x
        x = getattr(self, "out_norm", paddle.nn.Identity())(x)
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(paddle.nn.Module):
    """受 Detectron2 和 ConvNeXt 实现启发的 2D LayerNorm 模块。

    该类为 2D 特征图实现层归一化，在通道维度归一化，同时保留空间维度。

    属性:
        weight (nn.Parameter): 可学习缩放参数。
        bias (nn.Parameter): 可学习偏置参数。
        eps (float): 保持数值稳定的小常数。

    参考:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, num_channels: int, eps: float = 1e-06):
        """使用给定参数初始化 LayerNorm2d。

        参数:
            num_channels (int): 输入通道数。
            eps (float): 保持数值稳定的小常数。
        """
        super().__init__()
        self.weight = paddle.nn.Parameter(paddle.ones(num_channels))
        self.bias = paddle.nn.Parameter(paddle.zeros(num_channels))
        self.eps = eps

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """执行 2D 层归一化前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 归一化后的输出张量。
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(paddle.nn.Module):
    """基于 Deformable-DETR 和 PaddleDetection 实现的多尺度可变形注意力模块。

    该模块实现多尺度可变形注意力，可通过可学习采样位置和注意力权重关注多个尺度上的特征。

    属性:
        im2col_step (int): im2col 操作步长。
        d_model (int): 模型维度。
        n_levels (int): 特征层级数量。
        n_heads (int): 注意力头数量。
        n_points (int): 每个注意力头在每个特征层级上的采样点数量。
        sampling_offsets (nn.Linear): 生成采样偏移的线性层。
        attention_weights (nn.Linear): 生成注意力权重的线性层。
        value_proj (nn.Linear): 投影 value 的线性层。
        output_proj (nn.Linear): 投影输出的线性层。

    参考:
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        """使用给定参数初始化 MSDeformAttn。

        参数:
            d_model (int): 模型维度。
            n_levels (int): 特征层级数量。
            n_heads (int): 注意力头数量。
            n_points (int): 每个注意力头在每个特征层级上的采样点数量。
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model 必须能被 n_heads 整除，但得到的是 {d_model} 和 {n_heads}")
        _d_per_head = d_model // n_heads
        assert _d_per_head * n_heads == d_model, "`d_model` 必须能被 `n_heads` 整除"
        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = paddle.compat.nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = paddle.compat.nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = paddle.compat.nn.Linear(d_model, d_model)
        self.output_proj = paddle.compat.nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        """重置模块参数。"""
        paddle.nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = paddle.arange(self.n_heads, dtype=paddle.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs()._max(-1, keepdim=True)[0])
            .reshape(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with paddle.no_grad():
            self.sampling_offsets.bias = paddle.nn.Parameter(grid_init.view(-1))
        paddle.nn.init.constant_(self.attention_weights.weight.data, 0.0)
        paddle.nn.init.constant_(self.attention_weights.bias.data, 0.0)
        paddle.nn.init.xavier_uniform_(self.value_proj.weight.data)
        paddle.nn.init.constant_(self.value_proj.bias.data, 0.0)
        paddle.nn.init.xavier_uniform_(self.output_proj.weight.data)
        paddle.nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: paddle.Tensor,
        refer_bbox: paddle.Tensor,
        value: paddle.Tensor,
        value_shapes: list,
        value_mask: (paddle.Tensor | None) = None,
    ) -> paddle.Tensor:
        """执行多尺度可变形注意力的前向传播。

        参数:
            query (paddle.Tensor): query 张量，形状为 [bs, query_length, C]。
            refer_bbox (paddle.Tensor): 参考边界框，形状为 [bs, query_length, n_levels, 2 or 4]，范围为 [0, 1]，
                左上角为 (0,0)，右下角为 (1, 1)，包含 padding 区域。
            value (paddle.Tensor): value 张量，形状为 [bs, value_length, C]。
            value_shapes (list): 形状为 [n_levels, 2] 的列表，[(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]。
            value_mask (paddle.Tensor, optional): mask 张量，形状为 [bs, value_length]，padding 元素为 True，
                非 padding 元素为 False。

        返回:
            (paddle.Tensor): 输出张量，形状为 [bs, Length_{query}, C]。

        参考:
            https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v
        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.reshape(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).reshape(
            bs, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = paddle.compat.nn.functional.softmax(attention_weights, -1).reshape(
            bs, len_q, self.n_heads, self.n_levels, self.n_points
        )
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = paddle.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(axis=-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"reference_points 最后一维必须为 2 或 4，但得到 {num_points}。")
        output = multi_scale_deformable_attn_paddle(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(paddle.nn.Module):
    """受 PaddleDetection 和 Deformable-DETR 实现启发的可变形 Transformer 解码器层。

    该类实现单个解码器层，包含自注意力、使用多尺度可变形注意力的交叉注意力，以及前馈网络。

    属性:
        self_attn (nn.MultiheadAttention): 自注意力模块。
        dropout1 (nn.Dropout): 自注意力之后的 dropout。
        norm1 (nn.LayerNorm): 自注意力之后的层归一化。
        cross_attn (MSDeformAttn): 交叉注意力模块。
        dropout2 (nn.Dropout): 交叉注意力之后的 dropout。
        norm2 (nn.LayerNorm): 交叉注意力之后的层归一化。
        linear1 (nn.Linear): 前馈网络中的第一层线性层。
        act (nn.Module): 激活函数。
        dropout3 (nn.Dropout): 前馈网络中的 dropout。
        linear2 (nn.Linear): 前馈网络中的第二层线性层。
        dropout4 (nn.Dropout): 前馈网络之后的 dropout。
        norm3 (nn.LayerNorm): 前馈网络之后的层归一化。

    参考:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: paddle.nn.Module = paddle.nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """使用给定参数初始化 DeformableTransformerDecoderLayer。

        参数:
            d_model (int): 模型维度。
            n_heads (int): 注意力头数量。
            d_ffn (int): 前馈网络维度。
            dropout (float): dropout 概率。
            act (nn.Module): 激活函数。
            n_levels (int): 特征层级数量。
            n_points (int): 采样点数量。
        """
        super().__init__()
        self.self_attn = paddle.compat.nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = paddle.nn.Dropout(dropout)
        self.norm1 = paddle.nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = paddle.nn.Dropout(dropout)
        self.norm2 = paddle.nn.LayerNorm(d_model)
        self.linear1 = paddle.compat.nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = paddle.nn.Dropout(dropout)
        self.linear2 = paddle.compat.nn.Linear(d_ffn, d_model)
        self.dropout4 = paddle.nn.Dropout(dropout)
        self.norm3 = paddle.nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: paddle.Tensor, pos: (paddle.Tensor | None)) -> paddle.Tensor:
        """如果提供位置编码，则将其加到输入张量上。"""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: paddle.Tensor) -> paddle.Tensor:
        """执行该层中前馈网络部分的前向传播。

        参数:
            tgt (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 经过 FFN 后的输出张量。
        """
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        embed: paddle.Tensor,
        refer_bbox: paddle.Tensor,
        feats: paddle.Tensor,
        shapes: list,
        padding_mask: (paddle.Tensor | None) = None,
        attn_mask: (paddle.Tensor | None) = None,
        query_pos: (paddle.Tensor | None) = None,
    ) -> paddle.Tensor:
        """执行完整解码器层的前向传播。

        参数:
            embed (paddle.Tensor): 输入嵌入。
            refer_bbox (paddle.Tensor): 参考边界框。
            feats (paddle.Tensor): 特征图。
            shapes (list): 特征形状。
            padding_mask (paddle.Tensor, optional): padding mask。
            attn_mask (paddle.Tensor, optional): 注意力 mask。
            query_pos (paddle.Tensor, optional): query 位置嵌入。

        返回:
            (paddle.Tensor): 经过解码器层后的输出张量。
        """
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            embed.transpose(0, 1),
            attn_mask=attn_mask,
        )[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos),
            refer_bbox.unsqueeze(2),
            feats,
            shapes,
            padding_mask,
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(paddle.nn.Module):
    """基于 PaddleDetection 实现的可变形 Transformer 解码器。

    该类实现完整可变形 Transformer 解码器，包含多个解码器层以及用于边界框回归和分类的预测头。

    属性:
        layers (nn.ModuleList): 解码器层列表。
        num_layers (int): 解码器层数。
        hidden_dim (int): 隐藏维度。
        eval_idx (int): 评估时使用的层索引。

    参考:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: paddle.nn.Module,
        num_layers: int,
        eval_idx: int = -1,
    ):
        """使用给定参数初始化 DeformableTransformerDecoder。

        参数:
            hidden_dim (int): 隐藏维度。
            decoder_layer (nn.Module): 解码器层模块。
            num_layers (int): 解码器层数。
            eval_idx (int): 评估时使用的层索引。
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed: paddle.Tensor,
        refer_bbox: paddle.Tensor,
        feats: paddle.Tensor,
        shapes: list,
        bbox_head: paddle.nn.Module,
        score_head: paddle.nn.Module,
        pos_mlp: paddle.nn.Module,
        attn_mask: (paddle.Tensor | None) = None,
        padding_mask: (paddle.Tensor | None) = None,
    ):
        """执行完整解码器的前向传播。

        参数:
            embed (paddle.Tensor): 解码器嵌入。
            refer_bbox (paddle.Tensor): 参考边界框。
            feats (paddle.Tensor): 图像特征。
            shapes (list): 特征形状。
            bbox_head (nn.Module): 边界框预测头。
            score_head (nn.Module): 分数预测头。
            pos_mlp (nn.Module): 位置 MLP。
            attn_mask (paddle.Tensor, optional): 注意力 mask。
            padding_mask (paddle.Tensor, optional): padding mask。

        返回:
            dec_bboxes (paddle.Tensor): 解码后的边界框。
            dec_cls (paddle.Tensor): 解码后的分类分数。
        """
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                refer_bbox,
                feats,
                shapes,
                padding_mask,
                attn_mask,
                pos_mlp(refer_bbox),
            )
            bbox = bbox_head[i](output)
            refined_bbox = paddle.sigmoid(bbox + inverse_sigmoid(refer_bbox))
            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(paddle.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break
            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox
        return paddle.stack(dec_bboxes), paddle.stack(dec_cls)
