# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 基础卷积模块：Conv/DWConv/GhostConv 等封装。
@details
所有卷积类都包含 `autopad` 自动 padding 机制，确保输出尺寸匹配。
关键模块：
- `Conv(c1, c2, k, s, p, g, d, act)`：标准 BN+激活封装卷积
- `DWConv`：深度可分离卷积（group=c1）
- `GhostConv`：Ghost 卷积（廉价线性操作扩展特征图）
- `ConvTranspose`：反卷积（上采样）
- `RepConv`：推理时融合为单卷积的重参化结构
"""

from __future__ import annotations

import paddle

"""卷积模块。"""

import math

import numpy as np

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
)


def autopad(k, p=None, d=1):
    """自动补边，使输出尺寸保持为 'same'。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [(d * (x - 1) + 1) for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class Conv(paddle.nn.Module):
    """带批归一化和激活函数的标准卷积模块。

    属性:
        conv (nn.Conv2d): 卷积层。
        bn (nn.BatchNorm2d): 批归一化层。
        act (nn.Module): 激活函数层。
        default_act (nn.Module): 默认激活函数（SiLU）。
    """

    default_act = paddle.nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """使用给定参数初始化 Conv 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): stride。
            p (int, optional): padding。
            g (int): 分组数。
            d (int): dilation。
            act (bool | nn.Module): 激活函数。
        """
        super().__init__()
        self.conv = paddle.nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = paddle.nn.BatchNorm2D(num_features=c2)
        self.act = (
            self.default_act if act is True else act if isinstance(act, paddle.nn.Module) else paddle.nn.Identity()
        )

    def forward(self, x):
        """对输入张量应用卷积、批归一化和激活函数。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """在无批归一化时应用卷积和激活函数。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """带 Conv 融合的简化 RepConv 模块。

    属性:
        conv (nn.Conv2d): 主 3x3 卷积层。
        cv2 (nn.Conv2d): 额外 1x1 卷积层。
        bn (nn.BatchNorm2d): 批归一化层。
        act (nn.Module): 激活函数层。
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """使用给定参数初始化 Conv2 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): stride。
            p (int, optional): padding。
            g (int): 分组数。
            d (int): dilation。
            act (bool | nn.Module): 激活函数。
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = paddle.nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)

    def forward(self, x):
        """对输入张量应用卷积、批归一化和激活函数。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """对输入张量应用融合后的卷积、批归一化和激活函数。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """融合并行卷积。"""
        w = paddle.zeros_like(self.conv.weight.data)
        i = [(x // 2) for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(paddle.nn.Module):
    """由 1x1 卷积和深度卷积组成的轻量卷积模块。

    该实现基于 PaddleDetection HGNetV2 骨干网络。

    属性:
        conv1 (Conv): 1x1 卷积层。
        conv2 (DWConv): 深度卷积层。
    """

    def __init__(self, c1, c2, k=1, act=paddle.nn.ReLU()):
        """使用给定参数初始化 LightConv 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 深度卷积的卷积核大小。
            act (nn.Module): 激活函数。
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """对输入张量应用两个卷积。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """深度卷积模块。"""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """使用给定参数初始化深度卷积。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): stride。
            d (int): dilation。
            act (bool | nn.Module): 激活函数。
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(paddle.nn.Conv2DTranspose):
    """深度转置卷积模块。"""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """使用给定参数初始化深度转置卷积。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): stride。
            p1 (int): padding。
            p2 (int): output padding。
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(paddle.nn.Module):
    """可选批归一化和激活函数的转置卷积模块。

    属性:
        conv_transpose (nn.ConvTranspose2d): 转置卷积层。
        bn (nn.BatchNorm2d | nn.Identity): 批归一化层。
        act (nn.Module): 激活函数层。
        default_act (nn.Module): 默认激活函数（SiLU）。
    """

    default_act = paddle.nn.SiLU()

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """使用给定参数初始化 ConvTranspose 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): stride。
            p (int): padding。
            bn (bool): 是否使用批归一化。
            act (bool | nn.Module): 激活函数。
        """
        super().__init__()
        self.conv_transpose = paddle.nn.Conv2DTranspose(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            bias_attr=not bn,
        )
        self.bn = paddle.nn.BatchNorm2D(num_features=c2) if bn else paddle.nn.Identity()
        self.act = (
            self.default_act if act is True else act if isinstance(act, paddle.nn.Module) else paddle.nn.Identity()
        )

    def forward(self, x):
        """对输入应用转置卷积、批归一化和激活函数。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """对输入应用转置卷积和激活函数。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        return self.act(self.conv_transpose(x))


class Focus(paddle.nn.Module):
    """用于集中空间特征信息的 Focus 模块。

    将输入张量切成 4 份，并在通道维度拼接。

    属性:
        conv (Conv): 卷积层。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """使用给定参数初始化 Focus 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): stride。
            p (int, optional): padding。
            g (int): 分组数。
            act (bool | nn.Module): 激活函数。
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)

    def forward(self, x):
        """对输入张量应用 Focus 操作和卷积。

        输入形状为 (B, C, H, W)，输出形状为 (B, c2, H/2, W/2)。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        return self.conv(
            paddle.cat(
                (
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ),
                1,
            )
        )


class GhostConv(paddle.nn.Module):
    """Ghost 卷积模块。

    通过廉价操作以更少参数生成更多特征。

    属性:
        cv1 (Conv): 主卷积。
        cv2 (Conv): 廉价操作卷积。

    参考:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """使用给定参数初始化 Ghost 卷积模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): stride。
            g (int): 分组数。
            act (bool | nn.Module): 激活函数。
        """
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """对输入张量应用 Ghost 卷积。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 拼接特征后的输出张量。
        """
        y = self.cv1(x)
        return paddle.cat((y, self.cv2(y)), 1)


class RepConv(paddle.nn.Module):
    """支持训练和部署模式的 RepConv 模块。

    该模块用于 RT-DETR，并可在推理时融合卷积以提升效率。

    属性:
        conv1 (Conv): 3x3 卷积。
        conv2 (Conv): 1x1 卷积。
        bn (nn.BatchNorm2d, optional): identity 分支的批归一化。
        act (nn.Module): 激活函数。
        default_act (nn.Module): 默认激活函数（SiLU）。

    参考:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = paddle.nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """使用给定参数初始化 RepConv 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): stride。
            p (int): padding。
            g (int): 分组数。
            d (int): dilation。
            act (bool | nn.Module): 激活函数。
            bn (bool): 是否对 identity 分支使用批归一化。
            deploy (bool): 推理部署模式。
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = (
            self.default_act if act is True else act if isinstance(act, paddle.nn.Module) else paddle.nn.Identity()
        )
        self.bn = paddle.nn.BatchNorm2D(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=p - k // 2, g=g, act=False)

    def forward_fuse(self, x):
        """部署模式前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """训练模式前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """通过融合卷积计算等效卷积核和偏置。

        返回:
            (paddle.Tensor): 等效卷积核
            (paddle.Tensor): 等效偏置
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """将 1x1 卷积核补边为 3x3 大小。

        参数:
            kernel1x1 (paddle.Tensor): 1x1 卷积核。

        返回:
            (paddle.Tensor): 补边后的 3x3 卷积核。
        """
        if kernel1x1 is None:
            return 0
        else:
            return paddle.compat.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """将批归一化与卷积权重融合。

        参数:
            branch (Conv | nn.BatchNorm2d | None): 待融合分支。

        返回:
            kernel (paddle.Tensor): 融合后的卷积核。
            bias (paddle.Tensor): 融合后的偏置。
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        elif isinstance(branch, paddle.nn.BatchNorm2D):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """创建单个等效卷积，用于推理时融合卷积。"""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = paddle.nn.Conv2D(
            in_channels=self.conv1.conv._in_channels,
            out_channels=self.conv1.conv._out_channels,
            kernel_size=self.conv1.conv._kernel_size,
            stride=self.conv1.conv._stride,
            padding=self.conv1.conv._padding,
            dilation=self.conv1.conv._dilation,
            groups=self.conv1.conv._groups,
            bias_attr=True,
        )
        self.conv.weight.set_value(kernel)
        self.conv.bias.set_value(bias)
        for para in self.parameters():
            para.stop_gradient = True
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(paddle.nn.Module):
    """用于特征重校准的通道注意力模块。

    基于全局平均池化对通道施加注意力权重。

    属性:
        pool (nn.AdaptiveAvgPool2d): 全局平均池化。
        fc (nn.Conv2d): 以 1x1 卷积实现的全连接层。
        act (nn.Sigmoid): 用于注意力权重的 Sigmoid 激活。

    参考:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """初始化通道注意力模块。

        参数:
            channels (int): 输入通道数。
        """
        super().__init__()
        self.pool = paddle.nn.AdaptiveAvgPool2d(1)
        self.fc = paddle.nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = paddle.nn.Sigmoid()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """对输入张量应用通道注意力。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 经过通道注意力后的输出张量。
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(paddle.nn.Module):
    """用于特征重校准的空间注意力模块。

    基于通道统计对空间维度施加注意力权重。

    属性:
        cv1 (nn.Conv2d): 空间注意力卷积层。
        act (nn.Sigmoid): 用于注意力权重的 Sigmoid 激活。
    """

    def __init__(self, kernel_size=7):
        """初始化空间注意力模块。

        参数:
            kernel_size (int): 卷积核大小（3 或 7）。
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel_size 必须为 3 或 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = paddle.nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = paddle.nn.Sigmoid()

    def forward(self, x):
        """对输入张量应用空间注意力。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 经过空间注意力后的输出张量。
        """
        return x * self.act(
            self.cv1(
                paddle.cat(
                    [
                        paddle.mean(x, 1, keepdim=True),
                        paddle.compat.max(x, 1, keepdim=True)[0],
                    ],
                    1,
                )
            )
        )


class CBAM(paddle.nn.Module):
    """卷积块注意力模块（CBAM）。

    结合通道注意力和空间注意力机制，对特征进行综合细化。

    属性:
        channel_attention (ChannelAttention): 通道注意力模块。
        spatial_attention (SpatialAttention): 空间注意力模块。
    """

    def __init__(self, c1, kernel_size=7):
        """使用给定参数初始化 CBAM。

        参数:
            c1 (int): 输入通道数。
            kernel_size (int): 空间注意力卷积核大小。
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """依次对输入张量应用通道注意力和空间注意力。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 注意力处理后的输出张量。
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(paddle.nn.Module):
    """沿指定维度拼接张量列表。

    属性:
        d (int): 拼接张量的维度。
    """

    def __init__(self, dimension=1):
        """初始化 Concat 模块。

        参数:
            dimension (int): 拼接张量的维度。
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[paddle.Tensor]):
        """沿指定维度拼接输入张量。

        参数:
            x (list[paddle.Tensor]): 输入张量列表。

        返回:
            (paddle.Tensor): 拼接后的张量。
        """
        return paddle.cat(x, self.d)


class Index(paddle.nn.Module):
    """返回输入中的指定索引项。

    属性:
        index (int): 从输入中选择的索引。
    """

    def __init__(self, index=0):
        """初始化 Index 模块。

        参数:
            index (int): 从输入中选择的索引。
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[paddle.Tensor]):
        """从输入中选择并返回指定索引项。

        参数:
            x (list[paddle.Tensor]): 输入张量列表。

        返回:
            (paddle.Tensor): 选中的张量。
        """
        return x[self.index]
