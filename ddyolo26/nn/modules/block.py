# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 网络结构基础块：C2f/C3/SPPF/Proto 等模块定义。
@details
本文件定义了构建 YOLO26 骨干网络（Backbone）和颈部网络（Neck）所需的
所有复合模块，包括：
- C2f / C3：高效特征提取块（CSP + 残差结构）
- SPPF：空间金字塔池化（加速版）
- Proto / Proto26：实例分割 Prototype 生成（YOLO26 引入多尺度版）
- DFL：分布焦点损失卷积层（YOLO26 reg_max=1 时被 Identity 替代）
- Bottleneck / RepVGGBlock：瓶颈结构与可重参化模块
- 注意力模块（CBAM/AAttn 等）
"""

from __future__ import annotations
import sys
import paddle


from ddyolo26.paddle_utils import *

"""网络块模块。"""

from ddyolo26.utils.runtime import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
)


class DFL(paddle.nn.Module):
    """Distribution Focal Loss（DFL）的积分模块。

    由 Generalized Focal Loss 提出：https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """使用给定输入通道数初始化卷积层。

        参数:
            c1 (int): 输入通道数。
        """
        super().__init__()
        self.conv = paddle.nn.Conv2d(c1, 1, 1, bias=False)
        for p in self.conv.parameters():
            p.stop_gradient = True
        x = paddle.arange(c1, dtype=paddle.float32)
        self.conv.weight.data[:] = paddle.nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """对输入张量应用 DFL 模块，并返回变换后的输出。"""
        b, _, a = x.shape
        return self.conv(x.reshape(b, 4, self.c1, a).transpose(2, 1).softmax(1)).reshape(b, 4, a)


class Proto(paddle.nn.Module):
    """分割模型使用的 PaddleYOLO-RKNN Proto 模块。"""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """使用指定 proto 和 mask 数初始化 PaddleYOLO-RKNN Proto 模块。

        参数:
            c1 (int): 输入通道数。
            c_ (int): 中间通道数。
            c2 (int): 输出通道数（proto 数量）。
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = paddle.nn.Conv2DTranspose(
            in_channels=c_,
            out_channels=c_,
            kernel_size=2,
            stride=2,
            padding=0,
            bias_attr=True,
        )
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过上采样后的特征执行各层前向传播。"""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(paddle.nn.Module):
    """PPHGNetV2 的 StemBlock，包含 5 个卷积和一个 maxpool2d。

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """初始化 PPHGNetV2 的 StemBlock。

        参数:
            c1 (int): 输入通道数。
            cm (int): 中间通道数。
            c2 (int): 输出通道数。
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=paddle.nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=paddle.nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=paddle.nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=paddle.nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=paddle.nn.ReLU())
        self.pool = paddle.nn.MaxPool2D(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """PPHGNetV2 骨干层的前向传播。"""
        x = self.stem1(x)
        x = paddle.compat.nn.functional.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = paddle.compat.nn.functional.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = paddle.cat([x1, x2], axis=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(paddle.nn.Module):
    """PPHGNetV2 的 HG_Block，包含 2 个卷积和 LightConv。

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: paddle.nn.Module = paddle.nn.ReLU(),
    ):
        """使用指定参数初始化 HGBlock。

        参数:
            c1 (int): 输入通道数。
            cm (int): 中间通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            n (int): LightConv 或 Conv 块数量。
            lightconv (bool): 是否使用 LightConv。
            shortcut (bool): 是否使用 shortcut 连接。
            act (nn.Module): 激活函数。
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = paddle.nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """PPHGNetV2 骨干层的前向传播。"""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(paddle.cat(y, 1)))
        return y + x if self.add else y


class SPP(paddle.nn.Module):
    """Spatial Pyramid Pooling（SPP）层：https://arxiv.org/abs/1406.4729。"""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """使用输入/输出通道数和池化核大小初始化 SPP 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (tuple): 最大池化核大小。
        """
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = paddle.nn.ModuleList([paddle.nn.MaxPool2D(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """SPP 层前向传播，执行空间金字塔池化。"""
        x = self.cv1(x)
        return self.cv2(paddle.cat([x] + [m(x) for m in self.m], 1))


class SPPF(paddle.nn.Module):
    """YOLOv5 中的 Spatial Pyramid Pooling - Fast（SPPF）层。"""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """使用给定输入/输出通道数和卷积核大小初始化 SPPF 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            n (int): 池化迭代次数。
            shortcut (bool): 是否使用 shortcut 连接。

        说明:
            该模块等价于 SPP(k=(5, 9, 13))。
        """
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = paddle.nn.MaxPool2D(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """对输入应用连续池化操作，并返回拼接后的特征图。"""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(paddle.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(paddle.nn.Module):
    """带 1 个卷积的 CSP Bottleneck。"""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """初始化带 1 个卷积的 CSP Bottleneck。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): 卷积数量。
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = paddle.nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """对输入张量应用卷积和残差连接。"""
        y = self.cv1(x)
        return self.m(y) + y


class C2(paddle.nn.Module):
    """带 2 个卷积的 CSP Bottleneck。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化带 2 个卷积的 CSP Bottleneck。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = paddle.nn.Sequential(
            *(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过带 2 个卷积的 CSP Bottleneck 执行前向传播。"""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(paddle.cat((self.m(a), b), 1))


class C2f(paddle.nn.Module):
    """带 2 个卷积的 CSP Bottleneck 更快实现。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化带 2 个卷积的 CSP Bottleneck。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = paddle.nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 C2f 层执行前向传播。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(paddle.cat(y, 1))

    def forward_split(self, x: paddle.Tensor) -> paddle.Tensor:
        """使用 split() 而非 chunk() 执行前向传播。"""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(paddle.cat(y, 1))


class C3(paddle.nn.Module):
    """带 3 个卷积的 CSP Bottleneck。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化带 3 个卷积的 CSP Bottleneck。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = paddle.nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过带 3 个卷积的 CSP Bottleneck 执行前向传播。"""
        return self.cv3(paddle.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """使用交叉卷积的 C3 模块。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化使用交叉卷积的 C3 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = paddle.nn.Sequential(
            *(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n))
        )


class RepC3(paddle.nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """使用 RepConv 块初始化 RepC3 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): RepConv 块数量。
            e (float): 扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = paddle.nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else paddle.nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """RepC3 模块前向传播。"""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """带 TransformerBlock() 的 C3 模块。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        """使用 TransformerBlock 初始化 C3 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Transformer 块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """带 GhostBottleneck() 的 C3 模块。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        """使用 GhostBottleneck 初始化 C3 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Ghost bottleneck 块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = paddle.nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(paddle.nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """初始化 Ghost Bottleneck 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): stride。
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = paddle.nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            DWConv(c_, c_, k, s, act=False) if s == 2 else paddle.nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False),
        )
        self.shortcut = (
            paddle.nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False))
            if s == 2
            else paddle.nn.Identity()
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """对输入张量应用跳连和加法。"""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(paddle.nn.Module):
    """标准 bottleneck。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        """初始化标准 bottleneck 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            k (tuple): 卷积核大小。
            e (float): 扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """应用 bottleneck，并可选使用 shortcut 连接。"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(paddle.nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化 CSP Bottleneck。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = paddle.nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = paddle.nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = paddle.nn.BatchNorm2D(num_features=2 * c_)
        self.act = paddle.nn.SiLU()
        self.m = paddle.nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """应用带 4 个卷积的 CSP bottleneck。"""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(paddle.cat((y1, y2), 1))))


class ResNetBlock(paddle.nn.Module):
    """带标准卷积层的 ResNet 块。"""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """初始化 ResNet 块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            s (int): stride。
            e (int): 扩展比例。
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = (
            paddle.nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else paddle.nn.Identity()
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 ResNet 块执行前向传播。"""
        return paddle.nn.functional.relu(x=self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(paddle.nn.Module):
    """由多个 ResNet 块组成的 ResNet 层。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        s: int = 1,
        is_first: bool = False,
        n: int = 1,
        e: int = 4,
    ):
        """初始化 ResNet 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            s (int): stride。
            is_first (bool): 是否为第一层。
            n (int): ResNet 块数量。
            e (int): 扩展比例。
        """
        super().__init__()
        self.is_first = is_first
        if self.is_first:
            self.layer = paddle.nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True),
                paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = paddle.nn.Sequential(*blocks)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 ResNet 层执行前向传播。"""
        return self.layer(x)


class MaxSigmoidAttnBlock(paddle.nn.Module):
    """Max Sigmoid 注意力块。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        nh: int = 1,
        ec: int = 128,
        gc: int = 512,
        scale: bool = False,
    ):
        """初始化 MaxSigmoidAttnBlock。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            nh (int): head 数量。
            ec (int): 嵌入通道数。
            gc (int): guide 通道数。
            scale (bool): 是否使用可学习 scale 参数。
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = paddle.compat.nn.Linear(gc, ec)
        self.bias = paddle.nn.Parameter(paddle.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = paddle.nn.Parameter(paddle.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: paddle.Tensor, guide: paddle.Tensor) -> paddle.Tensor:
        """MaxSigmoidAttnBlock 前向传播。

        参数:
            x (paddle.Tensor): 输入张量。
            guide (paddle.Tensor): guide 张量。

        返回:
            (paddle.Tensor): 注意力处理后的输出张量。
        """
        bs, _, h, w = x.shape
        guide = self.gl(guide)
        guide = guide.reshape(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.reshape(bs, self.nh, self.hc, h, w)
        aw = paddle.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = (aw.max(axis=-1), aw.argmax(axis=-1))[0]
        aw = aw / self.hc**0.5
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale
        x = self.proj_conv(x)
        x = x.reshape(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.reshape(bs, -1, h, w)


class C2fAttn(paddle.nn.Module):
    """带额外注意力模块的 C2f 模块。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化带注意力机制的 C2f 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 块数量。
            ec (int): 注意力嵌入通道数。
            nh (int): 注意力 head 数量。
            gc (int): 注意力 guide 通道数。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = paddle.nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: paddle.Tensor, guide: paddle.Tensor) -> paddle.Tensor:
        """通过带注意力的 C2f 层执行前向传播。

        参数:
            x (paddle.Tensor): 输入张量。
            guide (paddle.Tensor): 注意力使用的 guide 张量。

        返回:
            (paddle.Tensor): 处理后的输出张量。
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(paddle.cat(y, 1))

    def forward_split(self, x: paddle.Tensor, guide: paddle.Tensor) -> paddle.Tensor:
        """使用 split() 而非 chunk() 执行前向传播。

        参数:
            x (paddle.Tensor): 输入张量。
            guide (paddle.Tensor): 注意力使用的 guide 张量。

        返回:
            (paddle.Tensor): 处理后的输出张量。
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(paddle.cat(y, 1))


class ImagePoolingAttn(paddle.nn.Module):
    """ImagePoolingAttn：使用图像感知信息增强文本嵌入。"""

    def __init__(
        self,
        ec: int = 256,
        ch: tuple[int, ...] = (),
        ct: int = 512,
        nh: int = 8,
        k: int = 3,
        scale: bool = False,
    ):
        """初始化 ImagePoolingAttn 模块。

        参数:
            ec (int): 嵌入通道数。
            ch (tuple): 特征图通道维度。
            ct (int): 文本嵌入通道维度。
            nh (int): 注意力头数量。
            k (int): 池化核大小。
            scale (bool): 是否使用可学习 scale 参数。
        """
        super().__init__()
        nf = len(ch)
        self.query = paddle.nn.Sequential(paddle.nn.LayerNorm(ct), paddle.compat.nn.Linear(ct, ec))
        self.key = paddle.nn.Sequential(paddle.nn.LayerNorm(ec), paddle.compat.nn.Linear(ec, ec))
        self.value = paddle.nn.Sequential(paddle.nn.LayerNorm(ec), paddle.compat.nn.Linear(ec, ec))
        self.proj = paddle.compat.nn.Linear(ec, ct)
        self.scale = paddle.nn.Parameter(paddle.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = paddle.nn.ModuleList(
            [paddle.nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch]
        )
        self.im_pools = paddle.nn.ModuleList([paddle.nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[paddle.Tensor], text: paddle.Tensor) -> paddle.Tensor:
        """ImagePoolingAttn 前向传播。

        参数:
            x (list[paddle.Tensor]): 输入特征图列表。
            text (paddle.Tensor): 文本嵌入。

        返回:
            (paddle.Tensor): 增强后的文本嵌入。
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).reshape(bs, -1, num_patches) for x, proj, pool in zip(x, self.projections, self.im_pools)]
        x = paddle.cat(x, axis=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)
        aw = paddle.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / self.hc**0.5
        aw = paddle.compat.nn.functional.softmax(aw, dim=-1)
        x = paddle.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class RepBottleneck(Bottleneck):
    """Rep bottleneck。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        """初始化 RepBottleneck。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            k (tuple): 卷积核大小。
            e (float): 扩展比例。
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """用于高效特征提取的可重复 Cross Stage Partial Network（RepCSP）模块。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化 RepCSP 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): RepBottleneck 块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = paddle.nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(paddle.nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """初始化 CSP-ELAN 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            c3 (int): 中间通道数。
            c4 (int): RepCSP 的中间通道数。
            n (int): RepCSP 块数量。
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = paddle.nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = paddle.nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 RepNCSPELAN4 层执行前向传播。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(paddle.cat(y, 1))

    def forward_split(self, x: paddle.Tensor) -> paddle.Tensor:
        """使用 split() 而非 chunk() 执行前向传播。"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(paddle.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """带 4 个卷积的 ELAN1 模块。"""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """初始化 ELAN1 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            c3 (int): 中间通道数。
            c4 (int): 卷积使用的中间通道数。
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)


class AConv(paddle.nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """初始化 AConv 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 AConv 层执行前向传播。"""
        x = paddle.nn.functional.avg_pool2d(
            x=x, kernel_size=2, stride=1, padding=0, ceil_mode=False, exclusive=not True
        )
        return self.cv1(x)


class ADown(paddle.nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """初始化 ADown 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 ADown 层执行前向传播。"""
        x = paddle.nn.functional.avg_pool2d(
            x=x, kernel_size=2, stride=1, padding=0, ceil_mode=False, exclusive=not True
        )
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = paddle.nn.functional.max_pool2d(x=x2, kernel_size=3, stride=2, padding=1)
        x2 = self.cv2(x2)
        return paddle.cat((x1, x2), 1)


class SPPELAN(paddle.nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """初始化 SPP-ELAN 块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            c3 (int): 中间通道数。
            k (int): 最大池化核大小。
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = paddle.nn.MaxPool2D(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = paddle.nn.MaxPool2D(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = paddle.nn.MaxPool2D(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 SPPELAN 层执行前向传播。"""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(paddle.cat(y, 1))


class CBLinear(paddle.nn.Module):
    """CBLinear."""

    def __init__(
        self,
        c1: int,
        c2s: list[int],
        k: int = 1,
        s: int = 1,
        p: (int | None) = None,
        g: int = 1,
    ):
        """初始化 CBLinear 模块。

        参数:
            c1 (int): 输入通道数。
            c2s (list[int]): 输出通道大小列表。
            k (int): 卷积核大小。
            s (int): stride。
            p (int | None): padding。
            g (int): 分组数。
        """
        super().__init__()
        self.c2s = c2s
        self.conv = paddle.nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: paddle.Tensor) -> list[paddle.Tensor]:
        """通过 CBLinear 层执行前向传播。"""
        return self.conv(x).split(self.c2s, axis=1)


class CBFuse(paddle.nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """初始化 CBFuse 模块。

        参数:
            idx (list[int]): 特征选择索引。
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[paddle.Tensor]) -> paddle.Tensor:
        """通过 CBFuse 层执行前向传播。

        参数:
            xs (list[paddle.Tensor]): 输入张量列表。

        返回:
            (paddle.Tensor): 融合后的输出张量。
        """
        target_size = xs[-1].shape[2:]
        res = [
            paddle.nn.functional.interpolate(x[self.idx[i]], size=target_size, mode="nearest")
            for i, x in enumerate(xs[:-1])
        ]
        return paddle.sum(paddle.stack(res + xs[-1:]), axis=0)


class C3f(paddle.nn.Module):
    """带 3 个卷积的 CSP Bottleneck 更快实现。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化带 3 个卷积的 CSP bottleneck 层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)
        self.m = paddle.nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 C3f 层执行前向传播。"""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(paddle.cat(y, 1))


class C3k2(C2f):
    """带 2 个卷积的 CSP Bottleneck 更快实现。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """初始化 C3k2 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): 块数量。
            c3k (bool): 是否使用 C3k 块。
            e (float): 扩展比例。
            attn (bool): 是否使用注意力块。
            g (int): 卷积分组数。
            shortcut (bool): 是否使用 shortcut 连接。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = paddle.nn.ModuleList(
            paddle.nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k 是可自定义卷积核大小的 CSP bottleneck 模块，用于神经网络特征提取。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ):
        """初始化 C3k 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
            k (int): 卷积核大小。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = paddle.nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(paddle.nn.Module):
    """RepVGGDW 表示 RepVGG 架构中的深度卷积块。"""

    def __init__(self, ed: int) -> None:
        """初始化 RepVGGDW 模块。

        参数:
            ed (int): 输入和输出通道数。
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = paddle.nn.SiLU()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """执行 RepVGGDW 块的前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 应用深度卷积后的输出张量。
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: paddle.Tensor) -> paddle.Tensor:
        """执行融合后 RepVGGDW 块的前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 应用深度卷积后的输出张量。
        """
        return self.act(self.conv(x))

    @paddle.no_grad()
    def fuse(self):
        """融合 RepVGGDW 块中的卷积层。

        该方法融合卷积层，并相应更新权重和偏置。
        """
        if not hasattr(self, "conv1"):
            return
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        conv1_w = paddle.compat.nn.functional.pad(conv1_w, [2, 2, 2, 2])
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        self.conv = conv
        del self.conv1


class CIB(paddle.nn.Module):
    """Compact Inverted Block（CIB）模块。

    参数:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        shortcut (bool, optional): 是否添加 shortcut 连接，默认 True。
        e (float, optional): 隐藏通道缩放因子，默认 0.5。
        lk (bool, optional): 第三个卷积层是否使用 RepVGGDW，默认 False。
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """初始化 CIB 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            shortcut (bool): 是否使用 shortcut 连接。
            e (float): 扩展比例。
            lk (bool): 是否使用 RepVGGDW。
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = paddle.nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """CIB 模块前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 输出张量。
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB 表示由 C2f 和 CIB 模块组成的卷积块。

    参数:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        n (int, optional): 堆叠的 CIB 模块数量，默认 1。
        shortcut (bool, optional): 是否使用 shortcut 连接，默认 False。
        lk (bool, optional): 是否使用大核，默认 False。
        g (int, optional): 分组卷积的分组数，默认 1。
        e (float, optional): CIB 模块扩展比例，默认 0.5。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        lk: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化 C2fCIB 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): CIB 模块数量。
            shortcut (bool): 是否使用 shortcut 连接。
            lk (bool): 是否使用大核。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = paddle.nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(paddle.nn.Module):
    """对输入张量执行自注意力的 Attention 模块。

    参数:
        dim (int): 输入张量维度。
        num_heads (int): 注意力头数量。
        attn_ratio (float): 注意力 key 维度与 head 维度的比例。

    属性:
        num_heads (int): 注意力头数量。
        head_dim (int): 每个注意力头的维度。
        key_dim (int): 注意力 key 维度。
        scale (float): 注意力分数缩放因子。
        qkv (Conv): 计算 query、key 和 value 的卷积层。
        proj (Conv): 投影注意力结果的卷积层。
        pe (Conv): 位置编码卷积层。
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """初始化多头注意力模块。

        参数:
            dim (int): 输入维度。
            num_heads (int): 注意力头数量。
            attn_ratio (float): key 维度的注意力比例。
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Attention 模块前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 自注意力后的输出张量。
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], axis=2
        )
        attn = q.transpose(-2, -1) @ k * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(paddle.nn.Module):
    """实现神经网络 Position-Sensitive Attention 块的 PSABlock。

    该类封装多头注意力和前馈神经网络层，并可选使用 shortcut 连接。

    属性:
        attn (Attention): 多头注意力模块。
        ffn (nn.Sequential): 前馈神经网络模块。
        add (bool): 是否添加 shortcut 连接的标志。

    方法:
        forward: 通过 PSABlock 执行前向传播，应用注意力和前馈层。

    示例:
        创建 PSABlock 并执行前向传播
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = paddle.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """初始化 PSABlock。

        参数:
            c (int): 输入和输出通道数。
            attn_ratio (float): key 维度的注意力比例。
            num_heads (int): 注意力头数量。
            shortcut (bool): 是否使用 shortcut 连接。
        """
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = paddle.nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 PSABlock 执行前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 经过注意力和前馈处理后的输出张量。
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(paddle.nn.Module):
    """用于在神经网络中实现 Position-Sensitive Attention 的 PSA 类。

    该类封装对输入张量应用位置敏感注意力和前馈网络的功能，用于增强特征提取和处理能力。

    属性:
        c (int): 初始卷积后的隐藏通道数。
        cv1 (Conv): 将输入通道数压缩到 2*c 的 1x1 卷积层。
        cv2 (Conv): 将输出通道数压缩到 c1 的 1x1 卷积层。
        attn (Attention): 位置敏感注意力模块。
        ffn (nn.Sequential): 用于进一步处理的前馈网络。

    方法:
        forward: 对输入张量应用位置敏感注意力和前馈网络。

    示例:
        创建 PSA 模块并应用到输入张量
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = paddle.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """初始化 PSA 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            e (float): 扩展比例。
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = paddle.nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """执行 PSA 模块前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 经过注意力和前馈处理后的输出张量。
        """
        a, b = self.cv1(x).split((self.c, self.c), axis=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(paddle.cat((a, b), 1))


class C2PSA(paddle.nn.Module):
    """带注意力机制的 C2PSA 模块，用于增强特征提取和处理。

    该模块实现带注意力机制的卷积块，以增强特征提取和处理能力。它包含一系列用于自注意力和前馈操作的
    PSABlock 模块。

    属性:
        c (int): 隐藏通道数。
        cv1 (Conv): 将输入通道数压缩到 2*c 的 1x1 卷积层。
        cv2 (Conv): 将输出通道数压缩到 c1 的 1x1 卷积层。
        m (nn.Sequential): PSABlock 模块顺序容器，用于注意力和前馈操作。

    方法:
        forward: 通过 C2PSA 模块执行前向传播，应用注意力和前馈操作。

    示例:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = paddle.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    说明:
        该模块本质上与 PSA 模块相同，但重构为可以堆叠更多 PSABlock 模块。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """初始化 C2PSA 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): PSABlock 模块数量。
            e (float): 扩展比例。
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = paddle.nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过一系列 PSA 块处理输入张量。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 处理后的输出张量。
        """
        a, b = self.cv1(x).split((self.c, self.c), axis=1)
        b = self.m(b)
        return self.cv2(paddle.cat((a, b), 1))


class C2fPSA(C2f):
    """使用 PSA 块增强特征提取的 C2fPSA 模块。

    该类通过引入 PSA 块扩展 C2f 模块，用于改进注意力机制和特征提取。

    属性:
        c (int): 隐藏通道数。
        cv1 (Conv): 将输入通道数压缩到 2*c 的 1x1 卷积层。
        cv2 (Conv): 将输出通道数压缩到 c2 的 1x1 卷积层。
        m (nn.ModuleList): 用于特征提取的 PSABlock 模块列表。

    方法:
        forward: 通过 C2fPSA 模块执行前向传播。
        forward_split: 使用 split() 而非 chunk() 执行前向传播。

    示例:
        >>> from ddyolo26.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = paddle.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """初始化 C2fPSA 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): PSABlock 模块数量。
            e (float): 扩展比例。
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = paddle.nn.ModuleList(
            PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)
        )


class SCDown(paddle.nn.Module):
    """使用可分离卷积执行下采样的 SCDown 模块。

    该模块结合逐点卷积和深度卷积执行下采样，在保留通道信息的同时高效降低输入张量空间尺寸。

    属性:
        cv1 (Conv): 降低通道数的逐点卷积层。
        cv2 (Conv): 执行空间下采样的深度卷积层。

    方法:
        forward: 将 SCDown 模块应用到输入张量。

    示例:
        >>> from ddyolo26.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = paddle.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        Shape([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """初始化 SCDown 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): stride。
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """对输入张量应用卷积和下采样。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 下采样后的输出张量。
        """
        return self.cv2(self.cv1(x))


class AAttn(paddle.nn.Module):
    """YOLO 模型的 Area-attention 模块，提供高效注意力机制。

    该模块实现基于区域的注意力机制，以空间感知方式处理输入特征，尤其适合目标检测任务。

    属性:
        area (int): 特征图划分的区域数量。
        num_heads (int): 注意力机制划分的 head 数量。
        head_dim (int): 每个注意力 head 的维度。
        qkv (Conv): 计算 query、key 和 value 张量的卷积层。
        proj (Conv): 投影卷积层。
        pe (Conv): 位置编码卷积层。

    方法:
        forward: 对输入张量应用 area-attention。

    示例:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = paddle.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        Shape([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """初始化 YOLO 模型的 Area-attention 模块。

        参数:
            dim (int): 隐藏通道数。
            num_heads (int): 注意力机制划分的 head 数量。
            area (int): 特征图划分的区域数量。
        """
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 area-attention 处理输入张量。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): area-attention 后的输出张量。
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.reshape(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], axis=2)
        )
        attn = q.transpose(-2, -1) @ k * self.head_dim**-0.5
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)
        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = x + self.pe(v)
        return self.proj(x)


class ABlock(paddle.nn.Module):
    """YOLO 模型中用于高效特征提取的 Area-attention 块模块。

    该模块结合 area-attention 机制和前馈网络来处理特征图。它使用基于区域的新型注意力方法，
    在保持效果的同时比传统自注意力更高效。

    属性:
        attn (AAttn): 用于处理空间特征的 area-attention 模块。
        mlp (nn.Sequential): 用于特征变换的多层感知机。

    方法:
        _init_weights: 使用截断正态分布初始化模块权重。
        forward: 对输入张量应用 area-attention 和前馈处理。

    示例:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = paddle.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        Shape([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """初始化 Area-attention 块模块。

        参数:
            dim (int): 输入通道数。
            num_heads (int): 注意力机制划分的 head 数量。
            mlp_ratio (float): MLP 隐藏维度扩展比例。
            area (int): 特征图划分的区域数量。
        """
        super().__init__()
        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = paddle.nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: paddle.nn.Module):
        """使用截断正态分布初始化权重。

        参数:
            m (nn.Module): 待初始化模块。
        """
        if isinstance(m, paddle.nn.Conv2d):
            paddle.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                paddle.nn.init.constant_(m.bias, 0)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 ABlock 执行前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 经过 area-attention 和前馈处理后的输出张量。
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(paddle.nn.Module):
    """Area-Attention C2f 模块，通过基于区域的注意力机制增强特征提取。

    该模块通过引入 area-attention 和 ABlock 层扩展 C2f 架构，以改进特征处理。它同时支持
    area-attention 和标准卷积模式。

    属性:
        cv1 (Conv): 将输入通道降到隐藏通道的初始 1x1 卷积层。
        cv2 (Conv): 处理拼接特征的最终 1x1 卷积层。
        gamma (nn.Parameter | None): 使用 area attention 时用于残差缩放的可学习参数。
        m (nn.ModuleList): 用于特征处理的 ABlock 或 C3k 模块列表。

    方法:
        forward: 通过 area-attention 或标准卷积路径处理输入。

    示例:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = paddle.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        Shape([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """初始化 Area-Attention C2f 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): 堆叠的 ABlock 或 C3k 模块数量。
            a2 (bool): 是否使用 area attention 块；若为 False，则改用 C3k 块。
            area (int): 特征图划分的区域数量。
            residual (bool): 是否使用带可学习 gamma 参数的残差连接。
            mlp_ratio (float): MLP 隐藏维度扩展比例。
            e (float): 隐藏通道扩展比例。
            g (int): 分组卷积分组数。
            shortcut (bool): C3k 块中是否使用 shortcut 连接。
        """
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "ABlock 维度必须是 32 的倍数。"
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        self.gamma = paddle.nn.Parameter(0.01 * paddle.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = paddle.nn.ModuleList(
            paddle.nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """通过 A2C2f 层执行前向传播。

        参数:
            x (paddle.Tensor): 输入张量。

        返回:
            (paddle.Tensor): 处理后的输出张量。
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(paddle.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.reshape(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(paddle.nn.Module):
    """用于 Transformer 架构的 SwiGLU 前馈网络。"""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """使用输入维度、输出维度和扩展因子初始化 SwiGLU FFN。

        参数:
            gc (int): guide 通道数。
            ec (int): 嵌入通道数。
            e (int): 扩展因子。
        """
        super().__init__()
        self.w12 = paddle.compat.nn.Linear(gc, e * ec)
        self.w3 = paddle.compat.nn.Linear(e * ec // 2, ec)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """对输入特征应用 SwiGLU 变换。"""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, axis=-1)
        hidden = paddle.nn.functional.silu(x1) * x2
        return self.w3(hidden)


class Residual(paddle.nn.Module):
    """神经网络模块的残差连接包装器。"""

    def __init__(self, m: paddle.nn.Module) -> None:
        """使用被包装模块初始化残差模块。

        参数:
            m (nn.Module): 需要添加残差连接的模块。
        """
        super().__init__()
        self.m = m
        paddle.nn.init.zeros_(self.m.w3.bias)
        paddle.nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """对输入特征应用残差连接。"""
        return x + self.m(x)


class Proto26(Proto):
    """分割模型使用的 PaddleYOLO-RKNN Proto26 模块。"""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """使用指定 proto 和 mask 数初始化 PaddleYOLO-RKNN Proto 模块。

        参数:
            ch (tuple): 骨干特征图各层通道数。
            c_ (int): 中间通道数。
            c2 (int): 输出通道数（proto 数量）。
            nc (int): 语义分割类别数。
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = paddle.nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = paddle.nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), paddle.nn.Conv2d(c_, nc, 1))

    def forward(self, x: paddle.Tensor, return_semseg: bool = True) -> paddle.Tensor:
        """通过融合多尺度特征图并生成 proto masks 来执行前向传播。"""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = paddle.nn.functional.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semseg:
            semseg = self.semseg(feat)
            return p, semseg
        return p

    def fuse(self):
        """移除语义分割头以融合模型用于推理。"""
        self.semseg = None


class RealNVP(paddle.nn.Module):
    """RealNVP：基于 flow 的生成模型。

    参考:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """获取单个可逆映射中的 scale 模型。"""
        return paddle.nn.Sequential(
            paddle.compat.nn.Linear(2, 64),
            paddle.nn.SiLU(),
            paddle.compat.nn.Linear(64, 64),
            paddle.nn.SiLU(),
            paddle.compat.nn.Linear(64, 2),
            paddle.nn.Tanh(),
        )

    @staticmethod
    def nett():
        """获取单个可逆映射中的 translation 模型。"""
        return paddle.nn.Sequential(
            paddle.compat.nn.Linear(2, 64),
            paddle.nn.SiLU(),
            paddle.compat.nn.Linear(64, 64),
            paddle.nn.SiLU(),
            paddle.compat.nn.Linear(64, 2),
        )

    @property
    def prior(self):
        """先验分布。"""
        return paddle.distribution.MultivariateNormal(loc=self.loc, covariance_matrix=self.cov)

    def __init__(self):
        super().__init__()
        self.register_buffer("loc", paddle.zeros(2))
        self.register_buffer("cov", paddle.eye(2))
        self.register_buffer("mask", paddle.tensor([[0, 1], [1, 0]] * 3, dtype=paddle.float32))
        self.s = paddle.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = paddle.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """初始化模型权重。"""
        for m in self.modules():
            if isinstance(m, paddle.compat.nn.Linear):
                paddle.nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """应用从数据空间到潜空间的映射，并计算雅可比矩阵行列式的对数。"""
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * paddle.exp(-s) + z_
            log_det_jacob -= s.sum(axis=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """计算给定样本在数据空间中的对数概率。"""
        if x.dtype == paddle.float32 and self.s[0][0].weight.dtype != paddle.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
