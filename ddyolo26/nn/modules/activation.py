# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 自定义激活函数模块。
@details
包含 SiLU/GELU 等标准激活函数的 Paddle 实现封装，
以及专为轻量边缘推理设计的自定义激活函数。
runtime 中的 `fuse_deconv_and_bn` 在 Paddle 迁移时有兼容性处理。
"""

from __future__ import annotations

import paddle

"""激活函数模块。"""


class AGLU(paddle.nn.Module):
    """来自 AGLU 的统一激活函数模块。

    该类基于 AGLU（Adaptive Gated Linear Unit）方法，实现带可学习 lambda 和 kappa 参数的参数化激活函数。

    属性:
        act (nn.Softplus): 使用负 beta 的 Softplus 激活函数。
        lambd (nn.Parameter): 以均匀分布初始化的可学习 lambda 参数。
        kappa (nn.Parameter): 以均匀分布初始化的可学习 kappa 参数。

    方法:
        forward: 计算统一激活函数的前向传播。

    示例:
        >>> m = AGLU()
        >>> input = paddle.randn(2)
        >>> output = m(input)
        >>> print(output.shape)
        Shape([2])

    参考:
        https://github.com/kostas1515/AGLU
    """

    def __init__(self, device=None, dtype=None) -> None:
        """初始化带可学习参数的统一激活函数。"""
        super().__init__()
        self.act = paddle.nn.Softplus(beta=-1.0)
        self.lambd = paddle.nn.Parameter(paddle.nn.init.uniform_(paddle.empty(1, device=device, dtype=dtype)))
        self.kappa = paddle.nn.Parameter(paddle.nn.init.uniform_(paddle.empty(1, device=device, dtype=dtype)))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """应用 Adaptive Gated Linear Unit（AGLU）激活函数。

        该前向方法使用可学习的 lambda 和 kappa 参数实现 AGLU 激活函数，自适应组合线性与非线性分量。

        参数:
            x (paddle.Tensor): 待应用激活函数的输入张量。

        返回:
            (paddle.Tensor): 应用 AGLU 激活函数后的输出张量，形状与输入相同。
        """
        lam = paddle.clamp(self.lambd, min=0.0001)
        return paddle.exp(1 / lam * self.act(self.kappa * x - paddle.log(lam)))
