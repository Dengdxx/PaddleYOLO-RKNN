# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief MuSGD 混合优化器：YOLO26 三大架构特性之一。
@details
实现了 Muon+SGD 混合更新策略：
- `zeropower_via_newtonschulz5`：基于 Newton-Schulz 迭代的矩阵正交化（梯度方向优化）
- `muon_update`：带 Nesterov 动量的 Muon 更新步，适用于 2D/4D 卷积参数
- `MuSGD`：可逐参数组切换 Muon 或纯 SGD 策略的 Paddle 优化器

Muon 优化器源自 Moonshot AI Kimi K2 的 LLM 训练经验，将大语言模型训练中的
优化方法移植到计算机视觉领域，实现更稳定的收敛与更快的训练速度。
对于迭代次数 > 10000 的任务，使用 `optimizer='auto'` 会自动选择 MuSGD。
"""

import paddle
from collections import defaultdict


def zeropower_via_newtonschulz5(G: paddle.Tensor, eps: float = 1e-07) -> paddle.Tensor:
    """使用 Newton-Schulz 迭代计算矩阵 G 的零次幂 / 正交化近似。

    该函数使用五次 Newton-Schulz 迭代来计算输入矩阵 G 的近似正交化。迭代系数经过优化，
    用于最大化零点处的收敛斜率，得到类似 SVD 中 UV^T 的结果（其中 USV^T = G）。
    它放宽了严格收敛保证，但经验上非常适合优化用途。

    参数:
        G (paddle.Tensor): 待正交化的输入 2D 张量/矩阵。
        eps (float, optional): 加到 norm 上的小 epsilon，用于数值稳定，默认 1e-7。

    返回:
        (paddle.Tensor): 与输入 G 形状相同的正交化矩阵。

    示例:
        >>> G = paddle.randn(128, 64)
        >>> G_ortho = zeropower_via_newtonschulz5(G)
        >>> print(G_ortho.shape)
        Shape([128, 64])

    说明:
        - 使用 bfloat16 精度计算。
        - 使用固定系数精确执行 5 次 Newton-Schulz 迭代。
        - 当行数大于列数时，为提升效率会自动转置。
        - 输出近似 US'V^T，其中 S' 的对角元素约服从 Uniform(0.5, 1.5)。
        - 不会产生精确 UV^T，但在神经网络优化中经验效果良好。
    """
    assert len(G.shape) == 2
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for a, b, c in [
        (3.4445, -4.775, 2.0315),
        (3.4445, -4.775, 2.0315),
        (3.4445, -4.775, 2.0315),
        (3.4445, -4.775, 2.0315),
        (3.4445, -4.775, 2.0315),
    ]:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(
    grad: paddle.Tensor,
    momentum: paddle.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> paddle.Tensor:
    """计算带动量和正交化的 Muon 优化器更新。

    该函数先对梯度应用动量，可选使用 Nesterov 加速，然后通过 Newton-Schulz 迭代对更新量做正交化。
    对卷积滤波器（4D 张量），会在正交化前 reshape，并根据参数维度缩放最终更新量。

    参数:
        grad (paddle.Tensor): 待更新的梯度张量，可以是 2D 或 4D（卷积滤波器）。
        momentum (paddle.Tensor): 动量缓冲张量，会通过 lerp 原地修改。
        beta (float, optional): 指数移动平均的动量系数，默认 0.95。
        nesterov (bool, optional): 是否使用 Nesterov 动量加速，默认 True。

    返回:
        (paddle.Tensor): 与输入 grad 形状相同的正交化更新张量。对于 4D 输入，返回 reshape 回原始维度的结果。

    示例:
        >>> grad = paddle.randn(64, 128)
        >>> momentum = paddle.zeros_like(grad)
        >>> update = muon_update(grad, momentum, beta=0.95, nesterov=True)
        >>> print(update.shape)
        Shape([64, 128])

    说明:
        - 动量缓冲会原地更新：momentum = beta * momentum + (1-beta) * grad。
        - 使用 Nesterov 时：update = beta * momentum + (1-beta) * grad。
        - 不使用 Nesterov 时：update = momentum。
        - 4D 张量（卷积滤波器）会 reshape 为 2D 的 (out_channels, in_channels*height*width) 再正交化。
        - 最终更新量会乘以 sqrt(max(1, dim[-2] / dim[-1]))，以适配参数维度。
    """
    # 将 grad 转为 momentum dtype（float32），避免 AMP 下 lerp_ 混合 dtype 报错
    g = grad.cast(momentum.dtype) if grad.dtype != momentum.dtype else grad
    momentum.lerp_(y=g, weight=1 - beta)
    update = g.lerp(y=momentum, weight=beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, g.size(-2) / g.size(-1)) ** 0.5
    return update


class MuSGD(paddle.optimizer.Optimizer):
    """结合 Muon 和 SGD 更新的神经网络训练混合优化器。

    该优化器结合 Muon（基于动量、并通过 Newton-Schulz 迭代做正交化的优化器）和标准带动量 SGD。
    不同参数组可以选择使用混合 Muon+SGD 策略或纯 SGD 策略。

    参数:
        params (Iterable): 待优化参数，或定义参数组的字典。
        muon (float, optional): 混合模式下 Muon 更新的权重因子，默认 0.5。
        sgd (float, optional): 混合模式下 SGD 更新的权重因子，默认 0.5。

    属性:
        muon (float): 作用于 Muon 学习率的缩放因子。
        sgd (float): 混合模式下作用于 SGD 学习率的缩放因子。

    示例:
        >>> param_groups = [
        ...     {
        ...         "params": model.conv_params,
        ...         "lr": 0.02,
        ...         "use_muon": True,
        ...         "momentum": 0.95,
        ...         "nesterov": True,
        ...         "weight_decay": 0.01,
        ...     },
        ...     {
        ...         "params": model.other_params,
        ...         "lr": 0.01,
        ...         "use_muon": False,
        ...         "momentum": 0.9,
        ...         "nesterov": False,
        ...         "weight_decay": 0,
        ...     },
        ... ]
        >>> optimizer = MuSGD(param_groups, muon=0.5, sgd=0.5)
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()

    说明:
        - 设置 'use_muon': True 的参数组会同时接收 Muon 和 SGD 更新。
        - 设置 'use_muon': False 的参数组只接收 SGD 更新。
        - Muon 更新使用正交化，对 2D 及以上参数张量效果最好。
    """

    def __init__(
        self,
        parameters,
        learning_rate: float = 0.001,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ):
        """初始化具备混合 Muon 和 SGD 能力的 MuSGD 优化器。

        参数:
            parameters: 待优化参数，或定义参数组的字典。
            learning_rate (float): 学习率。
            momentum (float): SGD 动量因子。
            weight_decay (float): 权重衰减（L2 惩罚）。
            nesterov (bool): 是否使用 Nesterov 动量。
            use_muon (bool): 是否启用 Muon 更新。
            muon (float): Muon 分量缩放因子。
            sgd (float): SGD 分量缩放因子。
        """
        defaults = dict(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
        )
        super().__init__(learning_rate=learning_rate, parameters=parameters)
        # 将自定义默认值合并到 dict 形式的参数组中。
        # 使用 `get(k) is None` 而不是 setdefault，因为 Paddle 的 Optimizer.__init__
        # 会向 group 注入 weight_decay=None/grad_clip=None，导致 setdefault 无法覆盖。
        for group in self._param_groups:
            if isinstance(group, dict):
                for k, v in defaults.items():
                    if group.get(k) is None:
                        group[k] = v
        self.param_groups = self._param_groups
        self.state = defaultdict(dict)
        self.muon = muon
        self.sgd = sgd

    @paddle.no_grad()
    def step(self, closure=None):
        """执行单步优化。

        根据每个参数组中的 'use_muon' 标志，应用混合 Muon+SGD 更新或纯 SGD 更新。
        对启用 Muon 的参数组，参数会同时接收正交化 Muon 更新和标准 SGD 动量更新。

        参数:
            closure (Callable, optional): 重新评估模型并返回 loss 的闭包，默认 None。

        返回:
            (paddle.Tensor | None): 若提供 closure 则返回 loss，否则返回 None。

        说明:
            - 梯度为 None 的参数会被跳过。
            - Muon 更新使用 Newton-Schulz 正交化，对 2D 及以上张量效果最好。
            - 混合模式下权重衰减只应用于 SGD 分量。
        """
        loss = None
        if closure is not None:
            with paddle.enable_grad():
                loss = closure()
        # AMP scaler 检测到 inf/nan 梯度时跳过更新。
        # Paddle 内置优化器在 C++ kernel 中检查这里，我们需要显式处理。
        found_inf = self._get_auxiliary_var("found_inf") if hasattr(self, "_get_auxiliary_var") else None
        if found_inf is not None and found_inf.item():
            return loss
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    lr = group["learning_rate"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = paddle.zeros_like(p)
                        state["momentum_buffer_SGD"] = paddle.zeros_like(p)
                    update = muon_update(
                        grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=group["nesterov"],
                    )
                    # 将 bfloat16 更新转回参数 dtype（float32），用于 add_
                    update = (
                        update.reshape(p.shape).cast(p.dtype) if update.dtype != p.dtype else update.reshape(p.shape)
                    )
                    p.add_(update, alpha=-(lr * self.muon))
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state["momentum_buffer_SGD"].scale_(group["momentum"]).add_(grad)
                    sgd_update = (
                        grad.add(state["momentum_buffer_SGD"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer_SGD"]
                    )
                    p.add_(sgd_update, alpha=-(lr * self.sgd))
            else:
                for p in group["params"]:
                    lr = group["learning_rate"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = paddle.zeros_like(p)
                    state["momentum_buffer"].scale_(group["momentum"]).add_(grad)
                    update = (
                        grad.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    p.add_(update, alpha=-lr)
        return loss


class Muon(paddle.optimizer.Optimizer):
    """用于非分布式场景的 Muon 优化器。

    该优化器实现 Muon 算法，将基于动量的更新与 Newton-Schulz 迭代正交化结合起来，
    并对参数更新应用权重衰减和学习率缩放。

    参数:
        params (iterable): 待优化参数，或定义参数组的字典。
        learning_rate (float, optional): 学习率，默认 0.02。
        weight_decay (float, optional): 权重衰减（L2 惩罚）系数，默认 0。
        momentum (float, optional): 指数移动平均动量系数，默认 0.95。

    属性:
        param_groups (list): 参数组及其优化设置列表。
        state (dict): 每个参数对应的优化器状态字典。

    示例:
        >>> model = YourModel()
        >>> optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.01, momentum=0.95)
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()

    说明:
        - 面向非分布式训练环境设计。
        - 对所有参数使用带正交化的 Muon 更新。
        - 权重衰减会在参数更新前以乘法形式应用。
        - 梯度为 None 的参数会被赋予零梯度以便同步。
    """

    def __init__(self, parameters, learning_rate: float = 0.02, weight_decay: float = 0, momentum: float = 0.95):
        """初始化基于正交化更新的 Muon 优化器。

        参数:
            parameters: 待优化参数，或定义参数组的字典。
            learning_rate (float): 学习率。
            weight_decay (float): 以乘法形式应用的权重衰减因子。
            momentum (float): 梯度累积动量因子。
        """
        defaults = dict(learning_rate=learning_rate, weight_decay=weight_decay, momentum=momentum)
        super().__init__(learning_rate=learning_rate, parameters=parameters)
        # 将自定义默认值合并到 dict 形式的参数组中
        for group in self._param_groups:
            if isinstance(group, dict):
                for k, v in defaults.items():
                    group.setdefault(k, v)
        self.param_groups = self._param_groups
        self.state = defaultdict(dict)

    @paddle.no_grad()
    def step(self, closure=None):
        """执行单步优化。

        对所有参数应用结合动量和正交化的 Muon 更新。权重衰减会在参数更新前以乘法形式应用。

        参数:
            closure (Callable[[], paddle.Tensor] | None, optional): 重新评估模型并返回 loss 的闭包，默认 None。

        返回:
            (paddle.Tensor | None): 若提供 closure 则返回 loss，否则返回 None。

        示例:
            >>> optimizer = Muon(model.parameters())
            >>> loss = model(inputs)
            >>> loss.backward()
            >>> optimizer.step()

        说明:
            - 梯度为 None 的参数会被赋予零梯度以便同步。
            - 权重衰减形式为：p *= (1 - lr * weight_decay)。
            - Muon 更新使用 Newton-Schulz 正交化，对 2D 及以上张量效果最好。
        """
        loss = None
        if closure is not None:
            with paddle.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = paddle.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = paddle.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.scale_(1 - group["learning_rate"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["learning_rate"])
        return loss
