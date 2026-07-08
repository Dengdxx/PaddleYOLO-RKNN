# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""
@file
@brief PaddlePaddle / PyTorch 兼容性补丁层：消除框架 API 差异。
@details
本文件是 PyTorch → PaddlePaddle 迁移的核心补丁，修复了：
1. **per-group lr 乘数语义**：Paddle 内置优化器 per-group lr 是相对乘数，
   本补丁将 Paddle 优化器封装为绝对 lr 语义，通过 `set_lr()` 统一调度
2. **Warmup 无效**：Paddle 运行时修改 per-group lr 不生效，
   通过 `optimizer.set_lr()` 单一全局 lr 实现 warmup
3. **EMA *= 创建新张量**：Paddle 中 *=/+= 操作创建新 Tensor，
   EMA 更新改用 `paddle.assign(new_val, output=p)` 原地写入
4. **BN 不支持 fp16 参数**：验证前强制 `model.float()` 转回 fp32
5. **`paddle.cat` 不自动提升 dtype**：手动 `.cast(bboxes.dtype)`
6. **DataLoader SHM 崩溃**：设 `use_shared_memory=False`
7. **auto base_lr 未传递**：在 `build_optimizer` 中存储 `_optimizer_base_lr`

大多数补丁以透明猴子补丁形式注入，用户代码无需感知。
"""

from __future__ import annotations

import warnings
import paddle

# 抑制 split/chunk 相关的 Paddle "Non compatible API" 兼容提示。
# Paddle 的提示以 \n 开头；filterwarnings 使用从字符串起点匹配的 re.match。
warnings.filterwarnings("ignore", message=r"\nNon compatible API")

# Paddle < 3.3 的 paddle.cuda 兼容层（3.2.x 仅有 paddle.device.cuda）
if not hasattr(paddle, "cuda"):
    paddle.cuda = paddle.device.cuda
    paddle.cuda.is_available = paddle.device.is_compiled_with_cuda

# Paddle 兼容别名（Paddle 使用 Layer/LayerList，而不是 Module/ModuleList）
paddle.nn.Module = paddle.nn.Layer
paddle.nn.ModuleList = paddle.nn.LayerList
paddle.nn.ModuleDict = paddle.nn.LayerDict
paddle.nn.SiLU = paddle.nn.Silu
paddle.nn.MultiheadAttention = paddle.nn.MultiHeadAttention

# Layer 方法别名（PyTorch Module → Paddle Layer）
if not hasattr(paddle.nn.Layer, "modules"):
    paddle.nn.Layer.modules = lambda self: self.sublayers(include_self=True)
if not hasattr(paddle.nn.Layer, "named_modules"):

    def _named_modules(self, memo=None, prefix="", remove_duplicate=True):
        for name, layer in self.named_sublayers(prefix=prefix, include_self=True):
            yield name, layer

    paddle.nn.Layer.named_modules = _named_modules

# paddle.compat.nn 兼容层：PaConvert 会生成 paddle.compat.nn.*，但 Paddle 3.x 没有该模块
import types

if not hasattr(paddle.compat, "nn"):
    _compat_nn = types.ModuleType("paddle.compat.nn")
    _compat_nn.Linear = paddle.nn.Linear
    _compat_nn.MultiheadAttention = paddle.nn.MultiHeadAttention
    _compat_nn_functional = types.ModuleType("paddle.compat.nn.functional")
    _compat_nn_functional.pad = paddle.nn.functional.pad
    _compat_nn_functional.softmax = paddle.nn.functional.softmax
    _compat_nn.functional = _compat_nn_functional
    paddle.compat.nn = _compat_nn


############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################


def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret


setattr(paddle.Tensor, "_max", _Tensor_max)


def device2int(device):
    if isinstance(device, str):
        device = device.replace("cuda", "gpu")
        device = device.replace("gpu:", "")
    return int(device)


def _Tensor_split(self, split_size, dim=None, axis=None):
    a = axis if axis is not None else (dim if dim is not None else 0)
    if isinstance(split_size, int):
        return paddle.split(self, self.shape[a] // split_size, a)
    else:
        return paddle.split(self, split_size, a)


setattr(paddle.Tensor, "split", _Tensor_split)


class PaddleFlag:
    cudnn_enabled = True
    cudnn_benchmark = False
    matmul_allow_tf32 = False
    cudnn_allow_tf32 = True
    cudnn_deterministic = False


def _Tensor_min(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret


setattr(paddle.Tensor, "_min", _Tensor_min)

import os


def _set_num_threads(int):
    os.environ["CPU_NUM"] = str(int)


############################## 相关utils函数，如上 ##############################
