# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief Paddle 深度学习工具函数。
@details
包含训练基础设施工具：
- `select_device()`：根据配置选择 GPU/CPU 并打印信息
- `time_sync()`：设备同步计时
- `ModelEMA`：指数移动平均权重（EMA = ema * decay + model * (1-decay)）
- `de_parallel()`：去除 DDP 包装层获取原始模型
- `get_flops()`：计算模型 FLOPs

注：Paddle 版 EMA 用 `paddle.assign()` 原地更新，避免 *= 创建新张量失效的 BUG。
"""

from __future__ import annotations
import sys
import paddle


import functools
import gc
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ddyolo26.paddle_utils import *

from ddyolo26 import __version__
from ddyolo26.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    NUM_THREADS,
    PYTHON_VERSION,
    PADDLE_VERSION,
    WINDOWS,
    colorstr,
)
from ddyolo26.utils.checks import check_version
from ddyolo26.utils.cpu import CPUInfo
from ddyolo26.utils.patches import checkpoint_load

if WINDOWS and check_version(PADDLE_VERSION, "==2.4.0"):
    LOGGER.warning("Windows CPU 环境下 paddle==2.4.0 存在已知问题，建议升级到更新的 PaddlePaddle release。")


@contextmanager
def distributed_zero_first(local_rank: int):
    """确保 distributed training 中所有进程等待 local master（rank 0）先完成任务。"""
    initialized = paddle.distributed.is_available() and paddle.distributed.is_initialized()
    if initialized and local_rank not in {-1, 0}:
        paddle.distributed.barrier()
    yield
    if initialized and local_rank == 0:
        paddle.distributed.barrier()


def smart_inference_mode():
    """为 inference mode 应用 paddle.no_grad() decorator。"""

    def decorate(fn):
        """为 inference mode 应用 paddle.no_grad decorator。"""
        return paddle.no_grad()(fn)

    return decorate


def autocast(enabled: bool, device: str = "cuda"):
    """根据 AMP setting 获取 PaddlePaddle autocast context manager。

    参数:
        enabled (bool): 是否启用 automatic mixed precision。
        device (str, optional): autocast 使用的 device。

    返回:
        (paddle.amp.auto_cast): 合适的 autocast context manager。

    示例:
        >>> with autocast(enabled=True):
        ...     # mixed precision operations
        ...     pass

    """
    # O1: white-list ops (conv, matmul) 使用 FP16，其余使用 FP32；匹配 PyTorch autocast default
    return paddle.amp.auto_cast(enable=enabled, level="O1")


@functools.lru_cache
def get_cpu_info():
    """返回包含 system CPU information 的 string，例如 'Apple M2'。"""
    from ddyolo26.utils import PERSISTENT_CACHE

    if "cpu_info" not in PERSISTENT_CACHE:
        try:
            PERSISTENT_CACHE["cpu_info"] = CPUInfo.name()
        except Exception:
            pass
    return PERSISTENT_CACHE.get("cpu_info", "unknown")


@functools.lru_cache
def get_gpu_info(index):
    """返回包含 system GPU information 的 string，例如 'Tesla T4, 15102MiB'。"""
    properties = paddle.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def select_device(device="", newline=False, verbose=True):
    """根据提供的 arguments 选择合适的 PaddlePaddle device。

    参数:
        device (str | paddle.CPUPlace | paddle.CUDAPlace, optional): device string 或 Paddle device object。
            可选 'cpu'、'cuda'、'0'、'0,1,2,3'、'mps'，或用 '-1' auto-select。默认 auto-select
            第一个可用 GPU；若没有 GPU，则使用 CPU。
        newline (bool, optional): 若为 True，在 log string 末尾添加换行。
        verbose (bool, optional): 若为 True，记录 device information。

    返回:
        (paddle.base.libpaddle.Place): 选中的 device。

    示例:
        >>> select_device("cuda:0")
        device(type='cuda', index=0)

        >>> select_device("cpu")
        device(type='cpu')

    说明:
        设置 'CUDA_VISIBLE_DEVICES' environment variable，用于指定使用哪些 GPUs。
    """
    if isinstance(device, (paddle.CUDAPlace, paddle.CPUPlace, paddle.CUDAPinnedPlace)) or str(device).startswith(
        ("tpu", "intel", "vulkan")
    ):
        return device
    from ddyolo26.utils.checks import PADDLE_VERSION

    s = f"PaddleYOLO-RKNN {__version__} 🚀 Python-{PYTHON_VERSION} paddle-{PADDLE_VERSION} "
    device = str(device).lower()
    for remove in ("cuda:", "none", "(", ")", "[", "]", "'", " "):
        device = device.replace(remove, "")
    if "-1" in device:
        from ddyolo26.utils.autodevice import GPUInfo

        parts = device.split(",")
        selected = GPUInfo().select_idle_gpu(count=parts.count("-1"), min_memory_fraction=0.2)
        for i in range(len(parts)):
            if parts[i] == "-1":
                parts[i] = str(selected.pop(0)) if selected else ""
        device = ",".join(p for p in parts if p)
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device:
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        if not (paddle.cuda.is_available() and paddle.cuda.device_count() >= len(device.split(","))):
            LOGGER.info(s)
            install = (
                "若未检测到 CUDA devices，请检查 PaddlePaddle CUDA installation。\n"
                if paddle.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"""请求了无效 CUDA 'device={device}'。请使用 'device=cpu'，或在可用时传入有效 CUDA device(s)，例如 'device=0'，Multi-GPU 使用 'device=0,1,2,3'。

paddle.cuda.is_available(): {paddle.cuda.is_available()}
paddle.cuda.device_count(): {paddle.cuda.device_count()}
os.environ['CUDA_VISIBLE_DEVICES']: {visible}
{install}"""
            )
    if not cpu and not mps and paddle.cuda.is_available():
        devices = device.split(",") if device else "0"
        space = " " * len(s)
        for i, d in enumerate(devices):
            s += f"{'' if i == 0 else space}CUDA:{d} ({get_gpu_info(i)})\n"
        arg = "cuda:0"
    elif False:
        pass  # PaddlePaddle 不支持 MPS
    else:
        s += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"
    if arg in {"cpu", "mps"}:
        os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return paddle.device(arg)


def time_sync():
    """返回 accelerator-synchronized wall-clock time。"""
    if paddle.cuda.is_available():
        paddle.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, bn):
    """融合 Conv2d 与 BatchNorm2d layers，用于 inference optimization。

    参数:
        conv (nn.Conv2d): 要 fuse 的 convolutional layer。
        bn (nn.BatchNorm2d): 要 fuse 的 batch normalization layer。

    返回:
        (nn.Conv2d): fused convolutional layer，gradients 已禁用。

    示例:
        >>> conv = nn.Conv2d(3, 16, 3)
        >>> bn = nn.BatchNorm2d(16)
        >>> fused_conv = fuse_conv_and_bn(conv, bn)
    """
    w_conv = conv.weight.view(conv._out_channels, -1)
    w_bn = paddle.diag(bn.weight.div(paddle.sqrt(bn._epsilon + bn._variance)))
    conv.weight.data = paddle.mm(input=w_bn, mat2=w_conv).view(conv.weight.shape)
    if hasattr(conv.weight, "place"):
        b_conv = (
            paddle.zeros([conv._out_channels]).cast(conv.weight.dtype).to(conv.weight.place)
            if conv.bias is None
            else conv.bias
        )
    else:
        b_conv = paddle.zeros([conv._out_channels]).cast(conv.weight.dtype) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn._mean).div(paddle.sqrt(bn._variance + bn._epsilon))
    fused_bias = paddle.mm(input=w_bn, mat2=b_conv.reshape([-1, 1])).reshape([-1]) + b_bn
    if conv.bias is None:
        conv.bias = paddle.nn.Parameter(fused_bias)
    else:
        conv.bias.data = fused_bias
    for p in conv.parameters():
        p.stop_gradient = True
    return conv


def fuse_deconv_and_bn(deconv, bn):
    """融合 ConvTranspose2d 与 BatchNorm2d layers，用于 inference optimization。

    参数:
        deconv (nn.ConvTranspose2d): 要 fuse 的 transposed convolutional layer。
        bn (nn.BatchNorm2d): 要 fuse 的 batch normalization layer。

    返回:
        (nn.ConvTranspose2d): fused transposed convolutional layer，gradients 已禁用。

    示例:
        >>> deconv = nn.ConvTranspose2d(16, 3, 3)
        >>> bn = nn.BatchNorm2d(3)
        >>> fused_deconv = fuse_deconv_and_bn(deconv, bn)
    """
    w_deconv = deconv.weight.view(deconv._out_channels, -1)
    w_bn = paddle.diag(bn.weight.div(paddle.sqrt(bn._epsilon + bn._variance)))
    deconv.weight.data = paddle.mm(input=w_bn, mat2=w_deconv).view(deconv.weight.shape)
    if hasattr(deconv.weight, "place"):
        b_conv = (
            paddle.zeros([deconv._out_channels]).cast(deconv.weight.dtype).to(deconv.weight.place)
            if deconv.bias is None
            else deconv.bias
        )
    else:
        b_conv = paddle.zeros([deconv._out_channels]).cast(deconv.weight.dtype) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn._mean).div(paddle.sqrt(bn._variance + bn._epsilon))
    fused_bias = paddle.mm(input=w_bn, mat2=b_conv.reshape([-1, 1])).reshape([-1]) + b_bn
    if deconv.bias is None:
        deconv.bias = paddle.nn.Parameter(fused_bias)
    else:
        deconv.bias.data = fused_bias
    for p in deconv.parameters():
        p.stop_gradient = True
    return deconv


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """逐 layer 打印并返回详细 model information。

    参数:
        model (nn.Module): 要分析的 model。
        detailed (bool, optional): 是否打印 detailed layer information。
        verbose (bool, optional): 是否打印 model information。
        imgsz (int | list, optional): input image size。

    返回:
        (tuple): 包含以下内容的 tuple:
            - n_l (int): layers 数量。
            - n_p (int): parameters 数量。
            - n_g (int): gradients 数量。
            - flops (float): GFLOPs.
    """
    if not verbose:
        return
    n_p = get_num_params(model)
    n_g = get_num_gradients(model)
    layers = __import__("collections").OrderedDict((n, m) for n, m in model.named_modules() if len(m._modules) == 0)
    n_l = len(layers)
    if detailed:
        h = f"{'layer':>5}{'name':>40}{'type':>20}{'gradient':>10}{'parameters':>12}{'shape':>20}{'mu':>10}{'sigma':>10}"
        LOGGER.info(h)
        for i, (mn, m) in enumerate(layers.items()):
            mn = mn.replace("module_list.", "")
            mt = m.__class__.__name__
            if len(m._parameters):
                for pn, p in m.named_parameters():
                    LOGGER.info(
                        f"{i:>5g}{f'{mn}.{pn}':>40}{mt:>20}{p.requires_grad!r:>10}{p.size:>12g}{list(p.shape)!s:>20}{p.mean():>10.3g}{p.std():>10.3g}{str(p.dtype).replace('paddle.', ''):>15}"
                    )
            else:
                LOGGER.info(f"{i:>5g}{mn:>40}{mt:>20}{False!r:>10}{0:>12g}{[]!s:>20}{'-':>10}{'-':>10}{'-':>15}")
    flops = get_flops(model, imgsz)
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """返回 YOLO model 的 parameters 总数。"""
    return sum(x.size for x in model.parameters())


def get_num_gradients(model):
    """返回 YOLO model 中带 gradients 的 parameters 总数。"""
    return sum(x.size for x in model.parameters() if x.requires_grad)


def model_info_for_loggers(trainer):
    """返回包含 useful model information 的 model info dict。

    参数:
        trainer (ddyolo26.engine.trainer.BaseTrainer): 包含 model 与 validation data 的 trainer object。

    返回:
        (dict): 包含 model parameters、GFLOPs 与 inference speeds 的 dictionary。

    示例:
        YOLOv8n info for loggers
        >>> results = {
        ...    "model/parameters": 3151904,
        ...    "model/GFLOPs": 8.746,
        ...    "model/speed_ONNX(ms)": 41.244,
        ...    "model/speed_TensorRT(ms)": 3.211,
        ...    "model/speed_Paddle(ms)": 18.755,
        ...}
    """
    if trainer.args.profile:
        from ddyolo26.utils.benchmarks import ProfileModels

        results = ProfileModels([trainer.last], device=trainer.device).run()[0]
        results.pop("model/name")
    else:
        results = {
            "model/parameters": get_num_params(trainer.model),
            "model/GFLOPs": round(get_flops(trainer.model), 3),
        }
    results["model/speed_Paddle(ms)"] = round(trainer.validator.speed["inference"], 3)
    return results


def get_flops(model, imgsz=640):
    """以 GFLOPs 为单位计算 model 的 FLOPs（floating point operations）。

    尝试两种计算方法：先使用 stride-based tensor 以提高效率；必要时（如 RTDETR models）回退到 full image size。
    如果 thop library 不可用或计算失败，则返回 0.0。

    参数:
        model (nn.Module): 要计算 FLOPs 的 model。
        imgsz (int | list, optional): input image size。

    返回:
        (float): model 的 GFLOPs（billions of floating point operations）。
    """
    try:
        import thop
    except ImportError:
        thop = None
    if not thop:
        return 0.0
    try:
        model = unwrap_model(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]
        try:
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
            im = paddle.empty((1, p.shape[1], stride, stride), device=p.device)
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1000000000.0 * 2
            return flops * imgsz[0] / stride * imgsz[1] / stride
        except Exception:
            im = paddle.empty((1, p.shape[1], *imgsz), device=p.device)
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1000000000.0 * 2
    except Exception:
        return 0.0


def get_flops_with_paddle_profiler(model, imgsz=640):
    """使用 profiler 计算 model FLOPs；当前返回 0.0。"""
    return 0.0


def initialize_weights(model):
    """将 model weights、biases 与 module settings 初始化为 default values。

    对 Conv2D weights 应用 PyTorch-compatible Kaiming Uniform init，以匹配原始 training behavior
    （Paddle 默认使用 Kaiming Normal，其 std 高 2.45 倍）。这里不处理 biases；biases 由 model constructor
    中先于本函数运行的 bias_init() 处理。
    """
    import math

    for m in model.modules():
        t = type(m)
        if t is paddle.nn.Conv2d:
            # 匹配 PyTorch nn.Conv2d 默认值：kaiming_uniform_(a=sqrt(5))
            fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
            gain = math.sqrt(2.0 / (1 + 5))  # a^2 = 5 → gain = sqrt(2/6)
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            paddle.nn.initializer.Uniform(low=-bound, high=bound)(m.weight)
        elif t is paddle.nn.BatchNorm2D:
            m._epsilon = 0.001
            # PyTorch BN 的 momentum 表示新 batch stats 权重；Paddle _momentum 表示 running stats 保留率。
            # 因此 PyTorch momentum=0.03 迁移到 Paddle 时应写成 1 - 0.03。
            m._momentum = 1.0 - 0.03
        elif t in {paddle.nn.Hardswish, paddle.nn.LeakyReLU, paddle.nn.ReLU, paddle.nn.ReLU6, paddle.nn.SiLU}:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """scale 并 pad image tensor，可选保持 aspect ratio 并 pad 到 gs 倍数。

    参数:
        img (paddle.Tensor): input image tensor。
        ratio (float, optional): scaling ratio。
        same_shape (bool, optional): 是否保持相同 shape。
        gs (int, optional): padding 使用的 grid size。

    返回:
        (paddle.Tensor): scale 并 pad 后的 image tensor。
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = int(h * ratio), int(w * ratio)
    img = paddle.nn.functional.interpolate(img, size=s, mode="bilinear", align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return paddle.compat.nn.functional.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def copy_attr(a, b, include=(), exclude=()):
    """将 attributes 从 object 'b' 复制到 object 'a'，可 include/exclude 特定 attributes。

    参数:
        a (Any): attributes 复制到的 destination object。
        b (Any): attributes 复制来源 source object。
        include (tuple, optional): 要 include 的 attributes。为空时 include 所有 attributes。
        exclude (tuple, optional): 要 exclude 的 attributes。
    """
    for k, v in b.__dict__.items():
        if len(include) and k not in include or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def intersect_dicts(da, db, exclude=()):
    """返回交集 key 且 shape 匹配的 dictionary，排除 'exclude' keys，并使用 da values。

    参数:
        da (dict): 第一个 dictionary。
        db (dict): 第二个 dictionary。
        exclude (tuple, optional): 要 exclude 的 keys。

    返回:
        (dict): 交集 key 且 shape 匹配的 dictionary。
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def is_parallel(model):
    """如果 model 类型为 DP 或 DDP，则返回 True。

    参数:
        model (nn.Module): 要检查的 model。

    返回:
        (bool): model 为 DataParallel 或 DistributedDataParallel 时为 True。
    """
    return isinstance(model, (paddle.DataParallel, paddle.DataParallel))


def unwrap_model(m: paddle.nn.Module) -> paddle.nn.Module:
    """unwrap compiled 与 parallel models，获取 base model。

    参数:
        m (nn.Module): 可能被 paddle.jit.to_static (._orig_mod) 或 DataParallel/DistributedDataParallel (.module)
            等 parallel wrappers 包装的 model。

    返回:
        (nn.Module): 去除 compile 或 parallel wrappers 后的 base model。
    """
    while True:
        if hasattr(m, "_orig_mod") and isinstance(m._orig_mod, paddle.nn.Module):
            m = m._orig_mod
        elif hasattr(m, "module") and isinstance(m.module, paddle.nn.Module):
            m = m.module
        else:
            return m


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """返回从 y1 到 y2 的 sinusoidal ramp lambda function，见 https://arxiv.org/pdf/1812.01187.pdf。

    参数:
        y1 (float, optional): initial value。
        y2 (float, optional): final value。
        steps (int, optional): steps 数量。

    返回:
        (function): 用于计算 sinusoidal ramp 的 lambda function。
    """
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def init_seeds(seed=0, deterministic=False):
    """初始化 random number generator (RNG) seeds。

    参数:
        seed (int, optional): random seed。
        deterministic (bool, optional): 是否设置 deterministic algorithms。
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)
    paddle.cuda.manual_seed(seed)
    paddle.cuda.manual_seed_all(seed)
    if deterministic:
        paddle.set_flags({"FLAGS_cudnn_deterministic": True})
    else:
        unset_deterministic()
        # 启用 cuDNN exhaustive search，以查找 optimal conv algorithms（类似 cudnn.benchmark）。
        paddle.set_flags(
            {
                "FLAGS_cudnn_exhaustive_search": True,
                "FLAGS_conv_workspace_size_limit": 4096,  # MB，允许更大 workspace 以使用更快 algorithms。
            }
        )
        # 通过 env vars 在 Ampere+ GPUs（compute capability >= 8.0）上启用 TF32。
        # FLAGS_enable_tf32_* 必须在 paddle import 前设置，因此这里使用 os.environ。
        if paddle.device.cuda.device_count() > 0:
            try:
                cc = paddle.device.cuda.get_device_capability(0)
                if cc[0] >= 8:
                    os.environ.setdefault("FLAGS_enable_tf32_on_cudnn", "1")
                    os.environ.setdefault("FLAGS_enable_tf32_on_cublas", "1")
            except Exception:
                pass


def unset_deterministic():
    """取消 deterministic training 应用的所有 configurations。"""
    paddle.set_flags({"FLAGS_cudnn_deterministic": False})
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ.pop("PYTHONHASHSEED", None)
    # Eager tensor garbage collection，尽早释放 GPU memory。
    paddle.set_flags({"FLAGS_eager_delete_tensor_gb": 0.0})


class ModelEMA:
    """更新后的 Exponential Moving Average (EMA) implementation。

    对 model state_dict 中所有内容（parameters 与 buffers）保持 moving average。EMA 细节见参考。

    如需禁用 EMA，将 `enabled` attribute 设为 `False`。

    属性:
        ema (nn.Module): evaluation mode 下的 model copy。
        updates (int): EMA updates 数量。
        decay (function): 决定 EMA weight 的 decay function。
        enabled (bool): EMA 是否启用。

    参考:
        - https://github.com/rwightman/pytorch-image-models
        - https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """使用给定 arguments 为 'model' 初始化 EMA。

        参数:
            model (nn.Module): 要创建 EMA 的 model。
            decay (float, optional): maximum EMA decay rate。
            tau (int, optional): EMA decay time constant。
            updates (int, optional): initial updates 数量。
        """
        self.ema = deepcopy(unwrap_model(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.stop_gradient = True
        self.enabled = True

    def update(self, model):
        """更新 EMA parameters。

        参数:
            model (nn.Module): EMA 更新来源 model。
        """
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)
            msd = unwrap_model(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if paddle.is_floating_point(v):
                    new_v = v * d + (1 - d) * msd[k].detach()
                    paddle.assign(new_v, v)

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """将 attributes 从 model 复制到 EMA，可 include/exclude 特定 attributes。

        参数:
            model (nn.Module): attributes 复制来源 model。
            include (tuple, optional): 要 include 的 attributes。
            exclude (tuple, optional): 要 exclude 的 attributes。
        """
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def strip_optimizer(
    f: (str | Path) = "best.pdparams", s: str = "", updates: (dict[str, Any] | None) = None, half: bool = True
) -> dict[str, Any]:
    """从 'f' 中剥离 optimizer 以 finalize training，可选另存为 's'。

    参数:
        f (str | Path): 要剥离 optimizer 的 model file path。
        s (str, optional): stripped optimizer 后 model 的保存 file path。未提供时会覆盖 'f'。
        updates (dict, optional): 保存前 overlay 到 checkpoint 上的 updates dictionary。
        half (bool): 是否将 model weights cast 为 float16。默认 True（AMP training）。
            设置为 False 可保留 float32 weights（non-AMP / FP32 training）。

    返回:
        (dict): 合并后的 checkpoint dictionary。

    示例:
        >>> from pathlib import Path
        >>> from ddyolo26.utils.runtime import strip_optimizer
        >>> for f in Path("path/to/model/checkpoints").rglob("*.pt"):
        ...     strip_optimizer(f)
    """
    try:
        x = checkpoint_load(f, map_location=paddle.device("cpu"))
        assert isinstance(x, dict), "checkpoint 不是 Python dictionary"
        assert "model" in x, "checkpoint 缺少 'model'"
    except Exception as e:
        LOGGER.warning(f"跳过 {f}，这不是有效的 PaddleYOLO-RKNN model: {e}")
        return {}
    metadata = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "GNU AGPL-3.0",
        "docs": "https://www.gnu.org/licenses/agpl-3.0.html",
    }
    if x.get("ema"):
        x["model"] = x["ema"]
    if isinstance(x["model"], dict):
        if half:
            x["model"] = convert_optimizer_state_dict_to_fp16(x["model"])
    else:
        if hasattr(x["model"], "args"):
            x["model"].args = dict(x["model"].args)
        if hasattr(x["model"], "criterion"):
            x["model"].criterion = None
        if half:
            x["model"].half()
        for p in x["model"].parameters():
            p.stop_gradient = not False
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}
    for k in ("optimizer", "best_fitness", "ema", "updates", "scaler"):
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    combined = {**metadata, **x, **(updates or {})}
    paddle.save(obj=combined, path=str(s or f))
    mb = os.path.getsize(s or f) / 1000000.0
    LOGGER.info(f"已从 {f} 剥离 optimizer，{f'保存为 {s}，' if s else ''}{mb:.1f}MB")
    return combined


def convert_optimizer_state_dict_to_fp16(state_dict):
    """将给定 optimizer 的 state_dict 转为 FP16，重点处理 'state' key 中的 tensor conversions。

    参数:
        state_dict (dict): optimizer state dictionary。

    返回:
        (dict): 包含 FP16 tensors 的 converted optimizer state dictionary。
    """
    import paddle

    for k, v in state_dict.items():
        if isinstance(v, paddle.Tensor) and getattr(v, "dtype", None) == paddle.float32:
            state_dict[k] = v.cast(paddle.float16)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, paddle.Tensor) and getattr(sub_v, "dtype", None) == paddle.float32:
                    v[sub_k] = sub_v.cast(paddle.float16)
    return state_dict


@contextmanager
def cuda_memory_usage(device=None):
    """监控并管理 CUDA memory usage。

    该函数检查 CUDA 是否可用；若可用，则清空 CUDA cache 以释放 unused memory。随后 yield 一个包含 memory usage
    information 的 dictionary，调用方可更新它。最后用指定 device 上 CUDA reserved memory 的数量更新该 dictionary。

    参数:
        device (str, optional): 要查询 memory usage 的 CUDA device。

    生成:
        (dict): 一个带 'memory' key 的 dictionary，初始值为 0，后续会更新为 reserved memory。
    """
    cuda_info = dict(memory=0)
    if paddle.cuda.is_available():
        paddle.cuda.empty_cache()
        try:
            yield cuda_info
        finally:
            cuda_info["memory"] = paddle.cuda.memory_reserved(device)
    else:
        yield cuda_info


def profile_ops(input, ops, n=10, device=None, max_num_obj=0):
    """PaddleYOLO-RKNN speed、memory 与 FLOPs profiler。

    参数:
        input (paddle.Tensor | list): 要 profile 的 input tensor(s)。
        ops (nn.Module | list): 要 profile 的 model 或 operations 列表。
        n (int, optional): 用于 average 的 iterations 数量。
        device (str | str, optional): profile 使用的 device。
        max_num_obj (int, optional): simulation 使用的 maximum objects 数量。

    返回:
        (list): 每个 operation 的 profile results。

    示例:
        >>> from ddyolo26.utils.runtime import profile_ops
        >>> input = paddle.randn(16, 3, 640, 640)
        >>> m1 = lambda x: x * paddle.nn.functional.sigmoid(x)
        >>> m2 = nn.SiLU()
        >>> profile_ops(input, [m1, m2], n=100)  # profile 100 iterations
    """
    try:
        import thop
    except ImportError:
        thop = None
    results = []
    if not isinstance(device, paddle.device):
        device = select_device(device)
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}{'input':>24s}{'output':>24s}"
    )
    gc.collect()
    paddle.cuda.empty_cache()
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.stop_gradient = not True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m
            m = m.half() if hasattr(m, "half") and isinstance(x, paddle.Tensor) and x.dtype is paddle.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]
            try:
                flops = thop.profile(deepcopy(m), inputs=[x], verbose=False)[0] / 1000000000.0 * 2 if thop else 0
            except Exception:
                flops = 0
            try:
                mem = 0
                for _ in range(n):
                    with cuda_memory_usage(device) as cuda_info:
                        t[0] = time_sync()
                        y = m(x)
                        t[1] = time_sync()
                        try:
                            (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                            t[2] = time_sync()
                        except Exception:
                            t[2] = float("nan")
                    mem += cuda_info["memory"] / 1000000000.0
                    tf += (t[1] - t[0]) * 1000 / n
                    tb += (t[2] - t[1]) * 1000 / n
                    if max_num_obj:
                        with cuda_memory_usage(device) as cuda_info:
                            paddle.randn(
                                x.shape[0],
                                max_num_obj,
                                int(sum(x.shape[-1] / s * (x.shape[-2] / s) for s in m.stride.tolist())),
                                device=device,
                                dtype=paddle.float32,
                            )
                        mem += cuda_info["memory"] / 1000000000.0
                s_in, s_out = (tuple(x.shape) if isinstance(x, paddle.Tensor) else "list" for x in (x, y))
                p = sum(x.size for x in m.parameters()) if isinstance(m, paddle.nn.Module) else 0
                LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{s_in!s:>24s}{s_out!s:>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                LOGGER.info(e)
                results.append(None)
            finally:
                gc.collect()
                paddle.cuda.empty_cache()
    return results


class EarlyStopping:
    """当指定数量 epochs 内没有 improvement 时停止 training 的 early stopping 类。

    属性:
        best_fitness (float): 已观测到的 best fitness value。
        best_epoch (int): 观测到 best fitness 的 epoch。
        patience (int): fitness 停止改善后等待多少 epochs 再停止。
        possible_stop (bool): 表示下一 epoch 是否可能停止的 flag。
    """

    def __init__(self, patience=50):
        """初始化 early stopping object。

        参数:
            patience (int, optional): fitness 停止改善后等待多少 epochs 再停止。
        """
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float("inf")
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        """检查是否应停止 training。

        参数:
            epoch (int): training 的 current epoch。
            fitness (float): current epoch 的 fitness value。

        返回:
            (bool): training 应停止时为 True，否则为 False。
        """
        if fitness is None:
            return False
        if fitness > self.best_fitness or self.best_fitness == 0:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= self.patience - 1
        stop = delta >= self.patience
        if stop:
            prefix = colorstr("EarlyStopping: ")
            LOGGER.info(
                f"""{prefix}training 已 early stop，因为最近 {self.patience} 个 epochs 没有观察到 improvement。Best results 出现在 epoch {self.best_epoch}，best model 已保存为 best.pdparams。
如需更新 EarlyStopping(patience={self.patience})，请传入新的 patience 值，例如 `patience=300`；或使用 `patience=0` 禁用 EarlyStopping。"""
            )
        return stop


def attempt_compile(
    model: paddle.nn.Module,
    device: paddle.device,
    imgsz: int = 640,
    use_autocast: bool = False,
    warmup: bool = False,
    mode: (bool | str) = "default",
) -> paddle.nn.Module:
    """使用 paddle.jit.to_static compile model，并可选 warm up graph 以降低 first-iteration latency。

    该工具会尝试使用 inductor backend compile 给定 model。如果 compilation 不可用或失败，则原样返回 original model。
    可选 warmup 会在 dummy input 上执行一次 forward pass，以预热 compiled graph 并测量 compile/warmup time。

    参数:
        model (paddle.nn.Layer): 要 compile 的 model。
        device (str): 用于 warmup 与 autocast decision 的 inference device。
        imgsz (int, optional): 用于创建 warmup dummy tensor 的 square input size，shape 为 (1, 3, imgsz, imgsz)。
        use_autocast (bool, optional): 是否在 CUDA 或 MPS devices 上使用 autocast 运行 warmup。
        warmup (bool, optional): 是否执行一次 dummy forward pass 来 warm up compiled model。
        mode (bool | str, optional): compile mode。True → "default"，False → 不 compile，或传入字符串如
            "default", "reduce-overhead", "max-autotune-no-cudagraphs".

    返回:
        (paddle.nn.Layer): compilation 成功时返回 compiled model，否则返回未修改的 original model。

    示例:
        >>> device = str("cuda:0" if paddle.device.cuda.device_count() > 0 else "cpu")
        >>> # 尝试 compile 并用 640x640 input warm up model
        >>> model = attempt_compile(model, device=device, imgsz=640, use_autocast=True, warmup=True)

    说明:
        - 如果 paddle.jit.to_static 不可用，该函数会立即返回 input model。
        - Warmup 在 paddle.no_grad 下运行，并可能对 CUDA/MPS 使用 autocast，以对齐 compute precision。
        - CUDA devices 会在 warmup 后同步，以计入 asynchronous kernel execution。
    """
    if not hasattr(paddle, "compile") or not mode:
        return model
    if mode is True:
        mode = "default"
    prefix = colorstr("compile:")
    LOGGER.info(f"{prefix} 正在以 '{mode}' mode 启动 paddle.jit.to_static...")
    if mode == "max-autotune":
        LOGGER.warning(f"{prefix} 不推荐 mode='{mode}'，改用 mode='max-autotune-no-cudagraphs'")
        mode = "max-autotune-no-cudagraphs"
    t0 = time.perf_counter()
    try:
        model = paddle.jit.to_static(model)
    except Exception as e:
        LOGGER.warning(f"{prefix} paddle.jit.to_static 失败，将以未 compiled 状态继续: {e}")
        return model
    t_compile = time.perf_counter() - t0
    t_warm = 0.0
    if warmup:
        dummy = paddle.zeros(1, 3, imgsz, imgsz, device=device)
        if use_autocast and device.type == "cuda":
            dummy = dummy.half()
        t1 = time.perf_counter()
        with paddle.no_grad():
            if use_autocast and device.type in {"cuda", "mps"}:
                with paddle.autocast(device.type):
                    _ = model(dummy)
            else:
                _ = model(dummy)
        if device.type == "cuda":
            paddle.cuda.synchronize(device)
        t_warm = time.perf_counter() - t1
    total = t_compile + t_warm
    if warmup:
        LOGGER.info(f"{prefix} 已完成，用时 {total:.1f}s (compile {t_compile:.1f}s + warmup {t_warm:.1f}s)")
    else:
        LOGGER.info(f"{prefix} compile 已完成，用时 {t_compile:.1f}s（未 warmup）")
    return model
