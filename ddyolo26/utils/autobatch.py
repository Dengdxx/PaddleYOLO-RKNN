# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 自动批大小估算：根据 GPU 显存动态推荐最优 batch_size。
@details
调用 `check_train_batch_size()` 在 GPU 上二分搜索最大可用 batch，
避免 OOM 同时最大化显存利用率。YOLO26 训练启动时如设 `batch=-1` 则触发此逻辑。
"""

from __future__ import annotations
import sys
import paddle

import os


from ddyolo26.paddle_utils import *

"""估算最佳 YOLO batch size 的函数，用于占用指定比例的可用 CUDA memory。"""

from copy import deepcopy

import numpy as np

from ddyolo26.utils import DEFAULT_CFG, LOGGER, colorstr
from ddyolo26.utils.runtime import autocast, profile_ops


def check_train_batch_size(
    model: paddle.nn.Module,
    imgsz: int = 640,
    amp: bool = True,
    batch: (int | float) = -1,
    max_num_obj: int = 1,
) -> int:
    """使用 autobatch() 计算 YOLO training 的最佳 batch size。

    参数:
        model (paddle.nn.Layer): 需要检查 batch size 的 YOLO model。
        imgsz (int, optional): training 使用的 image size。
        amp (bool, optional): 为 True 时启用 automatic mixed precision。
        batch (int | float, optional): 要使用的 GPU memory 比例；为 -1 时使用默认值。
        max_num_obj (int, optional): dataset 中单图最大 object 数。

    返回:
        (int): autobatch() 计算得到的最佳 batch size。

    说明:
        若 0.0 < batch < 1.0，则将其作为 GPU memory 使用比例；否则使用默认比例 0.6。
    """
    with autocast(enabled=amp):
        return autobatch(
            deepcopy(model).train(),
            imgsz,
            fraction=batch if 0.0 < batch < 1.0 else 0.6,
            max_num_obj=max_num_obj,
        )


def autobatch(
    model: paddle.nn.Module,
    imgsz: int = 640,
    fraction: float = 0.6,
    batch_size: int = DEFAULT_CFG.batch,
    max_num_obj: int = 1,
) -> int:
    """自动估算最佳 YOLO batch size，以占用指定比例的可用 CUDA memory。

    参数:
        model (paddle.nn.Layer): 需要计算 batch size 的 YOLO model。
        imgsz (int, optional): YOLO model 输入 image size。
        fraction (float, optional): 可用 CUDA memory 使用比例。
        batch_size (int, optional): 检测到错误时使用的默认 batch size。
        max_num_obj (int, optional): dataset 中单图最大 object 数。

    返回:
        (int): 最佳 batch size。
    """
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}正在为 imgsz={imgsz} 计算最佳 batch size，目标 CUDA memory 使用率 {fraction * 100}%。")
    device = next(model.parameters()).device
    if device.type in {"cpu", "mps"}:
        LOGGER.warning(f"{prefix}仅适用于 CUDA device，使用默认 batch-size {batch_size}")
        return batch_size
    if PaddleFlag.cudnn_benchmark:
        LOGGER.warning(f"{prefix}要求 cudnn.benchmark=False，使用默认 batch-size {batch_size}")
        return batch_size
    gb = 1 << 30
    d = f"CUDA:{os.getenv('CUDA_VISIBLE_DEVICES', '0').strip()[0]}"
    properties = paddle.cuda.get_device_properties(device)
    t = properties.total_memory / gb
    r = paddle.cuda.memory_reserved(device) / gb
    a = paddle.cuda.memory_allocated(device) / gb
    f = t - (r + a)
    LOGGER.info(f"{prefix}{d} ({properties.name}) 总量 {t:.2f}G, 已保留 {r:.2f}G, 已分配 {a:.2f}G, 空闲 {f:.2f}G")
    batch_sizes = [1, 2, 4, 8, 16] if t < 16 else [1, 2, 4, 8, 16, 32, 64]
    ch = model.yaml.get("channels", 3)
    try:
        img = [paddle.empty(b, ch, imgsz, imgsz) for b in batch_sizes]
        results = profile_ops(img, model, n=1, device=device, max_num_obj=max_num_obj)
        xy = [
            [x, y[2]]
            for i, (x, y) in enumerate(zip(batch_sizes, results))
            if y
            and isinstance(y[2], (int, float))
            and 0 < y[2] < t
            and (i == 0 or not results[i - 1] or y[2] > results[i - 1][2])
        ]
        fit_x, fit_y = zip(*xy) if xy else ([], [])
        p = np.polyfit(fit_x, fit_y, deg=1)
        b = int((round(f * fraction) - p[1]) / p[0])
        if None in results:
            i = results.index(None)
            if b >= batch_sizes[i]:
                b = batch_sizes[max(i - 1, 0)]
        if b < 1 or b > 1024:
            LOGGER.warning(f"{prefix}batch={b} 超出安全范围，使用默认 batch-size {batch_size}。")
            b = batch_size
        fraction = (np.polyval(p, b) + r + a) / t
        LOGGER.info(f"{prefix}为 {d} 使用 batch-size {b}，显存 {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
        return b
    except Exception as e:
        LOGGER.warning(f"{prefix}检测到错误: {e}，使用默认 batch-size {batch_size}。")
        return batch_size
    finally:
        paddle.cuda.empty_cache()
