# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 分布式训练工具：DDP 启动、端口查找、进程组管理。
@details
封装 Paddle DDP（DistributedDataParallel）的初始化逻辑：
- `find_free_port()`：动态分配可用端口，避免多任务冲突
- `generate_ddp_command()`：为子进程生成 `paddle.distributed.launch` 命令
- `ddp_cleanup()`：训练结束后清理进程组和临时文件
"""

from __future__ import annotations
import paddle
import os
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING

from . import USER_CONFIG_DIR

if TYPE_CHECKING:
    from ddyolo26.engine.trainer import BaseTrainer


def find_free_network_port() -> int:
    """在 localhost 上查找空闲端口。

    单机 training 中，如果不需要连接真实 main node，但必须设置 `MASTER_PORT` 环境变量，此函数会很有用。

    返回:
        (int): 可用网络端口号。
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def generate_ddp_file(trainer: BaseTrainer) -> str:
    """为 multi-GPU training 生成 DDP（Distributed Data Parallel）临时文件。

    该函数创建用于多 GPU distributed training 的临时 Python 文件，其中包含在分布式环境中初始化
    trainer 所需的配置。

    参数:
        trainer (ddyolo26.engine.trainer.BaseTrainer): 包含 training 配置和参数的 trainer，
            必须有 args 属性且为类实例。

    返回:
        (str): 生成的 DDP 临时文件路径。

    说明:
        生成文件会保存到 USER_CONFIG_DIR/DDP 目录，内容包括：
        - Trainer class import
        - 来自 trainer 参数的配置覆盖项
        - model path 配置
        - training 初始化代码
    """
    (
        module,
        name,
    ) = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)
    content = f"""
# PaddleYOLO-RKNN multi-GPU training 临时文件（使用后应自动删除）
from pathlib import Path, PosixPath  # 用于处理以 Path 而非 str 存储的 model 参数
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    from {module} import {name}
    from ddyolo26.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_ddp_command(trainer: BaseTrainer) -> tuple[list[str], str]:
    """生成 distributed training 命令。

    参数:
        trainer (ddyolo26.engine.trainer.BaseTrainer): 包含 distributed training 配置的 trainer。

    返回:
        cmd (list[str]): 执行 distributed training 的命令。
        file (str): 为 DDP training 创建的临时文件路径。
    """
    import __main__

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)
    file = generate_ddp_file(trainer)
    dist_cmd = "paddle.distributed.launch"
    port = find_free_network_port()
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nproc_per_node",
        f"{trainer.world_size}",
        "--master_port",
        f"{port}",
        file,
    ]
    return cmd, file


def ddp_cleanup(trainer: BaseTrainer, file: str) -> None:
    """删除 distributed data parallel（DDP）training 创建的临时文件。

    该函数检查给定文件名中是否包含 trainer ID，以判断它是否为 DDP training 临时文件；若是则删除。

    参数:
        trainer (ddyolo26.engine.trainer.BaseTrainer): 用于 distributed training 的 trainer。
        file (str): 可能需要删除的文件路径。

    示例:
        >>> trainer = YOLOTrainer()
        >>> file = "ddp_temp_123456789.py"
        >>> ddp_cleanup(trainer, file)
    """
    if f"{id(trainer)}.py" in file:
        os.remove(file)
