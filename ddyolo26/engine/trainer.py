# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 通用训练基类 BaseTrainer，管理 YOLO26 的完整训练生命周期。
@details
封装了 PaddlePaddle 版训练主循环，主要职责：
- 数据集加载、DataLoader 构建、AMP 混合精度初始化
- 优化器（含 MuSGD）构建与 LR 调度（Warmup + CosLR/LinearLR）
- EMA（指数移动平均）权重维护
- 每 epoch 的 train/val 循环、checkpoint 保存与断点恢复
- 训练前/后回调（callbacks）与 DDP 多卡支持

PaddlePaddle 迁移关键修复（相比 PyTorch 原版）：
- per-group lr 改为 1.0 乘数 + `set_lr()` 全局控制（避免 lr 被平方）
- EMA 参数用 `paddle.assign()` 原地更新（*= 会创建新张量导致更新无效）
- 验证前强制 `model.float()` 避免 BN fp16 崩溃
"""

from __future__ import annotations
import sys
import paddle

import os


from ddyolo26.paddle_utils import *

"""
在指定数据集上训练模型。

用法:
    $ yolo mode=train model=weights/yolov8/yolov8n.pdparams data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np

from ddyolo26 import __version__
from ddyolo26.cfg import get_cfg, get_save_dir
from ddyolo26.data.utils import check_cls_dataset, check_det_dataset
from ddyolo26.nn.tasks import load_checkpoint
from ddyolo26.optim import MuSGD
from ddyolo26.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ddyolo26.utils.autobatch import check_train_batch_size
from ddyolo26.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ddyolo26.utils.dist import ddp_cleanup, generate_ddp_command
from ddyolo26.utils.files import get_latest_run
from ddyolo26.utils.plotting import plot_results
from ddyolo26.utils.runtime import (
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    distributed_zero_first,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    unset_deterministic,
    unwrap_model,
)


class BaseTrainer:
    """用于创建训练器的基类。

    该类提供 YOLO 模型训练的基础能力，负责训练循环、验证、checkpoint 保存以及各类训练工具，
    支持单 GPU 和多 GPU 分布式训练。

    属性:
        args (SimpleNamespace): 训练器配置。
        validator (BaseValidator): 验证器实例。
        model (nn.Module): 模型实例。
        callbacks (defaultdict): 回调字典。
        save_dir (Path): 结果保存目录。
        wdir (Path): 权重保存目录。
        last (Path): 最新 checkpoint 路径。
        best (Path): 最优 checkpoint 路径。
        save_period (int): 每隔 x 个 epoch 保存 checkpoint（小于 1 时禁用）。
        batch_size (int): 训练 batch size。
        epochs (int): 训练 epoch 数。
        start_epoch (int): 开始训练的 epoch。
        device (paddle.base.libpaddle.Place): 训练设备。
        amp (bool): 是否启用 AMP（自动混合精度）。
        scaler (paddle.amp.GradScaler): AMP 梯度缩放器。
        data (dict): 包含路径和元数据的数据集字典。
        ema (ModelEMA): 模型 EMA（指数移动平均）。
        resume (bool): 是否从 checkpoint 恢复训练。
        lf (callable): 学习率调度函数。
        scheduler (paddle.optimizer.lr.LRScheduler): 学习率调度器。
        best_fitness (float): 已达到的最佳 fitness。
        fitness (float): 当前 fitness。
        loss (paddle.Tensor): 当前损失值。
        tloss (paddle.Tensor): 损失项运行均值。
        loss_names (list): 损失名称列表。
        csv (Path): 结果 CSV 文件路径。
        metrics (dict): 指标字典。
        plots (dict): 绘图字典。

    方法:
        train: 执行训练流程。
        validate: 在验证集上运行验证。
        save_model: 保存模型训练 checkpoint。
        get_dataset: 获取训练和验证数据集。
        setup_model: 加载、创建或下载模型。
        build_optimizer: 为模型构建优化器。

    示例:
        初始化训练器并开始训练
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 BaseTrainer。

        参数:
            cfg (str | dict | SimpleNamespace, optional): 配置文件路径或配置对象。
            overrides (dict, optional): 配置覆盖项。
            _callbacks (dict, optional): 回调函数字典。
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device)
        self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name
        self.wdir = self.save_dir / "weights"
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)
            self.args.save_dir = str(self.save_dir)
            args_dict = vars(self.args).copy()
            if args_dict.get("augmentations") is not None:
                args_dict["augmentations"] = [repr(t) for t in args_dict["augmentations"]]
            YAML.save(self.save_dir / "args.yaml", args_dict)
        self.last, self.best = self.wdir / "last.pdparams", self.wdir / "best.pdparams"
        self.save_period = self.args.save_period
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if isinstance(self.args.device, str) and len(self.args.device):
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:
            world_size = 0
        elif paddle.cuda.is_available():
            world_size = 1
        else:
            world_size = 0
        self.ddp = world_size > 1 and "LOCAL_RANK" not in os.environ
        self.world_size = world_size
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_pretrain_routine_start")
        self.model = check_model_file_from_stem(self.args.model)
        with distributed_zero_first(LOCAL_RANK):
            self.data = self.get_dataset()
        self.ema = None
        self.lf = None
        self.scheduler = None
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.args.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0

    def add_callback(self, event: str, callback):
        """向事件的回调列表追加指定回调。"""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """用指定回调覆盖某事件的现有回调。"""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """运行指定事件关联的所有现有回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """执行训练流程：多 GPU 使用 DDP 子进程，单 GPU 直接训练。"""
        if self.ddp:
            if self.args.rect:
                LOGGER.warning("'rect=True' 与多 GPU 训练不兼容，已设置为 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                raise ValueError(
                    f"多 GPU 训练不支持 batch<1 的 AutoBatch，请指定 GPU 数 {self.world_size} 的有效 batch size 倍数，例如 batch={self.world_size * 8}。"
                )
            cmd, file = generate_ddp_command(self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train()

    def _setup_scheduler(self):
        """初始化训练学习率调度器。

        Paddle 内置优化器（AdamW、SGD 等）会在构造时把每组 learning_rate 固化为 global_lr 的乘数，
        运行时修改 ``group["learning_rate"]`` 不会生效。因此：

        * 内置优化器：每组 lr = 1.0（固定乘数），所有 LR 控制都通过 ``optimizer.set_lr()``
          或调度器调整 global_lr。
        * MuSGD（自定义）：step 时读取 ``group["learning_rate"]``，因此每组 lr 就是绝对 lr，
          运行时更新可以生效。
        """
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        self._uses_custom_step = isinstance(self.optimizer, MuSGD)
        # 保存每组绝对 base LR，用于 warmup 目标计算。
        # MuSGD：每组 lr 就是绝对值（部分组可能不同，如 lr*3）。
        # 内置优化器：每组 lr 是固定 1.0 乘数，因此使用实际 optimizer base lr。
        base_lr = getattr(self, "_optimizer_base_lr", self.args.lr0)
        for g in self.optimizer._param_groups:
            if isinstance(g, dict):
                if self._uses_custom_step:
                    g.setdefault("initial_lr", g.get("learning_rate", base_lr))
                else:
                    g.setdefault("initial_lr", base_lr)

        if self._uses_custom_step:
            # MuSGD：global lr = 1.0（不使用），每组 lr 为绝对值（运行时可写）。
            tmp_lr = paddle.optimizer.lr.LambdaDecay(lr_lambda=lambda _: 1.0, learning_rate=1.0)
            self.optimizer.set_lr_scheduler(tmp_lr)
            self.scheduler = tmp_lr
        else:
            # 内置优化器：每组 lr = 1.0（固定），global lr 通过 set_lr() 作为绝对值控制。
            # 不注册调度器，由 _ManualScheduler 直接驱动 set_lr()。
            self.scheduler = self._ManualScheduler(self.optimizer, base_lr, self.lf)

    class _ManualScheduler:
        """通过 ``optimizer.set_lr()`` 驱动的学习率调度器替代实现。"""

        def __init__(self, optimizer, base_lr, lf):
            self._optimizer = optimizer
            self.base_lr = base_lr
            self.lf = lf
            self.last_epoch = -1
            self.last_lr = base_lr

        def step(self):
            self.last_epoch += 1
            self.last_lr = self.base_lr * self.lf(self.last_epoch)
            self._optimizer.set_lr(self.last_lr)

        def get_lr(self):
            return self.last_lr

    def _setup_ddp(self):
        """初始化并设置训练用 DistributedDataParallel 参数。"""
        paddle.cuda.set_device(RANK)
        self.device = paddle.device("cuda", RANK)
        paddle.distributed.init_parallel_env()

    def _build_train_pipeline(self):
        """为当前 batch size 构建 dataloader、优化器和调度器。"""
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()

    def _fix_bn_after_to(self):
        """修复 Paddle 3.x 中 .to() 导致 BatchNorm2D running stats 不更新的 bug。

        Paddle 3.x 的 ``model.to(device)`` 会断开 BatchNorm2D 底层 C++ 算子对
        ``_mean`` / ``_variance`` buffer 的内部引用，使前向传播时 running stats
        不再累积更新。本方法通过 ``set_value()`` 原地重写这些 buffer 来重建引用。
        """
        count = 0
        for m in self.model.sublayers():
            if isinstance(m, paddle.nn.BatchNorm2D):
                m._mean.set_value(m._mean.numpy())
                m._variance.set_value(m._variance.numpy())
                count += 1
        if count:
            LOGGER.info(f".to() 后已修复 {count} 个 BatchNorm2D 层的 running stats")

    def _setup_train(self):
        """在训练循环前配置模型、优化器、dataloader 和训练工具。"""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        # Paddle 3.x bug: .to() 会断开 BatchNorm2D 内部 C++ 对 _mean/_variance 的引用，
        # 导致前向传播时 running stats 不再更新。这里重新创建这些 buffer 来修复。
        self._fix_bn_after_to()
        self.set_model_attributes()
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        # Paddle 会在 named_parameters() 中暴露 BN running stats（_mean、_variance），
        # 但它们不应参与训练；解冻检查时跳过。
        _bn_buffer_suffixes = ("._mean", "._variance")
        frozen_layers = []
        unfrozen_layers = []
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                frozen_layers.append(k)
                v.stop_gradient = not False
            elif k.endswith(_bn_buffer_suffixes):
                pass  # BN running statistics，保持 stop_gradient=True
            elif not v.requires_grad and paddle.is_floating_point(v):
                unfrozen_layers.append(k)
                v.stop_gradient = not True
        if frozen_layers:
            LOGGER.info(f"冻结 {len(frozen_layers)} 层（匹配模式：{freeze_layer_names}）")
        if unfrozen_layers:
            LOGGER.warning(
                f"将 {len(unfrozen_layers)} 个此前冻结的层设置为 'requires_grad=True'。"
                f"如需自定义冻结层，请查看 ddyolo26.engine.trainer。"
            )
        self.amp = paddle.tensor(self.args.amp).to(self.device)
        if self.amp and RANK in {-1, 0}:
            callbacks_backup = callbacks.default_callbacks.copy()
            self.amp = paddle.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup
        if RANK > -1 and self.world_size > 1:
            paddle.distributed.broadcast(tensor=self.amp.int(), src=0)
        self.amp = bool(self.amp)
        self.scaler = paddle.amp.GradScaler(enable=self.amp, incr_every_n_steps=2000, init_loss_scaling=65536.0)
        # Paddle 3.x GradScaler bug：_scale 形状为 [1]，但 float() 需要 0-D。
        if self.scaler._scale is not None:
            self.scaler._scale = self.scaler._scale.squeeze()
        if self.world_size > 1:
            self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = paddle.DataParallel(layers=self.model, find_unused_parameters=True)
        gs = max(int(self.model.stride._max() if hasattr(self.model, "stride") else 32), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs
        if self.batch_size < 1 and RANK == -1:
            self.args.batch = self.batch_size = self.auto_batch()
        self._build_train_pipeline()
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self):
        """执行完整训练循环，包括初始化、epoch 迭代、验证和最终评估。"""
        # 启用 Paddle 自动调优：kernel 算法选择（等价于 cuDNN benchmark）。
        # 注意：layout autotune（NCHW→NHWC）会破坏 make_anchors 中的 .view()，保持禁用。
        try:
            from paddle.incubate.autotune import set_config

            set_config({"kernel": {"enable": True, "tuning_range": [1, 10]}})
        except Exception:
            pass
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()
        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"""图像尺寸 train={self.args.imgsz}, val={self.args.imgsz}
使用 {self.train_loader.num_workers * (self.world_size or 1)} 个 dataloader worker
结果记录到 {colorstr("bold", self.save_dir)}
开始训练 """
            + (f"{self.args.time} 小时..." if self.args.time else f"{self.epochs} 个 epoch...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.clear_grad()
        self._oom_retries = 0
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
            # MuSGD：每组 LR 需要手动同步（step 时读取）。
            # 内置优化器：_ManualScheduler.step() 已经调用 set_lr()。
            if self._uses_custom_step:
                lr_factor = self.lf(epoch)
                for g in self.optimizer._param_groups:
                    if isinstance(g, dict):
                        g["learning_rate"] = g.get("initial_lr", self.args.lr0) * lr_factor
            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == self.epochs - self.args.close_mosaic:
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(
                        1,
                        int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()),
                    )
                    for x in self.optimizer._param_groups:
                        # bias lr 从 warmup_bias_lr 降到 lr0，其余 lr 从 0.0 升到 lr0。
                        target_lr = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x.get("initial_lr", self.args.lr0) * self.lf(epoch),
                            ],
                        )
                        if self._uses_custom_step:
                            x["learning_rate"] = target_lr
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                    if not self._uses_custom_step:
                        # 内置优化器：通过 optimizer.set_lr() 驱动 LR（使用 weight 组目标值）。
                        _base = getattr(self, "_optimizer_base_lr", self.args.lr0)
                        wg_lr = np.interp(ni, xi, [0.0, _base * self.lf(epoch)])
                        self.optimizer.set_lr(float(wg_lr))
                        self.scheduler.last_lr = float(wg_lr)
                    else:
                        self.scheduler.last_lr = float(self.optimizer._param_groups[0]["learning_rate"])
                try:
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        if self.args.compile:
                            preds = self.model(batch["img"])
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        else:
                            loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= self.world_size
                        self.tloss = (
                            self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                        )
                    self.scaler.scale(self.loss).backward()
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise e
                    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                        raise
                    self._oom_retries += 1
                    old_batch = self.batch_size
                    self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                    LOGGER.warning(
                        f"batch={old_batch} 时 CUDA 显存不足，已降到 batch={self.batch_size} 并重试（{self._oom_retries}/3）。"
                    )
                    self._clear_memory()
                    self._build_train_pipeline()
                    self.scheduler.last_epoch = self.start_epoch - 1
                    nb = len(self.train_loader)
                    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
                    last_opt_step = -1
                    self.optimizer.clear_grad()
                    break
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    if self.args.time:
                        self.stop = time.time() - self.train_time_start > self.args.time * 3600
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            paddle.distributed.broadcast_object_list(object_list=broadcast_list, src=0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else paddle.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break
            else:
                self._oom_retries = 0
            if self._oom_retries and not self.stop:
                continue
            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()
            self.lr = {
                f"lr/pg{ir}": round(
                    x.get("learning_rate", self.scheduler.last_lr)
                    if (isinstance(x, dict) and self._uses_custom_step)
                    else self.scheduler.last_lr,
                    8,
                )
                for ir, x in enumerate(self.optimizer._param_groups)
            }
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(
                    self.model,
                    include=["yaml", "nc", "args", "names", "stride", "class_weights"],
                )
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                close = getattr(self.train_loader, "close", None)
                if callable(close):
                    close()
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()
            if self._handle_nan_recovery(epoch):
                continue
            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(
                    metrics={
                        **self.label_loss_items(self.tloss),
                        **self.metrics,
                        **self.lr,
                    }
                )
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= time.time() - self.train_time_start > self.args.time * 3600
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                paddle.distributed.broadcast_object_list(object_list=broadcast_list, src=0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1
        seconds = time.time() - self.train_time_start
        LOGGER.info(
            f"""
{epoch - self.start_epoch + 1} 个 epochs 已完成，耗时 {seconds / 3600:.3f} hours。"""
        )
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._close_dataloaders()
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """根据模型和设备显存约束计算最佳 batch size。"""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )

    def _get_memory(self, fraction=False):
        """获取加速器显存占用（GB）或总显存占比。"""
        memory, total = 0, 0
        if self.device.type != "cpu":
            memory = paddle.cuda.memory_reserved()
            if fraction:
                total = paddle.cuda.get_device_properties(self.device).total_memory
        return (memory / total if total > 0 else 0) if fraction else memory / 2**30

    def _clear_memory(self, threshold: (float | None) = None):
        """通过垃圾回收和清空缓存释放加速器显存。"""
        if threshold:
            assert 0 <= threshold <= 1, "threshold 必须位于 0 和 1 之间。"
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "cpu":
            return
        else:
            paddle.cuda.empty_cache()

    def _close_dataloaders(self) -> None:
        """关闭 persistent loader 持有的 Paddle DataLoader worker 进程。"""
        for name in ("train_loader", "test_loader"):
            loader = getattr(self, name, None)
            close = getattr(loader, "close", None)
            if callable(close):
                close()

    def read_results_csv(self):
        """使用 polars 将 results.csv 读取为字典。"""
        import polars as pl

        try:
            return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)
        except Exception:
            return {}

    def _model_train(self):
        """将模型设置为训练模式。"""
        self.model.train()
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, paddle.nn.BatchNorm2D):
                m.eval()

    def save_model(self):
        """保存带附加元数据的模型训练 checkpoint。"""
        import io

        buffer = io.BytesIO()
        paddle.save(
            obj={
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": unwrap_model(self.model).state_dict(),
                "ema": unwrap_model(self.ema.ema).state_dict(),
                "yaml": self.model.yaml,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,
                },
                "license": "GNU AGPL-3.0",
                "docs": "https://www.gnu.org/licenses/agpl-3.0.html",
            },
            path=buffer,
        )
        serialized_ckpt = buffer.getvalue()
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            (self.wdir / f"epoch{self.epoch}.pdparams").write_bytes(serialized_ckpt)

    def get_dataset(self):
        """从数据字典中获取训练和验证数据集。

        返回:
            (dict): 包含训练/验证/测试数据集和类别名称的字典。
        """
        try:
            data_str = str(self.args.data)
            if data_str.endswith(".ndjson") or data_str.startswith("ul://") and "/datasets/" in data_str:
                import asyncio

                from ddyolo26.data.converter import convert_ndjson_to_yolo
                from ddyolo26.utils.checks import check_file

                self.args.data = str(asyncio.run(convert_ndjson_to_yolo(check_file(self.args.data))))
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif str(self.args.data).rsplit(".", 1)[-1] in {
                "yaml",
                "yml",
            } or self.args.task in {"detect", "segment", "pose", "obb"}:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]
        except Exception as e:
            raise RuntimeError(emojis(f"数据集 '{clean_url(self.args.data)}' 出错 ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("使用单类别覆盖类别名称。")
            data["names"] = {(0): "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """为任意任务加载、创建或下载模型。

        返回:
            (dict | None): 用于恢复训练的 checkpoint；未加载 checkpoint 时为 None。
        """
        if isinstance(self.model, paddle.nn.Module):
            return
        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith((".pdparams", "_paddle.pt")):
            weights, ckpt = load_checkpoint(self.model, use_ema=not self.resume)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = load_checkpoint(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)
        return ckpt

    def optimizer_step(self):
        """执行一次训练优化器 step，并进行梯度裁剪和 EMA 更新。"""
        self.scaler.unscale_(self.optimizer)
        paddle.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.clear_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """允许按任务类型自定义模型输入和真值的预处理。"""
        return batch

    def validate(self):
        """使用 self.validator 在验证集上运行验证。

        返回:
            (tuple): 包含以下内容的元组：
                - metrics (dict | None): 验证指标字典；跳过验证时为 None。
                - fitness (float | None): 验证 fitness 分数；跳过验证时为 None。
        """
        if self.ema and self.world_size > 1:
            for buffer in self.ema.ema.buffers():
                paddle.distributed.broadcast(tensor=buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """获取模型；基类默认对 cfg 文件加载抛出 NotImplementedError。"""
        raise NotImplementedError("该任务训练器不支持加载 cfg 文件")

    def get_validator(self):
        """抛出 NotImplementedError（必须由子类实现）。"""
        raise NotImplementedError("训练器未实现 get_validator 函数")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """抛出 NotImplementedError（子类必须返回 `paddle.io.DataLoader`）。"""
        raise NotImplementedError("训练器未实现 get_dataloader 函数")

    def build_dataset(self, img_path, mode="train", batch=None):
        """构建数据集。"""
        raise NotImplementedError("训练器未实现 build_dataset 函数")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """返回带标签的训练损失项字典；loss_items 为 None 时返回损失名称列表。

        说明:
            分类任务不需要该方法，但分割和检测任务需要。
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """训练前设置或更新模型参数。"""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """为 YOLO 模型训练构建目标张量。"""
        pass

    def progress_string(self):
        """返回描述训练进度的字符串。"""
        return ""

    def plot_training_samples(self, batch, ni):
        """YOLO 训练期间绘制训练样本。"""
        pass

    def plot_training_labels(self):
        """绘制 YOLO 模型训练标签。"""
        pass

    def save_metrics(self, metrics):
        """将训练指标保存到 CSV 文件。"""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """根据 CSV 文件绘制指标。"""
        plot_results(file=self.csv, on_plot=self.on_plot)

    def on_plot(self, name, data=None):
        """注册绘图结果（例如供回调消费）。"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """对 YOLO 模型执行最终评估和验证。"""
        model = self.best if self.best.exists() else None
        with distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last, half=self.amp) if self.last.exists() else {}
                if model:
                    strip_optimizer(
                        self.best,
                        updates={"train_results": ckpt.get("train_results")},
                        half=self.amp,
                    )
        if model:
            LOGGER.info(f"\n正在验证 {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """检查 resume checkpoint 是否存在，并相应更新参数。"""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data
                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                ):
                    if k in overrides:
                        setattr(self.args, k, overrides[k])
                if ckpt_args.get("augmentations") is not None:
                    LOGGER.warning(
                        f"""原训练使用了自定义 Albumentations 变换，但当前不会自动恢复。若恢复训练时需要保留这些增强，请重新传入 'augmentations' 参数。示例：
model.train(resume=True, augmentations={ckpt_args["augmentations"]})"""
                    )
            except Exception as e:
                raise FileNotFoundError(
                    "未找到 resume checkpoint。请传入可用于恢复训练的有效 checkpoint，例如 'yolo train resume model=path/to/last.pdparams'"
                ) from e
        self.resume = resume

    def _load_checkpoint_state(self, ckpt):
        """从 checkpoint 加载 optimizer、scaler、EMA 和 best_fitness。"""
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(self._normalize_optimizer_state_dict(ckpt["optimizer"]))
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(self._normalize_scaler_state_dict(ckpt["scaler"]))
        if self.ema and ckpt.get("ema"):
            self.ema = ModelEMA(self.model)
            ema_sd = ckpt["ema"]
            ema_sd = {k: v.astype("float32") if hasattr(v, "astype") else v for k, v in ema_sd.items()}
            self.ema.ema.load_state_dict(ema_sd)
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness", 0.0)

    @staticmethod
    def _normalize_scaler_state_dict(state_dict):
        """规范化 GradScaler state_dict，兼容旧 checkpoint 中的 0 维 scale。"""
        scale = state_dict.get("scale")
        if scale is None:
            return state_dict

        normalized_scale = scale
        if isinstance(scale, np.ndarray):
            if scale.ndim == 0:
                normalized_scale = scale.reshape([1]).astype(np.float32, copy=False)
        elif isinstance(scale, paddle.Tensor):
            if len(scale.shape) == 0:
                normalized_scale = scale.reshape([1])
        elif np.isscalar(scale):
            normalized_scale = np.asarray([scale], dtype=np.float32)

        if normalized_scale is scale:
            return state_dict
        return {**state_dict, "scale": normalized_scale}

    @staticmethod
    def _optimizer_state_suffixes():
        """返回 Paddle 常见 optimizer state 的参数后缀。"""
        return (
            "_moment1_0",
            "_moment2_0",
            "_beta1_pow_acc_0",
            "_beta2_pow_acc_0",
            "_velocity_0",
            "_momentum_0",
            "_mean_square_0",
            "_mean_grad_0",
            "_inf_norm_0",
        )

    @classmethod
    def _split_optimizer_state_key(cls, key):
        """拆分 optimizer state key，返回 (参数名, 状态后缀)。"""
        for suffix in cls._optimizer_state_suffixes():
            if key.endswith(suffix):
                return key[: -len(suffix)], suffix
        return None, None

    def _current_optimizer_param_names(self):
        """返回当前 optimizer 参数名顺序。"""
        names = []
        for group in self.optimizer._param_groups:
            if isinstance(group, dict):
                names.extend(param.name for param in group.get("params", []))
        return names

    @staticmethod
    def _normalize_optimizer_state_value(value):
        """恢复 optimizer state 的数值 dtype，兼容保存时的 FP16 压缩。"""
        if isinstance(value, paddle.Tensor) and value.dtype == paddle.float16:
            return value.astype(paddle.float32)
        if isinstance(value, np.ndarray) and value.dtype == np.float16:
            return value.astype(np.float32, copy=False)
        return value

    def _normalize_optimizer_state_dict(self, state_dict):
        """按参数顺序重映射 optimizer state，兼容 resume 时的 Paddle 自动命名漂移。"""
        current_names = self._current_optimizer_param_names()
        if not current_names:
            return state_dict

        checkpoint_names = []
        seen = set()
        for key in state_dict.keys():
            param_name, _ = self._split_optimizer_state_key(key)
            if param_name is not None and param_name not in seen:
                seen.add(param_name)
                checkpoint_names.append(param_name)

        if not checkpoint_names:
            return state_dict
        remap_needed = checkpoint_names != current_names
        if len(checkpoint_names) != len(current_names):
            LOGGER.warning(
                "Optimizer state 参数数量与当前参数组不一致："
                f"checkpoint={len(checkpoint_names)}, current={len(current_names)}；跳过重映射。"
            )
            remap_needed = False

        name_map = dict(zip(checkpoint_names, current_names))
        if remap_needed:
            LOGGER.warning(
                "检测到 resume 训练的 optimizer 参数名发生漂移，按参数顺序重映射 checkpoint state 以恢复训练。"
            )

        dtype_fix_needed = False
        remapped = {}
        for key, value in state_dict.items():
            param_name, suffix = self._split_optimizer_state_key(key)
            normalized_value = self._normalize_optimizer_state_value(value)
            if normalized_value is not value:
                dtype_fix_needed = True
            if param_name is None:
                remapped[key] = normalized_value
                continue
            mapped_name = name_map.get(param_name, param_name)
            mapped_key = f"{mapped_name}{suffix}" if remap_needed else key
            remapped[mapped_key] = normalized_value

        if dtype_fix_needed:
            LOGGER.warning("检测到 checkpoint 中的 optimizer state 使用 FP16 压缩保存，恢复训练前已自动回升为 FP32。")
        return remapped

    def _handle_nan_recovery(self, epoch):
        """检测 NaN/Inf 损失或 fitness 崩溃，并通过加载 last checkpoint 恢复。"""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        corrupted = RANK in {-1, 0} and loss_nan and (fitness_nan or fitness_collapse)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
        if RANK != -1:
            broadcast_list = [corrupted if RANK == 0 else None]
            paddle.distributed.broadcast_object_list(object_list=broadcast_list, src=0)
            corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if epoch == self.start_epoch or not self.last.exists():
            LOGGER.warning(f"检测到 {reason}，但无法从 last.pdparams 恢复...")
            return False
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"训练失败：NaN 已持续 {self.nan_recovery_attempts} 个 epoch")
        LOGGER.warning(f"检测到 {reason}（第 {self.nan_recovery_attempts}/3 次尝试），正在从 last.pdparams 恢复...")
        self._model_train()
        _, ckpt = load_checkpoint(self.last)
        model_state = ckpt.get("model") or ckpt["ema"]
        model_state = {k: v.astype("float32") if hasattr(v, "astype") else v for k, v in model_state.items()}
        if not all(paddle.isfinite(v).all() for v in model_state.values() if isinstance(v, paddle.Tensor)):
            raise RuntimeError(f"Checkpoint {self.last} 已损坏，包含 NaN/Inf 权重")
        unwrap_model(self.model).load_state_dict(model_state)
        self._load_checkpoint_state(ckpt)
        del ckpt, model_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def resume_training(self, ckpt):
        """从指定 checkpoint 恢复 YOLO 训练。"""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert start_epoch > 0, f"""{self.args.model} 已完成 {self.epochs} 个 epoch 训练，无需恢复。
如需重新训练，请不要使用 resume，例如 'yolo train model={self.args.model}'"""
        LOGGER.info(f"正在恢复训练 {self.args.model}：从 epoch {start_epoch + 1} 到总计 {self.epochs} 个 epoch")
        if self.epochs < start_epoch:
            LOGGER.info(f"{self.model} 已训练 {ckpt['epoch']} 个 epoch，将继续微调 {self.epochs} 个 epoch。")
            self.epochs += ckpt["epoch"]
        self._load_checkpoint_state(ckpt)
        self.start_epoch = start_epoch
        self._truncate_csv(start_epoch)
        if start_epoch > self.epochs - self.args.close_mosaic:
            self._close_dataloader_mosaic()

    def _truncate_csv(self, start_epoch):
        """截断 results CSV，只保留截至 resume epoch 的行。

        恢复训练时，CSV 可能包含 checkpoint epoch 之后的旧记录（例如来自之前崩溃的训练）。
        这里会删除这些行，避免恢复训练继续追加时产生重复 epoch 记录。
        """
        if not self.csv.exists():
            return
        import csv as csv_module

        max_epoch = start_epoch  # 保留 epoch <= start_epoch 的行（CSV 中为 1 基索引）
        try:
            with open(self.csv, "r", encoding="utf-8") as f:
                reader = csv_module.reader(f)
                header = next(reader, None)
                if header is None:
                    return
                # 仅保留 epoch <= max_epoch 的行，同一 epoch 保留最后一次出现。
                seen = {}
                for row in reader:
                    try:
                        epoch_val = float(row[0])
                        if epoch_val <= max_epoch:
                            seen[epoch_val] = row  # 后出现的记录生效
                    except (ValueError, IndexError):
                        continue
                rows = [seen[k] for k in sorted(seen)]
            with open(self.csv, "w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
                for row in rows:
                    f.write(",".join(row) + "\n")
            LOGGER.info(f"已将 {self.csv.name} 截断到 {len(rows)} 行（epoch <= {max_epoch}）")
        except Exception as e:
            LOGGER.warning(f"截断 {self.csv.name} 失败：{e}")

    def _close_dataloader_mosaic(self):
        """更新 dataloader，使其停止使用 mosaic 增强。"""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("正在关闭 dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1e-05,
        iterations=100000.0,
    ):
        """为指定模型构建优化器。

        参数:
            model (paddle.nn.Layer): 需要构建优化器的模型。
            name (str, optional): 优化器名称；为 'auto' 时根据迭代次数自动选择。
            lr (float, optional): 优化器学习率。
            momentum (float, optional): 优化器动量系数。
            decay (float, optional): 优化器权重衰减。
            iterations (float, optional): 迭代次数；name 为 'auto' 时用于决定优化器。

        返回:
            (paddle.optimizer.Optimizer): 构建好的优化器。
        """
        g = [{}, {}, {}, {}]
        bn = tuple(v for k, v in paddle.nn.__dict__.items() if "Norm" in k)
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 检测到 'optimizer=auto'，将忽略 'lr0={self.args.lr0}' 和 'momentum={self.args.momentum}'，并自动确定最佳 'optimizer'、'lr0' 与 'momentum'... "
            )
            nc = self.data.get("nc", 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0
        use_muon = name == "MuSGD"
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                # 跳过不可训练参数（Paddle BN 会通过 named_parameters 暴露 _mean/_variance）。
                if param.stop_gradient:
                    continue
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim >= 2 and use_muon:
                    g[3][fullname] = param
                elif "bias" in fullname:
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:
                    g[1][fullname] = param
                else:
                    g[0][fullname] = param
        if not use_muon:
            g = [list(x.values()) for x in g[:3]]
        optimizers = {
            "Adam",
            "Adamax",
            "AdamW",
            "NAdam",
            "RAdam",
            "RMSProp",
            "SGD",
            "MuSGD",
            "auto",
        }
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        # 内置优化器中，每组 lr 是 global_lr 的固定乘数。
        # global lr 通过 set_lr() 以绝对值控制，因此每组 lr 必须为 1.0。
        # MuSGD（自定义）中，每组 lr 就是绝对 lr（运行时可写）。
        pg_lr = lr if use_muon else 1.0
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(learning_rate=pg_lr, weight_decay=0.0)
            ctor_args = dict(learning_rate=1.0, beta1=momentum, beta2=0.999)
        elif name == "RMSProp":
            optim_args = dict(learning_rate=pg_lr)
            ctor_args = dict(learning_rate=1.0, momentum=momentum)
        elif name == "SGD" or name == "MuSGD":
            optim_args = dict(learning_rate=pg_lr)
            ctor_args = dict(learning_rate=1.0, momentum=momentum, use_nesterov=True)
        else:
            raise NotImplementedError(
                f"优化器 '{name}' 不在可用优化器列表 {optimizers} 中。请通过 PaddleYOLO-RKNN 项目渠道反馈不支持的优化器。"
            )
        num_params = [len(g[0]), len(g[1]), len(g[2])]
        num_muon_params = len(g[3]) if use_muon else 0
        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {
            "params": g[0],
            **optim_args,
            "weight_decay": decay,
            "param_group": "weight",
        }
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        muon, sgd = 0.2, 1.0
        if use_muon:
            g[3] = {
                "params": g[3],
                **optim_args,
                "weight_decay": decay,
                "use_muon": True,
                "param_group": "muon",
            }
            import re

            pattern = re.compile("(?=.*23)(?=.*cv3)|proto\\.semseg")
            g_ = []
            for x in g:
                p = x.pop("params")
                p1 = [v for k, v in p.items() if pattern.search(k)]
                p2 = [v for k, v in p.items() if not pattern.search(k)]
                if p1:
                    g_.append({"params": p1, **x, "learning_rate": lr * 3})
                if p2:
                    g_.append({"params": p2, **x})
            g = g_
        # Paddle：lr/momentum 必须作为构造参数传入，不能放在参数组字典中。
        # 带 momentum 的 SGD 使用 paddle.optimizer.Momentum。
        paddle_name_map = {"SGD": "Momentum", "RMSProp": "RMSProp"}
        paddle_cls_name = paddle_name_map.get(name, name)
        if name == "MuSGD":
            optimizer = MuSGD(parameters=g, learning_rate=1.0, momentum=momentum, nesterov=True, muon=muon, sgd=sgd)
        else:
            optimizer = getattr(paddle.optimizer, paddle_cls_name)(parameters=g, **ctor_args)
        if use_muon:
            group_summary = (
                f"{num_muon_params} 个 Muon weight(decay={decay})，"
                f"{num_params[0]} 个 SGD weight(decay={decay})，"
                f"{num_params[1]} 个 norm weight(decay=0.0)，"
                f"{num_params[2]} 个 bias(decay=0.0)"
            )
        else:
            group_summary = (
                f"{num_params[1]} 个 weight(decay=0.0)，"
                f"{num_params[0]} 个 weight(decay={decay})，"
                f"{num_params[2]} 个 bias(decay=0.0)"
            )
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum})，参数组：{group_summary}"
        )
        # 保存实际 base lr（auto 模式下可能不同于 args.lr0）。
        self._optimizer_base_lr = lr
        return optimizer
