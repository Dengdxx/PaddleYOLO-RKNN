# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief DataLoader 构建：`build_dataloader()` 和 `build_yolo_dataset()` 工厂函数。
@details
根据模式（train/val）选择不同的数据增强策略并构建 Paddle DataLoader，
关键参数：
- `use_shared_memory=False`（Paddle Worker SHM 崩溃修复）
- 支持 rect 模式（矩形批次排序）减少 padding 比例
"""

import paddle

import math
import os
import random
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from PIL import Image

from ddyolo26.cfg import IterableSimpleNamespace
from ddyolo26.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from ddyolo26.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ddyolo26.data.utils import IMG_FORMATS, VID_FORMATS
from ddyolo26.utils import RANK, colorstr
from ddyolo26.utils.checks import check_file


class InfiniteDataLoader(paddle.io.DataLoader):
    """复用 workers 以支持 infinite iteration 的 DataLoader。

    该 dataloader 扩展 Paddle DataLoader，支持 workers 无限循环复用。对于需要多次遍历 dataset、
    且不希望反复重建 workers 的 training loops，可提升效率。

    属性:
        batch_sampler (_RepeatSampler): 无限 repeat 的 sampler。
        iterator (Iterator): parent DataLoader 返回的 iterator。

    方法:
        __len__: 返回 batch sampler 内部 sampler 的长度。
        __iter__: 从底层 iterator 生成 batches。
        __del__: 确保 workers 正确终止。
        reset: reset iterator，适用于 training 中修改 dataset settings 的场景。

    示例:
        为 training 创建 infinite DataLoader
        >>> dataset = YOLODataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # infinite iteration
        >>>     train_step(batch)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """使用与 DataLoader 相同的 arguments 初始化 InfiniteDataLoader。"""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = None

    def __len__(self) -> int:
        """返回 dataloader 中 batches 的数量。"""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """创建一个从底层 iterator 无限 yield 的 iterator。"""
        if self.iterator is None:
            self.iterator = super().__iter__()
        for _ in range(len(self)):
            yield next(self.iterator)

    def close(self, timeout: float = 5) -> None:
        """停止 Paddle worker processes，并允许下一轮 iteration 重新创建它们。"""
        iterators = []
        for name in ("iterator", "_iterator"):
            iterator = getattr(self, name, None)
            if iterator is not None and all(iterator is not x for x in iterators):
                iterators.append(iterator)
        for iterator in iterators:
            shutdown = getattr(iterator, "_try_shutdown_all", None)
            if shutdown is None:
                continue
            try:
                shutdown(timeout=timeout)
            except TypeError:
                shutdown()
            except Exception:
                pass
        self.iterator = None
        if hasattr(self, "_iterator"):
            self._iterator = None

    def __del__(self):
        """确保 DataLoader 删除时 workers 正确终止。"""
        try:
            self.close()
        except Exception:
            pass

    def reset(self):
        """reset iterator，使后续 iterations 可感知 close_mosaic() 等 dataset changes。"""
        # 必须清除 Paddle 内部 _iterator 引用，让 super().__iter__()
        # 创建新的 _DataLoaderIterMultiProcess，而不是在已 shutdown 的 persistent iterator 上调用 _reset()。
        self.close()
        self.iterator = super().__iter__()


class _RepeatSampler:
    """用于 infinite iteration 的无限 repeat sampler。

    该 sampler 包装另一个 sampler，并无限 yield 其内容，从而无需重建 sampler 即可无限遍历 dataset。

    属性:
        sampler (paddle.io.Sampler): 要 repeat 的 sampler。
    """

    def __init__(self, sampler: Any):
        """使用需要无限 repeat 的 sampler 初始化 _RepeatSampler。"""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """无限遍历 sampler，并 yield 其内容。"""
        while True:
            yield from iter(self.sampler)


class ContiguousDistributedSampler(paddle.io.Sampler):
    """将 dataset 中连续且 batch-aligned 的 chunks 分配给各 GPU 的 distributed sampler。

    不同于 PyTorch DistributedSampler 的 round-robin 分配方式（GPU 0 获取 indices [0,2,4,...]，
    GPU 1 获取 [1,3,5,...]），该 sampler 会把 dataset 的连续 batches 分配给每个 GPU（GPU 0 获取
    batches [0,1,2,...]，GPU 1 获取 [k,k+1,...]，以此类推）。这样可以保留 original dataset 中的排序或分组，
    当 samples 按相似性组织时尤其关键（例如使用 rect=True 时，images 按 size 排序以减少 padding 并提升 batching 效率）。

    该 sampler 会将余数 batches 分配给前几个 ranks，以处理 batch counts 不均的情况，确保所有 samples
    在全部 GPUs 上恰好覆盖一次。

    参数:
        dataset (Dataset): 要采样的 Dataset，必须实现 __len__。
        num_replicas (int, optional): distributed processes 数量，默认 world size。
        batch_size (int, optional): dataloader 使用的 batch size，默认 dataset.batch_size 或 1。
        rank (int, optional): 当前 process 的 rank，默认 current rank。
        shuffle (bool, optional): 是否在每个 rank 的 chunk 内 shuffle indices，默认 False。为 True 时，
            shuffling 是 deterministic 的，并由 set_epoch() 控制以保证 reproducibility。

    示例:
        >>> # 用于按 size 分组 images 的 validation
        >>> sampler = ContiguousDistributedSampler(val_dataset, batch_size=32, shuffle=False)
        >>> loader = DataLoader(val_dataset, batch_size=32, sampler=sampler)
        >>> # 用于带 shuffling 的 training
        >>> sampler = ContiguousDistributedSampler(train_dataset, batch_size=32, shuffle=True)
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in loader:
        ...         ...
    """

    def __init__(
        self,
        dataset: paddle.io.Dataset,
        num_replicas: (int | None) = None,
        batch_size: (int | None) = None,
        rank: (int | None) = None,
        shuffle: bool = False,
    ) -> None:
        """使用 dataset 与 distributed training 参数初始化 sampler。"""
        if num_replicas is None:
            num_replicas = paddle.distributed.get_world_size() if paddle.distributed.is_initialized() else 1
        if rank is None:
            rank = paddle.distributed.get_rank() if paddle.distributed.is_initialized() else 0
        if batch_size is None:
            batch_size = getattr(dataset, "batch_size", 1)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.total_size = len(dataset)
        self.batch_size = 1 if batch_size >= self.total_size else batch_size
        self.num_batches = math.ceil(self.total_size / self.batch_size)

    def _get_rank_indices(self) -> tuple[int, int]:
        """计算当前 rank 的 start/end sample indices。"""
        batches_per_rank_base = self.num_batches // self.num_replicas
        remainder = self.num_batches % self.num_replicas
        batches_for_this_rank = batches_per_rank_base + (1 if self.rank < remainder else 0)
        start_batch = self.rank * batches_per_rank_base + min(self.rank, remainder)
        end_batch = start_batch + batches_for_this_rank
        start_idx = start_batch * self.batch_size
        end_idx = min(end_batch * self.batch_size, self.total_size)
        return start_idx, end_idx

    def __iter__(self) -> Iterator:
        """为当前 rank 的 dataset 连续 chunk 生成 indices。"""
        start_idx, end_idx = self._get_rank_indices()
        indices = list(range(start_idx, end_idx))
        if self.shuffle:
            perm = np.random.RandomState(self.epoch % 2**32).permutation(len(indices))
            indices = [indices[i] for i in perm.tolist()]
        return iter(indices)

    def __len__(self) -> int:
        """返回当前 rank chunk 中的 samples 数量。"""
        start_idx, end_idx = self._get_rank_indices()
        return end_idx - start_idx

    def set_epoch(self, epoch: int) -> None:
        """设置该 sampler 的 epoch，确保不同 epochs 使用不同 shuffling patterns。

        参数:
            epoch (int): 用作 shuffling random seed 的 epoch number。
        """
        self.epoch = epoch


class DeterministicRandomSampler(paddle.io.Sampler):
    """带显式 local RNG state 的 sampler，用于可复现 single-GPU shuffling。

    Paddle 的 DataLoader shuffle path 使用 global RNG state，因此 DataLoader 创建前的 model initialization
    和其他 random draws 可能悄悄改变 batch order。该 sampler 将 sampling order 保持在 dataloader 本地，
    并在每次完整遍历 dataset 后 deterministic 地推进。
    """

    def __init__(self, dataset: paddle.io.Dataset, seed: int) -> None:
        self.dataset = dataset
        self.base_seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator:
        seed = (self.base_seed + self.epoch) % 2**32
        self.epoch += 1
        indices = np.random.RandomState(seed).permutation(len(self.dataset)).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def seed_worker(worker_id: int, base_seed: int = 0) -> None:
    """设置 dataloader worker seed，以保证 worker processes 间可复现。"""
    worker_seed = (int(base_seed) + int(worker_id)) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    paddle.seed(worker_seed)


def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
) -> paddle.io.Dataset:
    """基于 configuration parameters 构建并返回 YOLO dataset。"""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_grounding(
    cfg: IterableSimpleNamespace,
    img_path: str,
    json_file: str,
    batch: int,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    max_samples: int = 80,
) -> paddle.io.Dataset:
    """基于 configuration parameters 构建并返回 GroundingDataset。"""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        max_samples=max_samples,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> InfiniteDataLoader:
    """创建并返回用于 training 或 validation 的 InfiniteDataLoader。

    参数:
        dataset (Dataset): 加载 data 的 Dataset。
        batch (int): dataloader 的 batch size。
        workers (int): data loading 使用的 worker processes 数量。
        shuffle (bool, optional): 是否 shuffle dataset。
        rank (int, optional): distributed training 中的 process rank。-1 表示 single-GPU training。
        drop_last (bool, optional): 是否丢弃最后一个不完整 batch。
        pin_memory (bool, optional): dataloader 是否使用 pinned memory。

    返回:
        (InfiniteDataLoader): 可用于 training 或 validation 的 dataloader。

    示例:
        为 training 创建 dataloader
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    batch = min(batch, len(dataset))
    nd = paddle.cuda.device_count()
    nw = min(os.cpu_count() // max(nd, 1), workers)
    base_seed = 6148914691236517205 + RANK
    sampler = (
        DeterministicRandomSampler(dataset, seed=base_seed)
        if rank == -1 and shuffle
        else None
        if rank == -1
        else paddle.io.DistributedBatchSampler(dataset=dataset, shuffle=shuffle, batch_size=1)
        if shuffle
        else ContiguousDistributedSampler(dataset)
    )
    kwargs = {}
    if sampler is not None:
        if isinstance(sampler, paddle.io.BatchSampler) or "BatchSampler" in type(sampler).__name__:
            kwargs["batch_sampler"] = sampler
        else:
            kwargs["batch_sampler"] = paddle.io.BatchSampler(
                sampler=sampler, batch_size=batch, drop_last=drop_last and len(dataset) % batch != 0
            )
    else:
        kwargs["batch_size"] = batch
        kwargs["shuffle"] = shuffle
        kwargs["drop_last"] = drop_last and len(dataset) % batch != 0

    loader_kwargs = dict(
        dataset=dataset,
        num_workers=nw,
        use_shared_memory=False,  # Paddle 3.3 shared memory 可能导致 worker threads 崩溃
        persistent_workers=nw > 0,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=partial(seed_worker, base_seed=base_seed),
        **kwargs,
    )
    # Paddle DataLoader 在提供 prefetch_factor 时要求它是正整数。
    # 某些 Paddle versions 下 num_workers=0 且传入 None 会 raise TypeError。
    if nw > 0:
        loader_kwargs["prefetch_factor"] = 4  # increase over default 2

    return InfiniteDataLoader(**loader_kwargs)


def check_source(
    source: (str | int | Path | list | tuple | np.ndarray | Image.Image | paddle.Tensor),
) -> tuple[Any, bool, bool, bool, bool, bool]:
    """检查 input source 类型并返回对应 flag values。

    参数:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | paddle.Tensor): 要检查的 input source。

    返回:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | paddle.Tensor): processed source。
        webcam (bool): source 是否为 webcam。
        screenshot (bool): source 是否为 screenshot。
        from_img (bool): source 是否为 image 或 images 列表。
        in_memory (bool): source 是否为 in-memory object。
        tensor (bool): source 是否为 paddle.Tensor。

    示例:
        检查 file path source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source("image.jpg")

        检查 webcam source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source(0)
    """
    webcam, screenshot, from_img, in_memory, tensor = (
        False,
        False,
        False,
        False,
        False,
    )
    if isinstance(source, (str, int, Path)):
        source = str(source)
        source_lower = source.lower()
        is_url = source_lower.startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        is_file = (urlsplit(source_lower).path if is_url else source_lower).rpartition(".")[
            -1
        ] in IMG_FORMATS | VID_FORMATS
        webcam = source.isnumeric() or source.endswith(".streams") or is_url and not is_file
        screenshot = source_lower == "screen"
        if is_url and is_file:
            source = check_file(source)
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, paddle.Tensor):
        tensor = True
    else:
        raise TypeError("不支持的 image type。支持的 source types 请查看 PaddleYOLO-RKNN repository README。")
    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(
    source: (str | int | Path | list | tuple | np.ndarray | Image.Image | paddle.Tensor),
    batch: int = 1,
    vid_stride: int = 1,
    buffer: bool = False,
    channels: int = 3,
):
    """加载 object detection 的 inference source，并应用必要 transformations。

    参数:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | paddle.Tensor): inference 使用的 input source。
        batch (int, optional): dataloaders 使用的 batch size。
        vid_stride (int, optional): video sources 的 frame interval。
        buffer (bool, optional): 是否 buffer stream frames。
        channels (int, optional): model 的 input channels 数。

    返回:
        (Dataset): 指定 input source 对应的 dataset object，并附带 source_type attribute。

    示例:
        加载 image source 用于 inference
        >>> dataset = load_inference_source("image.jpg", batch=1)

        加载 video stream source
        >>> dataset = load_inference_source("rtsp://example.com/stream", vid_stride=2)
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer, channels=channels)
    elif screenshot:
        dataset = LoadScreenshots(source, channels=channels)
    elif from_img:
        dataset = LoadPilAndNumpy(source, channels=channels)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)
    setattr(dataset, "source_type", source_type)
    return dataset
