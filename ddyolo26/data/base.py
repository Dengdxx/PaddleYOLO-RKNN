# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 数据集基类 BaseDataset，提供懒加载与缓存机制。
@details
实现 YOLO 系列数据集的通用基础：
- 标注文件扫描与格式校验
- 以 `.cache` 文件（pickle + numpy memmap）缓存标注解析结果
- 支持 `rect`（矩形训练）批次排序
- 多进程安全的 numpy SHM（共享内存）相关处理

PaddlePaddle 迁移注意：DataLoader use_shared_memory=False（避免 Tensor 内存崩溃）。
"""

from __future__ import annotations
import sys
import paddle


import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ddyolo26.paddle_utils import *

from ddyolo26.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, check_file_speeds
from ddyolo26.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ddyolo26.utils.patches import imread


class BaseDataset(paddle.io.Dataset):
    """用于加载并处理 image data 的 base dataset 类。

    该类为 object detection 任务中的 images 加载、缓存，以及 training/inference data 准备提供核心功能。

    属性:
        img_path (str | list[str]): 包含 images 的 folder path。
        imgsz (int): resize 使用的 target image size。
        augment (bool): 是否应用 data augmentation。
        single_cls (bool): 是否将所有 objects 视为 single class。
        prefix (str): log messages 中打印的 prefix。
        fraction (float): 使用的 dataset fraction。
        channels (int): images 的 channels 数（1 为 grayscale，3 为 color）。OpenCV 加载的 color images 为 BGR channel order。
        cv2_flag (int): 读取 images 使用的 OpenCV flag。
        im_files (list[str]): image file paths 列表。
        labels (list[dict]): label data dictionaries 列表。
        ni (int): dataset 中 images 数量。
        rect (bool): 是否使用 rectangular training。
        batch_size (int): batches 大小。
        stride (int): model 使用的 stride。
        pad (float): padding value。
        buffer (list): mosaic images 使用的 buffer。
        max_buffer_length (int): 最大 buffer size。
        ims (list): loaded images 列表。
        im_hw0 (list): original image dimensions (h, w) 列表。
        im_hw (list): resized image dimensions (h, w) 列表。
        npy_files (list[Path]): numpy file paths 列表。
        cache (str | None): cache setting（'ram'、'disk' 或 None 表示不 cache）。
        transforms (callable): image transformation function。
        batch_shapes (np.ndarray): rectangular training 的 batch shapes。
        batch (np.ndarray): 每张 image 的 batch index。

    方法:
        get_img_files: 从指定 path 读取 image files。
        update_labels: 更新 labels，仅保留指定 classes。
        load_image: 从 dataset 加载 image。
        cache_images: 将 images cache 到 memory 或 disk。
        cache_images_to_disk: 将 image 保存为 *.npy file，以便更快加载。
        check_cache_disk: 检查 image caching 需求与 available disk space。
        check_cache_ram: 检查 image caching 需求与 available memory。
        set_rectangle: 按 aspect ratio 排序 images，并设置 rectangular training 的 batch shapes。
        get_image_and_label: 从 dataset 获取并返回 label 信息。
        update_labels_info: 子类实现的自定义 label format 方法。
        build_transforms: 子类实现的 transformation pipeline 构建方法。
        get_labels: 子类实现的 labels 获取方法。
    """

    def __init__(
        self,
        img_path: (str | list[str]),
        imgsz: int = 640,
        cache: (bool | str) = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: (list[int] | None) = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        """使用给定 configuration 与 options 初始化 BaseDataset。

        参数:
            img_path (str | list[str]): 包含 images 的 folder path，或 image paths 列表。
            imgsz (int): resize 使用的 image size。
            cache (bool | str): training 期间将 images cache 到 RAM 或 disk。
            augment (bool): 若为 True，应用 data augmentation。
            hyp (dict[str, Any]): 应用 data augmentation 的 hyperparameters。
            prefix (str): log messages 中打印的 prefix。
            rect (bool): 若为 True，使用 rectangular training。
            batch_size (int): batches 大小。
            stride (int): model 使用的 stride。
            pad (float): padding value。
            single_cls (bool): 若为 True，使用 single class training。
            classes (list[int], optional): 包含的 classes 列表。
            fraction (float): 使用的 dataset fraction。
            channels (int): images 的 channels 数（1 为 grayscale，3 为 color）。OpenCV 加载的 color images 为 BGR channel order。
        """
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)
        self.ni = len(self.labels)
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()
        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        self.ims, self.im_hw0, self.im_hw = (
            [None] * self.ni,
            [None] * self.ni,
            [None] * self.ni,
        )
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' 可能产生 non-deterministic training results。若 disk space 允许，可考虑使用 cache='disk' 作为 deterministic 替代方案。"
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path: (str | list[str])) -> list[str]:
        """从指定 path 读取 image files。

        参数:
            img_path (str | list[str]): image directories 或 files 的 path 或 paths 列表。

        返回:
            (list[str]): image file paths 列表。

        异常:
            FileNotFoundError: 当未找到 images 或 path 不存在时抛出。
        """
        try:
            f = []
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.is_file():
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [(x.replace("./", parent) if x.startswith("./") else x) for x in t]
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} 不存在")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            assert im_files, f"{self.prefix}在 {img_path} 中未找到 images。{FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}从 {img_path} 加载 data 时出错\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        check_file_speeds(im_files, prefix=self.prefix)
        return im_files

    def update_labels(self, include_class: (list[int] | None)) -> None:
        """更新 labels，仅保留指定 classes。

        参数:
            include_class (list[int], optional): 要保留的 classes 列表。若为 None，则包含所有 classes。
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """从 dataset index 'i' 加载 image。

        参数:
            i (int): 待加载 image 的 index。
            rect_mode (bool): 是否使用 rectangular resizing。

        返回:
            im (np.ndarray): NumPy array 形式的 loaded image。
            hw_original (tuple[int, int]): (height, width) format 的 original image dimensions。
            hw_resized (tuple[int, int]): (height, width) format 的 resized image dimensions。

        异常:
            FileNotFoundError: 当 image file 未找到时抛出。
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}正在移除损坏的 *.npy image file {fn}，原因: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)
            else:
                im = imread(f, flags=self.cv2_flag)
            if im is None:
                raise FileNotFoundError(f"未找到 image {f}")
            h0, w0 = im.shape[:2]
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not h0 == w0 == self.imgsz:
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self) -> None:
        """将 images cache 到 memory 或 disk，以加快 training。"""
        b, gb = 0, 1 << 30
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].size * self.ims[i].element_size()
                pbar.desc = f"{self.prefix}正在 cache images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        """将 image 保存为 *.npy file，以便更快加载。"""
        f = self.npy_files[i]
        if not f.exists():
            np.save(
                f.as_posix(),
                imread(self.im_files[i], flags=self.cv2_flag),
                allow_pickle=False,
            )

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """检查是否有足够 disk space cache images。

        参数:
            safety_margin (float): disk space 计算使用的 safety margin factor。

        返回:
            (bool): 若 disk space 足够，则返回 True；否则返回 False。
        """
        import shutil

        b, gb = 0, 1 << 30
        n = min(self.ni, 30)
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.size * im.element_size()
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}跳过将 images cache 到 disk：directory 不可写")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}需要 {disk_required / gb:.1f}GB disk space（包含 {int(safety_margin * 100)}% safety margin），但仅有 {free / gb:.1f}/{total / gb:.1f}GB free，不会将 images cache 到 disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """检查是否有足够 RAM cache images。

        参数:
            safety_margin (float): RAM 计算使用的 safety margin factor。

        返回:
            (bool): 若 RAM 足够，则返回 True；否则返回 False。
        """
        b, gb = 0, 1 << 30
        n = min(self.ni, 30)
        for _ in range(n):
            im = imread(random.choice(self.im_files))
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])
            b += im.size * im.element_size() * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}cache images 需要 {mem_required / gb:.1f}GB RAM（包含 {int(safety_margin * 100)}% safety margin），但仅有 {mem.available / gb:.1f}/{mem.total / gb:.1f}GB available，不会 cache images"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        """按 aspect ratio 排序 images，并设置 rectangular training 的 batch shapes。"""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)
        nb = bi[-1] + 1
        s = np.array([x.pop("shape") for x in self.labels])
        ar = s[:, 0] / s[:, 1]
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]
        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回给定 index 的 transformed label 信息。"""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """从 dataset 获取并返回 label 信息。

        参数:
            index (int): 要获取的 image index。

        返回:
            (dict[str, Any]): 包含 image 与 metadata 的 label dictionary。
        """
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        """返回 dataset 的 labels list 长度。"""
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """在这里自定义 label format。"""
        return label

    def build_transforms(self, hyp: (dict[str, Any] | None) = None):
        """用户可在这里自定义 augmentations。

        示例:
            >>> if self.augment:
            ...     # training transforms
            ...     return Compose([])
            >>> else:
            ...    # val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """用户可在这里自定义自己的 format。

        示例:
            确保 output 是包含以下 keys 的 dictionary：
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # format: (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # or False
            ...     bbox_format="xyxy",  # or xywh, ltwh
            ... )
        """
        raise NotImplementedError
