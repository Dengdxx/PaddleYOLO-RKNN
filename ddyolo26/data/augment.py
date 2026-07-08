# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 数据增强管道：Mosaic/CopyPaste/LetterBox 等变换实现。
@details
包含完整的训练期数据增强变换链：
- `LetterBox`：保持宽高比的等比缩放+填充，推理/验证必用
- `Mosaic`：4图/9图拼接（提升小目标检测）
- `RandomPerspective`：随机透视变换（旋转/缩放/剪切/位移）
- `CopyPaste`：实例复制粘贴（数据扩充）
- `Albumentations`：可选第三方增强库接口

在颜色敏感的数据集上，通常会调低部分激进增强（如 hsv_s/hsv_v），
以保持颜色类别信号的稳定性。
"""

from __future__ import annotations
import sys
import paddle


import math
import os
import random
from copy import deepcopy
from typing import Any

import cv2
import numpy as np

from ddyolo26.paddle_utils import *
from PIL import Image

from ddyolo26.data.utils import polygons2masks, polygons2masks_overlap
from ddyolo26.utils import LOGGER, IterableSimpleNamespace, colorstr
from ddyolo26.utils.checks import check_version
from ddyolo26.utils.instance import Instances
from ddyolo26.utils.metrics import bbox_ioa
from ddyolo26.utils.ops import segment2box, xywh2xyxy, xyxyxyxy2xywhr

DEFAULT_MEAN = 0.0, 0.0, 0.0
DEFAULT_STD = 1.0, 1.0, 1.0


class BaseTransform:
    """PaddleYOLO-RKNN image transform 的基础类。

    该类为各类 image 处理操作提供基础接口，兼容 classification 与 semantic segmentation 任务。

    方法:
        apply_image: 对 labels 应用 image transform。
        apply_instances: 对 labels 中的 object instances 应用 transform。
        apply_semantic: 对 image 应用 semantic segmentation transform。
        __call__: 对 image、instances 与 semantic masks 应用所有 label transform。

    示例:
        >>> transform = BaseTransform()
        >>> labels = {"image": np.array(...), "instances": [...], "semantic": np.array(...)}
        >>> transformed_labels = transform(labels)
    """

    def __init__(self) -> None:
        """初始化 BaseTransform 对象。

        该构造函数创建基础 transform 对象，可由具体 image 处理任务扩展；设计上兼容 classification
        与 semantic segmentation。
        """
        pass

    def apply_image(self, labels):
        """对 labels 应用 image transform。

        子类可重写该方法以实现具体 image transform 逻辑。基础实现保持输入 labels 不变。

        参数:
            labels (Any): 待 transform 的输入 labels；具体类型与结构取决于实现。

        返回:
            (Any): transform 后的 labels；基础实现中与输入相同。

        示例:
            >>> transform = BaseTransform()
            >>> original_labels = [1, 2, 3]
            >>> transformed_labels = transform.apply_image(original_labels)
            >>> print(transformed_labels)
            [1, 2, 3]
        """
        pass

    def apply_instances(self, labels):
        """对 labels 中的 object instances 应用 transform。

        该方法负责对给定 labels 内的 object instances 应用各类 transform；子类可重写以实现具体
        instance transform 逻辑。

        参数:
            labels (dict): 包含 label 信息的字典，其中包括 object instances。

        返回:
            (dict): 更新后的 labels 字典，其中 object instances 已完成 transform。

        示例:
            >>> transform = BaseTransform()
            >>> labels = {"instances": Instances(xyxy=paddle.rand(5, 4), cls=paddle.randint(0, 80, (5,)))}
            >>> transformed_labels = transform.apply_instances(labels)
        """
        pass

    def apply_semantic(self, labels):
        """对 image 应用 semantic segmentation transform。

        子类可重写该方法以实现具体 semantic segmentation transform。基础实现不执行操作。

        参数:
            labels (Any): 待 transform 的输入 labels 或 semantic segmentation mask。

        返回:
            (Any): transform 后的 semantic segmentation mask 或 labels。

        示例:
            >>> transform = BaseTransform()
            >>> semantic_mask = np.zeros((100, 100), dtype=np.uint8)
            >>> transformed_mask = transform.apply_semantic(semantic_mask)
        """
        pass

    def __call__(self, labels):
        """对 image、instances 与 semantic masks 应用所有 label transform。

        该方法按顺序调用 apply_image 与 apply_instances，协调 BaseTransform 中定义的各类 transform，
        分别处理 image 与 object instances。

        参数:
            labels (dict): 包含 image data 与 annotations 的字典。预期包含表示 image data 的 'img'
                和表示 object instances 的 'instances'。

        返回:
            (dict): 输入 labels 字典，其中 image 与 instances 已完成 transform。

        示例:
            >>> transform = BaseTransform()
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": []}
            >>> transformed_labels = transform(labels)
        """
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)


class Compose:
    """用于组合多个 image transform 的类。

    属性:
        transforms (list[Callable]): 需要顺序执行的 transform 函数列表。

    方法:
        __call__: 对输入数据应用一组 transform。
        append: 向现有 transform 列表追加新的 transform。
        insert: 在 transform 列表的指定位置插入新的 transform。
        __getitem__: 通过索引获取单个 transform 或一组 transform。
        __setitem__: 通过索引设置单个 transform 或一组 transform。
        tolist: 将 transform 列表转换为标准 Python list。

    示例:
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
        >>> compose.append(CenterCrop((224, 224)))
        >>> compose.insert(0, RandomFlip())
    """

    def __init__(self, transforms):
        """使用 transforms 列表初始化 Compose 对象。

        参数:
            transforms (list[Callable]): 需要顺序应用的 callable transform 对象列表。
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """对输入数据应用一系列 transform。

        该方法将 Compose 对象中的 transforms 顺序应用到输入数据。

        参数:
            data (Any): 待 transform 的输入数据；类型取决于列表中的 transforms。

        返回:
            (Any): 顺序应用所有 transforms 后的数据。

        示例:
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """向现有 transforms 列表追加新的 transform。

        参数:
            transform (BaseTransform): 要加入组合的 transform。

        示例:
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """在现有 transforms 列表的指定索引插入新的 transform。

        参数:
            index (int): 插入新 transform 的索引。
            transform (BaseTransform): 要插入的 transform 对象。

        示例:
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: (list | int)) -> Compose:
        """通过索引获取单个 transform 或一组 transforms。

        参数:
            index (int | list[int]): 要获取的 transform 索引或索引列表。

        返回:
            (Compose | Any): 若 index 为 list，返回新的 Compose 对象；若为 int，返回单个 transform。

        异常:
            AssertionError: 当 index 不是 int 或 list 类型时抛出。

        示例:
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
            >>> compose = Compose(transforms)
            >>> single_transform = compose[1]  # 直接返回 RandomPerspective transform
            >>> multiple_transforms = compose[[0, 1]]  # 返回包含 RandomFlip 与 RandomPerspective 的 Compose 对象
        """
        assert isinstance(index, (int, list)), f"indices 应为 list 或 int 类型，但得到 {type(index)}"
        return Compose([self.transforms[i] for i in index]) if isinstance(index, list) else self.transforms[index]

    def __setitem__(self, index: (list | int), value: (list | int)) -> None:
        """通过索引设置组合中的一个或多个 transforms。

        参数:
            index (int | list[int]): 要设置 transform 的索引或索引列表。
            value (Any | list[Any]): 要设置到指定索引处的 transform 或 transform 列表。

        异常:
            AssertionError: 当 index 类型无效、value 类型不匹配或 index 越界时抛出。

        示例:
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1] = NewTransform()  # 替换第二个 transform
            >>> compose[[0, 1]] = [NewTransform1(), NewTransform2()]  # 替换前两个 transforms
        """
        assert isinstance(index, (int, list)), f"indices 应为 list 或 int 类型，但得到 {type(index)}"
        if isinstance(index, list):
            assert isinstance(value, list), f"indices 与 values 应使用相同类型，但得到 {type(index)} 和 {type(value)}"
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} 超出范围 {len(self.transforms)}。"
            self.transforms[i] = v

    def tolist(self):
        """将 transforms 列表转换为标准 Python list。

        返回:
            (list): 包含 Compose 实例中所有 transform 对象的 list。

        示例:
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]
            >>> compose = Compose(transforms)
            >>> transform_list = compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    def __repr__(self):
        """返回 Compose 对象的字符串表示。

        返回:
            (str): Compose 对象的字符串表示，其中包含 transforms 列表。

        示例:
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose = Compose(transforms)
            >>> print(compose)
            Compose([
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform:
    """Cutmix、MixUp、Mosaic 等 mix transforms 的基础类。

    该类为 dataset 上的 mix transforms 提供基础实现，按概率决定是否应用 transform，并管理多张
    images 与 labels 的混合。

    属性:
        dataset (Any): 包含 images 与 labels 的 dataset 对象。
        pre_transform (Callable | None): mixing 前可选应用的 transform。
        p (float): 应用 mix transform 的概率。

    方法:
        __call__: 对输入 labels 应用 mix transform。
        _mix_transform: 子类实现的抽象方法，用于具体 mix 操作。
        get_indexes: 获取待混合 images 的索引。
        _update_label_text: 更新 mixed images 的 label text。

    示例:
        >>> class CustomMixTransform(BaseMixTransform):
        ...     def _mix_transform(self, labels):
        ...         # 在这里实现自定义 mix 逻辑
        ...         return labels
        ...
        ...     def get_indexes(self):
        ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        >>> dataset = YourDataset()
        >>> transform = CustomMixTransform(dataset, p=0.5)
        >>> mixed_labels = transform(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """初始化 CutMix、MixUp、Mosaic 等 mix transforms 的 BaseMixTransform 对象。

        该类是 image 处理 pipeline 中实现 mix transforms 的基础。

        参数:
            dataset (Any): 包含待 mixing 的 images 与 labels 的 dataset 对象。
            pre_transform (Callable | None): mixing 前可选应用的 transform。
            p (float): 应用 mix transform 的概率，应位于 [0.0, 1.0] 范围。
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对 labels data 应用预处理 transforms 与 cutmix/mixup/mosaic transforms。

        该方法根据概率决定是否应用 mix transform；若应用，则选择额外 images，按需执行 pre_transform，
        再执行 mix transform。

        参数:
            labels (dict[str, Any]): 包含某张 image 的 label data 的字典。

        返回:
            (dict[str, Any]): transform 后的 labels 字典，可能包含来自其他 images 的 mixed data。

        示例:
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """
        if random.uniform(0, 1) > self.p:
            return labels
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels
        labels = self._update_label_text(labels)
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _mix_transform(self, labels: dict[str, Any]):
        """对 label 字典应用 CutMix、MixUp 或 Mosaic augmentation。

        子类应实现该方法以执行 CutMix、MixUp 或 Mosaic 等具体 mix transforms；它会用 augmented data
        原地更新输入 label 字典。

        参数:
            labels (dict[str, Any]): 包含 image 与 label data 的字典。预期包含 'mix_labels' key，
                其中保存用于 mixing 的额外 image 与 label data 列表。

        返回:
            (dict[str, Any]): 应用 mix transform 后、包含 augmented data 的 labels 字典。

        示例:
            >>> transform = BaseMixTransform(dataset)
            >>> labels = {"image": img, "bboxes": boxes, "mix_labels": [{"image": img2, "bboxes": boxes2}]}
            >>> augmented_labels = transform._mix_transform(labels)
        """
        raise NotImplementedError

    def get_indexes(self):
        """为 mosaic augmentation 获取随机索引。

        返回:
            (int): 来自 dataset 的随机索引。

        示例:
            >>> transform = BaseMixTransform(dataset)
            >>> index = transform.get_indexes()
            >>> print(index)  # 7
        """
        return random.randint(0, len(self.dataset) - 1)

    @staticmethod
    def _update_label_text(labels: dict[str, Any]) -> dict[str, Any]:
        """更新 image augmentation 中 mixed labels 的 label text 与 class IDs。

        该方法处理输入 labels 字典及 mixed labels 中的 'texts' 与 'cls' 字段，创建统一的 text labels 集合，
        并相应更新 class IDs。

        参数:
            labels (dict[str, Any]): 包含 label 信息的字典，包括 'texts' 与 'cls' 字段，也可包含带有
                额外 label 字典的 'mix_labels' 字段。

        返回:
            (dict[str, Any]): 更新后的 labels 字典，其中 text labels 已统一且 class IDs 已更新。

        示例:
            >>> labels = {
            ...     "texts": [["cat"], ["dog"]],
            ...     "cls": paddle.to_tensor([[0], [1]]),
            ...     "mix_labels": [{"texts": [["bird"], ["fish"]], "cls": paddle.to_tensor([[0], [1]])}],
            ... }
            >>> updated_labels = BaseMixTransform._update_label_text(labels)
            >>> print(updated_labels["texts"])
            [['cat'], ['dog'], ['bird'], ['fish']]
            >>> print(updated_labels["cls"])
            tensor([[0],
                    [1]])
            >>> print(updated_labels["mix_labels"][0]["cls"])
            tensor([[2],
                    [3]])
        """
        if "texts" not in labels:
            return labels
        mix_texts = [
            *labels["texts"],
            *(item for x in labels["mix_labels"] for item in x["texts"]),
        ]
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}
        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    """用于 image dataset 的 Mosaic augmentation。

    该类将多张（4 或 9 张）images 合成为单张 mosaic image，并按给定概率应用到 dataset。

    属性:
        dataset: 应用 mosaic augmentation 的 dataset。
        imgsz (int): 单张 image 经过 mosaic pipeline 后的 image size（height 与 width）。
        p (float): 应用 mosaic augmentation 的概率，必须位于 0-1 范围。
        n (int): grid size，可为 4（2x2）或 9（3x3）。
        border (tuple[int, int]): height 与 width 的 border size。

    方法:
        get_indexes: 从 dataset 返回随机索引列表。
        _mix_transform: 对输入 image 与 labels 应用 mosaic transform。
        _mosaic3: 创建 1x3 image mosaic。
        _mosaic4: 创建 2x2 image mosaic。
        _mosaic9: 创建 3x3 image mosaic。
        _update_labels: 使用 padding 更新 labels。
        _cat_labels: 拼接 labels 并裁剪 mosaic border 外的 instances。

    示例:
        >>> from ddyolo26.data.augment import Mosaic
        >>> dataset = YourDataset(...)  # 你的 image dataset
        >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        """初始化 Mosaic augmentation 对象。

        该类将多张（4 或 9 张）images 合成为单张 mosaic image，并按给定概率应用到 dataset。

        参数:
            dataset (Any): 应用 mosaic augmentation 的 dataset。
            imgsz (int): 单张 image 经过 mosaic pipeline 后的 image size（height 与 width）。
            p (float): 应用 mosaic augmentation 的概率，必须位于 0-1 范围。
            n (int): grid size，可为 4（2x2）或 9（3x3）。
        """
        assert 0 <= p <= 1.0, f"probability 应位于 [0, 1] 范围，但得到 {p}。"
        assert n in {4, 9}, "grid 必须等于 4 或 9。"
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = -imgsz // 2, -imgsz // 2
        self.n = n
        self.buffer_enabled = self.dataset.cache != "ram"

    def get_indexes(self):
        """从 dataset 返回用于 mosaic augmentation 的随机索引列表。

        该方法根据 'buffer_enabled' 属性，从 buffer 或整个 dataset 中选择随机 image 索引，用于创建
        mosaic augmentations。

        返回:
            (list[int]): 随机 image 索引列表。列表长度为 n-1，其中 n 是 mosaic 使用的 images 数量
                （n 为 4 或 9 时分别是 3 或 8）。

        示例:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # 输出: 3
        """
        if self.buffer_enabled:
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对输入 image 与 labels 应用 mosaic augmentation。

        该方法根据 'n' 属性将多张 images（3、4 或 9 张）合成为单张 mosaic image，并确保不存在
        rectangular annotations，且存在可用于 mosaic augmentation 的其他 images。

        参数:
            labels (dict[str, Any]): 包含 image data 与 annotations 的字典。预期包含：
                - 'rect_shape': 应为 None，因为 rect 与 mosaic 互斥。
                - 'mix_labels': 字典列表，包含 mosaic 使用的其他 images 数据。

        返回:
            (dict[str, Any]): 包含 mosaic-augmented image 与更新后 annotations 的字典。

        异常:
            AssertionError: 当 'rect_shape' 非 None 或 'mix_labels' 为空时抛出。

        示例:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> augmented_data = mosaic._mix_transform(labels)
        """
        assert labels.get("rect_shape") is None, "rect 与 mosaic 互斥。"
        assert len(labels.get("mix_labels", [])), "没有可用于 mosaic augment 的其他 images。"
        return self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)

    def _mosaic3(self, labels: dict[str, Any]) -> dict[str, Any]:
        """组合三张 images 创建 1x3 image mosaic。

        该方法将三张 images 排成水平布局，主 image 位于中心，两侧各放一张额外 image；这是 object detection
        中 Mosaic augmentation 技术的一部分。

        参数:
            labels (dict[str, Any]): 包含主（中心）image 与 label 信息的字典。必须包含保存 image array 的
                'img' key，以及保存两侧 images 信息字典列表的 'mix_labels' key。

        返回:
            (dict[str, Any]): 包含 mosaic image 与更新后 labels 的字典。keys 包括：
                - 'img' (np.ndarray): shape 为 (H, W, C) 的 mosaic image array。
                - 输入 labels 中的其他 keys，会更新以反映新的 image dimensions。

        示例:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=3)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(2)],
            ... }
            >>> result = mosaic._mosaic3(labels)
            >>> print(result["img"].shape)
            (640, 640, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")
            if i == 0:
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
                h0, w0 = h, w
                c = s, s, s + w, s + h
            elif i == 1:
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:
                c = s - w, s + h0 - h, s, s + h0
            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)
            img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img3[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    def _mosaic4(self, labels: dict[str, Any]) -> dict[str, Any]:
        """由四张输入 images 创建 2x2 image mosaic。

        该方法将四张 images 放入 2x2 grid，合成为单张 mosaic image，并更新 mosaic 中每张 image
        对应的 labels。

        参数:
            labels (dict[str, Any]): 包含 base image（index 0）的 image data 与 labels 的字典，
                'mix_labels' key 中保存另外三张 images（indices 1-3）。

        返回:
            (dict[str, Any]): 包含 mosaic image 与更新后 labels 的字典。'img' key 保存 numpy array 形式的
                mosaic image，其他 keys 保存四张 images 合并并调整后的 labels。

        示例:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> labels = {
            ...     "img": np.random.rand(480, 640, 3),
            ...     "mix_labels": [{"img": np.random.rand(480, 640, 3)} for _ in range(3)],
            ... }
            >>> result = mosaic._mosaic4(labels)
            >>> assert result["img"].shape == (1280, 1280, 3)
        """
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")
            if i == 0:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _mosaic9(self, labels: dict[str, Any]) -> dict[str, Any]:
        """由输入 image 与八张额外 images 创建 3x3 image mosaic。

        该方法将九张 images 合成为单张 mosaic image。输入 image 放在中心，dataset 中的八张额外 images
        按 3x3 grid 排布在周围。

        参数:
            labels (dict[str, Any]): 包含输入 image 及其 labels 的字典，应包含以下 keys：
                - 'img' (np.ndarray): 输入 image。
                - 'resized_shape' (tuple[int, int]): resized image 的 shape（height, width）。
                - 'mix_labels' (list[dict]): 保存八张额外 images 信息的字典列表，每个字典与输入 labels 结构相同。

        返回:
            (dict[str, Any]): 包含 mosaic image 与更新后 labels 的字典，包含以下 keys：
                - 'img' (np.ndarray): 最终 mosaic image。
                - 输入 labels 中的其他 keys，会更新以反映新的 mosaic 排布。

        示例:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=9)
            >>> input_labels = dataset[0]
            >>> mosaic_result = mosaic._mosaic9(input_labels)
            >>> mosaic_image = mosaic_result["img"]
        """
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1
        for i in range(9):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")
            if i == 0:
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
                h0, w0 = h, w
                c = s, s, s + w, s + h
            elif i == 1:
                c = s, s - h, s + w, s
            elif i == 2:
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:
                c = s - w, s + h0 - hp - h, s, s + h0 - hp
            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)
            img9[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]
            hp, wp = h, w
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw: int, padh: int) -> dict[str, Any]:
        """使用 padding 值更新 label 坐标。

        该方法通过添加 padding 值调整 labels 中 object instances 的 bbox 坐标；若坐标此前为 normalized，
        也会执行 denormalize。

        参数:
            labels (dict[str, Any]): 包含 image 与 instance 信息的字典。
            padw (int): 要加到 x-coordinates 上的 padding width。
            padh (int): 要加到 y-coordinates 上的 padding height。

        返回:
            (dict): 更新后的 labels 字典，其中 instance 坐标已调整。

        示例:
            >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
            >>> padw, padh = 50, 50
            >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
        """
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
        """拼接并处理 mosaic augmentation 的 labels。

        该方法合并 mosaic augmentation 所用多张 images 的 labels，将 instances clip 到 mosaic border 内，
        并移除零面积 boxes。

        参数:
            mosaic_labels (list[dict[str, Any]]): mosaic 中每张 image 的 label 字典列表。

        返回:
            (dict[str, Any]): 包含 mosaic image 拼接并处理后 labels 的字典，包括：
                - im_file (str): mosaic 中第一张 image 的 file path。
                - ori_shape (tuple[int, int]): 第一张 image 的 original shape。
                - resized_shape (tuple[int, int]): mosaic image 的 shape（imgsz * 2, imgsz * 2）。
                - cls (np.ndarray): 拼接后的 class labels。
                - instances (Instances): 拼接后的 instance annotations。
                - mosaic_border (tuple[int, int]): Mosaic border size。
                - texts (list[str], optional): 若原始 labels 存在 text labels，则保留。

        示例:
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances', 'mosaic_border'])
        """
        if not mosaic_labels:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class MixUp(BaseMixTransform):
    """对 image dataset 应用 MixUp augmentation。

    该类实现论文 [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
    中描述的 MixUp augmentation 技术。MixUp 使用随机权重组合两张 images 及其 labels。

    属性:
        dataset (Any): 应用 MixUp augmentation 的 dataset。
        pre_transform (Callable | None): MixUp 前可选应用的 transform。
        p (float): 应用 MixUp augmentation 的概率。

    方法:
        _mix_transform: 对输入 labels 应用 MixUp augmentation。

    示例:
        >>> from ddyolo26.data.augment import MixUp
        >>> dataset = YourDataset(...)  # 你的 image dataset
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p: float = 0.0) -> None:
        """初始化 MixUp augmentation 对象。

        MixUp 是一种 image augmentation 技术，通过对两张 images 的 pixel values 与 labels 加权求和来组合它们。
        该实现面向 PaddleYOLO-RKNN 框架。

        参数:
            dataset (Any): 应用 MixUp augmentation 的 dataset。
            pre_transform (Callable | None): MixUp 前可选应用到 images 的 transform。
            p (float): 对 image 应用 MixUp augmentation 的概率，必须位于 [0, 1] 范围。
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对输入 labels 应用 MixUp augmentation。

        该方法实现论文 "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412)
        中描述的 MixUp augmentation 技术。

        参数:
            labels (dict[str, Any]): 包含 original image 与 label 信息的字典。

        返回:
            (dict[str, Any]): 包含 mixed-up image 与合并后 label 信息的字典。

        示例:
            >>> mixer = MixUp(dataset)
            >>> mixed_labels = mixer._mix_transform(labels)
        """
        r = np.random.beta(32.0, 32.0)
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


class CutMix(BaseMixTransform):
    """对 image dataset 应用论文 https://arxiv.org/abs/1905.04899 中描述的 CutMix augmentation。

    CutMix 使用另一张 image 的对应区域替换当前 image 的随机矩形区域来组合两张 images，并按 mixed region
    面积比例调整 labels。

    属性:
        dataset (Any): 应用 CutMix augmentation 的 dataset。
        pre_transform (Callable | None): CutMix 前可选应用的 transform。
        p (float): 应用 CutMix augmentation 的概率。
        beta (float): 用于采样 mixing ratio 的 Beta distribution 参数。
        num_areas (int): 尝试 cut 与 mix 的区域数量。

    方法:
        _mix_transform: 对输入 labels 应用 CutMix augmentation。
        _rand_bbox: 为 cut region 生成随机 bbox 坐标。

    示例:
        >>> from ddyolo26.data.augment import CutMix
        >>> dataset = YourDataset(...)  # 你的 image dataset
        >>> cutmix = CutMix(dataset, p=0.5)
        >>> augmented_labels = cutmix(original_labels)
    """

    def __init__(
        self,
        dataset,
        pre_transform=None,
        p: float = 0.0,
        beta: float = 1.0,
        num_areas: int = 3,
    ) -> None:
        """初始化 CutMix augmentation 对象。

        参数:
            dataset (Any): 应用 CutMix augmentation 的 dataset。
            pre_transform (Callable | None): CutMix 前可选应用的 transform。
            p (float): 应用 CutMix augmentation 的概率。
            beta (float): 用于采样 mixing ratio 的 Beta distribution 参数。
            num_areas (int): 尝试 cut 与 mix 的区域数量。
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.beta = beta
        self.num_areas = num_areas

    def _rand_bbox(self, width: int, height: int) -> tuple[int, int, int, int]:
        """为 cut region 生成随机 bbox 坐标。

        参数:
            width (int): image width。
            height (int): image height。

        返回:
            (tuple[int]): bbox 的 (x1, y1, x2, y2) 坐标。
        """
        lam = np.random.beta(self.beta, self.beta)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        return x1, y1, x2, y2

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对输入 labels 应用 CutMix augmentation。

        参数:
            labels (dict[str, Any]): 包含 original image 与 label 信息的字典。

        返回:
            (dict[str, Any]): 包含 mixed image 与调整后 labels 的字典。

        示例:
            >>> cutter = CutMix(dataset)
            >>> mixed_labels = cutter._mix_transform(labels)
        """
        h, w = labels["img"].shape[:2]
        cut_areas = np.asarray([self._rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32)
        ioa1 = bbox_ioa(cut_areas, labels["instances"].bboxes)
        idx = np.nonzero(ioa1.sum(axis=1) <= 0)[0]
        if len(idx) == 0:
            return labels
        labels2 = labels.pop("mix_labels")[0]
        area = cut_areas[np.random.choice(idx)]
        ioa2 = bbox_ioa(area[None], labels2["instances"].bboxes).squeeze(0)
        indexes2 = np.nonzero(ioa2 >= (0.01 if len(labels["instances"].segments) else 0.1))[0]
        if len(indexes2) == 0:
            return labels
        instances2 = labels2["instances"][indexes2]
        instances2.convert_bbox("xyxy")
        instances2.denormalize(w, h)
        x1, y1, x2, y2 = area.astype(np.int32)
        labels["img"][y1:y2, x1:x2] = labels2["img"][y1:y2, x1:x2]
        instances2.add_padding(-x1, -y1)
        instances2.clip(x2 - x1, y2 - y1)
        instances2.add_padding(x1, y1)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"][indexes2]], axis=0)
        labels["instances"] = Instances.concatenate([labels["instances"], instances2], axis=0)
        return labels


class RandomPerspective:
    """对 images 及对应 annotations 实现随机 perspective 与 affine transforms。

    该类对 images 及关联的 bboxes、segments、keypoints 应用随机旋转、平移、缩放、shearing 与
    perspective transforms，可作为 object detection 与 instance segmentation 任务 augmentation pipeline 的一部分。

    属性:
        degrees (float): 随机旋转的最大绝对角度范围。
        translate (float): 以 image size 比例表示的最大平移量。
        scale (float): scaling factor 范围，例如 scale=0.1 表示 0.9-1.1。
        shear (float): 最大 shear angle，单位为 degrees。
        perspective (float): perspective distortion factor。
        border (tuple[int, int]): 以 (y, x) 表示的 Mosaic border size。
        pre_transform (Callable | None): random perspective 前可选应用的 transform。

    方法:
        affine_transform: 对输入 image 应用 affine transforms。
        apply_bboxes: 使用 affine matrix transform bboxes。
        apply_segments: transform segments 并生成新的 bboxes。
        apply_keypoints: 使用 affine matrix transform keypoints。
        __call__: 对 images 与 annotations 应用 random perspective transform。
        box_candidates: 根据 size 与 aspect ratio 过滤 transform 后的 bboxes。

    示例:
        >>> transform = RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
        >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> labels = {"img": image, "cls": np.array([0, 1]), "instances": Instances(...)}
        >>> result = transform(labels)
        >>> transformed_image = result["img"]
        >>> transformed_instances = result["instances"]
    """

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: tuple[int, int] = (0, 0),
        pre_transform=None,
    ):
        """使用 transform 参数初始化 RandomPerspective 对象。

        该类对 images 及对应 bboxes、segments、keypoints 实现 random perspective 与 affine transforms。
        transform 包括 rotation、translation、scaling 与 shearing。

        参数:
            degrees (float): 随机 rotation 的 degree 范围。
            translate (float): 随机 translation 占总 width 与 height 的比例。
            scale (float): scaling factor 区间，例如 0.5 表示 resize 范围为 50%-150%。
            shear (float): shear 强度（angle in degrees）。
            perspective (float): perspective distortion factor。
            border (tuple[int, int]): 指定 mosaic border 的元组 (y, x)。
            pre_transform (Callable | None): 开始 random transform 前应用到 image 的函数/transform。
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.pre_transform = pre_transform

    def affine_transform(self, img: np.ndarray, border: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
        """以 image center 为中心应用一系列 affine transforms。

        该函数对输入 image 执行一系列 geometric transforms，包括 translation、perspective change、rotation、
        scaling 与 shearing。为保持一致性，这些 transforms 会按固定顺序应用。

        参数:
            img (np.ndarray): 待 transform 的输入 image。
            border (tuple[int, int]): transform 后 image 的 border dimensions。

        返回:
            img (np.ndarray): transform 后的 image。
            M (np.ndarray): 3x3 transformation matrix。
            s (float): transform 过程中应用的 scale factor。

        示例:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> border = (10, 10)
            >>> rp = RandomPerspective()
            >>> transformed_img, matrix, scale = rp.affine_transform(img, border)
        """
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2
        C[1, 2] = -img.shape[0] / 2
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]
        M = T @ S @ R @ P @ C
        if border[0] != 0 or border[1] != 0 or (M != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
            if img.ndim == 2:
                img = img[..., None]
        return img, M, s

    def apply_bboxes(self, bboxes: np.ndarray, M: np.ndarray) -> np.ndarray:
        """对 bboxes 应用 affine transform。

        该函数使用给定 transformation matrix 对一组 bboxes 应用 affine transform。

        参数:
            bboxes (np.ndarray): xyxy format 的 bboxes，shape 为 (N, 4)，其中 N 为 bboxes 数量。
            M (np.ndarray): shape 为 (3, 3) 的 affine transformation matrix。

        返回:
            (np.ndarray): transform 后 xyxy format 的 bboxes，shape 为 (N, 4)。

        示例:
            >>> rp = RandomPerspective()
            >>> bboxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32)
            >>> M = np.eye(3, dtype=np.float32)
            >>> transformed_bboxes = rp.apply_bboxes(bboxes, M)
        """
        n = len(bboxes)
        if n == 0:
            return bboxes
        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(self, segments: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对 segments 应用 affine transforms 并生成新的 bboxes。

        该函数对输入 segments 应用 affine transforms，并基于 transform 后的 segments 生成新的 bboxes。
        它还会 clip transform 后的 segments，使其位于新的 bboxes 内。

        参数:
            segments (np.ndarray): 输入 segments，shape 为 (N, M, 2)，其中 N 是 segments 数量，
                M 是每个 segment 的点数。
            M (np.ndarray): shape 为 (3, 3) 的 affine transformation matrix。

        返回:
            bboxes (np.ndarray): xyxy format 的新 bboxes，shape 为 (N, 4)。
            segments (np.ndarray): transform 并 clip 后的 segments，shape 为 (N, M, 2)。

        示例:
            >>> rp = RandomPerspective()
            >>> segments = np.random.rand(10, 500, 2)  # 10 个 segments，每个有 500 个点
            >>> M = np.eye(3)  # identity transformation matrix
            >>> new_bboxes, new_segments = rp.apply_segments(segments, M)
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments
        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints: np.ndarray, M: np.ndarray) -> np.ndarray:
        """对 keypoints 应用 affine transform。

        该方法使用给定 affine transformation matrix transform 输入 keypoints。必要时处理 perspective rescaling，
        并更新 transform 后落在 image boundaries 外的 keypoints visibility。

        参数:
            keypoints (np.ndarray): shape 为 (N, K, 3) 的 keypoints array，其中 N 为 instances 数量，
                K 为每个 instance 的 keypoints 数量，3 表示 (x, y, visibility)。
            M (np.ndarray): 3x3 affine transformation matrix。

        返回:
            (np.ndarray): transform 后的 keypoints array，shape 与输入相同，即 (N, K, 3)。

        示例:
            >>> random_perspective = RandomPerspective()
            >>> keypoints = np.random.rand(5, 17, 3)  # 5 个 instances，每个 17 个 keypoints
            >>> M = np.eye(3)  # identity transformation
            >>> transformed_keypoints = random_perspective.apply_keypoints(keypoints, M)
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T
        xy = xy[:, :2] / xy[:, 2:3]
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对 image 及其 labels 应用 random perspective 与 affine transforms。

        该方法对输入 image 执行 rotation、translation、scaling、shearing 与 perspective distortion 等一系列
        transforms，并相应调整 bboxes、segments 与 keypoints。

        参数:
            labels (dict[str, Any]): 包含 image data 与 annotations 的字典。

        返回:
            (dict[str, Any]): transform 后的 labels 字典，包含：
                - 'img' (np.ndarray): transform 后的 image。
                - 'cls' (np.ndarray): 更新后的 class labels。
                - 'instances' (Instances): 更新后的 object instances。
                - 'resized_shape' (tuple[int, int]): transform 后的新 image shape。

        示例:
            >>> transform = RandomPerspective()
            >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> labels = {
            ...     "img": image,
            ...     "cls": np.array([0, 1, 2]),
            ...     "instances": Instances(bboxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]])),
            ... }
            >>> result = transform(labels)
            >>> assert result["img"].shape[:2] == result["resized_shape"]

        说明:
            'labels' 参数必须包含：
                - 'img' (np.ndarray): 输入 image。
                - 'cls' (np.ndarray): class labels。
                - 'instances' (Instances): 带有 bboxes、segments 与 keypoints 的 object instances。
            可包含：
                - 'mosaic_border' (tuple[int, int]): mosaic augmentation 的 border size。
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)
        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])
        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2
        img, M, scale = self.affine_transform(img, border)
        bboxes = self.apply_bboxes(instances.bboxes, M)
        segments = instances.segments
        keypoints = instances.keypoints
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)
        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        new_instances.clip(*self.size)
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        i = self.box_candidates(
            box1=instances.bboxes.T,
            box2=new_instances.bboxes.T,
            area_thr=0.01 if len(segments) else 0.1,
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    @staticmethod
    def box_candidates(
        box1: np.ndarray,
        box2: np.ndarray,
        wh_thr: int = 2,
        ar_thr: int = 100,
        area_thr: float = 0.1,
        eps: float = 1e-16,
    ) -> np.ndarray:
        """根据 size 与 aspect ratio 条件计算待进一步处理的 candidate boxes。

        该方法比较 augmentation 前后的 boxes，判断其是否满足 width、height、aspect ratio 与 area 阈值；
        用于过滤被 augmentation 过度扭曲或缩小的 boxes。

        参数:
            box1 (np.ndarray): augmentation 前的 original boxes，shape 为 (4, N)，其中 N 是 boxes 数量。
                format 为 absolute coordinates 下的 [x1, y1, x2, y2]。
            box2 (np.ndarray): transform 后的 augmented boxes，shape 为 (4, N)。format 为 absolute coordinates
                下的 [x1, y1, x2, y2]。
            wh_thr (int): 以 pixels 为单位的 width 与 height 阈值。任一维度小于该值的 boxes 会被拒绝。
            ar_thr (int): aspect ratio 阈值。aspect ratio 大于该值的 boxes 会被拒绝。
            area_thr (float): area ratio 阈值。area ratio（new/old）小于该值的 boxes 会被拒绝。
            eps (float): 用于避免除零的小 epsilon 值。

        返回:
            (np.ndarray): shape 为 (N,) 的 boolean array，表示哪些 boxes 是 candidates。True 表示满足全部条件。

        示例:
            >>> random_perspective = RandomPerspective()
            >>> box1 = np.array([[0, 0, 100, 100], [0, 0, 50, 50]]).T
            >>> box2 = np.array([[10, 10, 90, 90], [5, 5, 45, 45]]).T
            >>> candidates = random_perspective.box_candidates(box1, box2)
            >>> print(candidates)
            [True True]
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


class RandomHSV:
    """随机调整 image 的 Hue、Saturation、Value（HSV）channels。

    该类在 hgain、sgain、vgain 设置的预定义范围内，对 images 应用 random HSV augmentation。

    属性:
        hgain (float): hue 的最大变化量，范围通常为 [0, 1]。
        sgain (float): saturation 的最大变化量，范围通常为 [0, 1]。
        vgain (float): value 的最大变化量，范围通常为 [0, 1]。

    方法:
        __call__: 对 image 应用 random HSV augmentation。

    示例:
        >>> import numpy as np
        >>> from ddyolo26.data.augment import RandomHSV
        >>> augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> labels = {"img": image}
        >>> labels = augmenter(labels)
        >>> augmented_image = labels["img"]
    """

    def __init__(self, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
        """初始化用于 random HSV（Hue、Saturation、Value）augmentation 的 RandomHSV 对象。

        该类在指定范围内对 image 的 HSV channels 进行随机调整。

        参数:
            hgain (float): hue 的最大变化量，应位于 [0, 1] 范围。
            sgain (float): saturation 的最大变化量，应位于 [0, 1] 范围。
            vgain (float): value 的最大变化量，应位于 [0, 1] 范围。
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """在预定义范围内对 image 应用 random HSV augmentation。

        该方法通过随机调整 Hue、Saturation、Value（HSV）channels 来修改输入 image。调整幅度由初始化时的
        hgain、sgain、vgain 限定。

        参数:
            labels (dict[str, Any]): 包含 image data 与 metadata 的字典。必须包含保存 numpy array image 的
                'img' key。

        返回:
            (dict[str, Any]): 包含 HSV-augmented image 的 labels 字典。

        示例:
            >>> hsv_augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> labels = {"img": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}
            >>> labels = hsv_augmenter(labels)
            >>> augmented_img = labels["img"]
        """
        img = labels["img"]
        if img.shape[-1] != 3:
            return labels
        if self.hgain or self.sgain or self.vgain:
            dtype = img.dtype
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
            lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
            lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
            lut_sat[0] = 0
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return labels


class RandomFlip:
    """按给定概率对 image 应用随机 horizontal 或 vertical flip。

    该类执行 random image flipping，并更新 bboxes、keypoints 等对应 instance annotations。

    属性:
        p (float): 应用 flip 的概率，必须位于 0 到 1 之间。
        direction (str): flip 方向，可为 'horizontal' 或 'vertical'。
        flip_idx (array-like): flip keypoints 时使用的索引映射（如适用）。

    方法:
        __call__: 对 image 及其 annotations 应用 random flip transform。

    示例:
        >>> transform = RandomFlip(p=0.5, direction="horizontal")
        >>> result = transform({"img": image, "instances": instances})
        >>> flipped_image = result["img"]
        >>> flipped_instances = result["instances"]
    """

    def __init__(
        self,
        p: float = 0.5,
        direction: str = "horizontal",
        flip_idx: (list[int] | None) = None,
    ) -> None:
        """使用概率与方向初始化 RandomFlip 类。

        该类按给定概率对 image 应用随机 horizontal 或 vertical flip，并相应更新所有 instances
        （bboxes、keypoints 等）。

        参数:
            p (float): 应用 flip 的概率，必须位于 0 到 1 之间。
            direction (str): 应用 flip 的方向，必须为 'horizontal' 或 'vertical'。
            flip_idx (list[int] | None): flip keypoints 使用的索引映射（如有）。

        异常:
            AssertionError: 当 direction 不是 'horizontal' 或 'vertical'，或 p 不在 0 到 1 之间时抛出。
        """
        assert direction in {
            "horizontal",
            "vertical",
        }, f"仅支持 direction `horizontal` 或 `vertical`，但得到 {direction}"
        assert 0 <= p <= 1.0, f"probability 应位于 [0, 1] 范围，但得到 {p}。"
        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对 image 应用 random flip，并相应更新 bboxes 或 keypoints 等 instances。

        该方法根据初始化的概率与方向，对输入 image 执行 horizontal 或 vertical random flip，并更新对应
        instances（bboxes、keypoints）以匹配 flipped image。

        参数:
            labels (dict[str, Any]): 包含以下 keys 的字典：
                - 'img' (np.ndarray): 待 flip 的 image。
                - 'instances' (ddyolo26.utils.instance.Instances): 包含 boxes 及可选 keypoints 的对象。

        返回:
            (dict[str, Any]): 同一个字典，其中包含 flipped image 与更新后的 instances：
                - 'img' (np.ndarray): flipped image。
                - 'instances' (ddyolo26.utils.instance.Instances): 与 flipped image 匹配的更新后 instances。

        示例:
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": Instances(...)}
            >>> random_flip = RandomFlip(p=0.5, direction="horizontal")
            >>> flipped_labels = random_flip(labels)
        """
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class LetterBox:
    """为 detection、instance segmentation、pose 执行 image resize 与 padding。

    该类在保持 aspect ratio 的同时将 images resize 并 pad 到指定 shape，同时更新对应 labels 与 bboxes。

    属性:
        new_shape (tuple): resize 的 target shape（height, width）。
        auto (bool): 是否使用 minimum rectangle。
        scale_fill (bool): 是否将 image 拉伸到 new_shape。
        scaleup (bool): 是否允许 scale up；若为 False，则只 scale down。
        stride (int): 用于 rounding padding 的 stride。
        center (bool): 是否居中 image，否则对齐到 top-left。

    方法:
        __call__: resize 并 pad image，同时更新 labels 与 bboxes。

    示例:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """初始化用于 resize 与 padding images 的 LetterBox 对象。

        该类用于 object detection、instance segmentation 与 pose estimation 任务中的 image resize 与 padding。
        支持 auto-sizing、scale-fill 与 letterboxing 等 resize 模式。

        参数:
            new_shape (tuple[int, int]): resized image 的 target size（height, width）。
            auto (bool): 若为 True，使用 minimum rectangle resize；若为 False，直接使用 new_shape。
            scale_fill (bool): 若为 True，将 image 拉伸到 new_shape，不添加 padding。
            scaleup (bool): 若为 True，允许 scaling up；若为 False，则只 scale down。
            center (bool): 若为 True，居中放置 image；若为 False，放在 top-left corner。
            stride (int): model 的 stride（例如 YOLOv5 为 32）。
            padding_value (int): padding image 使用的值，默认 114。
            interpolation (int): resize 使用的 interpolation method，默认 cv2.INTER_LINEAR。
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(self, labels: (dict[str, Any] | None) = None, image: np.ndarray = None) -> dict[str, Any] | np.ndarray:
        """为 object detection、instance segmentation 或 pose estimation 任务 resize 并 pad image。

        该方法对输入 image 应用 letterboxing：保持 aspect ratio resize image，并添加 padding 以适配新 shape。
        它也会相应更新关联 labels。

        参数:
            labels (dict[str, Any] | None): 包含 image data 与关联 labels 的字典；若为 None，则使用空 dict。
            image (np.ndarray | None): numpy array 形式的输入 image。若为 None，则从 'labels' 中取 image。

        返回:
            (dict[str, Any] | np.ndarray): 若提供 'labels'，返回包含 resized/padded image、更新后 labels
                与额外 metadata 的字典。若 'labels' 为空，则仅返回 resized/padded image。

        示例:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = new_shape, new_shape
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = new_shape[1], new_shape[0]
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        if self.center:
            dw /= 2
            dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]
        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)
        h, w, c = img.shape
        if c == 3:
            img = cv2.copyMakeBorder(
                img,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(self.padding_value,) * 3,
            )
        else:
            pad_img = np.full(
                (h + top + bottom, w + left + right, c),
                fill_value=self.padding_value,
                dtype=img.dtype,
            )
            pad_img[top : top + h, left : left + w] = img
            img = pad_img
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = labels["ratio_pad"], (left, top)
        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    @staticmethod
    def _update_labels(labels: dict[str, Any], ratio: tuple[float, float], padw: float, padh: float) -> dict[str, Any]:
        """对 image 应用 letterboxing 后更新 labels。

        该方法修改 labels 中 instances 的 bbox 坐标，以补偿 letterboxing 过程中执行的 resize 与 padding。

        参数:
            labels (dict[str, Any]): 包含 image labels 与 instances 的字典。
            ratio (tuple[float, float]): 应用到 image 的 scaling ratios（width, height）。
            padw (float): 添加到 image 的 padding width。
            padh (float): 添加到 image 的 padding height。

        返回:
            (dict[str, Any]): 更新后的 labels 字典，其中 instance 坐标已修改。

        示例:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {"instances": Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh)
        """
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class CopyPaste(BaseMixTransform):
    """用于对 image dataset 应用 Copy-Paste augmentation 的 CopyPaste 类。

    该类实现论文 "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
    (https://arxiv.org/abs/2012.07177) 中描述的 Copy-Paste augmentation 技术。它组合不同 images 中的
    objects 来创建新的 training samples。

    属性:
        dataset (Any): 应用 Copy-Paste augmentation 的 dataset。
        pre_transform (Callable | None): Copy-Paste 前可选应用的 transform。
        p (float): 应用 Copy-Paste augmentation 的概率。

    方法:
        _mix_transform: 对输入 labels 应用 Copy-Paste augmentation。
        __call__: 对 images 与 annotations 应用 Copy-Paste transform。

    示例:
        >>> from ddyolo26.data.augment import CopyPaste
        >>> dataset = YourDataset(...)  # 你的 image dataset
        >>> copypaste = CopyPaste(dataset, p=0.5)
        >>> augmented_labels = copypaste(original_labels)
    """

    def __init__(self, dataset=None, pre_transform=None, p: float = 0.5, mode: str = "flip") -> None:
        """使用 dataset、pre_transform 与应用 CopyPaste 的概率初始化 CopyPaste 对象。"""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        assert mode in {
            "flip",
            "mixup",
        }, f"`mode` 预期为 `flip` 或 `mixup`，但得到 {mode}。"
        self.mode = mode

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        """应用 Copy-Paste augmentation，将另一张 image 中的 objects 组合到当前 image。"""
        labels2 = labels["mix_labels"][0]
        return self._transform(labels, labels2)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对 image 及其 labels 应用 Copy-Paste augmentation。"""
        if len(labels["instances"].segments) == 0 or self.p == 0:
            return labels
        if self.mode == "flip":
            return self._transform(labels)
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels
        labels = self._update_label_text(labels)
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _transform(self, labels1: dict[str, Any], labels2: dict[str, Any] = {}) -> dict[str, Any]:
        """应用 Copy-Paste augmentation，将另一张 image 中的 objects 组合到当前 image。"""
        im = labels1["img"]
        if "mosaic_border" not in labels1:
            im = im.copy()
        cls = labels1["cls"]
        h, w = im.shape[:2]
        instances = labels1.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)
        im_new = np.zeros(im.shape[:2], np.uint8)
        instances2 = labels2.pop("instances", None)
        if instances2 is None:
            instances2 = deepcopy(instances)
            instances2.fliplr(w)
        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)
        indexes = np.nonzero((ioa < 0.3).all(1))[0]
        n = len(indexes)
        sorted_idx = np.argsort(ioa.max(1)[indexes])
        indexes = indexes[sorted_idx]
        for j in indexes[: round(self.p * n)]:
            cls = np.concatenate((cls, labels2.get("cls", cls)[[j]]), axis=0)
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, 1, cv2.FILLED)
        result = labels2.get("img", cv2.flip(im, 1))
        if result.ndim == 2:
            result = result[..., None]
        i = im_new.astype(bool)
        im[i] = result[i]
        labels1["img"] = im
        labels1["cls"] = cls
        labels1["instances"] = instances
        return labels1


class Albumentations:
    """用于 image augmentation 的 Albumentations transforms。

    该类使用 Albumentations library 应用各类 image transforms，包括 Blur、Median Blur、转 grayscale、
    Contrast Limited Adaptive Histogram Equalization（CLAHE）、brightness/contrast 随机变化、RandomGamma，
    以及通过 compression 降低 image quality。

    属性:
        p (float): 应用 transforms 的概率。
        transform (albumentations.Compose): 组合后的 Albumentations transforms。
        contains_spatial (bool): 表示 transforms 是否包含 spatial operations。

    方法:
        __call__: 对输入 labels 应用 Albumentations transforms。

    示例:
        >>> transform = Albumentations(p=0.5)
        >>> augmented_labels = transform(labels)

    说明:
        - 需要 Albumentations 1.0.3 或更高版本。
        - Spatial transforms 会以不同方式处理，以确保 bbox compatibility。
        - 部分 transforms 默认以极低概率（0.01）应用。
    """

    def __init__(self, p: float = 1.0, transforms: (list | None) = None) -> None:
        """初始化用于 YOLO bbox format 参数的 Albumentations transform 对象。

        该类使用 Albumentations library 应用各类 image augmentations，包括 Blur、Median Blur、转 grayscale、
        Contrast Limited Adaptive Histogram Equalization、brightness/contrast 随机变化、RandomGamma，
        以及通过 compression 降低 image quality。

        参数:
            p (float): 应用 augmentations 的概率，必须位于 0 到 1 之间。
            transforms (list | None): 自定义 Albumentations transforms 列表。若为 None，则使用默认 transforms。
        """
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            pass
            os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }
            T = (
                [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_range=(75, 100), p=0.0),
                ]
                if transforms is None
                else transforms
            )
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = (
                A.Compose(
                    T,
                    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
                )
                if self.contains_spatial
                else A.Compose(T)
            )
            if hasattr(self.transform, "set_random_seed"):
                self.transform.set_random_seed(paddle.get_rng_state()[0].current_seed())
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对输入 labels 应用 Albumentations transforms。

        该方法使用 Albumentations library 应用一系列 image augmentations。它可对输入 image 及其对应 labels
        执行 spatial 与 non-spatial transforms。

        参数:
            labels (dict[str, Any]): 包含 image data 与 annotations 的字典。预期 keys：
                - 'img': 表示 image 的 np.ndarray
                - 'cls': class labels 的 np.ndarray
                - 'instances': 包含 bboxes 与其他 instance 信息的对象

        返回:
            (dict[str, Any]): 输入字典，其中 image 已 augmentation，annotations 已更新。

        示例:
            >>> transform = Albumentations(p=0.5)
            >>> labels = {
            ...     "img": np.random.rand(640, 640, 3),
            ...     "cls": np.array([0, 1]),
            ...     "instances": Instances(bboxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 0.8, 0.8]])),
            ... }
            >>> augmented = transform(labels)
            >>> assert augmented["img"].shape == (640, 640, 3)

        说明:
            - 该方法按 self.p 的概率应用 transforms。
            - Spatial transforms 会更新 bboxes；non-spatial transforms 只修改 image。
            - 需要安装 Albumentations library。
        """
        if self.transform is None or random.random() > self.p:
            return labels
        im = labels["img"]
        if im.shape[2] != 3:
            return labels
        if self.contains_spatial:
            cls = labels["cls"]
            if len(cls):
                labels["instances"].convert_bbox("xywh")
                labels["instances"].normalize(*im.shape[:2][::-1])
                bboxes = labels["instances"].bboxes
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)
                if len(new["class_labels"]) > 0:
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"]).reshape(-1, 1)
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
                labels["instances"].update(bboxes=bboxes)
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]
        return labels


class Format:
    """用于格式化 object detection、instance segmentation 与 pose estimation 任务 image annotations 的类。

    该类标准化 image 与 instance annotations，以供 Paddle DataLoader 中的 `collate_fn` 使用。

    属性:
        bbox_format (str): bboxes 的 format，可选 'xywh' 或 'xyxy'。
        normalize (bool): 是否 normalize bboxes。
        return_mask (bool): 是否返回用于 segmentation 的 instance masks。
        return_keypoint (bool): 是否返回用于 pose estimation 的 keypoints。
        return_obb (bool): 是否返回 oriented bboxes。
        mask_ratio (int): masks 的 downsample ratio。
        mask_overlap (bool): masks 是否允许 overlap。
        batch_idx (bool): 是否保留 batch indexes。
        bgr (float): 返回 BGR images 的概率。

    方法:
        __call__: 格式化包含 image、classes、bboxes 以及可选 masks/keypoints 的 labels 字典。
        _format_img: 将 image 从 Numpy array 转为 Paddle tensor。
        _format_segments: 将 polygon points 转为 bitmap masks。

    示例:
        >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
        >>> formatted_labels = formatter(labels)
        >>> img = formatted_labels["img"]
        >>> bboxes = formatted_labels["bboxes"]
        >>> masks = formatted_labels["masks"]
    """

    def __init__(
        self,
        bbox_format: str = "xywh",
        normalize: bool = True,
        return_mask: bool = False,
        return_keypoint: bool = False,
        return_obb: bool = False,
        mask_ratio: int = 4,
        mask_overlap: bool = True,
        batch_idx: bool = True,
        bgr: float = 0.0,
    ):
        """使用给定参数初始化 Format 类，用于 image 与 instance annotation formatting。

        该类为 object detection、instance segmentation 与 pose estimation 任务标准化 image 和 instance
        annotations，使其可用于 Paddle DataLoader 的 `collate_fn`。

        参数:
            bbox_format (str): bboxes 的 format，可为 'xywh'、'xyxy' 等。
            normalize (bool): 是否将 bboxes normalize 到 [0,1]。
            return_mask (bool): 若为 True，返回 segmentation 任务的 instance masks。
            return_keypoint (bool): 若为 True，返回 pose estimation 任务的 keypoints。
            return_obb (bool): 若为 True，返回 oriented bboxes。
            mask_ratio (int): masks 的 downsample ratio。
            mask_overlap (bool): 若为 True，允许 mask overlap。
            batch_idx (bool): 若为 True，保留 batch indexes。
            bgr (float): 返回 BGR images 而非 RGB 的概率。
        """
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx
        self.bgr = bgr

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """为 object detection、instance segmentation 与 pose estimation 任务格式化 image annotations。

        该方法标准化 image 与 instance annotations，以供 Paddle DataLoader 中的 `collate_fn` 使用。
        它会处理输入 labels 字典，将 annotations 转为指定 format，并在需要时应用 normalization。

        参数:
            labels (dict[str, Any]): 包含 image 与 annotation data 的字典，含以下 keys：
                - 'img': numpy array 形式的输入 image。
                - 'cls': instances 的 class labels。
                - 'instances': 包含 bboxes、segments 与 keypoints 的 Instances 对象。

        返回:
            (dict[str, Any]): 包含格式化后 data 的字典，包括：
                - 'img': 格式化后的 image tensor。
                - 'cls': class labels tensor。
                - 'bboxes': 指定 format 的 bboxes tensor。
                - 'masks': instance masks tensor（若 return_mask 为 True）。
                - 'keypoints': keypoints tensor（若 return_keypoint 为 True）。
                - 'batch_idx': batch index tensor（若 batch_idx 为 True）。

        示例:
            >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
            >>> labels = {"img": np.random.rand(640, 640, 3), "cls": np.array([0, 1]), "instances": Instances(...)}
            >>> formatted_labels = formatter(labels)
            >>> print(formatted_labels.keys())
        """
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)
        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = paddle.from_numpy(masks)
                cls_tensor = paddle.from_numpy(cls.squeeze(1))
                if self.mask_overlap:
                    sem_masks = cls_tensor[masks[0].astype("int64") - 1]
                else:
                    sem_masks = (masks.astype("float32") * cls_tensor[:, None, None]).max(0)
                    overlap = masks.sum(axis=0) > 1
                    if overlap.any():
                        weights = masks.sum(axis=(1, 2)).astype("float32")
                        weighted_masks = masks.astype("float32") * weights[:, None, None]
                        weighted_masks[masks == 0] = weights.max() + 1
                        smallest_idx = weighted_masks.argmin(axis=0)
                        sem_masks[overlap] = cls_tensor[smallest_idx[overlap]]
            else:
                masks = paddle.zeros(
                    [1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio],
                    dtype="uint8",
                )
                sem_masks = paddle.zeros([img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio])
            labels["masks"] = masks
            labels["sem_masks"] = sem_masks.astype("float32")
        labels["img"] = self._format_img(img)
        labels["cls"] = paddle.from_numpy(cls) if nl else paddle.zeros(nl, 1)
        labels["bboxes"] = paddle.from_numpy(instances.bboxes) if nl else paddle.zeros((nl, 4))
        if self.return_keypoint:
            labels["keypoints"] = (
                paddle.empty(0, 3) if instances.keypoints is None else paddle.from_numpy(instances.keypoints)
            )
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(paddle.from_numpy(instances.segments))
                if len(instances.segments)
                else paddle.zeros((0, 5))
            )
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        if self.batch_idx:
            labels["batch_idx"] = paddle.zeros(nl)
        return labels

    def _format_img(self, img: np.ndarray) -> paddle.Tensor:
        """将 YOLO image 从 Numpy array 格式化为 Paddle tensor。

        该函数执行以下操作：
        1. 确保 image 为 3 维（必要时添加 channel 维度）。
        2. 将 image 从 HWC format 转为 CHW format。
        3. 根据 bgr 概率可选反转 color channels（例如 BGR 到 RGB）。
        4. 将 image 转为 contiguous array。
        5. 将 Numpy array 转为 Paddle tensor。

        参数:
            img (np.ndarray): Numpy array 形式的输入 image，shape 为 (H, W, C) 或 (H, W)。

        返回:
            (paddle.Tensor): 格式化后的 image，Paddle tensor，shape 为 (C, H, W)。

        示例:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> formatted_img = self._format_img(img)
            >>> print(formatted_img.shape)
            Shape([3, 100, 100])
        """
        if len(img.shape) < 3:
            img = img[..., None]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr and img.shape[0] == 3 else img)
        img = paddle.from_numpy(img)
        return img

    def _format_segments(
        self, instances: Instances, cls: np.ndarray, w: int, h: int
    ) -> tuple[np.ndarray, Instances, np.ndarray]:
        """将 polygon segments 转为 bitmap masks。

        参数:
            instances (Instances): 包含 segment 信息的对象。
            cls (np.ndarray): 每个 instance 的 class labels。
            w (int): image width。
            h (int): image height。

        返回:
            masks (np.ndarray): bitmap masks，shape 为 (N, H, W)；若 mask_overlap 为 True，则为 (1, H, W)。
            instances (Instances): 更新后的 instances 对象；若 mask_overlap 为 True，segments 已排序。
            cls (np.ndarray): 更新后的 class labels；若 mask_overlap 为 True，已排序。

        说明:
            - 若 self.mask_overlap 为 True，masks 会 overlap 并按 area 排序。
            - 若 self.mask_overlap 为 False，每个 mask 单独表示。
            - Masks 会按 self.mask_ratio downsample。
        """
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)
        return masks, instances, cls


class LoadVisualPrompt:
    """从 bboxes 或 masks 创建用于 model input 的 visual prompts。"""

    def __init__(self, scale_factor: float = 1 / 8) -> None:
        """使用 scale factor 初始化 LoadVisualPrompt。

        参数:
            scale_factor (float): 缩放输入 image dimensions 的 factor。
        """
        self.scale_factor = scale_factor

    @staticmethod
    def make_mask(boxes: paddle.Tensor, h: int, w: int) -> paddle.Tensor:
        """由 bboxes 创建 binary masks。

        参数:
            boxes (paddle.Tensor): xyxy format 的 bboxes，shape: (N, 4)。
            h (int): mask height。
            w (int): mask width。

        返回:
            (paddle.Tensor): shape 为 (N, h, w) 的 binary masks。
        """
        x1, y1, x2, y2 = paddle.chunk(boxes[:, :, None], 4, 1)
        r = paddle.arange(w)[None, None, :]
        c = paddle.arange(h)[None, :, None]
        return (r >= x1) * (r < x2) * (c >= y1) * (c < y2)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """处理 labels 以创建 visual prompts。

        参数:
            labels (dict[str, Any]): 包含 image data 与 annotations 的字典。

        返回:
            (dict[str, Any]): 添加 visual prompts 后的 labels。
        """
        imgsz = labels["img"].shape[1:]
        bboxes, masks = None, None
        if "bboxes" in labels:
            bboxes = labels["bboxes"]
            bboxes = xywh2xyxy(bboxes) * paddle.tensor(imgsz)[[1, 0, 1, 0]]
        cls = labels["cls"].squeeze(-1).to(paddle.int32)
        visuals = self.get_visuals(cls, imgsz, bboxes=bboxes, masks=masks)
        labels["visuals"] = visuals
        return labels

    def get_visuals(
        self,
        category: (int | np.ndarray | paddle.Tensor),
        shape: tuple[int, int],
        bboxes: (np.ndarray | paddle.Tensor) = None,
        masks: (np.ndarray | paddle.Tensor) = None,
    ) -> paddle.Tensor:
        """基于 bboxes 或 masks 生成 visual masks。

        参数:
            category (int | np.ndarray | paddle.Tensor): objects 的 category labels。
            shape (tuple[int, int]): image shape（height, width）。
            bboxes (np.ndarray | paddle.Tensor, optional): objects 的 bboxes，xyxy format。
            masks (np.ndarray | paddle.Tensor, optional): objects 的 masks。

        返回:
            (paddle.Tensor): 包含每个 category visual masks 的 tensor。

        异常:
            ValueError: 当 bboxes 与 masks 均未提供时抛出。
        """
        masksz = int(shape[0] * self.scale_factor), int(shape[1] * self.scale_factor)
        if bboxes is not None:
            if isinstance(bboxes, np.ndarray):
                bboxes = paddle.from_numpy(bboxes)
            bboxes *= self.scale_factor
            masks = self.make_mask(bboxes, *masksz).float()
        elif masks is not None:
            if isinstance(masks, np.ndarray):
                masks = paddle.from_numpy(masks)
            masks = paddle.nn.functional.interpolate(masks.unsqueeze(1), masksz, mode="nearest").squeeze(1).float()
        else:
            raise ValueError("LoadVisualPrompt 的 label 中必须包含 bboxes 或 masks")
        if not isinstance(category, paddle.Tensor):
            category = paddle.tensor(category, dtype=paddle.int32)
        cls_unique, inverse_indices = paddle.compat.unique(category, sorted=True, return_inverse=True)
        visuals = paddle.zeros(cls_unique.shape[0], *masksz)
        for idx, mask in zip(inverse_indices, masks):
            visuals[idx] = paddle.logical_or(visuals[idx], mask)
        return visuals


class RandomLoadText:
    """随机采样 positive/negative texts，并相应更新 class indices。

    该类负责从给定 class texts 集合中采样 texts，包括 positive（image 中存在）与 negative（image 中不存在）
    samples。它会更新 class indices 以反映采样后的 texts，并可选择将 text list pad 到固定长度。

    属性:
        prompt_format (str): text prompts 的 format string。
        neg_samples (tuple[int, int]): 随机采样 negative texts 的范围。
        max_samples (int): 单张 image 中不同 text samples 的最大数量。
        padding (bool): 是否将 texts pad 到 max_samples。
        padding_value (list[str]): padding 为 True 时使用的 padding text。

    方法:
        __call__: 处理输入 labels，并返回更新后的 classes 与 texts。

    示例:
        >>> loader = RandomLoadText(prompt_format="Object: {}", neg_samples=(5, 10), max_samples=20)
        >>> labels = {"cls": [0, 1, 2], "texts": [["cat"], ["dog"], ["bird"]], "instances": [...]}
        >>> updated_labels = loader(labels)
        >>> print(updated_labels["texts"])
        ['Object: cat', 'Object: dog', 'Object: bird', 'Object: elephant', 'Object: car']
    """

    def __init__(
        self,
        prompt_format: str = "{}",
        neg_samples: tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: list[str] = [""],
    ) -> None:
        """初始化用于随机采样 positive 与 negative texts 的 RandomLoadText 类。

        该类用于随机采样 positive texts 与 negative texts，并根据 samples 数量相应更新 class indices。
        可用于 text-based object detection 任务。

        参数:
            prompt_format (str): prompt 的 format string。format string 应包含一对花括号 {}，用于插入 text。
            neg_samples (tuple[int, int]): 随机采样 negative texts 的范围。第一个整数指定 negative samples
                最小数量，第二个整数指定最大数量。
            max_samples (int): 单张 image 中不同 text samples 的最大数量。
            padding (bool): 是否将 texts pad 到 max_samples。若为 True，texts 数量始终等于 max_samples。
            padding_value (list[str]): padding 为 True 时使用的 padding text。
        """
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """随机采样 positive/negative texts，并相应更新 class indices。

        该方法基于 image 中现有 class labels 采样 positive texts，并从剩余 classes 中随机选择 negative texts。
        随后更新 class indices，以匹配新的 sampled text 顺序。

        参数:
            labels (dict[str, Any]): 包含 image labels 与 metadata 的字典。必须包含 'texts' 与 'cls' keys。

        返回:
            (dict[str, Any]): 更新后的 labels 字典，包含新的 'cls' 与 'texts' entries。

        示例:
            >>> loader = RandomLoadText(prompt_format="A photo of {}", neg_samples=(5, 10), max_samples=20)
            >>> labels = {"cls": np.array([[0], [1], [2]]), "texts": [["dog"], ["cat"], ["bird"]]}
            >>> updated_labels = loader(labels)
        """
        assert "texts" in labels, "labels 中未找到 texts。"
        class_texts = labels["texts"]
        num_classes = len(class_texts)
        cls = np.asarray(labels.pop("cls"), dtype=int)
        pos_labels = np.unique(cls).tolist()
        if len(pos_labels) > self.max_samples:
            pos_labels = random.sample(pos_labels, k=self.max_samples)
        neg_samples = min(
            min(num_classes, self.max_samples) - len(pos_labels),
            random.randint(*self.neg_samples),
        )
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]
        neg_labels = random.sample(neg_labels, k=neg_samples)
        sampled_labels = pos_labels + neg_labels
        label2ids = {label: i for i, label in enumerate(sampled_labels)}
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
        new_cls = []
        for i, label in enumerate(cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])
        labels["instances"] = labels["instances"][valid_idx]
        labels["cls"] = np.array(new_cls)
        texts = []
        for label in sampled_labels:
            prompts = class_texts[label]
            assert len(prompts) > 0
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
            texts.append(prompt)
        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            if num_padding > 0:
                texts += random.choices(self.padding_value, k=num_padding)
        assert len(texts) == self.max_samples
        labels["texts"] = texts
        return labels


def v8_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace, stretch: bool = False):
    """为 training 应用一系列 image transforms。

    该函数创建 image augmentation 技术组合，为 YOLO training 准备 images。包含 mosaic、copy-paste、
    random perspective、mixup 与各类 color adjustments。

    参数:
        dataset (Dataset): 包含 image data 与 annotations 的 dataset 对象。
        imgsz (int): resize 使用的 target image size。
        hyp (IterableSimpleNamespace): 控制 transforms 各方面的 hyperparameters namespace。
        stretch (bool): 若为 True，对 image 应用 stretching；若为 False，使用 LetterBox resizing。

    返回:
        (Compose): 将应用到 dataset 的 image transforms 组合。

    示例:
        >>> from ddyolo26.data.dataset import YOLODataset
        >>> from ddyolo26.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = IterableSimpleNamespace(mosaic=1.0, copy_paste=0.5, degrees=10.0, translate=0.2, scale=0.9)
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])

        >>> # 使用自定义 albumentations
        >>> import albumentations as A
        >>> augmentations = [A.Blur(p=0.01), A.CLAHE(p=0.01)]
        >>> hyp.augmentations = augmentations
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )
    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0
            LOGGER.warning("data.yaml 中未定义 'flip_idx' array，已禁用 'fliplr' 与 'flipud' augmentations。")
        elif flip_idx and len(flip_idx) != kpt_shape[0]:
            raise ValueError(f"data.yaml flip_idx={flip_idx} 长度必须等于 kpt_shape[0]={kpt_shape[0]}")
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None)),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )


def classify_transforms(
    size: (tuple[int, int] | int) = 224,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    interpolation: str = "BILINEAR",
    crop_fraction: (float | None) = None,
):
    """Paddle-only detect/segment build 不支持 classification transforms。"""
    raise NotImplementedError("当前 PaddleYOLO-RKNN build 不支持 classification transforms。")


def classify_augmentations(
    size: int = 224,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    scale: (tuple[float, float] | None) = None,
    ratio: (tuple[float, float] | None) = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    auto_augment: (str | None) = None,
    hsv_h: float = 0.015,
    hsv_s: float = 0.4,
    hsv_v: float = 0.4,
    force_color_jitter: bool = False,
    erasing: float = 0.0,
    interpolation: str = "BILINEAR",
):
    """Paddle-only detect/segment build 不支持 classification augmentations。"""
    raise NotImplementedError("当前 PaddleYOLO-RKNN build 不支持 classification augmentations。")


class ClassifyLetterBox:
    """用于 classification 任务 resize 与 padding images 的类。

    该类设计为 transformation pipeline 的一部分，例如 T.Compose([LetterBox(size), ToTensor()])。
    它在保持 original aspect ratio 的同时，将 images resize 并 pad 到指定 size。

    属性:
        h (int): image 的 target height。
        w (int): image 的 target width。
        auto (bool): 若为 True，使用 stride 自动计算 short side。
        stride (int): 'auto' 为 True 时使用的 stride 值。

    方法:
        __call__: 对输入 image 应用 letterbox transform。

    示例:
        >>> transform = ClassifyLetterBox(size=(640, 640), auto=False, stride=32)
        >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> result = transform(img)
        >>> print(result.shape)
        (640, 640, 3)
    """

    def __init__(
        self,
        size: (int | tuple[int, int]) = (640, 640),
        auto: bool = False,
        stride: int = 32,
    ):
        """初始化用于 image preprocessing 的 ClassifyLetterBox 对象。

        该类设计为 image classification 任务 transformation pipeline 的一部分，在保持 original aspect ratio
        的同时将 images resize 并 pad 到指定 size。

        参数:
            size (int | tuple[int, int]): letterboxed image 的 target size。若为 int，创建 (size, size) 的
                square image；若为 tuple，应为 (height, width)。
            auto (bool): 若为 True，基于 stride 自动计算 short side。
            stride (int): 'auto' 为 True 时使用的 stride 值。
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto
        self.stride = stride

    def __call__(self, im: np.ndarray) -> np.ndarray:
        """使用 letterbox method resize 并 pad image。

        该方法在保持 aspect ratio 的同时 resize 输入 image，使其适配指定 dimensions，然后 pad resized image
        以匹配 target size。

        参数:
            im (np.ndarray): numpy array 形式的输入 image，shape 为 (H, W, C)。

        返回:
            (np.ndarray): resize 并 padding 后的 image，numpy array，shape 为 (hs, ws, 3)，其中 hs 与 ws
                分别是 target height 与 width。

        示例:
            >>> letterbox = ClassifyLetterBox(size=(640, 640))
            >>> image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            >>> resized_image = letterbox(image)
            >>> print(resized_image.shape)
            (640, 640, 3)
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)
        h, w = round(imh * r), round(imw * r)
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    """为 classification 任务对 images 应用 center cropping。

    该类对输入 images 执行 center cropping，并在保持 aspect ratio 的同时将其 resize 到指定 size。
    它设计为 transformation pipeline 的一部分，例如 T.Compose([CenterCrop(size), ToTensor()])。

    属性:
        h (int): cropped image 的 target height。
        w (int): cropped image 的 target width。

    方法:
        __call__: 对输入 image 应用 center crop transform。

    示例:
        >>> transform = CenterCrop(640)
        >>> image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        >>> cropped_image = transform(image)
        >>> print(cropped_image.shape)
        (640, 640, 3)
    """

    def __init__(self, size: (int | tuple[int, int]) = (640, 640)):
        """初始化用于 image preprocessing 的 CenterCrop 对象。

        该类设计为 transformation pipeline 的一部分，例如 T.Compose([CenterCrop(size), ToTensor()])。
        它会对输入 images 执行 center crop 到指定 size。

        参数:
            size (int | tuple[int, int]): crop 的期望 output size。若 size 为 int，则生成 (size, size) 的
                square crop；若为 (h, w) 这样的序列，则作为 output size。
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im: (Image.Image | np.ndarray)) -> np.ndarray:
        """对输入 image 应用 center cropping。

        该方法从 image 中裁出最大的 centered square，并 resize 到指定 dimensions。

        参数:
            im (np.ndarray | PIL.Image.Image): 输入 image，可为 shape (H, W, C) 的 numpy array 或 PIL Image 对象。

        返回:
            (np.ndarray): center-cropped 并 resized 后的 image，numpy array，shape 为 (self.h, self.w, C)。

        示例:
            >>> transform = CenterCrop(size=224)
            >>> image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            >>> cropped_image = transform(image)
            >>> assert cropped_image.shape == (224, 224, 3)
        """
        if isinstance(im, Image.Image):
            im = np.asarray(im)
        imh, imw = im.shape[:2]
        m = min(imh, imw)
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(
            im[top : top + m, left : left + m],
            (self.w, self.h),
            interpolation=cv2.INTER_LINEAR,
        )


class ToTensor:
    """将 image 从 numpy array 转为 Paddle tensor。

    该类设计为 transformation pipeline 的一部分，例如 T.Compose([LetterBox(size), ToTensor()])。

    属性:
        half (bool): 若为 True，将 image 转为 half precision（float16）。

    方法:
        __call__: 对输入 image 应用 tensor conversion。

    示例:
        >>> transform = ToTensor(half=True)
        >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> tensor_img = transform(img)
        >>> print(tensor_img.shape, tensor_img.dtype)
        Shape([3, 640, 640]) paddle.float16

    说明:
        输入 image 预期为 BGR format，shape 为 (H, W, C)。
        输出 tensor 为 BGR format，shape 为 (C, H, W)，并 normalize 到 [0, 1]。
    """

    def __init__(self, half: bool = False):
        """初始化用于将 images 转为 Paddle tensors 的 ToTensor 对象。

        该类用于 PaddleYOLO-RKNN framework 中 image preprocessing 的 transformation pipeline。
        它将 numpy arrays 或 PIL Images 转为 Paddle tensors，并可选择 half-precision（float16）conversion。

        参数:
            half (bool): 若为 True，将 tensor 转为 half precision（float16）。
        """
        super().__init__()
        self.half = half

    def __call__(self, im: np.ndarray) -> paddle.Tensor:
        """将 image 从 numpy array transform 为 Paddle tensor。

        该方法将输入 image 从 numpy array 转为 Paddle tensor，并应用可选 half-precision conversion 与
        normalization。image 会从 HWC format 转置为 CHW format。

        参数:
            im (np.ndarray): numpy array 形式的输入 image，shape 为 (H, W, C)，BGR order。

        返回:
            (paddle.Tensor): transform 后的 image，Paddle tensor，float32 或 float16，normalize 到 [0, 1]，
                shape 为 (C, H, W)，BGR order。

        示例:
            >>> transform = ToTensor(half=True)
            >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.shape, tensor_img.dtype)
            Shape([3, 640, 640]) paddle.float16
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1)))
        im = paddle.from_numpy(im)
        im = im.half() if self.half else im.float()
        im /= 255.0
        return im
