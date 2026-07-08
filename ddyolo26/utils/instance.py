# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 实例标注容器 Instances：边界框+分割+关键点的统一封装。
@details
训练数据管道中的核心数据结构，存储每张图片的所有目标实例标注，
提供统一的格式转换、透视变换、裁剪、翻转等操作接口，
确保 bbox/mask/keypoints 在各增强变换后保持空间一致性。
"""

from __future__ import annotations
import paddle

from collections import abc
from itertools import repeat
from numbers import Number

import numpy as np

from .ops import ltwh2xywh, ltwh2xyxy, resample_segments, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh


def _ntuple(n):
    """创建将输入转为 n-tuple 的函数；单值会重复 n 次。"""

    def parse(x):
        """解析输入并返回 n-tuple；单值会重复 n 次。"""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)
_formats = ["xyxy", "xywh", "ltwh"]
__all__ = "Bboxes", "Instances"


class Bboxes:
    """用于处理多种 formats bounding boxes 的类。

    该类支持 'xyxy'、'xywh'、'ltwh' 等 bbox formats，并提供 format conversion、scaling 与 area calculation
    方法。bounding box data 应以 numpy arrays 提供。

    属性:
        bboxes (np.ndarray): 存储在 shape 为 (N, 4) 的 2D numpy array 中的 bounding boxes。
        format (str): bounding boxes 的 format（'xyxy'、'xywh' 或 'ltwh'）。

    方法:
        convert: 将 bounding box format 从一种类型转换为另一种。
        areas: 计算 bounding boxes 的 area。
        mul: 将 bounding box coordinates 乘以 scale factor(s)。
        add: 为 bounding box coordinates 添加 offset。
        concatenate: 拼接多个 Bboxes objects。

    示例:
        创建 YOLO format 的 bounding boxes
        >>> bboxes = Bboxes(np.array([[100, 50, 150, 100]]), format="xywh")
        >>> bboxes.convert("xyxy")
        >>> print(bboxes.areas())

    说明:
        该类不处理 bounding boxes 的 normalization 或 denormalization。
    """

    def __init__(self, bboxes: np.ndarray, format: str = "xyxy") -> None:
        """使用指定 format 的 bounding box data 初始化 Bboxes 类。

        参数:
            bboxes (np.ndarray): shape 为 (N, 4) 或 (4,) 的 bounding boxes array。
            format (str): bounding boxes 的 format，可为 'xyxy'、'xywh' 或 'ltwh'。
        """
        assert format in _formats, f"无效 bounding box format: {format}，format 必须为 {_formats} 之一"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format

    def convert(self, format: str) -> None:
        """将 bounding box format 从一种类型转换为另一种。

        参数:
            format (str): conversion 的 target format，可为 'xyxy'、'xywh' 或 'ltwh'。
        """
        assert format in _formats, f"无效 bounding box format: {format}，format 必须为 {_formats} 之一"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self) -> np.ndarray:
        """计算 bounding boxes 的 area。"""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]
        )

    def mul(self, scale: (int | tuple | list)) -> None:
        """将 bounding box coordinates 乘以 scale factor(s)。

        参数:
            scale (int | tuple | list): 四个 coordinates 的 scale factor(s)。若为 int，则所有 coordinates 使用同一 scale。
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset: (int | tuple | list)) -> None:
        """为 bounding box coordinates 添加 offset。

        参数:
            offset (int | tuple | list): 四个 coordinates 的 offset(s)。若为 int，则所有 coordinates 使用同一 offset。
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self) -> int:
        """返回 bounding boxes 数量。"""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: list[Bboxes], axis: int = 0) -> Bboxes:
        """将 Bboxes objects 列表拼接为单个 Bboxes object。

        参数:
            boxes_list (list[Bboxes]): 要 concatenate 的 Bboxes objects 列表。
            axis (int, optional): concatenate bounding boxes 使用的 axis。

        返回:
            (Bboxes): 包含拼接后 bounding boxes 的新 Bboxes object。

        说明:
            输入应为 Bboxes objects 的 list 或 tuple。
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty((0, 4)))
        assert all(isinstance(box, Bboxes) for box in boxes_list)
        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index: (int | np.ndarray | slice)) -> Bboxes:
        """使用 indexing 获取指定 bounding box 或一组 bounding boxes。

        参数:
            index (int | slice | np.ndarray): 用于选择目标 bounding boxes 的 index、slice 或 boolean array。

        返回:
            (Bboxes): 包含已选择 bounding boxes 的新 Bboxes object。

        说明:
            使用 boolean indexing 时，请确保提供与 bounding boxes 数量相同长度的 boolean array。
        """
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].reshape(1, -1))
        b = self.bboxes[index]
        assert b.ndim == 2, f"使用 {index} 对 Bboxes indexing 未能返回 matrix！"
        return Bboxes(b)


class Instances:
    """image 中 detected objects 的 bounding boxes、segments 与 keypoints 容器。

    该类为 bounding boxes、segmentation masks、keypoints 等不同 object annotations 提供统一接口。
    支持 scaling、normalization、clipping 与 format conversion 等操作。

    属性:
        _bboxes (Bboxes): 处理 bounding box operations 的内部对象。
        keypoints (np.ndarray): shape 为 (N, 17, 3)、format 为 (x, y, visible) 的 keypoints。
        normalized (bool): 表示 bounding box coordinates 是否 normalized 的 flag。
        segments (np.ndarray): resampling 后 shape 为 (N, M, 2) 的 segments array。

    方法:
        convert_bbox: 转换 bounding box format。
        scale: 按给定 factors 缩放 coordinates。
        denormalize: 将 normalized coordinates 转为 absolute coordinates。
        normalize: 将 absolute coordinates 转为 normalized coordinates。
        add_padding: 为 coordinates 添加 padding。
        flipud: 垂直 flip coordinates。
        fliplr: 水平 flip coordinates。
        clip: clip coordinates，使其位于 image boundaries 内。
        remove_zero_area_boxes: 移除 zero-area boxes。
        update: 更新 instance variables。
        concatenate: 拼接多个 Instances objects。

    示例:
        创建带 bounding boxes 与 segments 的 instances
        >>> instances = Instances(
        ...     bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
        ...     segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
        ...     keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]]),
        ... )
    """

    def __init__(
        self,
        bboxes: np.ndarray,
        segments: np.ndarray = None,
        keypoints: np.ndarray = None,
        bbox_format: str = "xywh",
        normalized: bool = True,
    ) -> None:
        """使用 bounding boxes、segments 与 keypoints 初始化 Instances object。

        参数:
            bboxes (np.ndarray): shape 为 (N, 4) 的 bounding boxes。
            segments (np.ndarray, optional): segmentation masks。
            keypoints (np.ndarray, optional): shape 为 (N, 17, 3)、format 为 (x, y, visible) 的 keypoints。
            bbox_format (str): bboxes 的 format。
            normalized (bool): coordinates 是否 normalized。
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments

    def convert_bbox(self, format: str) -> None:
        """转换 bounding box format。

        参数:
            format (str): conversion 的 target format，可为 'xyxy'、'xywh' 或 'ltwh'。
        """
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self) -> np.ndarray:
        """计算 bounding boxes 的 area。"""
        return self._bboxes.areas()

    def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False):
        """按给定 factors 缩放 coordinates。

        参数:
            scale_w (float): width 的 scale factor。
            scale_h (float): height 的 scale factor。
            bbox_only (bool, optional): 是否只 scale bounding boxes。
        """
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        if bbox_only:
            return
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w: int, h: int) -> None:
        """将 normalized coordinates 转为 absolute coordinates。

        参数:
            w (int): image width。
            h (int): image height。
        """
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        self.segments[..., 0] *= w
        self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w: int, h: int) -> None:
        """将 absolute coordinates 转为 normalized coordinates。

        参数:
            w (int): image width。
            h (int): image height。
        """
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        self.segments[..., 0] /= w
        self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw: int, padh: int) -> None:
        """为 coordinates 添加 padding。

        参数:
            padw (int): padding width。
            padh (int): padding height。
        """
        assert not self.normalized, "应在 absolute coordinates 下添加 padding。"
        self._bboxes.add(offset=(padw, padh, padw, padh))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index: (int | np.ndarray | slice)) -> Instances:
        """使用 indexing 获取指定 instance 或一组 instances。

        参数:
            index (int | slice | np.ndarray): 用于选择目标 instances 的 index、slice 或 boolean array。

        返回:
            (Instances): 包含已选择 boxes、segments 以及可选 keypoints 的新 Instances object。

        说明:
            使用 boolean indexing 时，请确保提供与 instances 数量相同长度的 boolean array。
        """
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def flipud(self, h: int) -> None:
        """垂直 flip coordinates。

        参数:
            h (int): image height。
        """
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w: int) -> None:
        """水平 flip coordinates。

        参数:
            w (int): image width。
        """
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w: int, h: int) -> None:
        """clip coordinates，使其位于 image boundaries 内。

        参数:
            w (int): image width。
            h (int): image height。
        """
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)
        self.segments[..., 0] = self.segments[..., 0].clip(0, w)
        self.segments[..., 1] = self.segments[..., 1].clip(0, h)
        if self.keypoints is not None:
            self.keypoints[..., 2][
                (self.keypoints[..., 0] < 0)
                | (self.keypoints[..., 0] > w)
                | (self.keypoints[..., 1] < 0)
                | (self.keypoints[..., 1] > h)
            ] = 0.0
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)

    def remove_zero_area_boxes(self) -> np.ndarray:
        """移除 zero-area boxes，例如 clipping 后 width 或 height 为零的 boxes。

        返回:
            (np.ndarray): boolean array，表示哪些 boxes 被保留。
        """
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            if self.segments is not None and len(self.segments):
                self.segments = self.segments[good]
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        return good

    def update(
        self,
        bboxes: np.ndarray,
        segments: np.ndarray = None,
        keypoints: np.ndarray = None,
    ):
        """更新 instance variables。

        参数:
            bboxes (np.ndarray): 新 bounding boxes。
            segments (np.ndarray, optional): 新 segments。
            keypoints (np.ndarray, optional): 新 keypoints。
        """
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self) -> int:
        """返回 instances 数量。"""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: list[Instances], axis=0) -> Instances:
        """将 Instances objects 列表拼接为单个 Instances object。

        参数:
            instances_list (list[Instances]): 要 concatenate 的 Instances objects 列表。
            axis (int, optional): arrays concatenate 使用的 axis。

        返回:
            (Instances): 包含拼接后 bounding boxes、segments 以及可选 keypoints 的新 Instances object。

        说明:
            列表中的 `Instances` objects 应具有相同属性，例如 bounding boxes 的 format、是否包含 keypoints，
            以及 coordinates 是否 normalized。
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty((0, 4)))
        assert all(isinstance(instance, Instances) for instance in instances_list)
        if len(instances_list) == 1:
            return instances_list[0]
        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized
        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        seg_len = [b.segments.shape[1] for b in instances_list]
        if len(frozenset(seg_len)) > 1:
            max_len = max(seg_len)
            cat_segments = np.concatenate(
                [
                    (
                        resample_segments(list(b.segments), max_len)
                        if len(b.segments)
                        else np.zeros((0, max_len, 2), dtype=np.float32)
                    )
                    for b in instances_list
                ],
                axis=axis,
            )
        else:
            cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)

    @property
    def bboxes(self) -> np.ndarray:
        """返回 bounding boxes。"""
        return self._bboxes.bboxes

    def __repr__(self) -> str:
        """返回 Instances object 的字符串表示。"""
        attr_map = {"_bboxes": "bboxes"}
        parts = []
        for key, value in self.__dict__.items():
            name = attr_map.get(key, key)
            if name == "bboxes":
                value = self.bboxes
            if value is not None:
                parts.append(f"{name}={value!r}")
        return "Instances({})".format("\n".join(parts))
