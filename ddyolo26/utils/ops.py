# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 几何运算工具：坐标格式转换、NMS、裁剪等操作。
@details
常用函数：
- `xywh2xyxy` / `xyxy2xywh` / `xyxyn2xywhn`：坐标格式互转
- `non_max_suppression()`：多类 NMS 实现（one-to-many head 推理用）
- `scale_boxes()`：坐标从推理尺寸反变换回原图尺寸（补偿 LetterBox）
- `crop_mask()`：裁剪实例分割 mask
- `letterbox()`：图像等比缩放+填充
"""

from __future__ import annotations
import sys
import paddle


import contextlib
import math
import re
import time

import cv2
import numpy as np

from ddyolo26.paddle_utils import *

from ddyolo26.utils import NOT_MACOS14


class Profile(contextlib.ContextDecorator):
    """用于统计 code execution 耗时的 PaddleYOLO-RKNN profile 类。

    可通过 @Profile() 作为 decorator 使用，也可通过 'with Profile():' 作为 context manager 使用。
    支持 CUDA synchronization，可为 GPU operations 提供更准确的 timing measurements。

    属性:
        t (float): 累积耗时，单位 seconds。
        device (str): model inference 使用的 device。
        cuda (bool): 是否使用 CUDA 进行 timing synchronization。

    示例:
        作为 context manager 统计 code execution 耗时
        >>> with Profile(device=device) as dt:
        ...     pass  # slow operation
        >>> print(dt)  # 打印 "Elapsed time is 9.5367431640625e-07 s"

        作为 decorator 统计 function execution 耗时
        >>> @Profile()
        ... def slow_function():
        ...     time.sleep(0.1)
    """

    def __init__(self, t: float = 0.0, device: (paddle.device | None) = None):
        """初始化 Profile 类。

        参数:
            t (float): 初始累计耗时，单位 seconds。
            device (str, optional): model inference 使用的 device，用于启用 CUDA synchronization。
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """开始计时。"""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """停止计时。"""
        self.dt = self.time() - self.start
        self.t += self.dt

    def __str__(self):
        """返回表示累计 elapsed time 的 human-readable string。"""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """获取当前时间；适用时执行 CUDA synchronization。"""
        if self.cuda:
            paddle.cuda.synchronize(self.device)
        return time.perf_counter()


def segment2box(segment, width: int = 640, height: int = 640):
    """将 segment coordinates 转为 bounding box coordinates。

    通过查找 x/y coordinates 的最小值与最大值，将单个 segment label 转为 box label。
    必要时应用 inside-image 约束并 clip coordinates。

    参数:
        segment (np.ndarray): format 为 (N, 2) 的 segment coordinates，其中 N 为 points 数量。
        width (int): image width，单位 pixels。
        height (int): image height，单位 pixels。

    返回:
        (np.ndarray): xyxy format 的 bounding box coordinates [x1, y1, x2, y2]。
    """
    x, y = segment.T
    if np.array([x.min() < 0, y.min() < 0, x.max() > width, y.max() > height]).sum() >= 3:
        x = x.clip(0, width)
        y = y.clip(0, height)
    inside = (x > 0) & (y > 0) & (x < width) & (y < height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )


def scale_boxes(
    img1_shape,
    boxes,
    img0_shape,
    ratio_pad=None,
    padding: bool = True,
    xywh: bool = False,
):
    """将 bounding boxes 从一种 image shape rescale 到另一种。

    将 bounding boxes 从 img1_shape rescale 到 img0_shape，并补偿 padding 与 aspect ratio changes。
    支持 xyxy 与 xywh 两种 box formats。

    参数:
        img1_shape (tuple): source image 的 shape（height, width）。
        boxes (paddle.Tensor): 待 rescale 的 bounding boxes，format 为 (N, 4)。
        img0_shape (tuple): target image 的 shape（height, width）。
        ratio_pad (tuple, optional): scaling 使用的 (ratio, pad) tuple。若为 None，则由 image shapes 计算。
        padding (bool): boxes 是否基于带 padding 的 YOLO-style augmented images。
        xywh (bool): box format 是否为 xywh（True）或 xyxy（False）。

    返回:
        (paddle.Tensor): rescaled bounding boxes，format 与输入相同。
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]
    if padding:
        boxes = boxes.clone()
        boxes[..., 0] = boxes[..., 0] - pad_x
        boxes[..., 1] = boxes[..., 1] - pad_y
        if not xywh:
            boxes[..., 2] = boxes[..., 2] - pad_x
            boxes[..., 3] = boxes[..., 3] - pad_y
    boxes[..., :4] = boxes[..., :4] / gain
    return boxes if xywh else clip_boxes(boxes, img0_shape)


def make_divisible(x: int, divisor):
    """返回可被给定 divisor 整除的最近数值。

    参数:
        x (int): 待调整为可整除的数值。
        divisor (int | paddle.Tensor): divisor。

    返回:
        (int): 可被 divisor 整除的最近数值。
    """
    if isinstance(divisor, paddle.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def clip_boxes(boxes, shape):
    """将 bounding boxes clip 到 image boundaries 内。

    参数:
        boxes (paddle.Tensor | np.ndarray): 待 clip 的 bounding boxes。
        shape (tuple): HWC 或 HW 格式的 image shape（均支持）。

    返回:
        (paddle.Tensor | np.ndarray): clipped bounding boxes。
    """
    h, w = shape[:2]
    if isinstance(boxes, paddle.Tensor):
        if boxes.numel() == 0:
            return boxes
        if NOT_MACOS14:
            boxes[..., 0].clamp_(0, w)
            boxes[..., 1].clamp_(0, h)
            boxes[..., 2].clamp_(0, w)
            boxes[..., 3].clamp_(0, h)
        else:
            boxes[..., 0] = boxes[..., 0].clamp(0, w)
            boxes[..., 1] = boxes[..., 1].clamp(0, h)
            boxes[..., 2] = boxes[..., 2].clamp(0, w)
            boxes[..., 3] = boxes[..., 3].clamp(0, h)
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)
    return boxes


def clip_coords(coords, shape):
    """将 line coordinates clip 到 image boundaries 内。

    参数:
        coords (paddle.Tensor | np.ndarray): 待 clip 的 line coordinates。
        shape (tuple): HWC 或 HW 格式的 image shape（均支持）。

    返回:
        (paddle.Tensor | np.ndarray): clipped coordinates。
    """
    h, w = shape[:2]
    if isinstance(coords, paddle.Tensor):
        if NOT_MACOS14:
            coords[..., 0].clamp_(0, w)
            coords[..., 1].clamp_(0, h)
        else:
            coords[..., 0] = coords[..., 0].clamp(0, w)
            coords[..., 1] = coords[..., 1].clamp(0, h)
    else:
        coords[..., 0] = coords[..., 0].clip(0, w)
        coords[..., 1] = coords[..., 1].clip(0, h)
    return coords


def xyxy2xywh(x):
    """将 bounding box coordinates 从 (x1, y1, x2, y2) format 转为 (x, y, width, height) format。

    其中 (x1, y1) 为 top-left corner，(x2, y2) 为 bottom-right corner。

    参数:
        x (np.ndarray | paddle.Tensor): (x1, y1, x2, y2) format 的输入 bounding box coordinates。

    返回:
        (np.ndarray | paddle.Tensor): (x, y, width, height) format 的 bounding box coordinates。
    """
    assert x.shape[-1] == 4, f"input shape 最后一维预期为 4，但 input shape 为 {x.shape}"
    if str(x.dtype) in ("paddle.int64", "paddle.int32", "int64", "int32"):
        x = x.astype("float32")
    y = empty_like(x)
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2
    y[..., 1] = (y1 + y2) / 2
    y[..., 2] = x2 - x1
    y[..., 3] = y2 - y1
    return y


def xywh2xyxy(x):
    """将 bounding box coordinates 从 (x, y, width, height) format 转为 (x1, y1, x2, y2) format。

    其中 (x1, y1) 为 top-left corner，(x2, y2) 为 bottom-right corner。说明：每 2 channels 执行 ops 比逐 channel 更快。

    参数:
        x (np.ndarray | paddle.Tensor): (x, y, width, height) format 的输入 bounding box coordinates。

    返回:
        (np.ndarray | paddle.Tensor): (x1, y1, x2, y2) format 的 bounding box coordinates。
    """
    assert x.shape[-1] == 4, f"input shape 最后一维预期为 4，但 input shape 为 {x.shape}"
    if str(x.dtype) in ("paddle.int64", "paddle.int32", "int64", "int32"):
        x = x.astype("float32")
    y = empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def xywhn2xyxy(x, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0):
    """将 normalized bounding box coordinates 转为 pixel coordinates。

    参数:
        x (np.ndarray | paddle.Tensor): (x, y, w, h) format 的 normalized bounding box coordinates。
        w (int): image width，单位 pixels。
        h (int): image height，单位 pixels。
        padw (int): padding width，单位 pixels。
        padh (int): padding height，单位 pixels。

    返回:
        (np.ndarray | paddle.Tensor): (x1, y1, x2, y2) format 的 bounding box coordinates。
    """
    assert x.shape[-1] == 4, f"input shape 最后一维预期为 4，但 input shape 为 {x.shape}"
    y = empty_like(x)
    xc, yc, xw, xh = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    half_w, half_h = xw / 2, xh / 2
    y[..., 0] = w * (xc - half_w) + padw
    y[..., 1] = h * (yc - half_h) + padh
    y[..., 2] = w * (xc + half_w) + padw
    y[..., 3] = h * (yc + half_h) + padh
    return y


def xyxy2xywhn(x, w: int = 640, h: int = 640, clip: bool = False, eps: float = 0.0):
    """将 bbox coordinates 从 (x1, y1, x2, y2) format 转为 normalized (x, y, width, height) format。

    x、y、width 与 height 会按 image dimensions normalize。

    参数:
        x (np.ndarray | paddle.Tensor): (x1, y1, x2, y2) format 的输入 bounding box coordinates。
        w (int): image width，单位 pixels。
        h (int): image height，单位 pixels。
        clip (bool): 是否将 boxes clip 到 image boundaries。
        eps (float): box width 与 height 的最小值。

    返回:
        (np.ndarray | paddle.Tensor): (x, y, width, height) format 的 normalized bounding box coordinates。
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape 最后一维预期为 4，但 input shape 为 {x.shape}"
    y = empty_like(x)
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2 / w
    y[..., 1] = (y1 + y2) / 2 / h
    y[..., 2] = (x2 - x1) / w
    y[..., 3] = (y2 - y1) / h
    return y


def xywh2ltwh(x):
    """将 bounding box format 从 [x, y, w, h] 转为 [x1, y1, w, h]，其中 x1、y1 为 top-left coordinates。

    参数:
        x (np.ndarray | paddle.Tensor): xywh format 的输入 bounding box coordinates。

    返回:
        (np.ndarray | paddle.Tensor): ltwh format 的 bounding box coordinates。
    """
    y = x.clone() if isinstance(x, paddle.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    return y


def xyxy2ltwh(x):
    """将 bounding boxes 从 [x1, y1, x2, y2] 转为 [x1, y1, w, h] format。

    参数:
        x (np.ndarray | paddle.Tensor): xyxy format 的输入 bounding box coordinates。

    返回:
        (np.ndarray | paddle.Tensor): ltwh format 的 bounding box coordinates。
    """
    y = x.clone() if isinstance(x, paddle.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def ltwh2xywh(x):
    """将 bounding boxes 从 [x1, y1, w, h] 转为 [x, y, w, h]，其中 xy1=top-left，xy=center。

    参数:
        x (np.ndarray | paddle.Tensor): 输入 bounding box coordinates。

    返回:
        (np.ndarray | paddle.Tensor): xywh format 的 bounding box coordinates。
    """
    y = x.clone() if isinstance(x, paddle.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2
    y[..., 1] = x[..., 1] + x[..., 3] / 2
    return y


def xyxyxyxy2xywhr(x):
    """将 batched Oriented Bounding Boxes（OBB）从 [xy1, xy2, xy3, xy4] 转为 [xywh, rotation] format。

    参数:
        x (np.ndarray | paddle.Tensor): [xy1, xy2, xy3, xy4] format 的 input box corners，shape 为 (N, 8)。

    返回:
        (np.ndarray | paddle.Tensor): [cx, cy, w, h, rotation] format 的 converted data，shape 为 (N, 5)。
            rotation values 使用 radians，范围为 [-pi/4, 3pi/4)。
    """
    is_paddle = isinstance(x, paddle.Tensor)
    points = x.cpu().numpy() if is_paddle else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        theta = angle / 180 * np.pi
        if w < h:
            w, h = h, w
            theta += np.pi / 2
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi
        rboxes.append([cx, cy, w, h, theta])
    return paddle.tensor(rboxes, device=x.device, dtype=x.dtype) if is_paddle else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """将 batched Oriented Bounding Boxes（OBB）从 [xywh, rotation] 转为 [xy1, xy2, xy3, xy4] format。

    参数:
        x (np.ndarray | paddle.Tensor): [cx, cy, w, h, rotation] format 的 boxes，shape 为 (N, 5) 或 (B, N, 5)。
            rotation values 应使用 radians，范围为 [-pi/4, 3pi/4)。

    返回:
        (np.ndarray | paddle.Tensor): converted corner points，shape 为 (N, 4, 2) 或 (B, N, 4, 2)。
    """
    cos, sin, cat, stack = (
        (paddle.cos, paddle.sin, paddle.cat, paddle.stack)
        if isinstance(x, paddle.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )
    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    """将 bounding box 从 [x1, y1, w, h] 转为 [x1, y1, x2, y2]，其中 xy1=top-left，xy2=bottom-right。

    参数:
        x (np.ndarray | paddle.Tensor): 输入 bounding box coordinates。

    返回:
        (np.ndarray | paddle.Tensor): xyxy format 的 bounding box coordinates。
    """
    y = x.clone() if isinstance(x, paddle.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]
    y[..., 3] = x[..., 3] + x[..., 1]
    return y


def segments2boxes(segments):
    """将 segment coordinates 转为 xywh format 的 bounding box labels。

    参数:
        segments (list): segments 列表，每个 segment 是 points 列表，每个 point 为 [x, y] coordinates。

    返回:
        (np.ndarray): xywh format 的 bounding box coordinates。
    """
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))


def resample_segments(segments, n: int = 1000):
    """使用 linear interpolation 将每个 segment resample 到 n 个 points。

    参数:
        segments (list): (N, 2) arrays 列表，其中 N 为每个 segment 的 points 数量。
        n (int): 每个 segment resample 后的 points 数量。

    返回:
        (list): resampled segments，每个包含 n 个 points。
    """
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
    return segments


def crop_mask(masks: paddle.Tensor, boxes: paddle.Tensor) -> paddle.Tensor:
    """将 masks crop 到 bounding box regions。

    参数:
        masks (paddle.Tensor): shape 为 (N, H, W) 的 masks。
        boxes (paddle.Tensor): xyxy pixel format、shape 为 (N, 4) 的 bounding box coordinates。

    返回:
        (paddle.Tensor): cropped masks。
    """
    n, h, w = masks.shape
    if n < 50 and not paddle.is_compiled_with_cuda():
        for i, (x1, y1, x2, y2) in enumerate(boxes.round().int()):
            masks[i, :y1] = 0
            masks[i, y2:] = 0
            masks[i, :, :x1] = 0
            masks[i, :, x2:] = 0
        return masks
    else:
        x1, y1, x2, y2 = paddle.chunk(boxes[:, :, None], 4, 1)
        r = paddle.arange(w, dtype=x1.dtype)[None, None, :]
        c = paddle.arange(h, dtype=x1.dtype)[None, :, None]
        # Paddle 不支持 float * bool 自动类型提升，需显式转换
        crop = ((r >= x1) * (r < x2) * (c >= y1) * (c < y2)).astype(masks.dtype)
        return masks * crop


def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    """使用 mask head output 将 masks 应用到 bounding boxes。

    参数:
        protos (paddle.Tensor): shape 为 (mask_dim, mask_h, mask_w) 的 mask prototypes。
        masks_in (paddle.Tensor): shape 为 (N, mask_dim) 的 mask coefficients，其中 N 是 NMS 后 masks 数量。
        bboxes (paddle.Tensor): shape 为 (N, 4) 的 bounding boxes，其中 N 是 NMS 后 masks 数量。
        shape (tuple): (height, width) 格式的 input image size。
        upsample (bool): 是否将 masks upsample 到 original image size。

    返回:
        (paddle.Tensor): shape 为 [n, h, w] 的 binary mask tensor，其中 n 是 NMS 后 masks 数量，h/w 为 input image
            的 height/width。mask 会应用到 bounding boxes。
    """
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.astype("float32").reshape([c, -1])).reshape([-1, mh, mw])
    width_ratio = mw / shape[1]
    height_ratio = mh / shape[0]
    ratios = paddle.to_tensor([[width_ratio, height_ratio, width_ratio, height_ratio]])
    masks = crop_mask(masks, boxes=bboxes * ratios)
    if upsample:
        masks = paddle.nn.functional.interpolate(masks[None], shape, mode="bilinear")[0]
    return (masks > 0).astype("uint8")


def process_mask_native(protos, masks_in, bboxes, shape):
    """使用 mask head output 与 native upsampling 将 masks 应用到 bounding boxes。

    参数:
        protos (paddle.Tensor): shape 为 (mask_dim, mask_h, mask_w) 的 mask prototypes。
        masks_in (paddle.Tensor): shape 为 (N, mask_dim) 的 mask coefficients，其中 N 是 NMS 后 masks 数量。
        bboxes (paddle.Tensor): shape 为 (N, 4) 的 bounding boxes，其中 N 是 NMS 后 masks 数量。
        shape (tuple): (height, width) 格式的 input image size。

    返回:
        (paddle.Tensor): shape 为 (N, H, W) 的 binary mask tensor。
    """
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.astype("float32").reshape([c, -1])).reshape([-1, mh, mw])
    masks = scale_masks(masks[None], shape)[0]
    masks = crop_mask(masks, bboxes)
    return (masks > 0).astype("uint8")


def scale_masks(
    masks: paddle.Tensor,
    shape: tuple[int, int],
    ratio_pad: (tuple[tuple[int, int], tuple[int, int]] | None) = None,
    padding: bool = True,
) -> paddle.Tensor:
    """将 segment masks rescale 到 target shape。

    参数:
        masks (paddle.Tensor): shape 为 (N, C, H, W) 的 masks。
        shape (tuple[int, int]): (height, width) 格式的 target height 与 width。
        ratio_pad (tuple, optional): ((ratio_h, ratio_w), (pad_w, pad_h)) 格式的 ratio 与 padding values。
        padding (bool): masks 是否基于带 padding 的 YOLO-style augmented images。

    返回:
        (paddle.Tensor): rescaled masks。
    """
    im1_h, im1_w = masks.shape[2:]
    im0_h, im0_w = shape[:2]
    if im1_h == im0_h and im1_w == im0_w:
        return masks
    if ratio_pad is None:
        gain = min(im1_h / im0_h, im1_w / im0_w)
        pad_w, pad_h = im1_w - im0_w * gain, im1_h - im0_h * gain
        if padding:
            pad_w /= 2
            pad_h /= 2
    else:
        pad_w, pad_h = ratio_pad[1]
    top, left = (round(pad_h - 0.1), round(pad_w - 0.1)) if padding else (0, 0)
    bottom = im1_h - round(pad_h + 0.1)
    right = im1_w - round(pad_w + 0.1)
    return paddle.nn.functional.interpolate(
        masks[..., top:bottom, left:right].astype("float32"), shape, mode="bilinear"
    )


def scale_coords(
    img1_shape,
    coords,
    img0_shape,
    ratio_pad=None,
    normalize: bool = False,
    padding: bool = True,
):
    """将 segment coordinates 从 img1_shape rescale 到 img0_shape。

    参数:
        img1_shape (tuple): HWC 或 HW 格式的 source image shape（均支持）。
        coords (paddle.Tensor): 待 scale 的 coordinates，shape 为 (N, 2)。
        img0_shape (tuple): HWC 或 HW 格式的 image 0 shape（均支持）。
        ratio_pad (tuple, optional): ((ratio_h, ratio_w), (pad_w, pad_h)) 格式的 ratio 与 padding values。
        normalize (bool): 是否将 coordinates normalize 到 [0, 1]。
        padding (bool): coordinates 是否基于带 padding 的 YOLO-style augmented images。

    返回:
        (paddle.Tensor): scaled coordinates。
    """
    img0_h, img0_w = img0_shape[:2]
    if ratio_pad is None:
        img1_h, img1_w = img1_shape[:2]
        gain = min(img1_h / img0_h, img1_w / img0_w)
        pad = (img1_w - img0_w * gain) / 2, (img1_h - img0_h * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    if padding:
        coords[..., 0] -= pad[0]
        coords[..., 1] -= pad[1]
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_w
        coords[..., 1] /= img0_h
    return coords


def regularize_rboxes(rboxes):
    """将 rotated bounding boxes regularize 到 [0, pi/2) 范围。

    参数:
        rboxes (paddle.Tensor): xywhr format、shape 为 (N, 5) 的 input rotated boxes。

    返回:
        (paddle.Tensor): regularized rotated boxes。
    """
    x, y, w, h, t = rboxes.unbind(axis=-1)
    swap = t % math.pi >= math.pi / 2
    w_ = paddle.where(swap, h, w)
    h_ = paddle.where(swap, w, h)
    t = t % (math.pi / 2)
    return paddle.stack([x, y, w_, h_, t], axis=-1)


def _min_index(arr1: np.ndarray, arr2: np.ndarray) -> tuple[int, int]:
    """@brief 查找两组二维点中距离最近的点对索引。

    @param arr1 第一组二维点，形状为 `[N, 2]`。
    @param arr2 第二组二维点，形状为 `[M, 2]`。
    @return 最近点对在 `arr1` 与 `arr2` 中的索引。
    """
    distance = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(distance, axis=None), distance.shape)


def _merge_multi_segment(segments: list[np.ndarray]) -> list[np.ndarray]:
    """@brief 将多个分段轮廓拼接为一条连续分割曲线。

    @param segments 多段轮廓点集，每段形状为 `[K, 2]`。
    @return 按最短连接关系拼接后的轮廓序列列表。
    """
    merged: list[np.ndarray] = []
    normalized_segments = [np.asarray(segment, dtype=np.float32).reshape(-1, 2) for segment in segments]
    idx_list = [[] for _ in range(len(normalized_segments))]

    # 先记录相邻轮廓间的最近连接点索引。
    for i in range(1, len(normalized_segments)):
        idx1, idx2 = _min_index(normalized_segments[i - 1], normalized_segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # 两轮遍历分别处理正向与反向拼接，逻辑与上游 Ultralytics 一致。
    for round_idx in range(2):
        if round_idx == 0:
            for i, idx in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    normalized_segments[i] = normalized_segments[i][::-1, :]

                normalized_segments[i] = np.roll(normalized_segments[i], -idx[0], axis=0)
                normalized_segments[i] = np.concatenate([normalized_segments[i], normalized_segments[i][:1]])
                if i in {0, len(idx_list) - 1}:
                    merged.append(normalized_segments[i])
                else:
                    span = [0, idx[1] - idx[0]]
                    merged.append(normalized_segments[i][span[0] : span[1] + 1])
        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    tail_start = abs(idx[1] - idx[0])
                    merged.append(normalized_segments[i][tail_start:])
    return merged


def masks2segments(masks: (np.ndarray | paddle.Tensor), strategy: str = "all") -> list[np.ndarray]:
    """使用 contour detection 将 masks 转为 segments。

    参数:
        masks (np.ndarray | paddle.Tensor): shape 为 (N, H, W) 的 binary masks。
        strategy (str): Segmentation strategy，可为 'all' 或 'largest'。

    返回:
        (list): float32 arrays 形式的 segment masks 列表。
    """
    masks = masks.astype("uint8") if isinstance(masks, np.ndarray) else masks.astype("uint8").cpu().numpy()
    segments = []
    for x in np.ascontiguousarray(masks):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":
                c = (
                    np.concatenate(_merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))
        segments.append(c.astype("float32"))
    return segments


def convert_paddle2numpy_batch(batch: paddle.Tensor) -> np.ndarray:
    """将一批 FP32 Paddle tensors 转为 NumPy uint8 arrays，并从 BCHW layout 改为 BHWC layout。

    参数:
        batch (paddle.Tensor): shape 为 (Batch, Channels, Height, Width)、dtype 为 paddle.float32 的 input tensor batch。

    返回:
        (np.ndarray): shape 为 (Batch, Height, Width, Channels)、dtype 为 uint8 的 output NumPy array batch。
    """
    return (batch.transpose([0, 2, 3, 1]) * 255).clip(0, 255).astype("uint8").cpu().numpy()


def clean_str(s):
    """通过将 special characters 替换为 '_' 清理 string。

    参数:
        s (str): 需要替换 special characters 的 string。

    返回:
        (str): 将 special characters 替换为 underscore _ 后的 string。
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨`><+]", repl="_", string=s)


def empty_like(x):
    """创建与 input 具有相同 shape 和 dtype 的空 paddle.Tensor 或 np.ndarray。"""
    return paddle.empty_like(x, dtype=x.dtype) if isinstance(x, paddle.Tensor) else np.empty_like(x, dtype=x.dtype)
