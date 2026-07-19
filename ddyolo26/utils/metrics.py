# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 评估指标：mAP、精度-召回曲线、OKS 等计算。
@details
主要类和函数：
- `DetMetrics`：检测评估汇总（mAP50, mAP50-95, F1）
- `ConfusionMatrix`：混淆矩阵（检测任务各类误检分析）
- `bbox_iou()`：CIoU/DIoU/GIoU/SIoU 计算
- `ap_per_class()`：per-class AP 与精度-召回曲线
- `OKS_SIGMA`：关键点相似性 σ 常量（姿态估计评估）
"""

from __future__ import annotations
import sys
import paddle


from ddyolo26.paddle_utils import *

"""Model 验证指标。"""

import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ddyolo26.utils import LOGGER, DataExportMixin, SimpleClass, TryExcept, checks, plt_settings

OKS_SIGMA = (
    np.array(
        [
            0.26,
            0.25,
            0.25,
            0.35,
            0.35,
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
        ],
        dtype=np.float32,
    )
    / 10.0
)
RLE_WEIGHT = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.5,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.5,
    ]
)


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-07) -> np.ndarray:
    """给定 box1 与 box2，计算 intersection over box2 area。

    参数:
        box1 (np.ndarray): shape 为 (N, 4) 的 numpy array，表示 x1y1x2y2 format 的 N 个 bounding boxes。
        box2 (np.ndarray): shape 为 (M, 4) 的 numpy array，表示 x1y1x2y2 format 的 M 个 bounding boxes。
        iou (bool, optional): 为 True 时计算 standard IoU，否则返回 inter_area/box2_area。
        eps (float, optional): 避免除零的小值。

    返回:
        (np.ndarray): shape 为 (N, M) 的 numpy array，表示 intersection over box2 area。
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area
    return inter_area / (area + eps)


def box_iou(box1: paddle.Tensor, box2: paddle.Tensor, eps: float = 1e-07) -> paddle.Tensor:
    """计算 boxes 的 intersection-over-union (IoU)。

    参数:
        box1 (paddle.Tensor): shape 为 (N, 4) 的 tensor，表示 (x1, y1, x2, y2) format 的 N 个 bounding boxes。
        box2 (paddle.Tensor): shape 为 (M, 4) 的 tensor，表示 (x1, y1, x2, y2) format 的 M 个 bounding boxes。
        eps (float, optional): 避免除零的小值。

    返回:
        (paddle.Tensor): NxM tensor，包含 box1 与 box2 中每个元素的 pairwise IoU values。

    参考:
        Standard vectorized box IoU implementation。
    """
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (paddle.compat.min(a2, b2) - paddle.compat.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(
    box1: paddle.Tensor,
    box2: paddle.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-07,
) -> paddle.Tensor:
    """计算 bounding boxes 之间的 Intersection over Union (IoU)。

    只要最后一维为 4，该函数就支持 `box1` 与 `box2` 的多种 shapes。例如可传入 shape 为 (4,)、(N, 4)、
    (B, N, 4) 或 (B, N, 1, 4) 的 tensors。内部会在 `xywh=True` 时将最后一维拆为 (x, y, w, h)，
    在 `xywh=False` 时拆为 (x1, y1, x2, y2)。

    参数:
        box1 (paddle.Tensor): 表示一个或多个 bounding boxes 的 tensor，最后一维为 4。
        box2 (paddle.Tensor): 表示一个或多个 bounding boxes 的 tensor，最后一维为 4。
        xywh (bool, optional): 若为 True，input boxes 为 (x, y, w, h) format；若为 False，
            input boxes 为 (x1, y1, x2, y2) format。
        GIoU (bool, optional): 若为 True，计算 Generalized IoU。
        DIoU (bool, optional): 若为 True，计算 Distance IoU。
        CIoU (bool, optional): 若为 True，计算 Complete IoU。
        eps (float, optional): 避免除零的小值。

    返回:
        (paddle.Tensor): 根据指定 flags 返回 IoU、GIoU、DIoU 或 CIoU values。
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(y=b2_x2) - b1_x1.maximum(y=b2_x1)).clamp_(0) * (
        b1_y2.minimum(y=b2_y2) - b1_y1.maximum(y=b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(y=b2_x2) - b1_x1.minimum(y=b2_x1)
        ch = b1_y2.maximum(y=b2_y2) - b1_y1.minimum(y=b2_y1)
        if CIoU or DIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
            if CIoU:
                v = 4 / math.pi**2 * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with paddle.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


def mask_iou(mask1: paddle.Tensor, mask2: paddle.Tensor, eps: float = 1e-07) -> paddle.Tensor:
    """计算 masks IoU。

    参数:
        mask1 (paddle.Tensor): shape 为 (N, n) 的 tensor，其中 N 为 ground truth objects 数量，n 为 image width
            与 height 的乘积。
        mask2 (paddle.Tensor): shape 为 (M, n) 的 tensor，其中 M 为 predicted objects 数量，n 为 image width
            与 height 的乘积。
        eps (float, optional): 避免除零的小值。

    返回:
        (paddle.Tensor): shape 为 (N, M) 的 tensor，表示 masks IoU。
    """
    intersection = paddle.matmul(mask1, mask2.T).clamp_(0)
    union = mask1.sum(1)[:, None] + mask2.sum(1)[None] - intersection
    return intersection / (union + eps)


def kpt_iou(
    kpt1: paddle.Tensor,
    kpt2: paddle.Tensor,
    area: paddle.Tensor,
    sigma: list[float],
    eps: float = 1e-07,
) -> paddle.Tensor:
    """计算 Object Keypoint Similarity (OKS)。

    参数:
        kpt1 (paddle.Tensor): shape 为 (N, 17, 3) 的 tensor，表示 ground truth keypoints。
        kpt2 (paddle.Tensor): shape 为 (M, 17, 3) 的 tensor，表示 predicted keypoints。
        area (paddle.Tensor): shape 为 (N,) 的 tensor，表示 ground truth areas。
        sigma (list[float]): 包含 17 个 values 的 list，表示 keypoint scales。
        eps (float, optional): 避免除零的小值。

    返回:
        (paddle.Tensor): shape 为 (N, M) 的 tensor，表示 keypoint similarities。
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)
    sigma = paddle.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)
    kpt_mask = kpt1[..., 2] != 0
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def _get_covariance_matrix(
    boxes: paddle.Tensor,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """根据 oriented bounding boxes 生成 covariance matrix。

    参数:
        boxes (paddle.Tensor): shape 为 (N, 5) 的 tensor，表示 xywhr format 的 rotated bounding boxes。

    返回:
        (tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]): covariance matrix components (a, b, c)，其中 covariance
            matrix 为 [[a, c], [c, b]]，每项 shape 为 (N, 1)。
    """
    gbbs = paddle.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1: paddle.Tensor, obb2: paddle.Tensor, CIoU: bool = False, eps: float = 1e-07) -> paddle.Tensor:
    """计算 oriented bounding boxes 之间的 probabilistic IoU。

    参数:
        obb1 (paddle.Tensor): ground truth OBBs，shape 为 (N, 5)，format 为 xywhr。
        obb2 (paddle.Tensor): predicted OBBs，shape 为 (N, 5)，format 为 xywhr。
        CIoU (bool, optional): 若为 True，计算 CIoU。
        eps (float, optional): 避免除零的小值。

    返回:
        (paddle.Tensor): OBB similarities，shape 为 (N,)。

    说明:
        OBB format: [center_x, center_y, width, height, rotation_angle].

    参考:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        * 0.25
    )
    t2 = (c1 + c2) * (x2 - x1) * (y1 - y2) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = 4 / math.pi**2 * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with paddle.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha
    return iou


def batch_probiou(
    obb1: (paddle.Tensor | np.ndarray),
    obb2: (paddle.Tensor | np.ndarray),
    eps: float = 1e-07,
) -> paddle.Tensor:
    """计算 oriented bounding boxes 之间的 probabilistic IoU。

    参数:
        obb1 (paddle.Tensor | np.ndarray): shape 为 (N, 5) 的 tensor，表示 xywhr format 的 ground truth obbs。
        obb2 (paddle.Tensor | np.ndarray): shape 为 (M, 5) 的 tensor，表示 xywhr format 的 predicted obbs。
        eps (float, optional): 避免除零的小值。

    返回:
        (paddle.Tensor): shape 为 (N, M) 的 tensor，表示 obb similarities。

    参考:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = paddle.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = paddle.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        * 0.25
    )
    t2 = (c1 + c2) * (x2 - x1) * (y1 - y2) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth_bce(eps: float = 0.1) -> tuple[float, float]:
    """计算 smoothed positive 与 negative Binary Cross-Entropy targets。

    参数:
        eps (float, optional): label smoothing 使用的 epsilon value。

    返回:
        pos (float): positive label smoothing BCE target。
        neg (float): negative label smoothing BCE target。

    参考:
        Label smoothing for binary cross-entropy targets。
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix(DataExportMixin):
    """用于 object detection 与 classification tasks 中计算并更新 confusion matrix 的类。

    属性:
        task (str): task 类型，可为 'detect' 或 'classify'。
        matrix (np.ndarray): confusion matrix，其 dimensions 取决于 task。
        nc (int): classes 数量。
        names (dict[int, str]): classes names，用作 plot 上的 labels。
        matches (dict | None): 按 TP、FP 和 FN 分类保存 ground truths 与 predictions 的 indices。
    """

    def __init__(
        self,
        names: dict[int, str] = {},
        task: str = "detect",
        save_matches: bool = False,
    ):
        """初始化 ConfusionMatrix instance。

        参数:
            names (dict[int, str], optional): classes names，用作 plot labels。
            task (str, optional): task 类型，可为 'detect' 或 'classify'。
            save_matches (bool, optional): 保存 GTs、TPs、FPs、FNs 的 indices 以便 visualization。
        """
        self.task = task
        self.nc = len(names)
        self.matrix = np.zeros((self.nc, self.nc)) if self.task == "classify" else np.zeros((self.nc + 1, self.nc + 1))
        self.names = names
        self.matches = {} if save_matches else None

    def _append_matches(self, mtype: str, batch: dict[str, Any], idx: int) -> None:
        """将 matches 追加到 last batch 的 TP、FP、FN 或 GT list。

        该方法会将指定 batch data 追加到对应 match type（True Positive、False Positive 或 False Negative），
        以更新 matches dictionary。

        参数:
            mtype (str): match type identifier（'TP'、'FP'、'FN' 或 'GT'）。
            batch (dict[str, Any]): 包含 detection results 的 batch data，keys 如 'bboxes'、'cls'、'conf'、
                'keypoints'、'masks'。
            idx (int): 要从 batch 追加的指定 detection index。

        说明:
            对 masks，会同时处理 overlap 与 non-overlap cases。当 masks.max() > 1.0 时，表示 overlap_mask=True
            且 shape 为 (1, H, W)，否则使用 direct indexing。
        """
        if self.matches is None:
            return
        for k, v in batch.items():
            if k in {"bboxes", "cls", "conf", "keypoints"}:
                self.matches[mtype][k] += v[[idx]]
            elif k == "masks":
                self.matches[mtype][k] += [v[0] == idx + 1] if v.max() > 1.0 else [v[idx]]

    def process_cls_preds(self, preds: list[paddle.Tensor], targets: list[paddle.Tensor]) -> None:
        """为 classification task 更新 confusion matrix。

        参数:
            preds (list[paddle.Tensor]): predicted class labels。
            targets (list[paddle.Tensor]): ground truth class labels。
        """
        preds, targets = paddle.cat(preds)[:, 0], paddle.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(
        self,
        detections: dict[str, paddle.Tensor],
        batch: dict[str, Any],
        conf: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        """为 object detection task 更新 confusion matrix。

        参数:
            detections (dict[str, paddle.Tensor]): 包含 detected bounding boxes 及其关联信息的 dictionary。
                应包含 'cls'、'conf' 和 'bboxes' keys，其中 'bboxes' 可为 regular boxes 的 Array[N, 4]，
                或带 angle 的 OBB Array[N, 5]。
            batch (dict[str, Any]): 包含 ground truth data 的 batch dictionary，带 'bboxes'（Array[M, 4] |
                Array[M, 5]）和 'cls'（Array[M]）keys，其中 M 是 ground truth objects 数量。
            conf (float, optional): detections 的 confidence threshold。
            iou_thres (float, optional): matching detections 到 ground truth 的 IoU threshold。
        """
        gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]
        if self.matches is not None:
            self.matches = {k: defaultdict(list) for k in {"TP", "FP", "FN", "GT"}}
            for i in range(gt_cls.shape[0]):
                self._append_matches("GT", batch, i)
        is_obb = gt_bboxes.shape[1] == 5
        conf = 0.25 if conf in {None, 0.01 if is_obb else 0.001} else conf
        no_pred = detections["cls"].shape[0] == 0
        if gt_cls.shape[0] == 0:
            if not no_pred:
                detections = {k: detections[k][detections["conf"] > conf] for k in detections}
                detection_classes = detections["cls"].int().tolist()
                for i, dc in enumerate(detection_classes):
                    self.matrix[dc, self.nc] += 1
                    self._append_matches("FP", detections, i)
            return
        if no_pred:
            gt_classes = gt_cls.int().tolist()
            for i, gc in enumerate(gt_classes):
                self.matrix[self.nc, gc] += 1
                self._append_matches("FN", batch, i)
            return
        detections = {k: detections[k][detections["conf"] > conf] for k in detections}
        gt_classes = gt_cls.int().tolist()
        detection_classes = detections["cls"].int().tolist()
        bboxes = detections["bboxes"]
        iou = batch_probiou(gt_bboxes, bboxes) if is_obb else box_iou(gt_bboxes, bboxes)
        x = paddle.where(iou > iou_thres)
        if x[0].shape[0]:
            matches = paddle.cat((paddle.stack(x, 1).cast("float32"), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))
        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                dc = detection_classes[m1[j].item()]
                self.matrix[dc, gc] += 1
                if dc == gc:
                    self._append_matches("TP", detections, m1[j].item())
                else:
                    self._append_matches("FP", detections, m1[j].item())
                    self._append_matches("FN", batch, i)
            else:
                self.matrix[self.nc, gc] += 1
                self._append_matches("FN", batch, i)
        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1
                self._append_matches("FP", detections, i)

    def matrix(self):
        """返回 confusion matrix。"""
        return self.matrix

    def tp_fp(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 true positives 与 false positives。

        返回:
            tp (np.ndarray): true positives。
            fp (np.ndarray): false positives。
        """
        tp = self.matrix.diagonal()
        fp = self.matrix.sum(1) - tp
        return (tp, fp) if self.task == "classify" else (tp[:-1], fp[:-1])

    def plot_matches(self, img: paddle.Tensor, im_file: str, save_dir: Path) -> None:
        """为每张 image 绘制 GT、TP、FP、FN grid。

        参数:
            img (paddle.Tensor): 要绘制到其上的 image。
            im_file (str): 保存 visualizations 的 image filename。
            save_dir (Path): 保存 visualizations 的位置。
        """
        if not self.matches:
            return
        from .ops import xyxy2xywh
        from .plotting import plot_images

        labels = defaultdict(list)
        for i, mtype in enumerate(["GT", "FP", "TP", "FN"]):
            mbatch = self.matches[mtype]
            if "conf" not in mbatch:
                mbatch["conf"] = paddle.tensor([1.0] * len(mbatch["bboxes"]), device=img.device)
            mbatch["batch_idx"] = paddle.ones(len(mbatch["bboxes"]), device=img.device) * i
            for k in mbatch.keys():
                labels[k] += mbatch[k]
        labels = {k: (paddle.stack(v, 0) if len(v) else paddle.empty(0)) for k, v in labels.items()}
        if self.task != "obb" and labels["bboxes"].shape[0]:
            labels["bboxes"] = xyxy2xywh(labels["bboxes"])
        (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        plot_images(
            labels,
            img.repeat(4, 1, 1, 1),
            paths=[
                "真值",
                "误检",
                "正确检出",
                "漏检",
            ],
            fname=save_dir / "visualizations" / Path(im_file).name,
            names=self.names,
            max_subplots=4,
            conf_thres=0.001,
        )

    @TryExcept(msg="ConfusionMatrix 绘图失败")
    @plt_settings()
    def plot(self, normalize: bool = True, save_dir: str = "", on_plot=None):
        """使用 matplotlib 绘制 confusion matrix，并保存到 file。

        参数:
            normalize (bool, optional): 是否 normalize confusion matrix。
            save_dir (str, optional): plot 保存 directory。
            on_plot (callable, optional): plot render 后传递 plots path 与 data 的可选 callback。
        """
        import matplotlib.pyplot as plt

        array = self.matrix / (self.matrix.sum(0).reshape(1, -1) + 1e-09 if normalize else 1)
        array[array < 0.005] = np.nan
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        names, n = list(self.names.values()), self.nc
        if self.nc >= 100:
            k = max(2, self.nc // 60)
            keep_idx = slice(None, None, k)
            names = names[keep_idx]
            array = array[keep_idx, :][:, keep_idx]
            n = (self.nc + k - 1) // k
        nc = n if self.task == "classify" else n + 1
        ticklabels = "auto"
        if 0 < nc < 99:
            ticklabels = names if self.task == "classify" else [*names, "background"]
        xy_ticks = np.arange(len(ticklabels)) if ticklabels != "auto" else np.arange(nc)
        tick_fontsize = max(6, 15 - 0.1 * nc)
        label_fontsize = max(6, 12 - 0.1 * nc)
        title_fontsize = max(6, 12 - 0.1 * nc)
        btm = max(0.1, 0.25 - 0.001 * nc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.imshow(array, cmap="Blues", vmin=0.0, interpolation="none")
            ax.xaxis.set_label_position("bottom")
            if nc < 30:
                color_threshold = 0.45 * (1 if normalize else np.nanmax(array))
                for i, row in enumerate(array[:nc]):
                    for j, val in enumerate(row[:nc]):
                        val = array[i, j]
                        if np.isnan(val):
                            continue
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}" if normalize else f"{int(val)}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="white" if val > color_threshold else "black",
                        )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True", fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel("Predicted", fontsize=label_fontsize, labelpad=10)
        ax.set_title(title, fontsize=title_fontsize, pad=20)
        ax.set_xticks(xy_ticks)
        ax.set_yticks(xy_ticks)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
        if ticklabels != "auto":
            ax.set_xticklabels(ticklabels, fontsize=tick_fontsize, rotation=90, ha="center")
            ax.set_yticklabels(ticklabels, fontsize=tick_fontsize)
        for s in {"left", "right", "bottom", "top", "outline"}:
            if s != "outline":
                ax.spines[s].set_visible(False)
            cbar.ax.spines[s].set_visible(False)
        fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=btm)
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname, {"type": "confusion_matrix", "matrix": self.matrix.tolist()})

    def print(self):
        """将 confusion matrix 打印到 console。"""
        for i in range(self.matrix.shape[0]):
            LOGGER.info(" ".join(map(str, self.matrix[i])))

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, float]]:
        """以 dictionaries 列表生成 confusion matrix 的 summarized representation，可选 normalization。

        这对将 matrix 导出为 CSV、XML、HTML、JSON 或 SQL 等 formats 很有用。

        参数:
            normalize (bool): 是否 normalize confusion matrix values。
            decimals (int): output values 保留的小数位数。

        返回:
            (list[dict[str, float]]): dictionaries 列表，每个 dictionary 表示一个 predicted class 及所有 actual
                classes 对应 values。

        示例:
            >>> results = model.val(data="coco8.yaml", plots=True)
            >>> cm_dict = results.confusion_matrix.summary(normalize=True, decimals=5)
            >>> print(cm_dict)
        """
        import re

        names = list(self.names.values()) if self.task == "classify" else [*list(self.names.values()), "background"]
        clean_names, seen = [], set()
        for name in names:
            clean_name = re.sub("[^a-zA-Z0-9_]", "_", name)
            original_clean = clean_name
            counter = 1
            while clean_name.lower() in seen:
                clean_name = f"{original_clean}_{counter}"
                counter += 1
            seen.add(clean_name.lower())
            clean_names.append(clean_name)
        array = (self.matrix / (self.matrix.sum(0).reshape(1, -1) + 1e-09 if normalize else 1)).round(decimals)
        return [
            dict(
                {"Predicted": clean_names[i]},
                **{clean_names[j]: array[i, j] for j in range(len(clean_names))},
            )
            for i in range(len(clean_names))
        ]


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """fraction 为 f 的 box filter。"""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


@plt_settings()
def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] = {},
    on_plot=None,
):
    """绘制 precision-recall curve。

    参数:
        px (np.ndarray): PR curve 的 X values。
        py (np.ndarray): PR curve 的 Y values。
        ap (np.ndarray): average precision values。
        save_dir (Path, optional): 保存 plot 的 path。
        names (dict[int, str], optional): class indices 到 class names 的 mapping 字典。
        on_plot (callable, optional): plot 保存后调用的 function。
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)
    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")
    else:
        ax.plot(px, py, linewidth=1, color="gray")
    ax.plot(
        px,
        py.mean(1),
        linewidth=3,
        color="blue",
        label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(
            save_dir,
            {
                "type": "pr_curve",
                "x": px.tolist(),
                "y": py.T.tolist(),
                "ap": ap.tolist(),
            },
        )


@plt_settings()
def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] = {},
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    on_plot=None,
):
    """绘制 metric-confidence curve。

    参数:
        px (np.ndarray): metric-confidence curve 的 X values。
        py (np.ndarray): metric-confidence curve 的 Y values。
        save_dir (Path, optional): 保存 plot 的 path。
        names (dict[int, str], optional): class indices 到 class names 的 mapping 字典。
        xlabel (str, optional): X-axis label。
        ylabel (str, optional): Y-axis label。
        on_plot (callable, optional): plot 保存后调用的 function。
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")
    else:
        ax.plot(px, py.T, linewidth=1, color="gray")
    y = smooth(py.mean(0), 0.1)
    ax.plot(
        px,
        y,
        linewidth=3,
        color="blue",
        label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(
            save_dir,
            {"type": f"{ylabel.lower()}_curve", "x": px.tolist(), "y": py.tolist()},
        )


def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """根据 recall 与 precision curves 计算 average precision (AP)。

    参数:
        recall (list[float]): recall curve。
        precision (list[float]): precision curve。

    返回:
        ap (float): average precision。
        mpre (np.ndarray): precision envelope curve。
        mrec (np.ndarray): 在首尾添加 sentinel values 后的 modified recall curve。
    """
    # 与最新版 Ultralytics 对齐：在真实最大 recall 位置先把 precision
    # 降到 0，再延伸到 recall=1。旧实现直接连接到 (1, 0)，会在
    # 最大 recall 不足 1 时通过插值虚增 PR 曲线尾部面积。
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    method = "interp"
    if method == "interp":
        x = np.linspace(0, 1, 101)
        func = np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz
        ap = func(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    """为 object detection evaluation 计算每个 class 的 average precision。

    参数:
        tp (np.ndarray): binary array，表示 detection 是否正确（True/False）。
        conf (np.ndarray): detections 的 confidence scores array。
        pred_cls (np.ndarray): detections 的 predicted classes array。
        target_cls (np.ndarray): targets 的 true classes array。
        plot (bool, optional): 是否绘制 PR curves。
        on_plot (callable, optional): curves render 后传递 plots path 与 data 的 callback。
        save_dir (Path, optional): 保存 PR curves 的 directory。
        names (dict[int, str], optional): 用于绘制 PR curves 的 class names 字典。
        eps (float, optional): 避免除零的小值。
        prefix (str, optional): 保存 plot files 使用的 prefix string。

    返回:
        tp (np.ndarray): 每个 class 在 max F1 metric 给定 threshold 下的 true positive counts。
        fp (np.ndarray): 每个 class 在 max F1 metric 给定 threshold 下的 false positive counts。
        p (np.ndarray): 每个 class 在 max F1 metric 给定 threshold 下的 precision values。
        r (np.ndarray): 每个 class 在 max F1 metric 给定 threshold 下的 recall values。
        f1 (np.ndarray): 每个 class 在 max F1 metric 给定 threshold 下的 F1-score values。
        ap (np.ndarray): 每个 class 在不同 IoU thresholds 下的 average precision。
        unique_classes (np.ndarray): 有数据的 unique classes array。
        p_curve (np.ndarray): 每个 class 的 precision curves。
        r_curve (np.ndarray): 每个 class 的 recall curves。
        f1_curve (np.ndarray): 每个 class 的 F1-score curves。
        x (np.ndarray): curves 的 X-axis values。
        prec_values (np.ndarray): 每个 class 在 mAP@0.5 下的 precision values。
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x, prec_values = np.linspace(0, 1, 1000), []
    ap, p_curve, r_curve = (
        np.zeros((nc, tp.shape[1])),
        np.zeros((nc, 1000)),
        np.zeros((nc, 1000)),
    )
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))
    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}
    if plot:
        plot_pr_curve(
            x,
            prec_values,
            ap,
            save_dir / f"{prefix}PR_curve.png",
            names,
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            f1_curve,
            save_dir / f"{prefix}F1_curve.png",
            names,
            ylabel="F1",
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            p_curve,
            save_dir / f"{prefix}P_curve.png",
            names,
            ylabel="Precision",
            on_plot=on_plot,
        )
        plot_mc_curve(
            x,
            r_curve,
            save_dir / f"{prefix}R_curve.png",
            names,
            ylabel="Recall",
            on_plot=on_plot,
        )
    i = smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    return (
        tp,
        fp,
        p,
        r,
        f1,
        ap,
        unique_classes.astype(int),
        p_curve,
        r_curve,
        f1_curve,
        x,
        prec_values,
    )


class Metric(SimpleClass):
    """用于计算 PaddleYOLO-RKNN models evaluation metrics 的类。

    属性:
        p (list): 每个 class 的 precision。Shape: (nc,)。
        r (list): 每个 class 的 recall。Shape: (nc,)。
        f1 (list): 每个 class 的 F1 score。Shape: (nc,)。
        all_ap (list): 所有 classes 与所有 IoU thresholds 的 AP scores。Shape: (nc, 10)。
        ap_class_index (list): 每个 AP score 对应的 class index。Shape: (nc,)。
        nc (int): classes 数量。

    方法:
        ap50: 所有 classes 在 IoU threshold 0.5 下的 AP。
        ap: 所有 classes 在 IoU thresholds 0.5 到 0.95 下的 AP。
        mp: 所有 classes 的 mean precision。
        mr: 所有 classes 的 mean recall。
        map50: 所有 classes 在 IoU threshold 0.5 下的 mean AP。
        map75: 所有 classes 在 IoU threshold 0.75 下的 mean AP。
        map: 所有 classes 在 IoU thresholds 0.5 到 0.95 下的 mean AP。
        mean_results: results mean，返回 mp、mr、map50、map。
        class_result: class-aware result，返回 p[i]、r[i]、ap50[i]、ap[i]。
        maps: 每个 class 的 mAP。
        fitness: metrics 的 weighted combination 形式的 model fitness。
        update: 用新的 evaluation results 更新 metric attributes。
        curves: 提供访问 precision、recall、F1 等 specific metrics curves 的列表。
        curves_results: 提供访问 precision、recall、F1 等 specific metrics results 的列表。
    """

    def __init__(self) -> None:
        """初始化用于计算 YOLO model evaluation metrics 的 Metric instance。"""
        self.p = []
        self.r = []
        self.f1 = []
        self.all_ap = []
        self.ap_class_index = []
        self.nc = 0

    @property
    def ap50(self) -> np.ndarray | list:
        """返回所有 classes 在 IoU threshold 0.5 下的 Average Precision (AP)。

        返回:
            (np.ndarray | list): shape 为 (nc,) 的 array，包含每个 class 的 AP50 values；不可用时为空 list。
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self) -> np.ndarray | list:
        """返回所有 classes 在 IoU threshold 0.5-0.95 下的 Average Precision (AP)。

        返回:
            (np.ndarray | list): shape 为 (nc,) 的 array，包含每个 class 的 AP50-95 values；不可用时为空 list。
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self) -> float:
        """返回所有 classes 的 Mean Precision。

        返回:
            (float): 所有 classes 的 mean precision。
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        """返回所有 classes 的 Mean Recall。

        返回:
            (float): 所有 classes 的 mean recall。
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self) -> float:
        """返回 IoU threshold 0.5 下的 mean Average Precision (mAP)。

        返回:
            (float): IoU threshold 0.5 下的 mAP。
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self) -> float:
        """返回 IoU threshold 0.75 下的 mean Average Precision (mAP)。

        返回:
            (float): IoU threshold 0.75 下的 mAP。
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self) -> float:
        """返回 IoU thresholds 0.5 - 0.95（step 0.05）上的 mean Average Precision (mAP)。

        返回:
            (float): IoU thresholds 0.5 - 0.95（step 0.05）上的 mAP。
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self) -> list[float]:
        """返回 results mean：mp、mr、map50、map。"""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """返回 class-aware result：p[i]、r[i]、ap50[i]、ap[i]。"""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self) -> np.ndarray:
        """返回每个 class 的 mAP。"""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self) -> float:
        """以 metrics 的 weighted combination 返回 model fitness。"""
        w = [0.0, 0.0, 0.0, 1.0]
        return (np.nan_to_num(np.array(self.mean_results())) * w).sum()

    def update(self, results: tuple):
        """使用一组新的 results 更新 evaluation metrics。

        参数:
            results (tuple): 包含 evaluation metrics 的 tuple:
                - p (list): 每个 class 的 precision。
                - r (list): 每个 class 的 recall。
                - f1 (list): 每个 class 的 F1 score。
                - all_ap (list): 所有 classes 与所有 IoU thresholds 的 AP scores。
                - ap_class_index (list): 每个 AP score 对应的 class index。
                - p_curve (list): 每个 class 的 precision curve。
                - r_curve (list): 每个 class 的 recall curve。
                - f1_curve (list): 每个 class 的 F1 curve。
                - px (list): curves 的 X values。
                - prec_values (list): 每个 class 的 precision values。
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self) -> list:
        """返回用于访问 specific metrics curves 的 curves 列表。"""
        return []

    @property
    def curves_results(self) -> list[list]:
        """返回用于访问 specific metrics curves 的 curves results 列表。"""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


class DetMetrics(SimpleClass, DataExportMixin):
    """用于计算 precision、recall 与 mean average precision (mAP) 等 detection metrics 的工具类。

    属性:
        names (dict[int, str]): class names 字典。
        box (Metric): 用于存储 detection results 的 Metric class instance。
        speed (dict[str, float]): 存储 detection process 不同部分 execution times 的 dictionary。
        task (str): task 类型，固定为 'detect'。
        stats (dict[str, list]): 包含 true positives、confidence scores、predicted classes、target classes
            与 target images 列表的 dictionary。
        nt_per_class: 每个 class 的 targets 数量。
        nt_per_image: 每张 image 的 targets 数量。

    方法:
        update_stats: 向现有 stat collections 追加 new values 以更新 statistics。
        process: 处理 object detection predicted results 并更新 metrics。
        clear_stats: 清空已存储 statistics。
        keys: 返回用于访问 specific metrics 的 keys 列表。
        mean_results: 计算 detected objects 的 mean，并返回 precision、recall、mAP50 与 mAP50-95。
        class_result: 返回 object detection model 在指定 class 上的 performance evaluation result。
        maps: 返回每个 class 的 mean Average Precision (mAP) scores。
        fitness: 返回 box object 的 fitness。
        ap_class_index: 返回每个 class 的 average precision index。
        results_dict: 返回 computed performance metrics 与 statistics 的 dictionary。
        curves: 返回用于访问 specific metrics curves 的 curves 列表。
        curves_results: 返回 computed performance metrics 与 statistics 列表。
        summary: 以 dictionaries 列表生成 per-class detection metrics 的 summarized representation。
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """使用 class names 初始化 DetMetrics instance。

        参数:
            names (dict[int, str], optional): class names 字典。
        """
        self.names = names
        self.box = Metric()
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }
        self.task = "detect"
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.nt_per_class = None
        self.nt_per_image = None

    def update_stats(self, stat: dict[str, Any]) -> None:
        """向现有 stat collections 追加 new values 以更新 statistics。

        参数:
            stat (dict[str, Any]): 包含待追加 new statistical values 的 dictionary。Keys 应与 self.stats 中既有
                keys 匹配。
        """
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """处理 object detection predicted results 并更新 metrics。

        参数:
            save_dir (Path): 保存 plots 的 directory。默认 Path(".")。
            plot (bool): 是否绘制 precision-recall curves。默认 False。
            on_plot (callable, optional): plots 生成后调用的 function。默认 None。

        返回:
            (dict[str, np.ndarray]): 包含 concatenated statistics arrays 的 dictionary。
        """
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}
        if not stats:
            return stats
        results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            save_dir=save_dir,
            names=self.names,
            on_plot=on_plot,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=len(self.names))
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=len(self.names))
        return stats

    def clear_stats(self):
        """清空已存储 statistics。"""
        for v in self.stats.values():
            v.clear()

    @property
    def keys(self) -> list[str]:
        """返回用于访问 specific metrics 的 keys 列表。"""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ]

    def mean_results(self) -> list[float]:
        """计算 detected objects 的 mean，并返回 precision、recall、mAP50 与 mAP50-95。"""
        return self.box.mean_results()

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """返回 object detection model 在指定 class 上的 performance evaluation result。"""
        return self.box.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """返回每个 class 的 mean Average Precision (mAP) scores。"""
        return self.box.maps

    @property
    def fitness(self) -> float:
        """返回 box object 的 fitness。"""
        return self.box.fitness()

    @property
    def ap_class_index(self) -> list:
        """返回每个 class 的 average precision index。"""
        return self.box.ap_class_index

    @property
    def results_dict(self) -> dict[str, float]:
        """返回 computed performance metrics 与 statistics 的 dictionary。"""
        keys = [*self.keys, "fitness"]
        values = (float(x) if hasattr(x, "item") else x for x in [*self.mean_results(), self.fitness])
        return dict(zip(keys, values))

    @property
    def curves(self) -> list[str]:
        """返回用于访问 specific metrics curves 的 curves 列表。"""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """返回 computed performance metrics 与 statistics 列表。"""
        return self.box.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """以 dictionaries 列表生成 per-class detection metrics 的 summarized representation。

        包含 shared scalar metrics（mAP、mAP50、mAP75）以及每个 class 的 precision、recall 与 F1-score。

        参数:
            normalize (bool): 对 Detect metrics，所有内容默认 normalized 到 [0-1]。
            decimals (int): metrics values 保留的小数位数。

        返回:
            (list[dict[str, Any]]): dictionaries 列表，每个 dictionary 表示一个 class 及其对应 metric values。

        示例:
           >>> results = model.val(data="coco8.yaml")
           >>> detection_summary = results.summary()
           >>> print(detection_summary)
        """
        per_class = {"Box-P": self.box.p, "Box-R": self.box.r, "Box-F1": self.box.f1}
        return [
            {
                "Class": self.names[self.ap_class_index[i]],
                "Images": self.nt_per_image[self.ap_class_index[i]],
                "Instances": self.nt_per_class[self.ap_class_index[i]],
                **{k: round(v[i], decimals) for k, v in per_class.items()},
                "mAP50": round(self.class_result(i)[2], decimals),
                "mAP50-95": round(self.class_result(i)[3], decimals),
            }
            for i in range(len(per_class["Box-P"]))
        ]


class SegmentMetrics(DetMetrics):
    """检测+分割指标聚合器，在 DetMetrics 基础上增加掩码 mAP。

    属性:
        seg (Metric): 掩码分割指标实例。
        task (str): 任务类型，固定为 'segment'。
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """初始化 SegmentMetrics。

        参数:
            names (dict[int, str], optional): 类别名称字典。
        """
        DetMetrics.__init__(self, names)
        self.seg = Metric()
        self.task = "segment"
        self.stats["tp_m"] = []

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """处理检测与分割指标。

        参数:
            save_dir (Path): 保存图表的目录。
            plot (bool): 是否绘制 PR 曲线。
            on_plot (callable, optional): 图表生成后的回调。

        返回:
            (dict[str, np.ndarray]): 拼接后的统计数组字典。
        """
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)
        results_mask = ap_per_class(
            stats["tp_m"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Mask",
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        return stats

    @property
    def keys(self) -> list[str]:
        """返回所有指标键名列表。"""
        return [
            *DetMetrics.keys.fget(self),
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    def mean_results(self) -> list[float]:
        """返回框与掩码的平均指标。"""
        return DetMetrics.mean_results(self) + self.seg.mean_results()

    def class_result(self, i: int) -> list[float]:
        """返回指定类别的框+掩码评估结果。"""
        return DetMetrics.class_result(self, i) + self.seg.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """返回各类别的 mAP（框+掩码）。"""
        return DetMetrics.maps.fget(self) + self.seg.maps

    @property
    def fitness(self) -> float:
        """返回分割+框联合适应度分数。"""
        return self.seg.fitness() + DetMetrics.fitness.fget(self)

    @property
    def curves(self) -> list[str]:
        """返回所有曲线标识名。"""
        return [
            *DetMetrics.curves.fget(self),
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """返回所有曲线结果数据。"""
        return DetMetrics.curves_results.fget(self) + self.seg.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """生成每类分割指标摘要。

        参数:
            normalize (bool): 指标默认已归一化到 [0-1]。
            decimals (int): 小数位数。

        返回:
            (list[dict[str, Any]]): 每类指标字典列表。
        """
        per_class = {
            "Mask-P": self.seg.p,
            "Mask-R": self.seg.r,
            "Mask-F1": self.seg.f1,
        }
        summary = DetMetrics.summary(self, normalize, decimals)
        for i, s in enumerate(summary):
            s.update({**{k: round(v[i], decimals) for k, v in per_class.items()}})
        return summary
