# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief NMS 工具：YOLO26 one-to-many head 推理时的非极大值抑制后处理。
@details
YOLO26 的 end2end（one-to-one head）模式**不使用 NMS**，
本模块的 NMS 仅在以下情况调用：
- 传统模式推理（`end2end=False`）
- 精度对比评估（one-to-many head）
- 某些导出格式（如 CoreML with NMS）

实现了标准多类 NMS 和旋转框（OBB）NMS。
"""

from __future__ import annotations
import sys
import time

import paddle
from ddyolo26.paddle_utils import *

from ddyolo26.utils import LOGGER
from ddyolo26.utils.metrics import batch_probiou, box_iou
from ddyolo26.utils.ops import xywh2xyxy


def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """对 prediction results 执行非极大值抑制（NMS）。

    根据 confidence 与 IoU thresholds 过滤重叠 bounding boxes。支持标准框、旋转框和 masks 等多种检测格式。

    参数:
        prediction (paddle.Tensor): shape 为 (batch_size, num_classes + 4 + num_masks, num_boxes) 的 predictions，
            包含 boxes、classes 和可选 masks。
        conf_thres (float): 过滤 detections 的 confidence threshold，有效范围为 0.0 到 1.0。
        iou_thres (float): NMS 过滤使用的 IoU threshold，有效范围为 0.0 到 1.0。
        classes (list[int], optional): 需要考虑的 class indices；为 None 时考虑所有 classes。
        agnostic (bool): 是否执行 class-agnostic NMS。
        multi_label (bool): 每个 box 是否允许拥有多个 labels。
        labels (list[paddle.Tensor]): 每张 image 的先验 labels。
        max_det (int): 每张 image 最多保留的 detections 数。
        nc (int): classes 数量；之后的 indices 视为 masks。
        max_time_img (float): 单张 image 最大处理时间，单位为秒。
        max_nms (int): 进入 NMS 的最大 boxes 数。
        max_wh (int): box 最大宽高，单位为 pixels。
        rotated (bool): 是否处理 Oriented Bounding Boxes（OBB）。
        end2end (bool): model 是否为 end-to-end 且无需 NMS。
        return_idxs (bool): 是否返回保留 detections 的 indices。

    返回:
        (list[paddle.Tensor] | tuple[list[paddle.Tensor], list[paddle.Tensor]]): 每张 image 的 detections 列表，
            shape 为 (num_boxes, 6 + num_masks)，包含 (x1, y1, x2, y2, confidence, class, mask1, mask2, ...)。
            若 return_idxs=True，则返回 (output, keepi)，其中 keepi 包含保留 detections 的 indices。
    """
    assert 0 <= conf_thres <= 1, f"无效 Confidence threshold {conf_thres}，有效范围为 0.0 到 1.0"
    assert 0 <= iou_thres <= 1, f"无效 IoU {iou_thres}，有效范围为 0.0 到 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if classes is not None:
        classes = paddle.tensor(classes, device=prediction.device)
    if prediction.shape[-1] == 6 or end2end:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output
    bs = prediction.shape[0]
    nc = nc or prediction.shape[1] - 4
    extra = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres
    xinds = paddle.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]
    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1
    prediction = prediction.transpose(-1, -2)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    t = time.time()
    output = [paddle.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [paddle.zeros((0, 1), device=prediction.device)] * bs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        filt = xc[xi]
        x = x[filt]
        if return_idxs:
            xk = xk[filt]
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = paddle.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = paddle.cat((x, v), 0)
        if not x.shape[0]:
            continue
        box, cls, mask = x.split((4, nc, extra), 1)
        if multi_label:
            i, j = paddle.where(cls > conf_thres)
            x = paddle.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:
            conf, j = cls._max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = paddle.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        if rotated:
            boxes = paddle.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = PaddleNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4] + c
            i = PaddleNMS.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if time.time() - t > time_limit:
            LOGGER.warning(f"NMS 超过时间限制 {time_limit:.3f}s")
            break
    return (output, keepi) if return_idxs else output


class PaddleNMS:
    """面向 YOLO 优化的 PaddleYOLO-RKNN custom NMS implementation。

    该类提供对 bounding boxes 执行非极大值抑制（NMS）的静态方法，包括标准 NMS、fast NMS，
    以及多类别场景的 batched NMS。

    方法:
        fast_nms: 使用上三角矩阵操作的 Fast-NMS。
        nms: 支持提前终止且行为稳定的 optimized NMS。
        batched_nms: 用于 class-aware suppression 的 Batched NMS。

    示例:
        对 boxes 与 scores 执行标准 NMS
        >>> boxes = paddle.to_tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> scores = paddle.to_tensor([0.9, 0.8])
        >>> keep = PaddleNMS.nms(boxes, scores, 0.5)
    """

    @staticmethod
    def fast_nms(
        boxes: paddle.Tensor,
        scores: paddle.Tensor,
        iou_threshold: float,
        use_triu: bool = True,
        iou_func=box_iou,
        exit_early: bool = True,
    ) -> paddle.Tensor:
        """基于 https://arxiv.org/pdf/1904.02689 的 Fast-NMS 实现，使用上三角矩阵操作。

        参数:
            boxes (paddle.Tensor): xyxy format、shape 为 (N, 4) 的 bounding boxes。
            scores (paddle.Tensor): shape 为 (N,) 的 confidence scores。
            iou_threshold (float): suppression 使用的 IoU threshold。
            use_triu (bool): 是否使用 paddle.triu operator 执行上三角矩阵操作。
            iou_func (callable): 计算 boxes 间 IoU 的 function。
            exit_early (bool): 没有 boxes 时是否提前退出。

        返回:
            (paddle.Tensor): NMS 后保留 boxes 的 indices。

        示例:
            对一组 boxes 应用 NMS
            >>> boxes = paddle.to_tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = paddle.to_tensor([0.9, 0.8])
            >>> keep = PaddleNMS.fast_nms(boxes, scores, 0.5)
        """
        if boxes.size == 0 and exit_early:
            return paddle.empty((0,), dtype=paddle.int64, device=boxes.device)
        sorted_idx = paddle.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            pick = paddle.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(axis=-1)
        else:
            n = boxes.shape[0]
            row_idx = paddle.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = paddle.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            scores_ = scores[sorted_idx]
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            scores[sorted_idx] = scores_
            pick = paddle.topk(scores_, scores_.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: paddle.Tensor, scores: paddle.Tensor, iou_threshold: float) -> paddle.Tensor:
        """使用 Paddle 原生算子执行标准 NMS。

        参数:
            boxes (paddle.Tensor): xyxy format、shape 为 (N, 4) 的 bounding boxes。
            scores (paddle.Tensor): shape 为 (N,) 的 confidence scores。
            iou_threshold (float): suppression 使用的 IoU threshold。

        返回:
            (paddle.Tensor): NMS 后保留 boxes 的 indices。

        示例:
            对一组 boxes 应用 NMS
            >>> boxes = paddle.to_tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = paddle.to_tensor([0.9, 0.8])
            >>> keep = PaddleNMS.nms(boxes, scores, 0.5)
        """
        if boxes.size == 0:
            return paddle.empty((0,), dtype=paddle.int64, device=boxes.device)
        return paddle.vision.ops.nms(boxes, iou_threshold, scores=scores)

    @staticmethod
    def batched_nms(
        boxes: paddle.Tensor,
        scores: paddle.Tensor,
        idxs: paddle.Tensor,
        iou_threshold: float,
        use_fast_nms: bool = False,
    ) -> paddle.Tensor:
        """用于 class-aware suppression 的 Batched NMS。

        参数:
            boxes (paddle.Tensor): xyxy format、shape 为 (N, 4) 的 bounding boxes。
            scores (paddle.Tensor): shape 为 (N,) 的 confidence scores。
            idxs (paddle.Tensor): shape 为 (N,) 的 class indices。
            iou_threshold (float): suppression 使用的 IoU threshold。
            use_fast_nms (bool): 是否使用 Fast-NMS implementation。

        返回:
            (paddle.Tensor): NMS 后保留 boxes 的 indices。

        示例:
            跨多个 classes 应用 batched NMS
            >>> boxes = paddle.to_tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = paddle.to_tensor([0.9, 0.8])
            >>> idxs = paddle.to_tensor([0, 1])
            >>> keep = PaddleNMS.batched_nms(boxes, scores, idxs, 0.5)
        """
        if boxes.size == 0:
            return paddle.empty((0,), dtype=paddle.int64, device=boxes.device)
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        return (
            PaddleNMS.fast_nms(boxes_for_nms, scores, iou_threshold)
            if use_fast_nms
            else PaddleNMS.nms(boxes_for_nms, scores, iou_threshold)
        )
