# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief Task-Aligned Label Assignment (TAL)：YOLO26 动态标签分配算法。
@details
实现了基于任务对齐分数的动态锚点分配：
- `TaskAlignedAssigner`：标准 TAL（topk 正样本分配）
- `RotatedTaskAlignedAssigner`：旋转框 TAL（OBB 任务）
- `make_anchors()`：从特征图尺寸生成中心点锚点
- `dist2bbox()` / `bbox2dist()`：LTRB 分布 ↔ xyxy 坐标转换

TAL 对齐分数 = `cls_score ** alpha * iou_score ** beta`，综合分类+定位质量
选取 topk 候选，解决了 anchor-free 训练的一对多与一对一分配问题。
"""

from __future__ import annotations
import sys
import paddle


from ddyolo26.paddle_utils import *

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywh2xyxy, xywhr2xyxyxyxy, xyxy2xywh


class TaskAlignedAssigner(paddle.nn.Module):
    """用于 object detection 的 task-aligned assigner。

    该类基于 task-aligned metric 将 ground-truth (gt) objects 分配给 anchors，
    该 metric 同时结合 classification 与 localization 信息。

    属性:
        topk (int): 参与考虑的 top candidates 数量。
        topk2 (int): 用于额外 filtering 的 secondary topk 值。
        num_classes (int): object classes 数量。
        alpha (float): task-aligned metric 中 classification 分量的 alpha 参数。
        beta (float): task-aligned metric 中 localization 分量的 beta 参数。
        stride (list): 不同 feature levels 的 stride values 列表。
        stride_val (int): select_candidates_in_gts 使用的 stride value。
        eps (float): 防止除零的小值。
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list = [8, 16, 32],
        eps: float = 1e-09,
        topk2=None,
    ):
        """初始化可自定义 hyperparameters 的 TaskAlignedAssigner 对象。

        参数:
            topk (int, optional): 参与考虑的 top candidates 数量。
            num_classes (int, optional): object classes 数量。
            alpha (float, optional): task-aligned metric 中 classification 分量的 alpha 参数。
            beta (float, optional): task-aligned metric 中 localization 分量的 beta 参数。
            stride (list, optional): 不同 feature levels 的 stride values 列表。
            eps (float, optional): 防止除零的小值。
            topk2 (int, optional): 用于额外 filtering 的 secondary topk 值。
        """
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps

    @paddle.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """计算 task-aligned assignment。

        参数:
            pd_scores (paddle.Tensor): shape 为 (bs, num_total_anchors, num_classes) 的 predicted classification scores。
            pd_bboxes (paddle.Tensor): shape 为 (bs, num_total_anchors, 4) 的 predicted bounding boxes。
            anc_points (paddle.Tensor): shape 为 (num_total_anchors, 2) 的 anchor points。
            gt_labels (paddle.Tensor): shape 为 (bs, n_max_boxes, 1) 的 ground truth labels。
            gt_bboxes (paddle.Tensor): shape 为 (bs, n_max_boxes, 4) 的 ground truth boxes。
            mask_gt (paddle.Tensor): shape 为 (bs, n_max_boxes, 1) 的 valid ground truth boxes mask。

        返回:
            target_labels (paddle.Tensor): shape 为 (bs, num_total_anchors) 的 target labels。
            target_bboxes (paddle.Tensor): shape 为 (bs, num_total_anchors, 4) 的 target bounding boxes。
            target_scores (paddle.Tensor): shape 为 (bs, num_total_anchors, num_classes) 的 target scores。
            fg_mask (paddle.Tensor): shape 为 (bs, num_total_anchors) 的 foreground mask。
            target_gt_idx (paddle.Tensor): shape 为 (bs, num_total_anchors) 的 target ground truth indices。

        参考:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device
        if self.n_max_boxes == 0:
            return (
                paddle.full_like(pd_scores[..., 0], self.num_classes),
                paddle.zeros_like(pd_bboxes),
                paddle.zeros_like(pd_scores),
                paddle.zeros_like(pd_scores[..., 0]),
                paddle.zeros_like(pd_scores[..., 0]),
            )
        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                LOGGER.warning("TaskAlignedAssigner 发生 CUDA OutOfMemoryError，改用 CPU")
                cpu_tensors = [
                    t.cpu()
                    for t in (
                        pd_scores,
                        pd_bboxes,
                        anc_points,
                        gt_labels,
                        gt_bboxes,
                        mask_gt,
                    )
                ]
                result = self._forward(*cpu_tensors)
                return tuple(t.to(device) for t in result)
            raise

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """计算 task-aligned assignment。

        参数:
            pd_scores (paddle.Tensor): shape 为 (bs, num_total_anchors, num_classes) 的 predicted classification scores。
            pd_bboxes (paddle.Tensor): shape 为 (bs, num_total_anchors, 4) 的 predicted bounding boxes。
            anc_points (paddle.Tensor): shape 为 (num_total_anchors, 2) 的 anchor points。
            gt_labels (paddle.Tensor): shape 为 (bs, n_max_boxes, 1) 的 ground truth labels。
            gt_bboxes (paddle.Tensor): shape 为 (bs, n_max_boxes, 4) 的 ground truth boxes。
            mask_gt (paddle.Tensor): shape 为 (bs, n_max_boxes, 1) 的 valid ground truth boxes mask。

        返回:
            target_labels (paddle.Tensor): shape 为 (bs, num_total_anchors) 的 target labels。
            target_bboxes (paddle.Tensor): shape 为 (bs, num_total_anchors, 4) 的 target bounding boxes。
            target_scores (paddle.Tensor): shape 为 (bs, num_total_anchors, num_classes) 的 target scores。
            fg_mask (paddle.Tensor): shape 为 (bs, num_total_anchors) 的 foreground mask。
            target_gt_idx (paddle.Tensor): shape 为 (bs, num_total_anchors) 的 target ground truth indices。
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores.astype(norm_align_metric.dtype) * norm_align_metric
        return (
            target_labels,
            target_bboxes,
            target_scores,
            fg_mask.bool(),
            target_gt_idx,
        )

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """获取每个 ground truth box 的 positive mask。

        参数:
            pd_scores (paddle.Tensor): shape 为 (bs, num_total_anchors, num_classes) 的 predicted classification scores。
            pd_bboxes (paddle.Tensor): shape 为 (bs, num_total_anchors, 4) 的 predicted bounding boxes。
            gt_labels (paddle.Tensor): shape 为 (bs, n_max_boxes, 1) 的 ground truth labels。
            gt_bboxes (paddle.Tensor): shape 为 (bs, n_max_boxes, 4) 的 ground truth boxes。
            anc_points (paddle.Tensor): shape 为 (num_total_anchors, 2) 的 anchor points。
            mask_gt (paddle.Tensor): shape 为 (bs, n_max_boxes, 1) 的 valid ground truth boxes mask。

        返回:
            mask_pos (paddle.Tensor): shape 为 (bs, max_num_obj, h*w) 的 positive mask。
            align_metric (paddle.Tensor): shape 为 (bs, max_num_obj, h*w) 的 alignment metric。
            overlaps (paddle.Tensor): shape 为 (bs, max_num_obj, h*w) 的 predicted 与 ground truth boxes overlaps。
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """根据 predicted 和 ground truth bounding boxes 计算 alignment metric。

        参数:
            pd_scores (paddle.Tensor): shape 为 (bs, num_total_anchors, num_classes) 的 predicted classification scores。
            pd_bboxes (paddle.Tensor): shape 为 (bs, num_total_anchors, 4) 的 predicted bounding boxes。
            gt_labels (paddle.Tensor): shape 为 (bs, n_max_boxes, 1) 的 ground truth labels。
            gt_bboxes (paddle.Tensor): shape 为 (bs, n_max_boxes, 4) 的 ground truth boxes。
            mask_gt (paddle.Tensor): shape 为 (bs, n_max_boxes, h*w) 的 valid ground truth boxes mask。

        返回:
            align_metric (paddle.Tensor): 结合 classification 与 localization 的 alignment metric。
            overlaps (paddle.Tensor): predicted 与 ground truth boxes 之间的 IoU overlaps。
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = paddle.zeros(
            [self.bs, self.n_max_boxes, na],
            dtype=pd_bboxes.dtype,
            device=pd_bboxes.device,
        )
        bbox_scores = paddle.zeros(
            [self.bs, self.n_max_boxes, na],
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )
        ind = paddle.zeros([2, self.bs, self.n_max_boxes], dtype=paddle.long)
        ind[0] = paddle.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """计算 horizontal bounding boxes 的 IoU。

        参数:
            gt_bboxes (paddle.Tensor): Ground truth boxes。
            pd_bboxes (paddle.Tensor): Predicted boxes。

        返回:
            (paddle.Tensor): 每对 boxes 之间的 IoU values。
        """
        return paddle.clip(bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1), min=0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """基于给定 metrics 选择 top-k candidates。

        参数:
            metrics (paddle.Tensor): shape 为 (b, max_num_obj, h*w) 的 tensor，其中 b 是 batch size，
                max_num_obj 是最大 objects 数量，h*w 表示 anchor points 总数。
            topk_mask (paddle.Tensor, optional): shape 为 (b, max_num_obj, topk) 的可选 boolean tensor，
                其中 topk 是参与考虑的 top candidates 数量。未提供时会基于给定 metrics 自动计算 top-k values。

        返回:
            (paddle.Tensor): shape 为 (b, max_num_obj, h*w)、包含已选择 top-k candidates 的 tensor。
        """
        topk_metrics, topk_idxs = paddle.topk(metrics, self.topk, axis=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics._max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)
        count_tensor = paddle.zeros(metrics.shape, dtype=paddle.int32, device=topk_idxs.device)
        ones = paddle.ones_like(topk_idxs[:, :, :1], dtype=paddle.int32, device=topk_idxs.device)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """为 positive anchor points 计算 target labels、target bounding boxes 和 target scores。

        参数:
            gt_labels (paddle.Tensor): shape 为 (b, max_num_obj, 1) 的 ground truth labels，其中 b 是 batch size，
                max_num_obj 是最大 objects 数量。
            gt_bboxes (paddle.Tensor): shape 为 (b, max_num_obj, 4) 的 ground truth bounding boxes。
            target_gt_idx (paddle.Tensor): positive anchor points 分配到的 ground truth objects indices，
                shape 为 (b, h*w)，其中 h*w 是 anchor points 总数。
            fg_mask (paddle.Tensor): shape 为 (b, h*w) 的 boolean tensor，表示 positive (foreground) anchor points。

        返回:
            target_labels (paddle.Tensor): positive anchor points 的 target labels，shape 为 (b, h*w)。
            target_bboxes (paddle.Tensor): positive anchor points 的 target bounding boxes，shape 为 (b, h*w, 4)。
            target_scores (paddle.Tensor): positive anchor points 的 target scores，shape 为 (b, h*w, num_classes)。
        """
        batch_ind = paddle.arange(end=self.bs, dtype=paddle.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        target_labels = paddle.clip(target_labels, min=0, max=paddle.iinfo(target_labels.dtype).max)
        target_scores = paddle.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=paddle.int64,
            device=target_labels.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = paddle.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-09):
        """选择位于 ground truth bounding boxes 内的 positive anchor centers。

        参数:
            xy_centers (paddle.Tensor): anchor center coordinates，shape 为 (h*w, 2)。
            gt_bboxes (paddle.Tensor): ground truth bounding boxes，shape 为 (b, n_boxes, 4)。
            mask_gt (paddle.Tensor): valid ground truth boxes mask，shape 为 (b, n_boxes, 1)。
            eps (float, optional): 用于 numerical stability 的小值。

        返回:
            (paddle.Tensor): positive anchors 的 boolean mask，shape 为 (b, n_boxes, h*w)。

        说明:
            - b: batch size，n_boxes: ground truth boxes 数量，h: height，w: width。
            - Bounding box format: [x_min, y_min, x_max, y_max]。
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]
        gt_bboxes_xywh[..., 2:] = paddle.where(
            paddle.logical_and(wh_mask, mask_gt.bool()),
            paddle.tensor(
                self.stride_val,
                dtype=gt_bboxes_xywh.dtype,
                device=gt_bboxes_xywh.device,
            ),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = paddle.cat((xy_centers[None] - lt, rb - xy_centers[None]), axis=2).view(
            bs, n_boxes, n_anchors, -1
        )
        bbox_deltas_min = bbox_deltas.amin(3)
        return (bbox_deltas_min > eps).astype(bbox_deltas_min.dtype)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """当 anchor boxes 被分配给多个 ground truths 时，选择 IoU 最高者。

        参数:
            mask_pos (paddle.Tensor): positive mask，shape 为 (b, n_max_boxes, h*w)。
            overlaps (paddle.Tensor): IoU overlaps，shape 为 (b, n_max_boxes, h*w)。
            n_max_boxes (int): ground truth boxes 最大数量。
            align_metric (paddle.Tensor): 用于选择 best matches 的 alignment metric。

        返回:
            target_gt_idx (paddle.Tensor): assigned ground truths 的 indices，shape 为 (b, h*w)。
            fg_mask (paddle.Tensor): foreground mask，shape 为 (b, h*w)。
            mask_pos (paddle.Tensor): 更新后的 positive mask，shape 为 (b, n_max_boxes, h*w)。
        """
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = paddle.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = paddle.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos
            max_overlaps_idx = paddle.topk(align_metric, self.topk2, axis=-1, largest=True).indices
            topk_idx = paddle.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """使用 task-aligned metric 将 ground-truth objects 分配给 rotated bounding boxes。"""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """计算 rotated bounding boxes 的 IoU。"""
        return paddle.clip(probiou(gt_bboxes, pd_bboxes).squeeze(-1), min=0)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt):
        """为 rotated bounding boxes 选择 gt 内的 positive anchor center。

        参数:
            xy_centers (paddle.Tensor): shape 为 (h*w, 2) 的 anchor center coordinates。
            gt_bboxes (paddle.Tensor): shape 为 (b, n_boxes, 5) 的 ground truth bounding boxes。
            mask_gt (paddle.Tensor): shape 为 (b, n_boxes, 1) 的 valid ground truth boxes mask。

        返回:
            (paddle.Tensor): shape 为 (b, n_boxes, h*w) 的 positive anchors boolean mask。
        """
        wh_mask = gt_bboxes[..., 2:4] < self.stride[0]
        gt_bboxes[..., 2:4] = paddle.where(
            (wh_mask * mask_gt).bool(),
            paddle.tensor(self.stride_val, dtype=gt_bboxes.dtype, device=gt_bboxes.device),
            gt_bboxes[..., 2:4],
        )
        corners = xywhr2xyxyxyxy(gt_bboxes)
        a, b, _, d = corners.split(1, axis=-2)
        ab = b - a
        ad = d - a
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(axis=-1)
        norm_ad = (ad * ad).sum(axis=-1)
        ap_dot_ab = (ap * ab).sum(axis=-1)
        ap_dot_ad = (ap * ad).sum(axis=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """根据 features 生成 anchors。"""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = paddle.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = paddle.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = paddle.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(paddle.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(paddle.full((h * w, 1), stride, dtype=dtype, device=device))
    return paddle.cat(anchor_points), paddle.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """将 distance(ltrb) 转为 box(xywh 或 xyxy)。"""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return paddle.cat([c_xy, wh], dim)
    return paddle.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points: paddle.Tensor, bbox: paddle.Tensor, reg_max: (int | None) = None) -> paddle.Tensor:
    """将 bbox(xyxy) 转为 dist(ltrb)。"""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = paddle.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = paddle.clip(dist, min=0, max=reg_max - 0.01)
    return dist


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """根据 anchor points 与 distribution 解码 predicted rotated bounding box coordinates。

    参数:
        pred_dist (paddle.Tensor): shape 为 (bs, h*w, 4) 的 predicted rotated distance。
        pred_angle (paddle.Tensor): shape 为 (bs, h*w, 1) 的 predicted angle。
        anchor_points (paddle.Tensor): shape 为 (h*w, 2) 的 anchor points。
        dim (int, optional): 执行 split 的 dimension。

    返回:
        (paddle.Tensor): shape 为 (bs, h*w, 4) 的 predicted rotated bounding boxes。
    """
    lt, rb = pred_dist.split(2, axis=dim)
    cos, sin = paddle.cos(pred_angle), paddle.sin(pred_angle)
    xf, yf = ((rb - lt) / 2).split(1, axis=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = paddle.cat([x, y], axis=dim) + anchor_points
    return paddle.cat([xy, lt + rb], axis=dim)


def rbox2dist(
    target_bboxes: paddle.Tensor,
    anchor_points: paddle.Tensor,
    target_angle: paddle.Tensor,
    dim: int = -1,
    reg_max: (int | None) = None,
):
    """将 rotated bounding box (xywh) 转为 distance (ltrb)，是 dist2rbox 的逆操作。

    参数:
        target_bboxes (paddle.Tensor): shape 为 (bs, h*w, 4) 的 target rotated bounding boxes，format 为 [x, y, w, h]。
        anchor_points (paddle.Tensor): shape 为 (h*w, 2) 的 anchor points。
        target_angle (paddle.Tensor): shape 为 (bs, h*w, 1) 的 target angle。
        dim (int, optional): 执行 split 的 dimension。
        reg_max (int, optional): 用于 clamping 的 maximum regression value。

    返回:
        (paddle.Tensor): shape 为 (bs, h*w, 4) 的 rotated distance，format 为 [l, t, r, b]。
    """
    xy, wh = target_bboxes.split(2, axis=dim)
    offset = xy - anchor_points
    offset_x, offset_y = offset.split(1, axis=dim)
    cos, sin = paddle.cos(target_angle), paddle.sin(target_angle)
    xf = offset_x * cos + offset_y * sin
    yf = -offset_x * sin + offset_y * cos
    w, h = wh.split(1, axis=dim)
    target_l = w / 2 - xf
    target_t = h / 2 - yf
    target_r = w / 2 + xf
    target_b = h / 2 + yf
    dist = paddle.cat([target_l, target_t, target_r, target_b], axis=dim)
    if reg_max is not None:
        dist = paddle.clip(dist, min=0, max=reg_max - 0.01)
    return dist
