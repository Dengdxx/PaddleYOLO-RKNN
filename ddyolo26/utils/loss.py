# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 损失函数：涵盖 NMS-free 端到端训练所需的所有损失计算。
@details
核心损失类：
- `v8DetectionLoss`：经典 TAL（Task-Aligned）one-to-many 检测损失
- `E2EDetectLoss`：端到端双头损失（one2one 匈牙利 + one2many TAL 联合训练）
- `BboxLoss`：IoU 损失 + DFL 损失（YOLO26 reg_max=1 时 DFL 降为 L1）
- `VarifocalLoss` / `FocalLoss`：分类损失变体，用于处理正负样本不均衡
- `RLELoss`：残差对数似然估计损失（姿态估计精确关键点定位）

YOLO26 无 DFL 说明：BboxLoss 在 `reg_max=1` 时将 DFL 替换为标准化 L1 损失，
不改变训练目标语义但简化了模型结构与导出图。
"""

from __future__ import annotations
import sys
import paddle


import math
from typing import Any


from ddyolo26.paddle_utils import *

from ddyolo26.utils.metrics import OKS_SIGMA, RLE_WEIGHT
from ddyolo26.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ddyolo26.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ddyolo26.utils.runtime import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


class VarifocalLoss(paddle.nn.Module):
    """Zhang 等提出的 Varifocal loss。

    实现 Varifocal Loss，用于在 object detection 中通过关注 hard-to-classify examples 并平衡 positive/negative
    samples 来处理 class imbalance。

    属性:
        gamma (float): focusing parameter，控制 loss 对 hard-to-classify examples 的关注程度。
        alpha (float): 用于处理 class imbalance 的 balancing factor。

    参考:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """使用 focusing 与 balancing 参数初始化 VarifocalLoss 类。"""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: paddle.Tensor, gt_score: paddle.Tensor, label: paddle.Tensor) -> paddle.Tensor:
        """计算 predictions 与 ground truth 之间的 varifocal loss。"""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (
                    paddle.nn.functional.binary_cross_entropy_with_logits(
                        logit=pred_score.float(),
                        label=gt_score.float(),
                        reduction="none",
                    )
                    * weight
                )
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(paddle.nn.Module):
    """在现有 loss_fcn() 外包装 focal loss，例如 criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)。

    实现 Focal Loss，通过降低 easy examples 权重并在 training 中关注 hard negatives 来处理 class imbalance。

    属性:
        gamma (float): focusing parameter，控制 loss 对 hard-to-classify examples 的关注程度。
        alpha (paddle.Tensor): 用于处理 class imbalance 的 balancing factor。
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """使用 focusing 与 balancing 参数初始化 FocalLoss 类。"""
        super().__init__()
        self.gamma = gamma
        self.alpha = paddle.tensor(alpha)

    def forward(self, pred: paddle.Tensor, label: paddle.Tensor) -> paddle.Tensor:
        """使用 class imbalance 的 modulating factors 计算 focal loss。"""
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(logit=pred, label=label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(paddle.nn.Module):
    """用于计算 Distribution Focal Loss（DFL）的 criterion 类。"""

    def __init__(self, reg_max: int = 16) -> None:
        """使用 regularization maximum 初始化 DFL module。"""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        """返回 https://ieeexplore.ieee.org/document/9792391 中左右 DFL losses 之和。"""
        target = paddle.clip(target, min=0, max=self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr.astype(target.dtype) - target
        wr = 1 - wl
        return (
            paddle.nn.functional.cross_entropy(input=pred_dist, label=tl.view(-1), reduction="none").view(tl.shape) * wl
            + paddle.nn.functional.cross_entropy(input=pred_dist, label=tr.view(-1), reduction="none").view(tl.shape)
            * wr
        ).mean(-1, keepdim=True)


class BboxLoss(paddle.nn.Module):
    """用于计算 bounding boxes training losses 的 criterion 类。"""

    def __init__(self, reg_max: int = 16):
        """使用 regularization maximum 与 DFL settings 初始化 BboxLoss module。"""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: paddle.Tensor,
        pred_bboxes: paddle.Tensor,
        anchor_points: paddle.Tensor,
        target_bboxes: paddle.Tensor,
        target_scores: paddle.Tensor,
        target_scores_sum: paddle.Tensor,
        fg_mask: paddle.Tensor,
        imgsz: paddle.Tensor,
        stride: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """计算 bounding boxes 的 IoU 与 DFL losses。"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            scale_tensor = paddle.cast(
                paddle.stack([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], axis=-1), dtype=target_ltrb.dtype
            )
            target_ltrb = target_ltrb * stride / scale_tensor
            pred_dist = pred_dist * stride / scale_tensor
            loss_dfl = (
                paddle.nn.functional.l1_loss(
                    input=pred_dist[fg_mask],
                    label=target_ltrb[fg_mask],
                    reduction="none",
                ).mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl


class RLELoss(paddle.nn.Module):
    """Residual Log-Likelihood Estimation 损失。

    属性:
        size_average (bool): 是否按 batch_size 平均 loss。
        use_target_weight (bool): 是否使用 weighted loss。
        residual (bool): 是否添加 L1 loss，并让 flow 学习 residual error distribution。

    参考:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(
        self,
        use_target_weight: bool = True,
        size_average: bool = True,
        residual: bool = True,
    ):
        """使用 target weight 与 residual options 初始化 RLELoss。

        参数:
            use_target_weight (bool): loss calculation 是否使用 target weights。
            size_average (bool): 是否对 elements 上的 loss 求平均。
            residual (bool): 是否包含 residual log-likelihood term。
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self,
        sigma: paddle.Tensor,
        log_phi: paddle.Tensor,
        error: paddle.Tensor,
        target_weight: paddle.Tensor = None,
    ) -> paddle.Tensor:
        """
        参数:
            sigma (paddle.Tensor): output sigma，shape 为 (N, D)。
            log_phi (paddle.Tensor): output log_phi，shape 为 (N)。
            error (paddle.Tensor): error，shape 为 (N, D)。
            target_weight (paddle.Tensor): 不同 joint types 的 weights，shape 为 (N)。
        """
        log_sigma = paddle.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)
        if self.residual:
            loss += paddle.log(sigma * 2) + paddle.abs(error)
        if self.use_target_weight:
            assert target_weight is not None, "'use_target_weight' 为 True 时，'target_weight' 不应为 None。"
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight
        if self.size_average:
            loss /= len(loss)
        return loss.sum()


class v8DetectionLoss:
    """用于计算 YOLOv8 object detection training losses 的 criterion 类。"""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: (int | None) = None):
        """使用 model parameters 与 task-aligned assignment settings 初始化 v8DetectionLoss。"""
        device = model.parameters()[0].place
        h = model.args
        m = model.model[-1]
        self.bce = paddle.nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = paddle.arange(m.reg_max, dtype=paddle.float32, device=device)

    def preprocess(self, targets: paddle.Tensor, batch_size: int, scale_tensor: paddle.Tensor) -> paddle.Tensor:
        """将 targets 转为 tensor format 并 scale coordinates，完成 preprocessing。"""
        nl, ne = targets.shape
        if nl == 0:
            out = paddle.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=paddle.int32)
            out = paddle.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: paddle.Tensor, pred_dist: paddle.Tensor) -> paddle.Tensor:
        """根据 anchor points 与 distribution 解码 predicted object bbox coordinates。"""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.astype(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, paddle.Tensor], batch: dict[str, Any]) -> tuple:
        """计算 box、cls 与 dfl loss 之和，并返回 foreground mask 与 target indices。"""
        loss = paddle.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = paddle.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        bboxes = batch["bboxes"]
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape([0, 4])
        targets = paddle.cat(
            (batch["batch_idx"].view(-1, 1).cast(bboxes.dtype), batch["cls"].view(-1, 1), bboxes),
            1,
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = (gt_bboxes.sum(2, keepdim=True) > 0).astype(gt_bboxes.dtype)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).astype(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )

    def parse_output(
        self,
        preds: (dict[str, paddle.Tensor] | tuple[paddle.Tensor, dict[str, paddle.Tensor]]),
    ) -> paddle.Tensor:
        """解析 model predictions 以提取 features。"""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: (dict[str, paddle.Tensor] | tuple[paddle.Tensor, dict[str, paddle.Tensor]]),
        batch: dict[str, paddle.Tensor],
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """计算 box、cls 与 dfl loss 之和并乘以 batch size。"""
        return self.loss(self.parse_output(preds), batch)

    def loss(
        self, preds: dict[str, paddle.Tensor], batch: dict[str, paddle.Tensor]
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """使用 assigned targets 计算 detection loss。"""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class E2EDetectLoss:
    """用于计算 end-to-end detection training losses 的 criterion 类。"""

    def __init__(self, model):
        """使用给定 model 初始化包含 one-to-many 与 one-to-one detection losses 的 E2EDetectLoss。"""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, paddle.Tensor]) -> tuple[paddle.Tensor, paddle.Tensor]:
        """计算 box、cls 与 dfl loss 之和并乘以 batch size。"""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class MultiChannelDiceLoss(paddle.nn.Module):
    """多通道 Dice 损失，用于分割掩码训练。"""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """初始化 MultiChannelDiceLoss。

        参数:
            smooth (float): 平滑因子，避免除零。
            reduction (str): 归约方式（'mean'、'sum' 或 'none'）。
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        """计算预测与目标之间的多通道 Dice 损失。"""
        pred = pred.sigmoid()
        intersection = (pred * target).sum(axis=(2, 3))
        union = pred.sum(axis=(2, 3)) + target.sum(axis=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(axis=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(paddle.nn.Module):
    """组合 BCE + Dice 损失，用于语义分割。"""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """初始化 BCEDiceLoss。

        参数:
            weight_bce (float): BCE 损失权重。
            weight_dice (float): Dice 损失权重。
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = paddle.nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        """计算组合 BCE + Dice 损失。"""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):
            target = paddle.nn.functional.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class v8SegmentationLoss(v8DetectionLoss):
    """YOLO 分割损失：在检测损失基础上增加实例掩码损失和可选语义掩码损失。"""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: (int | None) = None):
        """初始化 v8SegmentationLoss。

        参数:
            model: 已去并行化的模型。
            tal_topk (int): TAL 匹配 topk。
            tal_topk2 (int | None): TAL 二次 topk。
        """
        super().__init__(model, tal_topk, tal_topk2)
        args = model.args
        self.overlap = args.overlap_mask if hasattr(args, "overlap_mask") else args.get("overlap_mask", True)
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(
        self, preds: dict[str, paddle.Tensor], batch: dict[str, paddle.Tensor]
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """计算检测+分割联合损失。

        返回:
            (paddle.Tensor): 各分量损失 (box, seg, cls, dfl, semseg) * batch_size。
            (paddle.Tensor): 各分量损失（detach，用于日志）。
        """
        pred_masks, proto = preds["mask_coefficient"].transpose([0, 2, 1]), preds["proto"]
        loss = paddle.zeros([5])
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape
        if fg_mask.sum():
            masks = batch["masks"].astype("float32")
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                proto = paddle.nn.functional.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = paddle.to_tensor(list(preds["feats"][0].shape[2:]), dtype=pred_masks.dtype) * self.stride[0]
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].reshape([-1, 1]),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"]
                sem_masks = (
                    paddle.nn.functional.one_hot(sem_masks.astype("int64"), num_classes=self.nc)
                    .transpose((0, 3, 1, 2))
                    .astype("float32")
                )

                if self.overlap:
                    mask_zero = masks == 0
                    sem_masks[mask_zero.unsqueeze(1).expand(sem_masks.shape)] = 0
                else:
                    batch_idx = batch["batch_idx"].reshape([-1])
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(axis=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.hyp.box
        return loss * batch_size, loss.detach()

    @staticmethod
    def single_mask_loss(
        gt_mask: paddle.Tensor,
        pred: paddle.Tensor,
        proto: paddle.Tensor,
        xyxy: paddle.Tensor,
        area: paddle.Tensor,
    ) -> paddle.Tensor:
        """计算单张图像的实例分割损失。

        参数:
            gt_mask (paddle.Tensor): 真值掩码，形状 (N, H, W)。
            pred (paddle.Tensor): 预测掩码系数，形状 (N, 32)。
            proto (paddle.Tensor): 原型掩码，形状 (32, H, W)。
            xyxy (paddle.Tensor): 真值框 xyxy 格式，归一化到 [0,1]，形状 (N, 4)。
            area (paddle.Tensor): 每个真值框面积，形状 (N,)。

        返回:
            (paddle.Tensor): 该图像的掩码损失标量。
        """
        pred_mask = paddle.einsum("in,nhw->ihw", pred, proto)
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(logit=pred_mask, label=gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(axis=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: paddle.Tensor,
        masks: paddle.Tensor,
        target_gt_idx: paddle.Tensor,
        target_bboxes: paddle.Tensor,
        batch_idx: paddle.Tensor,
        proto: paddle.Tensor,
        pred_masks: paddle.Tensor,
        imgsz: paddle.Tensor,
    ) -> paddle.Tensor:
        """计算实例分割损失。

        参数:
            fg_mask (paddle.Tensor): 前景锚点掩码，形状 (BS, N_anchors)。
            masks (paddle.Tensor): 真值掩码。
            target_gt_idx (paddle.Tensor): 每个锚点对应的 GT 索引，形状 (BS, N_anchors)。
            target_bboxes (paddle.Tensor): 锚点对应的 GT 框，形状 (BS, N_anchors, 4)。
            batch_idx (paddle.Tensor): 批次索引，形状 (N_labels, 1)。
            proto (paddle.Tensor): 原型掩码，形状 (BS, 32, H, W)。
            pred_masks (paddle.Tensor): 预测掩码系数，形状 (BS, N_anchors, 32)。
            imgsz (paddle.Tensor): 输入图像尺寸，形状 (2,)。

        返回:
            (paddle.Tensor): 实例分割损失。
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * paddle.to_tensor([mask_w, mask_h, mask_w, mask_h], dtype="float32")

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    # Paddle 需要显式类型对齐：masks 为 uint8/float，mask_idx 为 int64
                    masks_i_int = masks_i.astype("int64")
                    gt_mask = masks_i_int == (mask_idx + 1).reshape([-1, 1, 1])
                    gt_mask = gt_mask.astype("float32")
                else:
                    gt_mask = masks[batch_idx.reshape([-1]) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()

        return loss / fg_mask.sum()


class E2ELoss:
    """用于计算 end-to-end detection training losses 的 criterion 类。"""

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """使用给定 model 初始化包含 one-to-many 与 one-to-one detection losses 的 E2ELoss。"""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, paddle.Tensor]) -> tuple[paddle.Tensor, paddle.Tensor]:
        """计算 box、cls 与 dfl loss 之和并乘以 batch size。"""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """根据 decay schedule 更新 one-to-many 与 one-to-one losses 的 weights。"""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """基于当前 update step 计算 one-to-many loss 的 decayed weight。"""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """用于计算 text-visual prompt detection training losses 的 criterion 类。"""

    def __init__(self, model, tal_topk=10, tal_topk2: (int | None) = None):
        """使用给定 model 初始化带 task-prompt 与 visual-prompt criteria 的 TVPDetectLoss。"""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, paddle.Tensor]:
        """解析 model predictions 以提取 features。"""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, paddle.Tensor]) -> tuple[paddle.Tensor, paddle.Tensor]:
        """计算 text-visual prompt detection loss。"""
        return self.loss(self.parse_output(preds), batch)

    def loss(
        self, preds: dict[str, paddle.Tensor], batch: dict[str, paddle.Tensor]
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """计算 text-visual prompt detection loss。"""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = paddle.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()
        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, paddle.Tensor]) -> list[paddle.Tensor]:
        """从 model output 中提取 visual-prompt features。"""
        scores = preds["scores"]
        vnc = scores.shape[1]
        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores
