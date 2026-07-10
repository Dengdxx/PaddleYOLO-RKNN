# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 检测头实现：NMS-free end-to-end 双头结构。
@details
YOLO26 三大架构特性在此头部完整体现：

【NMS-free / end2end】
- `end2end=True` 时，Detect 创建 one2one_cv2/cv3（匈牙利匹配）分支
- one-to-one head 推理输出 `(N, 300, 6)` — 无需任何 NMS 后处理
- one-to-many head（训练辅助）输出 `(N, nc+4, 8400)`，仅训练期使用

【无 DFL】
- `reg_max=1` 时 `self.dfl = nn.Identity()`，直接回归坐标，移除分布焦点损失

【输出格式兼容】
- `export_raw_one2one=True`：导出 ONNX 时输出 one2one 已解码张量（单输出 `y` / `y+proto`），
    更接近推理侧 `yolo` / `one2one_raw` 契约
- `export_dual_raw=True`：导出 raw boxes/scores[/mask_coeff/proto]，对应当前 RKNN 主线
    `pre_dist` / `seg_pre_dist`
"""

from __future__ import annotations
import sys
import paddle


from ddyolo26.paddle_utils import *

"""模型头模块。"""

import copy
import math

from ddyolo26.utils import NOT_MACOS14
from ddyolo26.utils.tal import dist2bbox, dist2rbox, make_anchors
from ddyolo26.utils.runtime import fuse_conv_and_bn, smart_inference_mode

from .block import DFL, Proto, Proto26, RealNVP, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect", "Segment", "Segment26"


class Detect(paddle.nn.Module):
    """目标检测模型的 YOLO Detect 头。

    该类实现 YOLO 模型用于预测边界框和类别概率的检测头，支持训练与推理模式，并可选支持端到端检测。

    属性:
        dynamic (bool): 强制重建网格。
        export (bool): 导出模式标记。
        format (str): 导出格式。
        end2end (bool): 端到端检测模式。
        max_det (int): 每张图最大检测数。
        shape (tuple): 输入形状。
        anchors (paddle.Tensor): 锚点。
        strides (paddle.Tensor): 特征图 stride。
        legacy (bool): v3/v5/v8/v9/v11 模型向后兼容开关。
        xyxy (bool): 输出格式，xyxy 或 xywh。
        nc (int): 类别数。
        nl (int): 检测层数。
        reg_max (int): DFL 通道数。
        no (int): 每个 anchor 的输出数。
        stride (paddle.Tensor): 构建阶段计算得到的 stride。
        cv2 (nn.ModuleList): 边界框回归卷积层。
        cv3 (nn.ModuleList): 分类卷积层。
        dfl (nn.Module): Distribution Focal Loss 层。
        one2one_cv2 (nn.ModuleList): one-to-one 边界框回归卷积层。
        one2one_cv3 (nn.ModuleList): one-to-one 分类卷积层。

    方法:
        forward: 执行前向传播并返回预测。
        bias_init: 初始化检测头偏置。
        decode_bboxes: 从预测中解码边界框。
        postprocess: 后处理模型预测。

    示例:
        创建 80 类检测头
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [paddle.randn(1, 256, 80, 80), paddle.randn(1, 512, 40, 40), paddle.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False
    export = False
    format = None
    max_det = 300
    agnostic_nms = False
    export_raw_one2one = False
    export_use_one2many = False
    export_dual_raw = False  # 导出双输出 (boxes, scores) — 与 ultralytics pre_dist/pre_dfl 等价
    _feat_shape = None
    anchors = paddle.empty(0)
    strides = paddle.empty(0)
    legacy = False
    xyxy = False

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """使用指定类别数和通道数初始化 YOLO 检测层。

        参数:
            nc (int): 类别数。
            reg_max (int): DFL 通道数上限。
            end2end (bool): 是否使用端到端 NMS-free 检测。
            ch (tuple): 骨干特征图各层通道数。
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.stride = paddle.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = paddle.nn.ModuleList(
            paddle.nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                paddle.nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = (
            paddle.nn.ModuleList(
                paddle.nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), paddle.nn.Conv2d(c3, self.nc, 1)) for x in ch
            )
            if self.legacy
            else paddle.nn.ModuleList(
                paddle.nn.Sequential(
                    paddle.nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    paddle.nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    paddle.nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else paddle.nn.Identity()
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def sync_one2one_heads(self):
        """将 one2one 头重新同步为 one2many 头的深拷贝。"""
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """返回 one-to-many 头组件，用于 v3/v5/v8/v9/v11 向后兼容。"""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """返回 one-to-one 头组件。"""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """检查模型是否具备 one2one 头，用于 v3/v5/v8/v9/v11 向后兼容。"""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """覆盖端到端检测模式。"""
        self._end2end = value

    def forward_head(
        self,
        x: list[paddle.Tensor],
        box_head: paddle.nn.Module = None,
        cls_head: paddle.nn.Module = None,
    ) -> dict[str, paddle.Tensor]:
        """拼接并返回预测边界框和类别概率。"""
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = paddle.cat(
            [box_head[i](x[i]).reshape(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            axis=-1,
        )
        scores = paddle.cat([cls_head[i](x[i]).reshape(bs, self.nc, -1) for i in range(self.nl)], axis=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[paddle.Tensor]
    ) -> dict[str, paddle.Tensor] | paddle.Tensor | tuple[paddle.Tensor, dict[str, paddle.Tensor]]:
        """拼接并返回预测边界框和类别概率。"""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, paddle.Tensor]) -> paddle.Tensor:
        """基于多层特征图解码预测边界框和类别概率。

        参数:
            x (dict[str, paddle.Tensor]): 检测层输出的预测字典。

        返回:
            (paddle.Tensor): 解码后边界框和类别概率拼接而成的张量。
        """
        dbox = self._get_decode_boxes(x)
        return paddle.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, paddle.Tensor]) -> paddle.Tensor:
        """基于 anchors 和 strides 获取解码后的边界框。"""
        shape = x["feats"][0].shape
        if self.dynamic or self._feat_shape != shape or self.anchors.numel() == 0:
            _anchors, _strides = make_anchors(x["feats"], self.stride, 0.5)
            self.__dict__["anchors"] = _anchors.transpose([1, 0])
            self.__dict__["strides"] = _strides.transpose([1, 0])
            self._feat_shape = shape
        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def _get_decode_boxes_export(self, x: dict[str, paddle.Tensor]) -> paddle.Tensor:
        """导出友好的边界框解码，使用即时计算的 anchor。

        仅使用与 paddle.jit.to_static 兼容的操作从特征图计算 anchors，
        避免 self.__dict__ 赋值以及原始 make_anchors() 中的 .view() 和 device= kwargs。
        """
        # 从特征图计算 anchors（静态图兼容）
        anchor_points, stride_tensor = [], []
        dtype = x["feats"][0].dtype
        for i in range(self.nl):
            feat = x["feats"][i]
            h, w = feat.shape[2], feat.shape[3]
            s = self.stride[i]
            sx = paddle.arange(end=w).astype(dtype) + 0.5
            sy = paddle.arange(end=h).astype(dtype) + 0.5
            sy, sx = paddle.meshgrid(sy, sx)
            anchor_points.append(paddle.stack((sx, sy), -1).reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], s, dtype=dtype))
        anchors = paddle.concat(anchor_points).transpose([1, 0])
        strides = paddle.concat(stride_tensor).transpose([1, 0])
        dbox = self.decode_bboxes(self.dfl(x["boxes"]), anchors.unsqueeze(0)) * strides
        return dbox

    def bias_init(self):
        """初始化 Detect() 偏置。注意：需要 stride 已可用。"""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def decode_bboxes(self, bboxes: paddle.Tensor, anchors: paddle.Tensor, xywh: bool = True) -> paddle.Tensor:
        """从预测中解码边界框。"""
        return dist2bbox(bboxes, anchors, xywh=xywh and not self.end2end and not self.xyxy, dim=1)

    def postprocess(self, preds: paddle.Tensor) -> paddle.Tensor:
        """后处理 YOLO 模型预测。

        参数:
            preds (paddle.Tensor): 原始预测，形状为 (batch_size, num_anchors, 4 + nc)，最后一维格式为
                [x1, y1, x2, y2, class_probs]。

        返回:
            (paddle.Tensor): 处理后的预测，形状为 (batch_size, min(max_det, num_anchors), 6)，最后一维格式为
                [x1, y1, x2, y2, max_class_prob, class_index]。
        """
        boxes, scores = preds.split([4, self.nc], axis=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = paddle.take_along_axis(boxes, idx.repeat(1, 1, 4), axis=1)
        return paddle.cat([boxes, scores, conf], axis=-1)

    def get_topk_index(self, scores: paddle.Tensor, max_det: int) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """从 scores 中获取 top-k 索引。

        参数:
            scores (paddle.Tensor): 分数张量，形状为 (batch_size, num_anchors, num_classes)。
            max_det (int): 每张图最大检测数。

        返回:
            (paddle.Tensor, paddle.Tensor, paddle.Tensor): top 分数、类别索引和筛选后的索引。
        """
        batch_size, anchors, nc = scores.shape
        k = max_det if self.export else min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(keepdim=True, axis=-1), scores.argmax(keepdim=True, axis=-1)
            scores, indices = scores.topk(k, axis=1)
            labels = paddle.take_along_axis(labels, indices, axis=1)
            return scores, labels, indices
        ori_index = (scores.max(axis=-1), scores.argmax(axis=-1))[0].topk(k)[1].unsqueeze(-1)
        scores = paddle.take_along_axis(scores, ori_index.repeat(1, 1, nc), axis=1)
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[paddle.arange(batch_size)[..., None], index // nc]
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """移除 one2many 头以优化推理。"""
        if not self.end2end or self.export_use_one2many:
            # 非 end2end 或使用 one2many 头导出时，保留 cv2/cv3
            return
        self.cv2 = self.cv3 = None

    def _forward_export(self, x: list[paddle.Tensor]) -> paddle.Tensor:
        """导出友好的前向传播，兼容 paddle.jit.to_static。

        避免 paddle2onnx 无法转换的 .detach()（share_data_）、花式索引（broadcast_tensors）和取模（remainder）。
        """
        # 选择检测头：非 end2end 模型（如 YOLOv8）只有 one2many 头
        if not self.end2end or self.export_use_one2many:
            head_dict = self.one2many
        else:
            head_dict = self.one2one
        preds = self.forward_head(x, **head_dict)
        if self.export_dual_raw:
            # 双输出原始张量（与 ultralytics pre_dist / pre_dfl 等价）
            #   boxes  [B, 4*reg_max, A]   reg_max=1 → pre_dist；reg_max=16 → pre_dfl
            #   scores [B, nc, A]          未做 sigmoid
            return preds["boxes"], preds["scores"]
        y = self._inference(preds)
        if self.export_raw_one2one:
            return y
        y = self._postprocess_export(y.permute(0, 2, 1))
        return y

    def _forward_export_one2one_raw(self, x: list[paddle.Tensor]) -> paddle.Tensor:
        """导出 one2one 解码预测，不包含 e2e 后处理尾部。

        返回形状为 [B, 4 + nc, A] 的单个张量，其中 A 是各层 anchor 总数。
        该路径保留 one2one 分支和单输出契约，同时避开 RKNN 容易出问题的 TopK/Gather 导出尾部。
        """
        preds = self.forward_head(x, **self.one2one)
        return self._inference(preds)

    def _postprocess_export(self, preds: paddle.Tensor) -> paddle.Tensor:
        """导出友好的后处理，避开不支持的算子。"""
        boxes, scores = preds.split([4, self.nc], axis=-1)
        batch_size = scores.shape[0]
        nc = self.nc
        k = self.max_det

        # 步骤 1：按最大类别分数选出 top-k anchors
        score_max = scores.max(axis=-1)  # [B, A]
        _, ori_index = score_max.topk(k)  # [B, k]
        ori_index_3d = ori_index.unsqueeze(-1)  # [B, k, 1]

        # 步骤 2：收集 top-k anchors 的类别分数
        scores = paddle.take_along_axis(scores, ori_index_3d.expand([-1, -1, nc]), axis=1)  # [B, k, nc]

        # 步骤 3：展平并在所有类别上取 top-k
        scores_flat = scores.reshape([batch_size, -1])  # [B, k*nc]
        scores_topk, flat_index = scores_flat.topk(k)  # [B, k]

        # 步骤 4：还原 anchor 和类别索引（避免 % 和花式索引）
        anchor_idx = flat_index // nc  # [B, k] - top-k anchors 内的第几个 anchor
        class_idx = flat_index - anchor_idx * nc  # [B, k] - 类别索引（避免 %）

        # 步骤 5：将 top-k anchor 索引映射回原始 anchor 索引
        # 使用 take_along_axis 替代花式索引（避免 broadcast_tensors）
        final_idx = paddle.take_along_axis(ori_index, anchor_idx, axis=1).unsqueeze(-1)  # [B, k, 1]

        # 步骤 6：收集 boxes
        boxes = paddle.take_along_axis(boxes, final_idx.expand([-1, -1, 4]), axis=1)

        return paddle.concat([boxes, scores_topk.unsqueeze(-1), class_idx.unsqueeze(-1).astype("float32")], axis=-1)


class Segment(Detect):
    """YOLO Segment 头，用于实例分割模型。

    继承 Detect 头，增加 mask 预测能力（Proto + mask_coefficient）。

    属性:
        nm (int): mask 数量。
        npr (int): proto 数量。
        proto (Proto): Prototype 生成模块。
        cv4 (nn.ModuleList): mask 系数卷积层。
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """初始化 Segment 头。

        参数:
            nc (int): 类别数。
            nm (int): mask 数量。
            npr (int): proto 数量。
            reg_max (int): DFL 通道数上限。
            end2end (bool): 是否端到端检测。
            ch (tuple): 骨干特征图各层通道数。
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm  # mask 数量
        self.npr = npr  # proto 数量
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = paddle.nn.LayerList(
            [paddle.nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), paddle.nn.Conv2D(c4, self.nm, 1)) for x in ch]
        )
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    def sync_one2one_heads(self):
        """将 one2one 检测与分割头重新同步为 one2many 头的深拷贝。"""
        super().sync_one2one_heads()
        if self.end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """返回 one-to-many 头组件（含 mask_head）。"""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """返回 one-to-one 头组件（含 mask_head）。"""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward(self, x: list[paddle.Tensor]) -> tuple | list[paddle.Tensor] | dict[str, paddle.Tensor]:
        """前向传播，返回检测输出和 mask 原型。"""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # 训练 + 训练中验证
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, paddle.Tensor]) -> paddle.Tensor:
        """解码预测框和类别概率，拼接 mask 系数。

        内联 Detect._inference 避免 super() 调用导致 dy2static 递归。
        """
        dbox = self._get_decode_boxes(x)
        preds = paddle.cat((dbox, x["scores"].sigmoid()), 1)
        return paddle.concat([preds, x["mask_coefficient"]], axis=1)

    def forward_head(
        self, x: list[paddle.Tensor], box_head=None, cls_head=None, mask_head=None
    ) -> dict[str, paddle.Tensor]:
        """拼接并返回预测框、类别概率和 mask 系数。

        内联 Detect.forward_head 避免 super() 调用导致 Paddle dy2static 递归溢出。
        """
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = paddle.cat(
            [box_head[i](x[i]).reshape(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            axis=-1,
        )
        scores = paddle.cat([cls_head[i](x[i]).reshape(bs, self.nc, -1) for i in range(self.nl)], axis=-1)
        preds = dict(boxes=boxes, scores=scores, feats=x)
        if mask_head is not None:
            preds["mask_coefficient"] = paddle.concat(
                [mask_head[i](x[i]).reshape([bs, self.nm, -1]) for i in range(self.nl)], axis=2
            )
        return preds

    def postprocess(self, preds: paddle.Tensor) -> paddle.Tensor:
        """后处理 YOLO 分割模型预测结果。

        参数:
            preds (paddle.Tensor): 原始预测，形状 (batch_size, num_anchors, 4 + nc + nm)。

        返回:
            (paddle.Tensor): 处理后预测，形状 (batch_size, min(max_det, num_anchors), 6 + nm)。
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], axis=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = paddle.take_along_axis(boxes, idx.repeat(1, 1, 4), axis=1)
        mask_coefficient = paddle.take_along_axis(mask_coefficient, idx.repeat(1, 1, self.nm), axis=1)
        return paddle.concat([boxes, scores, conf, mask_coefficient], axis=-1)

    def fuse(self) -> None:
        """移除 one2many 头以优化推理。"""
        if not self.end2end or self.export_use_one2many:
            return
        self.cv2 = self.cv3 = self.cv4 = None

    def _forward_export(self, x: list[paddle.Tensor]) -> paddle.Tensor:
        """导出友好的前向传播，返回检测 + mask proto。"""
        if not self.end2end or self.export_use_one2many:
            head_dict = self.one2many
        else:
            head_dict = self.one2one
        preds = self.forward_head(x, **head_dict)
        proto = self.proto(x[0])
        if self.export_dual_raw:
            # 四输出仅作为导出瞬态张量，后续统一追加 score_sum 形成五输出。
            #   boxes        [B, 4*reg_max, A]
            #   scores       [B, nc, A]                (未 sigmoid)
            #   mask_coeff   [B, nm, A]
            #   proto        [B, nm, H/4, W/4]
            return preds["boxes"], preds["scores"], preds["mask_coefficient"], proto
        y = self._inference(preds)
        if self.export_raw_one2one:
            return y, proto
        y = self._postprocess_export(y.permute(0, 2, 1))
        return y, proto

    def _postprocess_export(self, preds: paddle.Tensor) -> paddle.Tensor:
        """导出友好的后处理，保留 mask 系数。"""
        boxes, scores, mc = preds.split([4, self.nc, self.nm], axis=-1)
        batch_size = scores.shape[0]
        nc = self.nc
        k = self.max_det

        score_max = scores.max(axis=-1)
        _, ori_index = score_max.topk(k)
        ori_index_3d = ori_index.unsqueeze(-1)

        scores = paddle.take_along_axis(scores, ori_index_3d.expand([-1, -1, nc]), axis=1)
        mc = paddle.take_along_axis(mc, ori_index_3d.expand([-1, -1, self.nm]), axis=1)

        scores_flat = scores.reshape([batch_size, -1])
        scores_topk, flat_index = scores_flat.topk(k)

        anchor_idx = flat_index // nc
        class_idx = flat_index - anchor_idx * nc

        final_idx = paddle.take_along_axis(ori_index, anchor_idx, axis=1).unsqueeze(-1)

        boxes = paddle.take_along_axis(boxes, final_idx.expand([-1, -1, 4]), axis=1)
        # mc 已在步骤 2 先裁剪到 top-k anchors，后续只能使用 anchor_idx（范围 [0, k)）
        # 不能再用映射回原始 anchors 的 final_idx / ori_index，否则导出 ONNX 后 GatherElements
        # 会拿 [0, A) 的索引去访问长度仅为 k 的轴，导致运行时 out-of-range。
        mc = paddle.take_along_axis(mc, anchor_idx.unsqueeze(-1).expand([-1, -1, self.nm]), axis=1)

        return paddle.concat([boxes, scores_topk.unsqueeze(-1), class_idx.unsqueeze(-1).astype("float32"), mc], axis=-1)


class Segment26(Segment):
    """YOLO26 Segment 头，使用 Proto26 多尺度 Prototype 生成。

    继承 Segment，将 Proto 替换为 Proto26 以支持多尺度特征融合。

    属性:
        nm (int): mask 数量。
        npr (int): proto 数量。
        proto (Proto26): 多尺度 Prototype 生成模块。
        cv4 (nn.ModuleList): mask 系数卷积层。
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """初始化 YOLO26 Segment 头。

        参数:
            nc (int): 类别数。
            nm (int): mask 数量。
            npr (int): proto 数量。
            reg_max (int): DFL 通道数上限。
            end2end (bool): 是否端到端检测。
            ch (tuple): 骨干特征图各层通道数。
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # 多尺度 protos

    def forward(self, x: list[paddle.Tensor]) -> tuple | list[paddle.Tensor] | dict[str, paddle.Tensor]:
        """前向传播，返回检测输出和多尺度 mask 原型。"""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # mask protos（多尺度）
        if isinstance(preds, dict):  # 训练 + 训练中验证
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        """移除 one2many 头和 proto 语义分割辅助头以优化推理。"""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()

    def _forward_export(self, x: list[paddle.Tensor]) -> paddle.Tensor:
        """导出友好的前向传播，Proto26 接收多尺度特征。"""
        # 选择检测头：非 end2end 模型只有 one2many 头
        if not self.end2end or self.export_use_one2many:
            head_dict = self.one2many
        else:
            head_dict = self.one2one
        preds = self.forward_head(x, **head_dict)
        proto = self.proto(x)  # Proto26 接收 list[Tensor]
        if self.export_dual_raw:
            return preds["boxes"], preds["scores"], preds["mask_coefficient"], proto
        y = self._inference(preds)
        if self.export_raw_one2one:
            return y, proto
        y = self._postprocess_export(y.permute(0, 2, 1))
        return y, proto
