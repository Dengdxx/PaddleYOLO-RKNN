# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 分割推理器 SegmentationPredictor。
@details
继承 DetectionPredictor，重写 `postprocess()` 处理分割模型输出：
- 从模型输出中提取 proto 掩码
- 结合掩码系数生成最终实例掩码
- 封装为包含 boxes + masks 的 Results 对象
"""

from __future__ import annotations

import paddle

from ddyolo26.engine.predictor import BasePredictor
from ddyolo26.engine.results import Results
from ddyolo26.models.yolo.detect.predict import DetectionPredictor
from ddyolo26.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """分割推理器，在检测推理器基础上处理掩码输出。

    属性:
        args (namespace): 推理配置参数。
        model: 加载的分割模型。

    示例:
        >>> from ddyolo26.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="weights/yolov8seg/yolov8n-seg.pdparams", source="test.jpg")
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 SegmentationPredictor。

        参数:
            cfg (dict): 推理配置。
            overrides (dict, optional): 覆盖配置。
            _callbacks (list, optional): 回调函数列表。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """后处理分割预测结果：提取 proto 并生成实例掩码。

        参数:
            preds (tuple): 模型输出，包含检测结果和掩码原型。
            img (paddle.Tensor): 预处理后的输入图像张量，形状 (B, C, H, W)。
            orig_imgs (list | paddle.Tensor | np.ndarray): 原始图像。
            **kwargs: 额外关键字参数。

        返回:
            (list[Results]): 包含检测框和分割掩码的 Results 对象列表。
        """
        # 提取 proto：PyTorch 模型为 tuple，导出模型为 array
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        """构建包含掩码的 Results 对象列表。

        参数:
            preds (list[paddle.Tensor]): NMS 后的检测结果列表。
            img (paddle.Tensor): 预处理后的图像。
            orig_imgs (list[np.ndarray]): 原始图像列表。
            protos (paddle.Tensor): 原型掩码张量，形状 (B, C, H, W)。

        返回:
            (list[Results]): Results 对象列表。
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """构建单张图像的 Results 对象。

        参数:
            pred (paddle.Tensor): 检测结果张量。
            img (paddle.Tensor): 预处理后的图像。
            orig_img (np.ndarray): 原始图像。
            img_path (str): 原始图像路径。
            proto (paddle.Tensor): 该图像的原型掩码。

        返回:
            (Results): 包含检测框和掩码的 Results 对象。
        """
        if pred.shape[0] == 0:
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            # 仅保留有效掩码的预测（掩码最大值 > 0）
            keep = masks.reshape([masks.shape[0], -1]).max(axis=-1) > 0
            if not keep.all():
                pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
