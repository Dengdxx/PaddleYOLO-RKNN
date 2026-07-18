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

import numpy as np
import paddle

from ddyolo26.engine.predictor import BasePredictor
from ddyolo26.engine.results import Results
from ddyolo26.models.yolo.detect.predict import DetectionPredictor
from ddyolo26.utils import DEFAULT_CFG, ops


def _is_rknn_seg_outputs(preds) -> bool:
    """!
    @brief 判断后端结果是否为支持的 RKNN 分割输出。
    @param preds 后端返回的原始输出对象。
    @return YOLO26-Seg 四输出或 YOLOv8-Seg 五输出时返回 true。
    """
    return isinstance(preds, (list, tuple)) and len(preds) in (4, 5)


def _normalize_rknn_imgsz(imgsz) -> tuple[int, int]:
    """!
    @brief 将模型输入尺寸规范化为静态高宽。
    @param imgsz 整数或包含高宽的序列。
    @return `(height, width)` 静态输入尺寸。
    @throw ValueError 当序列不是两维或尺寸非法时抛出。
    """
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) != 2:
            raise ValueError(f"RKNN 分割输入尺寸必须包含高宽，实际 imgsz={imgsz}")
        height, width = int(imgsz[0]), int(imgsz[1])
    else:
        height = width = int(imgsz)
    if height <= 0 or width <= 0:
        raise ValueError(f"RKNN 分割输入尺寸必须大于 0，实际 imgsz={imgsz}")
    return height, width


def _decode_rknn_seg_outputs(preds, imgsz, conf: float, iou: float, max_det: int):
    """!
    @brief 将 RKNN 分割输出解码为框架统一的检测张量与 proto。
    @details
    YOLO26-Seg `seg_pre_dist` 使用四输出并固定执行 NMS-free exact TopK；
    YOLOv8-Seg `seg_pre_dfl` 使用带 `score_sum` 的五输出并执行 NMS。
    @param preds RKNN 返回的四个或五个输出张量。
    @param imgsz 模型静态输入尺寸。
    @param conf 置信度阈值。
    @param iou YOLOv8-Seg NMS IoU 阈值；YOLO26-Seg 不使用。
    @param max_det 最大实例数。
    @return 二元组 `(detections, proto)`：detections 为 `[N,6+nm]`，proto 为
            `[1,nm,H,W]`，均为 Paddle FP32 张量。
    @throw ValueError 当输出数量、布局或关联维度不满足契约时抛出。
    """
    from tools.eval.cli import decode_seg_predfl, decode_seg_predist, detect_output_format

    if not _is_rknn_seg_outputs(preds):
        count = len(preds) if isinstance(preds, (list, tuple)) else type(preds).__name__
        raise ValueError(f"RKNN 分割部署要求 YOLO26 四输出或 YOLOv8 五输出，实际 {count}")

    arrays = [np.asarray(output) for output in preds]
    output_format = detect_output_format(arrays)
    size = _normalize_rknn_imgsz(imgsz)
    if output_format == "seg_pre_dist":
        result = decode_seg_predist(arrays, conf, max_det, imgsz=size)
    elif output_format == "seg_pre_dfl":
        result = decode_seg_predfl(arrays, conf, max_det, iou_thresh=iou, imgsz=size)
    else:
        shapes = [tuple(output.shape) for output in arrays]
        raise ValueError(f"RKNN 分割模型不满足四/五输出部署契约，实际 shapes={shapes}")

    detections = np.concatenate(
        [
            result["boxes"],
            result["scores"][:, None],
            result["classes"][:, None],
            result["coeffs"],
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    proto = np.asarray(result["proto"], dtype=np.float32)[None]
    return paddle.to_tensor(detections), paddle.to_tensor(proto)


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
        if getattr(self.model, "rknn", False):
            if not _is_rknn_seg_outputs(preds):
                count = len(preds) if isinstance(preds, (list, tuple)) else type(preds).__name__
                raise ValueError(f"RKNN 分割模型必须返回 YOLO26 四输出或 YOLOv8 五输出，实际 {count}")
            detections, protos = _decode_rknn_seg_outputs(
                preds,
                img.shape[2:],
                self.args.conf,
                self.args.iou,
                self.args.max_det,
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_paddle2numpy_batch(orig_imgs)[..., ::-1]
            return self.construct_results([detections], img, orig_imgs, protos)

        # Paddle/普通 ONNX 分割输出仍使用 det + proto 双输出契约。
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
            # `process_mask` 为节省内存返回 uint8；Paddle GPU 没有 uint8 reduce-max
            # kernel，先转 int32 保持 CPU/GPU 后端行为一致。
            keep = paddle.cast(masks.reshape([masks.shape[0], -1]), "int32").max(axis=-1) > 0
            if not keep.all():
                pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
