# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 检测推理器 DetectionPredictor。
@details
继承 BasePredictor，重写 `postprocess()` 对检测输出进行后处理：
- end2end 模式：直接解析 `(N, 300, 6)` 的无 NMS 输出
- 传统模式：对 `(N, nc+4, 8400)` 应用 NMS（兼容 one2many head）

输出封装为 `Results` 对象，包含 `.boxes` 属性供下游消费。
"""

from __future__ import annotations

import paddle

from ddyolo26.engine.predictor import BasePredictor
from ddyolo26.engine.results import Results
from ddyolo26.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """面向检测模型推理的 BasePredictor 子类。

    该推理器专用于目标检测任务，会将模型原始输出转换为带边界框和类别预测的检测结果。

    属性:
        args (namespace): 推理器配置参数。
        model (nn.Module): 用于推理的检测模型。
        batch (list): 待处理的图像与元数据批次。

    方法:
        postprocess: 将模型原始预测处理为检测结果。
        construct_results: 根据处理后的预测构建 Results 对象。
        construct_result: 根据单张图预测创建一个 Results 对象。
        get_obj_feats: 从特征图中提取目标特征。

    示例:
        >>> from ddyolo26.utils import ASSETS
        >>> from ddyolo26.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="weights/yolov8/yolov8n.pdparams", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """对预测结果做后处理，并返回 Results 对象列表。

        该方法会对模型原始预测执行非极大值抑制，并整理为便于可视化和后续分析的结构。

        参数:
            preds (paddle.Tensor): 模型原始预测。
            img (paddle.Tensor): 按模型输入格式处理后的图像张量。
            orig_imgs (paddle.Tensor | list): 预处理前的原始输入图像。
            **kwargs (Any): 其他关键字参数。

        返回:
            (list): 包含后处理预测结果的 Results 对象列表。

        示例:
            >>> predictor = DetectionPredictor(overrides=dict(model="weights/yolov8/yolov8n.pdparams"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_paddle2numpy_batch(orig_imgs)[..., ::-1]
        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]
        results = self.construct_results(preds, img, orig_imgs, **kwargs)
        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f
        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """从特征图中提取目标特征。"""
        s = min(x.shape[1] for x in feat_maps)
        obj_feats = paddle.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps],
            dim=1,
        )
        return [(feats[idx] if idx.shape[0] else []) for feats, idx in zip(obj_feats, idxs)]

    def construct_results(self, preds, img, orig_imgs):
        """根据模型预测构建 Results 对象列表。

        参数:
            preds (list[paddle.Tensor]): 每张图的预测边界框与分数列表。
            img (paddle.Tensor): 推理时使用的预处理图像批次。
            orig_imgs (list[np.ndarray]): 预处理前的原始图像列表。

        返回:
            (list[Results]): 每张图对应的检测结果对象列表。
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """根据单张图的预测构建一个 Results 对象。

        参数:
            pred (paddle.Tensor): 形状为 (N, 6) 的预测框和分数，N 为检测数量。
            img (paddle.Tensor): 推理时使用的预处理图像张量。
            orig_img (np.ndarray): 预处理前的原始图像。
            img_path (str): 原始图像文件路径。

        返回:
            (Results): 包含原图、图像路径、类别名和缩放后边界框的 Results 对象。
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
