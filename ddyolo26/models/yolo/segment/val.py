# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 分割验证器 SegmentationValidator。
@details
继承 DetectionValidator，增加掩码后处理与掩码 mAP 计算：
- `postprocess()`：从模型输出提取 proto 并生成实例掩码
- `_prepare_batch()`：准备真值掩码（overlap 模式 vs 独立掩码）
- `_process_batch()`：计算掩码 IoU 并返回 tp_m
- `plot_predictions()`：可视化分割结果
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import paddle

from ddyolo26.models.yolo.detect import DetectionValidator
from ddyolo26.models.yolo.segment.predict import _decode_rknn_seg_five_outputs, _is_seg_five_outputs
from ddyolo26.utils import LOGGER, ops
from ddyolo26.utils.checks import check_requirements
from ddyolo26.utils.metrics import SegmentMetrics, mask_iou
from ddyolo26.utils.plotting import plot_images


class SegmentationValidator(DetectionValidator):
    """分割验证器，在检测验证基础上增加掩码评估。

    属性:
        process (callable): 掩码处理函数（process_mask 或 process_mask_native）。
        metrics (SegmentMetrics): 分割指标计算器。

    示例:
        >>> from ddyolo26.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="weights/yolov8seg/yolov8n-seg.pdparams", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """初始化 SegmentationValidator。

        参数:
            dataloader: 验证数据加载器。
            save_dir (Path, optional): 结果保存目录。
            args (dict, optional): 验证器参数。
            _callbacks (list, optional): 回调函数列表。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """预处理批量数据，将掩码转为浮点型。

        参数:
            batch (dict[str, Any]): 包含图像和标注的批量数据。

        返回:
            (dict[str, Any]): 预处理后的批量数据。
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].cast(paddle.float32)
        return batch

    def init_metrics(self, model: paddle.nn.Module) -> None:
        """初始化指标并根据配置选择掩码处理函数。

        参数:
            model (paddle.nn.Layer): 待验证的模型。
        """
        super().init_metrics(model)
        self._rknn_backend = bool(getattr(model, "rknn", False))
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

    def get_desc(self) -> str:
        """返回分割评估指标的格式化描述字符串。"""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds: list[paddle.Tensor]) -> list[dict[str, paddle.Tensor]]:
        """后处理 YOLO 预测输出，提取 proto 并生成掩码。

        参数:
            preds (list[paddle.Tensor]): 模型原始预测。

        返回:
            (list[dict[str, paddle.Tensor]]): 处理后的检测结果字典列表，含 masks。
        """
        if getattr(self, "_rknn_backend", False):
            if not _is_seg_five_outputs(preds):
                count = len(preds) if isinstance(preds, (list, tuple)) else type(preds).__name__
                raise ValueError(f"RKNN 分割模型必须返回五输出，实际 {count}")
            detections, proto = _decode_rknn_seg_five_outputs(
                preds,
                self.args.imgsz,
                self.args.conf,
                self.args.iou,
                self.args.max_det,
            )
            preds = [
                {
                    "bboxes": detections[:, :4],
                    "conf": detections[:, 4],
                    "cls": detections[:, 5],
                    "extra": detections[:, 6:],
                }
            ]
        else:
            proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
            preds = super().postprocess(preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]
        for i, pred in enumerate(preds):
            coefficient = pred.pop("extra")
            if coefficient.shape[0]:
                pred["masks"] = self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
            else:
                if self.process is ops.process_mask_native:
                    shape = tuple(imgsz)
                else:
                    shape = tuple(proto.shape[2:])
                pred["masks"] = paddle.zeros(
                    (0, *shape),
                    dtype=paddle.uint8,
                )
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """准备验证批量数据，包含真值掩码。

        参数:
            si (int): 批量中的样本索引。
            batch (dict[str, Any]): 批量数据。

        返回:
            (dict[str, Any]): 包含真值掩码的准备数据。
        """
        prepared_batch = super()._prepare_batch(si, batch)
        nl = prepared_batch["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = paddle.arange(1, nl + 1).reshape([nl, 1, 1])
            masks = (masks.astype("int64") == index).cast(paddle.float32)
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared_batch["imgsz"]]
            if list(masks.shape[1:]) != mask_size:
                masks = paddle.nn.functional.interpolate(
                    masks.unsqueeze(0), mask_size, mode="bilinear", align_corners=False
                )[0]
                masks = (masks > 0.5).cast(masks.dtype)
        prepared_batch["masks"] = masks
        return prepared_batch

    def _process_batch(self, preds: dict[str, paddle.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """计算检测和掩码的正确预测矩阵。

        参数:
            preds (dict[str, paddle.Tensor]): 预测字典，含 'cls' 和 'masks'。
            batch (dict[str, Any]): 批量数据字典，含 'cls' 和 'masks'。

        返回:
            (dict[str, np.ndarray]): 包含 'tp' 和 'tp_m' 的正确预测矩阵。
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            iou = mask_iou(
                batch["masks"].reshape([batch["masks"].shape[0], -1]),
                preds["masks"].reshape([preds["masks"].shape[0], -1]).cast(paddle.float32),
            )
            tp_m = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_m": tp_m})
        return tp

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制带分割掩码的验证样本图。

        参数:
            batch (dict[str, Any]): 包含图像、标注和掩码的批量数据。
            ni (int): 批次索引。
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, paddle.Tensor]], ni: int) -> None:
        """绘制带掩码和检测框的批量预测结果。

        参数:
            batch (dict[str, Any]): 包含图像和标注的批量数据。
            preds (list[dict[str, paddle.Tensor]]): 模型预测列表。
            ni (int): 批次索引。
        """
        for p in preds:
            pred_masks = p["masks"]
            if pred_masks.shape[0] > self.args.max_det:
                LOGGER.warning(f"验证绘图最多保留 'max_det={self.args.max_det}' 个目标。")
            p["masks"] = pred_masks[: self.args.max_det]
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)

    def save_one_txt(
        self,
        predn: dict[str, paddle.Tensor],
        save_conf: bool,
        shape: tuple[int, int],
        file: Path,
    ) -> None:
        """@brief 将分割预测写入 TXT，并保留掩码多边形。

        @param predn 含边界框、置信度、类别与掩码的预测字典。
        @param save_conf 是否附带置信度。
        @param shape 原始图像尺寸 `(height, width)`。
        @param file 输出 TXT 文件路径。
        """
        from ddyolo26.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=paddle.cat(
                [
                    predn["bboxes"],
                    predn["conf"].unsqueeze(-1),
                    predn["cls"].unsqueeze(-1),
                ],
                axis=1,
            ),
            masks=predn["masks"].cast(paddle.uint8),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, paddle.Tensor], pbatch: dict[str, Any]) -> None:
        """@brief 将分割预测序列化为 COCO JSON，并写入掩码 RLE。

        @param predn 含边界框、置信度、类别与掩码的预测字典。
        @param pbatch 当前样本批数据，至少包含 `im_file`。
        """

        def to_string(counts: list[int]) -> str:
            """@brief 将 RLE 计数压缩为 COCO 兼容字符串。

            @param counts 原始游程长度列表。
            @return 压缩后的字符串表示。
            """
            result: list[str] = []
            for i, value in enumerate(counts):
                x = int(value)
                if i > 2:
                    x -= int(counts[i - 2])
                while True:
                    c = x & 0x1F
                    x >>= 5
                    more = (x != -1) if (c & 0x10) else (x != 0)
                    if more:
                        c |= 0x20
                    c += 48
                    result.append(chr(c))
                    if not more:
                        break
            return "".join(result)

        def multi_encode(pixels: np.ndarray) -> list[list[int]]:
            """@brief 对多张二值掩码执行 Run-Length Encoding。

            @param pixels 形状为 `[N, H*W]` 的二值掩码矩阵。
            @return 每张掩码对应的游程计数列表。
            """
            transitions = pixels[:, 1:] != pixels[:, :-1]
            row_idx, col_idx = np.where(transitions)
            col_idx = col_idx + 1
            counts: list[list[int]] = []
            for i in range(pixels.shape[0]):
                positions = col_idx[row_idx == i]
                if positions.size:
                    count = np.diff(positions).astype(np.int64).tolist()
                    count.insert(0, int(positions[0]))
                    count.append(int(len(pixels[i]) - positions[-1]))
                else:
                    count = [int(len(pixels[i]))]
                if int(pixels[i][0]) == 1:
                    count = [0, *count]
                counts.append(count)
            return counts

        pred_masks = predn["masks"].cast(paddle.uint8).numpy().transpose(0, 2, 1).reshape([predn["masks"].shape[0], -1])
        h, w = map(int, predn["masks"].shape[1:3])
        rles = [{"size": [h, w], "counts": to_string(c)} for c in multi_encode(pred_masks)]
        super().pred_to_json(predn, pbatch)
        for i, rle in enumerate(rles):
            self.jdict[-len(rles) + i]["segmentation"] = rle

    def scale_preds(self, predn: dict[str, paddle.Tensor], pbatch: dict[str, Any]) -> dict[str, paddle.Tensor]:
        """将预测缩放到原始图像尺寸，包含掩码。"""
        scaled = super().scale_preds(predn, pbatch)
        scaled["masks"] = ops.scale_masks(
            predn["masks"].unsqueeze(0), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )[0].cast(paddle.uint8)
        return scaled

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """返回 COCO 风格的实例分割评估指标。"""
        pred_json = self.save_dir / "predictions.json"
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])
