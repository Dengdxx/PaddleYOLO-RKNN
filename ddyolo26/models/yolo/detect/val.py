# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 检测验证器 DetectionValidator。
@details
继承 BaseValidator，实现 COCO 风格 mAP 评估：
- `postprocess()`：处理 end2end/传统NMS 双模式输出
- `update_metrics()`：逐图累计 TP/FP/FN
- `get_stats()`：计算 box mAP50 / mAP50-95
- `eval_json()`：生成 COCO 格式 JSON 结果并调用 pycocotools

验证时默认使用 one-to-one head（无 NMS），设 `end2end=False` 可切换到
one-to-many head + NMS 模式以对比精度差异。
"""

import paddle

import os
from pathlib import Path
from typing import Any

import numpy as np


from ddyolo26.data import build_dataloader, build_yolo_dataset
from ddyolo26.engine.validator import BaseValidator
from ddyolo26.utils import LOGGER, RANK, nms, ops
from ddyolo26.utils.checks import check_requirements
from ddyolo26.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ddyolo26.utils.plotting import plot_images


class DetectionValidator(BaseValidator):
    """面向检测模型验证的 BaseValidator 子类。

    该类实现目标检测任务专属的验证功能，包括指标计算、预测处理和结果可视化。

    属性:
        is_coco (bool): 数据集是否为 COCO。
        is_lvis (bool): 数据集是否为 LVIS。
        class_map (list[int]): 模型类别索引到数据集类别索引的映射。
        metrics (DetMetrics): 目标检测指标计算器。
        iouv (paddle.Tensor): 用于 mAP 计算的 IoU 阈值。
        niou (int): IoU 阈值数量。
        lb (list[Any]): 用于混合保存的真值标签列表。
        jdict (list[dict[str, Any]]): JSON 检测结果列表。
        stats (dict[str, list[paddle.Tensor]]): 验证过程中的统计信息字典。

    示例:
        >>> from ddyolo26.models.yolo.detect import DetectionValidator
        >>> args = dict(model="weights/yolov8/yolov8n.pdparams", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """使用必要变量和设置初始化检测验证器。

        参数:
            dataloader (paddle.io.DataLoader, optional): 验证使用的数据加载器。
            save_dir (Path, optional): 结果保存目录。
            args (dict[str, Any], optional): 验证器参数。
            _callbacks (list[Any], optional): 回调函数列表。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = paddle.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.size
        self.metrics = DetMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """为 YOLO 验证预处理图像批次。

        参数:
            batch (dict[str, Any]): 包含图像和标注的批次。

        返回:
            (dict[str, Any]): 预处理后的批次。
        """
        for k, v in batch.items():
            if isinstance(v, paddle.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        return batch

    def init_metrics(self, model: paddle.nn.Module) -> None:
        """初始化 YOLO 检测验证的评估指标。

        参数:
            model (paddle.nn.Layer): 待验证模型。
        """
        val = self.data.get(self.args.split, "")
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco
        self.class_map = list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics.names = model.names
        self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

    def get_desc(self) -> str:
        """返回汇总 YOLO 模型类别指标的格式化字符串。"""
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds: paddle.Tensor) -> list[dict[str, paddle.Tensor]]:
        """对预测输出执行非极大值抑制。

        参数:
            preds (paddle.Tensor): 模型原始预测。

        返回:
            (list[dict[str, paddle.Tensor]]): NMS 后的预测列表，每个字典包含 'bboxes'、'conf'、'cls'
                和 'extra' 张量。
        """
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """为验证准备单张图像及其标注。

        参数:
            si (int): batch 内的样本索引。
            batch (dict[str, Any]): 包含图像和标注的批次数据。

        返回:
            (dict[str, Any]): 标注已处理的样本数据。
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = (
                ops.xywh2xyxy(bbox).astype(paddle.float32)
                * paddle.tensor(imgsz, device=self.device, dtype=paddle.float32)[[1, 0, 1, 0]]
            )
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred: dict[str, paddle.Tensor]) -> dict[str, paddle.Tensor]:
        """准备与真值标注对比评估的预测结果。

        参数:
            pred (dict[str, paddle.Tensor]): 模型后处理后的预测。

        返回:
            (dict[str, paddle.Tensor]): 原始图像空间中的预测结果。
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: list[dict[str, paddle.Tensor]], batch: dict[str, Any]) -> None:
        """使用新的预测结果和真值标注更新指标。

        参数:
            preds (list[dict[str, paddle.Tensor]]): 模型预测结果列表。
            batch (dict[str, Any]): 包含真值标注的批次数据。
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)
            if no_pred:
                continue
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """设置速度指标和混淆矩阵的最终值。"""
        if self.args.plots:
            for normalize in (True, False):
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def gather_stats(self) -> None:
        """从所有 GPU 汇总统计信息。"""
        if RANK == 0:
            import pickle

            world_size = paddle.distributed.get_world_size()
            # 汇总指标统计
            local_stats_bytes = pickle.dumps(self.metrics.stats)
            local_stats_tensor = paddle.to_tensor(list(local_stats_bytes), dtype="uint8")
            local_len = paddle.to_tensor([len(local_stats_bytes)], dtype="int64")
            all_lens = [paddle.zeros([1], dtype="int64") for _ in range(world_size)]
            paddle.distributed.all_gather(all_lens, local_len)
            max_len = max(l.item() for l in all_lens)
            padded = paddle.zeros([max_len], dtype="uint8")
            padded[: len(local_stats_bytes)] = local_stats_tensor
            all_padded = [paddle.zeros([max_len], dtype="uint8") for _ in range(world_size)]
            paddle.distributed.all_gather(all_padded, padded)
            merged_stats = {key: [] for key in self.metrics.stats.keys()}
            for i, t in enumerate(all_padded):
                stats_dict = pickle.loads(bytes(t[: all_lens[i].item()].numpy().tolist()))
                for key in merged_stats:
                    merged_stats[key].extend(stats_dict[key])
            # 汇总 JSON 结果
            local_jdict_bytes = pickle.dumps(self.jdict)
            local_jdict_tensor = paddle.to_tensor(list(local_jdict_bytes), dtype="uint8")
            local_jlen = paddle.to_tensor([len(local_jdict_bytes)], dtype="int64")
            all_jlens = [paddle.zeros([1], dtype="int64") for _ in range(world_size)]
            paddle.distributed.all_gather(all_jlens, local_jlen)
            max_jlen = max(l.item() for l in all_jlens)
            jpadded = paddle.zeros([max_jlen], dtype="uint8")
            jpadded[: len(local_jdict_bytes)] = local_jdict_tensor
            all_jpadded = [paddle.zeros([max_jlen], dtype="uint8") for _ in range(world_size)]
            paddle.distributed.all_gather(all_jpadded, jpadded)
            self.jdict = []
            for i, t in enumerate(all_jpadded):
                jdict = pickle.loads(bytes(t[: all_jlens[i].item()].numpy().tolist()))
                self.jdict.extend(jdict)
            self.metrics.stats = merged_stats
            self.seen = len(self.dataloader.dataset)
        elif RANK > 0:
            import pickle

            world_size = paddle.distributed.get_world_size()
            # 参与指标统计汇总
            local_stats_bytes = pickle.dumps(self.metrics.stats)
            local_stats_tensor = paddle.to_tensor(list(local_stats_bytes), dtype="uint8")
            local_len = paddle.to_tensor([len(local_stats_bytes)], dtype="int64")
            all_lens = [paddle.zeros([1], dtype="int64") for _ in range(world_size)]
            paddle.distributed.all_gather(all_lens, local_len)
            max_len = max(l.item() for l in all_lens)
            padded = paddle.zeros([max_len], dtype="uint8")
            padded[: len(local_stats_bytes)] = local_stats_tensor
            all_padded = [paddle.zeros([max_len], dtype="uint8") for _ in range(world_size)]
            paddle.distributed.all_gather(all_padded, padded)
            # 参与 JSON 结果汇总
            local_jdict_bytes = pickle.dumps(self.jdict)
            local_jdict_tensor = paddle.to_tensor(list(local_jdict_bytes), dtype="uint8")
            local_jlen = paddle.to_tensor([len(local_jdict_bytes)], dtype="int64")
            all_jlens = [paddle.zeros([1], dtype="int64") for _ in range(world_size)]
            paddle.distributed.all_gather(all_jlens, local_jlen)
            max_jlen = max(l.item() for l in all_jlens)
            jpadded = paddle.zeros([max_jlen], dtype="uint8")
            jpadded[: len(local_jdict_bytes)] = local_jdict_tensor
            all_jpadded = [paddle.zeros([max_jlen], dtype="uint8") for _ in range(world_size)]
            paddle.distributed.all_gather(all_jpadded, jpadded)
            self.jdict = []
            self.metrics.clear_stats()

    def get_stats(self) -> dict[str, Any]:
        """计算并返回指标统计信息。

        返回:
            (dict[str, Any]): 包含指标结果的字典。
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """打印训练/验证集的逐类别指标。"""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(
            pf
            % (
                "all",
                self.seen,
                self.metrics.nt_per_class.sum(),
                *self.metrics.mean_results(),
            )
        )
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"{self.args.task} set 中未找到 labels，无法在没有 labels 的情况下计算 metrics")
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

    def _process_batch(self, preds: dict[str, paddle.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """返回预测是否正确的矩阵。

        参数:
            preds (dict[str, paddle.Tensor]): 包含 'bboxes' 和 'cls' 键的预测数据字典。
            batch (dict[str, Any]): 包含 'bboxes' 和 'cls' 键的真值数据字典。

        返回:
            (dict[str, np.ndarray]): 包含 'tp' 键的字典，值为形状 (N, 10) 的正确预测矩阵，对应 10 个 IoU 阈值。
        """
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def build_dataset(self, img_path: str, mode: str = "val", batch: (int | None) = None) -> paddle.io.Dataset:
        """构建 YOLO 数据集。

        参数:
            img_path (str): 图像文件夹路径。
            mode (str): `train` 或 `val` 模式，不同模式可配置不同增强策略。
            batch (int, optional): batch 大小，用于 `rect`。

        返回:
            (Dataset): YOLO 数据集。
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> paddle.io.DataLoader:
        """构建并返回数据加载器。

        参数:
            dataset_path (str): 数据集路径。
            batch_size (int): 每个 batch 的大小。

        返回:
            (paddle.io.DataLoader): 验证用 DataLoader。
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制验证图像样本。

        参数:
            batch (dict[str, Any]): 包含图像和标注的批次。
            ni (int): batch 索引。
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self,
        batch: dict[str, Any],
        preds: list[dict[str, paddle.Tensor]],
        ni: int,
        max_det: (int | None) = None,
    ) -> None:
        """在输入图像上绘制预测边界框并保存结果。

        参数:
            batch (dict[str, Any]): 包含图像和标注的批次。
            preds (list[dict[str, paddle.Tensor]]): 模型预测结果列表。
            ni (int): batch 索引。
            max_det (int | None): 最大绘制检测数量。
        """
        if not preds:
            return
        for i, pred in enumerate(preds):
            pred["batch_idx"] = paddle.ones_like(pred["conf"]) * i
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: paddle.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def save_one_txt(
        self,
        predn: dict[str, paddle.Tensor],
        save_conf: bool,
        shape: tuple[int, int],
        file: Path,
    ) -> None:
        """按指定格式将 YOLO 检测结果以归一化坐标保存到 txt 文件。

        参数:
            predn (dict[str, paddle.Tensor]): 包含 'bboxes'、'conf' 和 'cls' 键的预测字典。
            save_conf (bool): 是否保存置信度分数。
            shape (tuple[int, int]): 原始图像尺寸（高、宽）。
            file (Path): 保存检测结果的文件路径。
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
                dim=1,
            ),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, paddle.Tensor], pbatch: dict[str, Any]) -> None:
        """将 YOLO 预测序列化为 COCO JSON 格式。

        参数:
            predn (dict[str, paddle.Tensor]): 预测字典，包含带边界框坐标、置信度和类别预测的 'bboxes'、'conf'
                与 'cls' 键。
            pbatch (dict[str, Any]): 包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file' 的批次字典。

        示例:
             >>> result = {
             ...     "image_id": 42,
             ...     "file_name": "42.jpg",
             ...     "category_id": 18,
             ...     "bbox": [258.15, 41.29, 348.26, 243.78],
             ...     "score": 0.236,
             ... }
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])
        box[:, :2] -= box[:, 2:] / 2
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def scale_preds(self, predn: dict[str, paddle.Tensor], pbatch: dict[str, Any]) -> dict[str, paddle.Tensor]:
        """将预测结果缩放到原始图像尺寸。"""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """评估 JSON 格式的 YOLO 输出并返回性能统计。

        参数:
            stats (dict[str, Any]): 当前统计字典。

        返回:
            (dict[str, Any]): 加入 COCO/LVIS 评估结果后的统计字典。
        """
        pred_json = self.save_dir / "predictions.json"
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: (str | list[str]) = "bbox",
        suffix: (str | list[str]) = "Box",
    ) -> dict[str, Any]:
        """使用 faster-coco-eval 库评估 COCO/LVIS 指标。

        该方法会调用 faster-coco-eval 计算目标检测 mAP 指标，并把 mAP50、mAP50-95 以及适用的 LVIS
        专属指标写入传入的统计字典。

        参数:
            stats (dict[str, Any]): 用于保存计算指标和统计信息的字典。
            pred_json (str | Path): 包含 COCO 格式预测结果的 JSON 文件路径。
            anno_json (str | Path): 包含 COCO 格式真值标注的 JSON 文件路径。
            iou_types (str | list[str]): 评估用 IoU 类型，可为单个字符串或字符串列表，常见值包括
                "bbox"、"segm"、"keypoints"，默认 "bbox"。
            suffix (str | list[str]): 追加到 stats 指标名中的后缀；多类型评估时应与 iou_types 对应，默认 "Box"。

        返回:
            (dict[str, Any]): 包含 COCO/LVIS 评估指标的更新后统计字典。
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\n正在使用 {pred_json} 和 {anno_json} 通过 faster-coco-eval 评估 mAP...")
            try:
                for x in (pred_json, anno_json):
                    assert x.is_file(), f"{x} file 未找到"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno,
                        pred,
                        iouType=iou_type,
                        lvis_style=self.is_lvis,
                        print_function=LOGGER.info,
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                    val.evaluate()
                    val.accumulate()
                    val.summarize()
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]
                    stats["metrics/mAP_small(B)"] = val.stats_as_dict["AP_small"]
                    stats["metrics/mAP_medium(B)"] = val.stats_as_dict["AP_medium"]
                    stats["metrics/mAP_large(B)"] = val.stats_as_dict["AP_large"]
                    stats["fitness"] = 0.9 * val.stats_as_dict["AP_all"] + 0.1 * val.stats_as_dict["AP_50"]
                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]
                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval 无法运行: {e}")
        return stats
