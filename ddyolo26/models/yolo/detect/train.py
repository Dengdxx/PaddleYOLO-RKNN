# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 检测训练器 DetectionTrainer。
@details
继承 BaseTrainer，实现检测任务专属逻辑：
- `build_dataset()`：构建 YOLODataset 并注入检测任务配置
- `build_model()`：实例化 DetectionModel，挂载 E2EDetectLoss
- `preprocess_batch()`：图像归一化、bbox 坐标归一化到 [0,1]
- `plot_training_samples()`：每 epoch 采样绘制训练图检查标注质量

与 YOLO26 特性关联：E2EDetectLoss 使用匈牙利匹配训练 one2one head，
配合 `v8DetectionLoss` 用 TAL 训练 one2many head（训练时并行两路损失）。
"""

import paddle

import math
import random
from copy import copy
from typing import Any

import numpy as np


from ddyolo26.data import build_dataloader, build_yolo_dataset
from ddyolo26.engine.trainer import BaseTrainer
from ddyolo26.models import yolo
from ddyolo26.nn.tasks import DetectionModel
from ddyolo26.utils import DEFAULT_CFG, LOGGER, RANK
from ddyolo26.utils.patches import override_configs
from ddyolo26.utils.plotting import plot_images, plot_labels
from ddyolo26.utils.runtime import distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """面向检测模型训练的 BaseTrainer 子类。

    该训练器专用于目标检测任务，处理 YOLO 检测模型训练所需的数据集构建、数据加载、预处理和模型配置。

    属性:
        model (DetectionModel): 正在训练的 YOLO 检测模型。
        data (dict): 数据集信息字典，包括类别名和类别数。
        loss_names (tuple): 训练中使用的损失项名称（box_loss、cls_loss、dfl_loss）。

    方法:
        build_dataset: 为训练或验证构建 YOLO 数据集。
        get_dataloader: 构建并返回指定模式的数据加载器。
        preprocess_batch: 对图像批次做缩放和浮点转换。
        set_model_attributes: 根据数据集信息设置模型属性。
        get_model: 返回 YOLO 检测模型。
        get_validator: 返回用于模型评估的验证器。
        label_loss_items: 返回带标签的训练损失字典。
        progress_string: 返回格式化的训练进度字符串。
        plot_training_samples: 绘制带标注的训练样本。
        plot_training_labels: 创建 YOLO 模型训练标签统计图。
        auto_batch: 根据模型显存占用估算最佳 batch size。

    示例:
        >>> from ddyolo26.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="weights/yolov8/yolov8n.pdparams", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: (dict[str, Any] | None) = None,
        _callbacks=None,
    ):
        """初始化用于 YOLO 目标检测模型训练的 DetectionTrainer 对象。

        参数:
            cfg (dict, optional): 包含训练参数的默认配置字典。
            overrides (dict, optional): 默认配置的参数覆盖字典。
            _callbacks (list, optional): 训练过程中执行的回调函数列表。
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: (int | None) = None):
        """为训练或验证构建 YOLO 数据集。

        参数:
            img_path (str): 图像文件夹路径。
            mode (str): 'train' 或 'val' 模式，不同模式可配置不同增强策略。
            batch (int, optional): batch 大小，用于 'rect' 模式。

        返回:
            (Dataset): 按指定模式配置好的 YOLO 数据集对象。
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
        )

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        """构建并返回指定模式的数据加载器。

        参数:
            dataset_path (str): 数据集路径。
            batch_size (int): 每个 batch 的图像数量。
            rank (int): 分布式训练的进程 rank。
            mode (str): 'train' 表示训练加载器，'val' 表示验证加载器。

        返回:
            (DataLoader): 数据加载器对象。
        """
        assert mode in {"train", "val"}, f"mode 必须是 'train' 或 'val'，而不是 {mode}。"
        with distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' 与 DataLoader shuffle 不兼容，已设置 shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else 0,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """通过缩放和浮点转换预处理图像批次。

        参数:
            batch (dict): 包含 'img' 张量的批次数据字典。

        返回:
            (dict): 图像已归一化的预处理批次。
        """
        for k, v in batch.items():
            if isinstance(v, paddle.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [(math.ceil(x * sf / self.stride) * self.stride) for x in imgs.shape[2:]]
                imgs = paddle.nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """根据数据集信息设置模型属性。"""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)

    def get_model(
        self,
        cfg: (str | None) = None,
        weights: (str | None) = None,
        verbose: bool = True,
    ):
        """返回 YOLO 检测模型。

        参数:
            cfg (str, optional): 模型配置文件路径。
            weights (str, optional): 模型权重路径。
            verbose (bool): 是否显示模型信息。

        返回:
            (DetectionModel): YOLO 检测模型。
        """
        model = DetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """返回用于 YOLO 模型验证的 DetectionValidator。"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def label_loss_items(self, loss_items: (list[float] | None) = None, prefix: str = "train"):
        """返回带标签的训练损失项字典。

        参数:
            loss_items (list[float], optional): 损失值列表。
            prefix (str): 返回字典键名的前缀。

        返回:
            (dict | list): 提供 loss_items 时返回带标签的损失字典，否则返回键名列表。
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """返回包含 epoch、GPU 显存、损失、实例数和尺寸的训练进度字符串。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制带标注的训练样本。

        参数:
            batch (dict[str, Any]): 批次数据字典。
            ni (int): 用于命名输出文件的 batch 索引。
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """创建 YOLO 模型的训练标签统计图。"""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(
            boxes,
            cls.squeeze(),
            names=self.data["names"],
            save_dir=self.save_dir,
            on_plot=self.on_plot,
        )

    def auto_batch(self):
        """根据模型显存占用估算最佳 batch size。

        返回:
            (int): 最佳 batch size。
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        del train_dataset
        return super().auto_batch(max_num_obj)
