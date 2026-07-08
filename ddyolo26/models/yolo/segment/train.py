# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO26 分割训练器 SegmentationTrainer。
@details
继承 DetectionTrainer，重写模型构建和验证器获取，
使用 SegmentationModel 和 SegmentationValidator。
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import numpy as np

from ddyolo26.models import yolo
from ddyolo26.nn.tasks import SegmentationModel
from ddyolo26.utils import DEFAULT_CFG, RANK
from ddyolo26.utils.plotting import plot_images, plot_labels


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """分割训练器，在检测训练器基础上增加分割任务支持。

    属性:
        loss_names (tuple[str]): 训练损失分量名称。

    示例:
        >>> from ddyolo26.models.yolo.segment import SegmentationTrainer
        >>> args = dict(model="yolo26n-seg.yaml", data="coco8-seg.yaml", epochs=3)
        >>> trainer = SegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        """初始化 SegmentationTrainer。

        参数:
            cfg (dict): 包含默认训练参数的配置字典。
            overrides (dict, optional): 覆盖默认配置的参数字典。
            _callbacks (list, optional): 训练期间执行的回调函数列表。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ):
        """初始化并返回 SegmentationModel。

        参数:
            cfg (str, optional): 模型配置文件路径。
            weights (str | Path, optional): 预训练权重路径。
            verbose (bool): 是否显示模型信息。

        返回:
            (SegmentationModel): 初始化后的分割模型。
        """
        model = SegmentationModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """返回用于模型验证的 SegmentationValidator 实例。"""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制带分割掩码的训练样本图。

        参数:
            batch (dict[str, Any]): 包含图像、标注和掩码的批量数据。
            ni (int): 批次索引，用于生成文件名。
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self) -> None:
        """绘制训练集标签分布统计图（框分布 + 类别直方图）。"""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(
            boxes,
            cls.squeeze(),
            names=self.data["names"],
            save_dir=self.save_dir,
            on_plot=self.on_plot,
        )
