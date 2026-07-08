# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO 模型入口类：任务感知的统一门面。
@details
继承 `engine.model.Model`，绑定任务 → (predictor, trainer, validator) 三元组：
- detect → DetectionPredictor / DetectionTrainer / DetectionValidator
- segment → SegmentationPredictor / SegmentationTrainer / SegmentationValidator

提供 `YOLO(model_path)` 一行初始化接口，本仓库实现 detect 与 segment 任务，
其余任务（cls/pose/obb）已剔除，降低复杂度。
"""

import paddle

from pathlib import Path
from typing import Any


from ddyolo26.data.build import load_inference_source
from ddyolo26.engine.model import Model
from ddyolo26.models import yolo
from ddyolo26.nn.tasks import DetectionModel, SegmentationModel
from ddyolo26.utils import ROOT, YAML


class YOLO(Model):
    """YOLO (You Only Look Once) 目标检测/分割模型。

    该类为 YOLO 目标检测与分割模型提供统一入口，支持 detect 与 segment
    任务的训练、验证和推理。

    属性:
        model: 已加载的 YOLO 模型实例。
        task: 任务类型（detect 或 segment）。
        overrides: 模型配置覆盖项。

    方法:
        __init__: 初始化 YOLO 模型。
        task_map: 将任务映射到对应的模型、训练器、验证器和推理器类。

    示例:
        加载预训练 YOLOv8n 检测模型
        >>> model = YOLO("weights/yolov8/yolov8n.pdparams")

        从 YAML 配置初始化
        >>> model = YOLO("ddyolo26/cfg/models/v8/yolov8.yaml")
    """

    def __init__(
        self,
        model: (str | Path) = "weights/yolov8/yolov8n.pdparams",
        task: (str | None) = None,
        verbose: bool = False,
    ):
        """初始化用于检测或分割任务的 YOLO 模型。

        参数:
            model (str | Path): 模型名或模型文件路径，如 'weights/yolov8/yolov8n.pdparams'、'yolov8n'。
            task (str, optional): 任务类型；未指定时从模型结构或配置推断。
            verbose (bool): 加载时是否显示模型信息。
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """将任务映射到模型、训练器、验证器和推理器类。"""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
        }
