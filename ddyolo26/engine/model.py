# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""!
@file model.py
@brief `ddyolo26` 统一用户 API 的核心入口。
@details
本文件定义了 `Model` 基类，用统一接口封装了 YOLO Paddle 版的训练、验证、
推理、导出、调参等能力。用户通常通过更高层的 `YOLO(...)` 入口间接使用这里的
方法，因此这里相当于整个框架的“门面层”。

对当前仓库而言，这个门面层还承担了几项额外职责：
- 屏蔽 Paddle 重写版与上游 Ultralytics 风格 API 之间的差异；
- 统一处理 .pdparams、*_paddle.pt、.onnx 等多种模型来源；
- 在训练、验证、导出和部署之间共享配置与回调体系。
"""

from __future__ import annotations
import paddle
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from PIL import Image

from ddyolo26.cfg import TASK2DATA, get_cfg, get_save_dir
from ddyolo26.engine.results import Results
from ddyolo26.nn.tasks import guess_model_task, load_checkpoint, yaml_model_load
from ddyolo26.utils import ARGV, ASSETS, DEFAULT_CFG_DICT, LOGGER, RANK, SETTINGS, YAML, callbacks, checks


class Model(paddle.nn.Module):
    """!
    @brief YOLO26 模型统一抽象基类。
    @details
    该类把训练、验证、推理、导出、benchmark、超参搜索等能力统一暴露为稳定 API，
    让上层调用者无需关心底层究竟是 YAML 新建模型、Paddle 检查点，还是导出后的部署模型。

    在当前 Paddle 重写版中，它还有两个非常关键的作用：
    - 将 Ultralytics 风格的用户体验映射到 Paddle 实现细节；
    - 根据 task 动态装配 trainer / validator / predictor / exporter 等组件。
    """

    def __init__(
        self,
        model: (str | Path | Model) = "weights/yolov8/yolov8n.pdparams",
        task: (str | None) = None,
        verbose: bool = False,
    ) -> None:
        """!
        @brief 初始化统一模型入口。
        @param model 模型来源，可以是 YAML、Paddle 权重、导出模型路径，或已有 `Model` 实例。
        @param task 显式指定任务类型；为空时自动推断。
        @param verbose 是否输出更详细的初始化日志。
        @details
        初始化阶段会建立回调系统、识别模型来源，并进一步走 `_new()` 或 `_load()` 分支。
        这一步决定了后续 `train()`、`predict()`、`export()` 等 API 会挂接到哪套底层组件。
        """
        if isinstance(model, Model):
            self.__dict__ = model.__dict__
            return
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None
        self.model = None
        self.trainer = None
        self.ckpt = {}
        self.cfg = None
        self.ckpt_path = None
        self.overrides = {}
        self.metrics = None
        self.task = task
        self.model_name = None
        model = str(model).strip()
        if self.is_triton_model(model):
            self.model_name = self.model = model
            self.overrides["task"] = task or "detect"
            return
        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if str(model).endswith((".yaml", ".yml")):
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)
        del self.training

    def __call__(
        self,
        source: (str | Path | int | Image.Image | list | tuple | np.ndarray | paddle.Tensor) = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """!
        @brief `predict()` 的可调用别名。
        @param source 输入源，可以是路径、摄像头索引、数组、PIL 图像或张量。
        @param stream 是否按流式方式推理。
        @param kwargs 额外推理参数。
        @return `Results` 列表。
        @details 允许用户直接使用 `model(...)` 完成推理，保持与上游 YOLO 使用习惯一致。
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """检查模型字符串是否为 Triton Server URL。

        当前仓库暂不启用 Triton 推理，因此始终返回 False。

        参数:
            model (str): 待检查模型字符串。

        返回:
            (bool): 当前固定返回 False。

        示例:
            >>> Model.is_triton_model("http://localhost:8000/v2/models/yolo11n")
            False
            >>> Model.is_triton_model("weights/yolov8/yolov8n.pdparams")
            False
        """
        return False

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """根据 YAML 配置初始化新模型，并推断任务类型。

        会加载模型配置，在未显式指定 task 时自动推断任务，并从 task_map 中选择对应模型类。

        参数:
            cfg (str): YAML 模型配置文件路径。
            task (str, optional): 显式任务类型；为空时从配置推断。
            model (type[paddle.nn.Layer], optional): 自定义模型类；提供时替代 task_map 中的默认类。
            verbose (bool): 是否在加载时输出模型信息。

        异常:
            ValueError: 配置非法或无法推断任务类型时抛出。
            ImportError: 指定任务缺少依赖时抛出。

        示例:
            >>> model = Model()
            >>> model._new("yolo26n.yaml", task="detect", verbose=True)
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        """从权重或导出模型文件加载模型。

        Paddle 检查点会加载为动态图模型；ONNX/RKNN 等导出格式会保留路径并在推理后端中加载。
        普通 PyTorch `.pt` 在本 Paddle 分支中会被拒绝。

        参数:
            weights (str): 待加载权重或模型文件路径。
            task (str, optional): 模型任务；为空时自动推断。

        异常:
            FileNotFoundError: 指定文件不存在或不可访问时抛出。
            ValueError: 权重格式不受支持或无效时抛出。

        示例:
            >>> model = Model()
            >>> model._load("weights/yolov8/yolov8n.pdparams")
            >>> model._load("path/to/weights.pdparams", task="detect")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://", "ul://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])
        weights = checks.check_model_file_from_stem(weights)
        weights = checks.check_file(weights)
        weight_str = str(weights)
        if weight_str.endswith((".pdparams", "_paddle.pt")):
            self.model, self.ckpt = load_checkpoint(weights)
            self.task = self.model.task
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        elif weight_str.endswith(".pt"):
            raise ValueError(
                f"PyTorch 格式权重不被此 Paddle 分支支持: {weights}\n"
                "请使用 Paddle 格式权重 (*.pdparams 或 *_paddle.pt)。\n"
                "推荐预训练模型: yolov8n / yolov8s / yolov8m / yolov8l / yolov8x，"
                "以及对应的 yolov8*-seg 分割模型。\n"
                "用法: YOLO('weights/yolov8/yolov8n.pdparams')"
            )
        else:
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_paddle_model(self) -> None:
        """检查当前模型是否为 Paddle 模型，不满足时抛出 TypeError。

        训练、保存、融合等操作需要 Paddle 动态图模型或 Paddle 检查点；
        ONNX/RKNN 等导出格式只适合推理/验证后端。

        异常:
            TypeError: 当前模型不是 Paddle 模型或 Paddle 检查点时抛出。

        示例:
            >>> model = Model("weights/yolov8/yolov8n.pdparams")
            >>> model._check_is_paddle_model()  # 不抛异常
            >>> model = Model("yolov8n.onnx")
            >>> model._check_is_paddle_model()  # 抛出 TypeError
        """
        paddle_ckpt = isinstance(self.model, (str, Path)) and str(self.model).endswith((".pdparams", "_paddle.pt"))
        paddle_module = isinstance(self.model, paddle.nn.Module)
        if not (paddle_module or paddle_ckpt):
            raise TypeError(
                f"""要运行此方法，model='{self.model}' 应为 Paddle checkpoint（*.pdparams 或 *_paddle.pt）或已加载的 Paddle 模型，但当前是其他格式。Paddle checkpoint 支持 train、val、predict 和 export，例如 'model.train(data=...)'；ONNX、TensorRT 等导出格式仅支持 'predict' 和 'val' 模式，例如 'yolo predict model=yolov8n.onnx'。
如需运行 CUDA 或 MPS 推理，请在推理命令中直接传入 device 参数，例如 'model.predict(source=..., device=0)'"""
            )

    def reset_weights(self) -> Model:
        """将模型权重重置为初始化状态。

        会遍历所有子模块，调用其 `reset_parameters()`，并确保参数在训练中可更新。

        返回:
            (Model): 当前模型实例。

        异常:
            TypeError: 当前模型不是 Paddle 模型时抛出。

        示例:
            >>> model = Model("weights/yolov8/yolov8n.pdparams")
            >>> model.reset_weights()
        """
        self._check_is_paddle_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.stop_gradient = not True
        return self

    def load(self, weights: (str | Path) = "weights/yolov8/yolov8n.pdparams") -> Model:
        """将指定权重加载到当前模型中。

        支持从权重文件或权重对象加载，按名称和形状匹配参数。

        参数:
            weights (str | Path): 权重文件路径或权重对象。

        返回:
            (Model): 当前模型实例。

        异常:
            TypeError: 当前模型不是 Paddle 模型时抛出。

        示例:
            >>> model = Model()
            >>> model.load("weights/yolov8/yolov8n.pdparams")
            >>> model.load(Path("path/to/weights.pdparams"))
        """
        self._check_is_paddle_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights
            weights, self.ckpt = load_checkpoint(weights)
        self.model.load(weights)
        return self

    def save(self, filename: (str | Path) = "saved_model.pdparams") -> None:
        """将当前模型状态保存到文件。

        会把模型检查点与日期、版本、许可证等元数据一起写入指定文件。

        参数:
            filename (str | Path): 输出模型文件名。

        异常:
            TypeError: 当前模型不是 Paddle 模型时抛出。

        示例:
            >>> model = Model("weights/yolov8/yolov8n.pdparams")
            >>> model.save("my_model.pdparams")
        """
        self._check_is_paddle_model()
        from copy import deepcopy
        from datetime import datetime

        from ddyolo26 import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, paddle.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "GNU AGPL-3.0",
            "docs": "https://www.gnu.org/licenses/agpl-3.0.html",
        }
        paddle.save(obj={**self.ckpt, **updates}, path=filename)

    def info(
        self,
        detailed: bool = False,
        verbose: bool = True,
        imgsz: (int | list[int, int]) = 640,
    ):
        """显示模型信息。

        根据参数输出概要或详细层信息，并可返回模型层数、参数量、梯度数与 FLOPs。

        参数:
            detailed (bool): 是否显示详细层信息。
            verbose (bool): 是否打印信息；False 时通常返回 None。
            imgsz (int | list[int, int]): 计算 FLOPs 使用的输入尺寸。

        返回:
            (tuple): 层数、参数量、梯度数和 GFLOPs；verbose=False 时可能返回 None。

        示例:
            >>> model = Model("weights/yolov8/yolov8n.pdparams")
            >>> model.info()  # 打印模型概要并返回 tuple
            >>> model.info(detailed=True)  # 打印详细信息并返回 tuple
        """
        self._check_is_paddle_model()
        return self.model.info(detailed=detailed, verbose=verbose, imgsz=imgsz)

    def fuse(self) -> None:
        """融合模型中的 Conv2d 与 BatchNorm2d 层以优化推理。

        融合会把 BN 的均值、方差、权重和偏置折叠进前置卷积层，从而减少推理时的算子数量和内存访问。

        示例:
            >>> model = Model("weights/yolov8/yolov8n.pdparams")
            >>> model.fuse()
            >>> # 模型已融合，可用于优化推理
        """
        self._check_is_paddle_model()
        self.model.fuse()

    def embed(
        self,
        source: (str | Path | int | list | tuple | np.ndarray | paddle.Tensor) = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """基于输入源生成图像特征嵌入。

        这是 `predict()` 的包装入口，默认抽取模型靠后的特征层。

        参数:
            source (str | Path | int | list | tuple | np.ndarray | paddle.Tensor): 图像输入源，可为路径、URL、数组等。
            stream (bool): 是否流式返回结果。
            **kwargs (Any): 额外嵌入/推理配置。

        返回:
            (list[paddle.Tensor]): 图像嵌入列表。

        示例:
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> image = "ddyolo26/assets/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> print(embeddings[0].shape)
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: (str | Path | int | Image.Image | list | tuple | np.ndarray | paddle.Tensor) = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> list[Results]:
        """!
        @brief 对输入源执行推理。
        @param source 输入源，可以是文件、URL、摄像头、数组、张量或它们的列表。
        @param stream 是否按连续流模式处理输入。
        @param predictor 自定义预测器实例；为空时按 task 自动加载默认预测器。
        @param kwargs 推理配置，如 `conf`、`imgsz`、`device`、`save` 等。
        @return `Results` 对象列表。
        @details
        该方法会根据当前模型状态自动创建或复用预测器，并在 CLI 模式与 Python API 模式之间
        切换不同调用路径，是整套推理用户体验的真正入口。
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"缺少 'source'，使用 'source={source}'。")
        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ddyolo26")) and any(
            x in ARGV for x in ("predict", "mode=predict")
        )
        custom = {
            "conf": 0.25,
            "batch": 1,
            "save": is_cli,
            "mode": "predict",
            "rect": True,
        }
        args = {**self.overrides, **custom, **kwargs}
        prompts = args.pop("prompts", None)
        if not self.predictor or self.predictor.args.device != args.get("device", self.predictor.args.device):
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def val(self, validator=None, **kwargs: Any):
        """!
        @brief 在指定数据集上执行验证。
        @param validator 自定义验证器；为空时根据 task 自动装配默认验证器。
        @param kwargs 验证参数，如 `data`、`imgsz`、`batch` 等。
        @return 验证指标对象，通常包含 `box.map50`、`box.map` 等字段。
        @details 该方法会构造验证器、触发验证流程，并把最新指标缓存到 `self.metrics`。
        """
        custom = {"rect": True}
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}
        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(self, data=None, format="", verbose=False, **kwargs: Any):
        """在多种导出格式上 benchmark 模型以评估性能。

        该方法用于评估模型在不同导出格式（如 ONNX 等）下的性能，内部调用
        ddyolo26.utils.benchmarks 模块的 `benchmark` 函数。benchmark 配置由默认配置、模型参数、
        方法默认值和用户提供的关键字参数合并得到。

        参数:
            data (str | None): benchmark 使用的数据集路径；为 None 时使用任务默认数据集。
            format (str): 指定 benchmark 的导出格式名称。
            verbose (bool): 是否打印详细 benchmark 信息。
            **kwargs (Any): 自定义 benchmark 流程的关键字参数，常用选项包括：
                - imgsz (int | list[int]): benchmark 图像尺寸。
                - half (bool): 是否使用半精度（FP16）模式。
                - int8 (bool): 是否使用 int8 精度模式。
                - device (str): benchmark 运行设备（如 'cpu'、'cuda'）。

        返回:
            (polars.DataFrame): 包含各格式 benchmark 结果的 Polars DataFrame，包括文件大小、指标和推理耗时。

        异常:
            TypeError: 模型不是 Paddle 模型时抛出。

        示例:
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, half=True)
            >>> print(results)
        """
        self._check_is_paddle_model()
        from ddyolo26.utils.benchmarks import benchmark

        from .exporter import export_formats

        custom = {"verbose": False}
        args = {
            **DEFAULT_CFG_DICT,
            **self.model.args,
            **custom,
            **kwargs,
            "mode": "benchmark",
        }
        fmts = export_formats()
        export_args = set(dict(zip(fmts["Argument"], fmts["Arguments"])).get(format, [])) - {"batch"}
        export_kwargs = {k: v for k, v in args.items() if k in export_args}
        return benchmark(
            model=self,
            data=data,
            imgsz=args["imgsz"],
            device=args["device"],
            verbose=verbose,
            format=format,
            **export_kwargs,
        )

    def export(self, **kwargs: Any) -> str:
        """!
        @brief 将当前模型导出为部署格式。
        @param kwargs 导出参数，如 `format`、`imgsz`、`half`、`int8`、`simplify` 等。
        @return 导出后模型文件路径。
        @details
        导出流程会把模型当前配置、默认值和用户显式传参合并后交给 `Exporter`，
        是 ONNX、RKNN 前置 ONNX、TensorRT Engine 等部署链路的统一起点。
        """
        self._check_is_paddle_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,
            "verbose": False,
        }
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(self, trainer=None, **kwargs: Any):
        """!
        @brief 启动模型训练流程。
        @param trainer 自定义训练器实例；为空时根据 task 自动加载默认训练器。
        @param kwargs 训练参数，如 `data`、`epochs`、`batch`、`optimizer`、`lr0` 等。
        @return 训练完成后的指标对象；若不可用则返回 `None`。
        @details
        该方法会处理预训练权重加载、断点恢复、trainer 构造、
        训练结束后最佳权重回载等逻辑，是 YOLO Paddle 训练闭环的总调度入口。
        """
        self._check_is_paddle_model()
        checks.check_pip_update_available()
        pretrained_weights = None
        if isinstance(kwargs.get("pretrained", None), (str, Path)):
            # 先保留原始 checkpoint 模型，待 trainer 读取数据集类别数后再迁移。
            # 若此处提前加载到 YAML 默认类别数模型，类别相关 head 会因形状不匹配被丢弃，
            # 后续 trainer 只能从已损坏的中间模型继续构建目标模型。
            pretrained_weights, _ = load_checkpoint(kwargs["pretrained"])
        overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }
        args = {
            **overrides,
            **custom,
            **kwargs,
            "mode": "train",
        }
        if args.get("resume"):
            args["resume"] = self.ckpt_path
        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):
            weights = pretrained_weights if pretrained_weights is not None else (self.model if self.ckpt else None)
            self.trainer.model = self.trainer.get_model(weights=weights, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.train()
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = load_checkpoint(ckpt)
            self.overrides = self._reset_ckpt_args(self.model.args)
            self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics

    def tune(self, use_ray=False, iterations=10, *args: Any, **kwargs: Any):
        """对模型进行超参数调优，可选择使用 Ray Tune。

        该方法支持两种超参数调优模式：Ray Tune 或内部调优器。启用 Ray Tune 时，会调用
        ddyolo26.utils.tuner 模块的 `run_ray_tune`；否则使用内部 `Tuner` 类。方法会合并默认参数、
        覆盖参数和用户参数来配置调优流程。

        参数:
            use_ray (bool): 是否使用 Ray Tune 进行超参数调优；False 时使用内部调优方法。
            iterations (int): 调优迭代次数。
            *args (Any): 传给调优器的额外位置参数。
            **kwargs (Any): 调优配置的额外关键字参数，会与模型覆盖项和默认值合并。

        返回:
            (ray.tune.ResultGrid | None): use_ray=True 时返回包含搜索结果的 ResultGrid；use_ray=False
                时返回 None，并将最佳超参数保存为 YAML。

        异常:
            TypeError: 模型不是 Paddle 模型时抛出。

        示例:
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> results = model.tune(data="coco8.yaml", iterations=5)
            >>> print(results)

            # 使用 Ray Tune 进行更高级的超参数搜索
            >>> results = model.tune(use_ray=True, iterations=20, data="coco8.yaml")
        """
        self._check_is_paddle_model()
        if use_ray:
            from ddyolo26.utils.tuner import run_ray_tune

            return run_ray_tune(self, *args, max_samples=iterations, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}
            return Tuner(args=args, _callbacks=self.callbacks)(iterations=iterations)

    def _apply(self, fn) -> Model:
        """对模型参数、buffer 和张量应用函数。

        该方法在父类 _apply 的基础上额外重置 predictor，并更新模型 overrides 中的设备信息。
        通常用于移动模型设备或改变精度。

        参数:
            fn (Callable): 应用于模型张量的函数，通常是 to()、cpu()、cuda()、half() 或 float() 等方法。

        返回:
            (Model): 已应用函数并更新属性的模型实例。

        异常:
            TypeError: 模型不是 Paddle 模型时抛出。

        示例:
            >>> model = Model("weights/yolov8/yolov8n.pdparams")
            >>> model = model._apply(lambda t: t.cuda())  # 将模型移动到 GPU
        """
        self._check_is_paddle_model()
        self = super()._apply(fn)
        self.predictor = None
        self.overrides["device"] = self.device
        return self

    @property
    def names(self) -> dict[int, str]:
        """!
        @brief 获取当前模型的类别名映射。
        @return `dict[int, str]`，键为类别索引，值为类别名称。
        @details 若模型本体尚未暴露 `names`，则会临时构建预测器以完成类别名解析。
        """
        from ddyolo26.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:
            predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            predictor.setup_model(model=self.model, verbose=False)
            return predictor.model.names
        return self.predictor.model.names

    @property
    def device(self) -> paddle.device:
        """获取模型参数所在设备。

        该属性用于确定模型参数当前存储在 CPU 还是 GPU，仅适用于 paddle.nn.Layer 实例。

        返回:
            (str | None): 模型所在设备（CPU/GPU）；如果模型不是 paddle.nn.Layer 实例则返回 None。

        示例:
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> print(model.device)
            device(type='cuda', index=0)  # CUDA 可用时
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, paddle.nn.Module) else None

    @property
    def transforms(self):
        """获取已加载模型应用于输入数据的变换。

        如果模型中定义了 transforms，该属性会返回它们。transforms 通常包含输入进入模型前执行的 resize、
        normalization 和数据增强等预处理步骤。

        返回:
            (object | None): 模型的 transform 对象；不可用时返回 None。

        示例:
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"模型 transforms: {transforms}")
            ... else:
            ...     print("该模型未定义 transforms。")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """为指定事件添加回调函数。

        该方法允许注册自定义回调，在训练或推理等模型操作期间的指定事件触发。回调可用于扩展和定制模型
        生命周期各阶段的行为。

        参数:
            event (str): 要绑定回调的事件名称，必须是 PaddleYOLO-RKNN 框架识别的有效事件名。
            func (Callable): 要注册的回调函数，会在指定事件发生时调用。

        示例:
            >>> def on_train_start(trainer):
            ...     print("训练开始！")
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """清除指定事件注册的所有回调函数。

        该方法会移除给定事件关联的所有自定义和默认回调函数，将该事件的回调列表重置为空。

        参数:
            event (str): 要清除回调的事件名称，应为 PaddleYOLO-RKNN 回调系统识别的有效事件名。

        示例:
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> model.add_callback("on_train_start", lambda: print("训练开始"))
            >>> model.clear_callback("on_train_start")
            >>> # 'on_train_start' 的所有回调现已移除

        说明:
            - 该方法会同时影响用户添加的自定义回调和 PaddleYOLO-RKNN 框架提供的默认回调。
            - 调用后，在添加新回调前，该事件不会执行任何回调。
            - 请谨慎使用，因为它会移除所有回调，包括某些操作正常运行所需的关键回调。
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """将所有回调重置为默认函数。

        该方法会恢复所有事件的默认回调函数，并移除此前添加的自定义回调。它会遍历所有默认回调事件，
        用默认回调替换当前回调。

        默认回调定义在 `callbacks.default_callbacks` 字典中，包含模型生命周期中各类事件的预定义函数，
        如 on_train_start、on_epoch_end 等。

        当你在自定义修改后希望恢复原始回调集合时，该方法很有用，可确保不同运行或实验之间行为一致。

        示例:
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # 所有回调现已重置为默认函数
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict[str, Any]) -> dict[str, Any]:
        """加载 Paddle 模型 checkpoint 时重置指定参数。

        该方法会过滤输入参数字典，仅保留对模型加载重要的一组键，确保从 checkpoint 加载模型时只保留相关参数，
        丢弃不必要或可能冲突的设置。

        参数:
            args (dict[str, Any]): 包含各类模型参数和设置的字典。

        返回:
            (dict[str, Any]): 仅包含指定 include 键的新字典。

        示例:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls", "overlap_mask", "mask_ratio"}
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key: str):
        """根据模型任务智能加载合适模块。

        该方法会根据当前模型任务和给定 key 动态选择并返回正确模块（model、trainer、validator 或 predictor），
        通过 task_map 字典确定特定任务应加载的模块。

        参数:
            key (str): 要加载的模块类型，必须是 'model'、'trainer'、'validator' 或 'predictor' 之一。

        返回:
            (object): 与指定 key 和当前任务对应的模块类。

        异常:
            NotImplementedError: 当前任务不支持指定 key 时抛出。

        示例:
            >>> model = Model(task="detect")
            >>> predictor_class = model._smart_load("predictor")
            >>> trainer_class = model._smart_load("trainer")
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]
            raise NotImplementedError(f"'{name}' 模型不支持 task='{self.task}' 的 '{mode}' 模式。") from e

    @property
    def task_map(self) -> dict:
        """提供模型任务到不同模式对应类的映射。

        该属性返回一个字典，将每个支持的任务（detect 或 segment）映射到嵌套字典；嵌套字典中包含不同运行模式
        （model、trainer、validator、predictor）到对应类实现的映射。

        该映射允许根据模型任务和目标运行模式动态加载合适类，让 PaddleYOLO-RKNN 框架能够灵活处理不同任务
        和模式。

        返回:
            (dict[str, dict[str, Any]]): 任务名到嵌套字典的映射；每个嵌套字典包含 'model'、'trainer'、
                'validator'、'predictor' 键到该任务对应类实现的映射。

        示例:
            >>> model = Model("weights/yolov8/yolov8n.pdparams")
            >>> task_map = model.task_map
            >>> detect_predictor = task_map["detect"]["predictor"]
            >>> segment_trainer = task_map["segment"]["trainer"]
        """
        raise NotImplementedError("请为模型提供 task map！")

    def eval(self):
        """将模型设置为评估模式。

        该方法会将模型切换到评估模式，影响 dropout 和 batch normalization 等在训练/评估阶段行为不同的层。
        在评估模式中，这些层会使用 running statistics 而不是当前 batch 统计，dropout 也会被禁用。

        返回:
            (Model): 已设置为评估模式的模型实例。

        示例:
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> model.eval()
            >>> # 模型现在处于推理用评估模式
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """允许通过 Model 类直接访问底层模型属性。

        该方法提供从 Model 实例直接访问底层模型属性的方式。若请求属性为 'model'，则从模块字典返回模型；
        否则将属性查找委托给底层模型。

        参数:
            name (str): 要获取的属性名。

        返回:
            (Any): 请求的属性值。

        异常:
            AttributeError: 请求的属性不存在于模型中时抛出。

        示例:
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> print(model.stride)  # 访问 model.stride 属性
            >>> print(model.names)  # 访问 model.names 属性
        """
        return object.__getattribute__(self, "_sub_layers")["model"] if name == "model" else getattr(self.model, name)
