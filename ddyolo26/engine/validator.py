# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 通用验证基类 BaseValidator，管理模型评估流程。
@details
在 val/predict 时被调用，负责：
- 加载测试集 DataLoader
- 按批次运行模型推理
- 汇总统计量（mAP、精度、召回等）
- 输出表格日志与可视化绘图

子类（如 DetectionValidator）重写 `postprocess`、`update_metrics`、
`get_stats` 等方法以实现任务特定的指标计算。
"""

from __future__ import annotations

import paddle

"""
检查模型在数据集 test 或 val 划分上的精度。

用法:
    $ yolo mode=val model=weights/yolov8/yolov8n.pdparams data=coco8.yaml imgsz=640

用法 - 模型格式:
    $ yolo mode=val model=weights/yolov8/yolov8n.pdparams  # PaddlePaddle
                          yolov8n.onnx                    # ONNX Runtime 或 OpenCV DNN（dnn=True）
                          yolov8n.rknn                    # Rockchip RKNN
"""
import json
import time
from pathlib import Path

import numpy as np

from ddyolo26.cfg import get_cfg, get_save_dir
from ddyolo26.data.utils import check_cls_dataset, check_det_dataset
from ddyolo26.nn.autobackend import AutoBackend
from ddyolo26.utils import LOGGER, RANK, TQDM, callbacks, colorstr, emojis
from ddyolo26.utils.checks import check_imgsz
from ddyolo26.utils.ops import Profile
from ddyolo26.utils.runtime import attempt_compile, select_device, smart_inference_mode, unwrap_model


class BaseValidator:
    """用于创建验证器的基类。

    该类提供验证流程的基础能力，包括模型评估、指标计算和结果可视化。

    属性:
        args (SimpleNamespace): 验证器配置。
        dataloader (DataLoader): 验证使用的数据加载器。
        model (nn.Module): 待验证模型。
        data (dict): 包含数据集信息的数据字典。
        device (str): 验证使用的设备。
        batch_i (int): 当前 batch 索引。
        training (bool): 模型是否处于训练流程中。
        names (dict): 类别名映射。
        seen (int): 验证期间已经处理的图像数量。
        stats (dict): 验证期间收集的统计信息。
        confusion_matrix: 分类评估用混淆矩阵。
        nc (int): 类别数量。
        iouv (paddle.Tensor): 从 0.50 到 0.95、步长 0.05 的 IoU 阈值。
        jdict (list): JSON 验证结果列表。
        speed (dict): 包含 'preprocess'、'inference'、'loss'、'postprocess' 的耗时统计。
        save_dir (Path): 结果保存目录。
        plots (dict): 可视化绘图记录。
        callbacks (dict): 各类回调函数。
        stride (int): 用于 padding 计算的模型步长。
        loss (paddle.Tensor): 训练期验证累计损失。

    方法:
        __call__: 执行验证流程，在 dataloader 上推理并计算性能指标。
        match_predictions: 使用 IoU 匹配预测结果和真值目标。
        add_callback: 向指定事件追加回调。
        run_callbacks: 运行指定事件关联的所有回调。
        get_dataloader: 根据数据集路径和 batch size 获取数据加载器。
        build_dataset: 根据图像路径构建数据集。
        preprocess: 预处理输入 batch。
        postprocess: 后处理预测结果。
        init_metrics: 初始化 YOLO 模型性能指标。
        update_metrics: 根据预测和 batch 更新指标。
        finalize_metrics: 完成并返回所有指标。
        get_stats: 返回模型性能统计。
        print_results: 打印模型预测结果。
        get_desc: 获取 YOLO 模型描述。
        on_plot: 注册可视化绘图。
        plot_val_samples: 训练期间绘制验证样本。
        plot_predictions: 在 batch 图像上绘制 YOLO 模型预测。
        pred_to_json: 将预测转换为 JSON 格式。
        eval_json: 评估并返回 JSON 预测统计。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """初始化 BaseValidator 实例。

        参数:
            dataloader (paddle.io.DataLoader, optional): 验证使用的数据加载器。
            save_dir (Path, optional): 结果保存目录。
            args (SimpleNamespace, optional): 验证器配置。
            _callbacks (dict, optional): 各类回调函数字典。
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }
        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.01 if self.args.task == "obb" else 0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)
        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """执行验证流程，在 dataloader 上推理并计算性能指标。

        参数:
            trainer (object, optional): 包含待验证模型的训练器对象。
            model (nn.Module, optional): 不通过训练器验证时使用的模型。

        返回:
            (dict): 包含验证统计信息的字典。
        """
        self.training = trainer is not None
        augment = self.args.augment and not self.training
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Paddle BatchNorm 不支持 fp16 参数；验证阶段始终使用 fp32。
            self.args.half = False
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod
            model = model.float()
            self.loss = paddle.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or trainer.epoch == trainer.epochs - 1
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("正在验证未训练的 YAML 模型，结果 mAP 将为 0。")
            callbacks.add_integration_callbacks(self)
            if hasattr(model, "end2end"):
                if self.args.end2end is not None:
                    model.end2end = self.args.end2end
                if model.end2end:
                    model.set_head_attr(max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device) if RANK == -1 else paddle.device("cuda", RANK),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device
            self.args.half = model.fp16
            stride, pt, jit = model.stride, model.pt, model.jit
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(f"设置 batch={self.args.batch}，输入形状为 ({self.args.batch}, 3, {imgsz}, {imgsz})")
            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"未找到 task={self.args.task} 的数据集 '{self.args.data}' ❌"))
            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not (pt or getattr(model, "dynamic", False) and not model.imx):
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            model.warmup(
                imgsz=(
                    1 if pt else self.args.batch,
                    self.data["channels"],
                    imgsz,
                    imgsz,
                )
            )
        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []
        try:
            for batch_i, batch in enumerate(bar):
                self.run_callbacks("on_val_batch_start")
                self.batch_i = batch_i
                with dt[0]:
                    batch = self.preprocess(batch)
                with dt[1]:
                    preds = model(batch["img"], augment=augment)
                with dt[2]:
                    if self.training:
                        self.loss += model.loss(batch, preds)[1]
                with dt[3]:
                    preds = self.postprocess(preds)
                self.update_metrics(preds, batch)
                if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                    self.plot_val_samples(batch, batch_i)
                    self.plot_predictions(batch, preds, batch_i)
                self.run_callbacks("on_val_batch_end")
        finally:
            close = getattr(self.dataloader, "close", None)
            if callable(close):
                close()
        stats = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            self.speed = dict(
                zip(
                    self.speed.keys(),
                    (x.t / len(self.dataloader.dataset) * 1000.0 for x in dt),
                )
            )
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            loss = self.loss.clone().detach()
            if trainer.world_size > 1:
                paddle.distributed.reduce(tensor=loss, dst=0, op=paddle.distributed.ReduceOp.AVG)
            if RANK > 0:
                return
            results = {
                **stats,
                **trainer.label_loss_items(loss.cpu() / len(self.dataloader), prefix="val"),
            }
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            if RANK > 0:
                return stats
            LOGGER.info(
                "速度: 每张图 {:.1f}ms 预处理, {:.1f}ms 推理, {:.1f}ms 损失, {:.1f}ms 后处理".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"正在保存 {f.name}...")
                    json.dump(self.jdict, f)
                stats = self.eval_json(stats)
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"结果已保存到 {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(
        self,
        pred_classes: paddle.Tensor,
        true_classes: paddle.Tensor,
        iou: paddle.Tensor,
        use_scipy: bool = False,
    ) -> paddle.Tensor:
        """使用 IoU 将预测结果与真值目标匹配。

        参数:
            pred_classes (paddle.Tensor): 形状为 (N,) 的预测类别索引。
            true_classes (paddle.Tensor): 形状为 (M,) 的目标类别索引。
            iou (paddle.Tensor): 形状为 N x M 的张量，包含预测与真值之间的两两 IoU。
            use_scipy (bool, optional): 是否使用 scipy 匹配（更精确）。

        返回:
            (paddle.Tensor): 形状为 (N, 10) 的正确匹配矩阵，对应 10 个 IoU 阈值。
        """
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class.astype(iou.dtype)
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                import scipy

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return paddle.tensor(correct, dtype=paddle.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """向指定事件追加回调。"""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """运行指定事件关联的所有回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """根据数据集路径和 batch size 获取数据加载器。"""
        raise NotImplementedError("该验证器未实现 get_dataloader 函数")

    def build_dataset(self, img_path):
        """根据图像路径构建数据集。"""
        raise NotImplementedError("验证器未实现 build_dataset 函数")

    def preprocess(self, batch):
        """预处理输入 batch。"""
        return batch

    def postprocess(self, preds):
        """后处理预测结果。"""
        return preds

    def init_metrics(self, model):
        """初始化 YOLO 模型性能指标。"""
        pass

    def update_metrics(self, preds, batch):
        """根据预测和 batch 更新指标。"""
        pass

    def finalize_metrics(self):
        """完成并返回所有指标。"""
        pass

    def get_stats(self):
        """返回模型性能统计。"""
        return {}

    def gather_stats(self):
        """DDP 训练期间从所有 GPU 汇总统计信息到 0 号 GPU。"""
        pass

    def print_results(self):
        """打印模型预测结果。"""
        pass

    def get_desc(self):
        """获取 YOLO 模型描述。"""
        pass

    @property
    def metric_keys(self):
        """返回 YOLO 训练/验证使用的指标键。"""
        return []

    def on_plot(self, name, data=None):
        """注册可视化绘图，并按类型去重。"""
        plot_type = data.get("type") if data else None
        if plot_type and any((v.get("data") or {}).get("type") == plot_type for v in self.plots.values()):
            return
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def plot_val_samples(self, batch, ni):
        """训练期间绘制验证样本。"""
        pass

    def plot_predictions(self, batch, preds, ni):
        """在 batch 图像上绘制 YOLO 模型预测。"""
        pass

    def pred_to_json(self, preds, batch):
        """将预测转换为 JSON 格式。"""
        pass

    def eval_json(self, stats):
        """评估并返回 JSON 预测统计。"""
        pass
