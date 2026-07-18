# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 推理结果封装：Results、Boxes、Masks 等结果容器类。
@details
Results 类将模型单张图片的推理输出（边界框、置信度、类别标签等）
封装为易操作的对象，提供：
- `plot()`：将检测结果叠加绘制到图像上
- `save()` / `show()`：保存或显示结果图片
- `tojson()` / `todf()`：结果序列化
- `.boxes`：Boxes 子容器，提供 xyxy/xywh/xyxyn/xywhn 多坐标格式

支持 YOLO26 end2end 输出的 `(N, 300, 6)` 格式：[x1,y1,x2,y2,conf,cls]。
"""

from __future__ import annotations
import sys
import paddle


from ddyolo26.paddle_utils import *

"""
PaddleYOLO-RKNN 的 Results、Boxes、Masks、Keypoints、Probs 和 OBB 类，用于处理推理结果。

用法请参考 PaddleYOLO-RKNN 仓库 README。
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ddyolo26.data.augment import LetterBox
from ddyolo26.utils import LOGGER, DataExportMixin, SimpleClass, ops
from ddyolo26.utils.plotting import Annotator, colors, save_one_box


class BaseTensor(SimpleClass):
    """基础张量类，提供便捷操作和设备处理方法。

    该类为带设备管理能力的类张量对象提供基础实现，支持 Paddle 张量和 NumPy 数组，并提供设备迁移和
    张量类型转换方法。

    属性:
        data (paddle.Tensor | np.ndarray): 预测数据，如边界框、掩码或关键点。
        orig_shape (tuple[int, int]): 原始图像形状，通常为 (height, width)。

    方法:
        cpu: 返回存储在 CPU 内存中的张量副本。
        numpy: 返回转换为 numpy 数组的张量副本。
        cuda: 将张量移动到 GPU 内存，必要时返回新实例。
        to: 返回指定设备和 dtype 上的张量副本。

    示例:
        >>> data = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor = base_tensor.cuda()
    """

    def __init__(self, data: (paddle.Tensor | np.ndarray), orig_shape: tuple[int, int]) -> None:
        """使用预测数据和原始图像形状初始化 BaseTensor。

        参数:
            data (paddle.Tensor | np.ndarray): 预测数据，如边界框、掩码或关键点。
            orig_shape (tuple[int, int]): 原始图像形状，格式为 (height, width)。
        """
        assert isinstance(data, (paddle.Tensor, np.ndarray)), "data 必须是 paddle.Tensor 或 np.ndarray"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """返回底层数据张量的形状。

        返回:
            (tuple[int, ...]): 数据张量形状。

        示例:
            >>> data = paddle.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape

    def cpu(self):
        """返回存储在 CPU 内存中的张量副本。

        返回:
            (BaseTensor): 数据张量已移动到 CPU 内存的新 BaseTensor 对象。

        示例:
            >>> data = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """返回数据已转换为 NumPy 数组的对象副本。

        返回:
            (BaseTensor): `data` 为 NumPy 数组的新实例。

        示例:
            >>> data = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_tensor = base_tensor.numpy()
            >>> print(type(numpy_tensor.data))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """将张量移动到 GPU 内存。

            返回:
                (BaseTensor): 数据已移动到 GPU 内存的新 BaseTensor 实例。

        示例:
                >>> from ddyolo26.engine.results import BaseTensor
                >>> data = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
                >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
                >>> gpu_tensor = base_tensor.cuda()
                >>> print(gpu_tensor.data.device)
                cuda:0
        """
        return self.__class__(paddle.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """返回指定设备和 dtype 上的张量副本。

        参数:
            *args (Any): 传给 paddle.Tensor.to() 的可变位置参数。
            **kwargs (Any): 传给 paddle.Tensor.to() 的任意关键字参数。

        返回:
            (BaseTensor): 数据已移动到指定设备和/或 dtype 的新 BaseTensor 实例。

        示例:
            >>> base_tensor = BaseTensor(paddle.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=paddle.float16)
        """
        return self.__class__(paddle.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self) -> int:
        """返回底层数据张量长度。

        返回:
            (int): 数据张量第一维的元素数量。

        示例:
            >>> data = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx):
        """返回包含指定索引元素的新 BaseTensor 实例。

        参数:
            idx (int | list[int] | paddle.Tensor): 从数据张量中选择的索引。

        返回:
            (BaseTensor): 包含索引数据的新 BaseTensor 实例。

        示例:
            >>> data = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # 选择第一行
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass, DataExportMixin):
    """用于存储和操作推理结果的类。

    该类提供 PaddleYOLO-RKNN 模型推理结果处理能力，包括检测、分割、分类和姿态估计，支持可视化、
    数据导出以及多种坐标变换。

    属性:
        orig_img (np.ndarray): 原始图像，类型为 numpy 数组。
        orig_shape (tuple[int, int]): 原始图像形状，格式为 (height, width)。
        boxes (Boxes | None): 检测到的边界框。
        masks (Masks | None): 分割掩码。
        probs (Probs | None): 分类概率。
        keypoints (Keypoints | None): 检测到的关键点。
        obb (OBB | None): 有向边界框。
        speed (dict): 包含推理速度信息的字典。
        names (dict): 类别索引到类别名的映射字典。
        path (str): 输入图像文件路径。
        save_dir (str | None): 结果保存目录。

    方法:
        update: 使用新的检测数据更新 Results 对象。
        cpu: 返回所有张量已移动到 CPU 内存的 Results 副本。
        numpy: 将 Results 对象中的所有张量转换为 numpy 数组。
        cuda: 将 Results 对象中的所有张量移动到 GPU 内存。
        to: 将所有张量移动到指定设备和 dtype。
        new: 创建具有相同图像、路径、类别名和速度属性的新 Results 对象。
        plot: 在输入 BGR 图像上绘制检测结果。
        show: 显示带推理标注的图像。
        save: 将带标注的推理结果图像保存到文件。
        verbose: 返回结果中各任务的日志字符串。
        save_txt: 将检测结果保存到文本文件。
        save_crop: 将裁剪出的检测图像保存到指定目录。
        summary: 将推理结果转换为摘要字典。
        to_df: 将检测结果转换为 Polars DataFrame。
        to_json: 将检测结果转换为 JSON 格式。
        to_csv: 将检测结果转换为 CSV 格式。

    示例:
        >>> results = model("path/to/image.jpg")
        >>> result = results[0]  # 获取第一个结果
        >>> boxes = result.boxes  # 获取第一个结果的检测框
        >>> masks = result.masks  # 获取第一个结果的掩码
        >>> for result in results:
        ...     result.plot()  # 绘制检测结果
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict[int, str],
        boxes: (paddle.Tensor | None) = None,
        masks: (paddle.Tensor | None) = None,
        probs: (paddle.Tensor | None) = None,
        keypoints: (paddle.Tensor | None) = None,
        obb: (paddle.Tensor | None) = None,
        speed: (dict[str, float] | None) = None,
    ) -> None:
        """初始化用于存储和操作推理结果的 Results 类。

        参数:
            orig_img (np.ndarray): 原始图像，类型为 numpy 数组。
            path (str): 图像文件路径。
            names (dict): 类别名称字典。
            boxes (paddle.Tensor | None): 每个检测目标的边界框坐标二维张量。
            masks (paddle.Tensor | None): 检测掩码三维张量，每个掩码是一张二值图。
            probs (paddle.Tensor | None): 分类任务中每个类别概率的一维张量。
            keypoints (paddle.Tensor | None): 每个检测目标的关键点坐标二维张量。
            obb (paddle.Tensor | None): 每个检测目标的有向边界框坐标二维张量。
            speed (dict | None): 包含 preprocess、inference、postprocess 耗时的字典（ms/image）。

        说明:
            默认人体姿态模型的关键点索引为：
            0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
            5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
            9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
            13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """返回指定索引对应的 Results 对象。

        参数:
            idx (int | slice): 从 Results 对象中取出的索引或切片。

        返回:
            (Results): 包含指定推理结果子集的新 Results 对象。

        示例:
            >>> results = model("path/to/image.jpg")  # 执行推理
            >>> single_result = results[0]  # 获取第一个结果
            >>> subset_results = results[1:4]  # 获取结果切片
        """
        return self._apply("__getitem__", idx)

    def __len__(self) -> int:
        """返回 Results 对象中的检测数量。

        返回:
            (int): 检测数量，由 (boxes, masks, probs, keypoints, obb) 中第一个非空属性的长度决定。

        示例:
            >>> results = Results(orig_img, path, names, boxes=paddle.rand(5, 6))
            >>> len(results)
            5
        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(
        self,
        boxes: (paddle.Tensor | None) = None,
        masks: (paddle.Tensor | None) = None,
        probs: (paddle.Tensor | None) = None,
        obb: (paddle.Tensor | None) = None,
        keypoints: (paddle.Tensor | None) = None,
    ):
        """使用新的检测数据更新 Results 对象。

        该方法允许更新 Results 对象中的 boxes、masks、keypoints、probs 和 oriented bounding boxes (OBB)，
        并确保边界框裁剪到原始图像尺寸范围内。

        参数:
            boxes (paddle.Tensor | None): 形状为 (N, 6) 的张量，包含边界框坐标和置信度分数，
                格式为 (x1, y1, x2, y2, conf, class)。
            masks (paddle.Tensor | None): 形状为 (N, H, W) 的分割掩码张量。
            probs (paddle.Tensor | None): 形状为 (num_classes,) 的类别概率张量。
            obb (paddle.Tensor | None): 形状为 (N, 7) 或 (N, 8) 的有向边界框坐标张量。
            keypoints (paddle.Tensor | None): 形状为 (N, K, 3) 的关键点张量，人体任务中 K=17。

        示例:
            >>> results = model("image.jpg")
            >>> new_boxes = paddle.to_tensor([[100, 100, 200, 200, 0.9, 0]])
            >>> results[0].update(boxes=new_boxes)
        """
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)
        if keypoints is not None:
            self.keypoints = Keypoints(keypoints, self.orig_shape)

    def _apply(self, fn: str, *args, **kwargs):
        """对所有非空属性应用函数，并返回属性已修改的新 Results 对象。

        该方法由 .to()、.cuda()、.cpu() 等方法在内部调用。

        参数:
            fn (str): 要应用的函数名。
            *args (Any): 传给函数的可变位置参数。
            **kwargs (Any): 传给函数的任意关键字参数。

        返回:
            (Results): 应用函数后属性已修改的新 Results 对象。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result_cuda = result.cuda()
            ...     result_cpu = result.cpu()
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """返回所有张量已移动到 CPU 内存的 Results 对象副本。

        该方法会创建新的 Results 对象，并将所有张量属性（boxes、masks、probs、keypoints、obb）
        转移到 CPU 内存，便于后续处理或保存。

        返回:
            (Results): 所有张量属性都位于 CPU 内存的新 Results 对象。

        示例:
            >>> results = model("path/to/image.jpg")  # 执行推理
            >>> cpu_result = results[0].cpu()  # 将第一个结果移动到 CPU
            >>> print(cpu_result.boxes.device)  # 输出: cpu
        """
        return self._apply("cpu")

    def numpy(self):
        """将 Results 对象中的所有张量转换为 numpy 数组。

        返回:
            (Results): 所有张量都已转换为 numpy 数组的新 Results 对象。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> numpy_result = results[0].numpy()
            >>> type(numpy_result.boxes.data)
            <class 'numpy.ndarray'>

        说明:
            该方法会创建新的 Results 对象，不修改原对象；适用于 numpy 生态互操作或需要 CPU 运算的场景。
        """
        return self._apply("numpy")

    def cuda(self):
        """将 Results 对象中的所有张量移动到 GPU 内存。

        返回:
            (Results): 所有张量都已移动到 CUDA 设备的新 Results 对象。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> cuda_results = results[0].cuda()  # 将第一个结果移动到 GPU
            >>> for result in results:
            ...     result_cuda = result.cuda()  # 将每个结果移动到 GPU
        """
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """将 Results 对象中的所有张量移动到指定设备和 dtype。

        参数:
            *args (Any): 传给 paddle.Tensor.to() 的可变位置参数。
            **kwargs (Any): 传给 paddle.Tensor.to() 的任意关键字参数。

        返回:
            (Results): 所有张量都已移动到指定设备和 dtype 的新 Results 对象。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> result_cuda = results[0].to("cuda")  # 将第一个结果移动到 GPU
            >>> result_cpu = results[0].to("cpu")  # 将第一个结果移动到 CPU
            >>> result_half = results[0].to(dtype=paddle.float16)  # 将第一个结果转换为半精度
        """
        return self._apply("to", *args, **kwargs)

    def new(self):
        """创建具有相同图像、路径、类别名和速度属性的新 Results 对象。

        返回:
            (Results): 从原实例复制属性得到的新 Results 对象。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf: bool = True,
        line_width: (float | None) = None,
        font_size: (float | None) = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: (np.ndarray | None) = None,
        im_gpu: (paddle.Tensor | None) = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: (str | None) = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """在输入 BGR 图像上绘制检测结果。

        参数:
            conf (bool): 是否绘制检测置信度分数。
            line_width (float | None): 边界框线宽；为 None 时按图像尺寸缩放。
            font_size (float | None): 文本字号；为 None 时按图像尺寸缩放。
            font (str): 文本字体。
            pil (bool): 是否返回 PIL Image。
            img (np.ndarray | None): 绘制目标图像；为 None 时使用原始图像。
            im_gpu (paddle.Tensor | None): GPU 上的归一化图像，用于更快绘制掩码。
            kpt_radius (int): 关键点绘制半径。
            kpt_line (bool): 是否绘制连接关键点的线。
            labels (bool): 是否绘制边界框标签。
            boxes (bool): 是否绘制边界框。
            masks (bool): 是否绘制掩码。
            probs (bool): 是否绘制分类概率。
            show (bool): 是否显示带标注图像。
            save (bool): 是否保存带标注图像。
            filename (str | None): save 为 True 时的图像保存文件名。
            color_mode (str): 颜色模式，如 'instance' 或 'class'。
            txt_color (tuple[int, int, int]): 分类输出文本颜色，BGR 格式。

        返回:
            (np.ndarray | PIL.Image.Image): 带标注图像；`pil=True` 时为 PIL 图像（RGB），否则为 NumPy 数组（BGR）。

        示例:
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     im = result.plot()
            ...     im.show()
        """
        assert color_mode in {"instance", "class"}, f"color_mode 应为 'instance' 或 'class'，而不是 {color_mode}。"
        if img is None and isinstance(self.orig_img, paddle.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()
        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or pred_probs is not None and show_probs,
            example=names,
        )
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    paddle.as_tensor(img, dtype=paddle.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(axis=0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.is_track and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(
                pred_masks.data,
                colors_list=[colors(x, True) for x in idx],
                im_gpu=im_gpu,
            )
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c, d_conf, id = (
                    int(d.cls.flatten()[0]),
                    float(d.conf.flatten()[0]) if conf else None,
                    int(d.id.item()) if d.is_track else None,
                )
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {d_conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        c
                        if color_mode == "class"
                        else id
                        if id is not None
                        else i
                        if color_mode == "instance"
                        else None,
                        True,
                    ),
                )
        if pred_probs is not None and show_probs:
            text = "\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128))
        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.data)):
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )
        if show:
            annotator.show(self.path)
        if save:
            annotator.save(filename or f"results_{Path(self.path).name}")
        return annotator.result(pil)

    def show(self, *args, **kwargs):
        """显示带推理标注的图像。

        该方法会在原始图像上绘制检测结果并显示，便于直接可视化模型预测。

        参数:
            *args (Any): 传给 `plot()` 方法的可变位置参数。
            **kwargs (Any): 传给 `plot()` 方法的任意关键字参数。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> results[0].show()  # 显示第一个结果
            >>> for result in results:
            ...     result.show()  # 显示所有结果
        """
        self.plot(*args, show=True, **kwargs)

    def save(self, filename: (str | None) = None, *args, **kwargs) -> str:
        """将带标注的推理结果图像保存到文件。

        该方法会在原始图像上绘制检测结果，并通过 `plot` 生成带标注图像后保存到指定文件名。

        参数:
            filename (str | None): 带标注图像的保存文件名；为 None 时根据原始图像路径生成默认文件名。
            *args (Any): 传给 `plot` 方法的可变位置参数。
            **kwargs (Any): 传给 `plot` 方法的任意关键字参数。

        返回:
            (str): 图像保存文件名。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save("annotated_image.jpg")
            >>> # 使用自定义绘图参数
            >>> for result in results:
            ...     result.save("annotated_image.jpg", conf=False, line_width=2)
            >>> # 目录不存在时会自动创建
            >>> result.save("path/to/annotated_image.jpg")
        """
        if not filename:
            filename = f"results_{Path(self.path).name}"
        Path(filename).absolute().parent.mkdir(parents=True, exist_ok=True)
        self.plot(*args, save=True, filename=filename, **kwargs)
        return filename

    def verbose(self) -> str:
        """返回各任务结果的日志字符串，描述检测和分类输出。

        该方法会生成面向人的摘要字符串：检测任务包含每个类别的检测数量，分类任务包含 top 概率。

        返回:
            (str): 包含结果摘要的格式化字符串。检测任务包含逐类别检测数量，分类任务包含 top 5 类别概率。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     print(result.verbose())
            2 persons, 1 car, 3 traffic lights,
            dog 0.92, cat 0.78, horse 0.64,

        说明:
            - 检测任务无检测结果时返回 "(no detections), "。
            - 分类任务返回 top 5 类别概率及对应类别名。
            - 返回字符串以逗号分隔，并以逗号和空格结尾。
        """
        boxes = self.obb if self.obb is not None else self.boxes
        if len(self) == 0:
            return "" if self.probs is not None else "(no detections), "
        if self.probs is not None:
            return f"{', '.join(f'{self.names[j]} {self.probs.data[j]:.2f}' for j in self.probs.top5)}, "
        if boxes:
            counts = boxes.cls.int().bincount()
            return "".join(f"{n} {self.names[i]}{'s' * (n > 1)}, " for i, n in enumerate(counts) if n > 0)

    def save_txt(self, txt_file: (str | Path), save_conf: bool = False) -> str:
        """将检测结果保存到文本文件。

        参数:
            txt_file (str | Path): 输出文本文件路径。
            save_conf (bool): 输出中是否包含置信度分数。

        返回:
            (str): 已保存文本文件路径。

        示例:
            >>> from ddyolo26 import YOLO
            >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_txt("output.txt")

        说明:
            - 文件中每个检测或分类结果占一行，结构如下：
              - 检测：`class x_center y_center width height [confidence] [track_id]`
              - 分类：`confidence class_name`
              - 掩码和关键点的具体格式会按任务变化。
            - 输出目录不存在时会自动创建。
            - save_conf 为 False 时不输出置信度分数。
            - 文件已有内容不会被覆盖，新结果会追加写入。
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), int(d.id.item()) if d.is_track else None
                line = c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)
                    line = c, *seg
                if kpts is not None:
                    kpt = paddle.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)
        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(text + "\n" for text in texts)
        return str(txt_file)

    def save_crop(self, save_dir: (str | Path), file_name: (str | Path) = Path("im.jpg")):
        """将检测目标裁剪图保存到指定目录。

        该方法会将检测目标裁剪后保存到指定目录。每个裁剪图会保存到以目标类别命名的子目录中，
        文件名基于输入 file_name。

        参数:
            save_dir (str | Path): 裁剪图保存目录路径。
            file_name (str | Path): 保存裁剪图的基础文件名。

        示例:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_crop(save_dir="path/to/crops", file_name="detection")

        说明:
            - 该方法不支持分类或有向边界框（OBB）任务。
            - 裁剪图保存为 'save_dir/class_name/file_name.jpg'。
            - 必要子目录不存在时会自动创建。
            - 裁剪前会复制原图，避免修改原始图像。
        """
        if self.probs is not None:
            LOGGER.warning("分类任务不支持 `save_crop`。")
            return
        if self.obb is not None:
            LOGGER.warning("OBB 任务不支持 `save_crop`。")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / Path(file_name).with_suffix(".jpg"),
                BGR=True,
            )

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, Any]]:
        """将推理结果转换为摘要字典，可选择归一化框坐标。

        该方法会创建检测字典列表，每个字典包含一个检测或分类结果的信息。分类任务返回 top 5 类别及置信度；
        检测任务包含类别信息、边界框坐标，并可选包含掩码轮廓和关键点。

        参数:
            normalize (bool): 是否按图像尺寸归一化边界框坐标。
            decimals (int): 输出数值保留的小数位数。

        返回:
            (list[dict[str, Any]]): 摘要字典列表，每个字典对应一个检测或分类结果；结构会随任务类型
                （分类或检测）以及可用信息（boxes、masks、keypoints）变化。

        示例:
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     summary = result.summary()
            ...     print(summary)
        """
        results = []
        if self.probs is not None:
            for class_id, conf in zip(self.probs.top5, self.probs.top5conf.tolist()):
                class_id = int(class_id)
                results.append({"name": self.names[class_id], "class": class_id, "confidence": round(conf, decimals)})
            return results
        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):
            class_id, conf = int(row.cls), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                kpt = self.keypoints[i]
                if kpt.has_visible:
                    x, y, visible = kpt.data[0].cpu().unbind(dim=1)
                else:
                    x, y = kpt.data[0].cpu().unbind(dim=1)
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),
                    "y": (y / h).numpy().round(decimals).tolist(),
                }
                if kpt.has_visible:
                    result["keypoints"]["visible"] = visible.numpy().round(decimals).tolist()
            results.append(result)
        return results


class Boxes(BaseTensor):
    """用于管理和操作检测框的类。

    该类提供检测框处理能力，包括坐标、置信度分数、类别标签和可选跟踪 ID，支持多种框格式，并提供不同
    坐标系之间的便捷转换。

    属性:
        data (paddle.Tensor | np.ndarray): 包含检测框及相关数据的原始张量。
        orig_shape (tuple[int, int]): 原始图像尺寸 (height, width)。
        is_track (bool): 框数据中是否包含跟踪 ID。
        xyxy (paddle.Tensor | np.ndarray): [x1, y1, x2, y2] 格式的框。
        conf (paddle.Tensor | np.ndarray): 每个框的置信度分数。
        cls (paddle.Tensor | np.ndarray): 每个框的类别标签。
        id (paddle.Tensor | None): 每个框的跟踪 ID（如可用）。
        xywh (paddle.Tensor | np.ndarray): [x, y, width, height] 格式的框。
        xyxyn (paddle.Tensor | np.ndarray): 相对 orig_shape 归一化的 [x1, y1, x2, y2] 框。
        xywhn (paddle.Tensor | np.ndarray): 相对 orig_shape 归一化的 [x, y, width, height] 框。

    方法:
        cpu: 返回所有张量位于 CPU 内存的对象副本。
        numpy: 返回所有张量为 numpy 数组的对象副本。
        cuda: 返回所有张量位于 GPU 内存的对象副本。
        to: 返回张量位于指定设备和 dtype 的对象副本。

    示例:
        >>> boxes_data = paddle.to_tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # 高、宽
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """

    def __init__(self, boxes: (paddle.Tensor | np.ndarray), orig_shape: tuple[int, int]) -> None:
        """使用检测框数据和原始图像形状初始化 Boxes。

        该类管理检测框，可便捷访问和操作框坐标、置信度分数、类别标识符以及可选跟踪 ID，
        支持绝对坐标和归一化坐标等多种框格式。

        参数:
            boxes (paddle.Tensor | np.ndarray): 形状为 (num_boxes, 6) 或 (num_boxes, 7) 的检测框张量或
                numpy 数组，列应包含 [x1, y1, x2, y2, (optional) track_id, confidence, class]。
            orig_shape (tuple[int, int]): 原始图像形状 (height, width)，用于归一化。
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"期望 6 或 7 个值，但得到 {n} 个"
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> paddle.Tensor | np.ndarray:
        """返回 [x1, y1, x2, y2] 格式的边界框。

        返回:
            (paddle.Tensor | np.ndarray): 形状为 (n, 4) 的张量或 numpy 数组，包含 [x1, y1, x2, y2]
                格式的边界框坐标，n 为框数量。

        示例:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> xyxy = boxes.xyxy
            >>> print(xyxy)
        """
        return self.data[:, :4]

    @property
    def conf(self) -> paddle.Tensor | np.ndarray:
        """返回每个检测框的置信度分数。

        返回:
            (paddle.Tensor | np.ndarray): 一维张量或数组，包含每个检测的置信度分数，形状为 (N,)。

        示例:
            >>> boxes = Boxes(paddle.to_tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        return self.data[:, -2]

    @property
    def cls(self) -> paddle.Tensor | np.ndarray:
        """返回表示每个边界框类别预测的 class ID 张量。

        返回:
            (paddle.Tensor | np.ndarray): 包含每个检测框 class ID 的张量或 numpy 数组，形状为 (N,)。

        示例:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        return self.data[:, -1]

    @property
    def id(self) -> paddle.Tensor | np.ndarray | None:
        """返回每个检测框的跟踪 ID（如果可用）。

        返回:
            (paddle.Tensor | np.ndarray | None): 启用跟踪时返回包含每个框跟踪 ID 的张量或数组，否则返回 None。
                形状为 (N,)，N 为框数量。

        示例:
            >>> boxes = results[0].boxes
            >>> if boxes.is_track:
            ...     print(boxes.id)

        说明:
            - 仅当框张量中存在跟踪 ID 时，该属性才有值。
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xywh(self) -> paddle.Tensor | np.ndarray:
        """将边界框从 [x1, y1, x2, y2] 格式转换为 [x, y, width, height] 格式。

        返回:
            (paddle.Tensor | np.ndarray): [x_center, y_center, width, height] 格式的框，其中 x_center、y_center
                为边界框中心点坐标，width、height 为边界框尺寸；返回张量形状为 (N, 4)。

        示例:
            >>> boxes = Boxes(
            ...     paddle.to_tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]]), orig_shape=(480, 640)
            ... )
            >>> xywh = boxes.xywh
            >>> print(xywh)
            tensor([[125.0000,  75.0000,  50.0000,  50.0000],
                    [250.0000, 200.0000, 100.0000, 100.0000]])
        """
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self) -> paddle.Tensor | np.ndarray:
        """返回相对原始图像尺寸归一化的边界框坐标。

        该属性会计算并返回 [x1, y1, x2, y2] 格式的边界框坐标，并按原始图像尺寸归一化到 [0, 1]。

        返回:
            (paddle.Tensor | np.ndarray): 形状为 (N, 4) 的归一化边界框坐标，每行包含归一化到 [0, 1]
                的 [x1, y1, x2, y2]。

        示例:
            >>> boxes = Boxes(paddle.to_tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xyxyn
            >>> print(normalized)
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, paddle.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self) -> paddle.Tensor | np.ndarray:
        """返回 [x, y, width, height] 格式的归一化边界框。

        该属性会计算并返回 [x_center, y_center, width, height] 格式的归一化边界框坐标，所有值都相对原始图像尺寸。

        返回:
            (paddle.Tensor | np.ndarray): 形状为 (N, 4) 的归一化边界框，每行包含基于原始图像尺寸归一化到
                [0, 1] 的 [x_center, y_center, width, height]。

        示例:
            >>> boxes = Boxes(paddle.to_tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    """分割掩码容器类。

    属性:
        data (paddle.Tensor | np.ndarray): 掩码数据，形状 (N, H, W)。
        orig_shape (tuple[int, int]): 原始图像尺寸 (height, width)。

    属性:
        xy (list[np.ndarray]): 各掩码的多边形轮廓列表（像素坐标）。
        xyn (list[np.ndarray]): 各掩码的多边形轮廓列表（归一化坐标）。
    """

    def __init__(self, masks: "paddle.Tensor | np.ndarray", orig_shape: tuple[int, int]) -> None:
        """初始化 Masks。

        参数:
            masks (paddle.Tensor | np.ndarray): 掩码张量，形状 (N, H, W)。
            orig_shape (tuple[int, int]): 原始图像尺寸 (height, width)。
        """
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self) -> list:
        """归一化多边形轮廓列表。"""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self) -> list:
        """像素坐标多边形轮廓列表。"""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]
