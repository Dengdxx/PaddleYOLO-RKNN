# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 通用推理基类 BasePredictor，管理 YOLO26 单张/批量/视频推理流程。
@details
封装了端到端推理管道：
- 图像/视频/流/截图等多来源输入的统一处理
- 预处理（LetterBox 填充、归一化）
- 模型前向推理（支持 end2end 无 NMS 模式）
- 后处理（坐标反变换、结果封装为 Results 对象）
- 流式输出与结果保存

YOLO26 end2end 模式下，one-to-one head 直接输出 300 个检测框，
无需 NMS 后处理，显著降低推理延迟。
"""

from __future__ import annotations

import paddle

"""
对图像、视频、目录、通配符、摄像头、流媒体等来源执行推理。

用法 - 输入来源:
    $ yolo mode=predict model=weights/yolov8/yolov8n.pdparams source=0           # 摄像头
                                                img.jpg                         # 图像
                                                vid.mp4                         # 视频
                                                screen                          # 截屏
                                                path/                           # 目录
                                                list.txt                        # 图像列表
                                                list.streams                    # 流列表
                                                'path/*.jpg'                    # 通配符
                                                'rtsp://example.com/media.mp4'  # RTSP、RTMP、HTTP、TCP 流

用法 - 模型格式:
    $ yolo mode=predict model=weights/yolov8/yolov8n.pdparams  # PaddlePaddle
                              yolov8n.onnx                    # ONNX Runtime 或 OpenCV DNN（dnn=True）
                              yolov8n.rknn                    # Rockchip RKNN
"""

import platform
import re
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ddyolo26.cfg import get_cfg, get_save_dir
from ddyolo26.data import load_inference_source
from ddyolo26.data.augment import LetterBox
from ddyolo26.nn.autobackend import AutoBackend
from ddyolo26.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ddyolo26.utils.checks import check_imgsz, check_imshow
from ddyolo26.utils.files import increment_path
from ddyolo26.utils.runtime import attempt_compile, select_device, smart_inference_mode

STREAM_WARNING = """
未传入 `stream=True` 时，推理结果会持续累积在内存中；对于大规模输入源或长时间运行的视频/流媒体任务，
这可能导致内存不足。使用方式请参考 PaddleYOLO-RKNN 仓库 README。

示例:
    results = model(source=..., stream=True)  # Results 对象生成器
    for r in results:
        boxes = r.boxes  # bbox 输出对应的 Boxes 对象
        masks = r.masks  # 分割掩码输出对应的 Masks 对象
"""


class BasePredictor:
    """用于创建推理器的基类。

    该类提供推理功能的基础能力，负责不同输入源下的模型初始化、推理和结果处理。

    属性:
        args (SimpleNamespace): 推理器配置。
        save_dir (Path): 结果保存目录。
        done_warmup (bool): 推理器是否已完成预热。
        model (paddle.nn.Layer): 用于推理的模型。
        data (str): 数据配置。
        device (str): 推理设备。
        dataset (Dataset): 推理使用的数据集。
        vid_writer (dict[Path, cv2.VideoWriter]): 保存视频输出用的 {save_path: video_writer} 字典。
        plotted_img (np.ndarray): 最近一次绘制后的图像。
        source_type (SimpleNamespace): 输入源类型。
        seen (int): 已处理图像数量。
        windows (list[str]): 可视化窗口名称列表。
        batch (tuple): 当前批次数据。
        results (list[Any]): 当前批次结果。
        transforms (callable): 图像变换函数。
        callbacks (dict[str, list[callable]]): 各事件对应的回调函数。
        txt_path (Path): 文本结果保存路径。
        _lock (threading.Lock): 线程安全推理锁。

    方法:
        preprocess: 推理前准备输入图像。
        inference: 对指定图像执行推理。
        postprocess: 将原始预测处理为结构化结果。
        predict_cli: 执行命令行推理。
        setup_source: 设置输入源和推理模式。
        stream_inference: 对输入源执行流式推理。
        setup_model: 初始化并配置模型。
        write_results: 将推理结果写入文件。
        save_predicted_images: 保存预测可视化图像。
        show: 在窗口中显示结果。
        run_callbacks: 执行指定事件的已注册回调。
        add_callback: 注册新的回调函数。
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: (dict[str, Any] | None) = None,
        _callbacks: (dict[str, list[callable]] | None) = None,
    ):
        """初始化 BasePredictor。

        参数:
            cfg (str | Path | dict | SimpleNamespace): 配置文件路径或配置字典。
            overrides (dict, optional): 配置覆盖项。
            _callbacks (dict, optional): 回调函数字典。
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)
        self.model = None
        self.data = self.args.data
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im: (paddle.Tensor | list[np.ndarray])) -> paddle.Tensor:
        """推理前准备输入图像。

        参数:
            im (paddle.Tensor | list[np.ndarray]): 张量输入形状为 (N, 3, H, W)，列表输入为 [(H, W, 3) x N]。

        返回:
            (paddle.Tensor): 形状为 (N, 3, H, W) 的预处理图像张量。
        """
        not_tensor = not isinstance(im, paddle.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]
            im = im.transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = paddle.from_numpy(im)
        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        if not_tensor:
            im /= 255
        return im

    def inference(self, im: paddle.Tensor, *args, **kwargs):
        """使用指定模型和参数对图像执行推理。"""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and not self.source_type.tensor
            else False
        )
        return self.model(
            im,
            *args,
            augment=self.args.augment,
            visualize=visualize,
            embed=self.args.embed,
            **kwargs,
        )

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """推理前对输入图像执行预变换。

        参数:
            im (list[np.ndarray]): 形状为 [(H, W, 3) x N] 的图像列表。

        返回:
            (list[np.ndarray]): 变换后的图像列表。
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (self.model.pt or getattr(self.model, "dynamic", False) and not self.model.imx),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """后处理图像预测结果并返回。"""
        return preds

    def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
        """对图像或流执行推理。

        参数:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | paddle.Tensor, optional):
                推理输入源。
            model (str | Path | paddle.nn.Layer, optional): 推理模型。
            stream (bool): 是否流式返回推理结果；为 True 时返回生成器。
            *args (Any): 传给推理方法的额外位置参数。
            **kwargs (Any): 传给推理方法的额外关键字参数。

        返回:
            (list[ddyolo26.engine.results.Results] | generator): Results 对象列表或 Results 对象生成器。
        """
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))

    def predict_cli(self, source=None, model=None):
        """命令行（CLI）推理入口。

        该函数用于命令行推理：先设置输入源和模型，再以流式方式处理输入。它会消费生成器但不保存结果列表，
        从而避免长时间推理时输出在内存中累积。

        参数:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | paddle.Tensor, optional):
                推理输入源。
            model (str | Path | paddle.nn.Layer, optional): 推理模型。

        注意:
            不要改成收集生成器输出的形式。生成器能避免输出持续累积在内存中，这对长时间推理非常重要。
        """
        gen = self.stream_inference(source, model)
        for _ in gen:
            pass

    def setup_source(self, source, stride: (int | None) = None):
        """设置输入源和推理模式。

        参数:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | paddle.Tensor): Source for
                推理输入源。
            stride (int, optional): 用于图像尺寸检查的模型步长。
        """
        self.imgsz = check_imgsz(self.args.imgsz, stride=stride or self.model.stride, min_dim=2)
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
            channels=getattr(self.model, "ch", 3),
        )
        self.source_type = self.dataset.source_type
        if (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000
            or any(getattr(self.dataset, "video_flag", [False]))
        ):
            pass
            if not getattr(self, "stream", True):
                LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """对输入源执行流式推理并保存结果。

        参数:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | paddle.Tensor, optional):
                推理输入源。
            model (str | Path | paddle.nn.Layer, optional): 推理模型。
            *args (Any): 传给推理方法的额外位置参数。
            **kwargs (Any): 传给推理方法的额外关键字参数。

        生成:
            (ddyolo26.engine.results.Results): Results 对象。
        """
        if self.args.verbose:
            LOGGER.info("")
        if not self.model:
            self.setup_model(model)
        with self._lock:
            self.setup_source(source if source is not None else self.args.source)
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(
                        1 if self.model.pt or self.model.triton else self.dataset.bs,
                        self.model.ch,
                        *self.imgsz,
                    )
                )
                self.done_warmup = True
            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for batch in self.dataset:
                self.batch = batch
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                with profilers[0]:
                    im = self.preprocess(im0s)
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from ([preds] if isinstance(preds, paddle.Tensor) else preds)
                        continue
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1000.0 / n,
                            "inference": profilers[1].dt * 1000.0 / n,
                            "postprocess": profilers[2].dt * 1000.0 / n,
                        }
                        if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break
                if self.args.verbose:
                    LOGGER.info("\n".join(s))
                self.run_callbacks("on_predict_batch_end")
                yield from self.results
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()
        if self.args.show:
            cv2.destroyAllWindows()
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1000.0 for x in profilers)
            LOGGER.info(
                f"速度: 每张图 %.1fms preprocess, %.1fms inference, %.1fms postprocess，shape={min(self.args.batch, self.seen), getattr(self.model, 'ch', 3), *im.shape[2:]}"
                % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))
            s = f"\n{nl} 个 label 已保存到 {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"结果已保存到 {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose: bool = True):
        """按给定参数初始化 YOLO 模型，并切换到评估模式。

        参数:
            model (str | Path | paddle.nn.Layer): 要加载或使用的模型。
            verbose (bool): 是否打印详细输出。
        """
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if model.end2end:
                model.set_head_attr(max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms)
        self.model = AutoBackend(
            model=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=verbose,
        )
        self.device = self.model.device
        self.args.half = self.model.fp16
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.args.imgsz = self.model.imgsz
        self.model.eval()
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

    def write_results(self, i: int, p: Path, im: paddle.Tensor, s: list[str]) -> str:
        """将推理结果写入文件或目录。

        参数:
            i (int): 当前图像在 batch 中的索引。
            p (Path): 当前图像路径。
            im (paddle.Tensor): 预处理后的图像张量。
            s (list[str]): 结果字符串列表。

        返回:
            (str): 包含结果信息的字符串。
        """
        string = ""
        if len(im.shape) == 3:
            im = im[None]
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search("frame (\\d+)/", s[i])
            frame = int(match[1]) if match else None
        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)
        return string

    def save_predicted_images(self, save_path: Path, frame: int = 0):
        """将视频预测保存为 mp4/avi，或将图像预测保存为 jpg。

        参数:
            save_path (Path): 结果保存路径。
            frame (int): 视频模式下的帧号。
        """
        im = self.plotted_img
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = self.save_dir / f"{save_path.stem}_frames"
            if save_path not in self.vid_writer:
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,
                    frameSize=(im.shape[1], im.shape[0]),
                )
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)
        else:
            cv2.imwrite(str(save_path.with_suffix(".jpg")), im)

    def show(self, p: str = ""):
        """在窗口中显示图像。"""
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])
        cv2.imshow(p, im)
        if cv2.waitKey(300 if self.dataset.mode == "image" else 1) & 255 == ord("q"):
            raise StopIteration

    def run_callbacks(self, event: str):
        """运行指定事件的所有已注册回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func: callable):
        """为指定事件添加回调函数。"""
        self.callbacks[event].append(func)
