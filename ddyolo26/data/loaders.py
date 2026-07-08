# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 推理时数据加载器：图像/视频/流媒体等来源的统一 LoadImages/LoadStreams。
@details
推理管道的第一环，支持：
- 图片文件 / 目录 / glob 匹配
- 视频文件（逐帧读取）
- HTTP/RTSP 流（线程化采帧）
- 截图（桌面捕获）

输出统一接口：`(path, img, im0s, vid_cap, s)` 元组。
"""

from __future__ import annotations
import sys
import paddle


import glob
import math
import os
import time
import urllib
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
import numpy as np

from ddyolo26.paddle_utils import *
from PIL import Image, ImageOps

from ddyolo26.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from ddyolo26.utils import IS_COLAB, IS_KAGGLE, LOGGER, ops
from ddyolo26.utils.checks import check_requirements
from ddyolo26.utils.patches import imread


@dataclass
class SourceTypes:
    """表示 prediction 输入 sources 各种类型的类。

    该类使用 dataclass 定义不同 input source 类型的 boolean flags，这些 sources 可用于 YOLO models prediction。

    属性:
        stream (bool): 表示 input source 是否为 video stream 的 flag。
        screenshot (bool): 表示 input source 是否为 screenshot 的 flag。
        from_img (bool): 表示 input source 是否为 image file 的 flag。
        tensor (bool): 表示 input source 是否为 tensor 的 flag。

    示例:
        >>> source_types = SourceTypes(stream=True, screenshot=False, from_img=False)
        >>> print(source_types.stream)
        True
        >>> print(source_types.from_img)
        False
    """

    stream: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


class LoadStreams:
    """用于多种 video streams 的 stream loader。

    支持 RTSP、RTMP、HTTP 与 TCP streams。该类可同时加载并处理多个 video streams，适合 real-time video
    analysis 任务。

    属性:
        sources (list[str]): video streams 的 source input paths 或 URLs。
        vid_stride (int): video frame-rate stride。
        buffer (bool): 是否 buffer input streams。
        running (bool): 表示 streaming thread 是否正在运行的 flag。
        mode (str): 设为 'stream'，表示 real-time capture。
        imgs (list[list[np.ndarray]]): 每个 stream 的 image frames 列表。
        fps (list[float]): 每个 stream 的 FPS 列表。
        frames (list[int]): 每个 stream 的 total frames 列表。
        threads (list[Thread]): 每个 stream 的 threads 列表。
        shape (list[tuple[int, int, int]]): 每个 stream 的 shapes 列表。
        caps (list[cv2.VideoCapture]): 每个 stream 的 cv2.VideoCapture objects 列表。
        bs (int): processing 使用的 batch size。
        cv2_flag (int): image reading 使用的 OpenCV flag（grayscale 或 color/BGR）。

    方法:
        update: 在 daemon thread 中读取 stream frames。
        close: 关闭 stream loader 并释放 resources。
        __iter__: 返回该类的 iterator object。
        __next__: 返回用于 processing 的 source paths、transformed images 与 original images。
        __len__: 返回 sources object 的长度。

    示例:
        >>> stream_loader = LoadStreams("rtsp://example.com/stream1.mp4")
        >>> for sources, imgs, _ in stream_loader:
        ...     # 处理 images
        ...     pass
        >>> stream_loader.close()

    说明:
        - 该类使用 threading 高效同时加载多个 streams 的 frames。
        - 它会自动处理 YouTube links，并转换为最佳可用 stream URL。
        - 该类实现 buffer system，用于管理 frame storage 与 retrieval。
    """

    def __init__(
        self,
        sources: str = "file.streams",
        vid_stride: int = 1,
        buffer: bool = False,
        channels: int = 3,
    ):
        """初始化支持多种 stream types 的多 video sources stream loader。

        参数:
            sources (str): streams file path 或单个 stream URL。
            vid_stride (int): video frame-rate stride。
            buffer (bool): 是否 buffer input streams。
            channels (int): image channels 数（1 为 grayscale，3 为 color）。
        """
        PaddleFlag.cudnn_benchmark = True
        self.buffer = buffer
        self.running = True
        self.mode = "stream"
        self.vid_stride = vid_stride
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.bs = n
        self.fps = [0] * n
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n
        self.imgs = [[] for _ in range(n)]
        self.shape = [[] for _ in range(n)]
        self.sources = [ops.clean_str(x).replace(os.sep, "_") for x in sources]
        for i, s in enumerate(sources):
            st = f"{i + 1}/{n}: {s}... "
            if urllib.parse.urlparse(s).hostname in {
                "www.youtube.com",
                "youtube.com",
                "youtu.be",
            }:
                s = get_best_youtube_url(s)
            s = int(s) if s.isnumeric() else s
            if s == 0 and (IS_COLAB or IS_KAGGLE):
                raise NotImplementedError(
                    "'source=0' webcam 不支持 Colab 与 Kaggle notebooks。请在 local environment 中尝试运行 'source=0'。"
                )
            self.caps[i] = cv2.VideoCapture(s)
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}打开 {s} 失败")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float("inf")
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30
            success, im = self.caps[i].read()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None] if self.cv2_flag == cv2.IMREAD_GRAYSCALE else im
            if not success or im is None:
                raise ConnectionError(f"{st}从 {s} 读取 images 失败")
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=[i, self.caps[i], s], daemon=True)
            LOGGER.info(f"{st}成功 ✅ ({self.frames[i]} frames，shape {w}x{h}，{self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info("")

    def update(self, i: int, cap: cv2.VideoCapture, stream: str):
        """在 daemon thread 中读取 stream frames，并更新 image buffer。"""
        n, f = 0, self.frames[i]
        while self.running and cap.isOpened() and n < f - 1:
            if len(self.imgs[i]) < 30:
                n += 1
                cap.grab()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    im = (
                        cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None] if self.cv2_flag == cv2.IMREAD_GRAYSCALE else im
                    )
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning("Video stream 无响应，请检查 IP camera connection。")
                        cap.open(stream)
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)

    def close(self):
        """终止 stream loader、停止 threads，并释放 video capture resources。"""
        self.running = False
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)
        for cap in self.caps:
            try:
                cap.release()
            except Exception as e:
                LOGGER.warning(f"无法释放 VideoCapture object: {e}")

    def __iter__(self):
        """返回 iterator object，并 reset frame counter。"""
        self.count = -1
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """返回多个 video streams 的下一批 frames，用于 processing。"""
        self.count += 1
        images = []
        for i, x in enumerate(self.imgs):
            while not x:
                if not self.threads[i].is_alive():
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"正在等待 stream {i}")
            if self.buffer:
                images.append(x.pop(0))
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()
        return self.sources, images, [""] * self.bs

    def __len__(self) -> int:
        """返回 LoadStreams object 中的 video streams 数量。"""
        return self.bs


class LoadScreenshots:
    """用于捕获并处理 screen images 的 PaddleYOLO-RKNN screenshot dataloader。

    该类管理 screenshot images 的加载，以供 YOLO processing 使用。适合配合 `yolo predict source=screen` 使用。

    属性:
        screen (int): 要 capture 的 screen number。
        left (int): screen capture area 的 left coordinate。
        top (int): screen capture area 的 top coordinate。
        width (int): screen capture area 的 width。
        height (int): screen capture area 的 height。
        mode (str): 设为 'stream'，表示 real-time capture。
        frame (int): captured frames 计数器。
        sct (mss.mss): `mss` library 的 screen capture object。
        bs (int): batch size，设为 1。
        fps (int): frames per second，设为 30。
        monitor (dict[str, int]): monitor configuration details。
        cv2_flag (int): image reading 使用的 OpenCV flag（grayscale 或 color/BGR）。

    方法:
        __iter__: 返回 iterator object。
        __next__: capture 下一张 screenshot 并返回。

    示例:
        >>> loader = LoadScreenshots("0 100 100 640 480")  # screen 0, top-left (100,100), 640x480
        >>> for sources, imgs, info in loader:
        ...     print(f"Captured frame: {imgs[0].shape}")
    """

    def __init__(self, source: str, channels: int = 3):
        """使用指定 screen 与 region 参数初始化 screenshot capture。

        参数:
            source (str): screen capture source string，format 为 "screen_num left top width height"。
            channels (int): image channels 数（1 为 grayscale，3 为 color）。
        """
        check_requirements("mss")
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.mode = "stream"
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1
        self.fps = 30
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else monitor["top"] + top
        self.left = monitor["left"] if left is None else monitor["left"] + left
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    def __iter__(self):
        """返回 screenshot capture 的 iterator object。"""
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """使用 mss library capture 下一张 screenshot，并以 numpy array 返回。"""
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)[..., None] if self.cv2_flag == cv2.IMREAD_GRAYSCALE else im0
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "
        self.frame += 1
        return [str(self.screen)], [im0], [s]


class LoadImagesAndVideos:
    """用于加载并处理 YOLO object detection images 与 videos 的类。

    该类管理来自多种 sources 的 image/video data 加载与 pre-processing，包括单个 image files、video files，
    以及 image/video paths 列表。

    属性:
        files (list[str]): image 与 video file paths 列表。
        nf (int): files 总数（images 与 videos）。
        video_flag (list[bool]): 标记 file 是否为 video（True）或 image（False）的 flags。
        mode (str): 当前 mode，'image' 或 'video'。
        vid_stride (int): video frame-rate 的 stride。
        bs (int): batch size。
        cap (cv2.VideoCapture): OpenCV 的 video capture object。
        frame (int): video 的 frame counter。
        frames (int): video 的 total frames 数。
        count (int): iteration counter，在 __iter__() 中初始化为 0。
        ni (int): images 数量。
        cv2_flag (int): image reading 使用的 OpenCV flag（grayscale 或 color/BGR）。

    方法:
        __init__: 初始化 LoadImagesAndVideos object。
        __iter__: 返回 VideoStream 或 ImageFolder 的 iterator object。
        __next__: 返回下一批 images 或 video frames，以及对应 paths 与 metadata。
        _new_video: 为给定 path 创建新的 video capture object。
        __len__: 返回 object 中的 batches 数量。

    示例:
        >>> loader = LoadImagesAndVideos("path/to/data", batch=32, vid_stride=1)
        >>> for paths, imgs, info in loader:
        ...     # 处理 image 或 video frames batch
        ...     pass

    说明:
        - 支持包括 HEIC 在内的多种 image formats。
        - 同时处理 local files 与 directories。
        - 可从包含 image/video paths 的 text file 中读取。
    """

    def __init__(
        self,
        path: (str | Path | list),
        batch: int = 1,
        vid_stride: int = 1,
        channels: int = 3,
    ):
        """初始化支持多种 input formats 的 images/videos dataloader。

        参数:
            path (str | Path | list): images/videos path、directory 或 paths 列表。
            batch (int): processing 使用的 batch size。
            vid_stride (int): video frame-rate stride。
            channels (int): image channels 数（1 为 grayscale，3 为 color）。
        """
        parent = None
        if isinstance(path, str) and Path(path).suffix in {".txt", ".csv"}:
            parent, content = Path(path).parent, Path(path).read_text()
            path = content.splitlines() if Path(path).suffix == ".txt" else content.split(",")
            path = [p.strip() for p in path]
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))
            elif os.path.isfile(a):
                files.append(a)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).abs()))
            else:
                raise FileNotFoundError(f"{p} 不存在")
        images, videos = [], []
        for f in files:
            suffix = f.rpartition(".")[-1].lower()
            if suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv = len(images), len(videos)
        self.files = images + videos
        self.nf = ni + nv
        self.ni = ni
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "video" if ni == 0 else "image"
        self.vid_stride = vid_stride
        self.bs = batch
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        if any(videos):
            self._new_video(videos[0])
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"在 {p} 中未找到 images 或 videos。{FORMATS_HELP_MSG}")

    def __iter__(self):
        """遍历 image/video files，生成 source paths、images 与 metadata。"""
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """返回下一批 images 或 video frames，以及其 paths 与 metadata。"""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:
                if imgs:
                    return paths, imgs, info
                else:
                    raise StopIteration
            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)
                success = False
                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break
                if success:
                    success, im0 = self.cap.retrieve()
                    im0 = (
                        cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)[..., None]
                        if self.cv2_flag == cv2.IMREAD_GRAYSCALE
                        else im0
                    )
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames:
                            self.count += 1
                            self.cap.release()
                else:
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                self.mode = "image"
                im0 = imread(path, flags=self.cv2_flag)
                if im0 is None:
                    LOGGER.warning(f"Image 读取错误 {path}")
                else:
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1
                if self.count >= self.ni:
                    break
        return paths, imgs, info

    def _new_video(self, path: str):
        """为给定 path 创建新的 video capture object，并初始化 video 相关 attributes。"""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"打开 video {path} 失败")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self) -> int:
        """返回 dataset 中的 batches 数量。"""
        return math.ceil(self.nf / self.bs)


class LoadPilAndNumpy:
    """从 PIL 与 Numpy arrays 加载 images，用于 batch processing。

    该类管理 PIL 与 Numpy formats image data 的加载和 pre-processing。它会执行基础 validation 与 format
    conversion，以确保 images 符合 downstream processing 需要的 format。

    属性:
        paths (list[str]): image paths 或 autogenerated filenames 列表。
        im0 (list[np.ndarray]): 以 Numpy arrays 存储的 images 列表。
        mode (str): 正在处理的 data 类型，设为 'image'。
        bs (int): batch size，等于 `im0` 的长度。

    方法:
        _single_check: validate 并 format 单张 image 为 Numpy array。

    示例:
        >>> from PIL import Image
        >>> import numpy as np
        >>> pil_img = Image.new("RGB", (100, 100))
        >>> np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> loader = LoadPilAndNumpy([pil_img, np_img])
        >>> paths, images, _ = next(iter(loader))
        >>> print(f"已加载 {len(images)} images")
        Loaded 2 images
    """

    def __init__(self, im0: (Image.Image | np.ndarray | list), channels: int = 3):
        """初始化 PIL 与 Numpy images loader，并将输入转换为标准 format。

        参数:
            im0 (PIL.Image.Image | np.ndarray | list): PIL 或 numpy format 的单张 image 或 images 列表。
            channels (int): image channels 数（1 为 grayscale，3 为 color）。
        """
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [(getattr(im, "filename", "") or f"image{i}.jpg") for i, im in enumerate(im0)]
        pil_flag = "L" if channels == 1 else "RGB"
        self.im0 = [self._single_check(im, pil_flag) for im in im0]
        self.mode = "image"
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im: (Image.Image | np.ndarray), flag: str = "RGB") -> np.ndarray:
        """validate 并 format image 为 NumPy array。

        说明:
            - PIL inputs 会转为 NumPy；color images 返回 OpenCV-compatible BGR order。
            - NumPy inputs 会原样返回（不执行 channel-order conversion）。
        """
        assert isinstance(im, (Image.Image, np.ndarray)), f"预期 PIL/np.ndarray image type，但得到 {type(im)}"
        if isinstance(im, Image.Image):
            im = np.asarray(im.convert(flag))
            im = im[..., None] if flag == "L" else im[..., ::-1]
            im = np.ascontiguousarray(im)
        elif im.ndim == 2:
            im = im[..., None]
        return im

    def __len__(self) -> int:
        """返回 'im0' attribute 长度，即已加载 images 数量。"""
        return len(self.im0)

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """返回下一批 images、paths 与 metadata，用于 processing。"""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __iter__(self):
        """遍历 PIL/numpy images，生成 paths、raw images 与 processing metadata。"""
        self.count = 0
        return self


class LoadTensor:
    """用于 object detection 任务加载并处理 tensor data 的类。

    该类处理来自 Paddle tensors 的 image data 加载与 pre-processing，为 object detection pipelines 中的后续
    processing 做准备。

    属性:
        im0 (paddle.Tensor): 包含 image(s) 的 input tensor，shape 为 (B, C, H, W)。
        bs (int): batch size，由 `im0` shape 推断。
        mode (str): 当前 processing mode，设为 'image'。
        paths (list[str]): image paths 或 auto-generated filenames 列表。

    方法:
        _single_check: validate 并 format input tensor。

    示例:
        >>> tensor = paddle.rand(1, 3, 640, 640)
        >>> loader = LoadTensor(tensor)
        >>> paths, images, info = next(iter(loader))
        >>> print(f"已处理 {len(images)} images")
    """

    def __init__(self, im0: paddle.Tensor) -> None:
        """初始化用于处理 paddle.Tensor image data 的 LoadTensor object。

        参数:
            im0 (paddle.Tensor): shape 为 (B, C, H, W) 的 input tensor。
        """
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = "image"
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]

    @staticmethod
    def _single_check(im: paddle.Tensor, stride: int = 32) -> paddle.Tensor:
        """validate 并 format 单个 image tensor，确保 shape 与 normalization 正确。"""
        s = f"paddle.Tensor inputs 应为 BCHW，例如 shape(1, 3, 640, 640)，且可被 stride {stride} 整除。Input shape{tuple(im.shape)} 不兼容。"
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im._max() > 1.0 + paddle.finfo(im.dtype).eps:
            LOGGER.warning(
                f"paddle.Tensor inputs 应 normalize 到 0.0-1.0，但 max value 为 {im._max()}。将 input 除以 255。"
            )
            im = im.float() / 255.0
        return im

    def __iter__(self):
        """生成用于遍历 tensor image data 的 iterator object。"""
        self.count = 0
        return self

    def __next__(self) -> tuple[list[str], paddle.Tensor, list[str]]:
        """生成下一批 tensor images 与 metadata，用于 processing。"""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __len__(self) -> int:
        """返回 tensor input 的 batch size。"""
        return self.bs


def autocast_list(source: list[Any]) -> list[Image.Image | np.ndarray]:
    """将 sources 列表转换为 PaddleYOLO-RKNN prediction 使用的 numpy arrays 或 PIL images 列表。"""
    files = []
    for im in source:
        if isinstance(im, (str, Path)):
            files.append(
                ImageOps.exif_transpose(Image.open(urllib.request.urlopen(im) if str(im).startswith("http") else im))
            )
        elif isinstance(im, (Image.Image, np.ndarray)):
            files.append(im)
        else:
            raise TypeError(
                f"""type {type(im).__name__} 不是支持的 PaddleYOLO-RKNN prediction source type。
支持的 source types 请查看 PaddleYOLO-RKNN repository README。"""
            )
    return files


def get_best_youtube_url(url: str, method: str = "pytube") -> str | None:
    """从给定 YouTube video 获取最佳质量 MP4 video stream 的 URL。

    参数:
        url (str): YouTube video 的 URL。
        method (str): 提取 video info 使用的方法。可选 "pytube"、"pafy" 与 "yt-dlp"。

    返回:
        (str | None): 最佳质量 MP4 video stream 的 URL；若未找到合适 stream，则返回 None。

    示例:
        >>> url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        >>> best_url = get_best_youtube_url(url)
        >>> print(best_url)
        https://rr4---sn-q4flrnek.googlevideo.com/videoplayback?expire=...

    说明:
        - 根据所选 method，需要额外 libraries：pytubefix、pafy 或 yt-dlp。
        - 可用时，该函数优先选择至少 1080p resolution 的 streams。
        - 对 "yt-dlp" method，会查找带 video codec、无 audio、且 extension 为 *.mp4 的 formats。
    """
    if method == "pytube":
        check_requirements("pytubefix>=6.5.2")
        from pytubefix import YouTube

        streams = YouTube(url).streams.filter(file_extension="mp4", only_video=True)
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:
                return stream.url
    elif method == "pafy":
        check_requirements(("pafy", "youtube_dl==2020.12.2"))
        import pafy

        return pafy.new(url).getbestvideo(preftype="mp4").url
    elif method == "yt-dlp":
        check_requirements("yt-dlp")
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
        for f in reversed(info_dict.get("formats", [])):
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                return f.get("url")


LOADERS = LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadScreenshots
