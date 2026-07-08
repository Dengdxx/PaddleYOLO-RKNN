# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief Paddle/PyTorch 兼容性补丁：统一 checkpoint 存取接口。
@details
提供 `checkpoint_save()` 和 `checkpoint_load()` 两个函数，
封装 Paddle 的 `paddle.save` / `paddle.load`，兼容 `.pdparams` 和 `.pt` 格式，
以及部分 PyTorch 原版 API 风格的适配层。
"""

import paddle


"""用于更新/扩展现有 functions 功能的 monkey patches。"""
import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

_imshow = cv2.imshow


def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读取 image file，并支持多语言 filename。

    参数:
        filename (str): 要读取 file 的 path。
        flags (int, optional): 可取 cv2.IMREAD_* 的 flag，控制 image 读取方式。

    返回:
        (np.ndarray | None): 读取到的 image array；读取失败时为 None。

    示例:
        >>> img = imread("path/to/image.jpg")
        >>> img = imread("path/to/image.jpg", cv2.IMREAD_GRAYSCALE)
    """
    file_bytes = np.fromfile(filename, np.uint8)
    if filename.endswith((".tiff", ".tif")):
        success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:
            return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
        return None
    else:
        im = cv2.imdecode(file_bytes, flags)
        if im is None and filename.lower().endswith((".avif", ".heic")):
            im = _imread_pil(filename, flags)
        return im[..., None] if im is not None and im.ndim == 2 else im


_image_open = Image.open
_pil_plugins_registered = False


def image_open(filename, *args, **kwargs):
    """使用 PIL 打开 image，并在首次失败时 lazy 注册 HEIF plugin。

    该 monkey patch 通过 pi-heif（轻量、仅 decode）为 PIL.Image.open 增加 HEIC/HEIF 支持，
    避免在实际需要前引入该 package 带来的约 800ms startup cost。AVIF 由 Pillow 12+ 原生支持，不需要 plugin。

    参数:
        filename (str): image file 的 path。
        *args (Any): 传给 PIL.Image.open 的额外 positional arguments。
        **kwargs (Any): 传给 PIL.Image.open 的额外 keyword arguments。

    返回:
        (PIL.Image.Image): 已打开的 PIL image。
    """
    global _pil_plugins_registered
    if _pil_plugins_registered:
        return _image_open(filename, *args, **kwargs)
    try:
        return _image_open(filename, *args, **kwargs)
    except Exception:
        from ddyolo26.utils.checks import check_requirements

        check_requirements("pi-heif")
        from pi_heif import register_heif_opener

        register_heif_opener()
        _pil_plugins_registered = True
        return _image_open(filename, *args, **kwargs)


Image.open = image_open


def _imread_pil(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """对 OpenCV 不支持的 formats，使用 PIL 作为 fallback 读取 image。

    参数:
        filename (str): 要读取 file 的 path。
        flags (int, optional): OpenCV imread flags（用于决定 grayscale conversion）。

    返回:
        (np.ndarray | None): BGR format 的 image array；读取失败时为 None。
    """
    try:
        with Image.open(filename) as img:
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.asarray(img.convert("L"))
            return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def imwrite(filename: str, img: np.ndarray, params: (list[int] | None) = None) -> bool:
    """写入 image file，并支持多语言 filename。

    参数:
        filename (str): 要写入 file 的 path。
        img (np.ndarray): 要写入的 image。
        params (list[int], optional): image encoding 的额外 parameters。

    返回:
        (bool): file 成功写入时为 True，否则为 False。

    示例:
        >>> import numpy as np
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)  # 创建 black image
        >>> success = imwrite("output.jpg", img)  # 将 image 写入 file
        >>> print(success)
        True
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray) -> None:
    """在指定 window 中显示 image，并支持多语言 window name。

    该函数是 OpenCV imshow 的 wrapper，会在命名 window 中显示 image。它会对多语言 window names 进行适当编码，
    以兼容 OpenCV。

    参数:
        winname (str): 显示 image 的 window name。若该名称的 window 已存在，则 image 会显示在该 window 中。
        mat (np.ndarray): 要显示的 image，应为表示 image 的有效 numpy array。

    示例:
        >>> import numpy as np
        >>> img = np.zeros((300, 300, 3), dtype=np.uint8)  # 创建 black image
        >>> img[:100, :100] = [255, 0, 0]  # 添加 blue square
        >>> imshow("Example Window", img)  # 显示 image
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)


_checkpoint_save_native = paddle.save


def checkpoint_load(*args, **kwargs):
    kwargs.pop("map_location", None)
    kwargs.pop("weights_only", None)
    kwargs.pop("pickle_module", None)
    # paddle.load 需要 str path，而不是 PosixPath。
    if args and isinstance(args[0], Path):
        args = (str(args[0]), *args[1:])
    return paddle.load(*args, **kwargs)


def checkpoint_save(*args, **kwargs):
    """保存 checkpoint objects，并通过 retry mechanism 提高 robustness。

    该函数为 paddle.save 包装 3 次 retries 与 exponential backoff，以应对 device flushing delays
    或 antivirus scanning 可能导致的 save failures。

    参数:
        *args (Any): 传给 paddle.save 的 positional arguments。
        **kwargs (Any): 传给 paddle.save 的 keyword arguments。

    示例:
        >>> checkpoint_save({"epoch": 1}, "model.pdparams")
    """
    for i in range(4):
        try:
            return _checkpoint_save_native(*args, **kwargs)
        except RuntimeError as e:
            if i == 3:
                raise e
            time.sleep(2**i / 2)


@contextmanager
def arange_patch(args):
    """规避 ONNX paddle.arange 与 FP16 不兼容的问题。

    部分 ONNX exporters 会拒绝直接在 arange 中创建 FP16 dtype，因此先使用 default dtype 创建，再在之后 cast。
    """
    if args.dynamic and args.half and args.format == "onnx":
        func = paddle.arange

        def arange(*args, dtype=None, **kwargs):
            """包装 paddle.arange，在创建后 cast dtype，而不是直接传入 dtype。"""
            return func(*args, **kwargs).to(dtype)

        paddle.arange = arange
        yield
        paddle.arange = func
    else:
        yield


@contextmanager
def onnx_export_patch():
    """兼容旧 callers 的 placeholder。"""
    yield


@contextmanager
def override_configs(args, overrides: (dict[str, Any] | None) = None):
    """临时 override args 中 configurations 的 context manager。

    参数:
        args (IterableSimpleNamespace): 原始 configuration arguments。
        overrides (dict[str, Any] | None): 要应用的 overrides dictionary。

    生成:
        (IterableSimpleNamespace): 已应用 overrides 的 configuration arguments。
    """
    if overrides:
        original_args = copy(args)
        for key, value in overrides.items():
            setattr(args, key, value)
        try:
            yield args
        finally:
            args.__dict__.update(original_args.__dict__)
    else:
        yield args
