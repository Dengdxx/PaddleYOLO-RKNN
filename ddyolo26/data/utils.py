# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 数据处理工具函数：标注格式转换、数据集统计、缓存管理。
@details
常用功能：
- `check_det_dataset()`：自动下载/验证 COCO 等公开数据集
- `img2label_paths()`：推断标注文件路径
- `exif_transpose()`：根据 EXIF 信息自动旋转图片方向
- `polygons2masks()`：多边形标注转分割 mask
"""

from __future__ import annotations
import sys


import os
import random
import subprocess
import time
import zipfile
from pathlib import Path
from tarfile import is_tarfile
from typing import Any

import cv2
import numpy as np

from ddyolo26.paddle_utils import *
from PIL import Image, ImageOps

from ddyolo26.nn.autobackend import check_class_names
from ddyolo26.utils import (
    ASSETS_URL,
    DATASETS_DIR,
    LOGGER,
    PROJECT_SITE,
    ROOT,
    SETTINGS_FILE,
    YAML,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
)
from ddyolo26.utils.checks import check_file, check_font, is_ascii
from ddyolo26.utils.downloads import download, safe_download, unzip_file
from ddyolo26.utils.ops import segments2boxes

HELP_URL = f"dataset 格式说明见 {PROJECT_SITE}/tree/main/docs 和 README.md。"
IMG_FORMATS = {
    "avif",
    "bmp",
    "dng",
    "heic",
    "heif",
    "jp2",
    "jpeg",
    "jpeg2000",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
}
VID_FORMATS = {
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ts",
    "wmv",
    "webm",
}
FORMATS_HELP_MSG = f"""支持的格式:
images: {IMG_FORMATS}
videos: {VID_FORMATS}"""


def img2label_paths(img_paths: list[str]) -> list[str]:
    """将 image paths 转换为 label paths：把 'images' 替换为 'labels'，并将扩展名改为 '.txt'。"""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return [(sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt") for x in img_paths]


def check_file_speeds(
    files: list[str],
    threshold_ms: float = 10,
    threshold_mb: float = 50,
    max_files: int = 5,
    prefix: str = "",
):
    """检查 dataset 文件访问速度，并给出性能反馈。

    该函数通过测量 ping（stat call）时间和读取速度来测试 dataset 文件访问速度。它最多从输入列表中
    采样 `max_files` 个文件，并在访问时间超过阈值时发出 warning。

    参数:
        files (list[str]): 需要检查访问速度的文件路径列表。
        threshold_ms (float, optional): ping 时间 warning 阈值，单位毫秒。
        threshold_mb (float, optional): 读取速度 warning 阈值，单位 MB/s。
        max_files (int, optional): 最多检查的文件数量。
        prefix (str, optional): 添加到 log message 前面的前缀字符串。

    示例:
        >>> from pathlib import Path
        >>> image_files = list(Path("dataset/images").glob("*.jpg"))
        >>> check_file_speeds(image_files, threshold_ms=15)
    """
    if not files:
        LOGGER.warning(f"{prefix}Image speed checks: 没有可检查的文件")
        return
    files = random.sample(files, min(max_files, len(files)))
    ping_times = []
    file_sizes = []
    read_speeds = []
    for f in files:
        try:
            start = time.perf_counter()
            file_size = os.stat(f).st_size
            ping_times.append((time.perf_counter() - start) * 1000)
            file_sizes.append(file_size)
            start = time.perf_counter()
            with open(f, "rb") as file_obj:
                _ = file_obj.read()
            read_time = time.perf_counter() - start
            if read_time > 0:
                read_speeds.append(file_size / (1 << 20) / read_time)
        except Exception:
            pass
    if not ping_times:
        LOGGER.warning(f"{prefix}Image speed checks: 文件访问失败")
        return
    avg_ping = np.mean(ping_times)
    std_ping = np.std(ping_times, ddof=1) if len(ping_times) > 1 else 0
    size_msg = f", size: {np.mean(file_sizes) / (1 << 10):.1f} KB"
    ping_msg = f"ping: {avg_ping:.1f}±{std_ping:.1f} ms"
    if read_speeds:
        avg_speed = np.mean(read_speeds)
        std_speed = np.std(read_speeds, ddof=1) if len(read_speeds) > 1 else 0
        speed_msg = f", read: {avg_speed:.1f}±{std_speed:.1f} MB/s"
    else:
        speed_msg = ""
    if avg_ping < threshold_ms or avg_speed < threshold_mb:
        LOGGER.info(f"{prefix}Image 访问速度快 ✅ ({ping_msg}{speed_msg}{size_msg})")
    else:
        LOGGER.warning(
            f"{prefix}检测到 image 访问较慢（{ping_msg}{speed_msg}{size_msg}）。建议使用本地存储替代远程/挂载存储以提升性能。{HELP_URL}"
        )


def get_hash(paths: list[str]) -> str:
    """返回 path 列表（文件或目录）的单个 hash 值。"""
    size = 0
    for p in paths:
        try:
            size += os.stat(p).st_size
        except OSError:
            continue
    h = __import__("hashlib").sha256(str(size).encode())
    h.update("".join(paths).encode())
    return h.hexdigest()


def exif_size(img: Image.Image) -> tuple[int, int]:
    """返回经过 EXIF 修正的 PIL size。"""
    s = img.size
    if img.format == "JPEG":
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)
                if rotation in {6, 8}:
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def verify_image(args: tuple) -> tuple:
    """验证单张 image。"""
    (im_file, cls), prefix = args
    nf, nc, msg = 0, 0, ""
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = shape[1], shape[0]
        assert (shape[0] > 9) & (shape[1] > 9), f"图像尺寸 {shape} 小于 10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"无效 image format {im.format}。{FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: 已修复并保存损坏 JPEG"
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: 忽略损坏 image/label: {e}"
    return (im_file, cls), nf, nc, msg


def verify_image_label(args: tuple) -> list:
    """验证一组 image-label。"""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls = args
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = shape[1], shape[0]
        assert (shape[0] > 9) & (shape[1] > 9), f"图像尺寸 {shape} 小于 10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"无效 image format {im.format}。{FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: 已修复并保存损坏 JPEG"
        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and not keypoint:
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == 5 + nkpt * ndim, f"每个 label 需要 {5 + nkpt * ndim} 列"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"label 需要 5 列，检测到 {lb.shape[1]} 列"
                    points = lb[:, 1:]
                assert points.max() <= 1.01, f"坐标未归一化或越界 {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"存在负数 class label 或 coordinate {lb[lb < -0.01]}"
                max_cls = 0 if single_cls else lb[:, 0].max()
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} 超出 dataset 类别数 {num_cls}。合法 class labels 为 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}{im_file}: 已移除 {nl - len(i)} 个重复 labels"
            else:
                ne = 1
                lb = np.zeros((0, 5 + nkpt * ndim if keypoint else 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5 + nkpt * ndim if keypoint else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: 忽略损坏 image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def visualize_image_annotations(image_path: str, txt_path: str, label_map: dict[int, str]):
    """在 image 上可视化 YOLO annotations（bbox 和 class labels）。

    该函数读取 image 及其 YOLO format annotation 文件，绘制 detected objects 的 bbox，并标注对应
    class name。bbox 颜色根据 class ID 分配，文字颜色会根据背景亮度动态调整以保证可读性。

    参数:
        image_path (str): 待标注 image 文件路径，文件必须可由 PIL 读取。
        txt_path (str): YOLO format annotation 文件路径，每行对应一个 object。
        label_map (dict[int, str]): class ID（整数）到 class label（字符串）的映射。

    示例:
        >>> label_map = {0: "cat", 1: "dog", 2: "bird"}  # 应包含所有已标注 classes
        >>> visualize_image_annotations("path/to/image.jpg", "path/to/annotations.txt", label_map)
    """
    import matplotlib.pyplot as plt

    from ddyolo26.utils.plotting import colors

    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path, encoding="utf-8") as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, int(class_id)))
    _, ax = plt.subplots(1)
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(label, False))
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        ax.text(
            x,
            y - 5,
            label_map[label],
            color="white" if luminance < 0.5 else "black",
            backgroundcolor=color,
        )
    ax.imshow(img)
    plt.show()


def polygon2mask(
    imgsz: tuple[int, int],
    polygons: list[np.ndarray],
    color: int = 1,
    downsample_ratio: int = 1,
) -> np.ndarray:
    """将 polygons 列表转换为指定 image size 的 binary mask。

    参数:
        imgsz (tuple[int, int]): image size，格式为 (height, width)。
        polygons (list[np.ndarray]): polygons 列表。每个 polygon 是长度为 M 的一维坐标数组，
            其中 M % 2 = 0（x, y 交替）。
        color (int, optional): 在 mask 上填充 polygon 使用的颜色值。
        downsample_ratio (int, optional): mask 下采样倍率。

    返回:
        (np.ndarray): 指定 image size 的 binary mask，其中 polygons 已填充。
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio
    return cv2.resize(mask, (nw, nh))


def polygons2masks(
    imgsz: tuple[int, int],
    polygons: list[np.ndarray],
    color: int,
    downsample_ratio: int = 1,
) -> np.ndarray:
    """将 polygons 列表转换为指定 image size 的一组 binary masks。

    参数:
        imgsz (tuple[int, int]): image size，格式为 (height, width)。
        polygons (list[np.ndarray]): polygons 列表。每个 polygon 是可 reshape 为 (-1, 2) 的坐标数组，
            即 (x, y) 点对。
        color (int): 在 masks 上填充 polygons 使用的颜色值。
        downsample_ratio (int, optional): 每个 mask 的下采样倍率。

    返回:
        (np.ndarray): 指定 image size 的一组 binary masks，其中 polygons 已填充。
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(
    imgsz: tuple[int, int], segments: list[np.ndarray], downsample_ratio: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """返回下采样后的 overlap mask 和按面积排序的 indices。"""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for segment in segments:
        mask = polygon2mask(imgsz, [segment.reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def find_dataset_yaml(path: Path) -> Path:
    """查找并返回 Detect、Segment 或 Pose dataset 关联的 YAML 文件。

    该函数会先在给定目录根层级查找 YAML 文件；若未找到，则递归搜索。优先选择 stem 与给定 path 相同的 YAML。

    参数:
        path (Path): 搜索 YAML 文件的目录路径。

    返回:
        (Path): 找到的 YAML 文件路径。
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))
    assert files, f"未在 '{path.resolve()}' 中找到 YAML 文件"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]
    assert len(files) == 1, f"""期望在 '{path.resolve()}' 中找到 1 个 YAML 文件，但找到 {len(files)} 个。
{files}"""
    return files[0]


def check_det_dataset(dataset: str, autodownload: bool = True) -> dict[str, Any]:
    """当本地找不到 dataset 时，下载、验证和/或解压 dataset。

    该函数检查指定 dataset 是否可用；若找不到，可选择下载并解压。随后读取并解析随附 YAML，
    确保满足关键要求，并解析 dataset 相关路径。

    参数:
        dataset (str): dataset 或 dataset 描述文件（例如 YAML 文件）路径。
        autodownload (bool, optional): 找不到 dataset 时是否自动下载。

    返回:
        (dict[str, Any]): 解析后的 dataset 信息和路径。
    """
    file = check_file(dataset)
    extract_dir = ""
    if zipfile.is_zipfile(file) or is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False
    data = YAML.load(file, append_filename=True)
    for k in ("train", "val"):
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(
                    emojis(
                        f"""{dataset} 缺少 '{k}:' key ❌。
所有 data YAML 都必须包含 'train' 和 'val'。"""
                    )
                )
            LOGGER.warning("正在将 data YAML 的 'validation' key 重命名为 'val'，以匹配 YOLO format。")
            data["val"] = data.pop("validation")
    if "names" not in data and "nc" not in data:
        raise SyntaxError(
            emojis(
                f"""{dataset} 缺少 key ❌。
所有 data YAML 都必须包含 'names' 或 'nc'。"""
            )
        )
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' 长度 {len(data['names'])} 必须与 'nc: {data['nc']}' 匹配。"))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])
    data["names"] = check_class_names(data["names"])
    data["channels"] = data.get("channels", 3)
    yaml_dir = Path(data.get("yaml_file", "")).resolve().parent
    path = Path(extract_dir or data.get("path") or yaml_dir)
    if not path.is_absolute():
        yaml_relative_path = (yaml_dir / path).resolve()
        if yaml_relative_path.exists():
            path = yaml_relative_path
        elif not path.exists():
            path = (DATASETS_DIR / path).resolve()
    data["path"] = path
    for k in ("train", "val", "test", "minival"):
        if data.get(k):
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]
        if not all(x.exists() for x in val):
            name = clean_url(dataset)
            LOGGER.info("")
            m = f"Dataset '{name}' images 未找到，缺失路径 '{next(x for x in val if not x.exists())}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"""
注意 dataset download directory 为 '{DATASETS_DIR}'。可在 '{SETTINGS_FILE}' 中更新。"""
                raise FileNotFoundError(m)
            t = time.time()
            r = None
            if s.startswith("http") and s.endswith(".zip"):
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):
                LOGGER.info(f"正在运行 {s} ...")
                subprocess.run(s.split(), check=True)
            else:
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"成功 ✅ {dt}，已保存到 {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f"失败 {dt} ❌"
            LOGGER.info(f"Dataset 下载{s}\n")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")
    return data


def check_cls_dataset(dataset: (str | Path), split: str = "") -> dict[str, Any]:
    """检查分类 dataset，例如 Imagenet。

    该函数接收 `dataset` 名称并尝试获取对应 dataset 信息。若本地找不到 dataset，则尝试从网络下载并保存到本地。

    参数:
        dataset (str | Path): dataset 名称。
        split (str, optional): dataset split，可为 'val'、'test' 或 ''。

    返回:
        (dict[str, Any]): 包含以下 key 的字典：

            - 'train' (Path): dataset training set 目录路径。
            - 'val' (Path): dataset validation set 目录路径。
            - 'test' (Path): dataset test set 目录路径。
            - 'nc' (int): dataset 类别数。
            - 'names' (dict[int, str]): dataset class names 字典。
    """
    if str(dataset).startswith(("http:/", "https:/")):
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True, delete=False)
    elif str(dataset).endswith((".zip", ".tar", ".gz")):
        file = check_file(dataset)
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else DATASETS_DIR / dataset).resolve()
    if not data_dir.is_dir():
        if data_dir.suffix != "":
            raise ValueError(
                f'Classification dataset 必须是目录（data="path/to/dir"），不能是文件（data="{dataset}"），{HELP_URL}'
            )
        LOGGER.info("")
        LOGGER.warning(f"Dataset 未找到，缺失路径 {data_dir}，正在尝试 download...")
        t = time.time()
        if str(dataset) == "imagenet":
            subprocess.run(["bash", str(ROOT / "data/scripts/get_imagenet.sh")], check=True)
        else:
            download(f"{ASSETS_URL}/{dataset}.zip", dir=data_dir.parent)
        LOGGER.info(
            f"""Dataset 下载成功 ✅ ({time.time() - t:.1f}s)，已保存到 {colorstr("bold", data_dir)}
"""
        )
    train_set = data_dir / "train"
    if not train_set.is_dir():
        LOGGER.warning(f"Dataset 'split=train' 未在 {train_set} 找到")
        if image_files := list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png")):
            from ddyolo26.data.split import split_classify_dataset

            LOGGER.info(f"在子目录中找到 {len(image_files)} 张 images，正在尝试 split...")
            data_dir = split_classify_dataset(data_dir, train_ratio=0.8)
            train_set = data_dir / "train"
        else:
            LOGGER.error(f"在 {data_dir} 及其子目录中未找到 images。")
    val_set = (
        data_dir / "val"
        if (data_dir / "val").exists()
        else data_dir / "validation"
        if (data_dir / "validation").exists()
        else data_dir / "valid"
        if (data_dir / "valid").exists()
        else None
    )
    test_set = data_dir / "test" if (data_dir / "test").exists() else None
    if split == "val" and not val_set:
        LOGGER.warning("Dataset 'split=val' 未找到，改用 'split=test'。")
        val_set = test_set
    elif split == "test" and not test_set:
        LOGGER.warning("Dataset 'split=test' 未找到，改用 'split=val'。")
        test_set = val_set
    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])
    names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]
    names = dict(enumerate(sorted(names)))
    for k, v in {"train": train_set, "val": val_set, "test": test_set}.items():
        prefix = f"{colorstr(f'{k}:')} {v}..."
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
            nf = len(files)
            nd = len({file.parent for file in files})
            if nf == 0:
                if k == "train":
                    raise FileNotFoundError(f"{dataset} '{k}:' 未找到 training images")
                else:
                    LOGGER.warning(f"{prefix} 在 {nd} 个 classes 中找到 {nf} 张 images（未找到 images）")
            elif nd != nc:
                LOGGER.error(
                    f"{prefix} 在 {nd} 个 classes 中找到 {nf} 张 images（需要 {nc} 个 classes，而不是 {nd} 个）"
                )
            else:
                LOGGER.info(f"{prefix} 在 {nd} 个 classes 中找到 {nf} 张 images ✅ ")
    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "nc": nc,
        "names": names,
        "channels": 3,
    }


def compress_one_image(f: str, f_new: (str | None) = None, max_dim: int = 1920, quality: int = 50):
    """压缩单个 image 文件，在保持长宽比和质量的同时减小体积，可使用 PIL 或 OpenCV。
    如果输入 image 小于最大尺寸，则不会 resize。

    参数:
        f (str): 输入 image 文件路径。
        f_new (str, optional): 输出 image 文件路径；未指定时覆盖输入文件。
        max_dim (int, optional): 输出 image 最大边（width 或 height）。
        quality (int, optional): image 压缩质量百分比。

    示例:
        >>> from pathlib import Path
        >>> from ddyolo26.data.utils import compress_one_image
        >>> for f in Path("path/to/dataset").rglob("*.jpg"):
        >>>    compress_one_image(f)
    """
    try:
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(f)
        if im.mode in {"RGBA", "LA"}:
            im = im.convert("RGB")
        r = max_dim / max(im.height, im.width)
        if r < 1.0:
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)
    except Exception as e:
        LOGGER.warning(f"Image compression PIL 失败 {f}: {e}")
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)
        if r < 1.0:
            im = cv2.resize(
                im,
                (int(im_width * r), int(im_height * r)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(f_new or f), im)


def load_dataset_cache_file(path: Path) -> dict:
    """从 path 加载 PaddleYOLO-RKNN *.cache 字典。"""
    import gc

    gc.disable()
    cache = np.load(str(path), allow_pickle=True).item()
    gc.enable()
    return cache


def save_dataset_cache_file(prefix: str, path: Path, x: dict, version: str):
    """将 PaddleYOLO-RKNN dataset *.cache 字典 x 保存到 path。"""
    x["version"] = version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()
        with open(str(path), "wb") as file:
            np.save(file, x)
        LOGGER.info(f"{prefix}已创建新 cache: {path}")
    else:
        LOGGER.warning(f"{prefix}Cache directory {path.parent} 不可写，cache 未保存。")
