# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 数据集划分工具：将单一数据集按比例切分为 train/val/test 子集。
@details
提供 `autosplit()` 函数，将图像列表按指定比例（如 0.9/0.1/0）
随机划分并写入对应的 txt 文件列表。
"""

import paddle

import random
import shutil
from pathlib import Path

from ddyolo26.data.utils import IMG_FORMATS, img2label_paths
from ddyolo26.utils import DATASETS_DIR, LOGGER, TQDM


def split_classify_dataset(source_dir: (str | Path), train_ratio: float = 0.8) -> Path:
    """将分类数据集划分到新目录中的 train 和 val 子目录。

    创建新目录 '{source_dir}_split'，其中包含 train/val 子目录，并保留原始类别目录结构。
    默认按 80/20 比例划分。

    目录结构:
        划分前:
            caltech/
            ├── class1/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            ├── class2/
            │   ├── img1.jpg
            │   └── ...
            └── ...

        划分后:
            caltech_split/
            ├── train/
            │   ├── class1/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   ├── class2/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   └── ...
            └── val/
                ├── class1/
                │   ├── img2.jpg
                │   └── ...
                ├── class2/
                │   └── ...
                └── ...

    参数:
        source_dir (str | Path): 分类数据集根目录路径。
        train_ratio (float): 训练集比例，范围 0 到 1。

    返回:
        (Path): 创建出的划分目录路径。

    示例:
        使用默认 80/20 比例划分数据集
        >>> split_classify_dataset("path/to/caltech")

        使用自定义比例划分
        >>> split_classify_dataset("path/to/caltech", 0.75)
    """
    source_path = Path(source_dir)
    split_path = Path(f"{source_path}_split")
    train_path, val_path = split_path / "train", split_path / "val"
    split_path.mkdir(exist_ok=True)
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob("*.*"))) for d in class_dirs)
    stats = f"{len(class_dirs)} 个类别，{total_images} 张图片"
    LOGGER.info(f"正在将 {source_path}（{stats}）划分为 {train_ratio:.0%} train、{1 - train_ratio:.0%} val...")
    for class_dir in class_dirs:
        (train_path / class_dir.name).mkdir(exist_ok=True)
        (val_path / class_dir.name).mkdir(exist_ok=True)
        image_files = list(class_dir.glob("*.*"))
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)
        for img in image_files[:split_idx]:
            shutil.copy2(img, train_path / class_dir.name / img.name)
        for img in image_files[split_idx:]:
            shutil.copy2(img, val_path / class_dir.name / img.name)
    LOGGER.info(f"划分完成，输出目录: {split_path} ✅")
    return split_path


def autosplit(
    path: Path = DATASETS_DIR / "coco8/images",
    weights: tuple[float, float, float] = (0.9, 0.1, 0.0),
    annotated_only: bool = False,
) -> None:
    """自动将数据集划分为 train/val/test，并将结果保存为 autosplit_*.txt 文件。

    参数:
        path (Path): 图片目录路径。
        weights (tuple[float, float, float]): train、val 和 test 划分比例。
        annotated_only (bool): 若为 True，则只使用存在对应 txt 标注文件的图片。

    示例:
        使用默认比例划分图片
        >>> from ddyolo26.data.split import autosplit
        >>> autosplit()

        使用自定义比例，并且只划分带标注的图片
        >>> autosplit(path="path/to/images", weights=(0.8, 0.15, 0.05), annotated_only=True)
    """
    path = Path(path)
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)
    n = len(files)
    random.seed(0)
    indices = random.choices([0, 1, 2], weights=weights, k=n)
    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()
    LOGGER.info(f"正在自动划分图片目录 {path}" + "，仅使用带 *.txt 标注的图片" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            with open(path.parent / txt[i], "a", encoding="utf-8") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")


if __name__ == "__main__":
    split_classify_dataset("caltech101")
