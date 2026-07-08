# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief YOLO 格式数据集 YOLODataset，以及分割/姿态等子类。
@details
继承 BaseDataset，实现：
- `.txt` 格式标注解析（class x y w h per line）
- Mosaic、CopyPaste 等数据增强集成（通过 Compose/Callback 机制）
- 图像/标注缓存（RAM 或磁盘）
- 自动混合比例（mosaic_ratio、copy_paste_ratio）控制
"""

import paddle

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from PIL import Image

from ddyolo26.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ddyolo26.utils.instance import Instances
from ddyolo26.utils.ops import resample_segments, segments2boxes

from .augment import Compose, Format, LetterBox, RandomLoadText, v8_transforms
from .base import BaseDataset
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """用于加载 YOLO format object detection 和/或 segmentation labels 的 Dataset 类。

    该类支持使用 YOLO format 加载 object detection、segmentation、pose estimation 与 oriented bounding box（OBB）
    任务的数据。

    属性:
        use_segments (bool): 表示是否使用 segmentation masks。
        use_keypoints (bool): 表示是否使用 pose estimation 的 keypoints。
        use_obb (bool): 表示是否使用 oriented bboxes。
        data (dict): dataset configuration dictionary。

    方法:
        cache_labels: cache dataset labels、检查 images 并读取 shapes。
        get_labels: 返回 YOLO training 使用的 label dictionaries 列表。
        build_transforms: 构建并追加 transforms 到列表。
        close_mosaic: 禁用 mosaic、copy_paste、mixup 与 cutmix augmentations 并构建 transformations。
        update_labels_info: 为不同任务更新 label format。
        collate_fn: 将 data samples collate 为 batches。

    示例:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    def __init__(self, *args, data: (dict | None) = None, task: str = "detect", **kwargs):
        """初始化 YOLODataset。

        参数:
            data (dict, optional): dataset configuration dictionary。
            task (str): task type，可为 'detect'、'segment'、'pose' 或 'obb'。
            *args (Any): parent class 的额外 positional arguments。
            **kwargs (Any): parent class 的额外 keyword arguments。
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "不能同时使用 segments 和 keypoints。"
        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """cache dataset labels，检查 images 并读取 shapes。

        参数:
            path (Path): 保存 cache file 的 path。

        返回:
            (dict): 包含 cached labels 与相关信息的 dictionary。
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "data.yaml 中的 'kpt_shape' 缺失或不正确。应为 [keypoints 数量, dims 数量（2 表示 x,y；3 表示 x,y,visible）] 的列表，例如 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for (
                im_file,
                lb,
                shape,
                segments,
                keypoint,
                nm_f,
                nf_f,
                ne_f,
                nc_f,
                msg,
            ) in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],
                            "bboxes": lb[:, 1:],
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}在 {path} 中未找到 labels。{HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        """返回 YOLO training 使用的 label dictionaries 列表。

        该方法从 disk 或 cache 加载 labels，校验其完整性，并为 training 做准备。

        返回:
            (list[dict]): label dictionaries 列表，每项包含一张 image 及其 annotations 信息。
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False
        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))
        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"在 {cache_path} 中未找到有效 images。label format 不正确的 images 会被忽略。{HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box 与 segment 数量应相等，但得到 len(segments) = {len_segments}, len(boxes) = {len_boxes}。将只使用 boxes 并移除所有 segments。为避免该问题，请提供纯 detect 或纯 segment dataset，不要使用 detect-segment mixed dataset。"
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"{cache_path} 中 labels 缺失或为空，training 可能无法正常工作。{HELP_URL}")
        return labels

    def build_transforms(self, hyp: (dict | None) = None) -> Compose:
        """构建并追加 transforms 到列表。

        参数:
            hyp (dict, optional): transforms 使用的 hyperparameters。

        返回:
            (Compose): 组合后的 transforms。
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    def close_mosaic(self, hyp: dict) -> None:
        """通过将概率设为 0.0 禁用 mosaic、copy_paste、mixup 与 cutmix augmentations。

        参数:
            hyp (dict): transforms 使用的 hyperparameters。
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: dict) -> dict:
        """为不同任务更新 label format。

        参数:
            label (dict): 包含 bboxes、segments、keypoints 等的 label dictionary。

        返回:
            (dict): 更新后的 label dictionary，其中包含 instances。

        说明:
            现在 cls 不再与 bboxes 放在一起；classification 与 semantic segmentation 需要独立的 cls label。
            也可以通过在这里添加或删除 dict keys 支持 classification 与 semantic segmentation。
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            max_len = max(len(s) for s in segments)
            segment_resamples = max_len + 1 if segment_resamples < max_len else segment_resamples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """将 data samples collate 为 batches。

        参数:
            batch (list[dict]): 包含 sample data 的 dictionaries 列表。

        返回:
            (dict): collated batch，其中 tensors 已 stack。
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats", "sem_masks"}:
                value = paddle.stack(value, 0)
            elif k == "visuals":
                max_len = max(v.shape[0] for v in value)
                padded = []
                for v in value:
                    if v.shape[0] < max_len:
                        pad_size = [0] * (2 * len(v.shape))
                        pad_size[-1] = max_len - v.shape[0]
                        v = paddle.nn.functional.pad(v, pad_size)
                    padded.append(v)
                value = paddle.stack(padded, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = paddle.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = paddle.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """带 multi-modal 支持、用于加载 YOLO format object detection 和/或 segmentation labels 的 Dataset 类。

    该类扩展 YOLODataset，为 multi-modal model training 添加 text 信息，使 models 可同时处理 image 与 text data。

    方法:
        update_labels_info: 为 multi-modal model training 添加 text 信息。
        build_transforms: 使用 text augmentation 增强 data transformations。

    示例:
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # 应包含 'texts'
    """

    def __init__(self, *args, data: (dict | None) = None, task: str = "detect", **kwargs):
        """初始化 YOLOMultiModalDataset。

        参数:
            data (dict, optional): dataset configuration dictionary。
            task (str): task type，可为 'detect'、'segment'、'pose' 或 'obb'。
            *args (Any): parent class 的额外 positional arguments。
            **kwargs (Any): parent class 的额外 keyword arguments。
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: dict) -> dict:
        """为 multi-modal model training 添加 text 信息。

        参数:
            label (dict): 包含 bboxes、segments、keypoints 等的 label dictionary。

        返回:
            (dict): 更新后的 label dictionary，其中包含 instances 与 texts。
        """
        labels = super().update_labels_info(label)
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp: (dict | None) = None) -> Compose:
        """使用可选 text augmentation 增强 multi-modal training 的 data transformations。

        参数:
            hyp (dict, optional): transforms 使用的 hyperparameters。

        返回:
            (Compose): 组合后的 transforms；适用时包含 text augmentation。
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """返回 dataset 的 category names。

        返回:
            (set[str]): class names 集合。
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}

    @property
    def category_freq(self):
        """返回 dataset 中每个 category 的 frequency。"""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        """基于 frequency threshold 获取 negative text samples。"""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class GroundingDataset(YOLODataset):
    """使用 grounding format JSON file annotations 的 object detection 任务 Dataset 类。

    该 dataset 面向 grounding 任务：annotations 来自 JSON file，而不是标准 YOLO format text files。

    属性:
        json_file (str): 包含 annotations 的 JSON file path。

    方法:
        get_img_files: 返回空列表，因为 image files 在 get_labels 中读取。
        get_labels: 从 JSON file 加载 annotations 并为 training 做准备。
        build_transforms: 为 training 配置 augmentations，并可选加载 text。

    示例:
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # 带 annotations 的有效 images 数量
    """

    def __init__(
        self,
        *args,
        task: str = "detect",
        json_file: str = "",
        max_samples: int = 80,
        **kwargs,
    ):
        """初始化用于 object detection 的 GroundingDataset。

        参数:
            json_file (str): 包含 annotations 的 JSON file path。
            task (str): GroundingDataset 中必须为 'detect' 或 'segment'。
            max_samples (int): text augmentation 加载的最大 samples 数。
            *args (Any): parent class 的额外 positional arguments。
            **kwargs (Any): parent class 的额外 keyword arguments。
        """
        assert task in {
            "detect",
            "segment",
        }, "GroundingDataset 当前仅支持 `detect` 和 `segment` tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> list:
        """image files 会在 `get_labels` 函数中读取，因此这里返回空列表。

        参数:
            img_path (str): 包含 images 的 directory path。

        返回:
            (list): 空列表，因为 image files 在 get_labels 中读取。
        """
        return []

    def verify_labels(self, labels: list[dict[str, Any]]) -> None:
        """验证 dataset 中 instances 数量是否匹配预期数量。

        该方法检查给定 labels 中 bbox instances 总数是否匹配已知 datasets 的预期数量。
        它会基于一组预定义、已知 instance counts 的 datasets 执行 validation。

        参数:
            labels (list[dict[str, Any]]): label dictionaries 列表，每个 dictionary 包含 dataset annotations。
                每个 label dict 必须有 'bboxes' key，其中包含 bbox coordinates 的 numpy array 或 tensor。

        异常:
            AssertionError: 当已识别 dataset 的实际 instance 数量与预期数量不匹配时抛出。

        说明:
            对于未识别 datasets（不在预定义预期数量表中），会记录 warning 并跳过 verification。
        """
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }
        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' 有 {instance_count} 个 instances，预期 {count}。"
                return
        LOGGER.warning(f"未识别 dataset '{self.json_file}'，跳过 instance count verification")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict[str, Any]:
        """从 JSON file 加载 annotations，并为每张 image 过滤、normalize bboxes。

        参数:
            path (Path): 保存 cache file 的 path。

        返回:
            (dict[str, Any]): 包含 cached labels 与相关信息的 dictionary。
        """
        x = {"labels": []}
        LOGGER.info("正在加载 annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"正在读取 annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue
                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue
                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    bboxes.append(box)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)
            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
            lb = np.array(lb, dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        """从 cache 加载 labels，或从 JSON file 生成 labels。

        返回:
            (list[dict]): label dictionaries 列表，每项包含一张 image 及其 annotations 信息。
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.json_file)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, _ = self.cache_labels(cache_path), False
        [cache.pop(k) for k in ("hash", "version")]
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"从 cache file {cache_path} 加载 {self.json_file}")
        return labels

    def build_transforms(self, hyp: (dict | None) = None) -> Compose:
        """为 training 配置 augmentations，并可选加载 text。

        参数:
            hyp (dict, optional): transforms 使用的 hyperparameters。

        返回:
            (Compose): 组合后的 transforms；适用时包含 text augmentation。
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            transform = RandomLoadText(
                max_samples=min(self.max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """返回 dataset 中唯一的 category names。"""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """返回 dataset 中每个 category 的 frequency。"""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: dict, threshold: int = 100) -> list[str]:
        """基于 frequency threshold 获取 negative text samples。"""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(paddle.io.ConcatDataset):
    """由多个 datasets 拼接而成的 Dataset。

    该类用于组合不同的现有 datasets 以进行 YOLO training，并确保它们使用相同 collation function。

    方法:
        collate_fn: static method，使用 YOLODataset 的 collation function 将 data samples collate 为 batches。

    示例:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """将 data samples collate 为 batches。

        参数:
            batch (list[dict]): 包含 sample data 的 dictionaries 列表。

        返回:
            (dict): collated batch，其中 tensors 已 stack。
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: dict) -> None:
        """通过将概率设为 0.0 禁用 mosaic、copy_paste、mixup 与 cutmix augmentations。

        参数:
            hyp (dict): transforms 使用的 hyperparameters。
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)
