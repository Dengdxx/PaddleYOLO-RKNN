# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 可视化工具：标注绘制、训练曲线、混淆矩阵等图表。
@details
提供训练过程中的可视化支持：
- `Annotator`：在图像上绘制检测框+标签+置信度
- `plot_results()`：绘制训练指标折线图（loss/mAP/lr）
- `plot_labels()`：分析并可视化数据集中类别分布和框尺寸分布
- `output_to_target()`：将模型输出转换为可绘制格式
"""

from __future__ import annotations
import sys
import paddle


import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ddyolo26.paddle_utils import *
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from ddyolo26.utils import IS_COLAB, IS_KAGGLE, LOGGER, TryExcept, ops, plt_settings, threaded
from ddyolo26.utils.checks import check_font, check_version, is_ascii
from ddyolo26.utils.files import increment_path


class Colors:
    """用于 visualization 与 plotting 的 PaddleYOLO-RKNN color palette。

    该类提供 PaddleYOLO-RKNN color palette 相关方法，包括将 hex color codes 转为 RGB values，
    以及访问 object detection 和 pose estimation 的 predefined color schemes。

    ## PaddleYOLO-RKNN 调色板

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #042aff;"></i> | `#042aff` | (4, 42, 255)      |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #0bdbeb;"></i> | `#0bdbeb` | (11, 219, 235)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #f3f3f3;"></i> | `#f3f3f3` | (243, 243, 243)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #00dfb7;"></i> | `#00dfb7` | (0, 223, 183)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #111f68;"></i> | `#111f68` | (17, 31, 104)     |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #ff6fdd;"></i> | `#ff6fdd` | (255, 111, 221)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff444f;"></i> | `#ff444f` | (255, 68, 79)     |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #cced00;"></i> | `#cced00` | (204, 237, 0)     |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #00f344;"></i> | `#00f344` | (0, 243, 68)      |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #bd00ff;"></i> | `#bd00ff` | (189, 0, 255)     |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #00b4ff;"></i> | `#00b4ff` | (0, 180, 255)     |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #dd00ba;"></i> | `#dd00ba` | (221, 0, 186)     |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #00ffff;"></i> | `#00ffff` | (0, 255, 255)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #26c000;"></i> | `#26c000` | (38, 192, 0)      |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #01ffb3;"></i> | `#01ffb3` | (1, 255, 179)     |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #7d24ff;"></i> | `#7d24ff` | (125, 36, 255)    |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #7b0068;"></i> | `#7b0068` | (123, 0, 104)     |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #ff1b6c;"></i> | `#ff1b6c` | (255, 27, 108)    |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #fc6d2f;"></i> | `#fc6d2f` | (252, 109, 47)    |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #a2ff0b;"></i> | `#a2ff0b` | (162, 255, 11)    |

    ## Pose 调色板

    | Index | Color                                                             | HEX       | RGB               |
    |-------|-------------------------------------------------------------------|-----------|-------------------|
    | 0     | <i class="fa-solid fa-square fa-2xl" style="color: #ff8000;"></i> | `#ff8000` | (255, 128, 0)     |
    | 1     | <i class="fa-solid fa-square fa-2xl" style="color: #ff9933;"></i> | `#ff9933` | (255, 153, 51)    |
    | 2     | <i class="fa-solid fa-square fa-2xl" style="color: #ffb266;"></i> | `#ffb266` | (255, 178, 102)   |
    | 3     | <i class="fa-solid fa-square fa-2xl" style="color: #e6e600;"></i> | `#e6e600` | (230, 230, 0)     |
    | 4     | <i class="fa-solid fa-square fa-2xl" style="color: #ff99ff;"></i> | `#ff99ff` | (255, 153, 255)   |
    | 5     | <i class="fa-solid fa-square fa-2xl" style="color: #99ccff;"></i> | `#99ccff` | (153, 204, 255)   |
    | 6     | <i class="fa-solid fa-square fa-2xl" style="color: #ff66ff;"></i> | `#ff66ff` | (255, 102, 255)   |
    | 7     | <i class="fa-solid fa-square fa-2xl" style="color: #ff33ff;"></i> | `#ff33ff` | (255, 51, 255)    |
    | 8     | <i class="fa-solid fa-square fa-2xl" style="color: #66b2ff;"></i> | `#66b2ff` | (102, 178, 255)   |
    | 9     | <i class="fa-solid fa-square fa-2xl" style="color: #3399ff;"></i> | `#3399ff` | (51, 153, 255)    |
    | 10    | <i class="fa-solid fa-square fa-2xl" style="color: #ff9999;"></i> | `#ff9999` | (255, 153, 153)   |
    | 11    | <i class="fa-solid fa-square fa-2xl" style="color: #ff6666;"></i> | `#ff6666` | (255, 102, 102)   |
    | 12    | <i class="fa-solid fa-square fa-2xl" style="color: #ff3333;"></i> | `#ff3333` | (255, 51, 51)     |
    | 13    | <i class="fa-solid fa-square fa-2xl" style="color: #99ff99;"></i> | `#99ff99` | (153, 255, 153)   |
    | 14    | <i class="fa-solid fa-square fa-2xl" style="color: #66ff66;"></i> | `#66ff66` | (102, 255, 102)   |
    | 15    | <i class="fa-solid fa-square fa-2xl" style="color: #33ff33;"></i> | `#33ff33` | (51, 255, 51)     |
    | 16    | <i class="fa-solid fa-square fa-2xl" style="color: #00ff00;"></i> | `#00ff00` | (0, 255, 0)       |
    | 17    | <i class="fa-solid fa-square fa-2xl" style="color: #0000ff;"></i> | `#0000ff` | (0, 0, 255)       |
    | 18    | <i class="fa-solid fa-square fa-2xl" style="color: #ff0000;"></i> | `#ff0000` | (255, 0, 0)       |
    | 19    | <i class="fa-solid fa-square fa-2xl" style="color: #ffffff;"></i> | `#ffffff` | (255, 255, 255)   |

    !!! note "PaddleYOLO-RKNN Brand Colors"

        下方 palette 用于保持本仓库内部 visualization 一致性。
        PaddleYOLO-RKNN 暂无单独的公开 brand guideline 站点。

    属性:
        palette (list[tuple]): 通用 RGB color tuples 列表。
        n (int): palette 中的 colors 数量。
        pose_palette (np.ndarray): pose estimation 专用 color palette array，dtype 为 np.uint8。

    示例:
        >>> from ddyolo26.utils.plotting import Colors
        >>> colors = Colors()
        >>> colors(5, True)  # 返回 BGR format: (221, 111, 255)
        >>> colors(5, False)  # 返回 RGB format: (255, 111, 221)
    """

    def __init__(self):
        """用 hex colors 初始化 palette。"""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i: (int | paddle.Tensor), bgr: bool = False) -> tuple:
        """按 index 从 palette 返回一个 color。

        参数:
            i (int | paddle.Tensor): color index。
            bgr (bool, optional): 是否返回 BGR format 而不是 RGB。

        返回:
            (tuple): RGB 或 BGR color tuple。
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple:
        """将 hex color codes 转为 RGB values（即默认 PIL order）。"""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


class Annotator:
    """用于 train/val mosaics、JPGs 与 prediction annotations 的 PaddleYOLO-RKNN annotator。

    属性:
        im (Image.Image | np.ndarray): 需要 annotate 的 image。
        pil (bool): 是否使用 PIL 而不是 cv2 绘制 annotations。
        font (ImageFont.truetype | ImageFont.load_default): text annotations 使用的 font。
        lw (int): 绘制用 line width。
        skeleton (list[list[int]]): keypoints 的 skeleton structure。
        limb_color (np.ndarray): limbs 的 color palette。
        kpt_color (np.ndarray): keypoints 的 color palette。
        dark_colors (set): 为 text contrast 视为 dark 的 colors 集合。
        light_colors (set): 为 text contrast 视为 light 的 colors 集合。

    示例:
        >>> from ddyolo26.utils.plotting import Annotator
        >>> im0 = cv2.imread("test.png")
        >>> annotator = Annotator(im0, line_width=10)
        >>> annotator.box_label([10, 10, 100, 100], "person", (255, 0, 0))
    """

    def __init__(
        self,
        im,
        line_width: (int | None) = None,
        font_size: (int | None) = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """使用 image、line width 以及 keypoints/limbs color palette 初始化 Annotator。"""
        non_ascii = not is_ascii(example)
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if not input_is_pil:
            if im.shape[2] == 1:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 2:
                im = np.ascontiguousarray(np.dstack((im, np.zeros_like(im[..., :1]))))
            elif im.shape[2] > 3:
                im = np.ascontiguousarray(im[..., :3])
        if self.pil:
            self.im = im if input_is_pil else Image.fromarray(im)
            if self.im.mode not in {"RGB", "RGBA"}:
                self.im = self.im.convert("RGB")
            self.draw = ImageDraw.Draw(self.im, "RGBA")
            try:
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]
        else:
            assert im.data.contiguous, "Image 不连续。请对 Annotator input images 应用 np.ascontiguousarray(im)。"
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)
            self.sf = self.lw / 3
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]
        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(self, color: tuple = (128, 128, 128), txt_color: tuple = (255, 255, 255)) -> tuple:
        """根据 background color 分配 text color。

        参数:
            color (tuple, optional): text rectangle 的 background color。
            txt_color (tuple, optional): text 的 fallback color。

        返回:
            (tuple): label 使用的 text color。

        示例:
            >>> from ddyolo26.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_txt_color(color=(104, 31, 17))  # 返回 (255, 255, 255)
        """
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color

    def box_label(
        self,
        box,
        label: str = "",
        color: tuple = (128, 128, 128),
        txt_color: tuple = (255, 255, 255),
    ):
        """在 image 上绘制带给定 label 的 bounding box。

        参数:
            box (tuple): bounding box coordinates (x1, y1, x2, y2)。
            label (str, optional): 要显示的 text label。
            color (tuple, optional): rectangle 的 background color。
            txt_color (tuple, optional): text 的 color。

        示例:
            >>> from ddyolo26.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.box_label(box=[10, 20, 30, 40], label="person")
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, paddle.Tensor):
            box = box.tolist()
        multi_points = isinstance(box[0], list)
        p1 = [int(b) for b in box[0]] if multi_points else (int(box[0]), int(box[1]))
        if self.pil:
            self.draw.polygon(
                [tuple(b) for b in box], width=self.lw, outline=color
            ) if multi_points else self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = p1[1] >= h
                if p1[0] > self.im.size[0] - w:
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (
                        p1[0],
                        p1[1] - h if outside else p1[1],
                        p1[0] + w + 1,
                        p1[1] + 1 if outside else p1[1] + h + 1,
                    ),
                    fill=color,
                )
                self.draw.text(
                    (p1[0], p1[1] - h if outside else p1[1]),
                    label,
                    fill=txt_color,
                    font=self.font,
                )
        else:
            cv2.polylines(
                self.im, [np.asarray(box, dtype=int)], True, color, self.lw
            ) if multi_points else cv2.rectangle(
                self.im,
                p1,
                (int(box[2]), int(box[3])),
                color,
                thickness=self.lw,
                lineType=cv2.LINE_AA,
            )
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3
                outside = p1[1] >= h
                if p1[0] > self.im.shape[1] - w:
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )

    def masks(self, masks, colors_list, im_gpu=None, alpha=0.5):
        """在图像上绘制半透明分割掩码。

        参数:
            masks (np.ndarray): 二值掩码数组，形状 (N, H, W)，uint8。
            colors_list (list[tuple]): 每个掩码对应的 RGB 颜色列表。
            im_gpu: 未使用（兼容接口）。
            alpha (float): 掩码透明度，0~1。
        """
        if len(masks) == 0:
            return
        im = np.asarray(self.im).copy()
        for mask, color in zip(masks, colors_list):
            if mask.shape[:2] != im.shape[:2]:
                mask = cv2.resize(mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            colored = np.zeros_like(im)
            colored[:] = color
            region = mask.astype(bool)
            im[region] = cv2.addWeighted(im, 1 - alpha, colored, alpha, 0)[region]
        self.fromarray(im)

    def kpts(self, *args, **kwargs):
        pass

    def rectangle(self, xy, fill=None, outline=None, width: int = 1):
        """向 image 添加 rectangle（仅 PIL）。"""
        self.draw.rectangle(xy, fill, outline, width)

    def text(
        self,
        xy,
        text: str,
        txt_color: tuple = (255, 255, 255),
        anchor: str = "top",
        box_color: tuple = (),
    ):
        """使用 PIL 或 cv2 向 image 添加 text。

        参数:
            xy (list[int]): text 放置位置的 top-left coordinates。
            text (str): 要绘制的 text。
            txt_color (tuple, optional): text color。
            anchor (str, optional): text anchor position（'top' 或 'bottom'）。
            box_color (tuple, optional): 可带 alpha 的 box background color。
        """
        if self.pil:
            w, h = self.font.getsize(text)
            if anchor == "bottom":
                xy[1] += 1 - h
            for line in text.split("\n"):
                if box_color:
                    w, h = self.font.getsize(line)
                    self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=box_color)
                self.draw.text(xy, line, fill=txt_color, font=self.font)
                xy[1] += h
        else:
            if box_color:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3
                outside = xy[1] >= h
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, box_color, -1, cv2.LINE_AA)
            cv2.putText(
                self.im,
                text,
                xy,
                0,
                self.sf,
                txt_color,
                thickness=self.tf,
                lineType=cv2.LINE_AA,
            )

    def fromarray(self, im):
        """从 NumPy array 或 PIL image 更新当前绘图画布.

        @param im 新的 NumPy 或 PIL 图像。
        @return 无返回值；保持初始化时选择的 PIL/OpenCV 绘图后端。
        """
        if self.pil:
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
        else:
            self.im = np.ascontiguousarray(np.asarray(im)).copy()

    def result(self, pil=False):
        """以 array 或 PIL image 返回 annotated image。"""
        im = np.asarray(self.im)
        return Image.fromarray(im[..., ::-1]) if pil else im

    def show(self, title: (str | None) = None):
        """显示 annotated image。"""
        im = Image.fromarray(np.asarray(self.im)[..., ::-1])
        if IS_COLAB or IS_KAGGLE:
            try:
                display(im)
            except ImportError as e:
                LOGGER.warning(f"无法在 Jupyter notebooks 中显示 image: {e}")
        else:
            im.show(title=title)

    def save(self, filename: str = "image.jpg"):
        """将 annotated image 保存到 'filename'。"""
        cv2.imwrite(filename, np.asarray(self.im))

    @staticmethod
    def get_bbox_dimension(bbox: (tuple | list)):
        """计算 bounding box 的 dimensions 与 area。

        参数:
            bbox (tuple | list): format 为 (x_min, y_min, x_max, y_max) 的 bounding box coordinates。

        返回:
            width (float): bounding box 的 width。
            height (float): bounding box 的 height。
            area (float): bounding box 围成的 area。

        示例:
            >>> from ddyolo26.utils.plotting import Annotator
            >>> im0 = cv2.imread("test.png")
            >>> annotator = Annotator(im0, line_width=10)
            >>> annotator.get_bbox_dimension(bbox=[10, 20, 30, 40])
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height


@TryExcept()
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """绘制 training labels，包括 class histograms 与 box statistics。

    参数:
        boxes (np.ndarray): format 为 [x, y, width, height] 的 bounding box coordinates。
        cls (np.ndarray): class indices。
        names (dict, optional): class indices 到 class names 的 mapping 字典。
        save_dir (Path, optional): 保存 plot 的 directory。
        on_plot (Callable, optional): plot 保存后调用的 function。
    """
    import matplotlib.pyplot as plt
    import polars
    from matplotlib.colors import LinearSegmentedColormap

    LOGGER.info(f"正在绘制 labels 到 {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)
    boxes = boxes[:1000000]
    x = polars.DataFrame(boxes, schema=["x", "y", "width", "height"])
    subplot_3_4_color = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([(x / 255) for x in colors(i)])
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
        ax[0].bar_label(y[2])
    else:
        ax[0].set_xlabel("classes")
    boxes = np.column_stack([0.5 - boxes[:, 2:4] / 2, 0.5 + boxes[:, 2:4] / 2]) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for class_id, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box.tolist(), width=1, outline=colors(class_id))
    ax[1].imshow(img)
    ax[1].axis("off")
    ax[2].hist2d(x["x"], x["y"], bins=50, cmap=subplot_3_4_color)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[3].hist2d(x["width"], x["height"], bins=50, cmap=subplot_3_4_color)
    ax[3].set_xlabel("width")
    ax[3].set_ylabel("height")
    for a in {0, 1, 2, 3}:
        for s in {"top", "right", "left", "bottom"}:
            ax[a].spines[s].set_visible(False)
    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def save_one_box(
    xyxy,
    im,
    file: Path = Path("im.jpg"),
    gain: float = 1.02,
    pad: int = 10,
    square: bool = False,
    BGR: bool = False,
    save: bool = True,
):
    """以 crop size multiple {gain} 和 {pad} pixels 将 image crop 保存为 {file}，并保存和/或返回 crop。

    该函数接收 bounding box 和 image，并根据 bounding box 保存 image 的 cropped portion。
    可选地，crop 可被 square 化，也允许通过 gain 和 padding 调整 bounding box。

    参数:
        xyxy (paddle.Tensor | list): 表示 xyxy format bounding box 的 tensor 或 list。
        im (np.ndarray): input image。
        file (Path, optional): cropped image 保存路径。
        gain (float, optional): 增大 bounding box size 的乘法因子。
        pad (int, optional): 添加到 bounding box width 和 height 的 pixels 数量。
        square (bool, optional): 若为 True，则将 bounding box 转为 square。
        BGR (bool, optional): 若为 True，则以 BGR format 返回 image，否则为 RGB。
        save (bool, optional): 若为 True，则将 cropped image 保存到 disk。

    返回:
        (np.ndarray): cropped image。

    示例:
        >>> from ddyolo26.utils.plotting import save_one_box
        >>> xyxy = [50, 50, 150, 150]
        >>> im = cv2.imread("image.jpg")
        >>> cropped_im = save_one_box(xyxy, im, file="cropped.jpg", square=True)
    """
    if not isinstance(xyxy, paddle.Tensor):
        xyxy = paddle.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))
    if square:
        b[:, 2:] = b[:, 2:]._max(1)[0].unsqueeze(1)
    b[:, 2:] = b[:, 2:] * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    grayscale = im.shape[2] == 1
    crop = im[
        int(xyxy[0, 1]) : int(xyxy[0, 3]),
        int(xyxy[0, 0]) : int(xyxy[0, 2]),
        :: 1 if BGR or grayscale else -1,
    ]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)
        f = str(increment_path(file).with_suffix(".jpg"))
        crop = crop.squeeze(-1) if grayscale else crop[..., ::-1] if BGR else crop
        Image.fromarray(crop).save(f, quality=95, subsampling=0)
    return crop


@threaded
def plot_images(
    labels: dict[str, Any],
    images: (paddle.Tensor | np.ndarray) = np.zeros((0, 3, 640, 640), dtype=np.float32),
    paths: (list[str] | None) = None,
    fname: str = "images.jpg",
    names: (dict[int, str] | None) = None,
    on_plot: (Callable | None) = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.25,
) -> np.ndarray | None:
    """绘制包含 labels、bounding boxes、masks 和 keypoints 的 image grid。

    参数:
        labels (dict[str, Any]): 包含 detection data 的字典，keys 如 'cls'、'bboxes'、'conf'、'masks'、
            'keypoints'、'batch_idx'、'img'。
        images (paddle.Tensor | np.ndarray): 要绘制的 images batch。Shape: (batch_size, channels, height, width)。
        paths (list[str] | None): batch 中每张 image 的 file paths 列表。
        fname (str): 绘制后 image grid 的 output filename。
        names (dict[int, str] | None): class indices 到 class names 的 mapping 字典。
        on_plot (Callable | None): 保存 plot 后调用的 callback function。
        max_size (int): output image grid 的 maximum size。
        max_subplots (int): image grid 中 subplots 的 maximum number。
        save (bool): 是否将 plotted image grid 保存到 file。
        conf_thres (float): 显示 detections 的 confidence threshold。

    返回:
        (np.ndarray | None): save 为 False 时返回 numpy array 形式的 plotted image grid，否则返回 None。

    说明:
        该函数同时支持 tensor 和 numpy array inputs，并会自动将 tensor inputs 转为 numpy arrays 后处理。

        Channel Support:
        - 1 channel: Grayscale
        - 2 channels: 第三 channel 补零
        - 3 channels: 原样使用（standard RGB）
        - 4+ channels: 裁剪到前 3 channels
    """
    for k in {"cls", "bboxes", "conf", "masks", "keypoints", "batch_idx", "images"}:
        if k not in labels:
            continue
        if k == "cls" and labels[k].ndim == 2:
            labels[k] = labels[k].squeeze(1)
        if isinstance(labels[k], paddle.Tensor):
            labels[k] = labels[k].cpu().numpy()
    cls = labels.get("cls", np.zeros(0, dtype=np.int64))
    batch_idx = labels.get("batch_idx", np.zeros(cls.shape, dtype=np.int64))
    bboxes = labels.get("bboxes", np.zeros(0, dtype=np.float32))
    confs = labels.get("conf", None)
    masks = labels.get("masks", np.zeros(0, dtype=np.uint8))
    kpts = labels.get("keypoints", np.zeros(0, dtype=np.float32))
    images = labels.get("img", images)
    if len(images) and isinstance(images, paddle.Tensor):
        images = images.cpu().float().numpy()
    c = images.shape[1]
    if c == 2:
        zero = np.zeros_like(images[:, :1])
        images = np.concatenate((images, zero), axis=1)
    elif c > 3:
        images = images[:, :3]
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs**0.5)
    if np.max(images[0]) <= 1:
        images *= 255
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
    fs = int((h + w) * ns * 0.01)
    fs = max(fs, 18)
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=str(names))
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)
        if paths:
            annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None
            conf = confs[idx] if confs is not None else None

            # 绘制分割掩码（半透明叠加到子图区域）
            if len(masks) > 0:
                if idx.shape[0] == masks.shape[0] and masks.max() <= 1:
                    image_masks = masks[idx]
                else:
                    image_masks = masks[[i]]
                    nl = idx.sum()
                    index = np.arange(1, nl + 1).reshape((nl, 1, 1))
                    image_masks = (image_masks == index).astype(np.float32)

                mosaic_arr = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = cv2.resize(image_masks[j].astype(np.uint8), (w, h)).astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        try:
                            mosaic_arr[y : y + h, x : x + w, :][mask] = (
                                mosaic_arr[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                        except Exception:
                            pass
                annotator.fromarray(mosaic_arr)

            if len(bboxes):
                boxes = bboxes[idx]
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:
                        boxes[..., [0, 2]] *= w
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color)
            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    label = f"{c}" if labels else f"{c} {conf[0]:.1f}"
                    annotator.text([x, y], label, txt_color=color, box_color=(64, 64, 64, 128))
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results(file: str = "path/to/results.csv", dir: str = "", on_plot: (Callable | None) = None):
    """从 results CSV file 绘制 training results。

    该函数支持 segmentation、pose estimation 和 classification 等多种数据类型。
    Plots 会保存为 CSV 所在 directory 下的 'results.png'。

    参数:
        file (str, optional): 包含 training results 的 CSV file path。
        dir (str, optional): 未提供 'file' 时，CSV file 所在 directory。
        on_plot (Callable, optional): plotting 后执行的 callback function，以 filename 为参数。

    示例:
        >>> from ddyolo26.utils.plotting import plot_results
        >>> plot_results("path/to/results.csv")
    """
    import matplotlib.pyplot as plt
    import polars as pl
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"在 {save_dir.resolve()} 中未找到 results.csv files，无法绘图。"
    loss_keys, metric_keys = [], []
    for i, f in enumerate(files):
        try:
            data = pl.read_csv(f, infer_schema_length=None)
            if i == 0:
                for c in data.columns:
                    if "loss" in c:
                        loss_keys.append(c)
                    elif "metric" in c:
                        metric_keys.append(c)
                loss_mid, metric_mid = len(loss_keys) // 2, len(metric_keys) // 2
                columns = (
                    loss_keys[:loss_mid] + metric_keys[:metric_mid] + loss_keys[loss_mid:] + metric_keys[metric_mid:]
                )
                fig, ax = plt.subplots(
                    2,
                    len(columns) // 2,
                    figsize=(len(columns) + 2, 6),
                    tight_layout=True,
                )
                ax = ax.ravel()
            x = data.select(data.columns[0]).to_numpy().flatten()
            # 对 epochs 去重：每个 epoch value 保留最后一次出现。
            seen = {}
            for idx, epoch_val in enumerate(x):
                seen[epoch_val] = idx
            unique_mask = np.array(sorted(seen.values()))
            x = x[unique_mask]
            for i, j in enumerate(columns):
                y_raw = data.select(j).to_numpy().flatten().astype("float")
                y = y_raw[unique_mask]
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)
                ax[i].set_title(j, fontsize=12)
        except Exception as e:
            LOGGER.error(f"{f} 绘图出错: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def plt_color_scatter(
    v,
    f,
    bins: int = 20,
    cmap: str = "viridis",
    alpha: float = 0.8,
    edgecolors: str = "none",
):
    """绘制 scatter plot，并基于 2D histogram 为 points 着色。

    参数:
        v (array-like): x-axis values。
        f (array-like): y-axis values。
        bins (int, optional): histogram 的 bins 数量。
        cmap (str, optional): scatter plot 的 colormap。
        alpha (float, optional): scatter plot 的 alpha。
        edgecolors (str, optional): scatter plot 的 edge colors。

    示例:
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """
    import matplotlib.pyplot as plt

    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)
    colors = [
        hist[
            min(np.digitize(v[i], xedges, right=True) - 1, hist.shape[0] - 1),
            min(np.digitize(f[i], yedges, right=True) - 1, hist.shape[1] - 1),
        ]
        for i in range(len(v))
    ]
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)


@plt_settings()
def plot_tune_results(csv_file: str = "tune_results.csv", exclude_zero_fitness_points: bool = True):
    """绘制 'tune_results.csv' file 中保存的 evolution results。

    该函数会为 CSV 中每个 key 生成 scatter plot，并基于 fitness scores 着色；
    best-performing configurations 会在 plots 中高亮。

    参数:
        csv_file (str, optional): 包含 tuning results 的 CSV file path。
        exclude_zero_fitness_points (bool, optional): tuning plots 中不包含 zero fitness points。

    示例:
        >>> plot_tune_results("path/to/tune_results.csv")
    """
    import matplotlib.pyplot as plt
    import polars as pl
    from scipy.ndimage import gaussian_filter1d

    def _save_one_file(file):
        """将一个 matplotlib plot 保存到 'file'。"""
        plt.savefig(file, dpi=200)
        plt.close()
        LOGGER.info(f"已保存 {file}")

    csv_file = Path(csv_file)
    data = pl.read_csv(csv_file, infer_schema_length=None)
    num_metrics_columns = 1
    keys = [x.strip() for x in data.columns][num_metrics_columns:]
    x = data.to_numpy()
    fitness = x[:, 0]
    if exclude_zero_fitness_points:
        mask = fitness > 0
        x, fitness = x[mask], fitness[mask]
    if len(fitness) == 0:
        LOGGER.warning("没有可绘制的有效 fitness values（所有 iterations 可能都已失败）")
        return
    for _ in range(3):
        mean, std = fitness.mean(), fitness.std()
        lower_bound = mean - 3 * std
        mask = fitness >= lower_bound
        if mask.all():
            break
        x, fitness = x[mask], fitness[mask]
    j = np.argmax(fitness)
    n = math.ceil(len(keys) ** 0.5)
    plt.figure(figsize=(10, 10), tight_layout=True)
    for i, k in enumerate(keys):
        v = x[:, i + num_metrics_columns]
        mu = v[j]
        plt.subplot(n, n, i + 1)
        plt_color_scatter(v, fitness, cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, fitness.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})
        plt.tick_params(axis="both", labelsize=8)
        if i % n != 0:
            plt.yticks([])
    _save_one_file(csv_file.with_name("tune_scatter_plots.png"))
    x = range(1, len(fitness) + 1)
    plt.figure(figsize=(10, 6), tight_layout=True)
    plt.plot(x, fitness, marker="o", linestyle="none", label="fitness")
    plt.plot(x, gaussian_filter1d(fitness, sigma=3), ":", label="smoothed", linewidth=2)
    plt.title("Fitness vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    _save_one_file(csv_file.with_name("tune_fitness.png"))


@plt_settings()
def feature_visualization(
    x,
    module_type: str,
    stage: int,
    n: int = 32,
    save_dir: Path = Path("runs/detect/exp"),
):
    """在 inference 期间可视化给定 model module 的 feature maps。

    参数:
        x (paddle.Tensor): 待可视化的 features。
        module_type (str): module type。
        stage (int): model 内的 module stage。
        n (int, optional): 要绘制的 feature maps 最大数量。
        save_dir (Path, optional): 保存 results 的 directory。
    """
    import matplotlib.pyplot as plt

    for m in {"Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:
        if m in module_type:
            return
    if isinstance(x, paddle.Tensor):
        _, channels, height, width = x.shape
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.rsplit('.', 1)[-1]}_features.png"
            blocks = paddle.chunk(x[0].cpu(), channels, dim=0)
            n = min(n, channels)
            _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())
                ax[i].axis("off")
            LOGGER.info(f"正在保存 {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())
