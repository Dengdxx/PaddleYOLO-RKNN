# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""
@file tools/eval/backend_utils.py
@brief YOLO26 评估后端共享预处理工具库
@details
为 tools.eval.cli 和相关评估脚本提供统一的图像预处理、后处理和指标计算接口。
消除两个评估入口的代码重复，确保 FP32 基线与量化模型的评估流程一致。

@note 此模块无主函数，仅供评估/导出入口导入使用。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from image_shape import ImageShape, ImageShapeLike, parse_imgsz


class LetterboxTransform(NamedTuple):
    """!
    @brief 描述原图到固定 letterbox 画布的精确几何变换。

    @details 比例使用实际取整后的缩放宽高计算，避免使用名义单比例时的亚像素误差。
    """

    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    input_shape: ImageShape
    source_shape: ImageShape


def letterbox_image(
    image: np.ndarray,
    new_shape: ImageShapeLike,
    pad_value: int = 114,
    scaleup: bool = False,
) -> tuple[np.ndarray, LetterboxTransform]:
    """!
    @brief 将 HWC 图像等比缩放并居中填充到固定静态画布。
    @param image HWC 图像，颜色通道语义保持不变。
    @param new_shape 目标画布尺寸，内部统一按 `(height, width)` 解析。
    @param pad_value 填充像素值，默认为 YOLO 约定的 `114`。
    @param scaleup 是否允许放大小图；量化和部署默认关闭。
    @return 二元组 `(填充后图像, 精确几何变换)`。
    @throw ValueError 输入不是有效 HWC 图像或填充值越界时抛出。
    """
    import cv2

    if image.ndim != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"letterbox 要求非空 HWC 图像，实际 shape={image.shape}")
    if pad_value < 0 or pad_value > 255:
        raise ValueError(f"pad_value 必须位于 [0,255]，实际为 {pad_value}")

    target = parse_imgsz(new_shape)
    source = ImageShape(height=int(image.shape[0]), width=int(image.shape[1]))
    ratio = min(target.height / source.height, target.width / source.width)
    if not scaleup:
        ratio = min(ratio, 1.0)
    resized_width = max(1, min(target.width, round(source.width * ratio)))
    resized_height = max(1, min(target.height, round(source.height * ratio)))
    resized = (
        cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        if (source.width, source.height) != (resized_width, resized_height)
        else image
    )

    pad_width = target.width - resized_width
    pad_height = target.height - resized_height
    left = round(pad_width / 2 - 0.1)
    right = pad_width - left
    top = round(pad_height / 2 - 0.1)
    bottom = pad_height - top
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    if canvas.shape[:2] != (target.height, target.width):
        raise RuntimeError(f"letterbox 输出尺寸异常: {canvas.shape[:2]} != {(target.height, target.width)}")
    transform = LetterboxTransform(
        scale_x=resized_width / source.width,
        scale_y=resized_height / source.height,
        pad_x=left,
        pad_y=top,
        input_shape=target,
        source_shape=source,
    )
    return canvas, transform


def _bgr_to_rgb_if_needed(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """将 OpenCV 风格的 BGR 图像转为 RGB，非 3 通道输入保持不变。"""
    if img_hwc_uint8.ndim == 3 and img_hwc_uint8.shape[2] == 3:
        return img_hwc_uint8[..., ::-1]
    return img_hwc_uint8


def prepare_onnx_input(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """为 ONNX Runtime 推理准备单张 HWC uint8 图像。"""
    x = _bgr_to_rgb_if_needed(img_hwc_uint8).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(x, 0)


def prepare_rknn_input(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """为 RKNN 推理准备单张 HWC uint8 图像。"""
    x = _bgr_to_rgb_if_needed(img_hwc_uint8)
    return np.expand_dims(x, 0)


def make_rgb_calib_dataset(
    image_paths,
    input_shape: ImageShapeLike,
    prefix: str = "rknn_calib_rgb_",
) -> str:
    """! 将校准原图显式转换为部署同契约的固定 RGB letterbox 副本。

    RKNN Toolkit 在 `quant_img_RGB2BGR=False` 时按标准图片文件的 RGB 语义校准。
    OpenCV 解码和编码接口使用 BGR ndarray，因此必须在 BGR 空间完成
    letterbox 后直接 `cv2.imwrite`；若先转为 RGB ndarray 再交给
    `cv2.imwrite`，编码后文件的 R/B 通道会被二次交换。

    @param image_paths 校准原图路径迭代器（任意 cv2 可读格式）
    @param input_shape 模型静态输入尺寸，按 `(height, width)` 处理。
    @param prefix 临时目录前缀，便于排查与清理。
    @return 生成的 dataset.txt 绝对路径（同目录下含所有 RGB 副本）
    """
    import tempfile
    from pathlib import Path

    import cv2

    image_paths = [str(p) for p in image_paths]
    if not image_paths:
        raise ValueError("make_rgb_calib_dataset: image_paths 为空")

    tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    rgb_paths = []
    for idx, src in enumerate(image_paths):
        img_bgr = cv2.imread(src)
        if img_bgr is None:
            continue
        image_bgr, _ = letterbox_image(img_bgr, input_shape, pad_value=114, scaleup=False)
        dst = tmp_dir / f"calib_{idx:06d}.png"
        # PNG 无损保存，避免 JPEG 二次量化引入额外噪声影响校准 scale。
        cv2.imwrite(str(dst), image_bgr)
        rgb_paths.append(str(dst))

    if not rgb_paths:
        raise RuntimeError("make_rgb_calib_dataset: 未读取到有效图像")

    list_path = tmp_dir / "dataset.txt"
    list_path.write_text("\n".join(rgb_paths) + "\n", encoding="utf-8")
    return str(list_path)


def patch_onnx_strip_doc_string_for_protobuf7() -> None:
    """! 修补 onnx<1.17 与 protobuf>=5/7 的不兼容。

    `onnx.helper.strip_doc_string` 使用 `descriptor.label`，但该属性在 protobuf
    7.x（和部分 5.x）的 C++/upb 实现里被移除。RKNN-toolkit2 在
    `rknn.build()` 内部会调用此函数，触发：
        AttributeError: 'FieldDescriptor' object has no attribute 'label'

    用一个等价、不依赖 `descriptor.label` 的实现覆盖之。多次调用幂等。

    所有需要走 `rknn.build()` 的脚本都应在 `from rknn.api import RKNN` 之前
    调用本函数一次。
    """
    import google.protobuf.descriptor as _pb_desc
    import google.protobuf.message as _pb_msg
    import onnx.helper as _helper

    if getattr(_helper, "_yolo26_doc_patched", False):
        return

    repeated = _pb_desc.FieldDescriptor.LABEL_REPEATED
    type_message = _pb_desc.FieldDescriptor.TYPE_MESSAGE

    def _safe_strip(proto):
        if not isinstance(proto, _pb_msg.Message):
            raise TypeError(f"proto 必须是 {_pb_msg.Message} 的实例。")
        for descriptor in proto.DESCRIPTOR.fields:
            if descriptor.name == "doc_string":
                proto.ClearField(descriptor.name)
            elif descriptor.type == type_message:
                # protobuf 7+ 移除了 descriptor.label，用 HasField 抛异常
                # 是否抛 ValueError 来判断是否 repeated 字段。
                value = getattr(proto, descriptor.name)
                try:
                    has_field = proto.HasField(descriptor.name)
                    is_repeated = False
                except ValueError:
                    is_repeated = True
                    has_field = False
                if is_repeated:
                    for x in value:
                        _safe_strip(x)
                elif has_field:
                    _safe_strip(value)

    _helper.strip_doc_string = _safe_strip
    _helper._yolo26_doc_patched = True
