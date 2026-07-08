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

import numpy as np


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


def make_rgb_calib_dataset(image_paths, prefix: str = "rknn_calib_rgb_") -> str:
    """! 把磁盘上的校准图（cv2 默认按 BGR 读取）批量转换为 RGB 副本。

    YOLO26 模型权重训练时使用 RGB（Ultralytics dataloader 默认行为），因此
    RKNN INT8 量化校准也必须喂 RGB，否则 quantize scale 会按 BGR 像素分布
    估算并与权重期望的 RGB 通道顺序错位，导致 mask/seg 头精度严重下降
    （det 头容忍度高，但仍非最优）。本函数将每张图像 cv2.imread → BGR2RGB →
    cv2.imwrite 到临时目录，并生成 RKNN 所需的 dataset.txt。

    @param image_paths 校准原图路径迭代器（任意 cv2 可读格式）
    @param prefix 临时目录前缀，便于排查与清理
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
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        dst = tmp_dir / f"calib_{idx:06d}.png"
        # PNG 无损保存，避免 JPEG 二次量化引入额外噪声影响校准 scale。
        cv2.imwrite(str(dst), img_rgb)
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
