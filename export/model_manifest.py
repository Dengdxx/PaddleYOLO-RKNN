#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file model_manifest.py
@brief 生成与 ONNX/RKNN 产物哈希绑定的部署模型清单。
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

import onnx
import yaml

from export.det_onnx_routes import DEFAULT_STRIDES, expected_anchor_count, value_info_shape
from export.input_shape import StaticInputShape
from image_shape import validate_imgsz_stride


SCHEMA_VERSION = 1


def file_sha256(path: str | Path) -> str:
    """!
    @brief 计算文件 SHA-256 摘要。
    @param path 待计算的文件路径。
    @return 小写十六进制 SHA-256 字符串。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_manifest_path(model_path: str | Path) -> Path:
    """!
    @brief 返回模型产物对应的 `.model.yaml` 路径。
    @param model_path ONNX 或 RKNN 模型路径。
    @return 保留原扩展名的 sidecar 路径，例如 `model.rknn.model.yaml`。
    """
    path = Path(model_path)
    return path.with_suffix(path.suffix + ".model.yaml")


def _parse_onnx_metadata(model: onnx.ModelProto) -> dict[str, Any]:
    """!
    @brief 解析 ONNX metadata_props 中的 Python 字面量字段。
    @param model ONNX 模型。
    @return 元数据字典；无法解析的值保留为字符串。
    """
    result: dict[str, Any] = {}
    for item in model.metadata_props:
        try:
            result[item.key] = ast.literal_eval(item.value)
        except (SyntaxError, ValueError):
            result[item.key] = item.value
    return result


def _normalize_class_names(names: Any) -> list[str] | None:
    """!
    @brief 将列表或 ID 字典形式的类别名规范为有序列表。
    @param names 原始类别名对象。
    @return 规范后列表；类型不受支持时返回 `None`。
    """
    if isinstance(names, dict):
        try:
            return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
        except (TypeError, ValueError):
            return None
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return None


def _dataset_class_names(data_yaml: str | Path | None) -> list[str] | None:
    """!
    @brief 从数据集 YAML 读取权威类别名。
    @param data_yaml 数据集 YAML 路径；为空时返回 `None`。
    @return 按类别 ID 排序的名称列表。
    """
    if data_yaml is None:
        return None
    path = Path(data_yaml).resolve()
    if path.is_dir():
        class_root = path / "train" if (path / "train").is_dir() else path
        names = sorted(child.name for child in class_root.iterdir() if child.is_dir())
        if not names:
            raise ValueError(f"分类数据集目录 {path} 不包含类别子目录")
        return names
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = _normalize_class_names(payload.get("names"))
    if not names:
        raise ValueError(f"数据集 {path} 缺少有效 names")
    declared_nc = payload.get("nc")
    if declared_nc is not None and int(declared_nc) != len(names):
        raise ValueError(f"数据集 {path} 的 nc={declared_nc} 与 names={len(names)} 不一致")
    return names


def _is_numeric_placeholder_names(names: list[str]) -> bool:
    """!
    @brief 判断类别名是否仅为 `0..N-1` 占位符。
    @param names 待检查类别名列表。
    @return 全部名称均与其索引相同时返回 true。
    """
    return bool(names) and all(name == str(index) for index, name in enumerate(names))


def _class_names(
    metadata: dict[str, Any],
    class_count: int,
    data_yaml: str | Path | None = None,
    explicit_class_names: list[str] | tuple[str, ...] | dict | None = None,
) -> list[str]:
    """!
    @brief 将 ONNX 类别元数据规范为有序字符串列表。
    @param metadata ONNX 元数据。
    @param class_count 输出分类通道数。
    @return 长度严格等于 `class_count` 的类别名列表。
    """
    metadata_names = _normalize_class_names(metadata.get("names"))
    if metadata_names is not None and len(metadata_names) != class_count:
        raise ValueError(f"ONNX metadata names={len(metadata_names)} 与模型类别数 {class_count} 不一致")
    ordered = _normalize_class_names(explicit_class_names)
    if ordered is None:
        ordered = _dataset_class_names(data_yaml)
    if ordered is None:
        ordered = metadata_names
    if ordered is None or _is_numeric_placeholder_names(ordered):
        raise ValueError("ONNX metadata 未提供可信类别名；请通过 data_yaml 或 class_names 显式提供")
    if len(ordered) != class_count:
        raise ValueError(f"ONNX 类别名数量 {len(ordered)} 与输出类别数 {class_count} 不一致")
    return ordered


def infer_native_output_route(contract_onnx_path: str | Path) -> str:
    """!
    @brief 从 ONNX 元数据推断未裁剪模型的通用输出路由。
    @param contract_onnx_path ONNX 模型路径。
    @return `e2e` / `yolo` / `seg_e2e` / `seg_yolo` 之一。
    """
    metadata = _parse_onnx_metadata(onnx.load(contract_onnx_path))
    task = str(metadata.get("task", "detect"))
    end2end = bool(metadata.get("end2end", False))
    if task == "classify":
        return "classify"
    if task == "segment":
        return "seg_e2e" if end2end else "seg_yolo"
    return "e2e" if end2end else "yolo"


def build_model_manifest(
    model_path: str | Path,
    contract_onnx_path: str | Path,
    output_route: str,
    imgsz: StaticInputShape,
    strides: tuple[int, ...] = DEFAULT_STRIDES,
    data_yaml: str | Path | None = None,
    class_names: list[str] | tuple[str, ...] | dict | None = None,
) -> dict[str, Any]:
    """!
    @brief 从已校验 ONNX 契约构造部署模型清单。
    @param model_path 清单要绑定的最终 ONNX/RKNN 产物。
    @param contract_onnx_path 与最终产物共享输入输出契约的 ONNX 图。
    @param output_route 公开部署路由，如 `predist` 或 `seg_predfl`。
    @param imgsz 静态输入尺寸。
    @param strides 检测头特征层步幅。
    @return 可序列化为 YAML 的清单字典。
    """
    model_path = Path(model_path).resolve()
    contract_path = Path(contract_onnx_path).resolve()
    graph = onnx.load(contract_path)
    metadata = _parse_onnx_metadata(graph)
    input_h, input_w = validate_imgsz_stride(imgsz, 32)
    if not graph.graph.input:
        raise ValueError(f"{contract_path} 不包含 ONNX 输入")
    actual_input_shape = value_info_shape(graph.graph.input[0])
    expected_input_shape = [1, 3, input_h, input_w]
    if actual_input_shape != expected_input_shape:
        raise ValueError(
            f"清单输入契约不一致：期望 {expected_input_shape}，{contract_path} 实际为 {actual_input_shape}"
        )
    output_shapes = [value_info_shape(output) for output in graph.graph.output]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_file": model_path.name,
        "model_sha256": file_sha256(model_path),
        "contract_onnx_sha256": file_sha256(contract_path),
        "input_shape": [input_h, input_w],
        "color": "rgb",
        "resize_mode": "letterbox",
        "pad_value": 114,
        "scaleup": False,
        "tensor_encoding": {
            "logical_dtype": "uint8",
            "logical_layout": "hwc",
            "backend_binding": "query_model_attributes",
        },
        "normalization": "divide_by_255",
        "output_route": output_route,
        "output_shapes": output_shapes,
    }
    task = "segment" if output_route.startswith("seg_") else "classify" if output_route == "classify" else "detect"
    manifest["task"] = task

    routed_heads = {"predist", "predfl", "seg_predist", "seg_predfl"}
    if output_route not in routed_heads:
        metadata_names = _normalize_class_names(metadata.get("names"))
        if output_route == "classify":
            if len(output_shapes) != 1 or len(output_shapes[0]) != 2 or output_shapes[0][0] != 1:
                raise ValueError(f"classify 需要单输出 [1,nc]，实际为 {output_shapes}")
            class_count = int(output_shapes[0][1])
        elif output_route == "e2e":
            if (
                len(output_shapes) != 1
                or len(output_shapes[0]) != 3
                or output_shapes[0][0] != 1
                or output_shapes[0][2] != 6
            ):
                raise ValueError(f"e2e 需要单输出 [1,K,6]，实际为 {output_shapes}")
            if metadata_names is None:
                raise ValueError("e2e 无法从输出反推 nc，ONNX metadata 必须提供 names")
            class_count = len(metadata_names)
        elif output_route == "yolo":
            anchors = expected_anchor_count(imgsz, strides)
            if (
                len(output_shapes) != 1
                or len(output_shapes[0]) != 3
                or output_shapes[0][0] != 1
                or output_shapes[0][1] <= 4
                or output_shapes[0][2] != anchors
            ):
                raise ValueError(f"yolo 需要单输出 [1,4+nc,{anchors}]，实际为 {output_shapes}")
            class_count = int(output_shapes[0][1] - 4)
            manifest["head_strides"] = list(strides)
            manifest["anchor_count"] = anchors
        elif output_route in {"seg_e2e", "seg_yolo"}:
            if len(output_shapes) != 2:
                raise ValueError(f"{output_route} 需要预测+proto 双输出，实际为 {output_shapes}")
            pred_shape, proto_shape = output_shapes
            if (
                len(proto_shape) != 4
                or proto_shape[0] != 1
                or proto_shape[1] <= 0
                or proto_shape[2:] != [input_h // 4, input_w // 4]
            ):
                raise ValueError(
                    f"{output_route} proto 必须为 [1,nm,{input_h // 4},{input_w // 4}]，实际为 {proto_shape}"
                )
            mask_dim = int(proto_shape[1])
            if output_route == "seg_e2e":
                if len(pred_shape) != 3 or pred_shape[0] != 1 or pred_shape[2] != 6 + mask_dim:
                    raise ValueError(f"seg_e2e 预测必须为 [1,K,6+nm]，nm={mask_dim}，实际为 {pred_shape}")
                if metadata_names is None:
                    raise ValueError("seg_e2e 无法从输出反推 nc，ONNX metadata 必须提供 names")
                class_count = len(metadata_names)
            else:
                anchors = expected_anchor_count(imgsz, strides)
                if (
                    len(pred_shape) != 3
                    or pred_shape[0] != 1
                    or pred_shape[2] != anchors
                    or pred_shape[1] <= 4 + mask_dim
                ):
                    raise ValueError(f"seg_yolo 预测必须为 [1,4+nc+nm,{anchors}]，nm={mask_dim}，实际为 {pred_shape}")
                class_count = int(pred_shape[1] - 4 - mask_dim)
                manifest["head_strides"] = list(strides)
                manifest["anchor_count"] = anchors
            manifest["mask_dim"] = mask_dim
            manifest["proto_shape"] = [int(proto_shape[2]), int(proto_shape[3])]
        else:
            raise ValueError(f"不支持的 output_route={output_route}")
        manifest["class_names"] = _class_names(metadata, class_count, data_yaml, class_names)
        return manifest

    if len(output_shapes) < 2:
        raise ValueError(f"部署 route={output_route} 至少需要 box/cls 两个输出")
    is_segment = output_route.startswith("seg_")
    cls_shape = output_shapes[1]
    if len(cls_shape) != 3 or cls_shape[0] != 1 or cls_shape[1] <= 0:
        raise ValueError(f"分类输出必须为 [1,nc,anchors]，实际为 {cls_shape}")
    class_count = int(cls_shape[1])
    box_shape = output_shapes[0]
    anchors = expected_anchor_count(imgsz, strides)
    if len(box_shape) != 3 or box_shape[0] != 1:
        raise ValueError(f"box 输出必须为静态三维张量，实际为 {box_shape}")
    if box_shape[1] == anchors:
        box_channels = int(box_shape[2])
    elif box_shape[2] == anchors:
        box_channels = int(box_shape[1])
    else:
        raise ValueError(f"box 输出无法识别 anchor 轴: shape={box_shape}, anchors={anchors}")
    if cls_shape[2] != anchors:
        raise ValueError(f"分类输出 anchor 数应为 {anchors}，实际为 {cls_shape}")
    if box_channels < 4 or box_channels % 4 != 0:
        raise ValueError(f"box 通道必须为 4*reg_max，实际为 {box_channels}")
    manifest["head_strides"] = list(strides)
    manifest["anchor_count"] = anchors
    manifest["reg_max"] = box_channels // 4
    manifest["class_names"] = _class_names(metadata, class_count, data_yaml, class_names)
    if is_segment:
        if len(output_shapes) < 4:
            raise ValueError(f"分割 route={output_route} 至少需要四输出")
        coeff_shape = output_shapes[2]
        proto_shape = output_shapes[3]
        if (
            len(coeff_shape) != 3
            or coeff_shape[0] != 1
            or coeff_shape[1] <= 0
            or coeff_shape[2] != anchors
            or len(proto_shape) != 4
            or proto_shape[0] != 1
            or proto_shape[1] != coeff_shape[1]
            or proto_shape[2:] != [input_h // 4, input_w // 4]
        ):
            raise ValueError(
                f"分割 coeff/proto 契约不一致: coeff={coeff_shape}, proto={proto_shape}, input={input_h}x{input_w}"
            )
        manifest["mask_dim"] = int(coeff_shape[1])
        manifest["proto_shape"] = [int(proto_shape[2]), int(proto_shape[3])]
    return manifest


def write_model_manifest(
    model_path: str | Path,
    contract_onnx_path: str | Path,
    output_route: str,
    imgsz: StaticInputShape,
    strides: tuple[int, ...] = DEFAULT_STRIDES,
    data_yaml: str | Path | None = None,
    class_names: list[str] | tuple[str, ...] | dict | None = None,
) -> Path:
    """!
    @brief 原子生成与最终模型哈希绑定的 `.model.yaml`。
    @param model_path 最终 ONNX/RKNN 产物路径。
    @param contract_onnx_path 已校验的 ONNX 契约路径。
    @param output_route 公开部署路由。
    @param imgsz 静态输入尺寸。
    @param strides 检测头特征层步幅。
    @return 写入完成的 sidecar 路径。
    """
    output = model_manifest_path(model_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    manifest = build_model_manifest(
        model_path,
        contract_onnx_path,
        output_route,
        imgsz,
        strides,
        data_yaml=data_yaml,
        class_names=class_names,
    )
    temporary.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output
