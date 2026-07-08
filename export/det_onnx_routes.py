#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file det_onnx_routes.py
@brief 检测 ONNX 主线路由辅助函数。
@details
本模块只服务于两条检测主线：
- `pre_dist`：YOLO26 `reg_max=1`
- `pre_dfl`：DFL 检测模型

提供三类能力：
1. 识别当前 ONNX 属于哪条主线；
2. 将普通检测 ONNX 改写为 `pre_dist / pre_dfl` 双输出；
3. 统一一步式导出流程中的临时文件管理。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper


PREDIST_LTRB_TENSOR = "/model.23/Concat_output_0"
PREDIST_CLS_TENSOR = "/model.23/Concat_1_output_0"
PREDFL_DFL_TENSOR = "/model.22/Concat_output_0"
PREDFL_CLS_TENSOR = "/model.22/Concat_1_output_0"


def value_info_shape(value_info) -> list[int]:
    """!
    @brief 提取 ONNX `ValueInfoProto` 的静态形状。
    @param value_info ONNX 张量描述对象。
    @return 维度整数列表；未知维度统一返回 `0`。
    """
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(0)
    return dims


def _graph_tensor_names(graph) -> set[str]:
    """!
    @brief 收集图中所有可见张量名称。
    @param graph ONNX graph 对象。
    @return 张量名称集合。
    """
    tensor_names: set[str] = set()
    for node in graph.node:
        for out in node.output:
            tensor_names.add(out)
    for value_info in graph.value_info:
        tensor_names.add(value_info.name)
    return tensor_names


def _lookup_tensor_shape(graph, tensor_name: str) -> list[int]:
    """!
    @brief 查询指定张量在 ONNX 图中的静态形状。
    @param graph ONNX graph 对象。
    @param tensor_name 目标张量名称。
    @return 维度整数列表。
    @throw ValueError 当图中找不到该张量的形状描述时抛出。
    """
    for value_info in list(graph.value_info) + list(graph.output) + list(graph.input):
        if value_info.name == tensor_name:
            return value_info_shape(value_info)
    raise ValueError(f"缺少张量形状信息: {tensor_name}")


def detect_prepared_det_onnx_i8_route(onnx_model: onnx.ModelProto) -> str:
    """!
    @brief 识别已经裁剪好的检测 ONNX 所属主线。
    @param onnx_model 已加载的 ONNX 模型。
    @return `pre_dist` 或 `pre_dfl`。
    @throw ValueError 当输出契约不是受支持的双输出格式时抛出。
    """
    outputs = onnx_model.graph.output
    if len(outputs) != 2:
        raise ValueError("已准备的主线 ONNX 必须正好有两个输出")

    out0_shape = value_info_shape(outputs[0])
    out1_shape = value_info_shape(outputs[1])
    if len(out0_shape) != 3 or len(out1_shape) != 3:
        raise ValueError("已准备的主线 ONNX 必须是 3D 双输出")

    c0 = out0_shape[1]
    if c0 == 4:
        return "pre_dist"
    if c0 > 4 and c0 % 4 == 0:
        return "pre_dfl"
    raise ValueError("不支持的已准备输出契约")


def infer_det_onnx_i8_route(onnx_model: onnx.ModelProto) -> str:
    """!
    @brief 从普通或已裁剪的检测 ONNX 推断 INT8 主线。
    @param onnx_model 已加载的 ONNX 模型。
    @return `pre_dist` 或 `pre_dfl`。
    @throw ValueError 当模型不属于受支持主线时抛出。
    """
    try:
        return detect_prepared_det_onnx_i8_route(onnx_model)
    except ValueError:
        pass

    tensor_names = _graph_tensor_names(onnx_model.graph)
    has_predist = {PREDIST_LTRB_TENSOR, PREDIST_CLS_TENSOR}.issubset(tensor_names)
    has_predfl = {PREDFL_DFL_TENSOR, PREDFL_CLS_TENSOR}.issubset(tensor_names)

    if has_predist:
        return "pre_dist"
    if has_predfl:
        return "pre_dfl"
    raise ValueError("无法推断检测 ONNX INT8 主线；仅支持 pre_dist / pre_dfl")


def _rewrite_outputs(model: onnx.ModelProto, outputs: list[tuple[str, list[int]]]) -> onnx.ModelProto:
    """!
    @brief 重写 ONNX 图输出列表。
    @param model 目标 ONNX 模型。
    @param outputs 输出定义列表，每项为 `(张量名, 形状)`。
    @return 修改后的模型对象。
    """
    graph = model.graph
    while graph.output:
        graph.output.pop()
    for name, shape in outputs:
        graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
    return model


def make_predist_onnx(input_path: str, output_path: str | None = None) -> str:
    """!
    @brief 将普通检测 ONNX 裁剪为 `pre_dist` 双输出。
    @param input_path 输入 ONNX 路径。
    @param output_path 输出 ONNX 路径；为空时自动追加 `_predist.onnx`。
    @return 实际写出的 ONNX 路径。
    @throw ValueError 当图中缺少 `pre_dist` 所需关键张量时抛出。
    """
    model = onnx.load(input_path)
    tensor_names = _graph_tensor_names(model.graph)
    required = {PREDIST_LTRB_TENSOR, PREDIST_CLS_TENSOR}
    missing = required - tensor_names
    if missing:
        raise ValueError(f"缺少 pre_dist 所需张量: {sorted(missing)}")

    _rewrite_outputs(
        model,
        [
            (PREDIST_LTRB_TENSOR, _lookup_tensor_shape(model.graph, PREDIST_LTRB_TENSOR)),
            (PREDIST_CLS_TENSOR, _lookup_tensor_shape(model.graph, PREDIST_CLS_TENSOR)),
        ],
    )
    out_path = output_path or str(Path(input_path).with_name(Path(input_path).stem + "_predist.onnx"))
    onnx.save(model, out_path)
    return out_path


def make_predfl_onnx(input_path: str, output_path: str | None = None) -> str:
    """!
    @brief 将普通检测 ONNX 裁剪为 `pre_dfl` 双输出。
    @param input_path 输入 ONNX 路径。
    @param output_path 输出 ONNX 路径；为空时自动追加 `_predfl.onnx`。
    @return 实际写出的 ONNX 路径。
    @throw ValueError 当图中缺少 `pre_dfl` 所需关键张量时抛出。
    """
    model = onnx.load(input_path)
    tensor_names = _graph_tensor_names(model.graph)
    required = {PREDFL_DFL_TENSOR, PREDFL_CLS_TENSOR}
    missing = required - tensor_names
    if missing:
        raise ValueError(f"缺少 pre_dfl 所需张量: {sorted(missing)}")

    _rewrite_outputs(
        model,
        [
            (PREDFL_DFL_TENSOR, _lookup_tensor_shape(model.graph, PREDFL_DFL_TENSOR)),
            (PREDFL_CLS_TENSOR, _lookup_tensor_shape(model.graph, PREDFL_CLS_TENSOR)),
        ],
    )
    out_path = output_path or str(Path(input_path).with_name(Path(input_path).stem + "_predfl.onnx"))
    onnx.save(model, out_path)
    return out_path


def prepare_det_onnx_i8_input(
    weights_path: str,
    imgsz: int,
    export_onnx_func,
) -> tuple[str, str, list[str]]:
    """!
    @brief 为检测 INT8 一步式入口准备主线 ONNX。
    @param weights_path 用户输入权重路径，可为 `.pdparams / onnx`。
    @param imgsz 导出普通 ONNX 时使用的输入尺寸。
    @param export_onnx_func 普通 ONNX 导出函数，签名为 `(weights_path, imgsz) -> onnx_path`。
    @return 三元组 `(route, prepared_onnx_path, cleanup_paths)`：
      - `route`：推断出的 `pre_dist / pre_dfl`
      - `prepared_onnx_path`：可直接进入 INT8 导出的主线 ONNX 路径
      - `cleanup_paths`：后续需要清理的临时文件/目录列表
    """
    cleanup_paths: list[str] = []
    source_path = weights_path
    p = Path(weights_path)
    if p.suffix == ".pt" and not p.stem.endswith("_paddle"):
        raise ValueError(f"普通 .pt 权重不支持: {weights_path}. 请使用 Paddle 权重或 ONNX。")
    if p.suffix == ".pdparams" or (p.suffix == ".pt" and p.stem.endswith("_paddle")):
        source_path = export_onnx_func(weights_path, imgsz)
        cleanup_paths.append(source_path)

    onnx_model = onnx.load(source_path)
    try:
        route = detect_prepared_det_onnx_i8_route(onnx_model)
        return route, source_path, cleanup_paths
    except ValueError:
        pass

    route = infer_det_onnx_i8_route(onnx_model)
    tmp_dir = Path(tempfile.mkdtemp(prefix="det_i8_onnx_"))
    if route == "pre_dist":
        prepared = make_predist_onnx(source_path, str(tmp_dir / (Path(source_path).stem + "_predist.onnx")))
    else:
        prepared = make_predfl_onnx(source_path, str(tmp_dir / (Path(source_path).stem + "_predfl.onnx")))
    cleanup_paths.append(str(tmp_dir))
    return route, prepared, cleanup_paths


def cleanup_temp_paths(paths: list[str]) -> None:
    """!
    @brief 删除一步式导出过程中的临时文件和目录。
    @param paths 需要清理的路径列表。
    """
    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
