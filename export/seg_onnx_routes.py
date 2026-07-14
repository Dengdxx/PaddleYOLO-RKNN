#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file seg_onnx_routes.py
@brief Seg ONNX 主线路由辅助函数。
@details
本模块服务于两条 Seg INT8 主线：

- `seg_pre_dist`：YOLO26-Seg `reg_max=1`，四输出：
    raw_ltrb [1,4,N] + raw_cls_logits [1,nc,N] + mask_coeff [1,nm,N] + proto [1,nm,H,W]
- `seg_pre_dfl`：YOLOv8-Seg `reg_max=16`，五输出（转置 DFL + score-sum 快速过滤）：
    raw_dfl_transposed [1,N,4*reg_max] + raw_cls_logits [1,nc,N] + mask_coeff [1,nm,N]
    + proto [1,nm,H,W] + score_sum [1,1,N]

提供三类能力：
1. 识别当前 ONNX 属于哪条主线；
2. 将原始 seg ONNX 改写为对应模型族的部署契约；
3. 验证已准备好的 seg ONNX 的输出契约。
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper

from export.input_shape import StaticInputShape, static_imgsz_hw
from export.det_onnx_routes import DEFAULT_STRIDES, expected_anchor_count


# ─────────────────────────────────────────────────────────────────────────────
#  已知 seg_pre_dist 张量名（来自 yolo26n-seg 原始 ONNX 图）
# ─────────────────────────────────────────────────────────────────────────────

SEG_PREDIST_LTRB_TENSOR = "/model.23/Concat_output_0"
"""步幅空间 raw ltrb，形状 [1, 4, N]。"""

SEG_PREDIST_CLS_TENSOR = "/model.23/Concat_1_output_0"
"""原始分类 logits（无 sigmoid），形状 [1, nc, N]。"""

SEG_PREDIST_MASK_COEFF_TENSOR = "/model.23/Concat_2_output_0"
"""mask 系数，形状 [1, nm, N]。"""

SEG_PREDIST_PROTO_TENSOR = "output1"
"""proto 特征图，形状 [1, nm, H, W]。"""


# ─────────────────────────────────────────────────────────────────────────────
#  已知 seg_pre_dfl 张量名（来自 YOLOv8-Seg 原始 ONNX 图）
# ─────────────────────────────────────────────────────────────────────────────

SEG_PREDFL_DFL_TENSOR = "/model.22/Concat_output_0"
"""DFL logits，形状 [1, 4*reg_max, N]。"""

SEG_PREDFL_CLS_TENSOR = "/model.22/Concat_1_output_0"
"""原始分类 logits（无 sigmoid），形状 [1, nc, N]。"""

SEG_PREDFL_MASK_COEFF_TENSOR = "/model.22/Concat_2_output_0"
"""mask 系数，形状 [1, nm, N]。"""

SEG_PREDFL_PROTO_TENSOR = "output1"
"""proto 特征图，形状 [1, nm, H, W]。"""

# ─────────────────────────────────────────────────────────────────────────────
#  内部工具
# ─────────────────────────────────────────────────────────────────────────────


def _value_info_shape(value_info) -> list[int]:
    """!
    @brief 提取 ONNX `ValueInfoProto` 的静态形状。
    @param value_info ONNX 张量描述对象。
    @return 维度整数列表；未知维度统一返回 `-1`。
    """
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(-1)
    return dims


def _graph_tensor_names(graph) -> set[str]:
    """!
    @brief 收集 ONNX 图中所有中间张量名称（含 value_info）。
    @param graph ONNX graph 对象。
    @return 张量名称集合。
    """
    names: set[str] = set()
    for node in graph.node:
        for out in node.output:
            names.add(out)
    for vi in graph.value_info:
        names.add(vi.name)
    for o in graph.output:
        names.add(o.name)
    return names


def _validate_static_input_shape(onnx_model: onnx.ModelProto, imgsz: StaticInputShape, input_path: str) -> None:
    """!
    @brief 校验分割 ONNX 的首个输入与请求的静态尺寸一致。
    @param onnx_model 已加载的 ONNX 模型。
    @param imgsz 请求的静态输入尺寸。
    @param input_path 输入模型路径，用于错误信息。
    @throw ValueError 当模型输入不是静态 NCHW 或尺寸不一致时抛出。
    """
    if not onnx_model.graph.input:
        raise ValueError(f"{input_path} 不包含 ONNX 输入")
    input_shape = _value_info_shape(onnx_model.graph.input[0])
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
        raise ValueError(f"分割部署要求静态 NCHW 输入 [1,3,H,W]，{input_path} 实际为 {input_shape}")
    requested_h, requested_w = static_imgsz_hw(imgsz)
    if input_shape[2:] != [requested_h, requested_w]:
        raise ValueError(
            f"请求输入尺寸 {(requested_h, requested_w)} 与 ONNX 静态输入 {tuple(input_shape[2:])} 不一致: {input_path}"
        )


def _lookup_tensor_shape(graph, tensor_name: str) -> list[int]:
    """!
    @brief 查询指定张量在 ONNX 图中的静态形状。
    @param graph ONNX graph 对象。
    @param tensor_name 目标张量名称。
    @return 维度整数列表。
    @throw ValueError 当找不到该张量的形状描述时抛出。
    """
    for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
        if vi.name == tensor_name:
            return _value_info_shape(vi)
    raise ValueError(f"缺少张量形状信息: {tensor_name}")


def _lookup_tensor_elem_type(graph, tensor_name: str) -> int:
    """!
    @brief 查询指定张量的 ONNX 元素类型。
    @param graph ONNX graph 对象。
    @param tensor_name 目标张量名称。
    @return ONNX `TensorProto.DataType` 整数值。
    @throw ValueError 当找不到该张量的类型描述时抛出。
    """
    for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
        if vi.name == tensor_name:
            return vi.type.tensor_type.elem_type
    raise ValueError(f"缺少张量类型信息: {tensor_name}")


def _default_onnx_opset(model: onnx.ModelProto) -> int:
    """!
    @brief 读取 ONNX 默认算子域的 opset 版本。
    @param model ONNX 模型。
    @return 默认域 opset 版本；缺失时返回 `0`。
    """
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    return 0


def _node_producing(graph, tensor_name: str):
    """!
    @brief 查找生成指定张量的唯一 ONNX 节点。
    @param graph ONNX graph 对象。
    @param tensor_name 目标张量名称。
    @return 生成该张量的节点；找不到时返回 None。
    @throw ValueError 当多个节点生成同名张量时抛出。
    """
    producers = [node for node in graph.node if tensor_name in node.output]
    if len(producers) > 1:
        raise ValueError(f"张量 {tensor_name} 存在多个生产节点")
    return producers[0] if producers else None


def _attribute_ints(node, name: str) -> list[int] | None:
    """!
    @brief 读取 ONNX 节点的整数列表属性。
    @param node ONNX 节点。
    @param name 属性名称。
    @return 属性整数列表；属性不存在时返回 None。
    """
    for attribute in node.attribute:
        if attribute.name == name:
            return list(attribute.ints)
    return None


def _attribute_int(node, name: str, default: int) -> int:
    """!
    @brief 读取 ONNX 节点的单整数属性。
    @param node ONNX 节点。
    @param name 属性名称。
    @param default 属性不存在时的默认值。
    @return 属性整数值。
    """
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def _initializer_ints(graph, tensor_name: str) -> list[int] | None:
    """!
    @brief 读取整型 initializer 的扁平值。
    @param graph ONNX graph 对象。
    @param tensor_name initializer 张量名称。
    @return 整数列表；找不到时返回 None。
    """
    for initializer in graph.initializer:
        if initializer.name == tensor_name:
            return [int(value) for value in numpy_helper.to_array(initializer).reshape(-1)]
    return None


def _validate_score_sum_semantics(graph, cls_name: str, score_sum_name: str) -> None:
    """!
    @brief 验证第五输出由 `ReduceSum(axis=1, Sigmoid(cls))` 直接产生。
    @param graph ONNX graph 对象。
    @param cls_name 第二输出分类 logits 张量名称。
    @param score_sum_name 第五输出 score_sum 张量名称。
    @throw ValueError 当第五输出生产链不符合统一语义时抛出。
    """
    reduce_node = _node_producing(graph, score_sum_name)
    if reduce_node is None or reduce_node.op_type != "ReduceSum":
        raise ValueError("output[4] 必须由 ReduceSum 直接生成")
    if not reduce_node.input:
        raise ValueError("output[4] ReduceSum 缺少数据输入")
    if _attribute_int(reduce_node, "keepdims", 1) != 1:
        raise ValueError("output[4] ReduceSum 必须 keepdims=1")

    axes = _attribute_ints(reduce_node, "axes")
    if axes is None and len(reduce_node.input) >= 2:
        axes = _initializer_ints(graph, reduce_node.input[1])
    if axes != [1]:
        raise ValueError(f"output[4] ReduceSum 必须沿分类轴 axis=1，实际 axes={axes}")

    sigmoid_tensor = reduce_node.input[0]
    dequantize_node = _node_producing(graph, sigmoid_tensor)
    if dequantize_node is not None and dequantize_node.op_type == "DequantizeLinear":
        if not dequantize_node.input:
            raise ValueError("output[4] 的 DequantizeLinear 缺少数据输入")
        quantize_node = _node_producing(graph, dequantize_node.input[0])
        if quantize_node is None or quantize_node.op_type != "QuantizeLinear" or not quantize_node.input:
            raise ValueError("output[4] 仅允许透明的 QuantizeLinear→DequantizeLinear 校准链")
        sigmoid_tensor = quantize_node.input[0]

    sigmoid_node = _node_producing(graph, sigmoid_tensor)
    if sigmoid_node is None or sigmoid_node.op_type != "Sigmoid":
        raise ValueError("output[4] ReduceSum 输入必须由 Sigmoid 直接生成")
    if len(sigmoid_node.input) != 1 or sigmoid_node.input[0] != cls_name:
        raise ValueError("output[4] 必须等于 Sigmoid(output[1]).sum(axis=1)")


def _rewrite_outputs(model: onnx.ModelProto, outputs: list[tuple[str, list[int]]]) -> onnx.ModelProto:
    """!
    @brief 重写 ONNX 图的输出列表。
    @param model 目标 ONNX 模型。
    @param outputs 输出定义列表，每项为 `(张量名, 形状)`。
    @return 修改后的模型（原地修改）。
    """
    graph = model.graph
    while graph.output:
        graph.output.pop()
    for name, shape in outputs:
        graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
    return model


def _is_transient_seg_predfl_outputs(outputs) -> bool:
    """!
    @brief 判断 ONNX 公开输出是否为 `seg_pre_dfl` 瞬态四输出。
    @param outputs ONNX graph output 序列。
    @return 形状符合 `[1,4*reg_max,N] + cls + coeff + proto` 时返回 true。
    """
    if len(outputs) != 4:
        return False
    dfl_shape, cls_shape, coeff_shape, proto_shape = [_value_info_shape(output) for output in outputs]
    return all(
        (
            len(dfl_shape) == 3,
            dfl_shape[1] > 4,
            dfl_shape[1] % 4 == 0,
            len(cls_shape) == 3,
            len(coeff_shape) == 3,
            len(proto_shape) == 4,
            dfl_shape[2] == cls_shape[2],
            dfl_shape[2] == coeff_shape[2],
            coeff_shape[1] == proto_shape[1],
        )
    )


def _is_seg_predist_outputs(outputs) -> bool:
    """!
    @brief 判断 ONNX 公开输出是否为 `seg_pre_dist` 正式四输出。
    @param outputs ONNX graph output 序列。
    @return 形状符合 `[1,4,N] + cls + coeff + proto` 时返回 true。
    """
    if len(outputs) != 4:
        return False
    box_shape, cls_shape, coeff_shape, proto_shape = [_value_info_shape(output) for output in outputs]
    return all(
        (
            len(box_shape) == 3,
            box_shape[1] == 4,
            len(cls_shape) == 3,
            len(coeff_shape) == 3,
            len(proto_shape) == 4,
            box_shape[2] == cls_shape[2],
            box_shape[2] == coeff_shape[2],
            coeff_shape[1] == proto_shape[1],
        )
    )


def _append_score_sum(
    model: onnx.ModelProto,
    cls_name: str,
    cls_shape: list[int],
) -> tuple[str, list[int]]:
    """!
    @brief 为分类 logits 追加 `Sigmoid + ReduceSum` 快速过滤输出。
    @param model 待修改的 ONNX 模型。
    @param cls_name 分类 logits 张量名称。
    @param cls_shape 分类 logits 静态形状 `[1,nc,N]`。
    @return `score_sum` 张量名称与形状。
    """
    graph = model.graph
    sig_name = cls_name + "_sigmoid"
    sum_name = cls_name + "_score_sum"
    axes_name = sum_name + "_axes"
    sum_shape = [cls_shape[0], 1, cls_shape[2]]
    graph.node.append(helper.make_node("Sigmoid", inputs=[cls_name], outputs=[sig_name]))
    graph.initializer.append(helper.make_tensor(axes_name, TensorProto.INT64, [1], [1]))
    graph.node.append(
        helper.make_node(
            "ReduceSum",
            inputs=[sig_name, axes_name],
            outputs=[sum_name],
            keepdims=1,
        )
    )
    graph.value_info.append(helper.make_tensor_value_info(sig_name, TensorProto.FLOAT, cls_shape))
    graph.value_info.append(helper.make_tensor_value_info(sum_name, TensorProto.FLOAT, sum_shape))
    return sum_name, sum_shape


def _resolve_seg_predfl_source_outputs(graph, input_path: str) -> list[tuple[str, list[int]]]:
    """!
    @brief 解析 `seg_pre_dfl` 五输出改写所需的四个源张量。
    @details
    优先使用已裁剪 ONNX 的公开四输出，使 Paddle `dual_raw` 及其他
    等价导出器无需依赖特定张量名；普通 YOLOv8-Seg ONNX 则回退到
    已知的中间张量名。

    @param graph ONNX graph 对象。
    @param input_path 输入 ONNX 路径，用于错误信息。
    @return 按 DFL、分类 logits、mask 系数、proto 排列的张量名称与形状。
    @throw ValueError 当既不是合法四输出，又缺少已知中间张量时抛出。
    """
    if _is_transient_seg_predfl_outputs(graph.output):
        output_shapes = [_value_info_shape(output) for output in graph.output]
        return [(output.name, shape) for output, shape in zip(graph.output, output_shapes)]

    tensor_names = _graph_tensor_names(graph)
    known_sources = [
        SEG_PREDFL_DFL_TENSOR,
        SEG_PREDFL_CLS_TENSOR,
        SEG_PREDFL_MASK_COEFF_TENSOR,
        SEG_PREDFL_PROTO_TENSOR,
    ]
    missing = set(known_sources) - tensor_names
    if missing:
        raise ValueError(f"{input_path} 中缺少 seg_pre_dfl 所需张量: {sorted(missing)}")
    return [(name, _lookup_tensor_shape(graph, name)) for name in known_sources]


def _validate_seg_rewrite_inputs(
    model: onnx.ModelProto,
    source_outputs: list[tuple[str, list[int]]],
    input_path: str,
    require_score_sum: bool,
) -> None:
    """!
    @brief 校验分割五输出图改写的输入前提。
    @details
    RKNN INT8 量化前 ONNX 固定使用 FP32。YOLOv8-Seg 的 `score_sum`
    使用 opset 13+ 的 `ReduceSum` axes 输入形式；YOLO26-Seg
    不增加该旁路，因此不要求 opset 13。

    @param model ONNX 模型。
    @param source_outputs DFL、分类、mask 系数与 proto 源张量。
    @param input_path 输入 ONNX 路径，用于错误信息。
    @param require_score_sum 是否需要追加 `score_sum` 输出。
    @throw ValueError 当 opset 低于 13 或源张量非 FP32 时抛出。
    """
    opset = _default_onnx_opset(model)
    if require_score_sum and opset < 13:
        raise ValueError(f"分割五输出改写要求 ONNX opset >= 13，{input_path} 实际为 {opset}")

    non_fp32 = []
    for tensor_name, _ in source_outputs:
        elem_type = _lookup_tensor_elem_type(model.graph, tensor_name)
        if elem_type != TensorProto.FLOAT:
            type_name = TensorProto.DataType.Name(elem_type)
            non_fp32.append(f"{tensor_name}:{type_name}")
    if non_fp32:
        raise ValueError(f"分割 RKNN INT8 准备要求源输出为 FP32，{input_path} 实际为 {', '.join(non_fp32)}")


# ─────────────────────────────────────────────────────────────────────────────
#  公开 API
# ─────────────────────────────────────────────────────────────────────────────


def make_seg_predist_onnx(input_path: str, output_path: str | None = None) -> str:
    """!
    @brief 将原始 YOLO26-Seg ONNX 裁剪为 `seg_pre_dist` 四输出。
    @details
    输出顺序固定为：
      - `SEG_PREDIST_LTRB_TENSOR`      raw ltrb  [1, 4, N]
      - `SEG_PREDIST_CLS_TENSOR`       cls logits [1, nc, N]
      - `SEG_PREDIST_MASK_COEFF_TENSOR` mask coeff [1, nm, N]
      - `SEG_PREDIST_PROTO_TENSOR`     proto      [1, nm, H, W]

    @param input_path 输入 ONNX 路径（原始 seg ONNX，通常为 `*_raw_*.onnx`）。
    @param output_path 输出 ONNX 路径；为空时自动追加 `_predist.onnx` 后缀。
    @return 实际写出的 ONNX 文件路径。
    @throw ValueError 当图中缺少 `seg_pre_dist` 所需关键张量时抛出。
    """
    from onnx import shape_inference

    model = onnx.load(input_path)
    # 部分由 ultralytics 直接导出的 ONNX 不带中间张量 value_info，先做一次形状推断。
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        print(f"[seg_pre_dist] 警告：shape_inference 失败: {exc}")
    graph = model.graph

    if _is_seg_predist_outputs(graph.output):
        source_outputs = [(output.name, _value_info_shape(output)) for output in graph.output]
    else:
        source_names = [
            SEG_PREDIST_LTRB_TENSOR,
            SEG_PREDIST_CLS_TENSOR,
            SEG_PREDIST_MASK_COEFF_TENSOR,
            SEG_PREDIST_PROTO_TENSOR,
        ]
        missing = set(source_names) - _graph_tensor_names(graph)
        if missing:
            raise ValueError(f"{input_path} 中缺少 seg_pre_dist 所需张量: {sorted(missing)}")
        source_outputs = [(name, _lookup_tensor_shape(graph, name)) for name in source_names]

    _validate_seg_rewrite_inputs(model, source_outputs, input_path, require_score_sum=False)
    (
        (ltrb_name, ltrb_shape),
        (cls_name, cls_shape),
        (coeff_name, coeff_shape),
        (
            proto_name,
            proto_shape,
        ),
    ) = source_outputs
    _rewrite_outputs(
        model,
        [
            (ltrb_name, ltrb_shape),
            (cls_name, cls_shape),
            (coeff_name, coeff_shape),
            (proto_name, proto_shape),
        ],
    )

    out_path = output_path or str(Path(input_path).with_name(Path(input_path).stem + "_predist.onnx"))
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        raise ValueError(f"seg_pre_dist ONNX 校验失败: {input_path}: {exc}") from exc
    onnx.save(model, out_path)
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"[seg_pre_dist] ONNX 已保存: {out_path} ({size_mb:.1f} MB)")
    return out_path


def make_seg_predfl_onnx(input_path: str, output_path: str | None = None) -> str:
    """!
    @brief 将 YOLOv8-Seg 原始 ONNX 裁剪为 `seg_pre_dfl` 五输出。
    @details
    五输出顺序固定为：
      - DFL transposed  [1,N,4*reg_max]  — 转置布局，cache 友好
      - cls logits      [1,nc,N]
      - mask coeff      [1,nm,N]
      - proto           [1,nm,H,W]
      - score_sum       [1,1,N]          — Sigmoid(cls).sum，快速背景过滤

    既支持从普通 YOLOv8-Seg ONNX 的已知中间张量生成，也支持将
    Paddle `dual_raw` 等价四输出规范化为五输出。

    @param input_path 输入 ONNX 路径（原始或已裁剪四输出 ONNX）。
    @param output_path 输出 ONNX 路径；为空时自动追加 `_predfl.onnx` 后缀。
    @return 实际写出的 ONNX 文件路径。
    @throw ValueError 当图中缺少关键张量、opset/FP32 契约不符或模型校验失败时抛出。
    """
    from onnx import shape_inference

    model = onnx.load(input_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        print(f"[seg_pre_dfl] 警告：shape_inference 失败: {exc}")
    graph = model.graph

    source_outputs = _resolve_seg_predfl_source_outputs(graph, input_path)
    _validate_seg_rewrite_inputs(model, source_outputs, input_path, require_score_sum=True)
    (
        (dfl_name, dfl_shape),
        (cls_name, cls_shape),
        (coeff_name, coeff_shape),
        (
            proto_name,
            proto_shape,
        ),
    ) = source_outputs

    dfl_out_name = dfl_name
    dfl_out_shape: list[int] = list(dfl_shape)
    extra_outputs: list[tuple[str, list[int]]] = []

    # DFL 输出转置：[1, 4*reg_max, N] → [1, N, 4*reg_max]
    transposed_name = dfl_name + "_transposed"
    dfl_transposed_shape = [dfl_shape[0], dfl_shape[2], dfl_shape[1]]
    graph.node.append(
        helper.make_node(
            "Transpose",
            inputs=[dfl_name],
            outputs=[transposed_name],
            perm=[0, 2, 1],
        )
    )
    graph.value_info.append(helper.make_tensor_value_info(transposed_name, TensorProto.FLOAT, dfl_transposed_shape))
    dfl_out_name = transposed_name
    dfl_out_shape = dfl_transposed_shape
    print(f"[seg_pre_dfl] DFL transpose: {dfl_shape} → {dfl_transposed_shape}")

    # score-sum 快速过滤：Sigmoid(cls) → ReduceSum(axis=1) → [1,1,N]
    sum_name, sum_shape = _append_score_sum(model, cls_name, cls_shape)
    extra_outputs.append((sum_name, sum_shape))
    print(f"[seg_pre_dfl] score-sum output: {sum_shape}")

    _rewrite_outputs(
        model,
        [
            (dfl_out_name, dfl_out_shape),
            (cls_name, cls_shape),
            (coeff_name, coeff_shape),
            (proto_name, proto_shape),
            *extra_outputs,
        ],
    )

    out_path = output_path or str(Path(input_path).with_name(Path(input_path).stem + "_predfl.onnx"))
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        raise ValueError(f"seg_pre_dfl ONNX 校验失败: {input_path}: {exc}") from exc
    onnx.save(model, out_path)
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"[seg_pre_dfl] ONNX 已保存: {out_path} ({size_mb:.1f} MB)")
    return out_path


def detect_prepared_seg_route(onnx_model: onnx.ModelProto) -> str:
    """!
    @brief 识别已裁剪的 seg ONNX 所属主线。
    @details
    只认可两种可直接进入部署编译的正式契约：
    - `seg_pre_dist`: YOLO26-Seg 4 输出，output[0] = [1,4,N]
    - `seg_pre_dfl`: 5 输出，output[0] = [1,N,4*reg_max]，
      output[4] = score-sum [1,1,N]

    @param onnx_model 已加载的 ONNX 模型。
    @return `"seg_pre_dist"` 或 `"seg_pre_dfl"`。
    @throw ValueError 当输出契约不符合受支持 Seg 主线时抛出。
    """
    outputs = onnx_model.graph.output
    n_out = len(outputs)
    if n_out not in (4, 5):
        raise ValueError(f"分割部署模型要求 YOLO26 四输出或 YOLOv8 五输出，实际 {n_out}")

    for index, output in enumerate(outputs):
        elem_type = output.type.tensor_type.elem_type
        if elem_type != TensorProto.FLOAT:
            type_name = TensorProto.DataType.Name(elem_type)
            raise ValueError(f"output[{index}] 必须为 FP32，实际 {type_name}")

    s0 = _value_info_shape(outputs[0])
    s1 = _value_info_shape(outputs[1])
    s2 = _value_info_shape(outputs[2])
    s3 = _value_info_shape(outputs[3])
    if len(s0) != 3 or s0[0] != 1:
        raise ValueError(f"output[0] 必须是 3D raw box 张量，实际 {s0}")
    if len(s1) != 3 or s1[0] != 1 or s1[1] < 1 or s1[2] < 1:
        raise ValueError(f"output[1] 必须是 [1,nc,N]（cls logits），实际 {s1}")
    if len(s2) != 3 or s2[0] != 1 or s2[1] < 1 or s2[2] < 1:
        raise ValueError(f"output[2] 必须是 [1,nm,N]（mask coeff），实际 {s2}")
    if len(s3) != 4 or s3[0] != 1 or s3[1] < 1 or s3[2] < 1 or s3[3] < 1:
        raise ValueError(f"output[3] 必须是 4D [1,nm,H,W]（proto），实际 {s3}")
    if s3[1] != s2[1]:
        raise ValueError(f"output[2] nm={s2[1]} 与 output[3] nm={s3[1]} 必须一致")

    if n_out == 4 and s0[1] == 4 and s0[2] == s1[2] and s1[2] == s2[2]:
        return "seg_pre_dist"
    if n_out == 5:
        s4 = _value_info_shape(outputs[4])
        common_valid = all(
            (
                len(s4) == 3,
                s4[0] == 1,
                s4[1] == 1,
                s4[2] == s1[2],
                s1[2] == s2[2],
            )
        )
        if common_valid and s0[2] > 4 and s0[2] % 4 == 0 and s0[1] == s1[2]:
            _validate_score_sum_semantics(onnx_model.graph, outputs[1].name, outputs[4].name)
            return "seg_pre_dfl"
    raise ValueError(
        "YOLO26-Seg 需要 [1,4,N]+[1,nc,N]+[1,nm,N]+[1,nm,H,W] 四输出；"
        "YOLOv8-Seg 需要 [1,N,4*reg_max]+[1,nc,N]+[1,nm,N]+[1,nm,H,W]+[1,1,N] 五输出；"
        f"实际 {n_out} 个输出，s0={s0}"
    )


def validate_seg_deployment_contract(
    onnx_model: onnx.ModelProto,
    imgsz: StaticInputShape,
    route: str,
    input_path: str,
    strides: tuple[int, ...] = DEFAULT_STRIDES,
    proto_stride: int = 4,
) -> None:
    """!
    @brief 校验分割 ONNX 的输入、anchor 和 proto 静态几何契约。
    @param onnx_model 已加载的 ONNX 模型。
    @param imgsz 请求的静态输入尺寸。
    @param route 分割部署路由，`seg_pre_dist` 或 `seg_pre_dfl`。
    @param input_path 模型路径，用于错误信息。
    @param strides 检测头特征层步幅。
    @param proto_stride proto 相对模型输入的下采样倍数。
    @throw ValueError 任一静态契约不一致时抛出。
    """
    _validate_static_input_shape(onnx_model, imgsz, input_path)
    detected_route = detect_prepared_seg_route(onnx_model)
    if detected_route != route:
        raise ValueError(f"分割 route 不一致：期望 {route}，实际 {detected_route}: {input_path}")

    input_h, input_w = static_imgsz_hw(imgsz)
    anchors = expected_anchor_count(imgsz, strides)
    outputs = onnx_model.graph.output
    cls_shape = _value_info_shape(outputs[1])
    coeff_shape = _value_info_shape(outputs[2])
    proto_shape = _value_info_shape(outputs[3])
    box_shape = _value_info_shape(outputs[0])
    box_anchors = box_shape[1] if route == "seg_pre_dfl" else box_shape[2]
    if box_anchors != anchors or cls_shape[2] != anchors or coeff_shape[2] != anchors:
        raise ValueError(
            f"{route} anchor 数不一致：输出={box_anchors}/{cls_shape[2]}/{coeff_shape[2]}，"
            f"输入={input_h}x{input_w} 按 strides={list(strides)} 应为 {anchors}"
        )
    expected_proto = [1, coeff_shape[1], input_h // proto_stride, input_w // proto_stride]
    if proto_shape != expected_proto:
        raise ValueError(
            f"{route} proto 形状不一致：输出={proto_shape}，"
            f"输入={input_h}x{input_w} 按 proto_stride={proto_stride} 应为 {expected_proto}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  一步式输入准备（与 det_onnx_routes.prepare_det_onnx_i8_input 对称）
# ─────────────────────────────────────────────────────────────────────────────


def infer_seg_onnx_i8_route(onnx_model: onnx.ModelProto) -> str:
    """!
    @brief 从普通或已裁剪的 seg ONNX 推断 INT8 主线。
    @details
    优先识别 YOLO26 四输出或 YOLOv8 五输出正式契约；
    YOLOv8 的 DFL 四输出仅作为导出瞬态。

    @param onnx_model 已加载的 ONNX 模型。
    @return `"seg_pre_dist"` 或 `"seg_pre_dfl"`。
    @throw ValueError 当模型不属于受支持主线时抛出。
    """
    try:
        return detect_prepared_seg_route(onnx_model)
    except ValueError:
        pass

    if _is_seg_predist_outputs(onnx_model.graph.output):
        return "seg_pre_dist"
    if _is_transient_seg_predfl_outputs(onnx_model.graph.output):
        return "seg_pre_dfl"

    tensor_names = _graph_tensor_names(onnx_model.graph)
    predist_required = {
        SEG_PREDIST_LTRB_TENSOR,
        SEG_PREDIST_CLS_TENSOR,
        SEG_PREDIST_MASK_COEFF_TENSOR,
        SEG_PREDIST_PROTO_TENSOR,
    }
    predfl_required = {
        SEG_PREDFL_DFL_TENSOR,
        SEG_PREDFL_CLS_TENSOR,
        SEG_PREDFL_MASK_COEFF_TENSOR,
        SEG_PREDFL_PROTO_TENSOR,
    }
    if predist_required.issubset(tensor_names):
        return "seg_pre_dist"
    if predfl_required.issubset(tensor_names):
        return "seg_pre_dfl"
    raise ValueError(
        "seg INT8 主线仅支持 yolo26-seg 的 seg_pre_dist 或 yolov8-seg 的 seg_pre_dfl；当前 ONNX 不包含所需关键张量。"
    )


def prepare_seg_onnx_i8_input(
    weights_path: str,
    imgsz: StaticInputShape,
    export_onnx_func,
) -> tuple[str, str, list[str]]:
    """!
    @brief 为 seg INT8 一步式入口准备主线 ONNX。
    @details
    与 `det_onnx_routes.prepare_det_onnx_i8_input` 对称：
      - 输入可为 `.pdparams / onnx`；
      - Paddle 权重通过 `export_onnx_func` 自动导出普通 seg ONNX；
      - YOLO26 四输出 `seg_pre_dist` 与 YOLOv8 五输出 `seg_pre_dfl` 直接返回；
      - YOLOv8 瞬态四输出自动规范化为带 score-sum 的五输出；
      - 其他普通 ONNX 按 route 调用对应图改写函数。

    @param weights_path 用户输入权重路径，可为 `.pdparams / onnx`。
    @param imgsz 导出普通 ONNX 时使用的静态输入尺寸。
    @param export_onnx_func 普通 ONNX 导出函数，签名为 `(weights_path, imgsz) -> onnx_path`。
    @return 三元组 `(route, prepared_onnx_path, cleanup_paths)`：
            - `route`：`"seg_pre_dist"` 或 `"seg_pre_dfl"`；
            - `prepared_onnx_path`：可直接进入 RKNN INT8 编译的主线 ONNX；
            - `cleanup_paths`：后续需要清理的临时文件/目录列表。
    """
    import tempfile

    cleanup_paths: list[str] = []
    source_path = weights_path
    p = Path(weights_path)
    if p.suffix == ".pt" and not p.stem.endswith("_paddle"):
        raise ValueError(f"普通 .pt 权重不支持: {weights_path}. 请使用 Paddle 权重或 ONNX。")
    if p.suffix == ".pdparams" or (p.suffix == ".pt" and p.stem.endswith("_paddle")):
        source_path = export_onnx_func(weights_path, imgsz)
        cleanup_paths.append(source_path)

    onnx_model = onnx.load(source_path)
    _validate_static_input_shape(onnx_model, imgsz, source_path)
    try:
        route = detect_prepared_seg_route(onnx_model)
    except ValueError:
        route = None

    if route is not None:
        validate_seg_deployment_contract(onnx_model, imgsz, route, source_path)
        return route, source_path, cleanup_paths

    route = infer_seg_onnx_i8_route(onnx_model)
    tmp_dir = Path(tempfile.mkdtemp(prefix="seg_i8_onnx_"))
    if route == "seg_pre_dist":
        prepared = make_seg_predist_onnx(source_path, str(tmp_dir / (Path(source_path).stem + "_predist.onnx")))
    else:
        prepared = make_seg_predfl_onnx(source_path, str(tmp_dir / (Path(source_path).stem + "_predfl.onnx")))
    validate_seg_deployment_contract(onnx.load(prepared), imgsz, route, prepared)
    cleanup_paths.append(str(tmp_dir))
    return route, prepared, cleanup_paths
