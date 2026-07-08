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
2. 将原始 seg ONNX 改写为 `seg_pre_dist` 四输出或 `seg_pre_dfl` 五输出；
3. 验证已准备好的 seg ONNX 的输出契约。
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


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


# ─────────────────────────────────────────────────────────────────────────────
#  公开 API
# ─────────────────────────────────────────────────────────────────────────────


def make_seg_predist_onnx(input_path: str, output_path: str | None = None) -> str:
    """!
    @brief 将原始 seg ONNX 裁剪为 `seg_pre_dist` 四输出。
    @details
    只改写输出列表，不删除或新增图中的其他节点。四输出顺序固定为：
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

    tensor_names = _graph_tensor_names(graph)
    required = {
        SEG_PREDIST_LTRB_TENSOR,
        SEG_PREDIST_CLS_TENSOR,
        SEG_PREDIST_MASK_COEFF_TENSOR,
        SEG_PREDIST_PROTO_TENSOR,
    }
    missing = required - tensor_names
    if missing:
        raise ValueError(f"{input_path} 中缺少 seg_pre_dist 所需张量: {sorted(missing)}")

    ltrb_shape = _lookup_tensor_shape(graph, SEG_PREDIST_LTRB_TENSOR)
    cls_shape = _lookup_tensor_shape(graph, SEG_PREDIST_CLS_TENSOR)
    coeff_shape = _lookup_tensor_shape(graph, SEG_PREDIST_MASK_COEFF_TENSOR)
    proto_shape = _lookup_tensor_shape(graph, SEG_PREDIST_PROTO_TENSOR)

    _rewrite_outputs(
        model,
        [
            (SEG_PREDIST_LTRB_TENSOR, ltrb_shape),
            (SEG_PREDIST_CLS_TENSOR, cls_shape),
            (SEG_PREDIST_MASK_COEFF_TENSOR, coeff_shape),
            (SEG_PREDIST_PROTO_TENSOR, proto_shape),
        ],
    )

    out_path = output_path or str(Path(input_path).with_name(Path(input_path).stem + "_predist.onnx"))
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

    @param input_path 输入 ONNX 路径（YOLOv8-Seg 原始 ONNX）。
    @param output_path 输出 ONNX 路径；为空时自动追加 `_predfl.onnx` 后缀。
    @return 实际写出的 ONNX 文件路径。
    @throw ValueError 当图中缺少 `seg_pre_dfl` 所需关键张量时抛出。
    """
    from onnx import shape_inference

    model = onnx.load(input_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        print(f"[seg_pre_dfl] 警告：shape_inference 失败: {exc}")
    graph = model.graph

    tensor_names = _graph_tensor_names(graph)
    required = {
        SEG_PREDFL_DFL_TENSOR,
        SEG_PREDFL_CLS_TENSOR,
        SEG_PREDFL_MASK_COEFF_TENSOR,
        SEG_PREDFL_PROTO_TENSOR,
    }
    missing = required - tensor_names
    if missing:
        raise ValueError(f"{input_path} 中缺少 seg_pre_dfl 所需张量: {sorted(missing)}")

    dfl_shape = _lookup_tensor_shape(graph, SEG_PREDFL_DFL_TENSOR)
    cls_shape = _lookup_tensor_shape(graph, SEG_PREDFL_CLS_TENSOR)
    coeff_shape = _lookup_tensor_shape(graph, SEG_PREDFL_MASK_COEFF_TENSOR)
    proto_shape = _lookup_tensor_shape(graph, SEG_PREDFL_PROTO_TENSOR)

    dfl_out_name = SEG_PREDFL_DFL_TENSOR
    dfl_out_shape: list[int] = list(dfl_shape)
    extra_outputs: list[tuple[str, list[int]]] = []

    # DFL 输出转置：[1, 4*reg_max, N] → [1, N, 4*reg_max]
    transposed_name = SEG_PREDFL_DFL_TENSOR + "_transposed"
    dfl_transposed_shape = [dfl_shape[0], dfl_shape[2], dfl_shape[1]]
    graph.node.append(
        helper.make_node(
            "Transpose",
            inputs=[SEG_PREDFL_DFL_TENSOR],
            outputs=[transposed_name],
            perm=[0, 2, 1],
        )
    )
    graph.value_info.append(helper.make_tensor_value_info(transposed_name, TensorProto.FLOAT, dfl_transposed_shape))
    dfl_out_name = transposed_name
    dfl_out_shape = dfl_transposed_shape
    print(f"[seg_pre_dfl] DFL transpose: {dfl_shape} → {dfl_transposed_shape}")

    # score-sum 快速过滤：Sigmoid(cls) → ReduceSum(axis=1) → [1,1,N]
    sig_name = SEG_PREDFL_CLS_TENSOR + "_sigmoid"
    sum_name = SEG_PREDFL_CLS_TENSOR + "_score_sum"
    axes_name = sum_name + "_axes"
    sum_shape = [cls_shape[0], 1, cls_shape[2]]
    graph.node.append(
        helper.make_node(
            "Sigmoid",
            inputs=[SEG_PREDFL_CLS_TENSOR],
            outputs=[sig_name],
        )
    )
    # opset 13+ ReduceSum: axes 作为输入（initializer），非属性
    axes_init = helper.make_tensor(axes_name, TensorProto.INT64, [1], [1])
    model.graph.initializer.append(axes_init)
    graph.node.append(
        helper.make_node(
            "ReduceSum",
            inputs=[sig_name, axes_name],
            outputs=[sum_name],
            keepdims=1,
        )
    )
    graph.value_info.append(helper.make_tensor_value_info(sig_name, TensorProto.FLOAT, list(cls_shape)))
    graph.value_info.append(helper.make_tensor_value_info(sum_name, TensorProto.FLOAT, sum_shape))
    extra_outputs.append((sum_name, sum_shape))
    print(f"[seg_pre_dfl] score-sum output: {sum_shape}")

    _rewrite_outputs(
        model,
        [
            (dfl_out_name, dfl_out_shape),
            (SEG_PREDFL_CLS_TENSOR, cls_shape),
            (SEG_PREDFL_MASK_COEFF_TENSOR, coeff_shape),
            (SEG_PREDFL_PROTO_TENSOR, proto_shape),
            *extra_outputs,
        ],
    )

    out_path = output_path or str(Path(input_path).with_name(Path(input_path).stem + "_predfl.onnx"))
    onnx.save(model, out_path)
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"[seg_pre_dfl] ONNX 已保存: {out_path} ({size_mb:.1f} MB)")
    return out_path


def detect_prepared_seg_route(onnx_model: onnx.ModelProto) -> str:
    """!
    @brief 识别已裁剪的 seg ONNX 所属主线。
    @details
    支持两种输出契约：
    - `seg_pre_dist`: 4 输出，output[0] = [1,4,N]
    - `seg_pre_dfl`:  4 或 5 输出，output[0] = [1,4*reg_max,N] 或 [1,N,4*reg_max]
      （5 输出时第 5 个为 score-sum [1,1,N] 快速过滤旁路）

    两者 output[1:4] 均为 cls logits、mask coeff 和 proto。

    @param onnx_model 已加载的 ONNX 模型。
    @return `"seg_pre_dist"` 或 `"seg_pre_dfl"`。
    @throw ValueError 当输出契约不符合受支持 Seg 主线时抛出。
    """
    outputs = onnx_model.graph.output
    n_out = len(outputs)
    if n_out not in (4, 5):
        raise ValueError(f"seg_pre_dist 需要 4 个输出，seg_pre_dfl 需要 4 或 5 个输出，实际 {n_out}")

    s0 = _value_info_shape(outputs[0])
    s1 = _value_info_shape(outputs[1])
    s2 = _value_info_shape(outputs[2])
    s3 = _value_info_shape(outputs[3])

    if len(s0) != 3:
        raise ValueError(f"output[0] 必须是 3D raw box 张量，实际 {s0}")
    if len(s1) != 3 or s1[1] <= 1:
        raise ValueError(f"output[1] 必须是 [1,nc,N]（cls logits），实际 {s1}")
    if len(s2) != 3 or s2[1] <= 1:
        raise ValueError(f"output[2] 必须是 [1,nm,N]（mask coeff），实际 {s2}")
    if len(s3) != 4:
        raise ValueError(f"output[3] 必须是 4D [1,nm,H,W]（proto），实际 {s3}")
    if s3[1] != s2[1] and s2[1] > 0 and s3[1] > 0:
        raise ValueError(f"output[2] nm={s2[1]} 与 output[3] nm={s3[1]} 必须一致")

    if s0[1] == 4:
        return "seg_pre_dist"
    # seg_pre_dfl: 4 输出 legacy [1,4*reg_max,N] 或 5 输出 transposed [1,N,4*reg_max]
    if len(s0) == 3 and s0[1] > 4 and s0[1] % 4 == 0 and n_out == 4:
        return "seg_pre_dfl"
    if n_out == 5 and len(s0) == 3 and s0[2] > 4 and s0[2] % 4 == 0 and s0[1] > s0[2]:
        return "seg_pre_dfl"
    raise ValueError(
        f"seg_pre_dfl 需要 4 个输出 [1,4*reg_max,N] 或 5 个输出 [1,N,4*reg_max]，实际 {n_out} 个输出，s0={s0}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  一步式输入准备（与 det_onnx_routes.prepare_det_onnx_i8_input 对称）
# ─────────────────────────────────────────────────────────────────────────────


def infer_seg_onnx_i8_route(onnx_model: onnx.ModelProto) -> str:
    """!
    @brief 从普通或已裁剪的 seg ONNX 推断 INT8 主线。
    @details
    优先识别已经裁剪好的 `seg_pre_dist` 四输出或 `seg_pre_dfl` 四/五输出契约；
    否则在图中搜索对应关键张量是否存在。

    @param onnx_model 已加载的 ONNX 模型。
    @return `"seg_pre_dist"` 或 `"seg_pre_dfl"`。
    @throw ValueError 当模型不属于受支持主线时抛出。
    """
    try:
        return detect_prepared_seg_route(onnx_model)
    except ValueError:
        pass

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
    imgsz: int,
    export_onnx_func,
) -> tuple[str, str, list[str]]:
    """!
    @brief 为 seg INT8 一步式入口准备主线 ONNX。
    @details
    与 `det_onnx_routes.prepare_det_onnx_i8_input` 对称：
      - 输入可为 `.pdparams / onnx`；
      - Paddle 权重通过 `export_onnx_func` 自动导出普通 seg ONNX；
            - 若 ONNX 已经是 `seg_pre_dist` 四输出或 `seg_pre_dfl` 五输出，直接返回；
            - 否则按 route 调用 `make_seg_predist_onnx / make_seg_predfl_onnx` 完成图手术。

    @param weights_path 用户输入权重路径，可为 `.pdparams / onnx`。
    @param imgsz 导出普通 ONNX 时使用的输入尺寸。
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
    try:
        route = detect_prepared_seg_route(onnx_model)
        return route, source_path, cleanup_paths
    except ValueError:
        pass

    route = infer_seg_onnx_i8_route(onnx_model)
    tmp_dir = Path(tempfile.mkdtemp(prefix="seg_i8_onnx_"))
    if route == "seg_pre_dist":
        prepared = make_seg_predist_onnx(source_path, str(tmp_dir / (Path(source_path).stem + "_predist.onnx")))
    else:
        prepared = make_seg_predfl_onnx(source_path, str(tmp_dir / (Path(source_path).stem + "_predfl.onnx")))
    cleanup_paths.append(str(tmp_dir))
    return route, prepared, cleanup_paths
