#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""重写 Paddle 导出的 ONNX 后处理子图，使其兼容 RKNN NPU。

问题背景: PaddlePaddle 的 ONNX 导出器 (paddle2onnx) 在 YOLO end2end NMS
后处理中生成的算子模式与 PyTorch 不同：

  Paddle:  Cast(int→float) → Div(float) → Floor → Cast(float→int) → ... → Sub  (≡ idx % nc)
  PyTorch: Div(int) + Mod(int)

  Paddle:  Expand(x, target_shape)
  PyTorch: Tile(x, repeats)

两者数学等价，且 RKNN Toolkit2 声称支持 Expand/Floor/Cast，但 RK3588 NPU
实际运行时对 Paddle 的模式产生错误结果（bbox 全零）。
本脚本将 Paddle ONNX 转换为 PyTorch 等价的算子模式。

用法:
    python fix_paddle_onnx.py input.onnx output.onnx
    python fix_paddle_onnx.py input.onnx  # 就地覆写
"""

import argparse
import copy
import logging
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

logger = logging.getLogger(__name__)


def _get_initializer_value(graph, name):
    """根据名称获取 ONNX 图中的常量初始化值（numpy 数组）。"""
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _find_node_by_output(graph, output_name):
    """查找生产指定输出张量的节点。"""
    for node in graph.node:
        if output_name in node.output:
            return node
    return None


def _find_consumers(graph, tensor_name):
    """查找所有消费指定张量的节点。"""
    return [n for n in graph.node if tensor_name in n.input]


def _get_attr(node, name, default=None):
    """获取节点的指定属性值（支持 INT/FLOAT/INTS 类型）。"""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
    return default


def _remove_node(graph, node):
    """从 ONNX 图中移除指定节点。"""
    for i, n in enumerate(graph.node):
        if n is node:
            graph.node.pop(i)
            return i
    return -1


def _replace_node(graph, old_node, new_node):
    """在图的相同位置替换节点，保持拓扑顺序。"""
    for i, n in enumerate(graph.node):
        if n is old_node:
            graph.node[i].CopyFrom(new_node)
            return i
    return -1


def _insert_node_before(graph, ref_node, new_node):
    """在参考节点前插入新节点。"""
    for i, n in enumerate(graph.node):
        if n is ref_node:
            graph.node.insert(i, new_node)
            return i
    graph.node.append(new_node)
    return len(graph.node) - 1


def _remove_initializer(graph, name):
    """从图中移除指定名称的初始化器。"""
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            graph.initializer.pop(i)
            return True
    return False


def _add_initializer(graph, name, array):
    """将 numpy 数组作为常量初始化器添加到图中。"""
    tensor = numpy_helper.from_array(array, name=name)
    graph.initializer.append(tensor)
    return name


def _get_tensor_shape(graph, tensor_name):
    """从 value_info 或图输入中获取张量的静态形状。动态维度返回 None。"""
    for vi in list(graph.value_info) + list(graph.input):
        if vi.name == tensor_name:
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)  # dynamic
            return shape
    return None


def rewrite_expand_to_tile(graph):
    """将 Expand 节点替换为 Tile 节点。

    转换规则: Expand(input, target_shape) → Tile(input, repeats)
    其中 repeats = target_shape / input_shape（仅对 input 维度为 1 的轴进行平铺）。

    RKNN NPU 对 Expand 的实现有缺陷，而 Tile 表现正确。

    返回:
        int: 成功替换的节点数量
    """
    count = 0

    for node in list(graph.node):
        if node.op_type != "Expand":
            continue

        input_name = node.input[0]
        shape_name = node.input[1]
        output_name = node.output[0]

        # 从初始化器获取目标形状
        target_shape = _get_initializer_value(graph, shape_name)
        if target_shape is None:
            logger.warning(f"Expand {node.name}: 目标形状 {shape_name} 不是常量，跳过")
            continue

        # 计算重复次数：repeats[i] = target_shape[i] / input_shape[i]
        input_shape = _get_tensor_shape(graph, input_name)
        if input_shape is None:
            logger.warning(f"Expand {node.name}: 无法确定 {input_name} 的输入形状，跳过")
            continue

        repeats = np.ones(len(target_shape), dtype=np.int64)
        for i in range(len(target_shape)):
            if input_shape[i] is not None and input_shape[i] > 0:
                repeats[i] = target_shape[i] // input_shape[i]
            else:
                repeats[i] = target_shape[i]  # 假设输入维度为 1（动态）
        repeats_name = f"{node.name}_tile_repeats"
        _add_initializer(graph, repeats_name, repeats)

        # 创建 Tile 节点（原地替换保持拓扑顺序）
        tile_node = helper.make_node(
            "Tile",
            inputs=[input_name, repeats_name],
            outputs=[output_name],
            name=f"{node.name}_as_Tile",
        )

        _replace_node(graph, node, tile_node)
        count += 1
        logger.info(f"  Expand→Tile: {node.name}, repeats={repeats.tolist()}")

    return count


def rewrite_floordiv_to_intdiv_and_mod(graph):
    """将浮点域整除链 Cast→Div→Floor→Cast 替换为整数 Div + Mod。

    Paddle 导出的 ONNX 用浮点运算实现整数除法和取模（idx % nc）：
        Cast.14: idx(int64) → float32
        Div.0:   float32 / nc_float → float32
        Floor.0: floor(result) → float32           ← row = idx // nc
        Cast.16: float32 → int64
        Cast.17: int64 → float32
        Mul.105: float32 * nc_float → float32       ← row * nc
        Cast.18: result → int64
        Sub.2:   idx(int64) - Cast.18 → int64       ← col = idx - row*nc = idx % nc

    PyTorch 等价形式（NPU 可正确执行）：
        Div:  idx(int64) / nc(int64) → int64         （整数除法）
        Mod:  idx(int64) % nc(int64) → int64         （取余）

    返回:
        int: 成功替换的模式数量
    """
    count = 0

    # 查找 Floor 节点 — 每个都是整除模式的锚点
    for floor_node in [n for n in graph.node if n.op_type == "Floor"]:
        # 验证模式：Floor ← Div ← Cast(to=float)
        div_node = _find_node_by_output(graph, floor_node.input[0])
        if div_node is None or div_node.op_type != "Div":
            continue

        cast_to_float = _find_node_by_output(graph, div_node.input[0])
        if cast_to_float is None or cast_to_float.op_type != "Cast":
            continue
        if _get_attr(cast_to_float, "to") != TensorProto.FLOAT:
            continue

        # 原始整数索引输入
        original_idx = cast_to_float.input[0]
        # 浮点除数（如 80.0）
        divisor_float_name = div_node.input[1]
        divisor_val = _get_initializer_value(graph, divisor_float_name)
        if divisor_val is None:
            continue
        nc = int(divisor_val.item())

        # Floor 输出 → 应连接到 Cast(to=int64)
        floor_output = floor_node.output[0]
        cast_to_int64 = None
        for consumer in _find_consumers(graph, floor_output):
            if consumer.op_type == "Cast" and _get_attr(consumer, "to") == TensorProto.INT64:
                cast_to_int64 = consumer
                break

        if cast_to_int64 is None:
            continue

        # Cast.16 输出是整数行索引。现在查找 Mod 等价链：
        # Cast.17: row(int64) → float
        # Mul: row_float * nc_float → float
        # Cast.18: result → int64
        # Sub: original_idx - result → col_idx（即 idx % nc）
        row_int64_output = cast_to_int64.output[0]

        # 查找 Cast.17（row int64 → float）
        cast_row_to_float = None
        for consumer in _find_consumers(graph, row_int64_output):
            if consumer.op_type == "Cast" and _get_attr(consumer, "to") == TensorProto.FLOAT:
                cast_row_to_float = consumer
                break

        mul_node = None
        cast_product_to_int = None
        sub_node = None

        if cast_row_to_float is not None:
            # 查找 Mul(row_float * nc)
            for consumer in _find_consumers(graph, cast_row_to_float.output[0]):
                if consumer.op_type == "Mul":
                    mul_node = consumer
                    break

            if mul_node is not None:
                # 查找 Cast(Mul result → int64)
                for consumer in _find_consumers(graph, mul_node.output[0]):
                    if consumer.op_type == "Cast" and _get_attr(consumer, "to") == TensorProto.INT64:
                        cast_product_to_int = consumer
                        break

                if cast_product_to_int is not None:
                    # 查找 Sub(original_idx - product_int64)
                    for consumer in _find_consumers(graph, cast_product_to_int.output[0]):
                        if consumer.op_type == "Sub":
                            sub_node = consumer
                            break

        # 创建整数除数常量
        nc_int_name = f"{floor_node.name}_nc_int64"
        _add_initializer(graph, nc_int_name, np.array(nc, dtype=np.int64))

        # 创建整数 Div 节点：row = original_idx / nc（整数除法）
        # 放在原链的第一个节点（Cast.14）位置
        int_div_node = helper.make_node(
            "Div",
            inputs=[original_idx, nc_int_name],
            outputs=[cast_to_int64.output[0]],  # 复用 row_int64 输出名
            name=f"{floor_node.name}_intDiv",
        )

        if sub_node is not None:
            # 创建 Mod 节点：col = original_idx % nc
            mod_node = helper.make_node(
                "Mod",
                inputs=[original_idx, nc_int_name],
                outputs=[sub_node.output[0]],  # 复用 Sub 的输出名
                name=f"{floor_node.name}_Mod",
                fmod=0,
            )

            # 用 intDiv 替换 cast_to_float（第一个节点），Mod 放在其后
            _replace_node(graph, cast_to_float, int_div_node)
            _insert_node_before(graph, div_node, mod_node)

            # 移除旧链节点（cast_to_float 已被替换，不再重复移除）
            for n in [div_node, floor_node, cast_to_int64, cast_row_to_float, mul_node, cast_product_to_int, sub_node]:
                if n is not None:
                    _remove_node(graph, n)

            logger.info(f"  Floor+Sub→Mod: {floor_node.name}, nc={nc}")
        else:
            # 用 intDiv 替换 cast_to_float，移除其余节点
            _replace_node(graph, cast_to_float, int_div_node)
            for n in [div_node, floor_node, cast_to_int64]:
                _remove_node(graph, n)
            logger.info(f"  Floor→intDiv: {floor_node.name}, nc={nc}（未找到 Mod 链）")

        count += 1

    return count


def remove_dangling_cast_chains(graph):
    """移除仅供最终 Concat 消费的冗余 Cast(int64→float) 节点。

    RKNN 支持混合类型的 Concat 操作，因此部分类型转换是多余的。
    """
    count = 0

    # 构建消费者映射
    consumers = {}
    for n in graph.node:
        for inp in n.input:
            if inp not in consumers:
                consumers[inp] = []
            consumers[inp].append(n)

    for node in list(graph.node):
        if node.op_type != "Cast" or _get_attr(node, "to") != TensorProto.FLOAT:
            continue

        # 检查此 Cast 的输出是否仅流向 Concat（最终输出拼接）
        node_consumers = consumers.get(node.output[0], [])
        if len(node_consumers) == 1 and node_consumers[0].op_type == "Concat":
            concat_node = node_consumers[0]
            # 检查输入是否已经是 int64（来自 Div/Mod/Unsqueeze）
            # 用 Cast 的输入替换 Concat 的输入引用（跳过 Cast）
            for i, inp in enumerate(concat_node.input):
                if inp == node.output[0]:
                    # 在 concat 前插入新的 Cast to float 以保持类型一致
                    # 实际上 concat 输出是 float（坐标），所以需要这个 cast
                    # 但如果此 Cast 是冗余的（由我们的重写产生），则跳过
                    pass

    return count


def fix_paddle_onnx(model, verbose=True):
    """对 Paddle 导出的 ONNX 模型执行所有 RKNN NPU 兼容性修复。

    修复流程：
    1. 运行 shape inference 填充中间张量形状
    2. Pass 1: Expand → Tile（需要形状信息）
    3. Pass 2: 浮点整除链 → 整数 Div + Mod
    4. 清理未使用的初始化器
    5. 验证模型合法性

    参数:
        model: ONNX 模型对象
        verbose: 是否输出详细日志

    返回:
        修复后的 ONNX 模型（原地修改）
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    graph = model.graph

    logger.info(f"输入模型: {len(graph.node)} 个节点")

    # 运行形状推断以填充中间张量形状信息
    try:
        model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph
        logger.info("形状推断完成")
    except Exception as e:
        logger.warning(f"形状推断失败: {e}")

    # Pass 1: Expand → Tile（需要形状信息）
    n1 = rewrite_expand_to_tile(graph)
    logger.info(f"Pass 1 (Expand→Tile): {n1} 处替换")

    # Pass 2: Floor(Div(Cast)) → 整数 Div + Mod
    n2 = rewrite_floordiv_to_intdiv_and_mod(graph)
    logger.info(f"Pass 2 (FloorDiv→IntDiv+Mod): {n2} 处替换")

    # 清理未使用的初始化器
    used_inputs = set()
    for node in graph.node:
        used_inputs.update(node.input)
    removed_init = 0
    for init in list(graph.initializer):
        if init.name not in used_inputs:
            _remove_initializer(graph, init.name)
            removed_init += 1
    if removed_init:
        logger.info(f"移除了 {removed_init} 个未使用的初始化器")

    # 验证
    try:
        onnx.checker.check_model(model)
        logger.info("模型验证通过")
    except onnx.checker.ValidationError as e:
        logger.warning(f"模型验证警告（可能无害）: {e}")

    logger.info(f"输出模型: {len(graph.node)} 个节点")

    return model


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="修复 Paddle 导出的 ONNX 后处理子图，使其兼容 RKNN NPU")
    parser.add_argument("input", help="输入 ONNX 路径")
    parser.add_argument("output", nargs="?", help="输出 ONNX 路径；不填则就地覆写输入文件")
    return parser.parse_args()


def main():
    """命令行入口：读取 ONNX 文件 → 应用修复 → 保存。"""
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path

    model = onnx.load(input_path)
    model = fix_paddle_onnx(model)

    onnx.save(model, output_path)
    print(f"已保存修复模型至 {output_path}")
    print(f"  输入大小:  {os.path.getsize(input_path) / 1024 / 1024:.1f} MB")
    print(f"  输出大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
