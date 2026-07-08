# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ONNX 与 TensorRT Engine 导出工具函数。
@details
提供：
- `paddle2onnx_export()`：Paddle 模型 → ONNX 的核心导出逻辑
- `onnx2engine()`：ONNX → TensorRT .engine 序列化

YOLO26 ONNX 导出时需经过 PaddleONNX 修复（Expand→Tile 等价替换），
才能正常在 RKNN 等 NPU 工具链中编译。
"""

import paddle

import json
import os
import tempfile
import shutil
from pathlib import Path

import onnx

from ddyolo26.utils import IS_JETSON, LOGGER


def _fix_onnx_io(onnx_file, input_names, output_names, dynamic):
    """重命名 ONNX I/O tensors，并设置 dynamic dimension 名称以匹配部署约定。

    paddle2onnx 可能生成 'x' 或 'save_infer_model/scale_0.tmp_0' 这类内部名称。
    这里将其重命名为标准的 'images'、'output0'、'output1' 等。
    """
    model = onnx.load(onnx_file)
    graph = model.graph

    # 构建 rename map: old_name → new_name
    rename_map = {}

    # 重命名 inputs
    for i, (inp, target_name) in enumerate(zip(graph.input, input_names)):
        if inp.name != target_name:
            rename_map[inp.name] = target_name
            inp.name = target_name

    # 重命名 outputs
    for i, (out, target_name) in enumerate(zip(graph.output, output_names)):
        if out.name != target_name:
            rename_map[out.name] = target_name
            out.name = target_name

    # 将重命名应用到所有 nodes
    if rename_map:
        for node in graph.node:
            for j, name in enumerate(node.input):
                if name in rename_map:
                    node.input[j] = rename_map[name]
            for j, name in enumerate(node.output):
                if name in rename_map:
                    node.output[j] = rename_map[name]

    # 设置 dynamic dimension 名称（例如 "batch"、"height"、"anchors"）
    if isinstance(dynamic, dict):
        all_tensors = {t.name: t for t in list(graph.input) + list(graph.output)}
        for tensor_name, axes in dynamic.items():
            tensor = all_tensors.get(tensor_name)
            if tensor is None or not tensor.type.tensor_type.HasField("shape"):
                continue
            shape = tensor.type.tensor_type.shape
            for axis_idx, dim_name in axes.items():
                if axis_idx < len(shape.dim):
                    shape.dim[axis_idx].dim_param = dim_name
                    shape.dim[axis_idx].ClearField("dim_value")

    onnx.save(model, onnx_file)


def _patch_pir_json(json_path):
    """修复 paddle2onnx bug：nearest_interp 的 scale attrs 存成 f64，但 parser 期望 f32。"""
    with open(json_path) as f:
        model = json.load(f)

    ops = model["program"]["regions"][0]["blocks"][0]["ops"]
    patched = 0
    for op in ops:
        if op.get("#") == "1.nearest_interp":
            for attr in op.get("A", []):
                if attr.get("N") == "scale":
                    for item in attr["AT"]["D"]:
                        if item.get("#") == "0.a_f64":
                            item["#"] = "0.a_f32"
                            patched += 1

    if patched:
        with open(json_path, "w") as f:
            json.dump(model, f)
        LOGGER.info(f"已 patch {patched} 个 nearest_interp scale f64→f32 attrs，以兼容 paddle2onnx")


def paddle2onnx_export(
    paddle_model: paddle.nn.Module,
    im: paddle.Tensor,
    onnx_file: str,
    opset: int = 14,
    input_names: list[str] = ["images"],
    output_names: list[str] = ["output0"],
    dynamic: (bool | dict) = False,
) -> None:
    """将 PaddlePaddle model export 为 ONNX 格式。

    手动执行 jit.save → PIR JSON patch → paddle2onnx export，以绕过 paddle2onnx PIR parser
    对 float64 attributes 的兼容性问题。
    """
    import paddle2onnx

    # 根据示例 tensor 构建 InputSpec；dynamic dimensions 使用 None
    shape = list(im.shape)
    if dynamic:
        shape[0] = None  # batch
        if isinstance(dynamic, dict):
            for name, axes in dynamic.items():
                if name == input_names[0] if input_names else "images":
                    for idx in axes:
                        if idx < len(shape):
                            shape[idx] = None
        else:
            if len(shape) == 4:
                shape[2] = None
                shape[3] = None

    input_spec = [
        paddle.static.InputSpec(shape=shape, dtype="float32", name=input_names[0] if input_names else "images")
    ]

    if not onnx_file.endswith(".onnx"):
        onnx_file = onnx_file + ".onnx"

    paddle_model.eval()

    # 步骤 1：jit.save 得到 PIR static model
    tmp_dir = tempfile.mkdtemp()
    try:
        model_prefix = os.path.join(tmp_dir, "model")
        paddle.jit.save(paddle_model, model_prefix, input_spec)

        model_json = model_prefix + ".json"
        params_file = model_prefix + ".pdiparams"
        if not os.path.isfile(model_json):
            raise RuntimeError(f"jit.save 未生成 {model_json}")

        # 步骤 2：patch PIR JSON 以兼容 paddle2onnx
        _patch_pir_json(model_json)

        # 步骤 3：paddle2onnx export
        paddle2onnx.convert.export(
            model_json,
            params_file if os.path.isfile(params_file) else "",
            onnx_file,
            opset_version=opset,
            auto_upgrade_opset=True,
            enable_onnx_checker=True,
            enable_experimental_op=True,
            enable_optimize=True,
        )

        # 步骤 4：重命名 I/O 以匹配部署 ONNX 约定，并设置 dynamic dims
        _fix_onnx_io(onnx_file, input_names, output_names, dynamic)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def onnx2engine(
    onnx_file: str,
    engine_file: (str | None) = None,
    workspace: (int | None) = None,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dla: (int | None) = None,
    dataset=None,
    metadata: (dict | None) = None,
    verbose: bool = False,
    prefix: str = "",
) -> None:
    """将 YOLO model export 为 TensorRT engine 格式。

    参数:
        onnx_file (str): 待转换 ONNX 文件路径。
        engine_file (str | None): 生成的 TensorRT engine 文件保存路径。
        workspace (int | None): TensorRT workspace 大小，单位 GB。
        half (bool, optional): 启用 FP16 precision。
        int8 (bool, optional): 启用 INT8 precision。
        dynamic (bool, optional): 启用 dynamic input shapes。
        shape (tuple[int, int, int, int], optional): 输入 shape (batch, channels, height, width)。
        dla (int | None): 使用的 DLA core（仅 Jetson device）。
        dataset (ddyolo26.data.build.InfiniteDataLoader, optional): INT8 calibration dataset。
        metadata (dict | None): 写入 engine 文件的 metadata。
        verbose (bool, optional): 启用 verbose logging。
        prefix (str, optional): log message 前缀。

    异常:
        ValueError: 在非 Jetson device 启用 DLA，或未设置所需 precision 时抛出。
        RuntimeError: ONNX 文件无法解析时抛出。

    说明:
        TensorRT version 兼容性会在 workspace size 和 engine build 过程中处理。
        INT8 calibration 需要 dataset，并会生成 calibration cache。
        若提供 metadata，会序列化并写入 engine 文件。
    """
    import tensorrt as trt

    engine_file = engine_file or Path(onnx_file).with_suffix(".engine")
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace_bytes = int((workspace or 0) * (1 << 30))
    is_trt10 = int(trt.__version__.split(".", 1)[0]) >= 10
    if is_trt10 and workspace_bytes > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    elif workspace_bytes > 0:
        config.max_workspace_size = workspace_bytes
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    half = builder.platform_has_fast_fp16 and half
    int8 = builder.platform_has_fast_int8 and int8
    if dla is not None:
        if not IS_JETSON:
            raise ValueError("DLA 仅在 NVIDIA Jetson device 上可用")
        LOGGER.info(f"{prefix} 正在 core {dla} 上启用 DLA...")
        if not half and not int8:
            raise ValueError("DLA 要求启用 'half=True' (FP16) 或 'int8=True' (INT8)。请启用其中之一后重试。")
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_file):
        raise RuntimeError(f"加载 ONNX 文件失败: {onnx_file}")
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} 输入 "{inp.name}" shape={inp.shape} dtype={inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} 输出 "{out.name}" shape={out.shape} dtype={out.dtype}')
    if dynamic:
        profile = builder.create_optimization_profile()
        min_shape = 1, shape[1], 32, 32
        max_shape = *shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:])
        for inp in inputs:
            profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
        config.add_optimization_profile(profile)
        if int8 and not is_trt10:
            config.set_calibration_profile(profile)
    LOGGER.info(f"{prefix} 正在 build {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine: {engine_file}")
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        class EngineCalibrator(trt.IInt8Calibrator):
            """用于 TensorRT engine optimization 的自定义 INT8 calibrator。

            该 calibrator 提供 TensorRT 使用 dataset 执行 INT8 quantization calibration 所需接口，
            负责 batch 生成、cache 和 calibration algorithm 选择。

            属性:
                dataset: 用于 calibration 的 dataset。
                data_iter: calibration dataset iterator。
                algo (trt.CalibrationAlgoType): calibration algorithm 类型。
                batch (int): calibration batch size。
                cache (Path): calibration cache 保存路径。

            方法:
                get_algorithm: 获取使用的 calibration algorithm。
                get_batch_size: 获取 calibration batch size。
                get_batch: 获取下一批 calibration 数据。
                read_calibration_cache: 使用已有 cache，避免重复 calibration。
                write_calibration_cache: 将 calibration cache 写入磁盘。
            """

            def __init__(self, dataset, cache: str = "") -> None:
                """使用 dataset 和 cache path 初始化 INT8 calibrator。"""
                trt.IInt8Calibrator.__init__(self)
                self.dataset = dataset
                self.data_iter = iter(dataset)
                self.algo = (
                    trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                    if dla is not None
                    else trt.CalibrationAlgoType.MINMAX_CALIBRATION
                )
                self.batch = dataset.batch_size
                self.cache = Path(cache)

            def get_algorithm(self) -> trt.CalibrationAlgoType:
                """获取使用的 calibration algorithm。"""
                return self.algo

            def get_batch_size(self) -> int:
                """获取 calibration 使用的 batch size。"""
                return self.batch or 1

            def get_batch(self, names) -> list[int] | None:
                """获取 calibration 使用的下一批数据，以 device memory pointer 列表返回。"""
                try:
                    im0s = next(self.data_iter)["img"] / 255.0
                    im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                    return [int(im0s.data_ptr())]
                except StopIteration:
                    return None

            def read_calibration_cache(self) -> bytes | None:
                """使用已有 cache 以避免重复 calibration；否则隐式返回 None。"""
                if self.cache.exists() and self.cache.suffix == ".cache":
                    return self.cache.read_bytes()

            def write_calibration_cache(self, cache: bytes) -> None:
                """将 calibration cache 写入磁盘。"""
                _ = self.cache.write_bytes(cache)

        config.int8_calibrator = EngineCalibrator(dataset=dataset, cache=str(Path(onnx_file).with_suffix(".cache")))
    elif half:
        config.set_flag(trt.BuilderFlag.FP16)
    if is_trt10:
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build 失败，请检查 logs")
        with open(engine_file, "wb") as t:
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            t.write(engine)
    else:
        with builder.build_engine(network, config) as engine, open(engine_file, "wb") as t:
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            t.write(engine.serialize())
