# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 多后端自动推理：AutoBackend，按格式选择 Paddle/ONNX/RKNN 运行时。
@details
AutoBackend 在推理时自动检测模型文件格式并选择合适的后端：
- `.pdparams` / `*_paddle.pt`：PaddlePaddle 动态图
- `.onnx`：ONNX Runtime（PC 端 CPU/GPU 推理）
- `.rknn`：Rockchip RKNN NPU（RK3588 等）
- 普通 `.pt`：拒绝加载，提示使用 Paddle 版本

为基于 ONNX Runtime 的消费端提供兼容接口，
输出格式同 e2e ONNX：`[1, 300, 6]` 张量（x1,y1,x2,y2,conf,cls）。
"""

import paddle

import ast
import json
import os
import platform
import zipfile
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from PIL import Image

from ddyolo26.utils import ARM64, IS_JETSON, LINUX, LOGGER, PYTHON_VERSION, ROOT, YAML, is_jetson
from ddyolo26.utils.checks import check_requirements, check_suffix, check_version, check_yaml, is_rockchip
from ddyolo26.utils.downloads import attempt_download_asset, is_url
from ddyolo26.utils.nms import non_max_suppression


def check_class_names(names: (list | dict)) -> dict[int, str]:
    """检查类别名，并在需要时转换为 dict 格式。

    参数:
        names (list | dict): list 或 dict 格式的类别名。

    返回:
        (dict): 使用整数键和字符串值表示的类别名字典。

    异常:
        KeyError: 类别索引对数据集类别数无效时抛出。
    """
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n} 类数据集要求类别索引位于 0-{n - 1}，但数据集 YAML 中定义了无效类别索引 {min(names.keys())}-{max(names.keys())}。"
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):
            names_map = YAML.load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data: (str | Path | None) = None) -> dict[int, str]:
    """从 YAML 文件加载类别名，或返回数字类别名。

    参数:
        data (str | Path, optional): 包含类别名的 YAML 文件路径。

    返回:
        (dict): 类别索引到类别名的映射字典。
    """
    if data:
        try:
            return YAML.load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}


class AutoBackend(paddle.nn.Module):
    """为 PaddleYOLO-RKNN 模型推理动态选择后端。

    当前 Paddle 分支中，只有以下后端经过主动支持和测试；其它上游分支仅为兼容 Ultralytics 公共 API
    保留，不保证可用：

        支持格式与命名约定:
            | Format                | File Suffix       |
            | --------------------- | ----------------- |
            | PaddlePaddle          | *.pdparams / *_paddle.pt |
            | ONNX Runtime          | *.onnx            |
            | ONNX OpenCV DNN       | *.onnx (dnn=True) |
            | RKNN (Rockchip NPU)   | *.rknn            |

    属性:
        model (paddle.nn.Layer): 已加载的 YOLO 模型。
        device (str): 模型加载设备（CPU 或 GPU）。
        task (str): 模型任务类型（detect、segment）。
        names (dict): 模型可检测类别名字典。
        stride (int): 模型 stride，YOLO 模型通常为 32。
        fp16 (bool): 模型是否使用半精度（FP16）推理。
        nhwc (bool): 模型是否期望 NHWC 输入格式而非 NCHW。
        pt (bool): 模型是否为 Paddle 模型。
        onnx (bool): 模型是否为 ONNX 模型。
        rknn (bool): 模型是否为 RKNN 模型。

    方法:
        forward: 对输入图像运行推理。
        from_numpy: 将 NumPy 数组转换为模型设备上的张量。
        warmup: 使用虚拟输入预热模型。
        _model_type: 从文件路径判断模型类型。

    示例:
        >>> model = AutoBackend(model="weights/yolov8/yolov8n.pdparams", device="cuda")
        >>> results = model(img)
    """

    @paddle.no_grad()
    def __init__(
        self,
        model: (str | paddle.nn.Module) = "weights/yolov8/yolov8n.pdparams",
        device: str = "cpu",
        dnn: bool = False,
        data: (str | Path | None) = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """初始化用于推理的 AutoBackend。

        参数:
            model (str | paddle.nn.Layer): 模型权重文件路径或模块实例。
            device (str): 模型运行设备。
            dnn (bool): ONNX 推理是否使用 OpenCV DNN 模块。
            data (str | Path, optional): 额外 data.yaml 路径，用于读取类别名。
            fp16 (bool): 是否启用半精度推理，仅部分后端支持。
            fuse (bool): 是否融合 Conv2D + BatchNorm 层以优化推理。
            verbose (bool): 是否启用详细日志。
        """
        super().__init__()
        nn_module = isinstance(model, paddle.nn.Module)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            pd_model,
            mnn,
            ncnn,
            imx,
            rknn,
            pte,
            axelera,
            triton,
        ) = self._model_type("" if nn_module else model)
        fp16 &= pt or onnx or engine or nn_module
        nhwc = rknn
        stride, ch = 32, 3
        end2end, dynamic = False, False
        metadata, task = None, None
        cuda = isinstance(device, paddle.device.Device) and paddle.cuda.is_available() and device.type != "cpu"
        if cuda and not any([nn_module, pt, engine, onnx, pd_model]):
            device = paddle.device("cpu")
            cuda = False
        w = attempt_download_asset(model) if pt else model
        if nn_module or pt:
            if nn_module:
                pt = True
                if fuse:
                    if IS_JETSON and is_jetson(jetpack=5):
                        model = model.to(device)
                    model = model.fuse(verbose=verbose)
                model = model.to(device)
            else:
                from ddyolo26.nn.tasks import load_checkpoint

                model, _ = load_checkpoint(model, device=device, fuse=fuse)
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape
            stride = max(int(model.stride.max()), 32)
            names = model.module.names if hasattr(model, "module") else model.names
            model.half() if fp16 else model.float()
            ch = model.yaml.get("channels", 3)
            for p in model.parameters():
                p.stop_gradient = not False
            self.model = model
            end2end = getattr(model, "end2end", False)
        elif dnn:
            LOGGER.info(f"正在加载 {w} 用于 ONNX OpenCV DNN 推理...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:
            LOGGER.info(f"正在加载 {w} 用于 ONNX Runtime 推理...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            available = onnxruntime.get_available_providers()
            if cuda and "CUDAExecutionProvider" in available:
                providers = [
                    ("CUDAExecutionProvider", {"device_id": device.index}),
                    "CPUExecutionProvider",
                ]
            elif device.type == "mps" and "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                if cuda:
                    LOGGER.warning("请求了 CUDA，但 CUDAExecutionProvider 不可用。改用 CPU...")
                    device, cuda = paddle.device("cpu"), False
            LOGGER.info(
                f"使用 ONNX Runtime {onnxruntime.__version__}，provider={providers[0] if isinstance(providers[0], str) else providers[0][0]}"
            )
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map
            dynamic = isinstance(session.get_outputs()[0].shape[0], str)
            fp16 = "float16" in session.get_inputs()[0].type
            use_io_binding = not dynamic and cuda
            if use_io_binding:
                io = session.io_binding()
                bindings = []
                for output in session.get_outputs():
                    out_fp16 = "float16" in output.type
                    y_tensor = paddle.empty(
                        output.shape,
                        dtype=paddle.float16 if out_fp16 else paddle.float32,
                    ).to(device)
                    io.bind_output(
                        name=output.name,
                        device_type=device.type,
                        device_id=device.index if cuda else 0,
                        element_type=np.float16 if out_fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    bindings.append(y_tensor)
        elif xml:
            LOGGER.info(f"正在加载 {w} 用于 OpenVINO 推理...")
            check_requirements("openvino>=2024.0.0")
            import openvino as ov

            core = ov.Core()
            device_name = "AUTO"
            if isinstance(device, str) and device.startswith("intel"):
                device_name = device.split(":")[1].upper()
                device = paddle.device("cpu")
                if device_name not in core.available_devices:
                    LOGGER.warning(f"OpenVINO 设备 '{device_name}' 不可用，改用 'AUTO'。")
                    device_name = "AUTO"
            w = Path(w)
            if not w.is_file():
                w = next(w.glob("*.xml"))
            ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))
            metadata = w.parent / "metadata.yaml"
            if metadata.exists():
                metadata = YAML.load(metadata)
                batch = metadata["batch"]
                dynamic = metadata.get("args", {}).get("dynamic", dynamic)
            inference_mode = "CUMULATIVE_THROUGHPUT" if dynamic and batch > 1 else "LATENCY"
            ov_compiled_model = core.compile_model(
                ov_model,
                device_name=device_name,
                config={"PERFORMANCE_HINT": inference_mode},
            )
            LOGGER.info(
                f"使用 OpenVINO {inference_mode} 模式，在 {', '.join(ov_compiled_model.get_property('EXECUTION_DEVICES'))} 上执行 batch={batch} 推理..."
            )
            input_name = ov_compiled_model.input().get_any_name()
        elif engine:
            LOGGER.info(f"正在加载 {w} 用于 TensorRT 推理...")
            if IS_JETSON and check_version(PYTHON_VERSION, "<=3.8.10"):
                check_requirements("numpy==1.23.5")
            try:
                import tensorrt as trt
            except ImportError:
                if LINUX:
                    check_requirements("tensorrt>7.0.0,!=10.1.0")
                import tensorrt as trt
            check_version(trt.__version__, ">=7.0.0", hard=True)
            check_version(
                trt.__version__,
                "!=10.1.0",
                msg="https://github.com/ultralytics/ultralytics/pull/14239",  # upstream TensorRT compat fix
            )
            if device.type == "cpu":
                device = paddle.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")
                    metadata = json.loads(f.read(meta_len).decode("utf-8"))
                    dla = metadata.get("dla", None)
                    if dla is not None:
                        runtime.DLA_core = int(dla)
                except UnicodeDecodeError:
                    f.seek(0)
                model = runtime.deserialize_cuda_engine(f.read())
            try:
                context = model.create_execution_context()
            except Exception as e:
                LOGGER.error(
                    f"""TensorRT 模型由不同于当前 {trt.__version__} 的版本导出
"""
                )
                raise e
            bindings = OrderedDict()
            output_names = []
            fp16 = False
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    shape = tuple(model.get_tensor_shape(name))
                    profile_shape = tuple(model.get_tensor_profile_shape(name, 0)[2]) if is_input else None
                else:
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    shape = tuple(model.get_binding_shape(i))
                    profile_shape = tuple(model.get_profile_shape(0, i)[1]) if is_input else None
                if is_input:
                    if -1 in shape:
                        dynamic = True
                        if is_trt10:
                            context.set_input_shape(name, profile_shape)
                        else:
                            context.set_binding_shape(i, profile_shape)
                    if dtype == np.float16:
                        fp16 = True
                else:
                    output_names.append(name)
                shape = tuple(context.get_tensor_shape(name)) if is_trt10 else tuple(context.get_binding_shape(i))
                im = paddle.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        elif coreml:
            check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
            LOGGER.info(f"正在加载 {w} 用于 CoreML 推理...")
            import coremltools as ct

            model = ct.models.MLModel(w)
            dynamic = model.get_spec().description.input[0].type.HasField("multiArrayType")
            metadata = dict(model.user_defined_metadata)
        elif saved_model:
            LOGGER.info(f"正在加载 {w} 用于 TensorFlow SavedModel 推理...")
            import tensorflow as tf

            model = tf.saved_model.load(w)
            metadata = Path(w) / "metadata.yaml"
        elif pb:
            LOGGER.info(f"正在加载 {w} 用于 TensorFlow GraphDef 推理...")
            import tensorflow as tf

            from ddyolo26.utils.export.tensorflow import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """包装 frozen graph 以用于部署。"""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])
                ge = x.graph.as_graph_element
                return x.prune(
                    tf.nest.map_structure(ge, inputs),
                    tf.nest.map_structure(ge, outputs),
                )

            gd = tf.Graph().as_graph_def()
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
            try:
                metadata = next(Path(w).resolve().parent.rglob(f"{Path(w).stem}_saved_model*/metadata.yaml"))
            except StopIteration:
                pass
        elif tflite or edgetpu:
            try:
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:
                device = device[3:] if str(device).startswith("tpu") else ":0"
                LOGGER.info(f"正在设备 {device[1:]} 上加载 {w} 用于 TensorFlow Lite Edge TPU 推理...")
                delegate = {
                    "Linux": "libedgetpu.so.1",
                    "Darwin": "libedgetpu.1.dylib",
                    "Windows": "edgetpu.dll",
                }[platform.system()]
                interpreter = Interpreter(
                    model_path=w,
                    experimental_delegates=[load_delegate(delegate, options={"device": device})],
                )
                device = "cpu"
            else:
                LOGGER.info(f"正在加载 {w} 用于 TensorFlow Lite 推理...")
                interpreter = Interpreter(model_path=w)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            try:
                with zipfile.ZipFile(w, "r") as zf:
                    name = zf.namelist()[0]
                    contents = zf.read(name).decode("utf-8")
                    if name == "metadata.json":
                        metadata = json.loads(contents)
                    else:
                        metadata = ast.literal_eval(contents)
            except (zipfile.BadZipFile, SyntaxError, ValueError, json.JSONDecodeError):
                pass
        elif tfjs:
            raise NotImplementedError("PaddleYOLO-RKNN 当前不支持 TF.js 推理。")
        elif pd_model:
            LOGGER.info(f"正在加载 {w} 用于 PaddlePaddle 推理...")
            check_requirements(
                "paddlepaddle-gpu>=3.0.0,!=3.3.0"
                if paddle.cuda.is_available()
                else "paddlepaddle==3.0.0"
                if ARM64
                else "paddlepaddle>=3.0.0,!=3.3.0"
            )
            import paddle.inference as pdi

            w = Path(w)
            model_file, params_file = None, None
            if w.is_dir():
                model_file = next(w.rglob("*.json"), None)
                params_file = next(w.rglob("*.pdiparams"), None)
            elif w.suffix == ".pdiparams":
                model_file = w.with_name("model.json")
                params_file = w
            if not (model_file and params_file and model_file.is_file() and params_file.is_file()):
                raise FileNotFoundError(f"在 {w} 中未找到 Paddle 模型。需要同时存在 .json 和 .pdiparams 文件。")
            config = pdi.Config(str(model_file), str(params_file))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
            metadata = w / "metadata.yaml"
        elif mnn:
            LOGGER.info(f"正在加载 {w} 用于 MNN 推理...")
            check_requirements("MNN")
            pass
            import MNN

            config = {
                "precision": "low",
                "backend": "CPU",
                "numThread": (os.cpu_count() + 1) // 2,
            }
            rt = MNN.nn.create_runtime_manager((config,))
            net = MNN.nn.load_module_from_file(w, [], [], runtime_manager=rt, rearrange=True)

            def paddle_to_mnn(x):
                return MNN.expr.const(x.data_ptr(), x.shape)

            metadata = json.loads(net.get_info()["bizCode"])
        elif ncnn:
            LOGGER.info(f"正在加载 {w} 用于 NCNN 推理...")
            check_requirements("ncnn", cmds="--no-deps")
            import ncnn as pyncnn

            net = pyncnn.Net()
            if isinstance(cuda, paddle.device.Device):
                net.opt.use_vulkan_compute = cuda
            elif isinstance(device, str) and device.startswith("vulkan"):
                net.opt.use_vulkan_compute = True
                net.set_vulkan_device(int(device.split(":")[1]))
                device = paddle.device("cpu")
            w = Path(w)
            if not w.is_file():
                w = next(w.glob("*.param"))
            net.load_param(str(w))
            net.load_model(str(w.with_suffix(".bin")))
            metadata = w.parent / "metadata.yaml"
        elif triton:
            check_requirements("tritonclient[all]")
            from ddyolo26.utils.triton import TritonRemoteModel

            model = TritonRemoteModel(w)
            metadata = model.metadata
        elif rknn:
            if not is_rockchip():
                raise OSError("RKNN 推理仅支持 Rockchip 设备。")
            LOGGER.info(f"正在加载 {w} 用于 RKNN 推理...")
            check_requirements("rknn-toolkit-lite2")
            from rknnlite.api import RKNNLite

            w = Path(w)
            if not w.is_file():
                w = next(w.rglob("*.rknn"))
            rknn_model = RKNNLite()
            rknn_model.load_rknn(str(w))
            rknn_model.init_runtime()
            metadata = w.parent / "metadata.yaml"
        elif axelera:
            pass
            if not os.environ.get("AXELERA_RUNTIME_DIR"):
                LOGGER.warning(
                    """Axelera 运行时环境未激活。
请运行: source /opt/axelera/sdk/latest/axelera_activate.sh

如果仍失败，请检查驱动安装: https://www.axelera.ai/"""  # Axelera 官方站点
                )
            try:
                from axelera.runtime import op
            except ImportError:
                check_requirements(
                    "axelera_runtime2==0.1.2",
                    cmds="--extra-index-url https://software.axelera.ai/artifactory/axelera-runtime-pypi",
                )
            from axelera.runtime import op

            w = Path(w)
            if (found := next(w.rglob("*.axm"), None)) is None:
                raise FileNotFoundError(f"在以下路径未找到 .axm 文件: {w}")
            ax_model = op.load(str(found))
            metadata = found.parent / "metadata.yaml"
        else:
            from ddyolo26.engine.exporter import export_formats

            raise TypeError(
                f"""model='{w}' 不是受支持的模型格式。PaddleYOLO-RKNN 支持: {export_formats()["Format"]}
支持格式请参见 PaddleYOLO-RKNN 仓库 README。"""
            )
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = YAML.load(metadata)
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch", "channels"}:
                    metadata[k] = int(v)
                elif k in {
                    "imgsz",
                    "names",
                    "kpt_shape",
                    "kpt_names",
                    "args",
                    "end2end",
                } and isinstance(v, str):
                    metadata[k] = ast.literal_eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
            kpt_names = metadata.get("kpt_names")
            end2end = metadata.get("end2end", False) or metadata.get("args", {}).get("nms", False)
            dynamic = metadata.get("args", {}).get("dynamic", dynamic)
            ch = metadata.get("channels", 3)
        elif not (pt or triton or nn_module):
            LOGGER.warning(f"未找到 'model={w}' 的元数据")
        if "names" not in locals():
            names = default_class_names(data)
        names = check_class_names(names)
        self.__dict__.update(locals())

    def forward(
        self,
        im: paddle.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed: (list | None) = None,
        **kwargs: Any,
    ) -> paddle.Tensor | list[paddle.Tensor]:
        """在 AutoBackend 模型上运行推理。

        参数:
            im (paddle.Tensor): 用于推理的图像张量。
            augment (bool): 推理时是否执行数据增强。
            visualize (bool): 是否可视化输出预测。
            embed (list, optional): 需要返回嵌入特征的层索引列表。
            **kwargs (Any): 额外模型配置关键字参数。

        返回:
            (paddle.Tensor | list[paddle.Tensor]): 模型的原始输出张量。
        """
        _b, _ch, h, w = im.shape
        if self.fp16 and im.dtype != paddle.float16:
            im = im.half()
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)
        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed, **kwargs)
        elif self.dnn:
            im = im.cpu().numpy()
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx or self.imx:
            if self.use_io_binding:
                if not self.cuda:
                    im = im.cpu()
                self.io.bind_input(
                    name="images",
                    device_type=im.device.type,
                    device_id=im.device.index if im.device.type == "cuda" else 0,
                    element_type=np.float16 if self.fp16 else np.float32,
                    shape=tuple(im.shape),
                    buffer_ptr=im.data_ptr(),
                )
                self.session.run_with_iobinding(self.io)
                y = self.bindings
            else:
                im = im.cpu().numpy()
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            if self.imx:
                if self.task == "detect":
                    y = np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)
                elif self.task == "pose":
                    y = np.concatenate(
                        [y[0], y[1][:, :, None], y[2][:, :, None], y[3]],
                        axis=-1,
                        dtype=y[0].dtype,
                    )
                elif self.task == "segment":
                    y = (
                        np.concatenate(
                            [y[0], y[1][:, :, None], y[2][:, :, None], y[3]],
                            axis=-1,
                            dtype=y[0].dtype,
                        ),
                        y[4],
                    )
        elif self.xml:
            im = im.cpu().numpy()
            if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:
                n = im.shape[0]
                results = [None] * n

                def callback(request, userdata):
                    """根据 userdata 索引把结果放入预分配列表。"""
                    results[userdata] = request.results

                async_queue = self.ov.AsyncInferQueue(self.ov_compiled_model)
                async_queue.set_callback(callback)
                for i in range(n):
                    async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)
                async_queue.wait_all()
                y = [list(r.values()) for r in results]
                y = [np.concatenate(x) for x in zip(*y)]
            else:
                y = list(self.ov_compiled_model(im).values())
        elif self.engine:
            if self.dynamic and im.shape != self.bindings["images"].shape:
                if self.is_trt10:
                    self.context.set_input_shape("images", im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                else:
                    i = self.model.get_binding_index("images")
                    self.context.set_binding_shape(i, im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"输入尺寸 {im.shape} {'大于' if self.dynamic else '不等于'} 模型最大尺寸 {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:
            im = im.cpu().numpy()
            if self.dynamic:
                im = im.transpose(0, 3, 1, 2)
            else:
                im = Image.fromarray((im[0] * 255).astype("uint8"))
            y = self.model.predict({"image": im})
            if "confidence" in y:
                from ddyolo26.utils.ops import xywh2xyxy

                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])
                cls = y["confidence"].argmax(1, keepdims=True)
                y = np.concatenate((box, np.take_along_axis(y["confidence"], cls, axis=1), cls), 1)[None]
            else:
                y = list(y.values())
            if len(y) == 2 and len(y[1].shape) != 4:
                y = list(reversed(y))
        elif self.pd_model:
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.mnn:
            input_var = self.paddle_to_mnn(im)
            output_var = self.net.onForward([input_var])
            y = [x.read() for x in output_var]
        elif self.ncnn:
            mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
            with self.net.create_extractor() as ex:
                ex.input(self.net.input_names()[0], mat_in)
                y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]
        elif self.triton:
            im = im.cpu().numpy()
            y = self.model(im)
        elif self.rknn:
            im = (im.cpu().numpy() * 255).astype("uint8")
            im = im if isinstance(im, (list, tuple)) else [im]
            y = self.rknn_model.inference(inputs=im)
        elif self.axelera:
            y = self.ax_model(im.cpu())
        else:
            im = im.cpu().numpy()
            if self.saved_model:
                y = self.model.serving_default(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:
                y = self.frozen_func(x=self.tf.constant(im))
            else:
                details = self.input_details[0]
                is_int = details["dtype"] in {np.int8, np.int16}
                if is_int:
                    scale, zero_point = details["quantization"]
                    im = (im / scale + zero_point).astype(details["dtype"])
                self.interpreter.set_tensor(details["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if is_int:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale
                    if x.ndim == 3:
                        if x.shape[-1] == 6 or self.end2end:
                            x[:, :, [0, 2]] *= w
                            x[:, :, [1, 3]] *= h
                            if self.task == "pose":
                                x[:, :, 6::3] *= w
                                x[:, :, 7::3] *= h
                        else:
                            x[:, [0, 2]] *= w
                            x[:, [1, 3]] *= h
                            if self.task == "pose":
                                x[:, 5::3] *= w
                                x[:, 6::3] *= h
                    y.append(x)
            if self.task == "segment":
                if len(y[1].shape) != 4:
                    y = list(reversed(y))
                if y[1].shape[-1] == 6:
                    y = [y[1]]
                else:
                    y[1] = np.transpose(y[1], (0, 3, 1, 2))
            y = [(x if isinstance(x, np.ndarray) else x.numpy()) for x in y]
        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):
                nc = y[0].shape[1] - y[1].shape[1] - 4
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x: (np.ndarray | paddle.Tensor)) -> paddle.Tensor:
        """将 NumPy 数组转换为模型设备上的 Paddle 张量。

        参数:
            x (np.ndarray | paddle.Tensor): 输入数组或张量。

        返回:
            (paddle.Tensor): 位于 `self.device` 上的张量。
        """
        return paddle.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz: tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        """使用虚拟输入执行前向传播来预热模型。

        参数:
            imgsz (tuple[int, int, int, int]): 虚拟输入形状，格式为 (batch, channels, height, width)。
        """
        warmup_types = (
            self.pt,
            self.onnx,
            self.engine,
            self.saved_model,
            self.pb,
            self.triton,
            self.nn_module,
        )
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = paddle.empty(
                *imgsz,
                dtype=paddle.float16 if self.fp16 else paddle.float32,
                device=self.device,
            )
            for _ in range(2 if self.jit else 1):
                self.forward(im)
                warmup_boxes = paddle.rand(1, 84, 16, device=self.device)
                warmup_boxes[:, :4] *= imgsz[-1]
                non_max_suppression(warmup_boxes)

    @staticmethod
    def _model_type(p: str = "path/to/model.pdparams") -> list[bool]:
        """接收模型文件路径并返回模型类型。

        参数:
            p (str): 模型文件路径。

        返回:
            (list[bool]): 表示模型类型的一组布尔值。

        示例:
            >>> types = AutoBackend._model_type("path/to/model.onnx")
            >>> assert types[2]  # onnx
        """
        sf = ["_paddle.pt", ".pdparams", ".onnx", ".engine", "_paddle_model", ".rknn", "_rknn_model"]
        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, sf)
        name = Path(p).name
        is_paddle_ckpt = name.endswith(".pdparams") or name.endswith("_paddle.pt")
        is_onnx = name.endswith(".onnx")
        is_engine = name.endswith(".engine")
        is_paddle_static = "_paddle_model" in name
        is_rknn = name.endswith(".rknn") or "_rknn_model" in name
        types = [
            is_paddle_ckpt,  # pt: Paddle dynamic checkpoint in this fork
            False,  # jit
            is_onnx,
            False,  # xml
            is_engine,
            False,  # coreml
            False,  # saved_model
            False,  # pb
            False,  # tflite
            False,  # edgetpu
            False,  # tfjs
            is_paddle_static,
            False,  # mnn
            False,  # ncnn
            False,  # imx
            is_rknn,
            False,  # pte
            False,  # axelera
        ]
        if any(types):
            triton = False
        else:
            from urllib.parse import urlsplit

            url = urlsplit(p)
            triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}
        return [*types, triton]
