# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 模型导出引擎 Exporter，支持将 YOLO Paddle 模型导出为多种部署格式。
@details
主要导出格式及状态：
- ONNX（生产主力）：支持 FP32/FP16，含 end2end 和双输出两种格式
- TensorRT Engine：通过 ONNX 构建，不依赖 PyTorch
- RKNN（瑞芯微 NPU）：依赖 rknn-toolkit2

ONNX 导出特别注意事项：
- Paddle ONNX 使用 Expand+Floor 算子（标准 ONNX 用 Tile+Mod），需 fix_paddle_onnx 修复
- 双输出模式：移除输出端 sigmoid，让 NPU 输出原始 logits，CPU 侧用 float32 sigmoid
  （避免 INT8 量化后 sigmoid 输出全零的问题）
"""

from __future__ import annotations
import sys
import paddle

import os


from ddyolo26.paddle_utils import *

"""
导出 YOLO Paddle 模型。

当前 Paddle 分支仅支持以下导出格式：

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PaddlePaddle (dynamic)  | -                         | yolov8n.pdparams
ONNX                    | `onnx`                    | yolov8n.onnx
TensorRT                | `engine`                  | yolov8n.engine
RKNN (Rockchip NPU)     | `rknn`                    | yolov8n.rknn

依赖:
    $ pip install onnx onnxruntime onnxslim paddle2onnx

Python:
    from ddyolo26 import YOLO
    model = YOLO('yolov8n.pdparams')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolov8n.pdparams format=onnx

推理:
    $ yolo predict model=yolov8n.pdparams    # PaddlePaddle
                         yolov8n.onnx        # ONNX Runtime 或 OpenCV DNN（dnn=True）
                         yolov8n.rknn        # RKNN（Rockchip NPU）
"""
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from ddyolo26 import __version__
from ddyolo26.cfg import TASK2DATA, get_cfg
from ddyolo26.data import build_dataloader
from ddyolo26.data.dataset import YOLODataset
from ddyolo26.data.utils import check_det_dataset
from ddyolo26.nn.autobackend import check_class_names, default_class_names
from ddyolo26.nn.modules import C2f, Detect
from ddyolo26.nn.tasks import DetectionModel, SegmentationModel
from ddyolo26.utils import (
    DEFAULT_CFG,
    IS_COLAB,
    LOGGER,
    RKNN_CHIPS,
    YAML,
    callbacks,
    colorstr,
    get_default_args,
    is_dgx,
    is_jetson,
)
from ddyolo26.utils.checks import check_imgsz, check_requirements, check_tensorrt, check_version
from ddyolo26.utils.export import onnx2engine, paddle2onnx_export
from ddyolo26.utils.files import file_size
from ddyolo26.utils.nms import PaddleNMS
from ddyolo26.utils.ops import Profile
from ddyolo26.utils.patches import arange_patch
from ddyolo26.utils.runtime import select_device


def export_formats():
    """返回 PaddleYOLO-RKNN 支持的导出格式字典。"""
    x = [
        ["PaddlePaddle", "-", ".pdparams", True, True, []],
        [
            "ONNX",
            "onnx",
            ".onnx",
            True,
            True,
            ["batch", "dynamic", "half", "opset", "simplify", "nms", "dual_raw"],
        ],
        [
            "TensorRT",
            "engine",
            ".engine",
            False,
            True,
            ["batch", "dynamic", "half", "int8", "simplify", "nms", "fraction"],
        ],
        ["RKNN", "rknn", "_rknn_model", False, False, ["batch", "name"]],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x)))


def best_onnx_opset(onnx, cuda=False) -> int:
    """返回当前环境中可用的最佳 ONNX opset。

    对 paddle2onnx 导出，使用已安装 onnx 库支持的最大 opset，并在 CUDA 场景保留少量安全余量。
    """
    opset = onnx.defs.onnx_opset_version() - 1
    if cuda:
        opset -= 2
    return min(opset, onnx.defs.onnx_opset_version())


def validate_args(format, passed_args, valid_args):
    """根据导出格式校验参数。

    参数:
        format (str): 导出格式。
        passed_args (SimpleNamespace): 导出时使用的参数。
        valid_args (list): 该格式允许的参数列表。

    异常:
        AssertionError: 使用了不支持的参数，或该格式缺少受支持参数列表时抛出。
    """
    export_args = [
        "half",
        "int8",
        "dynamic",
        "nms",
        "batch",
        "fraction",
        "opset",
        "simplify",
        "workspace",
        "name",
        "dual_raw",
    ]
    assert valid_args is not None, f"错误 ❌️ 未列出 '{format}' 的有效参数。"
    custom = {"batch": 1, "data": None, "device": None}
    default_args = get_cfg(DEFAULT_CFG, custom)
    for arg in export_args:
        not_default = getattr(passed_args, arg, None) != getattr(default_args, arg, None)
        if not_default:
            assert arg in valid_args, f"错误 ❌️ format='{format}' 不支持参数 '{arg}'"


def try_export(inner_func):
    """YOLO 导出装饰器，即 @try_export。"""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """导出模型。"""
        prefix = inner_args["prefix"]
        dt = 0.0
        try:
            with Profile() as dt:
                f = inner_func(*args, **kwargs)
            path = f if isinstance(f, (str, Path)) else f[0]
            mb = file_size(path)
            assert mb > 0.0, "输出模型大小为 0.0 MB"
            LOGGER.info(f"{prefix} 导出成功 ✅ {dt.t:.1f}s，已保存为 '{path}' ({mb:.1f} MB)")
            return f
        except Exception as e:
            LOGGER.error(f"{prefix} 导出失败 {dt.t:.1f}s: {e}")
            raise e

    return outer_func


class Exporter:
    """用于将 YOLO 模型导出到多种格式的类。

    该类把 Paddle YOLO 模型导出为支持的部署格式：ONNX、TensorRT Engine 与 RKNN。它负责格式校验、
    设备选择、模型准备，以及各格式的实际导出流程。

    属性:
        args (SimpleNamespace): 导出器配置参数。
        callbacks (dict): 各导出事件对应的回调函数字典。
        im (paddle.Tensor): 导出期间用于模型推理的输入张量。
        model (paddle.nn.Layer): 待导出的 YOLO 模型。
        file (Path): 待导出模型文件路径。
        output_shape (tuple): 模型输出张量形状。
        pretty_name (str): 用于显示的格式化模型名。
        metadata (dict): 模型元数据，包括描述、作者、版本等。
        device (paddle.CUDAPlace | paddle.CPUPlace): 模型加载设备。
        imgsz (list): 模型输入图像尺寸。

    方法:
        __call__: 处理导出流程的主方法。
        get_int8_calibration_dataloader: 构建 INT8 校准 dataloader。
        export_onnx: 导出 ONNX 格式。
        export_engine: 导出 TensorRT 格式。
        export_rknn: 导出 RKNN 格式。

    示例:
        将 YOLO26 模型导出为 ONNX
        >>> from ddyolo26.engine.exporter import Exporter
        >>> exporter = Exporter()
        >>> exporter(model="weights/yolov8/yolov8n.pdparams")  # 导出为 yolov8n.onnx

        使用指定参数导出
        >>> args = {"format": "onnx", "dynamic": True, "half": True}
        >>> exporter = Exporter(overrides=args)
        >>> exporter(model="weights/yolov8/yolov8n.pdparams")
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 Exporter。

        参数:
            cfg (str | Path | dict | SimpleNamespace, optional): 配置文件路径或配置对象。
            overrides (dict, optional): 配置覆盖项。
            _callbacks (dict, optional): 回调函数字典。
        """
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def __call__(self, model=None) -> str:
        """导出模型，并返回最终导出路径字符串。

        返回:
            (str): 导出文件或目录路径（最后一个导出产物）。
        """
        t = time.time()
        fmt = self.args.format.lower()
        if fmt in {"tensorrt", "trt"}:
            fmt = "engine"
        fmts_dict = export_formats()
        fmts = tuple(fmts_dict["Argument"][1:])
        if fmt not in fmts:
            import difflib

            matches = difflib.get_close_matches(fmt, fmts, n=1, cutoff=0.6)
            if not matches:
                msg = "当前 Paddle 分支不支持 format='pt'。" if fmt == "pt" else f"导出格式 format='{fmt}' 无效。"
                raise ValueError(f"{msg} 有效格式为 {fmts}")
            LOGGER.warning(f"导出格式 format='{fmt}' 无效，已更新为 format='{matches[0]}'")
            fmt = matches[0]
        flags = [(x == fmt) for x in fmts]
        if sum(flags) != 1:
            raise ValueError(f"导出格式 format='{fmt}' 无效。有效格式为 {fmts}")
        onnx, engine, rknn = flags
        if rknn and getattr(model, "task", "detect") == "segment":
            raise ValueError(
                "公共 YOLO.export(format='rknn') 不直接编译分割模型；"
                "请使用 export/export_seg_rknn_i8.py，先生成对应模型族的部署输出再编译 RKNN。"
            )
        dla = None
        if engine and self.args.device is None:
            LOGGER.warning("TensorRT 需要在 GPU 上导出，已自动设置 device=0")
            self.args.device = "0"
        if engine and "dla" in str(self.args.device):
            device_str = str(self.args.device)
            dla = device_str.rsplit(":", 1)[-1]
            self.args.device = "0"
            assert dla in {
                "0",
                "1",
            }, f"期望 device 为 'dla:0' 或 'dla:1'，实际为 {device_str}。"
        self.device = select_device("cpu" if self.args.device is None else self.args.device)
        fmt_keys = fmts_dict["Arguments"][flags.index(True) + 1]
        validate_args(fmt, self.args, fmt_keys)
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if rknn and model.end2end:
                raise ValueError(
                    "公共 YOLO.export(format='rknn') 不直接编译 YOLO26 end2end 模型；"
                    "请使用 export/export_det_rknn_i8.py 生成 pre_dist NMS-free 部署模型。"
                )
        if self.args.half and self.args.int8:
            LOGGER.warning("half=True 与 int8=True 互斥，已设置 half=False。")
            self.args.half = False
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)
        if self.args.optimize:
            raise ValueError("Paddle-only 导出格式不支持 optimize=True。")
        if rknn:
            if not self.args.name:
                LOGGER.warning("Rockchip RKNN 导出需要 'name' 参数指定芯片类型，当前缺失；使用默认 name='rk3588'。")
                self.args.name = "rk3588"
            self.args.name = self.args.name.lower()
            assert self.args.name in RKNN_CHIPS, (
                f"Rockchip RKNN 导出的芯片名 '{self.args.name}' 无效。有效名称为 {RKNN_CHIPS}。"
            )
        if self.args.nms:
            if getattr(model, "end2end", False):
                LOGGER.warning("end2end 模型不支持 'nms=True'，已强制设置 'nms=False'。")
                self.args.nms = False
            self.args.conf = self.args.conf or 0.25
        if (engine or self.args.nms) and self.args.dynamic and self.args.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' 模型配合 '{'nms=True' if self.args.nms else f'format={self.args.format}'}' 时需要设置最大 batch size，例如 'batch=16'"
            )
        if self.args.int8 and not self.args.data:
            self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]
            LOGGER.warning(f"INT8 导出需要 'data' 参数用于校准，当前缺失；使用默认 'data={self.args.data}'。")
        im = paddle.zeros(self.args.batch, model.yaml.get("channels", 3), *self.imgsz).to(self.device)
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )
        if file.suffix in {".yaml", ".yml"}:
            file = Path(file.name)
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.stop_gradient = not False
        model.eval()
        model.float()
        model = model.fuse()
        for m in model.modules():
            if isinstance(m, Detect):
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
                if getattr(self.args, "dual_raw", False) and isinstance(m, Detect):
                    # 原始输出模式：返回 head_dict 的 boxes/scores（分割头另含 mask_coeff/proto）。
                    # YOLO26-Seg 四输出为正式契约；YOLOv8-Seg 四输出会追加 score_sum。
                    # end2end 模型自动走 one2one head，非 end2end 走 one2many（cv2/cv3）。
                    m.export_dual_raw = True
                if getattr(self.args, "export_raw_one2one", False) and isinstance(m, Detect):
                    # 原始单输出模式：返回 _inference() 的 [B, 4+nc(+nm), A] 张量（+proto）。
                    # 与 ultralytics e2e / yolov8_raw 契约等价，绕过 _postprocess_export。
                    m.export_raw_one2one = True
                anchors = sum(int(self.imgsz[0] / s) * int(self.imgsz[1] / s) for s in model.stride.tolist())
                m.max_det = min(self.args.max_det, anchors)
                m.agnostic_nms = self.args.agnostic_nms
                m.xyxy = self.args.nms
                m._feat_shape = None
                if hasattr(model, "pe") and hasattr(m, "fuse"):
                    m.fuse(model.pe.to(self.device))
            elif isinstance(m, C2f):
                m.forward = m.forward_split
        y = None
        for _ in range(2):
            y = NMSModel(model, self.args)(im) if self.args.nms else model(im)
        if self.args.half and onnx and self.device.type != "cpu":
            im, model = im.half(), model.half()
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = (
            tuple(y.shape)
            if isinstance(y, paddle.Tensor)
            else tuple(tuple(x.shape if isinstance(x, paddle.Tensor) else []) for x in y)
        )
        self.pretty_name = Path(self.model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")
        data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""
        description = f"PaddleYOLO-RKNN {self.pretty_name} model {f'trained on {data}' if data else ''}"
        self.metadata = {
            "description": description,
            "author": "Dengdxx <dengdx@tju.edu.cn>",
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "GNU AGPL-3.0",
            "docs": "https://www.gnu.org/licenses/agpl-3.0.html",
            "stride": int(max(model.stride)),
            "task": model.task,
            "batch": self.args.batch,
            "imgsz": self.imgsz,
            "names": model.names,
            "args": {k: v for k, v in self.args if k in fmt_keys},
            "channels": model.yaml.get("channels", 3),
            "end2end": getattr(model, "end2end", False),
        }
        if dla is not None:
            self.metadata["dla"] = dla
        LOGGER.info(
            f"""
{colorstr("PaddlePaddle:")} 从 '{file}' 开始导出，输入形状 {tuple(im.shape)} BCHW，输出形状 {self.output_shape} ({file_size(file):.1f} MB)"""
        )
        self.run_callbacks("on_export_start")
        exported = []
        if onnx:
            exported.append(self.export_onnx())
        if engine:
            exported.append(self.export_engine(dla=dla))
        if rknn:
            exported.append(self.export_rknn())
        if exported:
            f = str(Path(exported[-1]))
            square = self.imgsz[0] == self.imgsz[1]
            s = (
                ""
                if square
                else f"警告 ⚠️ 非 Paddle 验证要求方形输入，'imgsz={self.imgsz}' 不可用。如需验证，请使用导出参数 'imgsz={max(self.imgsz)}'。"
            )
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
            q = "int8" if self.args.int8 else "half" if self.args.half else ""
            LOGGER.info(
                f"""
导出完成 ({time.time() - t:.1f}s)
结果保存到 {colorstr("bold", file.parent.resolve())}
推理:           yolo predict task={model.task} model={f} imgsz={imgsz} {q}
验证:           yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}
可视化:         https://netron.app"""
            )
        else:
            f = ""
        self.run_callbacks("on_export_end")
        return f

    def get_int8_calibration_dataloader(self, prefix=""):
        """构建并返回 INT8 模型校准 dataloader。"""
        LOGGER.info(f"{prefix} 正在从 'data={self.args.data}' 收集 INT8 校准图片")
        data = check_det_dataset(self.args.data)
        dataset = YOLODataset(
            data[self.args.split or "val"],
            data=data,
            fraction=self.args.fraction,
            task=self.model.task,
            imgsz=self.imgsz[0],
            augment=False,
            batch_size=self.args.batch,
        )
        n = len(dataset)
        if n < self.args.batch:
            raise ValueError(f"校准数据集 ({n} 张图片) 的图片数必须不少于 batch size ('batch={self.args.batch}')。")
        elif self.args.format == "axelera" and n < 100:
            LOGGER.warning(f"{prefix} Axelera 校准需要超过 100 张图片，当前找到 {n} 张。")
        elif self.args.format != "axelera" and n < 300:
            LOGGER.warning(f"{prefix} INT8 校准建议超过 300 张图片，当前找到 {n} 张。")
        return build_dataloader(dataset, batch=self.args.batch, workers=0, drop_last=True)

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """将 YOLO 模型导出为 ONNX 格式。"""
        requirements = ["onnx>=1.12.0,<2.0.0"]
        if self.args.simplify:
            requirements += [
                "onnxslim>=0.1.71",
                "onnxruntime" + ("-gpu" if paddle.cuda.is_available() else ""),
            ]
        check_requirements(requirements)
        import onnx

        opset = self.args.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type)
        LOGGER.info(f"\n{prefix} 开始导出，onnx {onnx.__version__} opset {opset}...")
        f = str(self.file.with_suffix(".onnx"))
        dual_raw = bool(getattr(self.args, "dual_raw", False))
        if dual_raw:
            if self.model.task == "segment":
                # 顺序与 Segment._forward_export(dual_raw) 一致：
                #   boxes / scores / mask_coeff / proto
                output_names = [
                    "dual_raw_boxes",
                    "dual_raw_scores",
                    "dual_raw_mask_coeff",
                    "dual_raw_proto",
                ]
            else:
                output_names = ["dual_raw_boxes", "dual_raw_scores"]
        else:
            output_names = ["output0", "output1"] if self.model.task == "segment" else ["output0"]
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {"images": {(0): "batch", (2): "height", (3): "width"}}
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {(0): "batch", (2): "anchors"}
                dynamic["output1"] = {
                    (0): "batch",
                    (2): "mask_height",
                    (3): "mask_width",
                }
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {(0): "batch", (2): "anchors"}
            if self.args.nms:
                dynamic["output0"].pop(2)
        # 为 paddle.jit.to_static 兼容性设置导出友好的 forward：
        # to_static 无法在 BaseModel._predict_once 的循环变量上访问自定义 .i/.f 属性，
        # 因此替换为预计算路由版本。
        export_model = NMSModel(self.model, self.args) if self.args.nms else self.model
        if hasattr(export_model, "_setup_export_forward"):
            export_model._setup_export_forward()
            export_model.forward = export_model._forward_export
        # 移到 CPU，避免 paddle.jit.save 在 GPU 序列化时出现段错误。
        export_model = export_model.cpu()
        im = self.im.cpu()
        with arange_patch(self.args):
            paddle2onnx_export(
                export_model,
                im,
                f,
                opset=opset,
                input_names=["images"],
                output_names=output_names,
                dynamic=dynamic or None,
            )
        model_onnx = onnx.load(f)
        if self.args.simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} 正在使用 onnxslim {onnxslim.__version__} 简化模型...")
                model_onnx = onnxslim.slim(model_onnx)
            except Exception as e:
                LOGGER.warning(f"{prefix} 简化器失败: {e}")
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        if getattr(model_onnx, "ir_version", 0) > 10:
            LOGGER.info(f"{prefix} 为兼容 ONNXRuntime，将 IR version {model_onnx.ir_version} 限制为 10...")
            model_onnx.ir_version = 10
        if self.args.half and self.args.format == "onnx" and self.device.type == "cpu":
            try:
                from onnxruntime.transformers import float16

                LOGGER.info(f"{prefix} 正在转换为 FP16...")
                model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
            except Exception as e:
                LOGGER.warning(f"{prefix} FP16 转换失败: {e}")
        onnx.save(model_onnx, f)
        return f

    @try_export
    def export_engine(self, dla=None, prefix=colorstr("TensorRT:")):
        """将 YOLO 模型导出为 TensorRT 格式：https://developer.nvidia.com/tensorrt。"""
        assert self.im.device.type != "cpu", "导出正在 CPU 上运行，但 TensorRT 必须使用 GPU，例如设置 'device=0'"
        f_onnx = self.export_onnx()
        if is_jetson(jetpack=7) or is_dgx():
            check_tensorrt("10.15")
        try:
            import tensorrt as trt
        except ImportError:
            check_tensorrt()
            import tensorrt as trt
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(
            trt.__version__,
            "!=10.1.0",
            msg="https://github.com/ultralytics/ultralytics/pull/14239",  # upstream TensorRT compat fix
        )
        LOGGER.info(f"\n{prefix} 开始导出，TensorRT {trt.__version__}...")
        assert Path(f_onnx).exists(), f"ONNX 文件导出失败: {f_onnx}"
        f = self.file.with_suffix(".engine")
        onnx2engine(
            f_onnx,
            f,
            self.args.workspace,
            self.args.half,
            self.args.int8,
            self.args.dynamic,
            self.im.shape,
            dla=dla,
            dataset=self.get_int8_calibration_dataloader(prefix) if self.args.int8 else None,
            metadata=self.metadata,
            verbose=self.args.verbose,
            prefix=prefix,
        )
        return f

    @try_export
    def export_rknn(self, prefix=colorstr("RKNN:")):
        """将 YOLO 模型导出为 RKNN 格式。"""
        LOGGER.info(f"\n{prefix} 开始使用 rknn-toolkit2 导出...")
        check_requirements("rknn-toolkit2")
        check_requirements("onnx<1.19.0")
        if IS_COLAB:
            import builtins

            builtins.exit = lambda: None
        from rknn.api import RKNN

        self.args.opset = min(self.args.opset or 19, 19)
        f = self.export_onnx()
        export_path = Path(f"{Path(f).stem}_rknn_model")
        export_path.mkdir(exist_ok=True)
        rknn = RKNN(verbose=False)
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=self.args.name,
        )
        rknn.load_onnx(model=f)
        rknn.build(do_quantization=False)
        rknn.export_rknn(str(export_path / f"{Path(f).stem}-{self.args.name}.rknn"))
        YAML.save(export_path / "metadata.yaml", self.metadata)
        return export_path

    def add_callback(self, event: str, callback):
        """向指定事件追加回调。"""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """执行指定事件的所有回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)


class NMSModel(paddle.nn.Module):
    """为 detect 和 segment 导出嵌入 NMS 的模型包装器。"""

    def __init__(self, model, args):
        """初始化 NMSModel。

        参数:
            model (paddle.nn.Layer): 需要包装 NMS 后处理的模型。
            args (SimpleNamespace): 导出参数。
        """
        super().__init__()
        self.model = model
        self.args = args

    def forward(self, x):
        """对 detect 和 segment 模型执行带 NMS 后处理的推理。

        参数:
            x (paddle.Tensor): 形状为 (B, C, H, W) 的预处理张量。

        返回:
            (paddle.Tensor | tuple): 形状为 (B, max_det, 4 + 2 + extra_shape) 的张量，其中 B 为 batch size；
                分割模型返回 (detections, proto) 元组。
        """
        preds = self.model(x)
        pred = preds[0] if isinstance(preds, tuple) else preds
        kwargs = dict(device=pred.device, dtype=pred.dtype)
        bs = pred.shape[0]
        pred = pred.transpose(-1, -2)
        extra_shape = pred.shape[-1] - (4 + len(self.model.names))
        if self.args.dynamic and self.args.batch > 1:
            pad = paddle.zeros(
                paddle.compat.max(paddle.tensor(self.args.batch - bs), paddle.tensor(0)),
                *pred.shape[1:],
                **kwargs,
            )
            pred = paddle.cat((pred, pad))
        boxes, scores, extras = pred.split([4, len(self.model.names), extra_shape], dim=2)
        scores, classes = scores.max(axis=-1), scores.argmax(axis=-1)
        self.args.max_det = min(pred.shape[1], self.args.max_det)
        out = paddle.zeros(
            pred.shape[0],
            self.args.max_det,
            boxes.shape[-1] + 2 + extra_shape,
            **kwargs,
        )
        for i in range(bs):
            box, cls, score, extra = boxes[i], classes[i], scores[i], extras[i]
            mask = score > self.args.conf
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            multiplier = 1 / max(len(self.model.names), 1)
            nmsbox = multiplier * (nmsbox / paddle.tensor(x.shape[2:], **kwargs)._max())
            if not self.args.agnostic_nms:
                end = 4
                cls_offset = cls.view(cls.shape[0], 1).expand(cls.shape[0], end)
                offbox = nmsbox[:, :end] + cls_offset * multiplier
                nmsbox = paddle.cat((offbox, nmsbox[:, end:]), dim=-1)
            keep = PaddleNMS.nms(
                nmsbox,
                score,
                self.args.iou,
            )[: self.args.max_det]
            dets = paddle.cat(
                [
                    box[keep],
                    score[keep].view(-1, 1),
                    cls[keep].view(-1, 1).to(out.dtype),
                    extra[keep],
                ],
                dim=-1,
            )
            pad = 0, 0, 0, self.args.max_det - dets.shape[0]
            out[i] = paddle.compat.nn.functional.pad(dets, pad)
        return (out[:bs], preds[1]) if self.model.task == "segment" else out[:bs]
