# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 模型基准测试工具：评估各导出格式的推理速度和精度。
@details
提供 `ProfileModels` 和 `benchmark()` 函数，依次测试 Paddle/ONNX/RKNN 格式
的推理延迟（ms），输出格式对比表，辅助选择最优部署方案。
"""

from __future__ import annotations
import os
import paddle


"""
Benchmark YOLO model formats 的速度与精度。

用法:
    from ddyolo26.utils.benchmarks import ProfileModels, benchmark
    ProfileModels(['yolov8n.yaml', 'yolov8s.yaml']).run()
    benchmark(model='weights/yolov8/yolov8n.pdparams', imgsz=160)

格式                    | `format=argument`         | 模型
---                     | ---                       | ---
PaddlePaddle            | -                         | yolov8n.pdparams
ONNX                    | `onnx`                    | yolov8n.onnx
TensorRT                | `engine`                  | yolov8n.engine
RKNN                    | `rknn`                    | yolov8n_rknn_model/
"""

import glob
import platform
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from ddyolo26 import YOLO
from ddyolo26.cfg import TASK2DATA, TASK2METRIC
from ddyolo26.engine.exporter import export_formats
from ddyolo26.utils import ARM64, ASSETS, IS_JETSON, LINUX, LOGGER, MACOS, TQDM, WEIGHTS_DIR
from ddyolo26.utils.checks import IS_PYTHON_3_13, check_imgsz, check_requirements, check_yolo, is_rockchip
from ddyolo26.utils.files import file_size
from ddyolo26.utils.runtime import get_cpu_info, select_device


def benchmark(
    model=WEIGHTS_DIR / "yolov8" / "yolov8n.pdparams",
    data=None,
    imgsz=160,
    half=False,
    int8=False,
    device="cpu",
    verbose=False,
    eps=0.001,
    format="",
    **kwargs,
):
    """对 YOLO model 的不同 formats 进行 speed 与 accuracy benchmark。

    参数:
        model (str | Path): model file 或 directory 的 path。
        data (str | None): 用于 evaluate 的 dataset；未传入时继承 TASK2DATA。
        imgsz (int): benchmark 使用的 image size。
        half (bool): 为 True 时对 model 使用 half-precision。
        int8 (bool): 为 True 时对 model 使用 int8-precision。
        device (str): 运行 benchmark 的 device，可为 'cpu' 或 'cuda'。
        verbose (bool | float): 若为 True 或 float，则以给定 metric 断言 benchmarks 通过。
        eps (float): 防止除零的 epsilon value。
        format (str): benchmark 使用的 export format。未提供时 benchmark 所有 formats。
        **kwargs (Any): exporter 使用的额外 keyword arguments。

    返回:
        (polars.DataFrame): 包含各 format benchmark results 的 Polars DataFrame，包括 file size、metric 和
            inference time。

    示例:
        使用 default settings benchmark YOLO model:
        >>> from ddyolo26.utils.benchmarks import benchmark
        >>> benchmark(model="weights/yolov8/yolov8n.pdparams", imgsz=640)
    """
    imgsz = check_imgsz(imgsz)
    assert imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True, "benchmark() 仅支持 square imgsz。"
    import polars as pl

    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_width_chars(-1)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_formatting("ASCII_BORDERS_ONLY_CONDENSED")
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    data = data or TASK2DATA[model.task]
    key = TASK2METRIC[model.task]
    y = []
    t0 = time.time()
    format_arg = format.lower()
    if format_arg:
        formats = frozenset(export_formats()["Argument"])
        assert format in formats, f"format 应为 {formats} 之一，但得到 '{format_arg}'。"
    for name, format, suffix, cpu, gpu, _ in zip(*export_formats().values()):
        emoji, filename = "❌", None
        try:
            if format_arg and format_arg != format:
                continue
            if format == "rknn":
                assert LINUX, "RKNN 仅支持 Linux"
                assert not is_rockchip(), "RKNN inference 仅支持 Rockchip devices"
            if "cpu" in device.type:
                assert cpu, "CPU 不支持 inference"
            if "cuda" in device.type:
                assert gpu, "GPU 不支持 inference"
            if format == "-":
                filename = model.pt_path or model.ckpt_path or model.model_name
                exported_model = deepcopy(model)
            else:
                filename = deepcopy(model).export(
                    imgsz=imgsz,
                    format=format,
                    half=half,
                    int8=int8,
                    data=data,
                    device=device,
                    verbose=False,
                    **kwargs,
                )
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), "export 失败"
            emoji = "❎"
            assert model.task != "pose" or format != "pb", "GraphDef Pose inference 不受支持"
            exported_model.predict(ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half, verbose=False)
            results = exported_model.val(
                data=data,
                batch=1,
                imgsz=imgsz,
                plots=False,
                device=device,
                half=half,
                int8=int8,
                verbose=False,
                conf=0.001,
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            fps = round(1000 / (speed + eps), 2)
            y.append(
                [
                    name,
                    "✅",
                    round(file_size(filename), 1),
                    round(metric, 4),
                    round(speed, 2),
                    fps,
                ]
            )
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"{name} benchmark 失败: {e}"
            LOGGER.error(f"{name} benchmark 失败: {e}")
            y.append([name, emoji, round(file_size(filename), 1), None, None, None])
    check_yolo(device=device)
    df = pl.DataFrame(
        y,
        schema=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)", "FPS"],
        orient="row",
    )
    df = df.with_row_index(" ", offset=1)
    df_display = df.with_columns(pl.all().cast(pl.String).fill_null("-"))
    name = model.model_name
    dt = time.time() - t0
    legend = "Benchmarks 图例:  - ✅ 成功  - ❎ Export 通过但 validation 失败  - ❌️ Export 失败"
    s = f"""
{name} 在 {data} 上以 imgsz={imgsz} 完成 benchmarks ({dt:.2f}s)
{legend}
{df_display}
"""
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)
    if verbose and isinstance(verbose, float):
        metrics = df[key].to_numpy()
        floor = verbose
        assert all(x > floor for x in metrics if not np.isnan(x)), f"Benchmark 失败: metric(s) < floor {floor}"
    return df_display


class ProfileModels:
    """用于在 ONNX 与 TensorRT 上 profile 不同 models 的 ProfileModels 类。

    该类 profile 不同 models 的 performance，并返回 model speed、FLOPs 等 results。

    属性:
        paths (list[str]): 要 profile 的 models paths。
        num_timed_runs (int): profiling 的 timed runs 数量。
        num_warmup_runs (int): profiling 前的 warmup runs 数量。
        min_time (float): profile 的最小秒数。
        imgsz (int): models 使用的 image size。
        half (bool): 是否为 TensorRT profiling 使用 FP16 half-precision。
        trt (bool): 是否使用 TensorRT 进行 profile。
        device (str): profiling 使用的 device。

    方法:
        run: 跨多种 formats profile YOLO models 的 speed 与 accuracy。
        get_files: 获取所有相关 model files。
        get_onnx_model_info: 从 ONNX model 提取 metadata。
        iterative_sigma_clipping: 应用 sigma clipping 移除 outliers。
        profile_tensorrt_model: profile TensorRT model。
        profile_onnx_model: profile ONNX model。
        generate_table_row: 生成包含 model metrics 的 table row。
        generate_results_dict: 生成 profiling results 字典。
        print_table: 打印格式化 results table。

    示例:
        profile models 并打印 results
        >>> from ddyolo26.utils.benchmarks import ProfileModels
        >>> profiler = ProfileModels(["yolo26n.yaml", "yolov8s.yaml"], imgsz=640)
        >>> profiler.run()
    """

    def __init__(
        self,
        paths: list[str],
        num_timed_runs: int = 100,
        num_warmup_runs: int = 10,
        min_time: float = 60,
        imgsz: int = 640,
        half: bool = True,
        trt: bool = True,
        device: (paddle.device | str | None) = None,
    ):
        """初始化用于 profiling models 的 ProfileModels 类。

        参数:
            paths (list[str]): 待 profiled models 的 paths 列表。
            num_timed_runs (int): profiling 的 timed runs 数量。
            num_warmup_runs (int): actual profiling 开始前的 warmup runs 数量。
            min_time (float): profile 一个 model 的最短时间（秒）。
            imgsz (int): profiling 期间使用的 image size。
            half (bool): 是否为 TensorRT profiling 使用 FP16 half-precision。
            trt (bool): 是否使用 TensorRT 进行 profile。
            device (str | str | None): profiling 使用的 device。若为 None，则自动确定。

        说明:
            ONNX 已移除 FP16 'half' argument option，因为其在 CPU 上慢于 FP32。
        """
        self.paths = paths
        self.num_timed_runs = num_timed_runs
        self.num_warmup_runs = num_warmup_runs
        self.min_time = min_time
        self.imgsz = imgsz
        self.half = half
        self.trt = trt
        self.device = device if isinstance(device, paddle.device) else select_device(device)

    def run(self):
        """跨 ONNX、TensorRT 等 formats profile YOLO models 的 speed 与 accuracy。

        返回:
            (list[dict]): 包含每个 model profiling results 的 dictionaries 列表。

        示例:
            profile models 并打印 results
            >>> from ddyolo26.utils.benchmarks import ProfileModels
            >>> profiler = ProfileModels(["yolov8n.yaml", "yolov8s.yaml"])
            >>> results = profiler.run()
        """
        files = self.get_files()
        if not files:
            LOGGER.warning("未找到匹配的 *.pt 或 *.onnx files。")
            return []
        table_rows = []
        output = []
        for file in files:
            engine_file = file.with_suffix(".engine")
            if file.suffix in {".pt", ".yaml", ".yml"}:
                model = YOLO(str(file))
                model.fuse()
                model_info = model.info(imgsz=self.imgsz)
                if self.trt and self.device.type != "cpu" and not engine_file.is_file():
                    engine_file = model.export(
                        format="engine",
                        half=self.half,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                onnx_file = model.export(format="onnx", imgsz=self.imgsz, device=self.device, verbose=False)
            elif file.suffix == ".onnx":
                model_info = self.get_onnx_model_info(file)
                onnx_file = file
            else:
                continue
            t_engine = self.profile_tensorrt_model(str(engine_file))
            t_onnx = self.profile_onnx_model(str(onnx_file))
            table_rows.append(self.generate_table_row(file.stem, t_onnx, t_engine, model_info))
            output.append(self.generate_results_dict(file.stem, t_onnx, t_engine, model_info))
        self.print_table(table_rows)
        return output

    def get_files(self):
        """返回用户给定的所有相关 model files paths 列表。

        返回:
            (list[Path]): model files 的 Path objects 列表。
        """
        files = []
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                extensions = ["*.pt", "*.onnx", "*.yaml"]
                files.extend([file for ext in extensions for file in glob.glob(str(path / ext))])
            elif path.suffix in {".pt", ".yaml", ".yml"}:
                files.append(str(path))
            else:
                files.extend(glob.glob(str(path)))
        LOGGER.info(f"正在 profiling: {sorted(files)}")
        return [Path(file) for file in sorted(files)]

    @staticmethod
    def get_onnx_model_info(onnx_file: str):
        """从 ONNX model file 提取 metadata，包括 layers、parameters、gradients 与 FLOPs。"""
        return 0.0, 0.0, 0.0, 0.0

    @staticmethod
    def iterative_sigma_clipping(data: np.ndarray, sigma: float = 2, max_iters: int = 3):
        """对 data 应用 iterative sigma clipping 以移除 outliers。

        参数:
            data (np.ndarray): input data array。
            sigma (float): clipping 使用的 standard deviations 数量。
            max_iters (int): clipping process 的 maximum iterations 数。

        返回:
            (np.ndarray): 移除 outliers 后的 clipped data array。
        """
        data = np.array(data)
        for _ in range(max_iters):
            mean, std = np.mean(data), np.std(data)
            clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
            if len(clipped_data) == len(data):
                break
            data = clipped_data
        return data

    def profile_tensorrt_model(self, engine_file: str, eps: float = 0.001):
        """使用 TensorRT profile YOLO model performance，测量 average run time 与 standard deviation。

        参数:
            engine_file (str): TensorRT engine file 的 path。
            eps (float): 防止除零的小 epsilon value。

        返回:
            (tuple[float, float]): inference time 的 mean 与 standard deviation，单位为 milliseconds。
        """
        if not self.trt or not Path(engine_file).is_file():
            return 0.0, 0.0
        model = YOLO(engine_file)
        input_data = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                model(input_data, imgsz=self.imgsz, verbose=False)
            elapsed = time.time() - start_time
        num_runs = max(
            round(self.min_time / (elapsed + eps) * self.num_warmup_runs),
            self.num_timed_runs * 50,
        )
        run_times = []
        for _ in TQDM(range(num_runs), desc=engine_file):
            results = model(input_data, imgsz=self.imgsz, verbose=False)
            run_times.append(results[0].speed["inference"])
        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)
        return np.mean(run_times), np.std(run_times)

    @staticmethod
    def check_dynamic(tensor_shape):
        """检查 ONNX model 中的 tensor shape 是否为 dynamic。"""
        return not all(isinstance(dim, int) and dim >= 0 for dim in tensor_shape)

    def profile_onnx_model(self, onnx_file: str, eps: float = 0.001):
        """profile ONNX model，测量多次 runs 的 average inference time 与 standard deviation。

        参数:
            onnx_file (str): ONNX model file 的 path。
            eps (float): 防止除零的小 epsilon value。

        返回:
            (tuple[float, float]): inference time 的 mean 与 standard deviation，单位为 milliseconds。
        """
        check_requirements([("onnxruntime", "onnxruntime-gpu")])
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8
        sess = ort.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])
        input_data_dict = {}
        for input_tensor in sess.get_inputs():
            input_type = input_tensor.type
            if self.check_dynamic(input_tensor.shape):
                if len(input_tensor.shape) != 4 and self.check_dynamic(input_tensor.shape[1:]):
                    raise ValueError(f"{input_tensor.name} 的 dynamic shape {input_tensor.shape} 不受支持")
                input_shape = (
                    (1, 3, self.imgsz, self.imgsz) if len(input_tensor.shape) == 4 else (1, *input_tensor.shape[1:])
                )
            else:
                input_shape = input_tensor.shape
            if "float16" in input_type:
                input_dtype = np.float16
            elif "float" in input_type:
                input_dtype = np.float32
            elif "double" in input_type:
                input_dtype = np.float64
            elif "int64" in input_type:
                input_dtype = np.int64
            elif "int32" in input_type:
                input_dtype = np.int32
            else:
                raise ValueError(f"不支持的 ONNX datatype {input_type}")
            input_data = np.random.rand(*input_shape).astype(input_dtype)
            input_name = input_tensor.name
            input_data_dict[input_name] = input_data
        output_name = sess.get_outputs()[0].name
        elapsed = 0.0
        for _ in range(3):
            start_time = time.time()
            for _ in range(self.num_warmup_runs):
                sess.run([output_name], input_data_dict)
            elapsed = time.time() - start_time
        num_runs = max(
            round(self.min_time / (elapsed + eps) * self.num_warmup_runs),
            self.num_timed_runs,
        )
        run_times = []
        for _ in TQDM(range(num_runs), desc=onnx_file):
            start_time = time.time()
            sess.run([output_name], input_data_dict)
            run_times.append((time.time() - start_time) * 1000)
        run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=5)
        return np.mean(run_times), np.std(run_times)

    def generate_table_row(
        self,
        model_name: str,
        t_onnx: tuple[float, float],
        t_engine: tuple[float, float],
        model_info: tuple[float, float, float, float],
    ):
        """生成包含 model performance metrics 的 table row string。

        参数:
            model_name (str): model name。
            t_onnx (tuple): ONNX model inference time statistics（mean, std）。
            t_engine (tuple): TensorRT engine inference time statistics（mean, std）。
            model_info (tuple): model information（layers, params, gradients, flops）。

        返回:
            (str): 包含 model metrics 的 formatted table row string。
        """
        _layers, params, _gradients, flops = model_info
        return f"| {model_name:18s} | {self.imgsz} | - | {t_onnx[0]:.1f}±{t_onnx[1]:.1f} ms | {t_engine[0]:.1f}±{t_engine[1]:.1f} ms | {params / 1000000.0:.1f} | {flops:.1f} |"

    @staticmethod
    def generate_results_dict(
        model_name: str,
        t_onnx: tuple[float, float],
        t_engine: tuple[float, float],
        model_info: tuple[float, float, float, float],
    ):
        """生成 profiling results 字典。

        参数:
            model_name (str): model name。
            t_onnx (tuple): ONNX model inference time statistics（mean, std）。
            t_engine (tuple): TensorRT engine inference time statistics（mean, std）。
            model_info (tuple): model information（layers, params, gradients, flops）。

        返回:
            (dict): 包含 profiling results 的 dictionary。
        """
        _layers, params, _gradients, flops = model_info
        return {
            "model/name": model_name,
            "model/parameters": params,
            "model/GFLOPs": round(flops, 3),
            "model/speed_ONNX(ms)": round(t_onnx[0], 3),
            "model/speed_TensorRT(ms)": round(t_engine[0], 3),
        }

    @staticmethod
    def print_table(table_rows: list[str]):
        """打印 model profiling results 的 formatted table。

        参数:
            table_rows (list[str]): formatted table row strings 列表。
        """
        gpu = paddle.cuda.get_device_name(0) if paddle.cuda.is_available() else "GPU"
        headers = [
            "Model",
            "size<br><sup>(pixels)",
            "mAP<sup>val<br>50-95",
            f"Speed<br><sup>CPU ({get_cpu_info()}) ONNX<br>(ms)",
            f"Speed<br><sup>{gpu} TensorRT<br>(ms)",
            "params<br><sup>(M)",
            "FLOPs<br><sup>(B)",
        ]
        header = "|" + "|".join(f" {h} " for h in headers) + "|"
        separator = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"
        LOGGER.info(f"\n\n{header}")
        LOGGER.info(separator)
        for row in table_rows:
            LOGGER.info(row)
