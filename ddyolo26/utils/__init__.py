# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief ddyolo26 工具包顶层模块：全局常量、日志、settings 等基础设施。
@details
定义了整个框架通用的全局对象和常量：
- `LOGGER`：统一日志记录器
- `SETTINGS`：持久化用户设置（~/.config/Ultralytics/settings.yaml）
- `DEFAULT_CFG`、`DEFAULT_CFG_DICT`：默认超参数配置
- `ASSETS`、`ROOT`、`WEIGHTS_DIR` 等路径常量
- 环境检测标志：`IS_COLAB`、`IS_DOCKER`、`IS_JETSON` 等
- `callbacks`：全局回调注册表
"""

import paddle

import contextlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import re
import socket
import sys
import tempfile
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from urllib.parse import unquote

import cv2
import numpy as np

from ddyolo26.utils.git import GitRepo
from ddyolo26.utils.patches import checkpoint_save, imread, imshow, imwrite
from ddyolo26.utils.tqdm import TQDM

RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
ARGV = sys.argv or ["", ""]
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
ASSETS = ROOT / "assets"
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"  # upstream assets CDN
PROJECT_NAME = "PaddleYOLO-RKNN"
PROJECT_SITE = "https://github.com/Dengdxx/PaddleYOLO-RKNN"
PROJECT_COMMUNITY = "https://github.com/Dengdxx/PaddleYOLO-RKNN/issues"
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
AUTOINSTALL = str(os.getenv("YOLO_AUTOINSTALL", False)).lower() == "true"
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"
LOGGING_NAME = "ddyolo26"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None
NOT_MACOS14 = not (MACOS and MACOS_VERSION.startswith("14."))
ARM64 = platform.machine() in {"arm64", "aarch64"}
PYTHON_VERSION = platform.python_version()
PADDLE_VERSION = str(paddle.__version__)
IS_VSCODE = os.environ.get("TERM_PROGRAM", False) == "vscode"
RKNN_CHIPS = frozenset(
    {
        "rk3588",
        "rk3576",
        "rk3566",
        "rk3568",
        "rk3562",
        "rv1103",
        "rv1106",
        "rv1103b",
        "rv1106b",
        "rk2118",
        "rv1126b",
    }
)
HELP_MSG = f"""
    {PROJECT_NAME} 运行示例：

    1. 创建隔离环境，并安装仓库 README 中列出的依赖。

    2. 在 {PROJECT_NAME} 仓库根目录使用 Python SDK：

        from ddyolo26 import YOLO

        # 加载 model
        model = YOLO("ddyolo26/cfg/models/v8/yolov8.yaml")  # 从零构建新 model
        model = YOLO("weights/yolov8/yolov8n.pdparams")  # 加载 Paddle weights

        # 使用 model
        results = model.train(data="coco8.yaml", epochs=3)  # 训练 model
        results = model.val()  # 在 validation set 上评估 model performance
        results = model("ddyolo26/assets/bus.jpg")  # 对 image 做 predict
        success = model.export(format="onnx")  # export model 到 ONNX format

    3. 使用 command line interface（CLI）：

        {PROJECT_NAME} 的 'yolo' CLI command 使用以下语法：

            yolo TASK MODE ARGS

            其中    TASK（可选）为 [detect, segment] 之一
                    MODE（必选）为 [train, val, predict, export, benchmark] 之一
                    ARGS（可选）为任意数量自定义 "arg=value" 参数，例如 "imgsz=320"，用于覆盖默认值。
                        可在项目 config 参考或通过 "yolo cfg" 查看全部 ARGS。

        - 使用初始 learning_rate 0.01 train detection model 10 个 epochs
            yolo detect train data=coco8.yaml model=weights/yolov8/yolov8n.pdparams epochs=10 lr0=0.01

        - 使用 pretrained segmentation model，以 image size 320 predict YouTube video：
            yolo segment predict model=weights/yolov8seg/yolov8n-seg.pdparams source='https://youtu.be/LNwODJXcvt4' imgsz=320

        - 使用 batch-size 1 与 image size 640 val pretrained detection model：
            yolo detect val model=weights/yolov8/yolov8n.pdparams data=coco8.yaml batch=1 imgsz=640

        - 运行特殊 commands：
            yolo help
            yolo checks
            yolo version
            yolo settings
            yolo copy-cfg
            yolo cfg

    Repository: {PROJECT_SITE}
    Issues: {PROJECT_COMMUNITY}
    """
paddle.set_printoptions(linewidth=320, precision=4, threshold=1000, edgeitems=3)
np.set_printoptions(linewidth=320, formatter=dict(float_kind="{:11.5g}".format))
cv2.setNumThreads(0)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KINETO_LOG_LEVEL"] = "5"
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
# warnings.filterwarnings("ignore", category=UserWarning, message=".*prim::Constant.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="coremltools")
warnings.filterwarnings("ignore", message="When training, we now always track global mean and variance")
warnings.filterwarnings("ignore", message="No ccache found")
warnings.filterwarnings("ignore", message=".*Preheat PIR.*")  # Paddle eager_utils PIR 预热
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")  # multiprocessing fork 提示
warnings.filterwarnings("ignore", message=".*onnxruntime.*")  # onnxruntime quantization内部提示
FLOAT_OR_INT = float, int
STR_OR_PATH = str, Path


class DataExportMixin:
    """用于将 validation metrics 或 prediction results 导出为多种 formats 的 Mixin 类。

    该类提供工具函数，可将 detection 与 segmentation 任务的 performance metrics（例如 mAP、precision、recall）
    或 prediction results 导出为 Polars DataFrame、CSV 与 JSON 等 formats。

    方法:
        to_df: 将 summary 转为 Polars DataFrame。
        to_csv: 将 results 导出为 CSV string。
        to_json: 将 results 导出为 JSON string。
        tojson: Deprecated alias for `to_json()`.

    示例:
        >>> model = YOLO("weights/yolov8/yolov8n.pdparams")
        >>> results = model("image.jpg")
        >>> df = results.to_df()
        >>> print(df)
        >>> csv_data = results.to_csv()
    """

    def to_df(self, normalize=False, decimals=5):
        """由 prediction results summary 或 validation metrics 创建 Polars DataFrame。

        参数:
            normalize (bool, optional): normalize 数值，便于比较。
            decimals (int, optional): floats 四舍五入保留的小数位数。

        返回:
            (polars.DataFrame): 包含 summary data 的 Polars DataFrame。
        """
        import polars as pl

        return pl.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5):
        """将 results 或 metrics 导出为 CSV string format。

        参数:
            normalize (bool, optional): normalize numeric values。
            decimals (int, optional): decimal precision。

        返回:
            (str): string 形式的 CSV content。
        """
        import polars as pl

        df = self.to_df(normalize=normalize, decimals=decimals)
        try:
            return df.write_csv()
        except Exception:

            def _to_str_simple(v):
                if v is None:
                    return ""
                elif isinstance(v, (dict, list, tuple, set)):
                    return repr(v)
                else:
                    return str(v)

            df_str = df.select(
                [pl.col(c).map_elements(_to_str_simple, return_dtype=pl.String).alias(c) for c in df.columns]
            )
            return df_str.write_csv()

    def to_json(self, normalize=False, decimals=5):
        """将 results 导出为 JSON format。

        参数:
            normalize (bool, optional): normalize numeric values。
            decimals (int, optional): decimal precision。

        返回:
            (str): results 的 JSON-formatted string。
        """
        return self.to_df(normalize=normalize, decimals=decimals).write_json()


class SimpleClass:
    """用于创建带属性字符串表示对象的简单 base class。

    该类为可打印或可表示为字符串的对象提供基础，会显示所有 non-callable attributes。
    这对于 debugging 与 object state introspection 很有用。

    方法:
        __str__: 返回 human-readable 的对象字符串表示。
        __repr__: 返回 machine-readable 的对象字符串表示。
        __getattr__: 提供带帮助信息的自定义 attribute access error message。

    示例:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    说明:
        - 该类设计为被 subclass，提供方便的 object attributes 检查方式。
        - 字符串表示包含对象的 module 与 class name。
        - Callable attributes 与以下划线开头的 attributes 会从字符串表示中排除。
    """

    def __str__(self):
        """返回 human-readable 的对象字符串表示。"""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {v!r}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """返回 machine-readable 的对象字符串表示。"""
        return self.__str__()

    def __getattr__(self, attr):
        """提供带帮助信息的自定义 attribute access error message。"""
        name = self.__class__.__name__
        raise AttributeError(
            f"""'{name}' object 没有 attribute '{attr}'。可用 attributes 见下方。
{self.__doc__}"""
        )


class IterableSimpleNamespace(SimpleNamespace):
    """可迭代的 SimpleNamespace 类，增强 attribute access 与 iteration 功能。

    该类扩展 SimpleNamespace，增加 iteration、string representation 与 attribute access 方法，
    作为便捷容器用于存储和访问 configuration parameters。

    方法:
        __iter__: 返回 namespace attributes 的 key-value pairs iterator。
        __str__: 返回 human-readable 的对象字符串表示。
        __getattr__: 提供带帮助信息的自定义 attribute access error message。
        get: 获取指定 key 的值；若 key 不存在，则返回 default value。

    示例:
        >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    说明:
        相比标准 dictionary，该类特别适合以更易访问、可迭代的 format 存储 configuration parameters。
    """

    def __iter__(self):
        """返回 namespace attributes 的 key-value pairs iterator。"""
        return iter(vars(self).items())

    def __str__(self):
        """返回 human-readable 的对象字符串表示。"""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """提供带帮助信息的自定义 attribute access error message。"""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object 没有 attribute '{attr}'。这可能是因为 ddyolo26 的 'default.yaml' 文件被修改
            或已过期。
请更新代码，并在必要时用本地项目 default configuration file 的最新版本替换：
            {DEFAULT_CFG_PATH}
            """
        )

    def get(self, key, default=None):
        """若指定 key 存在则返回其值，否则返回 default value。"""
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend="Agg"):
    """临时设置 plotting function 的 rc parameters 与 backend 的 decorator。

    参数:
        rcparams (dict, optional): 要设置的 rc parameters 字典。
        backend (str, optional): 要使用的 backend 名称。

    返回:
        (Callable): 临时设置 rc parameters 与 backend 后的 decorated function。

    示例:
        >>> @plt_settings({"font.size": 12})
        ... def plot_function():
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()

        >>> with plt_settings({"font.size": 12}):
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()
    """
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """对函数应用临时 rc parameters 与 backend 的 decorator。"""

        def wrapper(*args, **kwargs):
            """设置 rc parameters 与 backend，调用原函数，然后恢复设置。"""
            import matplotlib.pyplot as plt

            if "font.sans-serif" not in rcparams and not wrapper._fonts_registered:
                from matplotlib import font_manager

                known = {f.fname for f in font_manager.fontManager.ttflist}
                for f in USER_CONFIG_DIR.glob("*.ttf"):
                    if str(f) not in known:
                        font_manager.fontManager.addfont(str(f))
                wrapper._fonts_registered = True
            rc = (
                rcparams
                if "font.sans-serif" in rcparams
                else {
                    **rcparams,
                    "font.sans-serif": [
                        "Arial Unicode MS",
                        *plt.rcParams.get("font.sans-serif", []),
                    ],
                }
            )
            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")
                plt.switch_backend(backend)
            try:
                with plt.rc_context(rc):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        wrapper._fonts_registered = False
        return wrapper

    return decorator


def set_logging(name="LOGGING_NAME", verbose=True):
    """设置 UTF-8 encoding 与可配置 verbosity 的 logging。

    该函数为 PaddleYOLO-RKNN library 配置 logging，会根据 verbosity flag 与当前 process rank 设置合适的
    logging level 与 formatter。它也处理 Windows 环境中 UTF-8 encoding 可能不是默认值的特殊情况。

    参数:
        name (str): logger 名称。
        verbose (bool): 若为 True，将 logging level 设为 INFO；否则设为 ERROR。

    返回:
        (logging.Logger): 配置后的 logger 对象。

    示例:
        >>> set_logging(name="ddyolo26", verbose=True)
        >>> logger = logging.getLogger("ddyolo26")
        >>> logger.info("这是一条 info message")

    说明:
        - 在 Windows 上，该函数会尽可能重新配置 stdout 以使用 UTF-8 encoding。
        - 若无法重新配置，则回退到可处理 non-UTF-8 环境的自定义 formatter。
        - 该函数会设置带合适 formatter 与 level 的 StreamHandler。
        - logger 的 propagate flag 会设为 False，避免 parent loggers 中重复 logging。
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR

    class PrefixFormatter(logging.Formatter):
        def format(self, record):
            """根据 level 为 log records 添加 prefix。"""
            if record.levelno == logging.WARNING:
                prefix = "WARNING" if WINDOWS else "WARNING ⚠️"
                record.msg = f"{prefix} {record.msg}"
            elif record.levelno == logging.ERROR:
                prefix = "ERROR" if WINDOWS else "ERROR ❌"
                record.msg = f"{prefix} {record.msg}"
            formatted_message = super().format(record)
            return emojis(formatted_message)

    formatter = PrefixFormatter("%(message)s")
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        with contextlib.suppress(Exception):
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)


def emojis(string=""):
    """返回按平台处理后的 emoji-safe string。"""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class ThreadingLocked:
    """确保 function 或 method thread-safe execution 的 decorator 类。

    该类可作为 decorator 使用，确保被装饰函数被多个 threads 调用时，同一时间只有一个 thread 能执行该函数。

    属性:
        lock (threading.Lock): 用于管理 decorated function 访问的 lock 对象。

    示例:
        >>> from ddyolo26.utils import ThreadingLocked
        >>> @ThreadingLocked()
        ... def my_function():
        ...    # 你的代码
    """

    def __init__(self):
        """使用 threading lock 初始化 decorator 类。"""
        self.lock = threading.Lock()

    def __call__(self, f):
        """以 thread-safe 方式执行 function 或 method。"""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """对 decorated function 或 method 应用 thread-safety。"""
            with self.lock:
                return f(*args, **kwargs)

        return decorated


class YAML:
    """用于高效 file operations、可自动检测 C implementation 的 YAML 工具类。

    该类使用 PyYAML 可用的最快 implementation（可用时使用 C-based）提供优化后的 YAML load/save 操作。
    它通过 lazy initialization 实现 singleton pattern，允许直接使用 class method 而无需显式实例化。
    该类会自动处理 file path 创建、validation 与 character encoding 问题。

    该实现通过以下方式优先考虑 performance：
        - 可用时自动选择 C-based loader/dumper
        - 通过 singleton pattern 复用同一实例
        - 使用 lazy initialization 将 import 成本延迟到真正需要时
        - 提供 fallback mechanisms 处理有问题的 YAML content

    属性:
        _instance: 内部 singleton instance storage。
        yaml: PyYAML module 引用。
        SafeLoader: 最佳可用 YAML loader（可用时为 CSafeLoader）。
        SafeDumper: 最佳可用 YAML dumper（可用时为 CSafeDumper）。

    示例:
        >>> data = YAML.load("config.yaml")
        >>> data["new_value"] = 123
        >>> YAML.save("updated_config.yaml", data)
        >>> YAML.print(data)
    """

    _instance = None

    @classmethod
    def _get_instance(cls):
        """首次使用时初始化 singleton instance。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """使用最优 YAML implementation 初始化（可用时使用 C-based）。"""
        import yaml

        self.yaml = yaml
        try:
            self.SafeLoader = yaml.CSafeLoader
            self.SafeDumper = yaml.CSafeDumper
        except (AttributeError, ImportError):
            self.SafeLoader = yaml.SafeLoader
            self.SafeDumper = yaml.SafeDumper

    @classmethod
    def save(cls, file="data.yaml", data=None, header=""):
        """将 Python object 保存为 YAML file。

        参数:
            file (str | Path): 保存 YAML file 的 path。
            data (dict | None): 要保存的 dict 或兼容对象。
            header (str): 可选 string，添加到 file 开头。
        """
        instance = cls._get_instance()
        if data is None:
            data = {}
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        valid_types = int, float, str, bool, list, tuple, dict, type(None)
        for k, v in data.items():
            if not isinstance(v, valid_types):
                data[k] = str(v)
        with open(file, "w", errors="ignore", encoding="utf-8") as f:
            if header:
                f.write(header)
            instance.yaml.dump(data, f, sort_keys=False, allow_unicode=True, Dumper=instance.SafeDumper)

    @classmethod
    def load(cls, file="data.yaml", append_filename=False):
        """通过稳健的 error handling 将 YAML file 加载为 Python object。

        参数:
            file (str | Path): YAML file 的 path。
            append_filename (bool): 是否将 filename 添加到返回的 dict。

        返回:
            (dict): 加载后的 YAML content。
        """
        instance = cls._get_instance()
        assert str(file).endswith((".yaml", ".yml")), f"不是 YAML file: {file}"
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()
        try:
            data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}
        except Exception as e:
            s = re.sub(
                "[^\\x09\\x0A\\x0D\\x20-\\x7E\\x85\\xA0-\\uD7FF\\uE000-\\uFFFD\\U00010000-\\U0010ffff]+",
                "",
                s,
            )
            try:
                data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}
            except Exception:
                raise ValueError(
                    f"""'{file}' 中存在 YAML syntax error: {e}
可使用 https://ray.run/tools/yaml-formatter 校验 YAML"""
                ) from None
        if "None" in data.values():
            data = {k: (None if v == "None" else v) for k, v in data.items()}
        if append_filename:
            data["yaml_file"] = str(file)
        return data

    @classmethod
    def print(cls, yaml_file):
        """将 YAML file 或 object pretty print 到 console。

        参数:
            yaml_file (str | Path | dict): 要打印的 YAML file path 或 dict。
        """
        instance = cls._get_instance()
        yaml_dict = cls.load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
        dump = instance.yaml.dump(
            yaml_dict,
            sort_keys=False,
            allow_unicode=True,
            width=-1,
            Dumper=instance.SafeDumper,
        )
        LOGGER.info(f"正在打印 '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


DEFAULT_CFG_DICT = YAML.load(DEFAULT_CFG_PATH)
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def read_device_model() -> str:
    """从系统读取 device model 信息。

    返回:
        (str): 小写 platform release string，用于识别 Jetson 或 Raspberry Pi 等 device models。
    """
    return platform.release().lower()


def is_ubuntu() -> bool:
    """检查 OS 是否为 Ubuntu。

    返回:
        (bool): 若 OS 为 Ubuntu，则返回 True；否则返回 False。
    """
    try:
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    except FileNotFoundError:
        return False


def is_debian(codenames: (list[str] | None | str) = None) -> list[bool] | bool:
    """检查 OS 是否为 Debian。

    参数:
        codenames (list[str] | None | str): 要检查的特定 Debian codename（例如 'buster'、'bullseye'）。
            若为 None，则只检查是否为 Debian。

    返回:
        (list[bool] | bool): boolean 列表，表示 OS 是否匹配各 Debian codename；若未提供 codenames，
            则返回单个 boolean。
    """
    try:
        with open("/etc/os-release") as f:
            content = f.read()
            if codenames is None:
                return "ID=debian" in content
            if isinstance(codenames, str):
                codenames = [codenames]
            return [
                (f"VERSION_CODENAME={codename}" in content if codename else "ID=debian" in content)
                for codename in codenames
            ]
    except FileNotFoundError:
        return [False] * len(codenames) if codenames else False


def is_colab():
    """检查当前 script 是否运行在 Google Colab notebook 内。

    返回:
        (bool): 若运行在 Colab notebook 内，则返回 True；否则返回 False。
    """
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    """检查当前 script 是否运行在 Kaggle kernel 内。

    返回:
        (bool): 若运行在 Kaggle kernel 内，则返回 True；否则返回 False。
    """
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_jupyter():
    """检查当前 script 是否运行在 Jupyter Notebook 内。

    返回:
        (bool): 若运行在 Jupyter Notebook 内，则返回 True；否则返回 False。

    说明:
        - 仅适用于 Colab 与 Kaggle，Jupyterlab、Paperspace 等其他环境无法可靠检测。
        - "get_ipython" in globals() 方法在手动安装 IPython package 时会出现 false positives。
    """
    return IS_COLAB or IS_KAGGLE


def is_docker() -> bool:
    """判断 script 是否运行在 Docker container 内。

    返回:
        (bool): 若 script 运行在 Docker container 内，则返回 True；否则返回 False。
    """
    try:
        return os.path.exists("/.dockerenv")
    except Exception:
        return False


def is_raspberrypi() -> bool:
    """判断 Python environment 是否运行在 Raspberry Pi 上。

    返回:
        (bool): 若运行在 Raspberry Pi 上，则返回 True；否则返回 False。
    """
    return "rpi" in DEVICE_MODEL


@lru_cache(maxsize=3)
def is_jetson(jetpack=None) -> bool:
    """判断 Python environment 是否运行在 NVIDIA Jetson device 上。

    参数:
        jetpack (int | None): 若指定，则检查特定 JetPack version（4、5、6）。

    返回:
        (bool): 若运行在 NVIDIA Jetson device 上，则返回 True；否则返回 False。
    """
    jetson = "tegra" in DEVICE_MODEL
    if jetson and jetpack:
        try:
            content = open("/etc/nv_tegra_release").read()
            version_map = {(4): "R32", (5): "R35", (6): "R36", (7): "R38"}
            return jetpack in version_map and version_map[jetpack] in content
        except Exception:
            return False
    return jetson


def is_dgx() -> bool:
    """检查当前 script 是否运行在 DGX（NVIDIA Data Center GPU）、DGX-Ready 或 DGX Spark system 内。

    返回:
        (bool): 若运行在 DGX、DGX-Ready 或 DGX Spark system 内，则返回 True；否则返回 False。
    """
    try:
        with open("/etc/dgx-release") as f:
            return "DGX" in f.read()
    except FileNotFoundError:
        return False


def is_online() -> bool:
    """使用 DNS（v4/v6）resolution（Cloudflare + Google）快速检查 online 状态。

    返回:
        (bool): 若 connection 成功，则返回 True；否则返回 False。
    """
    if str(os.getenv("YOLO_OFFLINE", "")).lower() == "true":
        return False
    for host in ("one.one.one.one", "dns.google"):
        try:
            socket.getaddrinfo(host, 0, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG)
            return True
        except OSError:
            continue
    return False


def is_pip_package(filepath: str = __name__) -> bool:
    """判断给定 filepath 处的 file 是否属于 pip package。

    参数:
        filepath (str): 要检查的 filepath。

    返回:
        (bool): 若 file 属于 pip package，则返回 True；否则返回 False。
    """
    import importlib.util

    spec = importlib.util.find_spec(filepath)
    return spec is not None and spec.origin is not None


def is_dir_writeable(dir_path: (str | Path)) -> bool:
    """检查 directory 是否可写。

    参数:
        dir_path (str | Path): directory 的 path。

    返回:
        (bool): 若 directory 可写，则返回 True；否则返回 False。
    """
    return os.access(str(dir_path), os.W_OK)


def is_github_action_running() -> bool:
    """判断当前 environment 是否为 GitHub Actions runner。

    返回:
        (bool): 若当前 environment 为 GitHub Actions runner，则返回 True；否则返回 False。
    """
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ


def get_default_args(func):
    """返回 function 默认参数的 dictionary。

    参数:
        func (callable): 要 inspect 的 function。

    返回:
        (dict): dictionary，其中每个 key 为参数名，每个 value 为该参数的 default value。
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_ubuntu_version():
    """若 OS 为 Ubuntu，则获取 Ubuntu version。

    返回:
        (str): Ubuntu version；若不是 Ubuntu OS，则返回 None。
    """
    if is_ubuntu():
        try:
            with open("/etc/os-release") as f:
                return re.search('VERSION_ID="(\\d+\\.\\d+)"', f.read())[1]
        except (FileNotFoundError, AttributeError):
            return None


def get_user_config_dir(sub_dir="DDYOLO26"):
    """返回可写 config dir，优先使用 YOLO_CONFIG_DIR，并感知 OS。

    参数:
        sub_dir (str): 要创建的 subdirectory 名称。

    返回:
        (Path): user config directory 的 path。
    """
    if env_dir := os.getenv("YOLO_CONFIG_DIR"):
        p = Path(env_dir).expanduser() / sub_dir
    elif LINUX:
        p = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / sub_dir
    elif WINDOWS:
        p = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:
        p = Path.home() / "Library" / "Application Support" / sub_dir
    else:
        raise ValueError(f"不支持的操作系统: {platform.system()}")
    if p.exists():
        return p
    if is_dir_writeable(p.parent):
        p.mkdir(parents=True, exist_ok=True)
        return p
    for alt in [Path(tempfile.gettempdir()) / sub_dir, Path.cwd() / sub_dir]:
        if alt.exists():
            return alt
        if is_dir_writeable(alt.parent):
            alt.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(f"user config directory '{p}' 不可写，改用 '{alt}'。可设置 YOLO_CONFIG_DIR 覆盖。")
            return alt
    p = Path.cwd() / sub_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


DEVICE_MODEL = read_device_model()
ONLINE = is_online()
IS_COLAB = is_colab()
IS_KAGGLE = is_kaggle()
IS_DOCKER = is_docker()
IS_JETSON = is_jetson()
IS_JUPYTER = is_jupyter()
IS_PIP_PACKAGE = is_pip_package()
IS_RASPBERRYPI = is_raspberrypi()
IS_DEBIAN, IS_DEBIAN_BOOKWORM, IS_DEBIAN_TRIXIE = is_debian([None, "bookworm", "trixie"])
IS_UBUNTU = is_ubuntu()
GIT = GitRepo()
USER_CONFIG_DIR = get_user_config_dir()
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"


def colorstr(*input):
    """使用 ANSI escape codes，根据提供的 color 与 style 参数为 string 着色。

    该函数可通过两种方式调用：
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    第二种形式会默认应用 'blue' 与 'bold'。

    参数:
        *input (str | Path): string 序列，其中前 n-1 个 string 是 color 与 style 参数，最后一个 string 是待着色内容。

    返回:
        (str): 使用指定 color 与 style 的 ANSI escape codes 包裹后的输入 string。

    示例:
        >>> colorstr("blue", "bold", "hello world")
        "\\033[34m\\033[1mhello world\\033[0m"

    说明:
        支持的 Colors 与 Styles：
        - Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        - Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        - Misc: 'end', 'bold', 'underline'

    参考:
        https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\x1b[30m",
        "red": "\x1b[31m",
        "green": "\x1b[32m",
        "yellow": "\x1b[33m",
        "blue": "\x1b[34m",
        "magenta": "\x1b[35m",
        "cyan": "\x1b[36m",
        "white": "\x1b[37m",
        "bright_black": "\x1b[90m",
        "bright_red": "\x1b[91m",
        "bright_green": "\x1b[92m",
        "bright_yellow": "\x1b[93m",
        "bright_blue": "\x1b[94m",
        "bright_magenta": "\x1b[95m",
        "bright_cyan": "\x1b[96m",
        "bright_white": "\x1b[97m",
        "end": "\x1b[0m",
        "bold": "\x1b[1m",
        "underline": "\x1b[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def remove_colorstr(input_string):
    """移除 string 中的 ANSI escape codes，即去除颜色。

    参数:
        input_string (str): 要移除 color 与 style 的 string。

    返回:
        (str): 移除所有 ANSI escape codes 后的新 string。

    示例:
        >>> remove_colorstr(colorstr("blue", "bold", "hello world"))
        "hello world"
    """
    ansi_escape = re.compile("\\x1B\\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)


class TryExcept(contextlib.ContextDecorator):
    """用于优雅处理 exceptions 的 PaddleYOLO-RKNN TryExcept 类。

    该类可作为 decorator 或 context manager 使用，用于捕获 exceptions 并可选打印 warning messages。
    它允许代码在 exception 发生后继续执行，适用于 non-critical operations。

    属性:
        msg (str): exception 发生时显示的可选 message。
        verbose (bool): 是否打印 exception message。

    示例:
        作为 decorator：
        >>> @TryExcept(msg="func 中发生 error", verbose=True)
        ... def func():
        ...     # function 逻辑
        ...     pass

        作为 context manager：
        >>> with TryExcept(msg="block 中发生 error", verbose=True):
        ...     # code block
        ...     pass
    """

    def __init__(self, msg="", verbose=True):
        """使用可选 message 与 verbosity settings 初始化 TryExcept 类。"""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """进入 TryExcept context 时执行，用于初始化 instance。"""
        pass

    def __exit__(self, exc_type, value, traceback):
        """定义退出 'with' block 时的行为，必要时打印 error message。"""
        if self.verbose and value:
            LOGGER.warning(f"{self.msg}{': ' if self.msg else ''}{value}")
        return True


class Retry(contextlib.ContextDecorator):
    """带 exponential backoff 的 function execution 重试类。

    该 decorator 可在 function 出现 exceptions 时重试，最多执行指定次数，并在重试之间使用指数增长的 delay。
    适用于处理 network operations 或其他不稳定流程中的 transient failures。

    属性:
        times (int): 最大 retry attempts 数。
        delay (int): retries 之间的 initial delay，单位 seconds。

    示例:
        作为 decorator 使用示例：
        >>> @Retry(times=3, delay=2)
        ... def test_func():
        ...     # 替换为可能 raise exceptions 的 function 逻辑
        ...     return True
    """

    def __init__(self, times=3, delay=2):
        """使用指定 retries 数量与 delay 初始化 Retry 类。"""
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):
        """带 exponential backoff 的 Retry decorator 实现。"""

        def wrapped_func(*args, **kwargs):
            """对 decorated function 或 method 应用 retries。"""
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    LOGGER.warning(f"Retry {self._attempts}/{self.times} 失败: {e}")
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * 2**self._attempts)

        return wrapped_func


def threaded(func):
    """默认以 multi-thread 执行 target function，并返回 thread 或 function result。

    该 decorator 为 target function 提供灵活执行方式，可在单独 thread 中执行，也可同步执行。
    默认情况下 function 会在 thread 中运行，但可通过 'threaded=False' keyword argument 控制；
    该参数会在调用 function 前从 kwargs 中移除。

    参数:
        func (callable): 可能在单独 thread 中执行的 function。

    返回:
        (callable): wrapper function，返回 daemon thread 或直接 function result。

    示例:
        >>> @threaded
        ... def process_data(data):
        ...     return data
        >>>
        >>> thread = process_data(my_data)  # 在 background thread 中运行
        >>> result = process_data(my_data, threaded=False)  # 同步运行，返回 function result
    """

    def wrapper(*args, **kwargs):
        """根据 'threaded' kwarg 对给定 function 使用 multi-thread，并返回 thread 或 function result。"""
        if kwargs.pop("threaded", True):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper


class JSONDict(dict):
    """为内容提供 JSON persistence 的 dictionary-like 类。

    该类扩展内置 dictionary，在内容被修改时自动保存到 JSON file。它使用 lock 确保 thread-safe operations，
    并处理 Path 对象的 JSON serialization。

    属性:
        file_path (Path): 用于 persistence 的 JSON file path。
        lock (threading.Lock): 确保 thread-safe operations 的 lock 对象。

    方法:
        _load: 从 JSON file 加载 data 到 dictionary。
        _save: 将 dictionary 当前状态保存到 JSON file。
        __setitem__: 存储 key-value pair 并持久化到 disk。
        __delitem__: 移除 item 并更新 persistent storage。
        update: 更新 dictionary 并持久化 changes。
        clear: 清空所有 entries 并更新 persistent storage。

    示例:
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        value
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """

    def __init__(self, file_path: (str | Path) = "data.json"):
        """使用指定 file path 初始化支持 JSON persistence 的 JSONDict 对象。"""
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        """从 JSON file 加载 data 到 dictionary。"""
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    super().update(json.load(f))
        except json.JSONDecodeError:
            LOGGER.warning(f"从 {self.file_path} 解码 JSON 时出错，将使用空 dictionary。")
        except Exception as e:
            LOGGER.error(f"读取 {self.file_path} 时出错: {e}")

    def _save(self):
        """将 dictionary 当前状态保存到 JSON file。"""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            LOGGER.error(f"写入 {self.file_path} 时出错: {e}")

    @staticmethod
    def _json_default(obj):
        """处理 Path 对象的 JSON 序列化。"""
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"{type(obj).__name__} 类型的 object 无法 JSON 序列化")

    def __setitem__(self, key, value):
        """存储 key-value pair 并持久化到 disk。"""
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        """移除 item 并更新 persistent storage。"""
        with self.lock:
            super().__delitem__(key)
            self._save()

    def __str__(self):
        """返回 dictionary 的 pretty-printed JSON string 表示。"""
        contents = json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)
        return f'JSONDict("{self.file_path}"):\n{contents}'

    def update(self, *args, **kwargs):
        """更新 dictionary 并持久化 changes。"""
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        """清空所有 entries 并更新 persistent storage。"""
        with self.lock:
            super().clear()
            self._save()


class SettingsManager(JSONDict):
    """用于管理并持久化 PaddleYOLO-RKNN settings 的 SettingsManager 类。

    该类扩展 JSONDict，为 settings 提供 JSON persistence，同时确保 thread-safe operations 与 default values。
    初始化时会 validate settings，并提供 update/reset settings 的方法。settings 包括 datasets、weights、
    runs 的 directories，以及各类 integration flags。

    属性:
        file (Path): 用于 persistence 的 JSON file path。
        version (str): settings schema 的 version。
        defaults (dict): 包含 default settings 的 dictionary。
        help_msg (str): 告知用户如何查看与更新 settings 的 help message。

    方法:
        _validate_settings: validate 当前 settings，必要时 reset。
        update: update settings，并 validate keys 与 types。
        reset: 将 settings reset 为默认值并保存。

    示例:
        初始化并更新 settings：
        >>> settings = SettingsManager()
        >>> settings.update(runs_dir="runs/custom")
        >>> print(settings["runs_dir"])
        runs/custom
    """

    def __init__(self, file=SETTINGS_FILE, version="0.0.9"):
        """使用 default settings 初始化 SettingsManager，并加载 user settings。"""
        import hashlib
        import uuid

        from ddyolo26.utils.runtime import distributed_zero_first

        root = GIT.root or Path()
        datasets_root = (root.parent if GIT.root and is_dir_writeable(root.parent) else root).resolve()
        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,
            "datasets_dir": str(datasets_root / "datasets"),
            "weights_dir": str(root / "weights"),
            "runs_dir": str(root / "runs"),
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),
            "vscode_msg": True,
        }
        self.help_msg = f"""
可通过 'yolo settings' 或 '{self.file}' 查看 PaddleYOLO-RKNN settings。
可通过 'yolo settings key=value' 更新 settings，例如 'yolo settings runs_dir=path/to/dir'。帮助见 {PROJECT_SITE}。"""
        with distributed_zero_first(LOCAL_RANK):
            super().__init__(self.file)
            if not self.file.exists() or not self:
                LOGGER.info(f"正在创建新的 PaddleYOLO-RKNN settings v{version} file ✅ {self.help_msg}")
                self.reset()
            self._validate_settings()

    def _validate_settings(self):
        """validate 当前 settings，必要时 reset。"""
        correct_keys = frozenset(self.keys()) == frozenset(self.defaults.keys())
        correct_types = all(isinstance(self.get(k), type(v)) for k, v in self.defaults.items())
        correct_version = self.get("settings_version", "") == self.version
        if not (correct_keys and correct_types and correct_version):
            LOGGER.warning(
                f"PaddleYOLO-RKNN settings 已 reset 为默认值。这可能是 settings 存在问题或近期 ddyolo26 package update 导致的。{self.help_msg}"
            )
            self.reset()
        if self.get("datasets_dir") == self.get("runs_dir"):
            LOGGER.warning(
                f"PaddleYOLO-RKNN setting 'datasets_dir: {self.get('datasets_dir')}' 必须不同于 'runs_dir: {self.get('runs_dir')}'。请修改其中一个，以避免 training 期间可能出现的问题。{self.help_msg}"
            )

    def __setitem__(self, key, value):
        """更新一个 key: value pair。"""
        self.update({key: value})

    def update(self, *args, **kwargs):
        """update settings，并 validate keys 与 types。"""
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
        for k, v in kwargs.items():
            if k not in self.defaults:
                raise KeyError(f"不存在 PaddleYOLO-RKNN setting '{k}'。{self.help_msg}")
            t = type(self.defaults[k])
            if not isinstance(v, t):
                raise TypeError(
                    f"PaddleYOLO-RKNN setting '{k}' 必须是 '{t.__name__}' 类型，而不是 '{type(v).__name__}'。{self.help_msg}"
                )
        super().update(*args, **kwargs)

    def reset(self):
        """将 settings reset 为默认值并保存。"""
        self.clear()
        self.update(self.defaults)


def deprecation_warn(arg, new_arg=None):
    """使用 deprecated argument 时发出 deprecation warning，并建议更新后的 argument。"""
    msg = f"'{arg}' 已 deprecated，将在未来移除。"
    if new_arg is not None:
        msg += f" 请改用 '{new_arg}'。"
    LOGGER.warning(msg)


def clean_url(url):
    """从 URL 中移除 auth，例如 https://url.com/file.txt?auth -> https://url.com/file.txt。"""
    url = Path(url).as_posix().replace(":/", "://")
    return unquote(url).split("?", 1)[0]


def url2file(url):
    """将 URL 转为 filename，例如 https://url.com/file.txt?auth -> file.txt。"""
    return Path(clean_url(url)).name


def vscode_msg(ext="ddyolo26.ddyolo26-snippets") -> str:
    """若尚未安装，则显示安装 PaddleYOLO-RKNN VS Code snippets 的消息。"""
    return ""

    PREFIX = colorstr("PaddleYOLO-RKNN: ")


SETTINGS = SettingsManager()
PERSISTENT_CACHE = JSONDict(USER_CONFIG_DIR / "persistent_cache.json")
DATASETS_DIR = Path(SETTINGS["datasets_dir"])
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])
RUNS_DIR = Path(SETTINGS["runs_dir"])
ENVIRONMENT = (
    "Colab"
    if IS_COLAB
    else "Kaggle"
    if IS_KAGGLE
    else "Jupyter"
    if IS_JUPYTER
    else "Docker"
    if IS_DOCKER
    else platform.system()
)
paddle.save = checkpoint_save
if WINDOWS:
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow
