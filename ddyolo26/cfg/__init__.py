# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief PaddleYOLO-RKNN 配置系统：超参数定义、CLI 解析、任务/模式/数据集映射。
@details
核心功能：
- `get_cfg()`：合并 YAML 默认配置与用户覆盖参数，返回 `OverrideDict`
- `get_save_dir()`：根据 task/name/project 构建实验保存路径
- `TASK2DATA`：任务 → 默认数据集映射（detect → coco8.yaml 等）
- `TASK2MODEL`：任务 → 默认预训练权重映射
- `TASK2METRIC`：任务 → 主评估指标名映射（detect → metrics/mAP50-95(B)）

CLI 解析：支持 `key=value` 覆盖语法，与常见 YOLO 命令行用法兼容。
"""

import paddle

import ast
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ddyolo26 import __version__
from ddyolo26.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    FLOAT_OR_INT,
    IS_VSCODE,
    LOGGER,
    PROJECT_COMMUNITY,
    PROJECT_SITE,
    RANK,
    ROOT,
    RUNS_DIR,
    SETTINGS,
    SETTINGS_FILE,
    STR_OR_PATH,
    YAML,
    IterableSimpleNamespace,
    checks,
    colorstr,
    deprecation_warn,
    vscode_msg,
)

SOLUTION_MAP = {
    "count": "ObjectCounter",
    "crop": "ObjectCropper",
    "blur": "ObjectBlurrer",
    "workout": "AIGym",
    "heatmap": "Heatmap",
    "isegment": "InstanceSegmentation",
    "visioneye": "VisionEye",
    "speed": "SpeedEstimator",
    "queue": "QueueManager",
    "analytics": "Analytics",
    "inference": "Inference",
    "trackzone": "TrackZone",
    "help": None,
}
MODES = frozenset({"train", "val", "predict", "export", "benchmark"})
TASKS = frozenset({"detect", "segment"})
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
}
TASK2MODEL = {
    "detect": "weights/yolov8/yolov8n.pdparams",
    "segment": "weights/yolov8seg/yolov8n-seg.pdparams",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
}
ARGV = sys.argv or ["", ""]
SOLUTIONS_HELP_MSG = f"""
    收到参数: {["yolo", *ARGV[1:]]!s}。PaddleYOLO-RKNN 不包含上游
    'solutions' 管线或 Streamlit demo 入口。

    当前支持的公开入口:
        yolo train ...
        yolo val ...
        yolo predict ...
        yolo export ...

    仓库: {PROJECT_SITE}
    问题反馈: {PROJECT_COMMUNITY}
    """
CLI_HELP_MSG = f"""
    收到参数: {["yolo", *ARGV[1:]]!s}。PaddleYOLO-RKNN 的 'yolo' 命令语法如下:

        yolo TASK MODE ARGS

        其中   TASK（可选）为 {list(TASKS)} 之一
                MODE（必选）为 {list(MODES)} 之一
                ARGS（可选）为任意数量的 'arg=value' 覆盖项，例如 'imgsz=320'。
                    可在项目配置参考或 'yolo cfg' 中查看全部 ARGS。

    1. 用初始学习率 0.01 训练检测模型 10 轮
        yolo train data=coco8.yaml model=weights/yolov8/yolov8n.pdparams epochs=10 lr0=0.01

    2. 使用预训练分割模型，以 320 输入尺寸预测 YouTube 视频:
        yolo predict model=weights/yolov8seg/yolov8n-seg.pdparams source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. 用 batch=1、imgsz=640 验证预训练检测模型:
        yolo val model=weights/yolov8/yolov8n.pdparams data=coco8.yaml batch=1 imgsz=640

    4. 运行特殊命令:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg

    仓库: {PROJECT_SITE}
    问题反馈: {PROJECT_COMMUNITY}
    """
CFG_FLOAT_KEYS = frozenset(
    {
        "warmup_epochs",
        "box",
        "cls",
        "dfl",
        "degrees",
        "shear",
        "time",
        "workspace",
        "batch",
    }
)
CFG_FRACTION_KEYS = frozenset(
    {
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_momentum",
        "warmup_bias_lr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "scale",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "conf",
        "iou",
        "fraction",
        "multi_scale",
    }
)
CFG_INT_KEYS = frozenset(
    {
        "epochs",
        "patience",
        "workers",
        "seed",
        "close_mosaic",
        "mask_ratio",
        "max_det",
        "vid_stride",
        "line_width",
        "nbs",
        "save_period",
    }
)
CFG_BOOL_KEYS = frozenset(
    {
        "save",
        "exist_ok",
        "verbose",
        "deterministic",
        "single_cls",
        "rect",
        "cos_lr",
        "overlap_mask",
        "val",
        "save_json",
        "half",
        "dnn",
        "plots",
        "show",
        "save_txt",
        "save_conf",
        "save_crop",
        "save_frames",
        "show_labels",
        "show_conf",
        "visualize",
        "augment",
        "agnostic_nms",
        "retina_masks",
        "show_boxes",
        "keras",
        "optimize",
        "int8",
        "dynamic",
        "simplify",
        "nms",
        "profile",
        "end2end",
        "dual_raw",
    }
)


def cfg2dict(cfg: (str | Path | dict | SimpleNamespace)) -> dict:
    """将配置对象转换为字典。

    参数:
        cfg (str | Path | dict | SimpleNamespace): 待转换配置；可以是文件路径、字符串、字典或 SimpleNamespace。

    返回:
        (dict): 字典格式配置。

    示例:
        将 YAML 文件路径转成字典:
        >>> config_dict = cfg2dict("config.yaml")

        将 SimpleNamespace 转成字典:
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        已经是字典时直接返回:
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    说明:
        - cfg 为路径或字符串时按 YAML 加载。
        - cfg 为 SimpleNamespace 时使用 vars() 转换。
        - cfg 已经是字典时保持不变。
    """
    if isinstance(cfg, STR_OR_PATH):
        cfg = YAML.load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg


def get_cfg(
    cfg: (str | Path | dict | SimpleNamespace) = DEFAULT_CFG_DICT,
    overrides: (dict | None) = None,
) -> SimpleNamespace:
    """从文件或字典加载配置，并合并可选覆盖项。

    参数:
        cfg (str | Path | dict | SimpleNamespace): 配置来源，可以是文件路径、字典或 SimpleNamespace。
        overrides (dict | None): 覆盖基础配置的键值对。

    返回:
        (SimpleNamespace): 合并后的配置命名空间。

    示例:
        >>> from ddyolo26.cfg import get_cfg
        >>> config = get_cfg()  # 加载默认配置
        >>> config_with_overrides = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch_size": 16})

    说明:
        - 同时提供 `cfg` 和 `overrides` 时，`overrides` 优先。
        - 会对配置做必要规整，例如把数值型 `project`/`name` 转成字符串。
        - 会检查配置键、类型和值是否合法。
    """
    cfg = cfg2dict(cfg)
    if overrides:
        overrides = cfg2dict(overrides)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}
    for k in ("project", "name"):
        if k in cfg and isinstance(cfg[k], FLOAT_OR_INT):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":
        cfg["name"] = str(cfg.get("model", "")).partition(".")[0]
        LOGGER.warning(f"'name=model' 已自动更新为 'name={cfg['name']}'。")
    check_cfg(cfg)
    return IterableSimpleNamespace(**cfg)


def check_cfg(cfg: dict, hard: bool = True) -> None:
    """检查 PaddleYOLO-RKNN 配置参数的类型和值。

    根据 `CFG_FLOAT_KEYS`、`CFG_FRACTION_KEYS`、`CFG_INT_KEYS`、`CFG_BOOL_KEYS`
    中定义的键集合验证配置；在非 hard 模式下会尝试自动转换类型。

    参数:
        cfg (dict): 待检查配置字典。
        hard (bool): True 时遇到非法类型/值直接抛错；False 时尝试转换。

    示例:
        >>> config = {
        ...     "epochs": 50,  # 有效 integer
        ...     "lr0": 0.01,  # 有效 float
        ...     "momentum": 1.2,  # 无效 float（超出 0.0-1.0 范围）
        ...     "save": "true",  # 无效 bool
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 1.2, 'save': False}  # 已修正 'save' key

    说明:
        - 本函数会原地修改输入字典。
        - None 通常来自可选参数，会被忽略。
        - fraction 类参数必须位于 [0.0, 1.0]。
    """
    for k, v in cfg.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and not isinstance(v, FLOAT_OR_INT):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' 类型无效: {type(v).__name__}。合法 '{k}' 类型为 int（例如 '{k}=0'）或 float（例如 '{k}=0.5'）"
                    )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' 类型无效: {type(v).__name__}。合法 '{k}' 类型为 int（例如 '{k}=0'）或 float（例如 '{k}=0.5'）"
                        )
                    cfg[k] = v = float(v)
                if not 0.0 <= v <= 1.0:
                    raise ValueError(f"'{k}={v}' 值无效。合法 '{k}' 取值范围为 0.0 到 1.0。")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(f"'{k}={v}' 类型无效: {type(v).__name__}。'{k}' 必须为 int（例如 '{k}=8'）")
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' 类型无效: {type(v).__name__}。'{k}' 必须为 bool（例如 '{k}=True' 或 '{k}=False'）"
                    )
                cfg[k] = bool(v)


def get_save_dir(args: SimpleNamespace, name: (str | None) = None) -> Path:
    """根据参数和默认设置返回输出保存目录。

    参数:
        args (SimpleNamespace): 配置命名空间，包含 'project'、'name'、'task'、'mode'、'save_dir' 等字段。
        name (str | None): 可选输出目录名；为空时使用 'args.name' 或 'args.mode'。

    返回:
        (Path): 输出保存目录。

    示例:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project="my_project", task="detect", mode="train", exist_ok=True)
        >>> save_dir = get_save_dir(args)
        >>> print(save_dir)
        runs/detect/my_project/train
    """
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ddyolo26.utils.files import increment_path

        project = args.project or ""
        if not Path(project).is_absolute():
            project = Path("runs") / args.task / project
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)
    return Path(save_dir).resolve()


def _handle_deprecation(custom: dict) -> dict:
    """处理已弃用配置键，并映射到当前等价键。

    参数:
        custom (dict): 可能包含弃用键的配置字典。

    返回:
        (dict): 替换弃用键后的配置字典。

    示例:
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        >>> print(custom_config)
        {'show_boxes': True, 'show_labels': True, 'line_width': 2}

    说明:
        本函数会原地修改输入字典，并在必要时转换值，例如对 'hide_labels' 和
        'hide_conf' 执行布尔取反。
    """
    deprecated_mappings = {
        "boxes": ("show_boxes", lambda v: v),
        "hide_labels": ("show_labels", lambda v: not bool(v)),
        "hide_conf": ("show_conf", lambda v: not bool(v)),
        "line_thickness": ("line_width", lambda v: v),
    }
    removed_keys = {"label_smoothing", "save_hybrid", "crop_fraction"}
    for old_key, (new_key, transform) in deprecated_mappings.items():
        if old_key not in custom:
            continue
        deprecation_warn(old_key, new_key)
        custom[new_key] = transform(custom.pop(old_key))
    for key in removed_keys:
        if key not in custom:
            continue
        deprecation_warn(key)
        custom.pop(key)
    return custom


def check_dict_alignment(
    base: dict,
    custom: dict,
    e: (Exception | None) = None,
    allowed_custom_keys: (set | None) = None,
) -> None:
    """检查自定义配置与基础配置是否对齐，并为错误键提供提示。

    参数:
        base (dict): 包含合法键的基础配置字典。
        custom (dict): 待检查的自定义配置字典。
        e (Exception | None): 调用方传入的可选异常实例。
        allowed_custom_keys (set | None): 自定义配置中额外允许的键集合。

    异常:
        SystemExit: 当发现非法配置键时抛出。

    示例:
        >>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch_size": 16}
        >>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch_size": 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SystemExit:
        ...     print("发现不匹配的 keys")

    说明:
        - 会基于相似度为非法键给出可能的修正建议。
        - 会自动替换自定义配置中的弃用键。
        - 会为每个非法键打印详细错误，帮助用户修正配置。
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (frozenset(x.keys()) for x in (base, custom))
    if allowed_custom_keys is None:
        allowed_custom_keys = {"augmentations", "save_dir"}
    if mismatched := [k for k in custom_keys if k not in base_keys and k not in allowed_custom_keys]:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)
            matches = [(f"{k}={base[k]}" if base.get(k) is not None else k) for k in matches]
            match_str = f"相似参数例如 {matches}。" if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' 不是合法 YOLO 参数。{match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


def merge_equals_args(args: list[str]) -> list[str]:
    """合并命令行中被空格拆开的 `=` 参数，并拼接括号片段。

    处理以下形式：
        1. ['arg', '=', 'val'] 变为 ['arg=val']
        2. ['arg=', 'val'] 变为 ['arg=val']
        3. ['arg', '=val'] 变为 ['arg=val']
        4. 拼接括号片段，如 ['imgsz=[3,', '640,', '640]'] 变为 ['imgsz=[3,640,640]']

    参数:
        args (list[str]): 命令行参数或片段列表。

    返回:
        (list[str]): 合并后的参数列表。

    示例:
        >>> args = ["arg1", "=", "value", "arg2=", "value2", "arg3", "=value3", "imgsz=[3,", "640,", "640]"]
        >>> merge_equals_args(args)
        ['arg1=value', 'arg2=value2', 'arg3=value3', 'imgsz=[3,640,640]']
    """
    new_args = []
    current = ""
    depth = 0
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "=" and 0 < i < len(args) - 1:
            new_args[-1] += f"={args[i + 1]}"
            i += 2
            continue
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:
            new_args.append(f"{arg}{args[i + 1]}")
            i += 2
            continue
        elif arg.startswith("=") and i > 0:
            new_args[-1] += arg
            i += 1
            continue
        depth += arg.count("[") - arg.count("]")
        current += arg
        if depth == 0:
            new_args.append(current)
            current = ""
        i += 1
    if current:
        new_args.append(current)
    return new_args


def handle_yolo_settings(args: list[str]) -> None:
    """处理 YOLO settings 命令行命令。

    支持 reset 和逐项更新 settings，用于 CLI 中的设置管理。

    参数:
        args (list[str]): settings 管理相关命令行参数。

    示例:
        >>> handle_yolo_settings(["reset"])  # 重置 YOLO settings
        >>> handle_yolo_settings(["default_cfg_path=yolo26n.yaml"])  # 更新指定 setting

    说明:
        - 不提供参数时显示当前 settings。
        - 'reset' 会删除现有 settings 文件并创建新的默认 settings。
        - 其它参数按键值对处理，用于更新指定 settings。
        - 更新前会检查新旧 settings 是否对齐。
        - 处理后会显示更新后的 settings。
                - 更多 settings 说明见仓库:
                    {PROJECT_SITE}
    """
    url = PROJECT_SITE
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_FILE.unlink()
                SETTINGS.reset()
                LOGGER.info("Settings 已成功重置")
            else:
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)
                for k, v in new.items():
                    LOGGER.info(f"✅ 已更新 '{k}={v}'")
        LOGGER.info(SETTINGS)
        LOGGER.info(f"💡 PaddleYOLO-RKNN settings 更多说明见 {url}")
    except Exception as e:
        LOGGER.warning(f"settings 错误: '{e}'。请查看 {url} 获取帮助。")


def handle_yolo_solutions(args: list[str]) -> None:
    """拒绝当前不支持的 'solutions' CLI 命令。"""
    if args and args[0] == "help":
        LOGGER.info(SOLUTIONS_HELP_MSG)
        return
    LOGGER.error("PaddleYOLO-RKNN 不包含 upstream 'solutions' 或 Streamlit demo 模块。")
    LOGGER.info(f"支持的 entrypoints 请参见 {PROJECT_SITE}")
    raise SystemExit(2)


def parse_key_value_pair(pair: str = "key=value") -> tuple:
    """将 `key=value` 字符串解析为键和值。

    参数:
        pair (str): 形如 "key=value" 的键值对字符串。

    返回:
        key (str): 解析得到的键。
        value (str): 解析得到的值。

    异常:
        AssertionError: 值缺失或为空时抛出。

    示例:
        >>> key, value = parse_key_value_pair("model=weights/yolov8/yolov8n.pdparams")
        >>> print(f"Key: {key}, Value: {value}")
        Key: model, Value: weights/yolov8/yolov8n.pdparams

        >>> key, value = parse_key_value_pair("epochs=100")
        >>> print(f"Key: {key}, Value: {value}")
        Key: epochs, Value: 100

    说明:
        - 仅按第一个 '=' 拆分。
        - 键和值都会去掉首尾空白。
        - 去空白后值为空会抛出断言错误。
    """
    k, v = pair.split("=", 1)
    k, v = k.strip(), v.strip()
    assert v, f"缺少 '{k}' 的值"
    return k, smart_value(v)


def smart_value(v: str) -> Any:
    """将字符串形式的值转换为合适的 Python 类型。

    会尝试把字符串转换为 None、bool、int、float 或其它可安全 literal_eval 的对象。

    参数:
        v (str): 待转换的字符串值。

    返回:
        (Any): 转换后的值；无法转换时返回原字符串。

    示例:
        >>> smart_value("42")
        42
        >>> smart_value("3.14")
        3.14
        >>> smart_value("True")
        True
        >>> smart_value("None")
        None
        >>> smart_value("some_string")
        'some_string'

    说明:
        - bool 和 None 使用大小写不敏感比较。
        - 其它类型尝试使用 ast.literal_eval() 安全解析。
        - 无法转换时返回原字符串。
    """
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return ast.literal_eval(v)
        except Exception:
            return v


def entrypoint(debug: str = "") -> None:
    """PaddleYOLO-RKNN 命令行入口：解析并执行命令行参数。

    该函数是 CLI 主入口，负责解析命令并执行训练、验证、推理、导出等任务。

    参数:
        debug (str): 调试用命令行字符串，参数以空格分隔。

    示例:
        用初始学习率 0.01 训练检测模型 10 轮:
        >>> entrypoint("train data=coco8.yaml model=weights/yolov8/yolov8n.pdparams epochs=10 lr0=0.01")

        使用预训练分割模型，以 320 输入尺寸预测 YouTube 视频:
        >>> entrypoint("predict model=weights/yolov8seg/yolov8n-seg.pdparams source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        使用 batch=1、imgsz=640 验证预训练检测模型:
        >>> entrypoint("val model=weights/yolov8/yolov8n.pdparams data=coco8.yaml batch=1 imgsz=640")

    说明:
        - 未传入参数时显示用法帮助。
        - 全部命令和参数见帮助信息，以及 PaddleYOLO-RKNN 仓库 {PROJECT_SITE}。
    """
    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:
        LOGGER.info(CLI_HELP_MSG)
        return
    special = {
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: YAML.print(DEFAULT_CFG_PATH),
        "copy-cfg": copy_default_cfg,
        "solutions": lambda: handle_yolo_solutions(args[1:]),
        "help": lambda: LOGGER.info(CLI_HELP_MSG),
    }
    full_args_dict = {
        **DEFAULT_CFG_DICT,
        **{k: None for k in TASKS},
        **{k: None for k in MODES},
        **special,
    }
    special.update({k[0]: v for k, v in special.items()})
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})
    special = {
        **special,
        **{f"-{k}": v for k, v in special.items()},
        **{f"--{k}": v for k, v in special.items()},
    }
    overrides = {}
    for a in merge_equals_args(args):
        if a.startswith("--"):
            LOGGER.warning(f"argument '{a}' 不需要前导 '--'，已更新为 '{a[2:]}'。")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"argument '{a}' 不需要结尾逗号 ','，已更新为 '{a[:-1]}'。")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:
                    LOGGER.info(f"使用 {v} 覆盖 {DEFAULT_CFG_PATH}")
                    overrides = {k: val for k, val in YAML.load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)
        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"""'{colorstr("red", "bold", a)}' 是合法 YOLO argument，但缺少用于赋值的 '='，例如可尝试 '{a}={DEFAULT_CFG_DICT[a]}'
{CLI_HELP_MSG}"""
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})
    check_dict_alignment(full_args_dict, overrides)
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"缺少 'mode' argument。合法 modes 为 {list(MODES)}。使用默认 'mode={mode}'。")
    elif mode not in MODES:
        raise ValueError(f"无效 'mode={mode}'。合法 modes 为 {list(MODES)}。\n{CLI_HELP_MSG}")
    task = overrides.pop("task", None)
    if task:
        if task not in TASKS:
            raise ValueError(
                f"""无效 'task={task}'。合法 tasks 为 {list(TASKS)}。
{CLI_HELP_MSG}"""
            )
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]
    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "weights/yolov8/yolov8n.pdparams"
        LOGGER.warning(f"缺少 'model' argument。使用默认 'model={model}'。")
    overrides["model"] = model
    from ddyolo26 import YOLO

    model = YOLO(model, task=task)
    if task != model.task:
        if task:
            LOGGER.warning(
                f"传入的 'task={task}' 与 model 的 'task={model.task}' 冲突。忽略 'task={task}'，并更新为 'task={model.task}' 以匹配 model。"
            )
        task = model.task
    if mode == "predict" and "source" not in overrides:
        overrides["source"] = DEFAULT_CFG.source or ASSETS
        LOGGER.warning(f"缺少 'source' argument。使用默认 'source={overrides['source']}'。")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"缺少 'data' argument。使用默认 'data={overrides['data']}'。")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "onnx"
            LOGGER.warning(f"缺少 'format' argument。使用默认 'format={overrides['format']}'。")
    getattr(model, mode)(**overrides)
    LOGGER.info(f"💡 更多信息见 {PROJECT_SITE}")
    if IS_VSCODE and SETTINGS.get("vscode_msg", True):
        LOGGER.info(vscode_msg())


def copy_default_cfg() -> None:
    """复制默认配置文件，并在当前目录创建带 `_copy` 后缀的新文件。

    该函数复制 `DEFAULT_CFG_PATH` 指向的默认配置，便于用户基于默认值创建自定义配置。

    示例:
        >>> copy_default_cfg()
        # 输出：default.yaml copied to path/to/current/directory/default_copy.yaml
        # 使用该自定义 cfg 的 YOLO 命令示例：
        #   yolo cfg='path/to/current/directory/default_copy.yaml' imgsz=320 batch=8

    说明:
        - 新配置文件创建在当前工作目录。
        - 复制完成后会打印新文件位置和示例 YOLO 命令。
        - 适合想修改默认配置但不想直接改原文件的用户。
    """
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"""{DEFAULT_CFG_PATH} 已复制到 {new_file}
使用该自定义 cfg 的 YOLO 命令示例：
    yolo cfg='{new_file}' imgsz=320 batch=8"""
    )


if __name__ == "__main__":
    entrypoint(debug="")
