# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 环境与依赖检查工具：验证 Python/Paddle 版本、包安装、文件存在等。
@details
提供：
- `check_version()`：强制或警告式的版本号对比
- `check_requirements()`：自动检测并安装缺失的 pip 包
- `check_imgsz()`：保证图像尺寸是 stride 的整数倍
- `check_file()` / `check_yaml()`：文件路径解析与验证
"""

from __future__ import annotations
import sys
import paddle


import ast
import functools
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ddyolo26.paddle_utils import *

from ddyolo26.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,
    AUTOINSTALL,
    GIT,
    IS_COLAB,
    IS_DOCKER,
    IS_JETSON,
    IS_KAGGLE,
    IS_PIP_PACKAGE,
    LINUX,
    LOGGER,
    MACOS,
    ONLINE,
    PROJECT_SITE,
    PYTHON_VERSION,
    RKNN_CHIPS,
    ROOT,
    PADDLE_VERSION,
    USER_CONFIG_DIR,
    WINDOWS,
    Retry,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    is_github_action_running,
    url2file,
)


def parse_requirements(file_path=ROOT.parent / "requirements-paddle.txt", package=""):
    """解析 requirements 文件，忽略以 '#' 开头的行以及 '#' 后的内容。

    参数:
        file_path (Path): requirements 文件的 path。
        package (str, optional): 用于替代 requirements 文件的 Python package。

    返回:
        requirements (list[SimpleNamespace]): 解析后的 requirements 列表，每项为带 `name` 与 `specifier`
            attributes 的 SimpleNamespace 对象。

    示例:
        >>> from ddyolo26.utils.checks import parse_requirements
        >>> parse_requirements(package="ddyolo26")
    """
    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        requires = Path(file_path).read_text().splitlines()
    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.partition("#")[0].strip()
            if match := re.match("([a-zA-Z0-9-_]+)\\s*([<>!=~]+.*)?", line):
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))
    return requirements


def get_distribution_name(import_name: str) -> str:
    """获取给定 import name 对应的 pip distribution name（例如 'cv2' -> 'opencv-python-headless'）。"""
    for dist in metadata.distributions():
        top_level = (dist.read_text("top_level.txt") or "").split()
        if import_name in top_level:
            return dist.metadata["Name"]
    return import_name


@functools.lru_cache
def parse_version(version="0.0.0") -> tuple:
    """将 version string 转为整数 tuple，忽略附加的非数字字符串。

    参数:
        version (str): version string，例如 '2.0.1+cpu'。

    返回:
        (tuple): 表示 version 数字部分的整数 tuple，例如 (2, 0, 1)。
    """
    try:
        return tuple(map(int, re.findall("\\d+", version)[:3]))
    except Exception as e:
        LOGGER.warning(f"parse_version({version}) 失败，返回 (0, 0, 0): {e}")
        return 0, 0, 0


def is_ascii(s) -> bool:
    """检查 string 是否仅由 ASCII characters 组成。

    参数:
        s (str | list | tuple | dict): 待检查输入（都会转为 string 后检查）。

    返回:
        (bool): 若 string 仅由 ASCII characters 组成，则返回 True；否则返回 False。
    """
    return all(ord(c) < 128 for c in str(s))


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """验证 image size 在每个维度上都是给定 stride 的倍数。

    若 image size 不是 stride 的倍数，则将其更新为大于等于给定 floor value 的最近 stride 倍数。

    参数:
        imgsz (int | list[int]): image size。
        stride (int): stride value。
        min_dim (int): 最小 dimensions 数。
        max_dim (int): 最大 dimensions 数。
        floor (int): image size 允许的最小值。

    返回:
        (list[int] | int): 更新后的 image size。
    """
    stride = int(stride.max() if isinstance(stride, paddle.Tensor) else stride)
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):
        imgsz = [int(imgsz)] if imgsz.isnumeric() else ast.literal_eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' 类型无效: {type(imgsz).__name__}。有效 imgsz 类型为 int，例如 'imgsz=640'；或 list，例如 'imgsz=[640,640]'"
        )
    if len(imgsz) > max_dim:
        msg = "'train' 与 'val' 的 imgsz 必须是 integer；'predict' 与 'export' 的 imgsz 可以是 [h, w] list 或 integer，例如 'yolo export imgsz=640,480' 或 'yolo export imgsz=640'"
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} 不是有效 image size。{msg}")
        LOGGER.warning(f"更新为 'imgsz={max(imgsz)}'。{msg}")
        imgsz = [max(imgsz)]
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]
    if sz != imgsz:
        LOGGER.warning(f"imgsz={imgsz} 必须是 max stride {stride} 的倍数，更新为 {sz}")
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz
    return sz


@functools.lru_cache
def check_uv():
    """检查 uv package manager 是否已安装且可正常运行。"""
    try:
        return subprocess.run(["uv", "-V"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


@functools.lru_cache
def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """检查 current version 是否满足 required version 或范围。

    参数:
        current (str): current version，或用于获取 version 的 package name。
        required (str): required version 或范围（pip-style format）。
        name (str): warning message 中使用的名称。
        hard (bool): 若为 True，requirement 不满足时 raise ModuleNotFoundError。
        verbose (bool): 若为 True，requirement 不满足时打印 warning message。
        msg (str): verbose 时显示的额外 message。

    返回:
        (bool): 若 requirement 满足，则返回 True；否则返回 False。

    示例:
        检查 current version 是否恰好为 22.04
        >>> check_version(current="22.04", required="==22.04")

        检查 current version 是否大于等于 22.04
        >>> check_version(current="22.10", required="22.04")  # 未传入不等式时默认按 '>=' 处理

        检查 current version 是否小于等于 22.04
        >>> check_version(current="22.04", required="<=22.04")

        检查 current version 是否位于 20.04（含）到 22.04（不含）之间
        >>> check_version(current="21.10", required=">20.04,<22.04")
    """
    if not current:
        LOGGER.warning(f"请求了无效的 check_version({current}, {required})，请检查取值。")
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(f"需要 {current} package，但尚未安装") from e
            else:
                return False
    if not required:
        return True
    if "sys_platform" in required and (
        WINDOWS
        and "win32" not in required
        or LINUX
        and "linux" not in required
        or MACOS
        and "macos" not in required
        and "darwin" not in required
    ):
        return True
    op = ""
    version = ""
    result = True
    c = parse_version(current)
    for r in required.strip(",").split(","):
        op, version = re.match("([^0-9]*)([\\d.]+)", r).groups()
        if not op:
            op = ">="
        v = parse_version(version)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not c >= v:
            result = False
        elif op == "<=" and not c <= v:
            result = False
        elif op == ">" and not c > v:
            result = False
        elif op == "<" and not c < v:
            result = False
    if not result:
        warning = f"需要 {name}{required}，但当前安装的是 {name}=={current} {msg}"
        if hard:
            raise ModuleNotFoundError(warning)
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name="ddyolo26"):
    """不下载或安装 package，返回 PyPI package 的 latest version。

    参数:
        package_name (str): 要查询 latest version 的 package 名称。

    返回:
        (str | None): package 的 latest version；若不可用则返回 None。
    """
    import requests

    try:
        requests.packages.urllib3.disable_warnings()
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]
    except Exception:
        return None


def check_pip_update_available():
    """检查 PyPI 上是否有新的 ddyolo26 package version。

    返回:
        (bool): 若存在 update，则返回 True；否则返回 False。
    """
    if ONLINE and IS_PIP_PACKAGE:
        try:
            from ddyolo26 import __version__

            latest = check_latest_pypi_version()
            if check_version(__version__, f"<{latest}"):
                LOGGER.info(
                    f"发现新的 https://pypi.org/project/ddyolo26/{latest} 😃 可使用 'pip install -U ddyolo26' 更新"
                )
                return True
        except Exception:
            pass
    return False


@ThreadingLocked()
@functools.lru_cache
def check_font(font="Arial.ttf"):
    """在本地查找 font；若不存在，则下载到用户配置目录。

    参数:
        font (str): font 的 path 或名称。

    返回:
        (Path | str): 解析后的 font file path。
    """
    from matplotlib import font_manager

    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]
    url = f"{ASSETS_URL}/{name}"
    from ddyolo26.utils.downloads import is_url, safe_download

    if is_url(url, check=True):
        safe_download(url=url, file=file)
        return file


def check_python(minimum: str = "3.8.0", hard: bool = True, verbose: bool = False) -> bool:
    """检查当前 python version 是否满足 required minimum version。

    参数:
        minimum (str): python 的 required minimum version。
        hard (bool): 若为 True，requirement 不满足时 raise ModuleNotFoundError。
        verbose (bool): 若为 True，requirement 不满足时打印 warning message。

    返回:
        (bool): 已安装 Python version 是否满足 minimum constraints。
    """
    return check_version(PYTHON_VERSION, minimum, name="Python", hard=hard, verbose=verbose)


@TryExcept()
def check_apt_requirements(requirements):
    """检查 apt packages 是否已安装，并安装缺失项。

    参数:
        requirements (list[str]): 要检查并安装的 apt package names 列表。
    """
    prefix = colorstr("red", "bold", "apt requirements:")
    missing_packages = []
    for package in requirements:
        try:
            result = subprocess.run(["dpkg", "-l", package], capture_output=True, text=True, check=False)
            if result.returncode != 0 or not any(
                line.startswith("ii") and package in line for line in result.stdout.splitlines()
            ):
                missing_packages.append(package)
        except Exception:
            missing_packages.append(package)
    if missing_packages:
        LOGGER.info(
            f"{prefix} 未找到 PaddleYOLO-RKNN requirement{'s' * (len(missing_packages) > 1)} {missing_packages}，正在尝试 AutoUpdate..."
        )
        cmd = (["sudo"] if is_sudo_available() else []) + ["apt", "update"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        cmd = (["sudo"] if is_sudo_available() else []) + ["apt", "install", "-y"] + missing_packages
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        LOGGER.info(f"{prefix} AutoUpdate 成功 ✅")
        LOGGER.warning(
            f"""{prefix} {colorstr("bold", "请重启 runtime 或重新运行 command，使更新生效")}
"""
        )


@TryExcept()
def check_requirements(requirements=ROOT.parent / "requirements-paddle.txt", exclude=(), install=True, cmds=""):
    """检查已安装 dependencies 是否满足 PaddleYOLO-RKNN requirements，并在需要时尝试 auto-update。

    参数:
        requirements (Path | str | list[str|tuple] | tuple[str]): requirements 文件 path、单个 package
            requirement string、package requirements string 列表，或包含 strings 与可互换 packages tuple 的列表。
        exclude (tuple): 要排除检查的 package names tuple。
        install (bool): 若为 True，尝试 auto-update 不满足 requirements 的 packages。
        cmds (str): auto-updating 时传给 pip install command 的额外 commands。

    示例:
        >>> from ddyolo26.utils.checks import check_requirements

        检查 requirements 文件
        >>> check_requirements("path/to/requirements-paddle.txt")

        检查单个 package
        >>> check_requirements("ddyolo26>=8.3.200")

        检查多个 packages
        >>> check_requirements(["numpy", "ddyolo26"])

        使用可互换 packages 检查
        >>> check_requirements([("onnxruntime", "onnxruntime-gpu"), "numpy"])
    """
    prefix = colorstr("red", "bold", "requirements:")
    if os.environ.get("DDYOLO26_SKIP_REQUIREMENTS_CHECKS", "0") == "1":
        LOGGER.info(f"{prefix} 检测到 DDYOLO26_SKIP_REQUIREMENTS_CHECKS=1，跳过 requirements check。")
        return True
    if isinstance(requirements, Path):
        file = requirements.resolve()
        assert file.exists(), f"{prefix} 未找到 {file}，检查失败。"
        requirements = [f"{x.name}{x.specifier}" for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]
    pkgs = []
    for r in requirements:
        candidates = r if isinstance(r, (list, tuple)) else [r]
        satisfied = False
        for candidate in candidates:
            r_stripped = candidate.rpartition("/")[-1].replace(".git", "")
            match = re.match("([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)
            name, required = match[1], match[2].strip() if match[2] else ""
            try:
                if check_version(metadata.version(name), required):
                    satisfied = True
                    break
            except (AssertionError, metadata.PackageNotFoundError):
                continue
        if not satisfied:
            pkg = candidates[0]
            if "git+" in pkg:
                url, sep, marker = pkg.partition(";")
                pkg = re.sub("[<>!=~]+.*$", "", url) + sep + marker
            pkgs.append(pkg)

    @Retry(times=2, delay=1)
    def attempt_install(packages, commands, use_uv):
        """可用时尝试使用 uv 安装 package，否则 fallback 到 pip。"""
        if use_uv:
            return subprocess.check_output(
                f'uv pip install --no-cache-dir --python "{sys.executable}" {packages} {commands} --index-strategy=unsafe-best-match --break-system-packages',
                shell=True,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return subprocess.check_output(
            f"pip install --no-cache-dir {packages} {commands}",
            shell=True,
            stderr=subprocess.STDOUT,
            text=True,
        )

    s = " ".join(f'"{x}"' for x in pkgs)
    if s:
        if install and AUTOINSTALL:
            n = len(pkgs)
            LOGGER.info(f"{prefix} 未找到 PaddleYOLO-RKNN requirement{'s' * (n > 1)} {pkgs}，正在尝试 AutoUpdate...")
            try:
                t = time.time()
                assert ONLINE, "已跳过 AutoUpdate（offline）"
                use_uv = not ARM64 and check_uv()
                LOGGER.info(attempt_install(s, cmds, use_uv=use_uv))
                dt = time.time() - t
                LOGGER.info(f"{prefix} AutoUpdate 成功 ✅ {dt:.1f}s")
                LOGGER.warning(
                    f"""{prefix} {colorstr("bold", "请重启 runtime 或重新运行 command，使更新生效")}
"""
                )
            except Exception as e:
                msg = f"{prefix} ❌ {e}"
                if hasattr(e, "output") and e.output:
                    msg += f"\n{e.output}"
                LOGGER.warning(msg)
                return False
        else:
            return False
    return True


def check_tensorrt(min_version: str = "7.0.0"):
    """检查并安装 TensorRT requirements，包括平台特定 dependencies。

    参数:
        min_version (str): 支持的最低 TensorRT version（默认 "7.0.0"）。
    """
    if LINUX:
        cuda_version = paddle.version.cuda().split(".")[0]
        check_requirements(f"tensorrt-cu{cuda_version}>={min_version},!=10.1.0")


def check_suffix(file="model.pt", suffix=".pt", msg=""):
    """检查 file(s) 是否使用 acceptable suffix。

    参数:
        file (str | list[str]): 要检查的 file 或 files 列表。
        suffix (str | tuple): acceptable suffix 或 suffixes tuple。
        msg (str): error 时显示的额外 message。
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = {suffix}
        for f in file if isinstance(file, (list, tuple)) else [file]:
            if s := str(f).rpartition(".")[-1].lower().strip():
                assert f".{s}" in suffix, f"{msg}{f} acceptable suffix 为 {suffix}，不是 .{s}"


def check_yolov5u_filename(file: str, verbose: bool = True) -> str:
    """Paddle-only build 中原样返回 filenames。"""
    return file


def check_model_file_from_stem(model: str = "yolo11n") -> str | Path:
    """由有效 model stem 返回 model filename。

    Paddle-only 分支只解析本地权重；不会下载或转换其它框架权重。
    支持的模型名: yolo11/yolo26/yolov8 的 n/s/m/l/x detect/segment 短名。

    参数:
        model (str): 要检查的 model stem。

    返回:
        (str | Path): 带有合适 suffix 的 model filename。
    """
    from ddyolo26.utils.downloads import (
        PADDLE_WEIGHT_FILENAMES,
        PADDLE_WEIGHT_SUPPORTED_NAMES,
        paddle_weight_group,
        paddle_weight_path,
        paddle_weight_lfs_pull_command,
        resolve_paddle_weight,
    )

    path = Path(model)

    _MIN_WEIGHT_BYTES = 1024  # Git LFS pointer files 约 130 bytes

    def _is_valid_weight(p: Path) -> bool:
        return p.exists() and p.stat().st_size > _MIN_WEIGHT_BYTES

    if path.suffix:
        if _is_valid_weight(path):
            return model
        if path.suffix == ".pt" and not path.stem.endswith("_paddle"):
            raise FileNotFoundError(
                f"PyTorch/普通 .pt 权重不被此 Paddle-only 分支支持: {model}\n"
                "请使用本地 Paddle 权重 (*.pdparams 或 *_paddle.pt)。"
            )
        return model

    # 无后缀短名（如 "yolo26n"）只解析本地 Paddle 权重。
    if not path.suffix:
        resolved = resolve_paddle_weight(model)
        if resolved:
            return resolved

        if path.stem in PADDLE_WEIGHT_FILENAMES:
            expected = paddle_weight_path(path.stem)
            legacy = Path("weights") / paddle_weight_group(path.stem) / f"{path.stem}_paddle.pt"
            raise FileNotFoundError(
                f"找不到本地 Paddle 权重 '{model}'。\n"
                f"请将权重放到 {expected}（或兼容旧格式 {legacy}）。\n"
                "当前版本不会自动下载或转换其它框架权重。\n"
                "如权重由 Git LFS 管理，可执行:\n"
                "  git lfs install\n"
                f"  {paddle_weight_lfs_pull_command(path.stem)}\n"
                f"支持的模型: {PADDLE_WEIGHT_SUPPORTED_NAMES}"
            )

        # 防呆：用户可能用了官方 YOLO 的名字格式
        import re

        m = re.match(r"^yolo(11|26)([nsmlx])$", path.stem)
        if m:
            raise FileNotFoundError(
                f"找不到本地 Paddle 权重 '{model}'。\n"
                f"请手动将权重放到 {paddle_weight_path(path.stem)}，"
                "或先执行 git lfs pull 拉取对应权重。\n"
                f"支持的模型: {PADDLE_WEIGHT_SUPPORTED_NAMES}"
            )

    # 完全无法识别的名字 → 提示支持的模型名
    if not path.exists() and not path.suffix:
        raise FileNotFoundError(
            f"未知的模型名 '{model}'。\n"
            f"ddyolo26 支持的 Paddle 预训练模型: {PADDLE_WEIGHT_SUPPORTED_NAMES}\n"
            f"用法示例: YOLO('yolo11n') 或 YOLO('weights/yolo11/yolo11n.pdparams')"
        )

    return model


def check_file(file, suffix="", download=True, download_dir=".", hard=True):
    """搜索/下载 file（如有必要），检查 suffix（如提供），并返回 path。

    参数:
        file (str): file name 或 path、URL、platform URI (ul://)，或 GCS path (gs://)。
        suffix (str | tuple): 用于校验 file 的 acceptable suffix 或 suffixes tuple。
        download (bool): 若 file 本地不存在，是否下载。
        download_dir (str): 下载 file 的 directory。
        hard (bool): 未找到 file 时是否 raise error。

    返回:
        (str | list): file path；若未找到则为空 list。
    """
    check_suffix(file, suffix)
    file = str(file).strip()
    file = check_yolov5u_filename(file)
    if not file or "://" not in file and Path(file).exists() or file.lower().startswith("grpc://"):
        return file
    elif False and download and file.lower().startswith("ul://"):
        return []
    elif download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://", "gs://")):
        if file.startswith("gs://"):
            file = "https://storage.googleapis.com/" + file[5:]
        url = file
        file = Path(download_dir) / url2file(file)
        if file.exists():
            LOGGER.info(f"已在本地 {file} 找到 {clean_url(url)}")
        else:
            from ddyolo26.utils.downloads import safe_download

            safe_download(url=url, file=file, unzip=False)
        return str(file)
    else:
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))
        if not files and hard:
            raise FileNotFoundError(f"'{file}' 不存在")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"多个 files 匹配 '{file}'，请指定 exact path: {files}")
        return files[0] if len(files) else []


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    """搜索/下载 YAML file（如有必要）并返回 path，同时检查 suffix。

    参数:
        file (str | Path): file name 或 path。
        suffix (tuple): acceptable YAML file suffixes tuple。
        hard (bool): 未找到 file 或找到多个 files 时是否 raise error。

    返回:
        (str): YAML file 的 path。
    """
    return check_file(file, suffix, hard=hard)


def check_is_path_safe(basedir: (Path | str), path: (Path | str)) -> bool:
    """检查 resolved path 是否位于目标 directory 下，以防 path traversal。

    参数:
        basedir (Path | str): 目标 directory。
        path (Path | str): 要检查的 path。

    返回:
        (bool): 若 path 安全，则返回 True；否则返回 False。
    """
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()
    return path_resolved.exists() and path_resolved.parts[: len(base_dir_resolved.parts)] == base_dir_resolved.parts


@functools.lru_cache
def check_imshow(warn=False):
    """检查 environment 是否支持 image displays。

    参数:
        warn (bool): environment 不支持 image displays 时是否 warning。

    返回:
        (bool): 若 environment 支持 image displays，则返回 True；否则返回 False。
    """
    try:
        if LINUX:
            assert not IS_COLAB and not IS_KAGGLE
            assert "DISPLAY" in os.environ, "DISPLAY environment variable 未设置。"
        cv2.imshow("test", np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"当前 environment 不支持 cv2.imshow() 或 PIL Image.show()\n{e}")
        return False


def check_yolo(verbose=True, device=""):
    """打印 human-readable 的 YOLO software 与 hardware summary。

    参数:
        verbose (bool): 是否打印 verbose information。
        device (str | str): YOLO 使用的 device。
    """
    import psutil

    from ddyolo26.utils.runtime import select_device

    if IS_COLAB:
        shutil.rmtree("sample_data", ignore_errors=True)
    if verbose:
        gib = 1 << 30
        ram = psutil.virtual_memory().total
        total, _used, free = shutil.disk_usage("/")
        s = f"({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)"
        try:
            from IPython import display

            display.clear_output()
        except ImportError:
            pass
    else:
        s = ""
    if GIT.is_repo:
        check_multiple_install()
    select_device(device=device, newline=False)
    LOGGER.info(f"环境检查完成 ✅ {s}")


def collect_system_info():
    """收集并打印相关 system information，包括 OS、Python、RAM、CPU 与 CUDA。

    返回:
        (dict): 包含 system information 的 dictionary。
    """
    import psutil

    from ddyolo26.utils import ENVIRONMENT
    from ddyolo26.utils.runtime import get_cpu_info, get_gpu_info

    gib = 1 << 30
    cuda = paddle.cuda.is_available()
    check_yolo()
    total, _, free = shutil.disk_usage("/")
    info_dict = {
        "OS": platform.platform(),
        "Environment": ENVIRONMENT,
        "Python": PYTHON_VERSION,
        "Install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
        "Path": str(ROOT),
        "RAM": f"{psutil.virtual_memory().total / gib:.2f} GB",
        "Disk": f"{(total - free) / gib:.1f}/{total / gib:.1f} GB",
        "CPU": get_cpu_info(),
        "CPU count": os.cpu_count(),
        "GPU": get_gpu_info(index=0) if cuda else None,
        "GPU count": paddle.cuda.device_count() if cuda else None,
        "CUDA": paddle.version.cuda() if cuda else None,
    }
    LOGGER.info("\n" + "\n".join(f"{k:<23}{v}" for k, v in info_dict.items()) + "\n")
    package_info = {}
    for r in parse_requirements(package=get_distribution_name("ddyolo26")):
        try:
            current = metadata.version(r.name)
            is_met = "✅ " if check_version(current, str(r.specifier), name=r.name, hard=True) else "❌ "
        except metadata.PackageNotFoundError:
            current = "(未安装)"
            is_met = "❌ "
        package_info[r.name] = f"{is_met}{current}{r.specifier}"
        LOGGER.info(f"{r.name:<23}{package_info[r.name]}")
    info_dict["Package Info"] = package_info
    if is_github_action_running():
        github_info = {
            "RUNNER_OS": os.getenv("RUNNER_OS"),
            "GITHUB_EVENT_NAME": os.getenv("GITHUB_EVENT_NAME"),
            "GITHUB_WORKFLOW": os.getenv("GITHUB_WORKFLOW"),
            "GITHUB_ACTOR": os.getenv("GITHUB_ACTOR"),
            "GITHUB_REPOSITORY": os.getenv("GITHUB_REPOSITORY"),
            "GITHUB_REPOSITORY_OWNER": os.getenv("GITHUB_REPOSITORY_OWNER"),
        }
        LOGGER.info("\n" + "\n".join(f"{k}: {v}" for k, v in github_info.items()))
        info_dict["GitHub Info"] = github_info
    return info_dict


def check_amp(model):
    """检查 YOLO model 的 PaddlePaddle Automatic Mixed Precision（AMP）功能。

    若检查失败，表示系统上的 AMP 存在异常，可能导致 NaN losses 或 zero-mAP results，因此 training
    期间将禁用 AMP。

    参数:
        model (paddle.nn.Layer): YOLO model instance。

    返回:
        (bool): 若 AMP 功能可与 YOLO model 正常工作，则返回 True；否则返回 False。

    示例:
        >>> from ddyolo26 import YOLO
        >>> from ddyolo26.utils.checks import check_amp
        >>> model = YOLO("weights/yolov8/yolov8n.pdparams").model.cuda()
        >>> check_amp(model)
    """
    from ddyolo26.utils.runtime import autocast

    device = model.parameters()[0].place if len(model.parameters()) > 0 else paddle.CPUPlace()
    prefix = colorstr("AMP: ")
    if getattr(device, "is_cpu_place", lambda: False)():
        return False
    else:
        pattern = re.compile(
            "(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)",
            re.IGNORECASE,
        )
        try:
            device_id = int(str(device).split(":")[1].replace(")", ""))
        except Exception:
            device_id = 0
        gpu = paddle.cuda.get_device_name(device_id)
        if bool(pattern.search(gpu)):
            LOGGER.warning(
                f"{prefix}检查失败 ❌。在 {gpu} GPU 上进行 AMP training 可能导致 NaN losses 或 zero-mAP results，因此 training 期间将禁用 AMP。"
            )
            return False

    LOGGER.info(f"{prefix}正在运行 Automatic Mixed Precision (AMP) 检查...")
    LOGGER.info(f"{prefix}检查通过 ✅")
    return True


def check_multiple_install():
    """检查是否存在多个 PaddleYOLO-RKNN installations。"""
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "ddyolo26"],
            capture_output=True,
            text=True,
        )
        install_msg = (
            f"请从 PaddleYOLO-RKNN repository root 运行 commands，并避免混用多个 local copies。见 {PROJECT_SITE}"
        )
        if result.returncode != 0:
            return
        yolo_path = (Path(re.findall("location:\\s+(.+)", result.stdout, flags=re.I)[-1]) / "ddyolo26").resolve()
        if not yolo_path.samefile(ROOT.resolve()):
            LOGGER.warning(
                f"检测到多个 PaddleYOLO-RKNN installations。`yolo` command 使用的是: {yolo_path}，但当前 session 从 {ROOT} import。这可能导致 version conflicts。{install_msg}"
            )
    except Exception:
        return


def print_args(args: (dict | None) = None, show_file=True, show_func=False):
    """打印 function arguments（可选 args dict）。

    参数:
        args (dict, optional): 要打印的 arguments。
        show_file (bool): 是否显示 file name。
        show_func (bool): 是否显示 function name。
    """

    def strip_auth(v):
        """清理较长 URL，移除潜在 authentication information。"""
        return clean_url(v) if isinstance(v, str) and v.startswith("http") and len(v) > 100 else v

    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={strip_auth(v)}" for k, v in sorted(args.items())))


def cuda_device_count() -> int:
    """获取 environment 中可用 NVIDIA GPUs 数量。

    返回:
        (int): 可用 NVIDIA GPUs 数量。
    """
    if IS_JETSON:
        return paddle.cuda.device_count()
    else:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"],
                encoding="utf-8",
            )
            first_line = output.strip().split("\n", 1)[0]
            return int(first_line)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return 0


def cuda_is_available() -> bool:
    """检查 environment 中 CUDA 是否可用。

    返回:
        (bool): 若存在一个或多个 NVIDIA GPUs，则返回 True；否则返回 False。
    """
    return cuda_device_count() > 0


def is_rockchip():
    """检查当前 environment 是否运行在 Rockchip SoC 上。

    返回:
        (bool): 若运行在 Rockchip SoC 上，则返回 True；否则返回 False。
    """
    if LINUX and ARM64:
        try:
            with open("/proc/device-tree/compatible") as f:
                dev_str = f.read()
                *_, soc = dev_str.split(",")
                if soc.replace("\x00", "").split("-", 1)[0] in RKNN_CHIPS:
                    return True
        except OSError:
            return False
    else:
        return False


def is_intel():
    """检查 system 是否具有 Intel hardware（CPU 或 GPU）。

    返回:
        (bool): 若检测到 Intel hardware，则返回 True；否则返回 False。
    """
    from ddyolo26.utils.runtime import get_cpu_info

    if "intel" in get_cpu_info().lower():
        return True
    try:
        result = subprocess.run(["xpu-smi", "discovery"], capture_output=True, text=True, timeout=5)
        return "intel" in result.stdout.lower()
    except Exception:
        return False


def is_sudo_available() -> bool:
    """检查 environment 中 sudo command 是否可用。

    返回:
        (bool): 若 sudo command 可用，则返回 True；否则返回 False。
    """
    if WINDOWS:
        return False
    cmd = "sudo --version"
    return subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


check_python("3.8", hard=False, verbose=True)
IS_PYTHON_3_8 = PYTHON_VERSION.startswith("3.8")
IS_PYTHON_3_9 = PYTHON_VERSION.startswith("3.9")
IS_PYTHON_3_10 = PYTHON_VERSION.startswith("3.10")
IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")
IS_PYTHON_3_13 = PYTHON_VERSION.startswith("3.13")
IS_PYTHON_MINIMUM_3_9 = check_python("3.9", hard=False)
IS_PYTHON_MINIMUM_3_10 = check_python("3.10", hard=False)
IS_PYTHON_MINIMUM_3_12 = check_python("3.12", hard=False)
