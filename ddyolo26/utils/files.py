# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 文件系统工具：Path 扩展、工作目录管理、文件大小等辅助函数。
@details
提供：
- `increment_path()`：自动递增文件名（run1 → run2 → …）
- `file_size()`：返回文件/目录大小（MB）
- `get_latest_run()`：查找最新训练结果目录
"""

import paddle

import contextlib
import glob
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class WorkingDirectory(contextlib.ContextDecorator):
    """用于临时切换 working directory 的 context manager 与 decorator。

    该类允许通过 context manager 或 decorator 临时切换 working directory，并确保 context 或被装饰函数结束后
    恢复原始 working directory。

    属性:
        dir (Path | str): 要切换到的新 directory。
        cwd (Path): 切换前的原始 current working directory。

    方法:
        __enter__: 将 current directory 切换到指定 directory。
        __exit__: context 退出时恢复原始 working directory。

    示例:
        作为 context manager 使用:
        >>> with WorkingDirectory("path/to/new/dir"):
        ...     # 在新 directory 中执行操作
        ...     pass

        作为 decorator 使用:
        >>> @WorkingDirectory("path/to/new/dir")
        ... def some_function():
        ...     # 在新 directory 中执行操作
        ...     pass
    """

    def __init__(self, new_dir: (str | Path)):
        """使用目标 directory 初始化 WorkingDirectory context manager。"""
        self.dir = new_dir
        self.cwd = Path.cwd().resolve()

    def __enter__(self):
        """进入 context 时将 current working directory 切换到指定 directory。"""
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 context 时恢复原始 working directory。"""
        os.chdir(self.cwd)


@contextmanager
def spaces_in_path(path: (str | Path)):
    """处理名称中带 spaces 的 paths 的 context manager。

    如果 path 包含 spaces，则将其替换为 underscores，把 file/directory 复制到新 path，执行 context code block，
    然后再将 file/directory 复制回原始位置。

    参数:
        path (str | Path): 可能包含 spaces 的原始 path。

    生成:
        (Path | str): 将 spaces 替换为 underscores 后的 temporary path。

    示例:
        >>> with spaces_in_path("path/with spaces") as new_path:
        ...     # 在这里执行代码
        ...     pass
    """
    if " " in str(path):
        string = isinstance(path, str)
        path = Path(path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / path.name.replace(" ", "_")
            if path.is_dir():
                shutil.copytree(path, tmp_path)
            elif path.is_file():
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, tmp_path)
            try:
                yield str(tmp_path) if string else tmp_path
            finally:
                if tmp_path.is_dir():
                    shutil.copytree(tmp_path, path, dirs_exist_ok=True)
                elif tmp_path.is_file():
                    shutil.copy2(tmp_path, path)
    else:
        yield path


def increment_path(path: (str | Path), exist_ok: bool = False, sep: str = "", mkdir: bool = False) -> Path:
    """递增 file 或 directory path，例如 runs/exp --> runs/exp{sep}2、runs/exp{sep}3 等。

    如果 path 已存在且 `exist_ok` 不为 True，则会在 path 末尾附加数字和 `sep` 进行递增。
    如果 path 是 file，会保留 file extension；如果 path 是 directory，会直接在 path 末尾追加数字。

    参数:
        path (str | Path): 要递增的 path。
        exist_ok (bool, optional): 若为 True，则不递增 path，直接原样返回。
        sep (str, optional): path 与递增数字之间使用的 separator。
        mkdir (bool, optional): 如果 directory 不存在则创建。

    返回:
        (Path): 递增后的 path。

    示例:
        递增 directory path:
        >>> from pathlib import Path
        >>> path = Path("runs/exp")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp2

        递增 file path:
        >>> path = Path("runs/exp/results.txt")
        >>> new_path = increment_path(path)
        >>> print(new_path)
        runs/exp/results2.txt
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def file_age(path: (str | Path) = __file__) -> int:
    """返回指定 file 自上次修改以来经过的天数。"""
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)
    return dt.days


def file_date(path: (str | Path) = __file__) -> str:
    """以 'YYYY-M-D' format 返回 file modification date。"""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path: (str | Path)) -> float:
    """返回 file 或 directory 的大小，单位为 mebibytes (MiB)。"""
    if isinstance(path, (str, Path)):
        mb = 1 << 20
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    return 0.0


def get_latest_run(search_dir: str = ".") -> str:
    """返回指定 directory 中最新 'last.pdparams' file 的 path，用于 resume training。"""
    last_list = glob.glob(f"{search_dir}/**/last*.pdparams", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def update_models(
    model_names: tuple = ("weights/yolov8/yolov8n.pdparams",),
    source_dir: Path = Path("."),
    update_names: bool = False,
):
    """更新指定 YOLO models，并重新保存到 'updated_models' subdirectory。

    参数:
        model_names (tuple, optional): 要更新的 model filenames。
        source_dir (Path, optional): 包含 models 与 target subdirectory 的 directory。
        update_names (bool, optional): 从 data YAML 更新 model names。

    示例:
        更新指定 YOLO models 并保存到 'updated_models' subdirectory:
        >>> from ddyolo26.utils.files import update_models
        >>> model_names = ("weights/yolov8/yolov8n.pdparams", "weights/yolov8/yolov8s.pdparams")
        >>> update_models(model_names, source_dir=Path("."), update_names=True)
    """
    from ddyolo26 import YOLO
    from ddyolo26.nn.autobackend import default_class_names
    from ddyolo26.utils import LOGGER

    target_dir = source_dir / "updated_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    for model_name in model_names:
        model_path = source_dir / model_name
        LOGGER.info(f"正在从 {model_path} 加载 model")
        model = YOLO(model_path)
        model.half()
        if update_names:
            model.model.names = default_class_names("coco8.yaml")
        save_path = target_dir / model_name
        LOGGER.info(f"正在将 {model_name} model 重新保存到 {save_path}")
        model.save(save_path)
