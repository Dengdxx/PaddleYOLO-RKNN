# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 文件下载工具：下载数据集和普通资源（GitHub release / Google Drive 等）。
@details
提供带进度条、重试、完整性校验的下载功能：
- `safe_download()`：支持断点续传、校验 SHA256
- `attempt_download_asset()`：普通资源从 GitHub Release 下载；模型权重不自动下载
- `get_github_assets()`：列举可用的 release 资源
"""

import re
import shutil
import subprocess
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib import parse, request

from ddyolo26.utils import ASSETS_URL, LOGGER, TQDM, checks, clean_url, emojis, is_online, url2file

GITHUB_ASSETS_REPO = "ultralytics/assets"  # upstream asset repository for sample data only
GITHUB_ASSETS_NAMES = frozenset(
    [
        "bus.jpg",
        "zidane.jpg",
        "calibration_image_sample_data_20x128x128x3_float32.npy.zip",
    ]
)
GITHUB_ASSETS_STEMS = frozenset(k.rpartition(".")[0] for k in GITHUB_ASSETS_NAMES)

# ── ddyolo26 Paddle 预训练权重注册表（仅本地解析，不自动下载） ────────────────
PADDLE_WEIGHT_FILENAMES = {
    "yolo11n": "yolo11n.pdparams",
    "yolo11s": "yolo11s.pdparams",
    "yolo11m": "yolo11m.pdparams",
    "yolo11l": "yolo11l.pdparams",
    "yolo11x": "yolo11x.pdparams",
    "yolo11n-seg": "yolo11n-seg.pdparams",
    "yolo11s-seg": "yolo11s-seg.pdparams",
    "yolo11m-seg": "yolo11m-seg.pdparams",
    "yolo11l-seg": "yolo11l-seg.pdparams",
    "yolo11x-seg": "yolo11x-seg.pdparams",
    "yolo26n": "yolo26n.pdparams",
    "yolo26s": "yolo26s.pdparams",
    "yolo26m": "yolo26m.pdparams",
    "yolo26l": "yolo26l.pdparams",
    "yolo26x": "yolo26x.pdparams",
    "yolo26n-seg": "yolo26n-seg.pdparams",
    "yolo26s-seg": "yolo26s-seg.pdparams",
    "yolo26m-seg": "yolo26m-seg.pdparams",
    "yolo26l-seg": "yolo26l-seg.pdparams",
    "yolo26x-seg": "yolo26x-seg.pdparams",
    "yolov8n": "yolov8n.pdparams",
    "yolov8s": "yolov8s.pdparams",
    "yolov8m": "yolov8m.pdparams",
    "yolov8l": "yolov8l.pdparams",
    "yolov8x": "yolov8x.pdparams",
    "yolov8n-seg": "yolov8n-seg.pdparams",
    "yolov8s-seg": "yolov8s-seg.pdparams",
    "yolov8m-seg": "yolov8m-seg.pdparams",
    "yolov8l-seg": "yolov8l-seg.pdparams",
    "yolov8x-seg": "yolov8x-seg.pdparams",
}
PADDLE_WEIGHT_SUPPORTED_NAMES = ", ".join(PADDLE_WEIGHT_FILENAMES.keys())
PADDLE_WEIGHT_BUNDLED_NAMES = tuple(name for name in PADDLE_WEIGHT_FILENAMES if name.startswith(("yolo26", "yolov8")))


def paddle_weight_group(model: str) -> str:
    """返回已注册 Paddle weight name 的 canonical directory group。"""
    stem = Path(model).stem
    base = re.sub(r"_paddle$", "", stem)
    match = re.match(r"^(yolo(?:11|26)|yolov8)[nslmx](?:-(seg|cls|pose|obb))?$", base)
    if not match:
        return base
    family, task = match.groups()
    return f"{family}{task or ''}"


def paddle_weight_path(model: str) -> Path:
    """返回已注册 model name 的 canonical local Paddle weight path。"""
    stem = Path(model).stem
    base = re.sub(r"_paddle$", "", stem)
    filename = PADDLE_WEIGHT_FILENAMES.get(base, f"{base}.pdparams")
    return Path("weights") / paddle_weight_group(base) / filename


def paddle_weight_lfs_pull_command(model: str) -> str:
    """返回拉取单个已注册 Paddle weight 的 Git LFS command。"""
    return f'git lfs pull --include="{paddle_weight_path(model).as_posix()}"'


def is_url(url: (str | Path), check: bool = False) -> bool:
    """验证给定 string 是否为 URL，并可选检查该 URL 是否 online 存在。

    参数:
        url (str | Path): 待验证为 URL 的 string。
        check (bool, optional): 若为 True，额外检查 URL 是否 online 存在。

    返回:
        (bool): 有效 URL 返回 True。若 'check' 为 True，则仅在 URL online 存在时返回 True。

    示例:
        >>> valid = is_url("https://www.example.com")
        >>> valid_and_exists = is_url("https://www.example.com", check=True)
    """
    try:
        url = str(url)
        result = parse.urlparse(url)
        if not (result.scheme and result.netloc):
            return False
        if check:
            r = request.urlopen(request.Request(url, method="HEAD"), timeout=3)
            return 200 <= r.getcode() < 400
        return True
    except Exception:
        return False


def delete_dsstore(path: (str | Path), files_to_delete: tuple[str, ...] = (".DS_Store", "__MACOSX")) -> None:
    """删除 directory 中所有指定 system files。

    参数:
        path (str | Path): 要删除 files 的 directory path。
        files_to_delete (tuple[str, ...]): 要删除的 files。

    示例:
        >>> from ddyolo26.utils.downloads import delete_dsstore
        >>> delete_dsstore("path/to/dir")

    说明:
        ".DS_Store" files 由 Apple operating system 创建，包含 folders 与 files 的 metadata。
        它们是 hidden system files，在不同 operating systems 间传输 files 时可能造成问题。
    """
    for file in files_to_delete:
        matches = list(Path(path).rglob(file))
        LOGGER.info(f"正在删除 {file} files: {matches}")
        for f in matches:
            f.unlink()


def zip_directory(
    directory: (str | Path),
    compress: bool = True,
    exclude: tuple[str, ...] = (".DS_Store", "__MACOSX"),
    progress: bool = True,
) -> Path:
    """压缩 directory 内容，并排除指定 files。

    生成的 zip file 以 directory 命名，并放在同级位置。

    参数:
        directory (str | Path): 待 zip 的 directory path。
        compress (bool): zipping 时是否压缩 files。
        exclude (tuple[str, ...], optional): 要排除的 filename strings tuple。
        progress (bool, optional): 是否显示 progress bar。

    返回:
        (Path): 生成 zip file 的 path。

    示例:
        >>> from ddyolo26.utils.downloads import zip_directory
        >>> file = zip_directory("path/to/dir")
    """
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"目录 '{directory}' 不存在。")
    files = [f for f in directory.rglob("*") if f.is_file() and all(x not in f.name for x in exclude)]
    zip_file = directory.with_suffix(".zip")
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, "w", compression) as f:
        for file in TQDM(
            files,
            desc=f"正在将 {directory} zip 到 {zip_file}...",
            unit="files",
            disable=not progress,
        ):
            f.write(file, file.relative_to(directory))
    return zip_file


def unzip_file(
    file: (str | Path),
    path: (str | Path | None) = None,
    exclude: tuple[str, ...] = (".DS_Store", "__MACOSX"),
    exist_ok: bool = False,
    progress: bool = True,
) -> Path:
    """将 *.zip file 解压到指定 path，并排除指定 files。

    若 zipfile 不包含单个 top-level directory，该函数会创建一个与 zipfile 同名（不含 extension）的新 directory，
    并将内容解压进去。若未提供 path，则默认使用 zipfile 的 parent directory。

    参数:
        file (str | Path): 待 extract 的 zipfile path。
        path (str | Path, optional): zipfile 解压到的 path。
        exclude (tuple[str, ...], optional): 要排除的 filename strings tuple。
        exist_ok (bool, optional): 若目标内容已存在，是否允许覆盖。
        progress (bool, optional): 是否显示 progress bar。

    返回:
        (Path): zipfile 解压后 directory 的 path。

    异常:
        BadZipFile: 当提供的 file 不存在或不是有效 zipfile 时抛出。

    示例:
        >>> from ddyolo26.utils.downloads import unzip_file
        >>> directory = unzip_file("path/to/file.zip")
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' 不存在或不是有效 zip file。")
    if path is None:
        path = Path(file).parent
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}
        unzip_as_dir = len(top_level_dirs) == 1
        if unzip_as_dir:
            extract_path = path
            path = Path(path) / next(iter(top_level_dirs))
        else:
            path = extract_path = Path(path) / Path(file).stem
        if path.exists() and any(path.iterdir()) and not exist_ok:
            LOGGER.warning(f"跳过 {file} unzip，因为目标 directory {path} 非空。")
            return path
        for f in TQDM(
            files,
            desc=f"正在将 {file} unzip 到 {Path(path).resolve()}...",
            unit="files",
            disable=not progress,
        ):
            if ".." in Path(f).parts:
                LOGGER.warning(f"潜在不安全 file path: {f}，跳过 extraction。")
                continue
            zipObj.extract(f, extract_path)
    return path


def check_disk_space(file_bytes: int, path: (str | Path) = Path.cwd(), sf: float = 1.5, hard: bool = True) -> bool:
    """检查是否有足够 disk space 下载并存储 file。

    参数:
        file_bytes (int): file size，单位 bytes。
        path (str | Path, optional): 用于检查 available free space 的 path 或 drive。
        sf (float, optional): safety factor，即 required free space 的倍数。
        hard (bool, optional): disk space 不足时是否抛出 error。

    返回:
        (bool): 若 disk space 足够，则返回 True；否则返回 False。
    """
    _total, _used, free = shutil.disk_usage(path)
    if file_bytes * sf < free:
        return True

    def fmt_bytes(b):
        return f"{b / (1 << 20):.1f} MB" if b < 1 << 30 else f"{b / (1 << 30):.3f} GB"

    text = f"可用磁盘空间不足：{fmt_bytes(free)} < 需要 {fmt_bytes(int(file_bytes * sf))}。请额外释放 {fmt_bytes(int(file_bytes * sf - free))} 磁盘空间后重试。"
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False


def get_google_drive_file_info(link: str) -> tuple[str, str | None]:
    """获取可分享 Google Drive file link 的 direct download link 与 filename。

    参数:
        link (str): Google Drive file 的 shareable link。

    返回:
        url (str): Google Drive file 的 direct download URL。
        filename (str | None): Google Drive file 的 original filename。若 filename 提取失败，则返回 None。

    示例:
        >>> from ddyolo26.utils.downloads import get_google_drive_file_info
        >>> link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        >>> url, filename = get_google_drive_file_info(link)
    """
    import requests

    file_id = link.split("/d/")[1].split("/view", 1)[0]
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    filename = None
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if "quota exceeded" in str(response.content.lower()):
            raise ConnectionError(
                emojis(f"❌  Google Drive file download quota 已超限。请稍后重试，或手动从 {link} 下载此 file。")
            )
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                drive_url += f"&confirm={v}"
        if cd := response.headers.get("content-disposition"):
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename


def safe_download(
    url: (str | Path),
    file: (str | Path | None) = None,
    dir: (str | Path | None) = None,
    unzip: bool = True,
    delete: bool = False,
    curl: bool = False,
    retry: int = 3,
    min_bytes: float = 1.0,
    exist_ok: bool = False,
    progress: bool = True,
) -> Path | str:
    """从 URL 下载 files，并支持 retry、unzip 与删除 downloaded file。

    该函数使用 Content-Length validation 增强 partial download detection。

    参数:
        url (str | Path): 待下载 file 的 URL。
        file (str | Path, optional): downloaded file 的 filename。若未提供，则使用 URL 同名保存。
        dir (str | Path, optional): 保存 downloaded file 的 directory。若未提供，则保存到当前 working directory。
        unzip (bool, optional): 是否 unzip downloaded file。
        delete (bool, optional): unzip 后是否删除 downloaded file。
        curl (bool, optional): 是否使用 curl command line tool 下载。
        retry (int, optional): 下载失败时的 retry 次数。
        min_bytes (float, optional): downloaded file 视为成功下载所需的最小 bytes 数。
        exist_ok (bool, optional): unzip 时是否覆盖已存在内容。
        progress (bool, optional): 下载期间是否显示 progress bar。

    返回:
        (Path | str): downloaded file 或 extracted directory 的 path。

    示例:
        >>> from ddyolo26.utils.downloads import safe_download
        >>> link = "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"
        >>> path = safe_download(link)
    """
    gdrive = url.startswith("https://drive.google.com/")
    if gdrive:
        url, file = get_google_drive_file_info(url)
    url = url.replace(" ", "%20")
    f = Path(dir or ".") / (file or url2file(url))
    if "://" not in str(url) and Path(url).is_file():
        f = Path(url)
    elif not f.is_file():
        uri = (url if gdrive else clean_url(url)).replace(ASSETS_URL, ASSETS_URL)
        desc = f"正在下载 {uri} 到 '{f}'"
        f.parent.mkdir(parents=True, exist_ok=True)
        curl_installed = shutil.which("curl")
        for i in range(retry + 1):
            try:
                if (curl or i > 0) and curl_installed:
                    s = "sS" * (not progress)
                    r = subprocess.run(
                        [
                            "curl",
                            "-#",
                            f"-{s}fL",
                            url,
                            "-o",
                            f,
                            "--retry",
                            "3",
                            "-C",
                            "-",
                        ]
                    ).returncode
                    assert r == 0, f"Curl return value {r}"
                    expected_size = None
                else:
                    with request.urlopen(url) as response:
                        expected_size = int(response.getheader("Content-Length", 0))
                        if i == 0 and expected_size > 1048576:
                            check_disk_space(expected_size, path=f.parent)
                        buffer_size = max(8192, min(1048576, expected_size // 1000)) if expected_size else 8192
                        with TQDM(
                            total=expected_size,
                            desc=desc,
                            disable=not progress,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as pbar:
                            with open(f, "wb") as f_opened:
                                while True:
                                    data = response.read(buffer_size)
                                    if not data:
                                        break
                                    f_opened.write(data)
                                    pbar.update(len(data))
                if f.exists():
                    file_size = f.stat().st_size
                    if file_size > min_bytes:
                        if expected_size and file_size != expected_size:
                            LOGGER.warning(
                                f"下载不完整: {file_size}/{expected_size} bytes ({file_size / expected_size * 100:.1f}%)"
                            )
                        else:
                            break
                    f.unlink()
            except MemoryError:
                raise
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(emojis(f"❌  下载 {uri} 失败。运行环境可能离线。")) from e
                elif i >= retry:
                    raise ConnectionError(emojis(f"❌  下载 {uri} 失败。已达到重试次数上限。{e}")) from e
                LOGGER.warning(f"下载失败，正在 retry {i + 1}/{retry} {uri}... {e}")
        else:  # for-else：循环未 break 即结束，表示 download 从未成功
            if not f.is_file():
                raise ConnectionError(emojis(f"❌  下载 {uri} 失败。{retry + 1} 次尝试后 file 仍未保存。"))
    if unzip and f.exists() and f.suffix in {"", ".zip", ".tar", ".gz"}:
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve()
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)
        elif f.suffix in {".tar", ".gz"}:
            LOGGER.info(f"正在将 {f} unzip 到 {unzip_dir}...")
            subprocess.run(
                [
                    "tar",
                    "xf" if f.suffix == ".tar" else "xfz",
                    f,
                    "--directory",
                    unzip_dir,
                ],
                check=True,
            )
        if delete:
            f.unlink()
        return unzip_dir
    return f


def get_github_assets(
    repo: str = GITHUB_ASSETS_REPO, version: str = "latest", retry: bool = False
) -> tuple[str, list[str]]:
    """从 GitHub repository 获取指定 version 的 tag 与 assets。

    若未指定 version，该函数会获取 latest release assets。

    参数:
        repo (str, optional): 'owner/repo' format 的 GitHub repository。
        version (str, optional): 获取 assets 的 release version。
        retry (bool, optional): request 失败时是否 retry。

    返回:
        tag (str): release tag。
        assets (list[str]): asset names 列表。

    示例:
        >>> tag, assets = get_github_assets(version="latest")
    """
    import requests

    if version != "latest":
        version = f"tags/{version}"
    url = f"https://api.github.com/repos/{repo}/releases/{version}"
    r = requests.get(url)
    if r.status_code != 200 and r.reason != "rate limit exceeded" and retry:
        r = requests.get(url)
    if r.status_code != 200:
        LOGGER.warning(f"GitHub assets check 失败: {url}: {r.status_code} {r.reason}")
        return "", []
    data = r.json()
    return data["tag_name"], [x["name"] for x in data["assets"]]


def attempt_download_asset(
    file: (str | Path),
    repo: str = GITHUB_ASSETS_REPO,
    release: str = "v8.4.0",
    **kwargs,
) -> str:
    """若本地未找到 file，则尝试从 GitHub release assets 下载。

    参数:
        file (str | Path): 待下载 filename 或 file path。
        repo (str, optional): 'owner/repo' format 的 GitHub repository。
        release (str, optional): 要下载的特定 release version。
        **kwargs (Any): download process 使用的额外 keyword arguments。

    返回:
        (str): downloaded file 的 path。

    示例:
        >>> file_path = attempt_download_asset("bus.jpg", release="v0.0.0")
    """
    from ddyolo26.utils import SETTINGS

    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ""))
    if file.exists():
        return str(file)
    elif (SETTINGS["weights_dir"] / file).exists():
        return str(SETTINGS["weights_dir"] / file)
    else:
        name = Path(parse.unquote(str(file))).name
        if name.endswith(".pt") and not name.endswith("_paddle.pt"):
            raise FileNotFoundError(
                f"模型权重不会自动下载: {name}. 请使用本地 Paddle 权重 (*.pdparams 或 *_paddle.pt)。"
            )
        download_url = f"https://github.com/{repo}/releases/download"
        if str(file).startswith(("http:/", "https:/")):
            url = str(file).replace(":/", "://")
            file = url2file(name)
            if Path(file).is_file():
                LOGGER.info(f"已在本地 {file} 找到 {clean_url(url)}")
            else:
                safe_download(url=url, file=file, min_bytes=100000.0, **kwargs)
        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(
                url=f"{download_url}/{release}/{name}",
                file=file,
                min_bytes=100000.0,
                **kwargs,
            )
        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)
            if name in assets:
                safe_download(
                    url=f"{download_url}/{tag}/{name}",
                    file=file,
                    min_bytes=100000.0,
                    **kwargs,
                )
        return str(file)


def resolve_paddle_weight(model: str) -> str | None:
    """解析本地 ddyolo26 Paddle 预训练权重，不执行网络下载。"""
    import re as _re

    stem = Path(model).stem  # "yolo26n_paddle" / "yolo26n" / ...
    base = _re.sub(r"_paddle$", "", stem)

    path = Path(model)
    min_weight_bytes = 1024
    if path.exists() and path.stat().st_size > min_weight_bytes:
        return str(path)

    if base not in PADDLE_WEIGHT_FILENAMES:
        return None

    canonical = PADDLE_WEIGHT_FILENAMES[base]
    candidates = [
        paddle_weight_path(base),
        Path("weights") / paddle_weight_group(base) / f"{base}_paddle.pt",
        Path("weights/pdparams") / canonical,
        Path("weights/pdparams") / f"{base}_paddle.pt",
        Path(canonical),
        Path(f"{base}_paddle.pt"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > min_weight_bytes:
            return str(candidate)
    return None


def _infer_yaml_cfg(stem: str) -> str:
    """根据模型名称推断对应的 YAML 配置路径。

    参数:
        stem (str): 模型文件名 stem（小写）。

    返回:
        (str): YAML 配置文件相对路径。
    """
    stem = stem.lower()
    if stem.startswith("yolov8") and "-seg" in stem:
        return "ddyolo26/cfg/models/v8/yolov8-seg.yaml"
    elif stem.startswith("yolov8"):
        return "ddyolo26/cfg/models/v8/yolov8.yaml"
    elif stem.startswith("yolo11") and "-seg" in stem:
        return "ddyolo26/cfg/models/11/yolo11-seg.yaml"
    elif stem.startswith("yolo11"):
        return "ddyolo26/cfg/models/11/yolo11.yaml"
    elif "-seg" in stem:
        return "ddyolo26/cfg/models/26/yolo26-seg.yaml"
    else:
        return "ddyolo26/cfg/models/26/yolo26.yaml"


def download(
    url: (str | list[str] | Path),
    dir: Path = Path.cwd(),
    unzip: bool = True,
    delete: bool = False,
    curl: bool = False,
    threads: int = 1,
    retry: int = 3,
    exist_ok: bool = False,
) -> None:
    """从指定 URLs 下载 files 到给定 directory。

    若指定多个 threads，则支持 concurrent downloads。

    参数:
        url (str | list[str] | Path): 待下载 files 的 URL 或 URL 列表。
        dir (Path, optional): 保存 files 的 directory。
        unzip (bool, optional): 下载后是否 unzip files。
        delete (bool, optional): extraction 后是否删除 zip files。
        curl (bool, optional): 是否使用 curl 下载。
        threads (int, optional): concurrent downloads 使用的 threads 数量。
        retry (int, optional): download 失败时的 retries 数。
        exist_ok (bool, optional): unzip 时是否覆盖已存在内容。

    示例:
        >>> download("https://github.com/ultralytics/assets/releases/download/v0.0.0/example.zip", dir="path/to/dir", unzip=True)
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    urls = [url] if isinstance(url, (str, Path)) else url
    if threads > 1:
        LOGGER.info(f"正在使用 {threads} threads 下载 {len(urls)} 个 file(s) 到 {dir}...")
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0],
                    dir=x[1],
                    unzip=unzip,
                    delete=delete,
                    curl=curl,
                    retry=retry,
                    exist_ok=exist_ok,
                    progress=True,
                ),
                zip(urls, repeat(dir)),
            )
            pool.close()
            pool.join()
    else:
        for u in urls:
            safe_download(
                url=u,
                dir=dir,
                unzip=unzip,
                delete=delete,
                curl=curl,
                retry=retry,
                exist_ok=exist_ok,
            )


if __name__ == "__main__":
    """命令行检查本地预训练权重。

    用法:
        # 解析单个本地模型
        python -m ddyolo26.utils.downloads yolov8n

        # 解析多个本地模型
        python -m ddyolo26.utils.downloads yolov8n yolov8s

        # 检查本仓已入库模型
        python -m ddyolo26.utils.downloads all
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="ddyolo26 Paddle 预训练权重本地解析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"支持的模型: {PADDLE_WEIGHT_SUPPORTED_NAMES}\n"
            "  all   - 检查本仓已入库的 20 个 Paddle 权重\n\n"
            "示例:\n"
            "  python -m ddyolo26.utils.downloads yolov8n\n"
            "  python -m ddyolo26.utils.downloads yolov8n yolov8s yolov8n-seg\n"
            "  python -m ddyolo26.utils.downloads all"
        ),
    )
    parser.add_argument(
        "models",
        nargs="+",
        metavar="MODEL",
        help=f"模型名称（{PADDLE_WEIGHT_SUPPORTED_NAMES}）或 all",
    )
    args = parser.parse_args()

    targets = list(PADDLE_WEIGHT_BUNDLED_NAMES) if "all" in args.models else args.models
    ok, fail = [], []
    for name in targets:
        result = resolve_paddle_weight(name)
        if result:
            ok.append(f"  ✓  {name}  →  {result}")
        else:
            fail.append(f"  ✗  {name}")

    print("\n本地权重解析结果:")
    for line in ok:
        print(line)
    for line in fail:
        print(line)
    if fail:
        print(
            f"\n{len(fail)} 个模型未找到。请将 Paddle 权重放到 weights/<model>/。"
            f"支持的模型: {PADDLE_WEIGHT_SUPPORTED_NAMES}"
        )
        raise SystemExit(1)
