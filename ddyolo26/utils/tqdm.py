# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 进度条工具：包装 tqdm，在 Jupyter/终端/CI 环境下自动适配。
@details
根据运行环境选择合适的 tqdm 变体，
在 Notebook 中使用 `tqdm.notebook`，在 CI 中禁用动画效果。
"""

import paddle

import os
import shutil
import sys
import time
import unicodedata
from functools import lru_cache
from typing import IO, Any


@lru_cache(maxsize=1)
def is_noninteractive_console() -> bool:
    """检查已知 non-interactive console environments。"""
    return "GITHUB_ACTIONS" in os.environ or "RUNPOD_POD_ID" in os.environ


class TQDM:
    """PaddleYOLO-RKNN 的轻量 zero-dependency progress bar。

    提供简洁的 rich-style progress bars，适用于 Weights & Biases、console outputs 和其他 logging systems 等环境。
    特性包括零 external dependencies、干净的 single-line output、带 Unicode block characters 的 rich-style
    progress bars、context manager 支持、iterator protocol 支持以及 dynamic description updates。

    属性:
        iterable (Any): 要用 progress bar 包装的 iterable。
        desc (str): progress bar 的 prefix description。
        total (int | None): 预期 iterations 数量。
        disable (bool): 是否禁用 progress bar。
        unit (str): iteration units 的字符串。
        unit_scale (bool): 是否 auto-scale units。
        unit_divisor (int): unit scaling 的 divisor。
        leave (bool): 完成后是否保留 progress bar。
        mininterval (float): 两次 updates 之间的 minimum time interval。
        initial (int): initial counter value。
        n (int): current iteration count。
        closed (bool): progress bar 是否已关闭。
        bar_format (str | None): custom bar format string。
        file (IO[str]): output file stream。

    方法:
        update: 按 n steps 更新 progress。
        set_description: 设置或更新 description。
        set_postfix: 设置 progress bar 的 postfix。
        close: 关闭 progress bar 并清理。
        refresh: 刷新 progress bar display。
        clear: 从 display 清除 progress bar。
        write: 写入 message 且不打断 progress bar。

    示例:
        iterator 基本用法:
        >>> for i in TQDM(range(100)):
        ...     time.sleep(0.01)

        使用 custom description:
        >>> pbar = TQDM(range(100), desc="Processing")
        >>> for i in pbar:
        ...     pbar.set_description(f"Processing item {i}")

        context manager 用法:
        >>> with TQDM(total=100, unit="B", unit_scale=True) as pbar:
        ...     for i in range(100):
        ...         pbar.update(1)

        手动 updates:
        >>> pbar = TQDM(total=100, desc="Training")
        >>> for epoch in range(100):
        ...     # 执行工作
        ...     pbar.update(1)
        >>> pbar.close()
    """

    MIN_RATE_CALC_INTERVAL = 0.01
    RATE_SMOOTHING_FACTOR = 0.3
    MAX_SMOOTHED_RATE = 1000000
    NONINTERACTIVE_MIN_INTERVAL = 60.0

    def __init__(
        self,
        iterable: Any = None,
        desc: (str | None) = None,
        total: (int | None) = None,
        leave: bool = True,
        file: (IO[str] | None) = None,
        mininterval: float = 0.1,
        disable: (bool | None) = None,
        unit: str = "it",
        unit_scale: bool = True,
        unit_divisor: int = 1000,
        bar_format: (str | None) = None,
        initial: int = 0,
        **kwargs,
    ) -> None:
        """使用指定 configuration options 初始化 TQDM progress bar。

        参数:
            iterable (Any, optional): 要用 progress bar 包装的 iterable。
            desc (str, optional): progress bar 的 prefix description。
            total (int, optional): 预期 iterations 数量。
            leave (bool, optional): 完成后是否保留 progress bar。
            file (IO[str], optional): progress display 的 output file stream。
            mininterval (float, optional): updates 之间的 minimum time interval（默认 0.1s，GitHub Actions 中 60s）。
            disable (bool, optional): 是否禁用 progress bar。None 时自动检测。
            unit (str, optional): iteration units 字符串（默认 "it" 表示 items）。
            unit_scale (bool, optional): bytes/data units 是否 auto-scale。
            unit_divisor (int, optional): unit scaling 的 divisor（默认 1000）。
            bar_format (str, optional): custom bar format string。
            initial (int, optional): initial counter value。
            **kwargs (Any): 用于兼容的额外 keyword arguments（忽略）。
        """
        if disable is None:
            try:
                from ddyolo26.utils import LOGGER, VERBOSE

                disable = not VERBOSE or LOGGER.getEffectiveLevel() > 20
            except ImportError:
                disable = False
        self.iterable = iterable
        self.desc = desc or ""
        self.total = total or (len(iterable) if hasattr(iterable, "__len__") else None) or None
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        self.leave = leave
        self.noninteractive = is_noninteractive_console()
        self.mininterval = max(mininterval, self.NONINTERACTIVE_MIN_INTERVAL) if self.noninteractive else mininterval
        self.initial = initial
        self.bar_format = bar_format
        self.file = file or sys.stdout
        self.n = self.initial
        self.last_print_n = self.initial
        self.last_print_t = time.time()
        self.start_t = time.time()
        self.last_rate = 0.0
        self.closed = False
        self.is_bytes = unit_scale and unit in {"B", "bytes"}
        self.scales = (
            [(1073741824, "GB/s"), (1048576, "MB/s"), (1024, "KB/s")]
            if self.is_bytes
            else [
                (1000000000.0, f"G{self.unit}/s"),
                (1000000.0, f"M{self.unit}/s"),
                (1000.0, f"K{self.unit}/s"),
            ]
        )
        if not self.disable and self.total and not self.noninteractive:
            self._display()

    def _format_rate(self, rate: float) -> str:
        """格式化带单位的 rate，并为可读性在 it/s 与 s/it 之间切换。"""
        if rate <= 0:
            return ""
        inv_rate = 1 / rate if rate else None
        if inv_rate and inv_rate > 1:
            return f"{inv_rate:.1f}s/B" if self.is_bytes else f"{inv_rate:.1f}s/{self.unit}"
        fallback = f"{rate:.1f}B/s" if self.is_bytes else f"{rate:.1f}{self.unit}/s"
        return next((f"{rate / t:.1f}{u}" for t, u in self.scales if rate >= t), fallback)

    def _format_num(self, num: (int | float)) -> str:
        """格式化 number，可选进行 unit scaling。"""
        if not self.unit_scale or not self.is_bytes:
            return str(num)
        for unit in ("", "K", "M", "G", "T"):
            if abs(num) < self.unit_divisor:
                return f"{num:3.1f}{unit}B" if unit else f"{num:.0f}B"
            num /= self.unit_divisor
        return f"{num:.1f}PB"

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化 time duration。"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}:{seconds % 60:02.0f}"
        else:
            h, m = int(seconds // 3600), int(seconds % 3600 // 60)
            return f"{h}:{m:02d}:{seconds % 60:02.0f}"

    def _generate_bar(self, width: int = 12) -> str:
        """生成 progress bar。"""
        if self.total is None:
            return "━" * width if self.closed else "─" * width
        frac = min(1.0, self.n / self.total)
        filled = int(frac * width)
        bar = "━" * filled + "─" * (width - filled)
        if filled < width and frac * width - filled > 0.5:
            bar = f"{bar[:filled]}╸{bar[filled + 1 :]}"
        return bar

    @staticmethod
    def _terminal_width() -> int:
        """返回当前 terminal width，并使用保守 fallback。"""
        return max(shutil.get_terminal_size(fallback=(100, 24)).columns, 40)

    @staticmethod
    def _truncate_text(text: str, max_length: int) -> str:
        """截断 text 以适配 available width。"""
        if max_length <= 0:
            return ""
        if TQDM._text_width(text) <= max_length:
            return text
        if max_length == 1:
            return "…"
        truncated = []
        current_width = 0
        target_width = max_length - 1
        for char in text:
            char_width = TQDM._char_width(char)
            if current_width + char_width > target_width:
                break
            truncated.append(char)
            current_width += char_width
        return "".join(truncated) + "…"

    @staticmethod
    def _char_width(char: str) -> int:
        """返回单个 character 的 terminal column width。"""
        return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1

    @classmethod
    def _text_width(cls, text: str) -> int:
        """返回 string 的 terminal column width。"""
        return sum(cls._char_width(char) for char in text)

    def _should_update(self, dt: float, dn: int) -> bool:
        """检查 display 是否应更新。"""
        if self.noninteractive:
            return False
        return self.total is not None and self.n >= self.total or dt >= self.mininterval

    def _display(self, final: bool = False) -> None:
        """显示 progress bar。"""
        if self.disable or self.closed and not final:
            return
        current_time = time.time()
        dt = current_time - self.last_print_t
        dn = self.n - self.last_print_n
        if not final and not self._should_update(dt, dn):
            return
        if dt > self.MIN_RATE_CALC_INTERVAL:
            rate = dn / dt if dt else 0.0
            if rate < self.MAX_SMOOTHED_RATE:
                self.last_rate = self.RATE_SMOOTHING_FACTOR * rate + (1 - self.RATE_SMOOTHING_FACTOR) * self.last_rate
                rate = self.last_rate
        else:
            rate = self.last_rate
        if self.total and self.n >= self.total:
            overall_elapsed = current_time - self.start_t
            if overall_elapsed > 0:
                rate = self.n / overall_elapsed
        self.last_print_n = self.n
        self.last_print_t = current_time
        elapsed = current_time - self.start_t
        remaining_str = ""
        if self.total and 0 < self.n < self.total and elapsed > 0:
            est_rate = rate or self.n / elapsed
            remaining_str = f"<{self._format_time((self.total - self.n) / est_rate)}"
        if self.total:
            percent = self.n / self.total * 100
            n_str = self._format_num(self.n)
            t_str = self._format_num(self.total)
            if self.is_bytes and n_str[-2] == t_str[-2]:
                n_str = n_str.rstrip("KMGTPB")
        else:
            percent = 0.0
            n_str, t_str = self._format_num(self.n), "?"
        elapsed_str = self._format_time(elapsed)
        rate_str = self._format_rate(rate) or (self._format_rate(self.n / elapsed) if elapsed > 0 else "")
        terminal_width = self._terminal_width()
        bar_width = 12
        desc = self.desc
        while True:
            bar = self._generate_bar(bar_width)
            if self.total:
                if self.is_bytes and self.n >= self.total:
                    suffix = f"{percent:.0f}% {bar} {t_str} {rate_str} {elapsed_str}"
                else:
                    suffix = f"{percent:.0f}% {bar} {n_str}/{t_str} {rate_str} {elapsed_str}{remaining_str}"
            else:
                suffix = f"{bar} {n_str} {rate_str} {elapsed_str}"
            separator = ": " if desc else ""
            progress_str = f"{desc}{separator}{suffix}"
            overflow = self._text_width(progress_str) - terminal_width
            if overflow <= 0:
                break
            if bar_width > 6:
                shrink = min(overflow, bar_width - 6)
                bar_width -= shrink
                continue
            available_desc = terminal_width - self._text_width(suffix) - self._text_width(separator)
            desc = self._truncate_text(desc, available_desc)
            progress_str = f"{desc}{separator if desc else ''}{suffix}"
            break
        try:
            if self.noninteractive:
                self.file.write(progress_str)
            else:
                self.file.write(f"\r\x1b[K{progress_str}")
            self.file.flush()
        except Exception:
            pass

    def update(self, n: int = 1) -> None:
        """按 n steps 更新 progress。"""
        if not self.disable and not self.closed:
            self.n += n
            self._display()

    def set_description(self, desc: (str | None)) -> None:
        """设置 description。"""
        self.desc = desc or ""
        if not self.disable:
            self._display()

    def set_postfix(self, **kwargs: Any) -> None:
        """设置 postfix（追加到 description）。"""
        if kwargs:
            postfix = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            base_desc = self.desc.split(" | ")[0] if " | " in self.desc else self.desc
            self.set_description(f"{base_desc} | {postfix}")

    def close(self) -> None:
        """关闭 progress bar。"""
        if self.closed:
            return
        self.closed = True
        if not self.disable:
            if self.total and self.n >= self.total:
                self.n = self.total
                if self.n != self.last_print_n:
                    self._display(final=True)
            else:
                self._display(final=True)
            if self.leave:
                self.file.write("\n")
            else:
                self.file.write("\r\x1b[K")
            try:
                self.file.flush()
            except Exception:
                pass

    def __enter__(self) -> "TQDM":
        """进入 context manager。"""
        return self

    def __exit__(self, *args: Any) -> None:
        """退出 context manager 并关闭 progress bar。"""
        self.close()

    def __iter__(self) -> Any:
        """遍历被包装 iterable，并同步更新 progress。"""
        if self.iterable is None:
            raise TypeError("'NoneType' object is not iterable")
        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __del__(self) -> None:
        """destructor，用于确保 cleanup。"""
        try:
            self.close()
        except Exception:
            pass

    def refresh(self) -> None:
        """刷新 display。"""
        if not self.disable:
            self._display()

    def clear(self) -> None:
        """清除 progress bar。"""
        if not self.disable:
            try:
                self.file.write("\r\x1b[K")
                self.file.flush()
            except Exception:
                pass

    @staticmethod
    def write(s: str, file: (IO[str] | None) = None, end: str = "\n") -> None:
        """写入内容且不打断 progress bar 的 static method。"""
        file = file or sys.stderr
        try:
            file.write(s + end)
            file.flush()
        except Exception:
            pass


if __name__ == "__main__":
    import time

    print("1. 带已知 total 的基础 progress bar:")
    for i in TQDM(range(3), desc="Known total"):
        time.sleep(0.05)
    print("\n2. 带已知 total 的手动 updates:")
    pbar = TQDM(total=300, desc="Manual updates", unit="files")
    for i in range(300):
        time.sleep(0.03)
        pbar.update(1)
        if i % 10 == 9:
            pbar.set_description(f"Processing batch {i // 10 + 1}")
    pbar.close()
    print("\n3. 未知 total 的 progress bar:")
    pbar = TQDM(desc="Unknown total", unit="items")
    for i in range(25):
        time.sleep(0.08)
        pbar.update(1)
        if i % 5 == 4:
            pbar.set_postfix(processed=i + 1, status="OK")
    pbar.close()
    print("\n4. 未知 total 的 context manager:")
    with TQDM(desc="Processing stream", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        for i in range(30):
            time.sleep(0.1)
            pbar.update(1024 * 1024 * i)
    print("\n5. 未知长度的 iterator:")

    def data_stream():
        """模拟未知长度的 data stream。"""
        import random

        for i in range(random.randint(10, 20)):
            yield f"data_chunk_{i}"

    for chunk in TQDM(data_stream(), desc="Stream processing", unit="chunks"):
        time.sleep(0.1)
    print("\n6. File processing simulation（未知 size）:")

    def process_files():
        """模拟处理未知数量的 files。"""
        return [f"file_{i}.txt" for i in range(18)]

    pbar = TQDM(desc="Scanning files", unit="files")
    files = process_files()
    for i, filename in enumerate(files):
        time.sleep(0.06)
        pbar.update(1)
        pbar.set_description(f"Processing {filename}")
    pbar.close()
