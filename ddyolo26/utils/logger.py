# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 日志工具：TensorBoard/WandB/Comet 等集成日志记录器。
@details
为 BaseTrainer 提供结构化指标记录接口，统一输出到：
- 本地 CSV 文件
- TensorBoard（可选）
- WandB / CometML（可选）
"""

from __future__ import annotations
import sys
import paddle

import logging
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


from ddyolo26.paddle_utils import *

from ddyolo26.utils import LOGGER, MACOS, RANK
from ddyolo26.utils.checks import check_requirements


class ConsoleLogger:
    """捕获 console output，并以 batched streaming 写入 file、API 或 custom callback。

    捕获 stdout/stderr output，并通过智能 deduplication 与可配置 batching 进行 streaming。

    属性:
        destination (str | Path | None): streaming 的 target destination（URL、Path，或 None 表示仅 callback）。
        batch_size (int): flush 前累积的 lines 数量（默认 1，表示 immediate）。
        flush_interval (float): automatic flushes 间隔秒数（默认 5.0）。
        on_flush (callable | None): flush 时以 batched content 调用的可选 callback function。
        active (bool): console capture 当前是否 active。

    示例:
        file logging（immediate）:
        >>> logger = ConsoleLogger("training.log")
        >>> logger.start_capture()
        >>> print("This will be logged")
        >>> logger.stop_capture()

        带 batching 的 API streaming:
        >>> logger = ConsoleLogger("https://api.example.com/logs", batch_size=10)
        >>> logger.start_capture()

        带 batching 的 custom callback:
        >>> def my_handler(content, line_count, chunk_id):
        ...     print(f"Received {line_count} lines")
        >>> logger = ConsoleLogger(on_flush=my_handler, batch_size=5)
        >>> logger.start_capture()
    """

    def __init__(self, destination=None, batch_size=1, flush_interval=5.0, on_flush=None):
        """初始化 console logger，可选 batching。

        参数:
            destination (str | Path | None): API endpoint URL (http/https)、local file path，或 None。
            batch_size (int): flush 前累积的 lines 数（1 = immediate，更大值 = batched）。
            flush_interval (float): batching 时两次 flushes 间的最大秒数。
            on_flush (callable | None): custom handling 使用的 Callback(content: str, line_count: int, chunk_id: int)。
        """
        self.destination = destination
        self.is_api = isinstance(destination, str) and destination.startswith(("http://", "https://"))
        if destination is not None and not self.is_api:
            self.destination = Path(destination)
        self.batch_size = max(1, batch_size)
        self.flush_interval = flush_interval
        self.on_flush = on_flush
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.active = False
        self._log_handler = None
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.flush_thread = None
        self.chunk_id = 0
        self.last_line = ""
        self.last_time = 0.0
        self.last_progress_line = ""
        self.last_was_progress = False

    def start_capture(self):
        """开始捕获 console output 并重定向 stdout/stderr。

        说明:
            在 DDP training 中，仅在 rank 0/-1 上启用，以避免 duplicate logging。
        """
        if self.active or RANK not in {-1, 0}:
            return
        self.active = True
        sys.stdout = self._ConsoleCapture(self.original_stdout, self._queue_log)
        sys.stderr = self._ConsoleCapture(self.original_stderr, self._queue_log)
        try:
            self._log_handler = self._LogHandler(self._queue_log)
            logging.getLogger("ddyolo26").addHandler(self._log_handler)
        except Exception:
            pass
        if self.batch_size > 1:
            self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
            self.flush_thread.start()

    def stop_capture(self):
        """停止捕获 console output，并 flush remaining buffer。"""
        if not self.active:
            return
        self.active = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self._log_handler:
            try:
                logging.getLogger("ddyolo26").removeHandler(self._log_handler)
            except Exception:
                pass
            self._log_handler = None
        self._flush_buffer()

    def _queue_log(self, text):
        """将 console text 入队，并进行 deduplication 与 timestamp processing。"""
        if not self.active:
            return
        current_time = time.time()
        if "\r" in text:
            text = text.split("\r")[-1]
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        for line in lines:
            line = line.rstrip()
            if "─" in line:
                continue
            if " ━━" in line:
                is_complete = "100%" in line
                if not is_complete:
                    continue
                parts = line.split()
                seq_key = ""
                if parts:
                    if "/" in parts[0] and parts[0].replace("/", "").isdigit():
                        seq_key = parts[0]
                    elif parts[0] == "Class" and len(parts) > 1:
                        seq_key = f"{parts[0]}_{parts[1]}"
                    elif parts[0] in ("train:", "val:"):
                        seq_key = parts[0]
                if seq_key and self.last_progress_line == f"{seq_key}:done":
                    continue
                if seq_key:
                    self.last_progress_line = f"{seq_key}:done"
                self.last_was_progress = True
            else:
                if not line and self.last_was_progress:
                    self.last_was_progress = False
                    continue
                self.last_was_progress = False
            if line == self.last_line and current_time - self.last_time < 0.1:
                continue
            self.last_line = line
            self.last_time = current_time
            if not line.startswith("[20"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = f"[{timestamp}] {line}"
            should_flush = False
            with self.buffer_lock:
                self.buffer.append(line)
                if len(self.buffer) >= self.batch_size:
                    should_flush = True
            if should_flush:
                self._flush_buffer()

    def _flush_worker(self):
        """周期性 flush buffer 的 background worker。"""
        while self.active:
            time.sleep(self.flush_interval)
            if self.active:
                self._flush_buffer()

    def _flush_buffer(self):
        """将 buffered lines flush 到 destination 和/或 callback。"""
        with self.buffer_lock:
            if not self.buffer:
                return
            lines = self.buffer.copy()
            self.buffer.clear()
            self.chunk_id += 1
            chunk_id = self.chunk_id
        content = "\n".join(lines)
        line_count = len(lines)
        if self.on_flush:
            try:
                self.on_flush(content, line_count, chunk_id)
            except Exception:
                pass
        if self.destination is not None:
            self._write_destination(content)

    def _write_destination(self, content):
        """将 content 写入 file 或 API destination。"""
        try:
            if self.is_api:
                import requests

                payload = {"timestamp": datetime.now().isoformat(), "message": content}
                requests.post(str(self.destination), json=payload, timeout=5)
            else:
                self.destination.parent.mkdir(parents=True, exist_ok=True)
                with self.destination.open("a", encoding="utf-8") as f:
                    f.write(content + "\n")
        except Exception as e:
            print(f"Console logger 写入出错: {e}", file=self.original_stderr)

    class _ConsoleCapture:
        """轻量 stdout/stderr capture。"""

        __slots__ = "callback", "original"

        def __init__(self, original, callback):
            """初始化 stream wrapper，将 writes 重定向到 callback，同时保留 original。"""
            self.original = original
            self.callback = callback

        def write(self, text):
            """将 text 写入 original stream，并转发给 capture callback。"""
            self.original.write(text)
            self.callback(text)

        def flush(self):
            """flush wrapped stream，确保 console capture 期间及时传播 buffered output。"""
            self.original.flush()

    class _LogHandler(logging.Handler):
        """轻量 logging handler。"""

        __slots__ = ("callback",)

        def __init__(self, callback):
            """初始化轻量 logging.Handler，将 log records 转发给给定 callback。"""
            super().__init__()
            self.callback = callback

        def emit(self, record):
            """格式化并转发 LogRecord messages 到 capture callback，用于统一 log streaming。"""
            self.callback(self.format(record) + "\n")


class SystemLogger:
    """记录用于 training monitoring 的 dynamic system metrics。

    捕获 real-time system metrics，包括 CPU、RAM、disk I/O、network I/O 和 NVIDIA GPU statistics，
    用于 training performance monitoring 与 analysis。

    属性:
        pynvml: 成功 import 时的 NVIDIA pynvml module instance，否则为 None。
        nvidia_initialized (bool): NVIDIA GPU monitoring 是否 available 且 initialized。
        net_start: 用于计算 cumulative usage 的 initial network I/O counters。
        disk_start: 用于计算 cumulative usage 的 initial disk I/O counters。

    示例:
        基本用法:
        >>> logger = SystemLogger()
        >>> metrics = logger.get_metrics()
        >>> print(f"CPU: {metrics['cpu']}%, RAM: {metrics['ram']}%")
        >>> if metrics["gpus"]:
        ...     gpu0 = metrics["gpus"]["0"]
        ...     print(f"GPU0: {gpu0['usage']}% usage, {gpu0['temp']}°C")

        training loop integration:
        >>> system_logger = SystemLogger()
        >>> for epoch in range(epochs):
        ...     # training code
        ...     metrics = system_logger.get_metrics()
        ...     # log 到 database/file
    """

    def __init__(self):
        """初始化 system logger。"""
        import psutil

        self.pynvml = None
        self.nvidia_initialized = self._init_nvidia()
        self.net_start = psutil.net_io_counters()
        self.disk_start = psutil.disk_io_counters()
        self._prev_net = self.net_start
        self._prev_disk = self.disk_start
        self._prev_time = time.time()

    def _init_nvidia(self):
        """使用 pynvml 初始化 NVIDIA GPU monitoring。"""
        if MACOS:
            return False
        try:
            check_requirements("nvidia-ml-py>=12.0.0")
            self.pynvml = __import__("pynvml")
            self.pynvml.nvmlInit()
            return True
        except Exception as e:
            if paddle.cuda.is_available():
                LOGGER.warning(f"SystemLogger NVML 初始化失败: {e}")
            return False

    def get_metrics(self, rates=False):
        """获取当前 system metrics，包括 CPU、RAM、disk、network 与 GPU usage。

        收集完整 system metrics，包括 CPU usage、RAM usage、disk I/O statistics、network I/O statistics
        以及 GPU metrics（如果 available）。

        示例输出（rates=False，默认）:
        ```python
        {
            "cpu": 45.2,
            "ram": 78.9,
            "disk": {"read_mb": 156.7, "write_mb": 89.3, "used_gb": 256.8},
            "network": {"recv_mb": 157.2, "sent_mb": 89.1},
            "gpus": {
                "0": {"usage": 95.6, "memory": 85.4, "temp": 72, "power": 285},
                "1": {"usage": 94.1, "memory": 82.7, "temp": 70, "power": 278},
            },
        }
        ```

        示例输出（rates=True）:
        ```python
        {
            "cpu": 45.2,
            "ram": 78.9,
            "disk": {"read_mbs": 12.5, "write_mbs": 8.3, "used_gb": 256.8},
            "network": {"recv_mbs": 5.2, "sent_mbs": 1.1},
            "gpus": {
                "0": {"usage": 95.6, "memory": 85.4, "temp": 72, "power": 285},
            },
        }
        ```

        参数:
            rates (bool): 若为 True，则以 MB/s rates 返回 disk/network，而不是 cumulative MB。

        返回:
            (dict): 包含 cpu、ram、disk、network 与 gpus keys 的 metrics dictionary。

        示例:
            >>> logger = SystemLogger()
            >>> logger.get_metrics()["cpu"]  # CPU 百分比
            >>> logger.get_metrics(rates=True)["network"]["recv_mbs"]  # MB/s 下载速率
        """
        import psutil

        net = psutil.net_io_counters()
        disk = psutil.disk_io_counters()
        memory = psutil.virtual_memory()
        disk_usage = shutil.disk_usage("/")
        now = time.time()
        metrics = {
            "cpu": round(psutil.cpu_percent(), 3),
            "ram": round(memory.percent, 3),
            "gpus": {},
        }
        elapsed = max(0.1, now - self._prev_time)
        if rates:
            metrics["disk"] = {
                "read_mbs": round(
                    max(
                        0,
                        (disk.read_bytes - self._prev_disk.read_bytes) / (1 << 20) / elapsed,
                    ),
                    3,
                ),
                "write_mbs": round(
                    max(
                        0,
                        (disk.write_bytes - self._prev_disk.write_bytes) / (1 << 20) / elapsed,
                    ),
                    3,
                ),
                "used_gb": round(disk_usage.used / (1 << 30), 3),
            }
            metrics["network"] = {
                "recv_mbs": round(
                    max(
                        0,
                        (net.bytes_recv - self._prev_net.bytes_recv) / (1 << 20) / elapsed,
                    ),
                    3,
                ),
                "sent_mbs": round(
                    max(
                        0,
                        (net.bytes_sent - self._prev_net.bytes_sent) / (1 << 20) / elapsed,
                    ),
                    3,
                ),
            }
        else:
            metrics["disk"] = {
                "read_mb": round((disk.read_bytes - self.disk_start.read_bytes) / (1 << 20), 3),
                "write_mb": round((disk.write_bytes - self.disk_start.write_bytes) / (1 << 20), 3),
                "used_gb": round(disk_usage.used / (1 << 30), 3),
            }
            metrics["network"] = {
                "recv_mb": round((net.bytes_recv - self.net_start.bytes_recv) / (1 << 20), 3),
                "sent_mb": round((net.bytes_sent - self.net_start.bytes_sent) / (1 << 20), 3),
            }
        self._prev_net = net
        self._prev_disk = disk
        self._prev_time = now
        if self.nvidia_initialized:
            metrics["gpus"].update(self._get_nvidia_metrics())
        return metrics

    def _get_nvidia_metrics(self):
        """获取 NVIDIA GPU metrics，包括 utilization、memory、temperature 与 power。"""
        gpus = {}
        if not self.nvidia_initialized or not self.pynvml:
            return gpus
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                power = self.pynvml.nvmlDeviceGetPowerUsage(handle) // 1000
                gpus[str(i)] = {
                    "usage": round(util.gpu, 3),
                    "memory": round(memory.used / memory.total * 100, 3),
                    "temp": temp,
                    "power": power,
                }
        except Exception:
            pass
        return gpus


if __name__ == "__main__":
    print("SystemLogger 实时指标监控")
    print("按 Ctrl+C 停止\n")
    logger = SystemLogger()
    try:
        while True:
            metrics = logger.get_metrics()
            print("\x1b[H\x1b[J", end="")
            print(f"CPU: {metrics['cpu']:5.1f}%")
            print(f"RAM: {metrics['ram']:5.1f}%")
            print(f"磁盘读取: {metrics['disk']['read_mb']:8.1f} MB")
            print(f"磁盘写入: {metrics['disk']['write_mb']:7.1f} MB")
            print(f"磁盘已用: {metrics['disk']['used_gb']:8.1f} GB")
            print(f"网络接收: {metrics['network']['recv_mb']:9.1f} MB")
            print(f"网络发送: {metrics['network']['sent_mb']:9.1f} MB")
            if metrics["gpus"]:
                print("\nGPU 指标:")
                for gpu_id, gpu_data in metrics["gpus"].items():
                    print(
                        f"  GPU {gpu_id}: {gpu_data['usage']:3}% | 显存: {gpu_data['memory']:5.1f}% | 温度: {gpu_data['temp']:2}°C | 功耗: {gpu_data['power']:3}W"
                    )
            else:
                print("\nGPU: 未检测到 NVIDIA GPUs")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n已停止 monitoring。")
