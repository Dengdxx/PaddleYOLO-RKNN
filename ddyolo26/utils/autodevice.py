# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 自动设备选择：GPU/CPU 检测与 DDP 多 GPU 设置工具。
@details
封装 Paddle 的设备检测逻辑，为 trainer 提供 `select_device(device_str)` 接口，
支持指定 CUDA ID（`device='0,1'`）或自动选择（`device=''`）。
"""

import paddle

import random
from typing import Any

from ddyolo26.utils import LOGGER
from ddyolo26.utils.checks import check_requirements


class GPUInfo:
    """通过 pynvml 管理 NVIDIA GPU 信息，并提供稳健的错误处理。

    提供查询详细 GPU stats（utilization、memory、temp、power）的方法，并可按可配置条件选择最空闲的 GPU。
    当 pynvml 缺失或初始化失败时，会记录 warning 并禁用相关功能，避免应用崩溃。

    GPU 选择过程中若 NVML 不可用，会通过 `paddle.device.cuda` fallback 获取基础 device 数量。
    NVML 初始化和关闭由本类内部管理。

    属性:
        pynvml (module | None): 成功 import 并初始化后的 `pynvml` module，否则为 `None`。
        nvml_available (bool): `pynvml` 是否可用。import 和 `nvmlInit()` 成功时为 True，否则为 False。
        gpu_stats (list[dict[str, Any]]): GPU stats 字典列表，每个元素对应一个 GPU，在初始化和
            `refresh_stats()` 时填充。key 包括: 'index', 'name', 'utilization' (%), 'memory_used' (MiB),
            'memory_total' (MiB), 'memory_free' (MiB), 'temperature' (C), 'power_draw' (W), 'power_limit' (W or 'N/A')。
            NVML 不可用或查询失败时为空。

    方法:
        refresh_stats: 查询 NVML 并刷新内部 gpu_stats 列表。
        print_status: 使用当前 stats 以紧凑表格打印 GPU status。
        select_idle_gpu: 基于 utilization 和 free memory 选择最空闲 GPU。
        shutdown: 若已初始化 NVML，则将其关闭。

    示例:
        初始化 GPUInfo 并打印 status
        >>> gpu_info = GPUInfo()
        >>> gpu_info.print_status()

        按最低 memory 要求选择 idle GPU
        >>> selected = gpu_info.select_idle_gpu(count=2, min_memory_fraction=0.2)
        >>> print(f"选中的 GPU indices: {selected}")
    """

    def __init__(self):
        """初始化 GPUInfo，并尝试 import 和初始化 pynvml。"""
        self.pynvml: Any | None = None
        self.nvml_available: bool = False
        self.gpu_stats: list[dict[str, Any]] = []
        try:
            check_requirements("nvidia-ml-py>=12.0.0")
            self.pynvml = __import__("pynvml")
            self.pynvml.nvmlInit()
            self.nvml_available = True
            self.refresh_stats()
        except Exception as e:
            LOGGER.warning(f"初始化 pynvml 失败，GPU stats 已禁用: {e}")

    def __del__(self):
        """对象被回收时确保关闭 NVML。"""
        self.shutdown()

    def shutdown(self):
        """若已初始化 NVML，则将其关闭。"""
        if self.nvml_available and self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml_available = False

    def refresh_stats(self):
        """查询 NVML 并刷新内部 gpu_stats 列表。"""
        self.gpu_stats = []
        if not self.nvml_available or not self.pynvml:
            return
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            self.gpu_stats.extend(self._get_device_stats(i) for i in range(device_count))
        except Exception as e:
            LOGGER.warning(f"device 查询时出错: {e}")
            self.gpu_stats = []

    def _get_device_stats(self, index: int) -> dict[str, Any]:
        """获取单个 GPU device 的 stats。"""
        handle = self.pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)

        def safe_get(func, *args, default=-1, divisor=1):
            try:
                val = func(*args)
                return val // divisor if divisor != 1 and isinstance(val, (int, float)) else val
            except Exception:
                return default

        temp_type = getattr(self.pynvml, "NVML_TEMPERATURE_GPU", -1)
        return {
            "index": index,
            "name": self.pynvml.nvmlDeviceGetName(handle),
            "utilization": util.gpu if util else -1,
            "memory_used": memory.used >> 20 if memory else -1,
            "memory_total": memory.total >> 20 if memory else -1,
            "memory_free": memory.free >> 20 if memory else -1,
            "temperature": safe_get(self.pynvml.nvmlDeviceGetTemperature, handle, temp_type),
            "power_draw": safe_get(self.pynvml.nvmlDeviceGetPowerUsage, handle, divisor=1000),
            "power_limit": safe_get(self.pynvml.nvmlDeviceGetEnforcedPowerLimit, handle, divisor=1000),
        }

    def print_status(self):
        """使用当前 stats 以紧凑表格打印 GPU status。"""
        self.refresh_stats()
        if not self.gpu_stats:
            LOGGER.warning("没有可用 GPU stats。")
            return
        stats = self.gpu_stats
        name_len = max(len(gpu.get("name", "N/A")) for gpu in stats)
        hdr = f"{'Idx':<3} {'Name':<{name_len}} {'Util':>6} {'显存(MiB)':>15} {'温度':>5} {'功耗(W)':>10}"
        LOGGER.info(f"\n--- GPU 状态 ---\n{hdr}\n{'-' * len(hdr)}")
        for gpu in stats:
            u = f"{gpu['utilization']:>5}%" if gpu["utilization"] >= 0 else " N/A "
            m = f"{gpu['memory_used']:>6}/{gpu['memory_total']:<6}" if gpu["memory_used"] >= 0 else " N/A / N/A "
            t = f"{gpu['temperature']}C" if gpu["temperature"] >= 0 else " N/A "
            p = f"{gpu['power_draw']:>3}/{gpu['power_limit']:<3}" if gpu["power_draw"] >= 0 else " N/A "
            LOGGER.info(f"{gpu.get('index'):<3d} {gpu.get('name', 'N/A'):<{name_len}} {u:>6} {m:>15} {t:>5} {p:>10}")
        LOGGER.info(f"{'-' * len(hdr)}\n")

    def select_idle_gpu(
        self,
        count: int = 1,
        min_memory_fraction: float = 0,
        min_util_fraction: float = 0,
    ) -> list[int]:
        """基于 utilization 和 free memory 选择最空闲的 GPU。

        参数:
            count (int): 要选择的 idle GPU 数量。
            min_memory_fraction (float): 所需最小 free memory，占 total memory 的比例。
            min_util_fraction (float): 所需最小 free utilization，范围 0.0 - 1.0。

        返回:
            (list[int]): 选中 GPU 的 index，按空闲程度排序（utilization 最低优先）。

        说明:
             如果满足条件或存在的 GPU 不足，返回数量会少于 'count'。
             如果 NVML stats 不可用，或没有 GPU 满足条件，则返回空列表。
        """
        assert min_memory_fraction <= 1.0, f"min_memory_fraction 必须 <= 1.0，当前为 {min_memory_fraction}"
        assert min_util_fraction <= 1.0, f"min_util_fraction 必须 <= 1.0，当前为 {min_util_fraction}"
        criteria = f"空闲显存 >= {min_memory_fraction * 100:.1f}% 且空闲利用率 >= {min_util_fraction * 100:.1f}%"
        LOGGER.info(f"正在搜索 {count} 个满足 {criteria} 的 idle GPU...")
        if count <= 0:
            return []
        self.refresh_stats()
        if not self.gpu_stats:
            LOGGER.warning("NVML stats 不可用。")
            return []
        eligible_gpus = [
            gpu
            for gpu in self.gpu_stats
            if gpu.get("memory_free", 0) / gpu.get("memory_total", 1) >= min_memory_fraction
            and 100 - gpu.get("utilization", 100) >= min_util_fraction * 100
        ]
        eligible_gpus.sort(
            key=lambda x: (
                x.get("utilization", 101),
                -x.get("memory_free", 0),
                random.random(),
            )
        )
        selected = [gpu["index"] for gpu in eligible_gpus[:count]]
        if selected:
            if len(selected) < count:
                LOGGER.warning(f"请求 {count} 个 GPU，但只有 {len(selected)} 个满足 idle 条件。")
            LOGGER.info(f"已选择 idle CUDA devices {selected}")
        else:
            LOGGER.warning(f"没有 GPU 满足条件（{criteria}）。")
        return selected


if __name__ == "__main__":
    required_free_mem_fraction = 0.2
    required_free_util_fraction = 0.2
    num_gpus_to_select = 1
    gpu_info = GPUInfo()
    gpu_info.print_status()
    if selected := gpu_info.select_idle_gpu(
        count=num_gpus_to_select,
        min_memory_fraction=required_free_mem_fraction,
        min_util_fraction=required_free_util_fraction,
    ):
        print(f"\n==> 使用选中的 GPU indices: {selected}")
        devices = [f"cuda:{idx}" for idx in selected]
        print(f"    目标 devices: {devices}")
