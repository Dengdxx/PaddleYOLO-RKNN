# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief CPU 性能检测：获取核心数、频率等 CPU 信息。
@details
为自动批大小选择和性能日志提供 CPU 信息，
在纯 CPU 环境（如 CI）中避免 CUDA API 调用崩溃。
"""

import paddle

import platform
import re
import subprocess
import sys
from pathlib import Path


class CPUInfo:
    """提供跨平台 CPU 品牌和型号信息。

    查询特定平台的信息源，获取可读的 CPU 描述，并进行规范化，保证在 macOS、Linux 和 Windows
    上展示一致。如果平台探测失败，则使用通用平台标识，确保始终返回稳定字符串。

    方法:
        name: 使用平台专用信息源和稳健 fallback 返回规范化 CPU 名称。
        _clean: 规范化并美化常见厂商品牌字符串和频率格式。
        __str__: 在字符串上下文中返回规范化 CPU 名称。

    示例:
        >>> CPUInfo.name()
        'Apple M4 Pro'
        >>> str(CPUInfo())
        'Intel Core i7-9750H 2.60GHz'
    """

    @staticmethod
    def name() -> str:
        """从平台专用信息源返回规范化 CPU 型号字符串。"""
        try:
            if sys.platform == "darwin":
                s = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if s:
                    return CPUInfo._clean(s)
            elif sys.platform.startswith("linux"):
                p = Path("/proc/cpuinfo")
                if p.exists():
                    for line in p.read_text(errors="ignore").splitlines():
                        if "model name" in line:
                            return CPUInfo._clean(line.split(":", 1)[1])
            elif sys.platform.startswith("win"):
                try:
                    import winreg as wr

                    with wr.OpenKey(
                        wr.HKEY_LOCAL_MACHINE,
                        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                    ) as k:
                        val, _ = wr.QueryValueEx(k, "ProcessorNameString")
                        if val:
                            return CPUInfo._clean(val)
                except Exception:
                    pass
            s = platform.processor() or getattr(platform.uname(), "processor", "") or platform.machine()
            return CPUInfo._clean(s or "Unknown CPU")
        except Exception:
            s = platform.processor() or platform.machine() or ""
            return CPUInfo._clean(s or "Unknown CPU")

    @staticmethod
    def _clean(s: str) -> str:
        """规范化并美化原始 CPU 描述字符串。"""
        s = re.sub("\\s+", " ", s.strip())
        s = s.replace("(TM)", "").replace("(tm)", "").replace("(R)", "").replace("(r)", "").strip()
        if m := re.search("(Intel.*?i\\d[\\w-]*) CPU @ ([\\d.]+GHz)", s, re.I):
            return f"{m.group(1)} {m.group(2)}"
        if m := re.search("(AMD.*?Ryzen.*?[\\w-]*) CPU @ ([\\d.]+GHz)", s, re.I):
            return f"{m.group(1)} {m.group(2)}"
        return s

    def __str__(self) -> str:
        """返回规范化 CPU 名称。"""
        return self.name()


if __name__ == "__main__":
    print(CPUInfo.name())
