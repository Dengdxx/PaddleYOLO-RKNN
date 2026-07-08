#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file _make_predfl_onnx.py
@brief `pre_dfl` ONNX 裁剪脚本入口。
@details
提供命令行包装，调用共享的 `make_predfl_onnx()` 完成普通检测 ONNX 到
`pre_dfl` 双输出 ONNX 的改写。
"""

from __future__ import annotations

import argparse

from det_onnx_routes import make_predfl_onnx


def parse_args() -> argparse.Namespace:
    """!
    @brief 解析命令行参数。
    @return 命令行参数对象。
    """
    p = argparse.ArgumentParser(description="将 detect ONNX 裁剪为 pre_dfl 双输出")
    p.add_argument("--input", required=True, help="输入 ONNX 路径")
    p.add_argument("--output", default=None, help="输出 ONNX 路径（默认自动加 _predfl 后缀）")
    return p.parse_args()


def main() -> int:
    """!
    @brief 脚本主入口。
    @return 成功返回 `0`。
    """
    args = parse_args()
    out_path = make_predfl_onnx(args.input, args.output)
    print(f"[OK] pre_dfl ONNX 已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
