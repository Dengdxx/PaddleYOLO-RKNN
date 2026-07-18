# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""eval_quant.py — 检测模型量化精度评估

流程：
  1. FP32 基线验证（model.val()）
  2. ImperativePTQ 校准（KL 激活量化 + per-channel 权重量化）
  3. INT8 (fake-quant 近似) 精度验证
  4. 在 5 张验证集图片上生成带检测框的可视化图
  5. 打印对比表 + 写入结果文件

用法：
    conda run -n pdrk python quant/eval_quant.py \\
        --weights runs/detect/exp/weights/best.pdparams \\
        --data data.yaml \\
        --calib-batches 50 --batch 4

注意：
  本脚本使用 Paddle ImperativePTQ 的 fake-quant 近似评估 INT8 精度。
  fake-quant 在 Python 侧仍以 FP32 进行浮点计算，但引入了量化舍入噪声，
  其精度与真实 INT8 部署精度高度相关（通常偏差 < 0.3 mAP）。
  真实 INT8 加速需导出 ONNX INT8 或 RKNN INT8，并在对应 runtime 中部署。
"""

import argparse
import glob
import os
import random
import sys
import time
import warnings
from pathlib import Path

# ── 在 import paddle 之前抑制已知的无害警告 ──────────────────────────
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", message="No ccache found")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export.input_shape import format_static_imgsz, normalize_static_imgsz


# ──────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────


def build_calib_loader(data_yaml: str, imgsz: int, batch: int, n_batches: int):
    """构建校准数据加载器（与 quantize.py 保持一致）。"""
    from quant.quantize import build_calib_loader as _shared_build_calib_loader

    return _shared_build_calib_loader(data_yaml, imgsz, batch, n_batches)


def run_val(yolo, data_yaml: str, imgsz: int, device: str):
    """运行 model.val()，返回 (metrics, elapsed_sec)。"""
    t0 = time.perf_counter()
    metrics = yolo.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False)
    elapsed = time.perf_counter() - t0
    return metrics, elapsed


def calibrate_ptq(model, loader, n_batches: int, device: str) -> int:
    """对 ImperativePTQ 包装后的模型执行校准前向传播。"""
    import paddle

    model.eval()
    count = 0
    with paddle.no_grad():
        for batch in loader:
            imgs = paddle.to_tensor(batch["img"], dtype="float32") / 255.0
            if device != "cpu":
                imgs = imgs.cuda()
            model(imgs)
            count += 1
            if count % 10 == 0:
                print(f"    校准进度: {count}/{n_batches}")
            if count >= n_batches:
                break
    return count


def patch_quant_model_for_val(quant_model):
    """
    AutoBackend 在 nn_module 路径下会调用 model.fuse()。
    PTQ 插入了假量化节点后不应再 fuse（会破坏量化统计），
    此处将 fuse 替换为无操作版本。
    """
    original_fuse = getattr(quant_model, "fuse", None)

    def _noop_fuse(verbose=False):
        """无操作版 fuse，防止破坏 PTQ 插入的假量化节点。"""
        return quant_model

    quant_model.fuse = _noop_fuse
    return quant_model


def pick_val_images(data_yaml: str, n: int, seed: int = 42) -> list:
    """从验证集随机抽取 n 张图片路径。"""
    from ddyolo26.data.utils import check_det_dataset

    data_dict = check_det_dataset(data_yaml)
    val_dir = data_dict["val"]
    imgs = sorted(
        glob.glob(str(Path(val_dir) / "*.jpg"))
        + glob.glob(str(Path(val_dir) / "*.png"))
        + glob.glob(str(Path(val_dir) / "*.jpeg"))
    )
    if not imgs:
        raise FileNotFoundError(f"验证集目录下找不到图片: {val_dir}")
    random.seed(seed)
    return random.sample(imgs, min(n, len(imgs)))


def predict_and_save(yolo, img_paths: list, output_dir: str, device: str, imgsz: int) -> list:
    """在给定图片上运行推理并保存带检测框的可视化图。"""
    os.makedirs(output_dir, exist_ok=True)
    results = yolo.predict(
        source=img_paths,
        imgsz=imgsz,
        device=device,
        conf=0.25,
        verbose=False,
    )
    saved = []
    for i, r in enumerate(results):
        out_path = str(Path(output_dir) / f"viz_{i + 1:02d}_{Path(r.path).name}")
        r.save(filename=out_path)
        n_boxes = len(r.boxes) if r.boxes is not None else 0
        print(f"    [{i + 1}] {Path(r.path).name}  → 检测框: {n_boxes}  已保存: {out_path}")
        saved.append(out_path)
    return saved


# ──────────────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────────────


def main():
    """主程序入口：执行 FP32 基线验证 → PTQ 校准 → INT8 精度验证 → 可视化。"""
    parser = argparse.ArgumentParser(
        description="检测模型 FP32 vs INT8 (ImperativePTQ) 量化精度评估",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/exp/weights/best.pdparams",
        help="训练好的 .pdparams 权重路径",
    )
    parser.add_argument("--data", type=str, default="data/your.yaml", help="数据集 yaml 路径")
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=50,
        help="校准 batch 数量（越多越精确，建议 50-200，默认 50）",
    )
    parser.add_argument("--batch", type=int, default=4, help="校准 batch size（默认 4）")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        default=["640"],
        metavar="SIZE",
        help="推理图像尺寸：SIZE、HxW 或 H W（默认 640）",
    )
    parser.add_argument("--device", type=str, default="0", help="推理设备，'0' 表示 GPU 0，'cpu'")
    parser.add_argument("--output", type=str, default="", help="输出目录；默认与输入权重同目录")
    parser.add_argument("--n-viz", type=int, default=5, help="可视化图片数量（默认 5）")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="同时导出 FP32 ONNX 并通过 onnxruntime 生成 INT8 ONNX（真正的 INT8 权重）",
    )
    args = parser.parse_args()
    parsed_imgsz = normalize_static_imgsz(args.imgsz)
    if parsed_imgsz.height != parsed_imgsz.width:
        raise ValueError("eval_quant 包含 Paddle val，当前仅支持方形 imgsz")
    args.imgsz = parsed_imgsz.height
    if not args.output:
        weights_path = Path(args.weights)
        args.output = str(weights_path.with_name(f"{weights_path.stem}_quant_eval"))

    import paddle
    from paddle.quantization import (
        ImperativePTQ,
        PTQConfig,
        KLQuantizer,
        PerChannelAbsmaxQuantizer,
    )
    from ddyolo26 import YOLO

    os.makedirs(args.output, exist_ok=True)

    # ── 步骤 1: FP32 基线验证 ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  步骤 1/4  FP32 基线验证")
    print("=" * 65)
    yolo_fp32 = YOLO(args.weights)
    fp32_metrics, fp32_time = run_val(yolo_fp32, args.data, args.imgsz, args.device)

    fp32_p = fp32_metrics.box.mp
    fp32_r = fp32_metrics.box.mr
    fp32_map50 = fp32_metrics.box.map50
    fp32_map = fp32_metrics.box.map

    print(
        f"\n  FP32: P={fp32_p:.4f}  R={fp32_r:.4f}  "
        f"mAP50={fp32_map50:.4f}  mAP50-95={fp32_map:.4f}  耗时={fp32_time:.1f}s"
    )

    # ── 步骤 2: PTQ 校准 ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  步骤 2/4  ImperativePTQ 校准")
    print("=" * 65)

    yolo_int8 = YOLO(args.weights)  # 独立加载，避免影响 FP32 实例
    fp32_model = yolo_int8.model

    ptq_config = PTQConfig(
        activation_quantizer=KLQuantizer(quant_bits=8),
        weight_quantizer=PerChannelAbsmaxQuantizer(quant_bits=8),
    )
    ptq = ImperativePTQ(quant_config=ptq_config)
    print("  正在插入 fake-quant 节点...")
    quant_model = ptq.quantize(fp32_model, inplace=False)
    quant_model.eval()

    print(f"  构建校准数据集（calib_batches={args.calib_batches}, batch={args.batch}）...")
    loader, n_actual = build_calib_loader(args.data, args.imgsz, args.batch, args.calib_batches)

    print(f"  开始校准（共 {n_actual} batches）...")
    t_cal = time.perf_counter()
    n_done = calibrate_ptq(quant_model, loader, n_actual, args.device)
    t_cal = time.perf_counter() - t_cal
    print(f"  校准完成: {n_done} batches，耗时 {t_cal:.1f}s")

    # _convert: 移除采样 hooks，计算量化阈值，固化 fake-quant 参数
    # 必须在推理前调用，否则 hooks 仍尝试 sample_data 导致 NaN
    print("  计算量化阈值并锁定 fake-quant 节点（ptq._convert）...")
    ptq._convert(quant_model)

    # ── 步骤 3: INT8 精度验证 ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  步骤 3/4  INT8 (fake-quant) 精度验证")
    print("=" * 65)

    # AutoBackend 会对 nn_module 调用 model.fuse()，PTQ 后不应 fuse
    quant_model = patch_quant_model_for_val(quant_model)
    yolo_int8.model = quant_model

    int8_metrics, int8_time = run_val(yolo_int8, args.data, args.imgsz, args.device)

    int8_p = int8_metrics.box.mp
    int8_r = int8_metrics.box.mr
    int8_map50 = int8_metrics.box.map50
    int8_map = int8_metrics.box.map

    print(
        f"\n  INT8: P={int8_p:.4f}  R={int8_r:.4f}  "
        f"mAP50={int8_map50:.4f}  mAP50-95={int8_map:.4f}  耗时={int8_time:.1f}s"
    )

    # ── 步骤 4: 可视化 ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  步骤 4/4  生成 {args.n_viz} 张可视化图（INT8 模型推理）")
    print("=" * 65)

    sample_imgs = pick_val_images(args.data, args.n_viz)
    viz_dir = str(Path(args.output) / "viz")
    saved = predict_and_save(yolo_int8, sample_imgs, viz_dir, args.device, args.imgsz)

    # ── 汇总表格 ───────────────────────────────────────────────────────
    def _pct_drop(a, b):
        """a=FP32, b=INT8, 返回精度下降百分比（正值=下降）。"""
        if a == 0:
            return 0.0
        return (a - b) / a * 100

    drop_map50 = _pct_drop(fp32_map50, int8_map50)
    drop_map = _pct_drop(fp32_map, int8_map)
    drop_p = _pct_drop(fp32_p, int8_p)
    drop_r = _pct_drop(fp32_r, int8_r)

    header = f"\n  {'指标':<16}{'FP32':>10}{'INT8 (PTQ)':>12}{'精度变化':>12}"
    sep = "  " + "-" * 52
    rows = [
        f"  {'Precision':<16}{fp32_p:>10.4f}{int8_p:>12.4f}{int8_p - fp32_p:>+10.4f}",
        f"  {'Recall':<16}{fp32_r:>10.4f}{int8_r:>12.4f}{int8_r - fp32_r:>+10.4f}",
        f"  {'mAP50':<16}{fp32_map50:>10.4f}{int8_map50:>12.4f}{-drop_map50:>+9.2f}%",
        f"  {'mAP50-95':<16}{fp32_map:>10.4f}{int8_map:>12.4f}{-drop_map:>+9.2f}%",
        f"  {'验证耗时(s)':<16}{fp32_time:>10.1f}{int8_time:>12.1f}{'':>12}",
    ]

    print("\n" + "=" * 65)
    print(f"  量化精度汇总（data={args.data}, imgsz={format_static_imgsz(args.imgsz)}）")
    print("=" * 65)
    print(header)
    print(sep)
    for row in rows:
        print(row)
    print(sep)
    print(f"\n  * 校准数据: {n_done} batches × batch_size={args.batch}")
    print(f"  * fake-quant 模式: KL 校准激活量化 + per-channel 权重量化")
    print(f"  * 真实 INT8 部署加速需导出至 ONNX / RKNN\n")

    # ── 写入结果文件 ───────────────────────────────────────────────────
    result_path = str(Path(args.output) / "quant_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"检测模型量化精度评估报告\n")
        f.write(f"{'=' * 55}\n")
        f.write(f"模型权重  : {args.weights}\n")
        f.write(f"数据集    : {args.data}\n")
        f.write(f"校准数据  : {n_done} batches × batch_size={args.batch}\n")
        f.write(f"图像尺寸  : {format_static_imgsz(args.imgsz)}\n")
        f.write(f"量化策略  : KL 激活量化 (8-bit) + per-channel absmax 权重量化 (8-bit)\n\n")
        f.write(f"{'指标':<16}{'FP32':>10}{'INT8 (PTQ)':>12}{'精度变化':>12}\n")
        f.write(f"{'-' * 52}\n")
        f.write(f"{'Precision':<16}{fp32_p:>10.4f}{int8_p:>12.4f}{int8_p - fp32_p:>+10.4f}\n")
        f.write(f"{'Recall':<16}{fp32_r:>10.4f}{int8_r:>12.4f}{int8_r - fp32_r:>+10.4f}\n")
        f.write(f"{'mAP50':<16}{fp32_map50:>10.4f}{int8_map50:>12.4f}{-drop_map50:>+9.2f}%\n")
        f.write(f"{'mAP50-95':<16}{fp32_map:>10.4f}{int8_map:>12.4f}{-drop_map:>+9.2f}%\n")
        f.write(f"{'验证耗时(s)':<16}{fp32_time:>10.1f}{int8_time:>12.1f}\n\n")
        f.write(f"可视化图片: {viz_dir}/\n")
        for p in saved:
            f.write(f"  {Path(p).name}\n")

    print(f"  结果文件已保存至: {result_path}")
    print(f"  可视化图片目录:   {viz_dir}/\n")

    # ── 保存量化模型权重 ───────────────────────────────────────────────
    import paddle as _paddle

    quant_weights_path = str(Path(args.output) / (Path(args.weights).stem + "_int8.pdparams"))
    _paddle.save(
        {
            "model": quant_model.state_dict(),
            "epoch": -1,
            "quantized": True,
            "quant_config": "KLQuantizer+PerChannelAbsmaxQuantizer",
            "fp32_weights": str(args.weights),
        },
        quant_weights_path,
    )
    print(f"  量化模型权重已保存: {quant_weights_path}")
    print(f"  加载方式: from quantize import load_quantized_model")
    print(f"           yolo = load_quantized_model('{quant_weights_path}', '{args.weights}')\n")

    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"\n量化权重: {quant_weights_path}\n")

    # ── 可选步骤：导出 ONNX INT8 ──────────────────────────────────────
    if args.export_onnx:
        print("\n" + "=" * 65)
        print("  附加步骤  导出 ONNX INT8 模型（真实 INT8 权重）")
        print("=" * 65)
        _export_onnx_int8(args, n_done, result_path)


def _export_onnx_int8(args, calib_count: int, result_path: str):
    """导出 FP32 ONNX → 预处理 → ONNX INT8 静态量化。"""
    import numpy as np

    # 1. FP32 ONNX 导出
    from ddyolo26 import YOLO

    print("  [1/3] 导出 FP32 ONNX 模型...")
    yolo_export = YOLO(args.weights)
    onnx_fp32 = yolo_export.export(format="onnx", imgsz=args.imgsz, simplify=True)
    if isinstance(onnx_fp32, str):
        onnx_fp32_path = onnx_fp32
    else:
        onnx_fp32_path = str(Path(args.weights).with_suffix(".onnx"))
    size_fp32 = os.path.getsize(onnx_fp32_path) / 1024 / 1024
    print(f"        FP32 ONNX: {onnx_fp32_path} ({size_fp32:.1f} MB)")

    # 2. 读取 ONNX 输入规格 & 收集校准数据
    import onnx as _onnx

    onnx_model = _onnx.load(onnx_fp32_path)
    inp = onnx_model.graph.input[0]
    input_name = inp.name
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    onnx_batch = dims[0] if dims[0] > 0 else 1
    onnx_h, onnx_w = dims[2], dims[3]

    print(f"  [2/3] 收集校准数据（{args.calib_batches} batches）...")
    loader, n_batches = build_calib_loader(
        args.data,
        (onnx_h, onnx_w),
        batch=onnx_batch,
        n_batches=args.calib_batches,
    )

    calib_data = []
    count = 0
    for batch_data in loader:
        imgs = batch_data["img"].astype("float32") / 255.0
        if imgs.shape[2] != onnx_h or imgs.shape[3] != onnx_w:
            raise ValueError(f"校准预处理尺寸 {imgs.shape[2:]} 与 ONNX 输入 {(onnx_h, onnx_w)} 不一致")
        if imgs.shape[0] != onnx_batch:
            for i in range(imgs.shape[0]):
                calib_data.append(imgs[i : i + 1])
        else:
            calib_data.append(imgs)
        count += 1
        if count >= n_batches:
            break

    from onnxruntime.quantization import (
        quantize_static,
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        preprocess as quant_preprocess,
    )

    class _CalibReader(CalibrationDataReader):
        """为 onnxruntime INT8 量化提供校准数据的迭代器。"""

        def __init__(self, data, name):
            """初始化校准读取器。"""
            self._data, self._name = data, name
            self._iter = iter(data)

        def get_next(self):
            """返回下一个校准数据字典。"""
            try:
                return {self._name: next(self._iter)}
            except StopIteration:
                return None

        def rewind(self):
            """重置迭代器。"""
            self._iter = iter(self._data)

    reader = _CalibReader(calib_data, input_name)

    # 3. 预处理 + 量化
    print("  [3/3] ONNX 预处理 + INT8 静态量化...")
    import tempfile

    preprocessed_path = tempfile.mktemp(suffix=".onnx")
    quant_preprocess.quant_pre_process(onnx_fp32_path, preprocessed_path, skip_symbolic_shape=True)

    onnx_int8_path = str(Path(args.output) / (Path(args.weights).stem + "_int8.onnx"))
    # 与 quant/quantize.py、scripts/export_all_models.py:step_int8_onnx_seg 对齐：
    # ORT CPU EP 的 QDQ→QLinearConv 融合器对「对称 INT8 激活」兼容性差，会让大量
    # Conv 退化成 fp32（DequantizeLinear→Conv→QuantizeLinear）。改用非对称 UInt8
    # 激活 + per-channel Int8 权重，保证 ≥95% Conv 走 QLinearConv 快路径。
    quantize_static(
        model_input=preprocessed_path,
        model_output=onnx_int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    size_int8 = os.path.getsize(onnx_int8_path) / 1024 / 1024
    print(f"\n  ONNX INT8 模型已保存: {onnx_int8_path} ({size_int8:.1f} MB)")
    print(f"  体积压缩: {size_fp32:.1f} MB → {size_int8:.1f} MB ({size_int8 / size_fp32 * 100:.0f}%)")
    print(f"  主机侧验证: onnxruntime.InferenceSession('{onnx_int8_path}')\n")

    with open(result_path, "a", encoding="utf-8") as f:
        f.write(f"\nONNX FP32: {onnx_fp32_path} ({size_fp32:.1f} MB)\n")
        f.write(f"ONNX INT8: {onnx_int8_path} ({size_int8:.1f} MB)\n")


if __name__ == "__main__":
    main()
