#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only
"""!
@file quantize.py
@brief 检测模型 INT8 量化入口。
@details
本仓库将检测模型部署主线放在 ONNX / RKNN，而非仅停留在 Paddle 训练态。
当前该脚本承担两条量化路径：
- **ONNX PTQ**：用户一步输入 Paddle 权重或 ONNX，内部自动路由到 `pre_dist / pre_dfl` 后输出真正的 INT8 ONNX；
- **Paddle ImperativePTQ**：调试路径，保存 fake-quant 模型状态。

本项目以 **ONNX** 为核心部署格式。对检测模型而言，
**只有 `pre_dist` / `pre_dfl` ONNX 路线可用于 INT8 量化部署**。

两种量化模式（默认 onnx）：
  --mode onnx    : （默认推荐）ONNX INT8 静态量化，输出真 INT8 ONNX 模型
  --mode paddle  : Paddle 原生 ImperativePTQ（fake-quant），主要用于调试

ONNX 模式特性：
  - 输入可以是 `.pdparams / onnx`
  - 内部自动导出普通 ONNX，并裁剪到 `pre_dist / pre_dfl`
  - 输出为真正的 INT8 ONNX 模型，权重直接以 INT8 存储
  - 部署使用 onnxruntime.InferenceSession 即可

用法示例：
    # 从训练权重一步得到 pre_dist / pre_dfl INT8 ONNX
    python quantize.py \
        --weights runs/detect/exp/weights/best.pdparams \
        --data data/your.yaml \
        --imgsz 640

    # 从已有普通 ONNX 一步得到 pre_dist / pre_dfl INT8 ONNX
    python quantize.py \
        --weights runs/detect/exp/weights/best.onnx \
        --data coco.yaml

    # Paddle PTQ（调试用，fake-quant）
    python quantize.py --mode paddle \
        --weights runs/detect/exp/weights/best.pdparams \
        --data data/your.yaml
    """

import argparse
import os
import sys
import warnings
from pathlib import Path

# ── 在 import paddle 之前抑制已知的无害警告 ──────────────────────────
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", message="No ccache found")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export.det_onnx_routes import (
    cleanup_temp_paths,
    detect_prepared_det_onnx_i8_route,
    infer_det_onnx_i8_route as _infer_det_onnx_i8_route,
    prepare_det_onnx_i8_input,
)
from export.input_shape import StaticInputShape, format_static_imgsz, normalize_static_imgsz, static_imgsz_hw


# ──────────────────────────────────────────────────────────────────────
# ONNX 图分析：仅允许 pre_dist / pre_dfl
# ──────────────────────────────────────────────────────────────────────


def detect_supported_onnx_i8_route(onnx_model) -> str:
    """!
    @brief 检查 ONNX 是否属于当前允许的 INT8 检测主线。
    @param onnx_model 已加载的 ONNX 模型。
    @return `pre_dist` 或 `pre_dfl`。
    @throw ValueError 当输出契约不属于受支持主线时抛出。
    """
    try:
        return detect_prepared_det_onnx_i8_route(onnx_model)
    except ValueError as exc:
        raise ValueError("ONNX INT8 仅支持 pre_dist / pre_dfl，当前 ONNX 输出契约不受支持。") from exc


def auto_export_onnx(weights_path: str, imgsz: StaticInputShape) -> str:
    """!
    @brief 将 Paddle 权重导出为普通 ONNX。
    @details
    `.pdparams` 通过 `ddyolo26.YOLO` 导出。
    @param weights_path 输入 Paddle 权重路径，必须以 `.pdparams` 结尾。
    @param imgsz 导出时使用的输入尺寸。
    @return 导出的普通 ONNX 路径。
    @throw ValueError 当 `weights_path` 不是 Paddle 权重时抛出。
    """
    suffix = Path(weights_path).suffix.lower()
    print(f"[AUTO-EXPORT] 将训练权重 {weights_path} 导出为 ONNX...")

    # 旧 Paddle checkpoint 虽是 .pt 后缀，但内容是 paddle.save() 写入的；
    # 这里按文件名再次判断一次，避免误走 ultralytics 加载路径。
    name_lower = Path(weights_path).stem.lower()
    is_paddle_pt = name_lower.endswith("_paddle") or name_lower.endswith("paddle")

    if suffix == ".pdparams" or (suffix == ".pt" and is_paddle_pt):
        # paddle 训练态权重，必须使用 ddyolo26（其 __init__ 会 import paddle）
        from ddyolo26 import YOLO  # noqa: PLC0415

        yolo = YOLO(weights_path)
        # dual_raw=True：输出 boxes/scores（detect）或 boxes/scores/mask_coeff/proto（segment）。
        # 分割四输出只作为瞬态图，随后由 prepare_seg_onnx_i8_input 规范化为五输出。
        onnx_path = yolo.export(format="onnx", imgsz=imgsz, simplify=True, dual_raw=True, opset=13)
    elif suffix == ".pt":
        raise ValueError(f"普通 .pt 权重不被 Paddle-only 量化入口支持: {weights_path}. 请使用 .pdparams。")
    else:
        raise ValueError(f"auto_export_onnx 只支持 .pdparams，收到: {weights_path}")

    if isinstance(onnx_path, str):
        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"[AUTO-EXPORT] ONNX 已保存: {onnx_path} ({size_mb:.1f} MB)")
        return onnx_path
    out = Path(weights_path).with_suffix(".onnx")
    print(f"[AUTO-EXPORT] ONNX 已保存: {out}")
    return str(out)


def prepare_onnx_i8_input(weights_path: str, imgsz: StaticInputShape) -> tuple[str, str, list[str]]:
    """!
    @brief 为 ONNX INT8 量化准备主线输入模型。
    @param weights_path 用户提供的输入权重路径。
    @param imgsz 自动导出普通 ONNX 时使用的输入尺寸。
    @return 三元组 `(route, prepared_onnx_path, cleanup_paths)`。
    """
    return prepare_det_onnx_i8_input(weights_path, imgsz, auto_export_onnx)


def infer_det_onnx_i8_route(onnx_model) -> str:
    """!
    @brief 从普通检测 ONNX 推断应走的 INT8 主线。
    @param onnx_model 已加载的 ONNX 模型。
    @return `pre_dist` 或 `pre_dfl`。
    """
    return _infer_det_onnx_i8_route(onnx_model)


# ──────────────────────────────────────────────────────────────────────
# 公共：构建校准数据集迭代器
# ──────────────────────────────────────────────────────────────────────


def build_calib_loader(data_yaml: str, imgsz: StaticInputShape, batch: int, n_batches: int):
    """!
    @brief 构建校准数据加载器。
    @details
    自包含实现：仅依赖 cv2 / numpy / pyyaml，不引入 ddyolo26 / paddle，
    在 yolo、paddle 两个 conda 环境均可运行。
    校准输入严格复用部署语义：固定目标画布居中 letterbox，且 `scaleup=False`。
    为避免按宽高比排序后只取头部样本，先按宽高比排序，再在完整分布上
    等距抽取 `batch * n_batches` 张代表性图片。
    返回的 batch dict 中 ``img`` 为 uint8 numpy 数组 [B, 3, H, W]。
    @param data_yaml 数据集 YAML 路径。
    @param imgsz 校准图像尺寸，正方形为单个边长，矩形为 `(H, W)`。
    @param batch 校准批大小。
    @param n_batches 计划使用的最大批次数。
    @return 二元组 `(batches, n_actual_batches)`，batches 是 list[dict]。
    """
    import yaml
    import cv2
    import numpy as np
    from pathlib import Path

    from tools.eval.backend_utils import letterbox_image

    if batch <= 0 or n_batches <= 0:
        raise ValueError("校准 batch 和 n_batches 必须大于 0")
    input_h, input_w = static_imgsz_hw(imgsz)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    yaml_path = Path(data_yaml).resolve()
    with open(yaml_path, "r", encoding="utf-8") as _f:
        cfg = yaml.safe_load(_f) or {}

    yaml_dir = yaml_path.parent
    dataset_root = Path(cfg.get("path") or yaml_dir)
    if not dataset_root.is_absolute():
        dataset_root = (yaml_dir / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()

    def _resolve_dataset_entry(entry: str) -> Path:
        path = Path(entry)
        if path.is_absolute():
            return path.resolve()
        resolved = (dataset_root / entry).resolve()
        if resolved.exists():
            return resolved
        if entry.startswith("../"):
            return (dataset_root / entry[3:]).resolve()
        return resolved

    def _collect_image_files(entry: str) -> list[Path]:
        resolved = _resolve_dataset_entry(entry)
        if resolved.is_file() and resolved.suffix.lower() == ".txt":
            items: list[Path] = []
            with open(resolved, "r", encoding="utf-8") as _f:
                for raw in _f:
                    line = raw.strip()
                    if not line:
                        continue
                    p = Path(line)
                    if p.is_absolute():
                        items.append(p.resolve())
                        continue
                    txt_relative = (resolved.parent / p).resolve()
                    if txt_relative.exists():
                        items.append(txt_relative)
                    else:
                        items.append((dataset_root / p).resolve())
            return items
        if resolved.is_dir():
            return sorted(p.resolve() for p in resolved.rglob("*") if p.is_file() and p.suffix.lower() in img_exts)
        if resolved.is_file() and resolved.suffix.lower() in img_exts:
            return [resolved.resolve()]
        return []

    val_spec = cfg.get("val", "images/val")
    val_entries = val_spec if isinstance(val_spec, list) else [val_spec]
    img_files: list[Path] = []
    for entry in val_entries:
        img_files.extend(_collect_image_files(str(entry)))
    img_files = list(dict.fromkeys(img_files))
    if not img_files:
        raise FileNotFoundError(f"未能从 {data_yaml} 的 val 划分找到校准图片")

    decoded_shapes: list[tuple[int, int]] = []
    valid_files: list[Path] = []
    unreadable = 0
    for p in img_files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            unreadable += 1
            continue
        decoded_shapes.append(img.shape[:2])
        valid_files.append(p)
    if unreadable:
        print(f"[PTQ-Loader] 警告：已跳过 {unreadable} 张无法读取的校准图片")
    if not valid_files:
        raise RuntimeError(f"{data_yaml} 的所有校准图片都无法读取")

    shapes = np.asarray(decoded_shapes, dtype=np.float32)
    aspect_ratios = shapes[:, 0] / shapes[:, 1]
    order = aspect_ratios.argsort()
    img_files = [valid_files[i] for i in order]
    aspect_ratios = aspect_ratios[order]

    sample_count = min(len(img_files), batch * n_batches)
    if batch > 1:
        sample_count -= sample_count % batch
    if sample_count == 0:
        raise ValueError(f"可用校准图片不足一个完整 batch：图片={len(img_files)}，batch={batch}")
    sample_indices = np.linspace(0, len(img_files) - 1, sample_count, dtype=np.int64)
    img_files = [img_files[int(i)] for i in sample_indices]
    sampled_ratios = aspect_ratios[sample_indices]
    print(
        "[PTQ-Loader] 固定 letterbox="
        f"{input_h}x{input_w}，代表性抽样={sample_count}/{len(valid_files)}，"
        f"宽高比范围={sampled_ratios.min():.3f}..{sampled_ratios.max():.3f}"
    )

    batches: list[dict[str, np.ndarray]] = []
    for batch_idx in range(0, len(img_files), batch):
        batch_files = img_files[batch_idx : batch_idx + batch]
        imgs = []
        for image_path in batch_files:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, _ = letterbox_image(img, (input_h, input_w), pad_value=114, scaleup=False)
            imgs.append(img.transpose(2, 0, 1))
        if imgs:
            batches.append({"img": np.stack(imgs)})
    return batches, len(batches)


# ──────────────────────────────────────────────────────────────────────
# 路径 1：Paddle 原生 ImperativePTQ
# ──────────────────────────────────────────────────────────────────────


def quantize_paddle(args):
    """!
    @brief 执行 Paddle ImperativePTQ 量化。
    @param args 命令行参数对象。
    @details
    该路径输出 fake-quant 权重，仅用于调试量化精度，不直接作为部署产物。
    """
    import paddle
    from paddle.quantization import ImperativePTQ, PTQConfig, KLQuantizer, PerChannelAbsmaxQuantizer

    print(f"[PTQ-Paddle] 加载模型: {args.weights}")
    from ddyolo26 import YOLO

    yolo = YOLO(str(args.weights))
    model = yolo.model
    model.eval()

    # 配置量化：激活用 KL（精度更高），权重用 per-channel absmax
    ptq_config = PTQConfig(
        activation_quantizer=KLQuantizer(quant_bits=8),
        weight_quantizer=PerChannelAbsmaxQuantizer(quant_bits=8),
    )
    ptq = ImperativePTQ(quant_config=ptq_config)
    quant_model = ptq.quantize(model, inplace=False)
    quant_model.eval()

    # 构建校准数据
    print(f"[PTQ-Paddle] 构建校准数据（{args.calib_batches} batches）...")
    loader, n_batches = build_calib_loader(args.data, args.imgsz, batch=args.batch, n_batches=args.calib_batches)

    # 校准前向传播（model 已插入 observer hook）
    print("[PTQ-Paddle] 开始校准前向传播...")
    count = 0
    with paddle.no_grad():
        for batch in loader:
            imgs = paddle.to_tensor(batch["img"], dtype="float32") / 255.0
            if paddle.is_compiled_with_cuda():
                imgs = imgs.cuda()
            quant_model(imgs)
            count += 1
            if count % 10 == 0:
                print(f"  校准进度: {count}/{n_batches}")
            if count >= n_batches:
                break
    print(f"[PTQ-Paddle] 校准完成，共 {count} batches")

    # 转换量化模型：移除 observer hook，计算量化阈值
    ptq._convert(quant_model)
    print("[PTQ-Paddle] 量化阈值计算完成")

    # 保存量化 state_dict（含 scale 参数）
    # 注意：save_quantized_model 需要 paddle.jit.to_static 跟踪，
    # 但本模型含动态控制流（m.i 属性），JIT 编译失败，故改用 dygraph 方式保存。
    output = str(args.output)
    if not output.endswith(".pdparams"):
        output_path = output + ".pdparams"
    else:
        output_path = output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    paddle.save(
        {
            "model": quant_model.state_dict(),
            "epoch": -1,
            "quantized": True,
            "quant_config": "KLQuantizer+PerChannelAbsmaxQuantizer",
            "fp32_weights": str(args.weights),
        },
        output_path,
    )
    print(f"[PTQ-Paddle] 完成！量化权重已保存至 {output_path}")
    print("  加载方式: from quantize import load_quantized_model")
    print(f"           yolo = load_quantized_model('{output_path}', '{args.weights}')")


# ──────────────────────────────────────────────────────────────────────
# 量化模型加载（对应 paddle 模式的输出）
# ──────────────────────────────────────────────────────────────────────


def load_quantized_model(quant_weights: str, fp32_weights: str):
    """!
    @brief 加载 fake-quant Paddle 权重并重建量化模型。
    @param quant_weights `quantize.py --mode paddle` 保存的 `.pdparams` 路径。
    @param fp32_weights 原始 FP32 权重路径，用于重建模型结构。
    @return 已替换 `.model` 的 YOLO 实例。
    """
    import paddle
    from paddle.quantization import ImperativePTQ, PTQConfig, KLQuantizer, PerChannelAbsmaxQuantizer
    from ddyolo26 import YOLO

    yolo = YOLO(str(fp32_weights))
    model = yolo.model
    model.eval()

    ptq_config = PTQConfig(
        activation_quantizer=KLQuantizer(quant_bits=8),
        weight_quantizer=PerChannelAbsmaxQuantizer(quant_bits=8),
    )
    ptq = ImperativePTQ(quant_config=ptq_config)
    quant_model = ptq.quantize(model, inplace=False)
    ptq._convert(quant_model)  # 必须先 _convert，结构才和 save 时一致

    ckpt = paddle.load(str(quant_weights))
    quant_model.set_state_dict(ckpt["model"])
    quant_model.eval()

    # AutoBackend 在 nn_module 路径会调用 model.fuse()，
    # QuantizedConv2D 不支持 fuse，将其替换为无操作版本
    def _noop_fuse(verbose=False):
        """!
        @brief 屏蔽量化模型上的 `fuse()` 调用。
        @param verbose 保留参数，兼容原接口。
        @return 当前量化模型对象。
        """
        return quant_model

    quant_model.fuse = _noop_fuse

    yolo.model = quant_model
    return yolo


# ──────────────────────────────────────────────────────────────────────
# 路径 2：ONNX INT8 静态量化 (onnxruntime)
# ──────────────────────────────────────────────────────────────────────


def quantize_onnx(args):
    """!
    @brief 执行 ONNX INT8 静态量化。
    @param args 命令行参数对象。
    @details
    对用户保持一步入口，但内部只允许 `pre_dist / pre_dfl` 两条主线参与
    INT8 量化，其他检测路线一律拒绝。
    """
    try:
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantFormat,
            QuantType,
        )
        import numpy as np
    except ImportError:
        print("错误：需要安装 onnxruntime + onnx:  pip install onnxruntime onnx")
        sys.exit(1)

    weights_path = str(args.weights)
    route, prepared_onnx_path, cleanup_paths = prepare_onnx_i8_input(weights_path, args.imgsz)
    public_route = "predfl" if route == "pre_dfl" else "predist"
    if args.route and args.route != public_route:
        cleanup_temp_paths(cleanup_paths)
        raise ValueError(f"请求 route={args.route}，但模型实际为 {public_route}")
    if args.output:
        output_path = str(args.output)
    else:
        weights = Path(weights_path)
        stem = weights.stem[:-7] if weights.stem.endswith("_paddle") else weights.stem
        shape_tag = format_static_imgsz(args.imgsz)
        output_path = str(weights.with_name(f"{stem}_paddle_{public_route}_int8_{shape_tag}.onnx"))
    if not output_path.endswith(".onnx"):
        output_path += ".onnx"
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    preprocessed_path = ""
    print(f"[PTQ-ONNX] 已确认路线: {route}")
    print(f"[PTQ-ONNX] 加载 ONNX 模型: {prepared_onnx_path}")

    # 读取 ONNX 模型输入规格
    import onnx

    onnx_model = onnx.load(prepared_onnx_path)
    inp = onnx_model.graph.input[0]
    input_name = inp.name  # 通常为 "images"
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    # dims 例如 [1, 3, 640, 640] 或 [0, 3, 640, 640]（0 表示 dynamic）
    onnx_batch = dims[0] if dims[0] > 0 else 1
    onnx_h, onnx_w = dims[2], dims[3]
    print(f"[PTQ-ONNX] ONNX 输入: {input_name}  shape={dims}  → 校准使用 ({onnx_batch}, 3, {onnx_h}, {onnx_w})")

    # 构建校准数据 — 必须匹配 ONNX 的固定输入形状
    print(f"[PTQ-ONNX] 构建校准数据（{args.calib_batches} batches）...")
    loader, n_batches = build_calib_loader(
        args.data,
        (onnx_h, onnx_w),
        batch=onnx_batch,
        n_batches=args.calib_batches,
    )

    # 收集校准张量，逐张 letterbox 到 ONNX 期望尺寸
    calib_data = []
    count = 0
    for batch_data in loader:
        imgs = batch_data["img"].astype("float32") / 255.0  # [B, 3, H, W]
        # loader 已按部署输入执行固定 letterbox；尺寸不一致代表调用配置错误。
        if imgs.shape[2] != onnx_h or imgs.shape[3] != onnx_w:
            raise ValueError(
                f"校准预处理尺寸 {imgs.shape[2:]} 与 ONNX 输入 {(onnx_h, onnx_w)} 不一致；"
                "禁止用拉伸 resize 修补校准数据"
            )
        # 确保 batch 维度匹配
        if imgs.shape[0] != onnx_batch:
            for i in range(imgs.shape[0]):
                calib_data.append(imgs[i : i + 1])
        else:
            calib_data.append(imgs)
        count += 1
        if count >= n_batches:
            break
    print(f"[PTQ-ONNX] 收集了 {len(calib_data)} 个校准样本（每个 shape={calib_data[0].shape}）")

    class YOLOCalibrationReader(CalibrationDataReader):
        """!
        @brief 向 onnxruntime 量化器提供校准数据。
        @details
        将预处理后的 numpy 数组列表逐项包装成
        `{input_name: numpy_array}` 字典。
        """

        def __init__(self, data, input_name):
            """!
            @brief 初始化校准数据读取器。
            @param data 校准样本列表。
            @param input_name ONNX 输入张量名称。
            """
            self._data = data
            self._input_name = input_name
            self._iter = iter(self._data)

        def get_next(self):
            """!
            @brief 返回下一份校准样本。
            @return 形如 `{input_name: numpy_array}` 的字典；耗尽时返回 `None`。
            """
            try:
                imgs = next(self._iter)
                return {self._input_name: imgs}
            except StopIteration:
                return None

        def rewind(self):
            """!
            @brief 将校准样本迭代器重置到起点。
            """
            self._iter = iter(self._data)

    reader = YOLOCalibrationReader(calib_data, input_name)

    # ONNX 预处理：shape inference + optimization，消除 onnxruntime 量化警告
    try:
        import tempfile
        from onnxruntime.quantization import preprocess as quant_preprocess

        preprocessed_path = tempfile.mktemp(suffix=".onnx")
        print("[PTQ-ONNX] 对 ONNX 模型执行预处理（shape inference + optimization）...")
        quant_preprocess.quant_pre_process(
            prepared_onnx_path,
            preprocessed_path,
            skip_symbolic_shape=True,
        )
        print("[PTQ-ONNX] 预处理完成。")

        # 执行静态量化（QDQ 格式，ONNXRuntime 推荐）
        # 注意：ORT CPU EP 的 QDQ→QOperator(QLinearConv) 融合器对「对称 INT8 激活」
        # 兼容性很差，会导致大部分 Conv 退化为 fp32（DequantizeLinear→Conv→QuantizeLinear），
        # 实测 102 个 Conv 仅 12 个能融合成 QLinearConv。
        # 与 seg 主线（step_int8_onnx_seg）对齐：
        #   activation = QUInt8（非对称） + weight = QInt8（per-channel）
        # 该组合是 ORT CPU EP QDQ 融合器的快路径，可达成 ≥95% 的 QLinearConv 比例。
        print("[PTQ-ONNX] 执行 INT8 静态量化...")
        quantize_static(
            model_input=preprocessed_path,
            model_output=output_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,  # QDQ 格式，onnxruntime 推荐
            activation_type=QuantType.QUInt8,  # 与 seg 对齐：非对称 UInt8 激活，
            # ORT CPU EP QDQ→QLinearConv 融合快路径
            weight_type=QuantType.QInt8,
            op_types_to_quantize=None,
            per_channel=True,  # per-channel 权重量化精度更高
            reduce_range=False,
        )
        quantized_model = onnx.load(output_path)
        validate_route = "pre_dfl" if route == "pre_dfl" else "pre_dist"
        from export.det_onnx_routes import validate_det_deployment_contract

        validate_det_deployment_contract(quantized_model, args.imgsz, validate_route, output_path)
        from export.model_manifest import write_model_manifest

        manifest_path = write_model_manifest(
            output_path,
            prepared_onnx_path,
            public_route,
            args.imgsz,
            data_yaml=args.data,
        )
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"[PTQ-ONNX] 量化完成！模型已保存至 {output_path}  ({size_mb:.1f} MB)")
        print(f"[PTQ-ONNX] 清单: {manifest_path}")
        print("  ONNX INT8 可用于主机侧验证；RKNN 部署请使用 export_det_rknn_i8.py / export_seg_rknn_i8.py。")
    finally:
        if preprocessed_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
        cleanup_temp_paths(cleanup_paths)


# ──────────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """!
    @brief 解析量化脚本命令行参数。
    @return 命令行参数对象。
    """
    parser = argparse.ArgumentParser(
        description="检测模型 INT8 量化",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["onnx", "paddle"],
        default="onnx",
        help=(
            "量化方式（默认 onnx）:\n"
            "  onnx   : ONNX INT8 静态量化，用户一步输入 Paddle 权重或 ONNX，内部自动转 pre_dist / pre_dfl\n"
            "  paddle : Paddle ImperativePTQ（fake-quant，调试用）"
        ),
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=(
            "模型权重路径:\n"
            "  onnx 模式   → 接受 .pdparams / .onnx，内部自动转 pre_dist / pre_dfl\n"
            "  paddle 模式 → .pdparams 文件"
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/your.yaml",
        help="数据集配置文件（用于采集校准数据，默认 data/your.yaml）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出路径；默认写到输入权重同目录",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        default=["640"],
        metavar="SIZE",
        help="校准图像尺寸：SIZE、HxW 或 H W（默认 640）",
    )
    parser.add_argument("--route", choices=["predist", "predfl"], help="ONNX 模式要求的部署 route")
    parser.add_argument("--batch", type=int, default=4, help="校准数据 batch size（默认 4）")
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=50,
        help="校准 batch 数量（越多精度越高，一般 50~200 足够，默认 50）",
    )
    return parser.parse_args()


def main() -> int:
    """!
    @brief 脚本主入口。
    @return 成功返回 `0`。
    """
    args = parse_args()
    args.imgsz = normalize_static_imgsz(args.imgsz)

    # 自动设置默认输出路径
    if args.output is None and args.mode == "paddle":
        weights_path = Path(args.weights)
        stem = weights_path.stem
        args.output = str(weights_path.with_name(f"{stem}_int8"))

    print("=" * 60)
    print(f"  模式:       {args.mode.upper()} PTQ")
    print(f"  权重:       {args.weights}")
    print(f"  数据集:     {args.data}")
    print(f"  输出:       {args.output}")
    print(f"  图像尺寸:   {args.imgsz}")
    print(f"  校准 batch: {args.calib_batches} × {args.batch}")
    print("=" * 60)

    if args.mode == "onnx":
        quantize_onnx(args)
    else:
        quantize_paddle(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
