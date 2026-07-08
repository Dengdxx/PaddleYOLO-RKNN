# RKNN 检测部署

> 默认命令在 `PaddleYOLO-RKNN/` 目录内执行。

## 范围

本页说明检测模型部署：
- Paddle 权重导出 ONNX
- RKNN FP16 通用导出
- RKNN INT8：`pre_dist` / `pre_dfl`
- ONNX INT8 PTQ

## 导出概览

```python
from ddyolo26 import YOLO

model = YOLO("runs/detect/exp/weights/best.pdparams")
model.export(format="onnx", imgsz=640, simplify=True)
```

`.pdparams` 是本仓库原生 Paddle 权重格式，不需要 `model.export(format="paddle")`。
进入 RKNN 部署前，只需要从 `.pdparams` 导出 ONNX。

### 导出产物位置

`model.export()` 默认把产物写到输入权重同目录：
- `runs/detect/exp/weights/best.onnx`

仓库内置的 `weights/` 目录只放 Git LFS 预训练权重。手动导出的 ONNX / RKNN
默认写到输入权重同目录；这些生成文件已被 `.gitignore` 忽略，不要提交。

## 检测输出路线

| 路线 | 适用模型 | 输出契约 | 用途 |
|------|----------|----------|--------|
| `e2e` | end2end 检测模型 | `[1,300,6]` | 适合 PC / ONNX 直接推理 |
| `pre_dist` | YOLO26 `reg_max=1` | `[1,4,N] + [1,nc,N]` | RKNN INT8 |
| `pre_dfl` | YOLOv8 `reg_max=16` | `[1,4*reg_max,N] + [1,nc,N]` | YOLOv8 RKNN INT8 |

## NEON 后处理

`bench/predist_tail_bench.cpp` 提供 `scalar` / `scalar_logit` / `neon_logit` 变体，
用于验证 YOLO26 `predist` 后处理尾部的数值一致性和耗时。

端到端评估仍以 `tools/eval/` 的 Python / NumPy 参考实现为准。

## RKNN FP16 通用导出

```bash
# 直接从 ONNX 导出
python export/export_rknn.py \
    --weights runs/detect/exp/weights/best_paddle_raw_fp32_640.onnx \
    --quantize fp16

# 从 .pdparams 自动导出
python export/export_rknn.py \
    --weights runs/detect/exp/weights/best.pdparams \
    --quantize fp16
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `--weights` | ONNX 或 `.pdparams` 模型路径 |
| `--output` | 输出 `.rknn` 路径 |
| `--target` | `rk3588` / `rk3588s` / `rk3576` / `rk3562` |
| `--quantize` | 通用入口仅支持 `fp16` |
| `--compress-weight` | 启用权重压缩 |
| `--model-pruning` | 启用编译期剪枝 |
| `--no-fix` | 跳过 Paddle ONNX 图修复 |

## Paddle ONNX 图修复

Paddle 导出的 end2end ONNX 在 RKNN 上需要额外图修复。通用入口会检测并调用
`tools/fix_paddle_onnx.py`：

- `Expand` → `Tile`
- `Cast → Div → Floor → Cast` → 整数 `Div`
- `Cast → Mul → Cast → Sub` → `Mod`

一般不需要手动运行修复脚本。

## RKNN INT8：`pre_dist`

### 适用范围

- YOLO26 检测
- `reg_max=1`
- 目标是 RK3588 / RK3576 / RK3562 NPU INT8 部署

### 核心思路

NPU 只保留 raw `ltrb` + `cls logits`，CPU 端执行：
- `dist2bbox + stride decode`
- `sigmoid`
- `nms` 或 `nmsfree_exact`

### 一步式导出

```bash
python export/export_det_rknn_i8.py \
    --weights runs/detect/exp/weights/best.pdparams \
    --data coco.yaml \
    --imgsz 640
```

### 参数

| 参数 | 说明 |
|------|------|
| `--weights` | `.pdparams / onnx`，内部自动裁到 `pre_dist`；普通 `.pt` 不支持 |
| `--data` | 校准数据集 YAML |
| `--target` | NPU 平台 |
| `--imgsz` | 输入尺寸 |
| `--calib-images` | 校准图数 |
| `--algorithm` | `auto / normal / mmse / kl_divergence` |
| `--auto-hybrid` | 启用 `auto_hybrid` |

## RKNN INT8：`pre_dfl`

### 适用范围

- YOLOv8 检测模型
- `reg_max > 1`

### 核心思路

把 RKNN 截断点提前到 DFL concat 之前，让 CPU 端执行：
- DFL softmax
- `dist2bbox`
- `sigmoid + NMS`

### 导出

```bash
python export/export_det_rknn_i8.py \
    --weights runs/detect/exp/weights/best_paddle_predfl_fp32_640.onnx \
    --data coco.yaml
```

输出示例：
- `runs/detect/exp/weights/best_paddle_predfl_int8_640.rknn`

## ONNX INT8 PTQ

### 路线

仅保留两条检测量化路线：
- `pre_dist`
- `pre_dfl`

### 导出

```bash
python quant/quantize.py \
    --weights runs/detect/exp/weights/best.pdparams \
    --data coco.yaml \
    --imgsz 640
```

```bash
python quant/quantize.py \
    --weights runs/detect/exp/weights/best.onnx \
    --data coco.yaml
```

### 调试用 Paddle PTQ

```bash
python quant/quantize.py --mode paddle \
    --weights runs/detect/exp/weights/best.pdparams \
    --data coco.yaml
```

### 量化精度评估

```bash
python quant/eval_quant.py \
    --weights runs/detect/exp/weights/best.pdparams \
    --data coco.yaml \
    --calib-batches 50
```

## ONNX FP16 混合精度

```python
from ddyolo26 import YOLO

model = YOLO("runs/detect/exp/weights/best.pdparams")
model.export(format="onnx", imgsz=640, simplify=True, half=True)
```

该路径内部使用 `onnxruntime.transformers.float16`，并对检测头后处理算子保留 FP32。

## 路线选择

1. 先确认模型属于 `pre_dist` 还是 `pre_dfl`
2. 再决定走 RKNN FP16、RKNN INT8 或 ONNX INT8
3. 如果需要具体精度 / FPS 数字，再看 [coco-baselines.md](coco-baselines.md)

## 相关文档

- [README](../README.md)
- [训练与评估](training.md)
- [RKNN 分割部署](deployment-rknn-seg.md)
- [COCO val2017 baseline](coco-baselines.md)
