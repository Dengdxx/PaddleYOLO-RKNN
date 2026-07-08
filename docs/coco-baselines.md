# COCO val2017 baseline

本页给出 YOLO26 / YOLOv8 在 **COCO val2017 子集**（前 1000 张，按文件名 sorted）下的 baseline：

- 模型：`yolo26n` / `yolo26n-seg` / `yolov8n` / `yolov8n-seg`
- 框架来源：`paddle`（本仓 `ddyolo26` Paddle 权重 export）
- 链路：`e2e`（端到端，含 NMS）/ `predist`（YOLO26 `reg_max=1` 预解码）/ `predfl`（YOLOv8 DFL 预解码）/ `seg_predist`（YOLO26-Seg）/ `seg_predfl`（YOLOv8-Seg DFL）
- 精度：`fp32` / `int8`
- 后端：`onnx`（host GPU，ONNX Runtime CUDAExecutionProvider）/ `rknn`（板端，OrangePi 5 RK3588S，NPU）

精度指标由 `python -m tools.eval` 产生，与 Ultralytics `model.val()` 的度量设置一致（`conf=0.001`、`iou=0.7`、letterbox 预处理、自动检测输出格式 + 解码 + NMS）。RKNN 速度指标来自 `bench/bench_rknn_perf.c`：`RKNN_QUERY_PERF_RUN` 给出纯 NPU 时间，外层计时给出 `rknn_inputs_set + rknn_run + rknn_outputs_get + CPU/NEON 后处理` 的端到端时间。RKNN 速度测试使用 RKNN runtime / toolkit `2.3.2`，CPU/NPU/DDR 锁到 max 频。

## 评测命令

```bash
# 1) 准备 1000 张子集
mkdir -p artifacts/coco/subsets
ls artifacts/coco/images/val2017 | sort | head -n 1000 > artifacts/coco/subsets/val1000.list

# 2) host ONNX（FP32 + INT8）
python -m tools.eval --backend onnx \
  --model artifacts/coco_baselines/<model>/<stem>.onnx \
  --data ddyolo26/cfg/datasets/coco-val2017-only.yaml \
  --imgsz 640 --conf 0.001 --iou 0.7 --max-images 1000 \
  --output artifacts/coco_baselines/<model>/_eval/<stem>.eval.json

# 3) 板端 RKNN 精度（INT8）
ssh orangepi@<board> "cd <repo>/PaddleYOLO-RKNN && python3 -m tools.eval \
  --backend rknn --model artifacts/coco_baselines/<model>/<stem>.rknn \
  --data ddyolo26/cfg/datasets/coco-val2017-only.yaml \
  --imgsz 640 --conf 0.001 --iou 0.7 --max-images 1000 \
  --output artifacts/coco_baselines/<model>/_eval/<stem>.rknn.eval.json"

# 4) 板端 RKNN 速度（需先确认 RKNN 2.3.2，并用板端项目自己的方式锁 CPU/NPU/DDR max 频）
ssh orangepi@<board> "cd <repo>/PaddleYOLO-RKNN && python scripts/board_bench_all.py \
  --bench ./bench/bench_rknn_perf \
  --models-root artifacts/coco_baselines/<model> \
  --summary-out artifacts/coco_baselines/<model>/_eval/<model>_board_bench_summary.json \
  --bench-status official \
  --frequency-profile cpu_npu_ddr_max"

# 5) 重建汇总表
python scripts/build_coco_baseline_table.py --markdown
```

> 板端拉回结果时加 `.rknn.eval.json` 后缀，避免与 host 同名 ONNX 结果冲突，
> 见 `scripts/build_coco_baseline_table.py`。

## 结果

下表由各模型 `_eval/*.json` 通过 `scripts/build_coco_baseline_table.py` 汇总生成。
需要重新生成 Markdown 快照时，可执行 `python scripts/build_coco_baseline_table.py --markdown`。

### `yolo26n`

| framework | 链路 | 精度 | 后端 | mAP50 | mAP50-95 | RKNN core0 NPU (ms) | RKNN core0 E2E (ms) | RKNN coreall NPU (ms) | RKNN coreall E2E (ms) | RKNN测速口径 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| paddle | e2e | fp32 | onnx | 0.5752 | 0.4213 | — | — | — | — | — |
| paddle | predist | int8 | onnx | 0.5379 | 0.3881 | — | — | — | — | — |
| paddle | predist | int8 | rknn | 0.5536 | 0.3994 | 21.8 | 22.5 | 22.6 | 23.1 | RKNN 2.3.2 + CPU/NPU/DDR max |

### `yolo26n-seg`

| framework | 链路 | 精度 | 后端 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | RKNN core0 NPU (ms) | RKNN core0 E2E (ms) | RKNN coreall NPU (ms) | RKNN coreall E2E (ms) | RKNN测速口径 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| paddle | e2e | fp32 | onnx | 0.5701 | 0.4140 | 0.5392 | 0.3515 | — | — | — | — | — |
| paddle | seg_predist | int8 | onnx | 0.5426 | 0.3870 | 0.5190 | 0.3332 | — | — | — | — | — |
| paddle | seg_predist | int8 | rknn | 0.5588 | 0.4004 | 0.5306 | 0.3455 | 25.8 | 26.9 | 26.6 | 27.7 | RKNN 2.3.2 + CPU/NPU/DDR max |

### `yolov8n`

| framework | 链路 | 精度 | 后端 | mAP50 | mAP50-95 | RKNN core0 NPU (ms) | RKNN core0 E2E (ms) | RKNN coreall NPU (ms) | RKNN coreall E2E (ms) | RKNN测速口径 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| paddle | e2e | fp32 | onnx | 0.5229 | 0.3758 | — | — | — | — | — |
| paddle | predfl | int8 | onnx | 0.5176 | 0.3713 | — | — | — | — | — |
| paddle | predfl | int8 | rknn | 0.5220 | 0.3751 | 16.2 | 21.6 | 16.0 | 21.5 | RKNN 2.3.2 + CPU/NPU/DDR max |

### `yolov8n-seg`

| framework | 链路 | 精度 | 后端 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | RKNN core0 NPU (ms) | RKNN core0 E2E (ms) | RKNN coreall NPU (ms) | RKNN coreall E2E (ms) | RKNN测速口径 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| paddle | e2e | fp32 | onnx | 0.5315 | 0.3837 | 0.5010 | 0.3200 | — | — | — | — | — |
| paddle | seg_predfl | int8 | onnx | 0.5190 | 0.3728 | 0.4895 | 0.3091 | — | — | — | — | — |
| paddle | seg_predfl | int8 | rknn | 0.5300 | 0.3811 | 0.4998 | 0.3168 | 19.6 | 25.8 | 19.5 | 25.8 | RKNN 2.3.2 + CPU/NPU/DDR max |

## 口径说明

- **精度入口**：ONNX 与 RKNN 精度均由 `python -m tools.eval` 产生；ONNX 使用 GPU provider，默认 `conf=0.001 / iou=0.7 / max_det=300`，box/mask AP 计算使用参考验证语义。
- **链路基线**：部署链路的量化和 RKNN 差异按同输出路线的 ONNX FP32 / ONNX INT8 对比，避免把 e2e 与预解码路线混作单变量比较。
- **RKNN 延迟字段**：`RKNN core0/coreall NPU (ms)` 来自 `RKNN_QUERY_PERF_RUN`，`RKNN core0/coreall E2E (ms)` 包含输入设置、推理、输出获取与 CPU/NEON 后处理。

## 兼容性说明

- `tools/eval/cli.py` 的 `detect_output_format` 会按 shape 自动选择解码：
  - `(1, 300, 6)` → `e2e`（自动识别 xyxy/cxcywh）
  - `(1, 4+nc, ~300)` → `one2one_raw`（YOLO26 头）
  - `(1, 4+nc, ≥1000)` → `yolov8_raw`（YOLOv8 8400 锚点 + per-class NMS）
  - 多输出 + reg_max=1 → `pre_dist`
  - 多输出 + DFL → `pre_dfl`
  - seg 多输出 → `seg_pre_dist`
  - seg 多输出 + DFL → `seg_pre_dfl`
- 解码语义：letterbox + sigmoid 概率 + `conf=0.001` + `iou=0.7` + `max_det=300`。YOLO26 one2one/e2e 保持 NMS-free top-k；YOLOv8 raw 与 Paddle TopK `cxcywh` e2e 路径按类别执行 NMS，避免把候选框编码差异计入精度差异。
