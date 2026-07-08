# 板端延迟与评估表

本文档说明 RK3588 板端 RKNN 延迟测量、频率锁定和评估结果汇总方式。

---

## 1. `bench_rknn_perf` —— 板端 NEON 加速延迟工具

源码：`PaddleYOLO-RKNN/bench/bench_rknn_perf.c`

### 功能

- 使用 NEON intrinsics（`int8x16_t -> float32x4_t`）加速 INT8 dequant
- 对分类分支使用 NEON 融合 `dequant logits + per-anchor max`，并用
  `logit(conf_thr)` 比较替代 `sigmoid + threshold`，数学等价且少一次全量
  `expf`/写回
- 内置四种后处理路径，与 [导出管线](export-pipeline.md) 第 1 节中的 `route` 一一对应：
  - `predist`：YOLO26，`reg_max=1`，4 个距离直接回归
  - `predfl`：YOLOv8，`reg_max=16`，含 DFL softmax-expectation
  - `seg_predist`：分割 predist + mask 后处理
  - `seg_predfl`：YOLOv8-Seg，分割 predfl + mask 后处理
  - `none`：仅推理，跳过后处理
- 输出延迟三件套（avg / min / max）：
  - `npu_pure_ms`：来自 `RKNN_QUERY_PERF_RUN`（NPU 纯耗时）
  - `postproc_ms`：CPU + NEON 后处理耗时
  - `e2e_ms`：端到端（拷贝 + infer + postproc）

### CLI

```
bench_rknn_perf
  --model M.rknn
  --core 0|1|2|all                # NPU core 绑定
  --postproc predist|predfl|seg_predist|seg_predfl|none
  --sram off|private|shared        # RKNN SRAM 初始化策略，默认 off
  --warmup N                       # 默认 10
  --runs N                         # 默认 200
  --fps-workers N                  # 多 context 离线 FPS；配合 --fps-core-map 使用
  --fps-core-map 0,1               # 2×NPU 锁频测试推荐 core0+core1
  --fps-seconds F                  # FPS 压测时长
  [--json OUT.json]                # 可选，机读输出
```

### 编译

在板端（RK3588 / cortex-a76）：

```bash
gcc -O3 -mcpu=cortex-a76 -ffast-math \
    -I<rknn-runtime include 路径> \
    bench_rknn_perf.c -o bench_rknn_perf \
    -lrknnrt -lm -pthread
```

PaddleYOLO-RKNN 仓库不内置 RKNN Runtime；编译时需要在板端提供本机可用的
`rknn_api.h` 与 `librknnrt.so` 路径。

### 单跑示例

```bash
# $MODELS_ROOT 指向部署产物目录（例如 <repo>/artifacts/coco_baselines）
./bench_rknn_perf \
  --model "$MODELS_ROOT/yolo26n/_eval/yolo26n_paddle_predist_int8_640.rknn" \
  --warmup 5 --runs 50 --core 0 --postproc predist --sram off

# 2×NPU context 吞吐 / 长压测示例（必须先锁频）
./bench_rknn_perf \
  --model "$MODELS_ROOT/yolov8n-seg/yolov8n-seg_paddle_seg_predfl_int8_640.rknn" \
  --fps-workers 2 --fps-core-map 0,1 --fps-seconds 600 \
  --warmup 50 --postproc seg_predfl --sram shared \
  --json "$BENCH_JSON_OUT"
```

## 2. 频率锁定

为了让延迟测量可复现，跑 bench 前必须把 **CPU / NPU / DDR** 锁到最高频。
RKNN runtime 版本固定后，CPU/NPU/DDR 均设为 max 频。

PaddleYOLO-RKNN 仓库不内置板端锁频脚本。手动 bench 时可在板端用
项目外部脚本或以下等价动作完成锁频：

- 写 `/sys/class/devfreq/fdab0000.npu/governor=userspace`，并将其设到 `available_frequencies` 中的最高档
- 写 `/sys/class/devfreq/dmc/governor=performance`，锁定 DDR/DMC 性能档
- 把所有 CPU 的 `scaling_governor` 设为 `performance`
- 读取对应 sysfs 节点核对 governor / 频率

> 手动 bench 时若发现结果波动大，先检查 CPU / NPU / DDR 是否都已锁到预期频率。

---

## 3. 板端 bench 矩阵

入口：`PaddleYOLO-RKNN/scripts/board_bench_all.py`

- 遍历 `<models-root>/*.rknn`、`<models-root>/*/*.rknn` 与 `<models-root>/*/_eval/*.rknn`，按文件名解析出 `route`
- `--models-root` 可指向总目录，也可直接指向单个模型目录
- 对每个模型分别跑 `core=0` 与 `core=all` 两次
- 每个模型目录的 `_eval/` 下写入 `<stem>.bench_core0.json` / `<stem>.bench_coreall.json`
- 聚合结果写入 `--summary-out` 指定的 JSON：

下面的 JSON 只展示字段结构，数值不是实测结论：

```jsonc
[
  {
    "model": "yolov8n-seg_paddle_seg_predfl_int8_640.rknn",
    "core": "0",
    "postproc": "seg_predfl",
    "npu_pure_ms": 19.7,
    "postproc_ms": 5.7,
    "e2e_avg_ms": 25.9,
    "bench_status": "official",
    "frequency_profile": "cpu_npu_ddr_max",
    "stem": "yolov8n-seg_paddle_seg_predfl_int8_640",
    "model_dir": "yolov8n-seg"
  }
]
```

`board_bench_all.py` 的 `fps = 1000 / e2e_avg_ms` 是单 context 延迟换算 FPS；
`bench_rknn_perf --fps-workers` 输出的 `offline_fps` 是多 context 并发离线吞吐，二者不能混作同一列。

### 使用方式

```bash
# 1. 板上锁频，并确认 RKNN runtime 为 2.3.2
#    PaddleYOLO-RKNN 仓库不内置锁频脚本；请使用板端项目自己的锁频方式。

# 2. 跑全模型矩阵（耗时几分钟到十几分钟，取决于模型数量）
#    路径通过 CLI 或环境变量传入；在 PaddleYOLO-RKNN/ 目录下执行
cd <repo>/PaddleYOLO-RKNN
python scripts/board_bench_all.py \
    --bench ./bench/bench_rknn_perf \
    --models-root "$MODELS_ROOT" \
    --summary-out "$MODELS_ROOT/_bench_summary.json" \
    --bench-status official \
    --frequency-profile cpu_npu_ddr_max \
    --sram off

# 或使用环境变量（适合批处理）
export BENCH_BIN=./bench/bench_rknn_perf
export BENCH_MODELS_ROOT="$MODELS_ROOT"
python scripts/board_bench_all.py

# 3. 按板端项目规范恢复 governor
```

---

## 4. 评估结果文件

每个模型目录的 `_eval/` 下保存评测与测速 JSON：

```
artifacts/coco_baselines/yolov8n/_eval/
```

| 文件 | 来源工具 | 数据来源 |
|----|---------|---------|
| `<stem>.eval.json` | `python -m tools.eval --backend onnx` | ONNX FP32 / INT8 模型评测 |
| `<stem>.rknn.eval.json` | 板端 `python -m tools.eval --backend rknn` | RKNN INT8 模型评测 |
| `<stem>.bench_core0.json` | `board_bench_all.py` | RKNN core0 延迟 |
| `<stem>.bench_coreall.json` | `board_bench_all.py` | RKNN coreall 延迟 |

> 评测使用 `PaddleYOLO-RKNN/tools/eval/`（与 ultralytics `ap_per_class` 对齐），host / 板端、detect / segment 都走这个入口。

---

## 5. Markdown 汇总

`scripts/build_coco_baseline_table.py` 读取 `_eval/*.json` 和 RKNN bench JSON，生成汇总表：

```bash
python scripts/build_coco_baseline_table.py --markdown
```

Markdown 汇总用于对比 ONNX 量化影响、RKNN 部署差异和板端延迟。

---

## 相关文档

- 导出管线（命名约定 / dual_raw / 工具脚本）：[export-pipeline.md](export-pipeline.md)
- 训练与评估：[training.md](training.md)
- COCO baseline：[coco-baselines.md](coco-baselines.md)
