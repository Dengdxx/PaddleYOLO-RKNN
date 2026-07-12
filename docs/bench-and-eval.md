# 板端延迟与评估表

本文档说明 RK3588 板端 RKNN 延迟测量、频率锁定和评估结果汇总方式。

---

## 1. `bench_rknn_perf` —— 板端 NEON 加速延迟工具

源码：`PaddleYOLO-RKNN/bench/bench_rknn_perf.cpp`

### 功能

- 五输出分割强制使用 full-IO zero-copy：输入和五个输出全部通过
  `rknn_set_io_mem()` 绑定，不保留 `rknn_outputs_get()` 分割兼容路径
- 允许量化模型的 logical input 为 `INT8`；bench 查询 native input 后
  以 `UINT8/NHWC` 图像缓冲绑定，与 RKNN 部署预处理契约一致
- 输出按 `score_sum → cls → box/DFL → mask_coeff + proto` 分阶段同步；
  无 survivor、无精确候选或无有效框时立即结束
- 分割 mask 直接消费原生 `NC1HWC2` proto：小 ROI 在原生布局内完成
  INT8 反量化与 NEON FMLA，大 ROI 融合恢复为 NCHW FP32；
  sigmoid 融合最终写回，resize 后单遍完成量化与二值化
- 每个 anchor 仅保留精确反量化后置信度最高的类别，与 Python 解码的
  `argmax` 语义一致；随后执行 class-aware NMS
- `seg_predfl` 对每个候选一次反量化四个 DFL 方向，使用 NEON
  max / fast-exp / 加权求和计算 softmax expectation
- 内置四种后处理路径，与 [导出管线](export-pipeline.md) 第 1 节中的 `route` 一一对应：
  - `predist`：YOLO26，`reg_max=1`，4 个距离直接回归
  - `predfl`：YOLOv8，`reg_max=16`，含 DFL softmax-expectation
  - `seg_predist`：分割 predist + mask 后处理
  - `seg_predfl`：YOLOv8-Seg，分割 predfl + mask 后处理
  - `none`：仅推理，跳过后处理
- 输出延迟统计（best / P50 / P90 / avg）：
  - `npu_pure_ms`：来自 `RKNN_QUERY_PERF_RUN`（NPU 纯耗时）
  - `io_wall_ms`：输入同步 + run + 实际需要的输出同步/布局准备墙钟耗时
  - `postproc_ms`：CPU + NEON 后处理耗时
  - `e2e_ms`：端到端墙钟耗时
- 五输出额外报告 `output_sync_ms`、`native_sync_bytes`、`ready_mask`
  和 `fetch_outcome`；同步耗时与字节数为全部计时轮次的帧均值
- mask 像素计数/hash 属于正确性校验，默认关闭；启用时单独报告
  `mask_verify_ms`，不计入单帧 `postproc_ms` 与 `e2e_ms`

### CLI

```
bench_rknn_perf
  --model M.rknn
  --core 0|1|2|all                # NPU core 绑定
  --postproc predist|predfl|seg_predist|seg_predfl|none
  --score-sum on|off               # 分割五输出预筛 A/B，默认 on
  --mask-verify on|off             # mask 计数/hash 校验，默认 off
  --mask-class-ids all|0,1         # 生成 mask 的类别，默认 0
  --mask-output-size WxH           # mask 输出坐标尺寸，默认 640x480
  --input F.rgb                    # 单 context 的 HWC RGB uint8 原始帧
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
cd bench
g++ -O3 -DNDEBUG -std=c++17 -mcpu=cortex-a76 \
    -I. \
    -I<rknn-runtime include 路径> \
    $(pkg-config --cflags opencv4) \
    bench_rknn_perf.cpp \
    postprocess/seg_class_selector.cpp \
    postprocess/five_output_runtime.cpp \
    postprocess/roi_mask_decoder.cpp \
    -o bench_rknn_perf \
    $(pkg-config --libs opencv4) -lrknnrt -pthread
```

PaddleYOLO-RKNN 仓库不内置 RKNN Runtime；编译时需要在板端提供本机可用的
`rknn_api.h` 与 `librknnrt.so` 路径，并安装 `g++`、`pkg-config`
与 OpenCV 4 开发包。

### 单跑示例

```bash
# $MODELS_ROOT 指向部署产物目录（例如 <repo>/artifacts/coco_baselines）
./bench_rknn_perf \
  --model "$MODELS_ROOT/yolo26n/_eval/yolo26n_paddle_predist_int8_640.rknn" \
  --warmup 5 --runs 50 --core 0 --postproc predist --sram off

# 分割五输出完整后处理；输入为恰好 640×640×3 字节的 HWC RGB uint8 原始帧
./bench_rknn_perf \
  --model "$MODELS_ROOT/yolov8n-seg/yolov8n-seg_paddle_seg_predfl_int8_640.rknn" \
  --input frame_640.rgb --warmup 20 --runs 100 --core all \
  --postproc seg_predfl --score-sum on --mask-verify off \
  --iou-thr 0.45 --mask-class-ids 0 --mask-output-size 640x480

# 2×NPU context 吞吐 / 长压测示例（必须先锁频）
./bench_rknn_perf \
  --model "$MODELS_ROOT/yolov8n-seg/yolov8n-seg_paddle_seg_predfl_int8_640.rknn" \
  --fps-workers 2 --fps-core-map 0,1 --fps-seconds 600 \
  --warmup 50 --postproc seg_predfl --sram shared \
  --json "$BENCH_JSON_OUT"
```

`seg_predist` / `seg_predfl` 会执行完整的 coeff×proto、上采样、sigmoid、
bbox crop 与二值化。正式延迟使用默认的 `--mask-verify off`，避免像素统计
和 hash 污染 E2E；需要验证输出一致性时，两组都传 `--mask-verify on`，
再切换 `--score-sum on|off`，候选数、最终实例数、有效像素与 hash 应一致。
启用校验后的耗时通过 `mask_verify_ms` 单列，仍从单帧 postproc/E2E 中扣除。

mask 类别白名单在 NMS 后、同步 `mask_coeff + proto` 前应用。若本帧没有
白名单类别，bench 以 `fetch_outcome=no_mask_classes` 提前结束，避免两块
mask 输出的 DMA 同步和解码。做跨实现 A/B 时必须显式对齐 `--iou-thr`、
`--mask-class-ids` 与 `--mask-output-size`；类别、NMS 和最终 mask 尺度不同的
结果不能直接比较 E2E。

当前 640 五输出模型的原生输出总计 `1,894,400 B`。按需同步的理论口径：

- `score_sum` 无 survivor：`134,400 B`，减少 `92.91%`
- 有 survivor 但精确分类无 seed：`268,800 B`，减少 `85.81%`
- 有 seed 但无有效框：`806,400 B`，减少 `57.43%`
- 需要 mask：`1,894,400 B`；同步量不减，但仍使用 full-IO 与原生 proto

survivor-first 差分验证（AArch64 NEON）：

- 10 类×8400 anchor 稀疏夹具中，20 个 survivor 的分类读取量从
  `84000` 降为 `200`，减少 `99.76%`；500 轮随机量化差分结果与完整扫描一致。
- 10 类 640 五输出模型的真实图像 A/B 中，`score_sum=on` 实际读取
  `49×10=490` 个分类值，`off` 读取 `84000` 个；两者均为 49 个候选、
  9 个 NMS 结果，mask hash 均为 `de998913f2e5b1c8`。

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
- 通过 `--iou-thr`、`--mask-class-ids`、`--mask-output-size` 统一下发
  分割 A/B 口径；默认分别为 `0.45`、`0`、`640x480`
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
    --input frame_640.rgb \
    --summary-out "$MODELS_ROOT/_bench_summary.json" \
    --bench-status official \
    --frequency-profile cpu_npu_ddr_max \
    --sram off \
    --score-sum on

# 或使用环境变量（适合批处理）
export BENCH_BIN=./bench/bench_rknn_perf
export BENCH_MODELS_ROOT="$MODELS_ROOT"
export BENCH_SCORE_SUM=on
export BENCH_INPUT="$PWD/frame_640.rgb"
python scripts/board_bench_all.py

# 3. 按板端项目规范恢复 governor
```

正式 E2E 数据必须通过 `--input` 或 `BENCH_INPUT` 传入同一张真实帧。
不传时 bench 使用全零输入，仅适合检查 NPU 运行，不能用于比较
依赖候选数和 ROI 面积的完整分割后处理延迟。

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

分割评估的 `--mask-eval fast|native` 对自定义数据集和 COCO 均生效；COCO
路径最终都会生成原图尺寸 RLE，但分别保留 proto 分辨率快速口径或 native
上采样口径，并在结果 JSON 中记录所选口径。

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
