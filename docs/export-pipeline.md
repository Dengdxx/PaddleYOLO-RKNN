# 导出管线

本文说明 YOLO26 / YOLOv8 系列模型从 Paddle 权重导出到 ONNX、RKNN（INT8）的命名和脚本。

> 配套阅读：[板端延迟与评估表](bench-and-eval.md)。

---

## 1. 权重命名

Paddle / ONNX / RKNN 产物使用下面的命名：

```
<stem>_<framework>_<route>_<precision>_<imgsz>.<ext>
```

| 字段 | 取值 | 说明 |
|------|------|------|
| `stem` | `yolo26n` / `yolov8n` / `yolo26n-seg_1` / `yolov8n-seg` | 模型基名 |
| `framework` | `paddle` | 训练框架来源 |
| `route` | `e2e` / `raw` / `predist` / `predfl` / `seg_predist` / `seg_predfl` | 头部导出路线（见下） |
| `precision` | `fp32` / `int8` | 数值精度 |
| `imgsz` | `640` / `480x640` | 静态输入尺寸，矩形按 `height x width` |
| `ext` | `.onnx` / `.rknn` / `.pdparams` | 产物格式 |

**route 取值含义：**

| route | 头部行为 | 适用模型 |
|-------|---------|----------|
| `e2e` | 端到端输出，含模型内 TopK / NMS-free 后处理 | YOLO26 系列普通 FP32 ONNX |
| `raw` | 原始 one2many 输出，CPU 侧再做 NMS | YOLOv8 系列普通 FP32 ONNX |
| `predist` | `reg_max = 1`，跳过 DFL，直接回归 4 个距离 | YOLO26 系列检测 |
| `predfl`  | `reg_max = 16`，保留 DFL 卷积 | YOLOv8 系列检测 |
| `seg_predist` | 分割版 predist，附 mask 系数 + proto | YOLO26-Seg |
| `seg_predfl` | 分割版 predfl，附 mask 系数 + proto | YOLOv8-Seg |

**Paddle 权重命名：** 使用 `<stem>.pdparams`。

`predict/export`、ONNX/RKNN 量化和静态 evaluator 支持
`--imgsz 640`、`--imgsz 480x640` 与 `--imgsz 480 640`。高度和宽度
必须分别对齐 `max(32, 模型最大 stride)`；不支持 RKNN 运行时动态尺寸。
标准训练与 Paddle `val` 仍使用标量方形 `imgsz`。

INT8 校准图像会在 Toolkit 编译前显式执行 RGB、居中 letterbox、
填充 114 和 `scaleup=false`，不依赖 RKNN Toolkit 隐式拉伸。
这与 Ultralytics `val`/INT8 校准的不放大语义对齐；通用
Paddle `predict` 仍保留上游默认 `scaleup=true`。静态部署评测和
SmartCar 必须使用 manifest 声明的 `scaleup=false` 契约。
每个部署产物同时生成 `<model>.<ext>.model.yaml`，记录 H/W、route、
anchor/proto 契约、类别名和最终模型 SHA-256。

**示例：**

```
weights/yolov8/yolov8n.pdparams                                  # Paddle 源权重
weights/yolov8/yolov8n_paddle_raw_fp32_640.onnx
weights/yolov8/yolov8n_paddle_predfl_fp32_640.onnx
weights/yolov8/yolov8n_paddle_predfl_int8_640.rknn
```

默认情况下，导出产物与输入模型放在同一个目录。需要集中保存时，再显式传
`--out` 或 `--output`。

---

## 2. `dual_raw` 导出标志

RKNN INT8 量化需要保留原始头输出，避免 NMS 子图进入量化图，也避免 DFL 在 NPU 上拖慢速度。导出时使用 `dual_raw`：

- 配置入口：
  - `ddyolo26/cfg/default.yaml`：新增 `dual_raw: False`
  - `ddyolo26/cfg/__init__.py` 的 `BOOL_KEYS` 中加入 `dual_raw`
- 头部支持（`ddyolo26/nn/modules/head.py`）：
  - `Detect.export_dual_raw` 类属性（约 L96）
  - `Detect._forward_export`（约 L347）：当 `export_dual_raw=True` 时，返回 `(boxes, scores)` 两个 raw 张量
  - `Segment._forward_export`（约 L555）：返回 `(boxes, scores, mask_coeff, proto)`
  - `Segment26._forward_export`（约 L677）：同上
- Exporter 串联（`ddyolo26/engine/exporter.py`）：
  - 在 Detect-iter 段为每个匹配 head 设置 `m.export_dual_raw = True`
  - `output_names` 分支选择：
    - 检测：`["dual_raw_boxes", "dual_raw_scores"]`
    - 分割：`["dual_raw_boxes", "dual_raw_scores", "dual_raw_mask_coeff", "dual_raw_proto"]`

**用途：** RKNN INT8 量化时保留 raw head 输出。YOLO26 由 CPU 执行 exact TopK，YOLOv8 执行 DFL 与 NMS。

YOLO26 `seg_predist` 的四个 `dual_raw` 张量就是正式部署契约，不追加
`score_sum`；YOLOv8 `seg_predfl` 才追加 `score_sum` 并形成五输出。
批量导出复用缓存前会按模型族分别验证四输出或五输出契约。

---

## 3. `load_checkpoint` 的类别数处理

文件：`ddyolo26/nn/tasks.py` 中的 `load_checkpoint`（约 L740）

Paddle 路径在 rebuild model 时会从 `ckpt['nc']` 注入类别数，确保 head 维度与权重一致。

> Paddle 重训权重导出 / 推理时依赖这个类别数注入。

---

## 4. RKNN INT8 使用 `opset=13`

文件：`quant/quantize.py` 中的 `auto_export_onnx`（paddle 分支）

RKNN INT8 的 Paddle 分支显式
`yolo.export(format='onnx', opset=13, ...)`，避免超出 RKNN-Toolkit2 的 opset 支持范围。

> opset=13 是与 RKNN 工具链验证可用的最低稳定档位。普通 FP32 ONNX 导出
> 仍走 `Exporter` 的默认 opset 选择；如果后续产物要进入 RKNN，需在对应脚本中显式
> 以源模型重新导出到受支持 opset。通用入口不会只修改版本号；仅在 ONNX
> version converter 能完成真实转换且 checker 通过时才接受高版本模型。

---

## 5. 工具脚本

编排脚本在 `PaddleYOLO-RKNN/scripts/`；跨 conda 环境的导出 helper 在 `PaddleYOLO-RKNN/export/`。
按导出 → 评估 → 板上 bench → 汇总的顺序排列：

| 脚本 | 作用 |
|------|------|
| `export_all_models.py` | 批量导出编排器；Paddle 环境负责 fp32 / int8 ONNX 和 route FP32 ONNX，RKNN 环境负责编译 RKNN；通过 `--python-paddle` / `--python-rknn` 分流到不同 conda 环境 |
| `export_coco_baselines.py` | COCO baseline 导出入口；使用本地 Paddle 权重，并以 `--python-paddle` / `--python-rknn` 分离导出与 RKNN 编译环境 |
| `export/export_fp32_onnx.py` | 单模型 e2e/raw FP32 ONNX helper；仅使用 `ddyolo26.YOLO` |
| `export/export_seg_onnx_i8.py` | segmentation INT8 ONNX helper；负责 `seg_predist / seg_predfl` 裁剪 + ORT 静态量化 |
| `tools/eval/` | host / 板端评测入口；支持 ONNX / RKNN、detect / segment；可用 `python -m tools.eval` 调用 |
| `eval_coco_baselines.py` | 批量调用评测入口评测 COCO baseline ONNX 产物，生成 `_eval/*.eval.json` 中间指标供汇总使用 |
| `board_bench_all.py` | 在板上跑 `bench_rknn_perf` 全模型矩阵（`core=0` / `core=all`），输出 `_bench_summary.json` |
| `build_coco_baseline_table.py` | 读取 COCO baseline `_eval/*.eval.json` / RKNN bench JSON，生成最终 `_host_eval_table.md` |

### 示例命令

```bash
# 1. 导出（Paddle-only，全精度）
python scripts/export_all_models.py \
  --weights weights/yolov8/yolov8n.pdparams \
  --data ddyolo26/cfg/datasets/coco8.yaml \
  --imgsz 640 --task detect --route predfl

# 2. 主机侧评估 ONNX（JSON 只作为 Markdown 汇总输入）
python -m tools.eval \
  --backend onnx \
  --model 'weights/yolov8/*.onnx' \
  --data ddyolo26/cfg/datasets/coco8.yaml \
  --output <eval-json-out>

# 3. 板上跑 RKNN 精度与延迟矩阵（详见 bench-and-eval.md）
ssh orangepi "cd <repo>/PaddleYOLO-RKNN && python scripts/board_bench_all.py ..."

# 4. 生成 Markdown 汇总
python scripts/build_coco_baseline_table.py --markdown
```

### Python 环境选择

按职责拆环境：

- `pdrk`：安装 `requirements-paddle.txt`，用于 Paddle 训练和 Paddle → ONNX 导出。
- `rknn`：在 Ubuntu/WSL 中安装 `requirements-rknn.txt`，只用于 ONNX → RKNN 编译。

`requirements-paddle.txt` 是 Paddle 侧环境定义；同一份文件可用于 Windows
训练环境，也可用于 Ubuntu/WSL 的 ONNX 导出环境。Windows 原生
`paddlepaddle-gpu==3.3.1` + `paddle2onnx==2.1.0` 组合可能在
`paddle2onnx_cpp2py_export` 上遇到 DLL/ABI 加载失败；若遇到该问题，先在
Ubuntu/WSL 的 `pdrk` 环境产出 route FP32 ONNX，再切到 `rknn` 环境执行
ONNX → RKNN。

Linux/WSL 的 `rknn` 环境不安装 Paddle，只消费 ONNX 并编译 RKNN。

RKNN 阶段消费量化前的 route FP32 ONNX，由 RKNN Toolkit 自己执行 INT8
量化，ORT QDQ ONNX 会被拒绝。拆分环境时，`--python-paddle` 指向 `pdrk` 解释器，`--python-rknn`
指向 Linux/WSL `rknn` 解释器。跨系统执行时，也可以先在 `pdrk` 侧产出 ONNX，
再在 WSL 中只运行 ONNX → RKNN 阶段。

```powershell
# pdrk: Paddle 权重 -> route FP32 ONNX
python scripts/export_all_models.py `
  --weights weights/yolo26/yolo26n.pdparams `
  --data ddyolo26/cfg/datasets/coco8.yaml `
  --imgsz 640 --task detect --route predist `
  --steps fp32predist_paddle
```

```bash
# WSL / rknn: route FP32 ONNX -> RKNN
python export/export_det_rknn_i8.py \
  --weights weights/yolo26/yolo26n_paddle_predist_fp32_640.onnx \
  --data ddyolo26/cfg/datasets/coco8.yaml \
  --imgsz 640 --mode int8 --calib-images 1
```

`scripts/export_coco_baselines.py` 支持同名参数，并会继续透传给
`export_all_models.py` 与 RKNN 导出脚本。

`[ONNX FP32]` / `[ONNX INT8]` 精度段、`[RKNN INT8]` 精度段需要按相同 framework 单独跑 `val`，工具会把结果幂等写回，重复执行不会破坏已有段。

---

## 相关入口

- 板端延迟与评估表：[bench-and-eval.md](bench-and-eval.md)
- 训练与评估：[training.md](training.md)
- RKNN 检测部署：[deployment-rknn-det.md](deployment-rknn-det.md)
- RKNN 分割部署：[deployment-rknn-seg.md](deployment-rknn-seg.md)
