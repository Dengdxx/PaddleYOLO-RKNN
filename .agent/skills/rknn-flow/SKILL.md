---
name: rknn-flow
description: 在 PaddleYOLO-RKNN 中把 YOLO 模型走完 Paddle 训练、ONNX 导出、RKNN 编译或 RK3588 板端验证时使用；当用户要求验证 README 的训练到 RKNN 流程、跑 smoke/full training、导出 RKNN、板端测速，或比较 Paddle/ONNX/RKNN 指标时使用。
---

# RKNN Flow

把单个模型沿本仓库真实部署链路跑通：Paddle 训练、route ONNX 导出、RKNN 编译，以及可选的 RK3588 板端验证。把它当成带硬门槛的检查清单，不要当成松散建议。

## 参数卡

执行命令前先整理参数卡。只询问无法从仓库或用户最新消息中安全推断的内容。

必需项：

- 模型：精确短名或权重路径，例如 `yolo26n`、`yolo26n-seg`、`yolov8n`、`yolov8n-seg`。
- 数据集：`data.yaml` 路径，或包含 `data.yaml` 的数据集根目录。
- 任务：`detect` 或 `segment`；只有模型名非常明确时才自动推断。
- 训练分辨率：`imgsz`。
- 训练预算：`epochs`、`batch`、`device`。
- 输出名：训练 run 名；导出产物默认放在输入权重或输入 ONNX 同目录。

按需询问：

- 用户要 Windows 训练、WSL 导出、RKNN 编译、板端测速、板端精度评估，还是完整链路。
- RKNN INT8 校准规模；快速验证常用 `50` 张图。
- 板卡连接信息：主机/IP、用户名、认证方式、板端工作目录和是否已有 RKNN Runtime。
- 是否把完整数据集同步到板端。做板端精度评估时，默认只同步验证集图片和标签，除非用户另有要求。

完成标准：参数卡已经列出模型、数据集 YAML、任务、图像尺寸、epochs、batch、device、导出 route、训练 run 名和用户要求的验证范围。

## Route 选择

优先使用本仓库文档中的成熟 RKNN 主线：

| 模型族 | 任务 | RKNN route |
| --- | --- | --- |
| `yolo26` | detect | `predist` |
| `yolo26-seg` / `yolo26seg` | segment | `seg_predist` |
| `yolov8` | detect | `predfl` |
| `yolov8-seg` / `yolov8seg` | segment | `seg_predfl` |

如果用户要求 YOLO11，说明它可以在 Paddle 侧运行，但除非本地文档或代码已经改变，否则它不是本仓库成熟 RKNN 部署主线。

完成标准：执行导出命令前已经写明选定 route。

## 环境分工

遵守本仓库的职责拆分：

- Windows `pdrk`：Paddle 训练和验证。
- WSL/Ubuntu `pdrk`：当 Windows 遇到 Paddle2ONNX DLL 问题时负责 ONNX 导出。
- WSL/Ubuntu `rknn`：只负责 ONNX 到 RKNN；不要在这里安装 Paddle。
- RK3588 板端：RKNN runtime/lite 推理、测速和可选精度评估。

长任务前先确认对应环境存在，并做必要 import 检查。除非用户要求配置环境，不要主动大修环境。

完成标准：已列出每个阶段使用的环境，并且没有用 `rknn` 环境执行 Paddle 导出。

## 数据集体检与路径处理

拿到数据集位置后，先做轻量体检，再进入训练或评估。用户的目标是跑通链路，不是单独检查数据集；因此数据集体检是本流程的内置门槛，不要拆成独立 skill。

体检内容：

- 找到并读取数据集 YAML；如果用户给的是根目录，优先寻找 `data.yaml`、`dataset.yaml`。
- 确认 `path`、`train`、`val` 能解析到真实目录；训练至少需要 `train` 和 `val`。
- 确认 `names` / `nc` 与标签类别范围一致；发现类别 id 越界时阻断训练。
- 统计 `images/train`、`images/val`、`labels/train`、`labels/val` 数量，并抽样检查图片与标签能按 stem 配对。
- 对 detect 标签，确认每行至少是 `class x y w h`，坐标是数值。
- 对 segment 标签，确认每行至少是 `class x1 y1 x2 y2 x3 y3`，点坐标数量为偶数且不少于 3 个点。
- 抽样检查空标签、缺标签、坏图片；Windows 中文路径下不要直接用 `cv2.imread()` 判坏图，改用 PIL 或 `np.fromfile + cv2.imdecode`。
- 小问题可以报告后继续，结构性问题要先停下。

如果 `path:` 是系统相关路径，给当前运行系统创建临时 YAML，不要直接修改用户原始数据集文件。

板端评估时：

- 只同步 `images/val`、`labels/val` 和板端专用 YAML。
- 不同步 `train`，除非用户明确要求。
- 板端 YAML 的 `path:` 指向板端本地数据集根目录。

完成标准：已经报告 train/val 图片与标签数量、类别范围、抽样标签格式结论；训练或评估命令使用的 YAML 在对应运行环境中路径真实存在。

## 训练

除非项目已有更合适的 CLI，否则使用本地 API：

```python
from ddyolo26 import YOLO

model = YOLO("<model-or-weight>")
model.train(
    data="<data-yaml>",
    epochs=<epochs>,
    imgsz=<imgsz>,
    batch=<batch>,
    device="<device>",
    name="<run-name>",
)
```

训练结束后读取 `results.csv`，报告最后一个 epoch 的指标。分割模型要同时报告 box 和 mask 指标。

完成标准：`runs/<task>/<run-name>/weights/best.pdparams` 存在，并已汇总最终 epoch 指标。

## ONNX 与 RKNN 导出

先导出用于 RKNN 编译的 route FP32 ONNX。任务明确时，优先使用更直接的任务脚本，而不是为了统一而强行走批量编排器。

分割 route ONNX：

```bash
python export/export_seg_onnx_i8.py \
  --weights <best.pdparams> \
  --data <data-yaml> \
  --imgsz <imgsz> \
  --skip-quant
```

默认产物：`<best.pdparams 同目录>/<stem>_paddle_<route>_fp32_<imgsz>.onnx`。

分割 RKNN INT8：

```bash
python export/export_seg_rknn_i8.py \
  --weights <route-fp32.onnx> \
  --data <data-yaml> \
  --imgsz <imgsz> \
  --calib-images <n> \
  --algorithm normal
```

默认产物：`<route-fp32.onnx 同目录>/<stem>_paddle_<route>_int8_<imgsz>.rknn`。

检测 RKNN INT8：

```bash
python export/export_det_rknn_i8.py \
  --weights <route-fp32.onnx-or-pdparams> \
  --data <data-yaml> \
  --imgsz <imgsz> \
  --mode int8 \
  --calib-images <n>
```

使用 YOLOv8 路线时，把文件名和 route 调整为 `predfl` / `seg_predfl`。如果结果后续要汇总成表，文件名尽量包含模型、框架（如已知）、route、精度和图像尺寸。

完成标准：ONNX 和 RKNN 文件存在，已检查大小和时间戳，RKNN build 日志显示 toolkit/runtime 兼容信息。

## 板端验证

只有用户要求时才做板端验证。先从项目文档/脚本发现连接参数，或询问用户。然后：

1. 把 RKNN 模型上传到板端隔离的 artifact 目录。
2. 确认 `librknnrt` / `rknnlite` 和 RKNN runtime 版本。
3. 延迟 benchmark 前用板端项目认可的方法锁 CPU/NPU/DDR 频率。
4. 用推断出的后处理 route 运行 `bench/bench_rknn_perf`，分别测 `core=0` 和 `core=all`。
5. 如果用户要求精度评估，只同步验证集数据，然后运行 `python -m tools.eval --backend rknn`。
6. 把 JSON 结果拉回本地 artifact 目录。
7. 无论评估是否失败，都恢复板端 governor。

完成标准：本地存在延迟 JSON；如用户要求精度评估，也存在 eval JSON。最终回复报告 runtime 版本、延迟数据、精度数据和 governor 恢复状态。

## 汇报

汇报路径和指标，不堆满所有命令。包含：

- 训练权重路径。
- ONNX 和 RKNN 产物路径。
- 最终训练指标。
- 如已运行，RKNN 板端精度和延迟。
- 环境偏差，尤其是 Windows ONNX 导出失败后切到 WSL 的情况。
- Git 状态影响：`runs/`、ONNX 和 RKNN 产物通常被忽略，不应提交，除非用户明确要求。

完成标准：用户能清楚看到什么已经成功、产物在哪里、还有什么未提交。
