# PaddleYOLO-RKNN

PaddleYOLO-RKNN 是面向 RKNN 部署的 PaddlePaddle YOLO 工具仓库，覆盖：

- Paddle 训练、验证与 ONNX 导出
- RKNN INT8 / FP16 导出
- RK3588 板端推理评估与延迟测试

> 默认命令在 `PaddleYOLO-RKNN/` 仓库根目录执行。
>
> 本仓库包含基于 Ultralytics YOLO 修改而来的派生源码，按 AGPL-3.0 分发。
> 具体归属规则见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。

## 克隆仓库

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Dengdxx/PaddleYOLO-RKNN.git
```

需要权重时运行 `git -C PaddleYOLO-RKNN lfs pull`。

## 支持路线

| 场景 | 推荐路线 |
|------|----------|
| 成熟 RKNN 主线 | `yolov8` / `yolov8-seg` |
| Paddle 训练 / 验证 | `yolov8` / `yolov8-seg` |
| Paddle → ONNX | Windows 或 Ubuntu/WSL 的 `pdrk` 环境 |
| 检测 RKNN INT8 | `pre_dfl` |
| 分割 RKNN INT8 | `seg_pre_dfl` |
| 板端运行 | RKNN Runtime `2.3.2` |

YOLO11 配置可运行，但本仓库没有把 YOLO11 纳入成熟 RKNN 部署主线。

## 安装环境

Paddle 训练与 ONNX 导出：

```bash
conda create -n pdrk python=3.12 -y
conda activate pdrk
pip install -r requirements-paddle.txt
```

RKNN 导出在 Ubuntu/WSL 中单独准备：

```bash
conda create -n rknn python=3.12 -y
conda activate rknn
pip install -r requirements-rknn.txt
```

`rknn` 环境只负责 ONNX → RKNN，不需要安装 Paddle。RKNN Toolkit2 不提供 Windows wheel。

## 预训练权重

仓库通过 Git LFS 提供 Paddle 预训练权重，位置为 `weights/<model>/`：

- `weights/yolov8/`
- `weights/yolov8seg/`

这些 `.pdparams` 的源头是 `ultralytics/assets` release 提供的上游预训练权重。

拉取仓库内权重：

```bash
git lfs install
git lfs pull --include="weights/*/*.pdparams"
```

## 最简全流程：训练到 RKNN

下面以 YOLOv8-Seg 为例，把 `data/your-seg.yaml` 换成自己的数据集 YAML 即可。

1. 在 Windows 的 `pdrk` 环境训练，得到 Paddle 权重：

```bash
conda activate pdrk
```

```python
from ddyolo26 import YOLO

model = YOLO("yolov8n-seg")
model.train(
    data="data/your-seg.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="0",
    name="exp",
)
```

训练完成后，权重位于 `runs/segment/exp/weights/best.pdparams`。

2. 在 Ubuntu/WSL 的 `pdrk` 环境导出 RKNN INT8 使用的 route FP32 ONNX：

```bash
conda activate pdrk
python export/export_seg_onnx_i8.py \
  --weights runs/segment/exp/weights/best.pdparams \
  --data data/your-seg.yaml \
  --imgsz 640 \
  --skip-quant
```

产物位于 `runs/segment/exp/weights/best_paddle_seg_predfl_fp32_640.onnx`。

3. 在 Ubuntu/WSL 的 `rknn` 环境量化并编译 RKNN：

```bash
conda activate rknn
python export/export_seg_rknn_i8.py \
  --weights runs/segment/exp/weights/best_paddle_seg_predfl_fp32_640.onnx \
  --data data/your-seg.yaml \
  --imgsz 640 \
  --calib-images 50 \
  --algorithm normal
```

产物位于 `runs/segment/exp/weights/best_paddle_seg_predfl_int8_640.rknn`。

更多数据集格式、训练参数、验证与推理说明见 [训练与评估](docs/training.md)。

## RKNN 部署

- 检测模型：[RKNN 检测部署](docs/deployment-rknn-det.md)
- 分割模型：[RKNN 分割部署](docs/deployment-rknn-seg.md)
- 导出链路与命名规则：[导出管线](docs/export-pipeline.md)
- 板端延迟与评估：[板端延迟与评估表](docs/bench-and-eval.md)
- COCO baseline：[COCO val2017 baseline](docs/coco-baselines.md)

## 其它说明

- 本仓库不作为 Python 包安装；安装依赖后，在仓库根目录直接执行命令。
- 仓库地址：<https://github.com/Dengdxx/PaddleYOLO-RKNN>
- 问题反馈：<https://github.com/Dengdxx/PaddleYOLO-RKNN/issues>
