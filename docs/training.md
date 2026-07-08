# 训练与评估

> 默认命令在 `PaddleYOLO-RKNN/` 目录内执行。

## 范围

本页说明训练、验证与推理：
- 检测：YOLO26 / YOLOv8
- 分割：YOLO26-Seg / YOLOv8-Seg
- 本地 Paddle 预训练权重加载

YOLO11 配置可以运行，但本仓库没有提供 YOLO11 的入库预训练权重，也没有把
YOLO11 纳入成熟 RKNN 部署主线。

部署与量化请看：
- [RKNN 检测部署](deployment-rknn-det.md)
- [RKNN 分割部署](deployment-rknn-seg.md)

## 模型变体

短名解析表表示代码支持的命名约定；仓库已随 Git LFS 入库的 Paddle
预训练权重位于 `weights/<model>/`，包括 `yolo26`、`yolo26seg`、
`yolov8`、`yolov8seg` 四组。其它短名需要先把对应 Paddle 权重放到
`weights/<model>/`，否则解析会失败。

### YOLO26

| 变体 | 短名 | 深度 | 宽度 | 本地路径 |
|------|------|------|------|----------|
| Nano | `yolo26n` | 0.50 | 0.25 | `weights/yolo26/yolo26n.pdparams` |
| Small | `yolo26s` | 0.50 | 0.50 | `weights/yolo26/yolo26s.pdparams` |
| Medium | `yolo26m` | 0.50 | 1.00 | `weights/yolo26/yolo26m.pdparams` |
| Large | `yolo26l` | 1.00 | 1.00 | `weights/yolo26/yolo26l.pdparams` |
| Extra | `yolo26x` | 1.00 | 1.50 | `weights/yolo26/yolo26x.pdparams` |

### YOLO26-Seg

| 变体 | 短名 | 本地路径 |
|------|------|----------|
| Nano | `yolo26n-seg` | `weights/yolo26seg/yolo26n-seg.pdparams` |
| Small | `yolo26s-seg` | `weights/yolo26seg/yolo26s-seg.pdparams` |
| Medium | `yolo26m-seg` | `weights/yolo26seg/yolo26m-seg.pdparams` |
| Large | `yolo26l-seg` | `weights/yolo26seg/yolo26l-seg.pdparams` |
| Extra | `yolo26x-seg` | `weights/yolo26seg/yolo26x-seg.pdparams` |

### YOLOv8

| 变体 | 短名 | 本地路径 |
|------|------|----------|
| Nano | `yolov8n` | `weights/yolov8/yolov8n.pdparams` |
| Small | `yolov8s` | `weights/yolov8/yolov8s.pdparams` |
| Medium | `yolov8m` | `weights/yolov8/yolov8m.pdparams` |
| Large | `yolov8l` | `weights/yolov8/yolov8l.pdparams` |
| Extra | `yolov8x` | `weights/yolov8/yolov8x.pdparams` |

### YOLOv8-Seg

| 变体 | 短名 | 本地路径 |
|------|------|----------|
| Nano | `yolov8n-seg` | `weights/yolov8seg/yolov8n-seg.pdparams` |
| Small | `yolov8s-seg` | `weights/yolov8seg/yolov8s-seg.pdparams` |
| Medium | `yolov8m-seg` | `weights/yolov8seg/yolov8m-seg.pdparams` |
| Large | `yolov8l-seg` | `weights/yolov8seg/yolov8l-seg.pdparams` |
| Extra | `yolov8x-seg` | `weights/yolov8seg/yolov8x-seg.pdparams` |

## 预训练权重

### 短名检查

先用短名解析入口检查权重是否已经可用：

```bash
# 解析单个本地模型
python -m ddyolo26.utils.downloads yolov8n

# 解析多个本地模型
python -m ddyolo26.utils.downloads yolov8n yolov8s yolov8m

# 检查已入库的常用检测 / 分割权重
python -m ddyolo26.utils.downloads yolov8n yolov8s yolov8m yolov8l yolov8x yolov8n-seg
```

短名入口只解析本地 Paddle 权重，不会自动下载或转换 `.pt`。如果权重缺失，
可以用封装函数生成精确的 Git LFS 拉取命令：

```bash
python -c "from ddyolo26.utils.downloads import paddle_weight_lfs_pull_command as cmd; [print(cmd(n)) for n in ('yolov8n', 'yolov8s', 'yolov8n-seg')]"
```

### Git LFS 拉取

仓库把已入库的 Paddle 预训练权重放在 Git LFS 中。clone 后如果
`weights/*/*.pdparams` 仍是 LFS pointer 文件，可以批量拉取：

```bash
git lfs install
git lfs pull --include="weights/*/*.pdparams"
```

也可以只拉取单个权重：

```bash
git lfs pull --include="weights/yolov8/yolov8n.pdparams"
```

### Python API

```python
from ddyolo26.utils.downloads import paddle_weight_lfs_pull_command, resolve_paddle_weight

path = resolve_paddle_weight("yolov8n")
print(path)
print(paddle_weight_lfs_pull_command("yolov8n"))
```

### 短名加载

```python
from ddyolo26 import YOLO

model = YOLO("yolov8n")
```

### 解析策略

| 优先级 | 来源 | 条件 |
|--------|------|------|
| 1 | 本地 `weights/<model>/` | 文件已存在且有效（>1KB）则直接返回 |
| 2 | Git LFS | 用户显式执行 `git lfs pull` 后，本地解析才会命中 |

## 快速开始

### 使用预训练权重训练

```python
from ddyolo26 import YOLO

model = YOLO("yolov8n")
model.train(
    data="ddyolo26/cfg/datasets/coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="auto",
    device="0",
    workers=4,
    name="exp",
)
```

### 从头训练

```python
from ddyolo26 import YOLO

model = YOLO("ddyolo26/cfg/models/v8/yolov8.yaml")
model.train(
    data="ddyolo26/cfg/datasets/coco8.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    pretrained=False,
    device="0",
    name="scratch_exp",
)
```

### 断点恢复

```python
from ddyolo26 import YOLO

model = YOLO("runs/detect/exp/weights/last.pdparams")
model.train(resume=True)
```

## 验证

```python
from ddyolo26 import YOLO

model = YOLO("runs/detect/exp/weights/best.pdparams")
metrics = model.val(
    data="ddyolo26/cfg/datasets/coco8.yaml",
    split="val",
    imgsz=640,
    batch=16,
    conf=0.001,
    iou=0.6,
)

print(metrics.box.map50, metrics.box.map)
```

## 推理

```python
from ddyolo26 import YOLO

model = YOLO("runs/detect/exp/weights/best.pdparams")
results = model.predict(
    source="artifacts/coco/images/val2017/",
    imgsz=640,
    conf=0.25,
    save=True,
    save_txt=True,
)
```

> YOLO26 的 `end2end=True` 路径为 NMS-free，`iou` 参数在该路径下不生效。

## 主要训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data` | 数据集配置文件路径 | 必选 |
| `epochs` | 训练轮数 | `100` |
| `imgsz` | 输入图像尺寸 | `640` |
| `batch` | Batch 大小 | `16` |
| `device` | `'0'` / `'0,1'` / `'cpu'` | `auto` |
| `optimizer` | `SGD` / `Adam` / `AdamW` / `MuSGD` / `auto` | `auto` |
| `pretrained` | 是否加载预训练权重 | `True` |
| `resume` | 从 `last.pdparams` 恢复训练 | `False` |
| `lr0` | 初始学习率 | `0.01` |
| `lrf` | 最终学习率衰减因子 | `0.01` |
| `mosaic` | Mosaic 增强概率 | `1.0` |
| `close_mosaic` | 最后 N 轮关闭 Mosaic | `10` |
| `box` | Box Loss 权重 | `7.5` |
| `cls` | 分类 Loss 权重 | `0.5` |
| `project` | 结果保存根目录 | `runs/detect` |
| `name` | 训练子目录名 | `train` |

> `reg_max=1` 的 YOLO26 检测模型中，`dfl` 参数不生效；YOLOv8 正常使用 DFL。

## 训练产物

```text
runs/detect/<name>/
├── weights/
│   ├── best.pdparams
│   └── last.pdparams
├── args.yaml
├── results.csv
├── results.png
├── confusion_matrix.png
├── train_batchX.jpg
└── val_batchX_pred.jpg
```

### 权重精度

| 模式 | 设置 | 训练精度 | 保存权重精度 |
|------|------|----------|-------------|
| AMP 混合精度 | `amp=True` | float16 + BN float32 | float16 |
| FP32 全精度 | `amp=False` | float32 | float32 |

```python
model.train(data="ddyolo26/cfg/datasets/coco8.yaml", epochs=100, amp=False)
model.train(data="ddyolo26/cfg/datasets/coco8.yaml", epochs=100)
```

## 最小化验证

```bash
python -c "from ddyolo26 import YOLO; YOLO('ddyolo26/cfg/models/v8/yolov8.yaml').train(data='ddyolo26/cfg/datasets/coco8.yaml', epochs=1, imgsz=320, device='0', project='runs/detect', name='train_yolov8')"
```

## 最佳实践

1. 显存不足时先降 `batch`，再降 `imgsz`。
2. AMP 默认开启，一般先保留。
3. 收敛后期可用 `close_mosaic` 缩短增强尾巴。
4. 多卡训练使用 `paddle.distributed.launch`。

## 相关文档

- [README](../README.md)
- [RKNN 检测部署](deployment-rknn-det.md)
- [RKNN 分割部署](deployment-rknn-seg.md)
