# RKNN 分割部署

> 默认命令在 `PaddleYOLO-RKNN/` 目录内执行。

## 范围

本页说明分割模型的 RKNN 部署：
- `seg_pre_dist` 导出与使用（YOLO26-Seg）
- `seg_pre_dfl` 导出与使用（YOLOv8-Seg）
- 分割后处理契约

## 输出路线

分割导出包括：
- `seg_pre_dist`：与检测 `pre_dist` 对称
- `seg_pre_dfl`：与检测 `pre_dfl` 对称，适用于 YOLOv8-Seg DFL head
- 两条路线都固定 5 个输出张量，并提供 score-sum 快速过滤
- CPU 端完成 `dist2bbox + coeff 聚合 + mask 重建`

| 方案 | 代号 | 输出数 | 后处理复杂度 | 说明 |
|------|------|--------|-------------|------|
| `seg_pre_dist` | `predist` | 5 | 简单 | `raw ltrb + cls logits + mask_coeff + proto + score_sum` |
| `seg_pre_dfl` | `seg_predfl` | 5 | 中等 | `raw DFL transposed + cls logits + mask_coeff + proto + score_sum` |

## 导出

分割 RKNN 只有本页专用入口受支持。公共 `YOLO.export(format="rknn")`
不直接编译分割模型，`export_one2many_onnx.py` 也只生成 ONNX 中间产物；
最终 RKNN 必须先规范化为统一五输出。

```bash
cd PaddleYOLO-RKNN
conda run --no-capture-output -n rknn python3 export/export_seg_rknn_i8.py \
    --weights runs/segment/exp/weights/best_paddle_seg_predist_fp32_640.onnx \
    --data coco.yaml \
    --imgsz 640 \
    --algorithm normal \
    --calib-images 50

# YOLOv8-Seg / seg_pre_dfl
conda run --no-capture-output -n rknn python3 export/export_seg_rknn_i8.py \
    --weights runs/segment/exp/weights/best_paddle_seg_predfl_fp32_640.onnx \
    --data coco.yaml \
    --imgsz 640 \
    --algorithm normal \
    --calib-images 50
```

### 参数

| 参数 | 说明 |
|------|------|
| `--weights` | route FP32 `.onnx`；普通 `.pt` 不支持 |
| `--route` | 要求的 `seg_predist / seg_predfl`；与模型实际 route 不一致时立即失败 |
| `--data` | 校准 YAML |
| `--output` | 输出 `.rknn` 路径 |
| `--target` | `rk3588` / `rk3588s` / `rk3576` / `rk3562` |
| `--imgsz` | 输入尺寸 |
| `--calib-images` | 校准图数 |
| `--algorithm` | `normal / mmse / kl_divergence` |
| `--optimization-level` | RKNN 编译优化级别 |

> 五输出 ONNX 会直接编译；导出过程产生的四输出 ONNX 会先自动规范化为五输出。
> ORT QDQ INT8 ONNX 不可作为 RKNN 输入；RKNN Toolkit 只消费量化前的
> route FP32 ONNX。Toolkit 中间文件在隔离临时目录生成，不会污染仓库。

## 示例产物

| 文件 | 说明 |
|------|------|
| `runs/segment/exp/weights/*_seg_predist_int8_640.rknn` | `seg_pre_dist` INT8 示例输出 |
| `runs/segment/exp/weights/*_seg_predfl_int8_640.rknn` | `seg_pre_dfl` INT8 示例输出 |

## 输出契约

`seg_pre_dist` 固定 5 个输出：

```text
raw_ltrb   [1, 4,  N]
cls_logits [1, nc, N]
mask_coeff [1, nm, N]
proto      [1, nm, PH, PW]
score_sum  [1, 1,  N]
```

`seg_pre_dfl` 固定 5 个输出：

```text
raw_dfl_transposed [1, N,         4*reg_max]
cls_logits         [1, nc,        N]
mask_coeff         [1, nm,        N]
proto              [1, nm,        PH, PW]
score_sum          [1, 1,         N]
```

## CPU 后处理流程

```text
1. full-IO zero-copy 运行 NPU
2. 同步 score_sum；无 survivor 直接结束
3. 同步 cls，仅对 survivor 做精确 best-class 分类；无 seed 直接结束
4. 同步 ltrb / raw_dfl，anchor decode + class-aware NMS；无框直接结束
5. 按 mask 类别白名单过滤；无目标类别直接结束，否则同步 mask_coeff +
   原生 NC1HWC2 proto
6. 小 ROI 直接读取 INT8 proto；大 ROI 融合恢复 NCHW FP32
7. coeff × ROI proto，并在最终写回融合 sigmoid
8. resize 后单遍量化、threshold → binary mask
```

- 评估工具会按严格五输出形状自动识别 `seg_pre_dist / seg_pre_dfl`，并使用与板端 bench 一致的每 anchor best-class 语义
- 外部消费端按本页的输出契约对接
- 五输出 C++ bench 对 full-IO 或原生 proto 布局不支持时直接失败，
  不回退旧取数路线
- 普通输出恢复同时处理 NCHW、NHWC 和逻辑布局未指定时的宽度 stride；
  输入缓冲必须与模型逻辑输入字节数严格一致

## 相关文档

- [README](../README.md)
- [训练与评估](training.md)
- [RKNN 检测部署](deployment-rknn-det.md)
- [COCO val2017 baseline](coco-baselines.md)
