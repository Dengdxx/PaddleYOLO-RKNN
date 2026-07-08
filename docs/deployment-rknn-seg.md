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
- 5 个输出张量（转置 DFL + score-sum 快速过滤）
- CPU 端完成 `dist2bbox + coeff 聚合 + mask 重建`

| 方案 | 代号 | 输出数 | 后处理复杂度 | 说明 |
|------|------|--------|-------------|------|
| `seg_pre_dist` | `predist` | 4 | 简单 | `raw ltrb + cls logits + mask_coeff + proto` |
| `seg_pre_dfl` | `seg_predfl` | 5 | 中等 | `raw DFL transposed + cls logits + mask_coeff + proto + score_sum` |

## 导出

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
| `--data` | 校准 YAML |
| `--output` | 输出 `.rknn` 路径 |
| `--target` | `rk3588` / `rk3588s` / `rk3576` / `rk3562` |
| `--imgsz` | 输入尺寸 |
| `--calib-images` | 校准图数 |
| `--algorithm` | `normal / mmse / kl_divergence` |
| `--optimization-level` | RKNN 编译优化级别 |

> 如果输入已经是 `seg_pre_dist` 四输出或 `seg_pre_dfl` 五输出 ONNX，脚本会跳过图手术直接编译。

## 示例产物

| 文件 | 说明 |
|------|------|
| `runs/segment/exp/weights/*_seg_predist_int8_640.rknn` | `seg_pre_dist` INT8 示例输出 |
| `runs/segment/exp/weights/*_seg_predfl_int8_640.rknn` | `seg_pre_dfl` INT8 示例输出 |

## 输出契约

`seg_pre_dist` 期望 4 个输出：

```text
raw_ltrb   [1, 4,  N]
cls_logits [1, nc, N]
mask_coeff [1, nm, N]
proto      [1, nm, PH, PW]
```

`seg_pre_dfl` 期望 5 个输出：

```text
raw_dfl_transposed [1, N,         4*reg_max]
cls_logits         [1, nc,        N]
mask_coeff         [1, nm,        N]
proto              [1, nm,        PH, PW]
score_sum          [1, 1,         N]
```

## CPU 后处理流程

```text
1. dequant ltrb / raw_dfl → anchor decode → xyxy（`seg_pre_dfl` 先做 DFL softmax-expectation）
2. dequant logits → sigmoid → conf / class
3. conf 过滤 + NMS（或 nmsfree_exact）
4. gather mask_coeff
5. dequant proto
6. coeff @ proto → sigmoid
7. resize + threshold → binary mask
```

- `seg_pre_dist / seg_pre_dfl` 不能依赖 `auto` 检测，需显式指定 `output_format`
- 外部消费端按本页的输出契约对接

## 相关文档

- [README](../README.md)
- [训练与评估](training.md)
- [RKNN 检测部署](deployment-rknn-det.md)
- [COCO val2017 baseline](coco-baselines.md)
