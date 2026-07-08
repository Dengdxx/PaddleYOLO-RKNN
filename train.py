#!/usr/bin/env python3
# Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>
# SPDX-License-Identifier: AGPL-3.0-only

"""!
@file train.py
@brief PaddleYOLO-RKNN 训练入口示例。
@details
该脚本展示如何通过 `ddyolo26` 的统一 `YOLO` API 启动目标检测训练。
默认使用 YOLOv8，因为它是当前推荐给新用户的训练与 RKNN 部署主线。
"""

from ddyolo26 import YOLO

# 支持的预训练模型：yolov8n、yolov8s、yolov8m、yolov8l、yolov8x
# 短名只解析本地 Paddle 权重；请先将权重放到 `weights/<model>/`


def main():
    """!
    @brief 执行一次完整的 YOLOv8 训练示例流程。
    @details 依次完成预训练权重加载、训练启动、验证评估与推理示例提示。
    @return 无返回值；结果通过终端日志输出。
    """
    # 1. 加载预训练权重（推荐）
    model = YOLO("yolov8n")  # 等价于 YOLO('weights/yolov8/yolov8n.pdparams')
    # （不推荐）若要从头训练，可改成：
    # model = YOLO('ddyolo26/cfg/models/v8/yolov8.yaml')

    # 2. 训练模型
    results = model.train(
        data="ddyolo26/cfg/datasets/coco8.yaml",  # 数据集配置文件
        epochs=100,  # 训练轮数
        imgsz=640,  # 图像尺寸
        batch=16,  # 批次大小
        workers=4,  # 数据加载线程数
        device="0",  # 使用单卡 GPU（或传入 'cpu' 改为 CPU）
        name="yolov8_train",  # 实验名称
        exist_ok=False,  # 覆盖已有实验
        optimizer="auto",  # 让训练器按模型与数据规模选择优化器
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率因子
        momentum=0.937,  # SGD 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,  # 预热轮数
        warmup_momentum=0.8,  # 预热阶段动量
        box=7.5,  # box 损失权重
        cls=0.5,  # 分类损失权重
        dfl=1.5,  # DFL 损失权重
        close_mosaic=10,  # 最后 10 个 epoch 禁用 mosaic
        amp=True,  # 是否使用混合精度训练
        resume=False,  # 是否恢复训练
    )

    # 3. 评估模型
    metrics = model.val(data="ddyolo26/cfg/datasets/coco8.yaml")
    print(f"验证结果: mAP50={metrics.box.map50}, mAP50-95={metrics.box.map}")

    # 4. 使用模型进行推理
    # results = model('path/to/image.jpg')
    # results[0].show()


if __name__ == "__main__":
    main()
