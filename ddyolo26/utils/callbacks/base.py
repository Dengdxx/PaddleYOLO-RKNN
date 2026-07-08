# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 训练回调机制基类：30+ 个训练生命周期钩子点定义。
@details
定义了 `default_callbacks` 字典，包含训练各阶段的钩子：
on_pretrain_routine_start/end、on_train_epoch_start/end、
on_train_batch_start/end、on_val_start/end 等。
第三方集成（Wandb/TensorBoard/Comet）通过 `add_integration_callbacks()` 注入。
"""

from __future__ import annotations

"""PaddleYOLO-RKNN train、val、predict 与 export 流程的基础 callbacks。"""
import paddle
from collections import defaultdict
from copy import deepcopy


def on_pretrain_routine_start(trainer):
    """pretraining routine 开始前调用。"""
    pass


def on_pretrain_routine_end(trainer):
    """pretraining routine 结束后调用。"""
    pass


def on_train_start(trainer):
    """training 开始时调用。"""
    pass


def on_train_epoch_start(trainer):
    """每个 training epoch 开始时调用。"""
    pass


def on_train_batch_start(trainer):
    """每个 training batch 开始时调用。"""
    pass


def optimizer_step(trainer):
    """optimizer 执行 step 时调用。"""
    pass


def on_before_zero_grad(trainer):
    """gradients 清零前调用。"""
    pass


def on_train_batch_end(trainer):
    """每个 training batch 结束时调用。"""
    pass


def on_train_epoch_end(trainer):
    """每个 training epoch 结束时调用。"""
    pass


def on_fit_epoch_end(trainer):
    """每个 fit epoch（train + val）结束时调用。"""
    pass


def on_model_save(trainer):
    """model 保存时调用。"""
    pass


def on_train_end(trainer):
    """training 结束时调用。"""
    pass


def on_params_update(trainer):
    """model parameters 更新时调用。"""
    pass


def teardown(trainer):
    """training process teardown 阶段调用。"""
    pass


def on_val_start(validator):
    """validation 开始时调用。"""
    pass


def on_val_batch_start(validator):
    """每个 validation batch 开始时调用。"""
    pass


def on_val_batch_end(validator):
    """每个 validation batch 结束时调用。"""
    pass


def on_val_end(validator):
    """validation 结束时调用。"""
    pass


def on_predict_start(predictor):
    """prediction 开始时调用。"""
    pass


def on_predict_batch_start(predictor):
    """每个 prediction batch 开始时调用。"""
    pass


def on_predict_batch_end(predictor):
    """每个 prediction batch 结束时调用。"""
    pass


def on_predict_postprocess_end(predictor):
    """prediction post-processing 结束后调用。"""
    pass


def on_predict_end(predictor):
    """prediction 结束时调用。"""
    pass


def on_export_start(exporter):
    """model export 开始时调用。"""
    pass


def on_export_end(exporter):
    """model export 结束时调用。"""
    pass


default_callbacks = {
    "on_pretrain_routine_start": [on_pretrain_routine_start],
    "on_pretrain_routine_end": [on_pretrain_routine_end],
    "on_train_start": [on_train_start],
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_batch_start": [on_train_batch_start],
    "optimizer_step": [optimizer_step],
    "on_before_zero_grad": [on_before_zero_grad],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_fit_epoch_end": [on_fit_epoch_end],
    "on_model_save": [on_model_save],
    "on_train_end": [on_train_end],
    "on_params_update": [on_params_update],
    "teardown": [teardown],
    "on_val_start": [on_val_start],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_end": [on_val_end],
    "on_predict_start": [on_predict_start],
    "on_predict_batch_start": [on_predict_batch_start],
    "on_predict_postprocess_end": [on_predict_postprocess_end],
    "on_predict_batch_end": [on_predict_batch_end],
    "on_predict_end": [on_predict_end],
    "on_export_start": [on_export_start],
    "on_export_end": [on_export_end],
}


def get_default_callbacks():
    """获取 PaddleYOLO-RKNN train、val、predict 与 export 流程的 default callbacks。

    返回:
        (dict): 各类 training events 的 default callbacks 字典。每个 key 表示 training process 中的一个事件，
            对应 value 是该事件发生时执行的 callback functions 列表。

    示例:
        >>> callbacks = get_default_callbacks()
        >>> print(list(callbacks.keys()))  # 显示所有可用 callback events
        ['on_pretrain_routine_start', 'on_pretrain_routine_end', ...]
    """
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance):
    """向 instance 的 callbacks 字典添加 integration callbacks。

    面向 detection-only training 的最小实现，不加载 external integrations。
    """
    pass
