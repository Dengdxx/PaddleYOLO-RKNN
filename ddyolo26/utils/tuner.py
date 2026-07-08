# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 超参数调优工具：Ray Tune / 进化算法自动搜索最优训练参数。
@details
提供 `Tuner` 类和 `run_ray_tune()` 函数，支持：
- 进化算法（内置）：对预定义搜索空间进行多代突变+选择
- Ray Tune 后端：支持 Optuna/BOHB/HyperOpt 等搜索算法
"""

import paddle

from ddyolo26.cfg import TASK2DATA, TASK2METRIC, get_cfg, get_save_dir
from ddyolo26.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks, colorstr


def run_ray_tune(
    model,
    space: (dict | None) = None,
    grace_period: int = 10,
    gpu_per_trial: (int | None) = None,
    max_samples: int = 10,
    **train_args,
):
    """使用 Ray Tune 运行 hyperparameter tuning。

    参数:
        model (YOLO): 需要运行 tuner 的 model。
        space (dict, optional): hyperparameter search space；未提供时使用默认 space。
        grace_period (int, optional): ASHA scheduler 的 grace period，单位为 epoch。
        gpu_per_trial (int, optional): 每个 trial 分配的 GPU 数量。
        max_samples (int, optional): 最大 trial 数。
        **train_args (Any): 传给 `train()` 方法的额外参数。

    返回:
        (ray.tune.ResultGrid): 包含 hyperparameter search 结果的 ResultGrid。

    示例:
        >>> from ddyolo26 import YOLO
        >>> model = YOLO("yolov8n")  # 加载 YOLOv8n model

        开始在 COCO8 dataset 上为 YOLOv8n training 调优 hyperparameters
        >>> result_grid = model.tune(data="coco8.yaml", use_ray=True)
    """
    LOGGER.info("💡 Ray Tune 用法请参见 PaddleYOLO-RKNN 仓库 README。")
    try:
        checks.check_requirements("ray[tune]")
        import ray
        from ray import tune
        from ray.air import RunConfig
        from ray.tune.schedulers import ASHAScheduler
    except ImportError:
        raise ModuleNotFoundError('需要 Ray Tune 但未找到。安装命令: pip install "ray[tune]"')
    checks.check_version(ray.__version__, ">=2.0.0", "ray")
    default_space = {
        "lr0": tune.uniform(1e-05, 0.1),
        "lrf": tune.uniform(0.01, 1.0),
        "momentum": tune.uniform(0.6, 0.98),
        "weight_decay": tune.uniform(0.0, 0.001),
        "warmup_epochs": tune.uniform(0.0, 5.0),
        "warmup_momentum": tune.uniform(0.0, 0.95),
        "box": tune.uniform(0.02, 0.2),
        "cls": tune.uniform(0.2, 4.0),
        "hsv_h": tune.uniform(0.0, 0.1),
        "hsv_s": tune.uniform(0.0, 0.9),
        "hsv_v": tune.uniform(0.0, 0.9),
        "degrees": tune.uniform(0.0, 45.0),
        "translate": tune.uniform(0.0, 0.9),
        "scale": tune.uniform(0.0, 0.9),
        "shear": tune.uniform(0.0, 10.0),
        "perspective": tune.uniform(0.0, 0.001),
        "flipud": tune.uniform(0.0, 1.0),
        "fliplr": tune.uniform(0.0, 1.0),
        "bgr": tune.uniform(0.0, 1.0),
        "mosaic": tune.uniform(0.0, 1.0),
        "mixup": tune.uniform(0.0, 1.0),
        "cutmix": tune.uniform(0.0, 1.0),
        "copy_paste": tune.uniform(0.0, 1.0),
    }
    task = model.task
    model_in_store = ray.put(model)
    base_name = train_args.get("name", "tune")

    def _tune(config):
        """使用指定 hyperparameters 训练 YOLO model 并返回结果。"""
        model_to_train = ray.get(model_in_store)
        model_to_train.trainer = None
        model_to_train.reset_callbacks()
        config.update(train_args)
        try:
            trial_id = tune.get_trial_id()
            trial_suffix = trial_id.split("_")[-1] if "_" in trial_id else trial_id
            config["name"] = f"{base_name}_{trial_suffix}"
        except Exception:
            config["name"] = base_name
        results = model_to_train.train(**config)
        return results.results_dict

    if not space and not train_args.get("resume"):
        space = default_space
        LOGGER.warning("未提供 search space，使用默认 search space。")
    data = train_args.get("data", TASK2DATA[task])
    space["data"] = data
    if "data" not in train_args:
        LOGGER.warning(f'未提供 data，使用默认 "data={data}"。')
    trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})
    asha_scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=TASK2METRIC[task],
        mode="max",
        max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,
        grace_period=grace_period,
        reduction_factor=3,
    )
    tune_dir = get_save_dir(
        get_cfg(DEFAULT_CFG, {**train_args, **{"exist_ok": train_args.pop("resume", False)}}),
        name=train_args.pop("name", "tune"),
    )
    tune_dir.mkdir(parents=True, exist_ok=True)
    if tune.Tuner.can_restore(tune_dir):
        LOGGER.info(f"{colorstr('Tuner: ')} 正在恢复 tuning run {tune_dir}...")
        tuner = tune.Tuner.restore(str(tune_dir), trainable=trainable_with_resources, resume_errored=True)
    else:
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=space,
            tune_config=tune.TuneConfig(
                scheduler=asha_scheduler,
                num_samples=max_samples,
                trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
                trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
            ),
            run_config=RunConfig(storage_path=tune_dir.parent, name=tune_dir.name),
        )
    tuner.fit()
    results = tuner.get_results()
    ray.shutdown()
    return results
