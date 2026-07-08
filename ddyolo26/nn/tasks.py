# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief 模型构建与 YAML 加载：从配置文件组装 YOLO26 网络结构。
@details
核心功能：
- `yaml_model_load()`：读取 YAML 并解析网络拓扑（[from, n, module, args]）
- `parse_model()`：将 YAML 中的层描述转换为 `nn.Sequential`，处理 C2f/C3/DFL 等
- `DetectionModel`：完整检测模型（含 one2one/one2many 双头结构）
- `guess_model_task()`：根据模型文件名/结构自动推断任务类型
- `load_checkpoint()`：Paddle 权重加载（兼容 `.pdparams` 和旧的 `*_paddle.pt`）

与 YOLO26 架构关联：
- `end2end: True`（yolo26.yaml）触发 Detect 头创建 one2one_cv2/cv3 副本
- `reg_max: 1` 使 DFL 被 Identity() 替换，等效移除 DFL
"""

from __future__ import annotations
import paddle

import contextlib
import re
from copy import deepcopy
from pathlib import Path


from ddyolo26.paddle_utils import *

from ddyolo26.nn.autobackend import check_class_names
from ddyolo26.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    Proto26,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    Segment,
    Segment26,
)
from ddyolo26.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, WINDOWS, YAML, colorstr, emojis
from ddyolo26.utils.checks import check_requirements, check_suffix, check_yaml
from ddyolo26.utils.loss import E2ELoss, v8DetectionLoss, v8SegmentationLoss
from ddyolo26.utils.ops import make_divisible
from ddyolo26.utils.patches import checkpoint_load
from ddyolo26.utils.plotting import feature_visualization
from ddyolo26.utils.runtime import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)


class BaseModel(paddle.nn.Module):
    """所有 PaddleYOLO-RKNN 模型变体的基类。

    该类为 YOLO 模型提供通用能力，包括前向流程处理、模型融合、信息展示和权重加载。

    属性:
        model (paddle.nn.Sequential): 神经网络模型。
        save (list): 需要保存输出的层索引列表。
        stride (paddle.Tensor): 模型 stride 值。

    方法:
        forward: 执行训练或推理前向。
        predict: 对输入张量执行推理。
        fuse: 融合 Conv/BatchNorm 层并重参数化以优化推理。
        info: 打印模型信息。
        load: 将权重加载到模型。
        loss: 计算训练损失。

    示例:
        创建 BaseModel 实例
        >>> model = BaseModel()
        >>> model.info()  # 显示模型信息
    """

    def forward(self, x, *args, **kwargs):
        """执行模型训练或推理前向。

        如果 x 是 dict，则计算并返回训练损失；否则返回推理预测。

        参数:
            x (paddle.Tensor | dict): 推理输入张量，或包含图像张量和标签的训练字典。
            *args (Any): 可变位置参数。
            **kwargs (Any): 任意关键字参数。

        返回:
            (paddle.Tensor): x 为 dict 时返回损失（训练），否则返回网络预测（推理）。
        """
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """执行一次网络前向。

        参数:
            x (paddle.Tensor): 模型输入张量。
            profile (bool): 为 True 时打印每层计算耗时。
            visualize (bool): 为 True 时保存模型特征图。
            augment (bool): 推理时是否使用增强。
            embed (list, optional): 需要返回嵌入的层索引列表。

        返回:
            (paddle.Tensor): 模型最后一层输出。
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """执行一次网络前向。

        参数:
            x (paddle.Tensor): 模型输入张量。
            profile (bool): 为 True 时打印每层计算耗时。
            visualize (bool): 为 True 时保存模型特征图。
            embed (list, optional): 需要返回嵌入的层索引列表。

        返回:
            (paddle.Tensor): 模型最后一层输出。
        """
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j]) for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(paddle.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return paddle.unbind(paddle.cat(embeddings, 1), dim=0)
        return x

    def _setup_export_forward(self):
        """通过预计算层路由信息为 ONNX 导出准备模型。

        paddle.jit.to_static（由 paddle.onnx.export 内部调用）无法处理循环变量上的自定义属性
        (.i, .f) 访问。本方法将路由抽取为普通 Python 列表，并把 forward() 替换为基于索引访问的
        导出友好版本。
        """
        from_list = []
        for m in self.model:
            f = m.f
            # 将列表型 f 中的 -1 解析为实际上一层索引。
            if isinstance(f, list):
                from_list.append([m.i - 1 if j == -1 else j for j in f])
            else:
                from_list.append(f)
        self._export_from = from_list
        self._export_n = len(from_list)

        # 将 Detect head 上的方法替换为导出友好的版本：
        # - _get_decode_boxes：避开 self.__dict__ 赋值和 make_anchors()
        # - forward：避开 .detach() (share_data_)、高级索引 (broadcast_tensors)，
        #   以及 paddle2onnx 无法转换的 modulo (remainder)
        from ddyolo26.nn.modules.head import Detect

        for m in self.model:
            if isinstance(m, Detect):
                m._get_decode_boxes = m._get_decode_boxes_export
                m.forward = m._forward_export
                # 从实例 __dict__ 中移除 warm-up 计算得到的 anchors/strides，
                # 防止 jit.save 尝试序列化未注册张量。
                for attr in ("anchors", "strides"):
                    m.__dict__.pop(attr, None)
                    if hasattr(m, "_buffers") and attr in m._buffers:
                        del m._buffers[attr]

    def _forward_export(self, x):
        """兼容 paddle.jit.to_static 的导出友好前向。

        使用 _setup_export_forward() 保存的预计算路由，避免在循环变量上访问自定义 .i/.f 属性。
        """
        y = []
        for i in range(self._export_n):
            f = self._export_from[i]
            if isinstance(f, int):
                if f != -1:
                    x = y[f]
            else:
                x = [y[j] for j in f]
            x = self.model[i](x)
            y.append(x)
        return x

    def _predict_augment(self, x):
        """对输入图像 x 执行增强，并返回增强推理结果。"""
        LOGGER.warning(f"{self.__class__.__name__} 不支持 'augment=True' 推理，已回退到单尺度推理。")
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """在给定输入上统计模型单层的计算耗时和 FLOPs。

        参数:
            m (paddle.nn.Layer): 待统计的层。
            x (paddle.Tensor): 该层输入数据。
            dt (list): 用于保存该层计算耗时的列表。
        """
        try:
            import thop
        except ImportError:
            thop = None
        c = m == self.model[-1] and isinstance(x, list)
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1000000000.0 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'耗时(ms)':>10s} {'GFLOPs':>10s} {'参数量':>10s}  模块")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  总计")

    def fuse(self, verbose=True):
        """融合 Conv/ConvTranspose 与 BatchNorm 层，并重参数化 RepConv/RepVGGDW 以提升效率。

        参数:
            verbose (bool): 融合后是否打印模型信息。

        返回:
            (paddle.nn.Layer): 融合后的模型。
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, Detect) and getattr(m, "end2end", False):
                    m.fuse()
            self.info(verbose=verbose)
        return self

    def is_fused(self, thresh=10):
        """检查模型中的归一化层数量是否少于指定阈值。

        参数:
            thresh (int, optional): 归一化层数量阈值。

        返回:
            (bool): 模型归一化层数量少于阈值时为 True，否则为 False。
        """
        bn = tuple(v for k, v in paddle.nn.__dict__.items() if "Norm" in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def info(self, detailed=False, verbose=True, imgsz=640):
        """打印模型信息。

        参数:
            detailed (bool): 为 True 时打印模型详细信息。
            verbose (bool): 为 True 时打印模型信息。
            imgsz (int): 计算模型信息时使用的图像尺寸。
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, transform, device=None, dtype=None, blocking=None, include_sublayers=True):
        """对模型中的所有张量应用函数，包括 Detect head 的 stride、anchors 等属性。

        参数:
            transform (function): 要应用到模型上的函数。

        返回:
            (BaseModel): 更新后的 BaseModel 对象。
        """
        super()._apply(transform, device, dtype, blocking, include_sublayers)
        m = self.model[-1]

        # 根据参数决定要应用的函数。
        if device is not None:
            fn = lambda t: paddle.to_tensor(t, place=device) if isinstance(t, paddle.Tensor) else t
        elif dtype is not None:
            fn = lambda t: paddle.to_tensor(t, dtype=dtype) if isinstance(t, paddle.Tensor) else t
        else:
            fn = lambda t: t

        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """将权重加载到模型中。

        参数:
            weights (dict | paddle.nn.Layer): 待加载的预训练权重。
            verbose (bool, optional): 是否记录权重迁移进度。
        """
        model = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        updated_csd = intersect_dicts(csd, self.state_dict())
        import warnings as _w

        with _w.catch_warnings():
            _w.filterwarnings("ignore", message="Skip loading for", category=UserWarning)
            self.load_state_dict(updated_csd, strict=False)
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict and first_conv in csd:
            dst = state_dict[first_conv]
            src = csd[first_conv]
            if isinstance(src, paddle.Tensor) and isinstance(dst, paddle.Tensor):
                dst_shape = [int(x) for x in dst.shape]
                src_shape = [int(x) for x in src.shape]
                if len(dst_shape) == 4 and len(src_shape) == 4:
                    c1, c2, h, w = dst_shape
                    cc1, cc2, ch, cw = src_shape
                    if ch == h and cw == w:
                        c1, c2 = min(c1, cc1), min(c2, cc2)
                        dst_np = dst.numpy()
                        src_np = src.numpy()
                        dst_np[:c1, :c2, :, :] = src_np[:c1, :c2, :, :].astype(dst_np.dtype, copy=False)
                        dst.set_value(dst_np)
                        len_updated_csd += 1
        if verbose:
            LOGGER.info(f"已从预训练权重迁移 {len_updated_csd}/{len(self.model.state_dict())} 项")

    def loss(self, batch, preds=None):
        """计算损失。

        参数:
            batch (dict): 用于计算损失的 batch。
            preds (paddle.Tensor | list[paddle.Tensor], optional): 预测结果。
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """初始化 BaseModel 的损失准则。"""
        raise NotImplementedError("compute_loss() 需要由任务 head 实现")


class DetectionModel(BaseModel):
    """YOLO 检测模型。

    该类实现 YOLO 检测架构，负责目标检测任务中的模型初始化、前向、增强推理和损失计算。

    属性:
        yaml (dict): 模型配置字典。
        model (paddle.nn.Sequential): 神经网络模型。
        save (list): 需要保存输出的层索引列表。
        names (dict): 类别名字典。
        inplace (bool): 是否使用 inplace 操作。
        end2end (bool): 模型是否使用端到端检测。
        stride (paddle.Tensor): 模型 stride 值。

    方法:
        __init__: 初始化 YOLO 检测模型。
        _predict_augment: 执行增强推理。
        _descale_pred: 对增强推理后的预测执行反缩放。
        _clip_augmented: 裁剪 YOLO 增强推理尾部。
        init_criterion: 初始化损失准则。

    示例:
        初始化检测模型
        >>> model = DetectionModel("yolo26n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        """使用给定配置和参数初始化 YOLO 检测模型。

        参数:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数。
            verbose (bool): 是否显示模型信息。
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` 模块已弃用，请改用 paddle.nn.Identity。请删除本地旧权重并重新下载最新模型 checkpoint。"
            )
            self.yaml["backbone"][0][2] = "nn.Identity"
        self.yaml["channels"] = ch
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"使用 nc={nc} 覆盖 model.yaml 中的 nc={self.yaml['nc']}")
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)
        self.args = deepcopy(DEFAULT_CFG)
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            def _forward(x):
                """执行模型前向，并按不同 Detect 子类类型处理输出。"""
                output = self.forward(x)
                if self.end2end:
                    output = output["one2many"]
                return output["feats"]

            self.model.eval()
            m.training = True
            m.stride = paddle.tensor([(s / x.shape[-2]) for x in _forward(paddle.zeros(1, ch, s, s))])
            self.stride = m.stride
            self.model.train()
            m.bias_init()
        else:
            self.stride = paddle.Tensor([32])
        initialize_weights(self)
        if isinstance(m, Detect):
            m.sync_one2one_heads()
        if verbose:
            self.info()
            LOGGER.info("")

    @property
    def end2end(self):
        """返回模型是否使用端到端无 NMS 检测。"""
        return getattr(self.model[-1], "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """覆盖端到端检测模式。"""
        self.set_head_attr(end2end=value)

    def set_head_attr(self, **kwargs):
        """设置模型 head（最后一层）的属性。

        参数:
            **kwargs: 表示待设置属性的任意关键字参数。
        """
        head = self.model[-1]
        for k, v in kwargs.items():
            if not hasattr(head, k):
                LOGGER.warning(f"Head 没有属性 '{k}'。")
                continue
            setattr(head, k, v)

    def _predict_augment(self, x):
        """对输入图像 x 执行增强，并返回增强推理输出和训练输出。

        参数:
            x (paddle.Tensor): 输入图像张量。

        返回:
            (tuple[paddle.Tensor, None]): 增强推理输出，以及训练输出占位 None。
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("模型不支持 'augment=True'，已回退到单尺度预测。")
            return self._predict_once(x)
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(axis=fi) if fi else x, si, gs=int(self.stride._max()))
            yi = super().predict(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return paddle.cat(y, -1), None

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """对增强推理后的预测执行反缩放（逆操作）。

        参数:
            p (paddle.Tensor): 预测张量。
            flips (int | None): 翻转类型（None=无，2=上下，3=左右）。
            scale (float): 缩放因子。
            img_size (tuple): 原始图像尺寸 (height, width)。
            dim (int): 拆分维度。

        返回:
            (paddle.Tensor): 反缩放后的预测。
        """
        p[:, :4] /= scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x
        return paddle.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """裁剪 YOLO 增强推理尾部。

        参数:
            y (list[paddle.Tensor]): 检测张量列表。

        返回:
            (list[paddle.Tensor]): 裁剪后的检测张量。
        """
        nl = self.model[-1].nl
        g = sum(4**x for x in range(nl))
        e = 1
        i = y[0].shape[-1] // g * sum(4**x for x in range(e))
        y[0] = y[0][..., :-i]
        i = y[-1].shape[-1] // g * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][..., i:]
        return y

    def init_criterion(self):
        """初始化 DetectionModel 的损失准则。"""
        return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class SegmentationModel(DetectionModel):
    """YOLO 分割模型。

    继承 DetectionModel，处理实例分割任务，提供像素级目标检测和分割的专用损失计算。

    方法:
        __init__: 初始化 YOLO 分割模型。
        init_criterion: 初始化分割损失函数。
    """

    def __init__(self, cfg="yolo26n-seg.yaml", ch=3, nc=None, verbose=True):
        """初始化 YOLO 分割模型。

        参数:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数。
            verbose (bool): 是否显示模型信息。
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化分割模型的损失函数。"""
        return E2ELoss(self, v8SegmentationLoss) if getattr(self, "end2end", False) else v8SegmentationLoss(self)


class Ensemble(paddle.nn.ModuleList):
    """模型集成容器。

    该类允许组合多个 YOLO 模型，通过模型平均或其它集成技术提升性能。

    方法:
        __init__: 初始化模型集成。
        forward: 从集成中的所有模型生成预测。

    示例:
        创建模型集成
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """初始化模型集成。"""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """运行集成前向，并拼接所有模型的预测。

        参数:
            x (paddle.Tensor): 输入张量。
            augment (bool): 是否对输入做增强。
            profile (bool): 是否统计模型耗时。
            visualize (bool): 是否可视化特征。

        返回:
            (paddle.Tensor): 所有模型拼接后的预测。
            (None): 集成推理始终返回 None 作为第二项。
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = paddle.cat(y, 2)
        return y, None


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """临时添加或修改 Python 模块缓存 (`sys.modules`) 的上下文管理器。

    该函数可在运行时改写模块路径。重构代码时，如果模块已经移动到新位置，但仍需要兼容旧导入路径，
    这个工具会很有用。

    参数:
        modules (dict, optional): 旧模块路径到新模块路径的映射字典。
        attributes (dict, optional): 旧模块属性到新模块属性的映射字典。

    示例:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # 此时会导入 new.module
        >>> from old.module import attribute  # 此时会导入 new.module.attribute

    说明:
        这些修改仅在上下文管理器内部生效，退出后会撤销。直接操作 `sys.modules` 可能产生难以预期的结果，
        尤其是在较大的应用或库中，请谨慎使用。
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))
        for old, new in modules.items():
            sys.modules[old] = import_module(new)
        yield
    finally:
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def paddle_safe_load(weight):
    """加载 Paddle 原生 checkpoint 文件。

    参数:
        weight (str | Path): 模型文件路径。

    返回:
        (dict): 加载得到的模型 checkpoint。
        (str): 加载到的文件名。

    示例:
        >>> from ddyolo26.nn.tasks import paddle_safe_load
        >>> ckpt, file = paddle_safe_load("path/to/best.pdparams")
    """
    from ddyolo26.utils.downloads import attempt_download_asset

    weight_str = str(weight)
    if not weight_str.endswith((".pdparams", "_paddle.pt")):
        raise ValueError(f"不支持的 checkpoint 格式: {weight}。请使用 Paddle checkpoint（*.pdparams 或 *_paddle.pt）。")
    check_suffix(file=weight, suffix=(".pt", ".pdparams"))
    file = attempt_download_asset(weight)
    # Paddle 原生 checkpoint：直接加载，不做 torch 模块重映射。
    ckpt = checkpoint_load(file, map_location="cpu")
    if not isinstance(ckpt, dict):
        LOGGER.warning(
            f"文件 '{weight}' 似乎保存方式或格式不正确。建议使用 model.save('filename.pdparams') 正确保存 YOLO 模型。"
        )
        ckpt = {"model": ckpt.model}
    return ckpt, file


def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """加载单模型权重。

    参数:
        weight (str | Path): 模型权重路径。
        device (str, optional): 模型加载设备。
        inplace (bool): 是否使用 inplace 操作。
        fuse (bool): 是否融合模型。

    返回:
        (paddle.nn.Layer): 加载后的模型。
        (dict): 模型 checkpoint 字典。
    """
    ckpt, weight = paddle_safe_load(weight)
    args = {**DEFAULT_CFG_DICT, **ckpt.get("train_args", {})}
    model_data = ckpt.get("ema") or ckpt["model"]
    if isinstance(model_data, dict):
        yaml_cfg = ckpt.get("yaml") or ckpt.get("train_args", {}).get("model", "yolo26n.yaml")
        # 确保 scale 已设置，避免 parse_model 发出警告。
        weight_scale = guess_model_scale(str(weight))
        if isinstance(yaml_cfg, str):
            yaml_cfg = yaml_model_load(yaml_cfg)
        if weight_scale and yaml_cfg.get("scales") and not yaml_cfg.get("scale"):
            yaml_cfg["scale"] = weight_scale
        # 根据配置中的 head 模块选择模型类（检测/分割）
        task = guess_model_task(yaml_cfg)
        # 安全兜底：yaml 声明 detect，但 state_dict 含有 proto（分割掩码头）相关键，
        # 说明 yaml 配置与权重不匹配，自动修正为 segment。
        if task == "detect" and any("proto" in k for k in model_data):
            filepath_task = guess_model_task(str(weight))
            if filepath_task == "segment":
                LOGGER.warning(
                    f"权重文件 '{weight}' 含 proto 分割键但嵌入 yaml 为 detect，"
                    "自动修正 task=segment。建议用正确 yaml 重新转换权重。"
                )
                yaml_cfg = yaml_model_load("ddyolo26/cfg/models/26/yolo26-seg.yaml")
                if weight_scale and yaml_cfg.get("scales") and not yaml_cfg.get("scale"):
                    yaml_cfg["scale"] = weight_scale
                task = "segment"
        model_cls = SegmentationModel if task == "segment" else DetectionModel
        # 优先使用 ckpt 中显式记录的 nc（PT→Paddle 转换时写入），避免回退到 yaml 默认 80。
        ckpt_nc = ckpt.get("nc")
        if ckpt_nc is not None:
            yaml_cfg = dict(yaml_cfg)
            yaml_cfg["nc"] = ckpt_nc
        model = model_cls(cfg=yaml_cfg, verbose=False)
        # 将加载的 state_dict 值转换为模型参数 dtype（如 fp16 → fp32）。
        model_state = model.state_dict()
        for k, v in model_data.items():
            if k in model_state and isinstance(v, paddle.Tensor) and v.dtype != model_state[k].dtype:
                model_data[k] = v.cast(model_state[k].dtype)
        import warnings as _w

        with _w.catch_warnings():
            _w.filterwarnings("ignore", message="Skip loading for", category=UserWarning)
            model.set_state_dict(model_data)
    else:
        model = model_data.float()
    # 从 YAML 配置/模型结构推断 task；优先级：YAML head > 模型模块 > 权重路径 > 默认 detect
    if isinstance(model_data, dict):
        args["task"] = task
    else:
        # 非 state_dict 模式：从模型结构或权重路径推出 task
        args["task"] = guess_model_task(model) if hasattr(model, "model") else guess_model_task(str(weight))
    model.args = args
    model.pt_path = str(weight)
    model.task = args["task"]
    if not hasattr(model, "stride"):
        model.stride = paddle.tensor([32.0])
    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)
    # 从 checkpoint 恢复类别名称
    if ckpt is not None and isinstance(ckpt, dict) and ckpt.get("names"):
        model.names = ckpt["names"]
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, paddle.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None
    return model, ckpt


def parse_model(d, ch, verbose=True):
    """将 YOLO model.yaml 字典解析为 Paddle 模型。

    参数:
        d (dict): 模型字典。
        ch (int): 输入通道数。
        verbose (bool): 是否打印模型详情。

    返回:
        (paddle.nn.Sequential): Paddle 模型。
        (list): 需要保存输出的层索引排序列表。
    """
    import ast

    legacy = True
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"未传入模型 scale，假定 scale='{scale}'。")
        depth, width, max_channels = scales[scale]
    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'参数量':>10}  {'模块':<45}{'参数':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    base_modules = frozenset(
        {
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            paddle.nn.Conv2DTranspose,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(
        {BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f}
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = getattr(paddle.nn, m[3:]) if "nn." in m else globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is paddle.nn.BatchNorm2D:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset({Detect, Segment, Segment26}):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m is Segment or m is Segment26:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, Segment, Segment26}:
                m.legacy = legacy
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
            args = [*args[1:]]
        else:
            c2 = ch[f]
        m_ = paddle.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.size for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return paddle.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """从 YAML 文件加载 YOLO 模型配置。

    参数:
        path (str | Path): YAML 文件路径。

    返回:
        (dict): 模型字典。
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub("(\\d+)([nslmx])6(.+)?$", "\\1\\2-p6\\3", path.stem)
        LOGGER.warning(f"PaddleYOLO-RKNN P6 模型现在使用 -p6 后缀，已将 {path.stem} 重命名为 {new_stem}。")
        path = path.with_name(new_stem + path.suffix)
    unified_path = re.sub("(\\d+)([nslmx])(.+)?$", "\\1\\3", str(path))
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """从模型路径中提取 scale 尺寸字符 n、s、m、l 或 x。

    参数:
        model_path (str | Path): YOLO 模型 YAML 文件路径。

    返回:
        (str): 模型 scale 的尺寸字符（n、s、m、l 或 x）；未找到时返回空字符串。
    """
    try:
        return re.search("yolo(e-)?[v]?\\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""


def guess_model_task(model):
    """根据模型架构或配置推断模型任务。

    参数:
        model (paddle.nn.Layer | dict | str | Path): 模型、配置字典或模型文件路径。

    返回:
        (str): 模型任务（'detect'、'segment'、'classify'、'pose'、'obb'）。
    """

    def cfg2task(cfg):
        """从 YAML 字典推断任务。"""
        m = cfg["head"][-1][-2].lower()
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if "pose" in m:
            return "pose"
        if "obb" in m:
            return "obb"

    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    if isinstance(model, paddle.nn.Module):
        for x in ("model.args", "model.model.args", "model.model.model.args"):
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in ("model.yaml", "model.model.yaml", "model.model.model.yaml"):
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            if isinstance(m, Detect):
                return "detect"
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"
    LOGGER.warning(
        "无法自动推断模型任务，假定为 'task=detect'。请为模型显式指定 task，例如 'task=detect'、'segment'、'classify'、'pose' 或 'obb'。"
    )
    return "detect"
