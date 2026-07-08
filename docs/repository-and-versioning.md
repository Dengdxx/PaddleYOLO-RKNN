# 仓库使用与版本策略

PaddleYOLO-RKNN 是独立仓库，不作为其它仓库的 Python 包、源码依赖或 git submodule 使用。
用户应单独克隆或复制本仓库，在 `PaddleYOLO-RKNN/` 根目录内运行训练、导出、评测和 bench
命令。

本仓库不提供 `pip install -e .`、wheel、editable install 或外部 `PYTHONPATH`
集成方案。用户进入 `PaddleYOLO-RKNN/` 后运行文档里的命令即可，不需要处理
Python packaging、跨仓库依赖或模块搜索路径。

## 标准使用方式

```bash
cd PaddleYOLO-RKNN

conda create -n pdrk python=3.12 -y
conda activate pdrk
pip install -r requirements-paddle.txt

git lfs install
git lfs pull --include="weights/*/*.pdparams"
python -m ddyolo26.utils.downloads yolov8n

python -c "from ddyolo26 import YOLO, __version__; print(__version__); YOLO('yolov8n')"
```

Paddle 训练和 Paddle → ONNX 导出依赖使用 `requirements-paddle.txt`。Windows
原生 `pdrk` 可用于训练；若 Windows 的 `paddle2onnx` wheel 遇到
`paddle2onnx_cpp2py_export` DLL/ABI 加载问题，用同一份
`requirements-paddle.txt` 在 Ubuntu/WSL 建 `pdrk` 环境产出 ONNX。

Ubuntu/WSL 侧 `requirements-rknn.txt` 只负责 ONNX → RKNN 编译，不安装 Paddle：

```bash
conda create -n rknn python=3.12 -y
conda activate rknn
pip install -r requirements-rknn.txt
```

所有文档命令默认都从仓库根目录执行。脚本如果需要访问模型配置、数据集配置、
预训练权重或导出产物，应使用仓库内相对路径，例如：

- `ddyolo26/cfg/models/v8/yolov8.yaml`
- `ddyolo26/cfg/datasets/coco8.yaml`
- `weights/yolov8/yolov8n.pdparams`
- `artifacts/coco_baselines/`

## 版本号

Python 侧版本号由 `ddyolo26.version.__version__` 提供，并在包根级重新导出：

```python
from ddyolo26 import __version__
```

版本号采用 `MAJOR.MINOR.PATCH`：

- `MAJOR`：训练、导出、评测或权重命名出现不兼容变化。
- `MINOR`：增加兼容能力，例如新模型规模、新导出路线、新评测表字段。
- `PATCH`：bugfix、文档、格式化、CI、非破坏性兼容修正。

`pyproject.toml` 中的项目版本必须与 `ddyolo26.version.__version__` 保持一致；CI/测试
会检查这个约束。

## 复现信息

为了让模型、导出产物和评测结果可复现，建议保留：

- PaddleYOLO-RKNN Git commit。
- `ddyolo26.__version__`。
- `requirements-paddle.txt` 中的 PaddlePaddle / Torch / paddle2onnx 版本，以及 `requirements-rknn.txt` 中的 RKNN Toolkit / Torch 版本。
- 预训练权重文件名与 Git LFS oid。
- 导出产物名、route、imgsz、opset。
- RKNN Toolkit / Runtime 版本和板端锁频口径。

这些信息只用于复现本仓库状态，不表示 PaddleYOLO-RKNN 可以作为其它仓库的运行时依赖。
