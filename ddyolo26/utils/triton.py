# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief Triton Inference Server 客户端适配器。
@details
为 YOLO26 推理提供 Triton HTTP/gRPC 远程推理后端，
实现与本地推理相同的 `__call__` 接口，便于在微服务架构中部署。
"""

import paddle

import ast
from urllib.parse import urlsplit

import numpy as np


class TritonRemoteModel:
    """用于与 remote Triton Inference Server model 交互的 client。

    该类提供便捷接口，用于向 Triton Inference Server 发送 inference requests 并处理 responses。
    支持 HTTP 与 gRPC 两种 communication protocols。

    属性:
        endpoint (str): Triton server 上的 model name。
        url (str): Triton server URL。
        triton_client: Triton client（HTTP 或 gRPC）。
        InferInput: Triton client 使用的 input class。
        InferRequestedOutput: Triton client 使用的 output request class。
        input_formats (list[str]): model inputs 的 data types。
        np_input_formats (list[type]): model inputs 的 numpy data types。
        input_names (list[str]): model inputs 的 names。
        output_names (list[str]): model outputs 的 names。
        metadata: 与 model 关联的 metadata。

    方法:
        __call__: 使用给定 inputs 调用 model 并返回 outputs。

    示例:
        使用 HTTP 初始化 Triton client
        >>> model = TritonRemoteModel(url="localhost:8000", endpoint="yolov8", scheme="http")

        使用 numpy arrays 执行 inference
        >>> outputs = model(np.random.rand(1, 3, 640, 640).astype(np.float32))
    """

    def __init__(self, url: str, endpoint: str = "", scheme: str = ""):
        """初始化用于与 remote Triton Inference Server 交互的 TritonRemoteModel。

        Arguments 可单独提供，也可从以下形式的聚合 'url' argument 中解析：
        <scheme>://<netloc>/<endpoint>/<task_name>

        参数:
            url (str): Triton server URL。
            endpoint (str, optional): Triton server 上的 model name。
            scheme (str, optional): communication scheme（'http' 或 'grpc'）。
        """
        if not endpoint and not scheme:
            splits = urlsplit(url)
            endpoint = splits.path.strip("/").split("/", 1)[0]
            scheme = splits.scheme
            url = splits.netloc
        self.endpoint = endpoint
        self.url = url
        if scheme == "http":
            import tritonclient.http as client

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint)
        else:
            import tritonclient.grpc as client

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint, as_json=True)["config"]
        config["output"] = sorted(config["output"], key=lambda x: x.get("name"))
        type_map = {
            "TYPE_FP32": np.float32,
            "TYPE_FP16": np.float16,
            "TYPE_UINT8": np.uint8,
        }
        self.InferRequestedOutput = client.InferRequestedOutput
        self.InferInput = client.InferInput
        self.input_formats = [x["data_type"] for x in config["input"]]
        self.np_input_formats = [type_map[x] for x in self.input_formats]
        self.input_names = [x["name"] for x in config["input"]]
        self.output_names = [x["name"] for x in config["output"]]
        self.metadata = ast.literal_eval(config.get("parameters", {}).get("metadata", {}).get("string_value", "None"))

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """使用给定 inputs 调用 model，并返回 inference results。

        参数:
            *inputs (np.ndarray): 输入 model 的 input data。每个 array 都应匹配对应 model input 的预期 shape
                与 type。

        返回:
            (list[np.ndarray]): cast 到第一个 input dtype 的 model outputs。列表中每个元素对应一个 model output tensor。

        示例:
            >>> model = TritonRemoteModel(url="localhost:8000", endpoint="yolov8", scheme="http")
            >>> outputs = model(np.random.rand(1, 3, 640, 640).astype(np.float32))
        """
        infer_inputs = []
        input_format = inputs[0].dtype
        for i, x in enumerate(inputs):
            if x.dtype != self.np_input_formats[i]:
                x = x.astype(self.np_input_formats[i])
            infer_input = self.InferInput(
                self.input_names[i],
                [*x.shape],
                self.input_formats[i].replace("TYPE_", ""),
            )
            infer_input.set_data_from_numpy(x)
            infer_inputs.append(infer_input)
        infer_outputs = [self.InferRequestedOutput(output_name) for output_name in self.output_names]
        outputs = self.triton_client.infer(model_name=self.endpoint, inputs=infer_inputs, outputs=infer_outputs)
        return [outputs.as_numpy(output_name).astype(input_format) for output_name in self.output_names]
