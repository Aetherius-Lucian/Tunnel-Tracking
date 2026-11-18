"""Inference backends for RT-DETR PyTorch, TorchScript and ONNX models."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from tunnel_rtdetr.utils import YAML

def check_class_names(names):
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}

def default_class_names(data=None):
    if data:
        names = YAML.load(data).get("names")
        if names:
            return check_class_names(names)
    return {i: f"class{i}" for i in range(999)}

class AutoBackend(nn.Module):
    @torch.no_grad()
    def __init__(self, model="rtdetr-l.pt", device=torch.device("cpu"), dnn=False, data=None,
                 fp16=False, batch=1, fuse=True, verbose=True):
        super().__init__()
        self.device, self.dnn = device, dnn
        self.pt, self.jit, self.onnx = isinstance(model, nn.Module), False, False
        self.dynamic, self.end2end, self.nhwc = False, False, False
        self.stride, self.ch, self.batch = 32, 3, batch
        self.fp16 = bool(fp16 and device.type != "cpu")
        self.metadata, self.task, self.names = {}, "detect", default_class_names(data)
        path = str(model) if not self.pt else ""
        if not self.pt:
            flags = self._model_type(path)
            self.pt, self.jit, self.onnx = flags[:3]
        if self.pt:
            from tunnel_rtdetr.nn.tasks import RTDETRDetectionModel, load_checkpoint
            if not isinstance(model, nn.Module):
                model, _ = load_checkpoint(model, device=device)
            if not isinstance(model, RTDETRDetectionModel):
                raise ValueError("Only RT-DETR models are supported.")
            model = model.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
            self.stride = max(int(model.stride.max()), 32)
            self.ch = model.yaml.get("channels", 3)
            self.names = model.names
            self.model = model.half() if self.fp16 else model.float()
            for param in self.model.parameters():
                param.requires_grad = False
        elif self.jit:
            files = {"config.txt": ""}
            self.model = torch.jit.load(path, map_location=device, _extra_files=files)
            self.model.half() if self.fp16 else self.model.float()
            if files["config.txt"]:
                self.metadata = json.loads(files["config.txt"])
        elif self.onnx:
            if dnn:
                self.net = cv2.dnn.readNetFromONNX(path)
                self.fp16 = False
            else:
                import onnxruntime as ort
                providers = ["CPUExecutionProvider"]
                if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, ("CUDAExecutionProvider", {"device_id": device.index or 0}))
                self.session = ort.InferenceSession(path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [o.name for o in self.session.get_outputs()]
                self.fp16 = "float16" in self.session.get_inputs()[0].type
                self.dynamic = any(isinstance(x, str) for x in self.session.get_inputs()[0].shape)
                self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        for key in ("stride", "batch", "imgsz", "names"):
            if key in self.metadata:
                value = self.metadata[key]
                if isinstance(value, str):
                    with __import__("contextlib").suppress(ValueError, SyntaxError):
                        value = ast.literal_eval(value)
                setattr(self, key, value)
        self.names = check_class_names(self.names)

    def forward(self, im, augment=False, visualize=False, embed=None, **kwargs):
        im = im.half() if self.fp16 else im.float()
        if self.pt:
            return self.model(im, augment=augment, visualize=visualize, embed=embed)
        if self.jit:
            return self.model(im)
        array = im.cpu().numpy()
        if self.dnn:
            self.net.setInput(array)
            output = self.net.forward()
        else:
            output = self.session.run(self.output_names, {self.input_name: array})[0]
        return torch.from_numpy(output).to(im.device)

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.device.type != "cpu":
            self.forward(torch.empty(imgsz, device=self.device, dtype=torch.float16 if self.fp16 else torch.float32))

    @staticmethod
    def _model_type(path="model.pt"):
        suffix = Path(path).suffix.lower()
        if suffix not in {".pt", ".torchscript", ".onnx"}:
            raise ValueError("Supported RT-DETR formats: .pt, .torchscript, .onnx")
        return [suffix == ".pt", suffix == ".torchscript", suffix == ".onnx"]
