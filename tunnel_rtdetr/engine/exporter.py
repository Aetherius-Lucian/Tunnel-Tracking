"""RT-DETR TorchScript and ONNX export."""
import json
from copy import deepcopy
from pathlib import Path
import torch
from tunnel_rtdetr.cfg import get_cfg
from tunnel_rtdetr.utils import DEFAULT_CFG, callbacks
from tunnel_rtdetr.utils.torch_utils import select_device

def export_formats():
    return {"Format": ["PyTorch", "TorchScript", "ONNX"], "Argument": ["-", "torchscript", "onnx"],
            "Suffix": [".pt", ".torchscript", ".onnx"], "CPU": [True]*3, "GPU": [True]*3,
            "Arguments": [[], ["imgsz", "half", "batch"], ["imgsz", "half", "batch", "dynamic", "opset"]]}

class Exporter:
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    def __call__(self, model=None):
        from tunnel_rtdetr.nn.tasks import RTDETRDetectionModel
        if not isinstance(model, RTDETRDetectionModel):
            raise ValueError("Only RT-DETR export is supported.")
        fmt = self.args.format
        if fmt not in {"onnx", "torchscript"}:
            raise ValueError("RT-DETR supports format=onnx or format=torchscript.")
        self.run_callbacks("on_export_start")
        device = select_device(self.args.device)
        model = deepcopy(model).to(device).eval().float()
        size = self.args.imgsz
        size = [size, size] if isinstance(size, int) else size
        if len(size) != 2 or size[0] != size[1]:
            raise ValueError("RT-DETR export requires a square image size.")
        sample = torch.zeros(self.args.batch, model.yaml.get("channels", 3), *size, device=device)
        if self.args.half:
            if device.type == "cpu":
                raise ValueError("Half precision export requires CUDA.")
            model, sample = model.half(), sample.half()
        head = model.model[-1]
        head.export = True
        head.dynamic = bool(self.args.dynamic)
        self.file = Path(getattr(model, "pt_path", model.yaml.get("yaml_file", "rtdetr.yaml")))
        output = self.file.with_suffix(".onnx" if fmt == "onnx" else ".torchscript")
        metadata = {"stride": 32, "task": "detect", "batch": self.args.batch,
                    "imgsz": size, "names": model.names, "end2end": False}
        with torch.no_grad():
            model(sample)
            if fmt == "torchscript":
                traced = torch.jit.trace(model, sample, strict=False)
                torch.jit.save(traced, str(output), _extra_files={"config.txt": json.dumps(metadata)})
            else:
                import onnx
                axes = {"images": {0: "batch"}, "output0": {0: "batch"}} if self.args.dynamic else None
                torch.onnx.export(model, sample, str(output), input_names=["images"], output_names=["output0"],
                                  dynamic_axes=axes, opset_version=self.args.opset or 16)
                exported = onnx.load(str(output))
                for key, value in metadata.items():
                    prop = exported.metadata_props.add()
                    prop.key, prop.value = key, str(value)
                onnx.checker.check_model(exported)
                onnx.save(exported, str(output))
        self.run_callbacks("on_export_end")
        return str(output)

    def run_callbacks(self, event):
        for callback in self.callbacks.get(event, []):
            callback(self)
