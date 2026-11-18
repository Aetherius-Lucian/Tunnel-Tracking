"""RT-DETR-only package and migration regressions."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

class RTDETROnlyTests(unittest.TestCase):
    def test_public_api_and_removed_packages(self):
        import tunnel_rtdetr
        self.assertEqual(tunnel_rtdetr.MODELS, ("RTDETR",))
        for name in ("YOLO", "SAM", "FastSAM", "NAS", "YOLOWorld", "YOLOE"):
            self.assertFalse(hasattr(tunnel_rtdetr, name))
        self.assertFalse((ROOT / "ultralytics").exists())
        self.assertFalse((ROOT / "tunnel_rtdetr/models/yolo").exists())

    def test_reject_non_rtdetr_configuration(self):
        from tunnel_rtdetr.nn.tasks import RTDETRDetectionModel
        with self.assertRaisesRegex(ValueError, "Only RT-DETR"):
            RTDETRDetectionModel({"nc": 3, "backbone": [], "head": [[-1, 1, "Detect", [3]]]}, verbose=False)

    def test_entrypoints_have_no_model_type(self):
        from Tunnelobject import train, val, predict
        for module in (train, val, predict):
            self.assertNotIn("--model-type", __import__("inspect").getsource(module))

    def test_backend_formats(self):
        from tunnel_rtdetr.nn.autobackend import AutoBackend
        for index, suffix in enumerate((".pt", ".torchscript", ".onnx")):
            flags = AutoBackend._model_type("rtdetr" + suffix)
            self.assertEqual(len(flags), 3)
            self.assertTrue(flags[index])
            self.assertEqual(sum(flags), 1)
        with self.assertRaises(ValueError):
            AutoBackend._model_type("other.engine")

    def test_legacy_checkpoint_module_remapping(self):
        import pickle
        import torch
        from tunnel_rtdetr.nn.tasks import RTDETRDetectionModel, torch_safe_load
        # Serialize a minimal real RT-DETR object using its original upstream module path.
        model = RTDETRDetectionModel.__new__(RTDETRDetectionModel)
        torch.nn.Module.__init__(model)
        old_module = RTDETRDetectionModel.__module__
        legacy = old_module.replace("tunnel_rtdetr", "ultralytics", 1)
        aliases = {"ultralytics": importlib.import_module("tunnel_rtdetr"),
                   "ultralytics.nn": importlib.import_module("tunnel_rtdetr.nn"),
                   legacy: importlib.import_module(old_module)}
        prior = {key: sys.modules.get(key) for key in aliases}
        sys.modules.update(aliases)
        try:
            RTDETRDetectionModel.__module__ = legacy
            with tempfile.TemporaryDirectory() as temp:
                weight = Path(temp) / "legacy.pt"
                torch.save({"model": model}, str(weight))
                RTDETRDetectionModel.__module__ = old_module
                checkpoint, _ = torch_safe_load(weight)
                self.assertIsInstance(checkpoint["model"], RTDETRDetectionModel)
        finally:
            RTDETRDetectionModel.__module__ = old_module
            for key, value in prior.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

if __name__ == "__main__":
    unittest.main()
