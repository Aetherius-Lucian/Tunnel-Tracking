"""Checks for removing demo assets and reducing runtime configuration."""
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

class ConfigCleanupTests(unittest.TestCase):
    def test_configuration_and_resources(self):
        from tunnel_rtdetr.cfg import get_cfg
        cfg = get_cfg()
        self.assertEqual(cfg.task, "detect")
        self.assertEqual(cfg.tracker, "bytetrack.yaml")
        self.assertFalse((ROOT / "tunnel_rtdetr/assets").exists())
        self.assertFalse((ROOT / "tunnel_rtdetr/cfg/datasets").exists())
        for key in ("pose", "dfl", "mask_ratio", "copy_paste", "retina_masks", "int8", "keras", "workspace"):
            self.assertFalse(hasattr(cfg, key), key)
        with self.assertRaises(ValueError):
            get_cfg(overrides={"conf": 1.5})
        with self.assertRaises(SyntaxError):
            get_cfg(overrides={"pose": 12.0})

    def test_source_and_dataset_must_be_explicit(self):
        import torch
        from tunnel_rtdetr import RTDETR
        torch.set_num_threads(2)
        model = RTDETR(str(ROOT / "Tunnelobject/model/rtdetr.yaml"))
        with self.assertRaisesRegex(ValueError, "explicit"):
            model.predict()
        with self.assertRaisesRegex(ValueError, "explicit"):
            model.train()

    def test_saved_prediction_and_legacy_options(self):
        import numpy as np
        import torch
        from tunnel_rtdetr import RTDETR
        torch.set_num_threads(2)
        model = RTDETR(str(ROOT / "Tunnelobject/model/rtdetr.yaml"))
        with tempfile.TemporaryDirectory() as temp:
            weights = Path(temp) / "legacy_options.pt"
            model.save(str(weights))
            checkpoint = torch.load(str(weights), map_location="cpu")
            checkpoint["train_args"] = {"pose": 12., "retina_masks": True, "imgsz": 128}
            torch.save(checkpoint, str(weights))
            loaded = RTDETR(str(weights))
            self.assertNotIn("pose", loaded.model.args)
            self.assertNotIn("retina_masks", loaded.model.args)
            result = loaded.predict(np.zeros((128, 128, 3), dtype=np.uint8), imgsz=128,
                                    conf=0.99, device="cpu", save=True, project=temp, name="predict")
            self.assertEqual(len(result), 1)
            self.assertTrue(list((Path(temp) / "predict").glob("*.jpg")))

if __name__ == "__main__":
    unittest.main()
