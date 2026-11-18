"""Regression checks for the RT-DETR dependency cleanup."""
import importlib
import pkgutil
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

class CleanupTests(unittest.TestCase):
    def test_all_retained_modules_import(self):
        import tunnel_rtdetr
        for item in pkgutil.walk_packages(tunnel_rtdetr.__path__, tunnel_rtdetr.__name__ + "."):
            with self.subTest(module=item.name):
                importlib.import_module(item.name)

    def test_unrelated_components_are_removed(self):
        from tunnel_rtdetr.engine import results
        from tunnel_rtdetr.data import dataset
        from tunnel_rtdetr.utils import metrics
        from tunnel_rtdetr.solutions.config import SolutionConfig
        from tunnel_rtdetr.solutions.solutions import SolutionAnnotator
        for name in ("Masks", "Keypoints", "Probs", "OBB"):
            self.assertFalse(hasattr(results, name))
        for name in ("ClassificationDataset", "GroundingDataset", "SemanticDataset", "MultiModalDetectionDataset"):
            self.assertFalse(hasattr(dataset, name))
        for name in ("ClassifyMetrics", "SegmentMetrics", "PoseMetrics", "OBBMetrics"):
            self.assertFalse(hasattr(metrics, name))
        for name in ("up_angle", "kpts", "analytics_type", "max_speed", "records", "crop_dir"):
            self.assertFalse(hasattr(SolutionConfig(), name))
        self.assertFalse(hasattr(SolutionAnnotator, "estimate_pose_angle"))
        for path in ("hub", "utils/export", "utils/tal.py", "utils/nms.py", "nn/text_model.py",
                     "nn/modules/MobileNet.py", "nn/modules/Efficientnet.py", "data/split_dota.py"):
            self.assertFalse((ROOT / "tunnel_rtdetr" / path).exists(), path)

    def test_research_architecture_builds(self):
        import gc
        import torch
        from tunnel_rtdetr.nn.tasks import RTDETRDetectionModel
        torch.set_num_threads(2)
        paths = [ROOT / "Tunnelobject/model/rtdetr.yaml"]
        for path in paths:
            with self.subTest(model=path.name):
                model = RTDETRDetectionModel(str(path), verbose=False).eval()
                if path.name == "rtdetr.yaml":
                    self.assertEqual(sum(p.numel() for p in model.parameters()), 13704434)
                with torch.no_grad():
                    prediction = model(torch.zeros(1, 3, 128, 128))[0]
                self.assertEqual(tuple(prediction.shape[:2]), (1, 300))
                self.assertEqual(prediction.shape[-1], model.yaml["nc"] + 4)
                self.assertTrue(torch.isfinite(prediction).all())
                del model
                gc.collect()

    def test_detection_result_output_and_real_tracking(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from tunnel_rtdetr.engine.results import Results
        from tunnel_rtdetr.trackers.byte_tracker import BYTETracker
        from tunnel_rtdetr.trackers.track import on_predict_postprocess_end
        from tunnel_rtdetr.utils import YAML, IterableSimpleNamespace
        cfg = IterableSimpleNamespace(**YAML.load(ROOT / "Tunnelobject/trackers/bytetrack.yaml"))
        tracker = BYTETracker(cfg)
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            predictor = SimpleNamespace(dataset=SimpleNamespace(mode="image"), trackers=[tracker], save_dir=Path(temp))
            ids = []
            for x in (40, 42):
                result = Results(image, "frame.jpg", {0: "crack"},
                                 boxes=torch.tensor([[x, 40, x + 20, 60, 0.9, 0]]))
                predictor.results = [result]
                on_predict_postprocess_end(predictor, persist=True)
                tracked = predictor.results[0]
                self.assertTrue(tracked.boxes.is_track)
                ids.append(int(tracked.boxes.id.item()))
            self.assertEqual(ids[0], ids[1])
            path = Path(temp) / "labels.txt"
            tracked.save_txt(path, save_conf=True)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.assertEqual(len(path.read_text().split()), 7)
            self.assertEqual(tracked.plot().shape, image.shape)
            self.assertEqual(tracked.summary()[0]["track_id"], ids[0])

if __name__ == "__main__":
    unittest.main()
