"""Run with: python -m unittest discover -s Tunnelobject/tests -v.

These smoke tests use temporary random weights and synthetic images/videos;
they verify execution, not the paper's detection accuracy.
"""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


class WorkflowSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2
        import numpy as np
        import torch
        from tunnel_rtdetr import RTDETR

        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix="tunnel_smoke_")
        cls.work = Path(cls.temp.name)
        cls.model = RTDETR(str(REPO_ROOT / "Tunnelobject/model/rtdetr.yaml"))
        cls.model.model.names = {0: "spalling", 1: "crack", 2: "leakage"}
        cls.weights = cls.work / "smoke.pt"
        cls.model.save(str(cls.weights))
        cls.source = cls.work / "input.avi"
        writer = cv2.VideoWriter(str(cls.source), cv2.VideoWriter_fourcc(*"MJPG"), 12.5, (128, 128))
        if not writer.isOpened():
            raise RuntimeError("Synthetic video writer could not open.")
        for value in (40, 80):
            writer.write(np.full((128, 128, 3), value, dtype=np.uint8))
        writer.release()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_entry_points_and_import_safety(self):
        for name in ("TrackCount", "train", "val", "predict", "prdict"):
            with self.subTest(entry=name):
                importlib.import_module("Tunnelobject." + name)
                run = subprocess.run(
                    [sys.executable, "-B", str(REPO_ROOT / "Tunnelobject" / (name + ".py")), "--help"],
                    cwd=self.work, capture_output=True, text=True,
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("usage:", run.stdout)

    def test_dataset_paths(self):
        import yaml
        from tunnel_rtdetr.data.utils import check_det_dataset

        data_path = REPO_ROOT / "Tunnelobject/dataset/all.yaml"
        config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        # The repository can be tested without distributing the real dataset.
        if not all((data_path.parent / config[key]).is_dir() for key in ("train", "val")):
            self.skipTest("Local research dataset is not present.")
        resolved = check_det_dataset(str(data_path), autodownload=False)
        self.assertEqual(resolved["nc"], 3)
        self.assertTrue(Path(resolved["train"]).is_dir())
        self.assertTrue(Path(resolved["val"]).is_dir())

    def test_invalid_input_preserves_source(self):
        from Tunnelobject.TrackCount import track_and_count_objects

        before = self.source.read_bytes()
        with self.assertRaises(ValueError):
            track_and_count_objects(self.source, self.source, self.weights)
        with self.assertRaises(FileNotFoundError):
            track_and_count_objects(self.work / "missing.mp4", self.work / "out.mp4", self.weights)
        self.assertEqual(self.source.read_bytes(), before)

    def test_real_video_pipeline(self):
        import cv2
        from Tunnelobject.TrackCount import track_and_count_objects

        for extension in ("mp4", "avi"):
            with self.subTest(extension=extension):
                output = self.work / ("output." + extension)
                result = track_and_count_objects(
                    self.source, output, self.weights, conf=0.99, device="cpu",
                    region_points=[(64, 0), (64, 128)],
                )
                self.assertEqual(result["frames"], 2)
                cap = cv2.VideoCapture(str(output))
                try:
                    self.assertTrue(cap.isOpened())
                    self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
                    self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 12.5, places=1)
                    self.assertTrue(cap.read()[0])
                finally:
                    cap.release()

    def test_crossing_count_and_overlay(self):
        import numpy as np
        from tunnel_rtdetr.solutions import ObjectCounter

        counter = ObjectCounter(
            model=str(self.weights), region=[(64, 0), (64, 128)],
            show=False, device="cpu", counts_direction=False, counts_position="top-left",
        )
        # Supply deterministic tracks to exercise the original counting/overlay logic.
        def tracks(frame):
            counter.profilers[0].dt = 0.0
            counter.boxes = [[60, 40, 80, 60]]
            counter.track_ids = [42]
            counter.clss = [0]
            counter.confs = [0.9]
        counter.extract_tracks = tracks
        counter.track_history[42] = [(50, 50)]
        result = counter(np.zeros((128, 128, 3), dtype=np.uint8))
        self.assertEqual(result.in_count + result.out_count, 1)
        result = counter(np.zeros((128, 128, 3), dtype=np.uint8))
        self.assertEqual(result.in_count + result.out_count, 1)
        self.assertEqual(result.classwise_count["spalling"]["IN"], 1)

    def test_torchscript_export_and_prediction(self):
        import numpy as np
        from tunnel_rtdetr import RTDETR
        model = RTDETR(str(self.weights))
        output = model.export(format="torchscript", imgsz=128, batch=1, device="cpu")
        self.assertTrue(Path(output).is_file())
        exported = RTDETR(output)
        results = exported.predict(np.zeros((128, 128, 3), dtype=np.uint8), imgsz=128, conf=0.99, device="cpu")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].orig_shape, (128, 128))

    def test_single_training_epoch(self):
        import cv2
        import numpy as np
        import yaml
        from tunnel_rtdetr import RTDETR
        from Tunnelobject.train import set_seed

        dataset = self.work / "dataset"
        for split in ("train", "val"):
            (dataset / split / "images").mkdir(parents=True)
            (dataset / split / "labels").mkdir()
            for index in range(2):
                image = np.full((128, 128, 3), 100 + index * 20, dtype=np.uint8)
                cv2.rectangle(image, (40, 40), (88, 88), (255, 255, 255), -1)
                cv2.imwrite(str(dataset / split / "images" / (str(index) + ".jpg")), image)
                (dataset / split / "labels" / (str(index) + ".txt")).write_text(
                    "0 0.5 0.5 0.375 0.375\n", encoding="utf-8"
                )
        config = dataset / "data.yaml"
        config.write_text(yaml.safe_dump({
            "train": str(dataset / "train/images"), "val": str(dataset / "val/images"),
            "names": {0: "spalling", 1: "crack", 2: "leakage"},
        }), encoding="utf-8")
        set_seed(42)
        model = RTDETR(str(REPO_ROOT / "Tunnelobject/model/rtdetr.yaml"))
        model.train(
            data=str(config), epochs=1, imgsz=128, batch=2, workers=0,
            device="cpu", optimizer="MuSGD", pretrained=False, augment=False,
            amp=False, plots=False, project=str(self.work / "runs"), name="train",
            save=True, val=True,
        )
        self.assertTrue((self.work / "runs/train/weights/best.pt").is_file())
        self.assertEqual(model.trainer.epoch, 0)


if __name__ == "__main__":
    os.environ.setdefault("TUNNEL_RTDETR_AUTOINSTALL", "false")
    os.environ.setdefault("TUNNEL_RTDETR_OFFLINE", "true")
    unittest.main()
