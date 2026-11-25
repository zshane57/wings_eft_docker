#!/usr/bin/env python3
"""Dependency-free tests for YOLO result selection and clean fallback.

The fake ultralytics module verifies the inference contract without installing
PyTorch. End-to-end weight inference is exercised by the Docker image.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import wing_detection as wd


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def item(self):
        return self.value


class FakeCoordinates:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tolist(self):
        return list(self.value)


class FakeBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = [FakeScalar(class_id)]
        self.conf = [FakeScalar(confidence)]
        self.xyxy = [FakeCoordinates(xyxy)]


class FakeYOLO:
    loads = 0
    next_results = []

    def __init__(self, weights_path):
        type(self).loads += 1
        self.weights_path = weights_path
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return type(self).next_results


fake_ultralytics = types.ModuleType("ultralytics")
fake_ultralytics.YOLO = FakeYOLO
sys.modules["ultralytics"] = fake_ultralytics


class WingDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.weights = os.path.join(self.temp.name, "yolo_wings_best.pt")
        with open(self.weights, "wb") as handle:
            handle.write(b"test weights")
        FakeYOLO.loads = 0
        FakeYOLO.next_results = []

    def tearDown(self):
        self.temp.cleanup()

    def detector(self):
        return wd.WingCellDetector(self.weights)

    def test_model_loads_once_and_predict_uses_training_settings(self):
        detector = self.detector()
        FakeYOLO.next_results = [types.SimpleNamespace(boxes=[])]
        image = object()
        detector.detect(image)
        detector.detect(image)
        self.assertEqual(FakeYOLO.loads, 1)
        self.assertEqual(len(detector.model.calls), 2)
        for actual_image, kwargs in detector.model.calls:
            self.assertIs(actual_image, image)
            self.assertEqual(kwargs, {
                "imgsz": 1024, "conf": 0.5, "verbose": False,
            })

    def test_class_mapping_xyxy_and_highest_confidence(self):
        detector = self.detector()
        FakeYOLO.next_results = [types.SimpleNamespace(boxes=[
            FakeBox(0, 0.51, [1, 2, 3, 4]),
            FakeBox(1, 0.72, [10.5, 20.25, 30.0, 40.75]),
            FakeBox(0, 0.93, [100, 200, 300, 400]),
            FakeBox(0, 0.80, [5, 6, 7, 8]),
            FakeBox(99, 0.99, [0, 0, 1, 1]),
        ])]
        boxes = detector.detect(object())
        self.assertEqual(set(boxes), {"pa2r", "dm"})
        self.assertEqual(boxes["pa2r"]["xyxy"], [100, 200, 300, 400])
        self.assertEqual(boxes["pa2r"]["confidence"], 0.93)
        self.assertEqual(boxes["pa2r"]["class_id"], 0)
        self.assertEqual(boxes["pa2r"]["mask_target"], "masks5")
        self.assertEqual(boxes["dm"]["xyxy"], [10.5, 20.25, 30.0, 40.75])
        self.assertEqual(boxes["dm"]["class_id"], 1)
        self.assertEqual(boxes["dm"]["mask_target"], "masks9")
        self.assertEqual(len(boxes), 2)

    def test_empty_or_none_predictions_are_clean_missing_results(self):
        detector = self.detector()
        FakeYOLO.next_results = []
        self.assertEqual(detector.detect(object()), {})
        FakeYOLO.next_results = [types.SimpleNamespace(boxes=None)]
        self.assertEqual(detector.detect(object()), {})

    def test_incomplete_detection_warns_with_filename_and_missing_cell(self):
        boxes = {
            "pa2r": {"class_id": 0, "cell": "pa2r", "mask_target": "masks5",
                     "xyxy": [1, 2, 3, 4], "confidence": 0.8},
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(wd.IncompleteDetectionError) as caught:
                wd.require_complete_boxes("/photos/bad wing.png", boxes)
        self.assertEqual(caught.exception.missing, ["dm"])
        self.assertIn("WARNING:", output.getvalue())
        self.assertIn("bad wing.png", output.getvalue())
        self.assertIn("missing dm", output.getvalue())
        self.assertIn("manual prompt fallback", output.getvalue())

    def test_partial_detection_json_is_saved_for_audit(self):
        boxes = {
            "dm": {"class_id": 1, "cell": "dm", "mask_target": "masks9",
                   "xyxy": [5, 6, 7, 8], "confidence": 0.75},
        }
        path = os.path.join(self.temp.name, "detections.json")
        wd.write_detection_json(
            path, "/photos/bad.png", "_aligned_bad.png",
            {"portrait_rotation_ccw": 0}, boxes, self.weights)
        with open(path) as handle:
            document = json.load(handle)
        self.assertFalse(document["complete"])
        self.assertEqual(document["image"], "bad.png")
        self.assertEqual(document["aligned_image"], "_aligned_bad.png")
        self.assertEqual(document["predict"], {"imgsz": 1024, "conf": 0.5})
        self.assertEqual(document["boxes"]["dm"]["xyxy"], [5, 6, 7, 8])
        self.assertEqual(document["class_mapping"]["0"]["mask_target"], "masks5")
        self.assertEqual(document["class_mapping"]["1"]["mask_target"], "masks9")

    def test_complete_boxes_pass_guard(self):
        boxes = {"pa2r": {}, "dm": {}}
        self.assertIs(wd.require_complete_boxes("ok.png", boxes), boxes)

    def test_alignment_uses_exif_rgb_and_upright_landscape(self):
        events = []

        class FakeImage:
            def __init__(self, size, exif=1):
                self.size = size
                self.width, self.height = size
                self.exif = exif

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def getexif(self):
                return {274: self.exif}

            def convert(self, mode):
                events.append(("convert", mode))
                return self

            def transpose(self, operation):
                events.append(("transpose", operation))
                return FakeImage((self.height, self.width))

        for exif, size_after_exif, expected_rotation in (
                (6, (1000, 600), 0),       # EXIF alone makes it landscape
                (1, (600, 1000), 90)):     # portrait fallback rotation
            events.clear()
            source = FakeImage((600, 1000), exif)
            after_exif = FakeImage(size_after_exif)
            fake_image = types.SimpleNamespace(
                Transpose=types.SimpleNamespace(ROTATE_90="rotate-90-ccw"),
                open=lambda path: source)
            fake_ops = types.SimpleNamespace(
                exif_transpose=lambda image: after_exif)
            fake_pil = types.ModuleType("PIL")
            fake_pil.Image = fake_image
            fake_pil.ImageOps = fake_ops
            with mock.patch.dict(sys.modules, {"PIL": fake_pil}):
                image, info = wd.load_aligned_image("wing.jpg")
            self.assertEqual(image.size, (1000, 600))
            self.assertEqual(info["aligned_size"], [1000, 600])
            self.assertEqual(info["exif_orientation"], exif)
            self.assertEqual(info["portrait_rotation_ccw"], expected_rotation)
            self.assertEqual(info["color_mode"], "RGB")
            self.assertIn(("convert", "RGB"), events)
            if expected_rotation:
                self.assertIn(("transpose", "rotate-90-ccw"), events)

    def test_bundled_weights_are_present(self):
        path = os.path.join(os.path.dirname(__file__),
                            "models", "yolo_wings_best.pt")
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 1_000_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
