"""The torchvision detector wrapper, with random weights (no download): builds, runs on CPU, returns Detections."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from ocgf.teachers.detector2d.mapping import RAW_CLASSES  # noqa: E402
from ocgf.teachers.detector2d.model import Detector  # noqa: E402


def test_detector_runs_and_keeps_only_our_coco_classes():
    det = Detector(device="cpu", min_size=64, max_size=256, score_floor=0.0, pretrained=False)
    assert det.categories[1] == "person" and det.categories[3] == "car"
    out = det([np.zeros((60, 200, 3), np.uint8), np.full((50, 120, 3), 128, np.uint8)])
    assert len(out) == 2
    for d in out:
        assert d.boxes.shape == (len(d), 4) and set(d.labels) <= set(RAW_CLASSES)
        if len(d):
            assert (d.boxes[:, 2] <= 200 + 1e-3).all()                       # boxes come back in input pixels
    assert det.settings["weights"] is None and det.weights_sha256() is None
