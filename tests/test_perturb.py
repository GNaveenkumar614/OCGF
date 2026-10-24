"""E5 test-time perturbations: LiDAR rings and dropout, camera degradations, box jitter and dropout; deterministic, and
applied by the inference dataset before any input is computed."""
from types import SimpleNamespace

import numpy as np
import pytest

from ocgf.datasets.objects import FrameObjects
from ocgf.evaluation.perturb import (parse_spec, perturb_image, perturb_objects, perturb_points, spec_name)

from geometry_fixtures import camera
from road_fixtures import street_scan


def test_specs_parse_and_name_folders():
    assert parse_spec("") == {} and parse_spec(None) == {}
    assert parse_spec("rings=2+box_drop=0.5") == {"rings": 2.0, "box_drop": 0.5}
    assert spec_name("rings=2+box_drop=0.5") == "rings2_box_drop0p5" and spec_name("") == "clean"
    with pytest.raises(ValueError):
        parse_spec("fog=1")


def test_lidar_rings_and_point_dropout():
    pts = street_scan()
    half = perturb_points(pts, {"rings": 2}, frame=3)
    assert set(np.unique(half[:, 4]).astype(int)) == set(range(0, 64, 2))
    quarter = perturb_points(pts, {"rings": 4}, frame=3)
    assert len(quarter) < len(half) < len(pts)
    a = perturb_points(pts, {"points": 0.5}, frame=3)
    assert abs(len(a) / len(pts) - 0.5) < 0.02
    assert np.array_equal(a, perturb_points(pts, {"points": 0.5}, frame=3))          # deterministic
    assert not np.array_equal(a, perturb_points(pts, {"points": 0.5}, frame=4))      # per frame


def test_camera_degradations():
    rng = np.random.default_rng(0)
    img = rng.integers(60, 200, (64, 96, 3), dtype=np.uint8)
    assert perturb_image(img, {}, 0) is img
    assert perturb_image(img, {"blur": 2}, 0).astype(float).std() < img.astype(float).std()
    assert abs(perturb_image(img, {"exposure": 0.5}, 0).mean() - img.mean() / 2) < 1
    assert (perturb_image(img, {"exposure": 1.8}, 0) == 255).mean() > 0.2                 # clipped
    noisy = perturb_image(img, {"noise": 0.05}, 0)
    assert noisy.dtype == np.uint8 and np.array_equal(noisy, perturb_image(img, {"noise": 0.05}, 0))
    patched = perturb_image(img, {"patches": 2}, 0)
    assert (patched == 0).all(axis=2).sum() >= 16 * 12


def objects():
    boxes = np.array([[20.0, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0], [30.0, 3.0, -1.0, 4.0, 1.8, 1.5, 0.3],
                      [np.nan] * 7])
    return FrameObjects(boxes, np.array([0, 0, 2]), np.array([0.9, 0.8, 0.5]), np.array([0.6, 0.5, np.nan]),
                        np.zeros((3, 4)))


def test_box_jitter_and_dropout():
    o = objects()
    j = perturb_objects(o, {"box_jitter": 0.5, "box_yaw": 0.2}, frame=1)
    assert not np.allclose(j.boxes[:2, :2], o.boxes[:2, :2]) and np.allclose(j.boxes[:2, 2:6], o.boxes[:2, 2:6])
    assert np.isnan(j.boxes[2]).all() and not np.allclose(j.boxes[:2, 6], o.boxes[:2, 6])
    assert len(perturb_objects(o, {"box_drop": 1.0}, frame=1)) == 0
    kept = perturb_objects(o, {"box_drop": 0.5}, frame=1)
    assert np.array_equal(kept.cls, perturb_objects(o, {"box_drop": 0.5}, frame=1).cls)   # same boxes for every arm
    assert perturb_objects(None, {"box_drop": 1.0}, 1) is None and perturb_objects(o, {"rings": 2}, 1) is o


def test_the_inference_dataset_recomputes_inputs_from_the_perturbed_scan():
    pytest.importorskip("torch")
    from ocgf.training.data import ScanFrames
    cam = camera()
    seq = SimpleNamespace(points=lambda f: street_scan(), image=lambda f: np.zeros((cam.height, cam.width, 3), np.uint8),
                          camera=cam)
    clean = ScanFrames(seq, [0])[0]
    sparse = ScanFrames(seq, [0], perturb={"rings": 4})[0]
    assert len(sparse["pt_feats"]) < 0.3 * len(clean["pt_feats"])
    assert sparse["Q_geo"][0].sum() < clean["Q_geo"][0].sum()           # fewer points per cell in Q
