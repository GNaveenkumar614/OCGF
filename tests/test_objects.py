"""MTrans objects in the student: reading object files, object channels O_B, boxes as occluders and as occupancy
pseudo-labels, the model's object input, and the detector's frame choice on KITTI-360."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ocgf.datasets.bev_inputs import BEVInputConfig, bev_inputs
from ocgf.datasets.objects import (CHANNELS, N_CHANNELS, OBJECT_CLASSES, FrameObjects, object_channels,
                                   read_object_file)
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.geometry.projection import project
from ocgf.training.targets import TargetConfig, build_targets

from geometry_fixtures import camera
from road_fixtures import ROAD_Z, street_scan

GRID = BEVGrid()
CAM = camera()
CAR = [20.0, 0.0, ROAD_Z + 0.75, 4.0, 1.8, 1.5, 0.0]          # centre, l, w, h, yaw: 1.5 m tall, on the road


def cell(x, y):
    ix, iy = GRID.index(x, y)
    return int(ix), int(iy)


def car_objects(conf=0.6, score=0.9, extra_2d=None):
    boxes, cls, s2d, cf, b2d = [CAR], [0], [score], [conf], [[600, 150, 760, 230]]
    if extra_2d is not None:                                   # a detection without a 3D box
        boxes.append([np.nan] * 7), cls.append(2), s2d.append(0.7), cf.append(np.nan), b2d.append(extra_2d)
    return FrameObjects(np.array(boxes, float), np.array(cls), np.array(s2d), np.array(cf, float), np.array(b2d, float))


def test_object_files_are_read_per_frame_with_boxes_and_detections_without_one(tmp_path):
    rows = [{"fid": "kitti360/s/0000000010", "cls": "car", "score_2d": 0.9, "box_2d": [1, 2, 3, 4], "n_points": 50,
             "box": CAR, "conf_iou_raw": 0.6, "status": "ok"},
            {"fid": "kitti360/s/0000000010", "cls": "pedestrian", "score_2d": 0.4, "box_2d": [5, 6, 7, 8],
             "n_points": 0, "box": None, "conf_iou_raw": None, "status": "too_few_points"},
            {"fid": "kitti360/s/0000000011", "cls": "car", "score_2d": 0.05, "box_2d": [1, 2, 3, 4], "n_points": 9,
             "box": CAR, "conf_iou_raw": 0.5, "status": "ok"}]
    path = tmp_path / "s.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    objs = read_object_file(path, min_score=0.1)
    assert set(objs) == {10}                                   # frame 11's only box is below the floor
    o = objs[10]
    assert len(o) == 2 and o.has_box.tolist() == [True, False] and o.cls.tolist() == [0, 2]
    assert o.quality == pytest.approx([0.54, 0.0])


def test_object_channels_mark_the_box_its_height_its_shadow_and_the_distance_to_it():
    z_g = np.full(GRID.shape, ROAD_Z, np.float32)
    ob = object_channels(car_objects(), z_g, CAM, GRID)
    assert ob.shape == (N_CHANNELS,) + GRID.shape and len(CHANNELS) == N_CHANNELS
    inside, behind, beside = cell(20, 0), cell(30, 0), cell(20, 6)
    assert ob[0][inside] == pytest.approx(0.54) and ob[2][inside] == 0 and ob[4][inside] == pytest.approx(0.54)
    assert ob[5][inside] == pytest.approx(1.5, abs=1e-3)       # top height above the ground
    assert ob[6][behind] == 1 and ob[6][beside] == 0 and ob[6][inside] == 0
    assert ob[7][inside] < 0 < ob[7][beside] and ob[7][cell(45, 15)] == 1.0
    empty = object_channels(FrameObjects.empty(), z_g, CAM, GRID)
    assert not empty[:7].any() and (empty[7] == 1).all()


def test_a_detection_without_a_3d_box_marks_the_cells_seen_through_its_2d_box():
    z_g = np.full(GRID.shape, ROAD_Z, np.float32)
    (u, v), = project(np.array([[15.0, 3.0, ROAD_Z]]), CAM)[0]
    ob = object_channels(car_objects(extra_2d=[u - 20, v - 40, u + 20, v + 5]), z_g, CAM, GRID)
    assert ob[8][cell(15, 3)] == pytest.approx(0.7) and ob[8][cell(15, -6)] == 0


def test_boxes_occlude_in_the_viewshed_only_when_switched_on():
    points = street_scan()                                     # the scan itself shows no car
    objs = car_objects()
    plain = bev_inputs(points, CAM, GRID, BEVInputConfig(), objs)
    boxed = bev_inputs(points, CAM, GRID, BEVInputConfig(box_occluders=True), objs)
    behind = cell(30, 0)
    assert plain["M_C"][behind] and not boxed["M_C"][behind] and boxed["shadow_cam"][behind]
    weak = bev_inputs(points, CAM, GRID, BEVInputConfig(box_occluders=True), car_objects(conf=0.1))
    assert weak["M_C"][behind]                                 # quality 0.09 is below the occluder threshold


def test_strict_boxes_become_occupancy_labels_and_no_box_holds_a_free_road_label():
    points = street_scan()
    inputs = bev_inputs(points, CAM, GRID)
    label = np.full((CAM.height, CAM.width), 0.95, np.float32)
    scores = np.full(len(points), 0.95, np.float32)
    on = TargetConfig(box_targets=True)
    t_strict = build_targets(inputs, label, scores, points, CAM, GRID, on, objects=car_objects())
    t_weak = build_targets(inputs, label, scores, points, CAM, GRID, on, objects=car_objects(conf=0.2))
    t_off = build_targets(inputs, label, scores, points, CAM, GRID, TargetConfig(), objects=car_objects())
    inside = cell(20, 0)
    assert t_strict["occ_t"][inside] == 1                      # quality 0.54 >= 0.3
    assert np.isnan(t_weak["occ_t"][inside])                   # quality 0.18: neither occupied nor a free negative
    assert t_off["occ_t"][inside] == 0                         # boxes ignored: the road label makes it a negative


def test_the_model_takes_object_channels_before_the_decoder():
    torch = pytest.importorskip("torch")
    from ocgf.models.student import N_OBJECT_CHANNELS, Student, StudentConfig
    from test_student import synthetic_sample
    from ocgf.training.data import collate
    assert N_OBJECT_CHANNELS == N_CHANNELS
    s = synthetic_sample()
    s["O_B"] = torch.zeros(N_CHANNELS, 16, 16)
    batch = collate([s, s])
    model = Student(StudentConfig(mode="lidar", channels=8, backbone="stub", grid_shape=(16, 16), objects=True)).eval()
    out = model(batch)
    assert out["road"].shape == (2, 16, 16)
    batch["O_B"][:, 0] = 1.0                                   # the object input changes the output
    assert not torch.allclose(model(batch)["occ"], out["occ"])


def test_the_detector_runs_on_every_student_frame():
    from ocgf.datasets.frame_index import sample_every
    from ocgf.teachers.detector2d.run import kitti360_frames
    odo = np.cumsum(np.random.default_rng(0).uniform(0.3, 1.5, 400))   # every frame stored, varying speed
    usable = np.ones(400, bool)
    index = {"frame": np.arange(400), "usable": usable, "kept": usable, "odometer_m": odo,
             "sample_5m": sample_every(odo, usable, 5.0)}
    frames = kitti360_frames(SimpleNamespace(index=index), every_m=2.5)
    assert np.isin(index["frame"][index["sample_5m"]], frames).all()
    assert not np.isin(index["frame"][index["sample_5m"]], index["frame"][sample_every(odo, usable, 2.5)]).all()
