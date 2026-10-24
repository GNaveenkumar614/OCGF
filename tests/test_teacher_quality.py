"""Teacher quality on KITTI-360 (G4) and the E3 oracle inputs: visible ground-truth objects (cyclists = rider + its
two-wheeler), detector recall, MTrans box quality, object files chosen by source."""
import json

import numpy as np
import pytest

from ocgf.datasets.boxes3d import FrameBoxes
from ocgf.datasets.locate import find_object_file
from ocgf.datasets.objects import FrameObjects
from ocgf.evaluation.teacher_quality import (BoxQuality, DetectorRecall, read_detection_file, visible_objects)
from ocgf.geometry.bev_grid import BEVGrid

from geometry_fixtures import camera

CAM = camera()
CAR = [15.0, 0.0, -0.9, 4.0, 1.8, 1.5, 0.0]
BIKE = [20.0, 3.0, -1.2, 1.8, 0.6, 1.0, 0.0]
RIDER = [20.1, 3.0, -0.6, 0.8, 0.6, 1.7, 0.0]
LONE_BIKE = [25.0, -3.0, -1.2, 1.8, 0.6, 1.0, 0.0]
FAR_CAR = [60.0, 0.0, -0.9, 4.0, 1.8, 1.5, 0.0]
HIDDEN_CAR = [30.0, -6.0, -0.9, 4.0, 1.8, 1.5, 0.0]


def frame_boxes():
    params = np.array([CAR, BIKE, RIDER, LONE_BIKE, FAR_CAR, HIDDEN_CAR])
    classes = np.array([0, 4, 3, 4, 0, 0])                    # car, other, cyclist (rider), other, car, car
    n = len(params)
    return FrameBoxes(params, ["car", "bicycle", "rider", "bicycle", "car", "car"], classes, np.zeros(n, int),
                      np.arange(n), np.zeros(n, bool), np.zeros(n, bool), np.array([-1, -1, 1, -1, -1, -1]))


def points_in(boxes, n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for b in boxes:
        local = rng.uniform(-0.45, 0.45, (n, 3)) * b[3:6]
        out.append(np.column_stack([local + b[:3], np.zeros(n), np.zeros(n)]))
    return np.concatenate(out)


@pytest.fixture(scope="module")
def gts():
    return visible_objects(frame_boxes(), CAM, points_in([CAR, BIKE, RIDER, LONE_BIKE, FAR_CAR]), BEVGrid())


def test_visible_objects_join_riders_to_their_bikes_and_skip_the_rest(gts):
    assert [g.cls for g in gts] == ["car", "cyclist"]          # no lone bike, far car (off the grid), hidden car
    cyc = gts[1]
    assert len(cyc.parts) == 2 and cyc.box[3] == pytest.approx(1.8) and cyc.box[5] > 1.7
    assert cyc.n_points == 80 and gts[0].distance == pytest.approx(15.0)
    l, t, r, b = gts[0].box_2d
    assert 0 <= l < r <= CAM.width and 0 <= t < b <= CAM.height


def test_detector_recall_by_any_class_and_its_class_at_two_thresholds(gts, tmp_path):
    car_2d, cyc_2d = gts[0].box_2d, gts[1].box_2d
    path = tmp_path / "d.txt"
    path.write_text(f"f car 0.9 {' '.join(map(str, car_2d))}\nf pedestrian 0.3 {' '.join(map(str, cyc_2d))}\n")
    dets = read_detection_file(path)["f"]
    rec = DetectorRecall()
    rec.add(gts, dets)
    r = rec.result()
    assert r["car"]["all"]["recall_class@0.5"] == 1.0
    cyc = r["cyclist"]["all"]
    assert cyc["recall_any@0.1"] == 1.0 and cyc["recall_class@0.1"] == 0.0 and cyc["recall_any@0.5"] == 0.0
    rec.add(gts, None)                                          # a frame with no detections: all missed
    assert rec.result()["car"]["all"]["recall_class@0.1"] == 0.5


def test_mtrans_box_quality_against_the_ground_truth_boxes(gts):
    shifted = np.array(CAR) + [0.5, 0, 0, 0, 0, 0, 0]
    objs = FrameObjects(np.array([shifted, [np.nan] * 7]), np.array([0, 3]), np.array([0.9, 0.6]),
                        np.array([0.7, np.nan]), np.array([gts[0].box_2d, gts[1].box_2d]))
    q = BoxQuality()
    q.add(gts, objs)
    r = q.result()
    car, cyc = r["car"]["all"], r["cyclist"]["all"]
    assert car["with_3d_box"] == 1 and car["mean_iou3d"] == pytest.approx(3.5 / 4.5, abs=1e-3)
    assert car["recall3d@0.7"] == 1.0 and car["conf_abs_err"] == pytest.approx(abs(0.7 - 3.5 / 4.5), abs=1e-3)
    assert cyc["matched"] == 1 and cyc["with_3d_box"] == 0 and cyc["recall3d@0.5"] == 0.0


def test_object_files_are_found_by_source(tmp_path):
    seq = "s"
    for folder, manifest in (("a", {"teacher": "mtrans"}), ("b", {"teacher": "mtrans", "box_source": "oracle2d"}),
                             ("c", {"teacher": "oracle_gt3d"})):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / f"{seq}.jsonl").write_text("")
        (tmp_path / folder / f"{seq}.manifest.json").write_text(json.dumps(manifest))
    assert find_object_file(tmp_path, seq) == tmp_path / "a" / f"{seq}.jsonl"
    assert find_object_file(tmp_path, seq, "oracle2d") == tmp_path / "b" / f"{seq}.jsonl"
    assert find_object_file(tmp_path, seq, "oracle3d") == tmp_path / "c" / f"{seq}.jsonl"
    with pytest.raises(ValueError):
        find_object_file(tmp_path, seq, "gt")


def test_the_oracle_script_marks_its_files_as_oracle(tmp_path):
    import sys
    from pathlib import Path
    from ocgf.datasets.prepare_kitti360 import PrepConfig, run_all
    from kitti360_fixtures import SEQ, make_fake_kitti360
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import make_oracle_objects
    paths = make_fake_kitti360(tmp_path, n_frames=40, starter_images=False)
    run_all(PrepConfig(sequence=SEQ, input_root=paths["input_root"], out_dir=paths["input_root"] / "supp" / "out",
                       scratch_dir=tmp_path / "scratch", sources=paths["sources"], role="train", keep_every_m=2.5,
                       min_future_m=20.0, n_overlays=0, dev_frames=0), log=lambda m: None)
    make_oracle_objects.main(["--input-root", str(paths["input_root"]), "--sequence", SEQ, "--out", str(tmp_path / "o")])
    manifest = json.loads((tmp_path / "o" / "oracle3d" / f"{SEQ}.manifest.json").read_text())
    assert manifest["teacher"] == "oracle_gt3d" and "ORACLE" in manifest["note"]
    assert (tmp_path / "o" / "oracle2d" / f"{SEQ}.txt").exists()
    assert find_object_file(tmp_path / "o", SEQ, "oracle3d") is not None and find_object_file(tmp_path / "o", SEQ) is None
