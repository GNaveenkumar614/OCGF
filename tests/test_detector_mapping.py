import numpy as np
import pytest

from ocgf.teachers.detector2d.mapping import (Detections, format_lines, iou, read_detections, suppress_duplicates,
                                              to_our_classes)


def dets(*rows):
    """rows: (label, score, l, t, r, b)"""
    return Detections(np.array([r[2:] for r in rows], dtype=float).reshape(-1, 4),
                      np.array([r[1] for r in rows], dtype=float), [r[0] for r in rows])


def test_coco_classes_map_to_ours():
    out = to_our_classes(dets(("car", 0.9, 0, 0, 10, 10), ("truck", 0.8, 50, 0, 70, 20), ("bus", 0.7, 100, 0, 140, 30),
                              ("train", 0.6, 200, 0, 300, 40), ("person", 0.5, 400, 0, 410, 30)))
    assert sorted(out.labels) == sorted(["car", "large_vehicle", "large_vehicle", "large_vehicle", "pedestrian"])


def test_a_person_on_a_bicycle_becomes_one_cyclist():
    out = to_our_classes(dets(("person", 0.8, 100, 50, 130, 140), ("bicycle", 0.6, 95, 100, 140, 150)))
    assert out.labels == ["cyclist"]
    np.testing.assert_allclose(out.boxes[0], [95, 50, 140, 150])            # the union of the two boxes
    assert out.scores[0] == pytest.approx(0.6)                              # the lower score


def test_parked_bikes_and_walkers_stay_apart():
    out = to_our_classes(dets(("person", 0.8, 0, 0, 20, 60), ("bicycle", 0.9, 300, 30, 350, 70),
                              ("motorcycle", 0.7, 500, 30, 560, 80)))
    assert out.labels == ["pedestrian"]                                     # unpaired two-wheelers are dropped


def test_pairs_are_one_to_one_best_overlap_first():
    # two riders side by side, each over their own bike
    out = to_our_classes(dets(("person", 0.9, 0, 0, 30, 80), ("person", 0.9, 40, 0, 70, 80),
                              ("bicycle", 0.8, 0, 50, 32, 100), ("bicycle", 0.8, 38, 50, 72, 100)))
    assert out.labels == ["cyclist", "cyclist"]
    assert sorted(out.boxes[:, 0].tolist()) == [0, 38]


def test_car_and_truck_on_one_vehicle_keep_the_better():
    out = to_our_classes(dets(("car", 0.6, 0, 0, 100, 50), ("truck", 0.9, 2, 1, 101, 52)))
    assert out.labels == ["large_vehicle"]


def test_suppression_keeps_separate_objects():
    out = suppress_duplicates(dets(("car", 0.9, 0, 0, 10, 10), ("car", 0.8, 20, 0, 30, 10)))
    assert len(out) == 2


def test_iou_values():
    np.testing.assert_allclose(iou([[0, 0, 10, 10]], [[5, 0, 15, 10], [0, 0, 10, 10], [20, 20, 30, 30]]),
                               [[1 / 3, 1.0, 0.0]])


def test_file_round_trip(tmp_path):
    a = dets(("car", 0.91234, 1.5, 2.25, 30.75, 40.5), ("pedestrian", 0.5, 100, 10, 120, 60))
    lines = format_lines("kitti/training/0000000007", a) + format_lines("kitti/training/0000000009", Detections.empty())
    assert lines[0] == "kitti/training/0000000007 car 0.9123 1.50 2.25 30.75 40.50"
    (tmp_path / "d.txt").write_text("\n".join(lines) + "\n")
    back = read_detections(tmp_path / "d.txt")
    assert list(back) == ["kitti/training/0000000007"]                      # frames without detections have no lines
    np.testing.assert_allclose(back["kitti/training/0000000007"].boxes, a.boxes, atol=0.005)
    assert back["kitti/training/0000000007"].labels == ["car", "pedestrian"]
