"""Rotated box IoU in numpy, against closed-form cases."""
import numpy as np
import pytest

from ocgf.geometry.box_iou import bev_iou, enclosing_box, footprint, iou_3d


def test_identical_disjoint_and_shifted_boxes():
    a = np.array([10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3])
    assert bev_iou(a, a) == pytest.approx(1.0) and iou_3d(a, a) == pytest.approx(1.0)
    assert bev_iou(a, a + [20, 0, 0, 0, 0, 0, 0]) == 0
    b = np.array([0.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0])
    shifted = b + [2, 0, 0, 0, 0, 0, 0]                       # overlap 2 x 2 of two 4 x 2 boxes
    assert bev_iou(b, shifted) == pytest.approx(4 / 12)
    raised = shifted + [0, 0, 0.5, 0, 0, 0, 0]                  # half the height overlaps too
    assert iou_3d(b, raised) == pytest.approx(2.0 / (8 + 8 - 2.0))


def test_rotation_by_90_degrees_of_a_square_changes_nothing_and_of_a_bar_gives_the_cross():
    sq = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0])
    assert bev_iou(sq, sq + [0, 0, 0, 0, 0, 0, np.pi / 2]) == pytest.approx(1.0)
    bar = np.array([0.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0])
    assert bev_iou(bar, bar + [0, 0, 0, 0, 0, 0, np.pi / 2]) == pytest.approx(1 / 7)   # 1 x 1 of 4 + 4 - 1
    assert bev_iou(sq, sq + [0, 0, 0, 0, 0, 0, np.pi / 4]) == pytest.approx(0.7071, abs=1e-3)


def test_the_enclosing_box_of_a_rider_and_its_bicycle():
    bike = np.array([10.0, 0.0, -1.2, 1.8, 0.6, 1.0, 0.0])
    rider = np.array([10.2, 0.1, -0.6, 0.8, 0.6, 1.7, 0.0])
    e = enclosing_box([bike, rider])
    assert e[3] == pytest.approx(1.8) and e[4] == pytest.approx(0.7)
    assert e[2] - e[5] / 2 == pytest.approx(-1.7) and e[2] + e[5] / 2 == pytest.approx(0.25)
    assert footprint(e).shape == (4, 2)
