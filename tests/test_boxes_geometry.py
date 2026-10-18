import numpy as np

from ocgf.geometry.boxes import EDGES, box_corners, points_in_box


def test_corners_of_an_axis_aligned_box():
    c = box_corners([[10.0, 2.0, 1.0, 4.0, 2.0, 1.5, 0.0]])[0]
    np.testing.assert_allclose(c[0], [12.0, 3.0, 0.25])          # bottom front-left
    np.testing.assert_allclose(c[6], [8.0, 1.0, 1.75])           # top rear-right
    assert len(EDGES) == 12 and EDGES[0] == (0, 1)


def test_yaw_turns_the_front_towards_the_heading():
    c = box_corners([[0.0, 0.0, 0.0, 4.0, 2.0, 1.0, np.pi / 2]])[0]
    front = c[[0, 1]].mean(axis=0)
    np.testing.assert_allclose(front[:2], [0.0, 2.0], atol=1e-12)    # heading +y: the front is at y = +l/2


def test_points_in_box_respects_yaw_and_margin():
    box = [5.0, 0.0, 0.0, 4.0, 1.0, 2.0, np.pi / 4]
    along = np.array([[5 + 1.9 * np.cos(np.pi / 4), 1.9 * np.sin(np.pi / 4), 0.0]])
    across = np.array([[5 - 0.9 * np.sin(np.pi / 4), 0.9 * np.cos(np.pi / 4), 0.0]])
    assert points_in_box(along, box)[0]                 # 1.9 m along the heading, inside l/2 = 2
    assert not points_in_box(across, box)[0]            # 0.9 m across, outside w/2 = 0.5
    assert points_in_box(across, box, margin=0.45)[0]
    assert not points_in_box(np.array([[5.0, 0.0, 1.2]]), box)[0]     # above h/2
