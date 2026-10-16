import numpy as np
import pytest

from ocgf.geometry.bev_grid import BEVGrid
from ocgf.geometry.raster import box_cells, count_points, rasterize_boxes

GRID = BEVGrid()


def brute_force(box, grid=GRID) -> np.ndarray:
    """Reference: test every cell centre of the grid against the box footprint."""
    x, y, _, l, w, _, yaw = box
    X, Y = grid.center_grid()
    dx, dy = X - x, Y - y
    c, s = np.cos(yaw), np.sin(yaw)
    return (np.abs(c * dx + s * dy) <= l / 2) & (np.abs(-s * dx + c * dy) <= w / 2)


def mask_of(box, **kw) -> np.ndarray:
    mask = np.zeros(GRID.shape, dtype=bool)
    ix, iy = box_cells(box, GRID, **kw)
    mask[ix, iy] = True
    return mask


@pytest.mark.parametrize("box", [
    (10.0, 0.0, 0.0, 4.0, 1.6, 1.5, 0.0),          # car ahead, axis-aligned
    (20.0, 5.0, 0.0, 4.0, 1.8, 1.5, 0.7),          # rotated
    (1.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0),           # straddles x = 0 (rear half behind the grid)
    (49.0, -19.5, 0.0, 4.5, 2.0, 1.5, 0.4),        # straddles the far-right corner
    (25.0, 20.0, 0.0, 12.0, 2.5, 3.5, np.pi / 2),  # bus straddling the left edge, turned 90 degrees
    (-3.0, 0.0, 0.0, 8.0, 2.0, 1.5, 0.0),          # mostly behind the grid
])
def test_box_cells_match_brute_force(box):
    np.testing.assert_array_equal(mask_of(box, min_one_cell=False), brute_force(box))


def test_random_boxes_match_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(200):
        box = (rng.uniform(-5, 55), rng.uniform(-25, 25), 0.0,
               rng.uniform(0.3, 12), rng.uniform(0.3, 3), 1.5, rng.uniform(-np.pi, np.pi))
        np.testing.assert_array_equal(mask_of(box, min_one_cell=False), brute_force(box))


def test_box_outside_the_grid_marks_nothing():
    assert not mask_of((60.0, 0.0, 0.0, 4.0, 1.6, 1.5, 0.0)).any()
    assert not mask_of((10.0, 30.0, 0.0, 4.0, 1.6, 1.5, 0.0)).any()


def test_box_smaller_than_a_cell_keeps_its_centre_cell():
    tiny = (12.4, -1.5, 0.0, 0.1, 0.1, 1.7, 0.0)     # inside cell (49, 74), contains no cell centre
    assert not mask_of(tiny, min_one_cell=False).any()
    assert list(zip(*np.nonzero(mask_of(tiny)))) == [(49, 74)]


def test_rasterize_keeps_the_maximum_value():
    boxes = [(10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0), (11.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0)]
    out = rasterize_boxes(boxes, GRID, values=[0.3, 0.8])
    ix, iy = GRID.index(10.5, 0.0)
    assert out[ix, iy] == pytest.approx(0.8)          # covered by both boxes
    ix, iy = GRID.index(8.5, 0.0)
    assert out[ix, iy] == pytest.approx(0.3)          # covered by the first box only
    assert out.dtype == np.float32


def test_rasterize_without_boxes_is_empty():
    assert not rasterize_boxes(np.zeros((0, 7)), GRID).any()


def test_count_points():
    points = np.array([[12.4, -1.5, 0.0], [12.3, -1.4, 1.0], [0.1, 0.1, 0.0], [-1.0, 0.0, 0.0], [10.0, 25.0, 0.0]])
    counts = count_points(points, GRID)
    assert counts[49, 74] == 2
    assert counts[0, 80] == 1
    assert counts.sum() == 3                            # the last two points are outside the grid
