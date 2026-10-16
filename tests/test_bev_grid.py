import numpy as np
import pytest

from ocgf.geometry.bev_grid import BEVGrid

GRID = BEVGrid()


def test_shape_is_200_by_160():
    assert GRID.shape == (200, 160)


def test_axis_orientation_example():
    # the reference case from CLAUDE.md
    ix, iy = GRID.index(12.4, -1.5)
    assert (int(ix), int(iy)) == (49, 74)


@pytest.mark.parametrize("x, y, cell", [
    (0.0, -20.0, (0, 0)),          # first cell: ahead of the LiDAR, on the far right
    (49.999, 19.999, (199, 159)),  # last cell: 50 m ahead, on the far left
    (0.2499, 0.0, (0, 80)),
    (0.25, -0.0001, (1, 79)),      # cell edges belong to the next cell up
])
def test_cell_edges(x, y, cell):
    ix, iy = GRID.index(x, y)
    assert (int(ix), int(iy)) == cell
    assert GRID.inside(x, y)


@pytest.mark.parametrize("x, y", [(-0.0001, 0.0), (50.0, 0.0), (10.0, -20.0001), (10.0, 20.0)])
def test_outside(x, y):
    assert not GRID.inside(x, y)


def test_centres_map_back_to_their_cells():
    X, Y = GRID.center_grid()
    ix, iy = GRID.index(X, Y)
    nx, ny = GRID.shape
    assert np.array_equal(ix, np.broadcast_to(np.arange(nx)[:, None], (nx, ny)))
    assert np.array_equal(iy, np.broadcast_to(np.arange(ny)[None, :], (nx, ny)))


def test_display_puts_forward_up_and_left_on_the_left():
    bev = np.zeros(GRID.shape)
    ix, iy = GRID.index(45.0, 15.0)            # far ahead, to the left
    bev[ix, iy] = 1
    row, col = np.argwhere(GRID.to_display(bev) == 1)[0]
    assert row < GRID.shape[0] // 2 and col < GRID.shape[1] // 2
