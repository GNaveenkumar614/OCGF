"""Pose warping of BEV maps between frames: identity, forward motion and the overlap mask."""
import numpy as np
import pytest

from ocgf.geometry.bev_grid import BEVGrid
from ocgf.geometry.transforms import rigid
from ocgf.geometry.warp import warp_grid

torch = pytest.importorskip("torch")
GRID = BEVGrid()


def sample(bev, coords):
    t = torch.from_numpy(bev.astype(np.float32))[None, None]
    return torch.nn.functional.grid_sample(t, torch.from_numpy(coords)[None], mode="nearest",
                                           align_corners=False)[0, 0].numpy()


def test_the_identity_warp_returns_the_map():
    bev = np.random.default_rng(0).uniform(size=GRID.shape)
    coords, inside = warp_grid(np.eye(4), np.eye(4), GRID)
    assert inside.all() and np.allclose(sample(bev, coords), bev)


def test_after_driving_5_m_forward_a_cell_was_5_m_further_ahead_before():
    T_prev = rigid(np.eye(3), [100.0, 50.0, 0.0])
    T_now = T_prev @ rigid(np.eye(3), [5.0, 0.0, 0.0])          # the car moved 5 m along its x axis
    bev = np.zeros(GRID.shape)
    ix, iy = (int(v) for v in GRID.index(25.1, 3.1))
    bev[ix, iy] = 1.0                                           # something at x = 25 m in the earlier frame
    coords, inside = warp_grid(T_now, T_prev, GRID)
    warped = sample(bev, coords)
    jx, jy = (int(v) for v in GRID.index(20.1, 3.1))
    assert warped[jx, jy] == 1.0 and warped.sum() == 1.0        # ... is at x = 20 m now
    assert not inside[GRID.index(47.0, 0.0)] and inside[GRID.index(10.0, 0.0)]   # beyond the earlier grid's reach
