"""Pose warping between two frames' BEV grids (architecture plan §4.7), for the temporal-consistency term.

Each cell centre of frame t goes to the world and into the LiDAR frame of an earlier frame t'; its position there,
as grid_sample coordinates over a [nx, ny] map (rows = x index, columns = y index, align_corners=False), lets the
student's prediction for t' be resampled at t's cells. Cells are taken at ground height (a 2D warp: the poses carry
the car's motion; the small effect of pitch on x, y is ignored).
"""
from __future__ import annotations

import numpy as np

from .bev_grid import BEVGrid
from .transforms import invert_rigid, transform_points


def warp_grid(T_world_lidar_t, T_world_lidar_prev, grid: BEVGrid = BEVGrid(), ground_z: float = -1.73):
    """(coords [nx, ny, 2] float32: (x, y) grid_sample coordinates in t' of t's cells; inside [nx, ny] bool)."""
    nx, ny = grid.shape
    xs, ys = grid.center_grid()
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, ground_z)])
    T_prev_t = invert_rigid(np.asarray(T_world_lidar_prev)) @ np.asarray(T_world_lidar_t)
    p = transform_points(T_prev_t, pts)
    rows = (p[:, 0] - grid.x_min) / grid.res               # continuous index along x (edges at integers)
    cols = (p[:, 1] - grid.y_min) / grid.res
    coords = np.stack([2 * cols / ny - 1, 2 * rows / nx - 1], axis=-1).astype(np.float32).reshape(nx, ny, 2)
    inside = ((rows >= 0) & (rows < nx) & (cols >= 0) & (cols < ny)).reshape(nx, ny)
    return coords, inside
