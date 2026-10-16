"""Rasterize boxes and points onto the BEV grid.

Box convention (LiDAR frame): (x, y, z, l, w, h, yaw) with (x, y, z) the box centre, l along the heading,
w across it, and yaw the heading angle about z measured from the x axis.
"""
from __future__ import annotations

import numpy as np

from .bev_grid import BEVGrid


def box_cells(box, grid: BEVGrid, min_one_cell: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(ix, iy) of the cells whose centre lies inside the box footprint.

    A box smaller than a cell can contain no cell centre. With min_one_cell it still marks the cell that holds
    its centre, so small objects never vanish from the occupancy map.
    """
    x, y, _, l, w, _, yaw = (float(v) for v in box[:7])
    c, s = np.cos(yaw), np.sin(yaw)
    hx = 0.5 * (abs(l * c) + abs(w * s))          # half extents of the axis-aligned bounding box
    hy = 0.5 * (abs(l * s) + abs(w * c))
    nx, ny = grid.shape
    ix0, iy0 = (int(v) for v in grid.index(x - hx, y - hy))
    ix1, iy1 = (int(v) for v in grid.index(x + hx, y + hy))
    ix0, iy0, ix1, iy1 = max(ix0, 0), max(iy0, 0), min(ix1, nx - 1), min(iy1, ny - 1)

    ix = iy = np.zeros(0, dtype=np.int64)
    if ix0 <= ix1 and iy0 <= iy1:
        xs = grid.x_min + (np.arange(ix0, ix1 + 1) + 0.5) * grid.res
        ys = grid.y_min + (np.arange(iy0, iy1 + 1) + 0.5) * grid.res
        dx, dy = xs[:, None] - x, ys[None, :] - y
        inside = (np.abs(c * dx + s * dy) <= 0.5 * l) & (np.abs(-s * dx + c * dy) <= 0.5 * w)
        ii, jj = np.nonzero(inside)
        ix, iy = ii + ix0, jj + iy0
    if min_one_cell and len(ix) == 0 and grid.inside(x, y):
        cx, cy = grid.index(x, y)
        ix, iy = np.array([cx]), np.array([cy])
    return ix, iy


def rasterize_boxes(boxes, grid: BEVGrid, values=None, min_one_cell: bool = True) -> np.ndarray:
    """[nx, ny] float32 map holding, per cell, the maximum value of the boxes that cover it (default value 1)."""
    out = np.zeros(grid.shape, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    values = np.ones(len(boxes)) if values is None else np.asarray(values, dtype=np.float64)
    for box, value in zip(boxes, values):
        ix, iy = box_cells(box, grid, min_one_cell)
        out[ix, iy] = np.maximum(out[ix, iy], value)
    return out


def count_points(points, grid: BEVGrid) -> np.ndarray:
    """[nx, ny] int32 number of points per cell (points outside the grid are dropped)."""
    points = np.asarray(points)
    keep = grid.inside(points[:, 0], points[:, 1])
    ix, iy = grid.index(points[keep, 0], points[keep, 1])
    counts = np.zeros(grid.shape, dtype=np.int32)
    np.add.at(counts, (ix, iy), 1)
    return counts
