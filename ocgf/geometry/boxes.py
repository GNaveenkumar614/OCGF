"""3D boxes in the LiDAR frame: (x, y, z, l, w, h, yaw) with (x, y, z) the centre, l along the heading.

Corner order: bottom face 0-3, then top face 4-7, each going front-left, front-right, rear-right, rear-left.
"""
from __future__ import annotations

import numpy as np

# 12 edges; the first two are the front edges (bottom, top)
EDGES = ((0, 1), (4, 5), (1, 2), (2, 3), (3, 0), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
_SIGNS = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                   [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]], dtype=np.float64) * 0.5


def box_corners(boxes) -> np.ndarray:
    """[K, 7] boxes -> [K, 8, 3] corners."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    local = _SIGNS[None] * boxes[:, None, 3:6]                       # [K, 8, 3]
    c, s = np.cos(boxes[:, 6])[:, None], np.sin(boxes[:, 6])[:, None]
    x = c * local[..., 0] - s * local[..., 1]
    y = s * local[..., 0] + c * local[..., 1]
    return np.stack([x, y, local[..., 2]], axis=-1) + boxes[:, None, :3]


def points_in_box(points, box, margin: float = 0.0) -> np.ndarray:
    """True for points [N, >=3] inside one box, grown by margin metres on every side."""
    x, y, z, l, w, h, yaw = (float(v) for v in box[:7])
    d = np.asarray(points)[:, :3] - (x, y, z)
    c, s = np.cos(yaw), np.sin(yaw)
    along, across = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return (np.abs(along) <= l / 2 + margin) & (np.abs(across) <= w / 2 + margin) & (np.abs(d[:, 2]) <= h / 2 + margin)
