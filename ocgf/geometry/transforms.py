"""Rigid transforms as 4x4 matrices. T_a_b maps points from frame b into frame a."""
from __future__ import annotations

import numpy as np


def rigid(R, t) -> np.ndarray:
    """4x4 transform from a 3x3 rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rigid_from_3x4(M) -> np.ndarray:
    """4x4 transform from a row-major 3x4 [R | t] matrix (the layout of KITTI calibration files)."""
    M = np.asarray(M, dtype=np.float64).reshape(3, 4)
    return rigid(M[:, :3], M[:, 3])


def invert_rigid(T) -> np.ndarray:
    """Inverse of a rigid transform, using R^T rather than a general matrix inverse."""
    R, t = T[:3, :3], T[:3, 3]
    return rigid(R.T, -R.T @ t)


def transform_points(T, points) -> np.ndarray:
    """Apply a 4x4 transform to points [..., 3]."""
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]
