"""Synthetic calibration shared by the geometry tests. The numbers are plausible, not taken from any dataset."""
import numpy as np

from ocgf.geometry.calib import CameraCalib
from ocgf.geometry.transforms import rigid

# LiDAR axes (x forward, y left, z up) -> camera axes (x right, y down, z forward)
AXES_CAM_LIDAR = np.array([[0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [1.0, 0.0, 0.0]])


def rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation from small Euler angles (radians), applied x, then y, then z."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def camera(width: int = 1408, height: int = 376) -> CameraCalib:
    """A front camera 0.8 m ahead of and 0.3 m below the LiDAR, slightly tilted."""
    K = np.array([[552.0, 0.0, 682.0], [0.0, 552.0, 238.0], [0.0, 0.0, 1.0]])
    R = rotation(0.01, -0.02, 0.005) @ AXES_CAM_LIDAR
    t = -R @ np.array([0.8, 0.05, -0.3])        # camera centre in the LiDAR frame: (0.8, 0.05, -0.3)
    return CameraCalib(K, rigid(R, t), width, height)


def points_in_front(n: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(3, 60, n), rng.uniform(-20, 20, n), rng.uniform(-2.5, 3, n)])
