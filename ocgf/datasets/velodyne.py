"""Velodyne HDL-64E scans (KITTI, KITTI-360): reading, ring recovery and cropping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_RINGS = 64


def read_bin(source) -> np.ndarray:
    """A KITTI-style .bin scan (path or raw bytes) -> float32 [N, 4]: x, y, z, intensity in the LiDAR frame."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = np.frombuffer(source, dtype=np.float32)
    else:
        data = np.fromfile(source, dtype=np.float32)
    if data.size % 4:
        raise ValueError(f"a scan holds 4 floats per point, got {data.size} floats")
    return data.reshape(-1, 4)


def quadrant(x, y) -> np.ndarray:
    """Azimuth quadrant 1..4, counter-clockwise from forward-left; 0 only at x = y = 0."""
    x, y = np.asarray(x), np.asarray(y)
    q = np.zeros(x.shape, dtype=np.int8)
    q[(x > 0) & (y >= 0)] = 1
    q[(x <= 0) & (y > 0)] = 2
    q[(x < 0) & (y <= 0)] = 3
    q[(x >= 0) & (y < 0)] = 4
    return q


def ring_index(points) -> np.ndarray:
    """Laser ring of every point, recovered from the raw point order: uint8 [N], 0 = the last ring stored.

    Vectorized form of the heuristic used by trajectory road autolabeling (eerik98/lidar-camera-road-autolabeling,
    MIT; originally VincentCheungM/Run_based_segmentation issue #3). Walk the points from last to first and start a
    new ring whenever the azimuth crosses the forward direction from forward-left (quadrant 1) to forward-right
    (quadrant 4), capped at 64 rings. Run it on the raw scan: cropping or filtering first breaks the point order.
    """
    q = quadrant(points[:, 0], points[:, 1])[::-1]
    previous = np.concatenate([[0], q[:-1]])
    ring = np.minimum(np.cumsum((q == 4) & (previous == 1)), N_RINGS - 1)
    return ring[::-1].astype(np.uint8)


def ring_profile(points, ring) -> tuple[np.ndarray, np.ndarray]:
    """Per ring: point count [64] and median elevation angle in degrees [64] (NaN for empty rings). For QA."""
    elevation = np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))
    counts = np.bincount(ring, minlength=N_RINGS)[:N_RINGS]
    medians = np.array([np.median(elevation[ring == r]) if counts[r] else np.nan for r in range(N_RINGS)])
    return counts, medians


@dataclass(frozen=True)
class CropBox:
    """Region kept from each scan: the BEV grid plus margins for ground fitting and the road teacher."""
    x_min: float = -10.0
    x_max: float = 80.0
    y_min: float = -40.0
    y_max: float = 40.0

    def mask(self, points) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]
        return (x >= self.x_min) & (x < self.x_max) & (y >= self.y_min) & (y < self.y_max)
