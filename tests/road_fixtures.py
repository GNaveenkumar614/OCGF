"""A synthetic street scanned by a 64-ring LiDAR (ring 0 = lowest beam), for the road-teacher tests.

Road at z = -1.73 for |y| < 4 m, sidewalks 0.15 m higher for 4 <= |y| < 8 m, walls at |y| = 8 m up to z = 1 m, and an
optional box on the street. The car drives straight ahead along x.
"""
from __future__ import annotations

import numpy as np

ROAD_Z, WALK_Z, KERB_Y, WALL_Y = -1.73, -1.58, 4.0, 8.0


def _ray_box(d, lo, hi):
    """Entry distance of rays d [N, 3] from the origin into the box [lo, hi] (inf if missed)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = lo / d, hi / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmax > 0), np.maximum(tmin, 0), np.inf)


def street_scan(obstacle=None, n_az: int = 900, fov_deg: float = 140.0) -> np.ndarray:
    """[N, 5] x, y, z, intensity, ring. obstacle: ((x0, y0, z0), (x1, y1, z1)) box corners, or None."""
    elev = np.radians(np.linspace(-24.9, 2.0, 64))
    az = np.radians(np.linspace(-fov_deg / 2, fov_deg / 2, n_az))
    E, A = np.meshgrid(elev, az, indexing="ij")
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1).reshape(-1, 3)
    ring = np.repeat(np.arange(64), n_az)
    t = np.full(len(d), np.inf)
    down = d[:, 2] < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        for z, ok in ((ROAD_Z, lambda y: np.abs(y) < KERB_Y), (WALK_Z, lambda y: (np.abs(y) >= KERB_Y) & (np.abs(y) < WALL_Y))):
            tz = np.where(down, z / d[:, 2], np.inf)
            hit = np.isfinite(tz) & ok(tz * d[:, 1])
            t = np.where(hit & (tz < t), tz, t)
        tw = np.where(np.abs(d[:, 1]) > 1e-9, WALL_Y / np.abs(d[:, 1]), np.inf)
        zw = tw * d[:, 2]
        wall = np.isfinite(tw) & (zw >= WALK_Z) & (zw <= 1.0)
        t = np.where(wall & (tw < t), tw, t)
    if obstacle is not None:
        tb = _ray_box(d, np.asarray(obstacle[0], float), np.asarray(obstacle[1], float))
        t = np.where(tb < t, tb, t)
    keep = np.isfinite(t) & (t < 120)
    xyz = d[keep] * t[keep, None]
    return np.column_stack([xyz, np.full(len(xyz), 0.5), ring[keep]])


def straight_path_positions(length: float = 80.0, step: float = 1.0):
    """(frames, world positions) of a car driving along +x from the origin; T_world_lidar = identity at frame 0."""
    x = np.arange(0.0, length, step)
    return np.arange(len(x)), np.column_stack([x, np.zeros_like(x), np.full_like(x, -0.8)])
