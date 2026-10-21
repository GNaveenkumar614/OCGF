"""Which BEV cells' ground a sensor can see (a viewshed).

The ground point of a cell (its centre at ground height z_g) is visible when the ray from the sensor to it passes above
every occluder that is closer to the sensor in that direction. Occluders are columns standing on the ground: each
occluder cell holds the top height of what stands there (e.g. points 0.3-3 m above ground, box tops). Tree canopies
and other overhangs must be left out by the caller: a column model would wrongly let them hide the ground below.

Rays are binned by azimuth around the sensor. An occluder cell covers every azimuth bin its footprint spans, and blocks
cells farther away than itself in those bins, up to its elevation angle. A cell never occludes itself or the cells
next to it (they are compared only with occluders at least `self_margin` closer), so a low curb hides about
the ground directly behind it and nothing in front.
"""
from __future__ import annotations

import numpy as np

from .bev_grid import BEVGrid


def viewshed(grid: BEVGrid, origin, ground_z: np.ndarray, occluder_top: np.ndarray, az_res_deg: float = 0.25,
             d_res: float = 0.1, self_margin: float | None = None, max_range: float = 80.0) -> np.ndarray:
    """Bool [nx, ny]: the ground of the cell is in line of sight of a sensor at origin (x, y, z, LiDAR frame).

    ground_z [nx, ny]: ground height per cell (NaN: unknown, never visible). occluder_top [nx, ny]: top height of the
    occluder in the cell (NaN: no occluder).
    """
    ox, oy, oz = (float(v) for v in origin)
    xs, ys = grid.center_grid()
    dist = np.hypot(xs - ox, ys - oy)
    az = np.arctan2(ys - oy, xs - ox)
    az_res = np.radians(az_res_deg)
    n_az = int(np.ceil(2 * np.pi / az_res))
    n_d = int(np.ceil(max_range / d_res)) + 1
    margin = grid.res * 1.2 if self_margin is None else self_margin

    # occluders -> max elevation angle per (azimuth bin, distance bin)
    occ = np.isfinite(occluder_top) & (dist > 1e-6)
    elev_max = np.full((n_az, n_d), -np.inf)
    if occ.any():
        d_o, az_o = dist[occ], az[occ]
        e_o = np.arctan2(occluder_top[occ] - oz, d_o)
        half = np.arctan2(grid.res * np.sqrt(0.5), d_o)                 # angular half-width of the cell footprint
        lo = np.floor((az_o - half + np.pi) / az_res).astype(np.int64)
        hi = np.floor((az_o + half + np.pi) / az_res).astype(np.int64)
        span = hi - lo + 1
        rows = np.repeat(np.arange(len(lo)), span)
        offsets = np.arange(span.sum()) - np.repeat(np.cumsum(span) - span, span)
        az_bins = (lo[rows] + offsets) % n_az
        d_bins = np.clip((d_o[rows] / d_res).astype(np.int64), 0, n_d - 1)
        np.maximum.at(elev_max, (az_bins, d_bins), e_o[rows])
    blocking = np.maximum.accumulate(elev_max, axis=1)                  # max elevation of anything up to distance d

    # targets: the ground point of every cell
    target = np.arctan2(ground_z - oz, np.maximum(dist, 1e-6))
    t_az = (np.floor((az + np.pi) / az_res).astype(np.int64)) % n_az
    look = np.floor((dist - margin) / d_res).astype(np.int64)            # only occluders clearly closer than the cell
    before = np.where(look >= 0, blocking[t_az, np.clip(look, 0, n_d - 1)], -np.inf)
    return np.isfinite(ground_z) & (target > before)
