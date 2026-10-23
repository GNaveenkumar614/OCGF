"""Ground surface under the BEV grid, from the current LiDAR scan only (no future poses, no ground truth).

The camera branch samples image features at each cell's ground point (x, y, z_g), so z_g sets where on the image a
cell looks; a 0.6 m error at 30 m moves the sample about 12 m along the ray. v1 heuristics:

1. Seeds: points are binned into coarse cells (1 m) over the grid plus a margin. A cell's seeds are its points within
   `seed_band` of its low percentile z, and its height is their median. Seeds far from the calibrated plane are not
   ground (canopies, cells that only hold an object's top).
2. Outliers: a cell more than `max_step` from the median of its valid neighbours is dropped (car bodies, walls and
   hedges whose base the scan does not see); twice, so a cluster of two bad cells goes too. Cells with no neighbours
   to compare with (the side of a parked car next to the ego, where the scan sees no ground) are dropped when they
   sit more than `max_wide_dev` from the wide-scale surface.
   The ego car stands on the calibrated plane, so its footprint (`ego_box`) holds plane-height seeds: the LiDAR sees
   no ground within about 4 m (about 4.7 m ahead, behind the bonnet).
3. Spreading: heights go onto the fine grid by normalized convolution, weighted by seed count: a local estimate
   (`sigma`), falling back to a wide one (`sigma_wide`) in holes, and to the calibrated plane far from any seed.
4. Reliability r_g in [0, 1]: local support times a seed-spread term. A heuristic proxy until calibrated.

The plane is z = a x + b y + c in the LiDAR frame, fitted once on training frames (fit_plane) and kept in the config;
it is also the flat ground of ablation A2.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .bev_grid import BEVGrid


@dataclass(frozen=True)
class GroundConfig:
    coarse: float = 1.0                 # m, seed cell size
    margin: float = 3.0                 # m fitted around the grid, so edge cells have neighbours
    percentile: float = 10.0            # low percentile of a cell's z that anchors its seeds
    seed_band: float = 0.2              # m, seeds lie within this of that percentile
    min_seeds: int = 3
    max_plane_dev: float = 2.0          # m, a cell height farther than this from the plane is not ground
    max_step: float = 0.3               # m, allowed difference to the median of the 8 neighbours (kerbs are ~0.15)
    seed_cap: float = 10.0              # seed count at which a cell gets full weight
    sigma: float = 1.0                  # m, local normalized convolution
    sigma_wide: float = 4.0             # m, wide estimate used in holes
    hole_weight: float = 0.05           # weight of the wide estimate inside the local one (seed-count units)
    plane_weight: float = 0.05          # weight of the plane inside the wide estimate
    support_full: float = 3.0           # local smoothed weight at which r_g's support term reaches 1
    spread_scale: float = 0.1           # m, r_g's spread term is exp(-(spread / spread_scale)^2)
    max_wide_dev: float = 0.4           # m, allowed difference to the wide-scale surface
    ego_box: tuple = (-1.0, 2.0, 0.8)   # (x_min, x_max, half width) m: ego footprint, seeded at plane height
    plane: tuple = (0.0, 0.0, -1.73)    # (a, b, c): z = a x + b y + c, LiDAR frame

    @classmethod
    def from_dict(cls, d: dict | None) -> GroundConfig:
        d = dict(d or {})
        for key in ("plane", "ego_box"):
            if key in d:
                d[key] = tuple(float(v) for v in d[key])
        return cls(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["plane"], d["ego_box"] = list(self.plane), list(self.ego_box)
        return d


def plane_z(plane, x, y) -> np.ndarray:
    a, b, c = plane
    return a * np.asarray(x) + b * np.asarray(y) + c


def _coarse_frame(grid: BEVGrid, cfg: GroundConfig):
    x0, y0 = grid.x_min - cfg.margin, grid.y_min - cfg.margin
    nx = int(np.ceil((grid.x_max + cfg.margin - x0) / cfg.coarse))
    ny = int(np.ceil((grid.y_max + cfg.margin - y0) / cfg.coarse))
    return x0, y0, nx, ny


def seed_cells(points, grid: BEVGrid, cfg: GroundConfig = GroundConfig()):
    """Per coarse cell: height [cx, cy] (NaN: no ground seed), seed count, seed spread (std, m)."""
    x0, y0, nx, ny = _coarse_frame(grid, cfg)
    p = np.asarray(points, dtype=np.float64)
    ix = np.floor((p[:, 0] - x0) / cfg.coarse).astype(np.int64)
    iy = np.floor((p[:, 1] - y0) / cfg.coarse).astype(np.int64)
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cell, z = ix[keep] * ny + iy[keep], p[keep, 2]
    height, count, spread = (np.full(nx * ny, np.nan), np.zeros(nx * ny), np.full(nx * ny, np.nan))
    if len(z):
        order = np.lexsort((z, cell))                       # by cell, then by z
        cell, z = cell[order], z[order]
        n = np.bincount(cell, minlength=nx * ny)
        start = np.cumsum(n) - n
        has = n > 0
        z_low = np.full(nx * ny, np.nan)
        z_low[has] = z[start[has] + np.floor(cfg.percentile / 100 * (n[has] - 1)).astype(np.int64)]
        dz = z - z_low[cell]
        below = np.bincount(cell[dz < -cfg.seed_band], minlength=nx * ny)
        seed = np.abs(dz) <= cfg.seed_band
        n_seed = np.bincount(cell[seed], minlength=nx * ny)
        ok = n_seed >= cfg.min_seeds
        # seeds are a contiguous run of the z-sorted cell: its middle element is their (lower) median
        height[ok] = z[start[ok] + below[ok] + (n_seed[ok] - 1) // 2]
        count[ok] = n_seed[ok]
        s1 = np.bincount(cell[seed], weights=z[seed], minlength=nx * ny)
        s2 = np.bincount(cell[seed], weights=z[seed] ** 2, minlength=nx * ny)
        spread[ok] = np.sqrt(np.maximum(s2[ok] / n_seed[ok] - (s1[ok] / n_seed[ok]) ** 2, 0.0))
    shape = (nx, ny)
    height, count, spread = height.reshape(shape), count.reshape(shape), spread.reshape(shape)
    cx = x0 + (np.arange(nx) + 0.5) * cfg.coarse
    cy = y0 + (np.arange(ny) + 0.5) * cfg.coarse
    far = np.abs(height - plane_z(cfg.plane, cx[:, None], cy[None, :])) > cfg.max_plane_dev
    height[far] = np.nan
    for _ in range(2):
        height[_step_outliers(height, cfg.max_step)] = np.nan
    plane = plane_z(cfg.plane, cx[:, None], cy[None, :])
    ex0, ex1, ehw = cfg.ego_box
    ego = (cx[:, None] >= ex0) & (cx[:, None] <= ex1) & (np.abs(cy[None, :]) <= ehw)
    height[ego], spread[ego] = plane[ego], 0.0
    count[ego] = np.maximum(count[ego], 1e3)
    count[np.isnan(height)] = 0
    wide = _wide_surface(height, count, plane, cfg)
    off = np.isfinite(height) & ~ego & (np.abs(height - wide) > cfg.max_wide_dev)
    height[off] = np.nan
    count[np.isnan(height)] = 0
    spread[np.isnan(height)] = np.nan
    return height, count, spread


def _wide_surface(height, count, plane, cfg: GroundConfig) -> np.ndarray:
    """Wide-scale ground on the coarse grid: seeds smoothed over sigma_wide, the plane far from any seed."""
    w = np.minimum(count, cfg.seed_cap) / cfg.seed_cap
    num, den = _smooth(np.where(w > 0, height, 0.0), w, cfg.sigma_wide / cfg.coarse)
    return (num + cfg.plane_weight * plane) / (den + cfg.plane_weight)


def _step_outliers(height: np.ndarray, max_step: float) -> np.ndarray:
    """Cells whose height differs from the median of their valid 8-neighbours by more than max_step."""
    padded = np.pad(height, 1, constant_values=np.nan)
    nx, ny = height.shape
    neigh = np.stack([padded[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny]
                      for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # all-NaN neighbourhoods
        med = np.nanmedian(neigh, axis=0)
    return np.isfinite(height) & np.isfinite(med) & (np.abs(height - med) > max_step)


def _smooth(values, weights, sigma_cells):
    num = gaussian_filter(np.where(weights > 0, values * weights, 0.0), sigma_cells, mode="constant")
    den = gaussian_filter(weights, sigma_cells, mode="constant")
    return num, den


def ground_surface(points, grid: BEVGrid = BEVGrid(), cfg: GroundConfig = GroundConfig()):
    """(z_g, r_g), float32 [nx, ny] each: ground height under every cell centre and its reliability in [0, 1]."""
    height, count, spread = seed_cells(points, grid, cfg)
    x0, y0, nx, ny = _coarse_frame(grid, cfg)
    cx = x0 + (np.arange(nx) + 0.5) * cfg.coarse
    cy = y0 + (np.arange(ny) + 0.5) * cfg.coarse
    plane = plane_z(cfg.plane, cx[:, None], cy[None, :])
    w = np.minimum(count, cfg.seed_cap) / cfg.seed_cap
    h = np.where(w > 0, height, 0.0)

    wide = _wide_surface(height, count, plane, cfg)
    num, den = _smooth(h, w, cfg.sigma / cfg.coarse)
    z = (num + cfg.hole_weight * wide) / (den + cfg.hole_weight)
    s_num, _ = _smooth(np.nan_to_num(spread), w, cfg.sigma / cfg.coarse)
    spread_local = np.where(den > 1e-6, s_num / np.maximum(den, 1e-6), np.inf)
    support = np.clip(den * (2 * np.pi * (cfg.sigma / cfg.coarse) ** 2) / cfg.support_full, 0.0, 1.0)
    r = support * np.exp(-(spread_local / cfg.spread_scale) ** 2)

    # fine cell centres -> continuous coarse indices (coarse cell k is centred on index k)
    xs, ys = grid.centers()
    fx = (xs - x0) / cfg.coarse - 0.5
    fy = (ys - y0) / cfg.coarse - 0.5
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    coords = np.stack([FX.ravel(), FY.ravel()])
    z_g = map_coordinates(z, coords, order=1, mode="nearest").reshape(grid.shape)
    r_g = map_coordinates(r, coords, order=1, mode="nearest").reshape(grid.shape)
    return z_g.astype(np.float32), np.clip(r_g, 0, 1).astype(np.float32)


def height_above_ground(points, z_g: np.ndarray, grid: BEVGrid = BEVGrid()) -> tuple[np.ndarray, np.ndarray]:
    """(inside [N] bool, h [N]): which points fall in the grid, and their height above their cell's z_g (NaN outside)."""
    p = np.asarray(points)
    inside = grid.inside(p[:, 0], p[:, 1])
    h = np.full(len(p), np.nan)
    ix, iy = grid.index(p[inside, 0], p[inside, 1])
    h[inside] = p[inside, 2] - z_g[ix, iy]
    return inside, h


def fit_plane(seed_xyz: np.ndarray, iters: int = 3, inlier: float = 0.15) -> tuple[float, float, float]:
    """Least-squares plane z = a x + b y + c through ground seed points [N, 3], refitted on inliers."""
    p = np.asarray(seed_xyz, dtype=np.float64)
    keep = np.ones(len(p), bool)
    coef = np.zeros(3)
    for _ in range(iters):
        A = np.column_stack([p[keep, 0], p[keep, 1], np.ones(keep.sum())])
        coef, *_ = np.linalg.lstsq(A, p[keep, 2], rcond=None)
        keep = np.abs(plane_z(coef, p[:, 0], p[:, 1]) - p[:, 2]) < inlier
    return tuple(float(v) for v in coef)


def seed_points(points, grid: BEVGrid = BEVGrid(), cfg: GroundConfig = GroundConfig()) -> np.ndarray:
    """Coarse-cell ground seeds as points [M, 3] (cell centre, cell height), for fit_plane over training frames."""
    height, count, _ = seed_cells(points, grid, cfg)
    x0, y0, nx, ny = _coarse_frame(grid, cfg)
    ix, iy = np.nonzero(np.isfinite(height) & (count < 1e3))          # measured seeds, not the ego prior
    return np.column_stack([x0 + (ix + 0.5) * cfg.coarse, y0 + (iy + 0.5) * cfg.coarse, height[ix, iy]])
