# Adapted from lidar-camera-road-autolabeling (github.com/eerik98, commit 2534254): lidar_autolabeling.py and utils.py.
# MIT License, Copyright (c) 2024 Eerik Alamikkotervo; full notice in LICENSE-lidar-camera-road-autolabeling here.
# Changes for OCGF: scores are kept per point of the stored scan (NaN where not scored), so they can be exported per
# point; projection through ocgf.geometry (with R_rect_00); NaN instead of an RGB-encoded unknown.
"""Step 2 of the road teacher: a road score for LiDAR points on the rings that hold a trajectory point.

- height score: points at or below the trajectory point's height score 1; higher points exp(-dh^2 / height_std^2);
  points far nearer or farther than it along the ring (> max_radial_dist) are not the road surface (0);
- gradient score: walking along the ring outward from the trajectory point in both directions, height steps larger
  than the largest step between the wheels accumulate; the score is exp(-sum^2 / gradient_std^2), so it drops at a kerb
  and stays low beyond it.
The two are averaged (both required), then interpolated over the image (linear, inside the points' hull); rows above
the highest pixel scored exactly 1 are unknown.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from ...geometry.projection import project
from .config import RoadConfig
from .trajectory import RingScan


def nanmean(a: np.ndarray, b: np.ndarray, require_both: bool) -> np.ndarray:
    """Mean of a and b; where only one is NaN, the other (unless require_both); NaN where both are."""
    out = np.full(np.shape(a), np.nan)
    na, nb = np.isnan(a), np.isnan(b)
    both = ~na & ~nb
    out[both] = (a[both] + b[both]) / 2
    if not require_both:
        out[na & ~nb] = b[na & ~nb]
        out[~na & nb] = a[~na & nb]
    return out


def height_score(scan: RingScan, wheels: np.ndarray, cfg: RoadConfig) -> np.ndarray:
    pts = scan.points
    score = np.full(len(pts), np.nan)
    rng = np.linalg.norm(pts[:, :3], axis=1)
    for ring, c, _, _ in wheels:
        ids = scan.rings[int(ring)]
        diff = pts[ids, 2] - pts[c, 2]
        s = np.where(diff > 0, np.exp(-diff ** 2 / cfg.height_std ** 2), 1.0)
        s[np.abs(rng[ids] - rng[c]) > cfg.max_radial_dist] = 0.0
        score[ids] = s
    return score


def gradient_score(scan: RingScan, wheels: np.ndarray, cfg: RoadConfig) -> np.ndarray:
    pts = scan.points
    score = np.full(len(pts), np.nan)
    for ring, c, left, right in wheels:
        ids = scan.rings[int(ring)]
        theta = np.mod(np.arctan2(pts[ids, 1], pts[ids, 0]) - np.arctan2(pts[c, 1], pts[c, 0]), 2 * np.pi)
        pos = {int(p): k for k, p in enumerate(ids)}
        t_left, t_right = theta[pos[int(left)]], theta[pos[int(right)]]
        order = np.argsort(theta)                     # from the trajectory point, counter-clockwise from 0 to 2 pi
        st, sz = theta[order], pts[ids[order], 2]
        diff = -np.diff(sz)
        diff = np.append(diff, diff[-1]) if len(diff) else np.zeros(1)
        diff[st < np.pi] = -diff[st < np.pi]
        between = (st > t_right) | (st < t_left)
        limit = diff[between].max() if between.any() else 0.0
        diff[diff < limit] = 0.0                      # adaptive threshold: ignore steps as small as on the road
        far = st > np.pi
        diff[far] = np.cumsum(diff[far][::-1])[::-1]
        diff[~far] = np.cumsum(diff[~far])
        score[ids[order]] = np.exp(-diff ** 2 / cfg.gradient_std ** 2)
    return score


def point_scores(scan: RingScan, wheels: np.ndarray, cfg: RoadConfig) -> np.ndarray:
    """Road score per point of the stored scan; NaN for points that were not scored."""
    if not (cfg.use_height or cfg.use_gradient):
        raise ValueError("use_height or use_gradient must be set")
    h = height_score(scan, wheels, cfg) if cfg.use_height else None
    g = gradient_score(scan, wheels, cfg) if cfg.use_gradient else None
    if h is not None and g is not None:
        return nanmean(h, g, require_both=True)
    return h if h is not None else g


def image_label(points: np.ndarray, scores: np.ndarray, camera, cfg: RoadConfig) -> np.ndarray:
    """Float [H, W]: point scores interpolated over the image (NaN outside their hull and above the road)."""
    H, W = camera.height, camera.width
    finite = np.isfinite(scores)
    uv, _, front = project(np.asarray(points)[finite, :3], camera, min_depth=0.1)
    coords, values = uv[front], scores[finite][front]
    wheels0, _, _ = project(np.array([cfg.origin_left, cfg.origin_right]), camera, min_depth=0.01)
    wheels0 = wheels0[np.isfinite(wheels0).all(axis=1)]
    coords = np.vstack([coords, wheels0])
    values = np.r_[values, np.ones(len(wheels0))]
    if len(coords) < 3:
        return np.full((H, W), np.nan)
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    label = griddata(coords, values, (gx, gy), method="linear")
    rows = np.flatnonzero(np.any(label == 1, axis=1))
    label[: rows[0] if len(rows) else H, :] = np.nan
    return label
