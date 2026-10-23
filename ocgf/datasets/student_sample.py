"""One student sample as numpy arrays: image, cell projections, LiDAR point features, BEV inputs and targets.

Image: the rectified left image, centre-cropped to IMAGE_W and padded with black rows at the bottom to IMAGE_H.
1400 x 392 is a multiple of both the ViT's 14-px patches (100 x 28 tokens) and the camera stem's stride 8 (175 x 49),
so both feature maps cover exactly the same pixels; the intrinsics follow the crop. KITTI-360: 4 px off each side,
16 rows of padding.

Cell projections ("grid_uv"): every BEV cell's ground point in that image, as grid_sample coordinates in [-1, 1]
(align_corners=False: x = 2 (u + 0.5) / W - 1). The feature maps span the whole padded image, so the same
coordinates serve the tokens and the stride-8 map. Two versions:
  grid_uv       at the ground surface z_g estimated from the current scan (the main, ground-guided lifting)
  grid_uv_flat  at the calibrated plane: flat-plane IPM, the camera-only baseline E0a (no LiDAR) and ablation A2
Cells whose ground point falls outside the real image rows (or behind the camera) get -2: outside, sampled as zero.

LiDAR points ("pt_feats" [N, 10], "pt_cell" [N]): per point x/50, y/20, z, intensity, height above z_g, offset to the
cell centre (x, y, in cells), offset to the mean of the cell's points (x, y, z); the flat cell index ix * ny + iy.
Points outside the grid are dropped.
"""
from __future__ import annotations

import numpy as np

from ..geometry.bev_grid import BEVGrid
from ..geometry.ground import plane_z
from ..geometry.projection import project

IMAGE_W, IMAGE_H = 1400, 392
POINT_FEATURES = 10
OUTSIDE = -2.0


def prepare_image(image: np.ndarray, camera, width: int = IMAGE_W, height: int = IMAGE_H):
    """(image [height, width, 3] uint8, camera of that image, real rows): centre crop in width, pad at the bottom."""
    H, W = image.shape[:2]
    left = (W - width) // 2
    rows = min(H, height)
    out = np.zeros((height, width, 3), np.uint8)
    src = image[:rows, max(left, 0):max(left, 0) + width]
    out[:rows, max(-left, 0):max(-left, 0) + src.shape[1]] = src
    return out, camera.crop(left, 0, width, height), rows


def cell_projection(camera, z: np.ndarray, grid: BEVGrid, real_rows: int):
    """(grid_uv [nx, ny, 2] float32 in grid_sample coordinates, OUTSIDE where invalid; valid [nx, ny] bool)."""
    xs, ys = grid.center_grid()
    uv, _, ok = project(np.column_stack([xs.ravel(), ys.ravel(), np.asarray(z, np.float64).ravel()]), camera)
    ok &= np.nan_to_num(uv[:, 1], nan=np.inf) < real_rows - 0.5          # not in the padding rows
    g = np.full((len(uv), 2), OUTSIDE, np.float32)
    g[ok, 0] = 2 * (uv[ok, 0] + 0.5) / camera.width - 1
    g[ok, 1] = 2 * (uv[ok, 1] + 0.5) / camera.height - 1
    return g.reshape(*grid.shape, 2), ok.reshape(grid.shape)


def point_features(points, z_g: np.ndarray, grid: BEVGrid):
    """(features [N, POINT_FEATURES] float32, flat cell index [N] int64) for the points inside the grid."""
    p = np.asarray(points, np.float64)
    p = p[grid.inside(p[:, 0], p[:, 1])]
    nx, ny = grid.shape
    ix, iy = grid.index(p[:, 0], p[:, 1])
    cell = ix * ny + iy
    n = np.bincount(cell, minlength=nx * ny)
    mean = [np.bincount(cell, weights=p[:, k], minlength=nx * ny)[cell] / n[cell] for k in range(3)]
    cx = grid.x_min + (ix + 0.5) * grid.res
    cy = grid.y_min + (iy + 0.5) * grid.res
    f = np.column_stack([p[:, 0] / grid.x_max, p[:, 1] / max(abs(grid.y_min), grid.y_max), p[:, 2], p[:, 3],
                         p[:, 2] - z_g[ix, iy], (p[:, 0] - cx) / grid.res, (p[:, 1] - cy) / grid.res,
                         p[:, 0] - mean[0], p[:, 1] - mean[1], p[:, 2] - mean[2]])
    return f.astype(np.float32), cell.astype(np.int64)


def build_sample(image: np.ndarray, points, camera, record: dict, grid: BEVGrid = BEVGrid(),
                 plane=(0.0, 0.0, -1.73)) -> dict:
    """The student's sample for one frame. record: ocgf.training.targets.load_record() of the same frame (BEV inputs
    and, for training frames, targets)."""
    img, cam, rows = prepare_image(image, camera)
    grid_uv, in_view = cell_projection(cam, record["z_g"], grid, rows)
    xs, ys = grid.center_grid()
    grid_uv_flat, fov_flat = cell_projection(cam, plane_z(plane, xs, ys), grid, rows)
    feats, cell = point_features(points, record["z_g"], grid)
    sample = {"image": img, "grid_uv": grid_uv, "M_C": record["M_C"] & in_view, "grid_uv_flat": grid_uv_flat,
              "fov_flat": fov_flat, "pt_feats": feats, "pt_cell": cell}
    for key in ("road_t", "unk_t", "occ_t", "M_L", "vis_lidar", "r_g", "n_pts", "n_above"):
        if key in record:
            sample[key] = record[key]
    return sample
