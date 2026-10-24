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

Reliability ("Q_geo" [6, nx, ny], "valid_L" [nx, ny]): the geometric reliability channels of the plan's Q, from the
same scan: log(1 + points in the cell), range / 50 m, ground reliability r_g, ground returns M_L, camera shadow, camera
validity M_C (heuristic proxies until calibrated). The model adds the camera's own road-score entropy H_cam and the
object confidence q_obj. valid_L marks where the LiDAR features mean something: the cell is in the LiDAR's line of sight
or holds points. (Not M_L: ground returns are ~0.4 m apart at 10 m, so most visible far cells have none, and an
obstacle cell has none at all.)

Road boundary ("road_bd" [nx, ny], training frames): distance (m) from each cell to the nearest cell on the road
target's 0.5 contour (1000 where the frame has no contour), for the boundary-weighted road loss.

Objects ("O_B" [9, nx, ny], with objects given): ocgf.datasets.objects.object_channels, from the MTrans boxes of the
frame; the original camera is used, because the detector's 2D boxes are in the original image.

LiDAR points ("pt_feats" [N, 10], "pt_cell" [N], "pt_uv" [N, 2]): per point x/50, y/20, z, intensity, height above z_g, offset to the
cell centre (x, y, in cells), offset to the mean of the cell's points (x, y, z); the flat cell index ix * ny + iy;
and each point's grid_sample coordinates in the image (OUTSIDE when it is not seen), for the LiDAR-anchored camera
lifting of ablation A1. Points outside the grid are dropped.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from ..geometry.bev_grid import BEVGrid
from ..geometry.ground import plane_z
from ..geometry.projection import project
from .objects import ObjectConfig, object_channels

IMAGE_W, IMAGE_H = 1400, 392
Q_GEO = ("log_points", "range", "r_g", "M_L", "shadow_cam", "M_C")
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


def point_projection(points, camera, real_rows: int) -> np.ndarray:
    """[N, 2] float32 grid_sample coordinates of the points in the image (OUTSIDE where not seen)."""
    uv, _, ok = project(np.asarray(points, np.float64)[:, :3], camera)
    ok &= np.nan_to_num(uv[:, 1], nan=np.inf) < real_rows - 0.5
    g = np.full((len(uv), 2), OUTSIDE, np.float32)
    g[ok, 0] = 2 * (uv[ok, 0] + 0.5) / camera.width - 1
    g[ok, 1] = 2 * (uv[ok, 1] + 0.5) / camera.height - 1
    return g


def reliability(record: dict, grid: BEVGrid, camera_valid: np.ndarray):
    """(Q_geo [6, nx, ny] float32, valid_L [nx, ny] bool) from a frame's BEV inputs (see the module docstring)."""
    xs, ys = grid.center_grid()
    q = np.stack([np.log1p(record["n_pts"]), np.hypot(xs, ys) / 50.0, record["r_g"], record["M_L"],
                  record["shadow_cam"], camera_valid]).astype(np.float32)
    return q, (record["vis_lidar"] | (record["n_pts"] > 0))


def boundary_distance(road_t: np.ndarray, grid: BEVGrid) -> np.ndarray:
    """[nx, ny] float32: distance (m) to the nearest cell on the 0.5 contour of road_t (NaN cells are neither side)."""
    finite = np.isfinite(road_t)
    inside = finite & (np.nan_to_num(road_t) >= 0.5)
    outside = finite & ~inside
    edge = (inside & binary_dilation(outside)) | (outside & binary_dilation(inside))
    if not edge.any():
        return np.full(grid.shape, 1000.0, np.float32)
    return (distance_transform_edt(~edge) * grid.res).astype(np.float32)


def build_sample(image: np.ndarray, points, camera, record: dict, grid: BEVGrid = BEVGrid(),
                 plane=(0.0, 0.0, -1.73), objects=None, obj_cfg: ObjectConfig = ObjectConfig()) -> dict:
    """The student's sample for one frame. record: ocgf.training.targets.load_record() of the same frame (BEV inputs
    and, for training frames, targets); objects: its FrameObjects, for models with object channels."""
    img, cam, rows = prepare_image(image, camera)
    grid_uv, in_view = cell_projection(cam, record["z_g"], grid, rows)
    xs, ys = grid.center_grid()
    grid_uv_flat, fov_flat = cell_projection(cam, plane_z(plane, xs, ys), grid, rows)
    feats, cell = point_features(points, record["z_g"], grid)
    pts = np.asarray(points, np.float64)
    pt_uv = point_projection(pts[grid.inside(pts[:, 0], pts[:, 1])], cam, rows)
    m_c = record["M_C"] & in_view
    q_geo, valid_l = reliability(record, grid, m_c)
    sample = {"image": img, "grid_uv": grid_uv, "M_C": m_c, "grid_uv_flat": grid_uv_flat, "fov_flat": fov_flat,
              "pt_feats": feats, "pt_cell": cell, "pt_uv": pt_uv, "Q_geo": q_geo, "valid_L": valid_l}
    if objects is not None:
        sample["O_B"] = object_channels(objects, record["z_g"], camera, grid, obj_cfg)
    for key in ("road_t", "unk_t", "occ_t", "M_L", "vis_lidar", "r_g", "n_pts", "n_above"):
        if key in record:
            sample[key] = record[key]
    if "road_t" in record:
        sample["road_bd"] = boundary_distance(record["road_t"], grid)
    return sample
