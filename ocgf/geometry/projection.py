"""Projection between the LiDAR frame and image / feature-map coordinates.

Pixel centres sit at integer coordinates, so an image of width W covers u in [-0.5, W - 0.5).
Feature maps of stride s use u_feat = (u + 0.5) / s - 0.5, which matches grid_sample(align_corners=False).
"""
from __future__ import annotations

import numpy as np

from .calib import CameraCalib
from .transforms import invert_rigid, transform_points


def in_image(uv, width: int, height: int) -> np.ndarray:
    """True where pixel coordinates fall inside the image. NaN coordinates are never inside."""
    uv = np.asarray(uv, dtype=np.float64)
    u, v = uv[..., 0], uv[..., 1]
    return (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)


def project(points, calib: CameraCalib, min_depth: float = 0.1):
    """LiDAR points [N, 3] -> pixel coordinates uv [N, 2], camera depth [N], valid [N].

    Points at depth <= min_depth (behind or too close to the camera) get NaN coordinates and are never valid.
    """
    cam = transform_points(calib.T_cam_lidar, points)
    depth = cam[:, 2]
    front = depth > min_depth
    uv = np.full((len(cam), 2), np.nan)
    pix = cam[front] @ calib.K.T
    uv[front] = pix[:, :2] / pix[:, 2:3]
    return uv, depth, front & in_image(uv, calib.width, calib.height)


def unproject(uv, depth, calib: CameraCalib) -> np.ndarray:
    """Pixel coordinates [N, 2] and camera depth [N] -> LiDAR-frame points [N, 3]."""
    uv = np.asarray(uv, dtype=np.float64)
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(calib.K).T     # rows are [x/z, y/z, 1]
    cam = rays * np.asarray(depth, dtype=np.float64)[:, None]
    return transform_points(invert_rigid(calib.T_cam_lidar), cam)


def to_feature_coords(uv, stride: int) -> np.ndarray:
    """Pixel coordinates -> coordinates on a stride-s feature map (same convention)."""
    return (np.asarray(uv, dtype=np.float64) + 0.5) / stride - 0.5


def to_grid_sample_coords(uv, stride: int, feat_width: int, feat_height: int) -> np.ndarray:
    """Pixel coordinates -> [-1, 1] coordinates for grid_sample(align_corners=False) on a stride-s feature map."""
    f = to_feature_coords(uv, stride)
    return np.stack([(f[..., 0] + 0.5) / feat_width * 2 - 1,
                     (f[..., 1] + 0.5) / feat_height * 2 - 1], axis=-1)
