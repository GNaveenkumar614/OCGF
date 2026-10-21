# Adapted from lidar-camera-road-autolabeling (github.com/eerik98, commit 2534254): pre_processing.py and utils.py.
# MIT License, Copyright (c) 2024 Eerik Alamikkotervo; full notice in LICENSE-lidar-camera-road-autolabeling here.
# Changes for OCGF:
#   - the future path comes from the SLAM poses (poses.txt), not from OXTS latitude/longitude through UTM;
#   - laser rings come from the stored ring index (recovered from the raw scan order before cropping, same rule);
#   - projection through ocgf.geometry, which includes R_rect_00 (the original KITTI-360 calibration omits it);
#   - point indices refer to the stored scan (so per-point scores line up with it); searches vectorised.
"""Step 1 of the road teacher: where the car will drive, found in the current scan.

The next `trajectory_length` metres of the car's path, in the LiDAR frame of the current frame, are matched ring by ring
(nearest ring first) to the scan's ground points: one trajectory point per ring, each farther out than the previous and
not much higher. Beside each, the points a wheel's width to the left and right are the wheel points. A wheel point that
nearer points hide in the image (something stands in the way) ends the trajectory there. The wheels' polygon in the
image is certainly road; the camera label learns what road looks like from it.
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy.spatial import cKDTree

from ...geometry.projection import project
from ...geometry.transforms import invert_rigid, transform_points
from .config import RoadConfig


def path_positions(T_world_lidar: dict, T_pose_lidar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(frames, positions [M, 3]) of the vehicle 'pose' frame (GPS/IMU) in world coordinates, for every posed frame."""
    frames = np.array(sorted(T_world_lidar), dtype=np.int64)
    T_lidar_pose = invert_rigid(np.asarray(T_pose_lidar, dtype=np.float64))
    positions = np.array([(T_world_lidar[f] @ T_lidar_pose)[:3, 3] for f in frames])
    return frames, positions


def future_path(frame: int, frames: np.ndarray, positions: np.ndarray, T_world_lidar_now, length: float) -> np.ndarray:
    """[K, 3] (x, y, heading) of the path from `frame` on, in the LiDAR frame at `frame`, until `length` metres are
    driven (the pose that passes the length is left out, as take_following_N_m does). Empty if the path is shorter."""
    start = int(np.searchsorted(frames, frame))
    p = positions[start:]
    if len(p) < 2:
        return np.zeros((0, 3))
    driven = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
    beyond = np.flatnonzero(driven[1:] > length)
    if not len(beyond):
        return np.zeros((0, 3))
    p = p[:beyond[0] + 1]
    local = transform_points(invert_rigid(np.asarray(T_world_lidar_now, dtype=np.float64)), p)
    step = np.diff(local[:, :2], axis=0)
    heading = np.arctan2(step[:, 1], step[:, 0])
    heading = np.r_[heading, heading[-1:]] if len(heading) else np.zeros(len(local))
    return np.column_stack([local[:, 0], local[:, 1], heading])


class RingScan:
    """The scan's points inside the LiDAR field of view, grouped by laser ring (lowest rings dropped)."""

    def __init__(self, points: np.ndarray, cfg: RoadConfig):
        self.points = np.asarray(points, dtype=np.float64)
        theta = np.arctan2(self.points[:, 1], self.points[:, 0])
        ring = self.points[:, 4].astype(np.int64)
        keep = (np.abs(theta) < np.radians(cfg.lidar_fov_deg) / 2) & (ring >= cfg.drop_rings)
        idx = np.flatnonzero(keep)
        order = np.argsort(ring[idx], kind="stable")
        idx = idx[order]
        r = ring[idx]
        starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        ends = np.r_[starts[1:], len(idx)]
        self.rings = {int(r[s]): idx[s:e] for s, e in zip(starts, ends)}     # ring -> indices into points

    def ring_ids(self) -> list[int]:
        return sorted(self.rings)

    def indices(self) -> np.ndarray:
        return np.concatenate(list(self.rings.values())) if self.rings else np.zeros(0, dtype=np.int64)


def find_trajectory(scan: RingScan, path: np.ndarray, cfg: RoadConfig) -> list[tuple[int, int, float]]:
    """[(ring, point index, heading)]: per ring, nearest ring first, the scan point closest to the path (closest_pcd_point)."""
    if not len(path):
        return []
    tree = cKDTree(path[:, :2])
    pts, prev = scan.points, np.asarray(cfg.origin_center, dtype=np.float64)
    out = []
    for ring in scan.ring_ids():
        ids = scan.rings[ring]
        p = pts[ids]
        ok = (p[:, 2] - prev[2] < cfg.height_filter) & \
             (np.linalg.norm(p[:, :2], axis=1) - np.linalg.norm(prev[:2]) > cfg.distance_filter)
        cand = ids[ok]
        if not len(cand):
            continue
        dist, j = tree.query(pts[cand, :2])
        k = int(np.argmin(dist))
        if dist[k] < cfg.max_scan2pose:
            out.append((ring, int(cand[k]), float(path[j[k], 2])))
            prev = pts[cand[k], :3]
    return out


def wheel_points(scan: RingScan, trajectory, cfg: RoadConfig) -> np.ndarray:
    """[K, 4] (ring, centre, left, right) point indices: on each trajectory ring, the points at the azimuths of the
    positions one wheel offset to either side of the path (wheel_contact_points)."""
    pts, out = scan.points, []
    for ring, c, heading in trajectory:
        ids = scan.rings[ring]
        az = np.arctan2(pts[ids, 1], pts[ids, 0])
        x, y = pts[c, :2]
        side = cfg.wheel_offset * np.array([np.cos(heading + np.pi / 2), np.sin(heading + np.pi / 2)])
        lx, ly = (x, y) + side
        rx, ry = (x, y) - side
        left = ids[int(np.argmin(np.abs(az - np.arctan2(ly, lx))))]
        right = ids[int(np.argmin(np.abs(az - np.arctan2(ry, rx))))]
        if (np.linalg.norm(pts[left, :2] - pts[c, :2]) < cfg.max_center2wheel
                and np.linalg.norm(pts[right, :2] - pts[c, :2]) < cfg.max_center2wheel):
            out.append((ring, c, left, right))
    return np.array(out, dtype=np.int64).reshape(-1, 4)


def occlusion_filter(scan: RingScan, wheels: np.ndarray, camera, cfg: RoadConfig) -> np.ndarray:
    """Cut the trajectory at the first wheel point that a nearer ring's point hides in the image: a nearer point in the
    same image column (within occlusion_px) that projects above the wheel point stands between it and the camera."""
    if len(wheels) < 2:
        return wheels
    pts = scan.points
    uv, _, front = project(pts[:, :3], camera, min_depth=0.1)
    ring_of = pts[:, 4].astype(np.int64)
    used = scan.indices()
    used = used[front[used]]
    for i in range(1, len(wheels)):
        ring, _, left, right = wheels[i]
        prev = used[ring_of[used] < ring]
        for w in (right, left):
            if not front[w]:
                continue
            near = np.abs(uv[prev, 0] - uv[w, 0]) < cfg.occlusion_px
            if near.any() and uv[prev[near], 1].min() < uv[w, 1]:
                return wheels[:i]
    return wheels


def trajectory_polygon(scan: RingScan, wheels: np.ndarray, camera, cfg: RoadConfig) -> np.ndarray:
    """Bool [H, W]: pixels inside the polygon of the wheel points (left side outward, right side back) and the
    front-axle wheel positions. Empty without wheel points."""
    mask = np.zeros((camera.height, camera.width), dtype=bool)
    if not len(wheels):
        return mask
    pts = scan.points
    corners = np.vstack([cfg.origin_left, pts[wheels[:, 2], :3], pts[wheels[::-1, 3], :3], cfg.origin_right])
    uv, depth, _ = project(corners, camera, min_depth=0.01)
    if not np.isfinite(uv).all():
        return mask
    gx, gy = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    inside = Path(uv).contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
    return inside.reshape(camera.height, camera.width)
