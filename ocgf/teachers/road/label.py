"""The road teacher for one frame, its output files, and the audit against independent ground truth (OCGF code).

label_frame runs the three steps (trajectory, LiDAR label, camera label) and combines them as the original's
post-processing does: the mean of the LiDAR and camera labels where both exist, the one that exists elsewhere. The
original then binarises with a dense CRF; we keep the soft label (the student trains on soft targets) and never use its
crf.py (CC BY-NC 4.0).

Output per frame (npz): label float16 [H, W] (NaN unknown), point_scores float16 [N] aligned with the stored scan,
path_mask bool [H, W], wheels float32 [K, 2, 3] (left / right wheel points, LiDAR frame), and with save_parts also
lidar_label and camera_label. status: ok, no_path (shorter future path), no_trajectory (fewer than 2 wheel pairs),
rings (scan without 64 recovered rings).
These are pseudo-labels for training only. They are audited against KITTI-360's 2D semantics, never used to evaluate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...datasets.kitti360_labels import ROAD
from .camera_label import camera_label
from .config import RoadConfig
from .lidar_label import image_label, nanmean, point_scores
from .trajectory import (RingScan, find_trajectory, future_path, occlusion_filter, trajectory_polygon,
                         wheel_points)

N_RINGS = 64


def label_frame(points: np.ndarray, image: np.ndarray, camera, path: np.ndarray, cfg: RoadConfig,
                features_fn=None, prototype=None) -> tuple[dict, object]:
    """points [N, 5] (x, y, z, intensity, ring) of the current scan; path [K, 3] from future_path. Returns
    (result, prototype for the next frame)."""
    H, W = camera.height, camera.width
    nan_img = np.full((H, W), np.nan)
    result = {"label": nan_img, "lidar_label": nan_img, "camera_label": nan_img,
              "point_scores": np.full(len(points), np.nan), "path_mask": np.zeros((H, W), bool),
              "wheels": np.zeros((0, 2, 3), np.float32), "status": "ok", "n_wheels": 0}
    if not len(path):
        result["status"] = "no_path"
        return result, prototype
    if len(points) == 0 or int(points[:, 4].max()) + 1 != N_RINGS:
        result["status"] = "rings"
        return result, prototype

    scan = RingScan(points, cfg)
    wheels = wheel_points(scan, find_trajectory(scan, path, cfg), cfg)
    wheels = occlusion_filter(scan, wheels, camera, cfg)
    result["n_wheels"] = int(len(wheels))
    if len(wheels) < 2:
        result["status"] = "no_trajectory"                  # the original writes an all-unknown LiDAR label
    else:
        scores = point_scores(scan, wheels, cfg)
        result["point_scores"] = scores
        result["lidar_label"] = image_label(points, scores, camera, cfg)
        result["path_mask"] = trajectory_polygon(scan, wheels, camera, cfg)
        result["wheels"] = np.stack([scan.points[wheels[:, 2], :3], scan.points[wheels[:, 3], :3]], axis=1).astype(np.float32)

    if features_fn is not None:
        result["camera_label"], prototype = camera_label(features_fn(image), result["path_mask"], prototype, cfg, (H, W))
    if result["status"] == "no_trajectory":
        result["label"] = nan_img                           # no path evidence in this frame: unknown, not camera-only
    else:
        result["label"] = nanmean(result["lidar_label"], result["camera_label"], require_both=False)
    return result, prototype


def save_result(path, result: dict, save_parts: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = {"label": result["label"].astype(np.float16), "point_scores": result["point_scores"].astype(np.float16),
           "path_mask": result["path_mask"], "wheels": result["wheels"], "status": np.array(result["status"])}
    if save_parts:
        out["lidar_label"] = result["lidar_label"].astype(np.float16)
        out["camera_label"] = result["camera_label"].astype(np.float16)
    np.savez_compressed(path, **out)


def load_result(path) -> dict:
    with np.load(path) as f:
        return {k: (str(f[k]) if k == "status" else f[k]) for k in f.files}


def audit_counts(label: np.ndarray, semantic: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """(TP, FP, FN) of label > threshold against KITTI-360 2D semantics (road = id 7), over the whole image, as the
    original's evaluate.py counts them (unknown counts as not road)."""
    pred = np.nan_to_num(label, nan=0.0) > threshold
    gt = semantic == ROAD
    return np.array([np.sum(pred & gt), np.sum(pred & ~gt), np.sum(~pred & gt)], dtype=np.int64)


def audit_summary(counts: np.ndarray) -> dict:
    tp, fp, fn = (int(v) for v in counts)
    div = lambda a, b: round(a / b, 4) if b else None
    return {"iou": div(tp, tp + fp + fn), "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn), "pixels": {"tp": tp, "fp": fp, "fn": fn}}
