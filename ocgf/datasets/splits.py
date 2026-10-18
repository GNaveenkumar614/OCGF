"""Scene-level splits.

Paper (configs/splits/kitti360_paper_v1.json, decided 2026-10-18): split by drive.
  train 0000, 0002, 0004, 0005 · val 0006 · test 0009, 0010 · unused 0003, 0007.
  Chosen with scripts/drive_stats.py. Test holds the most pedestrians/cyclists and parked cars and lies in its own
  area; val/test frames within 50 m (GPS) of any train frame are dropped, since 0006 passes 0004/0005 on 5-13 % of its
  route.

Starter phase (one drive, kitti360_d0000_dev_v1, debugging only): distance_split below. The drive is cut into
consecutive train | val | test stretches by odometer (default 70/15/15 of the usable distance). Frames within buffer_m/2
of a cut are dropped, so neighbouring splits are at least buffer_m apart along the route. Val/test frames closer than
revisit_m (world x/y) to any train frame are dropped too, which catches route revisits.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree


def to_ranges(frames) -> list[list[int]]:
    """Sorted frame ids -> [[start, end], ...] of consecutive runs."""
    frames = np.unique(np.asarray(frames, dtype=np.int64))
    if not len(frames):
        return []
    breaks = np.flatnonzero(np.diff(frames) > 1)
    starts, ends = frames[np.r_[0, breaks + 1]], frames[np.r_[breaks, len(frames) - 1]]
    return [[int(s), int(e)] for s, e in zip(starts, ends)]


def from_ranges(ranges) -> np.ndarray:
    return np.concatenate([np.arange(s, e + 1) for s, e in ranges]).astype(np.int64) if ranges else np.zeros(0, np.int64)


def content_hash(splits: dict) -> str:
    """sha256 of the split contents, independent of key order."""
    canonical = json.dumps({k: to_ranges(v) for k, v in splits.items()}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def distance_split(index: dict, fractions=(0.70, 0.15, 0.15), names=("train", "val", "test"),
                   buffer_m: float = 100.0, revisit_m: float = 50.0) -> tuple[dict, dict]:
    """Returns ({name: frame ids}, report) from a frame index (needs frame, usable, odometer_m, x_world, y_world)."""
    use = index["usable"] & np.isfinite(index["odometer_m"])
    frames, odo = index["frame"][use], index["odometer_m"][use]
    xy = np.column_stack([index["x_world"][use], index["y_world"][use]])
    lo, hi = float(odo.min()), float(odo.max())
    cuts = lo + (hi - lo) * np.cumsum(fractions)[:-1]
    part = np.digitize(odo, cuts)
    near_cut = np.zeros(len(odo), dtype=bool)
    for c in cuts:
        near_cut |= np.abs(odo - c) < buffer_m / 2
    splits = {n: frames[(part == k) & ~near_cut] for k, n in enumerate(names)}

    train = np.isin(frames, splits[names[0]])
    tree = cKDTree(xy[train])
    revisits = {}
    for n in names[1:]:
        rows = np.isin(frames, splits[n])
        dist, _ = tree.query(xy[rows])
        revisits[n] = int((dist < revisit_m).sum())
        splits[n] = frames[rows][dist >= revisit_m]

    report = {
        "usable_frames": int(use.sum()), "usable_distance_m": round(hi - lo, 1),
        "cuts_odometer_m": [round(float(c), 1) for c in cuts], "dropped_near_cuts": int(near_cut.sum()),
        "dropped_as_revisits": revisits, "counts": {n: int(len(f)) for n, f in splits.items()},
        "odometer_m": {n: [round(float(odo[np.isin(frames, f)].min()), 1), round(float(odo[np.isin(frames, f)].max()), 1)]
                       for n, f in splits.items() if len(f)},
    }
    return splits, report
