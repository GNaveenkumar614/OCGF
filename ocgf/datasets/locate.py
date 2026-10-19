"""Find datasets inside a Kaggle input folder (/kaggle/input), wherever Kaggle mounts them.

- KITTI 3D Object: the folders under 'training' (image_2, velodyne, calib, label_2).
- A prepared KITTI-360 drive: the output of data-prep notebook 01 (scans/<seq>, frame_index/<seq>.csv, image shards for
  every drive but 0000) combined with the raw inputs it still needs (calibration, poses, starter images for drive 0000,
  boxes, semantics), rediscovered here because mount paths may differ from the run that prepared it.
"""
from __future__ import annotations

import os
from pathlib import Path

from .dataset_map import discover_kitti360

KITTI_FOLDERS = ("image_2", "velodyne", "calib", "label_2")
_SKIP = {"image_2", "image_3", "velodyne", "velodyne_reduced", "calib", "label_2", "planes", "gt_database",
         "data_rect", "data_raw", "semantic", "semantic_rgb", "instance", "confidence"}


def find_kitti_object(root) -> dict[str, Path]:
    """{folder: path} for KITTI 3D Object 'training/<folder>' found under root (several datasets may share the work)."""
    found = {}
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        here = Path(dirpath)
        if here.name == "training":
            for name in KITTI_FOLDERS:
                if name in dirnames:
                    found.setdefault(name, here / name)
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP)
    return found


def find_prepared(root, sequence: str) -> Path | None:
    """The data-prep output folder for a sequence: it holds frame_index/<seq>.csv and scans/<seq>/."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        here = Path(dirpath)
        if here.name == "frame_index" and f"{sequence}.csv" in filenames and (here.parent / "scans" / sequence).is_dir():
            return here.parent
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP and d != "scans" and d != "images")
    return None


def prepared_drive(root, sequence: str) -> dict[str, Path]:
    """Dataset-map components of a prepared drive: the prepared folders plus the rediscovered raw inputs."""
    prepared = find_prepared(root, sequence)
    if prepared is None:
        raise FileNotFoundError(f"no prepared output for {sequence} under {root}: attach its kitti360-d<drive>-supplement")
    components = {k: v for k, v in discover_kitti360(root, sequence).items() if k != "semantics2d_zip"}
    components.update(scans=prepared / "scans" / sequence, frame_index=prepared / "frame_index" / f"{sequence}.csv")
    if (prepared / "images" / sequence).is_dir():
        components["image_shards"] = prepared / "images" / sequence
    for name in ("velodyne_timestamps", "image_timestamps"):
        if (prepared / name / f"{sequence}.txt").exists():
            components[name] = prepared / name / f"{sequence}.txt"
    if "images" not in components and "image_shards" not in components:
        raise FileNotFoundError(f"no images for {sequence}: neither starter images nor image shards were found")
    return components
