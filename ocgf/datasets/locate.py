"""Find datasets inside a Kaggle input folder (/kaggle/input), wherever Kaggle mounts them.

- KITTI 3D Object: the folders under 'training' (image_2, velodyne, calib, label_2).
- A prepared KITTI-360 drive: the output of data-prep notebook 01 (scans/<seq>, frame_index/<seq>.csv, image shards for
  every drive but 0000) combined with the raw inputs it still needs (calibration, poses, starter images for drive 0000,
  boxes, semantics), rediscovered here because mount paths may differ from the run that prepared it.
"""
from __future__ import annotations

import json
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


def find_all_prepared(root, sequence: str) -> list[Path]:
    """Every data-prep output folder for a sequence under root (each holds frame_index/<seq>.csv and scans/<seq>/)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        here = Path(dirpath)
        if here.name == "frame_index" and f"{sequence}.csv" in filenames and (here.parent / "scans" / sequence).is_dir():
            found.append(here.parent)
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP and d != "scans" and d != "images")
    return found


def prepared_info(prepared: Path, sequence: str) -> dict:
    """What a data-prep output holds: its settings (a trial has max_frames) and how many frames it stores."""
    from . import frame_index as fi
    idx = fi.read_csv(Path(prepared) / "frame_index" / f"{sequence}.csv")
    kept = idx["usable"] & idx.get("kept", True)
    settings = {}
    manifest = Path(prepared) / "manifest.json"
    if manifest.exists():
        try:
            settings = json.loads(manifest.read_text()).get("settings", {})
        except ValueError:
            pass
    return {"path": str(prepared), "max_frames": settings.get("max_frames"), "keep_every_m": settings.get("keep_every_m"),
            "usable_kept": int(kept.sum()), "sampled_5m": int((kept & idx["sample_5m"]).sum())}


def find_prepared(root, sequence: str) -> Path | None:
    """The data-prep output for a sequence. If several are attached (a trial and the full run), the one storing the
    most frames: picking the first found once trained a model on a 200-frame trial."""
    found = find_all_prepared(root, sequence)
    if len(found) <= 1:
        return found[0] if found else None
    return max(found, key=lambda p: prepared_info(p, sequence)["usable_kept"])


def describe_prepared(root, sequence: str) -> str:
    """One line for logs: which prepared output is used, its frame counts, and a warning for trials or duplicates."""
    found = find_all_prepared(root, sequence)
    if not found:
        return f"{sequence}: no prepared output found"
    use = find_prepared(root, sequence)
    info = prepared_info(use, sequence)
    line = (f"{sequence}: prepared output {info['path']} ({info['usable_kept']} usable stored frames, "
            f"{info['sampled_5m']} sampled every 5 m)")
    lines = [line]
    if info["max_frames"]:
        lines.append(f"  WARNING: a TRIAL data-prep run (max_frames = {info['max_frames']}), not the full drive")
    for other in found:
        if other != use:
            o = prepared_info(other, sequence)
            lines.append(f"  also attached, not used: {o['path']} ({o['usable_kept']} frames)")
    return "\n".join(lines)


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


def _pruned_walk(root):
    """os.walk over root that skips the large per-frame folders (images, scans, semantics, shards)."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP and d not in ("scans", "images", "qa", "preds")
                             and not d.startswith("shard_"))
        yield Path(dirpath), dirnames, filenames


def find_outputs(root, sequence: str, key: str, value: str) -> list[Path]:
    """Folders named <sequence> under root whose manifest.json has manifest[key] == value, e.g. the road teacher's
    labels (key 'teacher', value 'road') or the evaluation ground truth ('kind', 'eval_gt'). /kaggle/input is slow per
    file, so the walk never enters per-frame folders (a plain rglob over the inputs took minutes)."""
    found = []
    for here, dirnames, filenames in _pruned_walk(root):
        if here.name == sequence and "manifest.json" in filenames:
            try:
                if json.loads((here / "manifest.json").read_text()).get(key) == value:
                    found.append(here)
            except (OSError, ValueError):
                pass
            dirnames[:] = []
    return found


def find_runs(root, name: str) -> list[Path]:
    """Training-run folders named `name` under root (they hold ckpt/last.pt), e.g. a previous session's output."""
    found = []
    for here, dirnames, _ in _pruned_walk(root):
        if here.name == name and (here / "ckpt" / "last.pt").exists():
            found.append(here)
            dirnames[:] = []
    return found
