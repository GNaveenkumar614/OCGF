"""Locate the parts of a KITTI-360 sequence wherever they are mounted, and save them as a dataset map.

Loaders take a dataset map (one root per component) and never assume the official folder tree.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REQUIRED = ("images", "image_timestamps", "calibration", "poses", "semantics3d", "boxes3d")
OPTIONAL = ("semantics2d", "semantics2d_zip")
# folders with one file per frame: recognised from their parent, never listed
_LEAF_DIRS = {"data_rect", "data_raw", "semantic", "semantic_rgb", "instance", "confidence"}


def discover_kitti360(root, sequence: str) -> dict[str, Path]:
    """{component: path} for every component found under root. Missing components are simply absent."""
    found: dict[str, Path] = {}
    boxes: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        here = Path(dirpath)
        dirs, files = set(dirnames), set(filenames)
        if here.name == "image_00" and here.parent.name == sequence:
            if "data_rect" in dirs:
                found.setdefault("images", here / "data_rect")
                if "timestamps.txt" in files:
                    found.setdefault("image_timestamps", here / "timestamps.txt")
            if "semantic" in dirs:
                found.setdefault("semantics2d", here)
        if {"perspective.txt", "calib_cam_to_velo.txt"} <= files:
            found.setdefault("calibration", here)
        if here.name == sequence and "poses.txt" in files:
            found.setdefault("poses", here)
        if here.name == sequence and "static" in dirs:
            found.setdefault("semantics3d", here)
        if f"{sequence}.xml" in files:
            boxes.append(here / f"{sequence}.xml")
        if "data_2d_semantics.zip" in files:
            found.setdefault("semantics2d_zip", here / "data_2d_semantics.zip")
        dirnames[:] = sorted(d for d in dirnames if d not in _LEAF_DIRS)
    # train/ holds the object boxes with semanticId/instanceId; train_full/ adds annotation primitives
    boxes = [b for b in boxes if b.parent.name != "train_full"]
    if boxes:
        found["boxes3d"] = sorted(boxes, key=lambda b: (b.parent.name != "train", str(b)))[0]
    return found


def missing_required(found: dict) -> list[str]:
    return [c for c in REQUIRED if c not in found]


def write_dataset_map(path, sequence: str, components: dict) -> None:
    """Paths inside the map's own folder are stored relative to it, so a copied folder stays valid."""
    path = Path(path)
    base = path.parent.resolve()
    entries = {}
    for name, p in sorted(components.items()):
        p = Path(p).resolve()
        try:
            entries[name] = p.relative_to(base).as_posix()
        except ValueError:
            entries[name] = p.as_posix()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"sequence": sequence, "components": entries}, sort_keys=False))


def read_dataset_map(path) -> tuple[str, dict[str, Path]]:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    base = path.parent
    return data["sequence"], {k: (base / v if not Path(v).is_absolute() else Path(v)) for k, v in data["components"].items()}
