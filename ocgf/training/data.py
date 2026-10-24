"""Torch dataset and collate for the student (the arrays themselves come from ocgf.datasets.student_sample)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..datasets.bev_inputs import BEVInputConfig, bev_inputs
from ..datasets.objects import FrameObjects, ObjectConfig
from ..datasets.student_sample import build_sample
from ..geometry.bev_grid import BEVGrid
from .targets import load_record


def _frame_objects(objects: dict | None, frame: int):
    """The frame's objects (none detected: empty), or None when the model takes no object input."""
    if objects is None:
        return None
    return objects.get(frame, FrameObjects.empty())


class StudentFrames(torch.utils.data.Dataset):
    """Frames of one KITTI-360 sequence with a BEV record (<record_dir>/<frame>.npz). seq: Kitti360Sequence.
    objects: {frame: FrameObjects} for models with object channels (O_B), else None."""

    def __init__(self, seq, frames, record_dir, grid: BEVGrid = BEVGrid(), plane=(0.0, 0.0, -1.73), objects=None,
                 obj_cfg: ObjectConfig = ObjectConfig()):
        self.seq, self.frames = seq, [int(f) for f in frames]
        self.record_dir, self.grid, self.plane = Path(record_dir), grid, tuple(plane)
        self.objects, self.obj_cfg = objects, obj_cfg

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        record = load_record(self.record_dir / f"{f:010d}.npz")
        s = build_sample(self.seq.image(f), self.seq.points(f), self.seq.camera, record, self.grid, self.plane,
                         _frame_objects(self.objects, f), self.obj_cfg)
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in s.items()}
        out["image"] = out["image"].permute(2, 0, 1)                 # [3, H, W] uint8
        out["frame"] = torch.tensor(f)
        return out


class ScanFrames(torch.utils.data.Dataset):
    """Frames for inference and evaluation: the BEV inputs are computed from the scan (and the frame's detections) on
    the fly, as at test time, so a perturbed scan or box set (E5) changes them too. No targets, no ground truth.
    objects: {frame: FrameObjects} when the model takes objects or its viewshed uses box occluders, else None."""

    def __init__(self, seq, frames, grid: BEVGrid = BEVGrid(), inputs_cfg: BEVInputConfig = BEVInputConfig(),
                 objects=None, obj_cfg: ObjectConfig = ObjectConfig(), object_inputs: bool = False):
        self.seq, self.frames, self.grid, self.cfg = seq, [int(f) for f in frames], grid, inputs_cfg
        self.objects, self.obj_cfg, self.object_inputs = objects, obj_cfg, object_inputs

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        points = self.seq.points(f)
        objs = _frame_objects(self.objects, f)
        inputs = bev_inputs(points, self.seq.camera, self.grid, self.cfg, objs)
        s = build_sample(self.seq.image(f), points, self.seq.camera, inputs, self.grid, self.cfg.ground.plane,
                         objs if self.object_inputs else None, self.obj_cfg)
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in s.items()}
        out["image"] = out["image"].permute(2, 0, 1)
        out["frame"] = torch.tensor(f)
        return out


def collate(items: list[dict]) -> dict:
    """Stack the dense arrays; concatenate the points, offsetting each sample's cell indices by its grid size."""
    n_cells = items[0]["grid_uv"].shape[0] * items[0]["grid_uv"].shape[1]
    out = {k: torch.stack([it[k] for it in items]) for k in items[0] if k not in ("pt_feats", "pt_cell")}
    out["pt_feats"] = torch.cat([it["pt_feats"] for it in items])
    out["pt_cell"] = torch.cat([it["pt_cell"] + b * n_cells for b, it in enumerate(items)])
    return out


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
