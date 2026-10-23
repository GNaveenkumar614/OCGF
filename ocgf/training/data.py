"""Torch dataset and collate for the student (the arrays themselves come from ocgf.datasets.student_sample)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..datasets.student_sample import build_sample
from ..geometry.bev_grid import BEVGrid
from .targets import load_record


class StudentFrames(torch.utils.data.Dataset):
    """Frames of one KITTI-360 sequence with a BEV record (<record_dir>/<frame>.npz). seq: Kitti360Sequence."""

    def __init__(self, seq, frames, record_dir, grid: BEVGrid = BEVGrid(), plane=(0.0, 0.0, -1.73)):
        self.seq, self.frames = seq, [int(f) for f in frames]
        self.record_dir, self.grid, self.plane = Path(record_dir), grid, tuple(plane)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        record = load_record(self.record_dir / f"{f:010d}.npz")
        s = build_sample(self.seq.image(f), self.seq.points(f), self.seq.camera, record, self.grid, self.plane)
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in s.items()}
        out["image"] = out["image"].permute(2, 0, 1)                 # [3, H, W] uint8
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
