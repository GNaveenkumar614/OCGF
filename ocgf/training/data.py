"""Torch dataset and collate for the student (the arrays themselves come from ocgf.datasets.student_sample)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..datasets.bev_inputs import BEVInputConfig, bev_inputs
from ..datasets.objects import FrameObjects, ObjectConfig
from ..datasets.student_sample import build_sample
from ..evaluation.perturb import perturb_image, perturb_objects, perturb_points
from ..geometry.bev_grid import BEVGrid
from ..geometry.warp import warp_grid
from .targets import load_record

# what the model reads from a sample; a paired earlier frame (temporal consistency) carries these under "prev/"
MODEL_KEYS = ("image", "grid_uv", "M_C", "grid_uv_flat", "fov_flat", "pt_feats", "pt_cell", "pt_uv", "Q_geo",
              "valid_L", "O_B")


def previous_frames(seq, frames, available, min_m: float = 2.0, max_m: float = 7.5) -> dict[int, int]:
    """{frame: the most recent earlier frame among `available` at least min_m and at most max_m metres back along the
    drive} for the temporal-consistency term (the previous 5 m sample, when frames are 5 m apart). Frames without
    one are left out."""
    idx = seq.index
    odo = dict(zip(idx["frame"].tolist(), idx["odometer_m"].tolist()))
    avail = np.array(sorted(int(f) for f in available if int(f) in odo))
    out = {}
    for f in frames:
        f = int(f)
        if f not in odo:
            continue
        back = np.array([odo[f] - odo[int(g)] for g in avail[avail < f]])
        ok = np.flatnonzero(back >= min_m)
        if len(ok) and back[ok[-1]] <= max_m:
            out[f] = int(avail[avail < f][ok[-1]])
    return out


def _to_torch(s: dict) -> dict:
    out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in s.items()}
    out["image"] = out["image"].permute(2, 0, 1)                     # [3, H, W] uint8
    return out


def _frame_objects(objects: dict | None, frame: int):
    """The frame's objects (none detected: empty), or None when the model takes no object input."""
    if objects is None:
        return None
    return objects.get(frame, FrameObjects.empty())


class StudentFrames(torch.utils.data.Dataset):
    """Frames of one KITTI-360 sequence with a BEV record (<record_dir>/<frame>.npz). seq: Kitti360Sequence.
    objects: {frame: FrameObjects} for models with object channels (O_B), else None. prev: {frame: earlier frame}
    (previous_frames) for the temporal-consistency term: each sample then also carries the earlier frame's model inputs
    and unknown target under "prev/", the warp of its grid onto this frame's ("warp_grid", "warp_inside") and
    "has_prev" (False for frames without a pair, which then repeat their own inputs and drop out of the term)."""

    def __init__(self, seq, frames, record_dir, grid: BEVGrid = BEVGrid(), plane=(0.0, 0.0, -1.73), objects=None,
                 obj_cfg: ObjectConfig = ObjectConfig(), prev: dict | None = None):
        self.seq, self.frames = seq, [int(f) for f in frames]
        self.record_dir, self.grid, self.plane = Path(record_dir), grid, tuple(plane)
        self.objects, self.obj_cfg, self.prev = objects, obj_cfg, prev

    def __len__(self):
        return len(self.frames)

    def _sample(self, f: int) -> dict:
        record = load_record(self.record_dir / f"{f:010d}.npz")
        return _to_torch(build_sample(self.seq.image(f), self.seq.points(f), self.seq.camera, record, self.grid,
                                      self.plane, _frame_objects(self.objects, f), self.obj_cfg))

    def __getitem__(self, i):
        f = self.frames[i]
        out = self._sample(f)
        out["frame"] = torch.tensor(f)
        if self.prev is not None:
            g = self.prev.get(f)
            prev = self._sample(g) if g is not None else out
            for k in MODEL_KEYS:
                if k in prev:
                    out[f"prev/{k}"] = prev[k]
            out["prev/unk_t"] = prev["unk_t"]
            if g is not None:
                coords, inside = warp_grid(self.seq.T_world_lidar(f), self.seq.T_world_lidar(g), self.grid)
            else:
                coords, inside = warp_grid(np.eye(4), np.eye(4), self.grid)
            out["warp_grid"], out["warp_inside"] = torch.from_numpy(coords), torch.from_numpy(inside)
            out["has_prev"] = torch.tensor(g is not None)
        return out


class ScanFrames(torch.utils.data.Dataset):
    """Frames for inference and evaluation: the BEV inputs are computed from the scan (and the frame's detections) on
    the fly, as at test time, so a perturbed scan or box set (E5) changes them too. No targets, no ground truth.
    objects: {frame: FrameObjects} when the model takes objects or its viewshed uses box occluders, else None."""

    def __init__(self, seq, frames, grid: BEVGrid = BEVGrid(), inputs_cfg: BEVInputConfig = BEVInputConfig(),
                 objects=None, obj_cfg: ObjectConfig = ObjectConfig(), object_inputs: bool = False,
                 perturb: dict | None = None, seed: int = 0):
        self.seq, self.frames, self.grid, self.cfg = seq, [int(f) for f in frames], grid, inputs_cfg
        self.objects, self.obj_cfg, self.object_inputs = objects, obj_cfg, object_inputs
        self.perturb, self.seed = perturb or {}, seed

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        points = perturb_points(self.seq.points(f), self.perturb, f, self.seed)
        image = perturb_image(self.seq.image(f), self.perturb, f, self.seed)
        objs = perturb_objects(_frame_objects(self.objects, f), self.perturb, f, self.seed)
        inputs = bev_inputs(points, self.seq.camera, self.grid, self.cfg, objs)
        s = build_sample(image, points, self.seq.camera, inputs, self.grid, self.cfg.ground.plane,
                         objs if self.object_inputs else None, self.obj_cfg)
        out = _to_torch(s)
        out["frame"] = torch.tensor(f)
        return out


def collate(items: list[dict]) -> dict:
    """Stack the dense arrays; concatenate the points (also a paired earlier frame's, under "prev/"), offsetting each
    sample's cell indices by its grid size."""
    n_cells = items[0]["grid_uv"].shape[0] * items[0]["grid_uv"].shape[1]
    points = {k for k in items[0] if k.split("/")[-1].startswith("pt_")}          # per-point arrays
    out = {k: torch.stack([it[k] for it in items]) for k in items[0] if k not in points}
    for k in points:
        if k.endswith("pt_cell"):
            out[k] = torch.cat([it[k] + b * n_cells for b, it in enumerate(items)])
        else:
            out[k] = torch.cat([it[k] for it in items])
    return out


def previous_inputs(batch: dict) -> dict:
    """The paired earlier frames' model inputs from a batch (keys without the "prev/" prefix)."""
    return {k[5:]: v for k, v in batch.items() if k.startswith("prev/")}


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
