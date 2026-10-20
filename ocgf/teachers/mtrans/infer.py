"""Run a trained MTrans checkpoint on 2D boxes from the detector and write neutral object files (OCGF code).

One JSON line per 2D box, in the order of the input boxes of each frame:
  {"fid", "cls", "score_2d", "box_2d": [l, t, r, b], "n_points",
   "box": [x, y, z, l, w, h, yaw] in the LiDAR frame or null, "conf_iou_raw": float or null, "status"}
status "ok", or "too_few_points" (fewer frustum points than min_points: no 3D box; the 2D box is still listed, because
a detected object without a 3D box must never become free road downstream). conf_iou_raw is the head regressed to
the 3D IoU: a quality estimate, not a calibrated probability, and never rescaled by other objects in the batch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ...datasets.ids import parse_frame_id
from .data import ObjectDataset, ProjectedFrames, build_records, make_loader, to_device
from .model import MTrans
from .ops import to_attrdict


def load_model(checkpoint, device=None) -> tuple[MTrans, dict]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MTrans(to_attrdict(ckpt["model_config"]), ckpt["anchors"])
    model.load_state_dict(ckpt["model"])
    meta = {k: ckpt[k] for k in ("epoch", "epochs", "classes", "anchors", "data_config")}
    return model.to(device).eval(), meta


class Kitti360Frames:
    """Frame source over prepared KITTI-360 drives: {sequence: ocgf.datasets.kitti360.Kitti360Sequence}."""

    def __init__(self, sequences: dict):
        self.sequences = sequences

    def _get(self, fid):
        _, seq, frame = parse_frame_id(fid)
        return self.sequences[seq], frame

    def image(self, fid):
        seq, frame = self._get(fid)
        return seq.image(frame)

    def points(self, fid):
        seq, frame = self._get(fid)
        return seq.points(frame)[:, :4]

    def camera(self, fid):
        return self._get(fid)[0].camera


@torch.no_grad()
def infer(model: MTrans, meta: dict, frames: ProjectedFrames, objects: dict, out_file, batch_size: int = 32,
          min_points: int = 1, num_workers: int = 0, log=print) -> dict:
    """objects: {fid: objects} from data.detection_objects (2D boxes in our classes). Writes out_file (JSON lines)."""
    classes, d = tuple(meta["classes"]), meta["data_config"]
    device = next(model.parameters()).device
    records = build_records(frames, objects, classes, keep_all=True, log=log)
    runnable = [i for i, r in enumerate(records) if r["n_points"] >= min_points]
    dataset = ObjectDataset([records[i] for i in runnable], frames, meta["anchors"], d["out_img_size"],
                            d["out_cloud_size"])
    boxes = np.full((len(records), 7), np.nan)
    conf = np.full(len(records), np.nan)
    done = 0
    for data in make_loader(dataset, batch_size, num_workers=num_workers):
        b, c = model.predict(to_device(data, device))
        idx = [runnable[k] for k in data.record_index]
        boxes[idx], conf[idx] = b.cpu().numpy(), c.cpu().numpy()
        done += len(idx)
        if log and done % (batch_size * 50) < batch_size:
            log(f"  {done}/{len(runnable)} objects")

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for r, box, cf in zip(records, boxes, conf):
            ok = bool(np.isfinite(cf))
            f.write(json.dumps({"fid": r["fid"], "cls": r["cls"], "score_2d": round(r["score"], 4),
                                "box_2d": [round(v, 2) for v in r["box_2d"]], "n_points": r["n_points"],
                                "box": [round(float(v), 4) for v in box] if ok else None,
                                "conf_iou_raw": round(float(cf), 4) if ok else None,
                                "status": "ok" if ok else "too_few_points"}) + "\n")
    summary = {"objects": len(records), "with_3d_box": len(runnable), "too_few_points": len(records) - len(runnable),
               "frames": len(objects)}
    log(f"wrote {out_file}: {summary}")
    return summary


def read_objects(path) -> dict[str, list]:
    """{fid: rows} from an object file written by infer()."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out.setdefault(row["fid"], []).append(row)
    return out
