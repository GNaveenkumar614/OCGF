"""Teacher quality on KITTI-360 (gate G4), against the KITTI-360 3D boxes (ground truth, evaluation only). numpy only.

Visible ground-truth objects: the boxes of our four classes whose centre lies on the BEV grid, whose projection covers
at least min_area pixels of the image, and which hold at least min_points LiDAR points (a proxy for "not hidden").
A cyclist is a rider with the two-wheeler it rides (one enclosing box); two-wheelers without a rider are left out,
as the detector leaves them out.

Detector recall: a visible object counts as found when a detection at or above the score threshold overlaps its
projected 2D box with IoU >= 0.5 (any class, and of its class). Per class and distance band, at the inference floor
(0.1) and at 0.5. Every miss is a potential false-free cell downstream.
MTrans box quality: each visible object is matched one-to-one to the detection row (object file) whose 2D box overlaps
it most (IoU >= 0.5); for matched rows with a 3D box: 3D and BEV IoU against the ground-truth box, recall at 3D IoU
0.5 / 0.7, and how well conf_iou_raw tracks the 3D IoU (mean |conf - IoU| and their correlation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..datasets.boxes3d import CLASSES as GT_CLASSES
from ..datasets.objects import OBJECT_CLASSES, FrameObjects
from ..geometry.bev_grid import BEVGrid
from ..geometry.box_iou import bev_iou, enclosing_box, iou_3d
from ..geometry.boxes import box_corners, points_in_box
from ..geometry.projection import project

BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 55.0))
MATCH_IOU_2D = 0.5


@dataclass
class GTObject:
    cls: str
    box: np.ndarray                    # [7]; a cyclist's encloses rider and two-wheeler
    box_2d: np.ndarray                 # [4] l, t, r, b, clipped to the image
    distance: float
    n_points: int
    parts: list = field(default_factory=list)


def visible_objects(fb, camera, points, grid: BEVGrid = BEVGrid(), min_points: int = 5,
                    min_area: float = 20.0) -> list[GTObject]:
    """The visible ground-truth objects of our classes in one frame (fb: FrameBoxes in the LiDAR frame)."""
    out = []
    for k in range(len(fb)):
        name = GT_CLASSES[fb.classes[k]]
        if name not in OBJECT_CLASSES:
            continue
        parts = [fb.params[k]]
        if name == "cyclist" and fb.rider_of[k] >= 0:
            parts = [fb.params[fb.rider_of[k]], fb.params[k]]          # two-wheeler first: its heading
        box = enclosing_box(parts) if len(parts) > 1 else np.asarray(parts[0], np.float64)
        if not grid.inside(box[0], box[1]):
            continue
        inside = np.zeros(len(points), bool)
        for part in parts:                                   # the union: a rider overlaps its two-wheeler
            inside |= points_in_box(points, part)
        n = int(inside.sum())
        if n < min_points:
            continue
        uv, _, _ = project(box_corners(np.array(parts)).reshape(-1, 3), camera)
        uv = uv[np.isfinite(uv).all(axis=1)]
        if not len(uv):
            continue
        l, t = np.clip(uv.min(axis=0), 0, [camera.width, camera.height])
        r, b = np.clip(uv.max(axis=0), 0, [camera.width, camera.height])
        if (r - l) * (b - t) < min_area:
            continue
        out.append(GTObject(name, box, np.array([l, t, r, b]), float(np.hypot(box[0], box[1])), n, parts))
    return out


def read_detection_file(path) -> dict[str, tuple]:
    """{fid: (classes [K], scores [K], boxes [K, 4])} from a detector file ('fid cls score l t r b' lines)."""
    rows: dict[str, list] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 7:
            rows.setdefault(parts[0], []).append((parts[1], float(parts[2]), [float(v) for v in parts[3:]]))
    return {fid: (np.array([r[0] for r in rs]), np.array([r[1] for r in rs]), np.array([r[2] for r in rs]))
            for fid, rs in rows.items()}


def iou_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box a [4] with boxes b [K, 4] (l, t, r, b)."""
    b = np.asarray(b, np.float64).reshape(-1, 4)
    iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    union = area(a) + area(b) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def band_of(distance: float) -> str:
    for lo, hi in BANDS:
        if lo <= distance < hi:
            return f"{lo:g}-{hi:g}m"
    return "far"


def _key(cls, distance):
    return cls, band_of(distance)


class DetectorRecall:
    def __init__(self, thresholds=(0.1, 0.5)):
        self.thresholds = tuple(thresholds)
        self.n: dict = {}
        self.hit: dict = {}

    def add(self, gts: list[GTObject], dets) -> None:
        """dets: (classes, scores, boxes) of the frame, or None (no detection)."""
        for g in gts:
            k = _key(g.cls, g.distance)
            self.n[k] = self.n.get(k, 0) + 1
            for t in self.thresholds:
                any_hit = cls_hit = False
                if dets is not None and len(dets[1]):
                    keep = dets[1] >= t
                    ious = iou_2d(g.box_2d, dets[2][keep])
                    any_hit = bool((ious >= MATCH_IOU_2D).any())
                    cls_hit = bool(((ious >= MATCH_IOU_2D) & (dets[0][keep] == g.cls)).any())
                for name, hit in (("any", any_hit), ("class", cls_hit)):
                    self.hit[(k, t, name)] = self.hit.get((k, t, name), 0) + int(hit)

    def result(self) -> dict:
        out = {}
        for cls in OBJECT_CLASSES:
            rows = {}
            for band in [band_of((lo + hi) / 2) for lo, hi in BANDS] + ["all"]:
                keys = [k for k in self.n if k[0] == cls and (band == "all" or k[1] == band)]
                n = sum(self.n[k] for k in keys)
                row = {"objects": n}
                for t in self.thresholds:
                    for name in ("any", "class"):
                        hits = sum(self.hit.get((k, t, name), 0) for k in keys)
                        row[f"recall_{name}@{t:g}"] = round(hits / n, 4) if n else None
                rows[band] = row
            out[cls] = rows
        return out


class BoxQuality:
    def __init__(self):
        self.rows: list[tuple] = []        # (cls, band, matched, has_box, iou3d, iou_bev, conf)

    def add(self, gts: list[GTObject], objs: FrameObjects | None) -> None:
        if objs is None or not len(objs):
            self.rows += [(g.cls, band_of(g.distance), False, False, np.nan, np.nan, np.nan) for g in gts]
            return
        pairs = sorted(((float(iou_2d(g.box_2d, objs.box_2d[j:j + 1])[0]), i, j) for i, g in enumerate(gts)
                        for j in range(len(objs))), reverse=True)
        used_g, used_o, match = set(), set(), {}
        for iou, i, j in pairs:
            if iou < MATCH_IOU_2D:
                break
            if i not in used_g and j not in used_o:
                used_g.add(i), used_o.add(j)
                match[i] = j
        for i, g in enumerate(gts):
            j = match.get(i)
            if j is None:
                self.rows.append((g.cls, band_of(g.distance), False, False, np.nan, np.nan, np.nan))
            elif not objs.has_box[j]:
                self.rows.append((g.cls, band_of(g.distance), True, False, np.nan, np.nan, np.nan))
            else:
                b = objs.boxes[j]
                self.rows.append((g.cls, band_of(g.distance), True, True, iou_3d(b, g.box), bev_iou(b, g.box),
                                  float(objs.conf[j])))

    def result(self) -> dict:
        out = {}
        for cls in OBJECT_CLASSES:
            rows = {}
            for band in [band_of((lo + hi) / 2) for lo, hi in BANDS] + ["all"]:
                r = [x for x in self.rows if x[0] == cls and (band == "all" or x[1] == band)]
                boxed = [x for x in r if x[3]]
                iou = np.array([x[4] for x in boxed])
                conf = np.array([x[6] for x in boxed])
                rnd = lambda v: round(float(v), 4) if np.isfinite(v) else None
                rows[band] = {
                    "objects": len(r), "matched": sum(x[2] for x in r), "with_3d_box": len(boxed),
                    "mean_iou3d": rnd(iou.mean()) if len(iou) else None,
                    "mean_iou_bev": rnd(np.mean([x[5] for x in boxed])) if boxed else None,
                    "recall3d@0.5": round(float((iou >= 0.5).sum() / len(r)), 4) if r else None,
                    "recall3d@0.7": round(float((iou >= 0.7).sum() / len(r)), 4) if r else None,
                    "conf_abs_err": rnd(np.abs(conf - iou).mean()) if len(iou) else None,
                    "conf_iou_corr": rnd(np.corrcoef(conf, iou)[0, 1]) if len(iou) > 2 and iou.std() > 0 and conf.std() > 0 else None}
            out[cls] = rows
        return out
