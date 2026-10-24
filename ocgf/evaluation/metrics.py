"""Metrics against the evaluation ground truth (numpy only, deterministic). Never against pseudo-labels.

Prediction per frame: the student's probabilities p_road, p_occ, p_unk [nx, ny]. Predicted free road:
    D_free = (p_road >= t) & (p_occ < t) & (p_unk < t),        confidence p_free = p_road (1 - p_occ) (1 - p_unk).
Ground truth (ocgf.evaluation.gt_builder layers):
    free    = free_space(): drivable, unoccupied, no obstacle, labelled, observed by the LiDAR or the camera
    unknown = not labelled, or observed by neither sensor
    hazard  = occupied (any annotated box) | obstacle | unknown
False-free rate, both versions (the headline is chosen before the first E1 run):
    ffr_pred    |D_free & hazard| / |D_free|     when the model says free, how often is it wrong
    ffr_hazard  |D_free & hazard| / |hazard|     how much of the hazard the model calls free
Free road: IoU, precision, recall of D_free against free (and of the road head alone, p_road >= t). Occupancy: p_occ
against occupied | obstacle, on cells with ground truth. Unknown: p_unk against unknown. Per class (box classes of
the ground truth): cells, occupancy recall, share called free. Calibration of p_free against free: ECE and the
reliability bins; risk-coverage: ffr_pred of the cells above each confidence level against the share of the grid they
cover.

Regions: the whole grid, distance bands from the LiDAR, and the near-object band: cells within near_m of a box of
the four object classes (car, large vehicle, pedestrian, cyclist), the box cells included. Buildings, walls and
vegetation are left out of the band: with them, 2 m covers about 80 % of the grid (dev sample) and says nothing about
objects (H1). Counts are summed over frames (micro averages); per-frame values go
to the caller for dispersion and failure cases.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..datasets.kitti360_labels import ROAD
from ..geometry.bev_grid import BEVGrid
from .gt_builder import OCCUPIED_CLASSES, free_space


@dataclass(frozen=True)
class MetricConfig:
    threshold: float = 0.5
    near_m: float = 2.0
    bands: tuple = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 55.0))
    drivable: tuple = (ROAD,)            # parking as drivable: still open
    ece_bins: int = 15
    rc_bins: int = 100

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bands"], d["drivable"] = [list(b) for b in self.bands], list(self.drivable)
        return d


OBJECT_CLASSES = (1, 2, 3, 4)                # occupied codes of car, large vehicle, pedestrian, cyclist


def gt_masks(layers: dict, drivable=(ROAD,)) -> dict:
    observed = layers["observed_lidar"] | layers["observed_cam"]
    unknown = ~layers["labelled"] | ~observed
    occupied = (layers["occupied"] > 0) | layers["obstacle"]
    return {"free": free_space(layers, drivable), "unknown": unknown, "occupied": occupied,
            "objects": np.isin(layers["occupied"], OBJECT_CLASSES),
            "hazard": occupied | unknown, "has_gt": layers["labelled"] & (observed | occupied)}


def regions(grid: BEVGrid, objects: np.ndarray, cfg: MetricConfig) -> dict:
    xs, ys = grid.center_grid()
    dist = np.hypot(xs, ys)
    out = {"all": np.ones(grid.shape, bool)}
    for lo, hi in cfg.bands:
        out[f"d{lo:g}-{hi:g}m"] = (dist >= lo) & (dist < hi)
    near = distance_transform_edt(~objects, sampling=grid.res) <= cfg.near_m if objects.any() else np.zeros(grid.shape, bool)
    out[f"near_object_{cfg.near_m:g}m"] = near
    return out


def _ratio(a, b):
    return round(float(a) / float(b), 5) if b else None


def _iou(tp, fp, fn):
    return {"iou": _ratio(tp, tp + fp + fn), "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn)}


class Evaluator:
    """Add frames one by one (probabilities + ground-truth layers), then read result()."""

    COUNTS = ("cells", "pred_free", "hazard", "false_free", "free_tp", "free_fp", "free_fn", "road_tp", "road_fp",
              "road_fn", "occ_tp", "occ_fp", "occ_fn", "unk_tp", "unk_fp", "unk_fn")

    def __init__(self, grid: BEVGrid = BEVGrid(), cfg: MetricConfig = MetricConfig()):
        self.grid, self.cfg = grid, cfg
        self.counts: dict[str, dict[str, int]] = {}
        self.classes = {name: {"cells": 0, "occ_hit": 0, "called_free": 0} for name in OCCUPIED_CLASSES[1:]}
        self.cal = np.zeros((3, cfg.ece_bins))                    # count, sum of confidence, sum of free
        self.rc = np.zeros((2, cfg.rc_bins))                      # cells per confidence bin: all, hazard
        self.frames = 0

    def add(self, probs: dict, layers: dict) -> dict:
        t = self.cfg.threshold
        g = gt_masks(layers, self.cfg.drivable)
        pred_free = (probs["road"] >= t) & (probs["occ"] < t) & (probs["unk"] < t)
        road, occ, unk = probs["road"] >= t, probs["occ"] >= t, probs["unk"] >= t
        frame = {}
        for name, region in regions(self.grid, g["objects"], self.cfg).items():
            r, h = region, g["has_gt"] & region
            c = {"cells": r.sum(), "pred_free": (pred_free & r).sum(), "hazard": (g["hazard"] & r).sum(),
                 "false_free": (pred_free & g["hazard"] & r).sum(),
                 "free_tp": (pred_free & g["free"] & r).sum(), "free_fp": (pred_free & ~g["free"] & r).sum(),
                 "free_fn": (~pred_free & g["free"] & r).sum(),
                 "road_tp": (road & g["free"] & r).sum(), "road_fp": (road & ~g["free"] & r).sum(),
                 "road_fn": (~road & g["free"] & r).sum(),
                 "occ_tp": (occ & g["occupied"] & h).sum(), "occ_fp": (occ & ~g["occupied"] & h).sum(),
                 "occ_fn": (~occ & g["occupied"] & h).sum(),
                 "unk_tp": (unk & g["unknown"] & r).sum(), "unk_fp": (unk & ~g["unknown"] & r).sum(),
                 "unk_fn": (~unk & g["unknown"] & r).sum()}
            acc = self.counts.setdefault(name, dict.fromkeys(self.COUNTS, 0))
            for k, v in c.items():
                acc[k] += int(v)
            frame[name] = {"ffr_pred": _ratio(c["false_free"], c["pred_free"]),
                           "ffr_hazard": _ratio(c["false_free"], c["hazard"]),
                           "free_iou": _ratio(c["free_tp"], c["free_tp"] + c["free_fp"] + c["free_fn"])}
        for k, name in enumerate(OCCUPIED_CLASSES[1:], start=1):
            cells = layers["occupied"] == k
            self.classes[name]["cells"] += int(cells.sum())
            self.classes[name]["occ_hit"] += int((occ & cells).sum())
            self.classes[name]["called_free"] += int((pred_free & cells).sum())

        conf = probs["road"] * (1 - probs["occ"]) * (1 - probs["unk"])
        b = np.clip((conf * self.cfg.ece_bins).astype(int), 0, self.cfg.ece_bins - 1).ravel()
        self.cal[0] += np.bincount(b, minlength=self.cfg.ece_bins)
        self.cal[1] += np.bincount(b, weights=conf.ravel(), minlength=self.cfg.ece_bins)
        self.cal[2] += np.bincount(b, weights=g["free"].ravel().astype(float), minlength=self.cfg.ece_bins)
        rb = np.clip((conf * self.cfg.rc_bins).astype(int), 0, self.cfg.rc_bins - 1).ravel()
        self.rc[0] += np.bincount(rb, minlength=self.cfg.rc_bins)
        self.rc[1] += np.bincount(rb, weights=g["hazard"].ravel().astype(float), minlength=self.cfg.rc_bins)
        self.frames += 1
        return frame

    def result(self) -> dict:
        out = {"frames": self.frames, "config": self.cfg.to_dict(), "regions": {}}
        for name, c in self.counts.items():
            out["regions"][name] = {
                "cells": c["cells"], "pred_free_share": _ratio(c["pred_free"], c["cells"]),
                "ffr_pred": _ratio(c["false_free"], c["pred_free"]), "ffr_hazard": _ratio(c["false_free"], c["hazard"]),
                "free_road": _iou(c["free_tp"], c["free_fp"], c["free_fn"]),
                "road_head": _iou(c["road_tp"], c["road_fp"], c["road_fn"]),
                "occupancy": _iou(c["occ_tp"], c["occ_fp"], c["occ_fn"]),
                "unknown": _iou(c["unk_tp"], c["unk_fp"], c["unk_fn"])}
        out["classes"] = {name: {"cells": v["cells"], "occupancy_recall": _ratio(v["occ_hit"], v["cells"]),
                                 "called_free": _ratio(v["called_free"], v["cells"])}
                          for name, v in self.classes.items()}
        n, s_conf, s_free = self.cal
        out["calibration"] = {"ece": round(float(np.sum(np.abs(s_conf - s_free)) / max(n.sum(), 1)), 5),
                              "bins": [{"count": int(n[k]), "confidence": _ratio(s_conf[k], n[k]),
                                        "free_rate": _ratio(s_free[k], n[k])} for k in range(len(n))]}
        cells, hazard = self.rc[0][::-1].cumsum(), self.rc[1][::-1].cumsum()       # from the most confident bin down
        total = max(self.rc[0].sum(), 1)
        out["risk_coverage"] = [{"min_confidence": round(1 - (k + 1) / self.cfg.rc_bins, 4),
                                 "coverage": round(float(cells[k] / total), 5), "ffr_pred": _ratio(hazard[k], cells[k])}
                                for k in range(self.cfg.rc_bins)]
        return out
