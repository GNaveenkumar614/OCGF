"""How the OCGF gate uses the reliability channels: mean camera weight w_C per bin of each Q channel (numpy only).

This is the measurable use of Q the "Confidence-Guided" title needs (CLAUDE.md). Only cells where both sensors are
valid count, because only there does the gate choose (elsewhere one weight is forced to 0). A gate that ignores Q
shows a flat w_C across a channel's bins; a gate that leans on the LiDAR where the camera is uncertain shows w_C
falling as H_cam rises. Bins are fixed (not deciles of the data) so runs compare directly; empty bins report None.
"""
from __future__ import annotations

import numpy as np

Q_CHANNELS = ("log_points", "range", "r_g", "M_L", "shadow_cam", "M_C", "H_cam", "q_obj")   # ocgf.models.student
EDGES = {"log_points": np.linspace(0.0, 7.0, 11), "range": np.linspace(0.0, 1.1, 12),
         "r_g": np.linspace(0.0, 1.0, 11), "M_L": np.array([0.0, 0.5, 1.0]), "shadow_cam": np.array([0.0, 0.5, 1.0]),
         "M_C": np.array([0.0, 0.5, 1.0]), "H_cam": np.linspace(0.0, 1.0, 11), "q_obj": np.linspace(0.0, 1.0, 11)}


class GateLog:
    def __init__(self):
        self.sum = {c: np.zeros(len(EDGES[c]) - 1) for c in Q_CHANNELS}
        self.count = {c: np.zeros(len(EDGES[c]) - 1) for c in Q_CHANNELS}
        self.total = np.zeros(2)                        # sum of w_C, cells (both valid)
        self.cells = np.zeros(3)                        # both valid, camera only, LiDAR only

    def add(self, w_c: np.ndarray, q: np.ndarray, valid_c: np.ndarray, valid_l: np.ndarray) -> None:
        """w_c [nx, ny]; q [8, nx, ny]; valid masks [nx, ny] (the ones the gate used)."""
        both = valid_c & valid_l
        self.cells += [both.sum(), (valid_c & ~valid_l).sum(), (valid_l & ~valid_c).sum()]
        w = w_c[both].astype(np.float64)
        self.total += [w.sum(), len(w)]
        for k, c in enumerate(Q_CHANNELS):
            edges = EDGES[c]
            b = np.clip(np.searchsorted(edges, q[k][both], side="right") - 1, 0, len(edges) - 2)
            self.sum[c] += np.bincount(b, weights=w, minlength=len(edges) - 1)
            self.count[c] += np.bincount(b, minlength=len(edges) - 1)

    def result(self) -> dict:
        r = lambda a, b: round(float(a / b), 4) if b else None
        return {"mean_w_C": r(*self.total),
                "cells": {"both_valid": int(self.cells[0]), "camera_only": int(self.cells[1]),
                          "lidar_only": int(self.cells[2])},
                "by_channel": {c: [{"lo": round(float(EDGES[c][i]), 3), "hi": round(float(EDGES[c][i + 1]), 3),
                                    "cells": int(self.count[c][i]), "mean_w_C": r(self.sum[c][i], self.count[c][i])}
                                   for i in range(len(EDGES[c]) - 1)] for c in Q_CHANNELS}}
