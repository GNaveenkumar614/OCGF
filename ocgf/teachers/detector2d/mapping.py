"""COCO detections -> our four classes, and the detection files every teacher reads. numpy only.

Mapping (CLAUDE.md): car <- car · large_vehicle <- truck, bus, train · pedestrian <- person ·
cyclist <- a person box merged with the bicycle/motorcycle box it overlaps.
  - A person and a two-wheeler pair up when their intersection covers at least min_overlap of the smaller box; pairs
    are taken best overlap first, one to one. The cyclist box is their union, its score the lower of the two.
  - Unpaired persons are pedestrians. Unpaired two-wheelers (parked bikes) are not one of our classes: they stay in
    the raw file only. They still count as occupied in evaluation, and LiDAR sees them.
  - COCO often reports one vehicle as both 'car' and 'truck'; across our classes, a box overlapping a higher-scored box
    with IoU >= dup_iou is dropped.

File format, one detection per line: 'fid cls score l t r b' (canonical frame id, pixel box edges). The low score floor
is applied here; stricter thresholds belong to whoever reads the file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUR_CLASSES = ("car", "large_vehicle", "pedestrian", "cyclist")
COCO_TO_OURS = {"car": "car", "truck": "large_vehicle", "bus": "large_vehicle", "train": "large_vehicle",
                "person": "pedestrian"}
TWO_WHEELERS = ("bicycle", "motorcycle")
RAW_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck")     # kept in the raw file


@dataclass
class Detections:
    """Detections in one image: boxes [N, 4] as (l, t, r, b) pixels, scores [N], labels (class names)."""
    boxes: np.ndarray
    scores: np.ndarray
    labels: list

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4)), np.zeros(0), [])

    def __len__(self) -> int:
        return len(self.labels)

    def select(self, idx) -> "Detections":
        idx = np.asarray(idx, dtype=np.int64)
        return Detections(self.boxes[idx], self.scores[idx], [self.labels[i] for i in idx])


def area(boxes) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def intersection(a, b) -> np.ndarray:
    """[len(a), len(b)] pairwise intersection areas."""
    a, b = np.asarray(a, dtype=np.float64).reshape(-1, 4), np.asarray(b, dtype=np.float64).reshape(-1, 4)
    w = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    h = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    return np.clip(w, 0, None) * np.clip(h, 0, None)


def iou(a, b) -> np.ndarray:
    inter = intersection(a, b)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def to_our_classes(dets: Detections, min_overlap: float = 0.4, dup_iou: float = 0.7) -> Detections:
    """Map COCO detections to our classes (see the module docstring)."""
    persons = [i for i, l in enumerate(dets.labels) if l == "person"]
    wheels = [i for i, l in enumerate(dets.labels) if l in TWO_WHEELERS]
    boxes, scores, labels = [], [], []
    used_p, used_w = set(), set()
    if persons and wheels:
        inter = intersection(dets.boxes[persons], dets.boxes[wheels])
        smaller = np.minimum(area(dets.boxes[persons])[:, None], area(dets.boxes[wheels])[None, :])
        overlap = np.where(smaller > 0, inter / np.maximum(smaller, 1e-12), 0.0)
        for flat in np.argsort(-overlap, axis=None):
            pi, wi = np.unravel_index(flat, overlap.shape)
            if overlap[pi, wi] < min_overlap:
                break
            if pi in used_p or wi in used_w:
                continue
            used_p.add(pi)
            used_w.add(wi)
            p, w = dets.boxes[persons[pi]], dets.boxes[wheels[wi]]
            boxes.append([min(p[0], w[0]), min(p[1], w[1]), max(p[2], w[2]), max(p[3], w[3])])
            scores.append(min(dets.scores[persons[pi]], dets.scores[wheels[wi]]))
            labels.append("cyclist")
    for k, i in enumerate(persons):
        if k not in used_p:
            boxes.append(dets.boxes[i])
            scores.append(dets.scores[i])
            labels.append("pedestrian")
    for i, label in enumerate(dets.labels):
        if label in COCO_TO_OURS and label != "person":
            boxes.append(dets.boxes[i])
            scores.append(dets.scores[i])
            labels.append(COCO_TO_OURS[label])
    if not labels:
        return Detections.empty()
    out = Detections(np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.asarray(scores, dtype=np.float64), labels)
    return suppress_duplicates(out, dup_iou)


def suppress_duplicates(dets: Detections, dup_iou: float = 0.7) -> Detections:
    """Greedy, class-agnostic: keep the best-scored box, drop others overlapping it with IoU >= dup_iou."""
    order = np.argsort(-dets.scores, kind="stable")
    keep = []
    overlaps = iou(dets.boxes, dets.boxes)
    for i in order:
        if all(overlaps[i, j] < dup_iou for j in keep):
            keep.append(i)
    return dets.select(sorted(keep, key=lambda i: -dets.scores[i]))


def format_lines(fid: str, dets: Detections) -> list[str]:
    return [f"{fid} {label} {score:.4f} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}"
            for label, score, b in zip(dets.labels, dets.scores, dets.boxes)]


def read_detections(path) -> dict[str, Detections]:
    """{fid: Detections} from a detection file."""
    rows: dict[str, list] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            fid, label, score, l, t, r, b = line.split()
            rows.setdefault(fid, []).append((label, float(score), (float(l), float(t), float(r), float(b))))
    return {fid: Detections(np.array([x[2] for x in v]).reshape(-1, 4), np.array([x[1] for x in v]), [x[0] for x in v])
            for fid, v in rows.items()}
