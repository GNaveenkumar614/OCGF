"""IoU of rotated 3D boxes (x, y, z, l, w, h, yaw; centre convention) in numpy: BEV IoU by convex polygon clipping
(Sutherland-Hodgman), 3D IoU from the BEV intersection times the height overlap. For evaluation code, which must
not import the teachers (MTrans has its own torch IoU for training)."""
from __future__ import annotations

import numpy as np


def footprint(box) -> np.ndarray:
    """[4, 2] BEV corners, counter-clockwise."""
    x, y, _, l, w, _, yaw = (float(v) for v in box[:7])
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[l, w], [-l, w], [-l, -w], [l, -w]]) * 0.5
    return local @ np.array([[c, s], [-s, c]]) + (x, y)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Intersection of a polygon with a convex, counter-clockwise clip polygon."""
    out = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        if len(out) == 0:
            break
        inside = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12
        new = []
        for j in range(len(out)):
            p, q = out[j], out[(j + 1) % len(out)]
            ip, iq = inside(p), inside(q)
            if ip:
                new.append(p)
            if ip != iq:                                     # the edge p-q crosses the clip line a-b
                d1, d2 = q - p, b - a
                den = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(den) > 1e-12:
                    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
                    new.append(p + t * d1)
        out = np.array(new) if new else np.zeros((0, 2))
    return out


def bev_intersection(a, b) -> float:
    return polygon_area(clip_polygon(footprint(a), footprint(b)))


def bev_iou(a, b) -> float:
    inter = bev_intersection(a, b)
    union = a[3] * a[4] + b[3] * b[4] - inter
    return float(inter / union) if union > 0 else 0.0


def iou_3d(a, b) -> float:
    inter_bev = bev_intersection(a, b)
    za, zb = (a[2] - a[5] / 2, a[2] + a[5] / 2), (b[2] - b[5] / 2, b[2] + b[5] / 2)
    dz = max(0.0, min(za[1], zb[1]) - max(za[0], zb[0]))
    inter = inter_bev * dz
    union = a[3] * a[4] * a[5] + b[3] * b[4] * b[5] - inter
    return float(inter / union) if union > 0 else 0.0


def enclosing_box(boxes) -> np.ndarray:
    """One box around several (a rider and its two-wheeler), aligned with the first box's heading."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 7)
    yaw = boxes[0, 6]
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.concatenate([footprint(b) for b in boxes])
    along, across = pts @ np.array([c, s]), pts @ np.array([-s, c])
    zlo = float((boxes[:, 2] - boxes[:, 5] / 2).min())
    zhi = float((boxes[:, 2] + boxes[:, 5] / 2).max())
    ca, cc = (along.max() + along.min()) / 2, (across.max() + across.min()) / 2
    return np.array([ca * c - cc * s, ca * s + cc * c, (zlo + zhi) / 2, along.max() - along.min(),
                     across.max() - across.min(), zhi - zlo, yaw])
