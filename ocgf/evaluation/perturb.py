"""Test-time perturbations for the robustness experiments (E5), numpy only. Never used in training unless a config
says so (architecture plan §7.4).

A perturbation is a spec string of '+'-joined terms, e.g. "rings=2", "points=0.5+box_drop=0.25", "blur=2":
  LiDAR    rings=k       keep every k-th ring (2: a 32-beam sensor, 4: a 16-beam one)
           points=p      drop each point with probability p
  camera   blur=s        Gaussian blur, sigma s pixels
           noise=s       Gaussian noise, sigma s (fraction of 255)
           exposure=f    multiply the intensities by f (0.5 under-, 1.8 over-exposed), clipped
           patches=n     n black rectangles, each 1/8 of the width and 1/4 of the height
  boxes    box_jitter=s  move each 3D box centre by N(0, s) m in x and y
           box_yaw=s     rotate each 3D box by N(0, s) rad
           box_drop=p    drop each detection (with or without a 3D box) with probability p; 1.0 drops them all, so
                         E5 can check that a missed object does not become free road
The model's inputs are recomputed from the perturbed scan and boxes (ground, visibility, Q, object channels).
Randomness is seeded by (seed, frame, term), so a perturbed evaluation is deterministic, and the post-hoc box arm
sees exactly the boxes the object models see.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy.ndimage import gaussian_filter

from ..datasets.objects import FrameObjects

LIDAR_TERMS = ("rings", "points")
CAMERA_TERMS = ("blur", "noise", "exposure", "patches")
BOX_TERMS = ("box_jitter", "box_yaw", "box_drop")
TERMS = LIDAR_TERMS + CAMERA_TERMS + BOX_TERMS


def parse_spec(spec: str | None) -> dict[str, float]:
    """{term: value} from "rings=2+box_drop=0.5"; '' or None is the clean evaluation."""
    out = {}
    for part in (spec or "").split("+"):
        if not part.strip():
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in TERMS or not value:
            raise ValueError(f"unknown perturbation term {part!r}; terms: {', '.join(TERMS)} (as name=value)")
        out[name] = float(value)
    return out


def _rng(seed: int, frame: int, term: str) -> np.random.Generator:
    return np.random.default_rng([seed, int(frame), zlib.crc32(term.encode())])


def perturb_points(points: np.ndarray, spec: dict, frame: int, seed: int = 0) -> np.ndarray:
    p = np.asarray(points)
    if "rings" in spec:
        p = p[p[:, 4].astype(int) % int(spec["rings"]) == 0]
    if "points" in spec:
        p = p[_rng(seed, frame, "points").uniform(size=len(p)) >= spec["points"]]
    return p


def perturb_image(image: np.ndarray, spec: dict, frame: int, seed: int = 0) -> np.ndarray:
    if not any(t in spec for t in CAMERA_TERMS):
        return image
    img = image.astype(np.float32)
    if "blur" in spec:
        img = gaussian_filter(img, sigma=(spec["blur"], spec["blur"], 0))
    if "exposure" in spec:
        img = img * spec["exposure"]
    if "noise" in spec:
        img = img + _rng(seed, frame, "noise").normal(0.0, spec["noise"] * 255.0, img.shape)
    if "patches" in spec:
        rng = _rng(seed, frame, "patches")
        h, w = img.shape[:2]
        ph, pw = h // 4, w // 8
        for _ in range(int(spec["patches"])):
            top, left = rng.integers(0, h - ph + 1), rng.integers(0, w - pw + 1)
            img[top:top + ph, left:left + pw] = 0.0
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def perturb_objects(objs: FrameObjects | None, spec: dict, frame: int, seed: int = 0) -> FrameObjects | None:
    if objs is None or not any(t in spec for t in BOX_TERMS) or len(objs) == 0:
        return objs
    boxes = objs.boxes.copy()
    has = objs.has_box
    if "box_jitter" in spec:
        boxes[has, :2] += _rng(seed, frame, "box_jitter").normal(0.0, spec["box_jitter"], (int(has.sum()), 2))
    if "box_yaw" in spec:
        boxes[has, 6] += _rng(seed, frame, "box_yaw").normal(0.0, spec["box_yaw"], int(has.sum()))
    out = FrameObjects(boxes, objs.cls, objs.score_2d, objs.conf, objs.box_2d)
    if "box_drop" in spec:
        out = out.select(_rng(seed, frame, "box_drop").uniform(size=len(out)) >= spec["box_drop"])
    return out


def spec_name(spec: str | None) -> str:
    """A folder-safe name: 'clean', or the spec with '=' and '+' replaced."""
    return "clean" if not spec else spec.replace("=", "").replace("+", "_").replace(".", "p")
