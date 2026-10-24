"""MTrans objects for the student: reading the neutral object files and rasterising them onto the BEV grid.

numpy only. The files are read directly (JSON lines, ocgf/teachers/mtrans/infer.py writes them), so the student code
never imports ocgf.teachers. Every detector box at or above the inference floor has a row: a 3D box (LiDAR frame,
centre convention as ocgf.geometry.raster) with conf_iou_raw, or status "too_few_points" and only its 2D box.

Quality of an object = score_2d x conf_iou_raw (the plan's soft input value; heuristic until calibrated).

Object channels O_B [N_CHANNELS, nx, ny]:
  0-3  soft occupancy per class (car, large vehicle, pedestrian, cyclist): max quality of the boxes covering the cell
  4    q_obj: max quality over the classes
  5    box top height above the ground z_g (m), max over covering boxes
  6    object shadow: the cell's ground is hidden from the camera by a box (viewshed with boxes as the only occluders)
  7    signed distance to the nearest footprint, clipped to +-max_distance and divided by it (inside < 0)
  8    detections without a 3D box: max 2D score over the cells whose ground point projects inside their 2D box
       (no frustum point, so no depth; such an object must never become free road)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..geometry.bev_grid import BEVGrid
from ..geometry.projection import project
from ..geometry.raster import box_cells
from ..geometry.transforms import invert_rigid
from ..geometry.viewshed import viewshed

OBJECT_CLASSES = ("car", "large_vehicle", "pedestrian", "cyclist")
CHANNELS = ("occ_car", "occ_large_vehicle", "occ_pedestrian", "occ_cyclist", "q_obj", "top_height", "obj_shadow",
            "signed_distance", "unlocalized")
N_CHANNELS = len(CHANNELS)


@dataclass(frozen=True)
class ObjectConfig:
    min_score: float = 0.1             # detector boxes passed on as soft inputs (the inference floor)
    max_distance: float = 5.0          # m, signed-distance clip
    # the occluder threshold lives in BEVInputConfig, the occupancy pseudo-label threshold in TargetConfig

    @classmethod
    def from_dict(cls, d: dict | None) -> ObjectConfig:
        return cls(**dict(d or {}))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameObjects:
    boxes: np.ndarray                  # [K, 7] float64, NaN rows for detections without a 3D box
    cls: np.ndarray                    # [K] int64 index into OBJECT_CLASSES
    score_2d: np.ndarray               # [K]
    conf: np.ndarray                   # [K] conf_iou_raw, NaN without a 3D box
    box_2d: np.ndarray                 # [K, 4] l, t, r, b (pixels of the original image)

    @property
    def has_box(self) -> np.ndarray:
        return np.isfinite(self.conf)

    @property
    def quality(self) -> np.ndarray:
        return np.where(self.has_box, self.score_2d * np.nan_to_num(self.conf), 0.0)

    def __len__(self):
        return len(self.cls)

    @classmethod
    def empty(cls) -> FrameObjects:
        return cls(np.zeros((0, 7)), np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros((0, 4)))

    def select(self, keep) -> FrameObjects:
        return FrameObjects(self.boxes[keep], self.cls[keep], self.score_2d[keep], self.conf[keep], self.box_2d[keep])


def read_object_file(path, min_score: float = 0.0) -> dict[int, FrameObjects]:
    """{frame: FrameObjects} from an MTrans object file (one sequence). Rows below min_score are dropped."""
    rows: dict[int, list] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["cls"] in OBJECT_CLASSES and r["score_2d"] >= min_score:
                rows.setdefault(int(r["fid"].rsplit("/", 1)[1]), []).append(r)
    out = {}
    for frame, rs in rows.items():
        out[frame] = FrameObjects(
            boxes=np.array([r["box"] if r["box"] is not None else [np.nan] * 7 for r in rs], np.float64),
            cls=np.array([OBJECT_CLASSES.index(r["cls"]) for r in rs], np.int64),
            score_2d=np.array([r["score_2d"] for r in rs], np.float64),
            conf=np.array([r["conf_iou_raw"] if r["conf_iou_raw"] is not None else np.nan for r in rs], np.float64),
            box_2d=np.array([r["box_2d"] for r in rs], np.float64))
    return out


def footprint_max(objs: FrameObjects, grid: BEVGrid, values: np.ndarray, keep=None) -> np.ndarray:
    """[nx, ny]: max of values over the 3D boxes covering each cell (0 where none)."""
    out = np.zeros(grid.shape, np.float32)
    idx = np.flatnonzero(objs.has_box if keep is None else objs.has_box & keep)
    for k in idx:
        ix, iy = box_cells(objs.boxes[k], grid)
        out[ix, iy] = np.maximum(out[ix, iy], values[k])
    return out


def box_tops(objs: FrameObjects, grid: BEVGrid, min_quality: float) -> np.ndarray:
    """[nx, ny]: top height (LiDAR z) of the boxes at or above min_quality covering each cell, NaN where none."""
    out = np.full(grid.shape, -np.inf)
    for k in np.flatnonzero(objs.has_box & (objs.quality >= min_quality)):
        ix, iy = box_cells(objs.boxes[k], grid)
        out[ix, iy] = np.maximum(out[ix, iy], objs.boxes[k, 2] + objs.boxes[k, 5] / 2)
    out[~np.isfinite(out)] = np.nan
    return out


def object_channels(objs: FrameObjects, z_g: np.ndarray, camera, grid: BEVGrid = BEVGrid(),
                    cfg: ObjectConfig = ObjectConfig()) -> np.ndarray:
    """O_B [N_CHANNELS, nx, ny] float32 for one frame (see the module docstring)."""
    nx, ny = grid.shape
    out = np.zeros((N_CHANNELS, nx, ny), np.float32)
    if len(objs) == 0:
        out[7] = 1.0
        return out
    q = objs.quality
    for c in range(len(OBJECT_CLASSES)):
        out[c] = footprint_max(objs, grid, q, objs.cls == c)
    out[4] = out[:4].max(axis=0)
    tops = box_tops(objs, grid, 0.0)
    out[5] = np.where(np.isfinite(tops), np.maximum(tops - z_g, 0.0), 0.0)
    occupied = footprint_max(objs, grid, np.ones(len(objs))) > 0
    if occupied.any():
        cam_origin = invert_rigid(camera.T_cam_lidar)[:3, 3]
        out[6] = ~viewshed(grid, cam_origin, z_g, tops) & ~occupied & np.isfinite(z_g)
        outside = distance_transform_edt(~occupied, sampling=grid.res)
        inside = distance_transform_edt(occupied, sampling=grid.res)
        out[7] = np.clip(np.where(occupied, -inside, outside), -cfg.max_distance, cfg.max_distance) / cfg.max_distance
    else:
        out[7] = 1.0
    no_box = np.flatnonzero(~objs.has_box)
    if len(no_box):
        xs, ys = grid.center_grid()
        uv, _, ok = project(np.column_stack([xs.ravel(), ys.ravel(), z_g.ravel().astype(np.float64)]), camera)
        u, v = uv[:, 0].reshape(nx, ny), uv[:, 1].reshape(nx, ny)
        ok = ok.reshape(nx, ny)
        for k in no_box:
            l, t, r, b = objs.box_2d[k]
            inside = ok & (u >= l) & (u <= r) & (v >= t) & (v <= b)
            out[8] = np.maximum(out[8], inside * objs.score_2d[k])
    return out


def strict_footprints(objs: FrameObjects, grid: BEVGrid, strict_quality: float) -> np.ndarray:
    """Bool [nx, ny]: footprints of the boxes at or above strict_quality (occupancy pseudo-labels)."""
    return footprint_max(objs, grid, np.ones(len(objs)), objs.quality >= strict_quality) > 0


def any_footprints(objs: FrameObjects, grid: BEVGrid) -> np.ndarray:
    """Bool [nx, ny]: footprints of every 3D box (no road negatives there)."""
    return footprint_max(objs, grid, np.ones(len(objs))) > 0
