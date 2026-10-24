"""BEV training targets from the teachers' outputs and the student's own geometry (training frames only).

The road teacher's label lives on image pixels; the student predicts on the BEV grid. Each cell reads the label at its
ground point (x, y, z_g), with the same projection the camera branch uses to lift image features, so targets and
features line up cell for cell. Cells the camera cannot see never borrow a label (the pixel there shows the occluder).

road_t  float32 in [0, 1], NaN = no target. The image label at the cell's ground point where the camera sees the cell
        (M_C); else the mean LiDAR score of the cell's ground returns; else NaN (the road loss ignores it).
unk_t   float32 0/1: 1 where neither sensor observes the cell's ground (outside both views, or in both shadows) or the
        ground estimate is unreliable (r_g < min_r_g). As in the evaluation ground truth, a cell observed by either
        sensor can be free.
occ_t   float32, NaN = ignore: 1 where at least min_occ_points stand occluder_min..occluder_max above ground, and with
        box_targets on the footprints of MTrans boxes of quality >= strict_quality (pseudo-labels); 0 where
        road_t >= road_negative, nothing stands above ground and no detected box covers the cell.
Labels stay soft. numpy only; reads the teachers' files directly and never imports ocgf.teachers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..geometry.bev_grid import BEVGrid
from ..geometry.ground import height_above_ground
from ..geometry.projection import project
from ..datasets.objects import any_footprints, strict_footprints

TARGETS = ("road_t", "unk_t", "occ_t")


@dataclass(frozen=True)
class TargetConfig:
    min_r_g: float = 0.1               # ground reliability below this: unknown
    min_occ_points: int = 2            # as the evaluation ground truth's obstacles
    road_negative: float = 0.8         # road_t at or above this, with nothing above ground: an occupancy negative
    box_targets: bool = False          # MTrans boxes in occ_t (strict ones positive, none negative inside any box)
    strict_quality: float = 0.3        # score_2d x conf_iou at or above which a box footprint is an occupancy label

    @classmethod
    def from_dict(cls, d: dict | None) -> TargetConfig:
        return cls(**dict(d or {}))

    def to_dict(self) -> dict:
        return asdict(self)


def read_road_label(path) -> tuple[np.ndarray, np.ndarray, str]:
    """(image label [H, W] float32, per-point scores [N] float32, status) from a road-teacher file (<frame>.npz)."""
    with np.load(path) as f:
        return f["label"].astype(np.float32), f["point_scores"].astype(np.float32), str(f["status"])


def sample_image_label(label: np.ndarray, z_g: np.ndarray, valid: np.ndarray, camera, grid: BEVGrid) -> np.ndarray:
    """[nx, ny]: the image label [H, W] at each valid cell's ground point (nearest pixel), NaN elsewhere."""
    xs, ys = grid.center_grid()
    out = np.full(grid.shape, np.nan, np.float32)
    if not valid.any():
        return out
    pts = np.column_stack([xs[valid], ys[valid], z_g[valid].astype(np.float64)])
    uv, _, ok = project(pts, camera)
    u = np.clip(np.rint(uv[ok, 0]).astype(np.int64), 0, label.shape[1] - 1)
    v = np.clip(np.rint(uv[ok, 1]).astype(np.int64), 0, label.shape[0] - 1)
    vals = np.full(len(pts), np.nan, np.float32)
    vals[ok] = label[v, u]
    out[valid] = vals
    return out


def lidar_cell_scores(points, scores: np.ndarray, z_g: np.ndarray, grid: BEVGrid, band: float) -> np.ndarray:
    """[nx, ny]: mean finite per-point road score of each cell's ground returns (|z - z_g| <= band), NaN if none."""
    inside, h = height_above_ground(points, z_g, grid)
    use = inside & (np.abs(h) <= band) & np.isfinite(scores)
    p = np.asarray(points)[use]
    ix, iy = grid.index(p[:, 0], p[:, 1])
    s = np.zeros(grid.shape)
    n = np.zeros(grid.shape)
    np.add.at(s, (ix, iy), scores[use].astype(np.float64))
    np.add.at(n, (ix, iy), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan).astype(np.float32)


def build_targets(inputs: dict, label: np.ndarray, point_scores: np.ndarray, points, camera,
                  grid: BEVGrid = BEVGrid(), cfg: TargetConfig = TargetConfig(), ground_band: float = 0.15,
                  objects=None) -> dict:
    """road_t, unk_t, occ_t for one frame. inputs: ocgf.datasets.bev_inputs.bev_inputs() of the same scan; label: the
    road teacher's image label [H, W] (NaN = unknown); point_scores: its per-point LiDAR scores, in scan order;
    objects: the frame's FrameObjects (used when cfg.box_targets)."""
    cam = sample_image_label(np.asarray(label, np.float32), inputs["z_g"], inputs["M_C"], camera, grid)
    lid = lidar_cell_scores(points, np.asarray(point_scores, np.float32), inputs["z_g"], grid, ground_band)
    road_t = np.where(np.isfinite(cam), cam, lid).astype(np.float32)

    observed = inputs["M_C"] | inputs["vis_lidar"]
    unk_t = (~observed | (inputs["r_g"] < cfg.min_r_g)).astype(np.float32)
    road_t[unk_t > 0] = np.nan                                   # nothing to learn about road where nobody looks

    occ_t = np.full(grid.shape, np.nan, np.float32)
    boxes = cfg.box_targets and objects is not None and len(objects) > 0
    negative = (inputs["n_above"] == 0) & (road_t >= cfg.road_negative)
    if boxes:
        negative &= ~any_footprints(objects, grid)
    occ_t[negative] = 0.0
    occ_t[inputs["n_above"] >= cfg.min_occ_points] = 1.0
    if boxes:
        occ_t[strict_footprints(objects, grid, cfg.strict_quality)] = 1.0
    return {"road_t": road_t, "unk_t": unk_t, "occ_t": occ_t}


# ---------------------------------------------------------------- per-frame records (inputs + targets)

_DTYPES = {"z_g": np.float16, "r_g": np.float16, "n_pts": np.uint16, "ground_hits": np.uint16, "h_max": np.float16,
           "n_above": np.uint16, "occ_top": np.float16, "road_t": np.float16, "unk_t": np.uint8, "occ_t": np.float16}


def save_record(path, inputs: dict, targets: dict) -> None:
    """One frame's BEV inputs and targets in a compressed .npz (counts saturate at 65535; masks stay bool)."""
    out = {}
    for key, value in {**inputs, **targets}.items():
        dtype = _DTYPES.get(key, value.dtype)
        if np.issubdtype(dtype, np.integer) and not np.issubdtype(np.asarray(value).dtype, np.bool_):
            value = np.clip(value, 0, np.iinfo(dtype).max)
        out[key] = np.asarray(value).astype(dtype)
    np.savez_compressed(path, **out)


def load_record(path) -> dict:
    """The arrays of save_record, as float32 (heights, scores, targets), int32 (counts) or bool (masks)."""
    with np.load(path) as f:
        return {k: (f[k].astype(np.float32) if f[k].dtype == np.float16 or k == "unk_t"
                    else f[k].astype(np.int32) if f[k].dtype == np.uint16 else f[k]) for k in f.files}
