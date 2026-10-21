"""Evaluation ground truth in BEV for KITTI-360 frames, built from ground-truth data only.

Inputs: the accumulated semantic point clouds of the frame's window (the static cloud, plus the moving-object points
recorded at that frame), the 3D boxes valid at the frame, the frame's pose and the calibration. Never the LiDAR scan or
any geometry the student uses (no fitted ground, no camera lifting): the student's own geometry must not shape the
ground truth it is measured on. The accumulated clouds include points seen from other frames, so observability is
computed separately, by ray casting over ground-truth occluders.

Layers per frame on the BEV grid (200 x 160, indexed [ix, iy]):
  ground_z        float16  ground height: median of the cell's ground-surface points; holes filled from the nearest cell
  ground_class    uint8    majority label id of the cell's ground-surface points (kitti360_labels; 0 = none)
  labelled        bool     the cell holds ground-truth points (ground, or anything up to obstacle_max above it)
  occupied        uint8    object box covering the cell: 0 none, 1 car, 2 large vehicle, 3 pedestrian, 4 cyclist,
                           5 other (every annotated object is occupied, whatever its class; decided 2026-10-16)
  obstacle        bool     >= min_obstacle_points points obstacle_min..obstacle_max above ground (static scene, or
                           moving objects at this frame): walls, fences, bushes, poles, unboxed objects
  observed_lidar  bool     the cell's ground is in line of sight of the LiDAR, occluders = obstacle points and boxes
  observed_cam    bool     the same for the camera centre, and the ground point projects inside the image
free_space(): drivable surface & ~occupied & ~obstacle & labelled & (observed_lidar | observed_cam).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..datasets.kitti360_labels import GROUND_SURFACES, ROAD
from ..datasets.ply import read_ply
from ..geometry.bev_grid import BEVGrid
from ..geometry.projection import project
from ..geometry.raster import box_cells
from ..geometry.transforms import invert_rigid
from ..geometry.viewshed import viewshed

LAYERS = ("ground_z", "ground_class", "labelled", "occupied", "obstacle", "observed_lidar", "observed_cam")
OCCUPIED_CLASSES = ("none", "car", "large_vehicle", "pedestrian", "cyclist", "other")


@dataclass(frozen=True)
class GTConfig:
    obstacle_min: float = 0.3          # m above ground: lower things (curbs, ground noise) are not obstacles
    obstacle_max: float = 3.0          # m above ground: higher things (canopies) are not obstacles or occluders
    min_obstacle_points: int = 2
    margin: float = 5.0                # m around the grid kept when cropping the clouds
    fallback_ground: float = -1.73     # LiDAR-frame ground height when a frame has no ground points at all


# ---------------------------------------------------------------- clouds

def load_window(semantics3d: Path, window_start: int, window_end: int) -> tuple[dict, dict | None]:
    """Static and dynamic accumulated clouds of one window, as {x, y, z (world, float64), semantic[, timestamp]}."""
    name = f"{int(window_start):010d}_{int(window_end):010d}.ply"
    def read(part):
        path = Path(semantics3d) / part / name
        if not path.exists():
            return None
        c = read_ply(path)
        out = {"xyz": np.column_stack([c["x"], c["y"], c["z"]]).astype(np.float64),
               "semantic": c["semantic"].astype(np.int16)}
        if "timestamp" in c.dtype.names:
            out["timestamp"] = c["timestamp"].astype(np.int64)
        return out
    static = read("static")
    if static is None:
        raise FileNotFoundError(f"no static cloud {name} under {semantics3d}")
    return static, read("dynamic")


def to_lidar(xyz_world: np.ndarray, T_world_lidar, grid: BEVGrid, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """World points -> the frame's LiDAR frame, cropped to the grid plus margin. Returns (points, kept mask)."""
    T = invert_rigid(np.asarray(T_world_lidar, dtype=np.float64))
    p = xyz_world @ T[:3, :3].T + T[:3, 3]
    keep = ((p[:, 0] >= grid.x_min - margin) & (p[:, 0] < grid.x_max + margin)
            & (p[:, 1] >= grid.y_min - margin) & (p[:, 1] < grid.y_max + margin) & (np.abs(p[:, 2]) < 10))
    return p[keep], keep


# ---------------------------------------------------------------- per-cell statistics

def _flat_cells(grid: BEVGrid, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inside = grid.inside(p[:, 0], p[:, 1])
    ix, iy = grid.index(p[inside, 0], p[inside, 1])
    return ix * grid.shape[1] + iy, inside


def cell_median(cells: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    if not len(cells):
        return out
    order = np.lexsort((values, cells))
    c, v = cells[order], values[order]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    counts = np.diff(np.r_[starts, len(c)])
    out[c[starts]] = v[starts + (counts - 1) // 2]
    return out


def fill_nearest(z: np.ndarray, fallback: float) -> np.ndarray:
    known = np.isfinite(z)
    if not known.any():
        return np.full(z.shape, fallback)
    _, (ii, jj) = distance_transform_edt(~known, return_indices=True)
    return z[ii, jj]


# ---------------------------------------------------------------- one frame

def build_frame_gt(grid: BEVGrid, T_world_lidar, static: dict, dynamic: dict | None, frame: int, boxes, camera,
                   cfg: GTConfig = GTConfig()) -> dict:
    """GT layers of one frame. boxes: FrameBoxes valid at the frame (LiDAR frame); camera: CameraCalib."""
    nx, ny = grid.shape
    n = nx * ny
    S, keep = to_lidar(static["xyz"], T_world_lidar, grid, cfg.margin)
    s_sem = static["semantic"][keep]
    if dynamic is not None and "timestamp" in dynamic:
        now = dynamic["timestamp"] == int(frame)                         # moving objects where they are at this frame
        D, _ = to_lidar(dynamic["xyz"][now], T_world_lidar, grid, cfg.margin)
    else:
        D = np.zeros((0, 3))

    # ground surface: height and majority class
    is_ground = np.isin(s_sem, GROUND_SURFACES)
    g_cells, g_in = _flat_cells(grid, S[is_ground])
    ground_z = cell_median(g_cells, S[is_ground][g_in, 2], n).reshape(nx, ny)
    lut = np.zeros(max(GROUND_SURFACES) + 1, dtype=np.int64)
    lut[list(GROUND_SURFACES)] = np.arange(len(GROUND_SURFACES))
    votes = np.zeros((n, len(GROUND_SURFACES)), dtype=np.int64)
    np.add.at(votes, (g_cells, lut[s_sem[is_ground][g_in]]), 1)
    ground_class = np.where(votes.sum(1) > 0, np.array(GROUND_SURFACES)[votes.argmax(1)], 0).reshape(nx, ny)
    z_fill = fill_nearest(ground_z, cfg.fallback_ground)

    # heights above ground of every non-ground point (static) and every moving point at this frame
    above = np.concatenate([S[~is_ground], D])
    a_cells, a_in = _flat_cells(grid, above)
    h = above[a_in, 2] - z_fill.reshape(-1)[a_cells]
    band = (h > cfg.obstacle_min) & (h < cfg.obstacle_max)
    counts = np.bincount(a_cells[band], minlength=n)
    obstacle = (counts >= cfg.min_obstacle_points).reshape(nx, ny)
    occluder_top = np.full(n, -np.inf)
    np.maximum.at(occluder_top, a_cells[band], above[a_in, 2][band])
    occluder_top = np.where(obstacle.reshape(-1), occluder_top, np.nan).reshape(nx, ny)

    labelled = np.zeros(n, dtype=bool)
    labelled[g_cells] = True
    labelled[a_cells[h <= cfg.obstacle_max]] = True
    labelled = labelled.reshape(nx, ny)

    # object boxes: occupancy by class (specific classes drawn last, so they win over 'other') and box tops as occluders
    occupied = np.zeros((nx, ny), dtype=np.uint8)
    order = np.argsort(-np.asarray(boxes.classes)) if len(boxes) else []
    for k in order:
        ix, iy = box_cells(boxes.params[k], grid)
        occupied[ix, iy] = int(boxes.classes[k]) + 1
        top = boxes.params[k, 2] + boxes.params[k, 5] / 2
        occluder_top[ix, iy] = np.fmax(occluder_top[ix, iy], top)

    observed_lidar = viewshed(grid, (0.0, 0.0, 0.0), ground_z, occluder_top)
    cam_origin = invert_rigid(camera.T_cam_lidar)[:3, 3]
    xs, ys = grid.center_grid()
    _, _, in_fov = project(np.column_stack([xs.ravel(), ys.ravel(), z_fill.ravel()]), camera)
    observed_cam = viewshed(grid, cam_origin, ground_z, occluder_top) & in_fov.reshape(nx, ny)

    return {"ground_z": ground_z.astype(np.float16), "ground_class": ground_class.astype(np.uint8),
            "labelled": labelled, "occupied": occupied, "obstacle": obstacle,
            "observed_lidar": observed_lidar, "observed_cam": observed_cam}


def free_space(layers: dict, drivable=(ROAD,)) -> np.ndarray:
    """Ground-truth free, drivable, observed cells. drivable: ground classes that count (road only by default; parking
    is an open decision)."""
    return (np.isin(layers["ground_class"], drivable) & (layers["occupied"] == 0) & ~layers["obstacle"]
            & layers["labelled"] & (layers["observed_lidar"] | layers["observed_cam"]))


def save_layers(path, layers: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: layers[k] for k in LAYERS})


def load_layers(path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in LAYERS}


def settings(cfg: GTConfig) -> dict:
    return asdict(cfg)
