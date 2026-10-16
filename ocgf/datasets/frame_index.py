"""Per-frame index of one KITTI-360 sequence: which data each frame has, timing, travel and usability.

usable = image ∧ scan ∧ pose ∧ inside a semantic window ∧ ≥ min_future_m of future path ∧ image/scan aligned.
sample_5m marks usable frames picked every step_m metres of travel (teacher and training frames).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

BOOL_COLUMNS = ("has_image", "has_scan", "has_pose", "has_sem2d", "aligned", "usable", "sample_5m")
INT_COLUMNS = ("frame", "window_start", "window_end", "n_points_raw", "n_points_crop")
COLUMNS = ("frame", "has_image", "has_scan", "has_pose", "has_sem2d", "window_start", "window_end",
           "t_image", "t_scan", "dt_ms", "aligned", "odometer_m", "future_path_m", "x_world", "y_world",
           "n_points_raw", "n_points_crop", "usable", "sample_5m")


def parse_timestamps(lines) -> np.ndarray:
    """'2013-05-28 08:46:02.904668160' lines -> seconds since the Unix epoch (float64, sub-microsecond)."""
    stamps = np.array([line.strip() for line in lines if line.strip()], dtype="datetime64[ns]")
    return stamps.astype(np.int64) / 1e9


def read_timestamps(path) -> np.ndarray:
    """Line i of a KITTI-style timestamps.txt is frame i."""
    return parse_timestamps(Path(path).read_text().splitlines())


def read_poses(path) -> tuple[np.ndarray, np.ndarray]:
    """poses.txt ('frame' + 12 values of a 3x4 [R|t] per line) -> frames [M], T_world_pose [M, 4, 4]."""
    data = np.loadtxt(path, ndmin=2)
    T = np.tile(np.eye(4), (len(data), 1, 1))
    T[:, :3, :] = data[:, 1:13].reshape(-1, 3, 4)
    return data[:, 0].astype(np.int64), T


def read_cam0_to_world(path) -> tuple[np.ndarray, np.ndarray]:
    """cam0_to_world.txt ('frame' + 16 values of a 4x4 per line) -> frames [M], T_world_cam0 [M, 4, 4]."""
    data = np.loadtxt(path, ndmin=2)
    return data[:, 0].astype(np.int64), data[:, 1:17].reshape(-1, 4, 4)


_WINDOW = re.compile(r"(\d{10})_(\d{10})\.ply$")


def semantic_windows(names) -> np.ndarray:
    """Accumulated-semantics file names '{start:010d}_{end:010d}.ply' -> [W, 2] inclusive frame ranges, sorted."""
    windows = sorted({(int(m.group(1)), int(m.group(2))) for n in names if (m := _WINDOW.search(str(n)))})
    return np.array(windows, dtype=np.int64).reshape(-1, 2)


def window_of(frames, windows) -> np.ndarray:
    """Index of the window holding each frame, -1 if none. Where windows overlap, the one the frame is most central in."""
    frames = np.asarray(frames)
    index = np.full(frames.shape, -1, dtype=np.int64)
    best = np.full(frames.shape, -1, dtype=np.int64)
    for w, (start, end) in enumerate(windows):
        margin = np.minimum(frames - start, end - frames)
        better = (margin >= 0) & (margin > best)
        index[better], best[better] = w, margin[better]
    return index


def sample_every(distance, mask, step: float) -> np.ndarray:
    """Greedy pick of masked entries, in order, each at least `step` further along `distance` than the last pick."""
    picked = np.zeros(len(distance), dtype=bool)
    last = -np.inf
    for i in np.flatnonzero(mask):
        if distance[i] - last >= step:
            picked[i], last = True, distance[i]
    return picked


def compress_ranges(frames) -> str:
    """[0, 1, 2, 3, 7, 9, 10] -> '0-3, 7, 9-10'."""
    frames = np.unique(np.asarray(frames, dtype=np.int64))
    if not len(frames):
        return "none"
    breaks = np.flatnonzero(np.diff(frames) > 1)
    starts, ends = frames[np.r_[0, breaks + 1]], frames[np.r_[breaks, len(frames) - 1]]
    return ", ".join(str(s) if s == e else f"{s}-{e}" for s, e in zip(starts, ends))


def _by_frame(values, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    values = np.asarray(values, dtype=np.float64)[:n]
    out[: len(values)] = values
    return out


def build_frame_index(*, image_frames, scan_frames, image_times, scan_times, pose_frames, T_world_lidar,
                      windows, sem2d_frames=(), scan_stats=None, min_future_m: float = 50.0,
                      step_m: float = 5.0, max_dt_ms: float = 20.0) -> dict[str, np.ndarray]:
    """One row per frame from 0 to the last frame seen in any source. Times are indexed by frame (line i = frame i).

    scan_stats: optional {'frame', 'n_raw', 'n_crop'} arrays from the scan pass.
    """
    image_frames, scan_frames, pose_frames = (np.asarray(a, dtype=np.int64) for a in (image_frames, scan_frames, pose_frames))
    n = 1 + max(int(a.max()) if len(a) else -1 for a in (image_frames, scan_frames, pose_frames))
    frame = np.arange(n, dtype=np.int64)

    t_image, t_scan = _by_frame(image_times, n), _by_frame(scan_times, n)
    dt_ms = (t_scan - t_image) * 1000.0
    with np.errstate(invalid="ignore"):
        aligned = np.abs(dt_ms) <= max_dt_ms

    order = np.argsort(pose_frames)
    pose_frames, T_world_lidar = pose_frames[order], np.asarray(T_world_lidar)[order]
    position = np.full((n, 3), np.nan)
    position[pose_frames] = T_world_lidar[:, :3, 3]
    odometer = np.full(n, np.nan)
    if len(pose_frames):
        steps = np.linalg.norm(np.diff(position[pose_frames], axis=0), axis=1)
        odometer[pose_frames] = np.concatenate([[0.0], np.cumsum(steps)])
    future = odometer[pose_frames[-1]] - odometer if len(pose_frames) else np.full(n, np.nan)

    window = window_of(frame, windows)
    windows = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
    window_start = np.where(window >= 0, windows[window, 0] if len(windows) else -1, -1)
    window_end = np.where(window >= 0, windows[window, 1] if len(windows) else -1, -1)

    n_raw, n_crop = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    if scan_stats is not None:
        f = np.asarray(scan_stats["frame"], dtype=np.int64)
        n_raw[f], n_crop[f] = scan_stats["n_raw"], scan_stats["n_crop"]

    has_image, has_scan, has_pose = np.isin(frame, image_frames), np.isin(frame, scan_frames), np.isin(frame, pose_frames)
    with np.errstate(invalid="ignore"):
        usable = has_image & has_scan & has_pose & (window >= 0) & (future >= min_future_m) & aligned
    return {
        "frame": frame, "has_image": has_image, "has_scan": has_scan, "has_pose": has_pose,
        "has_sem2d": np.isin(frame, np.asarray(sem2d_frames, dtype=np.int64)),
        "window_start": window_start, "window_end": window_end,
        "t_image": t_image, "t_scan": t_scan, "dt_ms": dt_ms, "aligned": aligned,
        "odometer_m": odometer, "future_path_m": future, "x_world": position[:, 0], "y_world": position[:, 1],
        "n_points_raw": n_raw, "n_points_crop": n_crop,
        "usable": usable, "sample_5m": sample_every(odometer, usable, step_m),
    }


def write_csv(path, index: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [index[c] for c in COLUMNS]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in zip(*columns):
            writer.writerow([_fmt(name, v) for name, v in zip(COLUMNS, row)])


def _fmt(name: str, value) -> str:
    if name in BOOL_COLUMNS:
        return "1" if value else "0"
    if name in INT_COLUMNS:
        return str(int(value))
    return "nan" if np.isnan(value) else f"{value:.6f}"


def read_csv(path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    out = {}
    for j, name in enumerate(header):
        values = [r[j] for r in body]
        if name in BOOL_COLUMNS:
            out[name] = np.array([v == "1" for v in values], dtype=bool)
        elif name in INT_COLUMNS:
            out[name] = np.array(values, dtype=np.int64)
        else:
            out[name] = np.array(values, dtype=np.float64)
    return out


def select(index: dict[str, np.ndarray], mask) -> dict[str, np.ndarray]:
    """Rows of the index where mask is true."""
    return {k: v[mask] for k, v in index.items()}
