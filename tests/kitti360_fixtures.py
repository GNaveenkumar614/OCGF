"""Synthetic KITTI-360 data for tests: scans with real ring structure and a small fake input tree.

All numbers are made up; they only need the right formats and plausible geometry.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ocgf.datasets.ply import write_ply
from ocgf.datasets.velodyne import N_RINGS
from ocgf.geometry.transforms import rigid

from geometry_fixtures import AXES_CAM_LIDAR

SEQ = "2013_05_28_drive_0000_sync"


def synthetic_scan(points_per_ring: int = 360, seed: int = 0):
    """A scan stored ring by ring, top laser first; each ring sweeps counter-clockwise from just left of forward
    to just right of it. Returns points [N, 4] and the true ring (0 = lowest laser = last ring stored)."""
    rng = np.random.default_rng(seed)
    points, rings = [], []
    for stored in range(N_RINGS):
        true_ring = N_RINGS - 1 - stored
        elevation = np.radians(-24.9 + 26.9 * true_ring / (N_RINGS - 1))
        azimuth = np.linspace(0.01, 2 * np.pi - 0.01, points_per_ring)
        r = rng.uniform(5, 60, points_per_ring)
        points.append(np.column_stack([r * np.cos(azimuth), r * np.sin(azimuth), r * np.tan(elevation),
                                       rng.uniform(0, 1, points_per_ring)]))
        rings.append(np.full(points_per_ring, true_ring))
    return np.vstack(points).astype(np.float32), np.concatenate(rings)


def fmt(values) -> str:
    return " ".join(f"{v:.12e}" for v in np.ravel(values))


def timestamp_lines(times_s) -> list[str]:
    base = np.datetime64("2013-05-28T08:46:00", "ns")
    return [str(base + np.timedelta64(int(round(t * 1e9)), "ns")).replace("T", " ") for t in times_s]


CLOUD = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                  ("semantic", "<i4"), ("instance", "<i4"), ("visible", "u1"), ("confidence", "<f4")])


def cloud(n: int, x_range, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = np.zeros(n, dtype=CLOUD)
    c["x"], c["y"], c["z"] = rng.uniform(*x_range, n), rng.uniform(-30, 30, n), rng.uniform(-2, 5, n)
    c["semantic"] = rng.integers(0, 45, n)
    return c


def make_fake_kitti360(root, n_frames: int = 40, missing_scans=(7,), first_pose: int = 2, step_m: float = 2.0,
                       timestamps_in_velodyne_zip: bool = True) -> dict:
    """Write a fake starter, 2D semantics, boxes and velodyne zips under root. Returns their paths."""
    root = Path(root)
    k360 = root / "input" / "kitti-360-prepared-dataset" / "KITTI360"
    frames = np.arange(n_frames)

    calib = k360 / "calibration"
    calib.mkdir(parents=True)
    P = np.array([[552.0, 0.0, 682.0, 0.0], [0.0, 552.0, 238.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    (calib / "perspective.txt").write_text("\n".join([
        "calib_time: 09-Jan-2012 13:57:47", "S_rect_00: 1.408000e+03 3.760000e+02",
        "R_rect_00: " + fmt(np.eye(3)), "P_rect_00: " + fmt(P)]) + "\n")
    (calib / "calib_cam_to_velo.txt").write_text(fmt(rigid(AXES_CAM_LIDAR.T, [0.8, 0.3, -0.2])[:3]) + "\n")
    (calib / "calib_cam_to_pose.txt").write_text("image_00: " + fmt(rigid(np.eye(3), [1.5, 0.0, 0.9])[:3]) + "\n")

    image_times = frames * 0.1
    img = k360 / "data_2d_raw" / SEQ / "image_00"
    (img / "data_rect").mkdir(parents=True)
    (img / "timestamps.txt").write_text("\n".join(timestamp_lines(image_times)) + "\n")
    blank = Image.new("RGB", (1408, 376), (60, 60, 60))
    for f in frames:
        blank.save(img / "data_rect" / f"{f:010d}.png")

    poses = k360 / "data_poses" / SEQ
    poses.mkdir(parents=True)
    pose_frames = frames[first_pose:]
    T = [rigid(np.eye(3), [step_m * f, 0.0, 0.0]) for f in pose_frames]
    (poses / "poses.txt").write_text("\n".join(f"{f} " + fmt(t[:3]) for f, t in zip(pose_frames, T)) + "\n")
    (poses / "cam0_to_world.txt").write_text("\n".join(f"{f} " + fmt(t) for f, t in zip(pose_frames, T)) + "\n")

    sem3d = k360 / "data_3d_semantics" / SEQ
    for part, windows in (("static", [(0, 25), (20, n_frames - 1)]), ("dynamic", [(0, 25)])):
        (sem3d / part).mkdir(parents=True)
        for i, (s, e) in enumerate(windows):
            write_ply(sem3d / part / f"{s:010d}_{e:010d}.ply", cloud(3000, (step_m * s - 80, step_m * e + 80), seed=i))

    sem2d = root / "input" / "kitti360-2d-semantics" / "data_2d_semantics" / "train" / SEQ / "image_00"
    for sub in ("semantic", "confidence"):
        (sem2d / sub).mkdir(parents=True)
        for f in frames[::3]:
            Image.new("L", (16, 8)).save(sem2d / sub / f"{f:010d}.png")

    boxes = root / "input" / "kitti360-3d-bboxes"
    boxes.mkdir(parents=True)
    (boxes / f"{SEQ}.xml").write_text("<?xml version=\"1.0\"?>\n<opencv_storage>\n</opencv_storage>\n")

    downloads = root / "downloads"
    downloads.mkdir()
    scan_lines = "\n".join(timestamp_lines(image_times + 0.004)) + "\n"
    velodyne_zip = downloads / f"{SEQ}_velodyne.zip"
    with zipfile.ZipFile(velodyne_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in frames:
            if f not in missing_scans:
                points, _ = synthetic_scan(points_per_ring=40, seed=int(f))
                z.writestr(f"{SEQ}/velodyne_points/data/{f:010d}.bin", points.tobytes())
        if timestamps_in_velodyne_zip:
            z.writestr(f"{SEQ}/velodyne_points/timestamps.txt", scan_lines)
    timestamps_zip = downloads / "data_timestamps_velodyne.zip"
    with zipfile.ZipFile(timestamps_zip, "w") as z:
        z.writestr(f"{SEQ}/velodyne_points/timestamps.txt", scan_lines)

    sources = root / "sources.yaml"
    sources.write_text(yaml.safe_dump({"velodyne": {SEQ: str(velodyne_zip)}, "velodyne_timestamps": str(timestamps_zip)}))
    return {"input_root": root / "input", "starter": k360, "velodyne_zip": velodyne_zip,
            "timestamps_zip": timestamps_zip, "sources": sources}
