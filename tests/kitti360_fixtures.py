"""Synthetic KITTI-360 data for tests: scans with real ring structure and a small fake input tree.

All numbers are made up; they only need the right formats and plausible geometry.
"""
from __future__ import annotations

import io
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


UNIT_CUBE = np.array([[x, y, z] for x in (0.5, -0.5) for y in (0.5, -0.5) for z in (0.5, -0.5)])
NAME_OF = {13: "car", 14: "truck", 18: "bicycle", 19: "pedestrian", 20: "rider", 37: "unknownObject"}


def cube_transform(center, size, yaw) -> np.ndarray:
    """Unit cube -> world, the way KITTI-360 stores a box: [R_z(yaw) · diag(l, w, h) | centre]."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rigid(R @ np.diag(size), center)


def bbox_xml(objects) -> str:
    """A data_3d_bboxes XML. objects: dicts with kitti_id, instance, timestamp, start, end, center, size, yaw."""
    parts = ['<?xml version="1.0"?>', "<opencv_storage>"]
    for i, o in enumerate(objects, 1):
        M = cube_transform(o["center"], o["size"], o["yaw"])
        parts.append(
            f"<object{i}><index>{i}</index><label>{NAME_OF.get(o['kitti_id'], 'other')}</label>"
            f"<semanticId_orig>0</semanticId_orig><semanticId>{o['kitti_id']}</semanticId>"
            f"<instanceId>{o['instance']}</instanceId><category>instance</category><timestamp>{o['timestamp']}</timestamp>"
            f"<dynamic>{int(o['timestamp'] >= 0)}</dynamic>"
            f'<transform type_id="opencv-matrix"><rows>4</rows><cols>4</cols><dt>f</dt><data>{fmt(M)}</data></transform>'
            f'<vertices type_id="opencv-matrix"><rows>8</rows><cols>3</cols><dt>f</dt><data>{fmt(UNIT_CUBE)}</data></vertices>'
            f'<faces type_id="opencv-matrix"><rows>12</rows><cols>3</cols><dt>u</dt><data>0 2 1 2 3 1 4 6 5 6 7 5 4 5 1 5 0 1 '
            f"7 6 2 6 3 2 5 7 0 7 2 0 1 3 4 3 6 4</data></faces>"
            f"<start_frame>{o['start']}</start_frame><end_frame>{o['end']}</end_frame></object{i}>")
    parts.append("</opencv_storage>")
    return "\n".join(parts) + "\n"


def _obj(kitti_id, instance, timestamp, start, end, center, size, yaw):
    return dict(kitti_id=kitti_id, instance=instance, timestamp=timestamp, start=start, end=end,
                center=center, size=size, yaw=yaw)


FAKE_OBJECTS = [
    _obj(13, 1, -1, 0, 25, (30.0, 3.0, 0.8), (4.2, 1.8, 1.5), 0.1),            # parked car
    _obj(37, 2, -1, 0, 39, (40.0, -6.0, 0.5), (0.6, 0.6, 1.0), 0.0),           # unknown object
    *[_obj(19, 3, t, 5, 20, (26.0 + 0.5 * t, -2.0, 0.9), (0.6, 0.6, 1.8), 0.0) for t in (8, 9, 10, 13, 20)],
    *[_obj(18, 4, t, 8, 12, (24.0, 4.0, 0.6), (1.8, 0.6, 1.1), np.pi / 2) for t in range(8, 13)],   # ridden bicycle
    *[_obj(20, 5, t, 8, 12, (24.1, 4.1, 1.2), (0.8, 0.6, 1.8), np.pi / 2) for t in range(8, 13)],   # its rider
]


def make_fake_kitti360(root, n_frames: int = 40, missing_scans=(7,), first_pose: int = 2, step_m: float = 2.0,
                       timestamps_in_velodyne_zip: bool = True, starter_images: bool = True, windows=None) -> dict:
    """Write a fake starter, 2D semantics, boxes, and velodyne / image zips under root. Returns their paths.

    starter_images=False leaves the images out of the starter, as for every drive but 0000: they then come from the
    image_00 zip. windows: static semantic windows (default: (0, 25) and (20, n_frames - 1)).
    """
    root = Path(root)
    k360 = root / "input" / "kitti-360-prepared-dataset" / "KITTI360"
    frames = np.arange(n_frames)
    windows = windows or [(0, 25), (20, n_frames - 1)]

    calib = k360 / "calibration"
    calib.mkdir(parents=True)
    P = np.array([[552.0, 0.0, 682.0, 0.0], [0.0, 552.0, 238.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    (calib / "perspective.txt").write_text("\n".join([
        "calib_time: 09-Jan-2012 13:57:47", "S_rect_00: 1.408000e+03 3.760000e+02",
        "R_rect_00: " + fmt(np.eye(3)), "P_rect_00: " + fmt(P)]) + "\n")
    (calib / "calib_cam_to_velo.txt").write_text(fmt(rigid(AXES_CAM_LIDAR.T, [0.8, 0.3, -0.2])[:3]) + "\n")
    # camera -> vehicle 'pose' axes (x forward, y left, z up), as in the real calibration
    (calib / "calib_cam_to_pose.txt").write_text("image_00: " + fmt(rigid(AXES_CAM_LIDAR.T, [1.5, 0.0, 0.9])[:3]) + "\n")

    image_times = frames * 0.1
    image_lines = "\n".join(timestamp_lines(image_times)) + "\n"
    blank = Image.new("RGB", (1408, 376), (60, 60, 60))
    png = io.BytesIO()
    blank.save(png, format="PNG")
    if starter_images:
        img = k360 / "data_2d_raw" / SEQ / "image_00"
        (img / "data_rect").mkdir(parents=True)
        (img / "timestamps.txt").write_text(image_lines)
        for f in frames:
            (img / "data_rect" / f"{f:010d}.png").write_bytes(png.getvalue())

    poses = k360 / "data_poses" / SEQ
    poses.mkdir(parents=True)
    pose_frames = frames[first_pose:]
    T = [rigid(np.eye(3), [step_m * f, 0.0, 0.0]) for f in pose_frames]
    (poses / "poses.txt").write_text("\n".join(f"{f} " + fmt(t[:3]) for f, t in zip(pose_frames, T)) + "\n")
    (poses / "cam0_to_world.txt").write_text("\n".join(f"{f} " + fmt(t) for f, t in zip(pose_frames, T)) + "\n")

    sem3d = k360 / "data_3d_semantics" / SEQ
    for part, part_windows in (("static", windows), ("dynamic", windows[:1])):
        (sem3d / part).mkdir(parents=True)
        for i, (s, e) in enumerate(part_windows):
            write_ply(sem3d / part / f"{s:010d}_{e:010d}.ply", cloud(3000, (step_m * s - 80, step_m * e + 80), seed=i))

    sem2d = root / "input" / "kitti360-2d-semantics" / "data_2d_semantics" / "train" / SEQ / "image_00"
    for sub in ("semantic", "confidence"):
        (sem2d / sub).mkdir(parents=True)
        for f in frames[::3]:
            Image.new("L", (16, 8)).save(sem2d / sub / f"{f:010d}.png")

    boxes = root / "input" / "kitti360-3d-bboxes"
    boxes.mkdir(parents=True)
    (boxes / f"{SEQ}.xml").write_text(bbox_xml(FAKE_OBJECTS))

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
    image_zip = downloads / f"{SEQ}_image_00.zip"                 # layout of the official image_00 zips
    with zipfile.ZipFile(image_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in frames[::-1]:                                     # archive order differs from frame order, as there
            z.writestr(f"{SEQ}/image_00/data_rect/{f:010d}.png", png.getvalue())
    image_timestamps_zip = downloads / "data_timestamps_perspective.zip"
    with zipfile.ZipFile(image_timestamps_zip, "w") as z:
        z.writestr(f"{SEQ}/image_00/timestamps.txt", image_lines)

    sources = root / "sources.yaml"
    sources.write_text(yaml.safe_dump({"velodyne": {SEQ: str(velodyne_zip)}, "velodyne_timestamps": str(timestamps_zip),
                                       "images_00": {SEQ: str(image_zip)}, "image_timestamps": str(image_timestamps_zip)}))
    return {"input_root": root / "input", "starter": k360, "velodyne_zip": velodyne_zip,
            "timestamps_zip": timestamps_zip, "image_zip": image_zip, "sources": sources}
