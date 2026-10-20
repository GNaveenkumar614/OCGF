"""A small fake KITTI 3D Object 'training' folder: images, scans, calibration and labels of known LiDAR-frame boxes.

Labels are written with an independent forward conversion (LiDAR -> rectified camera 0 with R0_rect · Tr_velo_to_cam,
bottom centre, ry = -yaw - pi/2) and 2D boxes from projecting the corners with P2, which has a camera-2 offset, so a
reader that mixes up camera 0 and camera 2 fails the tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ocgf.geometry.boxes import box_corners

W, H = 1242, 375
K = np.array([[700.0, 0.0, 620.0], [0.0, 700.0, 187.0], [0.0, 0.0, 1.0]])
P2 = np.column_stack([K, K @ np.array([0.06, 0.0, 0.0])])                 # camera 2: 6 cm from camera 0
_a = np.radians(0.7)
R0 = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(_a), -np.sin(_a)], [0.0, np.sin(_a), np.cos(_a)]])
TR = np.array([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.08], [1.0, 0.0, 0.0, -0.27]])      # LiDAR -> camera 0

# (KITTI class, box x y z l w h yaw in the LiDAR frame)
DEFAULT_OBJECTS = [("Car", (15.0, 2.0, -0.95, 4.0, 1.7, 1.5, 0.3)),
                   ("Pedestrian", (10.0, -3.0, -0.85, 0.7, 0.6, 1.7, -1.2)),
                   ("Cyclist", (20.0, -1.0, -0.9, 1.8, 0.6, 1.7, 2.5)),
                   ("Truck", (28.0, 5.0, -0.2, 9.0, 2.5, 3.2, 0.1))]


def _fmt(values) -> str:
    return " ".join(f"{v:.6e}" for v in np.ravel(values))


def T_rect_lidar() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R0 @ TR[:, :3], R0 @ TR[:, 3]
    return T


def label_line(kitti_class: str, box) -> str:
    x, y, z, l, w, h, yaw = box
    bottom = T_rect_lidar() @ np.array([x, y, z - h / 2, 1.0])
    ry = -yaw - np.pi / 2
    ry = ry - 2 * np.pi * np.ceil((ry - np.pi) / (2 * np.pi))
    rect = (T_rect_lidar() @ np.column_stack([box_corners([box])[0], np.ones(8)]).T)
    uvw = P2 @ rect
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    l2d, t2d = max(0.0, u.min()), max(0.0, v.min())
    r2d, b2d = min(W - 1.0, u.max()), min(H - 1.0, v.max())
    return (f"{kitti_class} 0.00 0 0.00 {l2d:.2f} {t2d:.2f} {r2d:.2f} {b2d:.2f} "
            f"{h:.2f} {w:.2f} {l:.2f} {bottom[0]:.4f} {bottom[1]:.4f} {bottom[2]:.4f} {ry:.4f}")


def object_points(box, n: int, rng) -> np.ndarray:
    x, y, z, l, w, h, yaw = box
    local = (rng.random((n, 3)) - 0.5) * np.array([l, w, h]) * 0.95
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1], z + local[:, 2]])


def make_fake_kitti_object(root, frames=None, points_per_object: int = 300, seed: int = 0) -> dict:
    """frames: {frame number: [(KITTI class, box), ...]} (default: three frames of DEFAULT_OBJECTS).
    Returns the {folder: path} dict that KittiObjectFrames takes."""
    rng = np.random.default_rng(seed)
    frames = frames if frames is not None else {3: DEFAULT_OBJECTS, 5: DEFAULT_OBJECTS[:2], 8: DEFAULT_OBJECTS[1:]}
    training = Path(root) / "kitti" / "training"
    folders = {name: training / name for name in ("image_2", "velodyne", "calib", "label_2")}
    for p in folders.values():
        p.mkdir(parents=True, exist_ok=True)
    calib = "\n".join([f"P0: {_fmt(np.column_stack([K, np.zeros(3)]))}", f"P1: {_fmt(np.column_stack([K, np.zeros(3)]))}",
                       f"P2: {_fmt(P2)}", f"P3: {_fmt(P2)}", f"R0_rect: {_fmt(R0)}", f"Tr_velo_to_cam: {_fmt(TR)}",
                       f"Tr_imu_to_velo: {_fmt(TR)}"]) + "\n"
    gx, gy = np.meshgrid(np.arange(4.0, 40.0, 0.5), np.arange(-15.0, 15.0, 0.5))
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -1.73)])
    for frame, objects in frames.items():
        name = f"{frame:06d}"
        Image.fromarray(rng.integers(20, 235, (H, W, 3), dtype=np.uint8)).save(folders["image_2"] / f"{name}.png")
        pts = [ground] + [object_points(box, points_per_object, rng) for _, box in objects]
        pts.append(np.column_stack([rng.uniform(-20, 60, 500), rng.uniform(-30, 30, 500), rng.uniform(-2, 3, 500)]))
        xyz = np.concatenate(pts)
        np.column_stack([xyz, rng.random(len(xyz))]).astype(np.float32).tofile(folders["velodyne"] / f"{name}.bin")
        (folders["calib"] / f"{name}.txt").write_text(calib)
        lines = [label_line(c, box) for c, box in objects] + ["DontCare -1 -1 -10 500.0 150.0 520.0 170.0 -1 -1 -1 -1000 -1000 -1000 -10"]
        (folders["label_2"] / f"{name}.txt").write_text("\n".join(lines) + "\n")
    return folders


def write_detections(path, frames: dict, labels_dir, shift: float = 3.0) -> Path:
    """Detection lines (our classes) from the label 2D boxes of the given frames, shifted by a few pixels: stand-ins for
    zero-shot detections, so tests can tell detector boxes from label boxes."""
    from ocgf.datasets.kitti_object import KITTI_TO_OURS
    rows = []
    for frame in frames:
        for line in (Path(labels_dir) / f"{frame:06d}.txt").read_text().splitlines():
            f = line.split()
            if f and f[0] in KITTI_TO_OURS:
                l, t, r, b = (float(v) + shift for v in f[4:8])
                rows.append(f"kitti/training/{frame:010d} {KITTI_TO_OURS[f[0]]} 0.9000 {l:.2f} {t:.2f} {r:.2f} {b:.2f}")
    Path(path).write_text("\n".join(rows) + "\n")
    return Path(path)
