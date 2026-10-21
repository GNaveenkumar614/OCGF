"""KITTI 3D Object frames: image, LiDAR points, calibration and labels, with KITTI classes mapped to ours.

Labels are converted to our LiDAR-frame box convention (x, y, z, l, w, h, yaw): the centre (KITTI gives the bottom
centre in rectified camera-0 coordinates, so z += h / 2 after the transform) and yaw = -ry - pi/2 (as in MTrans).

Class mapping (draft in CLAUDE.md, "Class mapping KITTI <-> KITTI-360"): car <- Car, Van; large_vehicle <- Truck, Tram;
pedestrian <- Pedestrian, Person_sitting; cyclist <- Cyclist. Misc and DontCare have no class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..geometry.calib import CameraCalib, kitti_object_camera, kitti_object_rect_from_lidar
from ..geometry.transforms import invert_rigid, transform_points
from .ids import frame_id, parse_frame_id

KITTI_TO_OURS = {"Car": "car", "Van": "car", "Truck": "large_vehicle", "Tram": "large_vehicle",
                 "Pedestrian": "pedestrian", "Person_sitting": "pedestrian", "Cyclist": "cyclist"}


def kitti_fid(frame) -> str:
    return frame_id("kitti", "training", int(frame))


def label_to_box(fields, T_lidar_rect) -> np.ndarray:
    """One KITTI label line split into fields -> box (x, y, z, l, w, h, yaw) in the LiDAR frame."""
    h, w, l = (float(v) for v in fields[8:11])
    bottom = transform_points(T_lidar_rect, np.array([[float(v) for v in fields[11:14]]]))[0]
    return np.array([bottom[0], bottom[1], bottom[2] + h / 2, l, w, h, -float(fields[14]) - np.pi / 2])


def read_labels(label_file, calib_file) -> list[dict]:
    """Objects of our classes in a KITTI label file, as dicts with class, kitti_class, truncated, occluded, box_2d
    (l, t, r, b) and box (LiDAR frame)."""
    T_lidar_rect = invert_rigid(kitti_object_rect_from_lidar(calib_file))
    objects = []
    for line in Path(label_file).read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] not in KITTI_TO_OURS:
            continue
        objects.append({"class": KITTI_TO_OURS[fields[0]], "kitti_class": fields[0], "truncated": float(fields[1]),
                        "occluded": int(float(fields[2])), "box_2d": np.array([float(v) for v in fields[4:8]]),
                        "box": label_to_box(fields, T_lidar_rect)})
    return objects


def stage_kitti(folders: dict, fids, dst, threads: int = 32, log=print) -> dict:
    """Copy image_2, calib and label_2 of the given frames to a local folder with parallel readers; returns folders
    for KittiObjectFrames (velodyne stays where it is: ProjectedFrames caches the in-image points instead).

    On Kaggle, files under /kaggle/input are slow to read one after another (about 5 s per frame for a scan and an
    image header on 2026-10-20) but parallel reads add up, so a run copies its inputs once and then reads local files.
    Files already copied are skipped, so an interrupted copy resumes.
    """
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor

    dst = Path(dst)
    suffix = {"image_2": ".png", "calib": ".txt", "label_2": ".txt"}
    jobs = []
    for name, ext in suffix.items():
        (dst / name).mkdir(parents=True, exist_ok=True)
        for fid in fids:
            target = dst / name / f"{parse_frame_id(fid)[2]:06d}{ext}"
            if not target.exists():
                jobs.append((Path(folders[name]) / target.name, target))

    def copy(job):
        src, target = job
        tmp = target.with_name(target.name + ".part")
        shutil.copyfile(src, tmp)
        tmp.replace(target)
        return target.stat().st_size

    t0, done, size = time.time(), 0, 0
    with ThreadPoolExecutor(threads) as pool:
        for n in pool.map(copy, jobs):
            done, size = done + 1, size + n
            if log and (done % 2000 == 0 or done == len(jobs)):
                rate = size / 1e6 / max(time.time() - t0, 1e-6)
                log(f"  staged {done}/{len(jobs)} files, {size / 1e9:.2f} GB, {rate:.1f} MB/s")
    return {"image_2": dst / "image_2", "calib": dst / "calib", "label_2": dst / "label_2",
            "velodyne": Path(folders["velodyne"])}


class KittiObjectFrames:
    """Frames of KITTI's 'training' folder (which holds both the train and the val split), by canonical frame id."""

    def __init__(self, folders: dict):
        self.folders = {k: Path(v) for k, v in folders.items()}

    def _file(self, folder: str, fid: str, suffix: str) -> Path:
        return self.folders[folder] / f"{parse_frame_id(fid)[2]:06d}{suffix}"

    def image_size(self, fid: str) -> tuple[int, int]:
        with Image.open(self._file("image_2", fid, ".png")) as im:
            return im.size                                      # (width, height), from the header only

    def image(self, fid: str) -> np.ndarray:
        return np.asarray(Image.open(self._file("image_2", fid, ".png")).convert("RGB"))

    def points(self, fid: str) -> np.ndarray:
        """[N, 4] float32: x, y, z, intensity in the LiDAR frame."""
        return np.fromfile(self._file("velodyne", fid, ".bin"), dtype=np.float32).reshape(-1, 4)

    def camera(self, fid: str) -> CameraCalib:
        return kitti_object_camera(self._file("calib", fid, ".txt"), *self.image_size(fid))

    def labels(self, fid: str) -> list[dict]:
        return read_labels(self._file("label_2", fid, ".txt"), self._file("calib", fid, ".txt"))
