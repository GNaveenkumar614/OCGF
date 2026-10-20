"""Calibration for KITTI-360 and KITTI Object, reduced to one camera model.

Every camera is described by intrinsics K and a rigid transform T_cam_lidar into its rectified camera frame
(x right, y down, z forward), so that a LiDAR point p projects as  u ~ K · (T_cam_lidar · [p, 1])[:3].

- KITTI-360 (camera 00): T_cam_lidar = R_rect_00 · inv(cam_to_velo). The road-autolabeling repo omits R_rect_00.
- KITTI Object (camera 2): T_cam_lidar = R0_rect · Tr_velo_to_cam, with P2's 4th column folded in as a translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .transforms import invert_rigid, rigid, rigid_from_3x4


@dataclass(frozen=True)
class CameraCalib:
    K: np.ndarray            # [3, 3] intrinsics of the rectified image
    T_cam_lidar: np.ndarray  # [4, 4] LiDAR frame -> rectified camera frame
    width: int
    height: int

    def crop(self, left: int, top: int, width: int, height: int) -> CameraCalib:
        """Calibration of the window [left, left + width) x [top, top + height) of the image.

        Negative left/top or a window larger than the image means padding; padding never changes the intrinsics.
        """
        K = self.K.copy()
        K[0, 2] -= left
        K[1, 2] -= top
        return CameraCalib(K, self.T_cam_lidar, int(width), int(height))

    def resize(self, width: int, height: int) -> CameraCalib:
        """Calibration of the image resized to width x height (pixel centres at integer coordinates)."""
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        return CameraCalib(K, self.T_cam_lidar, int(width), int(height))

    @property
    def horizontal_fov_deg(self) -> float:
        fx, cx = self.K[0, 0], self.K[0, 2]
        return float(np.degrees(np.arctan((cx + 0.5) / fx) + np.arctan((self.width - 0.5 - cx) / fx)))


def read_calib_file(path) -> dict[str, np.ndarray]:
    """'key: v1 v2 ...' lines -> {key: float array}. Lines whose values are not all numbers (e.g. calib_time) are skipped."""
    values = {}
    for line in Path(path).read_text().splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        try:
            values[key.strip()] = np.array([float(v) for v in rest.split()], dtype=np.float64)
        except ValueError:
            continue
    return values


def split_projection(P) -> tuple[np.ndarray, np.ndarray]:
    """Split a rectified 3x4 projection P = K [I | t] into K and the translation t (zero for the reference camera)."""
    P = np.asarray(P, dtype=np.float64).reshape(3, 4)
    K = P[:, :3].copy()
    return K, np.linalg.solve(K, P[:, 3])


def kitti360_camera(calib_dir) -> CameraCalib:
    """Left perspective camera (image_00) of KITTI-360, from perspective.txt and calib_cam_to_velo.txt."""
    calib_dir = Path(calib_dir)
    perspective = read_calib_file(calib_dir / "perspective.txt")
    K, t = split_projection(perspective["P_rect_00"])
    R_rect = perspective["R_rect_00"].reshape(3, 3)
    width, height = (int(round(v)) for v in perspective["S_rect_00"])
    T_velo_cam = rigid_from_3x4(np.loadtxt(calib_dir / "calib_cam_to_velo.txt"))   # unrectified camera 0 -> Velodyne
    T_cam_lidar = rigid(np.eye(3), t) @ rigid(R_rect, np.zeros(3)) @ invert_rigid(T_velo_cam)
    return CameraCalib(K, T_cam_lidar, width, height)


def kitti360_pose_from_lidar(calib_dir) -> np.ndarray:
    """T_pose_lidar: Velodyne -> the GPS/IMU 'pose' frame of poses.txt, via the unrectified camera 0."""
    calib_dir = Path(calib_dir)
    T_pose_cam = rigid_from_3x4(read_calib_file(calib_dir / "calib_cam_to_pose.txt")["image_00"])
    T_velo_cam = rigid_from_3x4(np.loadtxt(calib_dir / "calib_cam_to_velo.txt"))
    return T_pose_cam @ invert_rigid(T_velo_cam)


def kitti_object_rect_from_lidar(calib_file) -> np.ndarray:
    """T_rect_lidar = R0_rect · Tr_velo_to_cam: LiDAR -> rectified camera 0, the frame KITTI label locations are in.

    Not kitti_object_camera().T_cam_lidar, which also holds P2's offset to camera 2 (about 6 cm) for projecting pixels.
    """
    calib = read_calib_file(calib_file)
    return rigid(calib["R0_rect"].reshape(3, 3), np.zeros(3)) @ rigid_from_3x4(calib["Tr_velo_to_cam"])


def kitti_object_camera(calib_file, width: int, height: int) -> CameraCalib:
    """Left colour camera (image_2) of a KITTI Object frame. Image sizes vary per frame, so they are passed in."""
    calib = read_calib_file(calib_file)
    K, t = split_projection(calib["P2"])
    R0 = calib["R0_rect"].reshape(3, 3)
    T_cam_velo = rigid_from_3x4(calib["Tr_velo_to_cam"])
    T_cam_lidar = rigid(np.eye(3), t) @ rigid(R0, np.zeros(3)) @ T_cam_velo
    return CameraCalib(K, T_cam_lidar, int(width), int(height))
