"""KITTI-360 sequence access through a dataset map.

frame() returns only what the model may see at inference (image, calibration, points) plus the world pose, which
training uses for temporal consistency and evaluation for ground truth. Ground truth (gt_boxes) is a separate call:
it is for evaluation and QA only, never a model input.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

from ..geometry.calib import CameraCalib, kitti360_camera, kitti360_pose_from_lidar
from . import frame_index as fi
from .boxes3d import Boxes3D, FrameBoxes
from .dataset_map import read_dataset_map
from .ids import frame_id
from .supplement_io import ShardReader

DATASET = "kitti360"


@dataclass
class Frame:
    fid: str
    image: np.ndarray               # [H, W, 3] uint8, rectified left camera
    camera: CameraCalib             # K and T_cam_lidar (includes R_rect_00)
    points: np.ndarray              # [N, 5] float32: x, y, z, intensity, ring in the LiDAR frame
    T_world_lidar: np.ndarray       # [4, 4]; training/evaluation only, never a model input


class Kitti360Sequence:
    def __init__(self, dataset_map=None, *, sequence: str | None = None, components: dict | None = None):
        """From a dataset_map.yaml, or from (sequence, components), e.g. ocgf.datasets.locate.prepared_drive()."""
        if dataset_map is not None:
            self.sequence, self.paths = read_dataset_map(Path(dataset_map))
        else:
            self.sequence, self.paths = sequence, {k: Path(v) for k, v in components.items()}
        self.camera = kitti360_camera(self.paths["calibration"])
        self.T_pose_lidar = kitti360_pose_from_lidar(self.paths["calibration"])
        frames, T_world_pose = fi.read_poses(self.paths["poses"] / "poses.txt")
        self._T_world_lidar = dict(zip(frames.tolist(), T_world_pose @ self.T_pose_lidar))
        self._scans = ShardReader(self.paths["scans"])
        # drive 0000 reads its images from the starter folder; the other paper drives from image shards
        self._image_shards = ShardReader(self.paths["image_shards"]) if "image_shards" in self.paths else None

    @cached_property
    def index(self) -> dict[str, np.ndarray]:
        return fi.read_csv(self.paths["frame_index"])

    @cached_property
    def _boxes(self) -> Boxes3D:
        return Boxes3D.from_xml(self.paths["boxes3d"])

    def fid(self, frame: int) -> str:
        return frame_id(DATASET, self.sequence, frame)

    def usable_frames(self, sampled: bool = False) -> np.ndarray:
        """Usable frames whose data is stored (all of drive 0000; every 2.5 m on the other paper drives)."""
        mask = self.index["usable"] & self.index.get("kept", True) & (self.index["sample_5m"] if sampled else True)
        return self.index["frame"][mask]

    def scan_frames(self) -> list[int]:
        return self._scans.frames()

    def image(self, frame: int) -> np.ndarray:
        if self._image_shards is not None:
            data = io.BytesIO(self._image_shards.read_bytes(frame, ".png"))
        else:
            data = self.paths["images"] / f"{frame:010d}.png"
        return np.asarray(Image.open(data).convert("RGB"))

    def points(self, frame: int) -> np.ndarray:
        return self._scans.read(frame)

    def T_world_lidar(self, frame: int) -> np.ndarray:
        if frame not in self._T_world_lidar:
            raise KeyError(f"{self.fid(frame)} has no pose")
        return self._T_world_lidar[frame]

    def frame(self, frame: int) -> Frame:
        return Frame(self.fid(frame), self.image(frame), self.camera, self.points(frame), self.T_world_lidar(frame))

    def gt_boxes(self, frame: int, max_gap: int = 5) -> FrameBoxes:
        """KITTI-360 ground-truth boxes in the frame's LiDAR frame. Evaluation and QA only."""
        return self._boxes.at(frame, self.T_world_lidar(frame), max_gap)
