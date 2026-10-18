"""KITTI-360 3D bounding boxes (data_3d_bboxes/train/<sequence>.xml). Evaluation data: never a model input.

Each XML object maps a unit cube (corners at +-0.5) into world coordinates with a 4x4 'transform' = [R·diag(l, w, h) | t]:
column lengths are the size, the first column is the heading. 'semanticId' is KITTI-360's kittiId.
Static objects (timestamp -1) are valid from start_frame to end_frame. Moving objects have one row per annotated frame;
gaps of up to max_gap frames between rows are filled by interpolation, longer gaps are left empty.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..geometry.transforms import invert_rigid

# kittiId -> label name, from kitti360Scripts helpers/labels.py (autonomousvision/kitti360Scripts, MIT)
KITTI_ID_NAMES = {
    11: "building", 12: "garage", 13: "car", 14: "truck", 15: "trailer", 16: "caravan", 17: "motorcycle",
    18: "bicycle", 19: "person", 20: "rider", 21: "pole", 22: "smallpole", 23: "traffic light", 24: "traffic sign",
    25: "lamp", 26: "trash bin", 27: "vending machine", 28: "box", 29: "stop", 33: "train", 34: "bus",
    35: "unknown construction", 36: "unknown vehicle", 37: "unknown object",
}
CLASSES = ("car", "large_vehicle", "pedestrian", "cyclist", "other")
_CLASS_OF = {"car": "car", "truck": "large_vehicle", "bus": "large_vehicle", "caravan": "large_vehicle",
             "trailer": "large_vehicle", "train": "large_vehicle", "person": "pedestrian", "rider": "cyclist"}
TWO_WHEELERS = ("bicycle", "motorcycle")
RIDER_DISTANCE_M = 1.0      # a two-wheeler with a rider this close (BEV centre distance) is part of a cyclist


@dataclass(frozen=True)
class Object3D:
    kitti_id: int
    instance_id: int
    name: str
    timestamp: int              # annotated frame of a moving object; -1 for static objects
    start_frame: int
    end_frame: int
    transform: np.ndarray       # [4, 4] unit cube -> world


def _matrix(node) -> np.ndarray:
    rows, cols = int(node.findtext("rows")), int(node.findtext("cols"))
    return np.array(node.findtext("data").split(), dtype=np.float64).reshape(rows, cols)


def parse_bboxes(path) -> list[Object3D]:
    objects = []
    for node in ET.parse(path).getroot():
        kitti_id = int(node.findtext("semanticId"))
        objects.append(Object3D(kitti_id, int(node.findtext("instanceId")), KITTI_ID_NAMES.get(kitti_id, f"kittiId {kitti_id}"),
                                int(node.findtext("timestamp")), int(node.findtext("start_frame")),
                                int(node.findtext("end_frame")), _matrix(node.find("transform"))))
    return objects


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def interpolate_transform(Ma, Mb, alpha: float) -> np.ndarray:
    """Between two unit-cube transforms: centre and size linearly, heading along the shorter arc."""
    Sa, Sb = np.linalg.norm(Ma[:3, :3], axis=0), np.linalg.norm(Mb[:3, :3], axis=0)
    Ra, Rb = Ma[:3, :3] / Sa, Mb[:3, :3] / Sb
    turn = np.arctan2(Rb[1, 0], Rb[0, 0]) - np.arctan2(Ra[1, 0], Ra[0, 0])
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    M = np.eye(4)
    M[:3, :3] = _rot_z(alpha * turn) @ Ra * ((1 - alpha) * Sa + alpha * Sb)
    M[:3, 3] = (1 - alpha) * Ma[:3, 3] + alpha * Mb[:3, 3]
    return M


@dataclass
class FrameBoxes:
    """Boxes valid at one frame, in that frame's LiDAR frame."""
    params: np.ndarray          # [K, 7] x, y, z, l, w, h, yaw
    names: list
    classes: np.ndarray         # [K] index into CLASSES
    kitti_ids: np.ndarray
    instance_ids: np.ndarray
    dynamic: np.ndarray         # [K] bool
    interpolated: np.ndarray    # [K] bool
    rider_of: np.ndarray        # [K] for riders: index of the two-wheeler they ride, else -1

    def __len__(self) -> int:
        return len(self.params)

    def class_names(self) -> list[str]:
        return [CLASSES[c] for c in self.classes]


class Boxes3D:
    def __init__(self, objects: list[Object3D]):
        self.static = [o for o in objects if o.timestamp < 0]
        self._ranges = np.array([(o.start_frame, o.end_frame) for o in self.static], dtype=np.int64).reshape(-1, 2)
        tracks = defaultdict(list)
        for o in objects:
            if o.timestamp >= 0:
                tracks[(o.kitti_id, o.instance_id)].append(o)
        self.tracks = {k: sorted(v, key=lambda o: o.timestamp) for k, v in tracks.items()}
        self._times = {k: np.array([o.timestamp for o in v]) for k, v in self.tracks.items()}

    @classmethod
    def from_xml(cls, path) -> "Boxes3D":
        return cls(parse_bboxes(path))

    def world_boxes(self, frame: int, max_gap: int = 5) -> list[tuple[Object3D, np.ndarray, bool]]:
        """(object, unit-cube -> world transform, interpolated) for every object valid at frame."""
        valid = np.flatnonzero((self._ranges[:, 0] <= frame) & (frame <= self._ranges[:, 1]))
        out = [(self.static[i], self.static[i].transform, False) for i in valid]
        for key, track in self.tracks.items():
            times = self._times[key]
            i = int(np.searchsorted(times, frame))
            if i < len(times) and times[i] == frame:
                out.append((track[i], track[i].transform, False))
            elif 0 < i < len(times) and times[i] - times[i - 1] <= max_gap:
                a, b = track[i - 1], track[i]
                alpha = (frame - a.timestamp) / (b.timestamp - a.timestamp)
                out.append((a, interpolate_transform(a.transform, b.transform, alpha), True))
        return out

    def at(self, frame: int, T_world_lidar, max_gap: int = 5) -> FrameBoxes:
        """Boxes valid at frame, as (x, y, z, l, w, h, yaw) in the LiDAR frame of that frame, with classes."""
        found = self.world_boxes(frame, max_gap)
        T_lidar_world = invert_rigid(np.asarray(T_world_lidar, dtype=np.float64))
        params = np.zeros((len(found), 7))
        for k, (_, M, _) in enumerate(found):
            T = T_lidar_world @ M
            params[k, :3] = T[:3, 3]
            params[k, 3:6] = np.linalg.norm(T[:3, :3], axis=0)
            params[k, 6] = np.arctan2(T[1, 0], T[0, 0])
        names = [o.name for o, _, _ in found]
        classes = np.array([CLASSES.index(_CLASS_OF.get(n, "other")) for n in names], dtype=np.int64)
        rider_of = np.full(len(found), -1, dtype=np.int64)
        riders = [k for k, n in enumerate(names) if n == "rider"]
        for k, n in enumerate(names):
            if n in TWO_WHEELERS and riders:
                d = np.hypot(*(params[riders, :2] - params[k, :2]).T)
                if d.min() < RIDER_DISTANCE_M:
                    classes[k] = CLASSES.index("cyclist")
                    rider_of[riders[int(np.argmin(d))]] = k
        return FrameBoxes(params, names, classes,
                          np.array([o.kitti_id for o, _, _ in found], dtype=np.int64),
                          np.array([o.instance_id for o, _, _ in found], dtype=np.int64),
                          np.array([o.timestamp >= 0 for o, _, _ in found], dtype=bool),
                          np.array([interp for _, _, interp in found], dtype=bool), rider_of)
