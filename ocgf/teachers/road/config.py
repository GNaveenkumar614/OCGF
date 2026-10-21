"""Settings of the road teacher. Values follow params_kitti360.yaml of lidar-camera-road-autolabeling (github.com/eerik98,
MIT; see LICENSE-lidar-camera-road-autolabeling), which its authors tuned on their own validation set."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class RoadConfig:
    # future path and its points in the scan (pre_processing)
    trajectory_length: float = 50.0        # m of future path
    lidar_fov_deg: float = 120.0           # points within +-60 deg of straight ahead are used
    drop_rings: int = 7                    # the 7 lowest beams land too close for the camera and are noisy
    distance_filter: float = 0.05          # m: each trajectory point must lie farther out than the previous one ...
    height_filter: float = 0.3             # m: ... and not more than this above it
    max_scan2pose: float = 1.0             # m between a scan point and the path for a match
    max_center2wheel: float = 2.0          # m between a trajectory point and its wheel points
    wheel_offset: float = 1.0              # m to each side of the path where the wheel points are looked for
    occlusion_px: float = 10.0             # px: nearer points in this column above a wheel point occlude it
    origin_center: tuple = (1.9, 0.0, -1.73)   # front-axle centre on the ground, LiDAR frame (KITTI-360)
    origin_left: tuple = (1.9, 0.8, -1.73)
    origin_right: tuple = (1.9, -0.8, -1.73)
    # LiDAR label (lidar_autolabeling)
    use_height: bool = True
    use_gradient: bool = True
    height_std: float = 0.1                # m
    gradient_std: float = 0.05             # m
    max_radial_dist: float = 5.0           # m: farther or nearer than this along the ring is not the road surface
    # camera label (camera_autolabeling)
    camera_model: str = "dinov2_vitg14_reg"
    camera_std: float = 0.5
    min_reference_pixels: int = 200        # fewer path pixels than this: reuse the previous frame's road prototype
    img_scale: float = 1.0

    @classmethod
    def from_dict(cls, values: dict) -> "RoadConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)
