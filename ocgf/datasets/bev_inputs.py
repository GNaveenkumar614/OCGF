"""The student's geometric inputs on the BEV grid, from the current scan and the calibration only.

Everything here is available at inference, so it is computed again on a perturbed scan in the robustness tests (E5:
fewer rings, dropped points). Ground truth never enters.

Layers (200 x 160, indexed [ix, iy]):
  z_g, r_g        ground height and its reliability (ocgf.geometry.ground)
  n_pts           points in the cell below occluder_max above ground
  ground_hits     points within ground_band of z_g (ground returns)
  h_max           highest point below occluder_max above ground (NaN: no point)
  n_above         points occluder_min..occluder_max above ground (obstacle evidence)
  occ_top         top of what stands occluder_min..occluder_max above ground (NaN: nothing); the viewshed occluders
  fov_cam         the cell's ground point projects into the image
  M_C             camera validity: fov_cam and in line of sight of the camera centre
  shadow_cam      fov_cam but hidden behind an occluder
  M_L             LiDAR validity: the cell has ground returns
  vis_lidar       the ground is in line of sight of the LiDAR and above its lowest beam
  shadow_lidar    above the lowest beam but hidden behind an occluder
Occluders are columns, as in the evaluation ground truth: canopies above occluder_max are left out.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..geometry.bev_grid import BEVGrid
from ..geometry.ground import GroundConfig, ground_surface, height_above_ground
from ..geometry.projection import project
from ..geometry.transforms import invert_rigid
from ..geometry.viewshed import above_lowest_beam, viewshed

LAYERS = ("z_g", "r_g", "n_pts", "ground_hits", "h_max", "n_above", "occ_top", "fov_cam", "M_C", "shadow_cam", "M_L",
          "vis_lidar", "shadow_lidar")


@dataclass(frozen=True)
class BEVInputConfig:
    ground: GroundConfig = field(default_factory=GroundConfig)
    ground_band: float = 0.15           # m around z_g: a ground return
    occluder_min: float = 0.3           # m above ground: lower things (kerbs) do not hide the ground
    occluder_max: float = 3.0           # m above ground: as in the evaluation ground truth
    lidar_min_elev_deg: float = -20.3   # the LiDAR's first ground return, about 4.7 m ahead (drive 0000)
    az_res_deg: float = 0.25

    @classmethod
    def from_dict(cls, d: dict | None) -> BEVInputConfig:
        d = dict(d or {})
        d["ground"] = GroundConfig.from_dict(d.get("ground"))
        return cls(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ground"] = self.ground.to_dict()
        return d


def bev_inputs(points, camera, grid: BEVGrid = BEVGrid(), cfg: BEVInputConfig = BEVInputConfig()) -> dict:
    """The layers above for one scan [N, >= 3] (LiDAR frame) and the camera calibration (CameraCalib)."""
    nx, ny = grid.shape
    z_g, r_g = ground_surface(points, grid, cfg.ground)
    inside, h = height_above_ground(points, z_g, grid)
    p = np.asarray(points)[inside]
    h = h[inside]
    ix, iy = grid.index(p[:, 0], p[:, 1])

    below = h < cfg.occluder_max
    n_pts = np.zeros((nx, ny), np.int32)
    np.add.at(n_pts, (ix[below], iy[below]), 1)
    hits = np.abs(h) <= cfg.ground_band
    ground_hits = np.zeros((nx, ny), np.int32)
    np.add.at(ground_hits, (ix[hits], iy[hits]), 1)
    h_max = np.full((nx, ny), -np.inf)
    np.maximum.at(h_max, (ix[below], iy[below]), h[below])
    h_max[~np.isfinite(h_max)] = np.nan
    occ = (h > cfg.occluder_min) & below
    n_above = np.zeros((nx, ny), np.int32)
    np.add.at(n_above, (ix[occ], iy[occ]), 1)
    occ_top = np.full((nx, ny), -np.inf)
    np.maximum.at(occ_top, (ix[occ], iy[occ]), p[occ, 2])
    occ_top[~np.isfinite(occ_top)] = np.nan

    xs, ys = grid.center_grid()
    _, _, fov = project(np.column_stack([xs.ravel(), ys.ravel(), z_g.ravel().astype(np.float64)]), camera)
    fov_cam = fov.reshape(nx, ny)
    cam_origin = invert_rigid(camera.T_cam_lidar)[:3, 3]
    vis_cam = viewshed(grid, cam_origin, z_g, occ_top, cfg.az_res_deg)
    origin = (0.0, 0.0, 0.0)
    beam = above_lowest_beam(grid, origin, z_g, cfg.lidar_min_elev_deg)
    los_lidar = viewshed(grid, origin, z_g, occ_top, cfg.az_res_deg)
    return {"z_g": z_g, "r_g": r_g, "n_pts": n_pts, "ground_hits": ground_hits,
            "h_max": h_max.astype(np.float32), "n_above": n_above, "occ_top": occ_top.astype(np.float32),
            "fov_cam": fov_cam, "M_C": fov_cam & vis_cam, "shadow_cam": fov_cam & ~vis_cam,
            "M_L": ground_hits > 0, "vis_lidar": beam & los_lidar, "shadow_lidar": beam & ~los_lidar}
