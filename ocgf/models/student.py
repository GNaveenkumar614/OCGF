"""The student: camera and/or LiDAR BEV features -> fusion -> U-Net decoder -> road, occupancy and unknown logits.

Modes (one code path, so the E0 baselines share every other part):
  camera_ipm  camera only, lifted by flat-plane IPM from the calibration; no LiDAR anywhere (E0a)
  camera      camera only, lifted onto the ground surface from the current scan (lifting ablations)
  lidar       LiDAR only (E0b)
  concat      camera (ground-guided) and LiDAR, concatenated and fused by a conv: the simple-fusion baseline (E0c)
The OCGF gate, object channels and reliability channels come later; F_C and F_L stay separate until the fusion step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera import CameraEncoder, lift, make_backbone
from .lidar import PillarEncoder

MODES = ("camera_ipm", "camera", "lidar", "concat")
HEADS = ("road", "occ", "unk")


@dataclass(frozen=True)
class StudentConfig:
    mode: str = "concat"
    channels: int = 64
    backbone: str = "dinov2_vits14_reg"      # "stub": a fixed random patch embedding (tests, CPU smoke runs)
    point_features: int = 10
    grid_shape: tuple = (200, 160)

    @classmethod
    def from_dict(cls, d: dict | None) -> StudentConfig:
        d = dict(d or {})
        if "grid_shape" in d:
            d["grid_shape"] = tuple(d["grid_shape"])
        cfg = cls(**d)
        if cfg.mode not in MODES:
            raise ValueError(f"mode {cfg.mode!r} is not one of {MODES}")
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grid_shape"] = list(self.grid_shape)
        return d


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Two downsamplings (200 x 160 -> 100 x 80 -> 50 x 40) and back, at channels c, 2c, 4c."""

    def __init__(self, cin: int, c: int = 64):
        super().__init__()
        self.enc0 = _block(cin, c)
        self.enc1 = nn.Sequential(nn.MaxPool2d(2), _block(c, 2 * c))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), _block(2 * c, 4 * c))
        self.dec1 = _block(4 * c + 2 * c, 2 * c)
        self.dec0 = _block(2 * c + c, c)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        d1 = self.dec1(torch.cat([F.interpolate(e2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.dec0(torch.cat([F.interpolate(d1, size=e0.shape[-2:], mode="bilinear", align_corners=False), e0], 1))


class Student(nn.Module):
    def __init__(self, cfg: StudentConfig = StudentConfig(), backbone: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        self.uses_camera = cfg.mode in ("camera_ipm", "camera", "concat")
        self.uses_lidar = cfg.mode in ("lidar", "concat")
        if self.uses_camera:
            self.camera = CameraEncoder(backbone if backbone is not None else make_backbone(cfg.backbone), c)
        if self.uses_lidar:
            self.lidar = PillarEncoder(cfg.point_features, c, cfg.grid_shape)
        cin = c * (int(self.uses_camera) + int(self.uses_lidar))
        self.fuse = (nn.Sequential(nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
                     if cin != c else nn.Identity())
        self.decoder = UNet(c, c)
        self.heads = nn.Conv2d(c, len(HEADS), 1)

    def forward(self, batch: dict) -> dict:
        feats = {}
        if self.uses_camera:
            img = self.camera(batch["image"])
            if self.cfg.mode == "camera_ipm":
                feats["F_C"] = lift(img, batch["grid_uv_flat"], batch["fov_flat"])
            else:
                feats["F_C"] = lift(img, batch["grid_uv"], batch["M_C"])
        if self.uses_lidar:
            feats["F_L"] = self.lidar(batch["pt_feats"], batch["pt_cell"], batch["grid_uv"].shape[0])
        x = self.fuse(torch.cat(list(feats.values()), dim=1))
        logits = self.heads(self.decoder(x))
        return {name: logits[:, k] for k, name in enumerate(HEADS)}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
