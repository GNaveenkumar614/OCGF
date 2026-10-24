"""The student: camera and/or LiDAR BEV features -> fusion -> U-Net decoder -> road, occupancy and unknown logits.

Modes (one code path, so the baselines share every other part):
  camera_ipm  camera only, lifted by flat-plane IPM from the calibration; no LiDAR anywhere (E0a)
  camera      camera only, lifted onto the ground surface from the current scan (lifting ablations)
  lidar       LiDAR only (E0b)
  concat      camera (ground-guided) and LiDAR, concatenated and fused by a conv: the simple-fusion baseline (E0c)
  ocgf        camera and LiDAR fused per cell by the OCGF gate (ocgf.models.ocgf): fixed weights or a learned gate
              from the features, with the reliability channels Q (gate_q) and the object channels (gate_objects)
With objects, the object channels O_B (ocgf.datasets.objects, 9 channels) are concatenated to the fused features
before the decoder (E2's concat arm, and in ocgf mode the plan's decoder input concat[F_road, O_B]).

Reliability channels Q (8): the sample's Q_geo (log points, range, r_g, ground returns, camera shadow, camera
validity), the camera road score's entropy H_cam (detached, from the camera auxiliary head) and q_obj (O_B's object
confidence; 0 without objects). Auxiliary road heads on F_C and F_L (1x1 convs) are trained on the road target where
each sensor is valid; H_cam comes from the camera one, and both will serve the cross-modal consistency term.
F_C and F_L stay separate until the fusion step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera import CameraEncoder, lift, make_backbone
from .lidar import PillarEncoder
from .ocgf import Gate

N_OBJECT_CHANNELS = 9            # ocgf.datasets.objects.N_CHANNELS (not imported: models stay free of the data code)
N_Q_GEO = 6                      # ocgf.datasets.student_sample.Q_GEO
Q_CHANNELS = ("log_points", "range", "r_g", "M_L", "shadow_cam", "M_C", "H_cam", "q_obj")

MODES = ("camera_ipm", "camera", "lidar", "concat", "ocgf")
HEADS = ("road", "occ", "unk")


@dataclass(frozen=True)
class StudentConfig:
    mode: str = "concat"
    channels: int = 64
    backbone: str = "dinov2_vits14_reg"      # "stub": a fixed random patch embedding (tests, CPU smoke runs)
    point_features: int = 10
    grid_shape: tuple = (200, 160)
    objects: bool = False                    # object channels O_B concatenated before the decoder
    gate: str = "learned"                    # ocgf mode: "learned" or "fixed" (E4)
    gate_q: bool = True                      # ocgf mode: the learned gate sees Q (E4: without)
    gate_objects: bool = True                # ocgf mode with objects: the gate sees O_B (E2)

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


def binary_entropy(logits):
    """Entropy of sigmoid(logits) in [0, 1] (divided by ln 2)."""
    p = torch.sigmoid(logits.float()).clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)) / 0.6931471805599453


class Student(nn.Module):
    def __init__(self, cfg: StudentConfig = StudentConfig(), backbone: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        self.uses_camera = cfg.mode in ("camera_ipm", "camera", "concat", "ocgf")
        self.uses_lidar = cfg.mode in ("lidar", "concat", "ocgf")
        if self.uses_camera:
            self.camera = CameraEncoder(backbone if backbone is not None else make_backbone(cfg.backbone), c)
            self.aux_camera = nn.Conv2d(c, 1, 1)
        if self.uses_lidar:
            self.lidar = PillarEncoder(cfg.point_features, c, cfg.grid_shape)
            self.aux_lidar = nn.Conv2d(c, 1, 1)
        if cfg.mode == "ocgf":
            extra = (len(Q_CHANNELS) if cfg.gate_q else 0) + (N_OBJECT_CHANNELS if cfg.objects and cfg.gate_objects else 0)
            self.gate = Gate(c, cfg.gate, extra if cfg.gate == "learned" else 0)
            self.fuse = nn.Identity()
        else:
            cin = c * (int(self.uses_camera) + int(self.uses_lidar))
            self.fuse = (nn.Sequential(nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                                       nn.ReLU(inplace=True)) if cin != c else nn.Identity())
        self.decoder = UNet(c + (N_OBJECT_CHANNELS if cfg.objects else 0), c)
        self.heads = nn.Conv2d(c, len(HEADS), 1)

    def reliability(self, batch: dict, aux_camera_logits):
        """Q [B, 8, nx, ny]: Q_geo, H_cam (detached), q_obj."""
        q_geo = batch["Q_geo"].float()
        h_cam = binary_entropy(aux_camera_logits.detach())[:, None] * batch["M_C"][:, None]
        q_obj = batch["O_B"][:, 4:5].float() if self.cfg.objects else torch.zeros_like(h_cam)
        return torch.cat([q_geo, h_cam, q_obj], dim=1)

    def forward(self, batch: dict) -> dict:
        out, feats = {}, {}
        if self.uses_camera:
            img = self.camera(batch["image"])
            ipm = self.cfg.mode == "camera_ipm"
            valid_c = batch["fov_flat"] if ipm else batch["M_C"]
            feats["F_C"] = lift(img, batch["grid_uv_flat" if ipm else "grid_uv"], valid_c)
            out["aux_road_camera"], out["valid_camera"] = self.aux_camera(feats["F_C"])[:, 0], valid_c
        if self.uses_lidar:
            feats["F_L"] = self.lidar(batch["pt_feats"], batch["pt_cell"], batch["grid_uv"].shape[0])
            out["aux_road_lidar"] = self.aux_lidar(feats["F_L"])[:, 0]
            out["valid_lidar"] = batch["valid_L"] if "valid_L" in batch else torch.ones_like(out["aux_road_lidar"], dtype=torch.bool)
        if self.cfg.mode == "ocgf":
            q = self.reliability(batch, out["aux_road_camera"])
            extra = []
            if self.cfg.gate_q:
                extra.append(q)
            if self.cfg.objects and self.cfg.gate_objects:
                extra.append(batch["O_B"].float())
            x, w = self.gate(feats["F_C"], feats["F_L"], batch["M_C"], out["valid_lidar"],
                             torch.cat(extra, dim=1) if extra else None)
            out["gate_w"], out["Q"] = w, q
        else:
            x = self.fuse(torch.cat(list(feats.values()), dim=1))
        if self.cfg.objects:
            x = torch.cat([x, batch["O_B"].to(x.dtype)], dim=1)
        logits = self.heads(self.decoder(x))
        out.update({name: logits[:, k] for k, name in enumerate(HEADS)})
        return out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
