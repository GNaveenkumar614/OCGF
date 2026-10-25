"""Camera branch: frozen self-supervised ViT tokens + a stride-8 conv stem -> image features -> lifted onto the BEV grid.

ViT tokens are 14 px wide (a token row covers about 15 m of ground depth at 30 m), so they are fused with a light conv
stem trained from scratch at stride 8 (CLAUDE.md: dense features at stride <= 8; the CLS token is never used). The
stem's 4x4 stride-2 convolutions keep the pixel-area convention (feature j covers pixels [8j, 8j + 8)), as the tokens
do, so lift() samples both at the same coordinates.

lift(): every BEV cell samples the image features at its ground point (bilinear, align_corners=False) and is zeroed
where the camera cannot see the cell (outside the view or in an occluder's shadow), so shadow cells never borrow the
occluder's appearance.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoBackbone(nn.Module):
    """Frozen DINOv2 from torch.hub (Apache-2.0, self-supervised): normalized [B, 3, H, W] -> tokens [B, C, H/14, W/14]."""
    patch = 14

    def __init__(self, name: str = "dinov2_vits14_reg"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        self.out_channels = int(self.model.embed_dim)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()                       # frozen: no dropout, fixed statistics
        return self

    def forward(self, x):
        with torch.no_grad():
            return self.model.get_intermediate_layers(x, n=1, reshape=True)[0]


class StubBackbone(nn.Module):
    """A stand-in for tests and CPU smoke runs: a fixed random 14x14 patch embedding. Seeded, so the same weights come
    back every time it is built (frozen parts are not stored in checkpoints)."""
    patch = 14

    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.out_channels = out_channels
        self.proj = nn.Conv2d(3, out_channels, self.patch, self.patch)
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in self.proj.parameters():
                p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) / (3 * self.patch * self.patch) ** 0.5)
                p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            return self.proj(x)


def make_backbone(name: str) -> nn.Module:
    return StubBackbone() if name == "stub" else DinoBackbone(name)


def _down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CameraEncoder(nn.Module):
    """uint8 image [B, 3, H, W] (H, W multiples of 8 and of the patch) -> features [B, channels, H/8, W/8]."""

    def __init__(self, backbone: nn.Module, channels: int = 64):
        super().__init__()
        self.backbone = backbone
        self.stem = nn.Sequential(_down(3, 32), _down(32, 64), _down(64, channels))
        self.tokens = nn.Sequential(nn.Conv2d(backbone.out_channels, channels, 1, bias=False),
                                    nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                  nn.Conv2d(channels, channels, 3, padding=1))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, image):
        x = (image.float() / 255.0 - self.mean) / self.std
        s8 = self.stem(x)
        tok = self.tokens(self.backbone(x).to(s8.dtype))
        tok = F.interpolate(tok, size=s8.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([s8, tok], dim=1))


def lift(features, grid_uv, valid):
    """features [B, C, h, w] spanning the whole image; grid_uv [B, nx, ny, 2] (x, y) in [-1, 1]; valid [B, nx, ny]
    -> BEV features [B, C, nx, ny], zero where not valid."""
    bev = F.grid_sample(features, grid_uv.to(features.dtype), mode="bilinear", padding_mode="zeros",
                        align_corners=False)
    return bev * valid.unsqueeze(1).to(bev.dtype)


def lift_at_points(features, pt_uv, pt_cell, batch_size: int, grid_shape):
    """LiDAR-anchored lifting (ablation A1): the image features at each LiDAR point's pixel, averaged per BEV cell.
    features [B, C, h, w]; pt_uv [N, 2] grid_sample coordinates (outside the image: -2); pt_cell [N] flat cell index
    over the batch. Returns (BEV features [B, C, nx, ny], valid [B, nx, ny]: the cell holds a seen point)."""
    nx, ny = grid_shape
    c = features.shape[1]
    seen = (pt_uv.abs() <= 1).all(dim=1)
    sums = features.new_zeros(batch_size * nx * ny, c, dtype=torch.float32)
    counts = features.new_zeros(batch_size * nx * ny, dtype=torch.float32)
    per_sample = nx * ny
    for b in range(batch_size):
        sel = seen & (pt_cell // per_sample == b)
        if not sel.any():
            continue
        f = F.grid_sample(features[b:b + 1].float(), pt_uv[sel][None, None].float(), mode="bilinear",
                          padding_mode="zeros", align_corners=False)[0, :, 0].T            # [n, C]
        sums.index_add_(0, pt_cell[sel], f)
        counts.index_add_(0, pt_cell[sel], torch.ones_like(f[:, 0]))
    bev = (sums / counts.clamp(min=1)[:, None]).view(batch_size, nx, ny, c).permute(0, 3, 1, 2).contiguous()
    return bev.to(features.dtype), (counts > 0).view(batch_size, nx, ny)

