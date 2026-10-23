"""LiDAR branch: per-point MLP, max over each BEV cell's points (pillars without a point cap), residual BEV convs."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
                                  nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class PillarEncoder(nn.Module):
    """Points (features [N, F], flat cell index [N] over the batch) -> BEV features [B, channels, nx, ny].
    Empty cells are zero."""

    def __init__(self, in_features: int, channels: int = 64, grid_shape=(200, 160)):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.point_mlp = nn.Sequential(nn.Linear(in_features, channels, bias=False), nn.BatchNorm1d(channels),
                                       nn.ReLU(inplace=True))
        self.bev = nn.Sequential(ResBlock(channels), ResBlock(channels))

    def forward(self, feats, cell, batch_size: int):
        nx, ny = self.grid_shape
        f = self.point_mlp(feats).float()                        # scatter in float32 under mixed precision too
        c = f.shape[1]
        cells = f.new_zeros(batch_size * nx * ny, c)
        cells.scatter_reduce_(0, cell.unsqueeze(1).expand(-1, c), f, reduce="amax", include_self=False)
        x = cells.view(batch_size, nx, ny, c).permute(0, 3, 1, 2).contiguous()
        return self.bev(x)
