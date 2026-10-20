# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# From MTrans (Liu et al., ECCV 2022; github.com/Cliu2/MTrans) via the team fork github.com/GNaveenkumar614/Transmodel, commit ebf8a90 (2026-10-06), file models/modules/modality_mapper.py (img2pc) and utils/point_ops.py (check_points_in_box, build_image_location_map_single).
# Modified for OCGF: only these three functions are kept; AttrDict added (replaces easydict).

"""Helpers shared by the MTrans model and its data pipeline."""
import numpy as np
import torch


class AttrDict(dict):
    """dict with attribute access (replaces easydict): d.key == d['key']."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


def to_attrdict(value):
    """Nested dicts -> nested AttrDicts (config files)."""
    if isinstance(value, dict):
        return AttrDict({k: to_attrdict(v) for k, v in value.items()})
    return value


def img2pc(feature_img, points_2d):
    # Bilinear interpolation to get point feature from the Image feature map
    # feature_img: (B, C_img, H, W); points_2d: (B, N, 2)
    B, C_img, H, W = feature_img.size()
    N = points_2d.size(1)
    batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, N).flatten()    # (B*N)
    points_2d = points_2d.view(-1, 2)                                   # (B*N)
    x, y = points_2d[:, 0], points_2d[:, 1]
    # query corresponding img feature with the 2D coordinates
    new_pc_features = torch.zeros(B*N, C_img, device=feature_img.device)
    total_weight = 0
    for ty in [torch.ceil(y), torch.floor(y)]:
        for tx in [torch.ceil(x), torch.floor(x)]:
            weight = (1-torch.abs(x-tx)) * (1-torch.abs(y-ty))
            temp_feature_img = feature_img[batch_idx, :, ty.long().clamp(max=H-1), tx.long().clamp(max=W-1)]
            new_pc_features = new_pc_features + weight.unsqueeze(1) * temp_feature_img
            total_weight = total_weight + weight.unsqueeze(1)
    new_pc_features = new_pc_features / total_weight
    new_pc_features = new_pc_features.view(B, N, C_img).transpose(-1, -2)
    return new_pc_features  # (B, C_img, N)


def check_points_in_box(points3d, location, dimension, yaw):
    # points3d: (N, 3); box3d: [x, y, z, l, w, h, yaw]
    cx, cy, cz = location
    l, w, h = dimension
    center = np.array([cx, cy, cz])[None, ...] # (1, 3)
    rot = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    points3d = points3d - center    # translation compensation
    points3d = np.matmul(points3d, rot.T)   # rotation compensation
    points3d = np.abs(points3d)
    x, y, z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    # by observation about the Kitti Dataset, some foregrounds are close to box but not exactly in it, so smooth the boundary a bit
    mask = np.logical_and.reduce((x<=l/2+0.1, y<=w/2+0.1, z<=h/2+0.04))   
    return mask


def build_image_location_map_single(H, W, device='cuda'):
    locations = torch.zeros(2, H, W, dtype=torch.float32, device=device)
    row_idx = torch.arange(H, dtype=torch.long, device=device)
    col_idx = torch.arange(W, dtype=torch.long, device=device)
    locations[1, row_idx, :] = row_idx.view(1, -1, 1).float()
    locations[0, :, col_idx] = col_idx.view(1, 1, -1).float()
    locations = locations.permute(1, 2, 0)  # H, W, 2 for (x, y) 2D coords
    return locations
