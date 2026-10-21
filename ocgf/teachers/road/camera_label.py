# Adapted from lidar-camera-road-autolabeling (github.com/eerik98, commit 2534254): camera_autolabeling.py.
# MIT License, Copyright (c) 2024 Eerik Alamikkotervo; full notice in LICENSE-lidar-camera-road-autolabeling here.
# Changes for OCGF: the feature extractor is injected (DINOv2 from torch.hub on Kaggle, a stand-in in tests); the
# path mask goes to the patch grid by area (a patch is road if at least half of it is); NaN when no prototype exists yet.
"""Step 3 of the road teacher: what road looks like in this image, learned from the path.

DINOv2 patch features (ViT-g/14 with registers, as in the original) are averaged over the patches inside the path
polygon: the road prototype. Every patch scores by cosine similarity to it, normalised by the best patch and mapped
through exp(-((1 - cos) / camera_std)^2), then upsampled to the image. A frame whose path covers fewer than
min_reference_pixels pixels reuses the previous frame's prototype (frames are labelled in driving order).
"""
from __future__ import annotations

import numpy as np

from .config import RoadConfig


def patch_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bool [h, w]: patches at least half covered by the pixel mask (the image is resized to h x w patches)."""
    H, W = mask.shape
    ys = np.minimum((np.arange(H) * h / H).astype(int), h - 1)
    xs = np.minimum((np.arange(W) * w / W).astype(int), w - 1)
    cover = np.zeros((h, w))
    count = np.zeros((h, w))
    np.add.at(cover, (ys[:, None], xs[None, :]), mask.astype(float))
    np.add.at(count, (ys[:, None], xs[None, :]), 1.0)
    return cover / np.maximum(count, 1) >= 0.5


class DinoFeatures:
    """DINOv2 patch features of an RGB image: [h, w, C] on the model's device (torch.hub, needs Internet once)."""

    def __init__(self, model: str = "dinov2_vitg14_reg", device=None, scale: float = 1.0, half: bool = True):
        import torch
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load("facebookresearch/dinov2", model, verbose=False).to(self.device).eval()
        self.half = half and self.device.type == "cuda"
        if self.half:
            self.model = self.model.half()
        self.scale, self.patch = scale, 14
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def __call__(self, image: np.ndarray):
        torch = self.torch
        H, W = image.shape[:2]
        h, w = int(H * self.scale) // self.patch, int(W * self.scale) // self.patch
        x = torch.from_numpy(np.ascontiguousarray(image)).to(self.device).permute(2, 0, 1)[None].float() / 255
        x = torch.nn.functional.interpolate(x, size=(h * self.patch, w * self.patch), mode="bilinear",
                                            align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            f = self.model.get_intermediate_layers(x.half() if self.half else x, n=1)[0]
        return f.reshape(h, w, -1).float()


def camera_label(features, path_mask: np.ndarray, prototype, cfg: RoadConfig, image_shape) -> tuple[np.ndarray, object]:
    """(label [H, W] in [0, 1] or all-NaN, prototype used). features: torch [h, w, C]; path_mask: bool [H, W]."""
    import torch
    import torch.nn.functional as F
    h, w = features.shape[:2]
    if path_mask.sum() >= cfg.min_reference_pixels or prototype is None:
        patches = torch.from_numpy(patch_mask(path_mask, h, w)).to(features.device)
        if patches.any():
            prototype = features[patches].mean(dim=0)
    if prototype is None:
        return np.full(image_shape, np.nan), None
    cos = F.cosine_similarity(features, prototype.view(1, 1, -1), dim=2)
    cos = cos / cos.max()
    score = torch.exp(-torch.pow((1 - cos) / cfg.camera_std, 2))
    up = F.interpolate(score[None, None], size=tuple(image_shape), mode="bilinear", align_corners=False)[0, 0]
    return up.cpu().numpy().astype(np.float64), prototype
