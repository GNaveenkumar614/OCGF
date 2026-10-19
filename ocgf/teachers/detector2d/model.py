"""The detector itself: torchvision fasterrcnn_resnet50_fpn_v2, COCO_V1 weights, zero-shot. torch.

Images are upscaled for small, distant objects: torchvision resizes so the short side reaches min_size unless the long
side would exceed max_size. With 800 / 2400 a 1408 x 376 KITTI-360 image runs at 2400 x 641 (x1.7) and a 1242 x 375
KITTI image at 2400 x 725 (x1.9); the defaults (800 / 1333) would shrink them instead. A recall choice for v1; the
recall evaluation on KITTI-360 checks it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

from .mapping import RAW_CLASSES, Detections


class Detector:
    def __init__(self, device: str | None = None, min_size: int = 800, max_size: int = 2400, score_floor: float = 0.05,
                 amp: bool = True, pretrained: bool = True):
        """pretrained=False builds the same network with random weights (tests, no download)."""
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.categories = self.weights.meta["categories"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = amp and self.device.type == "cuda"
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights if pretrained else None, weights_backbone=None, num_classes=len(self.categories),
            min_size=min_size, max_size=max_size, box_score_thresh=score_floor).eval().to(self.device)
        self.settings = {"model": "fasterrcnn_resnet50_fpn_v2", "weights": "COCO_V1" if pretrained else None,
                         "weights_url": self.weights.url if pretrained else None, "min_size": min_size,
                         "max_size": max_size, "score_floor": score_floor, "amp": self.amp}

    def weights_sha256(self) -> str | None:
        """sha256 of the downloaded weights file in the torch hub cache (provenance)."""
        if not self.settings["weights"]:
            return None
        path = Path(torch.hub.get_dir()) / "checkpoints" / self.weights.url.rsplit("/", 1)[-1]
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    @torch.inference_mode()
    def __call__(self, images: list) -> list[Detections]:
        """images: HxWx3 uint8 arrays -> raw detections of the COCO classes we use, in pixel coordinates."""
        tensors = [torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).to(self.device).float().div_(255)
                   for im in images]
        with torch.autocast("cuda", enabled=self.amp):
            outputs = self.model(tensors)
        results = []
        for out in outputs:
            labels = [self.categories[int(i)] for i in out["labels"].cpu()]
            keep = [k for k, l in enumerate(labels) if l in RAW_CLASSES]
            results.append(Detections(out["boxes"].float().cpu().numpy()[keep].astype(np.float64),
                                      out["scores"].float().cpu().numpy()[keep].astype(np.float64),
                                      [labels[k] for k in keep]))
        return results
