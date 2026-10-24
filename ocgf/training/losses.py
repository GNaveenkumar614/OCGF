"""Student losses: masked mean binary cross-entropy against soft targets (NaN = ignored), plus the boundary and the
self-supervised consistency terms (architecture plan §6.8).

L = road * L_road + occ * L_occ + unk * L_unk + aux * L_aux + boundary * L_boundary + cross * L_cross
    + temporal * L_temporal
  L_road, L_occ   ignore NaN targets (unknown cells, cells no teacher speaks about); L_unk covers every cell
  L_aux           the auxiliary road heads on F_C and F_L against the road target where each sensor is valid (the mean
                  of the two when both exist); the camera one gives the reliability channel H_cam
  L_boundary      the road BCE weighted by exp(-d^2 / 2 sigma^2), d = distance to the road target's 0.5 contour
  L_cross         cross-modal consistency: symmetric BCE between the camera and LiDAR auxiliary road heads, each
                  against the other's detached probability, where both sensors are valid (no labels needed)
  L_temporal      temporal consistency: L1 between this frame's road probability and the earlier frame's, warped by
                  the poses (detached), where both frames observed the cell and neither predicts it occupied
                  (moving objects); needs the earlier frame's prediction (prev_out)
Each term is a masked mean whose denominator is clamped to >= 1, so an all-ignored batch gives a zero, never a NaN.
The S3 ablation sets cross and temporal to 0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    road: float = 1.0
    occ: float = 1.0
    unk: float = 0.5
    aux: float = 0.2
    boundary: float = 0.5
    boundary_sigma: float = 0.5        # m
    cross: float = 0.1
    temporal: float = 0.1

    @classmethod
    def from_dict(cls, d: dict | None) -> LossWeights:
        return cls(**dict(d or {}))

    def to_dict(self) -> dict:
        return asdict(self)


def masked_bce(logits, target, mask=None, weight=None):
    """Mean BCE over the cells where the target is finite (and mask holds), optionally weighted; 0 when there are none."""
    logits, target = logits.float(), target.float()
    keep = torch.isfinite(target) if mask is None else torch.isfinite(target) & mask
    loss = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(target, nan=0.0).clamp(0, 1), reduction="none")
    w = keep.float() if weight is None else keep.float() * weight.float()
    return (loss * w).sum() / w.sum().clamp(min=1)


def masked_mean(values, mask):
    m = mask.float()
    return (values.float() * m).sum() / m.sum().clamp(min=1)


def cross_modal(out: dict):
    """Symmetric BCE between the camera and LiDAR auxiliary road heads where both sensors are valid."""
    a, b = out["aux_road_camera"].float(), out["aux_road_lidar"].float()
    both = out["valid_camera"].bool() & out["valid_lidar"].bool()
    pa, pb = torch.sigmoid(a).detach(), torch.sigmoid(b).detach()
    sym = 0.5 * (F.binary_cross_entropy_with_logits(a, pb, reduction="none")
                 + F.binary_cross_entropy_with_logits(b, pa, reduction="none"))
    return masked_mean(sym, both)


def temporal(out: dict, prev_out: dict, batch: dict):
    """L1 between this frame's road probability and the earlier frame's (warped into this frame, detached)."""
    grid = batch["warp_grid"].float()

    def warp(x):
        return F.grid_sample(x.float()[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]

    with torch.no_grad():
        road_prev = warp(torch.sigmoid(prev_out["road"]))
        occ_prev = warp(torch.sigmoid(prev_out["occ"]))
        unk_prev = warp(batch["prev/unk_t"])
        occ_now = torch.sigmoid(out["occ"].float())
    road_now = torch.sigmoid(out["road"].float())
    mask = (batch["has_prev"][:, None, None] & batch["warp_inside"] & (batch["unk_t"] < 0.5) & (unk_prev < 0.5)
            & (occ_now < 0.5) & (occ_prev < 0.5))
    return masked_mean((road_now - road_prev).abs(), mask)


def student_loss(out: dict, batch: dict, weights: LossWeights = LossWeights(), prev_out: dict | None = None):
    """(total, {term: value}) for the student's outputs and the batch targets; prev_out: the model's outputs on the
    batch's paired earlier frames (temporal consistency), or None."""
    parts = {"road": masked_bce(out["road"], batch["road_t"]),
             "occ": masked_bce(out["occ"], batch["occ_t"]),
             "unk": masked_bce(out["unk"], batch["unk_t"])}
    total = weights.road * parts["road"] + weights.occ * parts["occ"] + weights.unk * parts["unk"]
    aux = [masked_bce(out[f"aux_road_{s}"], batch["road_t"], out[f"valid_{s}"].bool())
           for s in ("camera", "lidar") if f"aux_road_{s}" in out]
    if aux:
        parts["aux"] = sum(aux) / len(aux)
        total = total + weights.aux * parts["aux"]
    if weights.boundary and "road_bd" in batch:
        w = torch.exp(-batch["road_bd"].float() ** 2 / (2 * weights.boundary_sigma ** 2))
        parts["boundary"] = masked_bce(out["road"], batch["road_t"], weight=w)
        total = total + weights.boundary * parts["boundary"]
    if weights.cross and "aux_road_camera" in out and "aux_road_lidar" in out:
        parts["cross"] = cross_modal(out)
        total = total + weights.cross * parts["cross"]
    if weights.temporal and prev_out is not None and "warp_grid" in batch:
        parts["temporal"] = temporal(out, prev_out, batch)
        total = total + weights.temporal * parts["temporal"]
    return total, parts
