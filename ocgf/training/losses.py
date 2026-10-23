"""Student losses: masked mean binary cross-entropy against soft targets (NaN = ignored).

L = w_road * L_road + w_occ * L_occ + w_unk * L_unk. L_road and L_occ ignore NaN targets (unknown cells, cells no
teacher speaks about); L_unk covers every cell. Each term is a masked mean whose denominator is clamped to >= 1, so an
all-ignored batch gives a zero, never a NaN. The boundary and consistency terms come later.
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

    @classmethod
    def from_dict(cls, d: dict | None) -> LossWeights:
        return cls(**dict(d or {}))

    def to_dict(self) -> dict:
        return asdict(self)


def masked_bce(logits, target, mask=None):
    """Mean BCE over the cells where the target is finite (and mask holds); 0 when there are none."""
    logits, target = logits.float(), target.float()
    keep = torch.isfinite(target) if mask is None else torch.isfinite(target) & mask
    loss = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(target, nan=0.0).clamp(0, 1), reduction="none")
    return (loss * keep).sum() / keep.sum().clamp(min=1)


def student_loss(out: dict, batch: dict, weights: LossWeights = LossWeights()):
    """(total, {term: value}) for the student's logits and the batch targets."""
    parts = {"road": masked_bce(out["road"], batch["road_t"]),
             "occ": masked_bce(out["occ"], batch["occ_t"]),
             "unk": masked_bce(out["unk"], batch["unk_t"])}
    total = weights.road * parts["road"] + weights.occ * parts["occ"] + weights.unk * parts["unk"]
    return total, parts
