"""OCGF v1: per-cell, confidence-guided gating of the camera and LiDAR BEV features (architecture plan §6.6).

    F_C, F_L = BN(F_C), BN(F_L)                 per-branch normalisation, so a weight means trust, not scale
    z        = Conv3x3(concat[F_C, F_L, (O_B), (Q)]) -> ReLU -> Conv1x1 -> logits [2, nx, ny]
    logits   masked where the camera (M_C) or the LiDAR (valid_L) has nothing to say about the cell
    w_C, w_L = softmax(logits)                 both 0 where both are invalid (the cell gets no evidence)
    F_road   = w_C * F_C + w_L * F_L

Without the normalisation the branches differ ~9x in magnitude (LiDAR larger; measured on the dev sample), so the sum
is dominated by the LiDAR whatever the weights, and a learned gate saturated within 100 steps (w_C 0.999 everywhere,
flat across every Q bin). The last layer starts at zero (w = 0.5 / 0.5), so the gate departs from equal trust only as
the data asks.

The weights are returned, so their use of Q is measurable (the "Confidence-Guided" title gate): evaluation logs mean
w_C by bins of each Q channel and its shift under perturbations (E5). Variants for the ablations:
  fixed    w = 0.5 / 0.5 where both are valid, 1 where only one is (E4's fixed fusion)
  learned  the gate above, from the features only, or with Q (E4), and with the object channels (E2)
"""
from __future__ import annotations

import torch
import torch.nn as nn

MASKED = -1.0e4                       # a masked logit: exp() underflows to 0, and it stays finite in float16


class Gate(nn.Module):
    def __init__(self, channels: int, mode: str = "learned", extra_channels: int = 0, hidden: int = 32):
        super().__init__()
        if mode not in ("fixed", "learned"):
            raise ValueError(f"gate mode {mode!r} is not 'fixed' or 'learned'")
        self.mode = mode
        self.norm_c, self.norm_l = nn.BatchNorm2d(channels), nn.BatchNorm2d(channels)
        if mode == "learned":
            self.net = nn.Sequential(nn.Conv2d(2 * channels + extra_channels, hidden, 3, padding=1),
                                     nn.ReLU(inplace=True), nn.Conv2d(hidden, 2, 1))
            nn.init.zeros_(self.net[2].weight)          # start neutral: w_C = w_L = 0.5 wherever both are valid
            nn.init.zeros_(self.net[2].bias)

    def forward(self, f_c, f_l, valid_c, valid_l, extra=None):
        """f_c, f_l [B, C, nx, ny]; valid_c, valid_l [B, nx, ny] bool; extra [B, E, nx, ny] (O_B, Q) or None.
        Returns (F_road [B, C, nx, ny], weights [B, 2, nx, ny]: w_C, w_L)."""
        valid = torch.stack([valid_c, valid_l], dim=1)
        f_c, f_l = self.norm_c(f_c), self.norm_l(f_l)
        if self.mode == "fixed":
            logits = torch.zeros_like(valid, dtype=torch.float32)
        else:
            parts = [f_c, f_l] + ([extra.to(f_c.dtype)] if extra is not None else [])
            logits = self.net(torch.cat(parts, dim=1)).float()
        logits = logits.masked_fill(~valid, MASKED)
        w = torch.softmax(logits, dim=1) * valid             # both invalid: 0 / 0
        fused = w[:, :1].to(f_c.dtype) * f_c + w[:, 1:].to(f_l.dtype) * f_l
        return fused, w
