# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# From MTrans (Liu et al., ECCV 2022; github.com/Cliu2/MTrans) via the team fork github.com/GNaveenkumar614/Transmodel, commit ebf8a90 (2026-10-06), file models/CNN_backbone/FPN.py.
# Modified for OCGF: none to the code.

"""
    A customized Feature Pyramid Network.
"""

import torch
from torch import nn

class FPN(nn.Module):
    def __init__(self, in_channel=3, inter_size=128, out_channel=None) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, inter_size, 3, padding=1),
            nn.BatchNorm2d(inter_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(inter_size, inter_size, 3, padding=1),
            nn.BatchNorm2d(inter_size),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(inter_size, inter_size, 3, padding=1),
            nn.BatchNorm2d(inter_size),
            nn.ReLU()
        )
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(inter_size, inter_size, 3, padding=1),
        #     nn.BatchNorm2d(inter_size),
        #     nn.ReLU()
        # )
        self.raise_channel = nn.Conv2d(inter_size*3, out_channel, 1)
        
    def forward(self, img):
        # img: (B, C, H, W)
        x1 = self.layer1(img)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # x4 = self.layer4(x3)

        fpn = torch.cat([x1, x2, x3], dim=1)
        fpn = self.raise_channel(fpn)
        return fpn, x3
