# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Written for mtrans_kaggle.ipynb in the team fork github.com/GNaveenkumar614/Transmodel (commit ebf8a90), as the
# pure-PyTorch stand-in for MTrans's compiled CUDA extension loss/cuda_op. Modified for OCGF: it is the only
# implementation (no CUDA build step), wrapped as sort_v; the tests check it against closed-form IoU values.
"""Pure-PyTorch version of the 'sort_vertices' CUDA extension used by the rotated-IoU loss.

Same contract as its sort_vertices_forward: for every polygon return the indices of its valid vertices in angular
order, then the first index again to close the polygon, then padding with an unused slot.
"""
import torch

MAX_NUM_VERT_IDX = 9
INTERSECTION_OFFSET = 8


def sort_vertices_forward(vertices, mask, num_valid):
    # vertices: (B, N, 24, 2), centred on each polygon's mean; mask: (B, N, 24) bool; num_valid: (B, N) int
    angle = torch.atan2(vertices[..., 1], vertices[..., 0])
    angle = torch.where(mask, angle, torch.full_like(angle, float('inf')))
    order = angle.argsort(dim=-1)[..., :MAX_NUM_VERT_IDX - 1]            # valid vertices first, by angle
    # an unused edge-intersection slot: its coordinates are zero, so it adds nothing to the shoelace area
    pad = (~mask[..., INTERSECTION_OFFSET:]).int().argmax(dim=-1, keepdim=True) + INTERSECTION_OFFSET
    count = num_valid.long().clamp(max=MAX_NUM_VERT_IDX - 1).unsqueeze(-1)
    slot = torch.arange(MAX_NUM_VERT_IDX, device=vertices.device).expand(*count.shape[:-1], -1)
    ring = torch.cat([order, order[..., :1]], dim=-1)                    # (B, N, 9)
    idx = torch.where(slot < count, ring, pad.expand_as(ring))
    idx = torch.where(slot == count, order[..., :1].expand_as(ring), idx)
    idx = torch.where(count < 3, pad.expand_as(ring), idx)               # fewer than 3 vertices: no polygon
    return idx.int()


def sort_v(vertices, mask, num_valid):
    """Indices only: not differentiable, as in the CUDA version."""
    with torch.no_grad():
        return sort_vertices_forward(vertices, mask, num_valid)
