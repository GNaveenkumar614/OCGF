"""The vendored rotated 3D IoU (pure-PyTorch vertex sorting) against values known in closed form."""
import math

import pytest

torch = pytest.importorskip("torch")

from ocgf.teachers.mtrans.iou import cal_diou_3d

OCTAGON = 2 * (math.sqrt(2) - 1)      # area shared by a unit square and the same square turned 45 degrees
CASES = [  # boxes are (x, y, z, l, w, h, yaw)
    ([0, 0, 0, 4, 1.6, 1.5, 0.3], [0, 0, 0, 4, 1.6, 1.5, 0.3], 1.0),
    ([0, 0, 0, 4, 2, 1.5, 0], [2, 0, 0, 4, 2, 1.5, 0], 1 / 3),
    ([0, 0, 0, 2, 2, 2, 0.5], [0, 0, 1, 2, 2, 2, 0.5], 1 / 3),
    ([0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1, math.pi / 4], OCTAGON / (2 - OCTAGON)),
    ([0, 0, 0, 1, 1, 1, 0], [5, 5, 0, 1, 1, 1, 0.2], 0.0),
]


def test_closed_form_ious_and_finite_gradients():
    pred = torch.tensor([c[0] for c in CASES], dtype=torch.float32).unsqueeze(1).requires_grad_()
    target = torch.tensor([c[1] for c in CASES], dtype=torch.float32).unsqueeze(1)
    loss, iou3d, _ = cal_diou_3d(pred, target)
    loss.mean().backward()
    assert torch.allclose(iou3d.flatten(), torch.tensor([c[2] for c in CASES]), atol=1e-3)
    assert torch.isfinite(pred.grad).all()


def test_a_typical_prediction_has_a_partial_overlap():
    _, iou3d, _ = cal_diou_3d(torch.tensor([[[0.3, -0.2, 0.1, 3.8, 1.7, 1.5, 0.4]]]),
                              torch.tensor([[[0.0, 0.0, 0.0, 4.0, 1.6, 1.5, 0.3]]]))
    assert 0.3 < float(iou3d) < 0.9
