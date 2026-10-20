"""The multi-class MTrans model: finite losses, class anchors and embedding in use, batch-independent predictions,
orientation helpers equal to the original loops."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ocgf.datasets.kitti_object import KittiObjectFrames, kitti_fid
from ocgf.teachers.detector2d.mapping import read_detections
from ocgf.teachers.mtrans.data import (ObjectDataset, ProjectedFrames, build_records, collate, detection_objects,
                                       label_objects, merge_batches)
from ocgf.teachers.mtrans.model import MTrans

from mtrans_fixtures import ANCHORS, fake_kitti, tiny_config


@pytest.fixture
def batches(tmp_path):
    folders, detections = fake_kitti(tmp_path)
    source = KittiObjectFrames(folders)
    frames = ProjectedFrames(source, tmp_path / "cache")
    lab = build_records(frames, label_objects(source, [kitti_fid(3)]), log=None)
    unl = build_records(frames, detection_objects(read_detections(detections), [kitti_fid(5)], 0.5), log=None)
    make = lambda recs, **kw: collate([ObjectDataset(recs, frames, ANCHORS, 32, 64, **kw)[i] for i in range(len(recs))])
    return make, lab, unl


def model(seed=0, anchors=ANCHORS):
    torch.manual_seed(seed)
    return MTrans(tiny_config().model, anchors)


def test_mixed_batch_gives_finite_loss_and_gradients(batches):
    make, lab, unl = batches
    m = model().train()
    data = make(lab + unl, mask_ratio=(0.0, 0.9), box2d_augmentation=True)
    loss, stats, iou, conf = m.get_loss(m(data), data)
    loss.backward()
    assert torch.isfinite(loss) and len(iou) == len(lab) == len(conf)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_a_batch_without_labelled_objects_still_trains_point_reconstruction(batches):
    make, _, unl = batches
    m = model().train()
    data = make(unl, mask_ratio=(0.5, 0.9))
    loss, stats, iou, _ = m.get_loss(m(data), data)
    loss.backward()
    assert torch.isfinite(loss) and len(iou) == 0 and "loss_box" not in stats and stats["loss_depth"][0] >= 0


def test_a_zeroed_box_head_predicts_each_objects_class_anchor(batches):
    make, lab, _ = batches
    m = model().eval()
    torch.nn.init.zeros_(m.box_head[-1].weight)
    torch.nn.init.zeros_(m.box_head[-1].bias)
    data = make(lab)
    with torch.no_grad():
        pred = m(data)
    assert torch.allclose(pred["dimension"], torch.tensor(ANCHORS)[data.classes])


def test_predictions_do_not_depend_on_the_rest_of_the_batch(batches):
    make, lab, unl = batches
    m = model().eval()
    np.random.seed(0)                                    # items sample points at random: build them once
    items = [make([rec]) for rec in lab + unl]
    batch = items[0]
    for one in items[1:]:
        batch = merge_batches(batch, one)
    boxes, conf = m.predict(batch)
    for i in range(len(items)):
        single = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in items[i].items()}
        b1, c1 = m.predict(type(items[i])(single))
        assert torch.allclose(b1[0], boxes[i], atol=1e-4) and torch.allclose(c1[0], conf[i], atol=1e-5)
    assert ((conf >= 0) & (conf <= 1)).all()


def test_the_class_embedding_changes_the_prediction(batches):
    make, lab, _ = batches
    m = model(anchors=[ANCHORS[0]] * 4).eval()          # same anchor for every class: only the embedding differs
    data = make(lab[:1])
    first = m.predict(data)[0]
    data = make(lab[:1])
    data.classes[:] = (data.classes + 1) % 4
    assert not torch.allclose(m.predict(data)[0], first)


def _clamp_loop(a):                                      # the original MTrans implementations, element by element
    a = a.clone()
    for i in range(a.size(0)):
        while a[i] > np.pi:
            a[i] = a[i] - np.pi * 2
        while a[i] <= -np.pi:
            a[i] = a[i] + np.pi * 2
    return a


def _adjust_loop(yaw, direction):
    yaw = _clamp_loop(yaw)
    for i in range(yaw.size(0)):
        if direction[i] == 1 and not (-np.pi / 2 <= yaw[i] < np.pi / 2):
            yaw[i] = yaw[i] + np.pi
        elif direction[i] == 0 and (-np.pi / 2 <= yaw[i] < np.pi / 2):
            yaw[i] = yaw[i] + np.pi
    return yaw


def test_orientation_helpers_match_the_original_loops():
    g = torch.Generator().manual_seed(0)
    angles = (torch.rand(500, 1, generator=g, dtype=torch.float64) - 0.5) * 8 * math.pi
    assert torch.allclose(MTrans.clamp_orientation_range(angles), _clamp_loop(angles))
    yaw = torch.asin(torch.tanh(torch.randn(500, 1, generator=g, dtype=torch.float64) * 2))
    direction = torch.randint(0, 2, (500,), generator=g)
    got, want = model().adjust_direction(yaw, direction), _adjust_loop(yaw, direction)
    assert torch.allclose(torch.cos(got), torch.cos(want)) and torch.allclose(torch.sin(got), torch.sin(want))
