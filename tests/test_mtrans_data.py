"""MTrans object data: labelled records come from labels, unlabelled ones from detector boxes, items have the shapes
the model expects."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ocgf.datasets.kitti_object import KittiObjectFrames, kitti_fid
from ocgf.teachers.detector2d.mapping import read_detections
from ocgf.teachers.mtrans.data import (CLASSES, ObjectDataset, ProjectedFrames, _overlapping, build_records,
                                       class_anchors, collate, detection_objects, label_objects, merge_batches)

from mtrans_fixtures import ANCHORS, fake_kitti

quiet = None


class SpySource(KittiObjectFrames):
    """Records every frame whose human labels are read."""

    def __init__(self, folders):
        super().__init__(folders)
        self.label_calls = []

    def labels(self, fid):
        self.label_calls.append(fid)
        return super().labels(fid)


@pytest.fixture
def setup(tmp_path):
    folders, detections = fake_kitti(tmp_path)
    source = SpySource(folders)
    return source, ProjectedFrames(source, tmp_path / "cache"), read_detections(detections)


def test_labelled_records_carry_3d_labels_and_unlabelled_ones_never_read_labels(setup):
    source, frames, dets = setup
    labelled = build_records(frames, label_objects(source, [kitti_fid(3)]), log=quiet)
    assert sorted(r["cls"] for r in labelled) == sorted(CLASSES)
    assert all(r["has_label"] and r["n_foreground"] >= 5 and len(r["box"]) == 7 for r in labelled)

    unlabelled = build_records(frames, detection_objects(dets, [kitti_fid(5)], 0.5), log=quiet)
    assert unlabelled and all(not r["has_label"] and r["box"] is None and r["n_foreground"] is None for r in unlabelled)
    assert np.allclose([r["box_2d"] for r in unlabelled], dets[kitti_fid(5)].boxes)    # detector boxes, not labels
    assert source.label_calls == [kitti_fid(3)]                                         # frame 5's labels never read


def test_detection_score_threshold(setup):
    _, frames, dets = setup
    assert detection_objects(dets, [kitti_fid(5)], min_score=0.95)[kitti_fid(5)] == []


def test_anchors_are_class_means_of_labelled_records_only():
    rec = lambda cls, dims, has_label=True: {"cls": cls, "has_label": has_label, "box": [0, 0, 0, *dims, 0]}
    records = [rec(c, d) for c, d in zip(CLASSES, ANCHORS)] + [rec("car", (6.0, 1.7, 1.5)),
                                                             rec("car", (99.0, 9.0, 9.0), has_label=False)]
    anchors = class_anchors(records)
    assert anchors[0] == [5.0, 1.7, 1.5] and anchors[1:] == ANCHORS[1:]
    with pytest.raises(ValueError):
        class_anchors(records[:3])                       # no labelled cyclist


def test_point_cache_round_trip(setup, tmp_path):
    _, frames, _ = setup
    first = frames.projected(kitti_fid(3))
    assert list((tmp_path / "cache").glob("*.npy"))
    second = frames.projected(kitti_fid(3))
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_items_have_the_shapes_the_model_expects(setup):
    source, frames, dets = setup
    records = (build_records(frames, label_objects(source, [kitti_fid(3)]), log=quiet)
               + build_records(frames, detection_objects(dets, [kitti_fid(5)], 0.5), log=quiet))
    ds = ObjectDataset(records, frames, ANCHORS, out_img_size=32, out_cloud_size=64, mask_ratio=(0.0, 0.9),
                       box2d_augmentation=True)
    for i, rec in enumerate(records):
        item = ds[i]
        assert item["frame_img"].shape == (1, 3, 32, 32) and item["overlap_mask"].shape == (1, 1, 32, 32)
        assert item["sub_cloud"].shape == (64, 4) and item["sub_cloud2d"].shape == (64, 2)
        assert set(np.unique(item["real_point_mask"])) <= {0, 1, 2}
        assert (item["sub_cloud2d"] >= 0).all() and (item["sub_cloud2d"] <= 31).all()
        if rec["has_label"]:
            assert (item["foreground_label"] == 1).any()
        else:
            assert (item["foreground_label"] == 2).all()
            assert np.allclose(item["dimensions"], ANCHORS[rec["class_idx"]])
    batch = collate([ds[0], ds[len(ds) - 1]])
    assert batch.images.shape == (2, 3, 32, 32) and batch.classes.dtype == torch.long
    assert merge_batches(batch, batch).sub_clouds.shape == (4, 64, 4)


def test_an_all_black_crop_still_gives_query_pixels(setup):
    source, frames, _ = setup
    frames.image = lambda fid: np.zeros((375, 1242, 3), dtype=np.uint8)
    records = build_records(frames, label_objects(source, [kitti_fid(3)]), log=quiet)
    item = ObjectDataset(records, frames, ANCHORS, out_img_size=32, out_cloud_size=64)[0]
    assert np.isfinite(item["sub_cloud2d"]).all()


def test_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30], [10, 0, 20, 10]], dtype=float)
    assert _overlapping(boxes) == [[1], [0, 3], [], [1]]      # touching edges do not overlap


def test_parallel_warm_up_fills_the_cache(setup, tmp_path):
    _, frames, _ = setup
    frames.warm([kitti_fid(3), kitti_fid(5), kitti_fid(8)], threads=3, log=None)
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 3
