"""MTrans inference: every detector box gets a row, with a 3D box and raw confidence when it has enough points."""
import pytest

torch = pytest.importorskip("torch")

from ocgf.datasets.kitti_object import KittiObjectFrames, kitti_fid
from ocgf.teachers.detector2d.mapping import read_detections
from ocgf.teachers.mtrans.data import ProjectedFrames, detection_objects
from ocgf.teachers.mtrans.infer import Kitti360Frames, infer, load_model, read_objects
from ocgf.teachers.mtrans.train import build_pools, train

from mtrans_fixtures import FRAME_LISTS, fake_kitti, tiny_config

quiet = lambda msg: None


@pytest.fixture
def trained(tmp_path):
    folders, detections = fake_kitti(tmp_path)
    cfg, out = tiny_config(), tmp_path / "out"
    pools = build_pools(cfg, folders, detections, tmp_path / "cache", out, log=quiet, frame_lists=FRAME_LISTS)
    train(cfg, pools, folders, tmp_path / "cache", out, device="cpu", smoke=3, log=quiet)
    model, meta = load_model(out / "ckpt" / "last.pt", device="cpu")
    frames = ProjectedFrames(KittiObjectFrames(folders), tmp_path / "cache")
    return model, meta, frames, read_detections(detections)


def test_every_detection_gets_a_row_with_a_lidar_box_and_raw_confidence(trained, tmp_path):
    model, meta, frames, dets = trained
    objects = detection_objects(dets, [kitti_fid(8)])
    summary = infer(model, meta, frames, objects, tmp_path / "objects.jsonl", batch_size=2, log=quiet)
    rows = read_objects(tmp_path / "objects.jsonl")[kitti_fid(8)]
    assert len(rows) == len(dets[kitti_fid(8)]) == summary["objects"]
    assert [r["cls"] for r in rows] == list(dets[kitti_fid(8)].labels)                # input order kept
    for r in rows:
        assert r["status"] == "ok" and len(r["box"]) == 7 and 0 <= r["conf_iou_raw"] <= 1 and r["box"][3] > 0


def test_boxes_without_enough_points_are_listed_without_a_3d_box(trained, tmp_path):
    model, meta, frames, dets = trained
    infer(model, meta, frames, detection_objects(dets, [kitti_fid(8)]), tmp_path / "o.jsonl", min_points=10 ** 9,
          log=quiet)
    rows = read_objects(tmp_path / "o.jsonl")[kitti_fid(8)]
    assert rows and all(r["status"] == "too_few_points" and r["box"] is None for r in rows)


def test_kitti360_frames_adapter():
    class Seq:
        camera = "cam"

        def image(self, frame):
            return f"image {frame}"

        def points(self, frame):
            import numpy as np
            return np.zeros((3, 5))

    frames = Kitti360Frames({"2013_05_28_drive_0009_sync": Seq()})
    fid = "kitti360/2013_05_28_drive_0009_sync/0000000042"
    assert frames.image(fid) == "image 42" and frames.points(fid).shape == (3, 4) and frames.camera(fid) == "cam"
