"""Detector runs end to end on fake KITTI and a fake prepared KITTI-360 drive, with a stand-in model (no torch)."""
import numpy as np
import pytest
from PIL import Image

from ocgf.datasets.kitti360 import Kitti360Sequence
from ocgf.datasets.locate import find_kitti_object, find_prepared, prepared_drive
from ocgf.datasets.prepare_kitti360 import PrepConfig, run_all
from ocgf.teachers.detector2d.mapping import Detections, read_detections
from ocgf.teachers.detector2d.run import kitti360_frames, kitti_object_items, load_png, run_source

from kitti360_fixtures import SEQ, make_fake_kitti360


class StandInDetector:
    """One car and one person-on-bicycle per image."""
    def __call__(self, images):
        return [Detections(np.array([[10, 10, 60, 40], [100, 20, 130, 90], [95, 60, 135, 100]], float),
                           np.array([0.9, 0.8, 0.7]), ["car", "person", "bicycle"]) for _ in images]


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("detector")
    paths = make_fake_kitti360(tmp, n_frames=40, starter_images=False)
    run_all(PrepConfig(sequence=SEQ, input_root=paths["input_root"], out_dir=paths["input_root"] / "supp" / "kitti360_supplement",
                       scratch_dir=tmp / "scratch", sources=paths["sources"], role="test", keep_every_m=2.5,
                       min_future_m=20.0, n_overlays=0, dev_frames=0), log=lambda m: None)
    kitti = paths["input_root"] / "kitti-3d" / "training"
    for name in ("image_2", "label_2", "calib", "velodyne"):
        (kitti / name).mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (200, 60), (90, 90, 90)).save(kitti / "image_2" / f"{i:06d}.png")
    return paths["input_root"]


def test_finds_kitti_and_the_prepared_drive(inputs):
    kitti = find_kitti_object(inputs)
    assert set(kitti) == {"image_2", "label_2", "calib", "velodyne"}
    assert find_prepared(inputs, SEQ).name == "kitti360_supplement"
    components = prepared_drive(inputs, SEQ)
    assert {"scans", "frame_index", "image_shards", "calibration", "poses"} <= set(components)


def test_run_on_kitti_writes_both_files(inputs, tmp_path):
    items = kitti_object_items(find_kitti_object(inputs)["image_2"])
    assert items[0][0] == "kitti/training/0000000000"
    stats = run_source(StandInDetector(), items, load_png, tmp_path, "kitti", "training", batch_size=2, log=lambda m: None)
    assert stats["images"] == 5 and stats["detections"] == {"car": 5, "large_vehicle": 0, "pedestrian": 0, "cyclist": 5}
    ours, raw = read_detections(tmp_path / "kitti" / "training.txt"), read_detections(tmp_path / "kitti" / "training_raw.txt")
    assert len(ours) == len(raw) == 5
    assert sorted(ours["kitti/training/0000000003"].labels) == ["car", "cyclist"]
    assert sorted(raw["kitti/training/0000000003"].labels) == ["bicycle", "car", "person"]


def test_run_on_a_prepared_drive(inputs, tmp_path):
    seq = Kitti360Sequence(sequence=SEQ, components=prepared_drive(inputs, SEQ))
    frames = kitti360_frames(seq, every_m=2.5)
    assert len(frames) and set(frames.tolist()) <= set(seq.usable_frames().tolist())
    items = [(seq.fid(int(f)), int(f)) for f in frames]
    run_source(StandInDetector(), items, seq.image, tmp_path, "kitti360", SEQ, batch_size=3, log=lambda m: None)
    assert set(read_detections(tmp_path / "kitti360" / f"{SEQ}.txt")) == {fid for fid, _ in items}


def test_drive_0000_style_index_is_thinned_to_2p5_m():
    # an index that keeps every frame (drive 0000): 1 m per frame, all usable
    n = 40
    index = {"frame": np.arange(n), "usable": np.ones(n, bool), "odometer_m": np.arange(n, dtype=float)}
    seq = type("S", (), {"index": index})()
    frames = kitti360_frames(seq, every_m=2.5)
    assert (np.diff(index["odometer_m"][frames]) >= 2.5).all() and len(frames) == 14
