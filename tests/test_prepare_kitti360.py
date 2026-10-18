"""End to end: the data-prep pipeline on a small fake KITTI-360 tree."""
import json
import zipfile

import numpy as np
import pytest

from ocgf.datasets import frame_index as fi
from ocgf.datasets.dataset_map import discover_kitti360, missing_required, read_dataset_map
from ocgf.datasets.kitti360 import Kitti360Sequence
from ocgf.datasets.ply import read_ply
from ocgf.datasets.prepare_kitti360 import PrepConfig, choose_dev_frames, run_all, stage_inputs
from ocgf.datasets.supplement_io import ShardReader, decode, encode
from ocgf.datasets.velodyne import CropBox, read_bin, ring_index

from kitti360_fixtures import SEQ, make_fake_kitti360


def config(tmp_path, paths, **kw):
    args = dict(sequence=SEQ, input_root=paths["input_root"], out_dir=tmp_path / "out", scratch_dir=tmp_path / "scratch",
                sources=paths["sources"], min_future_m=20.0, step_m=5.0, profile_every=5, n_overlays=2, dev_frames=5)
    args.update(kw)
    return PrepConfig(**args)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("prep")
    paths = make_fake_kitti360(tmp, n_frames=40)
    cfg = config(tmp, paths)
    return cfg, paths, run_all(cfg, log=lambda msg: None)


def test_scans_keep_rings_from_the_raw_order(run):
    cfg, paths, _ = run
    reader = ShardReader(cfg.scans_dir)
    assert reader.frames() == [f for f in range(40) if f != 7]
    with zipfile.ZipFile(paths["velodyne_zip"]) as z:
        raw = read_bin(z.read(f"{SEQ}/velodyne_points/data/{12:010d}.bin"))
    keep = CropBox().mask(raw)
    expected = decode(encode(raw[keep], ring_index(raw)[keep]))
    np.testing.assert_array_equal(reader.read(12), expected)
    reader.close()


def test_frame_index(run):
    cfg, _, result = run
    index = fi.read_csv(cfg.frame_index_csv)
    assert len(index["frame"]) == 40
    assert not index["has_scan"][7] and not index["has_pose"][:2].any()
    np.testing.assert_allclose(index["dt_ms"], 4.0, atol=1e-3)
    usable = np.flatnonzero(index["usable"])
    assert len(usable) and 7 not in usable and usable.min() >= 2
    assert (index["future_path_m"][usable] >= 20.0).all()
    assert index["has_sem2d"][::3].all() and not index["has_sem2d"][1]
    assert (np.diff(index["odometer_m"][index["sample_5m"]]) >= 5.0).all()


def test_report_and_manifest(run):
    cfg, _, result = run
    report = (cfg.out_dir / "report" / "report.md").read_text(encoding="utf-8")
    assert "## Frames" in report and "no scan: 7" in report
    for name in ("dt_hist.png", "points_per_frame.png", "ring_profile.png", "trajectory.png"):
        assert (cfg.out_dir / "report" / name).exists()
    assert len(list((cfg.out_dir / "report").glob("overlay_*.png"))) == 2
    manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
    assert manifest["sequence"] == SEQ and manifest["counts"]["has_scan"] == 39
    assert manifest["scan_format"]["format_version"] == 1
    assert "code" in manifest and "environment" in manifest


def test_dev_sample_is_a_working_mini_dataset(run, tmp_path):
    cfg, _, _ = run
    with zipfile.ZipFile(cfg.out_dir / "dev_sample.zip") as z:
        z.extractall(tmp_path)
    dev = tmp_path / "dev_sample"
    seq, components = read_dataset_map(dev / "dataset_map.yaml")
    assert seq == SEQ and missing_required(components) == []
    assert missing_required(discover_kitti360(dev / "KITTI360", SEQ)) == []
    rows = fi.read_csv(components["frame_index"])
    frames = rows["frame"]
    assert len(frames) == 5 and rows["usable"].all() and (np.diff(frames) == 1).all()
    dev_reader, full_reader = ShardReader(components["scans"]), ShardReader(cfg.scans_dir)
    assert dev_reader.frames() == list(frames)
    np.testing.assert_array_equal(dev_reader.read(int(frames[0])), full_reader.read(int(frames[0])))
    clouds = list((components["semantics3d"] / "static").glob("*.ply"))
    assert len(clouds) == 1 and len(read_ply(clouds[0])) > 0
    assert len(list(components["images"].glob("*.png"))) == 5


def test_timestamps_come_from_the_timestamps_zip_when_the_scan_zip_has_none(tmp_path):
    paths = make_fake_kitti360(tmp_path, n_frames=36, timestamps_in_velodyne_zip=False)
    cfg = config(tmp_path, paths)
    result = run_all(cfg, log=lambda msg: None)
    assert result["index"]["aligned"][result["index"]["has_scan"]].all()


def test_missing_inputs_stop_with_a_clear_message(tmp_path):
    paths = make_fake_kitti360(tmp_path, n_frames=4)
    (paths["input_root"] / "kitti-360-prepared-dataset" / "KITTI360" / "calibration" / "perspective.txt").unlink()
    logged = []
    with pytest.raises(FileNotFoundError, match="calibration"):
        stage_inputs(config(tmp_path, paths), log=logged.append)
    tree = "\n".join(logged)
    assert "calibration/  (2 files)  <- calib_cam_to_pose.txt, calib_cam_to_velo.txt" in tree   # shows what is there
    assert "kitti360-3d-bboxes/  (1 files)  <- 2013_05_28_drive_0000_sync.xml" in tree


# ---------- the paper's other drives: images from the image_00 zip, one frame every 2.5 m, role-dependent windows

@pytest.fixture(scope="module")
def train_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train_drive")
    paths = make_fake_kitti360(tmp, n_frames=40, starter_images=False, windows=[(0, 25)])
    cfg = config(tmp, paths, role="train", keep_every_m=2.5, dev_frames=0)
    return cfg, run_all(cfg, log=lambda msg: None)


def test_new_drive_keeps_and_stores_a_frame_every_2p5_m(train_run):
    cfg, _ = train_run
    index = fi.read_csv(cfg.frame_index_csv)
    kept = index["frame"][index["kept"]]
    assert len(kept) and (np.diff(index["odometer_m"][index["kept"]]) >= 2.5).all()
    assert index["usable"][index["kept"]].all()
    assert set(index["frame"][index["sample_5m"]]) <= set(kept.tolist())
    assert ShardReader(cfg.scans_dir).frames() == kept.tolist()               # only kept scans were read
    assert ShardReader(cfg.images_dir).frames(".png") == kept.tolist()        # and only kept images
    assert (index["n_points_raw"][~index["kept"]] == -1).all() and (index["n_points_raw"][index["kept"]] > 0).all()


def test_train_drives_may_leave_the_semantic_windows(train_run):
    cfg, _ = train_run
    index = fi.read_csv(cfg.frame_index_csv)
    assert index["usable"][index["window_start"] < 0].any()      # frames after the only window (0-25) count


def test_val_and_test_drives_keep_the_window_rule(tmp_path):
    paths = make_fake_kitti360(tmp_path, n_frames=40, starter_images=False, windows=[(0, 25)])
    index = run_all(config(tmp_path, paths, role="val", keep_every_m=2.5, dev_frames=0, n_overlays=0),
                    log=lambda msg: None)["index"]
    assert index["usable"].any() and not index["usable"][index["window_start"] < 0].any()


def test_new_drive_output_loads_with_the_loader(train_run):
    cfg, _ = train_run
    seq = Kitti360Sequence(cfg.out_dir / "dataset_map.yaml")
    f = int(seq.usable_frames()[0])
    assert seq.image(f).shape == (376, 1408, 3)                   # read back from the image shards
    assert len(seq.points(f)) > 0
    assert (cfg.out_dir / "image_timestamps" / f"{SEQ}.txt").exists()


def test_choose_dev_frames_stays_in_one_window_and_run():
    frame = np.arange(30)
    usable = np.ones(30, dtype=bool)
    usable[[4, 18]] = False
    window = np.where(frame < 12, 0, 10)
    picked = choose_dev_frames({"frame": frame, "usable": usable, "window_start": window}, 4)
    assert len(picked) == 4 and (np.diff(picked) == 1).all()
    assert usable[picked].all() and len(set(window[picked])) == 1
