"""stage_drive: a local copy of a prepared drive reads the same, whether its shards are zips or unpacked folders."""
import shutil
import zipfile

import numpy as np

from ocgf.datasets.kitti360 import Kitti360Sequence, stage_drive
from ocgf.datasets.locate import prepared_drive
from ocgf.datasets.prepare_kitti360 import PrepConfig, run_all

from kitti360_fixtures import SEQ, make_fake_kitti360


def _prepared(tmp_path):
    paths = make_fake_kitti360(tmp_path, n_frames=40, starter_images=False)
    run_all(PrepConfig(sequence=SEQ, input_root=paths["input_root"], out_dir=paths["input_root"] / "supp" / "out",
                       scratch_dir=tmp_path / "scratch", sources=paths["sources"], role="test", keep_every_m=2.5,
                       min_future_m=20.0, n_overlays=0, dev_frames=0), log=lambda m: None)
    return paths


def test_staged_drive_reads_the_same(tmp_path):
    paths = _prepared(tmp_path)
    parts = prepared_drive(paths["input_root"], SEQ)
    seq = Kitti360Sequence(sequence=SEQ, components=parts)
    frames = seq.usable_frames()[:3]
    staged = stage_drive(parts, frames, tmp_path / "local", threads=2, log=None)
    local = Kitti360Sequence(sequence=SEQ, components=staged)
    assert str(staged["scans"]).startswith(str(tmp_path / "local"))
    for f in frames:
        assert np.array_equal(local.points(int(f)), seq.points(int(f)))
        assert np.array_equal(local.image(int(f)), seq.image(int(f)))


def test_unpacked_shard_folders_are_staged_member_by_member(tmp_path):
    paths = _prepared(tmp_path)
    parts = prepared_drive(paths["input_root"], SEQ)
    for key in ("scans", "image_shards"):                   # what a Kaggle archive upload does to the shards
        for z in list(parts[key].glob("shard_*.zip")):
            with zipfile.ZipFile(z) as f:
                f.extractall(z.with_suffix(""))
            z.unlink()
    seq = Kitti360Sequence(sequence=SEQ, components=parts)
    frames = seq.usable_frames()[:2]
    staged = stage_drive(parts, frames, tmp_path / "local", threads=2, log=None)
    local = Kitti360Sequence(sequence=SEQ, components=staged)
    assert len(list((tmp_path / "local" / "scans").rglob("*.npy"))) == len(frames)
    assert all(np.array_equal(local.points(int(f)), seq.points(int(f))) for f in frames)
    shutil.rmtree(tmp_path / "local")
