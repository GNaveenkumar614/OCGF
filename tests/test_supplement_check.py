"""The backup check: a prepared drive must match its own manifest."""
import pytest

from ocgf.datasets.prepare_kitti360 import PrepConfig, run_all
from ocgf.datasets.supplement_check import check_supplement, find_root

from kitti360_fixtures import SEQ, make_fake_kitti360


@pytest.fixture
def prepared(tmp_path):
    paths = make_fake_kitti360(tmp_path, n_frames=40, starter_images=False)
    out = tmp_path / "copy" / "kitti360_supplement"           # nested, as in a downloaded dataset
    cfg = PrepConfig(sequence=SEQ, input_root=paths["input_root"], out_dir=out, scratch_dir=tmp_path / "scratch",
                     sources=paths["sources"], role="test", keep_every_m=2.5, min_future_m=20.0, n_overlays=1,
                     dev_frames=0)
    run_all(cfg, log=lambda msg: None)
    return tmp_path / "copy"


def test_a_complete_copy_passes(prepared):
    root = find_root(prepared)
    summary, problems = check_supplement(root, SEQ, deep=True)
    assert problems == []
    assert summary["role"] == "test" and summary["kept"] == summary["scans"] == summary["images"] > 0


def test_a_missing_shard_is_reported(prepared):
    root = find_root(prepared)
    next((root / "images" / SEQ).glob("shard_*.zip")).unlink()
    _, problems = check_supplement(root, SEQ)
    assert any("images stored" in p for p in problems)


def test_a_corrupt_shard_is_reported(prepared):
    root = find_root(prepared)
    shard = next((root / "scans" / SEQ).glob("shard_*.zip"))
    data = bytearray(shard.read_bytes())
    data[200:400] = bytes(200)                                  # damage member data, keep the directory intact
    shard.write_bytes(bytes(data))
    _, problems = check_supplement(root, SEQ, deep=True)
    assert any("corrupt" in p or "shard" in p for p in problems)


def test_the_wrong_drive_is_reported(prepared):
    _, problems = check_supplement(find_root(prepared), "2013_05_28_drive_0010_sync")
    assert any("expected" in p for p in problems)


def test_no_manifest(tmp_path):
    assert find_root(tmp_path) is None
    assert check_supplement(tmp_path)[1]
