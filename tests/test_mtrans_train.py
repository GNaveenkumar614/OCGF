"""MTrans training: object pools respect the label budget, a smoke run trains and reloads, a stopped run resumes."""
import json

import pytest

torch = pytest.importorskip("torch")

from ocgf.datasets.kitti_object import kitti_fid
from ocgf.teachers.mtrans.train import build_pools, train

from kitti_object_fixtures import write_detections
from mtrans_fixtures import FRAME_LISTS, fake_kitti, tiny_config

quiet = lambda msg: None


def test_pools_follow_the_label_budget_and_are_reused(tmp_path):
    folders, detections = fake_kitti(tmp_path)
    cfg, out = tiny_config(), tmp_path / "out"
    pools = build_pools(cfg, folders, detections, tmp_path / "cache", out, log=quiet, frame_lists=FRAME_LISTS)
    assert {r["fid"] for r in pools.labelled} == {kitti_fid(3)} and all(r["has_label"] for r in pools.labelled)
    assert {r["fid"] for r in pools.unlabelled} == {kitti_fid(5)}
    assert not any(r["has_label"] for r in pools.unlabelled)
    assert {r["fid"] for r in pools.val} == {kitti_fid(8)} and all(r["has_label"] for r in pools.val)
    assert len(pools.anchors) == 4

    messages = []
    build_pools(cfg, folders, detections, tmp_path / "cache", out, log=messages.append, frame_lists=FRAME_LISTS)
    assert any("reused" in m for m in messages)
    other = write_detections(tmp_path / "other.txt", [3, 5, 8], folders["label_2"], shift=5.0)
    messages.clear()
    build_pools(cfg, folders, other, tmp_path / "cache", out, log=messages.append, frame_lists=FRAME_LISTS)
    assert not any("reused" in m for m in messages)                     # new detections: built again


def test_labelled_frames_must_come_from_the_train_split(tmp_path):
    folders, detections = fake_kitti(tmp_path)
    lists = dict(FRAME_LISTS, labelled=[kitti_fid(8)])
    with pytest.raises(ValueError):
        build_pools(tiny_config(), folders, detections, tmp_path / "cache", tmp_path / "out", log=quiet, frame_lists=lists)


def test_smoke_run_trains_evaluates_and_reloads(tmp_path):
    folders, detections = fake_kitti(tmp_path)
    cfg, out = tiny_config(), tmp_path / "out"
    pools = build_pools(cfg, folders, detections, tmp_path / "cache", out, log=quiet, frame_lists=FRAME_LISTS)
    result = train(cfg, pools, folders, tmp_path / "cache", out, device="cpu", smoke=3, log=quiet)
    assert result["reload_ok"] and result["epochs_done"] == 1
    entry = json.loads((out / "metrics.jsonl").read_text().splitlines()[-1])
    assert entry["val"]["all"]["n"] == 3 and 0 <= entry["val"]["all"]["mean_iou3d"] <= 1
    assert len((out / "val_objects.jsonl").read_text().splitlines()) == 3


def test_a_run_stopped_by_the_time_budget_resumes(tmp_path):
    folders, detections = fake_kitti(tmp_path)
    cfg, out = tiny_config(**{"train.epochs": 2}), tmp_path / "out"
    pools = build_pools(cfg, folders, detections, tmp_path / "cache", out, log=quiet, frame_lists=FRAME_LISTS)
    first = train(cfg, pools, folders, tmp_path / "cache", out, device="cpu", time_budget_hours=0.0, log=quiet)
    assert first["epochs_done"] == 1
    messages = []
    second = train(cfg, pools, folders, tmp_path / "cache", out, device="cpu", log=messages.append)
    assert second["epochs_done"] == 2 and any("resumed" in m for m in messages)
    epochs = [json.loads(line)["epoch"] for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert epochs == [0, 1]
