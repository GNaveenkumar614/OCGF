import json
from pathlib import Path

import numpy as np
import pytest

from ocgf.datasets.splits import content_hash, distance_split, from_ranges, to_ranges

SPLIT_DIR = Path(__file__).resolve().parents[1] / "configs" / "splits"
# Pinned contents of the frozen splits. A change here must be a deliberate, reviewed decision.
FROZEN = {
    "kitti360_d0000_dev_v1": "b86b396f77d0d46d11957ec2ec2dde6655b4d0ce6dc2ecbca1617b2ab989fc11",
}


def index_along(xy, usable=None):
    xy = np.asarray(xy, dtype=np.float64)
    odo = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    n = len(xy)
    return {"frame": np.arange(n), "usable": np.ones(n, bool) if usable is None else usable,
            "odometer_m": odo, "x_world": xy[:, 0], "y_world": xy[:, 1]}


def test_ranges_round_trip():
    frames = np.array([3, 4, 5, 9, 11, 12])
    assert to_ranges(frames) == [[3, 5], [9, 9], [11, 12]]
    np.testing.assert_array_equal(from_ranges(to_ranges(frames)), frames)


def test_straight_drive_split_with_buffers():
    xy = np.column_stack([np.arange(2000.0), np.zeros(2000)])          # 1 m per frame
    splits, report = distance_split(index_along(xy))
    tr, va, te = splits["train"], splits["val"], splits["test"]
    assert va.min() - tr.max() >= 100 and te.min() - va.max() >= 100     # at least 100 m between stretches
    np.testing.assert_allclose(report["cuts_odometer_m"], [0.70 * 1999, 0.85 * 1999], atol=0.1)   # cuts by distance
    assert not (set(tr) & set(va) or set(va) & set(te) or set(tr) & set(te))
    assert report["dropped_near_cuts"] > 0


def test_revisits_of_train_areas_leave_val_and_test():
    # out 1000 m, then back along the same road: the way back passes the train stretch again
    out = np.column_stack([np.arange(1000.0), np.zeros(1000)])
    back = np.column_stack([np.arange(999.0, -1.0, -1.0), np.full(1000, 10.0)])
    splits, report = distance_split(index_along(np.vstack([out, back])))
    assert report["dropped_as_revisits"]["test"] > 0
    xy = np.vstack([out, back])
    tr = xy[splits["train"]]
    for name in ("val", "test"):
        d = np.min(np.linalg.norm(xy[splits[name]][:, None] - tr[None], axis=2), axis=1)
        assert (d >= 50).all()


def test_unusable_frames_are_never_split():
    xy = np.column_stack([np.arange(500.0), np.zeros(500)])
    usable = np.ones(500, bool)
    usable[::7] = False
    splits, _ = distance_split(index_along(xy, usable), buffer_m=20.0)
    assert not np.isin(np.concatenate(list(splits.values())), np.flatnonzero(~usable)).any()


def test_hash_ignores_key_order():
    a = {"train": np.array([1, 2]), "val": np.array([5])}
    assert content_hash(a) == content_hash({"val": np.array([5]), "train": np.array([1, 2])})


def test_paper_drive_split_is_the_decided_one():
    # kitti360_paper_v1, decided 2026-10-18 from scripts/drive_stats.py. Changing it means a new version, not an edit.
    record = json.loads((SPLIT_DIR / "kitti360_paper_v1.json").read_text())
    drive = lambda d: f"2013_05_28_drive_{d}_sync"
    assert record["train"] == [drive(d) for d in ("0000", "0002", "0004", "0005")]
    assert record["val"] == [drive("0006")]
    assert record["test"] == [drive(d) for d in ("0009", "0010")]
    parts = record["train"] + record["val"] + record["test"] + record["unused"]
    assert len(parts) == len(set(parts)) == 9                     # the nine KITTI-360 training drives, each once


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_frozen_splits_never_change(name):
    path = SPLIT_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not created yet")
    record = json.loads(path.read_text())
    splits = {k: from_ranges(v) for k, v in record["splits"].items()}
    assert content_hash(splits) == record["content_sha256"], "the split file was edited by hand"
    assert FROZEN[name] == record["content_sha256"], "frozen split changed: make a new version instead"
    frames = np.concatenate(list(splits.values()))
    assert len(frames) == len(set(frames.tolist())), "a frame is in two splits"
