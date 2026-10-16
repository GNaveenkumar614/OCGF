import numpy as np
import pytest

from ocgf.datasets import frame_index as fi
from ocgf.geometry.transforms import rigid


def test_parse_timestamps_keeps_nanosecond_differences():
    t = fi.parse_timestamps(["2013-05-28 08:46:02.904668160", "2013-05-28 08:46:03.008061952", ""])
    assert len(t) == 2
    assert (t[1] - t[0]) * 1e3 == pytest.approx(103.393792, abs=1e-3)


def test_read_poses_and_cam0_to_world(tmp_path):
    T = rigid(np.eye(3), [1.0, 2.0, 3.0])
    (tmp_path / "poses.txt").write_text("5 " + " ".join(map(str, T[:3].ravel())) + "\n")
    (tmp_path / "cam0_to_world.txt").write_text("5 " + " ".join(map(str, T.ravel())) + "\n")
    frames, Tp = fi.read_poses(tmp_path / "poses.txt")
    frames2, Tc = fi.read_cam0_to_world(tmp_path / "cam0_to_world.txt")
    assert list(frames) == [5] and list(frames2) == [5]
    np.testing.assert_allclose(Tp[0], T)
    np.testing.assert_allclose(Tc[0], T)


def test_semantic_windows_and_overlaps():
    windows = fi.semantic_windows(["0000000020_0000000039.ply", "0000000000_0000000025.ply", "notes.txt"])
    np.testing.assert_array_equal(windows, [[0, 25], [20, 39]])
    # frames 20-25 are in both windows: each goes to the one it is most central in
    np.testing.assert_array_equal(fi.window_of([0, 12, 21, 24, 30, 39, 40], windows), [0, 0, 0, 1, 1, 1, -1])


def test_sample_every_keeps_the_spacing():
    distance = np.arange(0, 30, 0.7)
    mask = np.ones(len(distance), dtype=bool)
    mask[5:8] = False
    picks = distance[fi.sample_every(distance, mask, 5.0)]
    assert picks[0] == 0.0
    assert (np.diff(picks) >= 5.0).all()
    assert not fi.sample_every(distance, mask, 5.0)[5:8].any()


def test_compress_ranges():
    assert fi.compress_ranges([3, 0, 1, 2, 7, 9, 10]) == "0-3, 7, 9-10"
    assert fi.compress_ranges([]) == "none"


def build(**overrides):
    n = 30
    args = dict(
        image_frames=np.arange(n), scan_frames=np.setdiff1d(np.arange(n), [7]),
        image_times=np.arange(n) * 0.1, scan_times=np.arange(n) * 0.1 + 0.004,
        pose_frames=np.arange(2, n), T_world_lidar=np.stack([rigid(np.eye(3), [3.0 * f, 0, 0]) for f in range(2, n)]),
        windows=np.array([[0, 29]]), min_future_m=20.0, step_m=5.0, max_dt_ms=20.0)
    args.update(overrides)
    return fi.build_frame_index(**args)


def test_frame_index_columns_and_usability():
    index = build()
    assert set(index) == set(fi.COLUMNS)
    assert not index["has_scan"][7] and not index["has_pose"][:2].any()
    np.testing.assert_allclose(index["dt_ms"], 4.0)
    assert index["odometer_m"][2] == 0.0 and index["odometer_m"][29] == pytest.approx(81.0)
    assert index["future_path_m"][2] == pytest.approx(81.0)
    usable = np.flatnonzero(index["usable"])
    # posed, scanned, and at least 20 m (about 7 frames of 3 m) before the end of the poses
    assert usable.min() == 2 and usable.max() == 22 and 7 not in usable
    picked = index["odometer_m"][index["sample_5m"]]
    assert (np.diff(picked) >= 5.0).all()


def test_misaligned_frames_are_not_usable():
    scan_times = np.arange(30) * 0.1 + 0.004
    scan_times[10] += 0.05                                   # 54 ms off
    index = build(scan_times=scan_times)
    assert not index["aligned"][10] and not index["usable"][10]
    assert index["usable"][11]


def test_csv_round_trip(tmp_path):
    index = build()
    fi.write_csv(tmp_path / "index.csv", index)
    back = fi.read_csv(tmp_path / "index.csv")
    for name in fi.COLUMNS:
        if back[name].dtype.kind == "f":
            np.testing.assert_allclose(back[name], index[name], atol=1e-6, equal_nan=True)
        else:
            np.testing.assert_array_equal(back[name], index[name])
