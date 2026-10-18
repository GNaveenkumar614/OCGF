"""The KITTI-360 loader on a dev sample produced by the data-prep pipeline from fake data."""
import zipfile

import numpy as np
import pytest

from ocgf.datasets.ids import frame_id, parse_frame_id
from ocgf.datasets.kitti360 import Kitti360Sequence
from ocgf.datasets.prepare_kitti360 import PrepConfig, run_all
from ocgf.geometry.transforms import transform_points

from kitti360_fixtures import SEQ, make_fake_kitti360


@pytest.fixture(scope="module")
def seq(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("loader")
    paths = make_fake_kitti360(tmp, n_frames=40)
    cfg = PrepConfig(sequence=SEQ, input_root=paths["input_root"], out_dir=tmp / "out", scratch_dir=tmp / "scratch",
                     sources=paths["sources"], min_future_m=20.0, n_overlays=0, dev_frames=6)
    run_all(cfg, log=lambda msg: None)
    with zipfile.ZipFile(cfg.out_dir / "dev_sample.zip") as z:
        z.extractall(tmp)
    return Kitti360Sequence(tmp / "dev_sample" / "dataset_map.yaml")


def test_frame_ids_round_trip():
    fid = frame_id("kitti360", SEQ, 42)
    assert fid == f"kitti360/{SEQ}/0000000042"
    assert parse_frame_id(fid) == ("kitti360", SEQ, 42)
    with pytest.raises(ValueError):
        parse_frame_id("kitti360/x/42")


def test_frame_holds_only_inference_inputs_and_the_pose(seq):
    frames = seq.usable_frames()
    assert len(frames) == 6 and seq.scan_frames() == frames.tolist()
    f = seq.frame(int(frames[0]))
    assert f.fid == seq.fid(int(frames[0]))
    assert f.image.shape == (376, 1408, 3) and f.image.dtype == np.uint8
    assert f.points.shape[1] == 5 and f.points.dtype == np.float32
    assert set(vars(f)) == {"fid", "image", "camera", "points", "T_world_lidar"}       # no ground truth in a Frame


def test_pose_is_the_vehicle_pose_times_the_lidar_mounting(seq):
    f = int(seq.usable_frames()[0])
    origin = transform_points(seq.T_world_lidar(f), np.zeros((1, 3)))[0]
    # fake poses move 2 m per frame along world x; the LiDAR sits at (0.7, -0.3, 1.1) in the vehicle frame
    np.testing.assert_allclose(origin, [2.0 * f + 0.7, -0.3, 1.1], atol=1e-9)
    with pytest.raises(KeyError):
        seq.T_world_lidar(0)                                    # frames 0-1 have no pose


def test_ground_truth_boxes_come_in_the_lidar_frame(seq):
    f = int(seq.usable_frames()[0])
    gt = seq.gt_boxes(f)
    car = gt.params[gt.names.index("car")]
    expected = transform_points(np.linalg.inv(seq.T_world_lidar(f)), np.array([[30.0, 3.0, 0.8]]))[0]
    np.testing.assert_allclose(car[:3], expected, atol=1e-6)
