"""Road teacher on a synthetic street: future path, trajectory and wheel points, occlusion, LiDAR and camera labels."""
import numpy as np
import pytest

from ocgf.geometry.projection import project
from ocgf.geometry.transforms import rigid
from ocgf.teachers.road.config import RoadConfig
from ocgf.teachers.road.label import audit_counts, audit_summary, label_frame, load_result, save_result
from ocgf.teachers.road.lidar_label import image_label, nanmean, point_scores
from ocgf.teachers.road.trajectory import (RingScan, find_trajectory, future_path, occlusion_filter, path_positions,
                                           trajectory_polygon, wheel_points)

from geometry_fixtures import camera, rotation
from road_fixtures import KERB_Y, ROAD_Z, WALK_Z, straight_path_positions, street_scan

CFG = RoadConfig()
CAM = camera()
EYE = np.eye(4)


def straight_path():
    frames, pos = straight_path_positions()
    return future_path(0, frames, pos, EYE, CFG.trajectory_length)


@pytest.fixture(scope="module")
def street():
    points = street_scan()
    scan = RingScan(points, CFG)
    wheels = occlusion_filter(scan, wheel_points(scan, find_trajectory(scan, straight_path(), CFG), CFG), CAM, CFG)
    return points, scan, wheels


def test_future_path_is_the_next_50_m_in_the_current_lidar_frame():
    frames, pos = straight_path_positions()
    T = rigid(rotation(0, 0, 0.6), [100.0, -40.0, 3.0])                  # the car somewhere in the world
    world = (T[:3, :3] @ pos.T).T + T[:3, 3]
    path = future_path(0, frames, world, T, 50.0)
    assert np.allclose(path[:, 1], 0, atol=1e-9) and np.allclose(path[:, 2], 0, atol=1e-9)    # straight ahead
    assert 49 <= path[-1, 0] <= 50
    assert len(future_path(60, frames, world, T, 50.0)) == 0                  # not enough path left


def test_path_positions_use_the_pose_frame():
    T_pose_lidar = rigid(np.eye(3), [0.5, 0.0, 0.9])
    frames, pos = path_positions({0: EYE, 1: rigid(np.eye(3), [2.0, 0, 0])}, T_pose_lidar)
    assert np.allclose(pos[0], [-0.5, 0, -0.9]) and np.allclose(pos[1], [1.5, 0, -0.9])


def test_trajectory_follows_the_path_ring_by_ring_with_wheels_either_side(street):
    points, scan, wheels = street
    assert len(wheels) >= 20
    c, l, r = points[wheels[:, 1]], points[wheels[:, 2]], points[wheels[:, 3]]
    assert np.all(np.abs(c[:, 1]) < 1.0) and np.all(np.diff(c[:, 0]) > 0)            # on the path, outward
    assert np.allclose(l[:, 1], 1.0, atol=0.35) and np.allclose(r[:, 1], -1.0, atol=0.35)
    assert np.all(wheels[:, 0] >= CFG.drop_rings)


def test_an_obstacle_on_the_path_ends_the_trajectory_before_it():
    box = ((20.0, -1.0, ROAD_Z), (23.0, 1.0, -0.2))
    points = street_scan(obstacle=box)
    scan = RingScan(points, CFG)
    wheels = occlusion_filter(scan, wheel_points(scan, find_trajectory(scan, straight_path(), CFG), CFG), CAM, CFG)
    # it may end on the box's bottom edge (at ground height), never on or beyond the box
    assert len(wheels) >= 5 and points[wheels[:, 1], 0].max() <= 20.0 + 1e-6


def test_occlusion_filter_cuts_at_a_wheel_hidden_by_a_nearer_point():
    near, far = np.array([6.0, 0.0, ROAD_Z]), np.array([12.0, 1.0, ROAD_Z])
    cam_centre = np.linalg.inv(CAM.T_cam_lidar)[:3, 3]
    blocker = cam_centre + 0.5 * (far - cam_centre) + [0, 0, 0.6]            # on the sight line, raised: hides `far`
    pts = np.array([[*near, 0.5, 8], [*(near + [0, 1, 0]), 0.5, 8], [*(near - [0, 1, 0]), 0.5, 8],
                    [*far, 0.5, 9], [*(far - [0, 2, 0]), 0.5, 9], [*blocker, 0.5, 8]])
    scan = RingScan(pts, CFG)
    wheels = np.array([[8, 0, 1, 2], [9, 3, 3, 4]])
    assert len(occlusion_filter(scan, wheels, CAM, CFG)) == 1
    pts[5, :3] = cam_centre + 0.5 * (far - cam_centre) - [0, 0, 0.6]          # below the sight line: not hidden
    assert len(occlusion_filter(RingScan(pts, CFG), wheels, CAM, CFG)) == 2


def test_road_scores_high_and_sidewalk_beyond_the_kerb_low(street):
    points, scan, wheels = street
    s = point_scores(scan, wheels, CFG)
    scored = np.isfinite(s)
    road = scored & (np.abs(points[:, 1]) < KERB_Y - 0.5) & np.isclose(points[:, 2], ROAD_Z, atol=0.01)
    walk = scored & (np.abs(points[:, 1]) > KERB_Y + 0.5) & np.isclose(points[:, 2], WALK_Z, atol=0.01)
    assert road.sum() > 500 and walk.sum() > 100
    assert np.median(s[road]) > 0.9 and np.median(s[walk]) < 0.2
    assert not np.isfinite(s[points[:, 4] < CFG.drop_rings]).any()           # dropped rings are never scored


def test_image_label_is_road_on_the_road_and_unknown_in_the_sky(street):
    points, scan, wheels = street
    label = image_label(points, point_scores(scan, wheels, CFG), CAM, CFG)
    (ur, vr), (uw, vw) = project(np.array([[15.0, 0.0, ROAD_Z], [15.0, 6.0, WALK_Z]]), CAM)[0].round().astype(int)
    assert label[vr, ur] > 0.8 and label[vw, uw] < 0.3
    assert np.isnan(label[:20]).all()
    mask = trajectory_polygon(scan, wheels, CAM, CFG)
    assert mask[vr, ur] and not mask[vw, uw]


def fake_features(image, h=26, w=100):
    """Stand-in for DINOv2: road patches (the street's lower centre) one vector, everything else another."""
    torch = pytest.importorskip("torch")
    f = torch.zeros(h, w, 8)
    f[..., 1] = 1.0
    f[h // 2:, 30:70] = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
    return f


def test_camera_label_learns_road_from_the_path_and_reuses_the_prototype():
    pytest.importorskip("torch")
    from ocgf.teachers.road.camera_label import camera_label, patch_mask
    cfg = RoadConfig()
    mask = np.zeros((376, 1408), bool)
    mask[250:376, 600:800] = True                                             # path pixels on the road patches
    label, proto = camera_label(fake_features(None), mask, None, cfg, (376, 1408))
    assert label[350, 700] > 0.9 and label[50, 100] < 0.1
    small = np.zeros_like(mask)
    small[300:305, 700:705] = True                                            # too few path pixels: reuse
    label2, proto2 = camera_label(fake_features(None), small, proto, cfg, (376, 1408))
    assert proto2 is proto and np.allclose(label2, label)
    assert np.isnan(camera_label(fake_features(None), np.zeros_like(mask), None, cfg, (376, 1408))[0]).all()
    assert patch_mask(mask, 26, 100)[20, 50] and not patch_mask(mask, 26, 100)[2, 5]


def test_label_frame_end_to_end_and_its_statuses(tmp_path):
    points = street_scan()
    path = straight_path()
    image = np.zeros((376, 1408, 3), np.uint8)
    result, proto = label_frame(points, image, CAM, path, CFG)
    assert result["status"] == "ok" and result["point_scores"].shape == (len(points),)
    assert np.isfinite(result["label"]).any() and result["path_mask"].any() and result["wheels"].shape[1:] == (2, 3)
    save_result(tmp_path / "f.npz", result, save_parts=True)
    back = load_result(tmp_path / "f.npz")
    assert back["status"] == "ok" and back["label"].dtype == np.float16 and "lidar_label" in back

    assert label_frame(points, image, CAM, np.zeros((0, 3)), CFG)[0]["status"] == "no_path"
    off_road = path + [0, 30.0, 0]                                            # the path far from the street
    r = label_frame(points, image, CAM, off_road, CFG)[0]
    assert r["status"] == "no_trajectory" and np.isnan(r["label"]).all()      # unknown, never camera-only
    fewer_rings = points[points[:, 4] < 63]
    assert label_frame(fewer_rings, image, CAM, path, CFG)[0]["status"] == "rings"


def test_nanmean_and_audit():
    a, b = np.array([1.0, np.nan, 0.2, np.nan]), np.array([0.0, 0.6, np.nan, np.nan])
    assert np.allclose(nanmean(a, b, False), [0.5, 0.6, 0.2, np.nan], equal_nan=True)
    assert np.allclose(nanmean(a, b, True), [0.5, np.nan, np.nan, np.nan], equal_nan=True)
    label = np.array([[0.9, 0.9], [0.1, np.nan]])
    semantic = np.array([[7, 8], [7, 7]])
    assert audit_counts(label, semantic).tolist() == [1, 1, 2]
    assert audit_summary(np.array([1, 1, 2]))["iou"] == 0.25


def test_adapted_files_keep_the_mit_notice():
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1] / "ocgf" / "teachers" / "road"
    for name in ("trajectory.py", "lidar_label.py", "camera_label.py"):
        assert "Eerik Alamikkotervo" in (folder / name).read_text(encoding="utf-8")[:600]
    assert "MIT License" in (folder / "LICENSE-lidar-camera-road-autolabeling").read_text(encoding="utf-8")
    assert not (folder / "crf.py").exists()                                   # CC BY-NC 4.0: never copied
