"""Calibration parsers, checked against the projection formulas written out in CLAUDE.md."""
import numpy as np
import pytest

from ocgf.geometry.calib import kitti360_camera, kitti360_pose_from_lidar, kitti_object_camera
from ocgf.geometry.projection import project
from ocgf.geometry.transforms import invert_rigid, rigid, rigid_from_3x4

from geometry_fixtures import AXES_CAM_LIDAR, points_in_front, rotation


def fmt(values) -> str:
    return " ".join(f"{v:.12e}" for v in np.ravel(values))


def project_homogeneous(P, T, points) -> np.ndarray:
    """u ~ P · T · [p, 1] with a 3x4 P, divided out. The reference formula."""
    h = np.column_stack([points, np.ones(len(points))]) @ T.T @ np.asarray(P).T
    return h[:, :2] / h[:, 2:3]


# ---------- KITTI-360 ----------

P_RECT_00 = np.array([[552.5, 0.0, 682.0, 0.0], [0.0, 552.5, 238.8, 0.0], [0.0, 0.0, 1.0, 0.0]])
R_RECT_00 = rotation(0.003, -0.0001, 0.007)
CAM_TO_VELO = rigid(AXES_CAM_LIDAR.T @ rotation(0.02, 0.04, -0.01), [0.80, 0.30, -0.18])   # camera 0 -> Velodyne
CAM_TO_POSE = rigid(rotation(0.01, 0.0, 1.57), [1.5, 0.0, 0.9])


@pytest.fixture
def kitti360_dir(tmp_path):
    (tmp_path / "perspective.txt").write_text("\n".join([
        "calib_time: 09-Jan-2012 13:57:47",
        "S_00: 1.392000e+03 5.120000e+02",
        "K_00: " + fmt(np.eye(3)),
        "S_rect_00: 1.408000e+03 3.760000e+02",
        "R_rect_00: " + fmt(R_RECT_00),
        "P_rect_00: " + fmt(P_RECT_00),
        "S_rect_01: 1.408000e+03 3.760000e+02",
        "R_rect_01: " + fmt(np.eye(3)),
        "P_rect_01: " + fmt(np.hstack([P_RECT_00[:, :3], [[-331.0], [0.0], [0.0]]])),
    ]) + "\n")
    (tmp_path / "calib_cam_to_velo.txt").write_text(fmt(CAM_TO_VELO[:3]) + "\n")
    (tmp_path / "calib_cam_to_pose.txt").write_text("\n".join(
        f"image_0{i}: " + fmt(CAM_TO_POSE[:3]) for i in range(4)) + "\n")
    return tmp_path


def test_kitti360_projection_matches_the_reference_formula(kitti360_dir):
    cam = kitti360_camera(kitti360_dir)
    assert (cam.width, cam.height) == (1408, 376)
    points = points_in_front()
    uv, _, valid = project(points, cam)
    # CLAUDE.md: u ~ P_rect_00[:, :3] · R_rect_00 · inv(calib_cam_to_velo) · [p, 1]
    T_ref = rigid(R_RECT_00, np.zeros(3)) @ invert_rigid(CAM_TO_VELO)
    expected = project_homogeneous(P_RECT_00[:, :3] @ np.hstack([np.eye(3), np.zeros((3, 1))]), T_ref, points)
    np.testing.assert_allclose(uv[valid], expected[valid], atol=1e-6)


def test_kitti360_projection_differs_without_r_rect(kitti360_dir):
    # guards against the known bug of dropping R_rect_00
    cam = kitti360_camera(kitti360_dir)
    points = points_in_front()
    uv, _, valid = project(points, cam)
    without = project_homogeneous(np.hstack([P_RECT_00[:, :3], np.zeros((3, 1))]), invert_rigid(CAM_TO_VELO), points)
    assert np.abs(uv[valid] - without[valid]).max() > 1.0


def test_kitti360_pose_from_lidar(kitti360_dir):
    T = kitti360_pose_from_lidar(kitti360_dir)
    np.testing.assert_allclose(T, CAM_TO_POSE @ invert_rigid(CAM_TO_VELO), atol=1e-12)


# ---------- KITTI Object ----------

P2 = np.array([[721.5, 0.0, 609.6, 44.86], [0.0, 721.5, 172.9, 0.2164], [0.0, 0.0, 1.0, 0.002746]])
R0_RECT = rotation(0.0005, -0.009, 0.0099)
TR_VELO_TO_CAM = rigid(rotation(0.01, 0.0, 0.0) @ AXES_CAM_LIDAR, [-0.004, -0.076, -0.27])


@pytest.fixture
def kitti_calib_file(tmp_path):
    path = tmp_path / "000123.txt"
    path.write_text("\n".join([
        "P0: " + fmt(np.hstack([P2[:, :3], np.zeros((3, 1))])),
        "P1: " + fmt(np.hstack([P2[:, :3], [[-386.1], [0.0], [0.0]]])),
        "P2: " + fmt(P2),
        "P3: " + fmt(np.hstack([P2[:, :3], [[-339.5], [2.2], [0.0027]]])),
        "R0_rect: " + fmt(R0_RECT),
        "Tr_velo_to_cam: " + fmt(TR_VELO_TO_CAM[:3]),
        "Tr_imu_to_velo: " + fmt(np.hstack([np.eye(3), np.zeros((3, 1))])),
    ]) + "\n")
    return path


def test_kitti_object_projection_matches_the_reference_formula(kitti_calib_file):
    cam = kitti_object_camera(kitti_calib_file, 1242, 375)
    points = points_in_front()
    uv, _, valid = project(points, cam)
    assert valid.sum() > 100
    # CLAUDE.md: u ~ P2 · R0_rect · Tr_velo_to_cam · [p, 1], with P2's non-zero 4th column
    expected = project_homogeneous(P2, rigid(R0_RECT, np.zeros(3)) @ TR_VELO_TO_CAM, points)
    np.testing.assert_allclose(uv[valid], expected[valid], atol=1e-6)


def test_rigid_from_3x4_round_trip():
    T = rigid(rotation(0.3, -0.2, 1.0), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rigid_from_3x4(T[:3].ravel()), T)
    np.testing.assert_allclose(invert_rigid(T) @ T, np.eye(4), atol=1e-12)
