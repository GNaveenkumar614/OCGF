import numpy as np
import pytest

from ocgf.geometry.projection import in_image, project, to_feature_coords, to_grid_sample_coords, unproject
from ocgf.geometry.transforms import invert_rigid, transform_points

from geometry_fixtures import camera, points_in_front

CAM = camera()


def test_round_trip_recovers_the_points():
    points = points_in_front()
    uv, depth, valid = project(points, CAM)
    assert valid.sum() > 100
    back = unproject(uv[valid], depth[valid], CAM)
    np.testing.assert_allclose(back, points[valid], atol=1e-9)


def test_points_behind_the_camera_are_invalid_and_nan():
    points = np.array([[-5.0, 0.0, 0.0], [0.8, 0.05, -0.3], [-30.0, 2.0, 1.0]])   # behind, at the camera centre, behind
    uv, depth, valid = project(points, CAM)
    assert not valid.any()
    assert np.isnan(uv).all()
    assert (depth <= 0.1).all()


def test_points_outside_the_image_are_invalid():
    points = np.array([[5.0, 30.0, 0.0], [5.0, -30.0, 0.0], [5.0, 0.0, 20.0]])    # far left, far right, high above
    uv, _, valid = project(points, CAM)
    assert not valid.any()
    assert np.isfinite(uv).all()


def test_point_on_the_optical_axis_hits_the_principal_point():
    centre = invert_rigid(CAM.T_cam_lidar)[:3, 3]
    axis = invert_rigid(CAM.T_cam_lidar)[:3, 2]                 # camera z axis in the LiDAR frame
    uv, _, valid = project((centre + 20.0 * axis)[None], CAM)
    assert valid[0]
    np.testing.assert_allclose(uv[0], CAM.K[:2, 2], atol=1e-9)


@pytest.mark.parametrize("u, v, inside", [
    (-0.5, 100.0, True), (-0.5 - 1e-9, 100.0, False),           # left edge of the first pixel
    (1408 - 0.5 - 1e-9, 100.0, True), (1408 - 0.5, 100.0, False),   # right edge of the last pixel
    (700.0, -0.5, True), (700.0, 376 - 0.5, False),
    (np.nan, 100.0, False),
])
def test_image_edges_follow_the_pixel_centre_convention(u, v, inside):
    assert bool(in_image(np.array([u, v]), 1408, 376)) is inside


@pytest.mark.parametrize("u, v, inside", [
    (-0.5 + 1e-6, 100.0, True), (-0.5 - 1e-6, 100.0, False),
    (1408 - 0.5 - 1e-6, 100.0, True), (1408 - 0.5 + 1e-6, 100.0, False),
])
def test_projected_points_near_the_image_edges(u, v, inside):
    point = unproject(np.array([[u, v]]), np.array([10.0]), CAM)
    _, _, valid = project(point, CAM)
    assert bool(valid[0]) is inside


def test_feature_coordinates():
    uv = np.array([[3.5, 3.5], [-0.5, -0.5], [11.5, 3.5]])
    np.testing.assert_allclose(to_feature_coords(uv, 8), [[0.0, 0.0], [-0.5, -0.5], [1.0, 0.0]])


def test_grid_sample_coordinates_span_the_feature_map():
    # a 1400 x 376 image at stride 8 -> a 175 x 47 feature map
    uv = np.array([[-0.5, -0.5], [1400 - 0.5, 376 - 0.5]])
    g = to_grid_sample_coords(uv, 8, 175, 47)
    np.testing.assert_allclose(g[0], [-1.0, -1.0])
    np.testing.assert_allclose(g[1], [1.0, 376 / (8 * 47) * 2 - 1])    # 376 rows cover 47 cells of 8 exactly


def test_crop_and_pad_shift_the_pixel_coordinates():
    points = points_in_front(200)
    uv, _, _ = project(points, CAM)
    cropped = CAM.crop(left=4, top=0, width=1400, height=378)         # 4 px off each side, 2 rows of padding
    uv2, _, _ = project(points, cropped)
    np.testing.assert_allclose(uv2, uv - [4.0, 0.0], atol=1e-9)


def test_resize_scales_about_pixel_edges():
    points = points_in_front(200)
    uv, _, _ = project(points, CAM)
    small = CAM.resize(704, 188)
    uv2, _, _ = project(points, small)
    np.testing.assert_allclose(uv2, (uv + 0.5) * 0.5 - 0.5, atol=1e-9)


def test_horizontal_fov():
    fov = CAM.horizontal_fov_deg
    assert 100.0 < fov < 106.0          # f = 552 px over 1408 px: about 104 degrees
    left = np.degrees(np.arctan((682.0 + 0.5) / 552.0))
    right = np.degrees(np.arctan((1408 - 0.5 - 682.0) / 552.0))
    assert fov == pytest.approx(left + right)


def test_transform_points_matches_matrix_product():
    points = points_in_front(10)
    T = CAM.T_cam_lidar
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ T.T
    np.testing.assert_allclose(transform_points(T, points), homogeneous[:, :3])
