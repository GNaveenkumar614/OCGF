"""Evaluation ground truth on a synthetic street: ground classes, occupancy, obstacles, observability and free space."""
import numpy as np
import pytest

from ocgf.datasets.boxes3d import CLASSES, FrameBoxes
from ocgf.datasets.kitti360_labels import ROAD
from ocgf.evaluation.gt_builder import build_frame_gt, free_space, load_layers, save_layers
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.geometry.transforms import rigid, transform_points

from geometry_fixtures import camera, rotation

GRID = BEVGrid()
Z = -1.73
T_WORLD_LIDAR = rigid(rotation(0.0, 0.0, 0.7), [1000.0, 3000.0, 115.0])   # the clouds come in world coordinates
CAR = (15.0, 3.0, Z + 0.75, 4.0, 1.8, 1.5, 0.0)


def _plane(x0, x1, y0, y1, z, step=0.1):
    x, y = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step), indexing="ij")
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


def street():
    road, walk = _plane(0, 50, -6, 6, Z), np.concatenate([_plane(0, 50, 6, 9, Z + 0.12), _plane(0, 50, -9, -6, Z + 0.12)])
    xw, zw = np.meshgrid(np.arange(0, 50, 0.1), np.arange(0.0, 3.0, 0.2), indexing="ij")
    wall = np.column_stack([xw.ravel(), np.full(xw.size, 9.5), Z + 0.12 + zw.ravel()])
    xyz = np.concatenate([road, walk, wall])
    sem = np.concatenate([np.full(len(road), ROAD), np.full(len(walk), 8), np.full(len(wall), 11)]).astype(np.int16)
    static = {"xyz": transform_points(T_WORLD_LIDAR, xyz), "semantic": sem}
    person = np.column_stack([np.full(30, 25.1), np.full(30, -2.1), np.linspace(Z + 0.3, Z + 1.7, 30)])
    dyn = np.concatenate([person, person + [0, 4.0, 0]])
    dynamic = {"xyz": transform_points(T_WORLD_LIDAR, dyn), "semantic": np.full(60, 24, np.int16),
               "timestamp": np.r_[np.full(30, 5), np.full(30, 7)]}
    return static, dynamic


def car_boxes():
    one = lambda v, dt=np.int64: np.array([v], dtype=dt)
    return FrameBoxes(params=np.array([CAR]), names=["car"], classes=one(CLASSES.index("car")), kitti_ids=one(13),
                      instance_ids=one(1), dynamic=one(False, bool), interpolated=one(False, bool), rider_of=one(-1))


def cell(x, y):
    ix, iy = GRID.index(x, y)
    return int(ix), int(iy)


@pytest.fixture(scope="module")
def frame5():
    static, dynamic = street()
    return build_frame_gt(GRID, T_WORLD_LIDAR, static, dynamic, 5, car_boxes(), camera())


def test_ground_height_and_class(frame5):
    assert frame5["ground_class"][cell(10, 0)] == ROAD and frame5["ground_class"][cell(10, 7.5)] == 8
    assert abs(float(frame5["ground_z"][cell(10, 0)]) - Z) < 0.01


def test_boxes_are_occupied_and_walls_and_moving_people_are_obstacles(frame5):
    assert frame5["occupied"][cell(15, 3)] == 1 and frame5["occupied"][cell(15, -3)] == 0
    assert frame5["obstacle"][cell(30, 9.5)] and not frame5["obstacle"][cell(30, 0)]
    assert frame5["obstacle"][cell(25.1, -2.1)]                       # the person where it is at frame 5
    assert not frame5["obstacle"][cell(25.1, 1.9)]                    # ... not where it is at frame 7


def test_observability_behind_the_car_and_outside_the_camera_view(frame5):
    behind = cell(25, 5)                                              # on the ray from the sensor through the car
    assert not frame5["observed_lidar"][behind] and not frame5["observed_cam"][behind]
    assert frame5["observed_lidar"][cell(25, -1)] and frame5["observed_cam"][cell(25, -1)]
    assert frame5["observed_lidar"][cell(1.0, 7.5)] and not frame5["observed_cam"][cell(1.0, 7.5)]   # beside the camera
    assert not frame5["observed_lidar"][cell(2.0, 0.0)] and not frame5["observed_cam"][cell(2.0, 0.0)]  # below the beams
    assert not frame5["labelled"][cell(40, -15)]                      # beyond the wall: no ground truth


def test_free_space_is_observed_unoccupied_road(frame5):
    free = free_space(frame5)
    assert free[cell(10, 0)] and not free[cell(15, 3)] and not free[cell(25, 5)] and not free[cell(25.1, -2.1)]
    assert not free[cell(10, 7.5)]                                    # sidewalk is not drivable
    assert free_space(frame5, drivable=(ROAD, 8))[cell(10, 7.5)]


def test_layers_round_trip(frame5, tmp_path):
    save_layers(tmp_path / "f.npz", frame5)
    back = load_layers(tmp_path / "f.npz")
    assert all(np.array_equal(back[k], frame5[k], equal_nan=True) for k in frame5)
