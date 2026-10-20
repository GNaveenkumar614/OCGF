"""KITTI 3D Object reading: labels become our LiDAR-frame boxes, classes map to ours, camera 2 projects the pixels."""
import numpy as np

from ocgf.datasets.kitti_object import KITTI_TO_OURS, KittiObjectFrames, kitti_fid
from ocgf.geometry.boxes import points_in_box
from ocgf.geometry.projection import project

from kitti_object_fixtures import DEFAULT_OBJECTS, make_fake_kitti_object


def test_labels_come_back_as_the_lidar_boxes_they_were_written_from(tmp_path):
    objects = KittiObjectFrames(make_fake_kitti_object(tmp_path)).labels(kitti_fid(3))
    assert [o["kitti_class"] for o in objects] == [c for c, _ in DEFAULT_OBJECTS]      # DontCare is dropped
    for o, (_, box) in zip(objects, DEFAULT_OBJECTS):
        assert np.allclose(o["box"][:6], box[:6], atol=2e-3)                          # labels have 4 decimals
        assert abs((o["box"][6] - box[6] + np.pi) % (2 * np.pi) - np.pi) < 1e-3


def test_class_mapping_follows_the_draft():
    assert KITTI_TO_OURS == {"Car": "car", "Van": "car", "Truck": "large_vehicle", "Tram": "large_vehicle",
                             "Pedestrian": "pedestrian", "Person_sitting": "pedestrian", "Cyclist": "cyclist"}


def test_each_objects_points_fall_inside_its_converted_box(tmp_path):
    frames = KittiObjectFrames(make_fake_kitti_object(tmp_path, points_per_object=300))
    points = frames.points(kitti_fid(3))
    for o in frames.labels(kitti_fid(3)):
        assert points_in_box(points, o["box"]).sum() >= 285


def test_box_centres_project_inside_their_2d_boxes(tmp_path):
    frames = KittiObjectFrames(make_fake_kitti_object(tmp_path))
    cam = frames.camera(kitti_fid(3))
    assert (cam.width, cam.height) == (1242, 375)
    for o in frames.labels(kitti_fid(3)):
        uv, _, valid = project(o["box"][None, :3], cam)
        l, t, r, b = o["box_2d"]
        assert valid[0] and l < uv[0, 0] < r and t < uv[0, 1] < b
