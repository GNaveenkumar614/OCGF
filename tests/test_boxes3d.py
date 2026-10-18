import numpy as np
import pytest

from ocgf.datasets.boxes3d import CLASSES, Boxes3D, interpolate_transform, parse_bboxes
from ocgf.geometry.transforms import rigid, transform_points

from geometry_fixtures import rotation
from kitti360_fixtures import FAKE_OBJECTS, bbox_xml, cube_transform


@pytest.fixture
def boxes(tmp_path):
    path = tmp_path / "boxes.xml"
    path.write_text(bbox_xml(FAKE_OBJECTS))
    return Boxes3D.from_xml(path)


def test_parse_reads_ids_names_and_transforms(tmp_path):
    path = tmp_path / "boxes.xml"
    path.write_text(bbox_xml(FAKE_OBJECTS))
    objects = parse_bboxes(path)
    assert len(objects) == len(FAKE_OBJECTS)
    car = objects[0]
    assert (car.kitti_id, car.name, car.timestamp, car.start_frame, car.end_frame) == (13, "car", -1, 0, 25)
    np.testing.assert_allclose(np.linalg.norm(car.transform[:3, :3], axis=0), [4.2, 1.8, 1.5])
    assert objects[1].name == "unknown object"


def test_static_objects_are_valid_within_their_frames(boxes):
    assert {o.name for o, _, _ in boxes.world_boxes(25)} >= {"car", "unknown object"}
    assert "car" not in {o.name for o, _, _ in boxes.world_boxes(26)}


def test_moving_objects_are_interpolated_only_across_short_gaps(boxes):
    def person(frame):
        hits = [(M, interp) for o, M, interp in boxes.world_boxes(frame) if o.name == "person"]
        return hits[0] if hits else None
    M, interp = person(9)
    assert not interp and M[0, 3] == pytest.approx(26.0 + 4.5)
    M, interp = person(11)                                   # between rows 10 and 13: gap 3 <= 5
    assert interp and M[0, 3] == pytest.approx(26.0 + 5.5)
    assert person(16) is None                                # between rows 13 and 20: gap 7 > 5
    assert person(16) is None and boxes.world_boxes(16, max_gap=7)


def test_interpolation_turns_the_short_way():
    a = cube_transform((0, 0, 0), (4, 2, 1), np.radians(170))
    b = cube_transform((2, 0, 0), (4, 2, 1), np.radians(-170))
    M = interpolate_transform(a, b, 0.5)
    assert np.degrees(np.arctan2(M[1, 0], M[0, 0])) == pytest.approx(180.0, abs=1e-6) or \
        np.degrees(np.arctan2(M[1, 0], M[0, 0])) == pytest.approx(-180.0, abs=1e-6)
    np.testing.assert_allclose(np.linalg.norm(M[:3, :3], axis=0), [4, 2, 1])


def test_boxes_in_the_lidar_frame(boxes):
    T_world_lidar = rigid(rotation(0.0, 0.0, 0.3), [5.0, 1.0, 1.7])          # the LiDAR turned 0.3 rad to the left
    fb = boxes.at(10, T_world_lidar)
    car = fb.params[fb.names.index("car")]
    expected = transform_points(np.linalg.inv(T_world_lidar), np.array([[30.0, 3.0, 0.8]]))[0]
    np.testing.assert_allclose(car[:3], expected, atol=1e-9)
    np.testing.assert_allclose(car[3:6], [4.2, 1.8, 1.5], atol=1e-9)
    assert car[6] == pytest.approx(0.1 - 0.3)


def test_classes_and_the_cyclist_merge(boxes):
    fb = boxes.at(10, np.eye(4))
    cls = dict(zip(fb.names, fb.class_names()))
    assert cls["car"] == "car" and cls["person"] == "pedestrian" and cls["unknown object"] == "other"
    assert cls["bicycle"] == "cyclist" and cls["rider"] == "cyclist"           # ridden: part of a cyclist
    rider, bike = fb.names.index("rider"), fb.names.index("bicycle")
    assert fb.rider_of[rider] == bike and fb.rider_of[bike] == -1
    assert set(fb.class_names()) <= set(CLASSES)
    assert fb.dynamic[fb.names.index("person")] and not fb.dynamic[fb.names.index("car")]
