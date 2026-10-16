from pathlib import Path

from ocgf.datasets.dataset_map import discover_kitti360, missing_required, read_dataset_map, write_dataset_map

from kitti360_fixtures import SEQ, make_fake_kitti360


def test_discovers_every_component(tmp_path):
    paths = make_fake_kitti360(tmp_path, n_frames=6)
    found = discover_kitti360(paths["input_root"], SEQ)
    assert missing_required(found) == []
    k = paths["starter"]
    assert found["images"] == k / "data_2d_raw" / SEQ / "image_00" / "data_rect"
    assert found["image_timestamps"].name == "timestamps.txt"
    assert found["calibration"] == k / "calibration"
    assert found["poses"] == k / "data_poses" / SEQ
    assert found["semantics3d"] == k / "data_3d_semantics" / SEQ
    assert found["semantics2d"].parts[-2:] == (SEQ, "image_00") and (found["semantics2d"] / "semantic").is_dir()
    assert found["boxes3d"].name == f"{SEQ}.xml"


def test_prefers_train_boxes_and_ignores_train_full(tmp_path):
    for sub in ("train_full", "train"):
        (tmp_path / "boxes" / sub).mkdir(parents=True)
        (tmp_path / "boxes" / sub / f"{SEQ}.xml").write_text("<opencv_storage/>")
    assert discover_kitti360(tmp_path, SEQ)["boxes3d"].parent.name == "train"


def test_reports_what_is_missing(tmp_path):
    found = discover_kitti360(tmp_path, SEQ)
    assert missing_required(found) == ["images", "image_timestamps", "calibration", "poses", "semantics3d", "boxes3d"]


def test_map_paths_inside_its_folder_are_relative(tmp_path):
    (tmp_path / "data" / "scans").mkdir(parents=True)
    write_dataset_map(tmp_path / "data" / "dataset_map.yaml", SEQ,
                      {"scans": tmp_path / "data" / "scans", "elsewhere": Path("/kaggle/input/x")})
    text = (tmp_path / "data" / "dataset_map.yaml").read_text()
    assert "scans: scans" in text
    seq, components = read_dataset_map(tmp_path / "data" / "dataset_map.yaml")
    assert seq == SEQ
    assert components["scans"].resolve() == (tmp_path / "data" / "scans").resolve()
