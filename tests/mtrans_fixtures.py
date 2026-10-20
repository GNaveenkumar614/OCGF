"""A tiny MTrans config and a fake KITTI setup, so the teacher trains and predicts on CPU in seconds."""
from __future__ import annotations

from ocgf.datasets.kitti_object import kitti_fid
from ocgf.teachers.mtrans.ops import to_attrdict

from kitti_object_fixtures import make_fake_kitti_object, write_detections

TINY = {
    "experiment": "mtrans_tiny", "seed": 3, "classes": ["car", "large_vehicle", "pedestrian", "cyclist"],
    "labelled_frames": "configs/splits/kitti_object/mtrans_labelled_500.txt",
    "data": {"out_img_size": 32, "out_cloud_size": 64, "train_mask_ratio": [0.0, 0.9], "box2d_augmentation": True,
             "max_truncation": 0.95, "min_points": 5, "min_foreground": 5, "unlabelled_min_score": 0.5,
             "class_balance_power": 0.5},
    "train": {"epochs": 1, "lr": 1.0e-4, "warmup_rate": 0.01, "batch_size": 2, "unlabelled_batch_size": 2,
              "eval_batch_size": 4, "eval_every": 0, "log_every": 100, "num_workers": 0},
    "model": {"sparse_query_rate": 4, "box_drop": 0.0, "class_embedding_dim": 8,
              "POINT_ATTENTION": {"use_cls_token": True, "fore_attn": False, "num_layers": 1, "pos_embedding": "SIN",
                                  "fuse_method": "CAT", "input_img_channel": 16, "input_pts_channel": 16,
                                  "position_embedding_channel": 16, "hidden_size": 32, "num_heads": 4,
                                  "dropout_rate": 0.0, "intermediate_size": 32}},
}
ANCHORS = [[4.0, 1.7, 1.5], [9.0, 2.5, 3.2], [0.7, 0.6, 1.7], [1.8, 0.6, 1.7]]
FRAME_LISTS = {"train": [kitti_fid(3), kitti_fid(5)], "val": [kitti_fid(8)], "labelled": [kitti_fid(3)]}


def tiny_config(**overrides):
    cfg = to_attrdict({k: (dict(v) if isinstance(v, dict) else v) for k, v in TINY.items()})
    for path, value in overrides.items():
        section, key = path.split(".")
        cfg[section][key] = value
    return cfg


def fake_kitti(tmp_path):
    """(KITTI folders, detections file covering every fake frame)."""
    folders = make_fake_kitti_object(tmp_path)
    detections = write_detections(tmp_path / "training.txt", [3, 5, 8], folders["label_2"])
    return folders, detections
