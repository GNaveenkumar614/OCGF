"""Boundary weighting and the self-supervised consistency terms (cross-modal, temporal), the frame pairing behind the
temporal term, and a training run that uses them."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ocgf.datasets.student_sample import boundary_distance
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.geometry.transforms import rigid
from ocgf.geometry.warp import warp_grid
from ocgf.training.data import collate, previous_frames
from ocgf.training.losses import LossWeights, cross_modal, student_loss, temporal

from test_student import small_model, synthetic_sample


def test_boundary_distance_is_zero_on_the_road_edge_and_grows_away_from_it():
    g = BEVGrid()
    road = np.zeros(g.shape, np.float32)
    road[:, 70:90] = 1.0                                        # a 5 m wide road along x
    road[:, :10] = np.nan
    d = boundary_distance(road, g)
    assert d[100, 70] == 0 and d[100, 89] == 0 and d[100, 80] == pytest.approx(2.25, abs=0.01)
    assert (boundary_distance(np.full(g.shape, np.nan, np.float32), g) == 1000).all()


def test_previous_frames_pair_each_frame_with_the_sample_about_5_m_back():
    idx = {"frame": np.arange(10), "odometer_m": np.array([0, 5, 10, 11, 16, 30, 35, 40, 45, 50], float)}
    seq = SimpleNamespace(index=idx)
    pairs = previous_frames(seq, [1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6])
    assert pairs == {1: 0, 2: 1, 3: 1, 4: 3, 6: 5}              # 3 is 1 m past 2: it pairs with 1; 5 is 14 m on: none
    dense = SimpleNamespace(index={"frame": np.arange(30), "odometer_m": np.arange(30) * 0.8})
    assert previous_frames(dense, [20], range(30)) == {20: 17}  # every frame stored: the one >= 2 m back


def test_cross_modal_consistency_vanishes_only_where_a_sensor_is_missing():
    out = {"aux_road_camera": torch.full((1, 4, 4), 3.0), "aux_road_lidar": torch.full((1, 4, 4), -3.0),
           "valid_camera": torch.ones(1, 4, 4, dtype=torch.bool), "valid_lidar": torch.ones(1, 4, 4, dtype=torch.bool)}
    assert cross_modal(out) > 2.0                                # they disagree everywhere
    out["valid_lidar"] = torch.zeros(1, 4, 4, dtype=torch.bool)
    assert cross_modal(out) == 0


def temporal_batch(shift_m: float, has_prev: bool = True):
    g = BEVGrid()
    T_prev = np.eye(4)
    T_now = rigid(np.eye(3), [shift_m, 0, 0])
    coords, inside = warp_grid(T_now, T_prev, g)
    return {"warp_grid": torch.from_numpy(coords)[None], "warp_inside": torch.from_numpy(inside)[None],
            "has_prev": torch.tensor([has_prev]), "unk_t": torch.zeros(1, *g.shape), "prev/unk_t": torch.zeros(1, *g.shape)}


def test_temporal_consistency_is_zero_for_a_static_scene_seen_consistently():
    g = BEVGrid()
    xs, _ = g.center_grid()
    road = torch.from_numpy(np.where(xs > 20, 4.0, -4.0).astype(np.float32))[None]   # road beyond x = 20 m (world)
    occ = torch.full((1, *g.shape), -5.0)
    prev_out = {"road": road, "occ": occ}
    shifted = torch.from_numpy(np.where(xs > 15, 4.0, -4.0).astype(np.float32))[None]  # after driving 5 m: beyond 15 m
    assert temporal({"road": shifted, "occ": occ}, prev_out, temporal_batch(5.0)) < 1e-3
    assert temporal({"road": road, "occ": occ}, prev_out, temporal_batch(5.0)) > 0.01   # ignores the motion: penalised
    assert temporal({"road": road, "occ": occ}, prev_out, temporal_batch(5.0, has_prev=False)) == 0


def paired(seed):
    s = synthetic_sample(seed=seed)
    p = synthetic_sample(seed=seed + 100)
    for k in ("image", "grid_uv", "M_C", "grid_uv_flat", "fov_flat", "pt_feats", "pt_cell", "Q_geo", "valid_L"):
        s[f"prev/{k}"] = p[k]
    s["prev/unk_t"] = p["unk_t"]
    s["road_bd"] = torch.rand(16, 16) * 3
    coords, inside = warp_grid(np.eye(4), np.eye(4), BEVGrid(x_max=4.0, y_min=-2.0, y_max=2.0))
    s["warp_grid"], s["warp_inside"], s["has_prev"] = torch.from_numpy(coords), torch.from_numpy(inside), torch.tensor(True)
    return s


def test_collate_offsets_the_paired_frames_points_too():
    a, b = paired(0), paired(1)
    batch = collate([a, b])
    assert torch.equal(batch["prev/pt_cell"][len(a["prev/pt_cell"]):], b["prev/pt_cell"] + 16 * 16)


def test_all_terms_join_the_loss_and_a_training_run_uses_them(tmp_path):
    from ocgf.training.train import TrainConfig, previous_outputs, train
    model = small_model("concat")
    batch = collate([paired(0), paired(1)])
    out = model(batch)
    total, parts = student_loss(out, batch, LossWeights(), previous_outputs(model, batch, False, torch.device("cpu")))
    assert {"road", "occ", "unk", "aux", "boundary", "cross", "temporal"} <= set(parts)
    assert all(torch.isfinite(v) for v in parts.values())
    _, off = student_loss(out, batch, LossWeights(cross=0.0, temporal=0.0, boundary=0.0), None)
    assert not {"cross", "temporal", "boundary"} & set(off)

    class Paired(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return paired(i)
    cfg = TrainConfig(epochs=1, batch_size=2, num_workers=0, amp=False, log_every=1, warmup_steps=1)
    lines = []
    train(small_model("concat"), Paired(), None, cfg, LossWeights(), tmp_path / "run", log=lines.append)
    assert any("temporal" in l and "cross" in l for l in lines)
