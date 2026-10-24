"""Student: sample building, lifting alignment, pillar pooling, forward shapes per mode, losses on empty targets, and a
training smoke run with checkpoint and resume (CPU, stub backbone)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ocgf.datasets.student_sample import IMAGE_H, IMAGE_W, OUTSIDE, cell_projection, point_features, prepare_image
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.models.camera import lift
from ocgf.models.lidar import PillarEncoder
from ocgf.models.student import HEADS, MODES, Student, StudentConfig
from ocgf.training.data import collate
from ocgf.training.losses import LossWeights, masked_bce, student_loss
from ocgf.training.train import TrainConfig, train

from geometry_fixtures import camera

GRID = BEVGrid()


def test_image_is_cropped_and_padded_to_a_multiple_of_8_and_14_with_the_intrinsics_following():
    cam = camera()
    img = np.random.default_rng(0).integers(0, 255, (cam.height, cam.width, 3), dtype=np.uint8)
    out, cam2, rows = prepare_image(img, cam)
    assert out.shape == (IMAGE_H, IMAGE_W, 3) and IMAGE_H % 56 == 0 and IMAGE_W % 56 == 0 and rows == cam.height
    assert np.array_equal(out[:rows], img[:, 4:4 + IMAGE_W]) and not out[rows:].any()
    assert cam2.K[0, 2] == cam.K[0, 2] - 4 and cam2.K[1, 2] == cam.K[1, 2] and (cam2.width, cam2.height) == (IMAGE_W, IMAGE_H)


def test_cell_projection_gives_grid_sample_coordinates_of_the_ground_point():
    cam = camera().crop(4, 0, IMAGE_W, IMAGE_H)
    z = np.full(GRID.shape, -1.73)
    g, ok = cell_projection(cam, z, GRID, real_rows=376)
    from ocgf.geometry.projection import project
    ix, iy = GRID.index(20.0, 1.0)
    (u, v), = project(np.array([[20.125, 1.125, -1.73]]), cam)[0]
    assert ok[ix, iy] and np.allclose(g[ix, iy], [2 * (u + 0.5) / IMAGE_W - 1, 2 * (v + 0.5) / IMAGE_H - 1], atol=1e-5)
    assert not ok[GRID.index(1.0, 0.0)] and (g[GRID.index(1.0, 0.0)] == OUTSIDE).all()   # below the image


def test_lift_samples_the_feature_at_each_cells_pixel():
    """A stride-8 map whose channels hold each feature cell's centre in pixels (pixel-area convention) gives back
    every cell's (u, v) where it projects inside the image."""
    h, w = IMAGE_H // 8, IMAGE_W // 8
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    feat = torch.tensor(np.stack([8 * jj + 3.5, 8 * ii + 3.5]), dtype=torch.float32)[None]
    cam = camera().crop(4, 0, IMAGE_W, IMAGE_H)
    g, ok = cell_projection(cam, np.full(GRID.shape, -1.73), GRID, real_rows=376)
    bev = lift(feat, torch.from_numpy(g)[None], torch.from_numpy(ok)[None])[0].numpy()
    u = (g[..., 0] + 1) * IMAGE_W / 2 - 0.5
    v = (g[..., 1] + 1) * IMAGE_H / 2 - 0.5
    inner = ok & (u > 4) & (u < IMAGE_W - 5) & (v > 4) & (v < 376 - 5)
    assert inner.sum() > 1000
    assert np.allclose(bev[0][inner], u[inner], atol=1e-3) and np.allclose(bev[1][inner], v[inner], atol=1e-3)
    assert (bev[:, ~ok] == 0).all()


def test_point_features_and_pillar_max_per_cell():
    pts = np.array([[10.1, 0.1, -1.7, 0.5], [10.2, 0.2, -1.0, 0.1], [30.0, -5.0, -1.7, 0.9], [60.0, 0.0, 0.0, 0.0]])
    f, cell = point_features(pts, np.full(GRID.shape, -1.73, np.float32), GRID)
    assert f.shape == (3, 10) and np.allclose(f[:, 4], pts[:3, 2] + 1.73, atol=1e-6)       # the far point is dropped
    assert np.all(np.abs(f[:, 5:7]) <= 0.5) and cell[0] == cell[1] != cell[2]
    enc = PillarEncoder(10, 8, GRID.shape).eval()
    x = enc.point_mlp(torch.from_numpy(f)).detach()
    out = torch.zeros(2 * 200 * 160, 8)
    out.scatter_reduce_(0, torch.from_numpy(cell)[:, None].expand(-1, 8), x, reduce="amax", include_self=False)
    assert torch.allclose(out[cell[0]], torch.maximum(x[0], x[1])) and (out.abs().sum(1) > 0).sum() <= 2


def synthetic_sample(grid_shape=(16, 16), hw=(56, 112), n_points=300, seed=0):
    rng = np.random.default_rng(seed)
    nx, ny = grid_shape
    g = rng.uniform(-1, 1, (nx, ny, 2)).astype(np.float32)
    road = rng.uniform(0, 1, (nx, ny)).astype(np.float32)
    road[rng.uniform(size=(nx, ny)) < 0.3] = np.nan
    occ = np.where(rng.uniform(size=(nx, ny)) < 0.5, np.nan, (road > 0.5)).astype(np.float32)
    s = {"image": torch.from_numpy(rng.integers(0, 255, (3, *hw), dtype=np.uint8)),
         "grid_uv": torch.from_numpy(g), "M_C": torch.from_numpy(rng.uniform(size=(nx, ny)) < 0.6),
         "grid_uv_flat": torch.from_numpy(g), "fov_flat": torch.from_numpy(rng.uniform(size=(nx, ny)) < 0.7),
         "pt_feats": torch.from_numpy(rng.normal(size=(n_points, 10)).astype(np.float32)),
         "pt_cell": torch.from_numpy(rng.integers(0, nx * ny, n_points)),
         "road_t": torch.from_numpy(road), "unk_t": torch.from_numpy((rng.uniform(size=(nx, ny)) < 0.5).astype(np.float32)),
         "occ_t": torch.from_numpy(occ), "frame": torch.tensor(seed),
         "Q_geo": torch.from_numpy(rng.uniform(size=(6, nx, ny)).astype(np.float32)),
         "valid_L": torch.from_numpy(rng.uniform(size=(nx, ny)) < 0.8)}
    return s


class Synthetic(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return synthetic_sample(seed=i)


def small_model(mode):
    return Student(StudentConfig(mode=mode, channels=8, backbone="stub", grid_shape=(16, 16)))


@pytest.mark.parametrize("mode", MODES)
def test_forward_shapes_per_mode(mode):
    batch = collate([synthetic_sample(seed=0), synthetic_sample(seed=1)])
    out = small_model(mode).eval()(batch)
    assert set(HEADS) <= set(out)
    assert all(out[h].shape == (2, 16, 16) and torch.isfinite(out[h]).all() for h in HEADS)


def test_the_camera_only_baseline_needs_no_lidar_input():
    batch = collate([synthetic_sample(seed=0)])
    camera_only = {k: batch[k] for k in ("image", "grid_uv_flat", "fov_flat")}         # calibration-only inputs
    out = small_model("camera_ipm").eval()(camera_only)
    assert out["road"].shape == (1, 16, 16)


def test_collate_offsets_each_samples_cells():
    a, b = synthetic_sample(seed=0), synthetic_sample(seed=1)
    batch = collate([a, b])
    assert torch.equal(batch["pt_cell"][len(a["pt_cell"]):], b["pt_cell"] + 16 * 16)
    assert batch["image"].shape == (2, 3, 56, 112)


def test_losses_are_finite_and_zero_with_no_targets():
    logits = torch.randn(2, 16, 16)
    assert masked_bce(logits, torch.full((2, 16, 16), float("nan"))) == 0
    t = torch.rand(2, 16, 16)
    t[0] = float("nan")
    manual = torch.nn.functional.binary_cross_entropy_with_logits(logits[1], t[1])
    assert torch.isclose(masked_bce(logits, t), manual)
    out = {h: torch.randn(1, 16, 16, requires_grad=True) for h in HEADS}
    empty = {"road_t": torch.full((1, 16, 16), float("nan")), "occ_t": torch.full((1, 16, 16), float("nan")),
             "unk_t": torch.zeros(1, 16, 16)}
    total, parts = student_loss(out, empty, LossWeights())
    total.backward()
    assert torch.isfinite(total) and parts["road"] == 0 and parts["occ"] == 0
    assert all(torch.isfinite(v.grad).all() for v in out.values())


def test_training_smoke_checkpoint_and_resume(tmp_path):
    cfg = TrainConfig(epochs=1, batch_size=2, num_workers=0, amp=False, log_every=1, warmup_steps=1)
    res = train(small_model("concat"), Synthetic(4), Synthetic(2), cfg, LossWeights(), tmp_path / "smoke", smoke=2,
                log=lambda m: None)
    assert res["reload_check"] == "ok" and (tmp_path / "smoke" / "ckpt" / "last.pt").exists()

    out = tmp_path / "run"
    first = train(small_model("concat"), Synthetic(4), Synthetic(2), cfg, LossWeights(), out, log=lambda m: None)
    assert first["finished"] and first["epochs_done"] == 1
    lines = []
    more = TrainConfig(**{**cfg.to_dict(), "epochs": 2})
    second = train(small_model("concat"), Synthetic(4), Synthetic(2), more, LossWeights(), out, log=lines.append)
    assert second["epochs_done"] == 2 and any("resumed" in l and "1 of 2" in l for l in lines)
    ckpt = torch.load(out / "ckpt" / "last.pt", weights_only=False)
    assert ckpt["epoch"] == 1 and not any(k.startswith("camera.backbone.") for k in ckpt["model"])
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 2
    import json
    assert json.loads((out / "ckpt" / "progress.json").read_text()) == {"epoch": 1, "epochs": 2}
