"""The OCGF gate: validity masking, fixed and learned weights, the reliability channels Q, the auxiliary road heads,
and the gate log that makes the use of Q measurable."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ocgf.evaluation.gate_log import Q_CHANNELS as LOG_CHANNELS, GateLog
from ocgf.models.ocgf import Gate
from ocgf.models.student import Q_CHANNELS, Student, StudentConfig
from ocgf.training.data import collate
from ocgf.training.losses import LossWeights, student_loss

from test_student import synthetic_sample


def masks():
    v_c = torch.tensor([[[True, False], [True, False]]])           # [1, 2, 2]
    v_l = torch.tensor([[[True, True], [False, False]]])
    return v_c, v_l


@pytest.mark.parametrize("mode", ["fixed", "learned"])
def test_the_gate_never_uses_an_invalid_sensor_and_gives_nothing_where_both_are_invalid(mode):
    v_c, v_l = masks()
    f_c, f_l = torch.full((1, 4, 2, 2), 2.0), torch.full((1, 4, 2, 2), -1.0)
    fused, w = Gate(4, mode)(f_c, f_l, v_c, v_l)
    assert torch.allclose(w.sum(1)[0], torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
    assert w[0, 0, 0, 1] == 0 and w[0, 1, 0, 1] == 1            # LiDAR only
    assert w[0, 0, 1, 0] == 1 and w[0, 1, 1, 0] == 0            # camera only
    assert (fused[0, :, 1, 1] == 0).all()                       # neither: no evidence
    assert w[0, 0, 0, 0] == 0.5                                  # fixed, and a fresh learned gate: equal trust


def test_a_learned_gate_with_q_changes_its_weights_with_q_and_trains():
    torch.manual_seed(0)
    gate = Gate(4, "learned", extra_channels=8)
    torch.nn.init.normal_(gate.net[2].weight)                   # as after some training (it starts at zero)
    v = torch.ones(1, 6, 6, dtype=torch.bool)
    f = torch.randn(1, 4, 6, 6)
    q0, q1 = torch.zeros(1, 8, 6, 6), torch.ones(1, 8, 6, 6)
    _, w0 = gate(f, f.clone(), v, v, q0)
    fused, w1 = gate(f, f.clone(), v, v, q1)
    assert not torch.allclose(w0, w1)
    fused.sum().backward()
    assert gate.net[0].weight.grad is not None and torch.isfinite(gate.net[0].weight.grad).all()


@pytest.mark.parametrize("gate,gate_q,objects", [("fixed", False, False), ("learned", False, False),
                                                 ("learned", True, False), ("learned", True, True)])
def test_the_ocgf_student_returns_gate_weights_and_q(gate, gate_q, objects):
    s = synthetic_sample()
    if objects:
        s["O_B"] = torch.rand(9, 16, 16)
    batch = collate([s, synthetic_sample(seed=1) | ({"O_B": torch.rand(9, 16, 16)} if objects else {})])
    cfg = StudentConfig(mode="ocgf", channels=8, backbone="stub", grid_shape=(16, 16), gate=gate, gate_q=gate_q,
                        objects=objects)
    out = Student(cfg).eval()(batch)
    assert out["road"].shape == (2, 16, 16) and out["gate_w"].shape == (2, 2, 16, 16)
    assert out["Q"].shape == (2, len(Q_CHANNELS), 16, 16)
    h_cam = out["Q"][:, Q_CHANNELS.index("H_cam")]
    assert (h_cam[~batch["M_C"]] == 0).all() and (h_cam >= 0).all() and (h_cam <= 1).all()
    assert (out["gate_w"][:, 0][~batch["M_C"]] == 0).all() and (out["gate_w"][:, 1][~batch["valid_L"]] == 0).all()
    if not objects:
        assert (out["Q"][:, Q_CHANNELS.index("q_obj")] == 0).all()


def test_auxiliary_road_heads_join_the_loss_where_each_sensor_is_valid():
    batch = collate([synthetic_sample(), synthetic_sample(seed=1)])
    model = Student(StudentConfig(mode="concat", channels=8, backbone="stub", grid_shape=(16, 16)))
    out = model(batch)
    assert {"aux_road_camera", "aux_road_lidar", "valid_camera", "valid_lidar"} <= set(out)
    total, parts = student_loss(out, batch, LossWeights())
    total.backward()
    assert "aux" in parts and torch.isfinite(parts["aux"])
    assert model.aux_camera.weight.grad is not None and model.aux_lidar.weight.grad is not None


def test_the_gate_log_shows_how_w_c_follows_a_q_channel():
    assert LOG_CHANNELS == Q_CHANNELS
    log = GateLog()
    nx, ny = 20, 20
    q = np.zeros((8, nx, ny))
    h = np.linspace(0, 0.999, nx * ny).reshape(nx, ny)
    q[Q_CHANNELS.index("H_cam")] = h
    valid = np.ones((nx, ny), bool)
    log.add(1.0 - h, q, valid, valid)                           # a gate that trusts the camera less as H_cam rises
    by_h = [b["mean_w_C"] for b in log.result()["by_channel"]["H_cam"]]
    assert all(a > b for a, b in zip(by_h, by_h[1:]))
    log.add(np.zeros((nx, ny)), q, valid, ~valid)              # nothing where only the camera is valid
    assert log.result()["cells"] == {"both_valid": nx * ny, "camera_only": nx * ny, "lidar_only": 0}


def test_q_geo_and_lidar_validity_come_from_the_scan_record():
    from ocgf.datasets.student_sample import Q_GEO, reliability
    from ocgf.geometry.bev_grid import BEVGrid
    grid = BEVGrid()
    rec = {"n_pts": np.zeros(grid.shape, np.int32), "r_g": np.full(grid.shape, 0.7, np.float32),
           "M_L": np.zeros(grid.shape, bool), "shadow_cam": np.zeros(grid.shape, bool),
           "vis_lidar": np.zeros(grid.shape, bool)}
    rec["n_pts"][100, 80] = 5                                   # an obstacle cell: points but no line of sight to ground
    q, valid_l = reliability(rec, grid, np.ones(grid.shape, bool))
    assert q.shape == (len(Q_GEO),) + grid.shape and Q_GEO == Q_CHANNELS[:6]
    assert q[0, 100, 80] == pytest.approx(np.log1p(5)) and valid_l[100, 80] and valid_l.sum() == 1
    assert q[1, 0, 80] == pytest.approx(np.hypot(0.125, 0.125) / 50, abs=1e-4)


def test_branch_features_are_normalised_before_gating():
    """The LiDAR branch is ~9x larger than the camera branch on real data: without normalisation the fused sum is
    the LiDAR's whatever the weights. After the gate's normalisation both have unit scale (training mode)."""
    gate = Gate(4, "fixed").train()
    v = torch.ones(2, 6, 6, dtype=torch.bool)
    f_c, f_l = 0.2 * torch.randn(2, 4, 6, 6), 9.0 * torch.randn(2, 4, 6, 6)
    with torch.no_grad():
        fused, _ = gate(f_c, f_l, v, v)
        assert abs(float(gate.norm_c(f_c).std()) - float(gate.norm_l(f_l).std())) < 0.05
    assert float(fused.std()) < 2.0

