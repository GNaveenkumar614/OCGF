"""Student-side geometry and training targets on a synthetic street: ground surface, visibility, road/unknown/occupancy
targets. The street (road_fixtures) is flat road at z = -1.73 for |y| < 4 m, sidewalks 0.15 m higher, walls at |y| = 8 m."""
import numpy as np
import pytest

from ocgf.datasets.bev_inputs import LAYERS, BEVInputConfig, bev_inputs
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.geometry.ground import GroundConfig, fit_plane, ground_surface, seed_points
from ocgf.training.targets import build_targets

from geometry_fixtures import camera
from road_fixtures import ROAD_Z, WALK_Z, street_scan

GRID = BEVGrid()
CAM = camera()
BOX = ((20.0, -1.0, ROAD_Z), (23.0, 1.0, -0.2))         # taller than the camera, on the street ahead


def cell(x, y):
    ix, iy = GRID.index(x, y)
    return int(ix), int(iy)


@pytest.fixture(scope="module")
def scene():
    points = street_scan(obstacle=BOX)
    inputs = bev_inputs(points, CAM, GRID)
    return points, inputs


def test_ground_follows_road_and_sidewalk_and_is_reliable_where_seen():
    z, r = ground_surface(street_scan(), GRID)
    assert z.shape == r.shape == GRID.shape and z.dtype == np.float32
    for x in (6, 15, 30):
        assert abs(z[cell(x, 0)] - ROAD_Z) < 0.03 and abs(z[cell(x, 6)] - WALK_Z) < 0.05
    assert r[cell(15, 0)] > 0.7
    assert r[cell(30, 15)] < 0.2 and abs(z[cell(30, 15)] - ROAD_Z) < 0.2      # behind the wall: plane, unreliable


def test_ground_follows_a_slope_that_a_flat_plane_would_miss():
    pts = street_scan()
    pts[:, 2] += 0.03 * pts[:, 0]                                            # 3 % uphill
    z, _ = ground_surface(pts, GRID)
    assert abs(z[cell(30, 0)] - (ROAD_Z + 0.9)) < 0.1


def test_ground_under_an_obstacle_and_next_to_the_ego_is_not_lifted():
    beside = ((-2.0, -3.2, ROAD_Z), (3.0, -1.6, -0.3))                       # a parked car beside the ego
    car = street_scan(obstacle=beside)
    pts = np.vstack([street_scan(obstacle=BOX), car[(car[:, 0] < 3.5) & (car[:, 2] > ROAD_Z + 0.2)]])
    z, _ = ground_surface(pts, GRID)
    assert abs(z[cell(21.5, 0)] - ROAD_Z) < 0.15                             # under the box
    assert abs(z[cell(1.0, -1.0)] - ROAD_Z) < 0.1                            # the ego footprint stays on the plane


def test_a_plane_fitted_to_seeds_recovers_the_ground():
    pts = street_scan()
    pts = pts[np.abs(pts[:, 1]) < 4]                                         # road only
    a, b, c = fit_plane(seed_points(pts, GRID))
    assert abs(a) < 0.005 and abs(b) < 0.005 and abs(c - ROAD_Z) < 0.03
    cfg = GroundConfig.from_dict(GroundConfig(plane=(a, b, c)).to_dict())
    assert cfg.plane == (a, b, c)


def test_visibility_shadows_behind_the_box_and_nothing_below_the_beams(scene):
    _, inp = scene
    assert set(inp) == set(LAYERS)
    near_road, behind = cell(10, 0), cell(30, 0)
    assert inp["M_C"][near_road] and inp["vis_lidar"][near_road]
    ix, iy = near_road
    assert inp["M_L"][ix - 2:ix + 3, iy - 2:iy + 3].any()                   # rings are ~0.4 m apart at 10 m
    assert inp["shadow_cam"][behind] and not inp["M_C"][behind]
    assert inp["shadow_lidar"][behind] and not inp["vis_lidar"][behind]
    assert not inp["vis_lidar"][cell(2, 0)] and not inp["fov_cam"][cell(2, 0)]        # below both sensors' view
    side = cell(6, 7)                                                        # sidewalk outside the camera's view
    assert not inp["fov_cam"][side] and inp["vis_lidar"][side]
    assert inp["M_L"][side[0] - 2:side[0] + 3, side[1] - 2:side[1] + 3].any()
    assert inp["n_above"][cell(20.1, 0)] >= 2 and inp["n_above"][near_road] == 0
    assert inp["n_above"][cell(30, 8.1)] >= 2                                # the wall (at y = 8.0)


def test_targets_read_the_camera_label_where_seen_and_lidar_scores_elsewhere(scene):
    points, inp = scene
    label = np.full((CAM.height, CAM.width), 0.9, np.float32)
    scores = np.full(len(points), 0.3, np.float32)
    t = build_targets(inp, label, scores, points, CAM, GRID)
    assert set(t) == {"road_t", "unk_t", "occ_t"} and all(v.shape == GRID.shape for v in t.values())
    assert t["road_t"][cell(10, 0)] == pytest.approx(0.9)                    # camera label
    assert t["road_t"][cell(6, 7)] == pytest.approx(0.3)                     # LiDAR score outside the camera's view
    behind = cell(30, 0)
    assert t["unk_t"][behind] == 1 and np.isnan(t["road_t"][behind])         # in both shadows: unknown, no road target
    assert t["unk_t"][cell(2, 0)] == 1 and np.isnan(t["road_t"][cell(2, 0)])  # next to the ego
    assert t["unk_t"][cell(10, 0)] == 0
    assert t["occ_t"][cell(20.1, 0)] == 1 and t["occ_t"][cell(10, 0)] == 0 and np.isnan(t["occ_t"][behind])


def test_targets_without_teacher_evidence_are_all_ignored(scene):
    points, inp = scene
    t = build_targets(inp, np.full((CAM.height, CAM.width), np.nan, np.float32), np.full(len(points), np.nan),
                      points, CAM, GRID)
    assert np.isnan(t["road_t"]).all() and np.isfinite(t["unk_t"]).all()
    assert np.nanmax(t["occ_t"]) == 1 and np.nanmin(t["occ_t"]) == 1         # obstacles still count, no negatives


def test_configs_round_trip():
    cfg = BEVInputConfig(occluder_max=2.5, ground=GroundConfig(sigma=2.0))
    assert BEVInputConfig.from_dict(cfg.to_dict()) == cfg


def test_records_round_trip_with_compact_dtypes(scene, tmp_path):
    from ocgf.training.targets import load_record, save_record
    points, inp = scene
    tgt = build_targets(inp, np.full((CAM.height, CAM.width), 0.9, np.float32), np.full(len(points), 0.3),
                        points, CAM, GRID)
    save_record(tmp_path / "r.npz", inp, tgt)
    back = load_record(tmp_path / "r.npz")
    assert set(back) == set(inp) | set(tgt)
    assert back["M_C"].dtype == bool and back["n_above"].dtype == np.int32 and back["road_t"].dtype == np.float32
    assert np.allclose(back["z_g"], inp["z_g"], atol=2e-3)
    assert np.array_equal(np.isnan(back["road_t"]), np.isnan(tgt["road_t"])) and np.array_equal(back["unk_t"], tgt["unk_t"])


def test_build_bev_targets_script_on_a_prepared_drive(tmp_path):
    import json
    import sys
    from pathlib import Path
    from ocgf.datasets.kitti360 import Kitti360Sequence
    from ocgf.datasets.locate import prepared_drive
    from ocgf.datasets.prepare_kitti360 import PrepConfig, run_all
    from ocgf.teachers.road.label import save_result
    from kitti360_fixtures import SEQ, make_fake_kitti360
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_bev_targets

    paths = make_fake_kitti360(tmp_path, n_frames=40, starter_images=False)
    run_all(PrepConfig(sequence=SEQ, input_root=paths["input_root"], out_dir=paths["input_root"] / "supp" / "out",
                       scratch_dir=tmp_path / "scratch", sources=paths["sources"], role="train", keep_every_m=2.5,
                       min_future_m=20.0, n_overlays=0, dev_frames=0), log=lambda m: None)
    seq = Kitti360Sequence(sequence=SEQ, components=prepared_drive(paths["input_root"], SEQ))
    frames = seq.usable_frames()[:3]
    for f in frames:                                         # stand-in road labels: everything road
        n = len(seq.points(int(f)))
        save_result(tmp_path / "road" / SEQ / f"{int(f):010d}.npz",
                    {"label": np.ones((seq.camera.height, seq.camera.width)), "point_scores": np.ones(n),
                     "path_mask": np.zeros((2, 2), bool), "wheels": np.zeros((0, 2, 3)), "status": "ok"})
    build_bev_targets.main(["--input-root", str(paths["input_root"]), "--sequence", SEQ, "--road-labels",
                            str(tmp_path / "road"), "--out", str(tmp_path / "out"), "--qa", "1",
                            "--stage-dir", str(tmp_path / "stage"), "--threads", "2"])
    out = tmp_path / "out" / SEQ
    assert sorted(p.stem for p in out.glob("*.npz")) == [f"{int(f):010d}" for f in frames]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["frames"] == 3 and len(list((out / "qa").glob("*.png"))) == 1
