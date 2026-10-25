"""Evaluation metrics on synthetic ground truth with known counts: false-free rates, free-road IoU, regions, classes,
calibration and determinism."""
import numpy as np
import pytest

from ocgf.evaluation.metrics import Evaluator, MetricConfig, gt_masks
from ocgf.geometry.bev_grid import BEVGrid

GRID = BEVGrid()
XS, YS = GRID.center_grid()
CAR = (XS >= 20) & (XS < 24) & (np.abs(YS) < 1)                     # 16 x 8 = 128 cells


def street_layers():
    road = np.abs(YS) < 4
    occupied = np.where(CAR, 1, 0).astype(np.uint8)
    occupied[(XS >= 10) & (XS < 10.5) & (YS >= 3) & (YS < 3.5)] = 3   # a pedestrian on the road edge
    shadow = (XS >= 24) & (np.abs(YS) < 1.5)
    return {"ground_z": np.full(GRID.shape, -1.73, np.float16),
            "ground_class": np.where(road, 7, 8).astype(np.uint8),
            "labelled": XS < 45, "occupied": occupied, "obstacle": np.abs(np.abs(YS) - 8.1) < 0.13,
            "observed_lidar": ~shadow & (XS >= 4), "observed_cam": ~shadow & (XS >= 5.5)}


def perfect(layers):
    g = gt_masks(layers)
    return {"road": g["free"].astype(float), "occ": g["occupied"].astype(float), "unk": g["unknown"].astype(float)}


def test_a_perfect_prediction_has_no_false_free_and_full_iou():
    ev = Evaluator()
    ev.add(perfect(street_layers()), street_layers())
    r = ev.result()["regions"]["all"]
    assert r["ffr_pred"] == 0 and r["ffr_hazard"] == 0
    assert r["free_road"]["iou"] == 1 and r["occupancy"]["iou"] == 1 and r["unknown"]["iou"] == 1
    assert ev.result()["calibration"]["ece"] == 0


def test_calling_the_car_free_counts_exactly_its_cells_in_both_rates():
    layers = street_layers()
    probs = perfect(layers)
    probs["road"][CAR], probs["occ"][CAR] = 1.0, 0.0                   # the model misses the car
    ev = Evaluator()
    ev.add(probs, layers)
    res = ev.result()
    r = res["regions"]["all"]
    g = gt_masks(layers)
    n_free, n_hazard = int(g["free"].sum()), int(g["hazard"].sum())
    assert r["ffr_pred"] == pytest.approx(128 / (n_free + 128), abs=1e-5)
    assert r["ffr_hazard"] == pytest.approx(128 / n_hazard, abs=1e-5)
    assert res["classes"]["car"]["called_free"] == 1.0 and res["classes"]["pedestrian"]["called_free"] == 0.0
    near = res["regions"]["near_object_2m"]
    assert near["ffr_pred"] > r["ffr_pred"]                            # the error sits in the near-object band
    assert res["regions"]["d20-30m"]["ffr_pred"] > 0 and res["regions"]["d0-10m"]["ffr_pred"] == 0


def test_the_near_object_band_surrounds_object_boxes_not_walls():
    from ocgf.evaluation.metrics import regions
    g = gt_masks(street_layers())
    near = regions(GRID, g["objects"], MetricConfig())["near_object_2m"]
    assert near[CAR].all() and near[GRID.index(25.5, 0.0)] and not near[GRID.index(27.0, 0.0)]
    assert not near[GRID.index(40.0, 8.1)]                              # beside the wall, far from any object


def test_unobserved_road_behind_the_car_is_a_hazard_not_free():
    layers = street_layers()
    g = gt_masks(layers)
    behind = (XS >= 25) & (np.abs(YS) < 1)
    assert g["hazard"][behind].all() and not g["free"][behind].any()
    probs = perfect(layers)
    probs["road"][behind], probs["unk"][behind] = 1.0, 0.0             # claims the hidden road is free
    ev = Evaluator()
    ev.add(probs, layers)
    assert ev.result()["regions"]["all"]["ffr_pred"] > 0


def test_results_do_not_depend_on_frame_order():
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(3):
        layers = street_layers()
        frames.append(({k: rng.uniform(size=GRID.shape) for k in ("road", "occ", "unk")}, layers))
    a, b = Evaluator(), Evaluator()
    for p, l in frames:
        a.add(p, l)
    for p, l in reversed(frames):
        b.add(p, l)
    assert a.result() == b.result()


def test_risk_coverage_ends_at_full_coverage_and_config_is_reported():
    ev = Evaluator(cfg=MetricConfig(near_m=1.0))
    rng = np.random.default_rng(1)
    ev.add({k: rng.uniform(size=GRID.shape) for k in ("road", "occ", "unk")}, street_layers())
    res = ev.result()
    rc = res["risk_coverage"]
    assert rc[-1]["coverage"] == 1.0 and all(a["coverage"] <= b["coverage"] for a, b in zip(rc, rc[1:]))
    assert "near_object_1m" in res["regions"] and res["config"]["near_m"] == 1.0
    assert sum(b["count"] for b in res["calibration"]["bins"]) == GRID.shape[0] * GRID.shape[1]


def test_bounded_aurc_and_matched_recall_reward_ranking_free_road_first():
    layers = street_layers()
    good, rand = Evaluator(), Evaluator()
    good.add(perfect(layers), layers)
    rng = np.random.default_rng(2)
    rand.add({k: rng.uniform(size=GRID.shape) for k in ("road", "occ", "unk")}, layers)
    g, r = good.result(), rand.result()
    assert g["aurc_free"] == 0 < r["aurc_free"]                   # a perfect ranking claims only free road first
    assert g["ffr_at_recall"]["0.6"] == 0 and r["ffr_at_recall"]["0.6"] > 0.1   # random: about the hazard share
    rc = g["risk_coverage"]
    assert all(a["free_recall"] <= b["free_recall"] for a, b in zip(rc, rc[1:])) and rc[-1]["free_recall"] == 1.0


def test_matched_recall_compares_a_cautious_and_a_bold_model_fairly():
    """Same claims at the 0.5 threshold, so the same plain ffr_pred; at equal free recall the confidence ranking
    decides: wrong claims ranked below the true free road cost nothing until that recall is passed."""
    layers = street_layers()
    g = gt_masks(layers)
    base = {k: v.copy() for k, v in perfect(layers).items()}
    noisy = {k: v.copy() for k, v in base.items()}
    wrong = g["hazard"] & (np.random.default_rng(3).uniform(size=GRID.shape) < 0.02)
    for k, v in (("road", 1.0), ("occ", 0.0), ("unk", 0.0)):           # confidently wrong: as sure as the free road
        noisy[k] = np.where(wrong, v, noisy[k])
    shy = {k: v.copy() for k, v in noisy.items()}
    for k, v in (("road", 0.97), ("occ", 0.0), ("unk", 0.0)):           # the same mistakes, less sure
        shy[k] = np.where(wrong, v, shy[k])
    res = {}
    for name, probs in (("noisy", noisy), ("shy", shy)):
        e = Evaluator()
        e.add(probs, layers)
        res[name] = e.result()
    assert res["noisy"]["regions"]["all"]["ffr_pred"] == res["shy"]["regions"]["all"]["ffr_pred"] > 0
    assert res["noisy"]["ffr_at_recall"]["0.7"] > 0 == res["shy"]["ffr_at_recall"]["0.7"]   # ranking decides
