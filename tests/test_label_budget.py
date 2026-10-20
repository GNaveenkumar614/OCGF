import numpy as np

from ocgf.teachers.detector2d.mapping import Detections
from ocgf.teachers.label_budget import select_labelled_frames


def fake_detections(n_frames=2000, seed=0):
    """Mostly cars; a few frames with cyclists, large vehicles, pedestrians; some low-score detections."""
    rng = np.random.default_rng(seed)
    frames = [f"kitti/training/{i:010d}" for i in range(n_frames)]
    out = {}
    for f in frames:
        labels, scores = ["car"], [0.9]
        for label, p in (("cyclist", 0.03), ("large_vehicle", 0.05), ("pedestrian", 0.1)):
            if rng.random() < p:
                labels.append(label)
                scores.append(rng.choice([0.9, 0.2]))           # some below the score threshold
        out[f] = Detections(np.zeros((len(labels), 4)), np.array(scores), labels)
    return frames, out


def test_sizes_nesting_and_no_duplicates():
    frames, dets = fake_detections()
    big, small, report = select_labelled_frames(dets, frames, 500, 125)
    assert len(big) == len(set(big)) == 500 and small == big[:125]
    assert set(big) <= set(frames)


def test_rare_classes_are_covered_far_above_chance():
    frames, dets = fake_detections()
    big, small, report = select_labelled_frames(dets, frames, 500, 125)
    for c in ("cyclist", "large_vehicle", "pedestrian"):
        available = report["frames_with"][c]
        assert report["selected_with"][c] >= min(available, 0.25 * 500) * 0.9       # quota met or pool exhausted
        assert report["first_small_with"][c] >= min(available, 25)                   # the 125 set has them too


def test_low_scores_do_not_count():
    frames = ["a", "b", "c"]
    dets = {"a": Detections(np.zeros((1, 4)), np.array([0.2]), ["cyclist"]),
            "b": Detections(np.zeros((1, 4)), np.array([0.8]), ["cyclist"])}
    _, _, report = select_labelled_frames(dets, frames, 2, 1, min_score=0.5)
    assert report["frames_with"]["cyclist"] == 1


def test_deterministic_for_a_seed():
    frames, dets = fake_detections()
    assert select_labelled_frames(dets, frames, seed=3)[0] == select_labelled_frames(dets, frames, seed=3)[0]
    assert select_labelled_frames(dets, frames, seed=3)[0] != select_labelled_frames(dets, frames, seed=4)[0]


def test_frozen_mtrans_label_lists():
    # Frozen 2026-10-20 from notebook 02 version 4 (detections sha256 91c6f621..., see mtrans_labelled.json).
    # The label budget is a reported number in the paper: changing these lists means a new version, not an edit.
    import hashlib
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / "configs" / "splits" / "kitti_object"
    big, small = (d / "mtrans_labelled_500.txt").read_text().split(), (d / "mtrans_labelled_125.txt").read_text().split()
    train, val = set((d / "train.txt").read_text().split()), set((d / "val.txt").read_text().split())
    assert len(set(big)) == len(big) == 500 and small == big[:125]
    assert set(big) <= train and not set(big) & val
    sha = lambda name: hashlib.sha256((d / name).read_bytes()).hexdigest()
    assert sha("mtrans_labelled_500.txt") == "01f286b25a324633c025234c1a6e68d3a0554f2412f1d99f8dd2d0da252e9e6e"
    assert sha("mtrans_labelled_125.txt") == "6901a14ad326a30c7b99cd186bcf1c5accab94f12d5d7afb6b45d9b50c7a6e76"
