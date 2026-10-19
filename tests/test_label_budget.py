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
