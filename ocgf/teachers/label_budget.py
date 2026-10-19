"""Which KITTI train frames MTrans may take 3D labels from: 500 (and a nested 125 for the S1 ablation).

Chosen from zero-shot 2D detections only, never from label_2, so the choice adds no human knowledge beyond the frames'
own labels. Frames are drawn in rounds: one frame with a detected cyclist, one with a large vehicle, one with a
pedestrian, then one at random; a class drops out once its frames run out or its quota is met. After the quotas, the
rest is random. The 125 set is the first 125 of the same order, so it covers the rare classes too.
"""
from __future__ import annotations

import numpy as np

RARE = ("cyclist", "large_vehicle", "pedestrian")


def select_labelled_frames(detections: dict, frames: list[str], n: int = 500, n_small: int = 125,
                           min_score: float = 0.5, quota_share: float = 0.25, seed: int = 0) -> tuple[list, list, dict]:
    """detections: {frame id: Detections in our classes}; frames: the eligible (train split) frame ids, ordered.
    Returns (n frames, its first n_small frames, report)."""
    rng = np.random.default_rng(seed)
    has = {c: set() for c in RARE}
    for f in frames:
        d = detections.get(f)
        if d is not None:
            for label, score in zip(d.labels, d.scores):
                if label in has and score >= min_score:
                    has[label].add(f)
    pools = {c: [frames[i] for i in rng.permutation(len(frames)) if frames[i] in has[c]] for c in RARE}
    pools["random"] = [frames[i] for i in rng.permutation(len(frames))]
    quota = {c: int(quota_share * n) for c in RARE}
    taken, order = {c: 0 for c in RARE}, []
    chosen = set()
    while len(order) < min(n, len(frames)):
        progressed = False
        for c in (*RARE, "random"):
            if len(order) >= n:
                break
            if c != "random" and taken[c] >= quota[c]:
                continue
            pool = pools[c]
            while pool and pool[0] in chosen:
                pool.pop(0)
            if pool:
                f = pool.pop(0)
                chosen.add(f)
                order.append(f)
                if c != "random":
                    taken[c] += 1
                progressed = True
        if not progressed:
            break
    report = {"eligible_frames": len(frames), "frames_with": {c: len(has[c]) for c in RARE},
              "selected": len(order), "min_score": min_score, "seed": seed,
              "selected_with": {c: len(set(order) & has[c]) for c in RARE},
              "first_small_with": {c: len(set(order[:n_small]) & has[c]) for c in RARE}}
    return order, order[:n_small], report
