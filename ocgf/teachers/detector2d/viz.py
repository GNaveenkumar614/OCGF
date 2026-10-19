"""Example images with their mapped detections, for a quick visual check of a run."""
from __future__ import annotations

from pathlib import Path

COLORS = {"car": "#1f6fd1", "large_vehicle": "#7a3fc4", "pedestrian": "#e0352b", "cyclist": "#f08c00"}


def draw_detections(image, dets, path, title: str = "", min_score: float = 0.3) -> Path:
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    h, w = image.shape[:2]
    fig = Figure(figsize=(w / 100, h / 100 + 0.3))
    ax = fig.add_axes([0, 0, 1, h / (h + 30)])
    ax.imshow(image)
    for (l, t, r, b), score, label in zip(dets.boxes, dets.scores, dets.labels):
        if score < min_score:
            continue
        color = COLORS.get(label, "#9aa0a6")
        ax.add_patch(Rectangle((l, t), r - l, b - t, fill=False, ec=color, lw=1.5))
        ax.text(l, t - 2, f"{label} {score:.2f}", color="white", fontsize=7, va="bottom",
                bbox={"facecolor": color, "alpha": 0.8, "pad": 1, "lw": 0})
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis("off")
    fig.suptitle(f"{title} (score >= {min_score})", fontsize=8, y=0.995)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    return path
