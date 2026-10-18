"""Projection QA: draw 3D boxes and LiDAR points into the camera image and onto the BEV grid.

Checks the whole transform chain (world -> LiDAR -> camera, BEV indexing, box conventions) before labels are built on
it, and counts LiDAR points inside each box as a numeric check. Box sources:
  gt      KITTI-360 ground truth (evaluation data: looked at, never trained on)
  (later) mtrans pseudo-label boxes, once MTrans runs on KITTI-360

    python scripts/qa_projection.py --dataset-map data/kitti360_dev/dev_sample/dataset_map.yaml --out data/qa --every 10
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from ocgf.datasets.boxes3d import CLASSES
from ocgf.datasets.kitti360 import Kitti360Sequence
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.geometry.boxes import EDGES, box_corners, points_in_box
from ocgf.geometry.projection import project
from ocgf.geometry.raster import count_points

COLORS = {"car": "#1f6fd1", "large_vehicle": "#7a3fc4", "pedestrian": "#e0352b", "cyclist": "#f08c00", "other": "#9aa0a6"}


def load_boxes(seq: Kitti360Sequence, frame: int, source: str):
    if source == "gt":
        return seq.gt_boxes(frame)
    raise NotImplementedError(f"box source {source!r} is not available yet (mtrans comes with the object teacher)")


def draw(seq, frame, boxes, grid, out_png, max_depth=50.0):
    image, points, cam = seq.image(frame), seq.points(frame), seq.camera
    uv, depth, valid = project(points[:, :3], cam)
    names = boxes.class_names()
    near = grid.inside(boxes.params[:, 0], boxes.params[:, 1]) if len(boxes) else np.zeros(0, bool)

    h, w = image.shape[:2]
    fig = Figure(figsize=(w / 100, h / 100 + 6.5))
    ax_img, ax_bev = fig.subplots(2, 1, gridspec_kw={"height_ratios": [h / 100, 6.5]})
    ax_img.imshow(image)
    ax_img.scatter(uv[valid, 0], uv[valid, 1], c=np.clip(depth[valid], 0, max_depth), s=0.3, cmap="turbo",
                   vmin=0, vmax=max_depth)
    corners = box_corners(boxes.params)
    for k in np.flatnonzero(near):
        cuv, cdepth, _ = project(corners[k], cam)
        for e, (a, b) in enumerate(EDGES):
            if cdepth[a] > 0.1 and cdepth[b] > 0.1:
                ax_img.plot(cuv[[a, b], 0], cuv[[a, b], 1], color=COLORS[names[k]], lw=2.0 if e < 2 else 1.0)
    ax_img.set_xlim(-0.5, w - 0.5)
    ax_img.set_ylim(h - 0.5, -0.5)
    ax_img.axis("off")
    ax_img.set_title(f"{seq.fid(frame)}: LiDAR by depth, boxes by class (thick edge = front)", fontsize=9)

    counts = grid.to_display(np.log1p(count_points(points, grid)))
    ax_bev.imshow(counts, cmap="Greys", extent=[grid.y_max, grid.y_min, grid.x_min, grid.x_max], aspect="equal")
    for k in np.flatnonzero(near):
        c = corners[k, :4]
        ax_bev.fill(c[:, 1], c[:, 0], color=COLORS[names[k]], alpha=0.45, lw=1.2, ec=COLORS[names[k]])
        ax_bev.plot(c[:2, 1], c[:2, 0], color="black", lw=1.2)                # front edge
    ax_bev.set_xlim(grid.y_max, grid.y_min)
    ax_bev.set_ylim(grid.x_min, grid.x_max)
    ax_bev.set_xlabel("y (m), left is +")
    ax_bev.set_ylabel("x (m), forward")
    ax_bev.set_title("BEV grid 50 x 40 m: LiDAR point count (log), box footprints", fontsize=9)
    for name, color in COLORS.items():
        ax_bev.plot([], [], color=color, lw=6, label=name)
    ax_bev.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-map", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--boxes", default="gt", choices=["gt", "mtrans"])
    p.add_argument("--frames", type=int, nargs="*", help="frames to draw (default: every n-th usable frame)")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--near-m", type=float, default=30.0, help="boxes closer than this enter the point-count check")
    args = p.parse_args(argv)

    seq, grid = Kitti360Sequence(args.dataset_map), BEVGrid()
    frames = args.frames or seq.usable_frames()[:: args.every].tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in frames:
        boxes = load_boxes(seq, frame, args.boxes)
        points = seq.points(frame)
        draw(seq, frame, boxes, grid, args.out / f"qa_{args.boxes}_{frame:010d}.png")
        for k in range(len(boxes)):
            x, y = boxes.params[k, :2]
            if grid.inside(x, y) and np.hypot(x, y) < args.near_m:
                rows.append({"frame": frame, "class": CLASSES[boxes.classes[k]], "name": boxes.names[k],
                             "instance": int(boxes.instance_ids[k]), "dynamic": bool(boxes.dynamic[k]),
                             "interpolated": bool(boxes.interpolated[k]), "range_m": round(float(np.hypot(x, y)), 1),
                             "points_inside": int(points_in_box(points, boxes.params[k]).sum()),
                             "points_inside_0.3m": int(points_in_box(points, boxes.params[k], margin=0.3).sum())})
        n_frame = [r for r in rows if r["frame"] == frame]
        hit = sum(r["points_inside"] >= 5 for r in n_frame)
        print(f"{seq.fid(frame)}: {len(boxes)} boxes valid, {len(n_frame)} within {args.near_m:.0f} m, "
              f"{hit} of them with >= 5 LiDAR points inside")
    if rows:
        with open(args.out / f"qa_{args.boxes}_boxes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for cls in CLASSES:
            sel = [r["points_inside"] for r in rows if r["class"] == cls]
            if sel:
                print(f"  {cls:>13}: {len(sel)} boxes, median {np.median(sel):.0f} points inside, "
                      f"{np.mean(np.array(sel) >= 5):.0%} with >= 5")


if __name__ == "__main__":
    main()
