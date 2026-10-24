"""Build the evaluation ground truth (BEV layers, see ocgf/evaluation/gt_builder.py) for KITTI-360 frames.

    python scripts/build_eval_gt.py --dataset-map data/kitti360_dev/dev_sample/dataset_map.yaml --out data/eval_gt \
        --all-frames --qa 4
    python scripts/build_eval_gt.py --input-root /kaggle/input --sequence 2013_05_28_drive_0009_sync \
        --out /kaggle/working/eval_gt --qa 6

Frames: usable frames inside a semantic window, sampled every 5 m (--all-frames: every stored usable frame); with
--split and --parts, only the frames in those parts of a frame-level split (starter phase: drive 0000's val and test).
Writes <out>/<sequence>/<frame>.npz, summary.jsonl (per-frame shares), manifest.json and qa/<frame>.png.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from ocgf.datasets.kitti360 import Kitti360Sequence
from ocgf.datasets.kitti360_labels import GROUND_SURFACES, LABEL_NAMES, ROAD
from ocgf.datasets.locate import describe_prepared, prepared_drive
from ocgf.evaluation.gt_builder import (OCCUPIED_CLASSES, GTConfig, build_frame_gt, free_space, load_window,
                                        save_layers, settings)
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.geometry.projection import project
from ocgf.utils.provenance import write_manifest
from ocgf.utils.staging import parallel_copy

GROUND_COLORS = {0: "#ffffff", 7: "#6b6f78", 8: "#e3a6d8", 9: "#c9b28a", 6: "#9c8a6d", 10: "#b0a090", 22: "#9fcf86"}
OCC_COLORS = ["#ffffff", "#1f6fd1", "#7a3fc4", "#e0352b", "#f08c00", "#9aa0a6"]


def frame_summary(layers: dict) -> dict:
    n = layers["labelled"].size
    share = lambda m: round(float(np.count_nonzero(m)) / n, 4)
    observed = layers["observed_lidar"] | layers["observed_cam"]
    return {"labelled": share(layers["labelled"]), "road": share(layers["ground_class"] == ROAD),
            "occupied": share(layers["occupied"] > 0), "obstacle": share(layers["obstacle"]),
            "observed": share(observed), "observed_lidar": share(layers["observed_lidar"]),
            "observed_cam": share(layers["observed_cam"]), "free": share(free_space(layers))}


def draw_qa(path, seq, frame, layers, grid):
    image, cam = seq.image(frame), seq.camera
    xs, ys = grid.center_grid()
    z = layers["ground_z"].astype(np.float64)
    free, occ = free_space(layers), layers["occupied"] > 0
    h, w = image.shape[:2]
    fig = Figure(figsize=(14, 9.2))
    ax = fig.add_axes([0.01, 0.56, 0.98, 0.42])
    ax.imshow(image)
    for mask, color in ((free, "#39ff5a"), (occ, "#ff3b30"), (layers["obstacle"], "#ffa500")):
        m = mask & np.isfinite(z)
        uv, _, valid = project(np.column_stack([xs[m], ys[m], z[m]]), cam)
        ax.scatter(uv[valid, 0], uv[valid, 1], s=3, c=color, alpha=0.8, linewidths=0)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis("off")
    ax.set_title(f"{seq.fid(frame)}: ground truth at ground level: free road (green), occupied (red), obstacle (orange)",
                 fontsize=10)

    ext = [grid.y_max, grid.y_min, grid.x_min, grid.x_max]
    observed = (layers["observed_lidar"].astype(int) + 2 * layers["observed_cam"].astype(int))
    ground = np.zeros(grid.shape, dtype=int)
    gids = [0] + list(GROUND_SURFACES)
    for k, gid in enumerate(gids):
        ground[layers["ground_class"] == gid] = k
    panels = [
        ("ground class", ground, ListedColormap([GROUND_COLORS[g] for g in gids]), len(gids) - 1),
        ("occupied by class, obstacle = black", np.where(layers["obstacle"] & ~occ, -1, layers["occupied"].astype(int)),
         ListedColormap(["#000000"] + OCC_COLORS), 5),
        ("observed: none / LiDAR / camera / both", observed,
         ListedColormap(["#ffffff", "#9cc3f0", "#f0c39c", "#5a9e6f"]), 3),
        ("free road (ground truth)", free.astype(int), ListedColormap(["#ffffff", "#2fbf4a"]), 1),
    ]
    for k, (title, data, cmap, vmax) in enumerate(panels):
        a = fig.add_axes([0.02 + k * 0.245, 0.03, 0.22, 0.49])
        vmin = -1 if title.startswith("occupied") else 0
        a.imshow(grid.to_display(data), extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                 aspect="equal")
        a.set_title(title, fontsize=9)
        a.set_xlabel("y (m), left is +", fontsize=8)
        a.tick_params(labelsize=7)
    fig.text(0.02, 0.535, "ground: " + ", ".join(f"{LABEL_NAMES[g]}={GROUND_COLORS[g]}" for g in GROUND_SURFACES)
             + " | occupied: " + ", ".join(f"{c}={OCC_COLORS[i]}" for i, c in enumerate(OCCUPIED_CLASSES) if i),
             fontsize=7)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-map", type=Path)
    src.add_argument("--input-root", type=Path)
    p.add_argument("--sequence", help="with --input-root: the prepared drive")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--all-frames", action="store_true", help="every stored usable frame, not one every 5 m")
    p.add_argument("--qa", type=int, default=4, help="QA figures for this many frames")
    p.add_argument("--stage-dir", type=Path, help="copy the semantic windows here first (slow input mounts)")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--split", type=Path, help="frame-level split JSON (e.g. configs/splits/kitti360_d0000_dev_v1.json)")
    p.add_argument("--parts", nargs="+", default=["val", "test"], help="with --split: the parts to build")
    args = p.parse_args(argv)

    if args.dataset_map:
        seq = Kitti360Sequence(args.dataset_map)
    else:
        seq = Kitti360Sequence(sequence=args.sequence, components=prepared_drive(args.input_root, args.sequence))
        print(describe_prepared(args.input_root, args.sequence), flush=True)
    if "semantics3d" not in seq.paths:
        raise SystemExit("no accumulated 3D semantics for this drive: attach the KITTI-360 3d semantics dataset")
    idx, grid, cfg = seq.index, BEVGrid(), GTConfig()
    sel = idx["usable"] & idx.get("kept", True) & (idx["window_start"] >= 0)
    if not args.all_frames:
        sel &= idx["sample_5m"]
    if args.split:
        from ocgf.datasets.splits import from_ranges
        split = json.loads(args.split.read_text())
        if split.get("sequence") != seq.sequence or "splits" not in split:
            raise SystemExit(f"{args.split} is not a frame-level split of {seq.sequence}")
        sel &= np.isin(idx["frame"], np.concatenate([from_ranges(split["splits"][p]) for p in args.parts]))
    frames = idx["frame"][sel]
    windows = sorted({(int(idx["window_start"][i]), int(idx["window_end"][i])) for i in np.flatnonzero(sel)})
    out = args.out / seq.sequence
    out.mkdir(parents=True, exist_ok=True)
    semantics3d = Path(seq.paths["semantics3d"])
    if args.stage_dir:
        local = args.stage_dir / seq.sequence
        names = [f"{ws:010d}_{we:010d}.ply" for ws, we in windows]
        parallel_copy([(semantics3d / part / n, local / part / n) for part in ("static", "dynamic") for n in names
                       if (semantics3d / part / n).exists()], args.threads, print, every=10)
        semantics3d = local
    qa = set(frames[np.linspace(0, len(frames) - 1, min(args.qa, len(frames))).astype(int)].tolist()) if len(frames) else set()
    print(f"{seq.sequence}: {len(frames)} frames in {len(windows)} semantic windows -> {out}", flush=True)

    t0, summaries = time.time(), []
    for ws, we in windows:
        static, dynamic = load_window(semantics3d, ws, we)
        in_window = frames[(idx["window_start"][sel] == ws) & (idx["window_end"][sel] == we)]
        for frame in in_window:
            frame = int(frame)
            layers = build_frame_gt(grid, seq.T_world_lidar(frame), static, dynamic, frame, seq.gt_boxes(frame),
                                    seq.camera, cfg)
            save_layers(out / f"{frame:010d}.npz", layers)
            summaries.append({"fid": seq.fid(frame), **frame_summary(layers)})
            if frame in qa:
                draw_qa(out / "qa" / f"{frame:010d}.png", seq, frame, layers, grid)
        print(f"  window {ws}-{we}: {len(in_window)} frames, {len(summaries)}/{len(frames)} done, "
              f"{(time.time() - t0) / max(len(summaries), 1):.2f} s per frame", flush=True)

    (out / "summary.jsonl").write_text("".join(json.dumps(s) + "\n" for s in summaries))
    mean = {k: round(float(np.mean([s[k] for s in summaries])), 4) for k in summaries[0] if k != "fid"} if summaries else {}
    write_manifest(out / "manifest.json", kind="eval_gt", sequence=seq.sequence, settings=settings(cfg),
                   grid={"x": [grid.x_min, grid.x_max], "y": [grid.y_min, grid.y_max], "res": grid.res},
                   frames=len(summaries), windows=windows, mean_shares=mean)
    print(f"done: {len(summaries)} frames; mean shares of the grid: {mean}")


if __name__ == "__main__":
    main()
