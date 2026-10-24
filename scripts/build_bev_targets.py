"""Student BEV inputs and training targets for the road-labelled frames of a prepared KITTI-360 training drive.

    python scripts/build_bev_targets.py --dataset-map data/kitti360_dev/dev_sample/dataset_map.yaml \
        --road-labels data/road_labels --out data/bev_targets --qa 4
    python scripts/build_bev_targets.py --input-root /kaggle/input --sequence 2013_05_28_drive_0000_sync \
        --road-labels /kaggle/input/ocgf-road-labels-v1 --out /kaggle/working/bev_targets --stage-dir /kaggle/temp/scans

Frames: every frame the road teacher labelled (<road-labels>/<sequence>/<frame>.npz). Per frame <out>/<sequence>/
<frame>.npz holds the inputs (ground surface, visibility, point statistics; ocgf.datasets.bev_inputs) and the targets
(road_t, unk_t, occ_t; ocgf.training.targets). Also frames.jsonl (mean share of each layer), manifest.json and
qa/<frame>.png. Uses no ground truth: the scan, the calibration and the teacher's labels only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from ocgf.datasets.bev_inputs import BEVInputConfig, bev_inputs
from ocgf.datasets.kitti360 import Kitti360Sequence, stage_drive
from ocgf.datasets.locate import describe_prepared, find_outputs, prepared_drive
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.geometry.ground import plane_z
from ocgf.geometry.projection import project
from ocgf.training.targets import TargetConfig, build_targets, read_road_label, save_record
from ocgf.utils.provenance import write_manifest

REPO = Path(__file__).resolve().parents[1]
SHARES = ("M_C", "shadow_cam", "M_L", "vis_lidar", "shadow_lidar")


def draw_qa(path, fid, image, camera, grid, inp, tgt, plane):
    fig = Figure(figsize=(15, 7.4))
    xs, ys = grid.center_grid()
    disp = BEVGrid.to_display
    ext = (grid.y_max, grid.y_min, grid.x_min, grid.x_max)               # forward up, left on the left
    panels = [
        ("ground z_g - plane (m)", disp(inp["z_g"] - plane_z(plane, xs, ys)), "coolwarm", -0.5, 0.5),
        ("ground reliability r_g", disp(inp["r_g"]), "viridis", 0, 1),
        ("camera: seen 2 / shadow 1 / out 0", disp(inp["M_C"] * 2.0 + inp["shadow_cam"]), "magma", 0, 2),
        ("LiDAR: seen 2 / shadow 1 / out 0", disp(inp["vis_lidar"] * 2.0 + inp["shadow_lidar"]),
         "magma", 0, 2),
        ("road_t (white: no target)", disp(np.ma.masked_invalid(tgt["road_t"])), "RdYlGn", 0, 1),
        ("unk_t (1 = unknown)", disp(tgt["unk_t"]), "gray_r", 0, 1),
        ("occ_t (1 occupied, 0 free road, white ignored)", disp(np.ma.masked_invalid(tgt["occ_t"])), "RdYlGn_r", 0, 1),
    ]
    for k, (title, img, cmap, lo, hi) in enumerate(panels):
        ax = fig.add_axes([0.005 + k * 0.142, 0.56, 0.135, 0.40])
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02).ax.tick_params(labelsize=6)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.50])
    ax.imshow(image)
    sel = np.isfinite(tgt["road_t"]) & inp["M_C"]
    sel &= (np.arange(sel.shape[0])[:, None] % 2 == 0) & (np.arange(sel.shape[1])[None, :] % 2 == 0)
    uv, _, ok = project(np.column_stack([xs[sel], ys[sel], inp["z_g"][sel].astype(np.float64)]), camera)
    ax.scatter(uv[ok, 0], uv[ok, 1], c=tgt["road_t"][sel][ok], cmap="RdYlGn", vmin=0, vmax=1, s=1.5, linewidths=0)
    ax.set_xlim(-0.5, camera.width - 0.5)
    ax.set_ylim(camera.height - 0.5, -0.5)
    ax.axis("off")
    ax.set_title(f"{fid}: road_t at each camera-seen cell's ground point (every 2nd cell)", fontsize=9)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-map", type=Path)
    src.add_argument("--input-root", type=Path)
    p.add_argument("--sequence")
    p.add_argument("--road-labels", type=Path, required=True, help="folder holding <sequence>/<frame>.npz")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=REPO / "configs" / "student" / "bev_targets_v1.yaml")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--qa", type=int, default=4)
    p.add_argument("--stage-dir", type=Path, help="copy the frames' scans here first")
    p.add_argument("--threads", type=int, default=32)
    args = p.parse_args(argv)

    import yaml
    raw = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    in_cfg, t_cfg = BEVInputConfig.from_dict(raw.get("bev_inputs")), TargetConfig.from_dict(raw.get("targets"))
    seq = (Kitti360Sequence(args.dataset_map) if args.dataset_map else
           Kitti360Sequence(sequence=args.sequence, components=prepared_drive(args.input_root, args.sequence)))
    if args.input_root:
        print(describe_prepared(args.input_root, args.sequence), flush=True)
    label_dir = args.road_labels / seq.sequence
    if not label_dir.is_dir():
        hits = find_outputs(args.road_labels, seq.sequence, "teacher", "road")      # wherever Kaggle mounted it
        label_dir = hits[0] if hits else label_dir
    frames = sorted(int(f.stem) for f in label_dir.glob("*.npz"))[: args.max_frames]
    if not frames:
        raise FileNotFoundError(f"no road labels for {seq.sequence} under {args.road_labels}")
    if args.stage_dir:
        scans_only = {k: v for k, v in seq.paths.items() if k not in ("images", "image_shards", "semantics2d")}
        staged = stage_drive(scans_only, frames, args.stage_dir / seq.sequence, args.threads)
        seq = Kitti360Sequence(sequence=seq.sequence, components={**seq.paths, "scans": staged["scans"]})
    grid = BEVGrid()
    out = args.out / seq.sequence
    out.mkdir(parents=True, exist_ok=True)
    qa = set(np.array(frames)[np.linspace(0, len(frames) - 1, min(args.qa, len(frames))).astype(int)].tolist())
    print(f"{seq.sequence}: {len(frames)} labelled frames -> {out}", flush=True)

    rows, t0 = [], time.time()
    for k, frame in enumerate(frames, start=1):
        points = seq.points(frame)
        label, scores, status = read_road_label(label_dir / f"{frame:010d}.npz")
        inp = bev_inputs(points, seq.camera, grid, in_cfg)
        tgt = build_targets(inp, label, scores, points, seq.camera, grid, t_cfg, in_cfg.ground_band)
        save_record(out / f"{frame:010d}.npz", inp, tgt)
        row = {"fid": seq.fid(frame), "teacher_status": status}
        row.update({key: round(float(inp[key].mean()), 4) for key in SHARES})
        row.update(road_target=round(float(np.isfinite(tgt["road_t"]).mean()), 4),
                   road=round(float((tgt["road_t"] >= 0.5).mean()), 4), unknown=round(float(tgt["unk_t"].mean()), 4),
                   occupied=round(float((tgt["occ_t"] == 1).mean()), 4))
        rows.append(row)
        if frame in qa:
            draw_qa(out / "qa" / f"{frame:010d}.png", seq.fid(frame), seq.image(frame), seq.camera, grid, inp, tgt,
                    in_cfg.ground.plane)
        if k % 100 == 0 or k == len(frames):
            print(f"  {k}/{len(frames)} frames, {(time.time() - t0) / k:.2f} s per frame", flush=True)

    (out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    keys = [k for k in rows[0] if k not in ("fid", "teacher_status")]
    means = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    write_manifest(out / "manifest.json", kind="bev_targets", sequence=seq.sequence, frames=len(rows),
                   bev_inputs=in_cfg.to_dict(), targets=t_cfg.to_dict(), road_labels=str(label_dir),
                   mean_shares=means)
    print(f"done: mean share of the grid: {means}")


if __name__ == "__main__":
    main()
