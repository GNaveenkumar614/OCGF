"""Road teacher: soft road labels for the training frames of a prepared KITTI-360 drive (see ocgf/teachers/road/).

    python scripts/run_road_teacher.py --dataset-map data/kitti360_dev/dev_sample/dataset_map.yaml \
        --out data/road_labels --all-frames --camera none --qa 4 --save-parts
    python scripts/run_road_teacher.py --input-root /kaggle/input --sequence 2013_05_28_drive_0002_sync \
        --out /kaggle/working/road_labels --qa 8

Frames: usable frames sampled every 5 m (--all-frames: every stored usable frame), in driving order (the camera step
reuses the previous frame's road prototype). Writes <out>/<sequence>/<frame>.npz, frames.jsonl (status, wheel pairs,
audit counts), manifest.json and qa/<frame>.png. Frames with KITTI-360 2D semantics are audited against them (road =
label 7); the audit is reported, never used to train or tune.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from ocgf.datasets.kitti360 import Kitti360Sequence, stage_drive
from ocgf.datasets.locate import describe_prepared, prepared_drive
from ocgf.geometry.projection import project
from ocgf.teachers.road.config import RoadConfig
from ocgf.teachers.road.label import N_RINGS, audit_counts, audit_summary, label_frame, save_result
from ocgf.teachers.road.trajectory import future_path, path_positions
from ocgf.utils.provenance import write_manifest

REPO = Path(__file__).resolve().parents[1]


def draw_qa(path, fid, image, points, result, camera, semantic=None):
    fig = Figure(figsize=(14, 7.6))
    ax1, ax2 = fig.add_axes([0.01, 0.51, 0.98, 0.45]), fig.add_axes([0.01, 0.02, 0.98, 0.45])
    for ax in (ax1, ax2):
        ax.imshow(image)
        ax.set_xlim(-0.5, camera.width - 0.5)
        ax.set_ylim(camera.height - 0.5, -0.5)
        ax.axis("off")
    scores = result["point_scores"]
    scored = np.isfinite(scores)
    uv, _, ok = project(points[scored, :3], camera)
    sc = ax1.scatter(uv[ok, 0], uv[ok, 1], c=scores[scored][ok], cmap="RdYlGn", vmin=0, vmax=1, s=2, linewidths=0)
    fig.colorbar(sc, ax=ax1, fraction=0.015, pad=0.005)
    if result["path_mask"].any():
        ax1.contour(result["path_mask"], levels=[0.5], colors="#00e5ff", linewidths=1.2)
    for wheel in result["wheels"].reshape(-1, 3)[:200]:
        wuv, _, wok = project(wheel[None], camera)
        if wok[0]:
            ax1.plot(wuv[0, 0], wuv[0, 1], "o", ms=3, mfc="none", mec="#00e5ff")
    ax1.set_title(f"{fid}: scored LiDAR points (red 0 .. green 1), path polygon (cyan); status {result['status']}",
                  fontsize=10)
    label = np.ma.masked_invalid(result["label"].astype(np.float64))
    ax2.imshow(label, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.5)
    title = "road label (unknown = no overlay)"
    if semantic is not None:
        ax2.contour(semantic == 7, levels=[0.5], colors="#1f6fd1", linewidths=1.0)
        title += "; KITTI-360 2D road outline (blue, audit only)"
    ax2.set_title(title, fontsize=10)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=85)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-map", type=Path)
    src.add_argument("--input-root", type=Path)
    p.add_argument("--sequence")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=REPO / "configs" / "teachers" / "road_v1.yaml")
    p.add_argument("--camera", default=None, help="DINOv2 hub model for the camera step, or 'none' (default: config)")
    p.add_argument("--all-frames", action="store_true")
    p.add_argument("--max-frames", type=int, help="stop after this many frames (trials)")
    p.add_argument("--qa", type=int, default=4)
    p.add_argument("--save-parts", action="store_true", help="also keep the LiDAR and camera labels per frame")
    p.add_argument("--device")
    p.add_argument("--stage-dir", type=Path, help="copy the selected frames' scans, images and 2D semantics here first")
    p.add_argument("--threads", type=int, default=32)
    args = p.parse_args(argv)

    import yaml
    cfg = RoadConfig.from_dict(yaml.safe_load(args.config.read_text()) if args.config.exists() else {})
    camera_model = cfg.camera_model if args.camera is None else args.camera
    seq = (Kitti360Sequence(args.dataset_map) if args.dataset_map else
           Kitti360Sequence(sequence=args.sequence, components=prepared_drive(args.input_root, args.sequence)))
    if args.input_root:
        print(describe_prepared(args.input_root, args.sequence), flush=True)
    idx = seq.index
    sel = idx["usable"] & idx.get("kept", True)
    if not args.all_frames:
        sel &= idx["sample_5m"]
    frames = idx["frame"][sel][: args.max_frames]
    if args.stage_dir:
        staged = stage_drive(seq.paths, frames, args.stage_dir / seq.sequence, args.threads)
        seq = Kitti360Sequence(sequence=seq.sequence, components=staged)
    pframes, positions = path_positions({f: seq.T_world_lidar(int(f)) for f in seq.posed_frames()}, seq.T_pose_lidar)
    features_fn = None
    if camera_model and camera_model != "none":
        from ocgf.teachers.road.camera_label import DinoFeatures
        features_fn = DinoFeatures(camera_model, device=args.device, scale=cfg.img_scale)
    out = args.out / seq.sequence
    out.mkdir(parents=True, exist_ok=True)
    qa = set(frames[np.linspace(0, len(frames) - 1, min(args.qa, len(frames))).astype(int)].tolist()) if len(frames) else set()
    sem_dir = Path(seq.paths["semantics2d"]) / "semantic" if "semantics2d" in seq.paths else None
    print(f"{seq.sequence}: {len(frames)} frames, camera step: {camera_model or 'none'} -> {out}", flush=True)

    prototype, rows, counts = None, [], {k: np.zeros(3, np.int64) for k in ("label", "lidar", "camera")}
    t0 = time.time()
    for k, frame in enumerate(frames, start=1):
        frame = int(frame)
        points, image = seq.points(frame), seq.image(frame)
        path = future_path(frame, pframes, positions, seq.T_world_lidar(frame), cfg.trajectory_length)
        result, prototype = label_frame(points, image, seq.camera, path, cfg, features_fn, prototype)
        save_result(out / f"{frame:010d}.npz", result, args.save_parts)
        row = {"fid": seq.fid(frame), "status": result["status"], "wheel_pairs": result["n_wheels"]}
        sem_file = sem_dir / f"{frame:010d}.png" if sem_dir else None
        semantic = np.asarray(Image.open(sem_file)) if sem_file is not None and sem_file.exists() else None
        if semantic is not None and result["status"] == "ok":
            for name, key in (("label", "label"), ("lidar", "lidar_label"), ("camera", "camera_label")):
                if not np.isfinite(result[key]).any():      # a step that did not run (e.g. --camera none)
                    continue
                c = audit_counts(result[key], semantic)
                counts[name] += c
                row[f"audit_{name}"] = c.tolist()
        rows.append(row)
        if frame in qa:
            draw_qa(out / "qa" / f"{frame:010d}.png", seq.fid(frame), image, points, result, seq.camera, semantic)
        if k % 100 == 0 or k == len(frames):
            print(f"  {k}/{len(frames)} frames, {(time.time() - t0) / k:.2f} s per frame", flush=True)

    (out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    statuses = {s: sum(r["status"] == s for r in rows) for s in ("ok", "no_trajectory", "no_path", "rings")}
    audit = {name: audit_summary(c) for name, c in counts.items() if c.sum()}
    write_manifest(out / "manifest.json", teacher="road", sequence=seq.sequence, config=cfg.to_dict(),
                   camera_model=camera_model, frames=len(rows), statuses=statuses, audit_vs_2d_semantics=audit,
                   n_rings=N_RINGS)
    print(f"done: {statuses}; audit vs KITTI-360 2D road: " + json.dumps({k: v['iou'] for k, v in audit.items()}))


if __name__ == "__main__":
    main()
