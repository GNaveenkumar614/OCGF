"""Run a trained student on any frames of a prepared drive, without ground truth: D_free maps, overlays and a video.

    python scripts/predict.py --run /kaggle/input/ocgf-runs-e0-s0/dev_e0_concat_s0 --input-root /kaggle/input \
        --sequence 2013_05_28_drive_0000_sync --frames 10354-10653 --out /kaggle/working/demo --video
    python scripts/predict.py --run runs/dev_e0_concat --dataset-map data/kitti360_dev/dev_sample/dataset_map.yaml \
        --out data/demo --backbone stub --workers 0 --video

The model sees only what it sees at inference: the image, the scan, the calibration (and, for object runs, the drive's
MTrans detections from --objects). Nothing is read from the ground truth, so any drive and any frame works.
Frames: the drive's stored usable frames, optionally within --frames A-B, every --every-th, at most --max-frames
(the default demo: consecutive frames give a smooth video).

Writes <out>/maps/<frame>.npz (float16 p_road, p_occ, p_unk; bool d_free = p_road >= t, p_occ < t, p_unk < t, as in
the evaluation), <out>/frames/<frame>.png (the camera image with the predicted free road and occupancy drawn at
ground level, and the BEV map), <out>/demo.gif (and demo.mp4 when ffmpeg is installed) with --video, manifest.json.
"""
import argparse
import dataclasses
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

from ocgf.datasets.kitti360 import Kitti360Sequence, stage_drive
from ocgf.datasets.locate import describe_prepared, find_object_file, prepared_drive
from ocgf.datasets.objects import read_object_file
from ocgf.datasets.student_sample import IMAGE_H, IMAGE_W
from ocgf.evaluation.perturb import parse_spec
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.models.student import HEADS, Student
from ocgf.training.data import ScanFrames, collate, to_device
from ocgf.training.experiment import resolve
from ocgf.training.train import load_trainable_state
from ocgf.utils.provenance import write_manifest

FREE, OCCUPIED, UNKNOWN, OTHER = (0.18, 0.75, 0.30), (0.88, 0.20, 0.20), (0.22, 0.22, 0.25), (0.62, 0.62, 0.62)


def d_free(probs: dict, t: float = 0.5) -> np.ndarray:
    return (probs["road"] >= t) & (probs["occ"] < t) & (probs["unk"] < t)


def bev_classes(probs: dict, t: float = 0.5) -> np.ndarray:
    """[nx, ny, 3] colours: free road green, occupied red, unknown dark grey, the rest (seen, not free) light grey."""
    img = np.empty(probs["road"].shape + (3,))
    img[:] = OTHER
    img[probs["unk"] >= t] = UNKNOWN
    img[(probs["occ"] >= t) & (probs["unk"] < t)] = OCCUPIED
    img[d_free(probs, t)] = FREE
    return img


def image_points(grid_uv: np.ndarray, crop_left: int):
    """Pixel coordinates in the original image of each cell's ground point (NaN where it is not in view)."""
    u = (grid_uv[..., 0] + 1) * IMAGE_W / 2 - 0.5 + crop_left
    v = (grid_uv[..., 1] + 1) * IMAGE_H / 2 - 0.5
    outside = (np.abs(grid_uv) > 1).any(axis=-1)
    return np.where(outside, np.nan, u), np.where(outside, np.nan, v)


def draw_frame(path, title, image, probs, grid_uv, grid: BEVGrid, crop_left: int, t: float = 0.5):
    fig = Figure(figsize=(16, 4.6))
    ax = fig.add_axes([0.005, 0.02, 0.76, 0.9])
    scale = 0.76 * 16 * 72 / image.shape[1]                 # points per image pixel in this figure
    ax.imshow(image)
    u, v = image_points(grid_uv, crop_left)
    xs, ys = grid.center_grid()
    dist = np.hypot(xs, ys)
    free, occ = d_free(probs, t), (probs["occ"] >= t) & (probs["unk"] < t)
    focal_px = 552.0                                         # about KITTI-360's focal length, for marker sizes only
    for mask, colour in ((free, FREE), (occ, OCCUPIED)):
        sel = mask & np.isfinite(u)
        side = focal_px * grid.res / np.maximum(dist[sel], 1.0) * scale   # a cell's projected width, in points
        ax.scatter(u[sel], v[sel], s=np.clip(side, 1.0, 40.0) ** 2, c=[colour], marker="s", alpha=0.30,
                   linewidths=0)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")
    bx = fig.add_axes([0.79, 0.12, 0.2, 0.8])
    bx.imshow(BEVGrid.to_display(bev_classes(probs, t).transpose(2, 0, 1)).transpose(1, 2, 0),
              extent=(grid.y_max, grid.y_min, grid.x_min, grid.x_max), interpolation="nearest")
    bx.plot([0], [0.6], marker="^", ms=9, color="white", mec="black")              # the ego car, facing up
    bx.set_title("BEV: free road green, occupied red, unknown dark", fontsize=8)
    bx.tick_params(labelsize=7)
    bx.set_xlabel("y (m, left)", fontsize=7)
    bx.set_ylabel("x (m, ahead)", fontsize=7)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)


def write_video(frames_dir: Path, out: Path, fps: int, width: int = 1100) -> list[str]:
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        return []
    imgs = []
    for p in pngs:
        im = Image.open(p).convert("RGB")
        imgs.append(im.resize((width, round(im.height * width / im.width))))
    made = []
    gif = out / "demo.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    made.append(str(gif))
    if shutil.which("ffmpeg"):                       # a file list works everywhere (glob patterns do not on Windows)
        listing = out / "frames.txt"
        listing.write_text("".join(f"file '{p.resolve().as_posix()}'\nduration {1 / fps:.4f}\n" for p in pngs)
                           + f"file '{pngs[-1].resolve().as_posix()}'\n")
        mp4 = out / "demo.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
               "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", str(mp4)]
        if subprocess.run(cmd).returncode == 0:
            made.append(str(mp4))
        listing.unlink(missing_ok=True)
    return made


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="a train_student output folder (ckpt/last.pt)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-map", type=Path)
    src.add_argument("--input-root", type=Path)
    p.add_argument("--sequence")
    p.add_argument("--frames", help="A-B: only frames in this range")
    p.add_argument("--every", type=int, default=1, help="take every n-th stored frame")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--objects", type=Path, help="where to find the drive's object file (object runs only)")
    p.add_argument("--perturb", default="", help="a test-time perturbation for the demo, e.g. rings=4")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", action="store_true", help="also write demo.gif (and demo.mp4 with ffmpeg)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device")
    p.add_argument("--backbone", help="override the run's backbone (stub: CPU tests only)")
    p.add_argument("--stage-dir", type=Path)
    p.add_argument("--threads", type=int, default=32)
    args = p.parse_args(argv)

    ckpt = torch.load(args.run / "ckpt" / "last.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    res = resolve(cfg)
    model_cfg = dataclasses.replace(res.model, **({"backbone": args.backbone} if args.backbone else {}))
    model = Student(model_cfg)
    load_trainable_state(model, ckpt["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    if args.dataset_map:
        seq = Kitti360Sequence(args.dataset_map)
    else:
        seq = Kitti360Sequence(sequence=args.sequence, components=prepared_drive(args.input_root, args.sequence))
        print(describe_prepared(args.input_root, args.sequence), flush=True)

    frames = seq.usable_frames()
    if args.frames:
        lo, hi = (int(v) for v in args.frames.split("-"))
        frames = frames[(frames >= lo) & (frames <= hi)]
    frames = [int(f) for f in frames[:: args.every][: args.max_frames]]
    if not frames:
        raise SystemExit("no stored usable frames in that range")
    objects = None
    if res.model.objects or res.inputs.box_occluders:
        if args.objects is None:
            raise SystemExit("this run uses MTrans objects: pass --objects")
        path = find_object_file(args.objects, seq.sequence, res.object_source)
        if path is None:
            raise SystemExit(f"no {res.object_source} object file for {seq.sequence} under {args.objects}")
        objects = read_object_file(path, res.objects.min_score)
    if args.stage_dir:
        keep = {k: v for k, v in seq.paths.items() if k != "semantics2d"}
        seq = Kitti360Sequence(sequence=seq.sequence, components=stage_drive(keep, frames, args.stage_dir / seq.sequence,
                                                                              args.threads))
    grid = BEVGrid()
    crop_left = (seq.camera.width - IMAGE_W) // 2
    print(f"run {args.run}: {model_cfg.mode}, checkpoint after epoch {ckpt['epoch']} of {ckpt['epochs']}; "
          f"{len(frames)} frames {frames[0]}..{frames[-1]}" + (f"; perturbation {args.perturb}" if args.perturb else ""),
          flush=True)

    data = ScanFrames(seq, frames, grid, res.inputs, objects, res.objects, object_inputs=res.model.objects,
                      perturb=parse_spec(args.perturb))
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    (args.out / "maps").mkdir(parents=True, exist_ok=True)
    shares, t0 = [], time.time()
    with torch.no_grad():
        for batch in loader:
            pred = model(to_device(batch, device))
            probs = {k: torch.sigmoid(pred[k].float()).cpu().numpy() for k in HEADS}
            for i, f in enumerate(batch["frame"].tolist()):
                p_i = {k: v[i] for k, v in probs.items()}
                free = d_free(p_i)
                np.savez_compressed(args.out / "maps" / f"{f:010d}.npz", d_free=free,
                                    **{f"p_{k}": v.astype(np.float16) for k, v in p_i.items()})
                draw_frame(args.out / "frames" / f"{f:010d}.png", f"{seq.fid(f)} · {model_cfg.mode}"
                           + (f" · {args.perturb}" if args.perturb else ""), seq.image(f), p_i,
                           batch["grid_uv"][i].numpy(), grid, crop_left)
                shares.append(float(free.mean()))
            done = len(shares)
            if done % 50 < args.batch_size or done == len(frames):
                print(f"  {done}/{len(frames)} frames, {(time.time() - t0) / done:.2f} s per frame", flush=True)
    videos = write_video(args.out / "frames", args.out, args.fps) if args.video else []
    write_manifest(args.out / "manifest.json", kind="student_predictions", run=str(args.run), mode=model_cfg.mode,
                   checkpoint_epoch=int(ckpt["epoch"]), sequence=seq.sequence, frames=[frames[0], frames[-1], len(frames)],
                   every=args.every, perturbation=args.perturb, mean_free_share=round(float(np.mean(shares)), 4),
                   videos=videos, note="no ground truth used")
    print(f"done: {len(frames)} frames -> {args.out}; mean share of the grid called free {np.mean(shares):.3f}"
          + (f"; videos {videos}" if videos else ""))


if __name__ == "__main__":
    main()
