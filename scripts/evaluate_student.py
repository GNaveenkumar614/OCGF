"""Evaluate a trained student against the evaluation ground truth (ocgf/evaluation/metrics.py).

    python scripts/evaluate_student.py --run runs/dev_e0_concat --dataset-map data/kitti360_dev/dev_sample/dataset_map.yaml \
        --eval-gt data/eval_gt --split configs/splits/kitti360_d0000_dev_v1.json --part train --backbone stub
    python scripts/evaluate_student.py --run /kaggle/working/dev_e0_concat_s0 --input-root /kaggle/input \
        --sequence 2013_05_28_drive_0000_sync --eval-gt /kaggle/temp/eval_gt \
        --split configs/splits/kitti360_d0000_dev_v1.json --part val --stage-dir /kaggle/temp/stage

Frames: the frames of the split part that have ground truth. The model sees only the image, the scan and the
calibration: its BEV inputs are recomputed from the scan, as at inference. The ground truth is read only to score.
Writes <out>/metrics.json (micro-averaged metrics, the checkpoint's epoch, the run's config), frames.jsonl (per-frame
values, for dispersion and failure cases), qa/<frame>.png and, with --save-preds, preds/<frame>.npz (float16 p_road,
p_occ, p_unk).

Objects: a run whose config uses object inputs or box occluders reads the drive's MTrans object file (--objects), the
same detections it would have at inference. --post-hoc-boxes is the E1 baseline arm: after prediction, every cell
inside a detected 3D box (the boxes the object models receive) is set occupied, so it is never called free.

Robustness (E5): --perturb applies a test-time perturbation (ocgf/evaluation/perturb.py: fewer LiDAR rings, point
dropout, camera blur/noise/exposure/patches, box jitter/dropout); every input is recomputed from the perturbed data,
and the post-hoc arm subtracts the same perturbed boxes.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from ocgf.datasets.kitti360 import Kitti360Sequence, stage_drive
from ocgf.datasets.locate import describe_prepared, find_object_file, find_outputs, prepared_drive
from ocgf.datasets.objects import FrameObjects, any_footprints, read_object_file
from ocgf.datasets.splits import from_ranges
from ocgf.evaluation.gate_log import GateLog
from ocgf.evaluation.gt_builder import load_layers
from ocgf.evaluation.perturb import parse_spec, perturb_objects, spec_name
from ocgf.evaluation.metrics import Evaluator, MetricConfig, gt_masks
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.models.student import HEADS, Student
from ocgf.training.data import ScanFrames, collate, to_device
from ocgf.training.experiment import resolve
from ocgf.training.train import load_trainable_state
from ocgf.utils.provenance import write_manifest


def find_gt_dir(root: Path, sequence: str) -> Path:
    found = find_outputs(root, sequence, "kind", "eval_gt")
    if not found:
        raise FileNotFoundError(f"no evaluation ground truth for {sequence} under {root}")
    return found[0]


def part_frames(split: dict, sequence: str, part: str, available: np.ndarray) -> np.ndarray:
    if "splits" in split:
        if split.get("sequence") != sequence:
            return available[:0]
        return available[np.isin(available, from_ranges(split["splits"][part]))]
    return available if sequence in split.get(part, []) else available[:0]


def draw_qa(path, fid, image, probs, layers, grid, threshold):
    g = gt_masks(layers)
    pred = (probs["road"] >= threshold) & (probs["occ"] < threshold) & (probs["unk"] < threshold)
    gt_img = np.zeros(grid.shape + (3,))
    gt_img[g["free"]] = (0.2, 0.7, 0.3)
    gt_img[g["unknown"]] = (0.75, 0.75, 0.75)
    gt_img[g["occupied"]] = (0.85, 0.2, 0.2)
    err = np.ones(grid.shape + (3,))
    err[pred & g["free"]] = (0.2, 0.7, 0.3)                        # correct free
    err[pred & g["hazard"]] = (0.9, 0.1, 0.1)                       # false free
    err[~pred & g["free"]] = (0.3, 0.45, 0.95)                      # missed free
    ext = (grid.y_max, grid.y_min, grid.x_min, grid.x_max)
    fig = Figure(figsize=(15, 7.4))
    panels = [("GT: free road green, occupied red, unknown grey, other surface black", BEVGrid.to_display(gt_img.transpose(2, 0, 1)).transpose(1, 2, 0), None),
              ("p_road", BEVGrid.to_display(probs["road"]), "RdYlGn"),
              ("p_occ", BEVGrid.to_display(probs["occ"]), "magma"),
              ("p_unk", BEVGrid.to_display(probs["unk"]), "gray_r"),
              ("D_free vs GT: correct green, false-free red, missed blue",
               BEVGrid.to_display(err.transpose(2, 0, 1)).transpose(1, 2, 0), None)]
    for k, (title, img, cmap) in enumerate(panels):
        ax = fig.add_axes([0.005 + k * 0.2, 0.56, 0.19, 0.40])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, extent=ext, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.50])
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(fid, fontsize=9)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="a train_student output folder (ckpt/last.pt)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-map", type=Path)
    src.add_argument("--input-root", type=Path)
    p.add_argument("--sequence", nargs="+")
    p.add_argument("--eval-gt", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--part", default="val")
    p.add_argument("--out", type=Path, help="default: <run>/eval_<part>")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device")
    p.add_argument("--backbone", help="override the run's backbone (stub: CPU tests only)")
    p.add_argument("--qa", type=int, default=6)
    p.add_argument("--save-preds", action="store_true")
    p.add_argument("--stage-dir", type=Path)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--objects", type=Path, help="where to find <sequence>.jsonl MTrans object files")
    p.add_argument("--post-hoc-boxes", action="store_true",
                   help="E1 baseline: cells inside detected 3D boxes are set occupied after prediction")
    p.add_argument("--perturb", default="", help="E5 test-time perturbation, e.g. rings=2 or box_drop=1.0")
    p.add_argument("--perturb-seed", type=int, default=0)
    args = p.parse_args(argv)

    ckpt = torch.load(args.run / "ckpt" / "last.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    res = resolve(cfg)
    model_cfg = dataclasses.replace(res.model, **({"backbone": args.backbone} if args.backbone else {}))
    model = Student(model_cfg)
    load_trainable_state(model, ckpt["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    in_cfg = res.inputs
    needs_objects = res.model.objects or res.inputs.box_occluders or args.post_hoc_boxes
    if needs_objects and args.objects is None:
        raise SystemExit("this run (or --post-hoc-boxes) uses MTrans objects: pass --objects")
    grid, mcfg = BEVGrid(), MetricConfig()
    split = json.loads(args.split.read_text())
    perturb = parse_spec(args.perturb)
    out = args.out or args.run / f"eval_{args.part}" / spec_name(args.perturb)
    out.mkdir(parents=True, exist_ok=True)
    print(f"run {args.run}: {model_cfg.mode}, objects {res.switches} from {res.object_source}"
          + (" (ORACLE: ground truth as input)" if res.object_source != "detector" and res.uses_objects else "")
          + f", checkpoint after epoch {ckpt['epoch']} of "
          f"{ckpt['epochs']}" + ("; post-hoc box subtraction" if args.post_hoc_boxes else "")
          + (f"; perturbation {args.perturb}" if args.perturb else ""), flush=True)

    seqs = ([Kitti360Sequence(args.dataset_map)] if args.dataset_map else
            [Kitti360Sequence(sequence=s, components=prepared_drive(args.input_root, s)) for s in args.sequence])
    for s in (args.sequence or []) if args.input_root else []:
        print(describe_prepared(args.input_root, s), flush=True)
    ev, rows, frames_info, object_files, t0 = Evaluator(grid, mcfg), [], {}, {}, time.time()
    gate_log = GateLog() if model_cfg.mode == "ocgf" else None
    for seq in seqs:
        gt_dir = find_gt_dir(args.eval_gt, seq.sequence)
        available = np.array(sorted(int(f.stem) for f in gt_dir.glob("*.npz")), np.int64)
        frames = part_frames(split, seq.sequence, args.part, available).tolist()
        frames_info[seq.sequence] = len(frames)
        if not frames:
            continue
        objects = None
        if needs_objects:
            path = find_object_file(args.objects, seq.sequence, res.object_source)
            if path is None:
                raise SystemExit(f"no {res.object_source} object file for {seq.sequence} under {args.objects}")
            objects, object_files[seq.sequence] = read_object_file(path, res.objects.min_score), str(path)
        if args.stage_dir:
            keep = {k: v for k, v in seq.paths.items() if k != "semantics2d"}
            seq = Kitti360Sequence(sequence=seq.sequence,
                                   components=stage_drive(keep, frames, args.stage_dir / seq.sequence, args.threads))
        qa = set(np.array(frames)[np.linspace(0, len(frames) - 1, min(args.qa, len(frames))).astype(int)].tolist())
        data = ScanFrames(seq, frames, grid, in_cfg, objects, res.objects, object_inputs=res.model.objects,
                          perturb=perturb, seed=args.perturb_seed)
        loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, num_workers=args.workers,
                                             collate_fn=collate)
        with torch.no_grad():
            for batch in loader:
                pred = model(to_device(batch, device))
                probs = {k: torch.sigmoid(pred[k].float()).cpu().numpy() for k in HEADS}
                if gate_log is not None:
                    w_c = pred["gate_w"][:, 0].float().cpu().numpy()
                    q = pred["Q"].float().cpu().numpy()
                    v_c, v_l = batch["M_C"].numpy(), batch["valid_L"].numpy()
                    for i in range(len(w_c)):
                        gate_log.add(w_c[i], q[i], v_c[i], v_l[i])
                for i, f in enumerate(batch["frame"].tolist()):
                    p_i = {k: v[i] for k, v in probs.items()}
                    if args.post_hoc_boxes:
                        seen = perturb_objects(objects.get(f, FrameObjects.empty()), perturb, f, args.perturb_seed)
                        boxes = any_footprints(seen, grid)
                        p_i["occ"] = np.where(boxes, 1.0, p_i["occ"])
                    layers = load_layers(gt_dir / f"{f:010d}.npz")
                    fr = ev.add(p_i, layers)
                    rows.append({"fid": seq.fid(f), **{f"{r}/{k}": v for r, d in fr.items() for k, v in d.items()}})
                    if args.save_preds:
                        (out / "preds").mkdir(exist_ok=True)
                        np.savez_compressed(out / "preds" / f"{f:010d}.npz",
                                            **{k: v.astype(np.float16) for k, v in p_i.items()})
                    if f in qa:
                        draw_qa(out / "qa" / f"{f:010d}.png", seq.fid(f), seq.image(f), p_i, layers, grid,
                                mcfg.threshold)
        print(f"{seq.sequence}: {len(frames)} frames, {(time.time() - t0) / max(len(rows), 1):.2f} s per frame", flush=True)
    if not rows:
        raise SystemExit(f"no frames to evaluate in part '{args.part}' of {split.get('name')}: ground-truth frames per "
                         f"drive in that part {frames_info}. A trial data-prep (e.g. 200 frames at the start of the "
                         f"drive) has frames in the train part only: use --part train for a pipeline check.")

    (out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = ev.result()
    write_manifest(out / "metrics.json", kind="student_eval", run=str(args.run), mode=model_cfg.mode,
                   objects=res.switches, object_source=res.object_source, post_hoc_boxes=args.post_hoc_boxes,
                   object_files=object_files,
                   perturbation=args.perturb, perturbation_seed=args.perturb_seed,
                   checkpoint_epoch=int(ckpt["epoch"]), epochs=int(ckpt["epochs"]), split=split.get("name"),
                   part=args.part, frames=frames_info, run_config=cfg, metrics=result,
                   gate=gate_log.result() if gate_log is not None else None)
    print(f"\n{'region':>20} {'ffr_pred':>9} {'ffr_hazard':>10} {'free IoU':>9} {'free P':>7} {'free R':>7}")
    for name, r in result["regions"].items():
        fr = r["free_road"]
        fmt = lambda v: "   n/a" if v is None else f"{v:.3f}"
        print(f"{name:>20} {fmt(r['ffr_pred']):>9} {fmt(r['ffr_hazard']):>10} {fmt(fr['iou']):>9} "
              f"{fmt(fr['precision']):>7} {fmt(fr['recall']):>7}")
    print(f"calibration ECE (p_free vs free): {result['calibration']['ece']:.4f}")
    if gate_log is not None:
        g = gate_log.result()
        print(f"gate: mean w_C {g['mean_w_C']} over {g['cells']['both_valid']:,} cells where both sensors are valid")
        for c in ("H_cam", "range", "log_points", "r_g"):
            print(f"  w_C by {c}: " + ", ".join(f"{b['lo']:g}-{b['hi']:g}: {b['mean_w_C']}" for b in g["by_channel"][c]
                                               if b["cells"]))
    print("per class, share of box cells called free: " +
          ", ".join(f"{k} {fmt(v['called_free'])} ({v['cells']} cells)" for k, v in result["classes"].items()))


if __name__ == "__main__":
    main()
