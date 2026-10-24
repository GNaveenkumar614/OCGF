"""Train the student (E0 baselines, E1/E2 object arms) on road-labelled KITTI-360 frames.

    python scripts/train_student.py --config configs/student/e0_concat.yaml \
        --dataset-map data/kitti360_dev/dev_sample/dataset_map.yaml --road-labels data/road_labels \
        --split configs/splits/kitti360_d0000_dev_v1.json --records data/bev_records --out runs/dev_e0_concat \
        --backbone stub --workers 0 --smoke 4
    python scripts/train_student.py --config configs/student/e1_objects.yaml --input-root /kaggle/input \
        --sequence 2013_05_28_drive_0000_sync --road-labels /kaggle/input --objects /kaggle/input \
        --split configs/splits/kitti360_d0000_dev_v1.json --records /kaggle/temp/bev_records \
        --stage-dir /kaggle/temp/stage --out /kaggle/working/runs/e1_objects_s0 --seed 0

Frames: the road-labelled frames in the split's train part (and its val part, for a loss against the teachers' labels
that only shows convergence). The split is frame-level for one drive (kitti360_d0000_dev_v1: starter phase, debugging
only) or drive-level (kitti360_paper_v1). BEV records are built for any frame that has none yet; a records folder
holds one configuration only (inputs, targets, object file). With objects (the config's `objects:` switches),
--objects is searched for each drive's MTrans object file. Writes train.log, metrics.jsonl, ckpt/last.pt and
manifest.json (resolved config, seed, git commit, frames, object file) to --out; the same command again resumes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ocgf.datasets.bev_inputs import bev_inputs
from ocgf.datasets.kitti360 import Kitti360Sequence, stage_drive
from ocgf.datasets.locate import describe_prepared, find_object_file, find_outputs, prepared_drive
from ocgf.datasets.objects import read_object_file
from ocgf.datasets.splits import from_ranges
from ocgf.geometry.bev_grid import BEVGrid
from ocgf.models.student import Student
from ocgf.training.data import StudentFrames
from ocgf.training.experiment import load_config, resolve
from ocgf.training.targets import build_targets, read_road_label, save_record
from ocgf.training.train import train
from ocgf.utils.provenance import write_manifest


def file_logger(path: Path):
    """Print and append to a timestamped log file (a saved Kaggle run shows its cell output only at the end)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    return log


def split_frames(split: dict, sequence: str, part: str) -> np.ndarray | None:
    """Frames of `sequence` in the split's part: an array (frame-level split), None for all (drive-level), or empty."""
    if "splits" in split:
        return from_ranges(split["splits"][part]) if split.get("sequence") == sequence else np.zeros(0, np.int64)
    return None if sequence in split.get(part, []) else np.zeros(0, np.int64)


def find_label_dir(root: Path, sequence: str) -> Path:
    """The road teacher's folder for the sequence under root (wherever Kaggle mounted it): its manifest says so."""
    found = find_outputs(root, sequence, "teacher", "road")
    if not found:
        raise FileNotFoundError(f"no road labels for {sequence} under {root}: attach the road teacher's output")
    return found[0]


def object_source(path: Path) -> dict:
    """Provenance of an object file: its MTrans checkpoint and detections (from the manifest beside it)."""
    m = json.loads(path.with_name(path.stem + ".manifest.json").read_text())
    return {"file": str(path), **{k: m.get(k) for k in ("checkpoint_sha256", "epoch", "detections_sha256", "min_score")}}


def ensure_records(seq, frames, label_dir: Path, rec_dir: Path, res, objects, obj_src, log) -> None:
    """Build the BEV record of every frame that has none; refuse records built with another configuration."""
    rec_dir.mkdir(parents=True, exist_ok=True)
    uses_boxes = res.inputs.box_occluders or res.targets.box_targets
    want = {"bev_inputs": res.inputs.to_dict(), "targets": res.targets.to_dict(),
            "objects": obj_src if uses_boxes else None}
    cfg_file = rec_dir / "config.json"
    if cfg_file.exists():
        if json.loads(cfg_file.read_text()) != json.loads(json.dumps(want)):
            raise ValueError(f"{rec_dir} holds records built with another configuration: use another --records folder")
    else:
        cfg_file.write_text(json.dumps(want, indent=1))
    todo = [f for f in frames if not (rec_dir / f"{f:010d}.npz").exists()]
    if not todo:
        return
    grid, t0 = BEVGrid(), time.time()
    log(f"{seq.sequence}: building {len(todo)} BEV records in {rec_dir}")
    for k, f in enumerate(todo, start=1):
        points = seq.points(f)
        objs = objects.get(f) if objects is not None else None
        label, scores, _ = read_road_label(label_dir / f"{f:010d}.npz")
        inp = bev_inputs(points, seq.camera, grid, res.inputs, objs)
        save_record(rec_dir / f"{f:010d}.npz", inp, build_targets(inp, label, scores, points, seq.camera, grid,
                                                                    res.targets, res.inputs.ground_band, objs))
        if k % 200 == 0 or k == len(todo):
            log(f"  {k}/{len(todo)} records, {(time.time() - t0) / k:.2f} s each")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-map", type=Path)
    src.add_argument("--input-root", type=Path)
    p.add_argument("--sequence", nargs="+", help="with --input-root: the training drives")
    p.add_argument("--road-labels", type=Path, required=True)
    p.add_argument("--objects", type=Path, help="where to find <sequence>.jsonl MTrans object files")
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True, help="BEV records folder, shared between runs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage-dir", type=Path, help="copy the frames' scans and images here first")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--time-budget-hours", type=float, help="stop cleanly before this many hours (resume later)")
    p.add_argument("--backbone", help="override the model backbone (e.g. stub)")
    p.add_argument("--max-frames", type=int, help="use at most this many train frames per drive (trials)")
    p.add_argument("--device")
    p.add_argument("--smoke", type=int, default=0, help="N steps of one epoch and a checkpoint reload check")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    for key, val in (("epochs", args.epochs), ("batch_size", args.batch_size), ("num_workers", args.workers),
                     ("time_budget_hours", args.time_budget_hours)):
        if val is not None:
            cfg["train"][key] = val
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    res = resolve(cfg)
    seed = int(cfg.get("seed", 0))
    log = file_logger(args.out / "train.log")
    log(f"config {args.config} (seed {seed}): {json.dumps({k: cfg.get(k) for k in ('model', 'objects', 'loss', 'train')})}")
    if res.uses_objects and args.objects is None:
        sys.exit(f"the config uses objects {res.switches}: pass --objects (where the MTrans object files are)")

    split = json.loads(args.split.read_text())
    if args.dataset_map:
        seqs = [Kitti360Sequence(args.dataset_map)]
    else:
        seqs = [Kitti360Sequence(sequence=s, components=prepared_drive(args.input_root, s)) for s in args.sequence]
        for s in args.sequence:
            log(describe_prepared(args.input_root, s))
    train_sets, val_sets, frames_info, object_files = [], [], {}, {}
    for seq in seqs:
        label_dir = find_label_dir(args.road_labels, seq.sequence)
        labelled = np.array(sorted(int(f.stem) for f in label_dir.glob("*.npz")), np.int64)
        parts = {}
        for part in ("train", "val"):
            allowed = split_frames(split, seq.sequence, part)
            parts[part] = labelled if allowed is None else labelled[np.isin(labelled, allowed)]
        parts["train"] = parts["train"][: args.max_frames]
        frames_info[seq.sequence] = {k: len(v) for k, v in parts.items()}
        needed = np.concatenate([parts["train"], parts["val"]]).astype(int).tolist()
        if not needed:
            continue
        objects, obj_src = None, None
        if res.uses_objects:
            path = find_object_file(args.objects, seq.sequence)
            if path is None:
                sys.exit(f"no MTrans object file for {seq.sequence} under {args.objects} (notebook 04's output)")
            objects, obj_src = read_object_file(path, res.objects.min_score), object_source(path)
            object_files[seq.sequence] = obj_src
            with_objects = float(np.mean([f in objects for f in needed]))
            log(f"{seq.sequence}: objects from {path} (MTrans epoch {obj_src['epoch']}); "
                f"{with_objects:.0%} of the frames have detections")
            if with_objects < 0.5:
                log("  WARNING: under half the frames have detections: was the detector run on these frames?")
        if args.stage_dir:
            keep = {k: v for k, v in seq.paths.items() if k != "semantics2d"}
            staged = stage_drive(keep, needed, args.stage_dir / seq.sequence, args.threads, log)
            seq = Kitti360Sequence(sequence=seq.sequence, components=staged)
        rec_dir = args.records / seq.sequence
        ensure_records(seq, needed, label_dir, rec_dir, res, objects, obj_src, log)
        plane = res.inputs.ground.plane
        model_objects = objects if res.model.objects else None
        for part, sets in (("train", train_sets), ("val", val_sets)):
            if len(parts[part]):
                sets.append(StudentFrames(seq, parts[part], rec_dir, plane=plane, objects=model_objects,
                                          obj_cfg=res.objects))
    log(f"frames per drive: {frames_info}")
    if not train_sets:
        sys.exit("no training frames: check --split, --road-labels and --sequence")
    train_set = torch.utils.data.ConcatDataset(train_sets)
    val_set = torch.utils.data.ConcatDataset(val_sets) if val_sets else None

    run = {"config": cfg, "seed": seed, "frames": frames_info, "split": split.get("name"),
           "road_labels": str(args.road_labels), "records": str(args.records), "object_files": object_files}
    write_manifest(args.out / "manifest.json", kind="student_run", **run)
    model = Student(res.model)
    result = train(model, train_set, val_set, res.train, res.loss, args.out, args.device, seed, args.smoke, run, log)
    write_manifest(args.out / "manifest.json", kind="student_run", result=result, **run)
    log(f"result: {result}")


if __name__ == "__main__":
    main()
