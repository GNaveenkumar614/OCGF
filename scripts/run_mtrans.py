"""Run a trained MTrans checkpoint on detector boxes and write neutral object files.

KITTI (e.g. the val split, for teacher quality with detector boxes):
    python scripts/run_mtrans.py --checkpoint <mtrans_v1>/ckpt/last.pt --input-root /kaggle/input \
        --detections <detector2d_v1>/kitti/training.txt --kitti-split val --out <objects>/kitti/val.jsonl

KITTI-360 (a prepared drive; detections from notebook 02's kitti360/<sequence>.txt):
    python scripts/run_mtrans.py --checkpoint ... --input-root /kaggle/input --sequence 2013_05_28_drive_0009_sync \
        --detections <detector2d_v1>/kitti360/2013_05_28_drive_0009_sync.txt --out <objects>/kitti360/<sequence>.jsonl

Every detector box at or above --min-score gets one line; see ocgf/teachers/mtrans/infer.py for the format.
"""
import argparse
from pathlib import Path

from ocgf.datasets.kitti_object import KittiObjectFrames, kitti_fid, stage_kitti
from ocgf.datasets.locate import find_kitti_object, prepared_drive
from ocgf.teachers.detector2d.mapping import read_detections
from ocgf.teachers.mtrans.data import ProjectedFrames, detection_objects
from ocgf.teachers.mtrans.infer import Kitti360Frames, infer, load_model
from ocgf.teachers.mtrans.train import KITTI_SPLITS, sha256_file
from ocgf.utils.provenance import write_manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="JSON-lines object file")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--kitti-split", choices=["train", "val"])
    group.add_argument("--sequence", help="a prepared KITTI-360 drive")
    p.add_argument("--min-score", type=float, default=0.1, help="2D score floor (recall-oriented)")
    p.add_argument("--min-points", type=int, default=1, help="frustum points needed for a 3D box")
    p.add_argument("--cache", type=Path, help="scratch folder for per-frame point caches")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device")
    p.add_argument("--stage-dir", type=Path, help="KITTI: copy images and calib here first (slow input mounts)")
    p.add_argument("--threads", type=int, default=32, help="parallel readers for staging and the point cache")
    p.add_argument("--splits-dir", type=Path, default=KITTI_SPLITS, help="folder with train.txt and val.txt (tests)")
    args = p.parse_args(argv)

    detections = read_detections(args.detections)
    if args.kitti_split:
        folders = find_kitti_object(args.input_root)
        fids = [kitti_fid(f) for f in (args.splits_dir / f"{args.kitti_split}.txt").read_text().split()]
        if args.stage_dir:
            folders = stage_kitti(folders, fids, args.stage_dir, args.threads)
        source = KittiObjectFrames(folders)
    else:
        from ocgf.datasets.kitti360 import Kitti360Sequence
        seq = Kitti360Sequence(sequence=args.sequence, components=prepared_drive(args.input_root, args.sequence))
        source = Kitti360Frames({args.sequence: seq})
        fids = sorted(f for f in detections if f.split("/")[1] == args.sequence)
    model, meta = load_model(args.checkpoint, args.device)
    objects = detection_objects(detections, fids, args.min_score, tuple(meta["classes"]))
    frames = ProjectedFrames(source, args.cache)
    frames.warm(fids, args.threads)
    summary = infer(model, meta, frames, objects, args.out, args.batch_size,
                    args.min_points, args.num_workers)
    write_manifest(args.out.with_suffix(".manifest.json"), teacher="mtrans", checkpoint=args.checkpoint.name,
                   checkpoint_sha256=sha256_file(args.checkpoint), epoch=meta["epoch"], classes=meta["classes"],
                   anchors=meta["anchors"], detections=args.detections.name,
                   detections_sha256=sha256_file(args.detections), min_score=args.min_score,
                   min_points=args.min_points, summary=summary)


if __name__ == "__main__":
    main()
