"""Train the multi-class MTrans teacher on KITTI 3D Object (500 labelled frames + detector boxes on the rest).

    python scripts/train_mtrans.py --input-root /kaggle/input --detections <detector2d_v1>/kitti/training.txt \
        --out /kaggle/working/mtrans_v1 --cache /kaggle/temp/mtrans_cache [--smoke 64] [--time-budget-hours 11]

Resumes from <out>/ckpt/last.pt. Writes records.json (object pools, anchors), metrics.jsonl, val_objects.jsonl
(per-object IoU and confidence on KITTI val) and manifest.json.
"""
import argparse
import json
import shutil
from pathlib import Path

from ocgf.datasets.kitti_object import kitti_fid
from ocgf.datasets.locate import find_kitti_object
from ocgf.teachers.mtrans.train import KITTI_SPLITS, REPO, build_pools, load_config, sha256_file, train
from ocgf.utils.provenance import write_manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", type=Path, required=True, help="folder holding KITTI's training/{image_2,...}")
    p.add_argument("--detections", type=Path, required=True, help="detector output for KITTI (kitti/training.txt)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True, help="scratch folder for per-frame point caches")
    p.add_argument("--config", type=Path, default=REPO / "configs" / "teachers" / "mtrans_v1.yaml")
    p.add_argument("--labelled-frames", help="override the config's list, e.g. configs/splits/kitti_object/mtrans_labelled_125.txt")
    p.add_argument("--epochs", type=int, help="override the config")
    p.add_argument("--smoke", type=int, default=0, metavar="N", help="one epoch on N objects per pool, then exit")
    p.add_argument("--time-budget-hours", type=float, default=11.0)
    p.add_argument("--device")
    p.add_argument("--splits-dir", type=Path, default=KITTI_SPLITS, help="folder with train.txt and val.txt (tests)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.labelled_frames:
        cfg.labelled_frames = args.labelled_frames
    if args.epochs:
        cfg.train.epochs = args.epochs
    folders = find_kitti_object(args.input_root)
    missing = [f for f in ("image_2", "velodyne", "calib", "label_2") if f not in folders]
    if missing:
        raise SystemExit(f"KITTI training folders not found under {args.input_root}: {missing}")
    out = args.out.with_name(args.out.name + "_smoke") if args.smoke else args.out
    if args.smoke and (args.out / "records.json").exists() and not (out / "records.json").exists():
        out.mkdir(parents=True, exist_ok=True)       # reuse the real run's pools (still checked against the inputs)
        shutil.copyfile(args.out / "records.json", out / "records.json")
    read = lambda path: [kitti_fid(f) for f in Path(path).read_text().split()]
    frame_lists = {"train": read(args.splits_dir / "train.txt"), "val": read(args.splits_dir / "val.txt"),
                   "labelled": read(REPO / cfg.labelled_frames)}
    pools = build_pools(cfg, folders, args.detections, args.cache, out, frame_lists=frame_lists)
    result = train(cfg, pools, folders, args.cache, out, device=args.device, time_budget_hours=args.time_budget_hours,
                   smoke=args.smoke)
    write_manifest(out / "manifest.json", teacher="mtrans", config=json.loads(json.dumps(cfg)),
                   labelled_frames=cfg.labelled_frames, labelled_sha256=sha256_file(REPO / cfg.labelled_frames),
                   detections=args.detections.name, detections_sha256=sha256_file(args.detections),
                   counts=pools.counts, anchors=pools.anchors, result=result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
