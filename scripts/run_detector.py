"""Run the zero-shot 2D detector on KITTI 3D Object and the prepared KITTI-360 paper drives (Kaggle notebook 02, GPU).

    python scripts/run_detector.py --input-root /kaggle/input --out /kaggle/working/detector2d_v1 \
        --kitti --drives 0000 0002 0004 0005 0006 0009 0010

Writes <out>/kitti/training.txt (+ _raw.txt), <out>/kitti360/<seq>.txt (+ _raw.txt), and manifest.json.
"""
import argparse
from pathlib import Path

from ocgf.datasets.kitti360 import Kitti360Sequence
from ocgf.datasets.locate import find_kitti_object, prepared_drive
from ocgf.teachers.detector2d.mapping import COCO_TO_OURS
from ocgf.teachers.detector2d.model import Detector
from ocgf.teachers.detector2d.run import kitti360_frames, kitti_object_items, load_png, run_source
from ocgf.utils.provenance import write_manifest

PAPER_DRIVES = ("0000", "0002", "0004", "0005", "0006", "0009", "0010")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kitti", action="store_true", help="run on the 7,481 KITTI 3D Object training images")
    p.add_argument("--drives", nargs="*", default=list(PAPER_DRIVES))
    p.add_argument("--every-m", type=float, default=2.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=2400)
    p.add_argument("--score-floor", type=float, default=0.05)
    p.add_argument("--max-images", type=int, help="per source, for a quick trial")
    args = p.parse_args(argv)

    detector = Detector(min_size=args.min_size, max_size=args.max_size, score_floor=args.score_floor)
    sources = {}
    if args.kitti:
        kitti = find_kitti_object(args.input_root)
        if "image_2" not in kitti:
            raise SystemExit(f"no KITTI training/image_2 under {args.input_root}")
        items = kitti_object_items(kitti["image_2"])[: args.max_images]
        sources["kitti/training"] = run_source(detector, items, load_png, args.out, "kitti", "training", args.batch_size)
    for drive in args.drives:
        seq_name = f"2013_05_28_drive_{drive}_sync"
        seq = Kitti360Sequence(sequence=seq_name, components=prepared_drive(args.input_root, seq_name))
        frames = kitti360_frames(seq, args.every_m)[: args.max_images]
        items = [(seq.fid(int(f)), int(f)) for f in frames]
        sources[f"kitti360/{seq_name}"] = run_source(detector, items, seq.image, args.out, "kitti360", seq_name,
                                                     args.batch_size)
    write_manifest(args.out / "manifest.json", teacher="detector2d", settings=detector.settings,
                   weights_sha256=detector.weights_sha256(), mapping=COCO_TO_OURS, every_m=args.every_m, sources=sources)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
