"""Prepare one KITTI-360 sequence: scans with ring indices, frame index, quality report, dev sample.

Runs every stage of ocgf.datasets.prepare_kitti360 in order. On Kaggle:

    python scripts/prepare_kitti360.py --input-root /kaggle/input \
        --out /kaggle/working/kitti360_supplement --scratch /kaggle/temp \
        --sequence 2013_05_28_drive_0009_sync --role test --keep-every-m 2.5 --dev-frames 0

Roles of the paper drives (kitti360_paper_v1): train 0000, 0002, 0004, 0005 · val 0006 · test 0009, 0010.
Drive 0000 was prepared with --role dev, every frame kept and a dev sample.
"""
import argparse
from pathlib import Path

from ocgf.datasets.prepare_kitti360 import PrepConfig, run_all

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequence", default="2013_05_28_drive_0000_sync")
    p.add_argument("--input-root", type=Path, required=True, help="folder holding the attached datasets")
    p.add_argument("--out", type=Path, required=True, help="output folder (kept as the notebook output)")
    p.add_argument("--scratch", type=Path, required=True, help="temporary folder, not kept")
    p.add_argument("--sources", type=Path, default=REPO / "configs" / "data_sources" / "kitti360.yaml")
    p.add_argument("--mode", choices=["auto", "download", "remote"], default="auto",
                   help="how to read the velodyne zip (auto: download if the disk has room)")
    p.add_argument("--role", choices=["dev", "train", "val", "test"], default="dev",
                   help="train drives may leave the semantic windows; val/test/dev may not")
    p.add_argument("--keep-every-m", type=float, help="keep one usable frame every this many metres (2.5 for the "
                                                      "paper drives other than 0000; default: every frame)")
    p.add_argument("--max-frames", type=int, help="stop after this many kept frames (for a quick trial)")
    p.add_argument("--dev-frames", type=int, default=50, help="0: no dev sample (only drive 0000 needs one)")
    args = p.parse_args(argv)
    run_all(PrepConfig(sequence=args.sequence, input_root=args.input_root, out_dir=args.out, scratch_dir=args.scratch,
                       sources=args.sources, role=args.role, keep_every_m=args.keep_every_m, mode=args.mode,
                       max_frames=args.max_frames, dev_frames=args.dev_frames))


if __name__ == "__main__":
    main()
