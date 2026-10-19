"""Choose the KITTI train frames MTrans may take 3D labels from: 500 and the nested 125 (S1 ablation).

Uses zero-shot detections only (notebook 02's kitti/training.txt), never label_2. Writes, once (frozen like the splits):
  configs/splits/kitti_object/mtrans_labelled_500.txt, mtrans_labelled_125.txt   (6-digit KITTI frame ids)
  configs/splits/kitti_object/mtrans_labelled.json                              (report and provenance)

    python scripts/select_labelled_frames.py --detections data/detector2d_v1/kitti/training.txt
"""
import argparse
import hashlib
import json
from pathlib import Path

from ocgf.datasets.ids import frame_id, parse_frame_id
from ocgf.teachers.detector2d.mapping import read_detections
from ocgf.teachers.label_budget import select_labelled_frames

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "configs" / "splits" / "kitti_object"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detections", type=Path, required=True, help="kitti/training.txt from the detector run")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--n-small", type=int, default=125)
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    outs = [SPLITS / f"mtrans_labelled_{args.n}.txt", SPLITS / f"mtrans_labelled_{args.n_small}.txt",
            SPLITS / "mtrans_labelled.json"]
    if any(o.exists() for o in outs):
        raise SystemExit("the labelled-frame lists exist already and are frozen; delete them deliberately to redo")
    train = (SPLITS / "train.txt").read_text().split()
    fids = [frame_id("kitti", "training", int(f)) for f in train]
    big, small, report = select_labelled_frames(read_detections(args.detections), fids, args.n, args.n_small,
                                                args.min_score, seed=args.seed)
    to_kitti = lambda fid: f"{parse_frame_id(fid)[2]:06d}"
    outs[0].write_text("\n".join(map(to_kitti, big)) + "\n")
    outs[1].write_text("\n".join(map(to_kitti, small)) + "\n")
    report.update(detections_file=args.detections.name,
                  detections_sha256=hashlib.sha256(args.detections.read_bytes()).hexdigest(),
                  rule="rounds of cyclist / large vehicle / pedestrian / random frames from zero-shot detections; "
                       "the small set is the first frames of the large one")
    outs[2].write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
