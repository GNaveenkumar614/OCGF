"""E3 ORACLE inputs from the KITTI-360 3D boxes (ground truth) for a prepared drive. Upper bounds, never the main result.

    python scripts/make_oracle_objects.py --input-root /kaggle/input --sequence 2013_05_28_drive_0000_sync \
        --out /kaggle/working/objects_v1

Writes
  <out>/oracle2d/<sequence>.txt            ground-truth 2D boxes in the detector's format (score 1.0): MTrans on them
                                           gives the oracle-2D arm (scripts/run_mtrans.py --box-source oracle2d)
  <out>/oracle3d/<sequence>.jsonl          the ground-truth 3D boxes as object rows (conf 1.0), with
  <out>/oracle3d/<sequence>.manifest.json  a manifest naming them an oracle (teacher "oracle_gt3d"): the GT-3D arm
Objects: the visible ground-truth objects of our classes (ocgf.evaluation.teacher_quality.visible_objects; a cyclist
is a rider with its two-wheeler), on the frames the detector runs on. The oracle-2D file sits in a folder not named
kitti360/, so it is never mistaken for the detector's output.
"""
import argparse
import json
from pathlib import Path

from ocgf.datasets.kitti360 import Kitti360Sequence
from ocgf.datasets.locate import describe_prepared, prepared_drive
from ocgf.evaluation.teacher_quality import visible_objects
from ocgf.teachers.detector2d.run import kitti360_frames
from ocgf.utils.provenance import write_manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-map", type=Path)
    src.add_argument("--input-root", type=Path)
    p.add_argument("--sequence")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every-m", type=float, default=2.5, help="frames as the detector picks them")
    p.add_argument("--min-points", type=int, default=1, help="LiDAR points an object needs to count as visible")
    args = p.parse_args(argv)

    if args.dataset_map:
        seq = Kitti360Sequence(args.dataset_map)
    else:
        seq = Kitti360Sequence(sequence=args.sequence, components=prepared_drive(args.input_root, args.sequence))
        print(describe_prepared(args.input_root, args.sequence), flush=True)
    if "boxes3d" not in seq.paths:
        raise SystemExit("no KITTI-360 3D boxes for this drive: attach data-3d-bboxes")
    frames = [int(f) for f in kitti360_frames(seq, args.every_m)]
    det_lines, rows, counts = [], [], {}
    for f in frames:
        fid = seq.fid(f)
        for g in visible_objects(seq.gt_boxes(f), seq.camera, seq.points(f), min_points=args.min_points):
            l, t, r, b = (float(v) for v in g.box_2d)
            det_lines.append(f"{fid} {g.cls} 1.0000 {l:.2f} {t:.2f} {r:.2f} {b:.2f}")
            rows.append({"fid": fid, "cls": g.cls, "score_2d": 1.0, "box_2d": [round(v, 2) for v in (l, t, r, b)],
                         "n_points": g.n_points, "box": [round(float(v), 4) for v in g.box], "conf_iou_raw": 1.0,
                         "status": "ok"})
            counts[g.cls] = counts.get(g.cls, 0) + 1
    (args.out / "oracle2d").mkdir(parents=True, exist_ok=True)
    (args.out / "oracle2d" / f"{seq.sequence}.txt").write_text("\n".join(det_lines) + ("\n" if det_lines else ""))
    (args.out / "oracle3d").mkdir(parents=True, exist_ok=True)
    (args.out / "oracle3d" / f"{seq.sequence}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_manifest(args.out / "oracle3d" / f"{seq.sequence}.manifest.json", teacher="oracle_gt3d",
                   note="ORACLE: KITTI-360 ground-truth boxes as the object input (E3 upper bound)",
                   sequence=seq.sequence, frames=len(frames), objects=counts, min_points=args.min_points,
                   boxes3d=str(seq.paths["boxes3d"]))
    print(f"{seq.sequence}: {len(frames)} frames, oracle objects {counts} -> {args.out}")


if __name__ == "__main__":
    main()
