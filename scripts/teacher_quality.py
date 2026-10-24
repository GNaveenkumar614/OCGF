"""Teacher quality on KITTI-360 (gate G4): the 2D detector's recall and MTrans's 3D boxes against the KITTI-360 boxes.

    python scripts/teacher_quality.py --input-root /kaggle/input --sequence 2013_05_28_drive_0000_sync \
        --objects /kaggle/working/objects_v1 --out /kaggle/working/objects_v1/quality/2013_05_28_drive_0000_sync.json

Frames: the drive's frames sampled every 5 m (the student's). Detections: the detector's kitti360/<sequence>.txt
(found under --detections, default --input-root); objects: the MTrans object file on detector boxes (under --objects).
Ground truth (evaluation only): the KITTI-360 3D boxes; see ocgf/evaluation/teacher_quality.py for the definitions.
"""
import argparse
import json
from pathlib import Path

from ocgf.datasets.kitti360 import Kitti360Sequence
from ocgf.datasets.locate import describe_prepared, find_detections, find_object_file, prepared_drive
from ocgf.datasets.objects import read_object_file
from ocgf.evaluation.teacher_quality import BoxQuality, DetectorRecall, read_detection_file, visible_objects
from ocgf.utils.provenance import write_manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-map", type=Path)
    src.add_argument("--input-root", type=Path)
    p.add_argument("--sequence")
    p.add_argument("--detections", type=Path, help="the detector file, or a folder to search (default: --input-root)")
    p.add_argument("--objects", type=Path, help="where to find the MTrans object file (detector boxes)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-points", type=int, default=5, help="LiDAR points an object needs to count as visible")
    args = p.parse_args(argv)

    if args.dataset_map:
        seq = Kitti360Sequence(args.dataset_map)
    else:
        seq = Kitti360Sequence(sequence=args.sequence, components=prepared_drive(args.input_root, args.sequence))
        print(describe_prepared(args.input_root, args.sequence), flush=True)
    if "boxes3d" not in seq.paths:
        raise SystemExit("no KITTI-360 3D boxes for this drive: attach data-3d-bboxes")
    det_path = args.detections or args.input_root
    if det_path is not None and det_path.is_dir():
        det_path = find_detections(det_path, seq.sequence)
    dets = read_detection_file(det_path) if det_path else None
    obj_path = find_object_file(args.objects, seq.sequence, "detector") if args.objects else None
    objects = read_object_file(obj_path) if obj_path else None
    if dets is None and objects is None:
        raise SystemExit("neither a detector file nor an MTrans object file was found")
    print(f"detections: {det_path}; objects: {obj_path}", flush=True)

    frames = [int(f) for f in seq.usable_frames(sampled=True)]
    recall, quality, n_gt = DetectorRecall(), BoxQuality(), 0
    for f in frames:
        gts = visible_objects(seq.gt_boxes(f), seq.camera, seq.points(f), min_points=args.min_points)
        n_gt += len(gts)
        if dets is not None:
            recall.add(gts, dets.get(seq.fid(f)))
        if objects is not None:
            quality.add(gts, objects.get(f))
    result = {"frames": len(frames), "visible_objects": n_gt}
    if dets is not None:
        result["detector_recall"] = recall.result()
    if objects is not None:
        result["mtrans_boxes"] = quality.result()
    write_manifest(args.out, kind="teacher_quality", sequence=seq.sequence, detections=str(det_path),
                   objects=str(obj_path), min_points=args.min_points, **result)

    fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
    print(f"\n{seq.sequence}: {len(frames)} frames, {n_gt} visible ground-truth objects")
    if dets is not None:
        print("detector recall (2D IoU >= 0.5, any class / its class), all distances:")
        for cls, rows in result["detector_recall"].items():
            a = rows["all"]
            print(f"  {cls:14s} {a['objects']:6d} objects  @0.1: {fmt(a['recall_any@0.1'])} / {fmt(a['recall_class@0.1'])}"
                  f"   @0.5: {fmt(a['recall_any@0.5'])} / {fmt(a['recall_class@0.5'])}")
    if objects is not None:
        print("MTrans boxes on detector boxes (matched objects): mean 3D IoU, recall at 3D IoU 0.5 / 0.7, |conf - IoU|:")
        for cls, rows in result["mtrans_boxes"].items():
            a = rows["all"]
            print(f"  {cls:14s} {a['objects']:6d} objects, {a['with_3d_box']:6d} with a box: {fmt(a['mean_iou3d'])}  "
                  f"{fmt(a['recall3d@0.5'])} / {fmt(a['recall3d@0.7'])}  {fmt(a['conf_abs_err'])}")


if __name__ == "__main__":
    main()
