"""Freeze a scene-level split from a frame index (frame ids only, no data) into configs/splits/.

    python scripts/make_splits.py --frame-index data/kitti360_d0000/frame_index/2013_05_28_drive_0000_sync.csv \
        --name kitti360_d0000_dev_v1

A split file is never rewritten: change the name (v2, ...) for a new split. tests/test_splits.py pins the hash.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from ocgf.datasets import frame_index as fi
from ocgf.datasets.splits import content_hash, distance_split, to_ranges

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame-index", type=Path, required=True)
    p.add_argument("--name", required=True, help="split name, e.g. kitti360_d0000_dev_v1")
    p.add_argument("--fractions", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    p.add_argument("--buffer-m", type=float, default=100.0)
    p.add_argument("--revisit-m", type=float, default=50.0)
    args = p.parse_args(argv)

    out = REPO / "configs" / "splits" / f"{args.name}.json"
    if out.exists():
        raise SystemExit(f"{out} exists. Splits are frozen: pick a new name instead of overwriting.")
    index = fi.read_csv(args.frame_index)
    splits, report = distance_split(index, tuple(args.fractions), buffer_m=args.buffer_m, revisit_m=args.revisit_m)
    record = {
        "name": args.name,
        "dataset": "kitti360",
        "sequence": args.frame_index.stem,
        "purpose": "starter phase: debugging on one drive only, never for paper results (the paper splits by drive)",
        "method": "consecutive train | val | test stretches by travelled distance over usable frames; "
                  "frames near a cut and val/test revisits of train areas dropped",
        "params": {"fractions": list(args.fractions), "buffer_m": args.buffer_m, "revisit_m": args.revisit_m},
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "frame_index_sha256": hashlib.sha256(args.frame_index.read_bytes()).hexdigest(),
        "report": report,
        "content_sha256": content_hash(splits),
        "splits": {n: to_ranges(f) for n, f in splits.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("name", "content_sha256", "report")}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
