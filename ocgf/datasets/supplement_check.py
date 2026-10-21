"""Check a prepared drive (the output of data-prep notebook 01) against its own manifest.

Used after copying an output anywhere (a Kaggle dataset, a local backup): the stored frames must match what the run
recorded, and every shard must open.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from . import frame_index as fi
from .supplement_io import ShardReader


def find_root(folder) -> Path | None:
    """The folder holding manifest.json, wherever the copy nested it (e.g. under kitti360_supplement/)."""
    hits = sorted(Path(folder).rglob("manifest.json"), key=lambda p: len(p.parts))
    return hits[0].parent if hits else None


def check_supplement(root, sequence: str | None = None, deep: bool = False) -> tuple[dict, list[str]]:
    """(summary, problems) for one prepared drive. deep=True also verifies every member's CRC (reads all bytes)."""
    root = Path(root)
    problems = []
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}, [f"no manifest.json in {root}"]
    manifest = json.loads(manifest_path.read_text())
    seq = manifest.get("sequence")
    if sequence and seq != sequence:
        problems.append(f"manifest is for {seq}, expected {sequence}")
    counts = manifest.get("counts", {})
    expected = counts.get("kept", counts.get("has_scan"))      # drive 0000's first run kept every scan, no 'kept'

    scans = ShardReader(root / "scans" / seq) if (root / "scans" / seq).is_dir() else None
    n_scans = len(scans.frames()) if scans else 0
    if n_scans != expected:
        problems.append(f"{n_scans} scans stored, manifest says {expected}")
    n_images = None
    if (root / "images" / seq).is_dir():
        n_images = len(ShardReader(root / "images" / seq).frames(".png"))
        if n_images != expected:
            problems.append(f"{n_images} images stored, manifest says {expected}")

    index_path = root / "frame_index" / f"{seq}.csv"
    if index_path.exists():
        index = fi.read_csv(index_path)
        kept = int(index["kept"].sum()) if "kept" in index else int(index["has_scan"].sum())
        if kept != expected:
            problems.append(f"frame index keeps {kept} frames, manifest says {expected}")
    else:
        problems.append(f"no frame index at {index_path}")
    if not (root / "report" / "report.md").exists():
        problems.append("no report/report.md")

    if deep:
        for shard in sorted(f for f in root.rglob("shard_*.zip") if f.is_file()):   # unpacked folders hold no zip
            try:
                with zipfile.ZipFile(shard) as z:
                    bad = z.testzip()
                if bad:
                    problems.append(f"{shard.relative_to(root)}: corrupt member {bad}")
            except zipfile.BadZipFile as err:
                problems.append(f"{shard.relative_to(root)}: {err}")

    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    summary = {"sequence": seq, "role": manifest.get("role", "dev"), "kept": expected, "scans": n_scans,
               "images": n_images, "gb": round(size / 1e9, 2), "code": manifest.get("code", {}).get("commit")}
    return summary, problems
