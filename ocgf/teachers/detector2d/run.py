"""Run the detector over frame sources and write the detection files.

Per source, under the output folder:  <dataset>/<sequence>.txt      our classes   'fid cls score l t r b'
                                      <dataset>/<sequence>_raw.txt  COCO classes  (same format, for audits/other mappings)
Sources:
  - KITTI 3D Object: all 7,481 'training' images (the train/val split is applied by whoever reads the file);
  - KITTI-360 paper drives: usable frames every 2.5 m of travel. The new drives store exactly those; drive 0000 keeps
    every frame, so it is thinned here to match.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ...datasets.frame_index import sample_every
from ...datasets.ids import frame_id
from .mapping import OUR_CLASSES, format_lines, to_our_classes


def kitti_object_items(image_dir) -> list[tuple[str, Path]]:
    """(fid, png path) for every KITTI 3D Object training image."""
    return [(frame_id("kitti", "training", int(p.stem)), p) for p in sorted(Path(image_dir).glob("*.png"))]


def kitti360_frames(seq, every_m: float = 2.5) -> np.ndarray:
    """Usable frames of a prepared drive, one every every_m metres among the stored ones."""
    index = seq.index
    stored = index["usable"] & index.get("kept", np.ones(len(index["frame"]), dtype=bool))
    return index["frame"][sample_every(index["odometer_m"], stored, every_m)]


def prefetched_batches(items, load, batch_size: int, workers: int = 4):
    """Yield (items, loaded) batches, loading the next batch in background threads while the current one runs."""
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    with ThreadPoolExecutor(workers) as pool:
        future = pool.map(load, batches[0]) if batches else None
        for k, batch in enumerate(batches):
            loaded = list(future)
            future = pool.map(load, batches[k + 1]) if k + 1 < len(batches) else None
            yield batch, loaded


def run_source(detector, items, load, out_dir, dataset: str, sequence: str, batch_size: int = 8,
               log=print) -> dict:
    """Detect on items [(fid, ref)] with load(ref) -> HxWx3 uint8; write both files; return per-class counts."""
    out_dir = Path(out_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {c: 0 for c in OUR_CLASSES}
    t0, done = time.time(), 0
    with open(out_dir / f"{sequence}.txt", "w") as ours, open(out_dir / f"{sequence}_raw.txt", "w") as raw:
        for batch, images in prefetched_batches(items, lambda it: load(it[1]), batch_size):
            for (fid, _), dets in zip(batch, detector(images)):
                raw.writelines(line + "\n" for line in format_lines(fid, dets))
                mapped = to_our_classes(dets)
                ours.writelines(line + "\n" for line in format_lines(fid, mapped))
                for label in mapped.labels:
                    counts[label] += 1
            done += len(batch)
            if done % (50 * batch_size) < batch_size or done == len(items):
                rate = done / (time.time() - t0)
                log(f"  {dataset}/{sequence}: {done}/{len(items)} images, {rate:.1f} img/s, "
                    f"{(len(items) - done) / rate / 60:.1f} min left")
    return {"images": len(items), "detections": counts, "seconds": round(time.time() - t0, 1)}


def load_png(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))
