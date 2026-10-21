"""Copy input files to local disk with many parallel readers.

On Kaggle, files under /kaggle/input are slow to read one after another (about 5 s per KITTI frame on 2026-10-20) but
parallel reads add up, so notebooks copy what they need to /kaggle/temp once and then read local files.
"""
from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parallel_copy(jobs, threads: int = 32, log=print, every: int = 2000) -> int:
    """jobs: (source, target) pairs. Existing targets are skipped, so an interrupted copy resumes. Returns bytes copied."""
    jobs = [(Path(s), Path(t)) for s, t in jobs if not Path(t).exists()]

    def copy(job):
        src, target = job
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        shutil.copyfile(src, tmp)
        tmp.replace(target)
        return target.stat().st_size

    t0, done, size = time.time(), 0, 0
    with ThreadPoolExecutor(max(1, threads)) as pool:
        for n in pool.map(copy, jobs):
            done, size = done + 1, size + n
            if log and (done % every == 0 or done == len(jobs)):
                rate = size / 1e6 / max(time.time() - t0, 1e-6)
                log(f"  staged {done}/{len(jobs)} files, {size / 1e9:.2f} GB, {rate:.1f} MB/s")
    return size
