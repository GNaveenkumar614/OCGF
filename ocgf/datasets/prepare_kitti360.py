"""Data-prep pipeline for one KITTI-360 sequence. The Kaggle notebook 01 runs these stages in order:

inputs -> velodyne source -> velodyne timestamps -> scans -> frame index -> report -> dev sample -> manifest

Every stage logs what it did, and the outputs land in cfg.out_dir:
  scans/<seq>/shard_*.zip, scan_stats.csv, velodyne_timestamps/<seq>.txt, frame_index/<seq>.csv,
  report/, dev_sample.zip, inputs_map.yaml, manifest.json
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..geometry.calib import kitti360_camera, kitti360_pose_from_lidar
from ..geometry.projection import project
from ..utils.provenance import write_manifest
from . import frame_index as fi
from .dataset_map import discover_kitti360, missing_required, write_dataset_map
from .ply import read_ply, write_ply
from .prep_report import make_figures, write_report
from .remote_zip import HTTPRangeFile, download, remote_size
from .supplement_io import ShardReader, ShardWriter, encode, format_info
from .velodyne import CropBox, read_bin, ring_index, ring_profile

GB = 1e9


@dataclass
class PrepConfig:
    sequence: str
    input_root: Path
    out_dir: Path
    scratch_dir: Path
    sources: Path                        # configs/data_sources/kitti360.yaml
    mode: str = "auto"                   # auto | download | remote: how to read the velodyne zip
    crop: CropBox = field(default_factory=CropBox)
    min_future_m: float = 50.0
    step_m: float = 5.0
    max_dt_ms: float = 20.0
    profile_every: int = 100             # ring profile of every n-th scan, for the report
    n_overlays: int = 10
    dev_frames: int = 50
    dev_margin_m: float = 60.0           # accumulated semantics kept around the dev-sample trajectory
    max_frames: int | None = None        # debugging: stop after this many scans

    def __post_init__(self):
        for name in ("input_root", "out_dir", "scratch_dir", "sources"):
            setattr(self, name, Path(getattr(self, name)))

    @property
    def scans_dir(self) -> Path:
        return self.out_dir / "scans" / self.sequence

    @property
    def velodyne_timestamps(self) -> Path:
        return self.out_dir / "velodyne_timestamps" / f"{self.sequence}.txt"

    @property
    def frame_index_csv(self) -> Path:
        return self.out_dir / "frame_index" / f"{self.sequence}.csv"

    def settings(self) -> dict:
        s = asdict(self)
        return {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in s.items()}


def _log(msg: str) -> None:
    print(msg, flush=True)


def dir_size(path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


# ---------------------------------------------------------------- inputs

def load_sources(path, sequence: str) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if sequence not in data.get("velodyne", {}):
        raise KeyError(f"{path} has no velodyne link for {sequence}")
    return {"velodyne": str(data["velodyne"][sequence]), "velodyne_timestamps": str(data["velodyne_timestamps"])}


def stage_inputs(cfg: PrepConfig, log=_log) -> dict[str, Path]:
    """Find every input component under cfg.input_root and stop early, with a clear message, if one is missing."""
    found = discover_kitti360(cfg.input_root, cfg.sequence)
    if "semantics2d" not in found and "semantics2d_zip" in found:
        found["semantics2d"] = extract_semantics2d(found["semantics2d_zip"], cfg.sequence, cfg.scratch_dir, log)
    for name in ("images", "image_timestamps", "calibration", "poses", "semantics3d", "boxes3d", "semantics2d"):
        log(f"{name:>17}: {found.get(name, 'NOT FOUND')}")
    missing = missing_required(found)
    if missing:
        raise FileNotFoundError(f"inputs not found under {cfg.input_root}: {', '.join(missing)}. Attach the KITTI-360 "
                                "starter and the kitti360-3d-bboxes dataset (Add Input), then run again.")
    if "semantics2d" not in found:
        log("warning: no 2D semantics found; has_sem2d will be false everywhere")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_map(cfg.out_dir / "inputs_map.yaml", cfg.sequence, found)
    return found


def extract_semantics2d(zip_path, sequence: str, scratch: Path, log=_log) -> Path:
    """Unpack only this sequence's image_00 folders from data_2d_semantics.zip. Returns the image_00 folder."""
    dest = Path(scratch) / "data_2d_semantics"
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if f"/{sequence}/image_00/" in f"/{m}" and not m.endswith("/")]
        log(f"extracting {len(members)} 2D-semantics files of {sequence} from {Path(zip_path).name}")
        for m in members:
            z.extract(m, dest)
    hits = list(dest.rglob(f"{sequence}/image_00"))
    if not hits:
        raise FileNotFoundError(f"{zip_path} has no {sequence}/image_00 folders")
    return hits[0]


# ---------------------------------------------------------------- velodyne zip

def open_zip(source: str, cfg: PrepConfig, log=_log) -> tuple[zipfile.ZipFile, str]:
    """Open a zip from a local path or a URL. For a URL, mode 'auto' downloads it to scratch if the disk has room
    (keeping room for the outputs when they share the disk) and otherwise reads it remotely."""
    if not source.startswith(("http://", "https://")):
        return zipfile.ZipFile(source), "local file"
    name = source.rsplit("/", 1)[-1]
    size = remote_size(source)
    mode = cfg.mode
    if mode == "auto":
        cfg.scratch_dir.mkdir(parents=True, exist_ok=True)
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(cfg.scratch_dir).free
        shared = os.stat(cfg.scratch_dir).st_dev == os.stat(cfg.out_dir).st_dev
        need = size + 2 * GB + (12 * GB if shared else 0)
        mode = "download" if free >= need else "remote"
        log(f"{name}: {size / GB:.2f} GB; scratch has {free / GB:.1f} GB free, needs {need / GB:.1f} GB -> {mode}")
    if mode == "download":
        step = max(size // 20, 1)
        marks = {"next": step}

        def progress(done, total):
            if done >= marks["next"] or done == total:
                log(f"  downloaded {done / GB:.2f} / {total / GB:.2f} GB")
                marks["next"] = done + step
        path = download(source, cfg.scratch_dir / name, progress=progress)
        return zipfile.ZipFile(path), f"downloaded to {path}"
    if mode == "remote":
        return zipfile.ZipFile(HTTPRangeFile(source)), "read remotely with HTTP range requests"
    raise ValueError(f"unknown mode {mode!r}: use auto, download or remote")


def stage_velodyne_timestamps(cfg: PrepConfig, velodyne_zip: zipfile.ZipFile, sources: dict, log=_log) -> Path:
    """This sequence's velodyne timestamps.txt, from the scan zip if it has one, else from the timestamps zip."""
    suffix = f"{cfg.sequence}/velodyne_points/timestamps.txt"
    for zf, label in ((velodyne_zip, "velodyne zip"), (None, "timestamps zip")):
        if zf is None:
            zf, _ = open_zip(sources["velodyne_timestamps"], cfg, log)
        member = next((m for m in zf.namelist() if m.endswith(suffix)), None)
        if member:
            cfg.velodyne_timestamps.parent.mkdir(parents=True, exist_ok=True)
            cfg.velodyne_timestamps.write_bytes(zf.read(member))
            log(f"velodyne timestamps from the {label}: {member}")
            return cfg.velodyne_timestamps
    raise FileNotFoundError(f"no {suffix} in the velodyne zip or the timestamps zip")


# ---------------------------------------------------------------- scans

def stage_scans(cfg: PrepConfig, velodyne_zip: zipfile.ZipFile, log=_log) -> tuple[dict, dict]:
    """Ring index on the raw scan, crop, encode, write shards. Returns per-scan stats and sampled ring profiles."""
    pattern = re.compile(rf"{re.escape(cfg.sequence)}/velodyne_points/data/(\d{{10}})\.bin$")
    members = [(int(m.group(1)), info) for info in velodyne_zip.infolist() if (m := pattern.search(info.filename))]
    if not members:
        raise FileNotFoundError(f"no {cfg.sequence} scans in the velodyne zip")
    members.sort(key=lambda fm: fm[0])
    if cfg.max_frames:
        members = members[: cfg.max_frames]
    members.sort(key=lambda fm: fm[1].header_offset)      # archive order: sequential reads
    if cfg.scans_dir.exists():
        shutil.rmtree(cfg.scans_dir)                       # a rerun starts clean: shards are append-only

    stats = {k: np.zeros(len(members), dtype=np.int64) for k in ("frame", "n_raw", "n_crop", "n_rings")}
    prof_frames, prof_counts, prof_elev = [], [], []
    t0 = time.time()
    with ShardWriter(cfg.scans_dir) as writer:
        for k, (frame, info) in enumerate(members):
            raw = read_bin(velodyne_zip.read(info))
            ring = ring_index(raw)                         # before any crop: the order carries the rings
            keep = cfg.crop.mask(raw)
            writer.write(frame, encode(raw[keep], ring[keep]))
            stats["frame"][k], stats["n_raw"][k], stats["n_crop"][k] = frame, len(raw), int(keep.sum())
            stats["n_rings"][k] = int(ring.max()) + 1 if len(ring) else 0
            if k % cfg.profile_every == 0:
                counts, elevation = ring_profile(raw, ring)
                prof_frames.append(frame)
                prof_counts.append(counts)
                prof_elev.append(elevation)
            if (k + 1) % 500 == 0 or k + 1 == len(members):
                rate = (k + 1) / (time.time() - t0)
                log(f"  {k + 1}/{len(members)} scans, {rate:.1f} scans/s, "
                    f"{(len(members) - k - 1) / rate / 60:.1f} min left")

    order = np.argsort(stats["frame"])
    stats = {k: v[order] for k, v in stats.items()}
    profiles = {"frame": np.array(prof_frames, dtype=np.int64),
                "counts": np.array(prof_counts).reshape(-1, 64), "elevation": np.array(prof_elev).reshape(-1, 64)}
    with open(cfg.out_dir / "scan_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(stats))
        w.writerows(zip(*stats.values()))
    report_dir = cfg.out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    np.savez(report_dir / "ring_profiles.npz", **profiles)
    log(f"scans written: {len(members)} frames, {dir_size(cfg.scans_dir) / GB:.2f} GB in {cfg.scans_dir}")
    return stats, profiles


# ---------------------------------------------------------------- frame index

def frames_in(folder, suffix: str = ".png") -> np.ndarray:
    folder = Path(folder)
    if not folder.is_dir():
        return np.zeros(0, dtype=np.int64)
    return np.array(sorted(int(e.name[:-len(suffix)]) for e in os.scandir(folder)
                           if e.name.endswith(suffix) and e.name[:-len(suffix)].isdigit()), dtype=np.int64)


def stage_frame_index(cfg: PrepConfig, found: dict, scan_stats: dict, log=_log) -> dict:
    pose_frames, T_world_pose = fi.read_poses(found["poses"] / "poses.txt")
    T_world_lidar = T_world_pose @ kitti360_pose_from_lidar(found["calibration"])
    sem2d = frames_in(found["semantics2d"] / "semantic") if "semantics2d" in found else np.zeros(0, np.int64)
    index = fi.build_frame_index(
        image_frames=frames_in(found["images"]), scan_frames=scan_stats["frame"],
        image_times=fi.read_timestamps(found["image_timestamps"]), scan_times=fi.read_timestamps(cfg.velodyne_timestamps),
        pose_frames=pose_frames, T_world_lidar=T_world_lidar,
        windows=fi.semantic_windows(p.name for p in (found["semantics3d"] / "static").glob("*.ply")),
        sem2d_frames=sem2d, scan_stats=scan_stats,
        min_future_m=cfg.min_future_m, step_m=cfg.step_m, max_dt_ms=cfg.max_dt_ms)
    fi.write_csv(cfg.frame_index_csv, index)
    log(f"frame index: {len(index['frame'])} frames, {int(index['usable'].sum())} usable, "
        f"{int(index['sample_5m'].sum())} sampled every {cfg.step_m} m -> {cfg.frame_index_csv}")
    return index


# ---------------------------------------------------------------- report

def stage_report(cfg: PrepConfig, found: dict, index: dict, scan_stats: dict, profiles: dict, log=_log) -> Path:
    cam = kitti360_camera(found["calibration"])
    usable = index["frame"][index["usable"] & index["has_scan"]]
    picks = usable[np.linspace(0, len(usable) - 1, min(cfg.n_overlays, len(usable))).astype(int)] if len(usable) else []
    reader, overlays = ShardReader(cfg.scans_dir), []
    from PIL import Image
    for frame in picks:
        image = np.asarray(Image.open(found["images"] / f"{frame:010d}.png").convert("RGB"))
        points = reader.read(int(frame))
        if len(points) > 30000:
            points = points[np.random.default_rng(0).choice(len(points), 30000, replace=False)]
        uv, depth, valid = project(points[:, :3], cam)
        overlays.append((int(frame), image, uv[valid], depth[valid]))
    reader.close()
    report_dir = cfg.out_dir / "report"
    figures = make_figures(report_dir, index, scan_stats, profiles, overlays)
    settings = cfg.settings()
    settings["crop"] = asdict(cfg.crop)
    write_report(report_dir / "report.md", cfg.sequence, index, scan_stats, profiles, figures, settings,
                 dir_size(cfg.scans_dir))
    log(f"report: {report_dir / 'report.md'} ({len(figures)} figures)")
    return report_dir / "report.md"


# ---------------------------------------------------------------- dev sample

def choose_dev_frames(index: dict, n: int) -> np.ndarray:
    """n consecutive usable frames that share one semantic window, from the run nearest the middle of the drive."""
    frame, usable, window = index["frame"], index["usable"], index["window_start"]
    runs, start = [], None                                  # runs of consecutive usable frames in one window
    for i in range(len(frame)):
        if start is not None and (not usable[i] or frame[i] != frame[i - 1] + 1 or window[i] != window[start]):
            runs.append((start, i))
            start = None
        if usable[i] and start is None:
            start = i
    if start is not None:
        runs.append((start, len(frame)))
    if not runs:
        raise ValueError("no usable frames to build a dev sample from")
    long_runs = [r for r in runs if r[1] - r[0] >= n] or [max(runs, key=lambda r: r[1] - r[0])]
    middle = np.median(np.flatnonzero(usable))
    a, b = min(long_runs, key=lambda r: abs((r[0] + r[1]) / 2 - middle))
    first = a + max(0, ((b - a) - n) // 2)
    return frame[first:min(first + n, b)]


def _crop_cloud(src: Path, dst: Path, xy_min, xy_max) -> int:
    cloud = read_ply(src)
    keep = (cloud["x"] >= xy_min[0]) & (cloud["x"] <= xy_max[0]) & (cloud["y"] >= xy_min[1]) & (cloud["y"] <= xy_max[1])
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_ply(dst, cloud[keep], comments=[f"cropped from {src.name} for the OCGF dev sample"])
    return int(keep.sum())


def stage_dev_sample(cfg: PrepConfig, found: dict, index: dict, log=_log) -> Path:
    """dev_sample.zip: a small copy of every component for consecutive frames, laid out like KITTI-360."""
    seq = cfg.sequence
    frames = choose_dev_frames(index, cfg.dev_frames)
    dev = cfg.scratch_dir / "dev_sample"
    if dev.exists():
        shutil.rmtree(dev)
    k360 = dev / "KITTI360"

    calib = k360 / "calibration"
    calib.mkdir(parents=True)
    for name in ("perspective.txt", "calib_cam_to_velo.txt", "calib_cam_to_pose.txt"):
        if (found["calibration"] / name).exists():
            shutil.copy2(found["calibration"] / name, calib / name)

    img_dir = k360 / "data_2d_raw" / seq / "image_00"
    (img_dir / "data_rect").mkdir(parents=True)
    shutil.copy2(found["image_timestamps"], img_dir / "timestamps.txt")
    for f in frames:
        shutil.copy2(found["images"] / f"{f:010d}.png", img_dir / "data_rect" / f"{f:010d}.png")

    pose_dir = k360 / "data_poses" / seq
    pose_dir.mkdir(parents=True)
    for name in ("poses.txt", "cam0_to_world.txt"):
        if (found["poses"] / name).exists():
            shutil.copy2(found["poses"] / name, pose_dir / name)

    rows = np.isin(index["frame"], frames)
    x, y = index["x_world"][rows], index["y_world"][rows]
    lo, hi = (x.min() - cfg.dev_margin_m, y.min() - cfg.dev_margin_m), (x.max() + cfg.dev_margin_m, y.max() + cfg.dev_margin_m)
    window = f"{index['window_start'][rows][0]:010d}_{index['window_end'][rows][0]:010d}.ply"
    for part in ("static", "dynamic"):
        src = found["semantics3d"] / part / window
        if src.exists():
            kept = _crop_cloud(src, k360 / "data_3d_semantics" / seq / part / window, lo, hi)
            log(f"dev sample: {part} cloud {window}, {kept} points kept")

    if "semantics2d" in found:
        sem_dst = k360 / "data_2d_semantics" / "train" / seq / "image_00"
        for sub in ("semantic", "semantic_rgb", "instance", "confidence"):
            for f in frames:
                src = found["semantics2d"] / sub / f"{f:010d}.png"
                if src.exists():
                    (sem_dst / sub).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, sem_dst / sub / src.name)

    box_dir = k360 / "data_3d_bboxes" / "train"
    box_dir.mkdir(parents=True)
    shutil.copy2(found["boxes3d"], box_dir / f"{seq}.xml")

    reader = ShardReader(cfg.scans_dir)
    with ShardWriter(dev / "scans" / seq) as writer:
        for f in frames:
            writer.write(int(f), reader.read_record(int(f)))
    reader.close()
    (dev / "velodyne_timestamps").mkdir()
    shutil.copy2(cfg.velodyne_timestamps, dev / "velodyne_timestamps" / f"{seq}.txt")
    fi.write_csv(dev / "frame_index" / f"{seq}.csv", fi.select(index, rows))

    components = discover_kitti360(k360, seq)
    components.update(scans=dev / "scans" / seq, velodyne_timestamps=dev / "velodyne_timestamps" / f"{seq}.txt",
                      frame_index=dev / "frame_index" / f"{seq}.csv")
    write_dataset_map(dev / "dataset_map.yaml", seq, components)

    out = cfg.out_dir / "dev_sample.zip"
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dev.rglob("*")):
            if p.is_file():
                z.write(p, Path("dev_sample") / p.relative_to(dev))
    log(f"dev sample: frames {frames[0]}-{frames[-1]} ({len(frames)}), {out.stat().st_size / 1e6:.1f} MB -> {out}")
    return out


# ---------------------------------------------------------------- whole run

def run_all(cfg: PrepConfig, log=_log) -> dict:
    timings = {}

    def timed(name, fn, *args):
        t = time.time()
        result = fn(*args)
        timings[name] = round(time.time() - t, 1)
        return result

    sources = load_sources(cfg.sources, cfg.sequence)
    found = timed("inputs", stage_inputs, cfg, log)
    velodyne_zip, how = timed("open_velodyne", open_zip, sources["velodyne"], cfg, log)
    log(f"velodyne zip: {how}")
    timed("velodyne_timestamps", stage_velodyne_timestamps, cfg, velodyne_zip, sources, log)
    scan_stats, profiles = timed("scans", stage_scans, cfg, velodyne_zip, log)
    index = timed("frame_index", stage_frame_index, cfg, found, scan_stats, log)
    timed("report", stage_report, cfg, found, index, scan_stats, profiles, log)
    timed("dev_sample", stage_dev_sample, cfg, found, index, log)
    total = stage_manifest(cfg, found, index, sources, how, timings, log)
    return {"found": found, "index": index, "scan_stats": scan_stats, "timings": timings, "output_bytes": total}


def stage_manifest(cfg: PrepConfig, found: dict, index: dict, sources: dict, how: str, timings: dict | None = None,
                   log=_log) -> int:
    """manifest.json with code version, settings, inputs, counts and timings. Returns the output size in bytes."""
    total = dir_size(cfg.out_dir)
    counts = {k: int(index[k].sum()) for k in ("has_image", "has_scan", "has_pose", "aligned", "usable", "sample_5m")}
    write_manifest(cfg.out_dir / "manifest.json", sequence=cfg.sequence, settings=cfg.settings(),
                   sources={k: v.rsplit("/", 1)[-1] for k, v in sources.items()}, velodyne_read=how,
                   inputs={k: str(v) for k, v in found.items()}, scan_format=format_info(), counts=counts,
                   timings_s=timings or {}, output_bytes=total)
    log(f"done: {total / GB:.2f} GB in {cfg.out_dir}"
        + ("  WARNING: above Kaggle's 20 GB output limit" if total > 19.5 * GB else ""))
    return total
