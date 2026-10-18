"""Data-prep pipeline for one KITTI-360 sequence. The Kaggle notebook 01 runs these stages in order:

inputs -> sources (velodyne zip, images, timestamps) -> frame selection -> scans -> images -> frame index
-> report -> dev sample -> manifest

Paper drives (split kitti360_paper_v1, decided 2026-10-18 from scripts/drive_stats.py):
  train 0000, 0002, 0004, 0005 · val 0006 · test 0009, 0010 · unused 0003, 0007 (fast roads, few near-obstacle cases)
  - drive 0000 (prepared first, role 'dev'): images from the Kaggle starter, every frame kept;
  - the other drives: images from the official image_00 zip, one usable frame kept every 2.5 m of travel
    (keep_every_m=2.5). Only those frames are read, so a remote zip costs about a quarter of its size;
  - the semantic-window rule applies to val/test (and dev) drives only, see frame_index.py.

Outputs in cfg.out_dir:
  scans/<seq>/shard_*.zip, images/<seq>/shard_*.zip (zip-sourced images only), scan_stats.csv,
  velodyne_timestamps/<seq>.txt, image_timestamps/<seq>.txt (if extracted), frame_index/<seq>.csv, report/,
  dev_sample.zip (if dev_frames), dataset_map.yaml, inputs_map.yaml, manifest.json
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ..geometry.calib import kitti360_camera, kitti360_pose_from_lidar
from ..geometry.projection import project
from ..utils.provenance import write_manifest
from . import frame_index as fi
from .dataset_map import describe_tree, discover_kitti360, missing_required, write_dataset_map
from .ply import read_ply, write_ply
from .prep_report import make_figures, write_report
from .remote_zip import HTTPRangeFile, download, remote_size
from .supplement_io import ShardReader, ShardWriter, encode, format_info
from .velodyne import CropBox, read_bin, ring_index, ring_profile

GB = 1e9
ROLES = ("dev", "train", "val", "test")


@dataclass
class PrepConfig:
    sequence: str
    input_root: Path
    out_dir: Path
    scratch_dir: Path
    sources: Path                        # configs/data_sources/kitti360.yaml
    role: str = "dev"                    # dev (drive 0000 starter phase) | train | val | test (kitti360_paper_v1)
    keep_every_m: float | None = None    # None: keep every frame (drive 0000); 2.5 for the other paper drives
    mode: str = "auto"                   # auto | download | remote: how to read a remote zip
    crop: CropBox = field(default_factory=CropBox)
    min_future_m: float = 50.0
    step_m: float = 5.0
    max_dt_ms: float = 20.0
    profile_every: int = 100             # ring profile of every n-th scan, for the report
    n_overlays: int = 10
    dev_frames: int = 50                 # 0: no dev sample (only drive 0000 needs one)
    dev_margin_m: float = 60.0           # accumulated semantics kept around the dev-sample trajectory
    max_frames: int | None = None        # debugging: stop after this many scans

    def __post_init__(self):
        for name in ("input_root", "out_dir", "scratch_dir", "sources"):
            setattr(self, name, Path(getattr(self, name)))
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")

    @property
    def require_window(self) -> bool:
        """Evaluation ground truth exists only inside semantic windows, so only train drives may leave them."""
        return self.role != "train"

    @property
    def scans_dir(self) -> Path:
        return self.out_dir / "scans" / self.sequence

    @property
    def images_dir(self) -> Path:
        return self.out_dir / "images" / self.sequence

    @property
    def velodyne_timestamps(self) -> Path:
        return self.out_dir / "velodyne_timestamps" / f"{self.sequence}.txt"

    @property
    def image_timestamps(self) -> Path:
        return self.out_dir / "image_timestamps" / f"{self.sequence}.txt"

    @property
    def frame_index_csv(self) -> Path:
        return self.out_dir / "frame_index" / f"{self.sequence}.csv"

    def settings(self) -> dict:
        s = asdict(self)
        s["require_window"] = self.require_window
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
    out = {"velodyne": str(data["velodyne"][sequence]), "velodyne_timestamps": str(data["velodyne_timestamps"])}
    if sequence in data.get("images_00", {}):
        out["images_00"] = str(data["images_00"][sequence])
    if "image_timestamps" in data:
        out["image_timestamps"] = str(data["image_timestamps"])
    return out


def required_inputs(cfg: PrepConfig) -> tuple[str, ...]:
    """Images may come from the official zip; semantic windows are needed only where the window rule applies."""
    return ("calibration", "poses", "boxes3d") + (("semantics3d",) if cfg.require_window else ())


def stage_inputs(cfg: PrepConfig, log=_log) -> dict[str, Path]:
    """Find every input component under cfg.input_root and stop early, with a clear message, if one is missing."""
    found = discover_kitti360(cfg.input_root, cfg.sequence)
    if "semantics2d" not in found and "semantics2d_zip" in found:
        found["semantics2d"] = extract_semantics2d(found["semantics2d_zip"], cfg.sequence, cfg.scratch_dir, log)
    for name in ("images", "image_timestamps", "calibration", "poses", "semantics3d", "boxes3d", "semantics2d"):
        log(f"{name:>17}: {found.get(name, 'NOT FOUND')}")
    missing = missing_required(found, required_inputs(cfg))
    if missing:
        log(f"\nWhat is under {cfg.input_root} (folders, file counts, and the files the search looks for):")
        for line in describe_tree(cfg.input_root):
            log(line)
        raise FileNotFoundError(f"inputs not found under {cfg.input_root}: {', '.join(missing)}. Check that the KITTI-360 "
                                "starter (calibration), the poses, the kitti360-3d-bboxes dataset and, for val/test "
                                "drives, the accumulated 3D semantics are attached (Add Input); if they are, the tree "
                                "above shows how their layout differs from what the search expects.")
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


# ---------------------------------------------------------------- sources

def open_zip(source: str, cfg: PrepConfig, log=_log, read_share: float = 1.0) -> tuple[zipfile.ZipFile, str]:
    """Open a zip from a local path or a URL. For a URL, mode 'auto' downloads it to scratch if the disk has room
    (keeping room for the outputs when they share the disk) and otherwise reads it remotely. With read_share < 1 (only
    some members will be read, e.g. frames every 2.5 m) 'auto' reads remotely: downloading everything would be waste."""
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
        mode = "download" if read_share >= 0.9 and free >= need else "remote"
        log(f"{name}: {size / GB:.2f} GB; scratch has {free / GB:.1f} GB free, needs {need / GB:.1f} GB to download; "
            f"about {read_share:.0%} of it will be read -> {mode}")
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


def zip_frames(zf: zipfile.ZipFile, pattern: str) -> dict[int, zipfile.ZipInfo]:
    """{frame: member} for members whose name matches pattern (with one (\\d{10}) group for the frame)."""
    rx = re.compile(pattern)
    return {int(m.group(1)): info for info in zf.infolist() if (m := rx.search(info.filename))}


def extract_member(cfg: PrepConfig, zips: list, suffix: str, dest: Path, log=_log) -> Path:
    """Copy the first member ending with suffix from the first zip that has one. zips: ZipFile or source string."""
    for z in zips:
        zf = z if isinstance(z, zipfile.ZipFile) else open_zip(z, cfg, log)[0]
        member = next((m for m in zf.namelist() if m.endswith(suffix)), None)
        if member:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))
            log(f"{suffix} <- {member}")
            return dest
    raise FileNotFoundError(f"no member ending with {suffix} in the given zips")


class ImageSource:
    """PNG images of one sequence: a folder (the Kaggle starter, drive 0000) or the official image_00 zip."""

    def __init__(self, sequence: str, folder: Path | None = None, zf: zipfile.ZipFile | None = None):
        self.folder, self.zf = folder, zf
        if folder is not None:
            self._members = None
            self._frames = sorted(int(p.stem) for p in Path(folder).glob("*.png") if p.stem.isdigit())
        else:
            self._members = zip_frames(zf, rf"{re.escape(sequence)}/image_00/data_rect/(\d{{10}})\.png$")
            self._frames = sorted(self._members)

    @property
    def kind(self) -> str:
        return "folder" if self.folder is not None else "zip"

    def frames(self) -> np.ndarray:
        return np.array(self._frames, dtype=np.int64)

    def read_bytes(self, frame: int) -> bytes:
        if self.folder is not None:
            return (Path(self.folder) / f"{frame:010d}.png").read_bytes()
        return self.zf.read(self._members[frame])

    def read(self, frame: int) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(self.read_bytes(frame))).convert("RGB"))

    def ordered(self, frames) -> list[int]:
        """Frames in archive order, so remote reads stay sequential."""
        if self._members is None:
            return sorted(int(f) for f in frames)
        return sorted((int(f) for f in frames), key=lambda f: self._members[f].header_offset)


def stage_sources(cfg: PrepConfig, found: dict, sources: dict, log=_log) -> dict:
    """Open the velodyne zip and the image source, and extract both timestamp files. Returns the open sources."""
    read_share = 0.3 if cfg.keep_every_m else 1.0
    velodyne_zip, how = open_zip(sources["velodyne"], cfg, log, read_share)
    log(f"velodyne zip: {how}")
    extract_member(cfg, [velodyne_zip, sources["velodyne_timestamps"]],
                   f"{cfg.sequence}/velodyne_points/timestamps.txt", cfg.velodyne_timestamps, log)
    if "images" in found:
        images = ImageSource(cfg.sequence, folder=found["images"])
        log(f"images: {len(images.frames())} PNGs in {found['images']}")
    else:
        if "images_00" not in sources:
            raise FileNotFoundError(f"no images for {cfg.sequence}: not in the inputs and no images_00 link in {cfg.sources}")
        image_zip, how = open_zip(sources["images_00"], cfg, log, read_share)
        images = ImageSource(cfg.sequence, zf=image_zip)
        log(f"images: {len(images.frames())} PNGs in the image_00 zip, {how}")
    if "image_timestamps" not in found:
        zips = [images.zf] if images.zf is not None else []
        found["image_timestamps"] = extract_member(cfg, zips + [sources["image_timestamps"]],
                                                   f"{cfg.sequence}/image_00/timestamps.txt", cfg.image_timestamps, log)
    scans = zip_frames(velodyne_zip, rf"{re.escape(cfg.sequence)}/velodyne_points/data/(\d{{10}})\.bin$")
    if not scans:
        raise FileNotFoundError(f"no {cfg.sequence} scans in the velodyne zip")
    return {"velodyne_zip": velodyne_zip, "velodyne_read": how, "scans": scans, "images": images}


# ---------------------------------------------------------------- frame selection and index

def frames_in(folder, suffix: str = ".png") -> np.ndarray:
    folder = Path(folder)
    if not folder.is_dir():
        return np.zeros(0, dtype=np.int64)
    return np.array(sorted(int(e.name[:-len(suffix)]) for e in os.scandir(folder)
                           if e.name.endswith(suffix) and e.name[:-len(suffix)].isdigit()), dtype=np.int64)


def stage_select(cfg: PrepConfig, found: dict, src: dict, log=_log) -> dict:
    """Frame index before any scan is read: usable frames and the kept ones (every frame, or one per keep_every_m)."""
    pose_frames, T_world_pose = fi.read_poses(found["poses"] / "poses.txt")
    T_world_lidar = T_world_pose @ kitti360_pose_from_lidar(found["calibration"])
    sem2d = frames_in(found["semantics2d"] / "semantic") if "semantics2d" in found else np.zeros(0, np.int64)
    windows = (fi.semantic_windows(p.name for p in (found["semantics3d"] / "static").glob("*.ply"))
               if "semantics3d" in found else np.zeros((0, 2), np.int64))
    index = fi.build_frame_index(
        image_frames=src["images"].frames(), scan_frames=np.array(sorted(src["scans"]), dtype=np.int64),
        image_times=fi.read_timestamps(found["image_timestamps"]), scan_times=fi.read_timestamps(cfg.velodyne_timestamps),
        pose_frames=pose_frames, T_world_lidar=T_world_lidar, windows=windows, sem2d_frames=sem2d,
        min_future_m=cfg.min_future_m, step_m=cfg.step_m, max_dt_ms=cfg.max_dt_ms,
        require_window=cfg.require_window, keep_every_m=cfg.keep_every_m)
    if cfg.max_frames:
        keep = np.flatnonzero(index["kept"])[: cfg.max_frames]
        index["kept"][:] = False
        index["kept"][keep] = True
        index["sample_5m"] &= index["kept"]
    log(f"frame selection ({cfg.role}): {len(index['frame'])} frames, {int(index['usable'].sum())} usable, "
        f"{int(index['kept'].sum())} kept, {int(index['sample_5m'].sum())} sampled every {cfg.step_m} m")
    return index


def stage_scans(cfg: PrepConfig, src: dict, frames, log=_log) -> tuple[dict, dict]:
    """Ring index on the raw scan, crop, encode, write shards, for the given frames only. Returns per-scan stats and
    sampled ring profiles."""
    velodyne_zip, members = src["velodyne_zip"], src["scans"]
    todo = sorted((members[int(f)].header_offset, int(f)) for f in frames)     # archive order: sequential reads
    if cfg.scans_dir.exists():
        shutil.rmtree(cfg.scans_dir)                       # a rerun starts clean: shards are append-only

    stats = {k: np.zeros(len(todo), dtype=np.int64) for k in ("frame", "n_raw", "n_crop", "n_rings")}
    prof_frames, prof_counts, prof_elev = [], [], []
    t0 = time.time()
    with ShardWriter(cfg.scans_dir) as writer:
        for k, (_, frame) in enumerate(todo):
            raw = read_bin(velodyne_zip.read(members[frame]))
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
            if (k + 1) % 500 == 0 or k + 1 == len(todo):
                rate = (k + 1) / (time.time() - t0)
                log(f"  {k + 1}/{len(todo)} scans, {rate:.1f} scans/s, {(len(todo) - k - 1) / rate / 60:.1f} min left")

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
    log(f"scans written: {len(todo)} frames, {dir_size(cfg.scans_dir) / GB:.2f} GB in {cfg.scans_dir}")
    return stats, profiles


def stage_images(cfg: PrepConfig, src: dict, frames, log=_log) -> Path | None:
    """Copy the kept frames' PNGs into image shards. Folder sources (the starter) are used in place."""
    images = src["images"]
    if images.kind == "folder":
        log("images: used in place from the starter")
        return None
    if cfg.images_dir.exists():
        shutil.rmtree(cfg.images_dir)
    todo = images.ordered(frames)
    t0 = time.time()
    with ShardWriter(cfg.images_dir) as writer:
        for k, frame in enumerate(todo):
            writer.write_bytes(frame, images.read_bytes(frame), ".png")
            if (k + 1) % 500 == 0 or k + 1 == len(todo):
                rate = (k + 1) / (time.time() - t0)
                log(f"  {k + 1}/{len(todo)} images, {rate:.1f} images/s, {(len(todo) - k - 1) / rate / 60:.1f} min left")
    log(f"images written: {len(todo)} frames, {dir_size(cfg.images_dir) / GB:.2f} GB in {cfg.images_dir}")
    return cfg.images_dir


def stage_frame_index(cfg: PrepConfig, index: dict, scan_stats: dict, log=_log) -> dict:
    """Add the scan pass's point counts and write the frame index."""
    fi.attach_scan_stats(index, scan_stats)
    fi.write_csv(cfg.frame_index_csv, index)
    log(f"frame index -> {cfg.frame_index_csv}")
    return index


# ---------------------------------------------------------------- report

def stage_report(cfg: PrepConfig, found: dict, index: dict, scan_stats: dict, profiles: dict, src: dict,
                 log=_log) -> Path:
    cam = kitti360_camera(found["calibration"])
    candidates = index["frame"][index["usable"] & index["kept"]]
    picks = (candidates[np.linspace(0, len(candidates) - 1, min(cfg.n_overlays, len(candidates))).astype(int)]
             if len(candidates) else [])
    reader, overlays = ShardReader(cfg.scans_dir), []
    for frame in picks:
        image = src["images"].read(int(frame))
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


def stage_dev_sample(cfg: PrepConfig, found: dict, index: dict, src: dict, log=_log) -> Path:
    """dev_sample.zip: a small copy of every component for consecutive frames, laid out like KITTI-360."""
    seq = cfg.sequence
    frames = choose_dev_frames({k: index[k] for k in ("frame", "window_start")} | {"usable": index["usable"] & index["kept"]},
                               cfg.dev_frames)
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
        (img_dir / "data_rect" / f"{f:010d}.png").write_bytes(src["images"].read_bytes(int(f)))

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
        cloud = found["semantics3d"] / part / window if "semantics3d" in found else None
        if cloud is not None and cloud.exists():
            kept = _crop_cloud(cloud, k360 / "data_3d_semantics" / seq / part / window, lo, hi)
            log(f"dev sample: {part} cloud {window}, {kept} points kept")

    if "semantics2d" in found:
        sem_dst = k360 / "data_2d_semantics" / "train" / seq / "image_00"
        for sub in ("semantic", "semantic_rgb", "instance", "confidence"):
            for f in frames:
                s = found["semantics2d"] / sub / f"{f:010d}.png"
                if s.exists():
                    (sem_dst / sub).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(s, sem_dst / sub / s.name)

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


# ---------------------------------------------------------------- manifest and whole run

def stage_manifest(cfg: PrepConfig, found: dict, index: dict, sources: dict, how: str, timings: dict | None = None,
                   log=_log) -> int:
    """dataset_map.yaml for the prepared components, and manifest.json with code version, settings, inputs, counts and
    timings. Returns the output size in bytes."""
    components = {k: v for k, v in found.items() if k != "semantics2d_zip"}
    components.update(scans=cfg.scans_dir, velodyne_timestamps=cfg.velodyne_timestamps, frame_index=cfg.frame_index_csv)
    if cfg.images_dir.exists():
        components.pop("images", None)
        components["image_shards"] = cfg.images_dir
    write_dataset_map(cfg.out_dir / "dataset_map.yaml", cfg.sequence, components)
    total = dir_size(cfg.out_dir)
    counts = {k: int(index[k].sum()) for k in ("has_image", "has_scan", "has_pose", "aligned", "usable", "kept", "sample_5m")}
    write_manifest(cfg.out_dir / "manifest.json", sequence=cfg.sequence, role=cfg.role, settings=cfg.settings(),
                   sources={k: v.rsplit("/", 1)[-1] for k, v in sources.items()}, velodyne_read=how,
                   inputs={k: str(v) for k, v in found.items()}, scan_format=format_info(), counts=counts,
                   timings_s=timings or {}, output_bytes=total)
    log(f"done: {total / GB:.2f} GB in {cfg.out_dir}"
        + ("  WARNING: above Kaggle's 20 GB output limit" if total > 19.5 * GB else ""))
    return total


def run_all(cfg: PrepConfig, log=_log) -> dict:
    timings = {}

    def timed(name, fn, *args):
        t = time.time()
        result = fn(*args)
        timings[name] = round(time.time() - t, 1)
        return result

    sources = load_sources(cfg.sources, cfg.sequence)
    found = timed("inputs", stage_inputs, cfg, log)
    src = timed("sources", stage_sources, cfg, found, sources, log)
    index = timed("select", stage_select, cfg, found, src, log)
    kept = index["frame"][index["kept"]]
    scan_stats, profiles = timed("scans", stage_scans, cfg, src, kept, log)
    timed("images", stage_images, cfg, src, kept, log)
    index = timed("frame_index", stage_frame_index, cfg, index, scan_stats, log)
    timed("report", stage_report, cfg, found, index, scan_stats, profiles, src, log)
    if cfg.dev_frames:
        timed("dev_sample", stage_dev_sample, cfg, found, index, src, log)
    total = stage_manifest(cfg, found, index, sources, src["velodyne_read"], timings, log)
    return {"found": found, "index": index, "scan_stats": scan_stats, "timings": timings, "output_bytes": total}
