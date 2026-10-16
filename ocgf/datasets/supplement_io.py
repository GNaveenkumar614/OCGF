"""Storage of prepared scans: compact point records in zip shards.

Each frame is one member '{frame:010d}.npy' holding a structured array with fields x, y, z (int16 centimetres),
intensity (uint8, 0..255 for 0..1) and ring (uint8): 8 bytes per point, at most 0.5 cm quantization error.
Frames are grouped SHARD_SIZE to a shard, 'shard_{frame // SHARD_SIZE:03d}.zip' (stored, not compressed), which keeps
Kaggle outputs to a few hundred files.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
SHARD_SIZE = 500
RECORD = np.dtype([("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("intensity", "u1"), ("ring", "u1")])


def format_info() -> dict:
    """What a reader needs to know about this format, for the dataset manifest."""
    return {"format_version": FORMAT_VERSION, "shard_size": SHARD_SIZE, "record": RECORD.descr,
            "units": {"x": "cm", "y": "cm", "z": "cm", "intensity": "0..255 for 0..1", "ring": "index"}}


def encode(points, ring) -> np.ndarray:
    """float [N, 4] (x, y, z in m, intensity in 0..1) and ring [N] -> RECORD [N]."""
    points = np.asarray(points, dtype=np.float64)
    cm = np.rint(points[:, :3] * 100.0)
    if len(cm) and np.abs(cm).max() > np.iinfo(np.int16).max:
        raise ValueError("coordinates beyond +-327 m do not fit int16 centimetres; crop the scan first")
    rec = np.empty(len(points), dtype=RECORD)
    rec["x"], rec["y"], rec["z"] = cm[:, 0], cm[:, 1], cm[:, 2]
    rec["intensity"] = np.clip(np.rint(points[:, 3] * 255.0), 0, 255)
    rec["ring"] = ring
    return rec


def decode(rec) -> np.ndarray:
    """RECORD [N] -> float32 [N, 5]: x, y, z (m), intensity (0..1), ring."""
    out = np.empty((len(rec), 5), dtype=np.float32)
    out[:, 0] = rec["x"] / 100.0
    out[:, 1] = rec["y"] / 100.0
    out[:, 2] = rec["z"] / 100.0
    out[:, 3] = rec["intensity"] / 255.0
    out[:, 4] = rec["ring"]
    return out


def shard_name(frame: int) -> str:
    return f"shard_{frame // SHARD_SIZE:03d}.zip"


def member_name(frame: int) -> str:
    return f"{frame:010d}.npy"


class ShardWriter:
    """Writes records frame by frame. Writing frames in increasing order opens each shard once."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._zip = None
        self._shard = None
        self._members = set()

    def write(self, frame: int, rec: np.ndarray) -> None:
        name = shard_name(frame)
        if name != self._shard:
            self.close()
            self._zip = zipfile.ZipFile(self.directory / name, mode="a", compression=zipfile.ZIP_STORED)
            self._shard = name
            self._members = set(self._zip.namelist())
        member = member_name(frame)
        if member in self._members:
            raise ValueError(f"frame {frame} is already in {name}")
        buf = io.BytesIO()
        np.save(buf, np.asarray(rec, dtype=RECORD), allow_pickle=False)
        self._zip.writestr(member, buf.getvalue())
        self._members.add(member)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
        self._zip, self._shard = None, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ShardReader:
    """Reads records by frame id. Zip handles are opened lazily and reopened in a forked worker process."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self._zips = {}
        self._pid = os.getpid()

    def frames(self) -> list[int]:
        frames = []
        for shard in sorted(self.directory.glob("shard_*.zip")):
            frames.extend(int(Path(m).stem) for m in self._open(shard.name).namelist())
        return sorted(frames)

    def read_record(self, frame: int) -> np.ndarray:
        data = self._open(shard_name(frame)).read(member_name(frame))
        return np.load(io.BytesIO(data), allow_pickle=False)

    def read(self, frame: int) -> np.ndarray:
        """float32 [N, 5]: x, y, z, intensity, ring."""
        return decode(self.read_record(frame))

    def _open(self, name: str) -> zipfile.ZipFile:
        if os.getpid() != self._pid:            # handles must not be shared across processes
            self._zips, self._pid = {}, os.getpid()
        if name not in self._zips:
            self._zips[name] = zipfile.ZipFile(self.directory / name, mode="r")
        return self._zips[name]

    def close(self) -> None:
        for z in self._zips.values():
            z.close()
        self._zips = {}
