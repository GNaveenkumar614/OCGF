"""Canonical frame ids: '{dataset}/{sequence}/{frame:010d}', used by every teacher output and cache."""
from __future__ import annotations


def frame_id(dataset: str, sequence: str, frame: int) -> str:
    return f"{dataset}/{sequence}/{int(frame):010d}"


def parse_frame_id(fid: str) -> tuple[str, str, int]:
    dataset, sequence, frame = fid.split("/")
    if len(frame) != 10 or not frame.isdigit():
        raise ValueError(f"not a canonical frame id: {fid!r}")
    return dataset, sequence, int(frame)
