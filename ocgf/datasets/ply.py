"""Minimal reader and writer for vertex-only binary little-endian PLY files (KITTI-360 accumulated point clouds)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}
_NAMES = {np.dtype(v).str: k for k, v in reversed(list(_TYPES.items()))}   # dtype -> the short PLY name


def read_ply(path) -> np.ndarray:
    """The vertex element as a structured array (one field per property). Other elements must come after it."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fields, count, in_vertex = [], None, False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: header has no end_header")
            words = line.decode("ascii").split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "format" and words[1] != "binary_little_endian":
                raise ValueError(f"{path}: only binary_little_endian PLY is supported, got {words[1]}")
            if words[0] == "element":
                in_vertex = words[1] == "vertex"
                if in_vertex:
                    count = int(words[2])
                elif count is None:
                    raise ValueError(f"{path}: the vertex element must come first")
            elif words[0] == "property" and in_vertex:
                if words[1] == "list":
                    raise ValueError(f"{path}: list properties in the vertex element are not supported")
                fields.append((words[2], _TYPES[words[1]]))
            elif words[0] == "end_header":
                break
        if count is None:
            raise ValueError(f"{path}: no vertex element")
        return np.fromfile(f, dtype=np.dtype(fields), count=count)


def write_ply(path, vertices: np.ndarray, comments=()) -> None:
    """Write a structured array as the vertex element of a binary little-endian PLY file."""
    vertices = np.ascontiguousarray(vertices, dtype=vertices.dtype.newbyteorder("<"))
    lines = ["ply", "format binary_little_endian 1.0", *(f"comment {c}" for c in comments),
             f"element vertex {len(vertices)}"]
    for name in vertices.dtype.names:
        lines.append(f"property {_NAMES[vertices.dtype[name].str]} {name}")
    lines.append("end_header")
    with open(Path(path), "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        vertices.tofile(f)
