import numpy as np
import pytest

from ocgf.datasets.ply import read_ply, write_ply

from kitti360_fixtures import cloud


def test_round_trip(tmp_path):
    c = cloud(500, (0, 100), seed=0)
    path = tmp_path / "0000000002_0000000385.ply"
    write_ply(path, c, comments=["test"])
    back = read_ply(path)
    assert back.dtype.names == c.dtype.names
    for name in c.dtype.names:
        np.testing.assert_array_equal(back[name], c[name])


def test_reads_a_header_written_by_another_tool(tmp_path):
    # long type names, a comment and an extra element after the vertices, as other writers produce
    vertices = np.zeros(3, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("semantic", "<i4")])
    vertices["x"] = [1, 2, 3]
    vertices["semantic"] = [7, 8, 9]
    header = ("ply\nformat binary_little_endian 1.0\ncomment made elsewhere\nelement vertex 3\n"
              "property float32 x\nproperty float32 y\nproperty float32 z\nproperty int32 semantic\n"
              "element face 0\nproperty list uchar int vertex_indices\nend_header\n")
    path = tmp_path / "other.ply"
    path.write_bytes(header.encode() + vertices.tobytes())
    back = read_ply(path)
    np.testing.assert_array_equal(back["x"], [1, 2, 3])
    np.testing.assert_array_equal(back["semantic"], [7, 8, 9])


def test_ascii_ply_is_refused(tmp_path):
    path = tmp_path / "ascii.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1.0\n")
    with pytest.raises(ValueError):
        read_ply(path)
