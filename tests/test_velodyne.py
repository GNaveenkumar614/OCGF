import numpy as np
import pytest

from ocgf.datasets.velodyne import N_RINGS, CropBox, quadrant, read_bin, ring_index, ring_profile

from kitti360_fixtures import synthetic_scan


def reference_quadrant(x, y) -> int:
    """Scalar quadrant rule as written in the reference implementation."""
    if x > 0 and y >= 0:
        return 1
    if x <= 0 and y > 0:
        return 2
    if x < 0 and y <= 0:
        return 3
    if x >= 0 and y < 0:
        return 4
    return 0


def reference_rings(points) -> np.ndarray:
    """The reference loop: walk the points backwards, new ring on a quadrant 1 -> 4 step, at most 64 rings."""
    ring, previous = 0, 0
    out = np.empty(len(points), dtype=np.int64)
    for i in range(len(points) - 1, -1, -1):
        q = reference_quadrant(points[i, 0], points[i, 1])
        if ring < N_RINGS - 1 and q == 4 and previous == 1:
            ring += 1
        out[i] = ring
        previous = q
    return out


def test_read_bin_from_bytes_and_file(tmp_path):
    points = np.arange(40, dtype=np.float32).reshape(10, 4)
    np.testing.assert_array_equal(read_bin(points.tobytes()), points)
    path = tmp_path / "0000000000.bin"
    points.tofile(path)
    np.testing.assert_array_equal(read_bin(path), points)
    with pytest.raises(ValueError):
        read_bin(points.tobytes()[:-4])


def test_quadrant_matches_the_reference_rule():
    rng = np.random.default_rng(0)
    xy = np.vstack([rng.normal(size=(500, 2)),
                    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]])   # include the axes and the origin
    expected = [reference_quadrant(x, y) for x, y in xy]
    np.testing.assert_array_equal(quadrant(xy[:, 0], xy[:, 1]), expected)


def test_rings_of_a_synthetic_scan_are_recovered_exactly():
    points, true_ring = synthetic_scan()
    np.testing.assert_array_equal(ring_index(points), true_ring)


@pytest.mark.parametrize("seed", range(5))
def test_vectorized_rings_equal_the_reference_loop(seed):
    rng = np.random.default_rng(seed)
    # shuffled points produce many quadrant 1 -> 4 steps, so the 64-ring cap is exercised too
    points = rng.normal(size=(3000, 4)).astype(np.float32)
    np.testing.assert_array_equal(ring_index(points), reference_rings(points))


def test_rings_survive_cropping():
    points, _ = synthetic_scan()
    ring = ring_index(points)                      # rings come from the raw order, before any crop
    keep = CropBox().mask(points)
    assert 0 < keep.sum() < len(points)
    np.testing.assert_array_equal(ring[keep], ring_index(points)[keep])


def test_ring_profile_elevation_rises_with_ring():
    points, _ = synthetic_scan()
    counts, elevation = ring_profile(points, ring_index(points))
    assert (counts == 360).all()
    assert (np.diff(elevation) > 0).all()


def test_crop_box_edges():
    pts = np.array([[-10.0, 0, 0, 0], [79.99, 39.99, 0, 0], [80.0, 0, 0, 0], [0, -40.0, 0, 0], [0, 40.0, 0, 0]])
    np.testing.assert_array_equal(CropBox().mask(pts), [True, True, False, True, False])
