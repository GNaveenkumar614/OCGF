import zipfile

import numpy as np
import pytest

from ocgf.datasets.supplement_io import RECORD, SHARD_SIZE, ShardReader, ShardWriter, decode, encode

from kitti360_fixtures import synthetic_scan


def scan(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    points = np.column_stack([rng.uniform(-10, 80, n), rng.uniform(-40, 40, n), rng.uniform(-3, 25, n),
                              rng.uniform(0, 1, n)]).astype(np.float32)
    return points, rng.integers(0, 64, n).astype(np.uint8)


def test_record_is_8_bytes_per_point():
    assert RECORD.itemsize == 8


def test_round_trip_error_is_at_most_half_a_centimetre():
    points, ring = scan()
    out = decode(encode(points, ring))
    assert out.dtype == np.float32 and out.shape == (len(points), 5)
    assert np.abs(out[:, :3] - points[:, :3]).max() <= 0.005 + 1e-6
    assert np.abs(out[:, 3] - points[:, 3]).max() <= 0.5 / 255 + 1e-6
    np.testing.assert_array_equal(out[:, 4], ring)


def test_intensity_outside_0_1_is_clipped():
    points = np.array([[1.0, 2.0, 3.0, -0.2], [1.0, 2.0, 3.0, 1.7]])
    rec = encode(points, np.zeros(2, np.uint8))
    np.testing.assert_array_equal(rec["intensity"], [0, 255])


def test_out_of_range_coordinates_are_refused():
    with pytest.raises(ValueError):
        encode(np.array([[400.0, 0.0, 0.0, 0.5]]), np.zeros(1, np.uint8))


def test_empty_scan():
    rec = encode(np.zeros((0, 4)), np.zeros(0, np.uint8))
    assert decode(rec).shape == (0, 5)


def test_shards_round_trip(tmp_path):
    frames = [0, 1, SHARD_SIZE - 1, SHARD_SIZE, 2 * SHARD_SIZE + 203]
    scans = {f: scan(50 + f % 7, seed=f) for f in frames}
    with ShardWriter(tmp_path) as writer:
        for f in frames:
            writer.write(f, encode(*scans[f]))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shard_000.zip", "shard_001.zip", "shard_002.zip"]
    reader = ShardReader(tmp_path)
    assert reader.frames() == frames
    for f in frames:
        np.testing.assert_array_equal(reader.read_record(f), encode(*scans[f]))
    reader.close()


def test_shards_are_stored_uncompressed(tmp_path):
    with ShardWriter(tmp_path) as writer:
        writer.write(3, encode(*scan()))
    with zipfile.ZipFile(tmp_path / "shard_000.zip") as z:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in z.infolist())


def test_a_later_session_appends_and_refuses_duplicates(tmp_path):
    with ShardWriter(tmp_path) as writer:
        writer.write(0, encode(*scan(seed=0)))
    with ShardWriter(tmp_path) as writer:
        writer.write(1, encode(*scan(seed=1)))
        with pytest.raises(ValueError):
            writer.write(0, encode(*scan(seed=2)))
    reader = ShardReader(tmp_path)
    assert reader.frames() == [0, 1]
    np.testing.assert_array_equal(reader.read_record(0), encode(*scan(seed=0)))
    reader.close()


@pytest.mark.parametrize("folder_name", ["shard_000", "shard_000.zip"])
def test_shards_unpacked_into_folders_read_the_same(tmp_path, folder_name):
    # Kaggle unpacks zip files inside an uploaded archive: shard_000.zip becomes a folder
    import zipfile
    points, ring = synthetic_scan(points_per_ring=10, seed=1)
    with ShardWriter(tmp_path / "zipped") as w:
        w.write(7, encode(points, ring))
        w.write_bytes(9, b"png bytes", ".png")
    unpacked = tmp_path / "unpacked" / folder_name
    with zipfile.ZipFile(tmp_path / "zipped" / "shard_000.zip") as z:
        z.extractall(unpacked)
    zipped, folder = ShardReader(tmp_path / "zipped"), ShardReader(tmp_path / "unpacked")
    assert folder.frames() == zipped.frames() == [7] and folder.frames(".png") == [9]
    assert np.array_equal(folder.read(7), zipped.read(7)) and folder.read_bytes(9, ".png") == b"png bytes"
