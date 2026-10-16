import io
import re
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from ocgf.datasets import remote_zip
from ocgf.datasets.remote_zip import HTTPRangeFile, download, fetch_range, remote_size


class RangeHandler(BaseHTTPRequestHandler):
    """Serves one payload, honours single byte ranges, and can fail the next few requests with 503."""
    payload = b""
    fail_next = 0
    ignore_range = False

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        cls = type(self)
        if cls.fail_next > 0:
            cls.fail_next -= 1
            self.send_error(503)
            return
        data, rng = self.payload, self.headers.get("Range")
        if rng and not cls.ignore_range:
            m = re.match(r"bytes=(\d+)-(\d*)$", rng)
            start = int(m[1])
            end = min(int(m[2]) if m[2] else len(data) - 1, len(data) - 1)
            body = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def serve(monkeypatch):
    monkeypatch.setattr(remote_zip, "BACKOFF_S", 0.0)
    servers = []

    def start(payload: bytes, **attrs):
        handler = type("Handler", (RangeHandler,), {"payload": payload, **attrs})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/archive.zip", handler

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def payload(n: int = 300_000, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def test_size_and_ranges(serve):
    data = payload()
    url, _ = serve(data)
    assert remote_size(url) == len(data)
    assert fetch_range(url, 1000, 1999) == data[1000:2000]


def test_range_file_reads_like_a_local_file(serve):
    data = payload()
    url, _ = serve(data)
    rng = np.random.default_rng(1)
    with HTTPRangeFile(url, block_size=4096, max_blocks=3) as f:
        assert f.seek(0, 2) == len(data)
        for _ in range(50):
            start, n = int(rng.integers(0, len(data))), int(rng.integers(0, 20_000))
            f.seek(start)
            assert f.read(n) == data[start:start + n]
            assert f.tell() == min(start + n, len(data))
        f.seek(-10, 2)
        assert f.read() == data[-10:]


def test_zipfile_over_http_reads_every_member(serve):
    buf = io.BytesIO()
    members = {f"seq/velodyne_points/data/{i:010d}.bin": payload(5000 + i, seed=i) for i in range(20)}
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    url, _ = serve(buf.getvalue())
    with zipfile.ZipFile(HTTPRangeFile(url, block_size=8192)) as z:
        infos = sorted(z.infolist(), key=lambda i: i.header_offset)
        assert {i.filename for i in infos} == set(members)
        for info in infos:
            assert z.read(info) == members[info.filename]          # zipfile also checks each CRC


def test_transient_errors_are_retried(serve):
    data = payload(10_000)
    url, handler = serve(data, fail_next=2)
    assert fetch_range(url, 0, 99, retries=3) == data[:100]


def test_server_without_range_support_is_refused(serve):
    url, _ = serve(payload(10_000), ignore_range=True)
    with pytest.raises(RuntimeError, match="range"):
        fetch_range(url, 0, 99)


def test_download_resumes_a_partial_file(serve, tmp_path):
    data = payload()
    url, _ = serve(data)
    dest = tmp_path / "archive.zip"
    (tmp_path / "archive.zip.part").write_bytes(data[:123_457])        # left over from an interrupted run
    seen = []
    assert download(url, dest, chunk_size=50_000, progress=lambda d, t: seen.append(d)) == dest
    assert dest.read_bytes() == data
    assert seen[0] > 123_457 and seen[-1] == len(data)
    assert not (tmp_path / "archive.zip.part").exists()
