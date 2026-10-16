"""Read large remote zip archives without storing them: a seekable file over HTTP range requests, plus a resumable
download for when the disk has room.

zipfile.ZipFile accepts any seekable file object, so ZipFile(HTTPRangeFile(url)) lists and reads members while only
fetching the bytes it needs. Reading members in archive order (by header_offset) keeps the requests sequential.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

MiB = 1 << 20
BACKOFF_S = 1.0          # first retry wait; doubles per attempt, capped at 30 s


def _wait(attempt: int) -> None:
    time.sleep(min(BACKOFF_S * 2 ** attempt, 30))


def _retry(fn, retries: int, what: str):
    for attempt in range(retries + 1):
        try:
            return fn()
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as err:
            if isinstance(err, urllib.error.HTTPError) and err.code < 500 and err.code != 429:
                raise                                   # 4xx other than rate limiting will not get better
            if attempt == retries:
                raise RuntimeError(f"{what} failed after {retries + 1} attempts: {err}") from err
            _wait(attempt)


def remote_size(url: str, timeout: float = 60, retries: int = 5) -> int:
    """Size in bytes of a remote file, from a HEAD request (or a one-byte range request if HEAD gives none)."""
    def head():
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout) as r:
            return int(r.headers.get("Content-Length", -1))
    size = _retry(head, retries, "HEAD")
    if size < 0:
        def probe():
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return int(r.headers["Content-Range"].rsplit("/", 1)[1])
        size = _retry(probe, retries, "size probe")
    return size


def fetch_range(url: str, start: int, end: int, timeout: float = 60, retries: int = 5) -> bytes:
    """Bytes [start, end] (inclusive) of a remote file. Refuses servers that ignore the Range header."""
    def get():
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 206:
                raise RuntimeError(f"server ignored the range request (HTTP {r.status}); remote reading needs ranges")
            data = r.read()
        if len(data) != end - start + 1:
            raise ConnectionError(f"short read: {len(data)} of {end - start + 1} bytes")
        return data
    return _retry(get, retries, f"range {start}-{end}")


class HTTPRangeFile:
    """Read-only, seekable file over HTTP range requests, with an LRU cache of fixed-size blocks."""

    def __init__(self, url: str, block_size: int = 16 * MiB, max_blocks: int = 8, timeout: float = 60, retries: int = 5):
        self.url, self.block_size, self.max_blocks = url, block_size, max_blocks
        self.timeout, self.retries = timeout, retries
        self.size = remote_size(url, timeout, retries)
        self._pos = 0
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self.closed = False
        self.requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self.size}[whence]
        if base + offset < 0:
            raise ValueError("negative seek position")
        self._pos = base + offset
        return self._pos

    def _block(self, index: int) -> bytes:
        if index in self._blocks:
            self._blocks.move_to_end(index)
            return self._blocks[index]
        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        data = fetch_range(self.url, start, end, self.timeout, self.retries)
        self.requests += 1
        self._blocks[index] = data
        if len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)
        return data

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        n = max(0, min(n, self.size - self._pos))
        parts = []
        while n > 0:
            index, offset = divmod(self._pos, self.block_size)
            chunk = self._block(index)[offset:offset + n]
            parts.append(chunk)
            self._pos += len(chunk)
            n -= len(chunk)
        return b"".join(parts)

    def close(self) -> None:
        self._blocks.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def download(url: str, dest, chunk_size: int = 16 * MiB, timeout: float = 60, retries: int = 5, progress=None) -> Path:
    """Download to dest, resuming a partial 'dest.part' left by an earlier attempt. Returns dest.

    progress(done_bytes, total_bytes) is called after every chunk if given.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url, timeout, retries)
    for attempt in range(retries + 1):
        done = part.stat().st_size if part.exists() else 0
        if done >= total:
            break
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "ab" if done else "wb") as f:
                if done and r.status != 206:
                    raise RuntimeError("server ignored the range request, cannot resume")
                while chunk := r.read(chunk_size):
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as err:
            if attempt == retries:
                raise RuntimeError(f"download failed after {retries + 1} attempts: {err}") from err
            _wait(attempt)
    if part.stat().st_size != total:
        raise RuntimeError(f"download incomplete: {part.stat().st_size} of {total} bytes")
    part.rename(dest)
    return dest
