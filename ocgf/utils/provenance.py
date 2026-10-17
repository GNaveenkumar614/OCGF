"""Provenance: which code, environment and inputs produced an output folder."""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

import ocgf

def git_commit(repo=None) -> dict:
    """{'commit': sha or None, 'dirty': bool or None} of the git checkout at repo (default: this package's repo)."""
    repo = Path(repo) if repo else Path(ocgf.__file__).resolve().parents[1]
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
    try:
        sha = git("rev-parse", "HEAD")
        if sha.returncode:
            return {"commit": None, "dirty": None}
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip())
        return {"commit": sha.stdout.strip(), "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
           "ocgf": ocgf.__version__}
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        env["torch"] = None
    return env


def write_manifest(path, **sections) -> dict:
    """Write manifest.json: creation time, code version, environment and any extra sections given."""
    manifest = {"created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "code": git_commit(), "environment": environment(), **sections}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=_jsonable))
    return manifest


def _jsonable(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
