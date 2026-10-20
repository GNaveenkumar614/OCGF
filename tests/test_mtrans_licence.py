"""Vendored MTrans code keeps its licence notices (MPL-2.0 is per file; the IoU loss is MIT)."""
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1] / "ocgf" / "teachers" / "mtrans"
OURS = {"infer.py", "__init__.py"}            # new OCGF files under the repository licence


def test_every_copied_file_carries_its_notice():
    for path in FOLDER.rglob("*.py"):
        if path.name in OURS:
            continue
        head = path.read_text(encoding="utf-8")[:1200]
        assert "https://mozilla.org/MPL/2.0/" in head or "Rotated_IoU" in head, f"{path.name} lost its licence notice"


def test_licence_texts_are_present():
    assert (FOLDER / "LICENSE").read_text(encoding="utf-8").startswith("Mozilla Public License Version 2.0")
    assert "Lanxiao Li" in (FOLDER / "iou" / "LICENSE-Rotated_IoU").read_text(encoding="utf-8")
