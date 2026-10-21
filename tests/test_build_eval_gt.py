"""scripts/build_eval_gt.py end to end on a fake prepared drive: one layer file per sampled frame, a manifest, QA."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from ocgf.datasets.prepare_kitti360 import PrepConfig, run_all
from ocgf.evaluation.gt_builder import LAYERS, load_layers

from kitti360_fixtures import SEQ, make_fake_kitti360

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_eval_gt  # noqa: E402


def test_builds_every_sampled_frame_of_a_prepared_drive(tmp_path):
    paths = make_fake_kitti360(tmp_path, n_frames=40, starter_images=False)
    run_all(PrepConfig(sequence=SEQ, input_root=paths["input_root"], out_dir=paths["input_root"] / "supp" / "out",
                       scratch_dir=tmp_path / "scratch", sources=paths["sources"], role="test", keep_every_m=2.5,
                       min_future_m=20.0, n_overlays=0, dev_frames=0), log=lambda m: None)
    out = tmp_path / "eval_gt"
    build_eval_gt.main(["--input-root", str(paths["input_root"]), "--sequence", SEQ, "--out", str(out), "--qa", "1",
                        "--stage-dir", str(tmp_path / "stage"), "--threads", "2"])
    files = sorted((out / SEQ).glob("*.npz"))
    manifest = json.loads((out / SEQ / "manifest.json").read_text())
    assert files and manifest["frames"] == len(files) == len((out / SEQ / "summary.jsonl").read_text().splitlines())
    assert set(load_layers(files[0])) == set(LAYERS) and len(list((out / SEQ / "qa").glob("*.png"))) == 1
