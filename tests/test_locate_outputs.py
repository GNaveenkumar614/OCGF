"""Finding teacher outputs, ground truth and training runs among the inputs without walking per-frame folders."""
import json

from ocgf.datasets.locate import find_outputs, find_runs

SEQ = "2013_05_28_drive_0000_sync"


def test_outputs_and_runs_are_found_by_manifest_and_per_frame_folders_are_skipped(tmp_path):
    def manifest(folder, **content):
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps(content))
    manifest(tmp_path / "labels" / "road_labels" / SEQ, teacher="road")
    manifest(tmp_path / "gt" / "eval_gt" / SEQ, kind="eval_gt")
    manifest(tmp_path / "prep" / "scans" / SEQ, teacher="road")           # inside scans/: never searched
    (tmp_path / "run" / "dev_e0_concat_s0" / "ckpt").mkdir(parents=True)
    (tmp_path / "run" / "dev_e0_concat_s0" / "ckpt" / "last.pt").write_bytes(b"")
    assert find_outputs(tmp_path, SEQ, "teacher", "road") == [tmp_path / "labels" / "road_labels" / SEQ]
    assert find_outputs(tmp_path, SEQ, "kind", "eval_gt") == [tmp_path / "gt" / "eval_gt" / SEQ]
    assert find_runs(tmp_path, "dev_e0_concat_s0") == [tmp_path / "run" / "dev_e0_concat_s0"]
    assert find_runs(tmp_path, "other") == []


def test_the_full_prepared_output_wins_over_a_trial_and_trials_are_flagged(tmp_path):
    from ocgf.datasets.locate import describe_prepared, find_prepared, prepared_info
    from ocgf.datasets.prepare_kitti360 import PrepConfig, run_all
    from kitti360_fixtures import SEQ as FSEQ, make_fake_kitti360
    paths = make_fake_kitti360(tmp_path, n_frames=40, starter_images=False)
    root = paths["input_root"]

    def prep(name, max_frames):
        run_all(PrepConfig(sequence=FSEQ, input_root=root, out_dir=root / name / "kitti360_supplement",
                           scratch_dir=tmp_path / f"scratch_{name}", sources=paths["sources"], role="train",
                           keep_every_m=2.5, min_future_m=20.0, n_overlays=0, dev_frames=0, max_frames=max_frames),
                log=lambda m: None)
        return root / name / "kitti360_supplement"
    trial = prep("a_trial", 4)                     # sorts first, as a trial dataset can on Kaggle
    assert "TRIAL" in describe_prepared(root, FSEQ)
    full = prep("b_full", None)
    assert prepared_info(full, FSEQ)["usable_kept"] > prepared_info(trial, FSEQ)["usable_kept"]
    assert find_prepared(root, FSEQ) == full
    line = describe_prepared(root, FSEQ)
    assert str(full) in line.splitlines()[0] and "also attached, not used" in line and "TRIAL" not in line
