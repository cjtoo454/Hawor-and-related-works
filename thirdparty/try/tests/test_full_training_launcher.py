"""Focused checks for full-training preflight and reproducible snapshots."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "try_full_training_launcher", PROJECT_ROOT / "scripts" / "start_full_training.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def _write_sequence(root: Path, sequence: str, provenance: dict | None = None) -> None:
    n = 4
    images = []
    for frame in range(n):
        image = root / f"{sequence}_{frame}.jpg"
        image.write_bytes(b"prepared image")
        images.append(str(image))
    np.savez_compressed(
        root / f"{sequence}.npz",
        segment_id=np.zeros(n, dtype=np.int32),
        timestamp_sec=np.arange(n, dtype=np.float64) / 30,
        coordinate_frame=np.asarray("per-interval local raw HaWoR SLAM world"),
        oracle_assisted=np.asarray(False),
        oracle_assisted_status=np.asarray("verified_no_oracle"),
        input_provenance_json=np.asarray(json.dumps(provenance or {
            "manifest_status": "verified", "oracle_assisted": False})),
        visual_valid=np.ones(n, dtype=bool),
        gt_valid=np.ones(n, dtype=bool),
        visual_state=np.zeros((n, 109), dtype=np.float32),
        gt_state=np.zeros((n, 109), dtype=np.float32),
        rotation_w2c=np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0),
        translation_w2c=np.zeros((n, 3), dtype=np.float32),
        focal_length=np.full(n, 100, dtype=np.float32),
        image_size=np.asarray([256, 256], dtype=np.int32),
        image_paths=np.asarray(images),
    )


def _prepared_data(tmp_path: Path, provenance: dict | None = None) -> tuple[Path, Path]:
    data_root = tmp_path / "prepared"
    data_root.mkdir()
    train = "T0001_train"
    calibration = "C0001_calibration"
    _write_sequence(data_root, train, provenance)
    _write_sequence(data_root, calibration)
    split_file = tmp_path / "split_manifest.json"
    split_file.write_text(json.dumps({"train": [train], "calibration": [calibration]}))
    return data_root, split_file


def test_args_default_to_gpu_one_and_no_legacy_prior(tmp_path):
    args = launcher.parse_args([
        "--run_dir", str(tmp_path / "run"),
        "--data_root", str(tmp_path / "prepared"),
        "--split_file", str(tmp_path / "split.json"),
    ])
    assert args.gpu == "1"
    assert args.world_checkpoint is None
    command = launcher.training_command(args, tmp_path / "run")
    assert "--world_checkpoint" not in command
    assert command[command.index("--output") + 1] == str(tmp_path / "run" / "checkpoints" / "dual_loop.pt")


def test_preflight_checks_both_splits_and_counts_contiguous_windows(tmp_path):
    data_root, split_file = _prepared_data(tmp_path)
    result = launcher.preflight(data_root, split_file, window=2, repository_root=tmp_path)
    assert result["sequences"] == {"train": 1, "calibration": 1}
    assert result["frames"] == {"train": 4, "calibration": 4}
    assert result["windows"] == {"train": 3, "calibration": 3}


def test_preflight_rejects_unverified_data_before_any_launch(tmp_path, monkeypatch):
    data_root, split_file = _prepared_data(
        tmp_path, {"manifest_status": "missing_unknown", "oracle_assisted": False})
    run_dir = tmp_path / "run"
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path / "repository")
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="origin is unverified"):
        launcher.main([
            "--run_dir", str(run_dir), "--data_root", str(data_root),
            "--split_file", str(split_file),
        ])
    assert not run_dir.exists()


def test_check_only_preflights_without_gpu_or_run_directory(tmp_path, monkeypatch):
    data_root, split_file = _prepared_data(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path / "repository")
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    assert launcher.main([
        "--run_dir", str(run_dir), "--data_root", str(data_root),
        "--split_file", str(split_file), "--window", "2", "--check_only",
    ]) == 0
    assert not run_dir.exists()


def test_source_snapshot_is_hashed_and_deterministic(tmp_path):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    source = project / "scripts" / "train.py"
    source.write_text("print('frozen')\n")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    records = launcher.write_source_snapshot(project, first)
    launcher.write_source_snapshot(project, second)
    assert records == [{
        "path": "scripts/train.py",
        "size_bytes": len(source.read_bytes()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }]
    assert launcher.sha256_file(first) == launcher.sha256_file(second)


def test_run_snapshot_captures_split_and_each_training_archive(tmp_path):
    data_root, split_file = _prepared_data(tmp_path)
    result = launcher.preflight(data_root, split_file, window=2, repository_root=tmp_path)
    args = launcher.parse_args([
        "--run_dir", str(tmp_path / "run"), "--data_root", str(data_root),
        "--split_file", str(split_file), "--window", "2",
    ])
    snapshot = tmp_path / "snapshot"
    hashes = launcher._write_snapshot(
        snapshot, args, result, ["python", "train.py"], {"python": "test"}, "numpy==test\n")
    records = json.loads((snapshot / "dataset_sha256.json").read_text())["archives"]
    assert len(records) == 2
    assert {record["sha256"] for record in records} == {
        launcher.sha256_file(data_root / "T0001_train.npz"),
        launcher.sha256_file(data_root / "C0001_calibration.npz"),
    }
    assert (snapshot / "split_manifest.json").read_bytes() == split_file.read_bytes()
    assert json.loads((snapshot / "source_manifest.json").read_text())["archive_sha256"] == hashes[
        "source_sha256"]
    assert json.loads((snapshot / "training_inputs.json").read_text())["inputs"][0]["sha256"] == hashes[
        "config_sha256"]

