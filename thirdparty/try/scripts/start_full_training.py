#!/usr/bin/env python3
"""Preflight and start a detached, reproducible dual-loop training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
FPS = 30.0
REQUIRED_FIELDS = {
    "segment_id", "timestamp_sec", "coordinate_frame", "oracle_assisted",
    "input_provenance_json", "oracle_assisted_status", "visual_valid",
    "gt_valid", "visual_state", "gt_state", "rotation_w2c",
    "translation_w2c", "focal_length", "image_size", "image_paths",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="New, unique directory for this run and its checkpoint")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Directory containing one prepared <sequence>.npz per split entry")
    parser.add_argument("--split_file", type=Path, required=True,
                        help="Prepared JSON split manifest with train and calibration lists")
    parser.add_argument("--gpu", default="1", help="Physical GPU(s) exposed to training")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--world_checkpoint", type=Path,
                        help="Optional compatible world-model prior; omitted runs train from scratch")
    parser.add_argument("--resume", type=Path, help="Optional full-training checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_epoch", type=int, default=300)
    parser.add_argument("--validation_steps", type=int, default=30)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--quality_error_m", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check_only", action="store_true",
                        help="Validate inputs and windows without requiring a GPU or launching training")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _participant(sequence: str) -> str:
    return sequence.split("_", 1)[0]


def _validate_manifest_split(split: object, split_file: Path) -> tuple[list[str], list[str]]:
    if not isinstance(split, dict):
        raise ValueError(f"Split file must contain a JSON object: {split_file}")
    train = split.get("train")
    calibration = split.get("calibration")
    if (not isinstance(train, list) or not isinstance(calibration, list)
            or not train or not calibration):
        raise ValueError("Nonempty train and calibration sequence lists are required")
    for name, sequences in (("train", train), ("calibration", calibration)):
        if any(not isinstance(seq, str) or not seq or Path(seq).name != seq
               or "/" in seq or "\\" in seq for seq in sequences):
            raise ValueError(f"{name} entries must be simple sequence names")
        if len(sequences) != len(set(sequences)):
            raise ValueError(f"Duplicate sequence in {name} split")
    if set(train) & set(calibration):
        raise ValueError("Nonempty disjoint train/calibration splits required")
    if ({_participant(s) for s in train}
            & {_participant(s) for s in calibration}):
        raise ValueError("Participant leakage between train and calibration")

    # Preparation manifests carry a record for every selected video, including
    # unusable/skipped videos. Ensure every eligible prepared sequence is used.
    records = split.get("sequence_records")
    if isinstance(records, list) and any(
            isinstance(record, dict) and "status" in record for record in records):
        eligible = {
            record.get("sequence") for record in records
            if isinstance(record, dict) and record.get("status") == "prepared"
            and int(record.get("eligible_frames", 0)) > 0
        }
        if eligible != set(train) | set(calibration):
            missing = sorted(eligible - (set(train) | set(calibration)))
            extra = sorted((set(train) | set(calibration)) - eligible)
            raise ValueError(
                f"Split does not cover eligible prepared sequences (missing={missing}, extra={extra})")
    for alias, sequences in (("train_sequences", train), ("calibration_sequences", calibration)):
        if alias in split and split[alias] != sequences:
            raise ValueError(f"Split manifest {alias} disagrees with its {alias.removesuffix('_sequences')} list")
    return train, calibration


def _load_sequence(path: Path, sequence: str, split_name: str,
                   repository_root: Path) -> tuple[dict[str, np.ndarray], int, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split_name} sequence archive: {path}")
    before = path.stat()
    # Read one archive at a time. This keeps preflight memory bounded while
    # touching every member so CRC/truncation errors are caught before launch.
    arrays: dict[str, np.ndarray] = {}
    member_shapes: dict[str, tuple[int, ...]] = {}
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_FIELDS - set(archive.files)
        if missing:
            raise ValueError(f"{sequence}: missing full-observer data fields: {sorted(missing)}")
        for key in archive.files:
            value = archive[key]
            member_shapes[key] = value.shape
            if key in {
                    "segment_id", "timestamp_sec", "coordinate_frame",
                    "oracle_assisted", "input_provenance_json",
                    "oracle_assisted_status", "visual_valid", "gt_valid",
                    "image_paths"}:
                arrays[key] = value.copy()
            del value
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"Dataset changed while preflighting: {path}")

    if bool(np.asarray(arrays["oracle_assisted"]).item()):
        raise ValueError(f"{sequence}: GT boxes forbidden for full observer training")
    try:
        provenance = json.loads(str(np.asarray(arrays["input_provenance_json"]).item()))
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{sequence}: invalid frontend provenance JSON") from error
    if (not isinstance(provenance, dict)
            or provenance.get("manifest_status") != "verified"
            or str(np.asarray(arrays["oracle_assisted_status"]).item()) != "verified_no_oracle"
            or provenance.get("oracle_assisted") is not False):
        raise ValueError(f"{sequence}: frontend origin is unverified; regenerate prepared observations")
    if provenance.get("raw_export_diagnostic"):
        raise ValueError(f"{sequence}: diagnostic raw export forbidden for formal training")

    timestamps = np.asarray(arrays["timestamp_sec"])
    segments = np.asarray(arrays["segment_id"])
    visual_valid = np.asarray(arrays["visual_valid"], dtype=bool)
    gt_valid = np.asarray(arrays["gt_valid"], dtype=bool)
    paths = np.asarray(arrays["image_paths"])
    if timestamps.ndim != 1 or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{sequence}: nonmonotonic or non-finite timestamps")
    n = len(timestamps)
    frame_fields = ("segment_id", "visual_valid", "gt_valid", "image_paths",
                    "visual_state", "gt_state", "rotation_w2c", "translation_w2c",
                    "focal_length")
    if (segments.shape != (n,) or any(
            not member_shapes.get(name) or member_shapes[name][0] != n
            for name in frame_fields)):
        raise ValueError(f"{sequence}: frame-aligned arrays have inconsistent lengths")
    if (visual_valid.shape != (n,) or gt_valid.shape != (n,) or paths.shape != (n,)
            or member_shapes["visual_state"] != member_shapes["gt_state"]
            or len(member_shapes["visual_state"]) != 2):
        raise ValueError(f"{sequence}: malformed state, validity, or image-path arrays")
    if member_shapes["rotation_w2c"] != (n, 3, 3) or member_shapes["translation_w2c"] != (n, 3):
        raise ValueError(f"{sequence}: malformed camera arrays")
    if member_shapes["image_size"] != (2,) or member_shapes["focal_length"] != (n,):
        raise ValueError(f"{sequence}: malformed camera intrinsics")

    for raw_image_path in paths:
        image_path = Path(str(raw_image_path))
        if not image_path.is_absolute():
            image_path = repository_root / image_path
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise FileNotFoundError(f"{sequence}: missing or empty prepared image: {image_path}")

    # Return only the small arrays required to reproduce train.window_index.
    return {
        "segment_id": segments,
        "timestamp_sec": timestamps,
        "visual_valid": visual_valid,
        "gt_valid": gt_valid,
    }, n, {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
          "inode": before.st_ino}


def _window_count(archives: list[dict[str, np.ndarray]], window: int) -> int:
    count = 0
    for data in archives:
        n = len(data["visual_valid"])
        good = ((np.diff(data["segment_id"]) == 0)
                & (np.abs(np.diff(data["timestamp_sec"]) - 1.0 / FPS) < 0.005))
        for start in range(max(0, n - window + 1)):
            if (data["visual_valid"][start]
                    and data["gt_valid"][start:start + window].any()
                    and good[start:start + window - 1].all()):
                count += 1
    return count


def preflight(data_root: Path, split_file: Path, window: int,
              repository_root: Path = REPOSITORY_ROOT) -> dict:
    if window < 2:
        raise ValueError("--window must be at least 2")
    data_root = data_root.resolve()
    split_file = split_file.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Prepared data directory does not exist: {data_root}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")
    split = json.loads(split_file.read_text(encoding="utf-8"))
    train, calibration = _validate_manifest_split(split, split_file)
    heldout = repository_root / "datasets" / "hot3d_valset_export" / "val.json"
    if heldout.is_file():
        val_sequences = json.loads(heldout.read_text(encoding="utf-8"))
        train_participants = {_participant(sequence) for sequence in train + calibration}
        if train_participants & {_participant(sequence) for sequence in val_sequences}:
            raise ValueError("Training/calibration participant overlaps held-out test")

    counts = {"train": 0, "calibration": 0}
    all_archives = {"train": [], "calibration": []}
    frame_counts = {"train": 0, "calibration": 0}
    archive_stats = {}
    for split_name, sequences in (("train", train), ("calibration", calibration)):
        for sequence in sequences:
            archive, frame_count, stat = _load_sequence(
                data_root / f"{sequence}.npz", sequence, split_name, repository_root)
            all_archives[split_name].append(archive)
            frame_counts[split_name] += frame_count
            archive_stats[sequence] = stat
        counts[split_name] = _window_count(all_archives[split_name], window)
        if not counts[split_name]:
            raise ValueError(f"No continuous {split_name} windows with observed initialization and GT")
    return {
        "data_root": str(data_root),
        "split_file": str(split_file),
        "window": window,
        "sequences": {"train": len(train), "calibration": len(calibration)},
        "frames": frame_counts,
        "windows": counts,
        "skipped_sequences": len(split.get("skipped_sequences", [])),
        "archive_stats": archive_stats,
        "split": split,
    }


def training_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "--data_root", str(args.data_root.resolve()),
        "--split_file", str(args.split_file.resolve()),
        "--config", str(args.config.resolve()),
        "--output", str(run_dir / "checkpoints" / "dual_loop.pt"),
        "--epochs", str(args.epochs),
        "--steps_per_epoch", str(args.steps_per_epoch),
        "--validation_steps", str(args.validation_steps),
        "--window", str(args.window),
        "--learning_rate", str(args.learning_rate),
        "--quality_error_m", str(args.quality_error_m),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if args.world_checkpoint is not None:
        command.extend(["--world_checkpoint", str(args.world_checkpoint.resolve())])
    if args.resume is not None:
        command.extend(["--resume", str(args.resume.resolve())])
    return command


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_record(path: Path, role: str) -> dict:
    path = path.resolve()
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return {
        "role": role,
        "path": str(path),
        "size_bytes": before.st_size,
        "sha256": digest,
    }


def _project_source_files(project: Path) -> list[tuple[Path, str]]:
    files: dict[Path, str] = {}
    for folder in ("dual_loop", "scripts", "configs", "tests"):
        base = project / folder
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
                    files[path] = path.relative_to(project).as_posix()
    for name in ("README.md", "DESIGN.md", "pyproject.toml", "requirements.txt"):
        path = project / name
        if path.is_file():
            files[path] = path.relative_to(project).as_posix()
    asset = project / "assets" / "mano_right.npz"
    if asset.is_file():
        files[asset] = asset.relative_to(project).as_posix()
    repository_root = project.parents[1] if len(project.parents) > 1 else project.parent
    mano_wrapper = repository_root / "lib" / "models" / "mano_wrapper.py"
    if mano_wrapper.is_file():
        files[mano_wrapper] = "repository/lib/models/mano_wrapper.py"
    return sorted(files.items(), key=lambda item: item[1])


def write_source_snapshot(project: Path, archive_path: Path) -> list[dict]:
    files = _project_source_files(project)
    records = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in files:
            info = zipfile.ZipInfo(relative, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as source, archive.open(info, "w") as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            records.append({"path": relative, "size_bytes": size, "sha256": digest.hexdigest()})
    return records


def _capture_environment() -> tuple[dict, str]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=False)
    metadata = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pip_freeze_returncode": freeze.returncode,
    }
    return metadata, freeze.stdout + ("\n[pip stderr]\n" + freeze.stderr if freeze.stderr else "")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_snapshot(staging: Path, args: argparse.Namespace,
                    preflight_result: dict, command: list[str],
                    environment: dict, packages: str) -> dict:
    staging.mkdir(parents=True, exist_ok=False)
    try:
        code_records = write_source_snapshot(PROJECT_ROOT, staging / "source.zip")
        source_archive_hash = sha256_file(staging / "source.zip")
        (staging / "source_manifest.json").write_text(json.dumps({
            "project_files": code_records,
            "archive_sha256": source_archive_hash,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        split_bytes = Path(preflight_result["split_file"]).read_bytes()
        (staging / "split_manifest.json").write_bytes(split_bytes)
        config_record = _stable_file_record(args.config, "config")
        shutil.copyfile(args.config, staging / "config.yaml")
        copied_config_hash = sha256_file(staging / "config.yaml")
        if copied_config_hash != config_record["sha256"]:
            raise RuntimeError(f"Config changed while snapshotting: {args.config}")
        input_records = [config_record]
        for role, path in (("world_checkpoint", args.world_checkpoint), ("resume", args.resume)):
            if path is not None:
                input_records.append(_stable_file_record(path, role))
        (staging / "training_inputs.json").write_text(json.dumps({
            "inputs": input_records,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        data_root = Path(preflight_result["data_root"])
        split = preflight_result["split"]
        dataset_records = []
        for split_name in ("train", "calibration"):
            for sequence in split[split_name]:
                path = data_root / f"{sequence}.npz"
                before = path.stat()
                checked = preflight_result["archive_stats"][sequence]
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                        checked["size_bytes"], checked["mtime_ns"], checked["inode"]):
                    raise RuntimeError(f"Dataset changed after preflight: {path}")
                digest = sha256_file(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                        after.st_size, after.st_mtime_ns, after.st_ino):
                    raise RuntimeError(f"Dataset changed while hashing: {path}")
                dataset_records.append({
                    "split": split_name,
                    "sequence": sequence,
                    "path": str(path),
                    "size_bytes": before.st_size,
                    "sha256": digest,
                })
        (staging / "dataset_sha256.json").write_text(json.dumps({
            "split_file_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "archives": dataset_records,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "args.json").write_text(json.dumps({
            "arguments": vars(args),
            "preflight": {key: value for key, value in preflight_result.items() if key != "split"},
            "command": command,
        }, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "environment.txt").write_text(packages, encoding="utf-8")
        return {
            "source_sha256": source_archive_hash,
            "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "dataset_archive_count": len(dataset_records),
            "config_sha256": config_record["sha256"],
            "world_checkpoint_sha256": next(
                (record["sha256"] for record in input_records
                 if record["role"] == "world_checkpoint"), None),
            "resume_sha256": next(
                (record["sha256"] for record in input_records if record["role"] == "resume"), None),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_launch_args(args: argparse.Namespace) -> None:
    if min(args.epochs, args.steps_per_epoch, args.validation_steps) < 1:
        raise ValueError("Epoch and step counts must be positive")
    if (args.window < 2 or not np.isfinite(args.learning_rate)
            or not np.isfinite(args.quality_error_m)
            or args.learning_rate <= 0 or args.quality_error_m <= 0):
        raise ValueError("Window, learning rate, and quality tolerance must be positive")
    if not args.gpu.strip():
        raise ValueError("--gpu must name at least one physical GPU")
    for name, path in (("config", args.config), ("world checkpoint", args.world_checkpoint),
                       ("resume checkpoint", args.resume)):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_launch_args(args)
    result = preflight(args.data_root, args.split_file, args.window,
                       repository_root=REPOSITORY_ROOT)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"split", "archive_stats"}},
                     indent=2, sort_keys=True))
    if args.check_only:
        return 0

    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    command = training_command(args, run_dir)
    environment, packages = _capture_environment()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=f".{run_dir.name}.snapshot-", dir=run_dir.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        hashes = _write_snapshot(staging, args, result, command, environment, packages)
        run_dir.mkdir(exist_ok=False)
        try:
            for path in staging.iterdir():
                shutil.move(str(path), str(run_dir / path.name))
            staging.rmdir()
            process_environment = os.environ.copy()
            process_environment.update(
                CUDA_VISIBLE_DEVICES=args.gpu,
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                PYTHONUNBUFFERED="1",
            )
            with (run_dir / "train.log").open("xb", buffering=0) as log:
                process = subprocess.Popen(
                    command, cwd=REPOSITORY_ROOT, env=process_environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            (run_dir / "train.pid").write_text(f"{process.pid}\n", encoding="utf-8")
            launch = {
                "stage": "dual_loop",
                "status": "spawned",
                "pid": process.pid,
                "physical_gpu": args.gpu,
                "command": command,
                **hashes,
                "training_log": str(run_dir / "train.log"),
                "split_manifest": str(Path(result["split_file"])),
            }
            (run_dir / "launch.json").write_text(
                json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as error:
            (run_dir / "launch.json").write_text(json.dumps({
                "stage": "dual_loop", "status": "launch_failed", "error": str(error),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise
    print(json.dumps(launch, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
