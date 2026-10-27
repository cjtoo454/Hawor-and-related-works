#!/usr/bin/env python3
"""Convert the raw HOT3D MANO training ZIPs to 30 Hz world-state sequences.

Only ``mano_hand_pose_trajectory.jsonl`` is read. No images or exported video
ground truth are used, so this stage remains a genuine sensor-trajectory prior
pretraining set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from dual_loop.state import STATE_DIM  # noqa: E402


DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "datasets" / "hot3d_train100"
DEFAULT_VAL_ROOT = REPOSITORY_ROOT / "datasets" / "hot3d_valset_export"
DEFAULT_MANO_ASSET = PROJECT_ROOT / "assets" / "mano_right.npz"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw_trajectories"
MANO_MEMBER = "mano_hand_pose_trajectory.jsonl"
DEFAULT_FPS = 30.0


class EmptyHandTrajectory(ValueError):
    """Raised when a hand-data ZIP contains no MANO trajectory rows."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare real HOT3D MANO trajectories for world-model training")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--val_root", type=Path, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--mano_asset", type=Path, default=DEFAULT_MANO_ASSET)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max_sequences", type=int, default=100)
    parser.add_argument("--history_length", type=int, default=16)
    parser.add_argument("--max_horizon", type=int, default=10)
    parser.add_argument("--calibration_participants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def participant_id(sequence: str) -> str:
    return sequence.split("_", 1)[0]


def _hand_record(record: dict[str, Any], sequence: str) -> dict[str, Any] | None:
    hand = record.get("hand_poses", {}).get("1")
    if hand is None:
        return None
    pose = np.asarray(hand.get("pose", []), dtype=np.float64)
    wrist = hand.get("wrist_xform", {})
    quaternion = np.asarray(wrist.get("q_wxyz", []), dtype=np.float64)
    translation = np.asarray(wrist.get("t_xyz", []), dtype=np.float64)
    betas = np.asarray(hand.get("betas", []), dtype=np.float64)
    if pose.shape != (15,) or quaternion.shape != (4,) or translation.shape != (3,):
        raise ValueError(f"Malformed right-hand MANO parameters in {sequence}")
    if betas.shape != (10,):
        raise ValueError(f"Expected 10 MANO betas in {sequence}, got {betas.shape}")
    if not all(np.isfinite(value).all() for value in (pose, quaternion, translation, betas)):
        raise ValueError(f"Non-finite MANO parameters in {sequence}")
    timestamp_ns = int(record["timestamp_ns"])
    if timestamp_ns < 0:
        raise ValueError(f"Negative timestamp in {sequence}: {timestamp_ns}")
    return {
        "timestamp_ns": timestamp_ns,
        "pose_pca": pose,
        "q_wxyz": quaternion,
        "t_xyz": translation,
        "betas": betas,
    }


def read_right_hand_trajectory(zip_path: Path, sequence: str) -> list[dict[str, Any]]:
    """Read and timestamp-sort right-hand samples from one raw data ZIP."""
    with zipfile.ZipFile(zip_path) as archive:
        try:
            info = archive.getinfo(MANO_MEMBER)
        except KeyError as error:
            raise ValueError(f"{MANO_MEMBER} missing from {zip_path}") from error
        if info.file_size == 0:
            raise EmptyHandTrajectory(f"empty {MANO_MEMBER}")
        samples_by_timestamp: dict[int, dict[str, Any]] = {}
        with archive.open(MANO_MEMBER) as source:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {zip_path}:{MANO_MEMBER}:{line_number}") from error
                sample = _hand_record(record, sequence)
                if sample is not None:
                    # HOT3D's official loader keeps the first record for a timestamp.
                    samples_by_timestamp.setdefault(sample["timestamp_ns"], sample)
    if not samples_by_timestamp:
        raise EmptyHandTrajectory("no right-hand (hand_poses['1']) records")
    return [samples_by_timestamp[key] for key in sorted(samples_by_timestamp)]


def sample_nearest_30hz(
    samples: list[dict[str, Any]], fps: float = DEFAULT_FPS,
) -> dict[str, np.ndarray]:
    """Select source rows nearest to a 30 Hz grid, leaving uncovered ticks invalid."""
    if not samples:
        raise ValueError("Cannot resample an empty trajectory")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    period_ns = 1_000_000_000.0 / fps
    max_offset_ns = int(math.ceil(period_ns / 2.0))
    source_times = np.asarray([sample["timestamp_ns"] for sample in samples], dtype=np.int64)
    duration_ns = int(source_times[-1] - source_times[0])
    count = int(math.floor(duration_ns * fps / 1_000_000_000.0)) + 1
    ticks = source_times[0] + np.rint(
        np.arange(count, dtype=np.float64) * period_ns).astype(np.int64)

    right = np.searchsorted(source_times, ticks, side="left")
    after = np.minimum(right, len(source_times) - 1)
    before = np.maximum(right - 1, 0)
    after_distance = np.abs(source_times[after] - ticks)
    before_distance = np.abs(source_times[before] - ticks)
    # On an exact tie, keep the earlier source timestamp for deterministic output.
    choose_before = before_distance <= after_distance
    selected = np.where(choose_before, before, after)
    selected_distance = np.minimum(before_distance, after_distance)
    valid = selected_distance <= max_offset_ns

    # A sparse source must not be duplicated to manufacture multiple target frames.
    used_indices: set[int] = set()
    for index in range(len(selected)):
        source_index = int(selected[index])
        if not valid[index]:
            continue
        if source_index in used_indices:
            valid[index] = False
            continue
        used_indices.add(source_index)

    timestamps_ns = np.full(len(ticks), -1, dtype=np.int64)
    timestamps_ns[valid] = source_times[selected[valid]]
    dt_seconds = np.full(len(ticks), np.nan, dtype=np.float32)
    dt_ok = np.zeros(len(ticks), dtype=bool)
    if len(ticks) > 1:
        adjacent = valid[1:] & valid[:-1]
        dt_ns = timestamps_ns[1:] - timestamps_ns[:-1]
        dt_seconds[1:][adjacent] = (dt_ns[adjacent] / 1e9).astype(np.float32)
        # The selected observations should still represent adjacent 30 Hz frames.
        dt_ok[1:] = adjacent & (dt_ns > 0) & (dt_ns <= int(1.5 * period_ns))

    return {
        "target_timestamps_ns": ticks.astype(np.int64),
        "timestamps_ns": timestamps_ns,
        "source_indices": selected.astype(np.int64),
        "source_offset_ns": np.where(valid, selected_distance, -1).astype(np.int64),
        "valid": valid.astype(bool),
        "dt_seconds": dt_seconds,
        "dt_ok": dt_ok,
    }


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternion(s) to rotation matrix(s)."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected q_wxyz with last dimension 4, got {q.shape}")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("MANO wrist quaternion has zero norm")
    q = q / norms
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    )
    return matrix.reshape(*q.shape[:-1], 3, 3).astype(np.float32)


def load_mano_pca(mano_asset: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the official use_pca=True right-hand basis and mean (no pickle)."""
    if not mano_asset.is_file():
        raise FileNotFoundError(
            f"MANO PCA asset missing: {mano_asset}. Run scripts/prepare_mano_assets.py first.")
    with np.load(mano_asset, allow_pickle=False) as archive:
        if "hands_components" not in archive or "hands_mean" not in archive:
            raise ValueError(
                f"Expected hands_components and hands_mean in {mano_asset}; got {archive.files}")
        components = np.asarray(archive["hands_components"], dtype=np.float32)
        mean = np.asarray(archive["hands_mean"], dtype=np.float32)
    if components.ndim != 2 or components.shape[1] != 45 or components.shape[0] < 15:
        raise ValueError(f"Expected MANO PCA components [>=15,45], got {components.shape}")
    if mean.shape != (45,):
        raise ValueError(f"Expected MANO use_pca=True hand mean [45], got {mean.shape}")
    return components[:15], mean


def make_states(
    samples: list[dict[str, Any]], source_indices: np.ndarray, valid: np.ndarray,
    components: np.ndarray, pose_mean: np.ndarray,
) -> np.ndarray:
    """Expand the official 15-D MANO PCA parameters and pack 109-D world states."""
    from torch import from_numpy

    from dual_loop.state import axis_angle_to_matrix, pack_state

    states = np.full((len(source_indices), STATE_DIM), np.nan, dtype=np.float32)
    output_rows = np.flatnonzero(valid)
    if not len(output_rows):
        return states
    picked = [samples[int(source_indices[index])] for index in output_rows]
    pose_pca = np.stack([sample["pose_pca"] for sample in picked]).astype(np.float32)
    full_pose = pose_pca @ components + pose_mean[None]
    hand_axis_angle = from_numpy(full_pose.reshape(-1, 15, 3))
    hand_rotation = axis_angle_to_matrix(hand_axis_angle)
    root_rotation = from_numpy(quaternion_wxyz_to_matrix(
        np.stack([sample["q_wxyz"] for sample in picked])))[:, None]
    translation = from_numpy(np.stack([sample["t_xyz"] for sample in picked]).astype(np.float32))
    betas = from_numpy(np.stack([sample["betas"] for sample in picked]).astype(np.float32))
    packed = pack_state(translation, root_rotation, hand_rotation, betas).numpy()
    states[output_rows] = packed.astype(np.float32)
    return states


def _zip_path(data_root: Path, sequence: str) -> Path:
    candidates = sorted((data_root / sequence).glob("*_hand_data.zip"))
    if len(candidates) != 1:
        if not candidates:
            raise FileNotFoundError(f"No *_hand_data.zip under {data_root / sequence}")
        raise ValueError(f"Expected one hand-data ZIP for {sequence}, got {len(candidates)}")
    return candidates[0]


def _segments(valid: np.ndarray, dt_ok: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open continuous frame ranges; no window may span a gap."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        if not is_valid:
            if start is not None:
                segments.append((start, index))
                start = None
            continue
        if start is None:
            start = index
        elif not dt_ok[index]:
            segments.append((start, index))
            start = index
    if start is not None:
        segments.append((start, len(valid)))
    return segments


def _assert_no_val_participant_overlap(
    sequences: list[str], val_root: Path,
) -> list[str]:
    val_file = val_root / "val.json"
    if not val_file.is_file():
        raise FileNotFoundError(f"Expected official participant list {val_file}")
    val_sequences = json.loads(val_file.read_text(encoding="utf-8"))
    train_participants = {participant_id(sequence) for sequence in sequences}
    val_participants = {participant_id(sequence) for sequence in val_sequences}
    overlap = sorted(train_participants & val_participants)
    if overlap:
        raise ValueError(f"Training sequences overlap official val participants: {overlap}")
    return sorted(val_participants)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.fps != DEFAULT_FPS:
        # This training recipe and temporal filters are explicitly calibrated at 30 Hz.
        raise ValueError(f"This stage requires fps={DEFAULT_FPS}, got {args.fps}")
    if args.max_sequences <= 0 or args.history_length <= 0 or args.max_horizon <= 0:
        raise ValueError("sequence, history, and horizon limits must be positive")
    train_file = args.data_root / "train.json"
    if not train_file.is_file():
        raise FileNotFoundError(train_file)
    sequence_ids = json.loads(train_file.read_text(encoding="utf-8"))
    selected = list(sequence_ids[: min(args.max_sequences, 100)])
    if len(selected) != min(args.max_sequences, 100):
        raise ValueError(
            f"Expected at least {min(args.max_sequences, 100)} train.json entries, got {len(selected)}")
    val_participants = _assert_no_val_participant_overlap(selected, args.val_root)
    components, pose_mean = load_mano_pca(args.mano_asset)
    args.output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for sequence in selected:
        record: dict[str, Any] = {
            "sequence": sequence,
            "participant": participant_id(sequence),
            "status": "skipped",
        }
        try:
            archive_path = _zip_path(args.data_root, sequence)
            record["hand_data_zip"] = str(archive_path.resolve())
            samples = read_right_hand_trajectory(archive_path, sequence)
            sampled = sample_nearest_30hz(samples, args.fps)
            states = make_states(
                samples, sampled["source_indices"], sampled["valid"], components, pose_mean)
            segments = _segments(sampled["valid"], sampled["dt_ok"])
            eligible_frames = sum(stop - start for start, stop in segments
                                  if stop - start >= args.history_length + args.max_horizon)
            output_path = args.output_root / f"{sequence}.npz"
            np.savez_compressed(
                output_path,
                sequence=np.asarray(sequence),
                participant=np.asarray(participant_id(sequence)),
                fps=np.asarray(args.fps, dtype=np.float32),
                states=states,
                valid=sampled["valid"],
                timestamps_ns=sampled["timestamps_ns"],
                target_timestamps_ns=sampled["target_timestamps_ns"],
                dt_seconds=sampled["dt_seconds"],
                dt_ok=sampled["dt_ok"],
                source_offset_ns=sampled["source_offset_ns"],
            )
            record.update({
                "status": "prepared",
                "raw_samples": len(samples),
                "sampled_frames": int(len(sampled["valid"])),
                "valid_frames": int(sampled["valid"].sum()),
                "continuous_segments": segments,
                "eligible_frames": int(eligible_frames),
                "output": str(output_path.resolve()),
            })
        except EmptyHandTrajectory as error:
            record["reason"] = str(error)
        except FileNotFoundError as error:
            record["reason"] = str(error)
        records.append(record)
        print(f"[{record['status']}] {sequence}: {record.get('reason', record.get('valid_frames'))}")

    prepared = [record for record in records if record["status"] == "prepared"]
    usable_participants = sorted({
        record["participant"] for record in prepared if record["eligible_frames"] > 0
    })
    if len(usable_participants) < 2:
        raise ValueError(
            f"Need at least two participants with continuous training windows, got {usable_participants}")
    calibration_count = args.calibration_participants
    if calibration_count not in (1, 2) or calibration_count >= len(usable_participants):
        raise ValueError(
            "calibration_participants must be 1 or 2 and leave at least one train participant")
    # This rule is fixed before any model calibration metric is seen.
    calibration_participants = usable_participants[-calibration_count:]
    train_participants = [
        participant for participant in usable_participants
        if participant not in calibration_participants
    ]
    train_sequences = sorted(
        record["sequence"] for record in prepared
        if record["participant"] in train_participants and record["eligible_frames"] > 0)
    calibration_sequences = sorted(
        record["sequence"] for record in prepared
        if record["participant"] in calibration_participants and record["eligible_frames"] > 0)
    manifest = {
        "schema": "hot3d_raw_mano_30hz_v1",
        "fps": args.fps,
        "state_dim": STATE_DIM,
        "state_coordinates": "HOT3D session world (sensor supervision), meters",
        "raw_member": MANO_MEMBER,
        "handedness": "right (hand_poses['1'])",
        "mano_pca": {
            "components": "first 15 rows of hands_components",
            "expansion": "pose_pca @ hands_components[:15] + hands_mean",
            "asset": str(args.mano_asset.resolve()),
            "mean_source": "smplx MANO use_pca=True flat_hand_mean=False",
        },
        "sampling": {
            "method": "nearest source sample on 30 Hz grid",
            "max_offset_ns": int(math.ceil(0.5e9 / args.fps)),
            "repeated_source_sample": "invalidated",
            "continuous_dt_max_seconds": 1.5 / args.fps,
            "missing_ticks": "remain invalid and break training windows",
        },
        "selected_train_json_entries": selected,
        "val_participants_checked_from_val_json_only": val_participants,
        "train_participants": train_participants,
        "calibration_participants": calibration_participants,
        "calibration_split_rule": f"last {calibration_count} usable participant IDs lexicographically",
        "train_sequences": train_sequences,
        "calibration_sequences": calibration_sequences,
        "train": train_sequences,
        "calibration": calibration_sequences,
        "skipped_sequences": [record for record in records if record["status"] == "skipped"],
        "sequence_records": records,
        "history_length_for_eligibility": args.history_length,
        "max_horizon_for_eligibility": args.max_horizon,
        "seed": args.seed,
    }
    output_manifest = args.output_root / "split_manifest.json"
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Prepared {len(prepared)}/{len(selected)} sequences; "
        f"train participants={train_participants}; calibration={calibration_participants}; "
        f"empty/skipped={sum(record['status'] == 'skipped' for record in records)}")
    print(f"[manifest] {output_manifest}")
    return manifest


def main() -> None:
    args = parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
