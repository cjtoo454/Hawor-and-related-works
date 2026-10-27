"""Load the fixed HOT3D train/calibration split for intra-frame visual training.

The prepared archives contain raw HaWoR observations and GT labels in the same
per-interval SLAM world. This module keeps those sources separate: observation
records expose RGB/camera/raw-state data, while labels are fetched only by the
loss path in ``train_visual.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRY = ROOT / "thirdparty" / "try"
sys.path[:0] = [str(ROOT), str(TRY), str(TRY / "scripts")]

from dual_loop.handloop_state import legacy_to_state  # noqa: E402
from dual_loop.io import _load_right_hand_tracks, _resolve_tracks_path  # noqa: E402
from prepare_handloop_data import fill_boxes  # noqa: E402


DEFAULT_DATA_ROOT = TRY / "data" / "prepared_train100"
DEFAULT_SPLIT = TRY / "data" / "raw_trajectories" / "split_manifest.json"
DEFAULT_EXPORT_SPLIT = ROOT / "datasets" / "hot3d_train100_export" / "training_split.json"
DEFAULT_VIDEO_ROOT = ROOT / "datasets" / "hot3d_train100_export"
DEFAULT_PROTOCOL = HERE / "records" / "protocol.json"


@dataclass
class PreparedSequence:
    """One verified sequence with GT retained outside the model-input record."""

    sequence: str
    split: str
    archive_path: Path
    image_paths: np.ndarray
    boxes: np.ndarray
    image_size_wh: np.ndarray
    focal_length: np.ndarray
    timestamp_sec: np.ndarray
    segment_id: np.ndarray
    rotation_w2c: np.ndarray
    translation_w2c: np.ndarray
    visual_valid: np.ndarray
    gt_valid: np.ndarray
    visual_legacy: np.ndarray
    target_legacy: np.ndarray
    focal_filled_count: int = 0
    raw_world: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.timestamp_sec)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON mapping {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_protocol(protocol_path: Path, experiment: dict[str, Any]) -> dict[str, Any]:
    """Bind the recorded experiment protocol to its frozen file and baseline."""
    frozen_path = Path(experiment.get("protocol", ""))
    expected_hash = experiment.get("protocol_sha256")
    if not frozen_path.is_file() or not isinstance(expected_hash, str):
        raise ValueError("Experiment protocol must name its frozen scoring protocol and SHA256")
    actual_hash = _sha256(frozen_path)
    if actual_hash != expected_hash:
        raise ValueError("Frozen scoring protocol SHA256 differs from records/protocol.json")
    frozen = _load_json(frozen_path)
    if frozen.get("test_sequences") != experiment.get("test_sequences"):
        raise ValueError("Held-out sequence list differs from the frozen scoring protocol")
    if (experiment.get("base_models_frozen") is not True
            or experiment.get("trainable_component") != "HandLoopVisual only"
            or experiment.get("rounds") != 2
            or experiment.get("reobserve") is not True
            or experiment.get("corrected_state_feedback") is not False):
        raise ValueError("Experiment protocol fields do not match the visual-only raw-history ablation")

    baseline_path = Path(experiment.get("baseline_result", ""))
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Missing first-row baseline result: {baseline_path}")
    baseline = _load_json(baseline_path)
    if (baseline.get("checkpoint_sha256") != experiment.get("base_twm_sha256")
            or baseline.get("hawor_checkpoint_sha256") != experiment.get("base_hawor_sha256")
            or baseline.get("checkpoint_stage") != "motion_pretrain"
            or int(baseline.get("checkpoint_global_step", 0)) != 5000
            or baseline.get("method") != "TWM only; raw HaWoR feedback"
            or int(baseline.get("num_videos", 0)) != len(experiment["test_sequences"])):
        raise ValueError("First-row baseline does not use the frozen 5000-step TWM/HaWoR weights")
    summary_path = baseline_path.parent / "onset" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing stored first-row onset summary: {summary_path}")
    summary = _load_json(summary_path)
    summary_hash = _sha256(summary_path)
    if summary.get("protocol_sha256") != actual_hash:
        raise ValueError("First-row onset summary was computed with a different frozen protocol")
    if (summary.get("test_sequences") not in (None, experiment.get("test_sequences"))
            or int(summary.get("num_videos", 0)) != len(experiment["test_sequences"])):
        raise ValueError("First-row onset summary does not cover the frozen held-out sequence list")
    threshold = frozen.get("prediction_threshold")
    if (threshold is None or baseline.get("prediction_threshold") != threshold
            or summary.get("prediction_threshold") != threshold
            or summary.get("reference_threshold") != frozen.get("reference_threshold")):
        raise ValueError("First-row onset summary thresholds differ from the frozen protocol")
    baseline_onset = float(baseline["onset_mae_ms"])
    summary_onset = float(summary["aggregate"]["onset_mae_ms"])
    if not math.isclose(baseline_onset, summary_onset, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Stored first-row onset metric disagrees with its detailed onset summary")
    return {
        "frozen_protocol_sha256": actual_hash,
        "first_row_onset_summary_sha256": summary_hash,
        "first_row_onset_mae_ms": baseline_onset,
        "first_row_prediction_threshold": float(threshold),
    }


def validate_split_manifests(
    split_manifest: Path = DEFAULT_SPLIT,
    export_split: Path = DEFAULT_EXPORT_SPLIT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    verify_frozen_protocol: bool = True,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return the exact permitted train/calibration lists after leakage checks."""
    split = _load_json(split_manifest)
    export = _load_json(export_split)
    protocol = _load_json(protocol_path)
    if verify_frozen_protocol:
        protocol["_validated_evidence"] = validate_frozen_protocol(protocol_path, protocol)
    train = list(split.get("train", []))
    calibration = list(split.get("calibration", []))
    test = list(protocol.get("test_sequences", []))
    if not train or not calibration or not test:
        raise ValueError("Train, calibration, and held-out test sequence lists must be non-empty")
    if train != list(export.get("train", [])) or calibration != list(export.get("calibration", [])):
        raise ValueError("Prepared-data and HOT3D export split manifests disagree")
    if len(set(train)) != len(train) or len(set(calibration)) != len(calibration):
        raise ValueError("A sequence appears more than once in a training split")
    groups = {"train": train, "calibration": calibration, "test": test}
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        overlap = set(groups[left]) & set(groups[right])
        if overlap:
            raise ValueError(f"Sequence leakage between {left} and {right}: {sorted(overlap)[:3]}")
        left_participants = {name.split("_", 1)[0] for name in groups[left]}
        right_participants = {name.split("_", 1)[0] for name in groups[right]}
        participant_overlap = left_participants & right_participants
        if participant_overlap:
            raise ValueError(
                f"Participant leakage between {left} and {right}: {sorted(participant_overlap)}")
    return train, calibration, protocol


def _read_detector_boxes(
    sequence: str,
    frame_count: int,
    image_size_wh: np.ndarray,
    segment_id: np.ndarray,
    video_root: Path,
) -> np.ndarray:
    sequence_root = Path(video_root) / sequence
    track_path, oracle_assisted = _resolve_tracks_path(sequence_root, allow_gt_boxes=False)
    if oracle_assisted:
        raise ValueError(f"GT-projected detections are forbidden for training inputs: {sequence}")
    tracks = _load_right_hand_tracks(track_path)
    if len(tracks) != frame_count:
        raise ValueError(
            f"{sequence}: detector track length {len(tracks)} != prepared frames {frame_count}")
    boxes = np.full((frame_count, 4), np.nan, dtype=np.float32)
    for frame, track in enumerate(tracks):
        if not isinstance(track, dict) or not track.get("det", False):
            continue
        candidate = np.asarray(track.get("det_box", []), dtype=np.float32)
        if candidate.size == 0 or candidate.shape[-1] < 4:
            continue
        box = candidate.reshape(-1, candidate.shape[-1])[0, :4]
        if np.isfinite(box).all() and np.all(box[2:] > box[:2]):
            boxes[frame] = box
    return fill_boxes(boxes, image_size_wh, segment_id)


def load_sequence(
    sequence: str,
    split: str,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    video_root: Path = DEFAULT_VIDEO_ROOT,
    check_images: bool = True,
) -> PreparedSequence:
    """Load one verified archive and its detector-only crop boxes."""
    archive_path = Path(data_root) / f"{sequence}.npz"
    required = {
        "sequence", "timestamp_sec", "segment_id", "visual_state", "gt_state",
        "visual_valid", "gt_valid", "rotation_w2c", "translation_w2c",
        "image_size", "image_paths", "focal_length", "coordinate_frame",
        "oracle_assisted", "oracle_assisted_status", "input_provenance_json",
    }
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{sequence}: prepared archive is missing {missing}")
        data = {key: archive[key].copy() for key in required}
    if str(data["sequence"].item()) != sequence:
        raise ValueError(f"Prepared archive name mismatch: {archive_path}")
    if str(data["coordinate_frame"].item()) != "per-interval local raw HaWoR SLAM world":
        raise ValueError(f"{sequence}: unexpected prepared coordinate frame")
    if bool(data["oracle_assisted"].item()) or str(data["oracle_assisted_status"].item()) != "verified_no_oracle":
        raise ValueError(f"{sequence}: oracle-assisted/unknown prepared input is forbidden")
    try:
        provenance = json.loads(str(data["input_provenance_json"].item()))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{sequence}: invalid frontend provenance") from error
    if (provenance.get("manifest_status") != "verified"
            or provenance.get("oracle_assisted") is not False
            or provenance.get("bbox_source") != "detected_tracks"):
        raise ValueError(f"{sequence}: raw frontend/detector provenance is not verified")

    timestamps = np.asarray(data["timestamp_sec"], dtype=np.float64)
    segments = np.asarray(data["segment_id"], dtype=np.int64)
    visual_valid = np.asarray(data["visual_valid"], dtype=bool)
    gt_valid = np.asarray(data["gt_valid"], dtype=bool)
    visual_legacy = np.asarray(data["visual_state"], dtype=np.float32)
    target_legacy = np.asarray(data["gt_state"], dtype=np.float32)
    frame_count = len(timestamps)
    if frame_count < 1 or timestamps.shape != (frame_count,) or segments.shape != (frame_count,):
        raise ValueError(f"{sequence}: malformed frame/time arrays")
    if (visual_valid.shape != (frame_count,) or gt_valid.shape != (frame_count,)
            or visual_legacy.shape != (frame_count, 109)
            or target_legacy.shape != (frame_count, 109)):
        raise ValueError(f"{sequence}: expected aligned legacy state/validity arrays")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{sequence}: timestamps must be finite and strictly increasing")
    if np.any(visual_valid & ~np.isfinite(visual_legacy).all(-1)):
        raise ValueError(f"{sequence}: valid raw HaWoR state contains non-finite values")
    if np.any(gt_valid & ~np.isfinite(target_legacy).all(-1)):
        raise ValueError(f"{sequence}: valid GT state contains non-finite values")

    image_size = np.asarray(data["image_size"], dtype=np.float32)
    if image_size.shape != (2,) or np.any(image_size <= 0):
        raise ValueError(f"{sequence}: invalid [width,height] image size")
    image_paths = np.asarray(data["image_paths"]).astype(str)
    if image_paths.shape != (frame_count,):
        raise ValueError(f"{sequence}: image path count differs from frame count")
    if check_images:
        missing_images = [path for path in image_paths if not Path(path).is_file()]
        if missing_images:
            raise FileNotFoundError(f"{sequence}: {len(missing_images)} image files are missing; first={missing_images[0]}")

    boxes = _read_detector_boxes(sequence, frame_count, image_size, segments, video_root)
    focal_path = Path(video_root) / sequence / "focal.txt"
    if not focal_path.is_file():
        raise FileNotFoundError(f"{sequence}: missing sensor focal calibration {focal_path}")
    try:
        sensor_focal = float(focal_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise ValueError(f"{sequence}: invalid sensor focal calibration {focal_path}") from error
    prepared_focal = np.asarray(data["focal_length"], dtype=np.float32)
    if prepared_focal.shape != (frame_count,):
        raise ValueError(f"{sequence}: malformed prepared focal array")
    prepared_focal_valid = np.isfinite(prepared_focal)
    if (not np.isfinite(sensor_focal) or sensor_focal <= 0
            or not prepared_focal_valid.any()
            or not np.allclose(prepared_focal[prepared_focal_valid], sensor_focal,
                               rtol=1e-5, atol=1e-3)):
        raise ValueError(f"{sequence}: prepared focal values disagree with sensor focal.txt")
    focal_filled_count = int((~prepared_focal_valid | (prepared_focal <= 0)).sum())
    focal_length = np.full(frame_count, sensor_focal, dtype=np.float32)
    rotation_w2c = np.asarray(data["rotation_w2c"], dtype=np.float32)
    translation_w2c = np.asarray(data["translation_w2c"], dtype=np.float32)
    if (focal_length.shape != (frame_count,) or rotation_w2c.shape != (frame_count, 3, 3)
            or translation_w2c.shape != (frame_count, 3)):
        raise ValueError(f"{sequence}: malformed camera calibration arrays")
    if not np.isfinite(boxes).all():
        raise ValueError(f"{sequence}: detector crop boxes must be finite after previous-only fill")

    return PreparedSequence(
        sequence=sequence,
        split=split,
        archive_path=archive_path,
        image_paths=image_paths,
        boxes=boxes,
        image_size_wh=image_size,
        focal_length=focal_length,
        timestamp_sec=timestamps,
        segment_id=segments,
        rotation_w2c=rotation_w2c,
        translation_w2c=translation_w2c,
        visual_valid=visual_valid,
        gt_valid=gt_valid,
        visual_legacy=visual_legacy,
        target_legacy=target_legacy,
        focal_filled_count=focal_filled_count,
    )


def load_sequences(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    video_root: Path = DEFAULT_VIDEO_ROOT,
    split_manifest: Path = DEFAULT_SPLIT,
    export_split: Path = DEFAULT_EXPORT_SPLIT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    splits: tuple[str, ...] = ("train", "calibration"),
    check_images: bool = True,
) -> tuple[dict[str, list[PreparedSequence]], dict[str, Any]]:
    """Load only requested split archives, after validating held-out isolation."""
    train_names, calibration_names, protocol = validate_split_manifests(
        split_manifest, export_split, protocol_path)
    split_names = {"train": train_names, "calibration": calibration_names}
    invalid = set(splits) - set(split_names)
    if invalid:
        raise ValueError(f"Unsupported data split(s): {sorted(invalid)}")
    loaded: dict[str, list[PreparedSequence]] = {name: [] for name in splits}
    for split in splits:
        for sequence in split_names[split]:
            loaded[split].append(load_sequence(
                sequence, split, data_root=data_root, video_root=video_root,
                check_images=check_images))
    return loaded, protocol


def convert_raw_history_states(
    sequences: list[PreparedSequence],
    geometry: torch.nn.Module,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> None:
    """Convert original HaWoR history to actual-wrist 99D states in memory."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    geometry.eval()
    for sequence in sequences:
        output = np.zeros((len(sequence), 99), dtype=np.float32)
        indices = np.flatnonzero(sequence.visual_valid)
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                selected = indices[start:start + batch_size]
                legacy = torch.as_tensor(
                    sequence.visual_legacy[selected], dtype=torch.float32, device=device)
                output[selected] = legacy_to_state(legacy, geometry).cpu().numpy()
        if not np.isfinite(output[indices]).all():
            raise ValueError(f"{sequence.sequence}: actual-wrist raw history conversion failed")
        sequence.raw_world = output


def context_start_indices(sequence: PreparedSequence) -> np.ndarray:
    """Return each frame's inference-aligned 16-frame HaWoR block start."""
    segments = sequence.segment_id
    starts = np.flatnonzero(np.r_[True, segments[1:] != segments[:-1]])
    frame_ids = np.arange(len(sequence), dtype=np.int64)
    segment_slots = np.searchsorted(starts, frame_ids, side="right") - 1
    segment_starts = starts[segment_slots]
    return segment_starts + ((frame_ids - segment_starts) // 16) * 16


def context_indices(sequence: PreparedSequence, frame_index: int) -> tuple[np.ndarray, int]:
    """Return the inference-aligned 16-frame HaWoR context without crossing SLAM resets."""
    if not 0 <= frame_index < len(sequence):
        raise IndexError(frame_index)
    segments = sequence.segment_id
    starts = np.flatnonzero(np.r_[True, segments[1:] != segments[:-1]])
    segment_slot = int(np.searchsorted(starts, frame_index, side="right") - 1)
    segment_start = int(starts[segment_slot])
    segment_stop = int(starts[segment_slot + 1]) if segment_slot + 1 < len(starts) else len(sequence)
    start = int(context_start_indices(sequence)[frame_index])
    stop = min(start + 16, segment_stop)
    ids = np.arange(start, stop, dtype=np.int64)
    if len(ids) < 16:
        ids = np.pad(ids, (0, 16 - len(ids)), mode="edge")
    return ids, int(frame_index - start)


def observation_record(
    sequence: PreparedSequence,
    frame_index: int,
) -> dict[str, Any]:
    """Expose model inputs for one frame; labels are deliberately absent."""
    ids, current_offset = context_indices(sequence, frame_index)
    record = {
        "sequence": sequence.sequence,
        "frame_index": int(frame_index),
        "context_indices": ids,
        "image_paths": sequence.image_paths[ids],
        "boxes": sequence.boxes[ids],
        "image_size_wh": sequence.image_size_wh,
        "focal_length": float(sequence.focal_length[frame_index]),
        "rotation_w2c": sequence.rotation_w2c[frame_index],
        "translation_w2c": sequence.translation_w2c[frame_index],
        "timestamp_sec": float(sequence.timestamp_sec[frame_index]),
        "segment_id": int(sequence.segment_id[frame_index]),
        "raw_available": bool(sequence.visual_valid[frame_index]),
        "image_available": True,
        "handedness": 1,
        "current_offset": current_offset,
    }
    return record


def history_indices(
    sequence: PreparedSequence,
    frame_index: int,
    history_length: int,
) -> list[int]:
    """Return the contiguous preceding raw-frame IDs after segment/gap resets."""
    if not 0 <= frame_index < len(sequence):
        raise IndexError(frame_index)
    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    if history_length == 0:
        return []
    selected: list[int] = []
    next_index = frame_index
    for candidate in range(frame_index - 1, max(-1, frame_index - history_length - 1), -1):
        gap = float(sequence.timestamp_sec[next_index] - sequence.timestamp_sec[candidate])
        if (sequence.segment_id[candidate] != sequence.segment_id[frame_index]
                or gap <= 0 or gap > 0.25):
            break
        selected.append(candidate)
        next_index = candidate
    selected.reverse()
    return selected


def calibration_samples(
    sequences: list[PreparedSequence],
) -> list[tuple[PreparedSequence, int]]:
    """Choose one fixed GT/raw-visual-valid frame per calibration video.

    The evaluation pipeline scores only frames with a valid raw visual
    prediction, so calibration uses the same frame eligibility while still
    selecting the frame by GT availability alone within that eligible set.
    """
    samples: list[tuple[PreparedSequence, int]] = []
    for sequence in sequences:
        candidates = np.flatnonzero(sequence.gt_valid & sequence.visual_valid)
        if not len(candidates):
            raise ValueError(
                f"{sequence.sequence}: calibration sequence has no frame with both GT and raw visual state")
        samples.append((sequence, int(candidates[len(candidates) // 2])))
    if not samples:
        raise ValueError("Calibration split has no representative frames")
    return samples


def _check_only(args: argparse.Namespace) -> None:
    groups, protocol = load_sequences(
        data_root=args.data_root, video_root=args.video_root,
        split_manifest=args.split_manifest, export_split=args.export_split,
        protocol_path=args.protocol, check_images=not args.skip_image_check)
    report = {
        "train_sequences": len(groups["train"]),
        "calibration_sequences": len(groups["calibration"]),
        "test_sequences_excluded": len(protocol["test_sequences"]),
        "train_frames": sum(len(sequence) for sequence in groups["train"]),
        "calibration_frames": sum(len(sequence) for sequence in groups["calibration"]),
        "train_gt_valid": sum(int(sequence.gt_valid.sum()) for sequence in groups["train"]),
        "train_visual_valid": sum(int(sequence.visual_valid.sum()) for sequence in groups["train"]),
        "focal_entries_filled_from_sensor_calibration": sum(
            sequence.focal_filled_count for sequence in groups["train"] + groups["calibration"]),
        "bbox_source": "detected_tracks with previous-only fill",
        "coordinate_frame": "per-interval local raw HaWoR SLAM world",
        "protocol_evidence": protocol.get("_validated_evidence"),
    }
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate split, provenance, crops, and image availability")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--export-split", type=Path, default=DEFAULT_EXPORT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        parser.error("This script is a data-validation module; pass --check-only")
    _check_only(args)


if __name__ == "__main__":
    main()
