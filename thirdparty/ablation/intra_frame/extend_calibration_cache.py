#!/usr/bin/env python3
"""Extend the v2 frozen feature cache with the v3 calibration event windows.

The parent cache is read-only. Its train items and unchanged calibration
contexts are referenced by absolute path and original SHA256; only new or
expanded calibration contexts run frozen HaWoR again.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRY = ROOT / "thirdparty" / "try"
TWM_ONLY = ROOT / "thirdparty" / "ablation" / "twm_only"
sys.path[:0] = [str(ROOT), str(TRY), str(TRY / "scripts"), str(TWM_ONLY), str(HERE)]

from dual_loop.handloop_runtime import build_system  # noqa: E402
from feedback_v2 import install_feedback_v2  # noqa: E402
from prepare_training import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPORT_SPLIT,
    DEFAULT_PROTOCOL,
    DEFAULT_SPLIT,
    DEFAULT_VIDEO_ROOT,
    PreparedSequence,
    context_start_indices,
    convert_raw_history_states,
    load_sequences,
)
from train_visual import (  # noqa: E402
    DEFAULT_HAWOR,
    DEFAULT_TWM,
    EXPECTED_HAWOR_SHA256,
    EXPECTED_TWM_SHA256,
    freeze_except_visual,
    load_base_twm,
    module_parameter_sha256,
    seed_everything,
    sha256_file as train_sha256,
)
from v2_cache import (  # noqa: E402
    CACHE_SCHEMA,
    CACHE_VERSION,
    _atomic_json,
    _atomic_torch_save,
    _inputs_for_row,
    _labels_for_frames,
    load_manifest,
)
import train_visual  # noqa: E402


EVENT_SCHEMA = "calibration_event_windows_v3"
PROTOCOL_REVISION = 3
TRAIN_ITEM_COUNT = 528
CALIBRATION_PARENT_CONTEXT_COUNT = 356
CALIBRATION_WINDOW_COUNT = 40
EXPECTED_FULL_VIDEO_REFERENCE_EVENTS = 224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-cache", type=Path,
        default=HERE / "runs" / "feedback_v2_train" / "cache")
    parser.add_argument(
        "--event-manifest", type=Path,
        default=HERE / "runs" / "calibration_events_v3" / "calibration_manifest.json")
    parser.add_argument(
        "--output-cache", type=Path,
        default=HERE / "runs" / "feedback_v3_train" / "cache")
    parser.add_argument("--twm-checkpoint", type=Path, default=DEFAULT_TWM)
    parser.add_argument("--hawor-checkpoint", type=Path, default=DEFAULT_HAWOR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--export-split", type=Path, default=DEFAULT_EXPORT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--plan-only", action="store_true",
                        help="Validate the parent/event manifests and print the extraction plan without writing files or loading models")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _resolve_from(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _context_start_for_frame(frame: int, starts: list[int], sequence: str) -> int:
    array = np.asarray(starts, dtype=np.int64)
    slot = int(np.searchsorted(array, int(frame), side="right") - 1)
    if slot < 0:
        raise ValueError(f"{sequence}: no required 16-frame context starts before frame {frame}")
    start = int(array[slot])
    if frame >= start + 16:
        raise ValueError(f"{sequence}: frame {frame} does not map to required context {start}")
    return start


def _window_frame_ids(window: dict[str, Any], *, source: str) -> np.ndarray:
    start = int(window["start_frame"])
    stop = int(window["stop_frame_exclusive"])
    if stop <= start:
        raise ValueError(f"Invalid {source} window range: {window}")
    frames = np.arange(start, stop, dtype=np.int64)
    expected_count = window.get("frame_count")
    if expected_count is not None and int(expected_count) != len(frames):
        raise ValueError(f"{source} window frame_count disagrees with its range: {window}")
    explicit = window.get("frame_indices")
    if explicit is not None and not np.array_equal(frames, np.asarray(explicit, dtype=np.int64)):
        raise ValueError(f"{source} window frame_indices disagree with its range: {window}")
    return frames


def _build_plan(
    original: dict[str, Any],
    event_manifest: dict[str, Any],
) -> dict[str, Any]:
    if original.get("schema") != CACHE_SCHEMA or original.get("version") != CACHE_VERSION:
        raise ValueError("Parent cache schema is not handloop_intra_v2_feature_cache version 1")
    if event_manifest.get("schema") != EVENT_SCHEMA:
        raise ValueError(f"Expected {EVENT_SCHEMA}, got {event_manifest.get('schema')!r}")
    if event_manifest.get("split") != "calibration" or event_manifest.get("sequence_count") != 20:
        raise ValueError("Event manifest must cover exactly the 20 calibration sequences")
    if event_manifest.get("coordinate_frame") != "per-interval local raw HaWoR SLAM world":
        raise ValueError("Event manifest coordinate frame differs from the frozen cache")
    if int(event_manifest.get("counts", {}).get("total_complete_gt_reference_events", -1)) \
            != EXPECTED_FULL_VIDEO_REFERENCE_EVENTS:
        raise ValueError(
            "Event manifest is not the corrected full-video GT reference set "
            f"({EXPECTED_FULL_VIDEO_REFERENCE_EVENTS} events expected)")
    if int(event_manifest.get("protocol_revision", -1)) != 2:
        raise ValueError("Event manifest does not use the corrected protocol revision 2")
    if not isinstance(event_manifest.get("prediction_eligibility"), dict):
        raise ValueError("Event manifest lacks the fixed full-video prediction eligibility rule")
    items = original.get("items", [])
    train_items = [item for item in items if item.get("split") == "train"]
    cal_items = [item for item in items if item.get("split") == "calibration"]
    if len(train_items) != TRAIN_ITEM_COUNT or len(cal_items) != CALIBRATION_PARENT_CONTEXT_COUNT:
        raise ValueError(
            f"Parent item counts changed: train={len(train_items)} cal={len(cal_items)}")
    if len(original.get("calibration_windows", [])) != CALIBRATION_WINDOW_COUNT:
        raise ValueError("Parent cache no longer contains the frozen 40 W-calibration windows")

    event_base = event_manifest.get("base_calibration_source", {})
    if (int(event_base.get("fixed_window_count", 0)) != CALIBRATION_WINDOW_COUNT
            or int(event_base.get("fixed_window_frames", 0)) != 128):
        raise ValueError("Event manifest is not anchored to the frozen 40 x 128 W windows")
    if event_base.get("cache_manifest_sha256") != event_manifest.get("_parent_manifest_sha256"):
        raise ValueError("Event manifest parent cache hash was not validated")
    if (event_base.get("baseline_reference_events")
            != event_manifest.get("counts", {}).get("baseline_reference_events_recomputed")):
        raise ValueError("Baseline event count in event manifest is inconsistent")

    event_sequences = set(event_manifest.get("sequence_names", []))
    parent_sequences = {str(item["sequence"]) for item in cal_items}
    if event_sequences != parent_sequences:
        raise ValueError("Event manifest sequence list differs from the parent calibration cache")

    old_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    old_contexts_by_sequence: dict[str, set[int]] = {}
    for item in cal_items:
        key = (str(item["sequence"]), int(item["context_start"]))
        if key in old_by_key:
            raise ValueError(f"Duplicate parent calibration context {key}")
        old_by_key[key] = item
        old_contexts_by_sequence.setdefault(key[0], set()).add(key[1])

    requested: dict[tuple[str, int], set[int]] = {}
    fixed_windows = event_base.get("fixed_windows", [])
    if len(fixed_windows) != CALIBRATION_WINDOW_COUNT:
        raise ValueError("Event manifest fixed_windows list must contain all 40 parent windows")
    expanded_windows = event_manifest.get("merged_windows", [])
    expected_event_window_count = int(event_manifest.get("counts", {}).get("merged_window_count", -1))
    if expected_event_window_count < 1 or len(expanded_windows) != expected_event_window_count:
        raise ValueError(
            "Merged event window count differs from its manifest: "
            f"{len(expanded_windows)} != {expected_event_window_count}")
    start_records = event_manifest.get("context_start_indices", {})

    for window in fixed_windows:
        sequence = str(window["sequence"])
        frames = _window_frame_ids(window, source="fixed W-MPJPE")
        starts = start_records[sequence]["required_context_start_indices"]
        derived = set()
        for frame in frames:
            context_start = _context_start_for_frame(int(frame), starts, sequence)
            requested.setdefault((sequence, context_start), set()).add(int(frame))
            derived.add(context_start)
        declared = {
            int(str(item_id).rsplit("ctx_", 1)[1])
            for item_id in window.get("context_item_ids", [])
        }
        if derived != declared:
            raise ValueError(f"{sequence}: fixed-window context ids differ from frame-derived ids")

    for window in expanded_windows:
        sequence = str(window["sequence"])
        frames = _window_frame_ids(window, source="expanded event")
        starts = start_records[sequence]["required_context_start_indices"]
        derived = set()
        for frame in frames:
            context_start = _context_start_for_frame(int(frame), starts, sequence)
            requested.setdefault((sequence, context_start), set()).add(int(frame))
            derived.add(context_start)

    for sequence in sorted(event_sequences):
        info = start_records.get(sequence)
        if not isinstance(info, dict):
            raise ValueError(f"Missing context_start_indices for {sequence}")
        required = {int(value) for value in info.get("required_context_start_indices", [])}
        baseline = {int(value) for value in info.get("baseline_context_start_indices", [])}
        new = {int(value) for value in info.get("new_context_start_indices", [])}
        if required != baseline | new or required != {
            context_start for name, context_start in requested if name == sequence}:
            raise ValueError(f"{sequence}: declared context-start union differs from W/event windows")
        if baseline != old_contexts_by_sequence.get(sequence, set()):
            raise ValueError(f"{sequence}: baseline context starts differ from the parent cache")
        if len(new) != int(info.get("new_context_count", -1)):
            raise ValueError(f"{sequence}: new context count disagrees with its manifest")

    expected_context_count = int(event_manifest["counts"].get(
        "required_unique_16_frame_contexts", -1))
    expected_new_context_count = int(event_manifest["counts"].get(
        "additional_16_frame_contexts_over_v2", -1))
    if len(requested) != expected_context_count:
        raise ValueError(
            f"Required context count differs from the event manifest: "
            f"{len(requested)} != {expected_context_count}")
    if sum(key not in old_by_key for key in requested) != expected_new_context_count:
        raise ValueError("New context count differs from the event manifest")
    requested_frames = {(sequence, frame) for (sequence, _start), frames in requested.items()
                        for frame in frames}
    expected_frames = int(event_manifest["counts"]["expanded_unique_frames"])
    if len(requested_frames) != expected_frames:
        raise ValueError(f"Expanded windows cover {len(requested_frames)} unique frames, expected {expected_frames}")

    context_rows = []
    reuse_count = expanded_count = new_count = 0
    for (sequence, context_start), frames in sorted(requested.items()):
        old = old_by_key.get((sequence, context_start))
        frame_ids = sorted(frames)
        if old is None:
            action = "extract_new"
            new_count += 1
        elif frame_ids == sorted(int(value) for value in old["frame_indices"]):
            action = "reuse_parent"
            reuse_count += 1
        else:
            action = "extract_expanded"
            expanded_count += 1
        context_rows.append({
            "sequence": sequence,
            "context_start": int(context_start),
            "frame_indices": frame_ids,
            "action": action,
            "parent_item_id": old["item_id"] if old else None,
        })
    return {
        "contexts": context_rows,
        "requested_frame_count": len(requested_frames),
        "parent_train_item_count": len(train_items),
        "parent_calibration_context_count": len(cal_items),
        "reuse_calibration_context_count": reuse_count,
        "expanded_calibration_context_count": expanded_count,
        "new_calibration_context_count": new_count,
        "total_calibration_context_count": len(context_rows),
        "fixed_window_count": len(fixed_windows),
        "expanded_event_window_count": len(expanded_windows),
        "full_video_reference_event_count": int(
            event_manifest["counts"]["total_complete_gt_reference_events"]),
    }


def _load_bound_manifests(
    original_cache: Path,
    event_manifest_path: Path,
    *,
    verify_parent_files: bool,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    original_cache = Path(original_cache).resolve()
    event_manifest_path = Path(event_manifest_path).resolve()
    original_manifest_path = original_cache / "manifest.json"
    original = load_manifest(original_cache, verify_files=verify_parent_files)
    event_manifest = _read_json(event_manifest_path)
    event_manifest_sha = sha256_file(event_manifest_path)
    parent_manifest_sha = sha256_file(original_manifest_path)

    event_base = event_manifest.get("base_calibration_source", {})
    referenced_parent = _resolve_from(event_base.get("cache_manifest", ""), event_manifest_path.parent)
    if referenced_parent != original_manifest_path.resolve():
        raise ValueError(f"Event manifest points at a different parent cache: {referenced_parent}")
    if event_base.get("cache_manifest_sha256") != parent_manifest_sha:
        raise ValueError("Event manifest parent cache SHA256 differs from the completed v2 cache")
    baseline_path = _resolve_from(event_base.get("baseline_result", ""), event_manifest_path.parent)
    if not baseline_path.is_file() or sha256_file(baseline_path) != event_base.get("baseline_result_sha256"):
        raise ValueError("Event manifest baseline result path/hash does not match the frozen v2 baseline")
    event_manifest["_parent_manifest_sha256"] = parent_manifest_sha
    return original, event_manifest, parent_manifest_sha, event_manifest_sha


def _verify_parent_source_inputs(
    args: argparse.Namespace,
    original: dict[str, Any],
    event_manifest: dict[str, Any],
) -> dict[str, str]:
    """Verify that split/protocol and calibration observation sources still match v2."""
    checks: dict[str, str] = {}
    source_hashes = original.get("source_hashes", {})
    recorded_files = source_hashes.get("protocol_and_splits", {})
    requested_files = {
        "split_manifest": args.split_manifest,
        "export_split": args.export_split,
        "protocol": args.protocol,
    }
    for name, requested_path in requested_files.items():
        record = recorded_files.get(name, {})
        path = Path(requested_path).resolve()
        if path != Path(record.get("path", "")).resolve() or not path.is_file():
            raise ValueError(f"{name} path differs from the frozen v2 source manifest: {path}")
        digest = sha256_file(path)
        if digest != record.get("sha256"):
            raise ValueError(f"{name} SHA256 differs from the frozen v2 source manifest: {path}")
        checks[name] = digest

    parent_sequences = source_hashes.get("sequences", {})
    for sequence, event_record in event_manifest.get("sequences", {}).items():
        parent_record = parent_sequences.get(sequence)
        if not isinstance(parent_record, dict):
            raise ValueError(f"{sequence}: no corresponding frozen v2 source record")
        archive_path = Path(event_record["source_archive"]).resolve()
        parent_archive = parent_record.get("archive", {})
        if (archive_path != Path(parent_archive.get("path", "")).resolve()
                or event_record.get("source_archive_sha256") != parent_archive.get("sha256")):
            raise ValueError(f"{sequence}: event and parent source archive provenance differs")
        for source_name in ("archive", "detector_tracks", "focal"):
            source_record = parent_record.get(source_name, {})
            source_path = Path(source_record.get("path", "")).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"{sequence}: missing {source_name} source {source_path}")
            digest = sha256_file(source_path)
            if digest != source_record.get("sha256"):
                raise ValueError(f"{sequence}: {source_name} SHA256 differs: {source_path}")
            checks[f"{sequence}:{source_name}"] = digest
    return checks


def _prepare_system(args: argparse.Namespace, device: torch.device,
                    original: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, str], str, str]:
    for path in (args.twm_checkpoint, args.hawor_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    twm_sha = sha256_file(args.twm_checkpoint)
    hawor_sha = sha256_file(args.hawor_checkpoint)
    if twm_sha != EXPECTED_TWM_SHA256 or hawor_sha != EXPECTED_HAWOR_SHA256:
        raise ValueError("Frozen TWM/HaWoR hashes do not match the parent v2 cache protocol")
    expected_base = {"twm_sha256": twm_sha, "hawor_sha256": hawor_sha}
    if original.get("base_checkpoints") != expected_base:
        raise ValueError("Parent cache uses different frozen base checkpoints")

    seed_everything(7)
    config, twm_state = train_visual.load_base_twm(args.twm_checkpoint, EXPECTED_TWM_SHA256)
    system = build_system(args.hawor_checkpoint, config, device)
    system.observer.twm.load_state_dict(twm_state, strict=True)
    install_feedback_v2(system)
    train_visual.freeze_except_visual(system)
    system.eval()
    system.hawor.eval()
    system.observer.twm.eval()
    frozen = {
        "hawor_parameters_sha256": module_parameter_sha256(system.hawor),
        "twm_parameters_sha256": module_parameter_sha256(system.observer.twm),
        "geometry_parameters_sha256": module_parameter_sha256(system.geometry),
    }
    if frozen != original.get("frozen_parameter_hashes"):
        raise ValueError("Fresh frozen model parameter hashes differ from the parent cache")
    return system, frozen, twm_sha, hawor_sha


def _load_calibration_sequences(args: argparse.Namespace,
                                event_manifest: dict[str, Any],
                                system: torch.nn.Module,
                                device: torch.device) -> dict[str, PreparedSequence]:
    groups, _protocol = load_sequences(
        data_root=args.data_root,
        video_root=args.video_root,
        split_manifest=args.split_manifest,
        export_split=args.export_split,
        protocol_path=args.protocol,
        splits=("calibration",),
        check_images=not args.skip_image_check,
    )
    sequences = groups["calibration"]
    expected_names = set(event_manifest["sequence_names"])
    by_name = {sequence.sequence: sequence for sequence in sequences}
    if set(by_name) != expected_names:
        raise ValueError("Loaded calibration sequences differ from the event manifest")
    for sequence_name, record in event_manifest["sequences"].items():
        sequence = by_name[sequence_name]
        archive_path = Path(record["source_archive"]).resolve()
        if archive_path != sequence.archive_path.resolve():
            raise ValueError(f"{sequence_name}: source archive path differs from event manifest")
        if sha256_file(archive_path) != record.get("source_archive_sha256"):
            raise ValueError(f"{sequence_name}: prepared source archive SHA256 changed")
        if int(record.get("frame_count", -1)) != len(sequence):
            raise ValueError(f"{sequence_name}: source frame count differs from event manifest")
        if record.get("coordinate_frame") != "per-interval local raw HaWoR SLAM world":
            raise ValueError(f"{sequence_name}: event labels use an unexpected coordinate frame")

    convert_raw_history_states(sequences, system.geometry, device)
    return by_name


def _verify_event_gt_sidecars(
    event_manifest_path: Path,
    event_manifest: dict[str, Any],
    sequences: dict[str, PreparedSequence] | None = None,
) -> dict[str, dict[str, Any]]:
    sidecars = {}
    for sequence_name, record in event_manifest["sequences"].items():
        sidecar_path = _resolve_from(record["gt_file"], Path(event_manifest_path).resolve().parent)
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"Missing event GT sidecar for {sequence_name}: {sidecar_path}")
        sidecar_sha = sha256_file(sidecar_path)
        with np.load(sidecar_path, allow_pickle=False) as archive:
            required = {"sequence", "split", "coordinate_frame", "frame_ids", "timestamp_sec",
                        "segment_id", "gt_valid", "gt_joints_world_m",
                        "visual_valid", "formal_reference_valid", "evaluation_mask",
                        "normalized_pinch_distance", "continuous_run_id"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"{sequence_name}: event GT sidecar missing {missing}")
            values = {key: archive[key].copy() for key in required}
        sequence_record = event_manifest["sequences"][sequence_name]
        count = int(sequence_record["frame_count"])
        if (str(values["sequence"].item()) != sequence_name
                or str(values["split"].item()) != "calibration"
                or str(values["coordinate_frame"].item()) != "per-interval local raw HaWoR SLAM world"):
            raise ValueError(f"{sequence_name}: event GT sidecar identity/split/frame mismatch")
        if (values["frame_ids"].shape != (count,)
                or not np.array_equal(values["frame_ids"], np.arange(count, dtype=np.int64))
                or values["timestamp_sec"].shape != (count,)
                or values["segment_id"].shape != (count,)
                or values["gt_valid"].shape != (count,)
                or values["gt_joints_world_m"].shape != (count, 21, 3)
                or values["visual_valid"].shape != (count,)
                or values["formal_reference_valid"].shape != (count,)
                or values["evaluation_mask"].shape != (count,)
                or values["normalized_pinch_distance"].shape != (count,)
                or values["continuous_run_id"].shape != (count,)):
            raise ValueError(f"{sequence_name}: malformed event GT sidecar arrays")
        if (not np.array_equal(values["formal_reference_valid"], values["gt_valid"])
                or np.any(values["evaluation_mask"]
                          & ~(values["gt_valid"] & values["visual_valid"]))):
            raise ValueError(f"{sequence_name}: sidecar reference/evaluation masks violate the fixed protocol")
        if sequences is not None:
            sequence = sequences[sequence_name]
            if (len(sequence) != count
                    or not np.allclose(values["timestamp_sec"], sequence.timestamp_sec,
                                       rtol=0.0, atol=1e-9)
                    or not np.array_equal(values["segment_id"], sequence.segment_id)
                    or not np.array_equal(values["visual_valid"], sequence.visual_valid)
                    or not np.array_equal(values["gt_valid"], sequence.gt_valid)):
                raise ValueError(
                    f"{sequence_name}: event sidecar validity/time/segment differs from observations")
        sidecars[sequence_name] = {
            "path": str(sidecar_path),
            "sha256": sidecar_sha,
            "coordinate_frame": str(values["coordinate_frame"].item()),
            "frame_count": count,
            "gt_valid_count": int(values["gt_valid"].sum()),
            "gt_joints_key": "gt_joints_world_m",
            "gt_valid_key": "gt_valid",
            "formal_reference_valid_key": "formal_reference_valid",
            "visual_valid_key": "visual_valid",
            "evaluation_mask_key": "evaluation_mask",
            "continuous_run_key": "continuous_run_id",
            "timestamps_key": "timestamp_sec",
            "segment_key": "segment_id",
            "event_distance_key": "normalized_pinch_distance",
            "evaluation_mask_true_count": int(values["evaluation_mask"].sum()),
        }
    return sidecars


def _segment_context_frame_ids(sequence: PreparedSequence, context_start: int) -> np.ndarray:
    if not 0 <= context_start < len(sequence):
        raise IndexError(context_start)
    segment = sequence.segment_id[context_start]
    stop = context_start + 1
    while stop < len(sequence) and sequence.segment_id[stop] == segment:
        stop += 1
    return np.arange(context_start, min(context_start + 16, stop), dtype=np.int64)


def _stack_context_inputs(model_inputs: dict[str, Any], rows: list[int]) -> dict[str, Any]:
    input_rows = [_inputs_for_row(model_inputs, row) for row in rows]
    return {
        key: torch.stack([row[key] for row in input_rows], dim=0)
        if key != "projector_args"
        else tuple(torch.stack([row[key][part] for row in input_rows], dim=0)
                   for part in range(4))
        for key in input_rows[0]
    }


def _extract_context(
    *,
    system: torch.nn.Module,
    sequence: PreparedSequence,
    context_start: int,
    requested_frames: list[int],
    device: torch.device,
    output_cache: Path,
    parent_item: dict[str, Any] | None,
) -> tuple[dict[str, Any], set[str]]:
    full_frame_ids = _segment_context_frame_ids(sequence, context_start)
    if len(full_frame_ids) > 16 or not len(full_frame_ids):
        raise RuntimeError(f"Invalid full context frame ids for {sequence.sequence}:{context_start}")
    if not set(requested_frames).issubset(set(full_frame_ids.tolist())):
        raise ValueError(f"Requested rows exceed HaWoR context {sequence.sequence}:{context_start}")
    full_samples = [(sequence, int(frame)) for frame in full_frame_ids]
    model_inputs, _beta, records = train_visual.build_live_batch(system, full_samples, device)
    row_by_frame = {int(record["frame_index"]): row for row, record in enumerate(records)}
    selected_rows = [row_by_frame[frame] for frame in requested_frames]
    cached_inputs = _stack_context_inputs(model_inputs, selected_rows)
    labels = _labels_for_frames(sequence, np.asarray(requested_frames, dtype=np.int64),
                                system.geometry, device)

    relative = Path("calibration") / sequence.sequence
    input_path = output_cache / relative / f"ctx_{context_start:06d}.inputs.pt"
    labels_path = output_cache / relative / f"ctx_{context_start:06d}.labels.pt"
    _atomic_torch_save(cached_inputs, input_path)
    _atomic_torch_save(labels, labels_path)

    frame_id_set = set(requested_frames)
    cached_records = [records[row_by_frame[frame]] for frame in requested_frames]
    item_id = f"calibration/{sequence.sequence}/ctx_{context_start:06d}"
    item = {
        "item_id": item_id,
        "split": "calibration",
        "sequence": sequence.sequence,
        "context_start": int(context_start),
        "frame_indices": list(requested_frames),
        "inputs_path": str(input_path.resolve()),
        "labels_path": str(labels_path.resolve()),
        "inputs_sha256": sha256_file(input_path),
        "labels_sha256": sha256_file(labels_path),
        "patch_dtype_on_disk": "float16",
        "input_keys": sorted(cached_inputs),
        "label_keys": sorted(labels),
        "record_timestamps": [float(record["timestamp_sec"]) for record in cached_records],
        "record_segment_ids": [int(record["segment_id"]) for record in cached_records],
        "cache_source": "new_frozen_hawor_extraction",
        "extraction_context_frame_indices": full_frame_ids.tolist(),
    }
    if parent_item is not None:
        item["replaced_parent_item_id"] = parent_item["item_id"]
        item["parent_inputs_sha256"] = parent_item["inputs_sha256"]
        item["parent_labels_sha256"] = parent_item["labels_sha256"]
    used_images = {str(path) for record in records for path in record["image_paths"]}
    if len(frame_id_set) != len(requested_frames):
        raise ValueError("Duplicate rows requested for a cache context")
    return item, used_images


def _absolute_parent_item(item: dict[str, Any], original_cache: Path) -> dict[str, Any]:
    value = copy.deepcopy(item)
    for path_key in ("inputs_path", "labels_path"):
        resolved = _resolve_from(value[path_key], original_cache)
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing parent cache file: {resolved}")
        value[path_key] = str(resolved)
    value["cache_source"] = "read_only_parent_cache"
    return value


def _build_manifest(
    *,
    original: dict[str, Any],
    event_manifest: dict[str, Any],
    plan: dict[str, Any],
    parent_cache: Path,
    event_manifest_path: Path,
    parent_manifest_sha: str,
    event_manifest_sha: str,
    frozen_hashes: dict[str, str],
    event_sidecars: dict[str, dict[str, Any]],
    new_items: dict[tuple[str, int], dict[str, Any]],
    incremental_images: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    old_by_key = {(str(item["sequence"]), int(item["context_start"])): item
                  for item in original["items"] if item["split"] == "calibration"}
    calibration_items = []
    for row in plan["contexts"]:
        key = (row["sequence"], int(row["context_start"]))
        if row["action"] == "reuse_parent":
            calibration_items.append(_absolute_parent_item(old_by_key[key], parent_cache))
        else:
            calibration_items.append(new_items[key])
    train_items = [_absolute_parent_item(item, parent_cache)
                   for item in original["items"] if item["split"] == "train"]
    items = sorted(train_items + calibration_items,
                   key=lambda item: (item["split"], item["sequence"], int(item["context_start"])))

    starts = event_manifest["context_start_indices"]
    event_windows = []
    for window in event_manifest["merged_windows"]:
        sequence = str(window["sequence"])
        frame_ids = _window_frame_ids(window, source="expanded event")
        context_item_ids = sorted({
            f"calibration/{sequence}/ctx_{_context_start_for_frame(int(frame), starts[sequence]['required_context_start_indices'], sequence):06d}"
            for frame in frame_ids
        })
        event_windows.append({
            **copy.deepcopy(window),
            "frame_indices": frame_ids.tolist(),
            "context_item_ids": context_item_ids,
        })

    source_hashes = copy.deepcopy(original.get("source_hashes", {}))
    source_hashes["parent_cache_manifest"] = {
        "path": str((parent_cache / "manifest.json").resolve()),
        "sha256": parent_manifest_sha,
    }
    source_hashes["calibration_event_manifest"] = {
        "path": str(Path(event_manifest_path).resolve()),
        "sha256": event_manifest_sha,
        "schema": EVENT_SCHEMA,
    }
    source_hashes["event_ground_truth_sidecars"] = copy.deepcopy(event_sidecars)
    source_hashes["incremental_rgb_images"] = copy.deepcopy(incremental_images)
    merged_rgb = source_hashes.setdefault("rgb_images", {})
    for path, source in incremental_images.items():
        previous = merged_rgb.get(path)
        if previous is not None and previous.get("sha256") != source["sha256"]:
            raise ValueError(f"Image SHA changed since the parent cache was built: {path}")
        merged_rgb[path] = source

    counts = copy.deepcopy(original.get("counts", {}))
    counts.update({
        "train_contexts": len(train_items),
        "calibration_contexts": len(calibration_items),
        "calibration_windows": len(original["calibration_windows"]),
        "calibration_window_frames": int(original["counts"]["calibration_window_frames"]),
        "calibration_event_windows": len(event_windows),
        "calibration_event_window_frames": int(event_manifest["counts"]["expanded_unique_frames"]),
        "unique_union_calibration_frames": int(plan["requested_frame_count"]),
        "reused_train_items": len(train_items),
        "reused_calibration_contexts": int(plan["reuse_calibration_context_count"]),
        "expanded_calibration_contexts": int(plan["expanded_calibration_context_count"]),
        "new_calibration_contexts": int(plan["new_calibration_context_count"]),
        "calibration_contexts_reextracted": len(new_items),
        "formal_prediction_eligible_frames": int(sum(
            sidecar["evaluation_mask_true_count"] for sidecar in event_sidecars.values())),
    })
    result = copy.deepcopy(original)
    result.update({
        "protocol_revision": PROTOCOL_REVISION,
        "items": items,
        "counts": counts,
        "calibration_windows": copy.deepcopy(original["calibration_windows"]),
        "event_windows": event_windows,
        "event_protocol": copy.deepcopy(event_manifest["event_protocol"]),
        "event_context_rule": copy.deepcopy(event_manifest["event_context_rule"]),
        "prediction_eligibility": copy.deepcopy(event_manifest["prediction_eligibility"]),
        "calibration_event_evaluation_metadata": {
            "prediction_eligibility": copy.deepcopy(event_manifest["prediction_eligibility"]),
            "fixed_evaluation_mask_key": "evaluation_mask",
            "evaluation_masks": {
                sequence: {
                    "sidecar_path": sidecar["path"],
                    "sidecar_sha256": sidecar["sha256"],
                    "array_key": sidecar["evaluation_mask_key"],
                    "frame_count": sidecar["frame_count"],
                    "eligible_frame_count": sidecar["evaluation_mask_true_count"],
                }
                for sequence, sidecar in event_sidecars.items()
            },
        },
        "event_ground_truth_sidecars": copy.deepcopy(event_sidecars),
        "parent_manifest": {
            "path": str((parent_cache / "manifest.json").resolve()),
            "sha256": parent_manifest_sha,
            "schema": CACHE_SCHEMA,
            "version": CACHE_VERSION,
        },
        "calibration_event_manifest": {
            "path": str(Path(event_manifest_path).resolve()),
            "sha256": event_manifest_sha,
            "schema": EVENT_SCHEMA,
            "event_count": int(event_manifest["counts"]["total_complete_gt_reference_events"]),
            "expanded_window_count": len(event_windows),
        },
        "feature_extension": {
            "protocol_revision": PROTOCOL_REVISION,
            "feature_source": "same frozen HaWoR IAM/model_inputs as parent cache",
            "label_policy": "GT stays in separate .labels.pt files and global GT sidecars; input files contain only observation/camera/raw-history features",
            "reused_train_items": len(train_items),
            "reused_calibration_contexts": int(plan["reuse_calibration_context_count"]),
            "expanded_calibration_contexts": int(plan["expanded_calibration_context_count"]),
            "new_calibration_contexts": int(plan["new_calibration_context_count"]),
            "reextracted_contexts": len(new_items),
            "frozen_parameter_hashes": frozen_hashes,
        },
        "source_hashes": source_hashes,
        "frozen_parameter_hashes": frozen_hashes,
    })
    return result


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    original_cache = Path(args.original_cache).resolve()
    event_manifest_path = Path(args.event_manifest).resolve()
    output_cache = Path(args.output_cache).resolve()
    original, event_manifest, parent_manifest_sha, event_manifest_sha = _load_bound_manifests(
        original_cache, event_manifest_path, verify_parent_files=not args.plan_only)
    verified_sources = _verify_parent_source_inputs(args, original, event_manifest)
    event_sidecars = _verify_event_gt_sidecars(event_manifest_path, event_manifest)
    plan = _build_plan(original, event_manifest)
    if args.plan_only:
        print(json.dumps({
            "plan_only": True,
            "original_cache": str(original_cache),
            "event_manifest": str(event_manifest_path),
            "output_cache": str(output_cache),
            "parent_manifest_sha256": parent_manifest_sha,
            "event_manifest_sha256": event_manifest_sha,
            "event_gt_sidecar_count": len(event_sidecars),
            "formal_prediction_eligible_frames": sum(
                sidecar["evaluation_mask_true_count"] for sidecar in event_sidecars.values()),
            "verified_parent_source_count": len(verified_sources),
            "verified_source_sha256_sample": dict(list(sorted(verified_sources.items()))[:4]),
            "counts": {key: value for key, value in plan.items() if key != "contexts"},
            "context_plan_sample": plan["contexts"][:4],
        }, sort_keys=True), flush=True)
        return

    if output_cache.joinpath("manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed extension cache: {output_cache / 'manifest.json'}")
    if not output_cache.exists():
        output_cache.mkdir(parents=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    system, frozen_hashes, twm_sha, hawor_sha = _prepare_system(args, device, original)
    if frozen_hashes != original.get("frozen_parameter_hashes"):
        raise ValueError("Frozen parameter hashes changed during extension setup")
    groups = _load_calibration_sequences(args, event_manifest, system, device)
    event_sidecars = _verify_event_gt_sidecars(event_manifest_path, event_manifest, groups)

    new_items: dict[tuple[str, int], dict[str, Any]] = {}
    incremental_images: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(plan["contexts"], start=1):
        if row["action"] == "reuse_parent":
            continue
        sequence = groups[row["sequence"]]
        item, image_paths = _extract_context(
            system=system,
            sequence=sequence,
            context_start=int(row["context_start"]),
            requested_frames=[int(frame) for frame in row["frame_indices"]],
            device=device,
            output_cache=output_cache,
            parent_item=next((value for value in original["items"]
                              if value.get("split") == "calibration"
                              and value.get("sequence") == row["sequence"]
                              and int(value.get("context_start", -1)) == int(row["context_start"])), None),
        )
        new_items[(row["sequence"], int(row["context_start"]))] = item
        for image_path in sorted(image_paths):
            if image_path not in incremental_images:
                path = Path(image_path)
                incremental_images[image_path] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        print(json.dumps({
            "event": "context_cached",
            "context_index": index,
            "context_count": len(plan["contexts"]),
            "action": row["action"],
            "sequence": row["sequence"],
            "context_start": row["context_start"],
            "requested_rows": len(row["frame_indices"]),
        }, sort_keys=True), flush=True)

    if len(new_items) != (plan["expanded_calibration_context_count"]
                          + plan["new_calibration_context_count"]):
        raise RuntimeError("Not all new/expanded calibration contexts were extracted")
    if {"twm_sha256": twm_sha, "hawor_sha256": hawor_sha} != original.get("base_checkpoints"):
        raise ValueError("Base checkpoints changed while extracting the extension cache")

    result = _build_manifest(
        original=original,
        event_manifest=event_manifest,
        plan=plan,
        parent_cache=original_cache,
        event_manifest_path=event_manifest_path,
        parent_manifest_sha=parent_manifest_sha,
        event_manifest_sha=event_manifest_sha,
        frozen_hashes=frozen_hashes,
        event_sidecars=event_sidecars,
        new_items=new_items,
        incremental_images=incremental_images,
    )
    manifest_path = output_cache / "manifest.json"
    _atomic_json(result, manifest_path)
    print(json.dumps({
        "complete": True,
        "output_cache": str(output_cache),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "counts": result["counts"],
        "source_hashes": {
            "parent_manifest_sha256": parent_manifest_sha,
            "event_manifest_sha256": event_manifest_sha,
            "event_gt_sidecar_count": len(event_sidecars),
            "incremental_rgb_image_count": len(incremental_images),
        },
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
