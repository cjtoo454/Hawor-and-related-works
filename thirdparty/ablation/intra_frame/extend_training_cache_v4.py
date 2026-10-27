#!/usr/bin/env python3
"""Extend the v3 frozen feature cache with 100-frame train-only windows.

The parent cache, calibration items, and calibration protocol are read-only.
This script adds four time-spread 128-frame windows per training sequence and
up to four GT-pinch-onset windows, storing requested rows in separate input
and label files while preserving each true 16-frame HaWoR context.
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

from dual_loop.geometry import ManoGeometry  # noqa: E402
from dual_loop.handloop_runtime import build_system  # noqa: E402
from dual_loop.io import _resolve_tracks_path  # noqa: E402
from dual_loop.oracle import normalized_pinch_distance  # noqa: E402
from dual_loop.pinch_evaluation import extract_events  # noqa: E402
from feedback_v2 import install_feedback_v2  # noqa: E402
from extend_calibration_cache import (  # noqa: E402
    _segment_context_frame_ids,
    _stack_context_inputs,
)
from prepare_training import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPORT_SPLIT,
    DEFAULT_PROTOCOL,
    DEFAULT_SPLIT,
    DEFAULT_VIDEO_ROOT,
    PreparedSequence,
    context_indices,
    context_start_indices,
    convert_raw_history_states,
    load_sequences,
)
from train_visual import (  # noqa: E402
    DEFAULT_HAWOR,
    DEFAULT_TWM,
    EXPECTED_HAWOR_SHA256,
    EXPECTED_TWM_SHA256,
    build_live_batch,
    freeze_except_visual,
    legacy_to_state,
    load_base_twm,
    module_parameter_sha256,
    seed_everything,
    sha256_file as train_sha256,
    state_to_legacy,
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


PROTOCOL_REVISION = 3
TRAINING_EXTENSION_REVISION = 4
UNIFORM_WINDOW_COUNT = 4
UNIFORM_WINDOW_FRAMES = 128
EVENT_WINDOW_LIMIT = 4
EVENT_WINDOW_FRAMES = 64
EVENT_HALF_WINDOW = EVENT_WINDOW_FRAMES // 2
PINCH_THRESHOLD = 0.2
PINCH_MIN_FRAMES = 3
MAX_GAP_SEC = 0.25
EXPECTED_TRAIN_SEQUENCE_COUNT = 33
EXPECTED_PARENT_TRAIN_ITEM_COUNT = 528
EXPECTED_PARENT_CALIBRATION_ITEM_COUNT = 1280
EXPECTED_PARENT_CALIBRATION_WINDOW_COUNT = 40


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-cache", type=Path,
        default=HERE / "runs" / "feedback_v3_train" / "cache")
    parser.add_argument(
        "--output-cache", type=Path,
        default=HERE / "runs" / "feedback_v4_train" / "cache")
    parser.add_argument("--twm-checkpoint", type=Path, default=DEFAULT_TWM)
    parser.add_argument("--hawor-checkpoint", type=Path, default=DEFAULT_HAWOR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--export-split", type=Path, default=DEFAULT_EXPORT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Validate sources, derive train windows, write a plan JSON, and exit without loading HaWoR/TWM")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _resolve_from(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _plan_sha256(plan: dict[str, Any]) -> str:
    payload = copy.deepcopy(plan)
    payload.pop("plan_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _continuous_runs(sequence: PreparedSequence) -> list[tuple[int, int]]:
    """Return half-open runs split by SLAM reset or a gap larger than 250ms."""
    if not len(sequence):
        return []
    starts = [0]
    for index in range(1, len(sequence)):
        delta = float(sequence.timestamp_sec[index] - sequence.timestamp_sec[index - 1])
        if (sequence.segment_id[index] != sequence.segment_id[index - 1]
                or delta <= 0 or delta > MAX_GAP_SEC):
            starts.append(index)
    starts.append(len(sequence))
    return list(zip(starts[:-1], starts[1:]))


def _longest_true_run(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if not len(values):
        return 0
    padded = np.pad(values.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return int(np.max(stops - starts)) if len(starts) else 0


def choose_uniform_windows(
    sequence: PreparedSequence,
    *,
    count: int = UNIFORM_WINDOW_COUNT,
    length: int = UNIFORM_WINDOW_FRAMES,
) -> list[dict[str, Any]]:
    """Select fixed windows uniformly over eligible start time.

    Each candidate is fully inside one segment/time-contiguous run, has at
    least half its rows GT-and-raw-valid, and contains a 16-frame supervised
    subrun. Nearest candidates to equally spaced time-bin centers are chosen.
    """
    if count < 1 or length < 1:
        raise ValueError("count and length must be positive")
    eligible = np.asarray(sequence.gt_valid, dtype=bool) & np.asarray(
        sequence.visual_valid, dtype=bool)
    candidates: list[dict[str, Any]] = []
    for run_start, run_stop in _continuous_runs(sequence):
        last_start = run_stop - length
        if last_start < run_start:
            continue
        for start in range(run_start, last_start + 1):
            stop = start + length
            local = eligible[start:stop]
            valid_count = int(local.sum())
            longest = _longest_true_run(local)
            if valid_count < (length + 1) // 2 or longest < 16:
                continue
            candidates.append({
                "start_frame": int(start),
                "stop_frame_exclusive": int(stop),
                "frame_indices": list(range(start, stop)),
                "segment_id": int(sequence.segment_id[start]),
                "eligible_frame_count": valid_count,
                "longest_eligible_run": longest,
                "center_time_sec": float(sequence.timestamp_sec[start + length // 2]),
            })
    if len(candidates) < count:
        raise ValueError(
            f"{sequence.sequence}: found only {len(candidates)} eligible {length}-frame "
            f"windows; {count} are required")

    candidates.sort(key=lambda item: (item["center_time_sec"], item["start_frame"]))
    centers = np.asarray([item["center_time_sec"] for item in candidates], dtype=np.float64)
    fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
    targets = float(centers[0]) + fractions * float(centers[-1] - centers[0])
    chosen: list[dict[str, Any]] = []
    available = set(range(len(candidates)))
    for target in targets:
        slot = min(available, key=lambda index: (
            abs(float(centers[index]) - float(target)),
            candidates[index]["start_frame"],
        ))
        available.remove(slot)
        chosen.append(copy.deepcopy(candidates[slot]))
    chosen.sort(key=lambda item: item["start_frame"])
    for window_index, item in enumerate(chosen):
        item["window_index"] = window_index
        item["kind"] = "uniform"
        item.pop("center_time_sec")
    return chosen


def _gt_joints_and_distances(
    sequence: PreparedSequence,
    geometry: torch.nn.Module,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute label-side GT joints through the repository state bridge."""
    joints = np.full((len(sequence), 21, 3), np.nan, dtype=np.float32)
    valid_indices = np.flatnonzero(sequence.gt_valid)
    geometry.eval()
    with torch.no_grad():
        for start in range(0, len(valid_indices), batch_size):
            indices = valid_indices[start:start + batch_size]
            legacy = torch.as_tensor(
                sequence.target_legacy[indices], dtype=torch.float32, device=device)
            state = legacy_to_state(legacy, geometry)
            reconstructed = state_to_legacy(state, legacy[..., 99:], geometry)
            joints[indices] = geometry.joints(reconstructed).detach().cpu().numpy()
    distances = normalized_pinch_distance(joints)
    return joints, distances


def extract_gt_pinch_events(
    sequence: PreparedSequence,
    distances: np.ndarray,
) -> list[dict[str, Any]]:
    """Extract train-label events separately within each continuous run."""
    values = np.asarray(distances, dtype=np.float64)
    if values.shape != (len(sequence),):
        raise ValueError(f"{sequence.sequence}: GT distance array has wrong shape")
    events: list[dict[str, Any]] = []
    for run_start, run_stop in _continuous_runs(sequence):
        local_events = extract_events(
            values[run_start:run_stop],
            np.asarray(sequence.timestamp_sec[run_start:run_stop], dtype=np.float64),
            np.asarray(sequence.gt_valid[run_start:run_stop], dtype=bool),
            threshold=PINCH_THRESHOLD,
            min_frames=PINCH_MIN_FRAMES,
            max_gap_s=MAX_GAP_SEC,
        )
        for event in local_events:
            onset = run_start + int(event.start)
            events.append({
                "start_frame": onset,
                "stop_frame_inclusive": run_start + int(event.stop),
                "gt_onset_times_sec": float(event.onset_sec),
                "continuous_run_start": int(run_start),
                "continuous_run_stop_exclusive": int(run_stop),
                "segment_id": int(sequence.segment_id[onset]),
            })
    events.sort(key=lambda item: (item["gt_onset_times_sec"], item["start_frame"]))
    return events


def choose_event_windows(
    sequence: PreparedSequence,
    events: list[dict[str, Any]],
    *,
    eligible_mask: np.ndarray,
    limit: int = EVENT_WINDOW_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Take supervised GT-onset windows, clipped to one continuous run.

    Events without any GT-and-raw-valid row, or with no usable row among the
    onset frame and its one-frame-before/three-frame-after neighborhood, are
    recorded and excluded before the four-window temporal selection.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    eligible = np.asarray(eligible_mask, dtype=bool)
    if eligible.shape != (len(sequence),):
        raise ValueError(f"{sequence.sequence}: event eligibility mask has wrong shape")
    candidates = []
    excluded_no_supervision = 0
    excluded_no_onset_supervision = 0
    for event in events:
        onset = int(event["start_frame"])
        run_start = int(event["continuous_run_start"])
        run_stop = int(event["continuous_run_stop_exclusive"])
        start = max(run_start, onset - EVENT_HALF_WINDOW)
        stop = min(run_stop, onset + EVENT_HALF_WINDOW)
        frame_ids = list(range(start, stop))
        if not frame_ids or onset not in frame_ids:
            raise ValueError(f"{sequence.sequence}: clipped event window lost its GT onset")
        supervised = eligible[start:stop]
        supervised_count = int(supervised.sum())
        longest_supervised = _longest_true_run(supervised)
        onset_neighborhood = [
            frame for frame in range(max(run_start, onset - 1), min(run_stop, onset + 4))
            if start <= frame < stop
        ]
        onset_supervision_count = int(sum(eligible[frame] for frame in onset_neighborhood))
        if supervised_count == 0:
            excluded_no_supervision += 1
            continue
        if onset_supervision_count == 0:
            excluded_no_onset_supervision += 1
            continue
        candidates.append({
            "sequence": sequence.sequence,
            "kind": "event",
            "start_frame": int(start),
            "stop_frame_exclusive": int(stop),
            "frame_indices": frame_ids,
            "segment_id": int(event["segment_id"]),
            "gt_onset_times_sec": float(event["gt_onset_times_sec"]),
            "supervised_frame_count": supervised_count,
            "longest_supervised_run": longest_supervised,
            "onset_neighborhood_frame_indices": onset_neighborhood,
            "onset_neighborhood_usable_count": onset_supervision_count,
        })
    candidates.sort(key=lambda item: (item["gt_onset_times_sec"], item["start_frame"]))
    if limit == 0 or not candidates:
        selected = []
    elif len(candidates) <= limit:
        selected = candidates
    else:
        times = np.asarray([item["gt_onset_times_sec"] for item in candidates], dtype=np.float64)
        fractions = (np.arange(limit, dtype=np.float64) + 0.5) / limit
        targets = float(times[0]) + fractions * float(times[-1] - times[0])
        available = set(range(len(candidates)))
        indices = []
        for target in targets:
            index = min(available, key=lambda value: (
                abs(float(times[value]) - float(target)), candidates[value]["start_frame"]))
            available.remove(index)
            indices.append(index)
        selected = [candidates[index] for index in sorted(indices)]
    windows = []
    selected_audit = []
    for window_index, candidate in enumerate(selected):
        item = {key: value for key, value in candidate.items()
                if key not in {"supervised_frame_count", "longest_supervised_run",
                               "onset_neighborhood_frame_indices",
                               "onset_neighborhood_usable_count"}}
        item["window_index"] = window_index
        windows.append(item)
        selected_audit.append({
            "window_index": window_index,
            "gt_onset_times_sec": candidate["gt_onset_times_sec"],
            "supervised_frame_count": candidate["supervised_frame_count"],
            "longest_supervised_run": candidate["longest_supervised_run"],
            "onset_neighborhood_usable_count": candidate["onset_neighborhood_usable_count"],
        })
    return windows, {
        "gt_event_count": len(events),
        "candidate_event_window_count": len(events),
        "eligible_event_window_count": len(candidates),
        "selected_event_window_count": len(windows),
        "excluded_no_supervised_rows": excluded_no_supervision,
        "excluded_no_onset_neighborhood_rows": excluded_no_onset_supervision,
        "selected_windows": selected_audit,
    }


def _attach_context_items(
    sequence: PreparedSequence,
    windows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str, int], list[int]]]:
    """Build per-window context IDs and exact, deduplicated row requests."""
    starts = context_start_indices(sequence)
    records: list[dict[str, Any]] = []
    requested: dict[tuple[int, str, int], list[int]] = {}
    for source in windows:
        frame_ids = [int(value) for value in source["frame_indices"]]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError(f"{sequence.sequence}: repeated frame ID within a window")
        context_rows: dict[int, list[int]] = {}
        for frame in frame_ids:
            if frame < 0 or frame >= len(sequence):
                raise ValueError(f"{sequence.sequence}: frame ID outside sequence: {frame}")
            context_start = int(starts[frame])
            context_rows.setdefault(context_start, []).append(frame)
        item_ids = []
        for context_start, rows in sorted(context_rows.items()):
            rows = sorted(set(rows))
            item_id = (
                f"train_v4/{sequence.sequence}/{source['kind']}_{int(source['window_index']):02d}/"
                f"ctx_{context_start:06d}")
            item_ids.append(item_id)
            key = (int(source["window_index"]), str(source["kind"]), context_start)
            requested[key] = rows
        record = {
            "sequence": sequence.sequence,
            "window_index": int(source["window_index"]),
            "kind": str(source["kind"]),
            "frame_indices": frame_ids,
            "context_item_ids": item_ids,
        }
        if source["kind"] == "event":
            record["gt_onset_times_sec"] = float(source["gt_onset_times_sec"])
        records.append(record)
    return records, requested


def _validate_window_item_coverage(
    windows: list[dict[str, Any]],
    requested_by_sequence: dict[str, dict[tuple[int, str, int], list[int]]],
) -> None:
    item_ids_by_key: dict[tuple[str, int, str, int], str] = {}
    rows_by_item_id: dict[str, list[int]] = {}
    for sequence_name, contexts in requested_by_sequence.items():
        for (window_index, kind, context_start), rows in contexts.items():
            item_id = (
                f"train_v4/{sequence_name}/{kind}_{window_index:02d}/"
                f"ctx_{context_start:06d}")
            item_ids_by_key[(sequence_name, window_index, kind, context_start)] = item_id
            rows_by_item_id[item_id] = rows
    for window in windows:
        sequence = str(window["sequence"])
        expected = [int(value) for value in window["frame_indices"]]
        actual: list[int] = []
        for item_id in window["context_item_ids"]:
            if item_id not in rows_by_item_id:
                raise ValueError(f"{sequence}: unknown training context item {item_id}")
            actual.extend(rows_by_item_id[item_id])
        if sorted(actual) != sorted(expected) or len(actual) != len(set(actual)):
            raise ValueError(
                f"{sequence}/{window['kind']}/{window['window_index']}: "
                "context items do not cover window rows exactly once")


def _load_parent_cache(parent_cache: Path) -> tuple[dict[str, Any], str]:
    parent_cache = Path(parent_cache).resolve()
    manifest_path = parent_cache / "manifest.json"
    original = load_manifest(parent_cache, verify_files=True)
    manifest_sha = sha256_file(manifest_path)
    if original.get("schema") != CACHE_SCHEMA or original.get("version") != CACHE_VERSION:
        raise ValueError("Parent cache schema is not handloop_intra_v2_feature_cache version 1")
    if int(original.get("protocol_revision", -1)) != PROTOCOL_REVISION:
        raise ValueError("Parent cache must preserve frozen protocol_revision 3")
    if original.get("training_windows", []) not in (None, []):
        raise ValueError("Parent cache already has training_windows; refusing ambiguous extension")
    items = original.get("items", [])
    train_items = [item for item in items if item.get("split") == "train"]
    calibration_items = [item for item in items if item.get("split") == "calibration"]
    if (len(train_items) != EXPECTED_PARENT_TRAIN_ITEM_COUNT
            or len(calibration_items) != EXPECTED_PARENT_CALIBRATION_ITEM_COUNT):
        raise ValueError(
            "Parent item counts changed: "
            f"train={len(train_items)} calibration={len(calibration_items)}")
    if len(original.get("calibration_windows", [])) != EXPECTED_PARENT_CALIBRATION_WINDOW_COUNT:
        raise ValueError("Parent cache no longer contains the frozen 40 calibration windows")
    train_sequences = {str(item["sequence"]) for item in train_items}
    if (len(train_sequences) != EXPECTED_TRAIN_SEQUENCE_COUNT
            or len(original.get("train_context_selection", {})) != EXPECTED_TRAIN_SEQUENCE_COUNT):
        raise ValueError("Parent train cache must preserve all 33 frozen train sequences")
    return original, manifest_sha


def _verify_file_record(name: str, record: dict[str, Any]) -> dict[str, str]:
    path = Path(record.get("path", "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {name} source: {path}")
    actual = sha256_file(path)
    if actual != record.get("sha256"):
        raise ValueError(f"Frozen {name} SHA256 changed: {path}")
    return {"path": str(path), "sha256": actual}


def _verify_parent_protocol_and_checkpoints(
    args: argparse.Namespace,
    original: dict[str, Any],
) -> dict[str, Any]:
    source_hashes = original.get("source_hashes", {})
    frozen_files = source_hashes.get("protocol_and_splits", {})
    requested = {
        "split_manifest": args.split_manifest,
        "export_split": args.export_split,
        "protocol": args.protocol,
    }
    verified_files: dict[str, Any] = {}
    for name, requested_path in requested.items():
        record = frozen_files.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Parent source hashes lack {name}")
        path = Path(requested_path).expanduser().resolve()
        if path != Path(record.get("path", "")).resolve():
            raise ValueError(f"{name} path differs from the frozen parent source: {path}")
        verified_files[name] = _verify_file_record(name, record)

    expected = {"twm_sha256": EXPECTED_TWM_SHA256,
                "hawor_sha256": EXPECTED_HAWOR_SHA256}
    if original.get("base_checkpoints") != expected:
        raise ValueError("Parent cache uses different frozen TWM/HaWoR checkpoints")
    if source_hashes.get("base_checkpoints") != expected:
        raise ValueError("Parent cache checkpoint source hashes disagree with its base_checkpoints")
    checkpoint_paths = {
        "twm_checkpoint": (Path(args.twm_checkpoint), EXPECTED_TWM_SHA256),
        "hawor_checkpoint": (Path(args.hawor_checkpoint), EXPECTED_HAWOR_SHA256),
    }
    verified_checkpoints = {}
    for name, (requested_path, expected_sha) in checkpoint_paths.items():
        path = requested_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen {name}: {path}")
        actual = sha256_file(path)
        if actual != expected_sha:
            raise ValueError(f"{name} SHA256 differs from the parent protocol: {path}")
        verified_checkpoints[name] = {"path": str(path), "sha256": actual}
    return {
        "protocol_and_splits": verified_files,
        "base_checkpoints": verified_checkpoints,
    }


def _verify_parent_sequence_sources(
    args: argparse.Namespace,
    original: dict[str, Any],
) -> dict[str, Any]:
    """Verify all frozen train/calibration archive, detector, and focal files."""
    source_sequences = original.get("source_hashes", {}).get("sequences", {})
    item_sequences = {
        str(item["sequence"]) for item in original.get("items", [])
        if item.get("split") in {"train", "calibration"}
    }
    if set(source_sequences) != item_sequences:
        raise ValueError("Parent source sequence records differ from its train/calibration items")
    data_root = Path(args.data_root).resolve()
    video_root = Path(args.video_root).resolve()
    verified: dict[str, Any] = {}
    for sequence_name in sorted(item_sequences):
        record = source_sequences[sequence_name]
        archive = record.get("archive", {})
        detector = record.get("detector_tracks", {})
        focal = record.get("focal", {})
        expected_archive = (data_root / f"{sequence_name}.npz").resolve()
        sequence_root = video_root / sequence_name
        expected_detector = (
            sequence_root / "preprocess" / "preprocess_tracks.npy").resolve()
        expected_focal = (sequence_root / "focal.txt").resolve()
        if (expected_archive != Path(archive.get("path", "")).resolve()
                or expected_detector != Path(detector.get("path", "")).resolve()
                or expected_focal != Path(focal.get("path", "")).resolve()):
            raise ValueError(
                f"{sequence_name}: archive/detector/focal paths differ from the explicit "
                "data_root/video_root; refusing stale CLI-root resolution")
        detected_path, oracle_assisted = _resolve_tracks_path(
            sequence_root, allow_gt_boxes=False)
        if oracle_assisted or Path(detected_path).resolve() != expected_detector:
            raise ValueError(f"{sequence_name}: parent source is not the real detected-track path")
        verified[sequence_name] = {
            "archive": _verify_file_record(f"{sequence_name} raw archive", archive),
            "detector_tracks": _verify_file_record(
                f"{sequence_name} detector tracks", detector),
            "focal": _verify_file_record(f"{sequence_name} focal", focal),
        }
    return verified


def _verify_parent_event_sources(original: dict[str, Any]) -> dict[str, Any]:
    """Verify frozen v2 manifest, calibration event manifest, and GT sidecars."""
    verified: dict[str, Any] = {}
    parent_manifest = original.get("parent_manifest")
    if not isinstance(parent_manifest, dict):
        raise ValueError("Parent v3 manifest lacks its read-only v2 parent reference")
    parent_path = Path(parent_manifest.get("path", "")).resolve()
    parent_record = {"path": str(parent_path), "sha256": parent_manifest.get("sha256")}
    parent_verified = _verify_file_record("v2 parent manifest", parent_record)
    verified["v2_parent_manifest"] = parent_verified

    source_hashes = original.get("source_hashes", {})
    event_manifest = source_hashes.get("calibration_event_manifest")
    if not isinstance(event_manifest, dict):
        raise ValueError("Parent source hashes lack the frozen calibration event manifest")
    if event_manifest.get("schema") != "calibration_event_windows_v3":
        raise ValueError("Parent event source does not use calibration_event_windows_v3")
    verified["calibration_event_manifest"] = _verify_file_record(
        "calibration event manifest", event_manifest)
    sidecars = source_hashes.get("event_ground_truth_sidecars")
    if not isinstance(sidecars, dict) or len(sidecars) != int(
            original.get("counts", {}).get("calibration_sequences", -1)):
        raise ValueError("Parent calibration event sidecars do not cover all calibration sequences")
    calibration_sequences = {
        str(item["sequence"]) for item in original.get("items", [])
        if item.get("split") == "calibration"
    }
    if set(sidecars) != calibration_sequences:
        raise ValueError("Parent calibration event sidecar sequence list changed")
    verified_sidecars = {}
    for sequence_name, record in sorted(sidecars.items()):
        verified_sidecars[sequence_name] = _verify_file_record(
            f"{sequence_name} calibration GT sidecar", record)
    verified["event_ground_truth_sidecars"] = verified_sidecars
    return verified


def _verify_parent_rgb_sources(
    original: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Verify every source RGB hash already recorded by the parent cache."""
    parent_images = original.get("source_hashes", {}).get("rgb_images", {})
    if not isinstance(parent_images, dict) or not parent_images:
        raise ValueError("Parent source hashes lack the frozen RGB image inventory")
    verified: dict[str, dict[str, Any]] = {}
    recorded_rows = []
    for recorded_path, record in sorted(parent_images.items()):
        path = Path(recorded_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing parent RGB source image: {path}")
        size = path.stat().st_size
        actual = sha256_file(path)
        if (size != int(record.get("size_bytes", -1))
                or actual != record.get("sha256")):
            raise ValueError(f"Parent RGB source image changed: {path}")
        verified[str(path)] = {"size_bytes": size, "sha256": actual}
        recorded_rows.append({"path": str(path), "size_bytes": size, "sha256": actual})
    digest = hashlib.sha256(_canonical_json(recorded_rows)).hexdigest()
    return verified, {"count": len(verified), "verified_inventory_sha256": digest}


def _load_verified_train_sequences(
    args: argparse.Namespace,
    original: dict[str, Any],
    verified_sources: dict[str, Any],
) -> tuple[list[PreparedSequence], dict[str, Any]]:
    groups, protocol = load_sequences(
        data_root=Path(args.data_root).resolve(),
        video_root=Path(args.video_root).resolve(),
        split_manifest=Path(args.split_manifest).resolve(),
        export_split=Path(args.export_split).resolve(),
        protocol_path=Path(args.protocol).resolve(),
        splits=("train",),
        check_images=True,
    )
    sequences = groups["train"]
    by_name = {sequence.sequence: sequence for sequence in sequences}
    expected_names = {str(item["sequence"]) for item in original["items"]
                      if item.get("split") == "train"}
    if len(by_name) != EXPECTED_TRAIN_SEQUENCE_COUNT or set(by_name) != expected_names:
        raise ValueError("Loaded train split differs from the verified parent cache")

    data_root = Path(args.data_root).resolve()
    video_root = Path(args.video_root).resolve()
    verified: dict[str, Any] = {}
    for sequence_name in sorted(expected_names):
        sequence = by_name[sequence_name]
        source_record = verified_sources.get(sequence_name)
        if not isinstance(source_record, dict):
            raise ValueError(f"{sequence_name}: no verified parent source record")
        expected_archive = (data_root / f"{sequence_name}.npz").resolve()
        if (sequence.archive_path.resolve() != expected_archive
                or expected_archive != Path(source_record["archive"]["path"]).resolve()):
            raise ValueError(f"{sequence_name}: prepared archive path differs from parent provenance")
        archive_verified = source_record["archive"]

        sequence_root = video_root / sequence_name
        detected_track_path, oracle_assisted = _resolve_tracks_path(
            sequence_root, allow_gt_boxes=False)
        expected_track = Path(source_record["detector_tracks"]["path"]).resolve()
        expected_detected_track = (
            sequence_root / "preprocess" / "preprocess_tracks.npy").resolve()
        expected_focal = Path(source_record["focal"]["path"]).resolve()
        if oracle_assisted:
            raise ValueError(f"{sequence_name}: oracle-assisted detector tracks are forbidden")
        if (Path(detected_track_path).resolve() != expected_track
                or expected_track != expected_detected_track):
            raise ValueError(
                f"{sequence_name}: detector-track path does not match explicit video_root "
                f"{sequence_root}: {detected_track_path}")
        if expected_focal != (sequence_root / "focal.txt").resolve():
            raise ValueError(
                f"{sequence_name}: focal.txt path does not match explicit video_root "
                f"{expected_focal}")
        detector_verified = source_record["detector_tracks"]
        focal_verified = source_record["focal"]
        if sequence.split != "train":
            raise ValueError(f"{sequence_name}: loaded non-train data in the train-only extension")
        verified[sequence_name] = {
            "archive": archive_verified,
            "detector_tracks": detector_verified,
            "focal": focal_verified,
            "frame_count": len(sequence),
        }
    return sequences, {"sequences": verified, "validated_protocol": protocol}


def _hash_requested_context_images(
    sequences: list[PreparedSequence],
    windows: list[dict[str, Any]],
    requested_by_sequence: dict[str, dict[tuple[int, str, int], list[int]]],
    original: dict[str, Any],
    verified_parent_images: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    sequence_by_name = {sequence.sequence: sequence for sequence in sequences}
    image_paths: set[str] = set()
    for sequence_name, contexts in requested_by_sequence.items():
        sequence = sequence_by_name[sequence_name]
        for (_window_index, _kind, context_start), rows in contexts.items():
            full_ids = _segment_context_frame_ids(sequence, context_start)
            if not set(rows).issubset(set(int(value) for value in full_ids)):
                raise ValueError(
                    f"{sequence_name}:{context_start}: selected frames exceed real HaWoR context")
            padded_ids, _offset = context_indices(sequence, context_start)
            if len(padded_ids) != 16:
                raise ValueError(f"{sequence_name}:{context_start}: HaWoR input is not 16 frames")
            image_paths.update(str(Path(value).expanduser().resolve())
                               for value in sequence.image_paths[padded_ids])

    hashed: dict[str, dict[str, Any]] = {}
    for value in sorted(image_paths):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing train image used by HaWoR context: {path}")
        previous = verified_parent_images.get(value)
        if previous is not None:
            hashed[value] = copy.deepcopy(previous)
        else:
            hashed[value] = {"size_bytes": path.stat().st_size,
                             "sha256": sha256_file(path)}
    return hashed


def _build_plan(
    *,
    args: argparse.Namespace,
    original: dict[str, Any],
    parent_manifest_sha: str,
    source_checks: dict[str, Any],
    sequences: list[PreparedSequence],
    sequence_source_checks: dict[str, Any],
    verified_parent_images: dict[str, dict[str, Any]],
    geometry: torch.nn.Module,
) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    requested_by_sequence: dict[str, dict[tuple[int, str, int], list[int]]] = {}
    event_window_audit: dict[str, dict[str, Any]] = {}
    for sequence in sequences:
        uniform = choose_uniform_windows(sequence)
        _joints, distances = _gt_joints_and_distances(
            sequence, geometry, torch.device("cpu"))
        events = extract_gt_pinch_events(sequence, distances)
        event_windows, event_audit = choose_event_windows(
            sequence,
            events,
            eligible_mask=np.asarray(sequence.gt_valid, dtype=bool)
            & np.asarray(sequence.visual_valid, dtype=bool),
        )
        event_window_audit[sequence.sequence] = event_audit
        source_windows = uniform + event_windows
        attached, requested = _attach_context_items(sequence, source_windows)
        windows.extend(attached)
        requested_by_sequence[sequence.sequence] = requested

    windows.sort(key=lambda item: (
        item["sequence"], 0 if item["kind"] == "uniform" else 1,
        item["window_index"],
    ))
    _validate_window_item_coverage(windows, requested_by_sequence)
    image_sources = _hash_requested_context_images(
        sequences, windows, requested_by_sequence, original, verified_parent_images)
    contexts = []
    for sequence_name, requested in sorted(requested_by_sequence.items()):
        for (window_index, kind, context_start), frame_ids in sorted(requested.items()):
            item_id = (
                f"train_v4/{sequence_name}/{kind}_{window_index:02d}/"
                f"ctx_{context_start:06d}")
            contexts.append({
                "sequence": sequence_name,
                "window_index": int(window_index),
                "kind": kind,
                "context_start": int(context_start),
                "frame_indices": frame_ids,
                "item_id": item_id,
            })

    parent_train_rows = sum(
        len(item.get("frame_indices", [])) for item in original["items"]
        if item.get("split") == "train")
    parent_unique_train_frames = sum(
        len({int(frame) for entry in rows for frame in entry.get("supervised_frame_indices", [])})
        for rows in original.get("train_context_selection", {}).values())
    extension_frame_rows = sum(len(window["frame_indices"]) for window in windows)
    unique_by_sequence: dict[str, set[int]] = {}
    for window in windows:
        unique_by_sequence.setdefault(window["sequence"], set()).update(
            int(frame) for frame in window["frame_indices"])
    counts = {
        "train_sequences": len(sequences),
        "parent_train_contexts": sum(item.get("split") == "train" for item in original["items"]),
        "parent_train_frame_rows": parent_train_rows,
        "parent_unique_train_frames": parent_unique_train_frames,
        "uniform_windows": sum(window["kind"] == "uniform" for window in windows),
        "uniform_window_frames": sum(
            len(window["frame_indices"]) for window in windows if window["kind"] == "uniform"),
        "event_windows": sum(window["kind"] == "event" for window in windows),
        "event_window_frames": sum(
            len(window["frame_indices"]) for window in windows if window["kind"] == "event"),
        "gt_event_onsets_detected": sum(
            value["gt_event_count"] for value in event_window_audit.values()),
        "event_windows_excluded_no_supervised_rows": sum(
            value["excluded_no_supervised_rows"] for value in event_window_audit.values()),
        "event_windows_excluded_no_onset_neighborhood_rows": sum(
            value["excluded_no_onset_neighborhood_rows"] for value in event_window_audit.values()),
        "extension_training_window_frame_rows": extension_frame_rows,
        "extension_unique_training_frames": sum(len(values) for values in unique_by_sequence.values()),
        "extension_context_items": len(contexts),
        "extension_unique_ha_wor_contexts": len({
            (row["sequence"], row["context_start"]) for row in contexts}),
        "requested_image_hashes": len(image_sources),
    }
    if counts["uniform_windows"] != EXPECTED_TRAIN_SEQUENCE_COUNT * UNIFORM_WINDOW_COUNT:
        raise ValueError("Uniform windows do not cover exactly four windows per train sequence")
    requested_frame_rows = sum(len(context["frame_indices"]) for context in contexts)
    if requested_frame_rows != extension_frame_rows:
        raise ValueError("Context frame rows disagree with the training-window frame count")

    plan: dict[str, Any] = {
        "schema": "handloop_intra_v4_training_cache_plan",
        "version": 1,
        "plan_only": True,
        "protocol_revision": PROTOCOL_REVISION,
        "training_extension_revision": TRAINING_EXTENSION_REVISION,
        "parent_cache": {
            "path": str(Path(args.parent_cache).resolve()),
            "manifest_path": str((Path(args.parent_cache).resolve() / "manifest.json")),
            "manifest_sha256": parent_manifest_sha,
            "schema": original["schema"],
            "version": original["version"],
        },
        "output_cache": str(Path(args.output_cache).resolve()),
        "source_checks": source_checks,
        "train_sequence_sources": sequence_source_checks["sequences"],
        "image_sources": image_sources,
        "window_policy": {
            "uniform_window_count_per_sequence": UNIFORM_WINDOW_COUNT,
            "uniform_window_frames": UNIFORM_WINDOW_FRAMES,
            "uniform_eligibility": {
                "minimum_gt_and_raw_valid_fraction": 0.5,
                "minimum_contiguous_gt_and_raw_valid_frames": 16,
                "one_slam_segment": True,
                "maximum_adjacent_dt_sec": MAX_GAP_SEC,
                "time_uniformity": "nearest eligible start to four equally spaced center-time targets",
            },
            "event_window_max_per_sequence": EVENT_WINDOW_LIMIT,
            "event_window_nominal_frames": EVENT_WINDOW_FRAMES,
            "event_window_clip": "one segment and one <=250ms-contiguous run",
            "gt_event_source": "train target_legacy -> legacy_to_state/state_to_legacy -> MANO joints",
            "gt_event_threshold": PINCH_THRESHOLD,
            "gt_event_min_frames": PINCH_MIN_FRAMES,
            "gt_event_max_gap_sec": MAX_GAP_SEC,
            "ha_wor_context": "real 16-frame inference context; resets at original SLAM segment boundaries",
            "label_rows": "all requested rows, including gt_valid=false or visual_valid=false; masks remain in labels",
        },
        "counts": counts,
        "training_windows": windows,
        "event_window_supervision": event_window_audit,
        "contexts": contexts,
        "estimated_patch_feature_bytes": requested_frame_rows * 192 * 1280 * 2,
        "estimated_patch_feature_gib": round(
            requested_frame_rows * 192 * 1280 * 2 / (1024 ** 3), 3),
    }
    plan["plan_sha256"] = _plan_sha256(plan)
    return plan


def _load_plan_and_sources(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, dict[str, Any], list[PreparedSequence], dict[str, Any],
           dict[str, dict[str, Any]]]:
    parent_cache = Path(args.parent_cache).resolve()
    output_cache = Path(args.output_cache).resolve()
    if output_cache == parent_cache or parent_cache in output_cache.parents:
        raise ValueError("Output cache must be separate from and outside the read-only parent cache")
    original, parent_manifest_sha = _load_parent_cache(parent_cache)
    source_checks = _verify_parent_protocol_and_checkpoints(args, original)
    all_sequence_sources = _verify_parent_sequence_sources(args, original)
    source_checks["parent_sequence_sources"] = all_sequence_sources
    source_checks["parent_event_sources"] = _verify_parent_event_sources(original)
    verified_parent_images, parent_image_audit = _verify_parent_rgb_sources(original)
    source_checks["parent_rgb_image_inventory"] = parent_image_audit
    sequences, sequence_source_checks = _load_verified_train_sequences(
        args, original, all_sequence_sources)
    return (original, parent_manifest_sha, source_checks, sequences,
            sequence_source_checks, verified_parent_images)


def _prepare_system(
    args: argparse.Namespace,
    original: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, str]]:
    seed_everything(7)
    config, twm_state = load_base_twm(args.twm_checkpoint, EXPECTED_TWM_SHA256)
    system = build_system(args.hawor_checkpoint, config, device)
    system.observer.twm.load_state_dict(twm_state, strict=True)
    install_feedback_v2(system)
    freeze_except_visual(system)
    system.eval()
    system.hawor.eval()
    system.observer.twm.eval()
    frozen = {
        "hawor_parameters_sha256": module_parameter_sha256(system.hawor),
        "twm_parameters_sha256": module_parameter_sha256(system.observer.twm),
        "geometry_parameters_sha256": module_parameter_sha256(system.geometry),
    }
    if frozen != original.get("frozen_parameter_hashes"):
        raise ValueError("Fresh frozen model parameter hashes differ from the parent v3 cache")
    return system, frozen


def _context_output_paths(
    output_cache: Path,
    context: dict[str, Any],
) -> tuple[Path, Path]:
    relative = Path("train_extension_v4") / context["sequence"] / (
        f"{context['kind']}_{int(context['window_index']):02d}")
    stem = f"ctx_{int(context['context_start']):06d}"
    inputs_path = (output_cache / relative / f"{stem}.inputs.pt").resolve()
    labels_path = (output_cache / relative / f"{stem}.labels.pt").resolve()
    if output_cache.resolve() not in inputs_path.parents or output_cache.resolve() not in labels_path.parents:
        raise ValueError("Resolved context paths escaped output_cache")
    return inputs_path, labels_path


def _extract_training_context(
    *,
    system: torch.nn.Module,
    sequence: PreparedSequence,
    context: dict[str, Any],
    device: torch.device,
    output_cache: Path,
) -> dict[str, Any]:
    context_start = int(context["context_start"])
    requested_frames = [int(frame) for frame in context["frame_indices"]]
    full_frame_ids = _segment_context_frame_ids(sequence, context_start)
    if (len(full_frame_ids) < 1 or len(full_frame_ids) > 16
            or not set(requested_frames).issubset(set(full_frame_ids.tolist()))):
        raise ValueError(
            f"{sequence.sequence}:{context_start}: requested rows exceed the original segment context")
    full_samples = [(sequence, int(frame)) for frame in full_frame_ids]
    model_inputs, _beta, records = build_live_batch(system, full_samples, device)
    row_by_frame = {int(record["frame_index"]): row
                    for row, record in enumerate(records)}
    selected_rows = [row_by_frame[frame] for frame in requested_frames]
    cached_inputs = _stack_context_inputs(model_inputs, selected_rows)
    labels = _labels_for_frames(
        sequence, np.asarray(requested_frames, dtype=np.int64), system.geometry, device)

    inputs_path, labels_path = _context_output_paths(output_cache, context)
    _atomic_torch_save(cached_inputs, inputs_path)
    _atomic_torch_save(labels, labels_path)
    cached_records = [records[row_by_frame[frame]] for frame in requested_frames]
    return {
        "item_id": str(context["item_id"]),
        "split": "train",
        "sequence": sequence.sequence,
        "context_start": context_start,
        "frame_indices": requested_frames,
        "inputs_path": str(inputs_path),
        "labels_path": str(labels_path),
        "inputs_sha256": sha256_file(inputs_path),
        "labels_sha256": sha256_file(labels_path),
        "patch_dtype_on_disk": "float16",
        "input_keys": sorted(cached_inputs),
        "label_keys": sorted(labels),
        "record_timestamps": [float(record["timestamp_sec"]) for record in cached_records],
        "record_segment_ids": [int(record["segment_id"]) for record in cached_records],
        "cache_source": "new_frozen_hawor_training_extension_v4",
        "window_kind": str(context["kind"]),
        "window_index": int(context["window_index"]),
        "extraction_context_frame_indices": full_frame_ids.tolist(),
    }


def _append_progress(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_verified_progress(
    output_cache: Path,
    contexts: list[dict[str, Any]],
    plan_sha256: str,
) -> dict[str, dict[str, Any]]:
    progress_path = output_cache / "training_extension_progress.jsonl"
    if not progress_path.exists():
        return {}
    planned = {str(context["item_id"]): context for context in contexts}
    completed: dict[str, dict[str, Any]] = {}
    with progress_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Corrupt extraction progress line {line_number}") from error
            if (record.get("schema") != "handloop_intra_v4_training_context_progress"
                    or record.get("plan_sha256") != plan_sha256):
                raise ValueError(f"Progress line {line_number} belongs to a different cache plan")
            item = record.get("item")
            if not isinstance(item, dict):
                raise ValueError(f"Progress line {line_number} lacks a cache item record")
            item_id = str(item.get("item_id", ""))
            context = planned.get(item_id)
            if context is None:
                raise ValueError(f"Progress log contains an unplanned context {item_id}")
            if (item.get("frame_indices") != context["frame_indices"]
                    or int(item.get("context_start", -1)) != int(context["context_start"])
                    or item.get("sequence") != context["sequence"]):
                raise ValueError(f"Progress item {item_id} differs from the current extraction plan")
            inputs_path = Path(item.get("inputs_path", "")).resolve()
            labels_path = Path(item.get("labels_path", "")).resolve()
            expected_inputs, expected_labels = _context_output_paths(output_cache, context)
            if inputs_path != expected_inputs or labels_path != expected_labels:
                raise ValueError(f"Progress item {item_id} points outside its planned output files")
            for file_path, hash_key in ((inputs_path, "inputs_sha256"),
                                         (labels_path, "labels_sha256")):
                if not file_path.is_file() or sha256_file(file_path) != item.get(hash_key):
                    raise ValueError(f"Progress cache file is missing or SHA-changed: {file_path}")
            previous = completed.get(item_id)
            if previous is not None and previous != item:
                raise ValueError(f"Progress log has conflicting records for {item_id}")
            completed[item_id] = item

    for context in contexts:
        item_id = str(context["item_id"])
        if item_id in completed:
            continue
        inputs_path, labels_path = _context_output_paths(output_cache, context)
        if inputs_path.exists() or labels_path.exists():
            raise ValueError(
                f"Unlogged partial context files exist for {item_id}; refusing silent reuse: "
                f"{inputs_path}, {labels_path}")
    return completed


def _verify_cached_context_files(item: dict[str, Any]) -> None:
    inputs_path = Path(item["inputs_path"])
    labels_path = Path(item["labels_path"])
    if sha256_file(inputs_path) != item["inputs_sha256"]:
        raise ValueError(f"New input cache SHA256 mismatch: {inputs_path}")
    if sha256_file(labels_path) != item["labels_sha256"]:
        raise ValueError(f"New label cache SHA256 mismatch: {labels_path}")
    inputs = torch.load(inputs_path, map_location="cpu", weights_only=True)
    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    if not isinstance(inputs, dict) or not isinstance(labels, dict):
        raise ValueError(f"Malformed new cache item {item['item_id']}")
    frame_ids = [int(value) for value in item["frame_indices"]]
    if (inputs["patches"].dtype != torch.float16
            or inputs["patches"].shape[0] != len(frame_ids)
            or labels["frame_indices"].tolist() != frame_ids):
        raise ValueError(f"Cached rows or patch dtype mismatch for {item['item_id']}")


def _build_final_manifest(
    *,
    original: dict[str, Any],
    parent_cache: Path,
    parent_manifest_sha: str,
    plan: dict[str, Any],
    frozen_parameter_hashes: dict[str, str],
    new_items: list[dict[str, Any]],
) -> dict[str, Any]:
    items = copy.deepcopy(original["items"]) + copy.deepcopy(new_items)
    counts = copy.deepcopy(original.get("counts", {}))
    extension_frame_rows = int(plan["counts"]["extension_training_window_frame_rows"])
    supervised_rows = 0
    for item in new_items:
        labels = torch.load(item["labels_path"], map_location="cpu", weights_only=True)
        supervised_rows += int((labels["gt_valid"] & labels["visual_valid"]).sum())
    counts.update({
        "train_contexts": EXPECTED_PARENT_TRAIN_ITEM_COUNT + len(new_items),
        "train_supervised_frames": int(counts.get("train_supervised_frames", 0)) + supervised_rows,
        "reused_train_items": EXPECTED_PARENT_TRAIN_ITEM_COUNT,
        "training_extension_revision": TRAINING_EXTENSION_REVISION,
        "training_extension_windows": len(plan["training_windows"]),
        "training_extension_contexts": len(new_items),
        "training_extension_frame_rows": extension_frame_rows,
        "training_extension_supervised_rows": supervised_rows,
        "training_extension_unique_frames": int(
            plan["counts"]["extension_unique_training_frames"]),
    })

    source_hashes = copy.deepcopy(original.get("source_hashes", {}))
    source_hashes["training_extension_v4_parent_cache_manifest"] = {
        "path": str((parent_cache / "manifest.json").resolve()),
        "sha256": parent_manifest_sha,
        "schema": original["schema"],
        "version": original["version"],
    }
    source_hashes["training_extension_v4"] = {
        "plan_sha256": plan["plan_sha256"],
        "source_checks": copy.deepcopy(plan["source_checks"]),
        "train_sequence_sources": copy.deepcopy(plan["train_sequence_sources"]),
        "rgb_images": copy.deepcopy(plan["image_sources"]),
        "ha_wor_checkpoint_sha256": plan["source_checks"]["base_checkpoints"]["hawor_checkpoint"]["sha256"],
        "twm_checkpoint_sha256": plan["source_checks"]["base_checkpoints"]["twm_checkpoint"]["sha256"],
    }

    result = copy.deepcopy(original)
    result.update({
        "protocol_revision": PROTOCOL_REVISION,
        "training_extension_revision": TRAINING_EXTENSION_REVISION,
        "training_extension_plan_sha256": plan["plan_sha256"],
        "training_windows": copy.deepcopy(plan["training_windows"]),
        "items": items,
        "counts": counts,
        "source_hashes": source_hashes,
        "frozen_parameter_hashes": frozen_parameter_hashes,
        "training_extension": {
            "revision": TRAINING_EXTENSION_REVISION,
            "parent_manifest": {
                "path": str((parent_cache / "manifest.json").resolve()),
                "sha256": parent_manifest_sha,
                "schema": original["schema"],
                "version": original["version"],
                "protocol_revision": PROTOCOL_REVISION,
            },
            "feature_source": "same frozen HaWoR IAM and feedback_v2 model-input protocol as parent",
            "labels": "GT-only targets in separate .labels.pt files; every requested row is retained with validity masks",
            "context_rule": "actual segment-reset 16-frame HaWoR input; only requested window rows are stored",
            "window_policy": copy.deepcopy(plan["window_policy"]),
            "counts": copy.deepcopy(plan["counts"]),
            "frozen_parameter_hashes": frozen_parameter_hashes,
        },
    })
    return result


def _write_plan(output_cache: Path, plan: dict[str, Any], *, replace: bool) -> Path:
    output_cache.mkdir(parents=True, exist_ok=True)
    plan_path = output_cache / "training_extension_plan.json"
    if plan_path.exists() and not replace:
        previous = _read_json(plan_path)
        if previous.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError(
                f"Existing plan differs from current verified source plan: {plan_path}")
        return plan_path
    _atomic_json(plan, plan_path)
    return plan_path


def _verify_plan_file(output_cache: Path, plan: dict[str, Any]) -> Path:
    path = output_cache / "training_extension_plan.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Run --plan-only and review its JSON before extraction: {path}")
    saved = _read_json(path)
    if (saved.get("schema") != "handloop_intra_v4_training_cache_plan"
            or saved.get("plan_sha256") != plan["plan_sha256"]):
        raise ValueError("Saved plan does not match current verified sources/windows")
    return path


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    (original, parent_manifest_sha, source_checks, sequences,
     sequence_source_checks, verified_parent_images) = _load_plan_and_sources(args)
    cpu_geometry = ManoGeometry(ROOT).to(torch.device("cpu"))
    cpu_geometry.eval()
    plan = _build_plan(
        args=args,
        original=original,
        parent_manifest_sha=parent_manifest_sha,
        source_checks=source_checks,
        sequences=sequences,
        sequence_source_checks=sequence_source_checks,
        verified_parent_images=verified_parent_images,
        geometry=cpu_geometry,
    )
    output_cache = Path(args.output_cache).resolve()
    if args.plan_only:
        plan_path = _write_plan(output_cache, plan, replace=True)
        summary = {
            "plan_only": True,
            "plan_path": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "parent_cache": plan["parent_cache"],
            "output_cache": str(output_cache),
            "counts": plan["counts"],
            "estimated_patch_feature_gib": plan["estimated_patch_feature_gib"],
            "training_window_sample": [
                {
                    "sequence": window["sequence"],
                    "window_index": window["window_index"],
                    "kind": window["kind"],
                    "frame_count": len(window["frame_indices"]),
                    "context_count": len(window["context_item_ids"]),
                    **({"gt_onset_times_sec": window["gt_onset_times_sec"]}
                       if "gt_onset_times_sec" in window else {}),
                }
                for window in plan["training_windows"][:4]
            ],
            "event_window_supervision": plan["event_window_supervision"],
            "verified_train_source_count": len(plan["train_sequence_sources"]),
            "requested_image_hash_count": len(plan["image_sources"]),
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    _verify_plan_file(output_cache, plan)
    final_manifest = output_cache / "manifest.json"
    if final_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite completed v4 cache: {final_manifest}")
    output_cache.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA extraction requested but CUDA is unavailable: {device}")
    if device.type == "cuda" and device.index not in (None, 0):
        raise ValueError("This cache extraction is assigned to physical GPU 1 as cuda:0")

    system, frozen_hashes = _prepare_system(args, original, device)
    convert_raw_history_states(sequences, system.geometry, device)
    sequence_by_name = {sequence.sequence: sequence for sequence in sequences}
    contexts = plan["contexts"]
    completed = _load_verified_progress(output_cache, contexts, plan["plan_sha256"])
    new_items: list[dict[str, Any]] = []
    progress_path = output_cache / "training_extension_progress.jsonl"
    for context_index, context in enumerate(contexts, start=1):
        item_id = str(context["item_id"])
        previous = completed.get(item_id)
        if previous is not None:
            _verify_cached_context_files(previous)
            new_items.append(previous)
            continue
        item = _extract_training_context(
            system=system,
            sequence=sequence_by_name[str(context["sequence"])],
            context=context,
            device=device,
            output_cache=output_cache,
        )
        _verify_cached_context_files(item)
        _append_progress(progress_path, {
            "schema": "handloop_intra_v4_training_context_progress",
            "plan_sha256": plan["plan_sha256"],
            "completed_context_index": context_index,
            "context_count": len(contexts),
            "item": item,
        })
        new_items.append(item)
        if context_index == 1 or context_index % 25 == 0 or context_index == len(contexts):
            print(json.dumps({
                "event": "context_complete",
                "completed": context_index,
                "total": len(contexts),
                "item_id": item_id,
                "inputs_sha256": item["inputs_sha256"],
                "labels_sha256": item["labels_sha256"],
            }, sort_keys=True), flush=True)

    item_by_id = {str(item["item_id"]): item for item in new_items}
    if len(item_by_id) != len(contexts):
        raise ValueError("Completed cache item count differs from the verified plan")
    coverage_rows = {item_id: item["frame_indices"] for item_id, item in item_by_id.items()}
    for window in plan["training_windows"]:
        actual = []
        for item_id in window["context_item_ids"]:
            if item_id not in coverage_rows:
                raise ValueError(f"Missing planned training item {item_id}")
            actual.extend(int(frame) for frame in coverage_rows[item_id])
        expected = [int(frame) for frame in window["frame_indices"]]
        if sorted(actual) != sorted(expected) or len(actual) != len(set(actual)):
            raise ValueError(
                f"{window['sequence']}/{window['kind']}/{window['window_index']}: "
                "final item labels do not cover requested frame IDs exactly once")

    manifest = _build_final_manifest(
        original=original,
        parent_cache=Path(args.parent_cache).resolve(),
        parent_manifest_sha=parent_manifest_sha,
        plan=plan,
        frozen_parameter_hashes=frozen_hashes,
        new_items=[item_by_id[str(context["item_id"])] for context in contexts],
    )
    # Revalidate every parent reference before atomically publishing the final manifest.
    for item in manifest["items"]:
        for path_key, hash_key in (("inputs_path", "inputs_sha256"),
                                   ("labels_path", "labels_sha256")):
            path = Path(item[path_key]).resolve()
            if not path.is_file() or sha256_file(path) != item[hash_key]:
                raise ValueError(f"Refusing to publish with missing/changed cache file: {path}")
    _atomic_json(manifest, final_manifest)
    print(json.dumps({
        "event": "cache_complete",
        "manifest": str(final_manifest),
        "manifest_sha256": sha256_file(final_manifest),
        "plan_sha256": plan["plan_sha256"],
        "counts": manifest["counts"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

