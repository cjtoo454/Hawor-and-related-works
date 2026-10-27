#!/usr/bin/env python3
"""Prepare fixed, GT-only calibration event coverage for the intra-frame study.

This reads only the frozen calibration split, decodes its original raw-SLAM
MANO states into MANO-21 joints, and combines complete reference-event windows
with the original fixed v2 calibration windows. It does not extract image
features or load any model beyond ManoGeometry.
"""
from __future__ import annotations

import argparse
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
sys.path[:0] = [str(ROOT), str(TRY), str(TRY / "scripts"), str(HERE)]

from dual_loop.geometry import ManoGeometry  # noqa: E402
from dual_loop.oracle import normalized_pinch_distance  # noqa: E402
from dual_loop.pinch_evaluation import extract_events  # noqa: E402
from prepare_training import context_start_indices, load_sequences  # noqa: E402


REFERENCE_THRESHOLD = 0.2
PREDICTION_THRESHOLD = 0.275
MIN_EVENT_FRAMES = 3
MAX_GAP_SEC = 0.25
CONTEXT_FRAMES_EACH_SIDE = 30
BASELINE_WINDOW_FRAMES = 128
BATCH_SIZE = 256
EVALUATION_MIN_FRAMES = 16
CACHE_MANIFEST = HERE / "runs" / "feedback_v2_train" / "cache" / "manifest.json"
BASELINE_RESULT = HERE / "runs" / "feedback_v2_train" / "baseline_calibration.json"
DEFAULT_OUTPUT = HERE / "runs" / "calibration_events_v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _continuous_time_runs(times: np.ndarray) -> list[tuple[int, int]]:
    """Half-open spans split only at non-positive or >250ms time steps."""
    if not len(times):
        return []
    boundaries = [0]
    for index in range(1, len(times)):
        delta = times[index] - times[index - 1]
        if delta <= 0.0 or delta > MAX_GAP_SEC:
            boundaries.append(index)
    boundaries.append(len(times))
    return list(zip(boundaries[:-1], boundaries[1:]))


def _formal_evaluation_mask(gt_valid: np.ndarray, visual_valid: np.ndarray) -> np.ndarray:
    """Mirror eval_hot3d.parse_chunks: contiguous valid frame IDs, min length 16."""
    valid = np.asarray(gt_valid, dtype=bool) & np.asarray(visual_valid, dtype=bool)
    frame_ids = np.flatnonzero(valid)
    boundaries = np.flatnonzero(np.diff(frame_ids) != 1) + 1
    chunks = np.split(frame_ids, boundaries)
    result = np.zeros(len(valid), dtype=bool)
    for chunk in chunks:
        if len(chunk) >= EVALUATION_MIN_FRAMES:
            result[chunk] = True
    return result


def _sequence_events(
    sequence: Any,
    joints: np.ndarray,
    distances: np.ndarray,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]], np.ndarray]:
    times = np.asarray(sequence.timestamp_sec, dtype=np.float64)
    segments = np.asarray(sequence.segment_id, dtype=np.int64)
    valid = (np.asarray(sequence.gt_valid, dtype=bool)
             & np.isfinite(joints).all(axis=(1, 2))
             & np.isfinite(distances))
    runs = _continuous_time_runs(times)
    rows: list[dict[str, Any]] = []
    run_id_for_frame = np.empty(len(times), dtype=np.int64)
    run_bounds: list[tuple[int, int]] = []
    for run_id, (left, right) in enumerate(runs):
        run_id_for_frame[left:right] = run_id
        run_bounds.append((left, right))
        found = extract_events(
            distances[left:right], times[left:right], valid[left:right],
            REFERENCE_THRESHOLD, min_frames=MIN_EVENT_FRAMES, max_gap_s=MAX_GAP_SEC,
        )
        for event in found:
            start, stop = left + int(event.start), left + int(event.stop)
            segment_ids = sorted(set(map(int, segments[start:stop + 1])))
            rows.append({
                "event_id": f"{sequence.sequence}:event_{len(rows):04d}",
                "sequence": sequence.sequence,
                "start_frame": start,
                "stop_frame": stop,
                "onset_sec": float(event.onset_sec),
                "offset_sec": float(event.offset_sec),
                "duration_frames": stop - start + 1,
                "segment_id": segment_ids[0] if len(segment_ids) == 1 else None,
                "segment_ids": segment_ids,
                "continuous_run_id": run_id,
                "valid_bracketed": True,
            })
    return rows, run_bounds, run_id_for_frame


def _events_in_frame_ids(
    frame_ids: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    distances: np.ndarray,
) -> list[dict[str, Any]]:
    selected = extract_events(
        distances[frame_ids], times[frame_ids], valid[frame_ids],
        REFERENCE_THRESHOLD, min_frames=MIN_EVENT_FRAMES, max_gap_s=MAX_GAP_SEC,
    )
    return [{"start_frame": int(frame_ids[event.start]),
             "stop_frame": int(frame_ids[event.stop]),
             "onset_sec": float(event.onset_sec),
             "offset_sec": float(event.offset_sec)} for event in selected]


def _merge_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping windows within time-continuous runs, including resets."""
    ordered = sorted(windows, key=lambda row: (
        row["sequence"], row["continuous_run_id"], row["start_frame"], row["stop_frame_exclusive"]))
    merged: list[dict[str, Any]] = []
    for source in ordered:
        row = {
            "sequence": source["sequence"],
            "segment_ids": sorted(set(map(int, source.get("segment_ids", [])))),
            "continuous_run_id": int(source["continuous_run_id"]),
            "start_frame": int(source["start_frame"]),
            "stop_frame_exclusive": int(source["stop_frame_exclusive"]),
            "event_ids": list(source.get("event_ids", [])),
            "baseline_window_ids": list(source.get("baseline_window_ids", [])),
        }
        if (merged and merged[-1]["sequence"] == row["sequence"]
                and merged[-1]["continuous_run_id"] == row["continuous_run_id"]
                and row["start_frame"] <= merged[-1]["stop_frame_exclusive"]):
            previous = merged[-1]
            previous["stop_frame_exclusive"] = max(
                previous["stop_frame_exclusive"], row["stop_frame_exclusive"])
            previous["event_ids"] = sorted(set(previous["event_ids"] + row["event_ids"]))
            previous["baseline_window_ids"] = sorted(
                set(previous["baseline_window_ids"] + row["baseline_window_ids"]))
            previous["segment_ids"] = sorted(set(previous["segment_ids"] + row["segment_ids"]))
        else:
            merged.append(row)
    for index, row in enumerate(merged):
        row["segment_id"] = row["segment_ids"][0] if len(row["segment_ids"]) == 1 else None
        row["window_id"] = f"merged_{index:04d}"
        row["frame_count"] = row["stop_frame_exclusive"] - row["start_frame"]
    return merged


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(arrays: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _window_segments(segments: np.ndarray, start: int, stop: int) -> list[int]:
    return sorted(set(map(int, np.asarray(segments)[start:stop])))


def _decode_gt(sequence: Any, geometry: ManoGeometry, device: torch.device) -> np.ndarray:
    """Decode only valid original 109D GT MANO states in the raw SLAM world."""
    count = len(sequence.timestamp_sec)
    joints = np.full((count, 21, 3), np.nan, dtype=np.float32)
    valid_ids = np.flatnonzero(sequence.gt_valid)
    with torch.inference_mode():
        for start in range(0, len(valid_ids), BATCH_SIZE):
            ids = valid_ids[start:start + BATCH_SIZE]
            states = torch.as_tensor(sequence.target_legacy[ids], dtype=torch.float32, device=device)
            decoded = geometry.joints(states)
            if decoded.shape != (len(ids), 21, 3):
                raise ValueError(f"{sequence.sequence}: unexpected MANO joint shape {tuple(decoded.shape)}")
            joints[ids] = decoded.detach().float().cpu().numpy()
    if np.any(sequence.gt_valid & ~np.isfinite(joints).all(axis=(1, 2))):
        bad = np.flatnonzero(sequence.gt_valid & ~np.isfinite(joints).all(axis=(1, 2)))
        raise FloatingPointError(f"{sequence.sequence}: non-finite GT MANO joints at {bad[:8].tolist()}")
    return joints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0 (CUDA requires CUDA_VISIBLE_DEVICES=1)")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-existing-gt", action="store_true",
                        help="Reuse emitted GT joints and only refresh masks/events/manifest")
    args = parser.parse_args()
    reuse_existing_gt = args.reuse_existing_gt
    if args.output_dir.exists() and not reuse_existing_gt:
        raise FileExistsError(f"Refusing to overwrite event calibration artifacts: {args.output_dir}")
    if reuse_existing_gt and not (args.output_dir / "gt").is_dir():
        raise FileNotFoundError("--reuse-existing-gt requires the existing gt/ output directory")
    if not CACHE_MANIFEST.is_file() or not BASELINE_RESULT.is_file():
        raise FileNotFoundError("Expected frozen v2 cache manifest and baseline calibration result")
    if args.device.startswith("cuda"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("For GPU work, expose only physical GPU 1 with CUDA_VISIBLE_DEVICES=1")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA device (physical GPU 1)")
    device = torch.device(args.device)

    # Deliberately request only calibration; test sequence names may be checked
    # by load_sequences' frozen split validation but their archives are not read.
    loaded, _protocol = load_sequences(splits=("calibration",))
    sequences = loaded["calibration"]
    if len(sequences) != 20 or any(sequence.split != "calibration" for sequence in sequences):
        raise ValueError(f"Expected the frozen 20-sequence calibration split, got {len(sequences)}")
    cache = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_RESULT.read_text(encoding="utf-8"))
    base_windows = cache.get("calibration_windows")
    if not isinstance(base_windows, list) or len(base_windows) != 40:
        raise ValueError("Expected exactly 40 fixed v2 calibration windows")
    if baseline.get("events", {}).get("reference_events") != 8:
        raise ValueError("Stored v2 baseline no longer reports its frozen 8 reference events")

    prior_events_path = args.output_dir / "events.json"
    prior_events: list[dict[str, Any]] = []
    if reuse_existing_gt and prior_events_path.is_file():
        prior_events = json.loads(prior_events_path.read_text(encoding="utf-8")).get("events", [])
    geometry = None if reuse_existing_gt else ManoGeometry(ROOT).to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=reuse_existing_gt)
    gt_dir = args.output_dir / "gt"
    gt_dir.mkdir(exist_ok=reuse_existing_gt)

    sequence_rows: dict[str, dict[str, Any]] = {}
    sequence_events: dict[str, list[dict[str, Any]]] = {}
    run_bounds_by_sequence: dict[str, list[tuple[int, int]]] = {}
    run_id_for_frame_by_sequence: dict[str, np.ndarray] = {}
    gt_arrays_by_sequence: dict[str, dict[str, np.ndarray]] = {}
    for index, sequence in enumerate(sequences, 1):
        gt_path = gt_dir / f"{sequence.sequence}.npz"
        sequence_json_path = gt_dir / f"{sequence.sequence}.json"
        expected_archive_hash = sha256_file(sequence.archive_path)
        if reuse_existing_gt:
            if not gt_path.is_file() or not sequence_json_path.is_file():
                raise FileNotFoundError(f"Cannot reuse missing GT artifact for {sequence.sequence}")
            prior_sequence = json.loads(sequence_json_path.read_text(encoding="utf-8"))
            if prior_sequence.get("source_archive_sha256") != expected_archive_hash:
                raise ValueError(f"Prepared source archive changed for {sequence.sequence}")
            with np.load(gt_path, allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in archive.files}
            frame_ids = np.arange(len(sequence), dtype=np.int64)
            if (not np.array_equal(saved["frame_ids"], frame_ids)
                    or not np.array_equal(saved["timestamp_sec"], sequence.timestamp_sec)
                    or not np.array_equal(saved["segment_id"], sequence.segment_id)
                    or not np.array_equal(saved["gt_valid"], sequence.gt_valid)):
                raise ValueError(f"Existing GT indices, times, segments, or GT mask changed for {sequence.sequence}")
            joints = saved["gt_joints_world_m"]
            distances = saved["normalized_pinch_distance"]
            if (joints.shape != (len(sequence), 21, 3)
                    or distances.shape != (len(sequence),)
                    or np.any(sequence.gt_valid & ~np.isfinite(joints).all(axis=(1, 2)))):
                raise ValueError(f"Existing GT joints/distances are malformed for {sequence.sequence}")
        else:
            assert geometry is not None
            joints = _decode_gt(sequence, geometry, device)
            distances = normalized_pinch_distance(joints).astype(np.float32)
            saved = {
                "sequence": np.asarray(sequence.sequence),
                "split": np.asarray("calibration"),
                "coordinate_frame": np.asarray("per-interval local raw HaWoR SLAM world"),
                "frame_ids": np.arange(len(sequence), dtype=np.int64),
                "timestamp_sec": np.asarray(sequence.timestamp_sec, dtype=np.float64),
                "segment_id": np.asarray(sequence.segment_id, dtype=np.int64),
                "gt_valid": np.asarray(sequence.gt_valid, dtype=bool),
                "gt_joints_world_m": joints,
                "normalized_pinch_distance": distances,
            }

        times = np.asarray(sequence.timestamp_sec, dtype=np.float64)
        segments = np.asarray(sequence.segment_id, dtype=np.int64)
        gt_valid = np.asarray(sequence.gt_valid, dtype=bool)
        visual_valid = np.asarray(sequence.visual_valid, dtype=bool)
        evaluation_mask = _formal_evaluation_mask(gt_valid, visual_valid)
        events, run_bounds, run_id_for_frame = _sequence_events(sequence, joints, distances)
        sequence_events[sequence.sequence] = events
        run_bounds_by_sequence[sequence.sequence] = run_bounds
        gt_arrays_by_sequence[sequence.sequence] = {
            "joints": joints,
            "distances": distances,
            "times": times,
            "valid": gt_valid,
            "visual_valid": visual_valid,
            "evaluation_mask": evaluation_mask,
            "segments": segments,
        }
        run_id_for_frame_by_sequence[sequence.sequence] = run_id_for_frame
        mask_hashes = {
            "timestamp_sha256": _array_sha256(times),
            "gt_valid_sha256": _array_sha256(gt_valid),
            "visual_valid_sha256": _array_sha256(visual_valid),
            "evaluation_mask_sha256": _array_sha256(evaluation_mask),
            "segment_id_sha256": _array_sha256(segments),
        }
        saved.update({
            "continuous_run_id": run_id_for_frame,
            "visual_valid": visual_valid,
            "formal_reference_valid": gt_valid,
            "evaluation_mask": evaluation_mask,
            "formal_prediction_eligible": evaluation_mask,
            "source_archive_sha256": np.asarray(expected_archive_hash),
            **{key: np.asarray(value) for key, value in mask_hashes.items()},
        })
        _atomic_npz(saved, gt_path)
        sequence_rows[sequence.sequence] = {
            "sequence": sequence.sequence,
            "split": sequence.split,
            "coordinate_frame": "per-interval local raw HaWoR SLAM world",
            "source_archive": str(sequence.archive_path),
            "source_archive_sha256": expected_archive_hash,
            "frame_count": len(sequence),
            "gt_valid_frame_count": int(gt_valid.sum()),
            "visual_valid_frame_count": int(visual_valid.sum()),
            "evaluation_mask_frame_count": int(evaluation_mask.sum()),
            "gt_joint_count": int(np.isfinite(joints).all(axis=(1, 2)).sum()),
            "complete_reference_event_count": len(events),
            "gt_file": f"gt/{sequence.sequence}.npz",
            **mask_hashes,
        }
        _atomic_json({**sequence_rows[sequence.sequence], "events": events}, sequence_json_path)
        print(f"[{index:02d}/20] {sequence.sequence}: gt_valid={int(gt_valid.sum())} "
              f"eval_mask={int(evaluation_mask.sum())} events={len(events)}",
              flush=True)

    sequence_by_name = {sequence.sequence: sequence for sequence in sequences}
    base_frame_sets: dict[str, set[int]] = {name: set() for name in sequence_by_name}
    base_context_starts: dict[str, set[int]] = {name: set() for name in sequence_by_name}
    base_source_rows: list[dict[str, Any]] = []
    baseline_reference_events = 0
    for window in base_windows:
        name = str(window["sequence"])
        if name not in sequence_by_name:
            raise ValueError(f"v2 calibration manifest includes non-calibration sequence {name}")
        sequence = sequence_by_name[name]
        frame_ids = np.asarray(window["frame_indices"], dtype=np.int64)
        if (frame_ids.shape != (BASELINE_WINDOW_FRAMES,)
                or not np.array_equal(frame_ids, np.arange(frame_ids[0], frame_ids[0] + BASELINE_WINDOW_FRAMES))):
            raise ValueError(f"Expected a contiguous 128-frame base window: {name} {window.get('window_index')}")
        if frame_ids[0] < 0 or frame_ids[-1] >= len(sequence):
            raise ValueError(f"v2 base window is out of range for {name}")
        segments = np.asarray(sequence.segment_id, dtype=np.int64)
        times = np.asarray(sequence.timestamp_sec, dtype=np.float64)
        if (np.any(segments[frame_ids] != segments[frame_ids[0]])
                or np.any(np.diff(times[frame_ids]) > MAX_GAP_SEC)):
            raise ValueError(f"v2 base window crosses a segment or >250ms time gap: {name}")
        base_frame_sets[name].update(map(int, frame_ids))
        for item_id in window["context_item_ids"]:
            base_context_starts[name].add(int(str(item_id).rsplit("ctx_", 1)[1]))
        base_id = f"{name}:window_{int(window['window_index']):02d}"
        gt_arrays = gt_arrays_by_sequence[name]
        local_events = _events_in_frame_ids(
            frame_ids, gt_arrays["times"], gt_arrays["valid"], gt_arrays["distances"],
        )
        baseline_reference_events += len(local_events)
        run_id = int(run_id_for_frame_by_sequence[name][frame_ids[0]])
        base_source_rows.append({
            "source": "fixed_v2_calibration_window",
            "sequence": name,
            "baseline_window_id": base_id,
            "window_index": int(window["window_index"]),
            "start_frame": int(frame_ids[0]),
            "stop_frame_exclusive": int(frame_ids[-1] + 1),
            "segment_id": int(segments[frame_ids[0]]),
            "segment_ids": [int(segments[frame_ids[0]])],
            "continuous_run_id": run_id,
            "context_item_ids": list(window["context_item_ids"]),
        })
    if baseline_reference_events != int(baseline["events"]["reference_events"]):
        raise ValueError(
            "Recomputed GT-only references on the fixed 40 windows do not match the frozen baseline: "
            f"{baseline_reference_events} != {baseline['events']['reference_events']}"
        )

    event_source_rows: list[dict[str, Any]] = []
    for name, events in sequence_events.items():
        for event in events:
            run_start, run_stop = run_bounds_by_sequence[name][event["continuous_run_id"]]
            start = max(run_start, event["start_frame"] - CONTEXT_FRAMES_EACH_SIDE)
            stop = min(run_stop, event["stop_frame"] + CONTEXT_FRAMES_EACH_SIDE + 1)
            segment_ids = _window_segments(sequence_by_name[name].segment_id, start, stop)
            context = {
                "source": "complete_gt_event_context",
                "sequence": name,
                "event_ids": [event["event_id"]],
                "start_frame": int(start),
                "stop_frame_exclusive": int(stop),
                "segment_id": segment_ids[0] if len(segment_ids) == 1 else None,
                "segment_ids": segment_ids,
                "continuous_run_id": int(event["continuous_run_id"]),
                "event_start_frame": int(event["start_frame"]),
                "event_stop_frame": int(event["stop_frame"]),
                "context_before_frames": int(event["start_frame"] - start),
                "context_after_frames": int(stop - event["stop_frame"] - 1),
            }
            event_source_rows.append(context)

    source_windows = []
    for row in base_source_rows:
        source_windows.append({
            "sequence": row["sequence"], "segment_id": row["segment_id"],
            "segment_ids": row["segment_ids"],
            "continuous_run_id": row["continuous_run_id"],
            "start_frame": row["start_frame"], "stop_frame_exclusive": row["stop_frame_exclusive"],
            "baseline_window_ids": [row["baseline_window_id"]],
        })
    for row in event_source_rows:
        source_windows.append({
            "sequence": row["sequence"], "segment_id": row["segment_id"],
            "segment_ids": row["segment_ids"],
            "continuous_run_id": row["continuous_run_id"],
            "start_frame": row["start_frame"], "stop_frame_exclusive": row["stop_frame_exclusive"],
            "event_ids": row["event_ids"],
        })
    merged_windows = _merge_windows(source_windows)

    expanded_frame_sets: dict[str, set[int]] = {name: set() for name in sequence_by_name}
    event_only_frame_sets: dict[str, set[int]] = {name: set() for name in sequence_by_name}
    for row in event_source_rows:
        event_only_frame_sets[row["sequence"]].update(
            range(row["start_frame"], row["stop_frame_exclusive"]))
    for row in merged_windows:
        name = row["sequence"]
        frame_ids = np.arange(row["start_frame"], row["stop_frame_exclusive"], dtype=np.int64)
        frames = set(map(int, frame_ids))
        expanded_frame_sets[name].update(frames)
        sequence = sequence_by_name[name]
        deltas = np.diff(sequence.timestamp_sec[frame_ids])
        if np.any(deltas <= 0) or np.any(deltas > MAX_GAP_SEC):
            raise ValueError(f"Merged window crosses a time gap: {row}")
        actual_segment_ids = _window_segments(
            sequence.segment_id, row["start_frame"], row["stop_frame_exclusive"])
        if actual_segment_ids != row["segment_ids"]:
            raise ValueError(f"Merged window segment metadata differs from frame indices: {row}")

    # Re-extract over the final union to prove all full-sequence events remain
    # bracketed and counted after clipping and interval merging.
    expanded_reference_events = 0
    expanded_event_keys: set[tuple[str, int, int]] = set()
    for row in merged_windows:
        name = row["sequence"]
        frames = np.arange(row["start_frame"], row["stop_frame_exclusive"], dtype=np.int64)
        gt_arrays = gt_arrays_by_sequence[name]
        found = _events_in_frame_ids(
            frames, gt_arrays["times"], gt_arrays["valid"], gt_arrays["distances"])
        expanded_reference_events += len(found)
        expanded_event_keys.update(
            (name, int(event["start_frame"]), int(event["stop_frame"]))
            for event in found
        )
    if expanded_reference_events != sum(map(len, sequence_events.values())):
        raise ValueError(
            "Expanded windows did not preserve every complete GT reference event: "
            f"{expanded_reference_events} != {sum(map(len, sequence_events.values()))}"
        )

    all_context_starts: dict[str, set[int]] = {name: set() for name in sequence_by_name}
    for name, frame_set in expanded_frame_sets.items():
        if frame_set:
            starts = context_start_indices(sequence_by_name[name])
            all_context_starts[name] = {int(starts[frame]) for frame in frame_set}
    context_rows: dict[str, dict[str, Any]] = {}
    for name in sequence_by_name:
        new_starts = sorted(all_context_starts[name] - base_context_starts[name])
        context_rows[name] = {
            "baseline_context_start_indices": sorted(base_context_starts[name]),
            "required_context_start_indices": sorted(all_context_starts[name]),
            "new_context_start_indices": new_starts,
            "new_context_count": len(new_starts),
        }

    all_events = [event for events in sequence_events.values() for event in events]
    event_ids = {event["event_id"] for event in all_events}
    full_event_keys = {
        (event["sequence"], int(event["start_frame"]), int(event["stop_frame"]))
        for event in all_events
    }
    if expanded_event_keys != full_event_keys:
        missing = sorted(full_event_keys - expanded_event_keys)[:8]
        extra = sorted(expanded_event_keys - full_event_keys)[:8]
        raise ValueError(f"Expanded windows changed full-video events: missing={missing}, extra={extra}")
    previous_event_keys = {
        (event["sequence"], int(event["start_frame"]), int(event["stop_frame"]))
        for event in prior_events
    }
    cross_segment_event_count = sum(len(event["segment_ids"]) > 1 for event in all_events)
    cross_segment_context_count = sum(len(row["segment_ids"]) > 1 for row in event_source_rows)
    cross_segment_merged_count = sum(len(row["segment_ids"]) > 1 for row in merged_windows)
    baseline_covered_event_ids = {
        event["event_id"] for event in all_events
        if event["start_frame"] in base_frame_sets[event["sequence"]]
        and all(frame in base_frame_sets[event["sequence"]]
                for frame in range(event["start_frame"], event["stop_frame"] + 1))
    }
    base_unique_frames = sum(len(values) for values in base_frame_sets.values())
    event_context_unique_frames = sum(len(values) for values in event_only_frame_sets.values())
    expanded_unique_frames = sum(len(values) for values in expanded_frame_sets.values())
    expanded_gt_valid_frames = sum(
        int(np.asarray(sequence.gt_valid, dtype=bool)[sorted(expanded_frame_sets[sequence.sequence])].sum())
        for sequence in sequences if expanded_frame_sets[sequence.sequence]
    )
    needed_context_count = sum(len(row["required_context_start_indices"]) for row in context_rows.values())
    additional_context_count = sum(row["new_context_count"] for row in context_rows.values())
    total_gt_valid_frames = sum(int(np.asarray(sequence.gt_valid, dtype=bool).sum())
                                for sequence in sequences)
    total_visual_valid_frames = sum(int(np.asarray(sequence.visual_valid, dtype=bool).sum())
                                    for sequence in sequences)
    total_evaluation_mask_frames = sum(
        int(gt_arrays_by_sequence[name]["evaluation_mask"].sum()) for name in sequence_by_name)

    manifest = {
        "schema": "calibration_event_windows_v3",
        "protocol_revision": 2,
        "source": "prepare_training.load_sequences(splits=('calibration',))",
        "split": "calibration",
        "sequence_count": len(sequences),
        "sequence_names": [sequence.sequence for sequence in sequences],
        "coordinate_frame": "per-interval local raw HaWoR SLAM world",
        "event_protocol": {
            "reference_scope": "full video per calibration sequence",
            "reference_threshold": REFERENCE_THRESHOLD,
            "prediction_threshold_fixed_for_later_scoring": PREDICTION_THRESHOLD,
            "min_frames": MIN_EVENT_FRAMES,
            "max_gap_sec": MAX_GAP_SEC,
            "extractor": "dual_loop.pinch_evaluation.extract_events",
            "event_rule": "complete below-threshold run with valid open brackets; split at GT-invalid samples and dt <= 0 or dt > 0.25s; never split at SLAM segment changes",
        },
        "event_context_rule": {
            "frames_each_side": CONTEXT_FRAMES_EACH_SIDE,
            "clipped_at_time_gap_only": True,
            "may_cross_segment_reset": True,
            "overlapping_or_adjoining_windows_merged": True,
        },
        "prediction_eligibility": {
            "source_mask": "prepared visual_valid AND gt_valid",
            "mask_field": "evaluation_mask",
            "reference_mask_field": "gt_valid only",
            "rule": "mirror thirdparty/try/scripts/eval_hot3d.py parse_chunks: split flat valid frame_ids only where diff != 1; retain complete chunks of at least 16 frames",
            "split_at_segment_reset": False,
            "split_at_timestamp_gap": False,
            "fixed_full_video_mask": True,
            "window_truncation_redefines_mask": False,
        },
        "context_start_rule": "prepare_training.context_start_indices remains aligned to SLAM segment resets; each full-video window may include multiple segments, with context indices grouped separately by the original alignment",
        "base_calibration_source": {
            "cache_manifest": str(CACHE_MANIFEST),
            "cache_manifest_sha256": sha256_file(CACHE_MANIFEST),
            "fixed_window_count": len(base_windows),
            "fixed_window_frames": BASELINE_WINDOW_FRAMES,
            "baseline_result": str(BASELINE_RESULT),
            "baseline_result_sha256": sha256_file(BASELINE_RESULT),
            "baseline_reference_events": baseline_reference_events,
            "fixed_windows": base_source_rows,
        },
        "counts": {
            "total_complete_gt_reference_events": len(all_events),
            "previous_segment_split_reference_events": len(prior_events),
            "fullvideo_events_not_exact_in_previous_segment_split": len(full_event_keys - previous_event_keys),
            "previous_segment_split_events_not_exact_in_fullvideo": len(previous_event_keys - full_event_keys),
            "fullvideo_events_spanning_multiple_segments": cross_segment_event_count,
            "event_context_windows_spanning_multiple_segments": cross_segment_context_count,
            "merged_windows_spanning_multiple_segments": cross_segment_merged_count,
            "baseline_reference_events_recomputed": baseline_reference_events,
            "baseline_full_gt_events_fully_covered": len(baseline_covered_event_ids),
            "event_context_windows_before_merge": len(event_source_rows),
            "event_context_unique_frames": event_context_unique_frames,
            "merged_window_count": len(merged_windows),
            "baseline_unique_frames": base_unique_frames,
            "expanded_unique_frames": expanded_unique_frames,
            "new_unique_frames_over_baseline": expanded_unique_frames - base_unique_frames,
            "expanded_gt_valid_frames": expanded_gt_valid_frames,
            "all_calibration_gt_valid_frames": total_gt_valid_frames,
            "all_calibration_visual_valid_frames": total_visual_valid_frames,
            "full_video_evaluation_mask_frames": total_evaluation_mask_frames,
            "expanded_reference_events_recomputed": expanded_reference_events,
            "expanded_complete_event_coverage": len(event_ids),
            "required_unique_16_frame_contexts": needed_context_count,
            "additional_16_frame_contexts_over_v2": additional_context_count,
        },
        "sequences": sequence_rows,
        "events": all_events,
        "event_context_windows": event_source_rows,
        "merged_windows": merged_windows,
        "context_start_indices": context_rows,
    }
    _atomic_json(manifest, args.output_dir / "calibration_manifest.json")
    _atomic_json({"schema": "calibration_event_gt_v3", "events": all_events},
                 args.output_dir / "events.json")
    _atomic_json({
        "schema": "calibration_event_gt_summary_v3",
        "protocol_revision": 2,
        "sequence_count": len(sequences),
        "events_by_sequence": {name: len(events) for name, events in sequence_events.items()},
        "counts": manifest["counts"],
        "baseline_events": {
            "v2_stored_reference_events": int(baseline["events"]["reference_events"]),
            "recomputed_from_fixed_40_windows": baseline_reference_events,
            "complete_full_gt_events_fully_inside_v2_frame_union": len(baseline_covered_event_ids),
            "previous_segment_split_reference_events": len(prior_events),
            "fullvideo_events_not_exact_in_previous_segment_split": len(full_event_keys - previous_event_keys),
            "previous_segment_split_events_not_exact_in_fullvideo": len(previous_event_keys - full_event_keys),
        },
        "fullvideo_cross_segment_events": cross_segment_event_count,
        "evaluation_mask_frames": total_evaluation_mask_frames,
        "fixed_prediction_threshold": PREDICTION_THRESHOLD,
    }, args.output_dir / "summary.json")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
