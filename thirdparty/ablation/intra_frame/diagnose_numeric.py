#!/usr/bin/env python3
"""Calibration-only numeric diagnostic for raw input and repeated TWM restore.

This deliberately reads only the prepared calibration split. It compares the
archived raw HaWoR state, the real one-pass TWM-only restoration path, and a
second TWM call whose current candidate is the first restoration. No image
features, visual network, or test labels are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT / "thirdparty" / "try"
TWM_ONLY = HERE.parent / "twm_only"
sys.path[:0] = [str(ROOT), str(PROJECT), str(PROJECT / "scripts"), str(TWM_ONLY), str(HERE)]

from dual_loop.geometry import ManoGeometry  # noqa: E402
from dual_loop.handloop_state import (  # noqa: E402
    make_anchor,
    state_to_legacy,
    to_local,
    to_world,
)
from dual_loop.state import BETAS  # noqa: E402
from lib.eval_utils.eval_utils import (  # noqa: E402
    batch_compute_similarity_transform_torch,
    compute_jpe,
    first_align_joints,
    global_align_joints,
)
from prepare_training import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPORT_SPLIT,
    DEFAULT_PROTOCOL,
    DEFAULT_SPLIT,
    DEFAULT_VIDEO_ROOT,
    PreparedSequence,
    convert_raw_history_states,
    load_sequences,
)
from runtime import raw_history_batch, sha256  # noqa: E402
import run_twm_only as twm_only  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "thirdparty/ablation/training_v2/checkpoints/latest.pt"
DEFAULT_RUN_DIR = HERE / "runs/diagnose_numeric_v2"
MIN_METRIC_CHUNK = 16
METRIC_BLOCK = 100
EXPECTED_COORDINATE_FRAME = "per-interval local raw HaWoR SLAM world"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--export-split", type=Path, default=DEFAULT_EXPORT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="TWM batch size; 64 is the conservative A10 setting")
    parser.add_argument("--conversion-batch-size", type=int, default=512)
    parser.add_argument("--resume", action="store_true",
                        help="Resume only from a compatible, atomically saved diagnostic.json")
    parser.add_argument("--skip-image-check", action="store_true",
                        help="Skip archive image-path existence checks")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return sha256(Path(path))


def _mapping_sha256(value: dict[str, str]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_archive_world(sequence: PreparedSequence) -> dict[str, Any]:
    """Bind raw/GT state coordinates to their per-segment raw SLAM source."""
    with np.load(sequence.archive_path, allow_pickle=False) as archive:
        frame_ids = archive["frame_id"].astype(np.int64, copy=True)
        segment_ids = archive["segment_id"].astype(np.int64, copy=True)
        interval = int(archive["segment_interval_frames"].item())
        coordinate_frame = str(archive["coordinate_frame"].item())
        position_unit = str(archive["position_unit"].item())
        provenance = json.loads(str(archive["input_provenance_json"].item()))
        visual_shape = tuple(archive["visual_state"].shape)
        target_shape = tuple(archive["gt_state"].shape)
    if coordinate_frame != EXPECTED_COORDINATE_FRAME:
        raise ValueError(f"{sequence.sequence}: unexpected coordinate frame {coordinate_frame!r}")
    if provenance.get("coordinate_frame") != EXPECTED_COORDINATE_FRAME:
        raise ValueError(f"{sequence.sequence}: provenance does not bind raw and GT to raw SLAM")
    if provenance.get("runtime_uses_gt_labels") is not False:
        raise ValueError(f"{sequence.sequence}: archive provenance does not exclude GT at runtime")
    if interval < 1 or not np.array_equal(frame_ids, np.arange(len(sequence))):
        raise ValueError(f"{sequence.sequence}: malformed frame ids or SLAM interval")
    if not np.array_equal(segment_ids, frame_ids // interval):
        raise ValueError(f"{sequence.sequence}: segment ids do not match SLAM reset intervals")
    if visual_shape != (len(sequence), 109) or target_shape != visual_shape:
        raise ValueError(f"{sequence.sequence}: malformed aligned raw/GT state arrays")
    if position_unit != "meter":
        raise ValueError(f"{sequence.sequence}: expected state positions in meters")

    bounds = []
    starts = np.r_[0, np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(sequence)]
    for start, stop in zip(starts, stops):
        bounds.append({"segment_id": int(segment_ids[start]),
                       "frame_start_inclusive": int(start),
                       "frame_stop_exclusive": int(stop)})
    return {
        "archive": str(sequence.archive_path.resolve()),
        "archive_sha256": _sha256(sequence.archive_path),
        "coordinate_frame": coordinate_frame,
        "position_unit": position_unit,
        "segment_interval_frames": interval,
        "raw_and_gt_share_per_segment_slam_world": True,
        "gt_transform_source": "prepare_training_data.transform_gt_state_to_slam",
        "raw_source": "prepared visual_state from HaWoR frontend output",
        "runtime_uses_gt_labels": False,
        "frontend_provenance": {
            key: provenance.get(key) for key in (
                "visual_frontend", "frontend_result_dir", "slam_camera_source",
                "segment_id_semantics", "bbox_source", "oracle_assisted",
                "manifest_status") if key in provenance
            },
        "segment_bounds": bounds,
        "frame_count": int(len(sequence)),
        "gt_valid_count": int(sequence.gt_valid.sum()),
        "visual_valid_count": int(sequence.visual_valid.sum()),
        "gt_and_visual_valid_count": int((sequence.gt_valid & sequence.visual_valid).sum()),
    }


def _raw_availability(sequence: PreparedSequence, device: torch.device) -> torch.Tensor:
    values = np.repeat(sequence.visual_valid[:, None], 17, axis=1)
    return torch.as_tensor(values, dtype=torch.bool, device=device)


def _restore_with_candidate_batched(
    twm: torch.nn.Module,
    raw_world: torch.Tensor,
    candidate_world: torch.Tensor,
    raw_available: torch.Tensor,
    timestamp_sec: torch.Tensor,
    segment_id: torch.Tensor,
    handedness: torch.Tensor,
    *,
    batch_size: int,
    candidate_available: torch.Tensor | None = None,
) -> torch.Tensor:
    """One TWM call per frame, with history built only from original raw states."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if raw_world.shape != candidate_world.shape or raw_world.ndim != 2 or raw_world.shape[-1] != 99:
        raise ValueError("raw_world and candidate_world must have matching [N,99] shapes")
    count = len(raw_world)
    if raw_available.shape != (count, 17):
        raise ValueError("raw_available must have shape [N,17]")
    if candidate_available is not None and candidate_available.shape != (count, 17):
        raise ValueError("candidate_available must have shape [N,17]")
    if any(value.shape != (count,) for value in (timestamp_sec, segment_id, handedness)):
        raise ValueError("timestamps, segment ids and handedness must have shape [N]")

    outputs: list[torch.Tensor] = []
    history_length = int(twm.config.history_length)
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            ids = torch.arange(start, stop, device=raw_world.device)
            raw = raw_world[ids]
            candidate = candidate_world[ids]
            available = raw_available[ids].bool()
            candidate_flags = (available if candidate_available is None
                               else candidate_available[ids].bool())
            past = raw_history_batch(
                raw_world,
                raw_available,
                timestamp_sec,
                segment_id,
                handedness,
                ids,
                history_length,
            )
            anchor_rotation, anchor_position = make_anchor(
                raw, past["history"], available, past["history_available"])
            history_anchor = to_local(past["history"], anchor_rotation, anchor_position)
            raw_anchor = to_local(raw, anchor_rotation, anchor_position)
            candidate_anchor = to_local(candidate, anchor_rotation, anchor_position)
            prediction = twm(
                history=history_anchor,
                raw=raw_anchor,
                candidate=candidate_anchor,
                history_available=past["history_available"],
                raw_available=available,
                candidate_available=candidate_flags,
                history_times=past["history_times"],
                dt=past["dt"],
                handedness=handedness[ids].reshape(-1).long().clamp(0, 1),
                history_mask=past["history_mask"],
                future_times=None,
            )
            outputs.append(to_world(
                prediction["current"]["state"], anchor_rotation, anchor_position))
    if not outputs:
        return raw_world.new_empty(0, 99)
    return torch.cat(outputs, dim=0)


def _verify_single_pass_equivalence(
    twm: torch.nn.Module,
    sequence: PreparedSequence,
    *,
    device: torch.device,
    batch_size: int,
    sample_frames: int,
) -> dict[str, Any]:
    count = min(len(sequence), sample_frames)
    raw_world = torch.as_tensor(sequence.raw_world[:count], dtype=torch.float32, device=device)
    available = _raw_availability(sequence, device)[:count]
    timestamps = torch.as_tensor(sequence.timestamp_sec[:count], dtype=torch.float64, device=device)
    segments = torch.as_tensor(sequence.segment_id[:count], dtype=torch.long, device=device)
    hands = torch.ones(count, dtype=torch.long, device=device)
    with torch.inference_mode():
        actual = twm_only.restore_raw_sequence_batched(
            twm, raw_world, available, timestamps, segments, hands, batch_size=batch_size)
        candidate_raw = _restore_with_candidate_batched(
            twm, raw_world, raw_world, available, timestamps, segments, hands,
            batch_size=batch_size)
    difference = (actual - candidate_raw).abs()
    max_error = float(difference.max().item()) if difference.numel() else 0.0
    mean_error = float(difference.mean().item()) if difference.numel() else 0.0
    if not torch.allclose(actual, candidate_raw, rtol=0.0, atol=1e-6):
        raise AssertionError(
            "candidate=raw path does not reproduce twm_only.restore_raw_sequence_batched: "
            f"max_abs_error={max_error}")
    return {
        "reference": "twm_only.restore_raw_sequence_batched",
        "candidate_path": "_restore_with_candidate_batched(candidate=raw)",
        "sample_sequence": sequence.sequence,
        "sample_frame_ids": list(range(count)),
        "includes_segment_reset": bool(
            np.any(sequence.segment_id[1:count] != sequence.segment_id[:count - 1])),
        "max_abs_state_error": max_error,
        "mean_abs_state_error": mean_error,
        "absolute_tolerance": 1e-6,
        "passed": True,
    }


def _state_joints(
    geometry: ManoGeometry,
    states: torch.Tensor,
    legacy_betas: np.ndarray,
    frame_ids: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(frame_ids), 21, 3), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(frame_ids), batch_size):
            ids = frame_ids[start:start + batch_size]
            state_batch = states[torch.as_tensor(ids, dtype=torch.long, device=device)]
            beta_batch = torch.as_tensor(legacy_betas[ids], dtype=torch.float32, device=device)
            legacy = state_to_legacy(state_batch, beta_batch, geometry)
            result[start:start + len(ids)] = geometry.joints(legacy).cpu().numpy().astype(np.float32)
    return result


def _gt_joints(
    geometry: ManoGeometry,
    sequence: PreparedSequence,
    frame_ids: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(frame_ids), 21, 3), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(frame_ids), batch_size):
            ids = frame_ids[start:start + batch_size]
            state = torch.as_tensor(sequence.target_legacy[ids], dtype=torch.float32, device=device)
            result[start:start + len(ids)] = geometry.joints(state).cpu().numpy().astype(np.float32)
    return result


def _eligible_chunks(sequence: PreparedSequence) -> list[np.ndarray]:
    valid_ids = np.flatnonzero(sequence.gt_valid & sequence.visual_valid)
    if not len(valid_ids):
        return []
    split_points = np.flatnonzero(
        (np.diff(valid_ids) != 1)
        | (sequence.segment_id[valid_ids[1:]] != sequence.segment_id[valid_ids[:-1]])
    ) + 1
    return [chunk for chunk in np.split(valid_ids, split_points) if len(chunk) >= MIN_METRIC_CHUNK]


def _score_joints(
    sequence: PreparedSequence,
    frame_ids: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
    raw_joints_by_id: dict[int, np.ndarray],
    *,
    need_offsets: bool,
) -> dict[str, Any]:
    """Use eval_hot3d's 16-frame chunks, first-two similarity and 100-frame W blocks."""
    id_to_position = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    eligible_chunks = _eligible_chunks(sequence)
    if not eligible_chunks:
        raise ValueError(f"{sequence.sequence}: no contiguous gt/raw-valid chunk has {MIN_METRIC_CHUNK} frames")
    metric_ids = np.concatenate(eligible_chunks)
    raw_positions = np.asarray([id_to_position[int(frame_id)] for frame_id in metric_ids], dtype=np.int64)
    pred_metric = predicted[raw_positions]
    gt_metric = target[raw_positions]
    per_joint_error = np.linalg.norm(pred_metric - gt_metric, axis=-1)
    root_error = per_joint_error[:, 0]
    metrics: dict[str, Any] = {
        "scored_frame_count": int(len(metric_ids)),
        "joint_mpjpe_mm": float(per_joint_error.mean() * 1000.0),
        "root_position_error_mm": float(root_error.mean() * 1000.0),
    }

    w_values: list[np.ndarray] = []
    wa_values: list[np.ndarray] = []
    pa_values: list[np.ndarray] = []
    for chunk_ids in eligible_chunks:
        positions = np.asarray([id_to_position[int(frame_id)] for frame_id in chunk_ids], dtype=np.int64)
        for start in range(0, len(positions), METRIC_BLOCK):
            block = positions[start:start + METRIC_BLOCK]
            gt_part = torch.as_tensor(target[block], dtype=torch.float32)
            pred_part = torch.as_tensor(predicted[block], dtype=torch.float32)
            w_values.append(compute_jpe(gt_part, first_align_joints(gt_part, pred_part)) * 1000.0)
            wa_values.append(compute_jpe(gt_part, global_align_joints(gt_part, pred_part)) * 1000.0)
        gt_chunk = torch.as_tensor(target[positions], dtype=torch.float32)
        pred_chunk = torch.as_tensor(predicted[positions], dtype=torch.float32)
        pa = batch_compute_similarity_transform_torch(pred_chunk, gt_chunk)
        pa_values.append(compute_jpe(gt_chunk, pa) * 1000.0)
    metrics["w_mpjpe_mm"] = float(np.concatenate(w_values).mean()) if w_values else None
    metrics["wa_mpjpe_mm"] = float(np.concatenate(wa_values).mean()) if wa_values else None
    metrics["pa_mpjpe_mm"] = float(np.concatenate(pa_values).mean()) if pa_values else None
    metrics["w_wa_pa_frame_count"] = int(sum(len(values) for values in w_values))

    if need_offsets:
        raw_metric_joints = np.stack([raw_joints_by_id[int(frame_id)] for frame_id in metric_ids])
        offset = np.linalg.norm(pred_metric - raw_metric_joints, axis=-1)
        metrics["restore_vs_raw_joint_offset_mm"] = float(offset.mean() * 1000.0)
        metrics["restore_vs_raw_root_offset_mm"] = float(offset[:, 0].mean() * 1000.0)
    else:
        metrics["restore_vs_raw_joint_offset_mm"] = 0.0
        metrics["restore_vs_raw_root_offset_mm"] = 0.0

    first_chunk = eligible_chunks[0]
    sample_ids = sorted(set([
        int(first_chunk[0]), int(first_chunk[min(1, len(first_chunk) - 1)]),
        int(metric_ids[len(metric_ids) // 2]), int(metric_ids[-1]),
        *(int(value) for value in metric_ids
          if sequence.segment_id[value] != sequence.segment_id[max(0, value - 1)]),
    ]))
    metrics["sample_frame_ids"] = sample_ids
    metrics["segment_ids_scored"] = sorted({int(sequence.segment_id[index]) for index in metric_ids})
    return metrics


def _aggregate(per_video: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "joint_mpjpe_mm", "root_position_error_mm", "w_mpjpe_mm", "wa_mpjpe_mm",
        "pa_mpjpe_mm", "restore_vs_raw_joint_offset_mm", "restore_vs_raw_root_offset_mm",
    )
    weighted: dict[str, Any] = {}
    for key in keys:
        rows = [row for row in per_video if row["metrics"].get(key) is not None]
        count_key = "w_wa_pa_frame_count" if key in {"w_mpjpe_mm", "wa_mpjpe_mm", "pa_mpjpe_mm"} else "scored_frame_count"
        weights = np.asarray([row["metrics"][count_key] for row in rows], dtype=np.float64)
        values = np.asarray([row["metrics"][key] for row in rows], dtype=np.float64)
        weighted[key] = float(np.average(values, weights=weights)) if weights.sum() else None
    weighted["scored_frame_count"] = int(sum(row["metrics"]["scored_frame_count"] for row in per_video))
    weighted["num_videos"] = len(per_video)
    macro = {
        key: (float(np.mean([row["metrics"][key] for row in per_video
                             if row["metrics"].get(key) is not None]))
              if any(row["metrics"].get(key) is not None for row in per_video) else None)
        for key in keys
    }
    macro["num_videos"] = len(per_video)
    return {"frame_weighted": weighted, "video_macro": macro}


def _metric_mask(sequence: PreparedSequence) -> np.ndarray:
    chunks = _eligible_chunks(sequence)
    if not chunks:
        return np.zeros(len(sequence), dtype=bool)
    mask = np.zeros(len(sequence), dtype=bool)
    mask[np.concatenate(chunks)] = True
    return mask


def _evaluate_sequence(
    sequence: PreparedSequence,
    geometry: ManoGeometry,
    twm: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    joint_batch_size: int,
    equivalence_sample: bool,
    archive_world: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if sequence.raw_world is None:
        raise ValueError(f"{sequence.sequence}: raw history conversion was not run")
    archive_world = archive_world or _validate_archive_world(sequence)
    frame_ids = np.flatnonzero(sequence.gt_valid & sequence.visual_valid)
    if not len(frame_ids):
        raise ValueError(f"{sequence.sequence}: no gt_valid & visual_valid frames")
    raw_world = torch.as_tensor(sequence.raw_world, dtype=torch.float32, device=device)
    available = _raw_availability(sequence, device)
    timestamps = torch.as_tensor(sequence.timestamp_sec, dtype=torch.float64, device=device)
    segments = torch.as_tensor(sequence.segment_id, dtype=torch.long, device=device)
    hands = torch.ones(len(sequence), dtype=torch.long, device=device)
    with torch.inference_mode():
        # This is the actual one-round TWM-only implementation used by row one.
        restored_once = twm_only.restore_raw_sequence_batched(
            twm, raw_world, available, timestamps, segments, hands, batch_size=batch_size)
        # The second call consumes only the previous restored candidate. Its
        # history remains the original raw sequence through raw_history_batch.
        restored_available = torch.isfinite(restored_once).all(dim=-1, keepdim=True).expand(-1, 17)
        restored_twice = _restore_with_candidate_batched(
            twm, raw_world, restored_once, available, timestamps, segments, hands,
            batch_size=batch_size, candidate_available=restored_available)

    onepass_check = None
    if equivalence_sample:
        interval = int(archive_world["segment_interval_frames"])
        onepass_check = _verify_single_pass_equivalence(
            twm, sequence, device=device, batch_size=batch_size,
            sample_frames=interval + min(8, batch_size))

    gt = _gt_joints(geometry, sequence, frame_ids, device=device, batch_size=joint_batch_size)
    raw = _state_joints(geometry, raw_world, sequence.visual_legacy[:, BETAS], frame_ids,
                        device=device, batch_size=joint_batch_size)
    once = _state_joints(geometry, restored_once, sequence.visual_legacy[:, BETAS], frame_ids,
                         device=device, batch_size=joint_batch_size)
    twice = _state_joints(geometry, restored_twice, sequence.visual_legacy[:, BETAS], frame_ids,
                          device=device, batch_size=joint_batch_size)

    # Check that the 99D actual-wrist conversion round-trips the archived raw
    # MANO joints before interpreting any recovery offset.
    # Compute archived raw frontend geometry directly for the round-trip.
    source_raw = np.empty_like(raw)
    with torch.inference_mode():
        for start in range(0, len(frame_ids), joint_batch_size):
            ids = frame_ids[start:start + joint_batch_size]
            legacy = torch.as_tensor(sequence.visual_legacy[ids], dtype=torch.float32, device=device)
            source_raw[start:start + len(ids)] = geometry.joints(legacy).cpu().numpy().astype(np.float32)
    roundtrip_error_mm = float(np.linalg.norm(raw - source_raw, axis=-1).max() * 1000.0)

    raw_by_id = {int(frame_id): raw[index] for index, frame_id in enumerate(frame_ids)}
    prediction_by_method = {"raw": raw, "twm_once_candidate_raw": once,
                            "twm_twice_candidate_previous_restore": twice}
    per_method = {}
    for method, prediction in prediction_by_method.items():
        per_method[method] = _score_joints(
            sequence, frame_ids, prediction, gt, raw_by_id,
            need_offsets=method != "raw")

    eligible_count = int(_metric_mask(sequence).sum())
    result = {
        "sequence": sequence.sequence,
        "split": "calibration",
        "frame_count": int(len(sequence)),
        "common_valid_frame_count": int(len(frame_ids)),
        "eval_eligible_frame_count": eligible_count,
        "excluded_short_fragment_frame_count": int(len(frame_ids) - eligible_count),
        "raw_roundtrip_max_joint_error_mm": roundtrip_error_mm,
        "metrics": per_method,
        "coordinate_provenance": archive_world,
    }
    if onepass_check is not None:
        result["single_pass_equivalence_check"] = onepass_check
    return result


def _aggregate_from_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    variants = ("raw", "twm_once_candidate_raw", "twm_twice_candidate_previous_restore")
    return {variant: _aggregate([
        {"sequence": row["sequence"], "metrics": row["metrics"][variant]}
        for row in results
    ]) for variant in variants}


METHODS = {
    "raw": "prepared raw HaWoR state, no TWM call",
    "twm_once_candidate_raw": "twm_only.restore_raw_sequence_batched; candidate=raw; history=raw",
    "twm_twice_candidate_previous_restore": (
        "one additional TWM call; candidate=previous restoration; history=original raw only"),
}
METRIC_PROTOCOL = {
    "scored_mask": "gt_valid & visual_valid, then contiguous per-segment chunks of at least 16 frames",
    "segment_resets": "segment_id changes split chunks and reset TWM history",
    "joint_mpjpe": "unaligned mean Euclidean error over 21 MANO joints on eligible frames",
    "root_position_error": "unaligned Euclidean error at MANO wrist joint 0",
    "w_mpjpe": "first-two-frame similarity alignment at each 100-frame block, matching eval_hot3d",
    "wa_mpjpe": "global similarity alignment at each 100-frame block, matching eval_hot3d",
    "pa_mpjpe": "per-frame similarity alignment over each contiguous segment chunk, matching eval_hot3d",
    "restoration_offset": "unaligned restored-vs-raw joint and wrist displacement",
    "units": "millimeters except frame counts",
}


def _resume_partial(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    twm: torch.nn.Module,
    checkpoint: dict[str, Any],
    checkpoint_step: int,
    sequences: list[PreparedSequence],
    protocol: dict[str, Any],
    archive_world_by_sequence: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    result_path = output_dir / "diagnostic.json"
    if not output_dir.is_dir() or not result_path.is_file():
        raise FileNotFoundError(f"--resume requires the existing diagnostic.json: {result_path}")
    base = json.loads(result_path.read_text(encoding="utf-8"))
    if base.get("schema") != "twm_numeric_calibration_diagnostic_v2":
        raise ValueError("Cannot resume an incompatible diagnostic schema")
    if base.get("status") != "running":
        raise ValueError(f"Cannot resume diagnostic with status={base.get('status')!r}")
    if base.get("methods") != METHODS or base.get("metric_protocol") != METRIC_PROTOCOL:
        raise ValueError("Resume method or metric protocol differs from the initial run")
    if MIN_METRIC_CHUNK != 16 or METRIC_BLOCK != 100:
        raise ValueError("Resume scoring chunk/block sizes differ from the initial metric protocol")
    if base.get("num_videos_expected") != len(sequences) or len(sequences) != 20:
        raise ValueError("Resume calibration split no longer contains the original 20 videos")
    if base.get("uses_test_labels") is not False or base.get("split") != "calibration":
        raise ValueError("Resume source is not a calibration-only diagnostic")

    current_checkpoint_hash = _sha256(args.checkpoint)
    if (base.get("checkpoint", {}).get("sha256") != current_checkpoint_hash
            or base.get("checkpoint", {}).get("path") != str(args.checkpoint.resolve())
            or base.get("checkpoint", {}).get("stage") != checkpoint.get("stage")
            or base.get("checkpoint", {}).get("global_step") != int(checkpoint_step)):
        raise ValueError("Resume TWM checkpoint path/hash/stage/step differs from the initial run")
    current_inputs = {
        "prepared_data_root": str(args.data_root.resolve()),
        "video_root": str(args.video_root.resolve()),
        "split_manifest": {"path": str(args.split_manifest.resolve()),
                            "sha256": _sha256(args.split_manifest)},
        "export_split": {"path": str(args.export_split.resolve()),
                         "sha256": _sha256(args.export_split)},
        "calibration_protocol": {"path": str(args.protocol.resolve()),
                                 "sha256": _sha256(args.protocol)},
    }
    old_inputs = base.get("inputs", {})
    for key, value in current_inputs.items():
        if old_inputs.get(key) != value:
            raise ValueError(f"Resume input path/hash differs from the initial run: {key}")
    old_parameters = base.get("run_parameters", {})
    expected_parameters = {
        "twm_batch_size": int(args.batch_size),
        "raw_conversion_batch_size": int(args.conversion_batch_size),
        "geometry_batch_size": int(args.conversion_batch_size),
        "history_length": int(twm.config.history_length),
        "visual_network_loaded": False,
        "ha_wor_frontend_rerun": False,
    }
    for key, value in expected_parameters.items():
        if old_parameters.get(key) != value:
            raise ValueError(f"Resume parameter differs from the initial run: {key}")

    source_paths = {
        "prepare_training.py": HERE / "prepare_training.py",
        "runtime.py": HERE / "runtime.py",
        "run_twm_only.py": TWM_ONLY / "run_twm_only.py",
        "prepare_training_data.py": PROJECT / "scripts/prepare_training_data.py",
    }
    old_source_hashes = base.get("source_hashes", {})
    for key, path in source_paths.items():
        if old_source_hashes.get(key) != _sha256(path):
            raise ValueError(f"Resume source module hash differs from the initial run: {key}")

    expected_order = [sequence.sequence for sequence in sequences]
    results = base.get("videos")
    if not isinstance(results, list) or len(results) >= len(expected_order):
        raise ValueError("Resume requires a non-empty, incomplete per-video result prefix")
    completed_names = [row.get("sequence") for row in results]
    if completed_names != expected_order[:len(completed_names)]:
        raise ValueError("Persisted videos are not an ordered prefix of the calibration split")
    if base.get("num_videos_completed") != len(results):
        raise ValueError("Persisted completion count disagrees with the per-video result list")

    current_archive_hashes = {
        sequence.sequence: archive_world_by_sequence[sequence.sequence]["archive_sha256"]
        for sequence in sequences
    }
    recorded_archive_hashes = old_inputs.get("calibration_archive_sha256")
    original_started = datetime.fromisoformat(base["created_utc"].replace("Z", "+00:00"))
    for row in results:
        name = row["sequence"]
        old_hash = row.get("coordinate_provenance", {}).get("archive_sha256")
        if old_hash != current_archive_hashes[name]:
            raise ValueError(f"Resume prepared-data SHA256 differs for completed video {name}")
    if recorded_archive_hashes is not None:
        if recorded_archive_hashes != current_archive_hashes:
            raise ValueError("Resume calibration archive hashes differ from the initial data manifest")
        old_fingerprint = old_inputs.get("calibration_archives_fingerprint_sha256")
        if old_fingerprint != _mapping_sha256(current_archive_hashes):
            raise ValueError("Resume prepared-data aggregate hash differs from the initial run")
        data_check = "all 20 prepared archive SHA256 values match the initial run"
    else:
        completed_set = set(completed_names)
        for sequence in sequences:
            if sequence.sequence in completed_set:
                continue
            if sequence.archive_path.stat().st_mtime > original_started.timestamp():
                raise ValueError(
                    f"Unscored calibration archive changed after initial run start: {sequence.sequence}")
        data_check = (
            "completed archive SHA256 values exactly match persisted rows; unscored archives were "
            "not hashed in the old partial JSON, so each was rehashed now and checked to have an "
            "mtime no later than the initial run start")
        old_inputs["resume_calibration_archive_sha256"] = current_archive_hashes
    base["resume_data_integrity"] = {
        "prepared_archive_check": data_check,
        "initial_run_started_utc": base["created_utc"],
        "resume_archive_sha256": current_archive_hashes,
        "unscored_archive_mtime_check": "passed" if recorded_archive_hashes is None else "not needed",
    }

    if base.get("num_videos_completed") != len(results):
        raise ValueError("Resume completion metadata is inconsistent")
    initial_source_hash = old_source_hashes.get("diagnose_numeric.py")
    phases = base.setdefault("execution_history", [])
    if not phases:
        phases.append({
            "phase": "initial_gpu1",
            "started_utc": base["created_utc"],
            "device": old_parameters.get("device"),
            "cuda_visible_devices": old_parameters.get("cuda_visible_devices"),
            "batch_size": old_parameters.get("twm_batch_size"),
            "completed_video_count_before_resume": len(results),
            "source_sha256": initial_source_hash,
            "status": "stopped_after_last_atomically_persisted_video",
        })
    current_script_hash = _sha256(Path(__file__).resolve())
    phases.append({
        "phase": "resume_gpu3",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": int(args.batch_size),
        "conversion_batch_size": int(args.conversion_batch_size),
        "source_sha256": current_script_hash,
        "skip_completed_videos": completed_names,
    })
    base.setdefault("resume_source_hashes", []).append(current_script_hash)
    base["resumed_utc"] = datetime.now(timezone.utc).isoformat()
    base["resumed_from_video_count"] = len(results)
    base["resume_execution"] = phases[-1]
    base["status"] = "running"
    base["num_videos_completed"] = len(results)
    base["aggregate"] = _aggregate_from_results(results)
    _write_json(result_path, base)
    return base, results, set(completed_names)


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.conversion_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    output_dir = args.run_dir.expanduser().resolve()
    if not output_dir.is_relative_to((HERE / "runs").resolve()):
        raise ValueError("run directory must remain under thirdparty/ablation/intra_frame/runs")
    if args.resume and not output_dir.is_dir():
        raise FileNotFoundError(f"--resume requires an existing run directory: {output_dir}")
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite numeric diagnostic: {output_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    torch.set_num_threads(2)
    loaded, protocol = load_sequences(
        data_root=args.data_root,
        video_root=args.video_root,
        split_manifest=args.split_manifest,
        export_split=args.export_split,
        protocol_path=args.protocol,
        splits=("calibration",),
        check_images=not args.skip_image_check,
    )
    sequences = loaded["calibration"]
    if len(sequences) != 20:
        raise ValueError(f"Expected the prepared 20-video calibration split, got {len(sequences)}")
    archive_world_by_sequence = {
        sequence.sequence: _validate_archive_world(sequence) for sequence in sequences
    }
    geometry = ManoGeometry(ROOT).to(device).eval()
    # load_twm validates the frozen trained checkpoint and extracts only TWM.
    twm, checkpoint = twm_only.load_twm(args.checkpoint, device)
    if checkpoint.get("stage") != "motion_pretrain":
        raise ValueError("Numeric diagnostic requires the frozen motion_pretrain TWM checkpoint")
    checkpoint_step = twm_only.validate_trained_checkpoint(checkpoint)

    current_archive_hashes = {
        name: archive["archive_sha256"]
        for name, archive in archive_world_by_sequence.items()
    }
    fresh_base = {
        "schema": "twm_numeric_calibration_diagnostic_v2",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_label": "calibration-local diagnostic; not an official held-out test result",
        "split": "calibration",
        "uses_test_labels": False,
        "num_videos_expected": 20,
        "num_videos_completed": 0,
        "methods": METHODS,
        "metric_protocol": METRIC_PROTOCOL,
        "run_parameters": {
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "twm_batch_size": int(args.batch_size),
            "raw_conversion_batch_size": int(args.conversion_batch_size),
            "geometry_batch_size": int(args.conversion_batch_size),
            "history_length": int(twm.config.history_length),
            "visual_network_loaded": False,
            "ha_wor_frontend_rerun": False,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint),
            "stage": checkpoint.get("stage"),
            "architecture": checkpoint.get("architecture"),
            "global_step": int(checkpoint_step),
        },
        "inputs": {
            "prepared_data_root": str(args.data_root.resolve()),
            "video_root": str(args.video_root.resolve()),
            "split_manifest": {"path": str(args.split_manifest.resolve()),
                                "sha256": _sha256(args.split_manifest)},
            "export_split": {"path": str(args.export_split.resolve()),
                             "sha256": _sha256(args.export_split)},
            "calibration_protocol": {"path": str(args.protocol.resolve()),
                                     "sha256": _sha256(args.protocol)},
            "calibration_archive_sha256": current_archive_hashes,
            "calibration_archives_fingerprint_sha256": _mapping_sha256(current_archive_hashes),
            "calibration_sequence_count": len(sequences),
            "heldout_test_sequence_count_excluded": len(protocol.get("test_sequences", [])),
            "raw_source_limitation": (
                "raw input is the archived visual_state from the prepared_train100 HaWoR frontend export; "
                "the live HaWoR frontend was not rerun"),
            "source_coordinate_evidence": (
                "prepare_training_data.py stores visual_state and transforms HOT3D gt_state through "
                "the same raw SLAM camera pose within each interval"),
        },
        "source_hashes": {
            "prepare_training.py": _sha256(HERE / "prepare_training.py"),
            "runtime.py": _sha256(HERE / "runtime.py"),
            "run_twm_only.py": _sha256(TWM_ONLY / "run_twm_only.py"),
            "prepare_training_data.py": _sha256(PROJECT / "scripts/prepare_training_data.py"),
            "diagnose_numeric.py": _sha256(Path(__file__).resolve()),
        },
        "one_pass_equivalence_check": None,
        "aggregate": {},
        "videos": [],
    }

    if args.resume:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "3" or device.type != "cuda":
            raise ValueError("--resume is configured for CUDA_VISIBLE_DEVICES=3 and --device cuda:0")
        base, results, completed = _resume_partial(
            output_dir=output_dir,
            args=args,
            device=device,
            twm=twm,
            checkpoint=checkpoint,
            checkpoint_step=checkpoint_step,
            sequences=sequences,
            protocol=protocol,
            archive_world_by_sequence=archive_world_by_sequence,
        )
        pending_sequences = [sequence for sequence in sequences
                             if sequence.sequence not in completed]
        equivalence_done = bool(
            base.get("one_pass_equivalence_check", {}).get("passed", False))
    else:
        pending_sequences = sequences
        results = []
        base = fresh_base
        output_dir.mkdir(parents=True)
        base["execution_history"] = [{
            "phase": "initial",
            "started_utc": base["created_utc"],
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "batch_size": int(args.batch_size),
            "conversion_batch_size": int(args.conversion_batch_size),
            "source_sha256": base["source_hashes"]["diagnose_numeric.py"],
        }]
        _write_json(output_dir / "diagnostic.json", base)
        equivalence_done = False

    convert_raw_history_states(
        pending_sequences, geometry, device, batch_size=args.conversion_batch_size)
    for sequence in pending_sequences:
        print(f"[numeric] starting {sequence.sequence}", flush=True)
        result = _evaluate_sequence(
            sequence, geometry, twm, device=device, batch_size=args.batch_size,
            joint_batch_size=args.conversion_batch_size,
            equivalence_sample=not equivalence_done,
            archive_world=archive_world_by_sequence[sequence.sequence],
        )
        if "single_pass_equivalence_check" in result:
            equivalence_done = True
            base["one_pass_equivalence_check"] = result["single_pass_equivalence_check"]
        results.append(result)
        base["videos"] = results
        base["num_videos_completed"] = len(results)
        base["aggregate"] = _aggregate_from_results(results)
        _write_json(output_dir / "diagnostic.json", base)
        once = result["metrics"]["twm_once_candidate_raw"]
        twice = result["metrics"]["twm_twice_candidate_previous_restore"]
        print(
            f"[numeric] {sequence.sequence}: frames={once['scored_frame_count']} "
            f"raw/W={result['metrics']['raw']['w_mpjpe_mm']:.2f}/"
            f"{once['w_mpjpe_mm']:.2f} -> {twice['w_mpjpe_mm']:.2f} mm",
            flush=True,
        )

    base["status"] = "complete"
    base["completed_utc"] = datetime.now(timezone.utc).isoformat()
    if len(results) != len(sequences):
        raise ValueError(f"Diagnostic ended with {len(results)}/{len(sequences)} videos")
    base["aggregate"] = _aggregate_from_results(results)
    _write_json(output_dir / "diagnostic.json", base)
    print(json.dumps({
        "status": base["status"],
        "output": str((output_dir / "diagnostic.json").resolve()),
        "num_videos": len(results),
        "aggregate": base["aggregate"],
        "one_pass_equivalence_check": base["one_pass_equivalence_check"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
