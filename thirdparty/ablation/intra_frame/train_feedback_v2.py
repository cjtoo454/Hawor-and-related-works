#!/usr/bin/env python3
"""Train bounded two-round intra-frame feedback from frozen real HaWoR caches."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRY = ROOT / "thirdparty" / "try"
TWM_ONLY = ROOT / "thirdparty" / "ablation" / "twm_only"
sys.path[:0] = [str(ROOT), str(TRY), str(TRY / "scripts"), str(TWM_ONLY), str(HERE)]

from dual_loop.handloop_probability import log_prob  # noqa: E402
from dual_loop.handloop_runtime import build_system  # noqa: E402
from dual_loop.handloop_state import (  # noqa: E402
    legacy_to_state,
    make_anchor,
    state_to_legacy,
    to_local,
    to_world,
)
from dual_loop.oracle import normalized_pinch_distance  # noqa: E402
from dual_loop.pinch_evaluation import extract_events, match_events  # noqa: E402
from feedback_v2 import (  # noqa: E402
    DEFAULT_FINGER_ROTATION_BOUND,
    DEFAULT_ROOT_ROTATION_BOUND,
    DEFAULT_TRANSLATION_BOUND,
    INTRA_ARCHITECTURE,
    install_feedback_v2,
)
from lib.eval_utils.eval_utils import first_align_joints  # noqa: E402
from prepare_training import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPORT_SPLIT,
    DEFAULT_PROTOCOL,
    DEFAULT_SPLIT,
    DEFAULT_VIDEO_ROOT,
    load_sequences,
    convert_raw_history_states,
)
from train_visual import (  # noqa: E402
    DEFAULT_HAWOR,
    DEFAULT_TWM,
    EXPECTED_HAWOR_SHA256,
    EXPECTED_TWM_SHA256,
    freeze_except_visual,
    load_base_twm,
    module_parameter_sha256,
    observer_forward,
    require_finite_metrics,
    seed_everything,
    sha256_file,
)
from v2_cache import (  # noqa: E402
    CACHE_SEED,
    CachedContext,
    build_cache,
    get_windows,
    load_cached_contexts,
    load_manifest,
)
import run_twm_only as twm_only  # noqa: E402


INTRA_FORMAT = "handloop_intra_training"
INTRA_FORMAT_VERSION = 1
DEFAULT_RUN_DIR = HERE / "runs" / "feedback_v2_train"
MAX_STEPS = 1500
SAVE_EVERY = 100
CALIBRATION_EVERY = 100
TWM_TRAIN_MICROBATCH_SIZE = 4
WMPJPE_BLOCK_FRAMES = 100
WMPJPE_MIN_FRAMES = 16
PINCH_PREDICTION_THRESHOLD = 0.275
PINCH_REFERENCE_THRESHOLD = 0.2
PINCH_MIN_FRAMES = 3
PINCH_ONSET_TOLERANCE_SEC = 0.2

LOSS_CONFIG = {
    "joint_mpjpe_scale_m": 0.05,
    "current_nll_weight": 1e-4,
    "relative_joint_mpjpe_weight": 0.5,
    "relative_joint_mpjpe_scale_m": 0.05,
    "joint_velocity_weight": 0.05,
    "joint_velocity_smooth_l1_beta_mps": 1.0,
    "pinch_distance_weight": 0.1,
    "pinch_contact_bce_weight": 0.1,
    "pinch_reference_threshold": 0.2,
    "pinch_prediction_threshold": 0.275,
    "pinch_sigmoid_temperature": 0.05,
}
SAMPLING_RULE = "uniform over cached context groups; each step uses all GT-and-visual-valid rows in one context (at most 16)"
CALIBRATION_RULE = (
    "two fixed uniform 128-frame windows per calibration sequence; split at segment or dt>0.25s; "
    "score GT-and-visual-valid runs in first_align_joints chunks of at most 100 frames, minimum 16"
)
SELECTION_RULE = (
    "qualified iff calibration W-MPJPE is strictly below fresh-zero baseline, event F1 is at least baseline, "
    "and onset MAE is no worse than baseline; best_w minimizes W-MPJPE; best_composite minimizes "
    "W/base_W + (1-F1) + onset_MAE/max(base_onset_MAE,200ms)"
)
TWM_CHECKPOINTING = {
    "enabled": True,
    "method": "batch chunks of 4; torch.utils.checkpoint.checkpoint(use_reentrant=False) per frozen TWM call and chunk",
    "activation_recomputation": "training with gradients enabled only; eval/no_grad uses the original forward",
    "balance_loss_aggregation": "approximate chunk-size-weighted mean of chunk-local scalars; diagnostic only and excluded from training loss",
}
TRAINING_MEMORY_POLICY = {
    "frozen_hawor": "offload to CPU after cache extraction; not used in cached training or calibration",
    "twm_microbatch_size": TWM_TRAIN_MICROBATCH_SIZE,
    "twm_gradient_checkpointing": TWM_CHECKPOINTING,
    "full_context_and_loss": "visual/trajectory/event losses use every selected row from the full context; only independent frozen TWM rows are microbatched",
    "twm_output_semantics": "current distribution tensors are concatenated by row; balance_loss is an approximate diagnostic aggregation and is excluded from the objective",
    "frozen_twm_input_gradients": "preserved through non-reentrant checkpointing while TWM parameters remain frozen",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twm-checkpoint", type=Path, default=DEFAULT_TWM)
    parser.add_argument("--hawor-checkpoint", type=Path, default=DEFAULT_HAWOR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--export-split", type=Path, default=DEFAULT_EXPORT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--cache-only", action="store_true",
                        help="Build/reuse frozen feature cache and compute the single-TWM calibration baseline, then exit")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Reuse the existing cache, compute the frozen baseline, then exit")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--smoke-toy", action="store_true",
                        help="Run CPU-only cache-selection and calibration-mask smoke checks")
    return parser.parse_args()


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _calibration_mask_sha256(
    contexts: list[CachedContext],
    windows: list[dict[str, Any]],
) -> str:
    """Hash the fixed per-frame GT/raw-visual masks and time/segment identity."""
    used = {(str(window["sequence"]), int(frame))
            for window in windows for frame in window["frame_indices"]}
    values: dict[tuple[str, int], dict[str, Any]] = {}
    for context in contexts:
        for row, frame in enumerate(context.frame_indices):
            key = (context.sequence, int(frame))
            if key not in used:
                continue
            values[key] = {
                "gt_valid": bool(context.labels["gt_valid"][row]),
                "visual_valid": bool(context.labels["visual_valid"][row]),
                "timestamp_sec": float(context.labels["timestamp_sec"][row]),
                "segment_id": int(context.labels["segment_id"][row]),
            }
    missing = used - values.keys()
    if missing:
        raise ValueError(f"Calibration mask cache is missing selected frames: {sorted(missing)[:8]}")
    records = [
        {"sequence": sequence, "frame_index": frame, **values[(sequence, frame)]}
        for sequence, frame in sorted(used)
    ]
    return _json_sha256(records)


def _validate_cache_protocol(
    manifest: dict[str, Any],
    calibration_contexts: list[CachedContext],
) -> None:
    """Fail closed if fixed per-video sampling or continuous windows drift."""
    counts = manifest.get("counts", {})
    if counts.get("train_sequences") != 33 or counts.get("calibration_sequences") != 20:
        raise ValueError(
            "Feedback v2 cache must contain the frozen 33 train and 20 calibration sequences")
    selection = manifest.get("train_context_selection", {})
    if len(selection) != 33:
        raise ValueError("Feedback v2 cache must record context selection for all 33 train videos")
    for sequence, rows in selection.items():
        starts = [int(row["context_start"]) for row in rows]
        if len(rows) != 16 or len(set(starts)) != 16:
            raise ValueError(f"{sequence}: expected 16 distinct train contexts, got {len(rows)}")
        for row in rows:
            if len(row["context_frame_indices"]) != 16:
                raise ValueError(f"{sequence}: a train HaWoR context is not exactly 16 frames")
            supervised = row["supervised_frame_indices"]
            if not supervised or len(supervised) > 16 or not set(supervised).issubset(row["context_frame_indices"]):
                raise ValueError(f"{sequence}: invalid GT/visual-valid train frame selection")
    expected_train_items = {
        f"train/{sequence}/ctx_{int(row['context_start']):06d}"
        for sequence, rows in selection.items() for row in rows
    }
    observed_train_items = {
        str(item["item_id"]) for item in manifest.get("items", [])
        if item.get("split") == "train"
    }
    if expected_train_items != observed_train_items:
        raise ValueError("Train context files do not match the fixed 16-context manifest selection")

    windows = get_windows(manifest)
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        by_sequence.setdefault(str(window["sequence"]), []).append(window)
        frames = np.asarray(window["frame_indices"], dtype=np.int64)
        if len(frames) != 128 or not np.all(np.diff(frames) == 1):
            raise ValueError(f"{window['sequence']}: calibration window must be exactly 128 consecutive frames")
    if len(by_sequence) != 20 or any(len(rows) != 2 for rows in by_sequence.values()):
        raise ValueError("Feedback v2 cache must contain two fixed calibration windows for each of 20 videos")
    for sequence, rows in by_sequence.items():
        first, second = sorted(rows, key=lambda row: int(row["start_frame"]))
        if int(first["stop_frame_exclusive"]) > int(second["start_frame"]):
            raise ValueError(f"{sequence}: the two calibration windows overlap")

    context_lookup = {(context.sequence, int(frame)): context.labels
                      for context in calibration_contexts for frame in context.frame_indices}
    for window in windows:
        frames = np.asarray(window["frame_indices"], dtype=np.int64)
        records = []
        for frame in frames:
            labels = context_lookup.get((str(window["sequence"]), int(frame)))
            if labels is None:
                raise ValueError(f"Calibration window cache misses {window['sequence']} frame {frame}")
            row = int(torch.nonzero(labels["frame_indices"] == int(frame), as_tuple=False)[0, 0])
            records.append((float(labels["timestamp_sec"][row]),
                            int(labels["segment_id"][row])))
        times = np.asarray([item[0] for item in records], dtype=np.float64)
        segments = np.asarray([item[1] for item in records], dtype=np.int64)
        if np.any(np.diff(times) <= 0) or np.any(np.diff(times) > 0.25) or np.any(segments != segments[0]):
            raise ValueError(f"{window['sequence']}: calibration window crosses a segment or time gap")
    if not observed_train_items:
        raise ValueError("Feedback v2 cache has no trainable context groups")


def _enable_twm_gradient_checkpointing(system: torch.nn.Module) -> None:
    """Checkpoint frozen TWM activations while preserving gradients to visual input."""
    twm = system.observer.twm
    if getattr(twm, "_feedback_v2_gradient_checkpointing", False):
        return
    original_forward = twm.forward

    def checkpointed_forward(*args: Any, **kwargs: Any) -> Any:
        if not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)
        batch_size = next(
            (int(value.shape[0]) for value in args
             if isinstance(value, torch.Tensor) and value.ndim > 0),
            None,
        )
        if batch_size is None:
            return original_forward(*args, **kwargs)

        def slice_value(value: Any, start: int, stop: int) -> Any:
            if (isinstance(value, torch.Tensor) and value.ndim > 0
                    and int(value.shape[0]) == batch_size):
                return value[start:stop]
            return value

        def merge_outputs(outputs: list[Any], sizes: list[int], key: str | None = None) -> Any:
            first = outputs[0]
            if isinstance(first, dict):
                keys = tuple(first)
                if any(tuple(value) != keys for value in outputs[1:]):
                    raise RuntimeError("TWM microbatch output keys changed between chunks")
                return {
                    child: merge_outputs([value[child] for value in outputs], sizes, child)
                    for child in keys
                }
            if isinstance(first, torch.Tensor):
                if first.ndim == 0:
                    weights = first.new_tensor(sizes)
                    stacked = torch.stack(outputs)
                    return (stacked * weights / weights.sum()).sum()
                return torch.cat(outputs, dim=0)
            if first is None:
                if any(value is not None for value in outputs):
                    raise RuntimeError("TWM microbatch None output changed between chunks")
                return None
            if all(value == first for value in outputs):
                return first
            raise TypeError(f"Unsupported TWM microbatch output at {key}: {type(first).__name__}")

        outputs: list[Any] = []
        chunk_sizes: list[int] = []
        for start in range(0, batch_size, TWM_TRAIN_MICROBATCH_SIZE):
            stop = min(start + TWM_TRAIN_MICROBATCH_SIZE, batch_size)
            chunk_args = tuple(slice_value(value, start, stop) for value in args)
            chunk_kwargs = {key: slice_value(value, start, stop) for key, value in kwargs.items()}

            def run_forward(*chunk_inputs: Any, _chunk_kwargs: dict[str, Any] = chunk_kwargs) -> Any:
                return original_forward(*chunk_inputs, **_chunk_kwargs)

            outputs.append(checkpoint(run_forward, *chunk_args, use_reentrant=False))
            chunk_sizes.append(stop - start)
        return merge_outputs(outputs, chunk_sizes)

    twm.forward = checkpointed_forward
    twm._feedback_v2_gradient_checkpointing = True


def _offload_hawor(system: torch.nn.Module, device: torch.device) -> None:
    """Move frozen HaWoR out of GPU memory after its cached features are ready."""
    if any(parameter.device.type != "cpu" for parameter in system.hawor.parameters()):
        system.hawor.to(device="cpu")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()


def _make_system(
    args: argparse.Namespace,
    device: torch.device,
    *,
    offload_hawor: bool = True,
):
    for path in (args.twm_checkpoint, args.hawor_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    twm_hash = sha256_file(args.twm_checkpoint)
    hawor_hash = sha256_file(args.hawor_checkpoint)
    if twm_hash != EXPECTED_TWM_SHA256 or hawor_hash != EXPECTED_HAWOR_SHA256:
        raise ValueError("Frozen TWM/HaWoR hashes do not match the approved intra-frame protocol")
    config, twm_state = load_base_twm(args.twm_checkpoint, EXPECTED_TWM_SHA256)
    system = build_system(args.hawor_checkpoint, config, device)
    system.observer.twm.load_state_dict(twm_state, strict=True)
    install_feedback_v2(
        system,
        translation_bound=DEFAULT_TRANSLATION_BOUND,
        root_rotation_bound=DEFAULT_ROOT_ROTATION_BOUND,
        finger_rotation_bound=DEFAULT_FINGER_ROTATION_BOUND,
    )
    freeze_except_visual(system)
    _enable_twm_gradient_checkpointing(system)
    system.eval()
    system.observer.twm.eval()
    system.hawor.eval()
    if offload_hawor:
        _offload_hawor(system, device)
    return system, config, twm_hash, hawor_hash


def _frozen_hashes(system: torch.nn.Module) -> dict[str, str]:
    return {
        "hawor_parameters_sha256": module_parameter_sha256(system.hawor),
        "twm_parameters_sha256": module_parameter_sha256(system.observer.twm),
        "geometry_parameters_sha256": module_parameter_sha256(system.geometry),
    }


def _bound_config(system: torch.nn.Module) -> dict[str, float]:
    observer = system.observer
    return {
        "translation_bound": float(observer.translation_bound),
        "root_rotation_bound": float(observer.root_rotation_bound),
        "finger_rotation_bound": float(observer.finger_rotation_bound),
    }


def _input_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device)
        if result.dtype == torch.float16:
            result = result.float()
        return result
    if isinstance(value, tuple):
        return tuple(_input_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_input_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _input_to_device(item, device) for key, item in value.items()}
    return value


def _labels_to_device(labels: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device) for key, value in labels.items()}


def _joint_velocity_loss(
    predicted_joints: torch.Tensor,
    target_joints: torch.Tensor,
    labels: dict[str, torch.Tensor],
    valid: torch.Tensor,
) -> torch.Tensor:
    """Compare real-time joint velocities using differences formed in float64."""
    velocity_loss = predicted_joints.new_zeros(())
    frame_ids = labels["frame_indices"]
    segments = labels["segment_id"]
    timestamps64 = labels["timestamp_sec"].to(dtype=torch.float64)
    if len(frame_ids) <= 1:
        return velocity_loss
    dt64 = timestamps64[1:] - timestamps64[:-1]
    dt = dt64.to(dtype=predicted_joints.dtype)
    edge_valid = (
        valid[1:] & valid[:-1]
        & (frame_ids[1:] == frame_ids[:-1] + 1)
        & (segments[1:] == segments[:-1])
        & (dt64 > 0.0) & (dt64 <= 0.25)
    )
    if edge_valid.any():
        pred_velocity = (predicted_joints[1:] - predicted_joints[:-1]) / dt[:, None, None].clamp_min(1e-6)
        target_velocity = (target_joints[1:] - target_joints[:-1]) / dt[:, None, None].clamp_min(1e-6)
        velocity_loss = F.smooth_l1_loss(
            pred_velocity[edge_valid], target_velocity[edge_valid],
            beta=LOSS_CONFIG["joint_velocity_smooth_l1_beta_mps"],
        )
    return velocity_loss


def _calibration_protocol() -> dict[str, Any]:
    return {
        "calibration_rule": CALIBRATION_RULE,
        "wmpjpe_block_frames": WMPJPE_BLOCK_FRAMES,
        "wmpjpe_min_frames": WMPJPE_MIN_FRAMES,
        "prediction_threshold": PINCH_PREDICTION_THRESHOLD,
        "reference_threshold": PINCH_REFERENCE_THRESHOLD,
        "minimum_event_frames": PINCH_MIN_FRAMES,
        "onset_tolerance_sec": PINCH_ONSET_TOLERANCE_SEC,
        "selection_rule": SELECTION_RULE,
    }


def _single_twm_state(system: torch.nn.Module, model_inputs: dict[str, Any]) -> torch.Tensor:
    """Compute the one-call raw-candidate TWM output matching a zero visual round."""
    raw_state = model_inputs["raw_state"]
    history = model_inputs["history"]
    raw_available = model_inputs["raw_available"].bool()
    history_available = model_inputs["history_available"].bool() & model_inputs["history_mask"].bool()[..., None]
    anchor_rotation, anchor_position = make_anchor(
        raw_state, history, raw_available, history_available)
    history_anchor = to_local(history, anchor_rotation, anchor_position)
    raw_anchor = to_local(raw_state, anchor_rotation, anchor_position)
    result = system.observer.twm(
        history_anchor,
        raw_anchor,
        raw_anchor,
        history_available,
        raw_available,
        raw_available,
        model_inputs["history_times"],
        model_inputs["dt"],
        model_inputs["handedness"].reshape(-1).long().clamp(0, 1),
        model_inputs["history_mask"].bool(),
        None,
    )
    return to_world(result["current"]["state"], anchor_rotation, anchor_position)


def _world_joints(system: torch.nn.Module, state: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    legacy = state_to_legacy(state, beta, system.geometry)
    return system.geometry.joints(legacy)


def _valid_chunk_slices(
    frame_indices: np.ndarray,
    segment_ids: np.ndarray,
    timestamps: np.ndarray,
    valid: np.ndarray,
    *,
    max_frames: int = WMPJPE_BLOCK_FRAMES,
    min_frames: int = WMPJPE_MIN_FRAMES,
) -> list[tuple[int, int]]:
    """Find valid, consecutive, gap-safe half-open scoring runs and chunk them."""
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    count = len(frame_indices)
    if any(array.shape != (count,) for array in (segment_ids, timestamps, valid)):
        raise ValueError("frame, segment, timestamp and validity arrays must align")
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index in range(count):
        connected = (
            index > 0
            and frame_indices[index] == frame_indices[index - 1] + 1
            and segment_ids[index] == segment_ids[index - 1]
            and 0.0 < timestamps[index] - timestamps[index - 1] <= 0.25
            and valid[index - 1]
        )
        if not valid[index]:
            if run_start is not None:
                runs.append((run_start, index))
                run_start = None
            continue
        if run_start is None:
            run_start = index
        elif not connected:
            runs.append((run_start, index))
            run_start = index
    if run_start is not None:
        runs.append((run_start, count))

    chunks = []
    for start, stop in runs:
        cursor = start
        while cursor < stop:
            next_cursor = min(cursor + max_frames, stop)
            if next_cursor - cursor >= min_frames:
                chunks.append((cursor, next_cursor))
            cursor = next_cursor
    return chunks


def _score_wmpjpe(
    predicted: np.ndarray,
    target: np.ndarray,
    labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    finite_target = np.isfinite(target).all(axis=(1, 2))
    eligible = labels["gt_valid"] & labels["visual_valid"] & finite_target
    finite_prediction = np.isfinite(predicted).all(axis=(1, 2))
    if np.any(eligible & ~finite_prediction):
        bad_frames = labels["frame_indices"][eligible & ~finite_prediction].tolist()
        raise FloatingPointError(f"Non-finite candidate joints on fixed W-MPJPE frames {bad_frames[:8]}")
    valid = eligible
    chunks = _valid_chunk_slices(
        labels["frame_indices"], labels["segment_id"], labels["timestamp_sec"], valid)
    total = 0.0
    point_count = 0
    chunk_metrics = []
    for start, stop in chunks:
        gt = torch.as_tensor(target[start:stop], dtype=torch.float32)
        pred = torch.as_tensor(predicted[start:stop], dtype=torch.float32)
        aligned = first_align_joints(gt, pred)
        error = torch.linalg.vector_norm(aligned - gt, dim=-1)
        total += float(error.sum())
        point_count += int(error.numel())
        chunk_metrics.append({
            "frame_start": int(labels["frame_indices"][start]),
            "frame_stop_exclusive": int(labels["frame_indices"][stop - 1]) + 1,
            "frames": int(stop - start),
            "w_mpjpe_mm": float(error.mean() * 1000.0),
        })
    return {
        "w_mpjpe_mm": total / point_count * 1000.0 if point_count else None,
        "scored_frames": int(sum(stop - start for start, stop in chunks)),
        "scored_chunks": len(chunks),
        "chunks": chunk_metrics,
    }


def _score_events(
    predicted: np.ndarray,
    target: np.ndarray,
    labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    finite_prediction = np.isfinite(predicted).all(axis=(1, 2))
    finite_target = np.isfinite(target).all(axis=(1, 2))
    if np.any(labels["visual_valid"] & ~finite_prediction):
        bad_frames = labels["frame_indices"][labels["visual_valid"] & ~finite_prediction].tolist()
        raise FloatingPointError(f"Non-finite candidate joints on fixed event frames {bad_frames[:8]}")
    if np.any(labels["gt_valid"] & ~finite_target):
        bad_frames = labels["frame_indices"][labels["gt_valid"] & ~finite_target].tolist()
        raise FloatingPointError(f"Non-finite GT joints on fixed event frames {bad_frames[:8]}")
    pred_valid = labels["visual_valid"]
    gt_valid = labels["gt_valid"]
    pred_distance = normalized_pinch_distance(predicted)
    gt_distance = normalized_pinch_distance(target)
    pred_events = extract_events(
        pred_distance, labels["timestamp_sec"], pred_valid,
        PINCH_PREDICTION_THRESHOLD, min_frames=PINCH_MIN_FRAMES)
    gt_events = extract_events(
        gt_distance, labels["timestamp_sec"], gt_valid,
        PINCH_REFERENCE_THRESHOLD, min_frames=PINCH_MIN_FRAMES)
    metrics = match_events(gt_events, pred_events,
                           onset_tolerance_sec=PINCH_ONSET_TOLERANCE_SEC)
    tp, fp, fn = int(metrics["tp"]), int(metrics["fp"]), int(metrics["fn"])
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": f1,
        "onset_mae_ms": metrics["onset_mae_ms"],
        "reference_events": len(gt_events),
        "predicted_events": len(pred_events),
        "matches": metrics["matches"],
    }


def _merge_event_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    onset_errors = [float(match["onset_error_ms"])
                    for row in rows for match in row["matches"]]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "reference_events": tp + fn,
        "predicted_events": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "onset_mae_ms": float(np.mean(np.abs(onset_errors))) if onset_errors else None,
        "onset_bias_ms": float(np.mean(onset_errors)) if onset_errors else None,
        "matches": len(onset_errors),
    }


def _predictions_by_frame(
    system: torch.nn.Module,
    contexts: list[CachedContext],
    device: torch.device,
    *,
    compare_zero_baseline: bool = False,
) -> tuple[dict[tuple[str, int], np.ndarray], float | None]:
    predictions: dict[tuple[str, int], np.ndarray] = {}
    maximum_difference = 0.0 if compare_zero_baseline else None
    with torch.no_grad():
        for context in contexts:
            model_inputs = _input_to_device(context.inputs, device)
            beta = model_inputs["projector_args"][0]
            v2_output = observer_forward(system, dict(model_inputs))
            v2_joints = _world_joints(system, v2_output["state"], beta)
            if compare_zero_baseline:
                one_twm_state = _single_twm_state(system, model_inputs)
                one_twm_joints = _world_joints(system, one_twm_state, beta)
                valid = context.labels["visual_valid"].to(device=device).bool()
                if valid.any():
                    delta = (v2_output["state"][valid] - one_twm_state[valid]).abs().max()
                    maximum_difference = max(float(maximum_difference), float(delta.cpu()))
                    if not torch.allclose(v2_output["state"][valid], one_twm_state[valid],
                                          atol=1e-5, rtol=1e-5):
                        raise AssertionError(
                            "Fresh zero-head v2 output differs from one-call TWM on valid rows: "
                            f"max_abs={float(delta.cpu()):.8g}")
                v2_joints = one_twm_joints
            values = v2_joints.detach().float().cpu().numpy()
            for row, frame in enumerate(context.frame_indices):
                key = (context.sequence, int(frame))
                if key in predictions:
                    raise ValueError(f"Duplicate cached calibration frame {key}")
                predictions[key] = values[row]
    return predictions, maximum_difference


def _window_arrays(
    window: dict[str, Any],
    predictions: dict[tuple[str, int], np.ndarray],
    contexts_by_id: dict[str, CachedContext],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sequence = str(window["sequence"])
    frame_ids = np.asarray(window["frame_indices"], dtype=np.int64)
    if len(frame_ids) != 128:
        raise ValueError(f"Calibration window must contain 128 frames: {window}")
    prediction_rows = []
    labels_by_frame: dict[int, dict[str, Any]] = {}
    for item_id in window["context_item_ids"]:
        context = contexts_by_id[item_id]
        for row, frame in enumerate(context.frame_indices):
            labels_by_frame[int(frame)] = {
                key: value[row].item() if value[row].ndim == 0 else value[row].numpy()
                for key, value in context.labels.items()
            }
    target_rows = []
    gt_valid, visual_valid, timestamps, segments = [], [], [], []
    for frame in frame_ids:
        key = (sequence, int(frame))
        if key not in predictions or int(frame) not in labels_by_frame:
            raise ValueError(f"Calibration cache does not cover selected frame {key}")
        label = labels_by_frame[int(frame)]
        prediction_rows.append(predictions[key])
        target_rows.append(label["target_joints"])
        gt_valid.append(bool(label["gt_valid"]))
        visual_valid.append(bool(label["visual_valid"]))
        timestamps.append(float(label["timestamp_sec"]))
        segments.append(int(label["segment_id"]))
    labels = {
        "frame_indices": frame_ids,
        "gt_valid": np.asarray(gt_valid, dtype=bool),
        "visual_valid": np.asarray(visual_valid, dtype=bool),
        "timestamp_sec": np.asarray(timestamps, dtype=np.float64),
        "segment_id": np.asarray(segments, dtype=np.int64),
    }
    return np.asarray(prediction_rows, dtype=np.float32), {
        **labels,
        "target_joints": np.asarray(target_rows, dtype=np.float32),
    }


def evaluate_calibration(
    system: torch.nn.Module,
    calibration_contexts: list[CachedContext],
    manifest: dict[str, Any],
    device: torch.device,
    *,
    compare_zero_baseline: bool = False,
) -> dict[str, Any]:
    """Score only the preselected train-independent calibration windows."""
    predictions, baseline_max_difference = _predictions_by_frame(
        system, calibration_contexts, device,
        compare_zero_baseline=compare_zero_baseline)
    contexts_by_id = {context.item_id: context for context in calibration_contexts}
    window_rows = []
    for window in get_windows(manifest):
        predicted, labels = _window_arrays(window, predictions, contexts_by_id)
        target = labels.pop("target_joints")
        wmpjpe = _score_wmpjpe(predicted, target, labels)
        events = _score_events(predicted, target, labels)
        window_rows.append({
            "sequence": window["sequence"],
            "window_index": int(window["window_index"]),
            "start_frame": int(window["start_frame"]),
            "metrics": {"w_mpjpe_mm": wmpjpe["w_mpjpe_mm"],
                        "scored_frames": wmpjpe["scored_frames"],
                        "scored_chunks": wmpjpe["scored_chunks"],
                        "events": events},
            "w_mpjpe_chunks": wmpjpe["chunks"],
        })

    wmpjpe_values = [(row["metrics"]["w_mpjpe_mm"], row["metrics"]["scored_frames"])
                     for row in window_rows if row["metrics"]["w_mpjpe_mm"] is not None]
    total_frames = sum(frames for _value, frames in wmpjpe_values)
    wmpjpe_mm = (sum(value * frames for value, frames in wmpjpe_values) / total_frames
                 if total_frames else None)
    event_summary = _merge_event_rows([row["metrics"]["events"] for row in window_rows])
    result: dict[str, Any] = {
        "w_mpjpe_mm": wmpjpe_mm,
        "w_mpjpe_scored_frames": total_frames,
        "w_mpjpe_scored_chunks": sum(row["metrics"]["scored_chunks"] for row in window_rows),
        "events": event_summary,
        "windows": window_rows,
        "window_count": len(window_rows),
        "window_rule": CALIBRATION_RULE,
        "eligibility": {
            "w_mpjpe": "GT-valid AND raw HaWoR visual-valid frames with finite GT; non-finite candidate output rejects calibration",
            "events_prediction": "raw HaWoR visual-valid frames; non-finite candidate output rejects calibration",
            "events_reference": "GT-valid frames; non-finite GT output rejects calibration",
        },
        "event_thresholds": {
            "prediction_threshold": PINCH_PREDICTION_THRESHOLD,
            "reference_threshold": PINCH_REFERENCE_THRESHOLD,
            "minimum_event_frames": PINCH_MIN_FRAMES,
            "onset_tolerance_sec": PINCH_ONSET_TOLERANCE_SEC,
        },
    }
    if compare_zero_baseline:
        result["zero_v2_equals_single_twm"] = {
            "max_abs_world_state_difference": float(baseline_max_difference or 0.0),
            "atol": 1e-5,
            "rtol": 1e-5,
            "checked_valid_rows_only": True,
        }
    require_finite_metrics({"w_mpjpe_mm": result["w_mpjpe_mm"]}, "calibration aggregate")
    if event_summary["f1"] is not None and not np.isfinite(event_summary["f1"]):
        raise FloatingPointError(f"Non-finite calibration event F1: {event_summary['f1']}")
    return result


def _write_baseline(
    system: torch.nn.Module,
    calibration_contexts: list[CachedContext],
    manifest: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    cache_manifest_sha256: str,
) -> dict[str, Any]:
    visual = system.observer.visual
    zero_heads = all(
        torch.count_nonzero(parameter.detach()).item() == 0
        for name in ("root_delta_head", "finger_delta_head")
        for parameter in getattr(visual, name).parameters()
    )
    if not zero_heads:
        raise ValueError("Single-TWM baseline must be computed from freshly initialized zero visual heads")
    system.eval()
    result = evaluate_calibration(
        system, calibration_contexts, manifest, device, compare_zero_baseline=True)
    result.update({
        "method": "fresh zero-head bounded v2 (verified equivalent to one raw-candidate TWM call)",
        "architecture": INTRA_ARCHITECTURE,
        "selection_rule": SELECTION_RULE,
        "visual_head_initialization": "fresh HandLoopVisual; root_delta_head and finger_delta_head are zero",
        "base_checkpoints": manifest["base_checkpoints"],
        "frozen_parameter_hashes": manifest["frozen_parameter_hashes"],
        "cache_manifest_sha256": cache_manifest_sha256,
        "calibration_windows_sha256": _json_sha256(manifest.get("calibration_windows", [])),
        "calibration_mask_sha256": _calibration_mask_sha256(
            calibration_contexts, get_windows(manifest)),
        "calibration_protocol": _calibration_protocol(),
        "calibration_protocol_sha256": _json_sha256(_calibration_protocol()),
    })
    path = run_dir / "baseline_calibration.json"
    _atomic_json(result, path)
    return result


def _load_baseline(
    path: Path,
    *,
    cache_manifest_sha256: str,
    calibration_windows_sha256: str,
    calibration_mask_sha256: str,
    calibration_protocol_sha256: str,
) -> dict[str, Any]:
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read calibration baseline {path}: {error}") from error
    if baseline.get("architecture") != INTRA_ARCHITECTURE:
        raise ValueError("Calibration baseline architecture differs from feedback v2")
    equivalence = baseline.get("zero_v2_equals_single_twm", {})
    if float(equivalence.get("max_abs_world_state_difference", float("inf"))) > 1e-5:
        raise ValueError("Stored baseline did not pass the fresh-zero versus single-TWM equivalence check")
    expected = {
        "cache_manifest_sha256": cache_manifest_sha256,
        "calibration_windows_sha256": calibration_windows_sha256,
        "calibration_mask_sha256": calibration_mask_sha256,
        "calibration_protocol_sha256": calibration_protocol_sha256,
    }
    if baseline.get("calibration_protocol") != _calibration_protocol():
        raise ValueError("Stored baseline calibration thresholds or selection rules differ from feedback v2")
    mismatches = {key: (baseline.get(key), value) for key, value in expected.items()
                  if baseline.get(key) != value}
    if mismatches:
        raise ValueError(f"Stored baseline is not bound to the current cache/windows/masks/protocol: {mismatches}")
    return baseline


def _score_is_qualified(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    base_w = baseline.get("w_mpjpe_mm")
    score_w = candidate.get("w_mpjpe_mm")
    base_event = baseline.get("events", {})
    score_event = candidate.get("events", {})
    base_f1, candidate_f1 = base_event.get("f1"), score_event.get("f1")
    base_onset, candidate_onset = base_event.get("onset_mae_ms"), score_event.get("onset_mae_ms")
    values = (base_w, score_w, base_f1, candidate_f1, base_onset, candidate_onset)
    if any(value is None or not np.isfinite(float(value)) for value in values):
        return False
    return (float(score_w) < float(base_w)
            and float(candidate_f1) >= float(base_f1)
            and float(candidate_onset) <= float(base_onset))


def _composite_score(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    w = candidate.get("w_mpjpe_mm")
    f1 = candidate.get("events", {}).get("f1")
    onset = candidate.get("events", {}).get("onset_mae_ms")
    if w is None or not np.isfinite(float(w)):
        return float("inf")
    base_w = max(float(baseline.get("w_mpjpe_mm") or 1.0), 1e-6)
    base_onset = baseline.get("events", {}).get("onset_mae_ms")
    onset_scale = max(float(base_onset) if base_onset is not None else 0.0, 200.0)
    # Missing event matches receive the worst finite event terms, so the run
    # still has a deterministic fallback candidate while remaining unqualified.
    f1_term = 1.0 - float(f1) if f1 is not None and np.isfinite(float(f1)) else 1.0
    onset_term = (float(onset) / onset_scale
                  if onset is not None and np.isfinite(float(onset)) else 1.0)
    return float(w) / base_w + f1_term + onset_term


def _checkpoint_payload(
    system: torch.nn.Module,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    rng: np.random.Generator,
    base_twm_hash: str,
    base_hawor_hash: str,
    frozen_hashes: dict[str, str],
    cache_manifest_sha256: str,
    baseline: dict[str, Any],
    last_calibration: dict[str, Any] | None,
    selection_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": INTRA_ARCHITECTURE,
        "format": INTRA_FORMAT,
        "format_version": INTRA_FORMAT_VERSION,
        "visual": {key: value.detach().cpu()
                   for key, value in system.observer.visual.state_dict().items()},
        "config": copy.deepcopy(config),
        "bounds": _bound_config(system),
        "base_twm_sha256": base_twm_hash,
        "base_hawor_sha256": base_hawor_hash,
        "trainer": {
            "global_step": int(step),
            "seed": CACHE_SEED,
            "optimizer": optimizer.state_dict(),
            "sampler_state_json": json.dumps(rng.bit_generator.state, sort_keys=True),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else [],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "loss_config": LOSS_CONFIG,
            "sampler": SAMPLING_RULE,
            "calibration_rule": CALIBRATION_RULE,
            "selection_rule": SELECTION_RULE,
            "twm_activation_checkpointing": TWM_CHECKPOINTING,
            "training_memory_policy": TRAINING_MEMORY_POLICY,
            "selection_state": selection_state,
            "last_calibration": last_calibration,
            "baseline_calibration": {
                "w_mpjpe_mm": baseline["w_mpjpe_mm"],
                "events": baseline["events"],
                "zero_v2_equals_single_twm": baseline["zero_v2_equals_single_twm"],
            },
            "frozen_parameter_hashes": frozen_hashes,
            "cache_manifest_sha256": cache_manifest_sha256,
            "trainable_component": "observer.visual only; bounded feedback transforms have no trainable parameters",
            "maximum_training_steps": MAX_STEPS,
            "calibration_every": CALIBRATION_EVERY,
            "checkpoint_every": SAVE_EVERY,
        },
    }


def _restore_resume(
    path: Path,
    system: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    *,
    base_twm_hash: str,
    base_hawor_hash: str,
    cache_manifest_sha256: str,
) -> tuple[int, np.random.Generator, dict[str, Any], dict[str, Any] | None]:
    checkpoint = twm_only.load_checkpoint_restricted(path)
    if (checkpoint.get("architecture") != INTRA_ARCHITECTURE
            or checkpoint.get("format") != INTRA_FORMAT
            or checkpoint.get("format_version") != INTRA_FORMAT_VERSION):
        raise ValueError("Resume checkpoint is not a feedback v2 intra-frame training checkpoint")
    if checkpoint.get("config") != config or checkpoint.get("bounds") != _bound_config(system):
        raise ValueError("Resume HandLoop configuration or physical bounds differ")
    if (checkpoint.get("base_twm_sha256") != base_twm_hash
            or checkpoint.get("base_hawor_sha256") != base_hawor_hash):
        raise ValueError("Resume checkpoint has different frozen base checkpoints")
    trainer = checkpoint.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("Resume checkpoint has no trainer metadata")
    if trainer.get("cache_manifest_sha256") != cache_manifest_sha256:
        raise ValueError("Resume checkpoint cache manifest differs from the current cache")
    step = int(trainer.get("global_step", 0))
    if step <= 0:
        raise ValueError("Resume checkpoint must have at least one completed step")
    system.observer.visual.load_state_dict(checkpoint["visual"], strict=True)
    optimizer_state = trainer.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ValueError("Resume checkpoint is missing AdamW state")
    optimizer.load_state_dict(optimizer_state)
    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(trainer["sampler_state_json"])
    torch_rng = trainer.get("torch_rng_state")
    if isinstance(torch_rng, torch.Tensor):
        torch.set_rng_state(torch_rng.cpu())
    cuda_rng = trainer.get("cuda_rng_state")
    if torch.cuda.is_available() and isinstance(cuda_rng, list) and cuda_rng:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
    return (step, rng, dict(trainer.get("selection_state") or {}),
            trainer.get("last_calibration"))


def _context_loss(
    system: torch.nn.Module,
    context: CachedContext,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = _input_to_device(context.inputs, device)
    beta = inputs["projector_args"][0]
    # Labels remain in a separate object and are fetched after the observer
    # forward, preserving the frozen observation/GT boundary.
    output = observer_forward(system, dict(inputs))
    labels = _labels_to_device(context.labels, device)
    target_legacy = labels["target_legacy"]
    target_joints = labels["target_joints"]
    valid = labels["gt_valid"].bool() & labels["visual_valid"].bool()
    valid &= torch.isfinite(target_joints).all(dim=(-1, -2))
    if not valid.any():
        raise ValueError(f"Training context has no valid GT/visual rows: {context.item_id}")

    with torch.no_grad():
        target_state = legacy_to_state(target_legacy, system.geometry)
    target_local = to_local(target_state, output["anchor_rotation"], output["anchor_position"])
    current_nll = -log_prob(output["current"], target_local).mean()

    predicted_joints = _world_joints(system, output["state"], beta)
    joint_m = (predicted_joints - target_joints).norm(dim=-1)[valid].mean()

    # Remove each context's first supervised wrist translation before comparing
    # the full hand trajectory. Rotations and articulation remain supervised.
    first_valid = torch.nonzero(valid, as_tuple=False)[0, 0]
    pred_relative = predicted_joints - predicted_joints[first_valid:first_valid + 1, :1]
    target_relative = target_joints - target_joints[first_valid:first_valid + 1, :1]
    relative_m = (pred_relative - target_relative).norm(dim=-1)[valid].mean()

    velocity_loss = _joint_velocity_loss(predicted_joints, target_joints, labels, valid)

    pred_pinch = torch.linalg.vector_norm(
        predicted_joints[:, 4] - predicted_joints[:, 8], dim=-1)
    pred_scale = torch.linalg.vector_norm(
        predicted_joints[:, 0] - predicted_joints[:, 9], dim=-1).clamp_min(1e-6)
    target_pinch = torch.linalg.vector_norm(
        target_joints[:, 4] - target_joints[:, 8], dim=-1)
    target_scale = torch.linalg.vector_norm(
        target_joints[:, 0] - target_joints[:, 9], dim=-1).clamp_min(1e-6)
    pred_distance = pred_pinch / pred_scale
    target_distance = target_pinch / target_scale
    pinch_distance_loss = F.smooth_l1_loss(pred_distance[valid], target_distance[valid])
    temperature = LOSS_CONFIG["pinch_sigmoid_temperature"]
    threshold = LOSS_CONFIG["pinch_reference_threshold"]
    target_contact = torch.sigmoid((threshold - target_distance[valid]) / temperature)
    predicted_contact = torch.sigmoid(
        (LOSS_CONFIG["pinch_prediction_threshold"] - pred_distance[valid]) / temperature)
    contact_bce = F.binary_cross_entropy(
        predicted_contact.clamp(1e-6, 1.0 - 1e-6), target_contact)

    loss = (
        joint_m / LOSS_CONFIG["joint_mpjpe_scale_m"]
        + LOSS_CONFIG["current_nll_weight"] * current_nll
        + LOSS_CONFIG["relative_joint_mpjpe_weight"]
        * relative_m / LOSS_CONFIG["relative_joint_mpjpe_scale_m"]
        + LOSS_CONFIG["joint_velocity_weight"] * velocity_loss
        + LOSS_CONFIG["pinch_distance_weight"] * pinch_distance_loss
        + LOSS_CONFIG["pinch_contact_bce_weight"] * contact_bce
    )
    metrics = {
        "loss": loss,
        "joint_m": joint_m.detach(),
        "current_nll": current_nll.detach(),
        "relative_joint_m": relative_m.detach(),
        "joint_velocity_smooth_l1_mps": velocity_loss.detach(),
        "pinch_distance_smooth_l1": pinch_distance_loss.detach(),
        "pinch_contact_bce": contact_bce.detach(),
        "valid_frames": valid.sum().detach(),
    }
    return metrics, loss


def _metric_float(metrics: dict[str, torch.Tensor]) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for key, value in metrics.items():
        scalar = float(value.detach().cpu())
        output[key] = int(scalar) if key == "valid_frames" else scalar
    return output


def _cpu_toy_smoke() -> dict[str, Any]:
    """Check only deterministic mask/window selection on a compact CPU fixture."""
    from prepare_training import PreparedSequence
    from v2_cache import choose_calibration_windows, choose_training_contexts

    count = 420
    times = np.arange(count, dtype=np.float64) / 30.0
    times[210:] += 1.0  # A timestamp discontinuity splits windows even in a segment.
    sequence = PreparedSequence(
        sequence="Ptoy_toy", split="calibration", archive_path=Path("toy.npz"),
        image_paths=np.asarray([f"frame-{index}" for index in range(count)]),
        boxes=np.zeros((count, 4), np.float32), image_size_wh=np.asarray([640, 480], np.float32),
        focal_length=np.full(count, 500.0, np.float32), timestamp_sec=times,
        segment_id=np.zeros(count, np.int64), rotation_w2c=np.repeat(np.eye(3, dtype=np.float32)[None], count, 0),
        translation_w2c=np.zeros((count, 3), np.float32),
        visual_valid=np.ones(count, bool), gt_valid=np.ones(count, bool),
        visual_legacy=np.zeros((count, 109), np.float32), target_legacy=np.zeros((count, 109), np.float32),
    )
    windows = choose_calibration_windows(sequence)
    assert len(windows) == 2
    for window in windows:
        ids = window["frame_indices"]
        assert len(ids) == 128
        assert np.all(np.diff(ids) == 1)
        assert np.all(np.diff(times[ids]) <= 0.25)

    train_seq = copy.copy(sequence)
    train_seq.split = "train"
    train_seq.segment_id = np.zeros(count, np.int64)
    train_seq.visual_valid[40:48] = False
    train_seq.gt_valid[96:104] = False
    selected = choose_training_contexts(train_seq, rng=np.random.default_rng(CACHE_SEED))
    assert len(selected) <= 16
    for context in selected:
        ids = context["frame_indices"]
        assert len(ids) > 0 and len(ids) <= 16
        assert np.all(train_seq.gt_valid[ids] & train_seq.visual_valid[ids])
        assert len(context["context_frame_indices"]) == 16

    frame_ids = np.arange(64)
    segments = np.zeros(64, np.int64)
    timestamps = frame_ids / 30.0
    valid = np.ones(64, bool)
    valid[20] = False
    chunks = _valid_chunk_slices(frame_ids, segments, timestamps, valid, max_frames=30, min_frames=16)
    assert chunks == [(0, 20), (21, 51)]

    # A candidate cannot improve a score by turning an eligible frame into NaN.
    labels = {
        "frame_indices": frame_ids[:16],
        "segment_id": segments[:16],
        "timestamp_sec": timestamps[:16],
        "gt_valid": np.ones(16, dtype=bool),
        "visual_valid": np.ones(16, dtype=bool),
    }
    target_joints = np.zeros((16, 21, 3), dtype=np.float32)
    bad_prediction = target_joints.copy()
    bad_prediction[5, 0, 0] = np.nan
    nonfinite_rejected = 0
    for scorer in (_score_wmpjpe, _score_events):
        try:
            scorer(bad_prediction, target_joints, labels)
        except FloatingPointError:
            nonfinite_rejected += 1
    assert nonfinite_rejected == 2

    # A stored baseline is rejected if its cache identity changes.
    baseline_fixture = {
        "architecture": INTRA_ARCHITECTURE,
        "zero_v2_equals_single_twm": {"max_abs_world_state_difference": 0.0},
        "cache_manifest_sha256": "old-cache",
        "calibration_windows_sha256": "windows",
        "calibration_mask_sha256": "mask",
        "calibration_protocol": _calibration_protocol(),
        "calibration_protocol_sha256": _json_sha256(_calibration_protocol()),
    }
    baseline_rejected = False
    with tempfile.TemporaryDirectory(prefix="feedback-v2-smoke-") as temporary:
        path = Path(temporary) / "baseline.json"
        path.write_text(json.dumps(baseline_fixture), encoding="utf-8")
        try:
            _load_baseline(
                path,
                cache_manifest_sha256="new-cache",
                calibration_windows_sha256="windows",
                calibration_mask_sha256="mask",
                calibration_protocol_sha256=_json_sha256(_calibration_protocol()),
            )
        except ValueError as error:
            baseline_rejected = "current cache/windows/masks" in str(error)
    assert baseline_rejected

    # Absolute timestamp origins must not change the measured joint velocity.
    predicted_joints = torch.zeros((3, 21, 3), dtype=torch.float32)
    target_joints_t = torch.zeros_like(predicted_joints)
    predicted_joints[1, :, 0] = 0.02
    predicted_joints[2, :, 0] = 0.06
    target_joints_t[1, :, 0] = 0.01
    target_joints_t[2, :, 0] = 0.04
    velocity_labels = {
        "frame_indices": torch.arange(3),
        "segment_id": torch.zeros(3, dtype=torch.long),
        "timestamp_sec": torch.tensor([57584.0, 57584.0 + 1 / 30, 57584.0 + 2 / 30], dtype=torch.float64),
    }
    velocity_valid = torch.ones(3, dtype=torch.bool)
    velocity_a = _joint_velocity_loss(predicted_joints, target_joints_t, velocity_labels, velocity_valid)
    shifted_labels = dict(velocity_labels)
    shifted_labels["timestamp_sec"] = velocity_labels["timestamp_sec"] + 1_000_000.0
    velocity_b = _joint_velocity_loss(predicted_joints, target_joints_t, shifted_labels, velocity_valid)
    assert torch.allclose(velocity_a, velocity_b, atol=1e-6, rtol=1e-6)
    return {
        "calibration_windows": len(windows),
        "train_contexts": len(selected),
        "valid_chunks": chunks,
        "nonfinite_candidate_paths_rejected": nonfinite_rejected,
        "stale_baseline_rejected": baseline_rejected,
        "timestamp_shift_velocity_equal": bool(torch.allclose(velocity_a, velocity_b, atol=1e-6, rtol=1e-6)),
    }


def main() -> None:
    args = _parse_args()
    if args.smoke_toy:
        print(json.dumps({"cpu_toy_smoke": _cpu_toy_smoke()}, sort_keys=True), flush=True)
        return
    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"--steps must be between 1 and the frozen {MAX_STEPS}-step cap")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip <= 0:
        raise ValueError("learning-rate and gradient-clip must be positive; weight-decay non-negative")
    if args.cache_only and args.baseline_only:
        raise ValueError("Choose only one of --cache-only and --baseline-only")
    if args.cache_only and args.resume is not None:
        raise ValueError("--cache-only cannot resume training")
    if args.rebuild_cache and args.baseline_only:
        raise ValueError("--baseline-only cannot rebuild the feature cache")
    torch.set_num_threads(max(1, args.cpu_threads))
    seed_everything(CACHE_SEED)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    cache_dir = Path(args.cache_dir or (args.run_dir / "cache"))
    manifest_path = cache_dir / "manifest.json"
    if args.baseline_only and not manifest_path.is_file():
        raise FileNotFoundError(f"--baseline-only requires a completed cache manifest: {manifest_path}")
    # A partial cache has no manifest. Keep HaWoR on device until extraction
    # finishes; completed caches never need the frontend on GPU for training.
    needs_feature_extraction = not manifest_path.is_file() or args.rebuild_cache
    system, config, base_twm_hash, base_hawor_hash = _make_system(
        args, device, offload_hawor=not needs_feature_extraction)
    frozen_hashes = _frozen_hashes(system)

    if args.cache_only or (not manifest_path.is_file() and not args.baseline_only and args.resume is None):
        groups, _protocol = load_sequences(
            data_root=args.data_root, video_root=args.video_root,
            split_manifest=args.split_manifest, export_split=args.export_split,
            protocol_path=args.protocol, check_images=not args.skip_image_check)
        all_sequences = groups["train"] + groups["calibration"]
        convert_raw_history_states(all_sequences, system.geometry, device)
        manifest = build_cache(
            system=system, groups=groups, device=device, cache_dir=cache_dir,
            base_hashes={"twm_sha256": base_twm_hash, "hawor_sha256": base_hawor_hash},
            frozen_parameter_hashes=frozen_hashes,
            split_manifest=args.split_manifest, export_split=args.export_split,
            protocol_path=args.protocol, video_root=args.video_root,
            rebuild=args.rebuild_cache,
        )
    else:
        if args.rebuild_cache:
            raise ValueError("--rebuild-cache requires --cache-only or a missing cache")
        manifest = load_manifest(cache_dir)
        if manifest.get("base_checkpoints") != {
            "twm_sha256": base_twm_hash, "hawor_sha256": base_hawor_hash}:
            raise ValueError("Cached features were built from different frozen checkpoints")
        if manifest.get("frozen_parameter_hashes") != frozen_hashes:
            raise ValueError("Cached features were built from different frozen in-memory model parameters")
    _offload_hawor(system, device)
    cache_manifest_sha = sha256_file(manifest_path)
    calibration_contexts = load_cached_contexts(cache_dir, "calibration")
    _validate_cache_protocol(manifest, calibration_contexts)
    calibration_windows_sha = _json_sha256(manifest.get("calibration_windows", []))
    calibration_mask_sha = _calibration_mask_sha256(
        calibration_contexts, get_windows(manifest))
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = run_dir / "baseline_calibration.json"
    if args.baseline_only or args.cache_only or not baseline_path.is_file():
        baseline = _write_baseline(
            system, calibration_contexts, manifest, device, run_dir, cache_manifest_sha)
    else:
        baseline = _load_baseline(
            baseline_path,
            cache_manifest_sha256=cache_manifest_sha,
            calibration_windows_sha256=calibration_windows_sha,
            calibration_mask_sha256=calibration_mask_sha,
            calibration_protocol_sha256=_json_sha256(_calibration_protocol()),
        )
        if baseline.get("base_checkpoints") != manifest.get("base_checkpoints"):
            raise ValueError("Stored baseline was scored with different base checkpoints")
    if args.cache_only or args.baseline_only:
        print(json.dumps({
            "mode": "cache_only" if args.cache_only else "baseline_only",
            "cache_dir": str(cache_dir.resolve()),
            "cache_counts": manifest["counts"],
            "baseline": {
                "w_mpjpe_mm": baseline["w_mpjpe_mm"],
                "events": baseline["events"],
                "zero_v2_equals_single_twm": baseline["zero_v2_equals_single_twm"],
            },
        }, sort_keys=True), flush=True)
        return

    train_contexts = load_cached_contexts(cache_dir, "train")
    if not train_contexts:
        raise ValueError("No training contexts in cache")
    optimizer = torch.optim.AdamW(
        system.observer.visual.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    visual_ids = {id(parameter) for parameter in system.observer.visual.parameters()
                  if parameter.requires_grad}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if not visual_ids or optimizer_ids != visual_ids:
        raise RuntimeError("AdamW must contain exactly the trainable visual parameters")

    initial_frozen_hashes = _frozen_hashes(system)
    rng = np.random.default_rng(CACHE_SEED)
    global_step = 0
    last_calibration: dict[str, Any] | None = None
    selection_state = {
        "best_w_mpjpe_mm": None,
        "best_composite_score": None,
        "qualified_best_w_mpjpe_mm": None,
        "qualified_best_step": None,
        "selection_qualified": False,
        "candidate_count": 0,
    }
    if args.resume is not None:
        global_step, rng, selection_state, last_calibration = _restore_resume(
            args.resume, system, optimizer, config,
            base_twm_hash=base_twm_hash, base_hawor_hash=base_hawor_hash,
            cache_manifest_sha256=cache_manifest_sha,
        )
    if global_step >= args.steps:
        raise ValueError(f"Resume global step {global_step} already reaches --steps={args.steps}")

    log_path = run_dir / "training.jsonl"
    if args.resume is None and log_path.exists():
        raise FileExistsError(f"Refusing to append a fresh run to existing log: {log_path}")
    if args.resume is not None and not log_path.exists():
        raise FileNotFoundError(f"Resume run is missing its append-only log: {log_path}")

    records_path = run_dir / "calibration_candidates.jsonl"
    if args.resume is None and records_path.exists():
        raise FileExistsError(f"Refusing to append a fresh run to existing candidates: {records_path}")
    system.eval()
    system.observer.visual.train()
    system.observer.twm.eval()
    system.hawor.eval()
    with log_path.open("a", encoding="utf-8") as log, records_path.open("a", encoding="utf-8") as candidate_log:
        for step in range(global_step + 1, args.steps + 1):
            context_index = int(rng.integers(len(train_contexts)))
            context = train_contexts[context_index]
            optimizer.zero_grad(set_to_none=True)
            metrics, loss = _context_loss(system, context, device)
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite loss at step {step} context={context.item_id}")
            loss.backward()
            parameters = [parameter for parameter in system.observer.visual.parameters()
                          if parameter.requires_grad and parameter.grad is not None]
            if not parameters:
                raise RuntimeError(f"No visual gradients at step {step}")
            if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"Non-finite visual gradient at step {step}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            if not torch.isfinite(gradient_norm).all() or float(gradient_norm.detach().cpu()) <= 0.0:
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"Zero or non-finite visual gradient norm at step {step}")
            optimizer.step()
            global_step = step

            should_calibrate = step % CALIBRATION_EVERY == 0 or step == args.steps
            should_save = step % SAVE_EVERY == 0 or step == args.steps
            candidate = None
            is_qualified = False
            composite_score = None
            improved_w = False
            improved_composite = False
            improved_qualified = False
            if should_calibrate:
                system.eval()
                calibration = evaluate_calibration(
                    system, calibration_contexts, manifest, device)
                last_calibration = calibration
                system.observer.visual.train()
                is_qualified = _score_is_qualified(calibration, baseline)
                composite_score = _composite_score(calibration, baseline)
                selection_state["candidate_count"] = int(selection_state.get("candidate_count", 0)) + 1
                current_w = calibration.get("w_mpjpe_mm")
                if current_w is not None and (
                        selection_state.get("best_w_mpjpe_mm") is None
                        or float(current_w) < float(selection_state["best_w_mpjpe_mm"])):
                    selection_state["best_w_mpjpe_mm"] = float(current_w)
                    selection_state["best_w_step"] = int(step)
                    improved_w = True
                if (np.isfinite(composite_score)
                        and (selection_state.get("best_composite_score") is None
                             or composite_score < float(selection_state["best_composite_score"]))):
                    selection_state["best_composite_score"] = float(composite_score)
                    selection_state["best_composite_step"] = int(step)
                    improved_composite = True
                if is_qualified and (
                        selection_state.get("qualified_best_w_mpjpe_mm") is None
                        or float(current_w) < float(selection_state["qualified_best_w_mpjpe_mm"])):
                    selection_state["qualified_best_w_mpjpe_mm"] = float(current_w)
                    selection_state["qualified_best_step"] = int(step)
                    selection_state["selection_qualified"] = True
                    improved_qualified = True
                candidate = {
                    "step": int(step),
                    "selection_qualified": bool(is_qualified),
                    "composite_score": composite_score,
                    "calibration": calibration,
                    "baseline": {
                        "w_mpjpe_mm": baseline["w_mpjpe_mm"],
                        "events": baseline["events"],
                    },
                    "selection_rule": SELECTION_RULE,
                    "training_memory_policy": TRAINING_MEMORY_POLICY,
                    "best_w_improved": improved_w,
                    "best_composite_improved": improved_composite,
                    "qualified_best_improved": improved_qualified,
                }
                candidate_log.write(json.dumps(candidate, sort_keys=True) + "\n")
                candidate_log.flush()

            record = {
                "step": int(step),
                "context_item_id": context.item_id,
                "context_sequence": context.sequence,
                "context_start": context.context_start,
                "context_frame_ids": context.frame_indices.tolist(),
                "batch_size": context.size,
                "train": _metric_float(metrics),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "calibration": candidate["calibration"] if candidate else None,
                "selection_qualified": bool(is_qualified) if should_calibrate else None,
                "composite_score": composite_score,
                "selection_state": selection_state,
                "training_memory_policy": TRAINING_MEMORY_POLICY,
                "loss_config": LOSS_CONFIG,
                "sampling_rule": SAMPLING_RULE,
                "base_twm_sha256": base_twm_hash,
                "base_hawor_sha256": base_hawor_hash,
            }
            if should_save:
                current_frozen_hashes = _frozen_hashes(system)
                record["frozen_parameter_hashes"] = current_frozen_hashes
                record["frozen_weights_unchanged"] = current_frozen_hashes == initial_frozen_hashes
                if not record["frozen_weights_unchanged"]:
                    raise RuntimeError("Frozen HaWoR/TWM/MANO parameters changed during feedback v2 training")
                payload = _checkpoint_payload(
                    system, config, optimizer, step=step, rng=rng,
                    base_twm_hash=base_twm_hash, base_hawor_hash=base_hawor_hash,
                    frozen_hashes=initial_frozen_hashes,
                    cache_manifest_sha256=cache_manifest_sha, baseline=baseline,
                    last_calibration=last_calibration,
                    selection_state=selection_state,
                )
                _atomic_save(payload, run_dir / "latest.pt")
                if improved_w:
                    _atomic_save(payload, run_dir / "best_w.pt")
                if improved_composite:
                    _atomic_save(payload, run_dir / "best_composite.pt")
                if improved_qualified:
                    _atomic_save(payload, run_dir / "qualified_best.pt")
            serialized = json.dumps(record, sort_keys=True)
            print(serialized, flush=True)
            log.write(serialized + "\n")
            log.flush()

    if not selection_state.get("selection_qualified"):
        selection_summary = {
            "selection_qualified": False,
            "qualified_best_written": False,
            "fallback_checkpoint": "best_composite.pt" if (run_dir / "best_composite.pt").is_file() else "best_w.pt",
            "reason": "No calibration candidate met all predeclared W-MPJPE, event-F1, and onset criteria",
        }
    else:
        selection_summary = {
            "selection_qualified": True,
            "qualified_best_written": (run_dir / "qualified_best.pt").is_file(),
            "qualified_best_step": selection_state.get("qualified_best_step"),
        }
    _atomic_json(selection_summary, run_dir / "selection.json")
    print(json.dumps({"final_selection": selection_summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
