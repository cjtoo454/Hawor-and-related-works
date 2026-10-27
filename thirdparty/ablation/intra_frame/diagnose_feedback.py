#!/usr/bin/env python3
"""Probe raw, zero-visual TWM, and trained visual feedback on calibration frames."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT / "thirdparty" / "try"
TWM_ONLY = ROOT / "thirdparty" / "ablation" / "twm_only"
sys.path[:0] = [str(ROOT), str(PROJECT), str(PROJECT / "scripts"), str(TWM_ONLY), str(HERE)]

from dual_loop.handloop_state import (  # noqa: E402
    make_anchor,
    so3_log,
    state_to_legacy,
    to_local,
    to_world,
    unpack_state,
)
from prepare_training import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPORT_SPLIT,
    DEFAULT_PROTOCOL,
    DEFAULT_SPLIT,
    DEFAULT_VIDEO_ROOT,
    PreparedSequence,
    calibration_samples,
    context_indices,
    convert_raw_history_states,
    load_sequences,
)
from runtime import load_intra_system, sha256  # noqa: E402
from train_visual import build_live_batch, observer_forward  # noqa: E402


DEFAULT_VISUAL = HERE / "runs" / "visual_train_gpu1" / "best.pt"
DEFAULT_TWM = ROOT / "thirdparty" / "ablation" / "training_v2" / "checkpoints" / "latest.pt"
DEFAULT_HAWOR = ROOT / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"
DEFAULT_OUTPUT = HERE / "runs" / "diagnose_feedback_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-checkpoint", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--twm-checkpoint", type=Path, default=DEFAULT_TWM)
    parser.add_argument("--hawor-checkpoint", type=Path, default=DEFAULT_HAWOR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--export-split", type=Path, default=DEFAULT_EXPORT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-video", type=int, choices=(1, 8), default=1,
                        help="1 uses calibration_samples midpoints; 8 uniformly samples each calibration video")
    parser.add_argument("--fix-controls", action="store_true",
                        help="Run four v1-weight feedback controls for causal repair attribution on the fixed 20 frames")
    parser.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _uniform_calibration_samples(
    sequences: list[PreparedSequence], samples_per_video: int,
) -> list[tuple[PreparedSequence, int]]:
    samples: list[tuple[PreparedSequence, int]] = []
    for sequence in sequences:
        eligible = np.flatnonzero(sequence.gt_valid & sequence.visual_valid)
        if len(eligible) < samples_per_video:
            raise ValueError(
                f"{sequence.sequence}: expected at least {samples_per_video} GT/raw-visual-valid frames, "
                f"found {len(eligible)}")
        if samples_per_video == 1:
            # This is exactly the fixed selection used by calibration_samples.
            selected = [int(eligible[len(eligible) // 2])]
        else:
            # Integer linspace over the eligible index list gives deterministic,
            # endpoint-inclusive, near-uniform frame selection.
            positions = np.linspace(0, len(eligible) - 1, samples_per_video, dtype=np.int64)
            selected = [int(eligible[position]) for position in positions]
            if len(set(selected)) != samples_per_video:
                raise RuntimeError(f"{sequence.sequence}: uniform sample indices are not unique")
        samples.extend((sequence, frame) for frame in selected)
    return samples


def _context_groups(
    samples: list[tuple[PreparedSequence, int]],
) -> list[list[tuple[PreparedSequence, int]]]:
    grouped: OrderedDict[tuple[str, tuple[int, ...]], list[tuple[PreparedSequence, int]]] = OrderedDict()
    for sequence, frame in samples:
        ids, _ = context_indices(sequence, frame)
        key = (sequence.sequence, tuple(int(index) for index in ids))
        grouped.setdefault(key, []).append((sequence, frame))
    return list(grouped.values())


@contextmanager
def _observer_rounds(system: torch.nn.Module, rounds: int) -> Iterator[None]:
    observer = system.observer
    original = observer.config
    observer.config = replace(original, rounds=rounds)
    try:
        yield
    finally:
        observer.config = original


def _zero_visual_forward(self: torch.nn.Module, **kwargs: torch.Tensor) -> torch.Tensor:
    candidate = kwargs["candidate_camera"]
    return candidate.new_zeros((candidate.shape[0], 51))


@contextmanager
def _zero_visual(system: torch.nn.Module) -> Iterator[None]:
    visual = system.observer.visual
    original = visual.forward
    visual.forward = MethodType(_zero_visual_forward, visual)
    try:
        yield
    finally:
        visual.forward = original


def _direct_raw_twm_once(system: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mirror run_twm_only.restore_raw_sequence's candidate=raw single call."""
    raw = inputs["raw_state"]
    available = inputs["raw_available"].bool()
    history = inputs["history"]
    history_available = inputs["history_available"].bool()
    anchor_rotation, anchor_position = make_anchor(raw, history, available, history_available)
    history_anchor = to_local(history, anchor_rotation, anchor_position)
    raw_anchor = to_local(raw, anchor_rotation, anchor_position)
    prediction = system.observer.twm(
        history=history_anchor,
        raw=raw_anchor,
        candidate=raw_anchor,
        history_available=history_available,
        raw_available=available,
        candidate_available=available,
        history_times=inputs["history_times"],
        dt=inputs["dt"],
        handedness=inputs["handedness"].reshape(-1).long().clamp(0, 1),
        history_mask=inputs["history_mask"].bool(),
        future_times=None,
    )
    return to_world(prediction["current"]["state"], anchor_rotation, anchor_position)


def _run_observer(
    system: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    rounds: int,
    zero_visual: bool,
) -> dict[str, Any]:
    with _observer_rounds(system, rounds):
        if zero_visual:
            with _zero_visual(system):
                return observer_forward(system, dict(inputs))
        return observer_forward(system, dict(inputs))


def _capture_v1_control(
    system: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    bound_visual: bool,
    feedback_v2: Any,
) -> tuple[dict[str, Any], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Run the original recursive observer while temporarily capturing deltas."""
    visual = system.observer.visual
    original_forward = visual.forward
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    bounds = {
        "translation_bound": feedback_v2.DEFAULT_TRANSLATION_BOUND,
        "root_rotation_bound": feedback_v2.DEFAULT_ROOT_ROTATION_BOUND,
        "finger_rotation_bound": feedback_v2.DEFAULT_FINGER_ROTATION_BOUND,
    }

    def wrapped_forward(self: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        raw_delta = original_forward(*args, **kwargs)
        applied_delta = (feedback_v2.bound_visual_delta(raw_delta, **bounds)
                         if bound_visual else raw_delta)
        captured.append((raw_delta.detach(), applied_delta.detach()))
        return applied_delta

    visual.forward = MethodType(wrapped_forward, visual)
    try:
        return observer_forward(system, dict(inputs)), captured
    finally:
        visual.forward = original_forward


def _capture_v2_control(
    system: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    bounded: bool,
    feedback_v2: Any,
) -> tuple[dict[str, Any], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Install the v2 observer, toggle only its bound function, then restore v1."""
    original_observer = system.observer
    original_rounds = original_observer.config.rounds
    original_reobserve = original_observer.config.reobserve
    original_bound = feedback_v2.bound_visual_delta
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    bounds = {
        "translation_bound": feedback_v2.DEFAULT_TRANSLATION_BOUND,
        "root_rotation_bound": feedback_v2.DEFAULT_ROOT_ROTATION_BOUND,
        "finger_rotation_bound": feedback_v2.DEFAULT_FINGER_ROTATION_BOUND,
    }

    def controlled_bound(delta: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if bounded:
            applied = original_bound(delta, **kwargs)
        else:
            # Preserve the v2 control flow while disabling just the nonlinear bound.
            applied = delta
        captured.append((delta.detach(), applied.detach()))
        return applied

    try:
        feedback_v2.install_feedback_v2(system, **bounds)
        feedback_v2.bound_visual_delta = controlled_bound
        output = observer_forward(system, dict(inputs))
        return output, captured
    finally:
        feedback_v2.bound_visual_delta = original_bound
        system.observer = original_observer
        original_observer.config.rounds = original_rounds
        original_observer.config.reobserve = original_reobserve


def _delta_row(
    captured: list[tuple[torch.Tensor, torch.Tensor]], row: int,
) -> list[dict[str, float]]:
    result = []
    for round_index, (raw, applied) in enumerate(captured, start=1):
        raw_position = raw[row, :3].norm().item()
        raw_root = raw[row, 3:6].norm().item()
        raw_fingers = raw[row, 6:].reshape(15, 3).norm(dim=-1)
        applied_position = applied[row, :3].norm().item()
        applied_root = applied[row, 3:6].norm().item()
        applied_fingers = applied[row, 6:].reshape(15, 3).norm(dim=-1)
        result.append({
            "round": round_index,
            "raw_translation_norm_m": float(raw_position),
            "applied_translation_norm_m": float(applied_position),
            "raw_root_rotation_norm_rad": float(raw_root),
            "applied_root_rotation_norm_rad": float(applied_root),
            "raw_finger_rotation_mean_norm_rad": float(raw_fingers.mean().item()),
            "applied_finger_rotation_mean_norm_rad": float(applied_fingers.mean().item()),
            "raw_finger_rotation_max_norm_rad": float(raw_fingers.max().item()),
            "applied_finger_rotation_max_norm_rad": float(applied_fingers.max().item()),
        })
    return result


def _metric_arrays(
    state: torch.Tensor,
    raw_state: torch.Tensor,
    betas: torch.Tensor,
    target_legacy: torch.Tensor,
    target_joints: torch.Tensor,
    geometry: torch.nn.Module,
) -> dict[str, np.ndarray]:
    legacy = state_to_legacy(state, betas, geometry)
    joints = geometry.joints(legacy)
    raw_joints = geometry.joints(state_to_legacy(raw_state, betas, geometry))
    joint_error = (joints - target_joints).norm(dim=-1).mean(dim=-1)
    wrist_error = (state[:, :3] - target_joints[:, 0]).norm(dim=-1)
    raw_joint_error = (raw_joints - target_joints).norm(dim=-1).mean(dim=-1)
    raw_wrist_error = (raw_state[:, :3] - target_joints[:, 0]).norm(dim=-1)
    return {
        "joint_mpjpe_mm": joint_error.detach().cpu().numpy() * 1000.0,
        "wrist_error_mm": wrist_error.detach().cpu().numpy() * 1000.0,
        "wrist_xyz_m": state[:, :3].detach().cpu().numpy(),
        "wrist_displacement_from_raw_mm": (
            (state[:, :3] - raw_state[:, :3]).norm(dim=-1).detach().cpu().numpy() * 1000.0),
        "joint_position_delta_from_raw_mm": (
            (joints - raw_joints).norm(dim=-1).mean(dim=-1).detach().cpu().numpy() * 1000.0),
        "joint_mpjpe_delta_vs_raw_mm": (
            (joint_error - raw_joint_error).detach().cpu().numpy() * 1000.0),
        "wrist_error_delta_vs_raw_mm": (
            (wrist_error - raw_wrist_error).detach().cpu().numpy() * 1000.0),
    }


def _metric_row(metrics: dict[str, np.ndarray], row: int) -> dict[str, Any]:
    return {key: (metrics[key][row].tolist() if metrics[key][row].ndim else float(metrics[key][row]))
            for key in metrics}


def _rotation_delta_degrees(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    _, left_rotation = unpack_state(left)
    _, right_rotation = unpack_state(right)
    relative = right_rotation[:, 0] @ left_rotation[:, 0].transpose(-1, -2)
    angle = torch.linalg.vector_norm(so3_log(relative), dim=-1) * (180.0 / math.pi)
    return angle.detach().cpu().numpy()


def _round_trace(
    output: dict[str, Any],
    raw_state: torch.Tensor,
    betas: torch.Tensor,
    target_legacy: torch.Tensor,
    target_joints: torch.Tensor,
    geometry: torch.nn.Module,
) -> tuple[list[list[dict[str, Any]]], list[torch.Tensor]]:
    traces: list[list[dict[str, Any]]] = [[] for _ in range(len(raw_state))]
    states: list[torch.Tensor] = []
    previous = raw_state
    for round_index, distribution in enumerate(output["rounds"], start=1):
        world = to_world(distribution["state"], output["anchor_rotation"], output["anchor_position"])
        states.append(world)
        metrics = _metric_arrays(world, raw_state, betas, target_legacy, target_joints, geometry)
        step_displacement = (world[:, :3] - previous[:, :3]).norm(dim=-1).detach().cpu().numpy() * 1000.0
        raw_displacement = (world[:, :3] - raw_state[:, :3]).norm(dim=-1).detach().cpu().numpy() * 1000.0
        step_rotation = _rotation_delta_degrees(previous, world)
        raw_rotation = _rotation_delta_degrees(raw_state, world)
        for row in range(len(raw_state)):
            trace = _metric_row(metrics, row)
            trace.update({
                "round": round_index,
                "root_step_displacement_mm": float(step_displacement[row]),
                "root_displacement_from_raw_mm": float(raw_displacement[row]),
                "root_rotation_step_delta_deg": float(step_rotation[row]),
                "root_rotation_delta_from_raw_deg": float(raw_rotation[row]),
            })
            traces[row].append(trace)
        previous = world
    return traces, states


def _summary(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    summary = {}
    for key, values in metrics.items():
        if key == "wrist_xyz_m":
            summary["mean_wrist_xyz_m"] = np.asarray(values, dtype=np.float64).mean(axis=0).tolist()
        else:
            summary[f"mean_{key}"] = float(np.asarray(values, dtype=np.float64).mean())
    return summary


def _sample_manifest(samples: list[tuple[PreparedSequence, int]]) -> list[dict[str, Any]]:
    return [{
        "sequence": sequence.sequence,
        "frame_index": int(frame),
        "timestamp_sec": float(sequence.timestamp_sec[frame]),
        "gt_valid": bool(sequence.gt_valid[frame]),
        "visual_valid": bool(sequence.visual_valid[frame]),
    } for sequence, frame in samples]


def run_fix_controls(args: argparse.Namespace) -> Path:
    """Isolate recursion and delta-bound effects using the same v1 visual weights."""
    if args.samples_per_video != 1:
        raise ValueError("--fix-controls requires the fixed one-midpoint-per-calibration-video sample plan")
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
        index_text = str(args.device).partition(":")[2]
        if index_text and int(index_text) >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {args.device}")
    device = torch.device(args.device)

    groups, _ = load_sequences(
        data_root=args.data_root,
        video_root=args.video_root,
        split_manifest=args.split_manifest,
        export_split=args.export_split,
        protocol_path=args.protocol,
        splits=("calibration",),
        check_images=not args.skip_image_check,
    )
    calibration = groups["calibration"]
    samples = _uniform_calibration_samples(calibration, 1)
    canonical = calibration_samples(calibration)
    if [(s.sequence, i) for s, i in canonical] != [(s.sequence, i) for s, i in samples]:
        raise RuntimeError("Fix-control sample plan differs from calibration_samples")

    system, metadata = load_intra_system(
        visual_checkpoint_path=args.visual_checkpoint,
        twm_checkpoint_path=args.twm_checkpoint,
        hawor_checkpoint_path=args.hawor_checkpoint,
        device=device,
    )
    system.eval()
    original_observer = system.observer
    convert_raw_history_states(calibration, system.geometry, device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "control_fixes_20frames.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_path}")

    import feedback_v2  # noqa: PLC0415 - imported only for this explicit control diagnostic

    mode_names = (
        "v1_recursive_unbounded",
        "nonrecursive_visual_q_unbounded",
        "v1_recursive_bounded",
        "v2_nonrecursive_visual_q_bounded",
    )
    frame_rows: list[dict[str, Any]] = []
    context_groups = _context_groups(samples)
    elapsed_start = time.perf_counter()
    bounds = {
        "translation_bound": feedback_v2.DEFAULT_TRANSLATION_BOUND,
        "root_rotation_bound": feedback_v2.DEFAULT_ROOT_ROTATION_BOUND,
        "finger_rotation_bound": feedback_v2.DEFAULT_FINGER_ROTATION_BOUND,
    }

    with torch.no_grad():
        for group_index, batch_samples in enumerate(context_groups, start=1):
            inputs, betas, records = build_live_batch(system, batch_samples, device)
            if not bool(inputs["raw_available"].all()):
                raise RuntimeError("The fixed control diagnostic contains a raw-invalid frame")

            outputs: dict[str, tuple[dict[str, Any], list[tuple[torch.Tensor, torch.Tensor]]]] = {}
            outputs[mode_names[0]] = _capture_v1_control(
                system, inputs, bound_visual=False, feedback_v2=feedback_v2)
            outputs[mode_names[2]] = _capture_v1_control(
                system, inputs, bound_visual=True, feedback_v2=feedback_v2)
            outputs[mode_names[1]] = _capture_v2_control(
                system, inputs, bounded=False, feedback_v2=feedback_v2)
            outputs[mode_names[3]] = _capture_v2_control(
                system, inputs, bounded=True, feedback_v2=feedback_v2)
            if system.observer is not original_observer:
                raise RuntimeError("Temporary feedback_v2 observer was not restored")

            targets = torch.as_tensor(
                np.stack([sequence.target_legacy[frame] for sequence, frame in batch_samples]),
                dtype=torch.float32, device=device)
            target_joints = system.geometry.joints(targets)
            raw_state = inputs["raw_state"]
            mode_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in mode_names}
            for mode in mode_names:
                output, captured = outputs[mode]
                metrics = _metric_arrays(
                    output["state"], raw_state, betas, targets, target_joints, system.geometry)
                traces, _ = _round_trace(
                    output, raw_state, betas, targets, target_joints, system.geometry)
                if len(captured) != 2 or len(traces) != len(batch_samples):
                    raise RuntimeError(f"{mode}: expected two visual updates and one trace per frame")
                for local_index in range(len(batch_samples)):
                    row = _metric_row(metrics, local_index)
                    row["round_trace"] = traces[local_index]
                    row["visual_delta_trace"] = _delta_row(captured, local_index)
                    row["terminal_root_displacement_from_raw_mm"] = float(
                        (output["state"][local_index, :3] - raw_state[local_index, :3]).norm().item()
                        * 1000.0)
                    row["terminal_root_rotation_from_raw_deg"] = float(
                        _rotation_delta_degrees(
                            raw_state[local_index:local_index + 1],
                            output["state"][local_index:local_index + 1],
                        )[0])
                    mode_rows[mode].append(row)

            for local_index, (sequence, frame) in enumerate(batch_samples):
                frame_rows.append({
                    "sequence": sequence.sequence,
                    "frame_index": int(frame),
                    "timestamp_sec": float(records[local_index]["timestamp_sec"]),
                    "context_indices": [int(value) for value in records[local_index]["context_indices"]],
                    "gt_valid": bool(sequence.gt_valid[frame]),
                    "visual_valid": bool(sequence.visual_valid[frame]),
                    "modes": {name: mode_rows[name][local_index] for name in mode_names},
                })
            if group_index == 1 or group_index % 5 == 0 or group_index == len(context_groups):
                print(f"fix-controls context {group_index}/{len(context_groups)} "
                      f"({len(frame_rows)}/20 frames, {time.perf_counter() - elapsed_start:.1f}s)",
                      flush=True)

    mode_summaries: dict[str, dict[str, Any]] = {}
    for mode in mode_names:
        values = [row["modes"][mode] for row in frame_rows]
        root_total = np.asarray(
            [item["terminal_root_displacement_from_raw_mm"] for item in values], dtype=np.float64)
        mode_summaries[mode] = {
            "mean_joint_mpjpe_mm": float(np.mean([item["joint_mpjpe_mm"] for item in values])),
            "mean_wrist_error_mm": float(np.mean([item["wrist_error_mm"] for item in values])),
            "mean_wrist_xyz_m": np.asarray(
                [item["wrist_xyz_m"] for item in values], dtype=np.float64).mean(axis=0).tolist(),
            "mean_terminal_root_displacement_from_raw_mm": float(root_total.mean()),
            "p95_terminal_root_displacement_from_raw_mm": float(np.percentile(root_total, 95)),
            "round_summaries": [],
            "visual_delta_summaries": [],
        }
        for round_index in range(2):
            round_rows = [item["round_trace"][round_index] for item in values]
            root_step = np.asarray(
                [item["root_step_displacement_mm"] for item in round_rows], dtype=np.float64)
            root_rotation = np.asarray(
                [item["root_rotation_step_delta_deg"] for item in round_rows], dtype=np.float64)
            delta_rows = [item["visual_delta_trace"][round_index] for item in values]
            mode_summaries[mode]["round_summaries"].append({
                "round": round_index + 1,
                "mean_root_step_displacement_mm": float(root_step.mean()),
                "p95_root_step_displacement_mm": float(np.percentile(root_step, 95)),
                "mean_root_rotation_step_delta_deg": float(root_rotation.mean()),
                "p95_root_rotation_step_delta_deg": float(np.percentile(root_rotation, 95)),
            })
            mode_summaries[mode]["visual_delta_summaries"].append({
                "round": round_index + 1,
                **{
                    f"mean_{key}": float(np.mean([item[key] for item in delta_rows]))
                    for key in (
                        "raw_translation_norm_m", "applied_translation_norm_m",
                        "raw_root_rotation_norm_rad", "applied_root_rotation_norm_rad",
                        "raw_finger_rotation_mean_norm_rad", "applied_finger_rotation_mean_norm_rad",
                    )
                },
                "p95_raw_translation_norm_m": float(np.percentile(
                    [item["raw_translation_norm_m"] for item in delta_rows], 95)),
                "p95_applied_translation_norm_m": float(np.percentile(
                    [item["applied_translation_norm_m"] for item in delta_rows], 95)),
                "p95_raw_root_rotation_norm_rad": float(np.percentile(
                    [item["raw_root_rotation_norm_rad"] for item in delta_rows], 95)),
                "p95_applied_root_rotation_norm_rad": float(np.percentile(
                    [item["applied_root_rotation_norm_rad"] for item in delta_rows], 95)),
            })

    report = {
        "schema": "intra_frame_feedback_v1_weight_fix_control_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "purpose": "Calibration-only causal attribution of nonrecursive visual-only candidate and bounded visual delta fixes; no training or test-set model selection.",
        "checkpoint_provenance": {
            "visual_checkpoint": str(args.visual_checkpoint.resolve()),
            "visual_checkpoint_sha256": metadata["visual_checkpoint_sha256"],
            "visual_global_step": metadata["visual_global_step"],
            "visual_weights_origin": "v1 recursive + unbounded visual feedback; reused unchanged in every control, including v2 transfer diagnostic",
            "twm_checkpoint_sha256": metadata["twm_checkpoint_sha256"],
            "hawor_checkpoint_sha256": metadata["hawor_checkpoint_sha256"],
            "v2_trained_checkpoint_used": False,
            "weights_written": False,
        },
        "sample_plan": {
            "split": "calibration",
            "rule": "calibration_samples: middle gt_valid & visual_valid frame per calibration video",
            "sample_count": len(samples),
            "matched_mask": "Same fixed valid frames and unchanged raw/history availability tensors for all four modes.",
            "test_prepared_data_loaded": False,
            "samples": _sample_manifest(samples),
        },
        "bounds": bounds,
        "mode_conditions": {
            mode_names[0]: "Original HandLoopObserver.forward_step: each round starts visual correction from previous TWM output; unbounded visual delta.",
            mode_names[1]: "feedback_v2 observer: independent visual-only q is updated; TWM output is reprojected but not accumulated into q; bound function temporarily replaced by identity.",
            mode_names[2]: "Original recursive HandLoopObserver.forward_step with the unchanged v1 visual forward wrapped by feedback_v2.bound_visual_delta.",
            mode_names[3]: "feedback_v2.install_feedback_v2 with default physical vector-norm bounds and independent visual-only q; v1 visual weights are transferred unchanged for diagnosis.",
        },
        "input_features": "build_live_batch runs once per calibration context; all four conditions reuse the same raw state, patches, camera, and history inputs.",
        "mode_summaries": mode_summaries,
        "elapsed_seconds": float(time.perf_counter() - elapsed_start),
        "frames": frame_rows,
    }
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_path}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({
        "output": str(output_path),
        "sample_count": len(samples),
        "mode_summaries": mode_summaries,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    return output_path


def run(args: argparse.Namespace) -> Path:
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
        index_text = str(args.device).partition(":")[2]
        if index_text and int(index_text) >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {args.device}")
    device = torch.device(args.device)

    groups, _ = load_sequences(
        data_root=args.data_root,
        video_root=args.video_root,
        split_manifest=args.split_manifest,
        export_split=args.export_split,
        protocol_path=args.protocol,
        splits=("calibration",),
        check_images=not args.skip_image_check,
    )
    calibration = groups["calibration"]
    samples = _uniform_calibration_samples(calibration, args.samples_per_video)
    if args.samples_per_video == 1:
        # Guard against accidentally drifting from the repository's fixed 20-frame selection.
        canonical = calibration_samples(calibration)
        if [(s.sequence, i) for s, i in canonical] != [(s.sequence, i) for s, i in samples]:
            raise RuntimeError("One-per-video sample plan differs from calibration_samples")

    system, metadata = load_intra_system(
        visual_checkpoint_path=args.visual_checkpoint,
        twm_checkpoint_path=args.twm_checkpoint,
        hawor_checkpoint_path=args.hawor_checkpoint,
        device=device,
    )
    system.eval()
    convert_raw_history_states(calibration, system.geometry, device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"feedback_{len(samples)}frames.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_path}")

    mode_states: dict[str, list[np.ndarray]] = {
        key: [] for key in ("raw", "twm_zero_round1", "twm_zero_round2", "trained_visual_round1", "trained_visual_round2")
    }
    frame_rows: list[dict[str, Any]] = []
    max_equivalence_error = 0.0
    max_one_vs_two_first_error = 0.0
    elapsed_start = time.perf_counter()
    context_groups = _context_groups(samples)

    with torch.no_grad():
        for group_index, batch_samples in enumerate(context_groups, start=1):
            # HaWoR and IAM features are computed once per context group and reused by all modes.
            inputs, betas, records = build_live_batch(system, batch_samples, device)
            raw_state = inputs["raw_state"]
            if not bool(inputs["raw_available"].all()):
                raise RuntimeError("The fixed calibration probe unexpectedly contains a raw-invalid frame")
            zero_one = _run_observer(system, inputs, rounds=1, zero_visual=True)
            zero_two = _run_observer(system, inputs, rounds=2, zero_visual=True)
            trained_one = _run_observer(system, inputs, rounds=1, zero_visual=False)
            trained_two = _run_observer(system, inputs, rounds=2, zero_visual=False)
            reference = _direct_raw_twm_once(system, inputs)

            group_equivalence = float((zero_one["state"] - reference).abs().max().item())
            max_equivalence_error = max(max_equivalence_error, group_equivalence)
            if not torch.allclose(zero_one["state"], reference, atol=1e-5, rtol=1e-5):
                raise RuntimeError(
                    "zero-visual rounds=1 differs from run_twm_only raw-candidate TWM logic: "
                    f"max_abs={group_equivalence}")
            one_vs_two_error = float((trained_one["state"] - to_world(
                trained_two["rounds"][0]["state"], trained_two["anchor_rotation"],
                trained_two["anchor_position"])).abs().max().item())
            max_one_vs_two_first_error = max(max_one_vs_two_first_error, one_vs_two_error)

            targets = torch.as_tensor(
                np.stack([sequence.target_legacy[frame] for sequence, frame in batch_samples]),
                dtype=torch.float32, device=device)
            target_joints = system.geometry.joints(targets)
            outputs = {
                "raw": None,
                "twm_zero_round1": zero_one,
                "twm_zero_round2": zero_two,
                "trained_visual_round1": trained_one,
                "trained_visual_round2": trained_two,
            }
            states: dict[str, torch.Tensor] = {"raw": raw_state}
            traces: dict[str, list[list[dict[str, Any]]]] = {}
            round_world: dict[str, list[torch.Tensor]] = {}
            for mode in list(outputs)[1:]:
                output = outputs[mode]
                assert output is not None
                states[mode] = output["state"]
                traces[mode], round_world[mode] = _round_trace(
                    output, raw_state, betas, targets, target_joints, system.geometry)

            metrics = {
                mode: _metric_arrays(state, raw_state, betas, targets, target_joints, system.geometry)
                for mode, state in states.items()
            }
            for mode in mode_states:
                mode_states[mode].append(metrics[mode]["joint_mpjpe_mm"])
            zero_one_metrics = metrics["twm_zero_round1"]
            trained_two_first_metrics = _metric_arrays(
                round_world["trained_visual_round2"][0], raw_state, betas, targets,
                target_joints, system.geometry)
            for local_index, (sequence, frame) in enumerate(batch_samples):
                per_mode: dict[str, Any] = {}
                for mode, values in metrics.items():
                    row = _metric_row(values, local_index)
                    row["delta_vs_twm_zero_round1"] = {
                        "joint_mpjpe_delta_mm": float(
                            values["joint_mpjpe_mm"][local_index] - zero_one_metrics["joint_mpjpe_mm"][local_index]),
                        "wrist_error_delta_mm": float(
                            values["wrist_error_mm"][local_index] - zero_one_metrics["wrist_error_mm"][local_index]),
                        "wrist_position_delta_mm": float((states[mode][local_index, :3]
                                                           - states["twm_zero_round1"][local_index, :3]).norm().item()
                                                          * 1000.0),
                    }
                    if mode == "trained_visual_round1":
                        row["root_displacement_vs_zero_visual_same_round_mm"] = float((
                            states[mode][local_index, :3]
                            - states["twm_zero_round1"][local_index, :3]).norm().item() * 1000.0)
                    elif mode == "trained_visual_round2":
                        row["root_displacement_vs_zero_visual_same_round_mm"] = float((
                            states[mode][local_index, :3]
                            - states["twm_zero_round2"][local_index, :3]).norm().item() * 1000.0)
                    if mode in traces:
                        row["round_trace"] = traces[mode][local_index]
                    per_mode[mode] = row
                trained_first = _metric_row(trained_two_first_metrics, local_index)
                trained_first["delta_vs_twm_zero_round1"] = {
                    "joint_mpjpe_delta_mm": float(
                        trained_two_first_metrics["joint_mpjpe_mm"][local_index]
                        - zero_one_metrics["joint_mpjpe_mm"][local_index]),
                    "wrist_error_delta_mm": float(
                        trained_two_first_metrics["wrist_error_mm"][local_index]
                        - zero_one_metrics["wrist_error_mm"][local_index]),
                    "wrist_position_delta_mm": float((
                        round_world["trained_visual_round2"][0][local_index, :3]
                        - states["twm_zero_round1"][local_index, :3]).norm().item() * 1000.0),
                }
                frame_rows.append({
                    "sequence": sequence.sequence,
                    "frame_index": int(frame),
                    "timestamp_sec": float(records[local_index]["timestamp_sec"]),
                    "context_indices": [int(value) for value in records[local_index]["context_indices"]],
                    "gt_valid": bool(sequence.gt_valid[frame]),
                    "visual_valid": bool(sequence.visual_valid[frame]),
                    "zero_visual_round1_twm_equivalence_max_abs_state_error": group_equivalence,
                    "trained_visual_round1_vs_round2_first_max_abs_state_error": one_vs_two_error,
                    "modes": per_mode,
                    "trained_visual_round2_first_round": trained_first,
                })

            if group_index == 1 or group_index % 10 == 0 or group_index == len(context_groups):
                elapsed = time.perf_counter() - elapsed_start
                print(f"processed context {group_index}/{len(context_groups)} "
                      f"({len(frame_rows)}/{len(samples)} frames, {elapsed:.1f}s)", flush=True)

    mode_summaries: dict[str, dict[str, float]] = {}
    for mode in mode_states:
        all_values = np.concatenate(mode_states[mode], axis=0)
        mode_summaries[mode] = {"mean_joint_mpjpe_mm": float(all_values.mean())}
    # Reconstruct complete summary arrays from the already retained per-frame rows.
    for mode in mode_summaries:
        values = [row["modes"][mode] for row in frame_rows]
        for metric_name in (
            "joint_mpjpe_mm", "wrist_error_mm", "wrist_displacement_from_raw_mm",
            "joint_position_delta_from_raw_mm", "joint_mpjpe_delta_vs_raw_mm",
            "wrist_error_delta_vs_raw_mm",
        ):
            mode_summaries[mode][f"mean_{metric_name}"] = float(
                np.mean([float(item[metric_name]) for item in values]))
        mode_summaries[mode]["mean_wrist_xyz_m"] = np.asarray(
            [item["wrist_xyz_m"] for item in values], dtype=np.float64).mean(axis=0).tolist()
        mode_summaries[mode]["mean_joint_mpjpe_delta_vs_twm_zero_round1_mm"] = float(
            np.mean([item["delta_vs_twm_zero_round1"]["joint_mpjpe_delta_mm"] for item in values]))
        mode_summaries[mode]["mean_wrist_error_delta_vs_twm_zero_round1_mm"] = float(
            np.mean([item["delta_vs_twm_zero_round1"]["wrist_error_delta_mm"] for item in values]))
        mode_summaries[mode]["mean_wrist_position_delta_vs_twm_zero_round1_mm"] = float(
            np.mean([item["delta_vs_twm_zero_round1"]["wrist_position_delta_mm"] for item in values]))

    trained_first_values = [row["trained_visual_round2_first_round"] for row in frame_rows]
    trained_first_summary = {
        "mean_joint_mpjpe_mm": float(np.mean([item["joint_mpjpe_mm"] for item in trained_first_values])),
        "mean_wrist_error_mm": float(np.mean([item["wrist_error_mm"] for item in trained_first_values])),
        "mean_joint_mpjpe_delta_vs_twm_zero_round1_mm": float(np.mean([
            item["delta_vs_twm_zero_round1"]["joint_mpjpe_delta_mm"] for item in trained_first_values])),
        "mean_wrist_error_delta_vs_twm_zero_round1_mm": float(np.mean([
            item["delta_vs_twm_zero_round1"]["wrist_error_delta_mm"] for item in trained_first_values])),
        "mean_wrist_position_delta_vs_twm_zero_round1_mm": float(np.mean([
            item["delta_vs_twm_zero_round1"]["wrist_position_delta_mm"] for item in trained_first_values])),
    }

    round_summaries: dict[str, list[dict[str, float]]] = {}
    for mode in ("twm_zero_round1", "twm_zero_round2", "trained_visual_round1", "trained_visual_round2"):
        round_count = 1 if mode.endswith("round1") else 2
        values = [row["modes"][mode]["round_trace"] for row in frame_rows]
        round_summaries[mode] = []
        for round_index in range(round_count):
            round_rows = [item[round_index] for item in values]
            displacement = np.asarray(
                [item["root_step_displacement_mm"] for item in round_rows], dtype=np.float64)
            rotation = np.asarray(
                [item["root_rotation_step_delta_deg"] for item in round_rows], dtype=np.float64)
            round_summaries[mode].append({
                "round": float(round_index + 1),
                "mean_root_step_displacement_mm": float(displacement.mean()),
                "p95_root_step_displacement_mm": float(np.percentile(displacement, 95)),
                "mean_root_rotation_step_delta_deg": float(rotation.mean()),
                "p95_root_rotation_step_delta_deg": float(np.percentile(rotation, 95)),
            })
    trained_visual_delta_summary = {}
    for mode in ("trained_visual_round1", "trained_visual_round2"):
        displacements = np.asarray([
            row["modes"][mode]["root_displacement_vs_zero_visual_same_round_mm"]
            for row in frame_rows
        ], dtype=np.float64)
        trained_visual_delta_summary[mode] = {
            "mean_root_displacement_vs_zero_visual_same_round_mm": float(displacements.mean()),
            "p95_root_displacement_vs_zero_visual_same_round_mm": float(np.percentile(displacements, 95)),
        }

    report = {
        "schema": "intra_frame_feedback_calibration_probe_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "sample_plan": {
            "split": "calibration",
            "samples_per_video": int(args.samples_per_video),
            "sample_count": len(samples),
            "rule": ("prepare_training.calibration_samples: middle gt_valid & visual_valid frame per video"
                     if args.samples_per_video == 1 else
                     "8 endpoint-inclusive uniform indices from each video's gt_valid & visual_valid frame list"),
            "matched_mask": "All modes score the same GT-valid & visual-valid sample list; original raw/history availability masks are preserved.",
            "test_prepared_data_loaded": False,
            "samples": _sample_manifest(samples),
        },
        "checkpoints": {
            "visual_checkpoint": str(args.visual_checkpoint.resolve()),
            "visual_checkpoint_sha256": metadata["visual_checkpoint_sha256"],
            "visual_global_step": metadata["visual_global_step"],
            "twm_checkpoint": str(args.twm_checkpoint.resolve()),
            "twm_checkpoint_sha256": metadata["twm_checkpoint_sha256"],
            "hawor_checkpoint": str(args.hawor_checkpoint.resolve()),
            "hawor_checkpoint_sha256": metadata["hawor_checkpoint_sha256"],
            "rounds_config_source": "dataclasses.replace(observer.config, rounds=N), restored in finally",
            "weights_written": False,
        },
        "input_features": "Each calibration context runs build_live_batch once; the same patches/raw state/history are reused for all modes.",
        "mode_summaries": mode_summaries,
        "round_summaries": round_summaries,
        "trained_visual_root_delta_vs_zero_visual": trained_visual_delta_summary,
        "trained_visual_round2_first_round_summary": trained_first_summary,
        "checks": {
            "zero_visual_round1_matches_run_twm_only_raw_candidate_call": {
                "max_abs_state_error": max_equivalence_error,
                "atol": 1e-5,
                "rtol": 1e-5,
                "passed": max_equivalence_error <= 1e-5,
            },
            "trained_visual_round1_matches_round2_first_round": {
                "max_abs_state_error": max_one_vs_two_first_error,
                "atol": 1e-5,
                "rtol": 1e-5,
                "passed": max_one_vs_two_first_error <= 1e-5,
            },
        },
        "elapsed_seconds": float(time.perf_counter() - elapsed_start),
        "frames": frame_rows,
    }
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_path}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({"output": str(output_path), "sample_count": len(samples),
                      "mode_summaries": mode_summaries,
                      "zero_visual_round1_max_abs_error": max_equivalence_error,
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    return output_path


def main() -> None:
    args = parse_args()
    if args.fix_controls:
        run_fix_controls(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
