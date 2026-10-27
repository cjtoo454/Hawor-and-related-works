#!/usr/bin/env python3
"""Pretrain the causal trajectory world model from raw HOT3D MANO states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from dual_loop.networks import (  # noqa: E402
    TrajectoryWorldModel,
    WORLD_MODEL_FORMAT_VERSION,
    WORLD_MODEL_VARIANTS,
)
from dual_loop.state import (  # noqa: E402
    STATE_DIM,
    rotation_6d_to_matrix,
    project_state,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw_trajectories"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "world_model_pretrain"
DEFAULT_HISTORY = 16
DEFAULT_HORIZON = 10
STAGE = "world_model_pretrain"
COORDINATE_NORMALIZATION = "first_history_wrist"
LEGACY_LOSS_CONFIG = {
    "name": "gaussian_nll_selected_horizons_v1",
    "horizons": [1, 5, 10],
    "weights": [1.0, 0.5, 0.25],
}
CV_RESIDUAL_LOSS_CONFIG = {
    "name": "cv_residual_balanced_all_steps_v1",
    "translation_weight": 10.0,
    "rotation_weight": 1.0,
    "shape_weight": 0.1,
    "uncertainty_weight": 0.01,
    "huber_beta_translation_shape": 0.01,
    "huber_beta_rotation_matrix": 0.05,
}
RESUME_CRITICAL_KEYS = (
    "stage", "coordinate_normalization", "state_dim", "model_config", "variant",
    "loss_config", "fps",
    "history", "horizon", "steps_per_epoch", "batch_size", "learning_rate",
    "weight_decay", "seed", "train_participants", "calibration_participants",
    "train_sequences", "calibration_sequences", "train_window_count",
    "calibration_window_count", "eval_windows", "device", "output_dir",
    "fast_motion_quantile", "fast_motion_threshold",
    "fast_motion_score", "fast_motion_threshold_source", "split_manifest_sha256",
    "prepared_data_sha256",
)


@dataclass
class SequenceData:
    sequence: str
    participant: str
    states: np.ndarray
    valid: np.ndarray
    timestamps_ns: np.ndarray
    dt_seconds: np.ndarray
    dt_ok: np.ndarray


@dataclass(frozen=True)
class Window:
    sequence_index: int
    start: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the real-trajectory world-model stage")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument(
        "--variant", choices=WORLD_MODEL_VARIANTS, default="legacy",
        help="legacy delta predictor or opt-in constant-velocity residual model",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_windows", type=int, default=512)
    parser.add_argument("--fast_motion_quantile", type=float, default=0.90)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu_threads", type=int, default=1,
                        help="maximum PyTorch CPU threads per training process")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def contiguous_segments(valid: np.ndarray, dt_ok: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs that never bridge invalid frames or time gaps."""
    valid = np.asarray(valid, dtype=bool)
    dt_ok = np.asarray(dt_ok, dtype=bool)
    if valid.ndim != 1 or dt_ok.shape != valid.shape:
        raise ValueError("valid and dt_ok must be same-length vectors")
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        if not is_valid:
            if start is not None:
                segments.append((start, index))
                start = None
        elif start is None:
            start = index
        elif not dt_ok[index]:
            segments.append((start, index))
            start = index
    if start is not None:
        segments.append((start, len(valid)))
    return segments


def load_prepared_dataset(
    data_root: Path,
) -> tuple[list[SequenceData], dict[str, Any], dict[str, str]]:
    manifest_path = data_root / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared split manifest not found: {manifest_path}; run prepare_raw_trajectories.py")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "hot3d_raw_mano_30hz_v1":
        raise ValueError(f"Unsupported raw trajectory schema: {manifest.get('schema')}")
    if manifest.get("state_dim") != STATE_DIM:
        raise ValueError(f"Expected state_dim={STATE_DIM}, got {manifest.get('state_dim')}")
    if manifest.get("fps") != 30.0:
        raise ValueError(f"Expected 30 Hz prepared trajectories, got {manifest.get('fps')}")
    if manifest.get("raw_member") != "mano_hand_pose_trajectory.jsonl":
        raise ValueError("Prepared data provenance does not name the raw MANO trajectory member")
    if manifest.get("handedness") != "right (hand_poses['1'])":
        raise ValueError("Prepared data does not contain the expected right-hand trajectory")
    if manifest.get("state_coordinates") != "HOT3D session world (sensor supervision), meters":
        raise ValueError("Prepared data has an incompatible world-coordinate convention")
    train_list = manifest.get("train_sequences", manifest.get("train", []))
    calibration_list = manifest.get(
        "calibration_sequences", manifest.get("calibration", []))
    if manifest.get("train", train_list) != train_list:
        raise ValueError("train alias differs from train_sequences")
    if manifest.get("calibration", calibration_list) != calibration_list:
        raise ValueError("calibration alias differs from calibration_sequences")
    train = set(train_list)
    calibration = set(calibration_list)
    if not train or not calibration:
        raise ValueError("The manifest must contain both train and calibration sequences")
    train_participants = set(manifest.get("train_participants", []))
    calibration_participants = set(manifest.get("calibration_participants", []))
    if train_participants & calibration_participants:
        raise ValueError("Participant leakage between train and calibration splits")
    val_participants = set(manifest.get("val_participants_checked_from_val_json_only", []))
    if (train_participants | calibration_participants) & val_participants:
        raise ValueError("An official valset participant entered raw training or calibration")

    sequences: list[SequenceData] = []
    prepared_hash = hashlib.sha256()
    prepared_hash.update(manifest_bytes)
    for sequence in sorted(train | calibration):
        path = data_root / f"{sequence}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared sequence missing: {path}")
        file_hash = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                file_hash.update(chunk)
        sequence_hash = file_hash.hexdigest()
        prepared_hash.update(sequence.encode("utf-8"))
        prepared_hash.update(sequence_hash.encode("ascii"))
        with np.load(path, allow_pickle=False) as archive:
            states = np.asarray(archive["states"], dtype=np.float32)
            valid = np.asarray(archive["valid"], dtype=bool)
            timestamps = np.asarray(archive["timestamps_ns"], dtype=np.int64)
            dt_seconds = np.asarray(archive["dt_seconds"], dtype=np.float32)
            dt_ok = np.asarray(archive["dt_ok"], dtype=bool)
            participant = str(np.asarray(archive["participant"]).item())
        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(f"{sequence}: expected states [T,{STATE_DIM}], got {states.shape}")
        if any(array.shape != (len(states),) for array in
               (valid, timestamps, dt_seconds, dt_ok)):
            raise ValueError(f"{sequence}: frame metadata length mismatch")
        if np.any(valid & ~np.isfinite(states).all(axis=1)):
            raise ValueError(f"{sequence}: valid frame has non-finite MANO state")
        if np.any(valid & (timestamps < 0)):
            raise ValueError(f"{sequence}: valid frame has invalid timestamp")
        expected_participant = sequence.split("_", 1)[0]
        if participant != expected_participant:
            raise ValueError(f"{sequence}: participant metadata mismatch ({participant})")
        sequences.append(SequenceData(
            sequence, participant, states, valid, timestamps, dt_seconds, dt_ok))
    fingerprints = {
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "prepared_data_sha256": prepared_hash.hexdigest(),
    }
    return sequences, manifest, fingerprints


def collect_windows(
    sequences: list[SequenceData], sequence_names: set[str], history: int, horizon: int,
) -> list[Window]:
    windows: list[Window] = []
    for sequence_index, sequence in enumerate(sequences):
        if sequence.sequence not in sequence_names:
            continue
        for start, stop in contiguous_segments(sequence.valid, sequence.dt_ok):
            last_start = stop - history - horizon
            windows.extend(Window(sequence_index, frame_start)
                           for frame_start in range(start, last_start + 1))
    return windows


def _stack_windows(
    sequences: list[SequenceData], windows: list[Window], history: int, horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    histories, futures, sequence_indices = [], [], []
    for window in windows:
        sequence = sequences[window.sequence_index]
        boundary = window.start + history
        histories.append(sequence.states[window.start:boundary])
        futures.append(sequence.states[boundary:boundary + horizon])
        sequence_indices.append(window.sequence_index)
    history_tensor = torch.from_numpy(np.stack(histories)).to(device=device)
    future_tensor = torch.from_numpy(np.stack(futures)).to(device=device)
    return history_tensor, future_tensor, np.asarray(sequence_indices, dtype=np.int64)


def _sample_window_indices(
    rng: np.random.Generator, batch_size: int, rank: int, world_size: int,
    window_count: int,
) -> np.ndarray:
    """Draw one deterministic global batch and return this rank's slice."""
    if world_size < 1 or batch_size <= 0 or batch_size % world_size:
        raise ValueError("global batch_size must be positive and divisible by world_size")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    global_indices = rng.integers(0, window_count, size=batch_size)
    local_batch_size = batch_size // world_size
    start = rank * local_batch_size
    return global_indices[start:start + local_batch_size]


def _initialize_distributed(args: argparse.Namespace) -> tuple[int, int, int, torch.device, bool]:
    """Initialize torchrun's process group and select the process-local device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    requested_device = torch.device(args.device)
    if world_size > 1:
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("torchrun requested CUDA DDP but CUDA is unavailable")
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
            backend = "nccl"
        else:
            device = requested_device
            backend = "gloo"
        owns_group = not dist.is_initialized()
        if owns_group:
            dist.init_process_group(backend=backend, init_method="env://")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        device = requested_device
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is unavailable")
            torch.cuda.set_device(device)
        owns_group = False
    if args.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(args.cpu_threads)
    except RuntimeError:
        # An embedding caller may already have started PyTorch's inter-op pool.
        pass
    return rank, local_rank, world_size, device, owns_group


def _sync() -> None:
    if dist.is_initialized():
        dist.barrier()


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mean.shape != target.shape or logvar.shape != target.shape:
        raise ValueError(
            f"NLL tensors must match; mean={mean.shape}, logvar={logvar.shape}, target={target.shape}")
    return 0.5 * ((target - mean).square() * torch.exp(-logvar) + logvar).mean()


def translation_regression_loss(
    prediction: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Direct translation regression, independent of predicted uncertainty."""
    if prediction.shape != target.shape or prediction.shape[-1] != STATE_DIM:
        raise ValueError("translation loss expects matching MANO-state tensors")
    return F.smooth_l1_loss(
        prediction[..., :3], target[..., :3], beta=0.01)


def cv_residual_objective(
    prediction: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced mean-state losses over every rollout step.

    The regression terms drive the predicted state directly. The small NLL
    trains uncertainty from a detached mean so variance cannot attenuate the
    translation objective.
    """
    if prediction.shape != target.shape or logvar.shape != target.shape:
        raise ValueError("CV residual loss tensors must have identical shapes")
    if prediction.ndim != 3 or prediction.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected [batch, horizon, {STATE_DIM}] state tensors")
    predicted_rotations = rotation_6d_to_matrix(
        prediction[..., 3:99].reshape(*prediction.shape[:-1], 16, 6))
    target_rotations = rotation_6d_to_matrix(
        target[..., 3:99].reshape(*target.shape[:-1], 16, 6))
    terms = {
        "translation": translation_regression_loss(prediction, target),
        "rotation": F.smooth_l1_loss(
            predicted_rotations, target_rotations, beta=0.05),
        "shape": F.smooth_l1_loss(
            prediction[..., 99:], target[..., 99:], beta=0.01),
        "uncertainty": gaussian_nll(prediction.detach(), logvar, target),
    }
    loss = (
        CV_RESIDUAL_LOSS_CONFIG["translation_weight"] * terms["translation"]
        + CV_RESIDUAL_LOSS_CONFIG["rotation_weight"] * terms["rotation"]
        + CV_RESIDUAL_LOSS_CONFIG["shape_weight"] * terms["shape"]
        + CV_RESIDUAL_LOSS_CONFIG["uncertainty_weight"] * terms["uncertainty"]
    )
    return loss, terms


def _normalize_prediction_shape(
    prediction: torch.Tensor, logvar: torch.Tensor, horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.ndim == 2:
        prediction = prediction[:, None]
    if logvar.ndim == 2:
        logvar = logvar[:, None]
    if prediction.shape[1] < horizon or logvar.shape[1] < horizon:
        raise ValueError(
            f"World model returned fewer than {horizon} steps: {prediction.shape}, {logvar.shape}")
    return prediction[:, :horizon], logvar[:, :horizon]


def _motion_score(history: torch.Tensor, dt_seconds: torch.Tensor) -> torch.Tensor:
    """Meters/second plus 5 cm per radian/second of mean MANO rotation speed."""
    current = history[:, -1]
    previous = history[:, -2]
    dt = dt_seconds.clamp_min(1e-6)
    translation_speed = torch.linalg.vector_norm(
        current[:, :3] - previous[:, :3], dim=-1) / dt
    current_rotation = rotation_6d_to_matrix(current[:, 3:99].reshape(-1, 16, 6))
    previous_rotation = rotation_6d_to_matrix(previous[:, 3:99].reshape(-1, 16, 6))
    relative = previous_rotation.transpose(-1, -2) @ current_rotation
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1, 1)
    angular_speed = torch.acos(cosine).mean(dim=-1) / dt
    return translation_speed + 0.05 * angular_speed


def _metric_totals() -> dict[str, float]:
    return {"translation_error_sum": 0.0, "rotation_error_sum": 0.0,
            "beta_squared_error_sum": 0.0, "state_nll_sum": 0.0, "count": 0.0}


def _accumulate_metrics(
    totals: dict[str, float], prediction: torch.Tensor, target: torch.Tensor,
    nll: torch.Tensor | None = None,
) -> None:
    translation_error = torch.linalg.vector_norm(prediction[..., :3] - target[..., :3], dim=-1)
    pred_rot = rotation_6d_to_matrix(prediction[..., 3:99].reshape(-1, 16, 6))
    target_rot = rotation_6d_to_matrix(target[..., 3:99].reshape(-1, 16, 6))
    relative = pred_rot.transpose(-1, -2) @ target_rot
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1, 1)
    rotation_error = torch.rad2deg(torch.acos(cosine)).mean(dim=-1)
    beta_squared_error = (prediction[..., 99:] - target[..., 99:]).square().mean(dim=-1)
    totals["translation_error_sum"] += float(translation_error.sum())
    totals["rotation_error_sum"] += float(rotation_error.sum())
    totals["beta_squared_error_sum"] += float(beta_squared_error.sum())
    if nll is not None:
        totals["state_nll_sum"] += float(nll.sum())
    totals["count"] += float(prediction.shape[0])


def _finalize_metrics(totals: dict[str, float]) -> dict[str, float | int | None]:
    count = totals["count"]
    if count == 0:
        return {
            "sample_count": 0,
            "translation_error_m": None,
            "rotation_error_deg": None,
            "beta_rmse": None,
            "gaussian_nll": None,
        }
    result = {
        "translation_error_m": totals["translation_error_sum"] / count,
        "rotation_error_deg": totals["rotation_error_sum"] / count,
        "beta_rmse": float(np.sqrt(totals["beta_squared_error_sum"] / count)),
        "sample_count": int(count),
    }
    result["gaussian_nll"] = (
        totals["state_nll_sum"] / count if totals["state_nll_sum"] else None)
    return result


def _subset_eval_windows(windows: list[Window], limit: int) -> list[Window]:
    if limit <= 0 or len(windows) <= limit:
        return windows
    indices = np.rint(np.linspace(0, len(windows) - 1, limit)).astype(np.int64)
    return [windows[index] for index in indices]


def motion_scores(
    sequences: list[SequenceData], windows: list[Window], history_length: int,
    device: torch.device,
) -> np.ndarray:
    scores: list[float] = []
    for offset in range(0, len(windows), 256):
        batch = windows[offset:offset + 256]
        histories, _, _ = _stack_windows(
            sequences, batch, history_length, 1, device)
        dt_values = []
        for window in batch:
            sequence = sequences[window.sequence_index]
            dt_values.append(float(sequence.dt_seconds[window.start + history_length - 1]))
        dt = torch.tensor(dt_values, dtype=histories.dtype, device=device)
        scores.extend(_motion_score(histories, dt).detach().cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


@torch.no_grad()
def evaluate(
    model: TrajectoryWorldModel, sequences: list[SequenceData], windows: list[Window],
    history_length: int, horizon: int, device: torch.device, fast_threshold: float,
    batch_size: int = 128, max_windows: int = 512,
) -> dict[str, Any]:
    model.eval()
    windows = _subset_eval_windows(windows, max_windows)
    methods = ("last", "constant_velocity", "recursive_constant_velocity", "learned")
    all_totals = {
        method: {step: _metric_totals() for step in (1, 5, 10) if step <= horizon}
        for method in methods
    }
    fast_totals = {
        method: {step: _metric_totals() for step in (1, 5, 10) if step <= horizon}
        for method in methods
    }
    fast_count = 0
    for offset in range(0, len(windows), batch_size):
        batch_windows = windows[offset:offset + batch_size]
        histories, target, sequence_indices = _stack_windows(
            sequences, batch_windows, history_length, horizon, device)
        predicted, predicted_logvar = model(histories, steps=horizon)
        predicted, predicted_logvar = _normalize_prediction_shape(
            predicted, predicted_logvar, horizon)
        previous = histories[:, -2]
        last = histories[:, -1]
        cv_steps = torch.arange(1, horizon + 1, dtype=last.dtype, device=device)[None, :, None]
        constant_velocity = project_state(
            last[:, None] + cv_steps * (last - previous)[:, None])
        recursive_history = histories
        recursive_predictions = []
        history_limit = histories.shape[1]
        for _ in range(horizon):
            recursive_last = recursive_history[:, -1]
            recursive_previous = (
                recursive_history[:, -2]
                if recursive_history.shape[1] >= 2 else recursive_last)
            recursive_prediction = project_state(
                recursive_last + (recursive_last - recursive_previous))
            recursive_predictions.append(recursive_prediction)
            recursive_history = torch.cat(
                (recursive_history, recursive_prediction[:, None]), dim=1
            )[:, -history_limit:]
        recursive_constant_velocity = torch.stack(recursive_predictions, dim=1)
        last_state = project_state(last)[:, None].expand(-1, horizon, -1)

        dt_values = []
        for window in batch_windows:
            sequence = sequences[window.sequence_index]
            dt_values.append(float(sequence.dt_seconds[window.start + history_length - 1]))
        motion = _motion_score(
            histories, torch.tensor(dt_values, dtype=histories.dtype, device=device))
        fast_mask = motion >= fast_threshold
        fast_count += int(fast_mask.sum())
        horizon_steps = [step for step in (1, 5, 10) if step <= horizon]
        for step in horizon_steps:
            target_state = target[:, step - 1]
            baseline_predictions = {
                "last": last_state[:, step - 1],
                "constant_velocity": constant_velocity[:, step - 1],
                "recursive_constant_velocity": recursive_constant_velocity[:, step - 1],
                "learned": predicted[:, step - 1],
            }
            for method, prediction in baseline_predictions.items():
                nll_values = None
                if method == "learned":
                    nll_values = 0.5 * (
                        (target_state - prediction).square()
                        * torch.exp(-predicted_logvar[:, step - 1])
                        + predicted_logvar[:, step - 1]
                    ).mean(dim=-1)
                _accumulate_metrics(
                    all_totals[method][step], prediction, target_state, nll_values)
                if bool(fast_mask.any()):
                    _accumulate_metrics(
                        fast_totals[method][step], prediction[fast_mask],
                        target_state[fast_mask],
                        nll_values[fast_mask] if nll_values is not None else None)

    return {
        "window_count": len(windows),
        "fast_window_count": fast_count,
        "fast_motion_threshold": float(fast_threshold),
        "all_motion": {
            method: {f"horizon_{step}": _finalize_metrics(totals)
                    for step, totals in by_step.items()}
            for method, by_step in all_totals.items()
        },
        "fast_motion": {
            method: {f"horizon_{step}": _finalize_metrics(totals)
                    for step, totals in by_step.items()}
            for method, by_step in fast_totals.items()
        },
    }


def _checkpoint(
    model: TrajectoryWorldModel, optimizer: torch.optim.Optimizer, args: argparse.Namespace,
    epoch: int, global_step: int, run_metadata: dict[str, Any],
    rng: np.random.Generator, best_metric: float,
) -> dict[str, Any]:
    return {
        "world_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": {
            "numpy_generator": rng.bit_generator.state,
            "numpy_global": np.random.get_state(),
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "stage": STAGE,
        "coordinate_normalization": COORDINATE_NORMALIZATION,
        "state_dim": STATE_DIM,
        "variant": args.variant,
        "world_model_format_version": WORLD_MODEL_FORMAT_VERSION[args.variant],
        "model_config": {
            "state_dim": STATE_DIM,
            "hidden_dim": args.hidden_dim,
            "variant": args.variant,
            "format_version": WORLD_MODEL_FORMAT_VERSION[args.variant],
        },
        "epoch": epoch,
        "global_step": global_step,
        "seed": args.seed,
        "best_metric": best_metric,
        "run": run_metadata,
        "device_topology": run_metadata.get("device_topology"),
        "topology_migration": run_metadata.get("topology_migration"),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _validate_resume_metadata(
    previous: dict[str, Any], current: dict[str, Any], source: str,
) -> None:
    mismatches = [
        key for key in RESUME_CRITICAL_KEYS
        if _resume_metadata_value(previous, key) != _resume_metadata_value(current, key)
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={_resume_metadata_value(previous, key)!r}, "
            f"current={_resume_metadata_value(current, key)!r}"
            for key in mismatches)
        raise ValueError(f"Resume {source} does not match this run ({details})")


def _resume_metadata_value(metadata: dict[str, Any], key: str) -> Any:
    """Normalize pre-versioned checkpoints as the unchanged legacy variant."""
    variant = metadata.get("variant", "legacy")
    if key == "variant":
        return variant
    if key == "model_config":
        raw_config = metadata.get(key, {})
        if not isinstance(raw_config, dict):
            return raw_config
        config = dict(raw_config)
        config.setdefault("variant", variant)
        config.setdefault(
            "format_version", WORLD_MODEL_FORMAT_VERSION.get(variant, -1))
        return config
    if key == "loss_config" and key not in metadata and variant == "legacy":
        return LEGACY_LOSS_CONFIG
    return metadata.get(key)


def _restore_rng_state(
    checkpoint: dict[str, Any], rng: np.random.Generator, *, rank: int = 0,
    seed: int = 7, device: torch.device | None = None,
) -> None:
    state = checkpoint.get("rng_state")
    if not isinstance(state, dict):
        raise ValueError("Resume checkpoint has no complete RNG state")
    required = {"numpy_generator", "numpy_global", "python", "torch_cpu", "torch_cuda"}
    if not required.issubset(state):
        raise ValueError(f"Resume checkpoint RNG state is missing {sorted(required - state.keys())}")
    rng.bit_generator.state = state["numpy_generator"]
    if rank == 0:
        np.random.set_state(state["numpy_global"])
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"].cpu())
    else:
        # Sampling uses the shared Generator state above; process-local model
        # randomness remains distinct after a single-GPU topology migration.
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        target_index = (device.index if device is not None and device.index is not None
                        else torch.cuda.current_device())
        source_index = target_index if target_index < len(cuda_states) else (0 if rank == 0 else None)
        if source_index is not None and cuda_states:
            torch.cuda.set_rng_state(cuda_states[source_index].cpu(), device=device)
        else:
            torch.cuda.manual_seed(seed + rank)


def train(
    args: argparse.Namespace,
    distributed: tuple[int, int, int, torch.device, bool] | None = None,
) -> dict[str, Any] | None:
    if distributed is None:
        distributed = _initialize_distributed(args)
    rank, local_rank, world_size, device, _owns_group = distributed
    is_main = rank == 0
    distributed_enabled = world_size > 1
    if args.epochs <= 0 or args.steps_per_epoch <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, steps_per_epoch, and batch_size must be positive")
    if args.batch_size % world_size:
        raise ValueError(
            f"global batch_size={args.batch_size} must be divisible by world_size={world_size}")
    if args.history < 2 or args.horizon < 10:
        raise ValueError("history must be >=2 and horizon must be >=10 for 1/5/10-step metrics")
    if not 0.5 <= args.fast_motion_quantile < 1.0:
        raise ValueError("fast_motion_quantile must be in [0.5,1)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError(
            f"Output directory is not empty: {args.output_dir}; use its matching --resume checkpoint")
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    local_batch_size = args.batch_size // world_size
    sequences, manifest, fingerprints = load_prepared_dataset(args.data_root)
    train_names = set(manifest["train_sequences"])
    calibration_names = set(manifest["calibration_sequences"])
    train_windows = collect_windows(sequences, train_names, args.history, args.horizon)
    calibration_windows = collect_windows(
        sequences, calibration_names, args.history, args.horizon)
    if not train_windows or not calibration_windows:
        raise ValueError(
            f"Need train and calibration windows at history={args.history}, "
            f"horizon={args.horizon}; got {len(train_windows)} and {len(calibration_windows)}")

    calibration_scores = motion_scores(
        sequences, calibration_windows, args.history, device)
    fast_threshold = float(np.quantile(calibration_scores, args.fast_motion_quantile))
    model_format_version = WORLD_MODEL_FORMAT_VERSION[args.variant]
    loss_config = (
        CV_RESIDUAL_LOSS_CONFIG if args.variant == "cv_residual"
        else LEGACY_LOSS_CONFIG)

    run_metadata = {
        "stage": STAGE,
        "coordinate_normalization": COORDINATE_NORMALIZATION,
        "state_dim": STATE_DIM,
        "variant": args.variant,
        "loss_config": loss_config,
        "fps": float(manifest["fps"]),
        "state_coordinates": manifest["state_coordinates"],
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "variant": args.variant,
            "format_version": model_format_version,
        },
        "history": args.history,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "local_batch_size": local_batch_size,
        "world_size": world_size,
        "device_topology": {
            "world_size": world_size,
            "requested_device": str(args.device),
            "backend": ("nccl" if device.type == "cuda" and distributed_enabled
                        else "gloo" if distributed_enabled else None),
        },
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "cpu_threads": args.cpu_threads,
        "seed": args.seed,
        # Keep the requested rank-zero device string stable for strict resume
        # provenance; the actual per-rank device is recorded separately below.
        "device": str(args.device),
        "eval_windows": args.eval_windows,
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "split_manifest": str((args.data_root / "split_manifest.json").resolve()),
        **fingerprints,
        "train_participants": manifest["train_participants"],
        "calibration_participants": manifest["calibration_participants"],
        "train_sequences": sorted(train_names),
        "calibration_sequences": sorted(calibration_names),
        "train_window_count": len(train_windows),
        "calibration_window_count": len(calibration_windows),
        "fast_motion_quantile": args.fast_motion_quantile,
        "fast_motion_threshold": fast_threshold,
        "fast_motion_score": "translation_speed_mps + 0.05 * mean_16_rotation_speed_radps",
        "fast_motion_threshold_source": "calibration participants only",
        "valset_participants_opened": False,
        "loss": loss_config["name"],
        "pretraining_source": "raw mano_hand_pose_trajectory.jsonl only",
    }
    run_metadata["topology_migration"] = None

    model = TrajectoryWorldModel(
        STATE_DIM, args.hidden_dim, variant=args.variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    checkpoint: dict[str, Any] | None = None
    previous_topology: dict[str, Any] | None = None
    topology_migration: dict[str, Any] | None = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != STAGE:
            raise ValueError(f"Cannot resume stage {checkpoint.get('stage')!r}; expected {STAGE!r}")
        if checkpoint.get("coordinate_normalization") != COORDINATE_NORMALIZATION:
            raise ValueError("Resume checkpoint has incompatible coordinate normalization")
        saved_model_config = dict(checkpoint.get("model_config", {}))
        saved_variant = checkpoint.get(
            "variant", saved_model_config.get("variant", "legacy"))
        saved_format_version = checkpoint.get(
            "world_model_format_version",
            saved_model_config.get(
                "format_version", WORLD_MODEL_FORMAT_VERSION.get(saved_variant, -1)))
        normalized_model_config = {
            "state_dim": saved_model_config.get("state_dim", STATE_DIM),
            "hidden_dim": saved_model_config.get("hidden_dim"),
            "variant": saved_model_config.get("variant", saved_variant),
            "format_version": saved_model_config.get(
                "format_version", saved_format_version),
        }
        if (saved_variant != args.variant
                or saved_format_version != model_format_version
                or normalized_model_config != {
                    "state_dim": STATE_DIM,
                    "hidden_dim": args.hidden_dim,
                    "variant": args.variant,
                    "format_version": model_format_version,
                }):
            raise ValueError("Resume checkpoint model configuration does not match this run")
        previous_run = checkpoint.get("run")
        if not isinstance(previous_run, dict):
            raise ValueError("Resume checkpoint has no run provenance metadata")
        previous_topology = previous_run.get("device_topology") or {
            "world_size": previous_run.get("world_size", 1),
            "requested_device": previous_run.get("device", args.device),
            "backend": None,
        }
        if previous_topology.get("world_size", 1) != world_size:
            topology_migration = {
                "from": previous_topology,
                "to": run_metadata["device_topology"],
            }
        else:
            topology_migration = previous_run.get("topology_migration")
        run_metadata["topology_migration"] = topology_migration
        _validate_resume_metadata(previous_run, run_metadata, "checkpoint provenance")
        if "optimizer" not in checkpoint:
            raise ValueError("Resume checkpoint has no optimizer state")
        model.load_state_dict(checkpoint["world_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint.get("best_metric", float("inf")))
        _restore_rng_state(checkpoint, rng, rank=rank, seed=args.seed, device=device)

    run_path = args.output_dir / "run.json"
    if args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()) and not run_path.is_file():
        raise ValueError("Cannot resume into a non-empty output directory without its run.json")
    if args.resume and run_path.is_file():
        existing_run = json.loads(run_path.read_text(encoding="utf-8"))
        _validate_resume_metadata(existing_run, run_metadata, "run.json provenance")
    _sync()
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_path, run_metadata)
    _sync()

    metrics_path = args.output_dir / "metrics.jsonl"
    best_path = args.output_dir / "best.pt"
    training_model: torch.nn.Module = model
    if distributed_enabled:
        training_model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )

    if args.variant == "cv_residual" and not args.resume:
        _sync()
        if is_main:
            initial_metrics = evaluate(
                model, sequences, calibration_windows, args.history, args.horizon,
                device, fast_threshold, max_windows=args.eval_windows)
            best_metric = initial_metrics["all_motion"]["learned"]["horizon_10"][
                "translation_error_m"]
            initial_row = {
                "epoch": 0, "global_step": 0,
                "phase": "zero_initialized_cv_baseline", "train": None,
                "calibration": initial_metrics,
                "checkpoint_selection_metric": (
                    "calibration learned horizon_10 translation_error_m"),
                "checkpoint_selection_value": best_metric,
            }
            with metrics_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(initial_row, sort_keys=True) + "\n")
            initial_checkpoint = _checkpoint(
                model, optimizer, args, 0, 0, run_metadata, rng, best_metric)
            _atomic_torch_save(initial_checkpoint, args.output_dir / "last.pt")
            _atomic_torch_save(initial_checkpoint, best_path)
            print(
                f"epoch=0 baseline_cal_h10_m={best_metric:.5f} "
                f"fast_n={initial_metrics['fast_window_count']}", flush=True)
        metric_value = torch.tensor([best_metric], dtype=torch.float64, device=device)
        if distributed_enabled:
            dist.broadcast(metric_value, src=0)
        best_metric = float(metric_value.item())
        _sync()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        recent_loss_sum = 0.0
        recent_steps = 0
        horizon_nll = {1: 0.0, 5: 0.0, 10: 0.0}
        cv_loss_totals = {key: 0.0 for key in
                          ("translation", "rotation", "shape", "uncertainty")}
        for _ in range(args.steps_per_epoch):
            chosen = [train_windows[int(index)] for index in _sample_window_indices(
                rng, args.batch_size, rank, world_size, len(train_windows))]
            histories, targets, _ = _stack_windows(
                sequences, chosen, args.history, args.horizon, device)
            prediction, logvar = training_model(histories, steps=args.horizon)
            prediction, logvar = _normalize_prediction_shape(
                prediction, logvar, args.horizon)
            terms = {
                step: gaussian_nll(
                    prediction[:, step - 1], logvar[:, step - 1], targets[:, step - 1])
                for step in (1, 5, 10)
            }
            if args.variant == "cv_residual":
                loss, objective_terms = cv_residual_objective(
                    prediction, logvar, targets)
                for key, value in objective_terms.items():
                    cv_loss_totals[key] += float(value.detach())
            else:
                loss = terms[1] + 0.5 * terms[5] + 0.25 * terms[10]
            if not torch.isfinite(prediction).all() or not torch.isfinite(logvar).all():
                raise FloatingPointError(f"Non-finite world-model output at global_step={global_step + 1}")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite world-model loss at global_step={global_step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at global_step={global_step + 1}")
            optimizer.step()
            global_step += 1
            loss_sum += float(loss.detach())
            recent_loss_sum += float(loss.detach())
            recent_steps += 1
            for step in horizon_nll:
                horizon_nll[step] += float(terms[step].detach())
            if global_step % 20 == 0:
                recent_loss = torch.tensor(
                    recent_loss_sum, dtype=torch.float64, device=device)
                if distributed_enabled:
                    dist.all_reduce(recent_loss, op=dist.ReduceOp.SUM)
                if is_main:
                    print(
                        f"epoch={epoch} global_step={global_step} "
                        f"train_loss_{recent_steps}="
                        f"{recent_loss.item() / (recent_steps * world_size):.6f}",
                        flush=True,
                    )
                recent_loss_sum = 0.0
                recent_steps = 0

        epoch_values = [loss_sum, *horizon_nll.values(), *cv_loss_totals.values()]
        epoch_tensor = torch.tensor(epoch_values, dtype=torch.float64, device=device)
        if distributed_enabled:
            dist.all_reduce(epoch_tensor, op=dist.ReduceOp.SUM)
            epoch_tensor /= world_size
        epoch_values = epoch_tensor.cpu().tolist()
        loss_sum = epoch_values[0]
        horizon_nll = dict(zip(horizon_nll, epoch_values[1:4]))
        cv_loss_totals = dict(zip(cv_loss_totals, epoch_values[4:]))
        train_metrics = {
            "loss": loss_sum / args.steps_per_epoch,
            **{f"gaussian_nll_{step}": value / args.steps_per_epoch
               for step, value in horizon_nll.items()},
        }
        if args.variant == "cv_residual":
            train_metrics.update({
                f"{key}_objective": value / args.steps_per_epoch
                for key, value in cv_loss_totals.items()
            })
        _sync()
        if is_main:
            # Evaluate the unwrapped module: DDP forward can broadcast buffers
            # and wait for ranks that are intentionally idle during evaluation.
            calibration_metrics = evaluate(
                model, sequences, calibration_windows, args.history, args.horizon,
                device, fast_threshold, max_windows=args.eval_windows)
            selection_metric = calibration_metrics["all_motion"]["learned"]["horizon_10"][
                "translation_error_m"]
            row = {
                "epoch": epoch, "global_step": global_step,
                "train": train_metrics, "calibration": calibration_metrics,
                "checkpoint_selection_metric": (
                    "calibration learned horizon_10 translation_error_m"),
                "checkpoint_selection_value": selection_metric,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            next_best = min(best_metric, selection_metric)
            improved = selection_metric < best_metric
        else:
            selection_metric = next_best = 0.0
            improved = False
        metric_packet = torch.tensor(
            [selection_metric, next_best, float(improved)],
            dtype=torch.float64, device=device)
        if distributed_enabled:
            dist.broadcast(metric_packet, src=0)
        selection_metric, best_metric, improved = metric_packet.tolist()
        improved = bool(improved)
        if is_main:
            checkpoint = _checkpoint(
                model, optimizer, args, epoch, global_step, run_metadata, rng, best_metric)
            _atomic_torch_save(checkpoint, args.output_dir / "last.pt")
            if improved:
                _atomic_torch_save(checkpoint, best_path)
            print(
                f"epoch={epoch}/{args.epochs} train_loss={train_metrics['loss']:.6f} "
                f"cal_h10_m={selection_metric:.5f} "
                f"fast_n={calibration_metrics['fast_window_count']} best={best_metric:.5f}")
        _sync()
    if not is_main:
        return None
    return {"run": run_metadata, "best_metric": best_metric,
            "last_checkpoint": str((args.output_dir / "last.pt").resolve()),
            "best_checkpoint": str(best_path.resolve())}


def main() -> None:
    args = parse_args()
    distributed = _initialize_distributed(args)
    rank, _local_rank, _world_size, _device, owns_group = distributed
    try:
        result = train(args, distributed)
        if rank == 0:
            print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if owns_group and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
