#!/usr/bin/env python3
"""Train only HandLoopVisual for the fixed-TWM intra-frame ablation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRY = ROOT / "thirdparty" / "try"
AB = ROOT / "thirdparty" / "ablation"
sys.path[:0] = [str(ROOT), str(TRY), str(TRY / "scripts"), str(AB / "twm_only"), str(HERE)]

from dual_loop.handloop_frontend import project_mano16  # noqa: E402
from dual_loop.handloop_probability import log_prob  # noqa: E402
from dual_loop.handloop_runtime import build_system, move_batch  # noqa: E402
from dual_loop.handloop_state import legacy_to_state, state_to_legacy, to_local, to_world  # noqa: E402
from dual_loop.handloop_data import crop_affine  # noqa: E402
from lib.datasets.track_dataset import TrackDatasetEval  # noqa: E402
from prepare_training import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_PROTOCOL,
    DEFAULT_SPLIT,
    DEFAULT_VIDEO_ROOT,
    DEFAULT_EXPORT_SPLIT,
    PreparedSequence,
    calibration_samples,
    convert_raw_history_states,
    context_start_indices,
    history_indices,
    load_sequences,
    observation_record,
)
from runtime import raw_history_batch  # noqa: E402
import run_twm_only as twm_only  # noqa: E402


INTRA_ARCHITECTURE = "handloop_intra_v1"
INTRA_FORMAT = "handloop_intra_training"
INTRA_FORMAT_VERSION = 1
CALIBRATION_SELECTION_RULE = "middle gt_valid & visual_valid frame per calibration sequence"
TRAIN_SAMPLING_RULE = (
    "context probability weighted by GT-valid train-frame count; sample current frames uniformly within context")
DEFAULT_TWM = AB / "training_v2" / "checkpoints" / "latest.pt"
DEFAULT_HAWOR = ROOT / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"
EXPECTED_TWM_SHA256 = "26078e9b5d587f38130d51316dad90b598d4a426dbc312f8db2360eb0d9a0ab2"
EXPECTED_HAWOR_SHA256 = "4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_parameter_sha256(module: torch.nn.Module) -> str:
    """Hash frozen in-memory parameters without copying a whole model at once."""
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        value = parameter.detach().contiguous().view(torch.uint8).cpu()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twm-checkpoint", type=Path, default=DEFAULT_TWM)
    parser.add_argument("--hawor-checkpoint", type=Path, default=DEFAULT_HAWOR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--export-split", type=Path, default=DEFAULT_EXPORT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=1e-4)
    parser.add_argument("--joint-scale", type=float, default=0.05)
    parser.add_argument("--calibration-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--skip-image-check", action="store_true",
                        help="Skip the startup file-existence check for the already verified RGB paths")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_twm(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    actual_hash = sha256_file(path)
    if actual_hash != expected_sha256:
        raise ValueError(f"Base TWM checkpoint hash differs from the frozen protocol: {actual_hash}")
    checkpoint = twm_only.load_checkpoint_restricted(path)
    step = twm_only.validate_trained_checkpoint(checkpoint)
    if (checkpoint.get("architecture") != "handloop_v2"
            or checkpoint.get("stage") != "motion_pretrain"
            or step != 5000):
        raise ValueError("The intra-frame run requires the fixed 5000-step HandLoop motion_pretrain checkpoint")
    config, state = twm_only.twm_state_from_checkpoint(checkpoint)
    config = copy.deepcopy(config)
    config["rounds"] = 2
    config["reobserve"] = True
    if not isinstance(config.get("visual"), dict) or not isinstance(config.get("twm"), dict):
        raise ValueError("The base checkpoint must contain the original HandLoop visual/TWM config")
    return config, state


def freeze_except_visual(system: torch.nn.Module) -> None:
    """Freeze HaWoR, MANO geometry, and TWM while leaving visual trainable."""
    system.requires_grad_(False)
    system.observer.visual.requires_grad_(True)
    if system.observer.config.rounds != 2 or not system.observer.config.reobserve:
        raise ValueError("Intra-frame training requires exactly two reobserving rounds")
    trainable = {id(parameter) for parameter in system.parameters() if parameter.requires_grad}
    visual = {id(parameter) for parameter in system.observer.visual.parameters()}
    if trainable != visual:
        raise RuntimeError("Only HandLoopObserver.visual parameters may be trainable")


def build_live_batch(
    system: torch.nn.Module,
    samples: list[tuple[PreparedSequence, int]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[dict[str, Any]]]:
    """Run frozen HaWoR on inference-aligned contexts and build visual inputs.

    The returned model-input dictionary is built only from detector crops,
    cameras, raw HaWoR state, and earlier raw HaWoR history. Ground-truth arrays
    are not read here.
    """
    if not samples:
        raise ValueError("Cannot build a visual batch with no samples")
    records = [observation_record(sequence, frame) for sequence, frame in samples]
    contexts = []
    context_keys: dict[tuple[str, tuple[int, ...]], int] = {}
    sample_context_rows: list[int] = []
    for sequence, record in zip((sample[0] for sample in samples), records):
        key = (sequence.sequence, tuple(int(index) for index in record["context_indices"]))
        context_index = context_keys.get(key)
        if context_index is not None:
            sample_context_rows.append(context_index)
            continue
        dataset = TrackDatasetEval(
            record["image_paths"].tolist(),
            record["boxes"].copy(),
            img_focal=record["focal_length"],
            img_center=record["image_size_wh"] / 2,
            normalization=True,
            dilate=1.2,
        )
        items = [dataset[index] for index in range(len(dataset))]
        contexts.append({
            key: torch.stack([item[key] for item in items])
            for key in ("img", "center", "scale", "img_focal", "img_center")
        })
        context_index = len(contexts) - 1
        context_keys[key] = context_index
        sample_context_rows.append(context_index)
        contexts[-1]["_items"] = items

    hawor_batch = {
        key: torch.stack([context[key] for context in contexts]).to(device)
        for key in ("img", "center", "scale", "img_focal", "img_center")
    }
    with torch.no_grad():
        frontend = system.hawor(hawor_batch)

    row = torch.arange(len(samples), dtype=torch.long, device=device)
    context_rows = torch.as_tensor(sample_context_rows, dtype=torch.long, device=device)
    offsets = torch.as_tensor([record["current_offset"] for record in records], device=device)
    rotation_w2c = torch.as_tensor(
        np.stack([record["rotation_w2c"] for record in records]), dtype=torch.float32, device=device)
    translation_w2c = torch.as_tensor(
        np.stack([record["translation_w2c"] for record in records]), dtype=torch.float32, device=device)
    rotation_c2w = rotation_w2c.transpose(-1, -2)
    translation_c2w = -(rotation_c2w @ translation_w2c.unsqueeze(-1)).squeeze(-1)
    raw_state = to_world(frontend["state_camera"][context_rows, offsets], rotation_c2w, translation_c2w)
    patches = frontend["patches"][context_rows, offsets]
    betas = frontend["betas"][context_rows, offsets]

    raw_available = torch.as_tensor(
        np.repeat(np.asarray([record["raw_available"] for record in records])[:, None], 17, axis=1),
        dtype=torch.bool, device=device)
    image_available = torch.as_tensor(
        [record["image_available"] for record in records], dtype=torch.bool, device=device)
    handedness = torch.as_tensor(
        [record["handedness"] for record in records], dtype=torch.long, device=device)
    focal = torch.as_tensor(
        [record["focal_length"] for record in records], dtype=torch.float32, device=device)
    principal = torch.as_tensor(
        np.stack([record["image_size_wh"] / 2 for record in records]), dtype=torch.float32, device=device)
    pixel_affines = []
    for context_index, record in zip(sample_context_rows, records):
        current = contexts[context_index]["_items"][record["current_offset"]]
        pixel_affines.append(
            crop_affine(current["center"].numpy(), float(current["scale"])))
    pixel_to_patch = torch.as_tensor(
        np.stack(pixel_affines), dtype=torch.float32, device=device)

    history_args = build_raw_history_args(
        samples, raw_state, raw_available, device,
        history_length=int(system.observer.config.twm.history_length))
    model_inputs = {
        "patches": patches.detach(),
        "raw_state": raw_state.detach(),
        "raw_available": raw_available,
        "image_available": image_available,
        "handedness": handedness,
        "rotation_w2c": rotation_w2c,
        "translation_w2c": translation_w2c,
        "projector_args": (betas.detach(), focal, principal, pixel_to_patch),
        **history_args,
    }
    return model_inputs, betas.detach(), records


def build_raw_history_args(
    samples: list[tuple[PreparedSequence, int]],
    current_raw: torch.Tensor,
    current_available: torch.Tensor,
    device: torch.device,
    history_length: int = 30,
) -> dict[str, torch.Tensor]:
    """Pack prior raw states and the current raw state with the runtime helper."""
    batch_size = len(samples)
    if current_raw.shape != (batch_size, 99) or current_available.shape != (batch_size, 17):
        raise ValueError("Current raw observations have incompatible batch shapes")
    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    width = history_length + 1
    timeline_states = np.zeros((batch_size, width, 99), dtype=np.float32)
    timeline_available = np.zeros((batch_size, width, 17), dtype=bool)
    timeline_times = np.zeros((batch_size, width), dtype=np.float64)
    timeline_segments = np.zeros((batch_size, width), dtype=np.int64)
    timeline_hands = np.ones((batch_size, width), dtype=np.int64)
    current_ids = []

    for row, (sequence, frame_index) in enumerate(samples):
        if sequence.raw_world is None:
            raise ValueError(f"{sequence.sequence}: raw history has not been converted")
        prior_ids = history_indices(sequence, frame_index, history_length)
        padding = history_length - len(prior_ids)
        current_time = float(sequence.timestamp_sec[frame_index])
        first_real_time = (float(sequence.timestamp_sec[prior_ids[0]])
                           if prior_ids else current_time)
        for offset in range(padding):
            slot_time = first_real_time - (padding - offset) / 30.0
            timeline_times[row, offset] = slot_time
            # Each padding slot starts a reset. The first actual raw row resets
            # once more, so the runtime helper returns history_mask=False here.
            timeline_segments[row, offset] = -((row + 1) * 1000 + offset + 1)
        if prior_ids:
            start = padding
            stop = history_length
            timeline_states[row, start:stop] = sequence.raw_world[prior_ids]
            timeline_available[row, start:stop] = np.repeat(
                sequence.visual_valid[np.asarray(prior_ids), None], 17, axis=1)
            timeline_times[row, start:stop] = sequence.timestamp_sec[prior_ids]
            timeline_segments[row, start:stop] = row + 1
        timeline_states[row, history_length] = current_raw[row].detach().cpu().numpy()
        timeline_available[row, history_length] = current_available[row].detach().cpu().numpy()
        timeline_times[row, history_length] = current_time
        timeline_segments[row, history_length] = row + 1
        current_ids.append(row * width + history_length)

    raw_world = torch.as_tensor(timeline_states.reshape(-1, 99), dtype=torch.float32, device=device)
    raw_available = torch.as_tensor(timeline_available.reshape(-1, 17), dtype=torch.bool, device=device)
    timestamp = torch.as_tensor(timeline_times.reshape(-1), dtype=torch.float64, device=device)
    segment = torch.as_tensor(timeline_segments.reshape(-1), dtype=torch.long, device=device)
    handedness = torch.as_tensor(timeline_hands.reshape(-1), dtype=torch.long, device=device)
    current_indices = torch.as_tensor(current_ids, dtype=torch.long, device=device)
    history = raw_history_batch(
        raw_world, raw_available, timestamp, segment, handedness,
        current_indices, history_length)
    return history


def observer_forward(system: torch.nn.Module, model_inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Call forward_step without ever adding a target field to its inputs."""
    beta, focal, principal, pixel_to_patch = model_inputs.pop("projector_args")
    rotation_w2c = model_inputs["rotation_w2c"]
    translation_w2c = model_inputs["translation_w2c"]

    def projector(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return project_mano16(
            state, beta, system.geometry, rotation_w2c, translation_w2c,
            focal, principal, pixel_to_patch)

    return system.observer.forward_step(projector=projector, **model_inputs)


def visual_loss(
    system: torch.nn.Module,
    output: dict[str, Any],
    target_legacy: torch.Tensor,
    beta: torch.Tensor,
    *,
    state_weight: float,
    joint_scale: float,
) -> dict[str, torch.Tensor]:
    """Supervise the final TWM state with MANO joint error and small current NLL."""
    if not 0.0 < joint_scale:
        raise ValueError("joint_scale must be positive")
    geometry = system.geometry
    with torch.no_grad():
        target_state = legacy_to_state(target_legacy, geometry)
        target_joints = geometry.joints(target_legacy)
    target_local = to_local(target_state, output["anchor_rotation"], output["anchor_position"])
    state_nll = -log_prob(output["current"], target_local).mean()

    predicted_legacy = state_to_legacy(output["state"], beta, geometry)
    predicted_joints = geometry.joints(predicted_legacy)
    valid = torch.isfinite(target_joints).all(dim=-1)
    if not valid.any():
        raise ValueError("Current visual batch has no finite GT MANO joints")
    joint_m = (predicted_joints - target_joints).norm(dim=-1)[valid].mean()
    total = joint_m / joint_scale + float(state_weight) * state_nll
    return {"loss": total, "joint_m": joint_m.detach(), "state_nll": state_nll.detach()}


def _labels_for_samples(
    samples: list[tuple[PreparedSequence, int]],
    device: torch.device,
) -> torch.Tensor:
    """Fetch GT only for loss computation, after model inputs are assembled."""
    labels = np.stack([sequence.target_legacy[index] for sequence, index in samples])
    return torch.as_tensor(labels, dtype=torch.float32, device=device)


def train_batch(
    system: torch.nn.Module,
    samples: list[tuple[PreparedSequence, int]],
    device: torch.device,
    *,
    state_weight: float,
    joint_scale: float,
) -> dict[str, torch.Tensor]:
    model_inputs, beta, _records = build_live_batch(system, samples, device)
    output = observer_forward(system, model_inputs)
    # The target is fetched only after the forward inputs have been consumed.
    target = _labels_for_samples(samples, device)
    return visual_loss(
        system, output, target, beta,
        state_weight=state_weight, joint_scale=joint_scale)


def _calibrate(
    system: torch.nn.Module,
    samples: list[tuple[PreparedSequence, int]],
    batch_size: int,
    device: torch.device,
    *,
    state_weight: float,
    joint_scale: float,
) -> dict[str, float]:
    system.eval()
    totals = {"loss": 0.0, "joint_m": 0.0, "state_nll": 0.0}
    count = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            result = train_batch(
                system, batch, device,
                state_weight=state_weight, joint_scale=joint_scale)
            require_finite_metrics(result, f"calibration samples {_sample_manifest(batch)}")
            size = len(batch)
            for key in totals:
                totals[key] += float(result[key].detach().cpu()) * size
            count += size
    if count == 0:
        raise ValueError("Calibration has no selected samples")
    system.observer.visual.train()
    metrics = {key: value / count for key, value in totals.items()}
    require_finite_metrics(metrics, "aggregated calibration")
    return metrics


def require_finite_metrics(metrics: dict[str, Any], context: str) -> None:
    """Reject invalid calibration scores before they can select a checkpoint."""
    for key, value in metrics.items():
        finite = torch.isfinite(value).all().item() if torch.is_tensor(value) else np.isfinite(value)
        if not finite:
            raise FloatingPointError(f"Non-finite {key} in {context}: {value}")


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_checkpoint(
    system: torch.nn.Module,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    base_twm_sha256: str,
    base_hawor_sha256: str,
    seed: int,
    sampler_state: dict[str, Any],
    best_calibration_loss: float | None,
    last_calibration: dict[str, float] | None,
    calibration_samples_json: list[dict[str, Any]],
    protocol_evidence: dict[str, Any],
) -> dict[str, Any]:
    if global_step < 1:
        raise ValueError("Untrained visual weights cannot be checkpointed")
    return {
        "architecture": INTRA_ARCHITECTURE,
        "format": INTRA_FORMAT,
        "format_version": INTRA_FORMAT_VERSION,
        "visual": {key: value.detach().cpu() for key, value in system.observer.visual.state_dict().items()},
        "config": copy.deepcopy(config),
        "trainer": {
            "global_step": int(global_step),
            "seed": int(seed),
            "optimizer": optimizer.state_dict(),
            "sampler_state_json": json.dumps(sampler_state, sort_keys=True),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else [],
            "best_calibration_loss": best_calibration_loss,
            "last_calibration": last_calibration,
            "calibration_samples": calibration_samples_json,
            "calibration_selection_rule": CALIBRATION_SELECTION_RULE,
            "training_candidate_rule": (
                "all GT-valid train frames; raw visual availability remains an input mask"),
            "training_sampler": TRAIN_SAMPLING_RULE,
            "protocol_evidence": protocol_evidence,
            "trainable_component": "observer.visual only",
            "history_commit": "raw HaWoR state only; dispersion unknown",
            "loss": "final-state joint MPJPE / 0.05m + 1e-4 * current NLL",
        },
        "base_twm_sha256": base_twm_sha256,
        "base_hawor_sha256": base_hawor_sha256,
    }


def _restore_resume(
    path: Path,
    system: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    base_twm_sha256: str,
    base_hawor_sha256: str,
) -> tuple[int, np.random.Generator, float | None, dict[str, float] | None]:
    checkpoint = twm_only.load_checkpoint_restricted(path)
    if (checkpoint.get("architecture") != INTRA_ARCHITECTURE
            or checkpoint.get("format") != INTRA_FORMAT
            or checkpoint.get("format_version") != INTRA_FORMAT_VERSION):
        raise ValueError("Resume checkpoint is not an intra-frame visual training checkpoint")
    if checkpoint.get("config") != config:
        raise ValueError("Resume checkpoint HandLoop config differs from the frozen base config")
    if (checkpoint.get("base_twm_sha256") != base_twm_sha256
            or checkpoint.get("base_hawor_sha256") != base_hawor_sha256):
        raise ValueError("Resume checkpoint was trained with different frozen base weights")
    trainer = checkpoint.get("trainer")
    if not isinstance(trainer, dict) or int(trainer.get("global_step", 0)) <= 0:
        raise ValueError("Resume checkpoint has no completed training step")
    system.observer.visual.load_state_dict(checkpoint["visual"], strict=True)
    optimizer_state = trainer.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ValueError("Resume checkpoint is missing optimizer state")
    optimizer.load_state_dict(optimizer_state)
    rng_state = json.loads(trainer["sampler_state_json"])
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    torch_rng = trainer.get("torch_rng_state")
    if isinstance(torch_rng, torch.Tensor):
        torch.set_rng_state(torch_rng.cpu())
    cuda_rng = trainer.get("cuda_rng_state")
    if torch.cuda.is_available() and isinstance(cuda_rng, list) and cuda_rng:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
    best = trainer.get("best_calibration_loss")
    last = trainer.get("last_calibration")
    return int(trainer["global_step"]), rng, best, last


def build_train_context_groups(
    sequences: list[PreparedSequence],
) -> list[dict[str, Any]]:
    """Group every supervised train frame by its inference-aligned HaWoR context."""
    groups: list[dict[str, Any]] = []
    for sequence in sequences:
        starts = context_start_indices(sequence)
        frames_by_start: dict[int, list[int]] = {}
        for frame in np.flatnonzero(sequence.gt_valid):
            frames_by_start.setdefault(int(starts[frame]), []).append(int(frame))
        for context_start, frame_indices in sorted(frames_by_start.items()):
            groups.append({
                "sequence": sequence,
                "context_start": context_start,
                "frame_indices": np.asarray(frame_indices, dtype=np.int64),
            })
    if not groups:
        raise ValueError("Training split has no GT-valid frames")
    return groups


class ContextWeightedSampler:
    """Sample batches from one context while preserving per-frame marginal weight."""

    def __init__(self, groups: list[dict[str, Any]]) -> None:
        if not groups or any(len(group["frame_indices"]) < 1 for group in groups):
            raise ValueError("Context sampler requires non-empty supervised groups")
        self.groups = groups
        self.weights = np.asarray([len(group["frame_indices"]) for group in groups], dtype=np.int64)
        self.cumulative = np.cumsum(self.weights)
        self.candidate_count = int(self.cumulative[-1])

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[tuple[PreparedSequence, int]]:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        group_index = int(np.searchsorted(self.cumulative, rng.integers(self.candidate_count), side="right"))
        group = self.groups[group_index]
        frames = group["frame_indices"]
        selected = rng.choice(frames, size=batch_size, replace=len(frames) < batch_size)
        return [(group["sequence"], int(frame)) for frame in selected]


def _sample_manifest(samples: list[tuple[PreparedSequence, int]]) -> list[dict[str, Any]]:
    return [{"sequence": sequence.sequence, "frame_index": int(frame)} for sequence, frame in samples]


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.calibration_every) < 1:
        raise ValueError("steps, batch-size, save-every, and calibration-every must be positive")
    if args.learning_rate <= 0 or args.joint_scale <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning-rate, joint-scale, and gradient-clip must be positive")
    torch.set_num_threads(max(1, args.cpu_threads))
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    for path in (args.twm_checkpoint, args.hawor_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_twm_sha256 = sha256_file(args.twm_checkpoint)
    base_hawor_sha256 = sha256_file(args.hawor_checkpoint)
    if base_twm_sha256 != EXPECTED_TWM_SHA256 or base_hawor_sha256 != EXPECTED_HAWOR_SHA256:
        raise ValueError("Frozen TWM/HaWoR hashes do not match the approved intra-frame protocol")
    config, twm_state = load_base_twm(args.twm_checkpoint, EXPECTED_TWM_SHA256)
    system = build_system(args.hawor_checkpoint, config, device)
    system.observer.twm.load_state_dict(twm_state, strict=True)
    freeze_except_visual(system)
    system.eval()

    groups, protocol = load_sequences(
        data_root=args.data_root,
        video_root=args.video_root,
        split_manifest=args.split_manifest,
        export_split=args.export_split,
        protocol_path=args.protocol,
        check_images=not args.skip_image_check,
    )
    all_sequences = groups["train"] + groups["calibration"]
    convert_raw_history_states(all_sequences, system.geometry, device)
    train_sampler = ContextWeightedSampler(build_train_context_groups(groups["train"]))
    calibration = calibration_samples(groups["calibration"])

    optimizer = torch.optim.AdamW(
        system.observer.visual.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    trainable_ids = {id(parameter) for parameter in system.observer.visual.parameters()
                     if parameter.requires_grad}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != trainable_ids:
        raise RuntimeError("Optimizer must contain exactly the HandLoopVisual parameters")
    frozen_before = {
        "hawor": module_parameter_sha256(system.hawor),
        "twm": module_parameter_sha256(system.observer.twm),
        "geometry": module_parameter_sha256(system.geometry),
    }
    rng = np.random.default_rng(args.seed)
    global_step = 0
    best_calibration_loss: float | None = None
    last_calibration: dict[str, float] | None = None
    if args.resume is not None:
        global_step, rng, best_calibration_loss, last_calibration = _restore_resume(
            args.resume, system, optimizer, config, base_twm_sha256, base_hawor_sha256)
    if global_step >= args.steps:
        raise ValueError(f"Resume global_step={global_step} already reaches --steps={args.steps}")

    calibration_json = _sample_manifest(calibration)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.jsonl"
    if args.resume is None and log_path.exists():
        raise FileExistsError(f"Refusing to append a fresh run to existing log: {log_path}")

    system.observer.visual.train()
    system.observer.twm.eval()
    system.hawor.eval()
    with log_path.open("a", encoding="utf-8") as log:
        for step in range(global_step + 1, args.steps + 1):
            samples = train_sampler.sample(args.batch_size, rng)
            optimizer.zero_grad(set_to_none=True)
            result = train_batch(
                system, samples, device,
                state_weight=args.state_weight,
                joint_scale=args.joint_scale,
            )
            if not torch.isfinite(result["loss"]).all():
                raise FloatingPointError(f"Non-finite visual loss at step {step}")
            result["loss"].backward()
            parameters = [parameter for parameter in system.observer.visual.parameters()
                          if parameter.grad is not None]
            if not parameters:
                raise RuntimeError("Visual loss produced no HandLoopVisual gradients")
            if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"Non-finite visual gradients at step {step}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            if (not torch.isfinite(gradient_norm).all()
                    or float(gradient_norm.detach().cpu()) <= 0.0):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"Visual gradient norm is zero or non-finite at step {step}")
            optimizer.step()
            global_step = step

            should_save = global_step % args.save_every == 0 or global_step == args.steps
            should_calibrate = global_step % args.calibration_every == 0 or global_step == args.steps
            if should_calibrate:
                last_calibration = _calibrate(
                    system, calibration, args.batch_size, device,
                    state_weight=args.state_weight,
                    joint_scale=args.joint_scale,
                )
            record = {
                "step": global_step,
                "train_loss": float(result["loss"].detach().cpu()),
                "train_joint_m": float(result["joint_m"].cpu()),
                "train_state_nll": float(result["state_nll"].cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "visual_grad_nonzero": True,
                "training_candidate_rule": (
                    "all GT-valid train frames; raw visual availability remains an input mask"),
                "training_sampler": TRAIN_SAMPLING_RULE,
                "calibration": last_calibration if should_calibrate else None,
                "base_twm_sha256": base_twm_sha256,
                "base_hawor_sha256": base_hawor_sha256,
            }
            if should_calibrate:
                record["calibration_samples"] = calibration_json
                record["calibration_selection_rule"] = CALIBRATION_SELECTION_RULE
                calibration_loss = float(last_calibration["loss"])
                is_best = best_calibration_loss is None or calibration_loss < best_calibration_loss
                if is_best:
                    best_calibration_loss = calibration_loss
            else:
                is_best = False
            if should_save or should_calibrate:
                record["protocol_evidence"] = protocol.get("_validated_evidence")
                if global_step == args.steps:
                    frozen_after = {
                        "hawor": module_parameter_sha256(system.hawor),
                        "twm": module_parameter_sha256(system.observer.twm),
                        "geometry": module_parameter_sha256(system.geometry),
                    }
                    record["frozen_parameter_sha256_before"] = frozen_before
                    record["frozen_parameter_sha256_after"] = frozen_after
                    record["frozen_weights_unchanged"] = frozen_before == frozen_after
                    if not record["frozen_weights_unchanged"]:
                        raise RuntimeError("Frozen HaWoR/TWM/geometry parameters changed during visual-only training")
                payload = make_checkpoint(
                    system, config, optimizer,
                    global_step=global_step,
                    base_twm_sha256=base_twm_sha256,
                    base_hawor_sha256=base_hawor_sha256,
                    seed=args.seed,
                    sampler_state=rng.bit_generator.state,
                    best_calibration_loss=best_calibration_loss,
                    last_calibration=last_calibration,
                    calibration_samples_json=calibration_json,
                    protocol_evidence=protocol.get("_validated_evidence") or {},
                )
                _atomic_save(payload, output_dir / "latest.pt")
                if is_best:
                    _atomic_save(payload, output_dir / "best.pt")
            serialized = json.dumps(record, sort_keys=True)
            print(serialized, flush=True)
            log.write(serialized + "\n")
            log.flush()


if __name__ == "__main__":
    main()
