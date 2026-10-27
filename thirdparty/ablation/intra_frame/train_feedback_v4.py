#!/usr/bin/env python3
"""Train centered v4 feedback on fixed long windows; select on frozen calibration."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import train_feedback_v2 as v2
import train_feedback_v3 as v3
from feedback_v4 import (
    DEFAULT_FINGER_ROTATION_BOUND, DEFAULT_ROOT_ROTATION_BOUND,
    DEFAULT_TRANSLATION_BOUND, INTRA_ARCHITECTURE, install_feedback_v4,
)
from lib.eval_utils.eval_utils import first_align_joints
from world_targets_v3 import WorldTargetMap, sha256_file


HERE = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = HERE / "runs/feedback_v4_train"
PROTOCOL_REVISION = 4
MAX_STEPS = 400
CHECK_EVERY = 100
TRAIN_FRAME_LIMIT = {"uniform": 100, "event": 64}
LOSS_CONFIG = {
    "aligned_world_weight": 1.0, "raw_camera_weight": .25,
    "canonical_relative_weight": .5, "velocity_weight": .05,
    "no_harm_weight": 1.0, "pinch_distance_weight": .1,
    "onset_bracket_weight": .2, "joint_scale_m": .05,
    "pinch_prediction_threshold": .275, "pinch_gt_threshold": .2,
    "onset_margin": .01, "event_min_frames": 3,
    "nll_weight": 0.0, "contact_bce_weight": 0.0,
}
SAMPLING_RULE = "uniform kind then uniform window; first 100 uniform rows or all <=64 event rows"
SELECTION_RULE = "W strictly below baseline; onset MAE strictly below baseline; F1 and recall at least baseline; baseline TP >=10"
MEMORY_POLICY = {
    "hawor": "frozen CPU offload; use cached 16-frame inputs",
    "twm": "frozen non-reentrant gradient checkpointing, microbatch 4",
    "loss": "one complete 100/64-frame window and one backward pass",
}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--twm-checkpoint", type=Path, default=v2.DEFAULT_TWM)
    p.add_argument("--hawor-checkpoint", type=Path, default=v2.DEFAULT_HAWOR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_RUN_DIR / "cache")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--camera-map", type=Path, default=v3.DEFAULT_MAP)
    p.add_argument("--event-manifest", type=Path, default=v3.DEFAULT_EVENT_MANIFEST)
    p.add_argument("--resume", type=Path)
    p.add_argument("--steps", type=int, default=MAX_STEPS)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--gradient-clip", type=float, default=1.)
    p.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--smoke-calibration-window", action="store_true",
                   help="Discarded 100-frame forward/backward on existing calibration cache")
    p.add_argument("--smoke-100", action="store_true",
                   help="Discarded 100-frame forward/backward on v4 training cache")
    p.add_argument("--emit-validation-checkpoint", action="store_true",
                   help="Save a fresh zero-head checkpoint for loader roundtrip only")
    p.add_argument("--baseline-only", action="store_true")
    return p.parse_args()


def _make_system(args: argparse.Namespace, device: torch.device):
    for path in (args.twm_checkpoint, args.hawor_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    twm_hash = sha256_file(args.twm_checkpoint)
    hawor_hash = sha256_file(args.hawor_checkpoint)
    if (twm_hash != v2.EXPECTED_TWM_SHA256
            or hawor_hash != v2.EXPECTED_HAWOR_SHA256):
        raise ValueError("Frozen TWM/HaWoR checkpoint SHA256 changed")
    config, twm_state = v2.load_base_twm(args.twm_checkpoint, v2.EXPECTED_TWM_SHA256)
    system = v2.build_system(args.hawor_checkpoint, config, device)
    system.observer.twm.load_state_dict(twm_state, strict=True)
    install_feedback_v4(system)
    v2.freeze_except_visual(system)
    v2._enable_twm_gradient_checkpointing(system)
    system.eval()
    v2._offload_hawor(system, device)
    config = copy.deepcopy(config)
    config["rounds"] = 2
    config["reobserve"] = True
    return system, config, twm_hash, hawor_hash


def _load_items(cache_dir: Path, manifest: dict[str, Any], ids: set[str]) -> dict[str, v2.CachedContext]:
    """Load exact manifest items, including invalid train rows needed for masks."""
    rows = {str(item["item_id"]): item for item in manifest["items"]}
    if len(rows) != len(manifest["items"]) or not ids.issubset(rows):
        raise ValueError("Cached context item IDs are missing or duplicated")
    contexts = {}
    for item_id in sorted(ids):
        item = rows[item_id]
        inputs = torch.load(cache_dir / item["inputs_path"], map_location="cpu", weights_only=True)
        labels = torch.load(cache_dir / item["labels_path"], map_location="cpu", weights_only=True)
        frames = np.asarray(item["frame_indices"], dtype=np.int64)
        if not isinstance(inputs, dict) or not isinstance(labels, dict):
            raise ValueError(f"Malformed item {item_id}")
        if inputs["patches"].dtype != torch.float16 or len(inputs["patches"]) != len(frames):
            raise ValueError(f"Cached patches missing or wrong dtype for {item_id}")
        if labels["frame_indices"].tolist() != frames.tolist():
            raise ValueError(f"Cached label frame IDs differ from manifest: {item_id}")
        inputs["patches"] = inputs["patches"].float()
        contexts[item_id] = v2.CachedContext(
            item_id=item_id, split=str(item["split"]), sequence=str(item["sequence"]),
            context_start=int(item["context_start"]), frame_indices=frames,
            inputs=inputs, labels=labels)
    return contexts


def _row(value: Any, row: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[row:row + 1]
    if isinstance(value, dict):
        return {key: _row(child, row) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_row(child, row) for child in value)
    if isinstance(value, list):
        return [_row(child, row) for child in value]
    raise TypeError(f"Unsupported cached input {type(value)}")


def compose_window(window: dict[str, Any], contexts: dict[str, v2.CachedContext],
                   *, limit: int | None = None) -> v2.CachedContext:
    """Keep each row's independent HaWoR context while restoring frame order."""
    kind = str(window["kind"])
    if kind not in TRAIN_FRAME_LIMIT:
        raise ValueError(f"Unknown training-window kind {kind}")
    frames = np.asarray(window["frame_indices"], dtype=np.int64)
    expected = 128 if kind == "uniform" else None
    if expected is not None and len(frames) != expected:
        raise ValueError("Uniform training window must have 128 cached frames")
    if kind == "event" and not 1 <= len(frames) <= 64:
        raise ValueError("Event training window must have 1..64 frames")
    if not np.all(np.diff(frames) == 1):
        raise ValueError("Training window frame IDs must be consecutive")
    if limit is None:
        limit = TRAIN_FRAME_LIMIT[kind]
    selected = frames[:limit]
    sequence = str(window["sequence"])
    by_frame: dict[int, tuple[v2.CachedContext, int]] = {}
    for item_id in window["context_item_ids"]:
        context = contexts[item_id]
        if context.sequence != sequence or context.split not in {"train", "calibration"}:
            raise ValueError(f"Window/context sequence or split mismatch: {item_id}")
        for row, frame in enumerate(context.frame_indices):
            frame = int(frame)
            if frame in by_frame:
                raise ValueError(f"Duplicate context row for frame {sequence}:{frame}")
            by_frame[frame] = context, row
    if set(by_frame) != set(map(int, frames)):
        raise ValueError("Referenced contexts do not cover exactly this window")
    input_rows = [_row(by_frame[int(frame)][0].inputs, by_frame[int(frame)][1]) for frame in selected]
    labels = {
        key: torch.stack([by_frame[int(frame)][0].labels[key][by_frame[int(frame)][1]]
                          for frame in selected])
        for key in by_frame[int(selected[0])][0].labels
    }
    result = v2.CachedContext(
        item_id=f"{sequence}/{kind}_{int(window['window_index']):02d}",
        split="train", sequence=sequence, context_start=int(selected[0]),
        frame_indices=selected.copy(), inputs=v3._cat_inputs(input_rows), labels=labels)
    if result.labels["frame_indices"].tolist() != selected.tolist():
        raise ValueError("Window labels lost frame order")
    return result


def _validate_training_windows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("protocol_revision") != 3 or manifest.get("training_extension_revision") != 4:
        raise ValueError("v4 requires revision-3 cache with training extension revision 4")
    windows = manifest.get("training_windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("v4 training windows are missing")
    by_seq: dict[str, dict[str, int]] = {}
    names = set()
    for window in windows:
        sequence, kind = str(window["sequence"]), str(window["kind"])
        key = (sequence, kind, int(window["window_index"]))
        if key in names or kind not in TRAIN_FRAME_LIMIT:
            raise ValueError(f"Duplicate or invalid training window {key}")
        names.add(key)
        by_seq.setdefault(sequence, {"uniform": 0, "event": 0})[kind] += 1
    if len(by_seq) != 33 or any(v["uniform"] != 4 or v["event"] > 4 for v in by_seq.values()):
        raise ValueError("Expected 33 train sequences with four uniform and up to four event windows each")
    return windows


def _valid_chunks(labels: dict[str, torch.Tensor], valid: torch.Tensor) -> list[tuple[int, int]]:
    return v2._valid_chunk_slices(
        labels["frame_indices"].detach().cpu().numpy(),
        labels["segment_id"].detach().cpu().numpy(),
        labels["timestamp_sec"].detach().cpu().numpy(),
        valid.detach().cpu().numpy(), max_frames=100, min_frames=16)


def aligned_point_errors(predicted: torch.Tensor, target: torch.Tensor,
                         labels: dict[str, torch.Tensor], valid: torch.Tensor
                         ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Formal 100-frame, minimum-16 first-two-frame alignment, with gradients."""
    chunks = _valid_chunks(labels, valid)
    errors = []
    for start, stop in chunks:
        pred = predicted[start:stop].to("cpu")
        gt = target[start:stop].to("cpu")
        point = pred[:2].reshape(-1, 3)
        singular = torch.linalg.svdvals((point - point.mean(0)).detach())
        if float(singular[-1]) < 1e-7 or float(singular[0] / singular[-1]) > 1e6:
            raise ValueError(f"Degenerate formal alignment anchor at frame {start}")
        aligned = first_align_joints(gt, pred)
        if not torch.isfinite(aligned).all():
            raise FloatingPointError("Nonfinite first-two-frame alignment")
        errors.append(torch.linalg.vector_norm(aligned - gt, dim=-1).reshape(-1))
    return (torch.cat(errors).to(predicted.device) if errors else predicted.new_empty(0), chunks)


def _continuous_runs(labels: dict[str, torch.Tensor], valid: torch.Tensor) -> list[tuple[int, int]]:
    frame = labels["frame_indices"].detach().cpu().numpy()
    seg = labels["segment_id"].detach().cpu().numpy()
    time = labels["timestamp_sec"].detach().cpu().numpy()
    ok = valid.detach().cpu().numpy()
    runs, start = [], None
    for i in range(len(ok)):
        connected = (i > 0 and ok[i-1] and frame[i] == frame[i-1] + 1
                     and seg[i] == seg[i-1] and 0 < time[i] - time[i-1] <= .25)
        if not ok[i]:
            if start is not None:
                runs.append((start, i))
            start = None
        elif start is None:
            start = i
        elif not connected:
            runs.append((start, i))
            start = i
    if start is not None:
        runs.append((start, len(ok)))
    return runs


def onset_bracket_loss(pred_distance: torch.Tensor, gt_distance: torch.Tensor,
                       labels: dict[str, torch.Tensor], valid: torch.Tensor
                       ) -> tuple[torch.Tensor, int]:
    """Supervise only complete GT onsets with a valid preceding and three contact rows."""
    gt = gt_distance.detach().cpu().numpy()
    losses = []
    for left, right in _continuous_runs(labels, valid):
        # Training needs the onset bracket, not the event offset. Formal
        # extract_events additionally requires an open frame after the whole
        # event, which censors long events at a 64-frame training-window edge.
        for start in range(left + 1, right - 2):
            if not (gt[start - 1] >= .2 and np.all(gt[start:start + 3] < .2)):
                continue
            before = F.relu((.275 + .01) - pred_distance[start-1])
            after = F.relu(pred_distance[start:start+3] - (.275 - .01)).mean()
            losses.append(before + after)
    return (torch.stack(losses).mean() if losses else pred_distance.new_zeros(()), len(losses))


def _delta_metrics(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    values = output["visual_deltas_camera"]
    delta = torch.stack(values)  # rounds, rows, 51
    pos = torch.linalg.vector_norm(delta[..., :3], dim=-1)
    root = torch.linalg.vector_norm(delta[..., 3:6], dim=-1)
    finger = torch.linalg.vector_norm(delta[..., 6:].reshape(*delta.shape[:2], 15, 3), dim=-1)
    return {
        "delta_translation_norm_m": pos.detach().mean(),
        "delta_root_norm_rad": root.detach().mean(),
        "delta_finger_norm_rad": finger.detach().mean(),
        "delta_saturation_rate": torch.cat((
            (pos >= .95 * DEFAULT_TRANSLATION_BOUND).flatten(),
            (root >= .95 * DEFAULT_ROOT_ROTATION_BOUND).flatten(),
            (finger >= .95 * DEFAULT_FINGER_ROTATION_BOUND).flatten())).float().mean().detach(),
    }


def window_loss(system: torch.nn.Module, context: v2.CachedContext,
                baseline_joints: torch.Tensor, world: WorldTargetMap,
                device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = v2._input_to_device(context.inputs, device)
    output = v2.observer_forward(system, dict(inputs))
    labels = v2._labels_to_device(context.labels, device)
    camera = labels["target_joints"]
    canonical = world.target(context).to(device)
    valid = labels["gt_valid"].bool() & labels["visual_valid"].bool()
    if not valid.any():
        raise ValueError(f"No supervised rows: {context.item_id}")
    for name, value in (("camera", camera), ("canonical", canonical)):
        if not torch.isfinite(value[valid]).all():
            raise FloatingPointError(f"Nonfinite valid {name} target in {context.item_id}")
    predicted = v2._world_joints(system, output["state"], inputs["projector_args"][0])
    base = baseline_joints.to(device)
    if predicted.shape != camera.shape or base.shape != camera.shape:
        raise ValueError("Candidate, baseline and GT joint shapes differ")
    if not torch.isfinite(predicted).all() or not torch.isfinite(base).all():
        raise FloatingPointError("Nonfinite candidate or baseline joints")
    aligned, chunks = aligned_point_errors(predicted, canonical, labels, valid)
    with torch.no_grad():
        base_aligned, base_chunks = aligned_point_errors(base, canonical, labels, valid)
    if chunks != base_chunks or aligned.numel() != base_aligned.numel():
        raise AssertionError("Baseline/candidate alignment masks differ")
    zero = predicted.new_zeros(())
    aligned_m = aligned.mean() if aligned.numel() else zero
    noharm_m = F.relu(aligned - base_aligned).mean() if aligned.numel() else zero
    camera_m = torch.linalg.vector_norm(predicted[valid]-camera[valid], dim=-1).mean()
    first = int(torch.nonzero(valid, as_tuple=False)[0, 0])
    pred_rel = predicted - predicted[first:first+1, :1]
    gt_rel = canonical - canonical[first:first+1, :1]
    relative_m = torch.linalg.vector_norm(pred_rel[valid]-gt_rel[valid], dim=-1).mean()
    velocity = v2._joint_velocity_loss(predicted, canonical, labels, valid)
    pred_dist = torch.linalg.vector_norm(predicted[:, 4]-predicted[:, 8], dim=-1) / torch.linalg.vector_norm(predicted[:, 0]-predicted[:, 9], dim=-1).clamp_min(1e-6)
    # Invalid GT rows may contain NaNs. Replace them before any norm so an
    # apparently masked loss cannot create a 0*NaN visual gradient.
    safe_camera = torch.where(valid[:, None, None], camera, predicted.detach())
    gt_dist = torch.linalg.vector_norm(safe_camera[:, 4]-safe_camera[:, 8], dim=-1) / torch.linalg.vector_norm(safe_camera[:, 0]-safe_camera[:, 9], dim=-1).clamp_min(1e-6)
    pinch = F.smooth_l1_loss(pred_dist[valid], gt_dist[valid])
    bracket, onset_count = onset_bracket_loss(pred_dist, gt_dist, labels, valid)
    loss = ((aligned_m + .25*camera_m + .5*relative_m + noharm_m)/.05
            + .05*velocity + .1*pinch + .2*bracket)
    metrics = {
        "loss": loss.detach(), "aligned_world_joint_m": aligned_m.detach(),
        "no_harm_m": noharm_m.detach(), "raw_camera_joint_m": camera_m.detach(),
        "canonical_relative_joint_m": relative_m.detach(),
        "velocity_smooth_l1_mps": velocity.detach(),
        "pinch_distance_smooth_l1": pinch.detach(),
        "onset_bracket_loss": bracket.detach(),
        "gt_onset_count": torch.tensor(onset_count),
        "aligned_valid_points": torch.tensor(aligned.numel()),
        "valid_frames": valid.sum().detach(), **_delta_metrics(output),
    }
    return metrics, loss


def _bound_sources(manifest: dict[str, Any], contexts: list[v2.CachedContext],
                   cache_sha: str, world: WorldTargetMap, event_sha: str) -> dict[str, str]:
    return {
        "cache_manifest_sha256": cache_sha,
        "world_reference_manifest_sha256": world.manifest_sha256,
        "event_manifest_sha256": event_sha,
        "calibration_windows_sha256": v2._json_sha256(manifest["calibration_windows"]),
        "calibration_mask_sha256": v2._calibration_mask_sha256(contexts, manifest["calibration_windows"]),
        "event_gt_sidecars_sha256": v2._json_sha256(manifest["event_ground_truth_sidecars"]),
        "event_mask_sha256": v3._fixed_event_mask_hash(manifest),
        "calibration_protocol_sha256": v2._json_sha256(calibration_protocol()),
    }


def calibration_protocol() -> dict[str, Any]:
    return {
        "revision": 4, "calibration_rule": v3.CALIBRATION_RULE,
        "selection_rule": SELECTION_RULE, "loss_config": LOSS_CONFIG,
        "w_window_count": 40, "w_block_frames": 100, "w_min_frames": 16,
        "event_reference_count": 224, "event_pred_threshold": .275,
        "event_gt_threshold": .2, "event_min_frames": 3,
        "onset_tolerance_sec": .2, "train_frame_limit": TRAIN_FRAME_LIMIT,
        "sampling_rule": SAMPLING_RULE,
    }


def _reference_file(run_dir: Path) -> Path:
    return run_dir / "reference_states.pt"


def _reference_joints(system: torch.nn.Module, context: v2.CachedContext,
                      device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        inputs = v2._input_to_device(context.inputs, device)
        state = v2._single_twm_state(system, inputs)
        return v2._world_joints(system, state, inputs["projector_args"][0]).detach().float().cpu()


def _ensure_references(system: torch.nn.Module, contexts: dict[str, v2.CachedContext],
                       run_dir: Path, source: dict[str, str], device: torch.device
                       ) -> dict[str, torch.Tensor]:
    path = _reference_file(run_dir)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("source") != source or set(payload.get("joints", {})) != set(contexts):
            raise ValueError("Reference predictions are stale or incomplete")
        return payload["joints"]
    joints = {item_id: _reference_joints(system, context, device)
              for item_id, context in contexts.items()}
    v2._atomic_save({"source": source, "joints": joints}, path)
    return joints


def _compose_reference(window: dict[str, Any], context: v2.CachedContext,
                       original: dict[str, v2.CachedContext],
                       reference: dict[str, torch.Tensor]) -> torch.Tensor:
    by_frame = {}
    for item_id in window["context_item_ids"]:
        for row, frame in enumerate(original[item_id].frame_indices):
            if int(frame) in by_frame:
                raise ValueError("Duplicate reference frame")
            by_frame[int(frame)] = reference[item_id][row]
    return torch.stack([by_frame[int(frame)] for frame in context.frame_indices])


def _calibrate_from_predictions(predictions: dict[tuple[str, int], np.ndarray],
                                contexts: list[v2.CachedContext], manifest: dict[str, Any],
                                device: torch.device, system: torch.nn.Module) -> dict[str, Any]:
    original = v3._predictions_by_frame
    try:
        v3._predictions_by_frame = lambda *_args, **_kwargs: (predictions, None)
        return v3.evaluate_calibration(system, contexts, manifest, device)
    finally:
        v3._predictions_by_frame = original


def _baseline(system: torch.nn.Module, contexts: list[v2.CachedContext],
              references: dict[str, torch.Tensor], manifest: dict[str, Any],
              bound: dict[str, str], run_dir: Path, device: torch.device) -> dict[str, Any]:
    path = run_dir / "baseline_calibration.json"
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        if any(result.get(k) != value for k, value in bound.items()):
            raise ValueError("Stored v4 baseline source or protocol binding changed")
        if result.get("architecture") != INTRA_ARCHITECTURE:
            raise ValueError("Stored baseline architecture changed")
        return result
    visual = system.observer.visual
    if any(torch.count_nonzero(p).item() for name in ("root_delta_head", "finger_delta_head")
           for p in getattr(visual, name).parameters()):
        raise ValueError("Baseline requires fresh zero visual heads")
    maximum = 0.0
    with torch.no_grad():
        for context in contexts[:8]:
            inputs = v2._input_to_device(context.inputs, device)
            candidate = v2.observer_forward(system, dict(inputs))["state"]
            direct = v2._single_twm_state(system, inputs)
            valid = context.labels["visual_valid"].to(device).bool()
            if bool(valid.any()):
                maximum = max(maximum, float((candidate[valid]-direct[valid]).abs().max().cpu()))
                if not torch.allclose(candidate[valid], direct[valid], atol=1e-5, rtol=1e-5):
                    raise AssertionError("Zero v4 differs from direct TWM")
    predictions = {(context.sequence, int(frame)): references[context.item_id][row].numpy()
                   for context in contexts for row, frame in enumerate(context.frame_indices)}
    result = _calibrate_from_predictions(predictions, contexts, manifest, device, system)
    result.update({**bound, "architecture": INTRA_ARCHITECTURE,
                   "protocol_revision": PROTOCOL_REVISION,
                   "experiment": "feedback_v4", "calibration_protocol": calibration_protocol(),
                   "method": "cached direct single-TWM reference; 8 zero-v4 context equivalence checks",
                   "zero_v4_equals_single_twm": {"max_abs_world_state_difference": maximum,
                                                   "contexts_checked": 8, "atol": 1e-5, "rtol": 1e-5},
                   "base_checkpoints": manifest["base_checkpoints"],
                   "frozen_parameter_hashes": manifest["frozen_parameter_hashes"]})
    if result["events"]["reference_events"] != 224:
        raise ValueError("Baseline lacks the fixed 224 GT events")
    v2._atomic_json(result, path)
    return result


def qualified(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    b, c = baseline["events"], candidate["events"]
    values = (baseline["w_mpjpe_mm"], candidate["w_mpjpe_mm"],
              b["onset_mae_ms"], c["onset_mae_ms"], b["f1"], c["f1"],
              b["recall"], c["recall"])
    if b["tp"] < 10 or any(value is None or not np.isfinite(float(value)) for value in values):
        return False
    return (candidate["w_mpjpe_mm"] < baseline["w_mpjpe_mm"]
            and c["onset_mae_ms"] < b["onset_mae_ms"]
            and c["f1"] >= b["f1"] and c["recall"] >= b["recall"])


_score_is_qualified = qualified


def _checkpoint(system: torch.nn.Module, config: dict[str, Any], optimizer: torch.optim.Optimizer,
                step: int, rng: np.random.Generator, hashes: dict[str, str],
                bound: dict[str, str], baseline: dict[str, Any],
                last_calibration: dict[str, Any] | None, selection: dict[str, Any]
                ) -> dict[str, Any]:
    return {
        "architecture": INTRA_ARCHITECTURE,
        "format": v2.INTRA_FORMAT, "format_version": v2.INTRA_FORMAT_VERSION,
        "experiment": "feedback_v4", "protocol_revision": PROTOCOL_REVISION,
        "visual": {key: value.detach().cpu() for key, value in system.observer.visual.state_dict().items()},
        "config": copy.deepcopy(config), "bounds": v2._bound_config(system),
        "base_twm_sha256": hashes["twm_sha256"],
        "base_hawor_sha256": hashes["hawor_sha256"],
        "trainer": {
            "protocol_revision": PROTOCOL_REVISION, "experiment": "feedback_v4",
            "global_step": step, "seed": 7, "optimizer": optimizer.state_dict(),
            "sampler_state_json": json.dumps(rng.bit_generator.state, sort_keys=True),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "loss_config": LOSS_CONFIG, "sampler": SAMPLING_RULE,
            "calibration_rule": v3.CALIBRATION_RULE, "selection_rule": SELECTION_RULE,
            "training_memory_policy": MEMORY_POLICY,
            "baseline_calibration": baseline,
            "last_calibration": last_calibration, "selection_state": selection,
            "frozen_parameter_hashes": hashes["frozen_parameter_hashes"], **bound,
            "maximum_training_steps": MAX_STEPS, "calibration_every": CHECK_EVERY,
            "checkpoint_every": CHECK_EVERY,
            "trainable_component": "observer.visual only",
        },
    }


def _restore(path: Path, system: torch.nn.Module, optimizer: torch.optim.Optimizer,
             config: dict[str, Any], hashes: dict[str, str], bound: dict[str, str]
             ) -> tuple[int, np.random.Generator, dict[str, Any], dict[str, Any] | None]:
    checkpoint = v2.twm_only.load_checkpoint_restricted(path)
    trainer = checkpoint.get("trainer", {})
    if (checkpoint.get("format") != v2.INTRA_FORMAT
            or checkpoint.get("format_version") != 1
            or checkpoint.get("architecture") != INTRA_ARCHITECTURE
            or checkpoint.get("protocol_revision") != 4
            or trainer.get("protocol_revision") != 4):
        raise ValueError("Resume checkpoint format/architecture/protocol differs from v4")
    if checkpoint.get("config") != config or checkpoint.get("bounds") != v2._bound_config(system):
        raise ValueError("Resume configuration or bounds differ")
    if any(trainer.get(k) != value for k, value in bound.items()):
        raise ValueError("Resume source/protocol hashes differ")
    if checkpoint.get("base_twm_sha256") != hashes["twm_sha256"] or checkpoint.get("base_hawor_sha256") != hashes["hawor_sha256"]:
        raise ValueError("Resume base checkpoint hashes differ")
    system.observer.visual.load_state_dict(checkpoint["visual"], strict=True)
    optimizer.load_state_dict(trainer["optimizer"])
    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(trainer["sampler_state_json"])
    torch.set_rng_state(trainer["torch_rng_state"].cpu())
    if torch.cuda.is_available() and trainer.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(trainer["cuda_rng_state"])
    step = int(trainer["global_step"])
    if step <= 0:
        raise ValueError("Resume step must be positive")
    return step, rng, dict(trainer["selection_state"]), trainer.get("last_calibration")


def main() -> None:
    args = _args()
    if not 1 <= args.steps <= MAX_STEPS or args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip <= 0:
        raise ValueError("Invalid frozen training hyperparameters")
    torch.set_num_threads(max(1, args.cpu_threads))
    v2.seed_everything(7)
    device = torch.device(args.device)
    run_dir, cache_dir = args.run_dir.resolve(), args.cache_dir.resolve()
    manifest = v2.load_manifest(cache_dir)
    if manifest.get("protocol_revision") != 3:
        raise ValueError("v4 requires revision-3 calibration cache")
    cache_sha = sha256_file(cache_dir / "manifest.json")
    system, config, twm_hash, hawor_hash = _make_system(args, device)
    frozen = v2._frozen_hashes(system)
    hashes = {"twm_sha256": twm_hash, "hawor_sha256": hawor_hash,
              "frozen_parameter_hashes": frozen}
    if manifest["base_checkpoints"] != {"twm_sha256": twm_hash, "hawor_sha256": hawor_hash} or manifest["frozen_parameter_hashes"] != frozen:
        raise ValueError("Cache is bound to different frozen models")
    world = WorldTargetMap(args.camera_map)
    event_sha = sha256_file(args.event_manifest)
    event_manifest = json.loads(args.event_manifest.read_text(encoding="utf-8"))
    if event_manifest.get("protocol_revision") != 2 or manifest["calibration_event_manifest"]["sha256"] != event_sha:
        raise ValueError("Frozen calibration event manifest changed")
    v3.WORLD, v3.EVENT_MANIFEST, v3.EVENT_HASH, v3.CACHE_DIR = world, event_manifest, event_sha, cache_dir
    cal_ids = {str(item["item_id"]) for item in manifest["items"] if item["split"] == "calibration"}
    cal = _load_items(cache_dir, manifest, cal_ids)
    calibration_contexts = list(cal.values())
    if len(manifest["calibration_windows"]) != 40 or len(manifest["event_windows"]) != len(event_manifest["merged_windows"]):
        raise ValueError("Frozen calibration windows changed")
    parent_view = dict(manifest)
    parent_view["items"] = [item for item in manifest["items"]
                            if item["split"] != "train" or not str(item["item_id"]).startswith("train_v4/")]
    v2._ORIGINAL_VALIDATE_CACHE = v2._validate_cache_protocol
    v3._validate_cache_protocol(parent_view, calibration_contexts)
    bound = _bound_sources(manifest, calibration_contexts, cache_sha, world, event_sha)
    v3.BOUND_HASHES = bound

    if args.emit_validation_checkpoint:
        source_baseline = HERE / "runs/feedback_v3_train/baseline_calibration.json"
        baseline = json.loads(source_baseline.read_text(encoding="utf-8"))
        optimizer = torch.optim.AdamW(system.observer.visual.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        payload = _checkpoint(system, config, optimizer, 1, np.random.default_rng(7),
                              hashes, bound, baseline, None,
                              {"selection_qualified": False, "validation_only": True})
        payload["validation_only"] = True
        payload["trainer"]["validation_only"] = True
        payload["trainer"]["optimizer_step_completed"] = False
        path = run_dir / "validation/discarded_validation_checkpoint.pt"
        if path.exists():
            raise FileExistsError(path)
        v2._atomic_save(payload, path)
        print(json.dumps({"validation_checkpoint": str(path),
                          "source": bound, "not_for_selection_or_test": True}), flush=True)
        return

    if args.smoke_calibration_window:
        window = dict(manifest["calibration_windows"][0], kind="uniform")
        context = compose_window(window, cal, limit=100)
        baseline_joints = _reference_joints(system, context, device)
        system.observer.visual.train()
        metrics, loss = window_loss(system, context, baseline_joints, world, device)
        loss.backward()
        grads = [p.grad for p in system.observer.visual.parameters() if p.requires_grad and p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError("Discarded calibration smoke has no finite visual gradient")
        print(json.dumps({"smoke": "discarded_calibration_100", "frames": context.size,
                          "metrics": v2._metric_float(metrics),
                          "gpu_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}), flush=True)
        return

    windows = _validate_training_windows(manifest)
    train_ids = {str(item_id) for window in windows for item_id in window["context_item_ids"]}
    train = _load_items(cache_dir, manifest, train_ids)
    if any(context.split != "train" for context in train.values()):
        raise ValueError("Training window refers to non-train cache item")
    # Validate every extended window before any optimizer step; unsupervised
    # event windows cannot contribute a loss and are excluded deterministically.
    trainable_windows = []
    excluded_unsupervised = []
    for window in windows:
        composed = compose_window(window, train)
        if bool((composed.labels["gt_valid"] & composed.labels["visual_valid"]).any()):
            trainable_windows.append(window)
        else:
            excluded_unsupervised.append((window["sequence"], window["kind"], window["window_index"]))
    windows = trainable_windows
    if not any(window["kind"] == "uniform" for window in windows):
        raise ValueError("No supervised uniform training window")
    if args.smoke_100:
        window = next(row for row in windows if row["kind"] == "uniform")
        context = compose_window(window, train)
        baseline_joints = _reference_joints(system, context, device)
        system.observer.visual.train()
        metrics, loss = window_loss(system, context, baseline_joints, world, device)
        loss.backward()
        grads = [p.grad for p in system.observer.visual.parameters() if p.requires_grad and p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError("Discarded train smoke has no finite visual gradient")
        print(json.dumps({"smoke": "discarded_train_100", "frames": context.size,
                          "metrics": v2._metric_float(metrics),
                          "gpu_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}), flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    reference_source = {"cache_manifest_sha256": cache_sha,
                        "twm_sha256": twm_hash, "hawor_sha256": hawor_hash}
    refs = _ensure_references(system, {**cal, **train}, run_dir, reference_source, device)
    baseline = _baseline(system, calibration_contexts, refs, manifest, bound, run_dir, device)
    if args.baseline_only:
        print(json.dumps({"baseline": {"w_mpjpe_mm": baseline["w_mpjpe_mm"],
                                        "events": baseline["events"]}}, sort_keys=True), flush=True)
        return
    optimizer = torch.optim.AdamW(system.observer.visual.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    trainable = {id(p) for p in system.observer.visual.parameters() if p.requires_grad}
    if not trainable or trainable != {id(p) for group in optimizer.param_groups for p in group["params"]}:
        raise RuntimeError("Only visual parameters may be optimized")
    rng = np.random.default_rng(7)
    selection: dict[str, Any] = {"candidate_count": 0, "selection_qualified": False,
                                 "qualified_best_step": None, "best_w_step": None}
    step0, last = 0, None
    if args.resume:
        step0, rng, selection, last = _restore(args.resume, system, optimizer, config, hashes, bound)
    if step0 >= args.steps:
        raise ValueError("Resume step already reached requested steps")
    logs = run_dir / "training.jsonl"
    candidates = run_dir / "calibration_candidates.jsonl"
    if (args.resume is None and (logs.exists() or candidates.exists())) or (args.resume and (not logs.exists() or not candidates.exists())):
        raise FileExistsError("Training logs conflict with fresh/resume mode")
    by_kind = {kind: [w for w in windows if w["kind"] == kind] for kind in TRAIN_FRAME_LIMIT}
    kinds = [kind for kind in TRAIN_FRAME_LIMIT if by_kind[kind]]
    with logs.open("a", encoding="utf-8") as log, candidates.open("a", encoding="utf-8") as candidate_log:
        for step in range(step0+1, args.steps+1):
            kind = kinds[int(rng.integers(len(kinds)))]
            window = by_kind[kind][int(rng.integers(len(by_kind[kind])))]
            context = compose_window(window, train)
            reference = _compose_reference(window, context, train, refs)
            system.observer.visual.train()
            optimizer.zero_grad(set_to_none=True)
            metrics, loss = window_loss(system, context, reference, world, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            parameters = [p for p in system.observer.visual.parameters() if p.requires_grad and p.grad is not None]
            if not parameters or any(not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError(f"Missing or nonfinite visual gradient at step {step}")
            gradnorm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            if not torch.isfinite(gradnorm) or float(gradnorm) <= 0:
                raise FloatingPointError(f"Zero or nonfinite visual gradient at step {step}")
            optimizer.step()
            candidate = None
            if step % CHECK_EVERY == 0:
                system.eval()
                last = v3.evaluate_calibration(system, calibration_contexts, manifest, device)
                is_qualified = qualified(last, baseline)
                selection["candidate_count"] += 1
                score_w = float(last["w_mpjpe_mm"])
                if selection.get("best_w_mpjpe_mm") is None or score_w < selection["best_w_mpjpe_mm"]:
                    selection["best_w_mpjpe_mm"], selection["best_w_step"] = score_w, step
                if is_qualified and (selection.get("qualified_best_w_mpjpe_mm") is None or score_w < selection["qualified_best_w_mpjpe_mm"]):
                    selection["qualified_best_w_mpjpe_mm"] = score_w
                    selection["qualified_best_step"] = step
                    selection["selection_qualified"] = True
                candidate = {"step": step, "qualified": is_qualified,
                             "calibration": last, "baseline": baseline,
                             "selection_state": selection.copy()}
                candidate_log.write(json.dumps(candidate, sort_keys=True) + "\n")
                candidate_log.flush()
                frozen_unchanged = v2._frozen_hashes(system) == frozen
                if not frozen_unchanged:
                    raise RuntimeError("Frozen model parameters changed")
                payload = _checkpoint(system, config, optimizer, step, rng, hashes,
                                      bound, baseline, last, selection.copy())
                checkpoint_path = run_dir / f"checkpoint_{step:04d}.pt"
                if checkpoint_path.exists():
                    raise FileExistsError(checkpoint_path)
                v2._atomic_save(payload, checkpoint_path)
                if is_qualified and selection["qualified_best_step"] == step:
                    v2._atomic_save(payload, run_dir / "qualified_best.pt")
            record = {"step": step, "window": {"sequence": context.sequence,
                                               "kind": kind, "window_index": window["window_index"],
                                               "frame_indices": context.frame_indices.tolist()},
                      "train": v2._metric_float(metrics), "gradient_norm": float(gradnorm.detach().cpu()),
                      "calibration": candidate, "source": bound,
                      "excluded_unsupervised_windows": excluded_unsupervised,
                      "frozen_weights_unchanged": frozen_unchanged if candidate is not None else None}
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    v2._atomic_json({"protocol_revision": 4, **selection,
                     "qualified_checkpoint": (f"checkpoint_{selection['qualified_best_step']:04d}.pt"
                                              if selection["selection_qualified"] else None),
                     "formal_test_authorized": bool(selection["selection_qualified"])},
                    run_dir / "selection.json")


if __name__ == "__main__":
    main()
