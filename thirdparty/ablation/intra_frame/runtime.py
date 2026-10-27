"""Runtime helpers for the trained visual + fixed-TWM intra-frame ablation.

Only the original HaWoR world state is committed as feedback. The observer
performs two candidate-projection rounds on the same image frame.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "try"
ROOT = HERE.parents[2]
TWM_ONLY = HERE.parent / "twm_only"
sys.path[:0] = [str(ROOT), str(PROJECT), str(PROJECT / "scripts"), str(TWM_ONLY)]

from dual_loop.handloop_frontend import project_mano16  # noqa: E402
from dual_loop.handloop_model import HandLoopConfig  # noqa: E402
from dual_loop.handloop_runtime import build_system, move_batch  # noqa: E402
from dual_loop.handloop_state import state_to_legacy, to_world  # noqa: E402
from dual_loop.io import _empty_predictions, save_predictions  # noqa: E402
import run_twm_only as twm_only  # noqa: E402
from run_handloop import context_batch, load_inputs  # noqa: E402

INTRA_ARCHITECTURE = "handloop_intra_v1"
INTRA_FORMAT = "handloop_intra_training"
INTRA_FORMAT_VERSION = 1
INTRA_METHOD = "TWM + Intra-frame Feedback; raw HaWoR history"
BASELINE_METHOD = "TWM only; raw HaWoR feedback"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_intra_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[int, dict[str, Any], dict[str, torch.Tensor]]:
    """Validate the independent visual checkpoint and return usable fields."""
    if checkpoint.get("architecture") != INTRA_ARCHITECTURE:
        raise ValueError(f"Expected architecture={INTRA_ARCHITECTURE!r}")
    if checkpoint.get("format") != INTRA_FORMAT:
        raise ValueError(f"Expected format={INTRA_FORMAT!r}")
    if checkpoint.get("format_version") != INTRA_FORMAT_VERSION:
        raise ValueError(f"Expected format_version={INTRA_FORMAT_VERSION}")
    trainer = checkpoint.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("Intra-frame checkpoint has no trainer metadata")
    try:
        global_step = int(trainer.get("global_step", 0))
    except (TypeError, ValueError):
        global_step = 0
    if global_step <= 0:
        raise ValueError("Intra-frame visual checkpoint has global_step=0; untrained weights are rejected")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Intra-frame checkpoint must contain the original HandLoop config mapping")
    visual = checkpoint.get("visual")
    if not isinstance(visual, dict) or not visual:
        raise ValueError("Intra-frame checkpoint must contain a non-empty visual state dictionary")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in visual.items()):
        raise ValueError("Intra-frame visual weights must be string-keyed tensors")
    for name in ("base_twm_sha256", "base_hawor_sha256"):
        value = checkpoint.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Intra-frame checkpoint must include a SHA256 {name}")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"Invalid SHA256 value for {name}") from error
    return global_step, config, visual


def _inference_config(config: dict[str, Any]) -> dict[str, Any]:
    """Preserve trained module dimensions while fixing this ablation to 2 rounds."""
    result = dict(config)
    result["rounds"] = 2
    result["reobserve"] = True
    return result


def load_intra_system(
    *,
    visual_checkpoint_path: Path,
    twm_checkpoint_path: Path,
    hawor_checkpoint_path: Path,
    device: str | torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load trained visual weights beside byte-identical frozen base models."""
    visual_checkpoint_path = Path(visual_checkpoint_path)
    twm_checkpoint_path = Path(twm_checkpoint_path)
    hawor_checkpoint_path = Path(hawor_checkpoint_path)
    requested = (visual_checkpoint_path, twm_checkpoint_path, hawor_checkpoint_path)
    missing = [str(path) for path in requested if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inference checkpoint(s): {missing}")

    visual_checkpoint = twm_only.load_checkpoint_restricted(visual_checkpoint_path)
    global_step, config, visual_state = validate_intra_checkpoint(visual_checkpoint)
    visual_hash = sha256(visual_checkpoint_path)
    twm_hash = sha256(twm_checkpoint_path)
    hawor_hash = sha256(hawor_checkpoint_path)
    if visual_checkpoint["base_twm_sha256"] != twm_hash:
        raise ValueError("Intra-frame visual checkpoint was trained with a different TWM checkpoint")
    if visual_checkpoint["base_hawor_sha256"] != hawor_hash:
        raise ValueError("Intra-frame visual checkpoint was trained with a different HaWoR checkpoint")

    twm_checkpoint = twm_only.load_checkpoint_restricted(twm_checkpoint_path)
    twm_step = twm_only.validate_trained_checkpoint(twm_checkpoint)
    if twm_checkpoint.get("stage") != "motion_pretrain":
        raise ValueError("The intra-frame ablation requires the fixed motion_pretrain TWM checkpoint")
    if twm_checkpoint.get("architecture") != "handloop_v2":
        raise ValueError("The fixed TWM checkpoint must use architecture='handloop_v2'")
    baseline_config, twm_state = twm_only.twm_state_from_checkpoint(twm_checkpoint)

    observer_config = HandLoopConfig.from_dict(_inference_config(config))
    if observer_config.architecture != "handloop_v2":
        raise ValueError("The visual checkpoint config must describe the original handloop_v2 modules")
    if observer_config.rounds != 2 or not observer_config.reobserve:
        raise ValueError("The Intra-frame observer must use exactly two reobserving rounds")
    frozen_config = twm_only.TWMConfig(**dict(baseline_config.get("twm", {})))
    if asdict(observer_config.twm) != asdict(frozen_config):
        raise ValueError("Visual checkpoint TWM config does not match the frozen TWM checkpoint")

    # build_system loads only the base HaWoR checkpoint. The independent
    # Intra checkpoint contributes visual weights, never alternate frontend weights.
    system = build_system(hawor_checkpoint_path, _inference_config(config), device)
    system.observer.twm.load_state_dict(twm_state, strict=True)
    system.observer.visual.load_state_dict(visual_state, strict=True)
    system.observer.twm.requires_grad_(False)
    system.eval()
    metadata = {
        "architecture": INTRA_ARCHITECTURE,
        "method": INTRA_METHOD,
        "visual_checkpoint": str(visual_checkpoint_path.resolve()),
        "visual_checkpoint_sha256": visual_hash,
        "visual_global_step": global_step,
        "twm_checkpoint": str(twm_checkpoint_path.resolve()),
        "twm_checkpoint_sha256": twm_hash,
        "twm_checkpoint_stage": twm_checkpoint["stage"],
        "twm_global_step": twm_step,
        "hawor_checkpoint": str(hawor_checkpoint_path.resolve()),
        "hawor_checkpoint_sha256": hawor_hash,
        "rounds": 2,
        "reobserve": True,
    }
    return system, metadata


def raw_history_batch(
    raw_world: torch.Tensor,
    raw_available: torch.Tensor,
    timestamp_sec: torch.Tensor,
    segment_id: torch.Tensor,
    handedness: torch.Tensor,
    current_indices: torch.Tensor,
    history_length: int,
) -> dict[str, torch.Tensor]:
    """Build causal, left-padded raw history for a batch of sequence frames.

    Reset rules and padding layout match the first-row batched TWM runner.
    No restored state or predicted uncertainty can enter this input.
    """
    if raw_world.ndim != 2 or raw_world.shape[-1] != 99:
        raise ValueError("raw_world must have shape [N,99]")
    count = len(raw_world)
    if raw_available.shape != (count, 17):
        raise ValueError("raw_available must have shape [N,17]")
    if any(value.shape != (count,) for value in (timestamp_sec, segment_id, handedness)):
        raise ValueError("timestamp_sec, segment_id, and handedness must have shape [N]")
    if current_indices.ndim != 1 or current_indices.numel() == 0:
        raise ValueError("current_indices must be a non-empty [B] tensor")
    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    if int(current_indices.min()) < 0 or int(current_indices.max()) >= count:
        raise IndexError("current_indices fall outside the raw sequence")

    device = raw_world.device
    ids = current_indices.to(device=device, dtype=torch.long)
    history_width = min(int(history_length), int(ids.max().item()))
    resets = torch.zeros(count, dtype=torch.bool, device=device)
    if count > 1:
        gaps = timestamp_sec[1:] - timestamp_sec[:-1]
        resets[1:] = ((gaps <= 0) | (gaps > 0.25)
                      | (segment_id[1:] != segment_id[:-1])
                      | (handedness[1:] != handedness[:-1]))
    frame_ids = torch.arange(count, device=device)
    reset_ids = torch.where(resets, frame_ids, torch.zeros_like(frame_ids))
    latest_reset = torch.cummax(reset_ids, dim=0).values

    if history_width == 0:
        history = raw_world.new_empty(len(ids), 0, 99)
        history_available = torch.empty(len(ids), 0, 17, dtype=torch.bool, device=device)
        history_mask = torch.empty(len(ids), 0, dtype=torch.bool, device=device)
        history_times = raw_world.new_empty(len(ids), 0)
    else:
        offsets = torch.arange(history_width, device=device)
        past_ids = ids[:, None] - history_width + offsets[None, :]
        in_history = (past_ids >= 0) & (past_ids < ids[:, None])
        safe_ids = past_ids.clamp(0, count - 1)
        history = raw_world[safe_ids]
        history_mask = in_history & (past_ids >= latest_reset[ids, None])
        history_available = raw_available[safe_ids].bool() & history_mask[..., None]
        history_times = (timestamp_sec[safe_ids] - timestamp_sec[ids, None]).float()

    previous_ids = (ids - 1).clamp_min(0)
    gaps = (timestamp_sec[ids] - timestamp_sec[previous_ids]).float()
    default_dt = torch.full_like(gaps, 1.0 / 30.0)
    dt = torch.where((ids == 0) | resets[ids], default_dt, gaps)
    return {
        "history": history,
        "history_available": history_available,
        "history_mask": history_mask,
        "history_times": history_times,
        "history_dispersion": raw_world.new_zeros(len(ids), history_width, 17),
        "dispersion_known": torch.zeros(len(ids), history_width, dtype=torch.bool, device=device),
        "dt": dt,
    }


def slam_source_manifest(sequence_root: Path, frame_count: int,
                         interval: int) -> list[dict[str, str]]:
    """Hash exact raw SLAM chunks consumed by the shared HOT3D loader."""
    sources = []
    for start in range(0, frame_count, interval):
        stop = min(start + interval, frame_count)
        path = (Path(sequence_root) / "preprocess" / "SLAM"
                / f"hawor_slam_w_scale_{start}_{stop}.npz")
        if not path.is_file():
            raise FileNotFoundError(f"Missing SLAM source chunk: {path}")
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
    return sources


def _finite_compare(reference: np.ndarray, actual: np.ndarray, *, atol: float,
                    rtol: float, name: str) -> dict[str, float | int]:
    reference = np.asarray(reference)
    actual = np.asarray(actual)
    if reference.shape != actual.shape:
        raise ValueError(f"Baseline {name} shape mismatch: {reference.shape} vs {actual.shape}")
    finite_reference = np.isfinite(reference)
    finite_actual = np.isfinite(actual)
    if not np.array_equal(finite_reference, finite_actual):
        raise ValueError(f"Baseline {name} finite-value mask differs from current frontend output")
    if not finite_reference.any():
        return {"finite_values": 0, "max_abs_error": 0.0, "max_abs_relative_error": 0.0}
    left = reference[finite_reference].astype(np.float64)
    right = actual[finite_actual].astype(np.float64)
    difference = np.abs(left - right)
    if not np.allclose(left, right, atol=atol, rtol=rtol):
        raise ValueError(
            f"Current {name} does not reproduce archived TWM-only raw frontend input; "
            f"max_abs_error={float(difference.max())}, atol={atol}, rtol={rtol}")
    relative = difference / np.maximum(np.abs(left), 1e-12)
    return {"finite_values": int(len(left)), "max_abs_error": float(difference.max()),
            "max_abs_relative_error": float(relative.max())}


def verify_baseline_inputs(
    baseline_sequence_dir: Path,
    *,
    sequence: str,
    inputs: dict[str, Any],
    raw_world: np.ndarray,
    betas: np.ndarray,
    twm_checkpoint_sha256: str,
    hawor_checkpoint_sha256: str,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Require the live frontend and camera trajectory to match row one's inputs."""
    baseline_sequence_dir = Path(baseline_sequence_dir)
    metadata_path = baseline_sequence_dir / "try_inference.json"
    trace_path = baseline_sequence_dir / "twm_only_trace.npz"
    if not metadata_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError(f"Missing archived TWM-only baseline for {sequence}: {baseline_sequence_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "method": BASELINE_METHOD,
        "checkpoint_sha256": twm_checkpoint_sha256,
        "hawor_checkpoint_sha256": hawor_checkpoint_sha256,
        "frontend_source_sha256": hawor_checkpoint_sha256,
        "track_sha256": inputs["track_sha256"],
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items()
                  if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"Baseline metadata does not bind same TWM/HaWoR/tracks: {mismatches}")
    if not metadata.get("oracle_assisted") or not metadata.get("baseline_frontend_scope_matches_frozen_pinch_protocol"):
        raise ValueError("Frozen HOT3D row-two protocol requires verified GT-projected boxes")
    if not inputs.get("oracle_assisted"):
        raise ValueError("Current inference did not load the frozen GT-projected track boxes")

    with np.load(trace_path, allow_pickle=False) as archive:
        saved_raw = archive["raw_state_world"].copy()
        saved_betas = archive["betas"].copy()
        saved_ns = archive["timestamp_ns"].copy()
        saved_seconds = archive["timestamp_sec"].copy()
        saved_segments = archive["segment_id"].copy()
    if len(saved_raw) != len(raw_world) or len(saved_betas) != len(betas):
        raise ValueError(f"Baseline frame count differs for {sequence}")
    if not np.array_equal(saved_ns, inputs["ns"]) or not np.array_equal(saved_segments, inputs["segments"]):
        raise ValueError(f"Baseline timestamps/SLAM interval segments differ for {sequence}")
    if not np.allclose(saved_seconds, inputs["seconds"], rtol=0, atol=1e-9):
        raise ValueError(f"Baseline timestamps differ for {sequence}")
    return {
        "sequence": sequence,
        "baseline_metadata_sha256": sha256(metadata_path),
        "baseline_trace_sha256": sha256(trace_path),
        "raw_state_world": _finite_compare(saved_raw, raw_world, atol=atol, rtol=rtol,
                                           name="raw_state_world"),
        "betas": _finite_compare(saved_betas, betas, atol=atol, rtol=rtol, name="betas"),
    }


def infer_sequence(
    system: torch.nn.Module,
    sequence_root: Path,
    *,
    sequence: str,
    baseline_sequence_dir: Path,
    output_dir: Path,
    device: str | torch.device,
    interval: int,
    max_frames: int | None,
    base_metadata: dict[str, Any],
    visual_checkpoint_path: Path,
    allow_gt_boxes: bool = True,
    history_batch_size: int = 16,
) -> dict[str, Any]:
    """Run two-round visual feedback and save row-one-compatible predictions."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite intra-frame results: {output_dir}")
    if interval < 1 or history_batch_size < 1:
        raise ValueError("interval and history_batch_size must be positive")
    inputs = load_inputs(Path(sequence_root), allow_gt_boxes, interval)
    frame_count = len(inputs["images"])
    frames = min(frame_count, max_frames) if max_frames is not None else frame_count
    if frames < 1:
        raise ValueError("max_frames must be positive")
    if frames != frame_count:
        raise ValueError("Baseline-verified intra inference requires every sequence frame")

    geometry = system.geometry
    timestamps = torch.as_tensor(inputs["seconds"], device=device, dtype=torch.float64)
    segments = torch.as_tensor(inputs["segments"], device=device, dtype=torch.long)
    handedness = torch.ones(frame_count, dtype=torch.long, device=device)
    raw_world = torch.full((frame_count, 99), float("nan"), device=device)
    raw_available = torch.from_numpy(
        np.repeat(inputs["valid"][:, None], 17, axis=1)).to(device=device, dtype=torch.bool)
    raw_betas = torch.full((frame_count, 10), float("nan"), device=device)
    raw_pred_cam = torch.full((frame_count, 3), float("nan"), device=device)
    restored = torch.full_like(raw_world, float("nan"))
    predictions = _empty_predictions(frame_count, inputs["size"])
    predictions["bbox"][:] = inputs["boxes"]
    predictions["focal_length"][:] = inputs["focal"]

    # Match the first-row 16-frame HaWoR context and SLAM reset boundaries.
    with torch.inference_mode():
        start = 0
        while start < frame_count:
            end = min(start + 16, ((start // interval) + 1) * interval, frame_count)
            real = end - start
            ids = list(range(start, end)) + [end - 1] * (16 - real)
            batch = move_batch(context_batch(inputs, ids), device)
            front = system.hawor(batch["hawor"])
            raw_camera = front["state_camera"][0, :real]
            rwc = batch["rotation_w2c"][0, :real]
            twc = batch["translation_w2c"][0, :real]
            rcw = rwc.transpose(-1, -2)
            tcw = -(rcw @ twc[..., None]).squeeze(-1)
            chunk_raw = to_world(raw_camera, rcw, tcw)
            raw_world[start:end] = chunk_raw
            raw_betas[start:end] = front["betas"][0, :real]
            raw_pred_cam[start:end] = front["pred_cam"][0, :real]
            if not torch.isfinite(chunk_raw).all() or not torch.isfinite(raw_betas[start:end]).all():
                raise ValueError(f"HaWoR produced a non-finite raw frame in {sequence} at {start}:{end}")

            frame_ids = torch.arange(start, end, device=device)
            history = raw_history_batch(
                raw_world[:end], raw_available[:end], timestamps[:end], segments[:end],
                handedness[:end], frame_ids, system.observer.config.twm.history_length)
            for sub_start in range(0, real, history_batch_size):
                sub_stop = min(sub_start + history_batch_size, real)
                local_ids = frame_ids[sub_start:sub_stop]
                select = slice(sub_start, sub_stop)
                current_rotation = rwc[select]
                current_translation = twc[select]
                beta = front["betas"][0, :real][select]
                focal = batch["hawor"]["img_focal"][0, :real][select]
                principal = batch["hawor"]["img_center"][0, :real][select]
                pixel_transform = batch["pixel_to_patch"][0, :real][select]

                def projector(state, *, beta=beta, rotation=current_rotation,
                              translation=current_translation, f=focal, center=principal,
                              transform=pixel_transform):
                    return project_mano16(state, beta, geometry, rotation, translation,
                                          f, center, transform)

                subset_history = {key: value[select] for key, value in history.items()
                                  if key != "dt"}
                output = system.observer.forward_step(
                    patches=front["patches"][0, :real][select],
                    raw_state=raw_world[local_ids],
                    history=subset_history["history"],
                    raw_available=raw_available[local_ids],
                    history_available=subset_history["history_available"],
                    history_mask=subset_history["history_mask"],
                    history_times=subset_history["history_times"],
                    dt=history["dt"][select],
                    handedness=handedness[local_ids],
                    history_dispersion=subset_history["history_dispersion"],
                    dispersion_known=subset_history["dispersion_known"],
                    rotation_w2c=current_rotation,
                    translation_w2c=current_translation,
                    projector=projector,
                    image_available=batch["image_available"][0, :real][select],
                    future_times=None,
                )
                for offset, index in enumerate(local_ids.tolist()):
                    state = output["state"][offset]
                    restored[index] = state
                    if inputs["valid"][index]:
                        legacy = state_to_legacy(state[None], beta[offset:offset + 1], geometry)
                        trans, root, hand_pose, shape = geometry.world_to_camera_parameters(
                            legacy, current_rotation[offset:offset + 1],
                            current_translation[offset:offset + 1])
                        predictions["camera_translation"][index] = trans[0].cpu().numpy()
                        predictions["global_orient"][index] = root[0].cpu().numpy()
                        predictions["hand_pose"][index] = hand_pose[0].cpu().numpy()
                        predictions["betas"][index] = shape[0].cpu().numpy()
                        predictions["pred_cam"][index] = raw_pred_cam[index].cpu().numpy()
                        predictions["valid"][index] = True
            start = end
            print(f"[{sequence}] Intra-frame raw/restore {end}/{frame_count}", flush=True)

    baseline_check = verify_baseline_inputs(
        baseline_sequence_dir, sequence=sequence, inputs=inputs,
        raw_world=raw_world.cpu().numpy(), betas=raw_betas.cpu().numpy(),
        twm_checkpoint_sha256=base_metadata["twm_checkpoint_sha256"],
        hawor_checkpoint_sha256=base_metadata["hawor_checkpoint_sha256"],
    )
    slam_sources = slam_source_manifest(Path(sequence_root), frame_count, interval)
    output_dir.mkdir(parents=True)
    save_predictions(output_dir / "try_predictions.npz", predictions)
    np.savez_compressed(
        output_dir / "intra_frame_trace.npz",
        frame_id=np.arange(frame_count),
        timestamp_ns=inputs["ns"],
        timestamp_sec=inputs["seconds"],
        segment_id=inputs["segments"],
        state_world=restored.cpu().numpy(),
        raw_state_world=raw_world.cpu().numpy(),
        visual_valid=predictions["valid"],
        betas=raw_betas.cpu().numpy(),
        architecture=np.asarray(INTRA_ARCHITECTURE),
        coordinate_frame=np.asarray("per-interval raw SLAM world; eval_hot3d aligns camera trajectory"),
    )
    metadata = {
        **base_metadata,
        "sequence": sequence,
        "method": INTRA_METHOD,
        "trained_checkpoint_loaded": True,
        "visual_loop_bypassed": False,
        "twm_predictions_per_frame": 2,
        "twm_calls_per_frame": 2,
        "twm_calls_per_observer_batch": 2,
        "twm_batch_size": history_batch_size,
        "history_commit": "original raw HaWoR world state; observer output and dispersion are never committed",
        "history_source": "raw_state_world only",
        "history_dispersion_input": "zeros with dispersion_known=false; no uncertainty feedback",
        "hawor_bidirectional_context": 16,
        "total_frames": frame_count,
        "processed_frames": frame_count,
        "oracle_assisted": bool(inputs["oracle_assisted"]),
        "allow_gt_boxes": bool(allow_gt_boxes),
        "track_file": inputs["track_file"],
        "track_sha256": inputs["track_sha256"],
        "slam_interval": interval,
        "slam_source_chunks": slam_sources,
        "baseline_sequence_dir": str(Path(baseline_sequence_dir).resolve()),
        "baseline_raw_input_verification": baseline_check,
        "baseline_frontend_scope_matches_frozen_pinch_protocol": bool(inputs["oracle_assisted"]),
        "raw_trace": str((output_dir / "intra_frame_trace.npz").resolve()),
        "visual_checkpoint_used": str(Path(visual_checkpoint_path).resolve()),
        "frontend_weights_source": "--hawor_checkpoint; independently trained visual state only",
        "frontend_source_sha256": base_metadata["hawor_checkpoint_sha256"],
        "coordinate_frame": "per-interval raw SLAM world; eval_hot3d aligns camera trajectory",
    }
    (output_dir / "try_inference.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata

