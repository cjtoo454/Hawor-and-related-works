#!/usr/bin/env python3
"""Run the HandLoop v2 TWM once per frame on original HaWoR states.

The visual correction loop is bypassed. Every completed frame commits the raw
HaWoR world state to history, regardless of the restored TWM output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "try"
ROOT = HERE.parents[2]
SCRIPTS = PROJECT / "scripts"
sys.path[:0] = [str(ROOT), str(PROJECT), str(SCRIPTS)]

from dual_loop.geometry import ManoGeometry  # noqa: E402
from dual_loop.handloop_frontend import HaWoRFrontend, load_hawor  # noqa: E402
from dual_loop.handloop_state import make_anchor, state_to_legacy, to_local, to_world  # noqa: E402
from dual_loop.handloop_twm import TWMConfig, TrajectoryRestorer  # noqa: E402
from dual_loop.io import _empty_predictions, _resolve_tracks_path, save_predictions  # noqa: E402
from run_handloop import context_batch, load_inputs  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy_checkpoint_globals() -> list[Any]:
    """Allow only NumPy globals used by the repository's saved RNG state."""
    allowed = [
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    for module_name in ("numpy.core.multiarray", "numpy._core.multiarray"):
        try:
            allowed.append(importlib.import_module(module_name)._reconstruct)
        except (ImportError, AttributeError):
            continue
    return allowed


def load_checkpoint_restricted(path: Path) -> dict[str, Any]:
    """Load a normal HandLoop training checkpoint without unrestricted pickle."""
    with torch.serialization.safe_globals(_numpy_checkpoint_globals()):
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    return value


def twm_state_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if checkpoint.get("architecture") != "handloop_v2":
        raise ValueError("Expected architecture='handloop_v2'")
    stage = checkpoint.get("stage")
    config = checkpoint.get("config")
    if stage not in {"motion_pretrain", "joint_visual"}:
        raise ValueError("Checkpoint stage must be motion_pretrain or joint_visual")
    if not isinstance(config, dict) or not isinstance(config.get("twm", {}), dict):
        raise ValueError("Checkpoint must contain a HandLoop v2 config with a twm mapping")
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise ValueError("Checkpoint has no model state dictionary")

    if stage == "motion_pretrain":
        state = model
    else:
        prefix = "observer.twm."
        state = {key[len(prefix):]: tensor for key, tensor in model.items()
                 if isinstance(key, str) and key.startswith(prefix)}
        if not state:
            raise ValueError("joint_visual checkpoint has no observer.twm.* weights")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("TWM checkpoint weights must be tensors")
    return config, state


def frontend_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return jointly trained HaWoRFrontend weights, excluding non-frontend modules."""
    if checkpoint.get("stage") != "joint_visual":
        return {}
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise ValueError("joint_visual checkpoint has no model state dictionary")
    prefix = "hawor."
    state = {key[len(prefix):]: value for key, value in model.items()
             if isinstance(key, str) and key.startswith(prefix)}
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("joint_visual checkpoint has no valid hawor.* weights")
    return state


def validate_trained_checkpoint(checkpoint: dict[str, Any]) -> int:
    if (checkpoint.get("format") != "handloop_training"
            or checkpoint.get("format_version") != 2):
        raise ValueError("Expected a HandLoop v2 training checkpoint")
    trainer = checkpoint.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("Checkpoint has no training-step metadata")
    try:
        step = int(trainer.get("global_step", 0))
    except (TypeError, ValueError):
        step = 0
    if step <= 0:
        raise ValueError("Checkpoint has global_step=0; untrained weights are not accepted")
    return step


def load_twm(checkpoint_path: Path, device: str | torch.device):
    checkpoint = load_checkpoint_restricted(checkpoint_path)
    validate_trained_checkpoint(checkpoint)
    config, state = twm_state_from_checkpoint(checkpoint)
    model = TrajectoryRestorer(TWMConfig(**dict(config.get("twm", {})))).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


class RawFeedbackHistory:
    """Bounded causal history whose only commit source is the raw observation."""

    def __init__(self, length: int) -> None:
        if length < 0:
            raise ValueError("history length must be non-negative")
        self.length = int(length)
        self.states: list[torch.Tensor] = []
        self.available: list[torch.Tensor] = []
        self.times: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []
        self.last_time: torch.Tensor | None = None
        self.last_segment: torch.Tensor | None = None
        self.last_hand: torch.Tensor | None = None

    def before(self, time: torch.Tensor, segment: torch.Tensor,
               hand: torch.Tensor) -> torch.Tensor:
        reset = torch.zeros_like(time, dtype=torch.bool)
        if self.last_time is None:
            dt = time.new_full(time.shape, 1.0 / 30.0).float()
        else:
            gap = time - self.last_time
            reset = ((gap <= 0) | (gap > 0.25)
                     | (segment != self.last_segment) | (hand != self.last_hand))
            self.masks = [mask & ~reset for mask in self.masks]
            dt = gap.float()
            dt = torch.where(reset, torch.full_like(dt, 1.0 / 30.0), dt)
        self.last_time = time.detach()
        self.last_segment = segment.detach()
        self.last_hand = hand.detach()
        return dt

    def tensors(self, raw: torch.Tensor, time: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = raw.shape[0]
        device = raw.device
        if not self.states:
            return {
                "history": raw.new_empty(batch, 0, 99),
                "history_available": torch.empty(batch, 0, 17, dtype=torch.bool, device=device),
                "history_mask": torch.empty(batch, 0, dtype=torch.bool, device=device),
                "history_times": raw.new_empty(batch, 0),
            }
        states = torch.stack(self.states, dim=1)
        available = torch.stack(self.available, dim=1)
        masks = torch.stack(self.masks, dim=1)
        times = torch.stack(self.times, dim=1)
        return {
            "history": states,
            "history_available": available & masks[..., None],
            "history_mask": masks,
            "history_times": (times - time[:, None]).float(),
        }

    def commit_raw(self, raw: torch.Tensor, available: torch.Tensor,
                   time: torch.Tensor) -> None:
        if self.length == 0:
            return
        for store, value in (
            (self.states, raw),
            (self.available, available.bool()),
            (self.times, time),
            (self.masks, torch.ones_like(time, dtype=torch.bool)),
        ):
            store.append(value.detach())
            if len(store) > self.length:
                del store[:-self.length]


def restore_raw_sequence(
    twm: torch.nn.Module,
    raw_world: torch.Tensor,
    raw_available: torch.Tensor,
    timestamp_sec: torch.Tensor,
    segment_id: torch.Tensor,
    handedness: torch.Tensor,
) -> torch.Tensor:
    """Restore frames once, using original raw states for both current and history."""
    if raw_world.ndim != 2 or raw_world.shape[-1] != 99:
        raise ValueError("raw_world must have shape [N,99]")
    count = len(raw_world)
    if raw_available.shape != (count, 17):
        raise ValueError("raw_available must have shape [N,17]")
    if any(value.shape != (count,) for value in (timestamp_sec, segment_id, handedness)):
        raise ValueError("timestamp_sec, segment_id, and handedness must have shape [N]")
    history = RawFeedbackHistory(int(twm.config.history_length))
    restored: list[torch.Tensor] = []
    for index in range(count):
        raw = raw_world[index:index + 1]
        available = raw_available[index:index + 1].bool()
        time = timestamp_sec[index:index + 1]
        segment = segment_id[index:index + 1]
        hand = handedness[index:index + 1]
        dt = history.before(time, segment, hand)
        past = history.tensors(raw, time)
        anchor_rotation, anchor_position = make_anchor(
            raw, past["history"], available, past["history_available"])
        history_anchor = to_local(past["history"], anchor_rotation, anchor_position)
        raw_anchor = to_local(raw, anchor_rotation, anchor_position)
        prediction = twm(
            history=history_anchor,
            raw=raw_anchor,
            candidate=raw_anchor,
            history_available=past["history_available"],
            raw_available=available,
            candidate_available=available,
            history_times=past["history_times"],
            dt=dt,
            handedness=hand.reshape(1).long().clamp(0, 1),
            history_mask=past["history_mask"],
            future_times=None,
        )
        current = prediction["current"]["state"]
        restored_world = to_world(current, anchor_rotation, anchor_position)
        restored.append(restored_world[0])
        # This is intentionally the raw input, never restored_world/current.
        history.commit_raw(raw, available, time)
    if not restored:
        return raw_world.new_empty(0, 99)
    return torch.stack(restored)


def restore_raw_sequence_batched(
    twm: torch.nn.Module,
    raw_world: torch.Tensor,
    raw_available: torch.Tensor,
    timestamp_sec: torch.Tensor,
    segment_id: torch.Tensor,
    handedness: torch.Tensor,
    batch_size: int = 8,
) -> torch.Tensor:
    """Restore raw frames in batches while preserving causal per-frame inputs.

    Every row still gets exactly one TWM call-equivalent prediction.  Its
    history is assembled exclusively from earlier raw observations, and a
    discontinuity invalidates every history entry before that boundary just as
    ``RawFeedbackHistory.before`` does in :func:`restore_raw_sequence`.
    """
    if raw_world.ndim != 2 or raw_world.shape[-1] != 99:
        raise ValueError("raw_world must have shape [N,99]")
    count = len(raw_world)
    if raw_available.shape != (count, 17):
        raise ValueError("raw_available must have shape [N,17]")
    if any(value.shape != (count,) for value in (timestamp_sec, segment_id, handedness)):
        raise ValueError("timestamp_sec, segment_id, and handedness must have shape [N]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if count == 0:
        return raw_world.new_empty(0, 99)

    history_length = int(twm.config.history_length)
    if history_length < 0:
        raise ValueError("TWM history_length must be non-negative")

    # The old sequential buffer clears its existing masks when the current
    # frame is discontinuous, then commits the current raw frame as valid.
    # Therefore only entries at or after the latest reset can be used.
    frame_ids = torch.arange(count, device=raw_world.device)
    reset = torch.zeros(count, dtype=torch.bool, device=raw_world.device)
    if count > 1:
        gaps = timestamp_sec[1:] - timestamp_sec[:-1]
        reset[1:] = (
            (gaps <= 0)
            | (gaps > 0.25)
            | (segment_id[1:] != segment_id[:-1])
            | (handedness[1:] != handedness[:-1])
        )
    reset_at = torch.where(reset, frame_ids, torch.zeros_like(frame_ids))
    latest_reset = torch.cummax(reset_at, dim=0).values

    restored: list[torch.Tensor] = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        current_ids = frame_ids[start:stop]
        raw = raw_world[start:stop]
        available = raw_available[start:stop].bool()
        time = timestamp_sec[start:stop]
        segments = segment_id[start:stop]
        hands = handedness[start:stop]

        max_history = min(history_length, stop - 1)
        if max_history == 0:
            # A zero-width history is valid for a configured history length
            # of zero and for the first frame of the sequence.
            history = raw.new_empty(len(raw), 0, 99)
            history_available = torch.empty(
                len(raw), 0, 17, dtype=torch.bool, device=raw.device)
            history_mask = torch.empty(len(raw), 0, dtype=torch.bool, device=raw.device)
            history_times = raw.new_empty(len(raw), 0)
        else:
            # Use one rectangular tensor per model call.  Rows with shorter
            # histories are left-padded, which is inert because all padding
            # masks are false and the TWM's temporal scan skips invalid rows.
            offsets = torch.arange(max_history, device=raw.device)
            past_ids = current_ids[:, None] - max_history + offsets[None, :]
            in_history = (past_ids >= 0) & (past_ids < current_ids[:, None])
            safe_ids = past_ids.clamp(0, count - 1)
            history = raw_world[safe_ids]
            history_mask = in_history & (past_ids >= latest_reset[current_ids, None])
            history_available = raw_available[safe_ids].bool() & history_mask[..., None]
            history_times = (timestamp_sec[safe_ids] - time[:, None]).float()

        previous_ids = (current_ids - 1).clamp_min(0)
        gaps = (time - timestamp_sec[previous_ids]).float()
        default_dt = torch.full_like(gaps, 1.0 / 30.0)
        dt = torch.where((current_ids == 0) | reset[current_ids], default_dt, gaps)

        anchor_rotation, anchor_position = make_anchor(
            raw, history, available, history_available)
        history_anchor = to_local(history, anchor_rotation, anchor_position)
        raw_anchor = to_local(raw, anchor_rotation, anchor_position)
        prediction = twm(
            history=history_anchor,
            raw=raw_anchor,
            candidate=raw_anchor,
            history_available=history_available,
            raw_available=available,
            candidate_available=available,
            history_times=history_times,
            dt=dt,
            handedness=hands.reshape(-1).long().clamp(0, 1),
            history_mask=history_mask,
            future_times=None,
        )
        restored.append(to_world(
            prediction["current"]["state"], anchor_rotation, anchor_position))

    return torch.cat(restored, dim=0)


def _encode_raw(frontend: HaWoRFrontend, hawor: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    frames = hawor["img"].shape[1]
    if frames % 16:
        raise ValueError("HaWoR frontend batch must be padded to a multiple of 16")
    outputs = []
    for start in range(0, frames, 16):
        outputs.append(frontend({key: value[:, start:start + 16]
                                 for key, value in hawor.items()}))
    return {key: torch.cat([output[key] for output in outputs], dim=1)
            for key in outputs[0]}


def _runtime_checks(args: argparse.Namespace) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for name, path in (
        ("checkpoint", args.checkpoint),
        ("HaWoR checkpoint", args.hawor_checkpoint),
        ("HOT3D video root", args.video_root),
        ("MANO right-hand asset", ROOT / "_DATA/data/mano/MANO_RIGHT.pkl"),
    ):
        checks.append({"name": name, "path": str(path),
                       "status": "ok" if path.exists() else "missing"})
    set_path = args.video_root / args.set_file
    checks.append({"name": "sequence set file", "path": str(set_path),
                   "status": "ok" if set_path.is_file() else "missing"})
    if args.interval < 1:
        checks.append({"name": "SLAM interval", "path": str(args.interval), "status": "invalid: must be positive"})
    if args.twm_batch_size < 1:
        checks.append({"name": "TWM batch size", "path": str(args.twm_batch_size),
                       "status": "invalid: must be positive"})
    if str(args.device).startswith("cuda"):
        available = torch.cuda.is_available()
        index_text = str(args.device).partition(":")[2]
        valid_index = (not index_text or (index_text.isdigit()
                       and int(index_text) < torch.cuda.device_count()))
        checks.append({"name": "CUDA device", "path": str(args.device),
                       "status": "ok" if available and valid_index else "unavailable"})
    if set_path.is_file():
        try:
            sequences = json.loads(set_path.read_text(encoding="utf-8"))
            if args.video:
                unknown = sorted(set(args.video) - set(sequences))
                if unknown:
                    checks.append({"name": "selected sequences", "path": str(set_path),
                                   "status": f"invalid: unknown sequence(s) {unknown}"})
                    sequences = []
                else:
                    sequences = [sequence for sequence in sequences if sequence in args.video]
            for sequence in sequences:
                sequence_root = args.video_root / sequence
                images = sorted((sequence_root / "extracted_images").glob("*.jpg"))
                checks.append({"name": f"{sequence} images", "path": str(sequence_root / "extracted_images"),
                               "status": "ok" if images else "missing"})
                for name in ("focal.txt", "tt_list.pkl"):
                    path = sequence_root / name
                    checks.append({"name": f"{sequence} {name}", "path": str(path),
                                   "status": "ok" if path.is_file() else "missing"})
                try:
                    track_path, oracle_assisted = _resolve_tracks_path(sequence_root, args.allow_gt_boxes)
                    checks.append({"name": f"{sequence} tracks", "path": str(track_path),
                                   "status": "ok" if track_path.is_file() else "missing",
                                   "source": "GT-projected" if oracle_assisted else "detected"})
                except (FileNotFoundError, ValueError) as error:
                    checks.append({"name": f"{sequence} tracks", "path": str(sequence_root / "preprocess"),
                                   "status": f"missing: {error}"})
                if images:
                    for start in range(0, len(images), max(1, args.interval)):
                        stop = min(start + max(1, args.interval), len(images))
                        path = (sequence_root / "preprocess" / "SLAM"
                                / f"hawor_slam_w_scale_{start}_{stop}.npz")
                        checks.append({"name": f"{sequence} SLAM {start}:{stop}", "path": str(path),
                                       "status": "ok" if path.is_file() else "missing"})
        except (OSError, json.JSONDecodeError, TypeError) as error:
            checks.append({"name": "sequence inputs", "path": str(set_path),
                           "status": f"invalid: {type(error).__name__}: {error}"})
    if args.checkpoint.is_file():
        try:
            checkpoint = load_checkpoint_restricted(args.checkpoint)
            validate_trained_checkpoint(checkpoint)
            twm_state_from_checkpoint(checkpoint)
            frontend_state_from_checkpoint(checkpoint)
            checks.append({"name": "HandLoop v2 checkpoint format", "path": str(args.checkpoint),
                           "status": "ok"})
        except Exception as error:
            checks.append({"name": "HandLoop v2 checkpoint format", "path": str(args.checkpoint),
                           "status": f"invalid: {type(error).__name__}: {error}"})
    return checks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="HandLoop v2 motion_pretrain or joint_visual checkpoint")
    parser.add_argument("--hawor_checkpoint", type=Path,
                        default=ROOT / "weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--video_root", type=Path,
                        default=ROOT / "datasets/hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--video", action="append")
    parser.add_argument("--output_root", type=Path,
                        default=ROOT / "eval_log_twm_only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--twm_batch_size", type=int, default=8,
                        help="Number of independent raw-history frames per TWM call (default: 8)")
    parser.add_argument("--allow_gt_boxes", action="store_true",
                        help="Accept legacy GT-projected boxes; metadata marks the run oracle-assisted")
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--preflight", action="store_true",
                        help="Print resource/checkpoint status without initializing HaWoR or MANO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checks = _runtime_checks(args)
    report = {"ready": all(item["status"] == "ok" for item in checks), "checks": checks}
    if args.preflight:
        print(json.dumps(report, indent=2))
        if not report["ready"]:
            raise SystemExit(2)
        return
    if not report["ready"]:
        print(json.dumps(report, indent=2), file=sys.stderr)
        raise SystemExit("TWM-only inference preflight failed; provide all listed resources")
    if args.interval < 1:
        raise ValueError("interval must be positive")
    if args.twm_batch_size < 1:
        raise ValueError("twm_batch_size must be positive")
    checkpoint, loaded = load_twm(args.checkpoint, args.device)
    checkpoint_hash = sha256(args.checkpoint)
    hawor_hash = sha256(args.hawor_checkpoint)
    set_path = args.video_root / args.set_file
    sequences = json.loads(set_path.read_text(encoding="utf-8"))
    if args.video:
        unknown = set(args.video) - set(sequences)
        if unknown:
            raise ValueError(f"Unknown sequences: {sorted(unknown)}")
        sequences = [sequence for sequence in sequences if sequence in args.video]
    if not sequences:
        raise ValueError("No sequences selected")

    geometry = ManoGeometry(ROOT).to(args.device).eval()
    frontend = HaWoRFrontend(load_hawor(args.hawor_checkpoint, args.device), geometry).to(args.device).eval()
    frontend_state = frontend_state_from_checkpoint(loaded)
    if frontend_state:
        frontend.load_state_dict(frontend_state, strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        output_dir = args.output_root / sequence
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite TWM-only results: {output_dir}")
        inputs = load_inputs(args.video_root / sequence, args.allow_gt_boxes, args.interval)
        frame_count = len(inputs["images"])
        frames = min(frame_count, args.max_frames) if args.max_frames is not None else frame_count
        if frames < 1:
            raise ValueError("max_frames must be positive")

        camera_state = np.full((frames, 99), np.nan, np.float32)
        betas = np.full((frames, 10), np.nan, np.float32)
        pred_cam = np.full((frames, 3), np.nan, np.float32)
        with torch.inference_mode():
            start = 0
            while start < frames:
                # Keep each 16-frame HaWoR context within a SLAM segment.
                end = min(start + 16, ((start // args.interval) + 1) * args.interval, frames)
                real = end - start
                ids = list(range(start, end)) + [end - 1] * (16 - real)
                batch = context_batch(inputs, ids)
                hawor = {key: value.to(args.device) for key, value in batch["hawor"].items()}
                output = _encode_raw(frontend, hawor)
                camera_state[start:end] = output["state_camera"][0, :real].cpu().numpy()
                betas[start:end] = output["betas"][0, :real].cpu().numpy()
                pred_cam[start:end] = output["pred_cam"][0, :real].cpu().numpy()
                start = end
                print(f"[{sequence}] HaWoR raw {end}/{frames}", flush=True)

        rotation_w2c = inputs["rotation_w2c"][:frames].to(args.device)
        translation_w2c = inputs["translation_w2c"][:frames].to(args.device)
        rotation_c2w = rotation_w2c.transpose(-1, -2)
        translation_c2w = -(rotation_c2w @ translation_w2c[..., None]).squeeze(-1)
        raw_camera = torch.as_tensor(camera_state, device=args.device)
        raw_world = to_world(raw_camera, rotation_c2w, translation_c2w)
        raw_available = torch.from_numpy(
            np.repeat(inputs["valid"][:frames, None], 17, axis=1)).to(args.device)
        timestamps = torch.from_numpy(inputs["seconds"][:frames].copy()).to(args.device)
        segments = torch.from_numpy(inputs["segments"][:frames].copy()).to(args.device)
        hands = torch.ones(frames, dtype=torch.long, device=args.device)
        with torch.inference_mode():
            restored = restore_raw_sequence_batched(
                checkpoint, raw_world, raw_available, timestamps, segments, hands,
                batch_size=args.twm_batch_size)

        predictions = _empty_predictions(frame_count, inputs["size"])
        predictions["bbox"][:] = inputs["boxes"]
        predictions["focal_length"][:] = inputs["focal"]
        predictions["pred_cam"][:frames] = pred_cam
        valid_ids = np.flatnonzero(inputs["valid"][:frames])
        if len(valid_ids):
            ids = torch.as_tensor(valid_ids, device=args.device)
            world_state = restored[ids]
            shape = torch.as_tensor(betas[valid_ids], device=args.device)
            legacy = state_to_legacy(world_state, shape, geometry)
            cam_translation, root, hand_pose, shape = geometry.world_to_camera_parameters(
                legacy, rotation_w2c[ids], translation_w2c[ids])
            predictions["camera_translation"][valid_ids] = cam_translation.cpu().numpy()
            predictions["global_orient"][valid_ids] = root.cpu().numpy()
            predictions["hand_pose"][valid_ids] = hand_pose.cpu().numpy()
            predictions["betas"][valid_ids] = shape.cpu().numpy()
            predictions["valid"][valid_ids] = True

        output_dir.mkdir(parents=True)
        save_predictions(output_dir / "try_predictions.npz", predictions)
        trace = np.full((frame_count, 99), np.nan, np.float32)
        trace[:frames] = restored.cpu().numpy()
        raw_trace = np.full((frame_count, 99), np.nan, np.float32)
        raw_trace[:frames] = raw_world.cpu().numpy()
        np.savez_compressed(
            output_dir / "twm_only_trace.npz",
            frame_id=np.arange(frame_count),
            timestamp_ns=inputs["ns"],
            timestamp_sec=inputs["seconds"],
            segment_id=inputs["segments"],
            state_world=trace,
            raw_state_world=raw_trace,
            visual_valid=predictions["valid"],
            betas=np.pad(betas, ((0, frame_count - frames), (0, 0)), constant_values=np.nan),
        )
        metadata = {
            "sequence": sequence,
            "method": "TWM only; raw HaWoR feedback",
            "architecture": "handloop_v2",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_stage": loaded["stage"],
            "checkpoint_architecture": loaded["architecture"],
            "checkpoint_global_step": validate_trained_checkpoint(loaded),
            "hawor_checkpoint": str(args.hawor_checkpoint.resolve()),
            "hawor_checkpoint_sha256": hawor_hash,
            "frontend_weights_source": (
                "joint_visual checkpoint hawor.*" if frontend_state else "--hawor_checkpoint"),
            "frontend_source_sha256": checkpoint_hash if frontend_state else hawor_hash,
            "trained_checkpoint_loaded": True,
            "visual_loop_bypassed": True,
            "twm_predictions_per_frame": 1,
            "twm_calls_per_batch": 1,
            "twm_batch_size": args.twm_batch_size,
            "history_commit": "original raw HaWoR world state; restored output is never committed",
            "total_frames": frame_count,
            "processed_frames": frames,
            "oracle_assisted": bool(inputs["oracle_assisted"]),
            "allow_gt_boxes": bool(args.allow_gt_boxes),
            "track_file": inputs["track_file"],
            "track_sha256": inputs["track_sha256"],
            "coordinate_frame": "per-interval raw SLAM world; eval_hot3d aligns camera trajectory",
            "baseline_frontend_scope_matches_frozen_pinch_protocol": bool(inputs["oracle_assisted"]),
        }
        (output_dir / "try_inference.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[{sequence}] saved TWM-only predictions", flush=True)


if __name__ == "__main__":
    main()
