#!/usr/bin/env python3
"""Run the planned three-clip legacy dual-loop inference ablation.

The model only receives RGB, HaWoR front-end state, and raw HaWoR SLAM
cameras. Ground-truth cameras, cached ground-truth joints, and frozen pinch
references are loaded only after every variant has finished inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "scripts"),
    str(PROJECT_ROOT),
    str(REPOSITORY_ROOT),
]

from run_hot3d import (  # noqa: E402
    image_tensor,
    load_checkpoint,
    load_config,
    neutral_state,
)
from dual_loop.geometry import ManoGeometry  # noqa: E402
from dual_loop.io import export_hawor_frontend, load_slam_c2w, load_timestamps  # noqa: E402
from dual_loop.model import DualLoopConfig, DualLoopObserver  # noqa: E402
from dual_loop.oracle import normalized_pinch_distance  # noqa: E402
from dual_loop.pinch_evaluation import extract_events, match_events  # noqa: E402
from dual_loop.state import pack_state  # noqa: E402


PLAN_DEFAULT = REPOSITORY_ROOT / "diagnostics" / "mini_ablation_plan.json"
OUTPUT_DEFAULT = REPOSITORY_ROOT / "diagnostics" / "mini_ablation_results"
VIDEO_ROOT_DEFAULT = REPOSITORY_ROOT / "datasets" / "hot3d_valset_export"
FRONTEND_ROOT_DEFAULT = REPOSITORY_ROOT / "eval_log"
EVAL_LOG_ROOT_DEFAULT = REPOSITORY_ROOT / "eval_log"
SIGNAL_ROOT_DEFAULT = (
    REPOSITORY_ROOT / "diagnostics" / "pinch_hawor_vs_ema_10videos_20261018_final"
)
SLAM_INTERVAL = 1000
WARMUP_FRAMES = 32
FRAMES_PER_CLIP = 900
MIN_METRIC_CHUNK = 16
METRIC_BLOCK = 100
EXPECTED_W_FRAMES = {
    "P0001_9b6feab7": 733,
    "P0011_76ea6d47": 893,
    "P0014_84ea2dcc": 900,
}

VARIANTS = (
    {
        "name": "Fixed-evidence correction, raw history",
        "slug": "fixed_evidence_raw_history",
        "intra_frame_reobserve": False,
        "corrected_state_feedback": False,
    },
    {
        "name": "+Intra-frame re-observation",
        "slug": "intra_frame_reobservation",
        "intra_frame_reobserve": True,
        "corrected_state_feedback": False,
    },
    {
        "name": "+Full dual-loop",
        "slug": "full_dual_loop",
        "intra_frame_reobserve": True,
        "corrected_state_feedback": True,
    },
)

SOURCE_PATHS = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "scripts" / "run_hot3d.py",
    PROJECT_ROOT / "scripts" / "eval_hot3d.py",
    PROJECT_ROOT / "dual_loop" / "model.py",
    PROJECT_ROOT / "dual_loop" / "networks.py",
    PROJECT_ROOT / "dual_loop" / "state.py",
    PROJECT_ROOT / "dual_loop" / "geometry.py",
    PROJECT_ROOT / "dual_loop" / "io.py",
    PROJECT_ROOT / "dual_loop" / "oracle.py",
    PROJECT_ROOT / "dual_loop" / "pinch_evaluation.py",
    REPOSITORY_ROOT / "lib" / "eval_utils" / "custom_utils.py",
    REPOSITORY_ROOT / "lib" / "eval_utils" / "eval_utils.py",
)


@dataclass
class ClipInputs:
    sequence: str
    sequence_root: Path
    image_paths: list[Path]
    total_frames: int
    frontend: dict[str, np.ndarray]
    frontend_provenance: dict[str, Any]
    raw_rotation_c2w: torch.Tensor
    raw_translation_c2w: torch.Tensor
    raw_rotation_w2c: torch.Tensor
    raw_translation_w2c: torch.Tensor
    segment_start: int
    segment_stop: int
    timestamp_ns: np.ndarray
    timestamp_sec: np.ndarray
    default_focal: float
    slam_path: Path


@dataclass
class VariantTrace:
    variant: dict[str, Any]
    config: DualLoopConfig
    frame_ids: np.ndarray
    states_raw: np.ndarray
    joints_raw: np.ndarray
    frontend_valid: np.ndarray
    quality: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed three-clip legacy dual-loop inference ablation"
    )
    parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    parser.add_argument("--video_root", "--video-root", type=Path, default=VIDEO_ROOT_DEFAULT)
    parser.add_argument(
        "--frontend_log_root", "--frontend-log-root", type=Path,
        default=FRONTEND_ROOT_DEFAULT,
    )
    parser.add_argument(
        "--eval_log_root", "--eval-log-root", type=Path, default=EVAL_LOG_ROOT_DEFAULT,
    )
    parser.add_argument(
        "--frozen_signal_root", "--frozen-signal-root", type=Path,
        default=SIGNAL_ROOT_DEFAULT,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml",
    )
    parser.add_argument("--output_root", "--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--preflight", action="store_true",
        help="Write an input audit and exit without model inference",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_hash(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def ensure_fresh_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path is not a directory: {path}")
        existing = list(path.iterdir())
        if existing:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Choose a new --output_root; "
                "existing results are never overwritten."
            )
    else:
        path.mkdir(parents=True)


def _resolve_protocol_path(plan_path: Path, plan: dict[str, Any]) -> Path:
    protocol = Path(plan["fixed_event_protocol"])
    return protocol if protocol.is_absolute() else REPOSITORY_ROOT / protocol


def _resolve_checkpoint_path(args: argparse.Namespace, plan: dict[str, Any]) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    checkpoint = Path(plan["legacy_checkpoint"])
    return checkpoint if checkpoint.is_absolute() else REPOSITORY_ROOT / checkpoint


def _protocol_issues(protocol: dict[str, Any]) -> list[str]:
    expected = {
        "reference_threshold": 0.2,
        "prediction_threshold": 0.275,
        "min_frames": 3,
        "onset_tolerance_sec": 0.2,
    }
    issues = []
    for key, value in expected.items():
        actual = protocol.get(key)
        if key == "min_frames":
            matches = actual == value and not isinstance(actual, bool)
        else:
            try:
                matches = abs(float(actual) - value) <= 1e-12
            except (TypeError, ValueError):
                matches = False
        if not matches:
            issues.append(f"frozen protocol {key} must be {value}, got {actual!r}")
    return issues


def _plan_issues(plan: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    clips = plan.get("clips")
    if not isinstance(clips, list) or len(clips) != 3:
        return ["plan must contain exactly three clips"]
    if plan.get("frames_per_clip") != FRAMES_PER_CLIP:
        issues.append(f"plan frames_per_clip must equal {FRAMES_PER_CLIP}")
    variants = plan.get("legacy_variants")
    expected = [
        (False, False),
        (True, False),
        (True, True),
    ]
    actual = [
        (v.get("intra_frame_reobserve"), v.get("corrected_state_feedback"))
        for v in variants or []
    ]
    if actual != expected:
        issues.append(f"plan variants must have R/F combinations {expected}, got {actual}")
    for item in clips:
        sequence = item.get("sequence", "<unknown>")
        start = item.get("start_frame")
        stop = item.get("stop_frame_exclusive")
        warmup = item.get("warmup_start_frame")
        if not all(isinstance(value, int) for value in (start, stop, warmup)):
            issues.append(f"{sequence}: start/stop/warmup frames must be integers")
            continue
        if stop - start != FRAMES_PER_CLIP:
            issues.append(f"{sequence}: target must contain exactly {FRAMES_PER_CLIP} frames")
        if start - warmup != WARMUP_FRAMES:
            issues.append(f"{sequence}: warm-up must contain exactly {WARMUP_FRAMES} frames")
        if warmup // SLAM_INTERVAL != (stop - 1) // SLAM_INTERVAL:
            issues.append(f"{sequence}: clip and warm-up must stay in one {SLAM_INTERVAL}-frame SLAM interval")
    return issues


def collect_missing_inputs(
    args: argparse.Namespace,
    plan: dict[str, Any] | None,
    plan_error: str | None = None,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    def need(path: Path, reason: str) -> None:
        if not path.exists():
            issues.append({"path": str(path), "reason": reason})

    if plan_error is not None:
        issues.append({"path": str(args.plan), "reason": plan_error})
        return issues
    assert plan is not None
    for message in _plan_issues(plan):
        issues.append({"path": str(args.plan), "reason": message})

    checkpoint_path = _resolve_checkpoint_path(args, plan)
    protocol_path = _resolve_protocol_path(args.plan, plan)
    need(args.config, "missing model config")
    need(checkpoint_path, "missing planned legacy full-model checkpoint")
    need(protocol_path, "missing frozen pinch-event protocol")
    need(REPOSITORY_ROOT / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl", "missing MANO right-hand asset")
    need(PROJECT_ROOT / "assets" / "mano_right.npz", "missing dual-loop MANO geometry asset")

    expected_hash = plan.get("checkpoint_sha256")
    if checkpoint_path.is_file() and expected_hash:
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash != expected_hash:
            issues.append({
                "path": str(checkpoint_path),
                "reason": f"checkpoint SHA256 mismatch: expected {expected_hash}, got {actual_hash}",
            })
    expected_protocol_hash = plan.get("fixed_event_protocol_sha256")
    if protocol_path.is_file() and expected_protocol_hash:
        actual_protocol_hash = sha256_file(protocol_path)
        if actual_protocol_hash != expected_protocol_hash:
            issues.append({
                "path": str(protocol_path),
                "reason": (
                    "frozen protocol SHA256 mismatch: expected "
                    f"{expected_protocol_hash}, got {actual_protocol_hash}"
                ),
            })
    if protocol_path.is_file():
        try:
            protocol = load_json(protocol_path)
            for message in _protocol_issues(protocol):
                issues.append({"path": str(protocol_path), "reason": message})
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            issues.append({"path": str(protocol_path), "reason": f"invalid frozen protocol: {exc}"})

    clips = plan.get("clips", []) if isinstance(plan.get("clips"), list) else []
    if not args.video_root.is_dir():
        issues.append({"path": str(args.video_root), "reason": "validation video root is missing"})
    for clip in clips:
        sequence = str(clip.get("sequence", "<unknown>"))
        sequence_root = args.video_root / sequence
        if not sequence_root.is_dir():
            issues.append({"path": str(sequence_root), "reason": "planned validation sequence is missing"})
            continue

        image_paths = sorted((sequence_root / "extracted_images").glob("*.jpg"))
        stop = int(clip.get("stop_frame_exclusive", 0))
        warmup = int(clip.get("warmup_start_frame", 0))
        if len(image_paths) < stop:
            issues.append({
                "path": str(sequence_root / "extracted_images"),
                "reason": f"need image frames through {stop - 1}, found {len(image_paths)} images",
            })
        need(sequence_root / "focal.txt", "missing sequence focal length")
        need(sequence_root / "tt_list.pkl", "missing sequence timestamps")
        need(sequence_root / "head_pose.pkl", "missing GT camera head trajectory for post-inference alignment")
        need(sequence_root / "ego_extrinsics.pkl", "missing GT camera extrinsics for post-inference alignment")
        track_detected = sequence_root / "preprocess" / "preprocess_tracks.npy"
        track_legacy = sequence_root / "preprocess" / "preprocess_gtdet_tracks.npy"
        if not track_detected.is_file() and not track_legacy.is_file():
            issues.append({
                "path": str(track_detected),
                "reason": "missing HaWoR front-end track file (detected or allowed legacy GT-projected tracks)",
            })
        chunk_dir = args.frontend_log_root / sequence / "1"
        if not chunk_dir.is_dir() or not any(chunk_dir.glob("*.json")):
            issues.append({"path": str(chunk_dir), "reason": "missing HaWoR right-hand frontend JSON chunks"})
        segment_start = (warmup // SLAM_INTERVAL) * SLAM_INTERVAL
        segment_stop = min(segment_start + SLAM_INTERVAL, len(image_paths))
        slam_path = (
            sequence_root / "preprocess" / "SLAM" /
            f"hawor_slam_w_scale_{segment_start}_{segment_stop}.npz"
        )
        need(slam_path, "missing raw HaWoR SLAM chunk for the planned clip interval")
        need(args.eval_log_root / sequence / "evaluation_details.npz", "missing cached baseline GT joints/common frame IDs")
        need(args.frozen_signal_root / sequence / "signals.npz", "missing frozen pinch reference and baseline-valid signals")

    return issues


def _read_plan(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    if not args.plan.is_file():
        return None, "missing ablation plan"
    try:
        return load_json(args.plan), None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f"invalid ablation plan: {exc}"


def _failure_report(
    output_root: Path,
    status: str,
    issues: list[dict[str, str]],
    plan_path: Path,
) -> None:
    ensure_fresh_output_dir(output_root)
    payload = {
        "status": status,
        "inference_started": False,
        "plan": str(plan_path),
        "missing_inputs": issues,
    }
    write_json_atomic(output_root / "missing_inputs.json", payload)
    write_json_atomic(output_root / "run_status.json", payload)


def audit_paths(args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    plan, plan_error = _read_plan(args)
    return plan, collect_missing_inputs(args, plan, plan_error)


def _load_config_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid model config: {path}")
    runtime = value.get("runtime", {})
    if int(runtime.get("slam_interval", SLAM_INTERVAL)) != SLAM_INTERVAL:
        raise ValueError(f"This plan requires runtime.slam_interval={SLAM_INTERVAL}")
    return value


def prepare_clips(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, ClipInputs]:
    prepared: dict[str, ClipInputs] = {}
    for clip in plan["clips"]:
        sequence = clip["sequence"]
        sequence_root = args.video_root / sequence
        image_paths = sorted((sequence_root / "extracted_images").glob("*.jpg"))
        total_frames = len(image_paths)
        if total_frames < int(clip["stop_frame_exclusive"]):
            raise ValueError(
                f"{sequence}: expected at least {clip['stop_frame_exclusive']} images, found {total_frames}"
            )

        frontend, frontend_provenance = export_hawor_frontend(
            sequence_root,
            args.frontend_log_root / sequence,
            allow_gt_boxes=True,
            return_provenance=True,
        )
        if len(frontend["valid"]) != total_frames:
            raise ValueError(f"{sequence}: frontend/image length mismatch")

        warmup_start = int(clip["warmup_start_frame"])
        segment_start = (warmup_start // SLAM_INTERVAL) * SLAM_INTERVAL
        segment_stop = min(segment_start + SLAM_INTERVAL, total_frames)
        slam_path = (
            sequence_root / "preprocess" / "SLAM" /
            f"hawor_slam_w_scale_{segment_start}_{segment_stop}.npz"
        )
        rotation_c2w, translation_c2w = load_slam_c2w(slam_path)
        expected_segment_frames = segment_stop - segment_start
        if len(rotation_c2w) < expected_segment_frames or len(translation_c2w) < expected_segment_frames:
            raise ValueError(
                f"{sequence}: raw SLAM segment has {len(rotation_c2w)} poses; "
                f"expected {expected_segment_frames} for {slam_path}"
            )
        rotation_c2w = rotation_c2w[:expected_segment_frames].float()
        translation_c2w = translation_c2w[:expected_segment_frames].float()
        rotation_w2c = rotation_c2w.transpose(-1, -2)
        translation_w2c = -torch.einsum("bij,bj->bi", rotation_w2c, translation_c2w)
        timestamp_ns, timestamp_sec = load_timestamps(sequence_root, total_frames)
        with (sequence_root / "focal.txt").open("r", encoding="utf-8") as handle:
            default_focal = float(handle.read())

        for frame_id in range(warmup_start, int(clip["stop_frame_exclusive"])):
            if not image_paths[frame_id].is_file():
                raise FileNotFoundError(f"Missing image frame {frame_id}: {image_paths[frame_id]}")
            if cv2.imread(str(image_paths[frame_id])) is None:
                raise ValueError(f"Unreadable image frame {frame_id}: {image_paths[frame_id]}")
        prepared[sequence] = ClipInputs(
            sequence=sequence,
            sequence_root=sequence_root,
            image_paths=image_paths,
            total_frames=total_frames,
            frontend=frontend,
            frontend_provenance=frontend_provenance,
            raw_rotation_c2w=rotation_c2w,
            raw_translation_c2w=translation_c2w,
            raw_rotation_w2c=rotation_w2c,
            raw_translation_w2c=translation_w2c,
            segment_start=segment_start,
            segment_stop=segment_stop,
            timestamp_ns=timestamp_ns,
            timestamp_sec=timestamp_sec,
            default_focal=default_focal,
            slam_path=slam_path,
        )
    return prepared


def variant_configs(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> list[tuple[dict[str, Any], DualLoopConfig]]:
    result: list[tuple[dict[str, Any], DualLoopConfig]] = []
    for variant in VARIANTS:
        override_args = argparse.Namespace(
            config=args.config,
            intra_frame_reobserve=variant["intra_frame_reobserve"],
            corrected_state_feedback=variant["corrected_state_feedback"],
        )
        config, raw_config, _source = load_config(override_args, checkpoint)
        if int(raw_config.get("runtime", {}).get("slam_interval", SLAM_INTERVAL)) != SLAM_INTERVAL:
            raise ValueError(
                f"Checkpoint runtime must use slam_interval={SLAM_INTERVAL} for this plan"
            )
        if config.correction_steps != 2:
            raise ValueError(f"Expected exactly two correction steps, got {config.correction_steps}")
        if config.world_model_variant != "legacy":
            raise ValueError(
                f"The planned checkpoint must use the legacy world model, got {config.world_model_variant!r}"
            )
        result.append((variant, config))
    return result


def verify_checkpoint_load(
    config: DualLoopConfig,
    state_dict: dict[str, torch.Tensor],
) -> None:
    model = DualLoopObserver(config).cpu()
    model.load_state_dict(state_dict, strict=True)
    del model


def run_variant(
    inputs: ClipInputs,
    clip_spec: dict[str, Any],
    variant: dict[str, Any],
    config: DualLoopConfig,
    state_dict: dict[str, torch.Tensor],
    geometry: ManoGeometry,
    device: torch.device,
) -> VariantTrace:
    model = DualLoopObserver(config).to(device).eval()
    model.load_state_dict(state_dict, strict=True)
    frontend = inputs.frontend
    image_size = torch.as_tensor(frontend["image_size"], dtype=torch.float32, device=device)
    frame_ids = np.arange(
        int(clip_spec["warmup_start_frame"]),
        int(clip_spec["stop_frame_exclusive"]),
        dtype=np.int32,
    )
    states = np.full((len(frame_ids), config.state_dim), np.nan, dtype=np.float32)
    joints = np.full((len(frame_ids), 21, 3), np.nan, dtype=np.float32)
    frontend_valid = np.zeros(len(frame_ids), dtype=bool)
    quality_trace = np.zeros(len(frame_ids), dtype=np.float32)
    history: list[torch.Tensor] = []
    last_quality = 0.0
    previous_segment: int | None = None

    with torch.inference_mode():
        for output_index, frame_id_value in enumerate(frame_ids):
            frame_id = int(frame_id_value)
            segment_id = frame_id // SLAM_INTERVAL
            if previous_segment is not None and segment_id != previous_segment:
                history.clear()
                last_quality = 0.0
            previous_segment = segment_id
            local_id = frame_id - inputs.segment_start
            if not 0 <= local_id < len(inputs.raw_rotation_c2w):
                raise ValueError(f"{inputs.sequence}: frame {frame_id} is outside its loaded SLAM segment")

            valid = bool(frontend["valid"][frame_id])
            image, blur_score = image_tensor(
                inputs.image_paths[frame_id], config.image_size, device)
            if valid:
                camera_state = pack_state(
                    torch.as_tensor(frontend["camera_translation"][frame_id:frame_id + 1], device=device),
                    torch.as_tensor(frontend["global_orient"][frame_id:frame_id + 1], device=device),
                    torch.as_tensor(frontend["hand_pose"][frame_id:frame_id + 1], device=device),
                    torch.as_tensor(frontend["betas"][frame_id:frame_id + 1], device=device),
                ).float()
                visual_state = geometry.camera_to_world_state(
                    camera_state,
                    inputs.raw_rotation_c2w[local_id:local_id + 1].to(device),
                    inputs.raw_translation_c2w[local_id:local_id + 1].to(device),
                )
            elif history:
                visual_state = history[-1]
            else:
                visual_state = neutral_state(device)

            if history:
                history_tensor = torch.stack(history, dim=1)
            else:
                history_tensor = visual_state[:, None, :0].reshape(1, 0, config.state_dim)
            frame_focal = frontend["focal_length"][frame_id]
            focal = inputs.default_focal if not np.isfinite(frame_focal) else float(frame_focal)
            focal_tensor = torch.tensor([focal], dtype=torch.float32, device=device)
            image_size_batch = image_size.unsqueeze(0)

            def projector(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return geometry.project_world_joints(
                    state,
                    inputs.raw_rotation_w2c[local_id:local_id + 1].to(device),
                    inputs.raw_translation_w2c[local_id:local_id + 1].to(device),
                    focal_tensor,
                    image_size_batch,
                )

            result = model.forward_step(
                image=image,
                visual_state=visual_state,
                history=history_tensor,
                projector=projector,
                visual_valid=torch.tensor([valid], dtype=torch.bool, device=device),
                quality_hint=torch.tensor([[blur_score if valid else 0.0]], device=device),
            )
            final_state = result["state"]
            frame_quality = float(result["quality"].item())
            if not valid:
                frame_quality = min(frame_quality, last_quality * config.quality_decay_on_missing)
            last_quality = frame_quality
            feedback = final_state if config.corrected_state_feedback else visual_state
            history.append(feedback.detach())
            history = history[-config.history_length:]

            states[output_index] = final_state[0].cpu().numpy()
            joints[output_index] = geometry.joints(final_state)[0].cpu().numpy()
            frontend_valid[output_index] = valid
            quality_trace[output_index] = frame_quality
            if output_index % 100 == 0 or output_index + 1 == len(frame_ids):
                print(f"[{variant['slug']}:{inputs.sequence}] {output_index + 1}/{len(frame_ids)}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return VariantTrace(
        variant=variant,
        config=config,
        frame_ids=frame_ids,
        states_raw=states,
        joints_raw=joints,
        frontend_valid=frontend_valid,
        quality=quality_trace,
    )


def _continuous_chunks(frame_ids: np.ndarray) -> list[np.ndarray]:
    ids = np.asarray(frame_ids, dtype=np.int64)
    if ids.size == 0:
        return []
    splits = np.flatnonzero(np.diff(ids) != 1) + 1
    return [part for part in np.split(ids, splits) if len(part)]


def baseline_mask_from_frames(
    frame_ids: np.ndarray,
    prediction_valid: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    """Require the two archived baseline-valid representations to agree."""
    ids = np.asarray(frame_ids, dtype=np.int64)
    signal_mask = np.asarray(prediction_valid, dtype=bool)
    if signal_mask.shape != (frame_count,):
        raise ValueError("frozen prediction_valid must have one value per sequence frame")
    if np.any(ids < 0) or np.any(ids >= frame_count):
        raise ValueError("cached common frame IDs exceed the sequence")
    frame_mask = np.zeros(frame_count, dtype=bool)
    frame_mask[ids] = True
    if not np.array_equal(frame_mask, signal_mask):
        raise ValueError("cached evaluation frame IDs do not match frozen baseline prediction_valid")
    return frame_mask


def _torch_float(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device="cpu")


def fixed_prediction_mask(common_mask, frontend_valid, frame_ids):
    common = np.asarray(common_mask, dtype=bool)
    available = np.asarray(frontend_valid, dtype=bool)
    if common.shape != available.shape:
        raise ValueError("common and frontend masks must have identical shape")
    missing = common & ~available
    if missing.any():
        ids = np.asarray(frame_ids)[missing][:10].tolist()
        raise ValueError(f"frontend lost frozen common frames {ids}")
    return common.copy()


def score_w_mpjpe(
    frame_ids: np.ndarray,
    predicted_joints: np.ndarray,
    target_joints: np.ndarray,
    valid: np.ndarray,
) -> tuple[float | None, int, np.ndarray]:
    """Match eval_hot3d's contiguous >=16 frame chunks and 100-frame W blocks."""
    from lib.eval_utils.eval_utils import compute_jpe, first_align_joints

    ids = np.asarray(frame_ids, dtype=np.int64)
    predicted = np.asarray(predicted_joints, dtype=np.float32)
    target = np.asarray(target_joints, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[1:] != (21, 3):
        raise ValueError("W-MPJPE joints must both have shape [frames, 21, 3]")
    if len(ids) != len(predicted) or mask.shape != (len(ids),):
        raise ValueError("W-MPJPE frame IDs and validity mask must match joint frames")
    finite = np.isfinite(predicted).all(axis=(1, 2)) & np.isfinite(target).all(axis=(1, 2))
    if np.any(mask & ~finite):
        bad_ids = ids[mask & ~finite][:10].tolist()
        raise ValueError(f"non-finite W-MPJPE joints on valid frames {bad_ids}")
    by_frame = np.full(len(ids), np.nan, dtype=np.float32)
    all_scores: list[np.ndarray] = []
    id_to_position = {int(frame_id): index for index, frame_id in enumerate(ids)}
    valid_ids = ids[mask]
    for chunk_ids in _continuous_chunks(valid_ids):
        if len(chunk_ids) < MIN_METRIC_CHUNK:
            continue
        positions = np.asarray([id_to_position[int(frame_id)] for frame_id in chunk_ids], dtype=np.int64)
        for start in range(0, len(positions), METRIC_BLOCK):
            block = positions[start:start + METRIC_BLOCK]
            gt_part = _torch_float(target[block])
            pred_part = _torch_float(predicted[block])
            aligned = first_align_joints(gt_part, pred_part)
            errors_mm = np.asarray(compute_jpe(gt_part, aligned), dtype=np.float32) * 1000.0
            by_frame[block] = errors_mm
            all_scores.append(errors_mm)
    if not all_scores:
        return None, 0, by_frame
    values = np.concatenate(all_scores)
    return float(values.mean()), int(len(values)), by_frame


def align_variant_to_gt_world(
    inputs: ClipInputs,
    trace: VariantTrace,
    args: argparse.Namespace,
    device: torch.device,
    geometry: ManoGeometry,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Align raw SLAM to GT cameras and map the already-inferred states afterwards."""
    from lib.eval_utils.custom_utils import algin_cam_traj_wo_scale, load_gt_cam

    _, _, gt_rotation_c2w, gt_translation_c2w = load_gt_cam(
        str(args.video_root), inputs.sequence, inputs.segment_start, inputs.segment_stop,
    )
    _, _, aligned_rotation_c2w, aligned_translation_c2w = algin_cam_traj_wo_scale(
        inputs.raw_rotation_c2w,
        inputs.raw_translation_c2w,
        gt_rotation_c2w,
        gt_translation_c2w,
    )
    local_ids = trace.frame_ids.astype(np.int64) - inputs.segment_start
    if np.any(local_ids < 0) or np.any(local_ids >= len(inputs.raw_rotation_w2c)):
        raise ValueError(f"{inputs.sequence}: trace includes frames outside one raw SLAM interval")
    with torch.inference_mode():
        raw_state = torch.as_tensor(trace.states_raw, dtype=torch.float32, device=device)
        raw_r_w2c = inputs.raw_rotation_w2c[local_ids].to(device)
        raw_t_w2c = inputs.raw_translation_w2c[local_ids].to(device)
        camera_translation, camera_root, camera_hand, camera_betas = geometry.world_to_camera_parameters(
            raw_state, raw_r_w2c, raw_t_w2c,
        )
        camera_state = pack_state(camera_translation, camera_root, camera_hand, camera_betas)
        aligned_state = geometry.camera_to_world_state(
            camera_state,
            aligned_rotation_c2w[local_ids].to(device),
            aligned_translation_c2w[local_ids].to(device),
        )
        aligned_joints = geometry.joints(aligned_state).cpu().numpy().astype(np.float32)
        aligned_state_np = aligned_state.cpu().numpy().astype(np.float32)
    camera_files = {
        "head_pose_sha256": sha256_file(inputs.sequence_root / "head_pose.pkl"),
        "ego_extrinsics_sha256": sha256_file(inputs.sequence_root / "ego_extrinsics.pkl"),
    }
    return aligned_state_np, aligned_joints, camera_files


def score_clip_variant(
    clip_spec: dict[str, Any],
    inputs: ClipInputs,
    trace: VariantTrace,
    aligned_joints: np.ndarray,
    eval_log_root: Path,
    frozen_signal_root: Path,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    sequence = inputs.sequence
    eval_path = eval_log_root / sequence / "evaluation_details.npz"
    signals_path = frozen_signal_root / sequence / "signals.npz"
    with np.load(eval_path, allow_pickle=False) as archive:
        baseline_frame_ids = archive["frame_ids"].astype(np.int64, copy=True)
        gt_world = archive["gt_world_joints_m"].astype(np.float32, copy=True)
    with np.load(signals_path, allow_pickle=False) as archive:
        timestamp_sec = archive["timestamp_sec"].astype(np.float64, copy=True)
        baseline_prediction_valid = archive["prediction_valid"].astype(bool, copy=True)
        reference_distance = archive["reference_distance"].astype(np.float64, copy=True)
        reference_valid = archive["reference_valid"].astype(bool, copy=True)

    arrays = (timestamp_sec, baseline_prediction_valid, reference_distance, reference_valid)
    if any(value.ndim != 1 or len(value) != inputs.total_frames for value in arrays):
        raise ValueError(f"{sequence}: frozen signal arrays do not match the validation sequence")
    if gt_world.shape != (inputs.total_frames, 21, 3):
        raise ValueError(f"{sequence}: cached GT joint array has unexpected shape {gt_world.shape}")
    try:
        common_mask = baseline_mask_from_frames(
            baseline_frame_ids, baseline_prediction_valid, inputs.total_frames,
        )
    except ValueError as exc:
        raise ValueError(f"{sequence}: {exc}") from exc
    frame_ids = trace.frame_ids.astype(np.int64)
    target_mask = (frame_ids >= int(clip_spec["start_frame"])) & (
        frame_ids < int(clip_spec["stop_frame_exclusive"])
    )
    target_positions = np.flatnonzero(target_mask)
    target_ids = frame_ids[target_positions]
    frontend_valid = trace.frontend_valid[target_positions]
    pred_valid = fixed_prediction_mask(common_mask[target_ids], frontend_valid, target_ids)
    common_count = int(common_mask[target_ids].sum())
    expected_common = clip_spec.get("common_valid_frames")
    if expected_common is not None and common_count != int(expected_common):
        raise ValueError(
            f"{sequence}: common mask covers {common_count} target frames; "
            f"plan expects {expected_common}"
        )

    target_prediction_joints = aligned_joints[target_positions]
    target_gt_joints = gt_world[target_ids]
    finite_prediction = np.isfinite(target_prediction_joints).all(axis=(1, 2))
    finite_gt = np.isfinite(target_gt_joints).all(axis=(1, 2))
    bad_prediction = pred_valid & ~finite_prediction
    bad_gt = common_mask[target_ids] & ~finite_gt
    if np.any(bad_prediction):
        bad_ids = target_ids[bad_prediction][:10].tolist()
        raise ValueError(
            f"{sequence}: non-finite model joints on expected-valid frames {bad_ids}"
        )
    if np.any(bad_gt):
        bad_ids = target_ids[bad_gt][:10].tolist()
        raise ValueError(
            f"{sequence}: cached GT joints are non-finite on common frames {bad_ids}"
        )

    w_valid = pred_valid.copy()
    w_mean, w_count, w_by_processed_frame = score_w_mpjpe(
        target_ids,
        target_prediction_joints,
        target_gt_joints,
        w_valid,
    )
    expected_w_count = EXPECTED_W_FRAMES.get(sequence)
    if expected_w_count is not None and w_count != expected_w_count:
        raise ValueError(
            f"{sequence}: W-MPJPE scored {w_count} frames; "
            f"the frozen common-mask protocol expects {expected_w_count}"
        )
    # Metric eligibility can also be restricted by short-chunk filtering.
    w_by_target = np.full(len(target_ids), np.nan, dtype=np.float32)
    w_by_target[:] = w_by_processed_frame

    pred_distance = normalized_pinch_distance(aligned_joints[target_positions])
    prediction_events = extract_events(
        pred_distance,
        timestamp_sec[target_ids],
        pred_valid & np.isfinite(pred_distance),
        threshold=float(protocol["prediction_threshold"]),
        min_frames=int(protocol["min_frames"]),
    )
    reference_events = extract_events(
        reference_distance[target_ids],
        timestamp_sec[target_ids],
        reference_valid[target_ids],
        threshold=float(protocol["reference_threshold"]),
        min_frames=int(protocol["min_frames"]),
    )
    event_metrics = match_events(
        reference_events,
        prediction_events,
        onset_tolerance_sec=float(protocol["onset_tolerance_sec"]),
    )
    short_reference_count = sum(
        3 <= (event.stop - event.start + 1) <= 10 for event in reference_events
    )
    expected_ref = clip_spec.get("reference_events")
    expected_short = clip_spec.get("short_reference_events")
    if expected_ref is not None and len(reference_events) != int(expected_ref):
        raise ValueError(
            f"{sequence}: frozen protocol produced {len(reference_events)} reference events; "
            f"plan expects {expected_ref}"
        )
    if expected_short is not None and short_reference_count != int(expected_short):
        raise ValueError(
            f"{sequence}: frozen protocol produced {short_reference_count} short events; "
            f"plan expects {expected_short}"
        )

    result = {
        "sequence": sequence,
        "variant": trace.variant["name"],
        "variant_slug": trace.variant["slug"],
        "intra_frame_reobserve": trace.config.intra_frame_reobserve,
        "corrected_state_feedback": trace.config.corrected_state_feedback,
        "W-MPJPE_mm": w_mean,
        "w_mpjpe_scored_frames": w_count,
        "common_frontend_valid_frames": int(pred_valid.sum()),
        "event_precision": event_metrics["precision"],
        "event_recall": event_metrics["recall"],
        "onset_MAE_ms": event_metrics["onset_mae_ms"],
        "short_event_recall": event_metrics["short_event_recall"],
        "matched_events_count": int(event_metrics["tp"]),
        "reference_events": len(reference_events),
        "predicted_events": len(prediction_events),
        "true_positives": int(event_metrics["tp"]),
        "false_positives": int(event_metrics["fp"]),
        "false_negatives": int(event_metrics["fn"]),
        "short_reference_events": short_reference_count,
        "short_matched_events": int(event_metrics["short_matched_events"]),
        "event_matches": event_metrics["matches"],
        "reference_valid_frames": int(reference_valid[target_ids].sum()),
        "baseline_common_mask_frames": common_count,
        "frontend_valid_frames": int(frontend_valid.sum()),
    }
    return result, w_by_target


def aggregate_rows(clip_results: list[dict[str, Any]], w_values: list[np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        results = [item for item in clip_results if item["variant_slug"] == variant["slug"]]
        # w_values is ordered the same way as clip_results (clip-major).
        variant_index = [i for i, item in enumerate(clip_results) if item["variant_slug"] == variant["slug"]]
        selected_w = [w_values[i][np.isfinite(w_values[i])] for i in variant_index]
        selected_w = [array for array in selected_w if len(array)]
        flattened = np.concatenate(selected_w) if selected_w else np.asarray([], dtype=np.float32)
        tp = sum(int(result["true_positives"]) for result in results)
        fp = sum(int(result["false_positives"]) for result in results)
        fn = sum(int(result["false_negatives"]) for result in results)
        short_refs = sum(int(result["short_reference_events"]) for result in results)
        short_tp = sum(int(result["short_matched_events"]) for result in results)
        onset_errors = [
            abs(float(match["onset_error_ms"]))
            for result in results
            for match in result["event_matches"]
        ]
        rows.append({
            "variant": variant["name"],
            "variant_slug": variant["slug"],
            "intra_frame_reobserve": variant["intra_frame_reobserve"],
            "corrected_state_feedback": variant["corrected_state_feedback"],
            "clip_count": len(results),
            "W-MPJPE_mm": float(flattened.mean()) if len(flattened) else None,
            "w_mpjpe_scored_frames": int(len(flattened)),
            "event_precision": tp / (tp + fp) if tp + fp else None,
            "event_recall": tp / (tp + fn) if tp + fn else None,
            "onset_MAE_ms": float(np.mean(onset_errors)) if onset_errors else None,
            "short_event_recall": short_tp / short_refs if short_refs else None,
            "matched_events_count": tp,
            "reference_events": tp + fn,
            "predicted_events": tp + fp,
            "short_reference_events": short_refs,
            "short_matched_events": short_tp,
        })
    return rows


def source_provenance(plan_path: Path, protocol_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    source_hashes = {
        path.relative_to(REPOSITORY_ROOT).as_posix(): sha256_file(path)
        for path in SOURCE_PATHS
    }
    return {
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "fixed_event_protocol_path": str(protocol_path),
        "fixed_event_protocol_sha256": sha256_file(protocol_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_sha256": combined_hash(source_hashes),
        "source_file_sha256": source_hashes,
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "variant", "intra_frame_reobserve", "corrected_state_feedback", "clip_count",
        "W-MPJPE_mm", "w_mpjpe_scored_frames", "event_precision", "event_recall",
        "onset_MAE_ms", "short_event_recall", "matched_events_count",
        "reference_events", "predicted_events", "short_reference_events",
        "short_matched_events",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def run_experiment(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    checkpoint_path = _resolve_checkpoint_path(args, plan)
    protocol_path = _resolve_protocol_path(args.plan, plan)
    protocol = load_json(protocol_path)
    protocol_mismatches = _protocol_issues(protocol)
    if protocol_mismatches:
        raise ValueError("; ".join(protocol_mismatches))
    _load_config_yaml(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Validate every input and checkpoint before creating result files.
    prepared = prepare_clips(args, plan)
    checkpoint, state_dict = load_checkpoint(checkpoint_path, torch.device("cpu"))
    configured_variants = variant_configs(args, checkpoint)
    for _variant, model_config in configured_variants:
        verify_checkpoint_load(model_config, state_dict)
    geometry = ManoGeometry(REPOSITORY_ROOT).to(device).eval()
    provenance = source_provenance(args.plan, protocol_path, checkpoint_path)

    ensure_fresh_output_dir(args.output_root)
    status: dict[str, Any] = {
        "status": "running",
        "inference_started": True,
        "completed_variants": [],
        "completed_clips": [],
        "output_root": str(args.output_root),
        "device": str(device),
        "provenance": provenance,
    }
    write_json_atomic(args.output_root / "run_status.json", status)

    variant_traces: dict[str, dict[str, VariantTrace]] = {v["slug"]: {} for v in VARIANTS}
    try:
        for variant, model_config in configured_variants:
            for clip_spec in plan["clips"]:
                sequence = clip_spec["sequence"]
                trace = run_variant(
                    prepared[sequence], clip_spec, variant, model_config,
                    state_dict, geometry, device,
                )
                variant_traces[variant["slug"]][sequence] = trace
                status["completed_clips"].append({
                    "variant": variant["slug"], "sequence": sequence,
                })
                write_json_atomic(args.output_root / "run_status.json", status)
            status["completed_variants"].append(variant["slug"])
            write_json_atomic(args.output_root / "run_status.json", status)

        # GT camera alignment and all metrics start after all nine clip/variant
        # inference runs have completed.
        clip_results: list[dict[str, Any]] = []
        w_values: list[np.ndarray] = []
        for clip_spec in plan["clips"]:
            sequence = clip_spec["sequence"]
            inputs = prepared[sequence]
            per_clip: dict[str, Any] = {
                "sequence": sequence,
                "start_frame": int(clip_spec["start_frame"]),
                "stop_frame_exclusive": int(clip_spec["stop_frame_exclusive"]),
                "warmup_start_frame": int(clip_spec["warmup_start_frame"]),
                "target_frame_count": FRAMES_PER_CLIP,
                "warmup_frame_count": WARMUP_FRAMES,
                "coordinate_protocol": "raw SLAM segment during inference; GT camera aligned after all inference",
                "frontend_provenance": inputs.frontend_provenance,
                "input_files": {
                    "slam_path": str(inputs.slam_path),
                    "slam_sha256": sha256_file(inputs.slam_path),
                    "timestamps_path": str(inputs.sequence_root / "tt_list.pkl"),
                    "timestamps_sha256": sha256_file(inputs.sequence_root / "tt_list.pkl"),
                },
                "provenance": provenance,
                "variants": [],
            }
            for variant in VARIANTS:
                trace = variant_traces[variant["slug"]][sequence]
                aligned_state, aligned_joints, camera_hashes = align_variant_to_gt_world(
                    inputs, trace, args, device, geometry,
                )
                result, w_by_target = score_clip_variant(
                    clip_spec,
                    inputs,
                    trace,
                    aligned_joints,
                    args.eval_log_root,
                    args.frozen_signal_root,
                    protocol,
                )
                target_mask = (trace.frame_ids >= int(clip_spec["start_frame"])) & (
                    trace.frame_ids < int(clip_spec["stop_frame_exclusive"])
                )
                w_by_processed = np.full(len(trace.frame_ids), np.nan, dtype=np.float32)
                w_by_processed[np.flatnonzero(target_mask)] = w_by_target
                output_dir = args.output_root / variant["slug"]
                output_dir.mkdir(parents=True, exist_ok=True)
                npz_path = output_dir / f"{sequence}.npz"
                np.savez_compressed(
                    npz_path,
                    frame_ids=trace.frame_ids,
                    target_mask=target_mask,
                    state_world_raw_slam=trace.states_raw,
                    joints_world_raw_slam_m=trace.joints_raw,
                    state_world_gt_aligned=aligned_state,
                    joints_world_gt_aligned_m=aligned_joints,
                    frontend_valid=trace.frontend_valid,
                    quality=trace.quality,
                    w_mpjpe_by_frame_mm=w_by_processed,
                    input_provenance_json=np.asarray(json.dumps({
                        **provenance,
                        "frontend": inputs.frontend_provenance,
                        "gt_camera_sha256": camera_hashes,
                    }, sort_keys=True)),
                    model_config_json=np.asarray(json.dumps(asdict(trace.config), sort_keys=True)),
                )
                result["poses_joints_npz"] = str(npz_path)
                result["model_config"] = asdict(trace.config)
                result["gt_camera_sha256"] = camera_hashes
                result["evaluation_details_sha256"] = sha256_file(
                    args.eval_log_root / sequence / "evaluation_details.npz"
                )
                result["frozen_signals_sha256"] = sha256_file(
                    args.frozen_signal_root / sequence / "signals.npz"
                )
                per_clip["variants"].append(result)
                clip_results.append(result)
                w_values.append(w_by_target)
            clip_json_path = args.output_root / f"{sequence}.json"
            write_json_atomic(clip_json_path, per_clip)

        rows = aggregate_rows(clip_results, w_values)
        summary = {
            "status": "complete",
            "scope": "user-authorized legacy inference-time ablation; exploratory, no retraining",
            "clip_count": len(plan["clips"]),
            "frames_per_clip": FRAMES_PER_CLIP,
            "warmup_frames_per_clip": WARMUP_FRAMES,
            "target_frame_count_per_variant": FRAMES_PER_CLIP * len(plan["clips"]),
            "all_variant_target_frame_count_total": FRAMES_PER_CLIP * len(plan["clips"]) * len(VARIANTS),
            "inference_frame_count_per_variant": (FRAMES_PER_CLIP + WARMUP_FRAMES) * len(plan["clips"]),
            "inference_frame_count_total": (FRAMES_PER_CLIP + WARMUP_FRAMES) * len(plan["clips"]) * len(VARIANTS),
            "rows": rows,
            "per_clip_metrics": clip_results,
            "provenance": provenance,
        }
        write_json_atomic(args.output_root / "summary.json", summary)
        _write_summary_csv(args.output_root / "summary.csv", rows)
        status.update({"status": "complete", "summary": str(args.output_root / "summary.json")})
        write_json_atomic(args.output_root / "run_status.json", status)
        print(f"[done] {args.output_root / 'summary.json'}")
        return 0
    except Exception as exc:
        status.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json_atomic(args.output_root / "run_status.json", status)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan, issues = audit_paths(args)
    if issues:
        _failure_report(
            args.output_root,
            "blocked_missing_inputs",
            issues,
            args.plan,
        )
        print(f"[blocked] {len(issues)} required input issue(s); see {args.output_root / 'missing_inputs.json'}")
        return 2
    assert plan is not None
    if args.preflight:
        ensure_fresh_output_dir(args.output_root)
        report = {
            "status": "ready_for_inference",
            "inference_started": False,
            "preflight_mode": "path-presence only; no model inference or GT metric loading",
            "plan": str(args.plan),
            "checkpoint": str(_resolve_checkpoint_path(args, plan)),
            "video_root": str(args.video_root),
            "missing_inputs": [],
        }
        write_json_atomic(args.output_root / "missing_inputs.json", report)
        write_json_atomic(args.output_root / "run_status.json", report)
        print(f"[ready] preflight passed; see {args.output_root / 'run_status.json'}")
        return 0
    try:
        return run_experiment(args, plan)
    except FileExistsError:
        raise
    except Exception as exc:
        # Full content checks (frontend chunks, timestamps, SLAM archive, and
        # strict checkpoint loading) also precede any inference outputs.
        if not args.output_root.exists() or not any(args.output_root.iterdir()):
            _failure_report(
                args.output_root,
                "blocked_input_validation",
                [{"path": "input_validation", "reason": f"{type(exc).__name__}: {exc}"}],
                args.plan,
            )
        print(f"[blocked] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
