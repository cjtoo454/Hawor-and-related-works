#!/usr/bin/env python3
"""Run the dual-loop observer on HOT3D sequences using the HaWoR front-end."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from dual_loop.geometry import ManoGeometry  # noqa: E402
from dual_loop.io import (  # noqa: E402
    PREDICTION_KEYS,
    export_hawor_frontend,
    load_slam_c2w,
    load_timestamps,
    save_predictions,
    save_trace,
    write_json,
)
from dual_loop.model import DualLoopConfig, DualLoopObserver  # noqa: E402
from dual_loop.state import pack_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-loop HOT3D inference")
    parser.add_argument(
        "--video_root", type=Path,
        default=REPOSITORY_ROOT / "datasets" / "hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--video", action="append")
    parser.add_argument(
        "--frontend_log_root", type=Path, default=REPOSITORY_ROOT / "eval_log",
        help="HaWoR chunk-JSON root; this is the shared visual front-end")
    parser.add_argument(
        "--output_root", type=Path, default=REPOSITORY_ROOT / "eval_log_try")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow_untrained", action="store_true",
                        help="Run identity-safe initialization for format/smoke tests only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow_gt_boxes", action="store_true",
        help=("Allow legacy GT-projected boxes or unverified frontend provenance; "
              "the run is marked oracle-assisted/unknown"))
    parser.add_argument(
        "--intra_frame_reobserve", action=argparse.BooleanOptionalAction,
        default=None)
    parser.add_argument(
        "--corrected_state_feedback", action=argparse.BooleanOptionalAction,
        default=None)
    return parser.parse_args()


def load_config(
    args: argparse.Namespace,
    checkpoint: dict | None = None,
) -> tuple[DualLoopConfig, dict, str]:
    with args.config.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    raw = copy.deepcopy(base_config)
    config_source = str(args.config)
    if checkpoint is not None:
        saved_config = checkpoint.get("config", checkpoint.get("model_config"))
        if not isinstance(saved_config, dict):
            raise ValueError(
                "The full checkpoint must include its model config under 'config' "
                "or 'model_config'; refusing to infer architecture from the CLI YAML.")
        saved_model = saved_config.get("model", saved_config.get("model_config"))
        if saved_model is None:
            model_fields = set(DualLoopConfig.__dataclass_fields__)
            if model_fields.intersection(saved_config):
                saved_model = saved_config
            else:
                raise ValueError("Checkpoint config has no 'model' section")
        if not isinstance(saved_model, dict):
            raise ValueError("Checkpoint model config must be a mapping")
        raw["model"] = dict(saved_model)
        for key in ("runtime", "output"):
            if isinstance(saved_config.get(key), dict):
                raw[key] = {**raw.get(key, {}), **saved_config[key]}
        config_source = "checkpoint"
    model_values = dict(raw["model"])
    if args.intra_frame_reobserve is not None:
        model_values["intra_frame_reobserve"] = args.intra_frame_reobserve
    if args.corrected_state_feedback is not None:
        model_values["corrected_state_feedback"] = args.corrected_state_feedback
    return DualLoopConfig(**model_values), raw, config_source


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if checkpoint.get("stage") == "world_model_pretrain":
        raise ValueError(
            "Checkpoint stage='world_model_pretrain' contains prior-only weights; "
            "provide a full dual-loop checkpoint for inference.")
    state_dict = checkpoint.get(
        "model", checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no model state dictionary: {path}")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value
                      for key, value in state_dict.items()}
    keys = list(state_dict)
    required_modules = {
        "image_encoder", "reobserver", "world_model", "quality_estimator", "corrector"
    }
    present_modules = {key.split(".", 1)[0] for key in keys if "." in key}
    missing_modules = sorted(required_modules - present_modules)
    if missing_modules:
        raise ValueError(
            f"Checkpoint is not a full dual-loop model (missing {missing_modules}); "
            "a world-model-only checkpoint requires full training before inference.")
    return checkpoint, state_dict


def load_sequences(args: argparse.Namespace) -> list[str]:
    with (args.video_root / args.set_file).open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    if args.video:
        selected = set(args.video)
        unknown = sorted(selected - set(sequences))
        if unknown:
            raise ValueError(f"Sequences not found in {args.set_file}: {unknown}")
        sequences = [sequence for sequence in sequences if sequence in selected]
    return sequences


def load_slam_cameras(
    sequence_root: Path,
    frame_count: int,
    interval: int,
    sequence_frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if interval < 1:
        raise ValueError("slam_interval must be positive")
    rotations_c2w, translations_c2w = [], []
    for start in range(0, frame_count, interval):
        # Keep the dataset's real chunk end in the filename when max_frames truncates
        # the requested prefix (for example, read 0_1000 for a 32-frame smoke run).
        stop = min(start + interval, sequence_frame_count)
        path = (
            sequence_root / "preprocess" / "SLAM" /
            f"hawor_slam_w_scale_{start}_{stop}.npz")
        if not path.exists():
            raise FileNotFoundError(f"Missing HaWoR SLAM trajectory: {path}")
        rotation, translation = load_slam_c2w(path)
        required = min(stop - start, frame_count - start)
        if len(rotation) < required or len(translation) < required:
            raise ValueError(
                f"SLAM trajectory {path} has too few frames for requested prefix: "
                f"need {required}, got rotations={len(rotation)}, "
                f"translations={len(translation)}")
        rotations_c2w.append(rotation[:required].float())
        translations_c2w.append(translation[:required].float())
    rotation_c2w = torch.cat(rotations_c2w)
    translation_c2w = torch.cat(translations_c2w)
    rotation_w2c = rotation_c2w.transpose(-1, -2)
    translation_w2c = -torch.einsum(
        "bij,bj->bi", rotation_w2c, translation_c2w)
    return rotation_c2w, translation_c2w, rotation_w2c, translation_w2c


def image_tensor(path: Path, image_size: int, device: torch.device) -> tuple[torch.Tensor, float]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(np.clip(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()) / 7.0, 0, 1))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device), blur_score


def require_matching_existing_run(
    metadata_path: Path,
    expected: dict,
    prediction_path: Path,
) -> None:
    if not metadata_path.is_file():
        raise ValueError(
            f"Predictions already exist without matching run metadata: {prediction_path}. "
            "Pass --overwrite to regenerate them.")
    with metadata_path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    mismatches = [
        key for key, value in expected.items()
        if existing.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Existing output metadata does not match this run for {mismatches}. "
            "Pass --overwrite to regenerate the outputs.")


def neutral_state(device: torch.device) -> torch.Tensor:
    root = torch.eye(3, device=device).reshape(1, 1, 3, 3)
    hand = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(1, 15, 1, 1)
    return pack_state(
        torch.zeros(1, 3, device=device), root, hand,
        torch.zeros(1, 10, device=device))


def run_sequence(
    args: argparse.Namespace,
    config: DualLoopConfig,
    raw_config: dict,
    config_source: str,
    sequence: str,
    checkpoint: dict | None,
    state_dict: dict | None,
) -> None:
    sequence_root = args.video_root / sequence
    output_dir = args.output_root / sequence
    prediction_path = output_dir / raw_config["output"]["prediction_filename"]
    image_paths = sorted((sequence_root / "extracted_images").glob("*.jpg"))
    total_frames = len(image_paths)
    process_frames = total_frames if args.max_frames is None else min(args.max_frames, total_frames)
    if process_frames < 1:
        raise ValueError("No frames selected")
    metadata_path = output_dir / raw_config["output"]["metadata_filename"]
    if prediction_path.exists() and not args.overwrite:
        expected_metadata = {
            "checkpoint": str(args.checkpoint) if checkpoint is not None else None,
            "model_config_source": config_source,
            "model_config": asdict(config),
            "intra_frame_reobserve": config.intra_frame_reobserve,
            "corrected_state_feedback": config.corrected_state_feedback,
            "total_frames": total_frames,
            "processed_frames": process_frames,
        }
        require_matching_existing_run(metadata_path, expected_metadata, prediction_path)
        print(f"[skip] matching predictions exist: {prediction_path}")
        return

    frontend, track_provenance = export_hawor_frontend(
        sequence_root,
        args.frontend_log_root / sequence,
        allow_gt_boxes=args.allow_gt_boxes,
        return_provenance=True,
    )

    device = torch.device(args.device)
    model = DualLoopObserver(config).to(device).eval()
    trained = checkpoint is not None
    if trained:
        assert state_dict is not None
        model.load_state_dict(state_dict, strict=True)
    elif not args.allow_untrained:
        raise ValueError(
            "A trained --checkpoint is required. Use --allow_untrained only for "
            "pipeline and format validation; its correction heads are identity-initialized.")

    geometry = ManoGeometry(REPOSITORY_ROOT).to(device).eval()
    interval = int(raw_config["runtime"]["slam_interval"])
    rotation_c2w, translation_c2w, rotation_w2c, translation_w2c = [
        value.to(device) for value in load_slam_cameras(
            sequence_root, process_frames, interval, total_frames)]
    timestamp_ns, timestamp_sec = load_timestamps(sequence_root, total_frames)
    segment_id = np.arange(total_frames, dtype=np.int32) // interval
    image_size = torch.from_numpy(frontend["image_size"].astype(np.float32)).to(device)
    with (sequence_root / "focal.txt").open("r", encoding="utf-8") as handle:
        default_focal = float(handle.read())

    corrected = {key: value.copy() for key, value in frontend.items()}
    corrected["valid"][process_frames:] = False
    for key in ("global_orient", "hand_pose", "betas", "camera_translation"):
        corrected[key][process_frames:] = np.nan
    trace_state = np.full((total_frames, config.state_dim), np.nan, dtype=np.float32)
    trace_joints = np.full((total_frames, 21, 3), np.nan, dtype=np.float32)
    trace_quality = np.zeros(total_frames, dtype=np.float32)
    trace_source = np.full(total_frames, "unprocessed", dtype="U16")
    history: list[torch.Tensor] = []
    last_quality = 0.0

    with torch.inference_mode():
        for frame_id in range(process_frames):
            if frame_id > 0 and frame_id % interval == 0:
                history.clear()
                last_quality = 0.0
            valid = bool(frontend["valid"][frame_id])
            image, blur_score = image_tensor(
                image_paths[frame_id], config.image_size, device)
            if valid:
                camera_state = pack_state(
                    torch.from_numpy(frontend["camera_translation"][frame_id:frame_id + 1]).to(device),
                    torch.from_numpy(frontend["global_orient"][frame_id:frame_id + 1]).to(device),
                    torch.from_numpy(frontend["hand_pose"][frame_id:frame_id + 1]).to(device),
                    torch.from_numpy(frontend["betas"][frame_id:frame_id + 1]).to(device),
                ).float()
                visual_state = geometry.camera_to_world_state(
                    camera_state,
                    rotation_c2w[frame_id:frame_id + 1],
                    translation_c2w[frame_id:frame_id + 1],
                )
            elif history:
                visual_state = history[-1]
            else:
                visual_state = neutral_state(device)
            history_tensor = (
                torch.stack(history, dim=1)
                if history else visual_state[:, None, :0].reshape(1, 0, config.state_dim))
            frame_focal = frontend["focal_length"][frame_id]
            focal = default_focal if not np.isfinite(frame_focal) else float(frame_focal)
            focal_tensor = torch.tensor([focal], device=device)
            size_tensor = image_size.unsqueeze(0)

            def projector(state: torch.Tensor):
                return geometry.project_world_joints(
                    state,
                    rotation_w2c[frame_id:frame_id + 1],
                    translation_w2c[frame_id:frame_id + 1],
                    focal_tensor,
                    size_tensor,
                )

            result = model.forward_step(
                image=image,
                visual_state=visual_state,
                history=history_tensor,
                projector=projector,
                visual_valid=torch.tensor([valid], device=device),
                quality_hint=torch.tensor([[blur_score if valid else 0.0]], device=device),
            )
            final_state = result["state"]
            if valid:
                quality = float(result["quality"].item())
                source = "visual_corrected"
            else:
                quality = min(
                    float(result["quality"].item()),
                    last_quality * config.quality_decay_on_missing,
                )
                source = "world_prior"
            last_quality = quality
            feedback = final_state if config.corrected_state_feedback else visual_state
            history.append(feedback.detach())
            history = history[-config.history_length:]

            trace_state[frame_id] = final_state[0].cpu().numpy()
            trace_joints[frame_id] = geometry.joints(final_state)[0].cpu().numpy()
            trace_quality[frame_id] = quality
            trace_source[frame_id] = source
            if valid:
                translation, root, hand, betas = geometry.world_to_camera_parameters(
                    final_state,
                    rotation_w2c[frame_id:frame_id + 1],
                    translation_w2c[frame_id:frame_id + 1],
                )
                corrected["camera_translation"][frame_id] = translation[0].cpu().numpy()
                corrected["global_orient"][frame_id] = root[0].cpu().numpy()
                corrected["hand_pose"][frame_id] = hand[0].cpu().numpy()
                corrected["betas"][frame_id] = betas[0].cpu().numpy()
            if frame_id % 100 == 0 or frame_id + 1 == process_frames:
                print(f"[{sequence}] {frame_id + 1}/{process_frames}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(prediction_path, corrected)
    input_provenance = {
        "visual_frontend": "HaWoR chunk JSON",
        "frontend_result_dir": str(args.frontend_log_root / sequence),
        **track_provenance,
        "slam_camera_source": (
            "raw preprocess/SLAM/hawor_slam_w_scale_<start>_<stop>.npz"),
        "coordinate_frame": "per-interval local raw HaWoR SLAM world",
        "timestamp_source": "tt_list.pkl integer nanoseconds converted to seconds",
        "segment_id_semantics": (
            f"frame_id // {interval}; different SLAM segments are not directly comparable"),
    }
    save_trace(
        output_dir / raw_config["output"]["trace_filename"],
        frame_id=np.arange(total_frames),
        timestamp_ns=timestamp_ns,
        timestamp_sec=timestamp_sec,
        segment_id=segment_id,
        state_world=trace_state,
        joints_world_m=trace_joints,
        visual_valid=corrected["valid"],
        quality=trace_quality,
        source=trace_source,
        input_provenance=input_provenance,
    )
    write_json(
        metadata_path,
        {
            "sequence": sequence,
            "method": "Dual-Loop State Reconciliation",
            "visual_frontend": "HaWoR",
            "checkpoint": str(args.checkpoint) if trained else None,
            "trained_checkpoint_loaded": trained,
            "untrained_identity_smoke_test": not trained,
            "model_config_source": config_source,
            "model_config": asdict(config),
            "total_frames": total_frames,
            "processed_frames": process_frames,
            "valid_right_hand_frames": int(corrected["valid"].sum()),
            "state_dim": config.state_dim,
            "history_length": config.history_length,
            "correction_steps": config.correction_steps,
            "intra_frame_reobserve": config.intra_frame_reobserve,
            "corrected_state_feedback": config.corrected_state_feedback,
            "coordinate_frame": "per-interval local raw HaWoR SLAM world",
            "segments_directly_comparable": False,
            "timestamp_source": "tt_list.pkl integer nanoseconds converted to seconds",
            "segment_id_semantics": (
                f"frame_id // {interval}; different SLAM segments are not directly comparable"),
            "bbox_source": track_provenance["bbox_source"],
            "oracle_assisted": track_provenance["oracle_assisted"],
            "input_provenance": input_provenance,
            "prediction_schema": list(PREDICTION_KEYS),
        },
    )
    print(f"[done] {prediction_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = None
    state_dict = None
    if args.checkpoint is not None:
        checkpoint, state_dict = load_checkpoint(args.checkpoint, device)
    config, raw_config, config_source = load_config(args, checkpoint)
    for sequence in load_sequences(args):
        run_sequence(
            args, config, raw_config, config_source, sequence, checkpoint, state_dict)


if __name__ == "__main__":
    main()
