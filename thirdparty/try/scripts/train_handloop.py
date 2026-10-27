#!/usr/bin/env python3
"""Pretrain and jointly adapt the HandLoop v2 numerical/visual model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from dual_loop.handloop_training import (  # noqa: E402
    HandLoopMotionDataset,
    HandLoopBatchStream,
    HandLoopTrainer,
    make_optimizer,
    manifest_fingerprint,
    motion_training_step,
    parameter_groups_for_system,
    save_checkpoint,
    validate_manifest,
    validate_cross_manifest_splits,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HandLoop v2")
    parser.add_argument("--stage", choices=("motion", "joint"), default="motion")
    parser.add_argument("--manifest", "--motion_manifest", dest="motion_manifest", type=Path,
                        required=True, help="handloop_v2 motion manifest")
    parser.add_argument("--visual_manifest", type=Path,
                        help="handloop_v2 live RGB manifest; required for joint stage")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "handloop_v2.yaml")
    parser.add_argument("--hawor_checkpoint", type=Path,
                        help="HaWoR checkpoint required for joint visual adaptation")
    parser.add_argument("--motion_checkpoint", type=Path,
                        help="v2 trajectory prior checkpoint used to initialize joint training")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "handloop_v2" / "latest.pt")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size_motion", type=int, default=32)
    parser.add_argument("--batch_size_visual", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--cpu_threads", type=int, default=1)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"HandLoop v2 config not found: {path}")
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("HandLoop config must contain a YAML mapping")
    if values.get("architecture", values.get("version", "handloop_v2")) != "handloop_v2":
        raise ValueError("train_handloop.py only accepts architecture: handloop_v2")
    return values


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset, batch_size: int, workers: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        persistent_workers=False,
    )


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def _load_twm(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    from dual_loop.handloop_twm import TWMConfig, TrajectoryRestorer

    twm_config = TWMConfig(**dict(config.get("twm", {})))
    return TrajectoryRestorer(twm_config).to(device)


def _initialize_motion_prior(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "handloop_v2":
        raise ValueError("Joint adaptation requires a HandLoop v2 motion checkpoint")
    if checkpoint.get("stage") != "motion_pretrain":
        raise ValueError("--motion_checkpoint must point to a motion-pretraining checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_batches(dataset, batch_size: int, workers: int, seed: int) -> DataLoader:
    if len(dataset) < 1:
        raise ValueError("Calibration split has no usable windows")
    fixed = Subset(dataset, range(min(8, len(dataset))))
    return _loader(fixed, batch_size, workers, seed, shuffle=False)


def _validate_motion(model, loader, device) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            values.append(float(motion_training_step(model, batch)["loss"].cpu()))
    model.train()
    if not values:
        raise ValueError("Motion calibration has no batches")
    return float(np.mean(values))


def _validate_visual(visual_loss, system, loader, device, loss_config) -> float:
    system.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            values.append(float(visual_loss(system, batch, loss_config)["loss"].cpu()))
    system.train()
    if not values:
        raise ValueError("Visual calibration has no batches")
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.save_every < 1:
        raise ValueError("steps and save_every must be positive")
    torch.set_num_threads(max(1, args.cpu_threads))
    _seed_everything(args.seed)
    config = _load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    motion_manifest, _, motion_splits = validate_manifest(
        args.motion_manifest, required_kind="motion")
    split_hashes = {
        "motion": manifest_fingerprint(motion_manifest),
        "motion_train_count": len(motion_splits["train"]),
        "motion_calibration_count": len(motion_splits["calibration"]),
        "motion_test_count": len(motion_splits["test"]),
    }
    train_motion = HandLoopMotionDataset(
        args.motion_manifest, "train", seed=args.seed,
        max_gap_s=float(config.get("training", {}).get("max_gap_s", 0.25)),
    )
    calibration_motion = HandLoopMotionDataset(
        args.motion_manifest, "calibration", seed=args.seed + 1,
        max_gap_s=float(config.get("training", {}).get("max_gap_s", 0.25)),
    )
    motion_stream = HandLoopBatchStream(
        train_motion, args.batch_size_motion, seed=args.seed,
        num_workers=args.num_workers, shuffle=True,
        sampling_mode="source_balanced" if args.stage == "joint" else "full_coverage")
    calibration_motion_loader = _validation_batches(
        calibration_motion, args.batch_size_motion, args.num_workers, args.seed + 1)

    if args.stage == "motion":
        model = _load_twm(config, device)
        named_parameters = {"twm": model.parameters()}
        rates = {"twm": float(config.get("training", {}).get("twm_lr", 1e-5))}
        visual_loss = None
        visual_stream = None
        visual_validation = None
        joint_system = None
    else:
        if args.visual_manifest is None or args.hawor_checkpoint is None or args.motion_checkpoint is None:
            raise ValueError(
                "Joint training requires --visual_manifest, --hawor_checkpoint, and --motion_checkpoint")
        from dual_loop.handloop_data import VisualDataset
        from dual_loop.handloop_runtime import build_system, visual_clip_loss

        visual_manifest, _, visual_splits = validate_manifest(
            args.visual_manifest, required_kind="visual")
        validate_cross_manifest_splits(motion_splits, visual_splits)
        split_hashes["visual"] = manifest_fingerprint(visual_manifest)
        split_hashes["visual_train_count"] = len(visual_splits["train"])
        split_hashes["visual_calibration_count"] = len(visual_splits["calibration"])
        split_hashes["visual_test_count"] = len(visual_splits["test"])
        joint_system = build_system(args.hawor_checkpoint, config, device)
        _initialize_motion_prior(joint_system.twm, args.motion_checkpoint)
        model = joint_system
        named_parameters = parameter_groups_for_system(joint_system)
        rates = {
            "hawor": float(config.get("training", {}).get("hawor_lr", 1e-6)),
            "twm": float(config.get("training", {}).get("twm_lr", 1e-5)),
            "visual": float(config.get("training", {}).get("visual_lr", 1e-4)),
        }
        visual_loss = visual_clip_loss
        visual_window = int(config.get("training", {}).get("window", 32))
        visual_warmup = int(config.get("training", {}).get("warmup_frames", 0))
        visual_train = VisualDataset(
            args.visual_manifest, split="train", window=visual_window, warmup=visual_warmup)
        visual_calibration = VisualDataset(
            args.visual_manifest, split="calibration", window=visual_window,
            warmup=visual_warmup)
        visual_stream = HandLoopBatchStream(
            visual_train, args.batch_size_visual, seed=args.seed + 2,
            num_workers=args.num_workers, shuffle=True)
        visual_validation = _validation_batches(
            visual_calibration, args.batch_size_visual, args.num_workers, args.seed + 3)

    optimizer = make_optimizer(
        named_parameters,
        rates,
        weight_decay=float(config.get("training", {}).get("weight_decay", 0.01)),
    )
    trainer = HandLoopTrainer(
        model,
        optimizer,
        total_steps=args.steps,
        warmup_steps=int(config.get("training", {}).get("warmup_steps", 20)),
        final_lr_ratio=float(config.get("training", {}).get("final_lr_ratio", 0.1)),
        max_grad_norm=float(config.get("training", {}).get("max_grad_norm", 1.0)),
    )
    stage_tag = "motion_pretrain" if args.stage == "motion" else "joint_visual"
    training_config = config.get("training", {})
    loss_config = dict(config.get("visual_loss", training_config.get("visual_loss", {})))
    if not loss_config:
        loss_config = {
            key: training_config[key]
            for key in ("state_weight", "wrist_weight", "joint_weight", "wrist_scale", "joint_scale")
            if key in training_config
        }
    base_checkpoint = {}
    if args.stage == "joint":
        base_checkpoint = {
            "hawor_checkpoint": str(args.hawor_checkpoint.resolve()),
            "hawor_sha256": _sha256(args.hawor_checkpoint),
            "motion_checkpoint": str(args.motion_checkpoint.resolve()),
            "motion_sha256": _sha256(args.motion_checkpoint),
        }
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            trainer=trainer,
            optimizer=optimizer,
            stage=stage_tag,
            split_fingerprint=json.dumps(split_hashes, sort_keys=True),
            map_location="cpu",
        )
        if checkpoint.get("config") != config:
            raise ValueError("Resume checkpoint was created with a different HandLoop config")
        if checkpoint.get("base_checkpoint", {}) != base_checkpoint:
            raise ValueError("Resume checkpoint base HaWoR/TWM checkpoints differ")
        stream_states = checkpoint.get("extra", {}).get("streams")
        if not isinstance(stream_states, dict) or "motion" not in stream_states:
            raise ValueError("Resume checkpoint is missing the exact training data-stream cursor")
        motion_stream.load_state_dict(stream_states["motion"])
        if visual_stream is not None:
            if "visual" not in stream_states:
                raise ValueError("Resume checkpoint is missing the visual data-stream cursor")
            visual_stream.load_state_dict(stream_states["visual"])
    last_validation: dict[str, float] = {}
    while trainer.global_step < trainer.total_steps:
        if args.stage == "motion":
            batch = _move(next(motion_stream), device)
            results = motion_training_step(model, batch)
        else:
            task = HandLoopTrainer.task_for_step(trainer.global_step)
            if task == "motion":
                batch = _move(next(motion_stream), device)
                results = motion_training_step(joint_system.twm, batch)
            else:
                batch = _move(next(visual_stream), device)
                results = visual_loss(joint_system, batch, loss_config)
        stats = trainer.optimize(results["loss"])
        current_step = trainer.global_step

        if current_step % args.save_every == 0 or current_step == trainer.total_steps:
            if args.stage == "motion":
                validation = _validate_motion(model, calibration_motion_loader, device)
            elif HandLoopTrainer.task_for_step(current_step - 1) == "motion":
                validation = _validate_motion(joint_system.twm, calibration_motion_loader, device)
            else:
                validation = _validate_visual(
                    visual_loss, joint_system, visual_validation, device, loss_config)
            last_validation = {"calibration_loss": validation}
            print(json.dumps({
                "step": current_step,
                "stage": args.stage,
                "loss": stats["loss"],
                "validation": validation,
                "grad_norms": {k: v for k, v in stats.items() if k.startswith("grad_norm/")},
            }, sort_keys=True), flush=True)
            stream_states = {"motion": motion_stream.state_dict()}
            if visual_stream is not None:
                stream_states["visual"] = visual_stream.state_dict()
            save_checkpoint(
                args.output,
                model=model,
                trainer=trainer,
                optimizer=optimizer,
                stage=stage_tag,
                config=config,
                split_fingerprint=json.dumps(split_hashes, sort_keys=True),
                base_checkpoint=base_checkpoint,
                extra={
                    "last_validation": last_validation,
                    "split_counts": split_hashes,
                    "streams": stream_states,
                    "resume_sampling": "exact_epoch_permutation_and_batch_cursor",
                },
            )


if __name__ == "__main__":
    main()
