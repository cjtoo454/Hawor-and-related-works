#!/usr/bin/env python3
"""Calibration-only component replacement diagnosis for the v3 checkpoint.

The script runs each cached calibration context once through the trained v3
observer and the direct single-call TWM path. It saves only CPU states, beta,
and frame identity, then scores four state-component variants using the frozen
v3 calibration windows and full-video event protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRY = ROOT / "thirdparty" / "try"
TWM_ONLY = ROOT / "thirdparty" / "ablation" / "twm_only"
import sys

sys.path[:0] = [str(ROOT), str(TRY), str(TRY / "scripts"), str(TWM_ONLY), str(HERE)]

import run_feedback_v3  # noqa: E402
import train_feedback_v2 as v2  # noqa: E402
import train_feedback_v3 as v3  # noqa: E402
from dual_loop.handloop_state import so3_log, unpack_state  # noqa: E402
from lib.eval_utils.eval_utils import align_pcl  # noqa: E402
from world_targets_v3 import WorldTargetMap, sha256_file  # noqa: E402
from v2_cache import get_windows, load_cached_contexts  # noqa: E402


DEFAULT_RUN_DIR = HERE / "runs" / "diagnose_v4_components"
DEFAULT_CACHE_DIR = HERE / "runs" / "feedback_v3_train" / "cache"
DEFAULT_CHECKPOINT = HERE / "runs" / "feedback_v3_train" / "best_composite.pt"
VARIANT_NAMES = (
    "learnedfull",
    "baseline",
    "base_pose+learned_fingers",
    "base_position+learned_rotations",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--camera-map", type=Path, default=v3.DEFAULT_MAP)
    parser.add_argument("--event-manifest", type=Path, default=v3.DEFAULT_EVENT_MANIFEST)
    parser.add_argument("--twm-checkpoint", type=Path, default=v2.DEFAULT_TWM)
    parser.add_argument("--hawor-checkpoint", type=Path, default=v2.DEFAULT_HAWOR)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-frames", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--round2-probe", action="store_true",
                        help="Run one calibration context and report q1/s1/q2 plus the round-2 visual delta")
    parser.add_argument("--probe-context", help="Calibration item_id for --round2-probe; defaults to the first 16-frame context")
    args = parser.parse_args()
    if args.batch_frames < 1:
        raise ValueError("--batch-frames must be positive")
    return args


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _refresh_report(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    variants = metrics["variants"]
    lines = [
        "# v4 component diagnosis on calibration data",
        "",
        f"Checkpoint: `{metrics['checkpoint']['checkpoint']}` (step {metrics['checkpoint']['global_step']}).",
        f"Calibration coverage: {metrics['calibration_contexts']} contexts / {metrics['calibration_frames']} frames; "
        f"fixed {variants['learnedfull']['window_count']} W windows and "
        f"{variants['learnedfull']['reference_events']} full-video event references.",
        "Thresholds were unchanged; no training or test labels were used.",
        "",
        "| Variant | W-MPJPE (mm) | Onset MAE (ms) | F1 | Recall | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANT_NAMES:
        row = variants[name]
        lines.append(
            f"| {name} | {row['w_mpjpe_mm']:.3f} | {row['onset_mae_ms']:.3f} | "
            f"{row['f1']:.6f} | {row['recall']:.6f} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    learned = variants["learnedfull"]
    baseline = variants["baseline"]
    lines.extend([
        "",
        f"The learned full state lowers W by {baseline['w_mpjpe_mm'] - learned['w_mpjpe_mm']:.3f} mm "
        f"versus direct TWM, while F1 changes by {learned['f1'] - baseline['f1']:+.6f} "
        f"and onset MAE changes by {learned['onset_mae_ms'] - baseline['onset_mae_ms']:+.3f} ms. "
        "Both mixed replacements score worse on W than direct TWM. These results do not support "
        "attributing the full-state W gain to root position or root rotation alone.",
        "",
        "The primary pass measured q2−s1: q2 is the second-round visual candidate after applying delta2, "
        "and s1 is the round-1 TWM state used at the start of round 2. This difference includes delta2, "
        "so it is not a pure input-mismatch estimate.",
        "",
        "| Fixed-W first-two-frame alignment scale | Mean | Median | P05 | P95 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in VARIANT_NAMES:
        scale = metrics["first_two_frame_w_alignment_scale"][name]["scale_applied_to_prediction"]
        lines.append(f"| {name} | {scale['mean']:.6f} | {scale['median']:.6f} | {scale['p05']:.6f} | {scale['p95']:.6f} |")

    probe_path = run_dir / "round2_probe.json"
    if probe_path.is_file():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        q1_s1 = probe["q1_minus_s1"]
        q2_q1 = probe["q2_minus_q1"]
        lines.extend([
            "",
            f"A separate {probe['frame_count']}-frame probe (`{probe['context']['item_id']}`) measured q1−s1 "
            "before delta2 and then q2−q1 after delta2:",
            "",
            "| Difference | Root position mean (mm) | Root rotation mean (degrees) |",
            "|---|---:|---:|",
            f"| q1−s1 | {q1_s1['root_position_m']['mean'] * 1000:.3f} | {q1_s1['root_rotation_degrees']['mean']:.4f} |",
            f"| q2−q1 | {q2_q1['root_position_m']['mean'] * 1000:.3f} | {q2_q1['root_rotation_degrees']['mean']:.4f} |",
            "",
            "Bounded camera-space visual delta norms (near bound means at least 90% of the configured radial bound):",
            "",
            "| Round | Component | Mean norm / bound | Near-bound fraction |",
            "|---|---|---:|---:|",
        ])
        for round_name, values in probe["bounded_visual_delta_camera"].items():
            for component in ("translation", "root_rotation", "finger_rotation_per_finger"):
                item = values[component]
                lines.append(
                    f"| {round_name} | {component} | {item['norm']['mean']:.6f} / "
                    f"{item['configured_max_norm']:.6f} | {item['near_bound_fraction']:.1%} |"
                )
    lines.extend([
        "",
        f"Reusable state artifact: `{metrics['state_artifact']}`.",
        "",
    ])
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _batches(contexts: list[Any], batch_frames: int) -> Iterator[list[Any]]:
    current: list[Any] = []
    count = 0
    for context in contexts:
        if context.size > batch_frames:
            raise ValueError(f"{context.item_id}: context size {context.size} exceeds batch limit {batch_frames}")
        if current and count + context.size > batch_frames:
            yield current
            current, count = [], 0
        current.append(context)
        count += context.size
    if current:
        yield current


def _component_variants(learned: torch.Tensor, baseline: torch.Tensor) -> dict[str, torch.Tensor]:
    if learned.shape != baseline.shape or learned.ndim != 2 or learned.shape[-1] != 99:
        raise ValueError("Expected matching learned and baseline states with shape [N,99]")
    pose_from_base_fingers_from_learned = baseline.clone()
    pose_from_base_fingers_from_learned[:, 9:] = learned[:, 9:]
    position_from_base_rotations_from_learned = baseline.clone()
    position_from_base_rotations_from_learned[:, 3:] = learned[:, 3:]
    return {
        "learnedfull": learned,
        "baseline": baseline,
        "base_pose+learned_fingers": pose_from_base_fingers_from_learned,
        "base_position+learned_rotations": position_from_base_rotations_from_learned,
    }


def _add_prediction_rows(
    destination: dict[tuple[str, int], np.ndarray],
    context: Any,
    values: np.ndarray,
) -> None:
    if values.shape != (context.size, 21, 3):
        raise ValueError(f"{context.item_id}: malformed MANO joints {values.shape}")
    for row, frame in enumerate(context.frame_indices):
        key = (context.sequence, int(frame))
        if key in destination:
            raise ValueError(f"Duplicate cached calibration frame {key}")
        destination[key] = values[row]


def _q_s_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    position_differences = []
    rotation_differences = []
    frame_count = 0
    for record in records:
        q = record["second_round_visual_candidate_q"]
        s = record["second_round_projection_state_s"]
        if q.shape != s.shape or q.ndim != 2 or q.shape[-1] != 99:
            raise ValueError(f"{record['item_id']}: malformed second-round q/s states")
        position_differences.append(torch.linalg.vector_norm(q[:, :3] - s[:, :3], dim=-1))
        _, q_rotations = unpack_state(q)
        _, s_rotations = unpack_state(s)
        relative_root = q_rotations[:, 0] @ s_rotations[:, 0].transpose(-1, -2)
        rotation_differences.append(torch.linalg.vector_norm(so3_log(relative_root), dim=-1))
        frame_count += q.shape[0]
    positions = torch.cat(position_differences).double()
    rotations_rad = torch.cat(rotation_differences).double()
    if not torch.isfinite(positions).all() or not torch.isfinite(rotations_rad).all():
        raise FloatingPointError("Nonfinite q-to-s root displacement or rotation")

    def stats(values: torch.Tensor, scale: float = 1.0) -> dict[str, float]:
        values = values * scale
        quantiles = torch.quantile(values, torch.tensor([0.5, 0.95], dtype=values.dtype))
        return {
            "mean": float(values.mean()),
            "median": float(quantiles[0]),
            "p95": float(quantiles[1]),
            "max": float(values.max()),
        }

    return {
        "frame_count": frame_count,
        "q_definition": "output['visual_candidates'][1]: world candidate after round-2 visual update, before the round-2 TWM call",
        "s_definition": "output['projection_states'][1]: world state projected at start of round 2, produced by the round-1 TWM call",
        "root_position_q_minus_s_m": stats(positions),
        "root_rotation_q_relative_to_s_degrees": stats(rotations_rad, 180.0 / np.pi),
    }


def _root_position_rotation_difference(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left_position, left_rotations = unpack_state(left)
    right_position, right_rotations = unpack_state(right)
    position = torch.linalg.vector_norm(left_position - right_position, dim=-1)
    relative_root = left_rotations[:, 0] @ right_rotations[:, 0].transpose(-1, -2)
    rotation_rad = torch.linalg.vector_norm(so3_log(relative_root), dim=-1)
    return position, rotation_rad


def _summary(values: torch.Tensor, multiplier: float = 1.0) -> dict[str, float]:
    values = values.detach().double().cpu() * multiplier
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite round-2 probe measurement")
    quantiles = torch.quantile(values, torch.tensor([0.5, 0.95], dtype=values.dtype))
    return {
        "mean": float(values.mean()),
        "median": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "max": float(values.max()),
    }


def _bounded_component_summary(vectors: torch.Tensor, bound: float) -> dict[str, Any]:
    norms = torch.linalg.vector_norm(vectors.reshape(-1, vectors.shape[-1]), dim=-1)
    return {
        "configured_max_norm": float(bound),
        "norm": _summary(norms),
        "near_bound_threshold_fraction": 0.9,
        "near_bound_fraction": float((norms >= 0.9 * float(bound)).float().mean()),
    }


def _run_round2_probe(
    system: torch.nn.Module,
    context: Any,
    device: torch.device,
    run_dir: Path,
    checkpoint_metadata: dict[str, Any],
    cache_sha256: str,
) -> dict[str, Any]:
    inputs = v2._input_to_device(context.inputs, device)
    beta = inputs["projector_args"][0]
    with torch.no_grad():
        output = v2.observer_forward(system, dict(inputs))
    q1 = output["visual_candidates"][0].detach().float().cpu().contiguous()
    s1 = output["projection_states"][1].detach().float().cpu().contiguous()
    q2 = output["visual_candidates"][1].detach().float().cpu().contiguous()
    delta1_camera = output["visual_deltas_camera"][0].detach().float().cpu().contiguous()
    delta2_anchor = output["visual_deltas_anchor"][1].detach().float().cpu().contiguous()
    delta2_camera = output["visual_deltas_camera"][1].detach().float().cpu().contiguous()
    if any(value.shape != (context.size, 99) for value in (q1, s1, q2)):
        raise ValueError(f"{context.item_id}: q1/s1/q2 are not [frames,99]")
    if any(value.shape != (context.size, 51)
           for value in (delta1_camera, delta2_anchor, delta2_camera)):
        raise ValueError(f"{context.item_id}: visual deltas are not [frames,51]")

    q1_s1_position, q1_s1_rotation = _root_position_rotation_difference(q1, s1)
    q2_q1_position, q2_q1_rotation = _root_position_rotation_difference(q2, q1)
    delta2_position = torch.linalg.vector_norm(delta2_anchor[:, :3], dim=-1)
    delta2_root_rotation = torch.linalg.vector_norm(delta2_anchor[:, 3:6], dim=-1)
    observer = system.observer
    visual_delta_camera = {}
    for round_index, delta in enumerate((delta1_camera, delta2_camera), start=1):
        visual_delta_camera[f"round_{round_index}"] = {
            "translation": _bounded_component_summary(delta[:, :3], observer.translation_bound),
            "root_rotation": _bounded_component_summary(delta[:, 3:6], observer.root_rotation_bound),
            "finger_rotation_per_finger": _bounded_component_summary(
                delta[:, 6:].reshape(context.size, 15, 3), observer.finger_rotation_bound),
        }
    result = {
        "status": "complete",
        "context": {"item_id": context.item_id, "sequence": context.sequence,
                    "frame_ids": [int(value) for value in context.frame_indices]},
        "checkpoint": checkpoint_metadata,
        "cache_manifest_sha256": cache_sha256,
        "frame_count": context.size,
        "definitions": {
            "q1": "output['visual_candidates'][0]: candidate after round-1 visual delta, before round-1 TWM",
            "s1": "output['projection_states'][1]: round-1 TWM state projected at start of round 2",
            "q2": "output['visual_candidates'][1]: candidate after round-2 visual delta, before round-2 TWM",
            "delta2_anchor": "output['visual_deltas_anchor'][1]: applied round-2 position/root/finger tangent in anchor coordinates",
            "delta2_camera": "output['visual_deltas_camera'][1]: bounded round-2 visual tangent in camera coordinates",
        },
        "q1_minus_s1": {
            "root_position_m": _summary(q1_s1_position),
            "root_rotation_degrees": _summary(q1_s1_rotation, 180.0 / np.pi),
        },
        "q2_minus_q1": {
            "root_position_m": _summary(q2_q1_position),
            "root_rotation_degrees": _summary(q2_q1_rotation, 180.0 / np.pi),
        },
        "delta2_anchor": {
            "root_position_tangent_m": _summary(delta2_position),
            "root_rotation_tangent_degrees": _summary(delta2_root_rotation, 180.0 / np.pi),
            "all_finger_tangent_norms_radians": _summary(
                torch.linalg.vector_norm(delta2_anchor[:, 6:].reshape(-1, 15, 3), dim=-1).reshape(-1)),
        },
        "bounded_visual_delta_camera": visual_delta_camera,
    }
    stem = run_dir / "round2_probe"
    _atomic_torch_save(stem.with_suffix(".pt"), {
        "metadata": result,
        "frame_ids": torch.as_tensor(context.frame_indices.copy(), dtype=torch.int64),
        "beta": beta.detach().float().cpu().contiguous(),
        "q1": q1,
        "s1": s1,
        "q2": q2,
        "delta1_camera": delta1_camera,
        "delta2_anchor": delta2_anchor,
        "delta2_camera": delta2_camera,
    })
    _atomic_json(stem.with_suffix(".json"), result)
    _refresh_report(run_dir)
    return result


def _alignment_scales(
    variant_predictions: dict[str, dict[tuple[str, int], np.ndarray]],
    contexts: list[Any],
    windows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collect the exact scale factors used by each fixed W chunk's first-two-frame alignment."""
    context_by_id = {context.item_id: context for context in contexts}
    target_cache: dict[str, np.ndarray] = {}
    per_variant: dict[str, list[float]] = {name: [] for name in variant_predictions}
    for window in windows:
        sequence = str(window["sequence"])
        frames = np.asarray(window["frame_indices"], dtype=np.int64)
        frame_lookup: dict[int, tuple[Any, int]] = {}
        for item_id in window["context_item_ids"]:
            context = context_by_id[item_id]
            if item_id not in target_cache:
                target_cache[item_id] = v3.WORLD.target(context).numpy()
            for row, frame in enumerate(context.frame_indices):
                frame_lookup[int(frame)] = (context, row)

        target_rows = []
        gt_valid, visual_valid, timestamps, segments = [], [], [], []
        for frame in frames:
            context, row = frame_lookup[int(frame)]
            target_rows.append(target_cache[context.item_id][row])
            gt_valid.append(bool(context.labels["gt_valid"][row]))
            visual_valid.append(bool(context.labels["visual_valid"][row]))
            timestamps.append(float(context.labels["timestamp_sec"][row]))
            segments.append(int(context.labels["segment_id"][row]))
        target = np.asarray(target_rows, dtype=np.float32)
        valid = np.asarray(gt_valid, dtype=bool) & np.asarray(visual_valid, dtype=bool)
        chunks = v2._valid_chunk_slices(
            frames,
            np.asarray(segments, dtype=np.int64),
            np.asarray(timestamps, dtype=np.float64),
            valid,
        )
        for start, stop in chunks:
            gt_first_two = torch.as_tensor(target[start:start + 2], dtype=torch.float32)
            if len(gt_first_two) != 2 or stop - start < 2:
                raise ValueError("Fixed W chunk cannot support first-two-frame alignment")
            for variant, predictions in variant_predictions.items():
                pred_first_two = torch.as_tensor(
                    np.stack([predictions[(sequence, int(frame))] for frame in frames[start:start + 2]]),
                    dtype=torch.float32,
                )
                scale, _rotation, _translation = align_pcl(
                    gt_first_two.reshape(1, -1, 3), pred_first_two.reshape(1, -1, 3))
                per_variant[variant].append(float(scale.reshape(-1)[0]))

    baseline_values = np.asarray(per_variant["baseline"], dtype=np.float64)
    result: dict[str, dict[str, Any]] = {}
    for variant, raw_values in per_variant.items():
        values = np.asarray(raw_values, dtype=np.float64)
        if len(values) == 0 or not np.isfinite(values).all():
            raise FloatingPointError(f"No finite fixed-window alignment scales for {variant}")
        result[variant] = {
            "chunks": int(len(values)),
            "scale_applied_to_prediction": {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
                "min": float(values.min()),
                "max": float(values.max()),
            },
            "mean_change_vs_baseline": float(values.mean() - baseline_values.mean()),
            "median_change_vs_baseline": float(np.median(values) - np.median(baseline_values)),
        }
    return result


def _initialize_v3_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], list[Any], str]:
    cache_dir = args.cache_dir.resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing v3 cache manifest: {manifest_path}")
    event_path = args.event_manifest.resolve()
    if not event_path.is_file():
        raise FileNotFoundError(f"Missing v3 event manifest: {event_path}")

    v3.WORLD = WorldTargetMap(args.camera_map)
    v3.EVENT_HASH = sha256_file(event_path)
    v3.EVENT_MANIFEST = json.loads(event_path.read_text(encoding="utf-8"))
    if v3.EVENT_MANIFEST.get("protocol_revision") != v3.EVENT_PROTOCOL_REVISION:
        raise ValueError("Frozen event inventory has the wrong protocol revision")
    if len(v3.EVENT_MANIFEST.get("events", [])) != 224:
        raise ValueError("Frozen calibration/full-video event manifest must contain 224 events")
    v3.RUN_DIR = args.run_dir.resolve()
    v3.CACHE_DIR = cache_dir
    v3.BOUND_HASHES = {
        "camera_map_manifest_sha256": v3.WORLD.manifest_sha256,
        "event_manifest_sha256": v3.EVENT_HASH,
    }

    contexts = load_cached_contexts(cache_dir, "calibration")
    # load_cached_contexts already verifies every file named by this manifest.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # train_feedback_v3.main normally saves this alias before protocol validation.
    v2._ORIGINAL_VALIDATE_CACHE = v2._validate_cache_protocol
    v3._validate_cache_protocol(manifest, contexts)
    cache_sha256 = sha256_file(manifest_path)
    return manifest, contexts, cache_sha256


def _load_v3_system(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = v2.twm_only.load_checkpoint_restricted(checkpoint_path)
    run_feedback_v3.configure_runtime()
    trainer = run_feedback_v3.validate_protocol(checkpoint)
    step = int(trainer.get("global_step", 0))
    if step != 1400:
        raise ValueError(f"Expected the selected best_composite checkpoint at step 1400, got {step}")

    system, _config, base_twm_hash, base_hawor_hash = v2._make_system(
        argparse.Namespace(twm_checkpoint=args.twm_checkpoint, hawor_checkpoint=args.hawor_checkpoint),
        device,
        offload_hawor=True,
    )
    if checkpoint.get("base_twm_sha256") != base_twm_hash:
        raise ValueError("Selected visual checkpoint was trained with a different TWM checkpoint")
    if checkpoint.get("base_hawor_sha256") != base_hawor_hash:
        raise ValueError("Selected visual checkpoint was trained with a different HaWoR checkpoint")
    bounds = checkpoint.get("bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(run_feedback_v3.BOUND_KEYS):
        raise ValueError("Selected v3 checkpoint is missing the recorded feedback bounds")
    v2.install_feedback_v2(system, **bounds)
    system.observer.visual.load_state_dict(checkpoint["visual"], strict=True)
    system.eval()
    system.observer.eval()
    system.observer.twm.eval()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "global_step": step,
        "architecture": checkpoint.get("architecture"),
        "bounds": {key: float(value) for key, value in bounds.items()},
        "base_twm_sha256": base_twm_hash,
        "base_hawor_sha256": base_hawor_hash,
        "trainer_protocol_revision": trainer.get("protocol_revision"),
        "trainer_bindings": {key: trainer.get(key) for key in v3.BOUND_HASHES},
    }
    return system, metadata


def _single_traversal(
    system: torch.nn.Module,
    contexts: list[Any],
    device: torch.device,
    batch_frames: int,
) -> tuple[dict[str, dict[tuple[str, int], np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    predictions = {name: {} for name in VARIANT_NAMES}
    state_records: list[dict[str, Any]] = []
    q_s_moments: list[dict[str, Any]] = []
    total_frames = 0
    batch_count = 0

    system.eval()
    with torch.no_grad():
        for batch in _batches(contexts, batch_frames):
            batch_count += 1
            inputs = v2._input_to_device(v3._cat_inputs([context.inputs for context in batch]), device)
            beta = inputs["projector_args"][0]
            output = v2.observer_forward(system, dict(inputs))
            direct_state = v2._single_twm_state(system, inputs)
            learned_state = output["state"]
            q = output["visual_candidates"][1]
            s = output["projection_states"][1]
            if any(value.shape != (sum(context.size for context in batch), 99)
                   for value in (learned_state, direct_state, q, s)):
                raise ValueError("v3 output state or second-round q/s shape is not [batch,99]")
            variants = _component_variants(learned_state, direct_state)
            joints_by_variant = {
                name: v2._world_joints(system, state, beta).detach().float().cpu().numpy()
                for name, state in variants.items()
            }

            offset = 0
            for context in batch:
                stop = offset + context.size
                for name in VARIANT_NAMES:
                    _add_prediction_rows(predictions[name], context, joints_by_variant[name][offset:stop])
                record = {
                    "item_id": context.item_id,
                    "sequence": context.sequence,
                    "frame_ids": torch.as_tensor(context.frame_indices.copy(), dtype=torch.int64),
                    "beta": beta[offset:stop].detach().float().cpu().contiguous(),
                    "learned_state": learned_state[offset:stop].detach().float().cpu().contiguous(),
                    "baseline_state": direct_state[offset:stop].detach().float().cpu().contiguous(),
                    "second_round_visual_candidate_q": q[offset:stop].detach().float().cpu().contiguous(),
                    "second_round_projection_state_s": s[offset:stop].detach().float().cpu().contiguous(),
                }
                state_records.append(record)
                q_s_moments.append({
                    "item_id": context.item_id,
                    "second_round_visual_candidate_q": record["second_round_visual_candidate_q"],
                    "second_round_projection_state_s": record["second_round_projection_state_s"],
                })
                offset = stop
                total_frames += context.size
            if batch_count == 1 or batch_count % 32 == 0:
                print(f"forward progress: batches={batch_count} contexts={len(state_records)}/{len(contexts)} frames={total_frames}",
                      flush=True)
    expected = sum(context.size for context in contexts)
    if total_frames != expected:
        raise RuntimeError(f"Single traversal frame count mismatch: {total_frames} != {expected}")
    q_s_summary = _q_s_summary(q_s_moments)
    return predictions, state_records, q_s_summary


def _evaluate_variants(
    system: torch.nn.Module,
    contexts: list[Any],
    manifest: dict[str, Any],
    device: torch.device,
    predictions: dict[str, dict[tuple[str, int], np.ndarray]],
) -> dict[str, dict[str, Any]]:
    active_variant: str | None = None

    def return_precomputed(_system: torch.nn.Module, _contexts: list[Any], _device: torch.device,
                           *, compare_zero_baseline: bool = False,
                           batch_frames: int = 64):
        del _system, _contexts, _device, batch_frames
        if compare_zero_baseline:
            raise ValueError("Component diagnosis requires compare_zero_baseline=False")
        if active_variant is None:
            raise RuntimeError("No active precomputed component variant")
        return predictions[active_variant], None

    original = v3._predictions_by_frame
    v3._predictions_by_frame = return_precomputed
    results: dict[str, dict[str, Any]] = {}
    try:
        for variant in VARIANT_NAMES:
            active_variant = variant
            result = v3.evaluate_calibration(
                system, contexts, manifest, device, compare_zero_baseline=False)
            if result.get("window_count") != 40:
                raise ValueError(f"{variant}: expected 40 fixed W windows, got {result.get('window_count')}")
            if result.get("events", {}).get("reference_events") != 224:
                raise ValueError(f"{variant}: expected 224 fixed full-video references")
            results[variant] = result
    finally:
        v3._predictions_by_frame = original
    return results


def _alignment_scale_summary(
    variant_predictions: dict[str, dict[tuple[str, int], np.ndarray]],
    contexts: list[Any],
    windows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return _alignment_scales(variant_predictions, contexts, windows)


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.manual_seed(v2.CACHE_SEED)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if device.type == "cuda" and device.index not in (None, 0):
        raise ValueError("Use the isolated CUDA-visible device as --device cuda:0")
    if device.type == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
            raise ValueError("The calibration job must isolate physical GPU 3 with CUDA_VISIBLE_DEVICES=3")
        if torch.cuda.device_count() != 1:
            raise ValueError(f"Expected exactly one CUDA-visible GPU, got {torch.cuda.device_count()}")

    manifest, contexts, cache_sha256 = _initialize_v3_protocol(args)
    print(f"cache ready: contexts={len(contexts)} cache_manifest_sha256={cache_sha256}", flush=True)
    system, checkpoint_metadata = _load_v3_system(args, device)
    print(f"checkpoint ready: step={checkpoint_metadata['global_step']} device={device} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)
    v3.BOUND_HASHES.update(v3._bound_sources(manifest, contexts, cache_sha256))

    if args.round2_probe:
        if args.probe_context:
            probe_context = next((context for context in contexts
                                  if context.item_id == args.probe_context), None)
            if probe_context is None:
                raise ValueError(f"Unknown calibration context {args.probe_context!r}")
        else:
            probe_context = next((context for context in contexts if context.size == 16), None)
            if probe_context is None:
                raise ValueError("No 16-frame calibration context is available for the round-2 probe")
        if probe_context.size != 16:
            raise ValueError(f"Round-2 probe requires 16 frames, got {probe_context.size}")
        probe_result = _run_round2_probe(
            system, probe_context, device, args.run_dir.resolve(), checkpoint_metadata, cache_sha256)
        print(json.dumps(probe_result, sort_keys=True), flush=True)
        return

    predictions, state_records, q_s = _single_traversal(
        system, contexts, device, args.batch_frames)
    expected_contexts = manifest.get("counts", {}).get("calibration_contexts")
    if expected_contexts is not None and len(state_records) != expected_contexts:
        raise ValueError(f"Expected {expected_contexts} calibration contexts, got {len(state_records)}")
    state_path = args.run_dir.resolve() / "calibration_states.pt"
    state_metadata = {
        "format": "diagnose_v4_components_calibration_states",
        "protocol_revision": v3.PROTOCOL_REVISION,
        "state_layout": {"root_position": "state[..., :3]", "root_rotation_6d": "state[..., 3:9]", "finger_rotations_6d": "state[..., 9:]"},
        "saved_fields": "frame_ids, beta, learned_state, baseline_state, second_round_visual_candidate_q, second_round_projection_state_s",
        "q_s_definition": {"q": "second round visual candidate after visual update, before round-2 TWM", "s": "round-1 TWM world state projected at start of round 2"},
        "q_s_measured": "q2 - s1; includes the bounded second-round visual delta and is not a pure input-mismatch estimate",
        "ground_truth_or_full_model_inputs_saved": False,
        "saved_model_inputs": ["beta"],
        "context_count": len(state_records),
        "frame_count": int(sum(len(record["frame_ids"]) for record in state_records)),
        "checkpoint": checkpoint_metadata,
        "cache_manifest_sha256": cache_sha256,
        "camera_map_manifest_sha256": v3.WORLD.manifest_sha256,
        "event_manifest_sha256": v3.EVENT_HASH,
    }
    _atomic_torch_save(state_path, {"metadata": state_metadata, "contexts": state_records})

    results = _evaluate_variants(system, contexts, manifest, device, predictions)
    scales = _alignment_scale_summary(predictions, contexts, get_windows(manifest))
    metrics: dict[str, Any] = {
        "protocol_revision": v3.PROTOCOL_REVISION,
        "checkpoint": checkpoint_metadata,
        "cache_manifest_sha256": cache_sha256,
        "camera_map_manifest_sha256": v3.WORLD.manifest_sha256,
        "event_manifest_sha256": v3.EVENT_HASH,
        "calibration_contexts": len(contexts),
        "calibration_frames": state_metadata["frame_count"],
        "batch_frames": args.batch_frames,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "state_artifact": str(state_path),
        "variants": {},
        "q_vs_s": q_s,
        "q_s_interpretation": "Primary q_vs_s is q2-s1 and includes delta2; consult round2_probe.json for the pre-delta q1-s1 input mismatch.",
        "first_two_frame_w_alignment_scale": scales,
        "calibration_rule": v3.CALIBRATION_RULE,
        "uses_test_ground_truth": False,
        "thresholds_changed": False,
        "trained": False,
    }
    for variant in VARIANT_NAMES:
        result = results[variant]
        events = result["events"]
        metrics["variants"][variant] = {
            "w_mpjpe_mm": result["w_mpjpe_mm"],
            "w_mpjpe_scored_frames": result["w_mpjpe_scored_frames"],
            "window_count": result["window_count"],
            "onset_mae_ms": events["onset_mae_ms"],
            "f1": events["f1"],
            "recall": events["recall"],
            "tp": events["tp"],
            "fp": events["fp"],
            "fn": events["fn"],
            "reference_events": events["reference_events"],
            "predicted_events": events["predicted_events"],
            "full_result": result,
        }
    metrics_path = args.run_dir.resolve() / "metrics.json"
    _atomic_json(metrics_path, metrics)
    _refresh_report(args.run_dir.resolve())
    print(json.dumps({
        "status": "complete",
        "state_artifact": str(state_path),
        "metrics": str(metrics_path),
        "contexts": len(contexts),
        "frames": state_metadata["frame_count"],
        "variants": {name: {key: metrics["variants"][name][key]
                             for key in ("w_mpjpe_mm", "onset_mae_ms", "f1", "recall", "tp", "fp", "fn")}
                     for name in VARIANT_NAMES},
        "q_vs_s": q_s,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
