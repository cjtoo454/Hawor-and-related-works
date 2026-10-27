#!/usr/bin/env python3
"""Train v3 bounded feedback against canonical world GT and full-video events."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import train_feedback_v2 as v2
from lib.eval_utils.eval_utils import first_align_joints
from dual_loop.pinch_evaluation import extract_events, match_events
from dual_loop.oracle import normalized_pinch_distance
from world_targets_v3 import DEFAULT_MAP, WorldTargetMap, sha256_file


HERE = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = HERE / "runs/feedback_v3_train"
DEFAULT_EVENT_MANIFEST = HERE / "runs/calibration_events_v3/calibration_manifest.json"
PROTOCOL_REVISION = 3
EVENT_PROTOCOL_REVISION = 2  # frozen full-video event inventory consumed by v3
CALIBRATION_RULE = "fixed original 40 clips for canonical raw-world W; full-video GT event references and fixed formal eligibility intersected with expanded feature coverage"
SELECTION_RULE = "W strictly below fresh-zero baseline; onset MAE no worse; F1 and recall no worse; baseline event TP >= 10"
LOSS_CONFIG = {
    **v2.LOSS_CONFIG,
    "camera_joint_weight": 0.25,
    "aligned_world_joint_weight": 1.0,
    "camera_relative_nll_weight": 1e-4,
    "world_alignment": "first two valid consecutive frames per run; differentiable similarity alignment",
}
WORLD: WorldTargetMap | None = None
EVENT_MANIFEST: dict[str, Any] | None = None
EVENT_HASH: str | None = None
RUN_DIR: Path | None = None
CACHE_DIR: Path | None = None
BOUND_HASHES: dict[str, str] = {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twm-checkpoint", type=Path, default=v2.DEFAULT_TWM)
    parser.add_argument("--hawor-checkpoint", type=Path, default=v2.DEFAULT_HAWOR)
    parser.add_argument("--data-root", type=Path, default=v2.DEFAULT_DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=v2.DEFAULT_VIDEO_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=v2.DEFAULT_SPLIT)
    parser.add_argument("--export-split", type=Path, default=v2.DEFAULT_EXPORT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=v2.DEFAULT_PROTOCOL)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--camera-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--event-manifest", type=Path, default=DEFAULT_EVENT_MANIFEST)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=v2.MAX_STEPS)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--cache-only", action="store_true", help="Unsupported in v3; use extend_calibration_cache.py")
    parser.add_argument("--rebuild-cache", action="store_true", help="Unsupported in v3; use extend_calibration_cache.py")
    parser.add_argument("--smoke-toy", action="store_true", help="Run the inherited v2 CPU smoke")
    args = parser.parse_args()
    if args.rebuild_cache or args.cache_only:
        raise ValueError("v3 consumes a completed revision-3 cache; build it with extend_calibration_cache.py")
    cache = Path(args.cache_dir or (args.run_dir / "cache"))
    if not (cache / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing v3 cache manifest: {cache / 'manifest.json'}")
    global WORLD, EVENT_MANIFEST, EVENT_HASH, RUN_DIR, CACHE_DIR, BOUND_HASHES
    WORLD = WorldTargetMap(Path(args.camera_map))
    event_path = Path(args.event_manifest).resolve()
    EVENT_HASH = sha256_file(event_path)
    EVENT_MANIFEST = json.loads(event_path.read_text(encoding="utf-8"))
    if EVENT_MANIFEST.get("protocol_revision") != EVENT_PROTOCOL_REVISION:
        raise ValueError("Frozen event inventory has the wrong protocol revision")
    RUN_DIR = Path(args.run_dir).resolve()
    CACHE_DIR = cache.resolve()
    BOUND_HASHES = {
        "camera_map_manifest_sha256": WORLD.manifest_sha256,
        "event_manifest_sha256": EVENT_HASH,
    }
    return args


def _validate_cache_protocol(manifest: dict[str, Any], contexts: list[Any]) -> None:
    v2._ORIGINAL_VALIDATE_CACHE(manifest, contexts)
    if manifest.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("v3 requires a revision-3 extended cache")
    if len(manifest.get("event_windows", [])) != len(EVENT_MANIFEST["merged_windows"]):
        raise ValueError("Extended event windows do not match event manifest")
    binding = manifest.get("calibration_event_manifest", {})
    if binding.get("sha256") != EVENT_HASH:
        raise ValueError("Extended cache is bound to another event manifest")
    if WORLD is None:
        raise RuntimeError("World map not initialized")
    expected = {str(item["sequence"]) for item in manifest["items"]}
    if expected != set(WORLD._files):
        raise ValueError("Camera maps do not cover exactly the cached train/calibration sequences")
    actual_video_root = Path(manifest["source_hashes"]["protocol_and_splits"]["export_split"]["path"]).resolve().parent
    fixed_video_root = (HERE.parents[2] / "datasets/hot3d_train100_export").resolve()
    if actual_video_root != fixed_video_root:
        raise ValueError("Revision-3 cache source video root is not the audited HOT3D export")
    parent = manifest.get("parent_manifest", {})
    parent_path = Path(parent["path"])
    if sha256_file(parent_path) != parent["sha256"]:
        raise ValueError("Parent cache manifest changed")
    parent_manifest = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_video_root = Path(parent_manifest["source_hashes"]["protocol_and_splits"]["export_split"]["path"]).resolve().parent
    if parent_video_root != actual_video_root:
        raise ValueError("Parent and extended cache video roots differ")
    for sequence, sources in manifest["source_hashes"]["sequences"].items():
        for key in ("detector_tracks", "focal"):
            path = Path(sources[key]["path"]).resolve()
            if not path.is_relative_to(fixed_video_root / sequence):
                raise ValueError(f"{sequence}: cache {key} source escapes audited video root")
    assert CACHE_DIR is not None
    BOUND_HASHES.update(_bound_sources(manifest, contexts, sha256_file(CACHE_DIR / "manifest.json")))


def _cat_inputs(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        if any(value.ndim != first.ndim or value.shape[1:] != first.shape[1:] for value in values):
            raise ValueError("Calibration contexts have incompatible trailing input shapes")
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        if any(value.keys() != first.keys() for value in values):
            raise ValueError("Calibration contexts have different input keys")
        return {key: _cat_inputs([value[key] for value in values]) for key in first}
    if isinstance(first, (tuple, list)):
        if any(len(value) != len(first) for value in values):
            raise ValueError("Calibration input tuple lengths differ")
        result = [_cat_inputs([value[index] for value in values]) for index in range(len(first))]
        return tuple(result) if isinstance(first, tuple) else result
    raise TypeError(f"Unsupported calibration input type {type(first)}")


def _predictions_by_frame(system: torch.nn.Module, contexts: list[Any], device: torch.device,
                          *, compare_zero_baseline: bool = False,
                          batch_frames: int = 64) -> tuple[dict[tuple[str, int], np.ndarray], float | None]:
    """Concatenate independent, unpadded contexts; every input is row local."""
    if batch_frames < 1:
        raise ValueError("batch_frames must be positive")
    predictions: dict[tuple[str, int], np.ndarray] = {}
    maximum_difference = 0.0 if compare_zero_baseline else None
    batches: list[list[Any]] = []
    current: list[Any] = []
    size = 0
    for context in contexts:
        if context.size > batch_frames:
            raise ValueError(f"Context {context.item_id} exceeds calibration batch limit")
        if current and size + context.size > batch_frames:
            batches.append(current)
            current, size = [], 0
        current.append(context)
        size += context.size
    if current:
        batches.append(current)
    with torch.no_grad():
        for batch in batches:
            inputs = v2._input_to_device(_cat_inputs([context.inputs for context in batch]), device)
            beta = inputs["projector_args"][0]
            output = v2.observer_forward(system, dict(inputs))
            states = output["state"]
            if compare_zero_baseline:
                direct = v2._single_twm_state(system, inputs)
                offset = 0
                for context in batch:
                    part = slice(offset, offset + context.size)
                    valid = context.labels["visual_valid"].to(device=device).bool()
                    if bool(valid.any()):
                        delta = (states[part][valid] - direct[part][valid]).abs().max()
                        maximum_difference = max(float(maximum_difference), float(delta.cpu()))
                        if not torch.allclose(states[part][valid], direct[part][valid], atol=1e-5, rtol=1e-5):
                            raise AssertionError(f"Fresh zero head differs from direct TWM: {context.item_id}")
                    offset += context.size
                states = direct
            joints = v2._world_joints(system, states, beta).detach().float().cpu().numpy()
            offset = 0
            for context in batch:
                for row, frame in enumerate(context.frame_indices):
                    key = (context.sequence, int(frame))
                    if key in predictions:
                        raise ValueError(f"Duplicate calibration frame {key}")
                    predictions[key] = joints[offset + row]
                offset += context.size
    return predictions, maximum_difference


def _valid_aligned_runs(labels: dict[str, torch.Tensor], valid: torch.Tensor) -> list[slice]:
    frames = labels["frame_indices"].detach().cpu().numpy()
    segments = labels["segment_id"].detach().cpu().numpy()
    times = labels["timestamp_sec"].detach().cpu().numpy()
    mask = valid.detach().cpu().numpy()
    starts = []
    start = None
    for i in range(len(mask)):
        connected = (i > 0 and mask[i-1] and frames[i] == frames[i-1] + 1
                     and segments[i] == segments[i-1] and 0 < times[i] - times[i-1] <= .25)
        if mask[i] and start is None:
            start = i
        elif mask[i] and not connected:
            if i - start >= 2:
                starts.append(slice(start, i))
            start = i
        elif not mask[i] and start is not None:
            if i - start >= 2:
                starts.append(slice(start, i))
            start = None
    if start is not None and len(mask) - start >= 2:
        starts.append(slice(start, len(mask)))
    return starts


def _aligned_world_loss(predicted: torch.Tensor, target: torch.Tensor,
                        labels: dict[str, torch.Tensor], valid: torch.Tensor) -> tuple[torch.Tensor, int]:
    errors = []
    for rows in _valid_aligned_runs(labels, valid):
        pred, gt = predicted[rows], target[rows]
        if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
            raise FloatingPointError("Nonfinite aligned world loss input")
        # first_align_joints currently constructs its SVD reflection matrix on
        # CPU; moving these <=16 rows to CPU preserves autograd to CUDA heads.
        pred_cpu, gt_cpu = pred.to("cpu"), gt.to("cpu")
        point = pred_cpu[:2].reshape(-1, 3)
        centered = point - point.mean(0)
        singular = torch.linalg.svdvals(centered.detach())
        if float(singular[-1]) < 1e-7 or float(singular[0] / singular[-1]) > 1e6:
            continue  # explicitly exclude geometrically degenerate first pairs
        aligned = first_align_joints(gt_cpu, pred_cpu)
        if not torch.isfinite(aligned).all():
            raise FloatingPointError("Nonfinite first-two-frame similarity alignment")
        errors.append(torch.linalg.vector_norm(aligned - gt_cpu, dim=-1).reshape(-1))
    if not errors:
        return predicted.new_zeros(()), 0
    return torch.cat(errors).mean().to(predicted.device), sum(len(value) for value in errors)


def _context_loss(system: torch.nn.Module, context: Any,
                  device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if WORLD is None:
        raise RuntimeError("World map unavailable")
    inputs = v2._input_to_device(context.inputs, device)
    output = v2.observer_forward(system, dict(inputs))
    labels = v2._labels_to_device(context.labels, device)
    canonical = WORLD.target(context).to(device)
    camera = labels["target_joints"]
    valid = labels["gt_valid"].bool() & labels["visual_valid"].bool()
    if not torch.isfinite(camera[valid]).all() or not torch.isfinite(canonical[valid]).all():
        raise FloatingPointError(f"{context.item_id}: nonfinite fixed supervised GT")
    if not bool(valid.any()):
        raise ValueError(f"No valid supervised rows: {context.item_id}")
    with torch.no_grad():
        target_state = v2.legacy_to_state(labels["target_legacy"], system.geometry)
    target_local = v2.to_local(target_state, output["anchor_rotation"], output["anchor_position"])
    current_nll = -v2.log_prob(output["current"], target_local).mean()
    predicted = v2._world_joints(system, output["state"], inputs["projector_args"][0])
    if not torch.isfinite(predicted[valid]).all():
        raise FloatingPointError("Nonfinite supervised predicted joints")
    camera_m = torch.linalg.vector_norm(predicted - camera, dim=-1)[valid].mean()
    aligned_m, aligned_points = _aligned_world_loss(predicted, canonical, labels, valid)
    first = int(torch.nonzero(valid, as_tuple=False)[0, 0])
    pred_relative = predicted - predicted[first:first+1, :1]
    gt_relative = canonical - canonical[first:first+1, :1]
    relative_m = torch.linalg.vector_norm(pred_relative - gt_relative, dim=-1)[valid].mean()
    velocity = v2._joint_velocity_loss(predicted, canonical, labels, valid)
    pred_distance = torch.linalg.vector_norm(predicted[:, 4] - predicted[:, 8], dim=-1) / torch.linalg.vector_norm(predicted[:, 0] - predicted[:, 9], dim=-1).clamp_min(1e-6)
    gt_distance = torch.linalg.vector_norm(camera[:, 4] - camera[:, 8], dim=-1) / torch.linalg.vector_norm(camera[:, 0] - camera[:, 9], dim=-1).clamp_min(1e-6)
    pinch = F.smooth_l1_loss(pred_distance[valid], gt_distance[valid])
    temperature = LOSS_CONFIG["pinch_sigmoid_temperature"]
    contact = F.binary_cross_entropy(torch.sigmoid((.275 - pred_distance[valid]) / temperature).clamp(1e-6, 1-1e-6),
                                     torch.sigmoid((.2 - gt_distance[valid]) / temperature))
    loss = (LOSS_CONFIG["aligned_world_joint_weight"] * aligned_m / .05
            + LOSS_CONFIG["camera_joint_weight"] * camera_m / .05
            + LOSS_CONFIG["camera_relative_nll_weight"] * current_nll
            + .5 * relative_m / .05 + .05 * velocity + .1 * pinch + .1 * contact)
    metrics = {"loss": loss, "aligned_world_joint_m": aligned_m.detach(),
               "aligned_world_joint_points": torch.tensor(aligned_points),
               "camera_joint_m": camera_m.detach(), "camera_relative_current_nll": current_nll.detach(),
               "world_relative_joint_m": relative_m.detach(), "world_joint_velocity_smooth_l1_mps": velocity.detach(),
               "pinch_distance_smooth_l1": pinch.detach(), "pinch_contact_bce": contact.detach(),
               "valid_frames": valid.sum().detach()}
    return metrics, loss


def _fixed_event_mask_hash(manifest: dict[str, Any]) -> str:
    rows = []
    for sequence, record in sorted(manifest["event_ground_truth_sidecars"].items()):
        rows.append([sequence, record["sha256"], record["evaluation_mask_true_count"]])
    return v2._json_sha256(rows)


def _event_results(predictions: dict[tuple[str, int], np.ndarray],
                   manifest: dict[str, Any]) -> dict[str, Any]:
    assert EVENT_MANIFEST is not None
    coverage: dict[str, set[int]] = {sequence: set() for sequence in EVENT_MANIFEST["sequence_names"]}
    for window in manifest["event_windows"]:
        coverage[str(window["sequence"])].update(int(frame) for frame in window["frame_indices"])
    declared_events: dict[str, dict[tuple[int, int], dict[str, Any]]] = {sequence: {} for sequence in coverage}
    for event in EVENT_MANIFEST["events"]:
        declared_events[event["sequence"]][(int(event["start_frame"]), int(event["stop_frame"]))] = event
    results = []
    all_errors = []
    total = {"tp": 0, "fp": 0, "fn": 0, "reference_events": 0,
             "predicted_events": 0, "covered_frames": 0, "eligible_covered_frames": 0}
    for sequence in sorted(coverage):
        record = manifest["event_ground_truth_sidecars"][sequence]
        path = Path(record["path"])
        if not path.is_absolute():
            path = HERE.parents[2] / path
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{sequence}: event GT sidecar SHA256 changed")
        with np.load(path, allow_pickle=False) as archive:
            times = archive["timestamp_sec"].astype(np.float64)
            gt_valid = archive["formal_reference_valid"].astype(bool)
            eligible = archive["formal_prediction_eligible"].astype(bool)
            fixed = archive["evaluation_mask"].astype(bool)
            gt_distance = archive["normalized_pinch_distance"].astype(np.float64)
            frame_ids = archive["frame_ids"].astype(np.int64)
        count = len(times)
        if not np.array_equal(frame_ids, np.arange(count)) or not np.array_equal(eligible, fixed):
            raise ValueError(f"{sequence}: formal eligibility or frame index changed")
        source_valid = np.zeros(count, dtype=bool)
        with np.load(path, allow_pickle=False) as archive:
            source_valid = archive["gt_valid"].astype(bool) & archive["visual_valid"].astype(bool)
        reconstructed = np.zeros(count, dtype=bool)
        starts = np.flatnonzero(source_valid & ~np.r_[False, source_valid[:-1]])
        stops = np.flatnonzero(source_valid & ~np.r_[source_valid[1:], False]) + 1
        for start, stop in zip(starts, stops):
            if stop - start >= 16:
                reconstructed[start:stop] = True
        if not np.array_equal(eligible, reconstructed):
            raise ValueError(f"{sequence}: formal eligibility differs from global 16-frame parse_chunks")
        expected_mask_hash = EVENT_MANIFEST["sequences"][sequence]["evaluation_mask_sha256"]
        digest = hashlib.sha256()
        digest.update(str(reconstructed.dtype).encode("ascii"))
        digest.update(json.dumps(reconstructed.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(reconstructed).tobytes())
        if expected_mask_hash != digest.hexdigest():
            raise ValueError(f"{sequence}: event eligibility hash differs from fixed manifest")
        covered = np.zeros(count, dtype=bool)
        if any(frame < 0 or frame >= count for frame in coverage[sequence]):
            raise ValueError(f"{sequence}: event coverage out of video bounds")
        covered[list(coverage[sequence])] = True
        pred_valid = eligible & covered
        joints = np.full((count, 21, 3), np.nan, dtype=np.float32)
        for frame in coverage[sequence]:
            key = (sequence, frame)
            if key not in predictions:
                raise ValueError(f"{sequence}: uncovered cached event frame {frame}")
            joints[frame] = predictions[key]
        if not np.isfinite(joints[pred_valid]).all():
            bad = np.flatnonzero(pred_valid & ~np.isfinite(joints).all((1, 2)))[:8].tolist()
            raise FloatingPointError(f"{sequence}: nonfinite candidate on fixed event frames {bad}")
        if not np.isfinite(gt_distance[gt_valid]).all():
            raise FloatingPointError(f"{sequence}: nonfinite full-video reference distance")
        distance = np.full(count, np.nan, dtype=np.float64)
        distance[pred_valid] = normalized_pinch_distance(joints[pred_valid])
        references = extract_events(gt_distance, times, gt_valid, .2, min_frames=3)
        predicted = extract_events(distance, times, pred_valid, .275, min_frames=3)
        reference_ids = []
        for event in references:
            key = (event.start, event.stop)
            if key not in declared_events[sequence]:
                raise ValueError(f"{sequence}: full-video GT event {key} missing from fixed manifest")
            declared = declared_events[sequence][key]
            if (abs(event.onset_sec - float(declared["onset_sec"])) > 1e-7
                    or abs(event.offset_sec - float(declared["offset_sec"])) > 1e-7
                    or not declared["valid_bracketed"]):
                raise ValueError(f"{sequence}: fixed GT event {key} onset/bracket changed")
            reference_ids.append(declared["event_id"])
        if len(references) != len(declared_events[sequence]):
            raise ValueError(f"{sequence}: full-video reference event count drifted")
        metrics = match_events(references, predicted, onset_tolerance_sec=.2)
        matched = []
        for match in metrics["matches"]:
            reference = references[match["reference_index"]]
            prediction = predicted[match["prediction_index"]]
            matched.append({"event_id": reference_ids[match["reference_index"]],
                            "reference_start_frame": reference.start,
                            "predicted_start_frame": prediction.start,
                            "reference_onset_sec": reference.onset_sec,
                            "predicted_onset_sec": prediction.onset_sec,
                            "onset_error_ms": match["onset_error_ms"]})
            all_errors.append(abs(float(match["onset_error_ms"])))
        row = {"sequence": sequence, "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
               "reference_events": len(references), "predicted_events": len(predicted),
               "covered_frames": int(covered.sum()), "eligible_covered_frames": int(pred_valid.sum()),
               "formal_eligible_frames": int(eligible.sum()),
               "matched_gt_event_ids": [item["event_id"] for item in matched],
               "matches": matched,
               "unmatched_gt_event_ids": [reference_ids[i] for i in metrics["unmatched_reference_indices"]],
               "onset_mae_ms": metrics["onset_mae_ms"]}
        results.append(row)
        for key in total:
            total[key] += row[key]
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    total.update({"precision": tp / (tp + fp) if tp + fp else None,
                  "recall": tp / (tp + fn) if tp + fn else None,
                  "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                  "onset_mae_ms": float(np.mean(all_errors)) if all_errors else None,
                  "sequences": results,
                  "matched_gt_event_ids": [event_id for row in results for event_id in row["matched_gt_event_ids"]]})
    return total


def evaluate_calibration(system: torch.nn.Module, calibration_contexts: list[Any],
                         manifest: dict[str, Any], device: torch.device,
                         *, compare_zero_baseline: bool = False) -> dict[str, Any]:
    if WORLD is None:
        raise RuntimeError("World map unavailable")
    predictions, difference = _predictions_by_frame(system, calibration_contexts, device,
                                                       compare_zero_baseline=compare_zero_baseline)
    context_lookup = {context.item_id: context for context in calibration_contexts}
    windows = []
    total_error = 0.0
    total_points = 0
    for window in manifest["calibration_windows"]:
        sequence = str(window["sequence"])
        labels: dict[int, tuple[Any, int]] = {}
        for item_id in window["context_item_ids"]:
            context = context_lookup[item_id]
            for row, frame in enumerate(context.frame_indices):
                labels[int(frame)] = (context, row)
        frames = np.asarray(window["frame_indices"], dtype=np.int64)
        target_cache: dict[str, np.ndarray] = {}
        predicted_rows, target_rows, valid, timestamps, segments = [], [], [], [], []
        for frame in frames:
            context, row = labels[int(frame)]
            if context.item_id not in target_cache:
                target_cache[context.item_id] = WORLD.target(context).numpy()
            predicted_rows.append(predictions[(sequence, int(frame))])
            target_rows.append(target_cache[context.item_id][row])
            valid.append(bool(context.labels["gt_valid"][row] & context.labels["visual_valid"][row]))
            timestamps.append(float(context.labels["timestamp_sec"][row]))
            segments.append(int(context.labels["segment_id"][row]))
        target = np.asarray(target_rows)
        score = v2._score_wmpjpe(np.asarray(predicted_rows), target,
                                  {"frame_indices": frames, "segment_id": np.asarray(segments),
                                   "timestamp_sec": np.asarray(timestamps), "gt_valid": np.asarray(valid),
                                   "visual_valid": np.asarray(valid)})
        frames_scored = int(score["scored_frames"])
        if score["w_mpjpe_mm"] is not None:
            total_error += score["w_mpjpe_mm"] * frames_scored
            total_points += frames_scored
        windows.append({"sequence": sequence, "window_index": window["window_index"],
                        "start_frame": window["start_frame"], "w_mpjpe_mm": score["w_mpjpe_mm"],
                        "scored_frames": frames_scored, "chunks": score["chunks"]})
    result = {"w_mpjpe_mm": total_error / total_points if total_points else None,
              "w_mpjpe_scored_frames": total_points,
              "window_count": len(windows), "windows": windows,
              "events": _event_results(predictions, manifest),
              "calibration_rule": CALIBRATION_RULE, **BOUND_HASHES}
    if result["w_mpjpe_mm"] is None or not np.isfinite(result["w_mpjpe_mm"]):
        raise FloatingPointError("No finite fixed-window W score")
    if compare_zero_baseline:
        result["zero_v2_equals_single_twm"] = {"max_abs_world_state_difference": difference,
                                                 "atol": 1e-5, "rtol": 1e-5,
                                                 "checked_valid_rows_only": True}
    return result


def _calibration_protocol() -> dict[str, Any]:
    return {"revision": PROTOCOL_REVISION, "calibration_rule": CALIBRATION_RULE,
            "selection_rule": SELECTION_RULE, "loss_config": LOSS_CONFIG,
            "w_window_count": 40, "event_reference_count": 224,
            "w_min_frames": 16, "w_block_frames": 100,
            "event_pred_threshold": .275, "event_gt_threshold": .2,
            "event_min_frames": 3, "onset_tolerance_sec": .2}


def _bound_sources(manifest: dict[str, Any], contexts: list[Any],
                   cache_manifest_sha256: str) -> dict[str, str]:
    assert WORLD is not None and EVENT_HASH is not None
    return {"cache_manifest_sha256": cache_manifest_sha256,
            "world_reference_manifest_sha256": WORLD.manifest_sha256,
            "event_manifest_sha256": EVENT_HASH,
            "event_gt_sidecars_sha256": v2._json_sha256(manifest["event_ground_truth_sidecars"]),
            "calibration_windows_sha256": v2._json_sha256(manifest["calibration_windows"]),
            "calibration_mask_sha256": v2._calibration_mask_sha256(contexts, manifest["calibration_windows"]),
            "event_mask_sha256": _fixed_event_mask_hash(manifest),
            "calibration_protocol_sha256": v2._json_sha256(_calibration_protocol())}


def _write_baseline(system: torch.nn.Module, calibration_contexts: list[Any],
                    manifest: dict[str, Any], device: torch.device, run_dir: Path,
                    cache_manifest_sha256: str) -> dict[str, Any]:
    visual = system.observer.visual
    if not all(torch.count_nonzero(parameter.detach()).item() == 0
               for name in ("root_delta_head", "finger_delta_head")
               for parameter in getattr(visual, name).parameters()):
        raise ValueError("v3 baseline must start from fresh zero heads")
    system.eval()
    result = evaluate_calibration(system, calibration_contexts, manifest, device,
                                  compare_zero_baseline=True)
    bound = _bound_sources(manifest, calibration_contexts, cache_manifest_sha256)
    BOUND_HASHES.update(bound)
    result.update(bound)
    result.update({"architecture": v2.INTRA_ARCHITECTURE,
                   "experiment": "feedback_v3", "protocol_revision": PROTOCOL_REVISION,
                   "calibration_protocol": _calibration_protocol(),
                   "base_checkpoints": manifest["base_checkpoints"],
                   "frozen_parameter_hashes": manifest["frozen_parameter_hashes"],
                   "selection_rule": SELECTION_RULE,
                   "method": "fresh zero-head bounded feedback v3, verified direct TWM equivalent"})
    if result["events"]["reference_events"] != 224:
        raise ValueError("Baseline did not include all 224 fixed GT reference events")
    v2._atomic_json(result, run_dir / "baseline_calibration.json")
    return result


def _load_baseline(path: Path, *, cache_manifest_sha256: str,
                   calibration_windows_sha256: str, calibration_mask_sha256: str,
                   calibration_protocol_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline.get("protocol_revision") != PROTOCOL_REVISION or baseline.get("architecture") != v2.INTRA_ARCHITECTURE:
        raise ValueError("Stored baseline is not feedback v3")
    if baseline.get("calibration_protocol") != _calibration_protocol():
        raise ValueError("Stored v3 calibration protocol changed")
    if baseline.get("zero_v2_equals_single_twm", {}).get("max_abs_world_state_difference", float("inf")) > 1e-5:
        raise ValueError("Stored baseline failed zero-head equivalence")
    expected = {"cache_manifest_sha256": cache_manifest_sha256,
                "calibration_windows_sha256": calibration_windows_sha256,
                "calibration_mask_sha256": calibration_mask_sha256,
                "calibration_protocol_sha256": v2._json_sha256(_calibration_protocol()),
                **BOUND_HASHES}
    if any(baseline.get(key) != value for key, value in expected.items()):
        raise ValueError("Stored v3 baseline source/protocol binding changed")
    if baseline["events"]["reference_events"] != 224:
        raise ValueError("Stored baseline lacks fixed 224 GT events")
    return baseline


def _score_is_qualified(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    b, c = baseline["events"], candidate["events"]
    values = (baseline["w_mpjpe_mm"], candidate["w_mpjpe_mm"],
              b["f1"], c["f1"], b["recall"], c["recall"],
              b["onset_mae_ms"], c["onset_mae_ms"])
    if b["tp"] < 10 or any(value is None or not np.isfinite(value) for value in values):
        return False
    return (candidate["w_mpjpe_mm"] < baseline["w_mpjpe_mm"]
            and c["f1"] >= b["f1"] and c["recall"] >= b["recall"]
            and c["onset_mae_ms"] <= b["onset_mae_ms"])


def _checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = v2._ORIGINAL_CHECKPOINT_PAYLOAD(*args, **kwargs)
    payload["experiment"] = "feedback_v3"
    payload["protocol_revision"] = PROTOCOL_REVISION
    payload["trainer"].update({"protocol_revision": PROTOCOL_REVISION,
                               "experiment": "feedback_v3", **BOUND_HASHES})
    return payload


def _restore_resume(path: Path, system: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    config: dict[str, Any], *, base_twm_hash: str, base_hawor_hash: str,
                    cache_manifest_sha256: str) -> tuple[int, np.random.Generator, dict[str, Any], dict[str, Any] | None]:
    checkpoint = v2.twm_only.load_checkpoint_restricted(path)
    trainer = checkpoint.get("trainer", {})
    if checkpoint.get("protocol_revision") != PROTOCOL_REVISION or trainer.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("Resume checkpoint is not feedback v3")
    if any(trainer.get(key) != value for key, value in BOUND_HASHES.items()):
        raise ValueError("Resume checkpoint source/protocol binding changed")
    return v2._ORIGINAL_RESTORE_RESUME(path, system, optimizer, config,
                                       base_twm_hash=base_twm_hash,
                                       base_hawor_hash=base_hawor_hash,
                                       cache_manifest_sha256=cache_manifest_sha256)


def main() -> None:
    # Reuse the audited v2 system construction, memory policy and step loop.
    # Only process-local v3 protocol hooks change; no v2 file is edited.
    v2._ORIGINAL_VALIDATE_CACHE = v2._validate_cache_protocol
    v2._ORIGINAL_CHECKPOINT_PAYLOAD = v2._checkpoint_payload
    v2._ORIGINAL_RESTORE_RESUME = v2._restore_resume
    v2._parse_args = _parse_args
    v2._validate_cache_protocol = _validate_cache_protocol
    v2._predictions_by_frame = _predictions_by_frame
    v2.evaluate_calibration = evaluate_calibration
    v2._write_baseline = _write_baseline
    v2._load_baseline = _load_baseline
    v2._score_is_qualified = _score_is_qualified
    v2._context_loss = _context_loss
    v2._checkpoint_payload = _checkpoint_payload
    v2._restore_resume = _restore_resume
    v2._calibration_protocol = _calibration_protocol
    v2.CALIBRATION_RULE = CALIBRATION_RULE
    v2.SELECTION_RULE = SELECTION_RULE
    v2.LOSS_CONFIG = LOSS_CONFIG
    v2.main()
    if RUN_DIR is not None:
        selection_path = RUN_DIR / "selection.json"
        baseline_path = RUN_DIR / "baseline_calibration.json"
        if selection_path.is_file() and baseline_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            selection["baseline_tp"] = baseline["events"]["tp"]
            selection["baseline_tp_at_least_10"] = baseline["events"]["tp"] >= 10
            selection["protocol_revision"] = PROTOCOL_REVISION
            v2._atomic_json(selection, selection_path)


if __name__ == "__main__":
    main()
