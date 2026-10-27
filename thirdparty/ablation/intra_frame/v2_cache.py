#!/usr/bin/env python3
"""Frozen HaWoR feature and label cache for intra-frame feedback v2.

Each cached item is one inference-aligned HaWoR context. The model inputs and
supervision are written to different files so GT never enters the observer
forward path. Patch features are stored as fp16 and restored as fp32 by the
trainer.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRY = ROOT / "thirdparty" / "try"
sys.path[:0] = [str(ROOT), str(TRY), str(TRY / "scripts"), str(HERE)]

from dual_loop.io import _resolve_tracks_path  # noqa: E402
from prepare_training import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_EXPORT_SPLIT,
    DEFAULT_PROTOCOL,
    DEFAULT_SPLIT,
    DEFAULT_VIDEO_ROOT,
    PreparedSequence,
    context_indices,
    context_start_indices,
    convert_raw_history_states,
    load_sequences,
)
from dual_loop.handloop_runtime import build_system  # noqa: E402
from feedback_v2 import install_feedback_v2  # noqa: E402
import train_visual  # noqa: E402


CACHE_SCHEMA = "handloop_intra_v2_feature_cache"
CACHE_VERSION = 1
TRAIN_CONTEXTS_PER_VIDEO = 16
CALIBRATION_WINDOWS_PER_VIDEO = 2
CALIBRATION_WINDOW_FRAMES = 128
CACHE_SEED = 7


@dataclass
class CachedContext:
    item_id: str
    split: str
    sequence: str
    context_start: int
    frame_indices: np.ndarray
    inputs: dict[str, Any]
    labels: dict[str, torch.Tensor]

    @property
    def size(self) -> int:
        return len(self.frame_indices)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _stratified_uniform_indices(count: int, choose: int, rng: np.random.Generator) -> np.ndarray:
    """Choose unique items across the full ordered list using seeded strata."""
    if count < 1 or choose < 1:
        return np.empty(0, dtype=np.int64)
    choose = min(count, choose)
    if count == choose:
        return np.arange(count, dtype=np.int64)
    edges = np.linspace(0, count, choose + 1)
    selected = []
    for slot in range(choose):
        low = int(np.floor(edges[slot]))
        high = max(low + 1, int(np.floor(edges[slot + 1])))
        high = min(high, count)
        selected.append(int(rng.integers(low, high)))
    return np.asarray(selected, dtype=np.int64)


def choose_training_contexts(
    sequence: PreparedSequence,
    *,
    count: int = TRAIN_CONTEXTS_PER_VIDEO,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Return seeded, spread-out full 16-frame contexts with usable labels."""
    starts = context_start_indices(sequence)
    candidates: list[dict[str, Any]] = []
    for start in np.unique(starts):
        start = int(start)
        stop = min(start + 16, len(sequence))
        frame_ids = np.arange(start, stop, dtype=np.int64)
        # Training uses only complete HaWoR contexts; padded boundary contexts
        # are kept out of the supervised sampler.
        if len(frame_ids) != 16 or not np.all(sequence.segment_id[frame_ids] == sequence.segment_id[start]):
            continue
        usable = frame_ids[sequence.gt_valid[frame_ids] & sequence.visual_valid[frame_ids]]
        if len(usable):
            candidates.append({
                "context_start": start,
                "context_frame_indices": frame_ids,
                "frame_indices": usable,
            })
    selected = _stratified_uniform_indices(len(candidates), count, rng)
    return [candidates[int(index)] for index in selected]


def _continuous_runs(sequence: PreparedSequence) -> list[tuple[int, int]]:
    """Return half-open runs with adjacent IDs, one segment, and dt <= 250ms."""
    if not len(sequence):
        return []
    starts = [0]
    for index in range(1, len(sequence)):
        gap = float(sequence.timestamp_sec[index] - sequence.timestamp_sec[index - 1])
        if (sequence.segment_id[index] != sequence.segment_id[index - 1]
                or gap <= 0.0 or gap > 0.25):
            starts.append(index)
    starts.append(len(sequence))
    return list(zip(starts[:-1], starts[1:]))


def choose_calibration_windows(
    sequence: PreparedSequence,
    *,
    count: int = CALIBRATION_WINDOWS_PER_VIDEO,
    length: int = CALIBRATION_WINDOW_FRAMES,
) -> list[dict[str, Any]]:
    """Choose fixed, roughly uniform, continuous windows without crossing gaps."""
    candidate_starts: list[int] = []
    run_for_start: dict[int, tuple[int, int]] = {}
    for run_start, run_stop in _continuous_runs(sequence):
        last_start = run_stop - length
        if last_start < run_start:
            continue
        for start in range(run_start, last_start + 1):
            candidate_starts.append(start)
            run_for_start[start] = (run_start, run_stop)
    candidate_starts.sort()
    if not candidate_starts:
        return []

    # Targets sit at one-third and two-thirds of the available candidate list.
    # Pick nearest valid starts while keeping the two windows disjoint whenever
    # the sequence has enough continuous footage.
    target_positions = np.linspace(0.25, 0.75, min(count, len(candidate_starts)))
    chosen: list[int] = []
    for fraction in target_positions:
        target_index = int(round(float(fraction) * (len(candidate_starts) - 1)))
        target_start = candidate_starts[target_index]
        eligible = [start for start in candidate_starts if start not in chosen]
        disjoint = [
            start for start in eligible
            if all(start + length <= prior or prior + length <= start
                   or run_for_start[start] != run_for_start[prior]
                   for prior in chosen)
        ]
        pool = disjoint or eligible
        if not pool:
            continue
        selected_start = min(pool, key=lambda value: (abs(value - target_start), value))
        chosen.append(selected_start)

    chosen.sort()
    windows = []
    for window_index, start in enumerate(chosen):
        run_start, run_stop = run_for_start[start]
        frame_ids = np.arange(start, start + length, dtype=np.int64)
        if frame_ids[-1] >= run_stop:
            raise RuntimeError("Calibration window selection crossed a discontinuity")
        windows.append({
            "window_index": window_index,
            "start_frame": int(start),
            "stop_frame_exclusive": int(start + length),
            "frame_indices": frame_ids,
            "segment_id": int(sequence.segment_id[start]),
            "run_start": int(run_start),
            "run_stop_exclusive": int(run_stop),
        })
    return windows


def _item_id(split: str, sequence: str, context_start: int) -> str:
    return f"{split}/{sequence}/ctx_{context_start:06d}"


def _feature_tensor(value: torch.Tensor, *, key: str) -> torch.Tensor:
    result = value.detach().cpu().contiguous()
    if key == "patches":
        return result.to(torch.float16)
    if result.dtype.is_floating_point and result.dtype != torch.float64:
        return result.to(torch.float32)
    return result


def _inputs_for_row(model_inputs: dict[str, Any], row: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in model_inputs.items():
        if key == "projector_args":
            output[key] = tuple(_feature_tensor(item[row], key=f"projector_{index}")
                                for index, item in enumerate(value))
        else:
            output[key] = _feature_tensor(value[row], key=key)
    return output


def _labels_for_frames(
    sequence: PreparedSequence,
    frame_indices: np.ndarray,
    geometry: torch.nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    ids = np.asarray(frame_indices, dtype=np.int64)
    valid = sequence.gt_valid[ids]
    target = torch.as_tensor(sequence.target_legacy[ids], dtype=torch.float32, device=device)
    target_joints = torch.full((len(ids), 21, 3), float("nan"), dtype=torch.float32, device=device)
    with torch.no_grad():
        valid_ids = torch.as_tensor(np.flatnonzero(valid), dtype=torch.long, device=device)
        if len(valid_ids):
            target_joints[valid_ids] = geometry.joints(target[valid_ids])
    return {
        "target_legacy": target.cpu(),
        "target_joints": target_joints.cpu(),
        "gt_valid": torch.as_tensor(sequence.gt_valid[ids], dtype=torch.bool),
        "visual_valid": torch.as_tensor(sequence.visual_valid[ids], dtype=torch.bool),
        "timestamp_sec": torch.as_tensor(sequence.timestamp_sec[ids].copy(), dtype=torch.float64),
        "segment_id": torch.as_tensor(sequence.segment_id[ids].copy(), dtype=torch.long),
        "frame_indices": torch.as_tensor(ids.copy(), dtype=torch.long),
    }


def _source_manifest(
    groups: dict[str, list[PreparedSequence]],
    *,
    base_hashes: dict[str, str],
    split_manifest: Path,
    export_split: Path,
    protocol_path: Path,
    video_root: Path,
    selected_contexts: dict[str, list[dict[str, Any]]],
    calibration_windows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_files = {
        "split_manifest": Path(split_manifest),
        "export_split": Path(export_split),
        "protocol": Path(protocol_path),
    }
    files = {key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
             for key, path in source_files.items()}
    sequences: dict[str, Any] = {}
    image_paths: set[str] = set()
    all_sequences = groups["train"] + groups["calibration"]
    for sequence in all_sequences:
        # The prepared archive and detector/focal metadata fully identify the
        # non-RGB observation sources used to build each item.
        sequence_root = Path(video_root) / sequence.sequence
        tracks_path, oracle_assisted = _resolve_tracks_path(sequence_root, allow_gt_boxes=False)
        if oracle_assisted:
            raise ValueError(f"Refusing oracle-assisted track source for {sequence.sequence}")
        focal_path = sequence_root / "focal.txt"
        sequences[sequence.sequence] = {
            "archive": {"path": str(sequence.archive_path.resolve()),
                        "sha256": sha256_file(sequence.archive_path)},
            "detector_tracks": {"path": str(tracks_path.resolve()),
                                "sha256": sha256_file(tracks_path)},
            "focal": {"path": str(focal_path.resolve()), "sha256": sha256_file(focal_path)},
        }
        requested_contexts = selected_contexts.get(sequence.sequence, [])
        requested_contexts = list(requested_contexts)
        for window in calibration_windows.get(sequence.sequence, []):
            requested_contexts.extend(np.asarray(window, dtype=np.int64).tolist())
        starts = context_start_indices(sequence)
        for item in requested_contexts:
            if isinstance(item, dict):
                frame_ids = item["context_frame_indices"]
            else:
                context_start = int(starts[int(item)])
                frame_ids, _ = context_indices(sequence, context_start)
            image_paths.update(str(path) for path in sequence.image_paths[frame_ids])

    # Hash selected RGB source bytes, not merely their paths. This binds cached
    # IAM features to the actual images used by HaWoR.
    images = {}
    for value in sorted(image_paths):
        path = Path(value)
        images[value] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "base_checkpoints": base_hashes,
        "protocol_and_splits": files,
        "sequences": sequences,
        "rgb_images": images,
    }


def _requested_contexts(
    groups: dict[str, list[PreparedSequence]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rng = np.random.default_rng(CACHE_SEED)
    train_selection: dict[str, list[dict[str, Any]]] = {}
    calibration_selection: dict[str, list[dict[str, Any]]] = {}
    windows_manifest: list[dict[str, Any]] = []
    for sequence in groups["train"]:
        selected = choose_training_contexts(sequence, rng=rng)
        if not selected:
            raise ValueError(f"{sequence.sequence}: no full train context with GT and visual valid frames")
        train_selection[sequence.sequence] = selected
    for sequence in groups["calibration"]:
        windows = choose_calibration_windows(sequence)
        if not windows:
            raise ValueError(f"{sequence.sequence}: no continuous 128-frame calibration window")
        calibration_selection[sequence.sequence] = windows
        for window in windows:
            windows_manifest.append({
                "sequence": sequence.sequence,
                "window_index": int(window["window_index"]),
                "start_frame": int(window["start_frame"]),
                "stop_frame_exclusive": int(window["stop_frame_exclusive"]),
                "segment_id": int(window["segment_id"]),
                "run_start": int(window["run_start"]),
                "run_stop_exclusive": int(window["run_stop_exclusive"]),
                "frame_indices": window["frame_indices"].tolist(),
            })
    return train_selection, calibration_selection, {"calibration_windows": windows_manifest}


def _context_requests(
    groups: dict[str, list[PreparedSequence]],
    train_selection: dict[str, list[dict[str, Any]]],
    calibration_selection: dict[str, list[dict[str, Any]]],
) -> tuple[dict[tuple[str, str, int], set[int]], dict[str, PreparedSequence]]:
    sequence_by_name = {sequence.sequence: sequence
                        for split in ("train", "calibration") for sequence in groups[split]}
    requests: dict[tuple[str, str, int], set[int]] = {}
    for sequence in groups["train"]:
        for selected in train_selection[sequence.sequence]:
            key = ("train", sequence.sequence, int(selected["context_start"]))
            requests.setdefault(key, set()).update(int(value) for value in selected["frame_indices"])
    for sequence in groups["calibration"]:
        starts = context_start_indices(sequence)
        for window in calibration_selection[sequence.sequence]:
            for frame in window["frame_indices"]:
                start = int(starts[int(frame)])
                key = ("calibration", sequence.sequence, start)
                requests.setdefault(key, set()).add(int(frame))
    return requests, sequence_by_name


def build_cache(
    *,
    system: torch.nn.Module,
    groups: dict[str, list[PreparedSequence]],
    device: torch.device,
    cache_dir: Path,
    base_hashes: dict[str, str],
    frozen_parameter_hashes: dict[str, str],
    split_manifest: Path,
    export_split: Path,
    protocol_path: Path,
    video_root: Path,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Extract each selected HaWoR context exactly once and save X/Y apart."""
    cache_dir = Path(cache_dir).resolve()
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not rebuild:
        manifest = load_manifest(cache_dir)
        if manifest.get("base_checkpoints") != base_hashes:
            raise ValueError("Existing cache was built from different frozen base checkpoints")
        return manifest
    if manifest_path.exists() and rebuild:
        # Only files listed by the old cache are removed; unrelated user files
        # under the cache root are left intact.
        old = load_manifest(cache_dir, verify_files=False)
        for item in old.get("items", []):
            for key in ("inputs_path", "labels_path"):
                (cache_dir / item[key]).unlink(missing_ok=True)
        manifest_path.unlink()

    train_selection, calibration_selection, windows_record = _requested_contexts(groups)
    requests, sequence_by_name = _context_requests(groups, train_selection, calibration_selection)
    selected_contexts: dict[str, list[dict[str, Any]]] = {
        name: values for name, values in train_selection.items()
    }
    source_manifest = _source_manifest(
        groups, base_hashes=base_hashes, split_manifest=split_manifest,
        export_split=export_split, protocol_path=protocol_path, video_root=video_root,
        selected_contexts=selected_contexts,
        calibration_windows={
            name: [window["frame_indices"] for window in windows]
            for name, windows in calibration_selection.items()
        },
    )

    item_records: list[dict[str, Any]] = []
    for (split, sequence_name, context_start), requested_ids in sorted(requests.items()):
        sequence = sequence_by_name[sequence_name]
        frame_ids = np.asarray(sorted(requested_ids), dtype=np.int64)
        samples = [(sequence, int(frame)) for frame in frame_ids]
        # Exactly one build_live_batch invocation per source context. It runs
        # actual frozen HaWoR over the aligned 16 RGB crops and never reads GT.
        model_inputs, _beta, records = train_visual.build_live_batch(system, samples, device)
        input_rows = [_inputs_for_row(model_inputs, row) for row in range(len(frame_ids))]
        batched_inputs = {
            key: torch.stack([row[key] for row in input_rows], dim=0)
            if key != "projector_args"
            else tuple(torch.stack([row[key][part] for row in input_rows], dim=0)
                       for part in range(4))
            for key in input_rows[0]
        }
        labels = _labels_for_frames(sequence, frame_ids, system.geometry, device)
        if split == "train" and not bool((labels["gt_valid"] & labels["visual_valid"]).all()):
            raise RuntimeError("Training context cache contains an invalid supervised frame")

        item_id = _item_id(split, sequence_name, context_start)
        relative = Path(split) / sequence_name
        input_path = cache_dir / relative / f"ctx_{context_start:06d}.inputs.pt"
        labels_path = cache_dir / relative / f"ctx_{context_start:06d}.labels.pt"
        _atomic_torch_save(batched_inputs, input_path)
        _atomic_torch_save(labels, labels_path)
        item_records.append({
            "item_id": item_id,
            "split": split,
            "sequence": sequence_name,
            "context_start": int(context_start),
            "frame_indices": frame_ids.tolist(),
            "inputs_path": str(input_path.relative_to(cache_dir)),
            "labels_path": str(labels_path.relative_to(cache_dir)),
            "inputs_sha256": sha256_file(input_path),
            "labels_sha256": sha256_file(labels_path),
            "patch_dtype_on_disk": "float16",
            "input_keys": sorted(batched_inputs),
            "label_keys": sorted(labels),
            "record_timestamps": [float(record["timestamp_sec"]) for record in records],
            "record_segment_ids": [int(record["segment_id"]) for record in records],
        })

    item_by_key = {(item["split"], item["sequence"], item["context_start"]): item
                   for item in item_records}
    calibration_windows = []
    for sequence_name, windows in calibration_selection.items():
        starts = context_start_indices(sequence_by_name[sequence_name])
        for window in windows:
            context_keys = []
            for frame in window["frame_indices"]:
                key = ("calibration", sequence_name, int(starts[int(frame)]))
                record = item_by_key[key]
                context_keys.append(record["item_id"])
            calibration_windows.append({
                "sequence": sequence_name,
                "window_index": int(window["window_index"]),
                "start_frame": int(window["start_frame"]),
                "stop_frame_exclusive": int(window["stop_frame_exclusive"]),
                "segment_id": int(window["segment_id"]),
                "frame_indices": window["frame_indices"].tolist(),
                "context_item_ids": sorted(set(context_keys)),
            })

    manifest = {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "sampling_seed": CACHE_SEED,
        "train_contexts_per_video_requested": TRAIN_CONTEXTS_PER_VIDEO,
        "calibration_windows_per_video_requested": CALIBRATION_WINDOWS_PER_VIDEO,
        "calibration_window_frames": CALIBRATION_WINDOW_FRAMES,
        "feature_policy": "frozen actual HaWoR IAM and model_inputs; patches fp16 on disk, fp32 in training",
        "label_policy": "GT labels are stored in separate .labels.pt files and never enter model_inputs",
        "train_context_selection": {
            name: [{"context_start": int(item["context_start"]),
                    "context_frame_indices": item["context_frame_indices"].tolist(),
                    "supervised_frame_indices": item["frame_indices"].tolist()}
                   for item in values]
            for name, values in train_selection.items()
        },
        "calibration_windows": calibration_windows,
        "calibration_window_candidates": windows_record["calibration_windows"],
        "source_hashes": source_manifest,
        "base_checkpoints": base_hashes,
        "frozen_parameter_hashes": frozen_parameter_hashes,
        "items": item_records,
        "counts": {
            "train_sequences": len(groups["train"]),
            "calibration_sequences": len(groups["calibration"]),
            "train_contexts": sum(item["split"] == "train" for item in item_records),
            "train_supervised_frames": sum(
                len(item["frame_indices"]) for item in item_records if item["split"] == "train"),
            "calibration_windows": len(calibration_windows),
            "calibration_window_frames": sum(len(window["frame_indices"])
                                              for window in calibration_windows),
            "unique_calibration_frames": len({(window["sequence"], frame)
                                               for window in calibration_windows
                                               for frame in window["frame_indices"]}),
        },
    }
    _atomic_json(manifest, manifest_path)
    return manifest


def load_manifest(cache_dir: Path, *, verify_files: bool = True) -> dict[str, Any]:
    manifest_path = Path(cache_dir) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read v2 feature cache manifest {manifest_path}: {error}") from error
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("version") != CACHE_VERSION:
        raise ValueError(f"Unsupported v2 cache schema in {manifest_path}")
    if manifest.get("sampling_seed") != CACHE_SEED:
        raise ValueError("Cache sampling seed differs from the fixed v2 seed 7")
    if verify_files:
        for item in manifest.get("items", []):
            for path_key, hash_key in (("inputs_path", "inputs_sha256"),
                                       ("labels_path", "labels_sha256")):
                path = Path(cache_dir) / item[path_key]
                if not path.is_file() or sha256_file(path) != item.get(hash_key):
                    raise ValueError(f"Cache file is missing or changed: {path}")
    return manifest


def load_cached_contexts(cache_dir: Path, split: str) -> list[CachedContext]:
    """Load verified cached tensors; patch maps are promoted back to fp32."""
    if split not in {"train", "calibration"}:
        raise ValueError(f"Unsupported cache split {split!r}")
    cache_dir = Path(cache_dir).resolve()
    manifest = load_manifest(cache_dir)
    contexts = []
    for item in manifest.get("items", []):
        if item.get("split") != split:
            continue
        inputs_path = cache_dir / item["inputs_path"]
        labels_path = cache_dir / item["labels_path"]
        inputs = torch.load(inputs_path, map_location="cpu", weights_only=True)
        labels = torch.load(labels_path, map_location="cpu", weights_only=True)
        if not isinstance(inputs, dict) or not isinstance(labels, dict):
            raise ValueError(f"Malformed cache item {item['item_id']}")
        if inputs.get("patches").dtype != torch.float16:
            raise ValueError(f"Cached IAM patches must be fp16 on disk: {item['item_id']}")
        inputs["patches"] = inputs["patches"].float()
        frame_ids = np.asarray(item.get("frame_indices", []), dtype=np.int64)
        if inputs["patches"].shape[0] != len(frame_ids):
            raise ValueError(f"Input frame count mismatch for {item['item_id']}")
        if labels["frame_indices"].tolist() != frame_ids.tolist():
            raise ValueError(f"Label frame IDs differ from cache manifest for {item['item_id']}")
        if split == "train" and not bool((labels["gt_valid"] & labels["visual_valid"]).all()):
            raise ValueError(f"Training cache group has invalid supervised rows: {item['item_id']}")
        contexts.append(CachedContext(
            item_id=str(item["item_id"]), split=split,
            sequence=str(item["sequence"]), context_start=int(item["context_start"]),
            frame_indices=frame_ids, inputs=inputs, labels=labels,
        ))
    if not contexts:
        raise ValueError(f"v2 feature cache has no {split} contexts: {cache_dir}")
    return contexts


def get_windows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return list(manifest.get("calibration_windows", []))


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twm-checkpoint", type=Path, default=train_visual.DEFAULT_TWM)
    parser.add_argument("--hawor-checkpoint", type=Path, default=train_visual.DEFAULT_HAWOR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--export-split", type=Path, default=DEFAULT_EXPORT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-dir", type=Path,
                        default=HERE / "runs" / "feedback_v2_train" / "cache")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run only frozen feature extraction; training has its own entrypoint."""
    args = _parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    for path in (args.twm_checkpoint, args.hawor_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    twm_hash = sha256_file(args.twm_checkpoint)
    hawor_hash = sha256_file(args.hawor_checkpoint)
    if (twm_hash != train_visual.EXPECTED_TWM_SHA256
            or hawor_hash != train_visual.EXPECTED_HAWOR_SHA256):
        raise ValueError("Frozen TWM/HaWoR hashes do not match the approved intra-frame protocol")

    train_visual.seed_everything(CACHE_SEED)
    config, twm_state = train_visual.load_base_twm(
        args.twm_checkpoint, train_visual.EXPECTED_TWM_SHA256)
    system = build_system(args.hawor_checkpoint, config, device)
    system.observer.twm.load_state_dict(twm_state, strict=True)
    observer = install_feedback_v2(system)
    train_visual.freeze_except_visual(system)
    system.eval()
    system.observer.twm.eval()
    system.hawor.eval()
    zero_heads = all(
        torch.count_nonzero(parameter.detach()).item() == 0
        for name in ("root_delta_head", "finger_delta_head")
        for parameter in getattr(observer.visual, name).parameters()
    )
    if not zero_heads:
        raise RuntimeError("Fresh v2 visual heads must be zero initialized for the frozen cache run")

    groups, protocol = load_sequences(
        data_root=args.data_root, video_root=args.video_root,
        split_manifest=args.split_manifest, export_split=args.export_split,
        protocol_path=args.protocol, check_images=not args.skip_image_check)
    all_sequences = groups["train"] + groups["calibration"]
    convert_raw_history_states(all_sequences, system.geometry, device)
    base_hashes = {"twm_sha256": twm_hash, "hawor_sha256": hawor_hash}
    frozen_hashes = {
        "hawor_parameters_sha256": train_visual.module_parameter_sha256(system.hawor),
        "twm_parameters_sha256": train_visual.module_parameter_sha256(system.observer.twm),
        "geometry_parameters_sha256": train_visual.module_parameter_sha256(system.geometry),
    }
    manifest = build_cache(
        system=system, groups=groups, device=device, cache_dir=args.cache_dir,
        base_hashes=base_hashes, frozen_parameter_hashes=frozen_hashes,
        split_manifest=args.split_manifest, export_split=args.export_split,
        protocol_path=args.protocol, video_root=args.video_root,
        rebuild=args.rebuild_cache,
    )
    summary = {
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_schema": manifest["schema"],
        "counts": manifest["counts"],
        "base_checkpoints": base_hashes,
        "frozen_parameter_hashes": frozen_hashes,
        "visual_head_initialization": "fresh HandLoopVisual with zero root/finger delta heads",
        "test_sequences_excluded": len(protocol.get("test_sequences", [])),
        "cache_only": True,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

