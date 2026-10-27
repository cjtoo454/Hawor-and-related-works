#!/usr/bin/env python3
"""Build detector-backed HaWoR observations and masked DROID-SLAM inputs.

This is deliberately separate from the repository's evaluation scripts: those
scripts parse arguments at import time and some paths read GT annotations.
Only ``extracted_images/``, ``focal.txt`` and ``tt_list.pkl`` are sequence
inputs. Detector tracks, HaWoR chunks, masks and SLAM outputs carry hashes for
the exact sources used to generate them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
TRACK_RELATIVE_PATH = Path("preprocess/preprocess_tracks.npy")
TRACK_PROVENANCE_NAME = "preprocess_tracks_provenance.json"
FRONTEND_STATE_NAME = "producer_state.json"
FRONTEND_MANIFEST_NAME = "frontend_provenance.json"
MASKS_RELATIVE_PATH = Path("preprocess/model_masks.npy")
MASKS_PROVENANCE_NAME = "model_masks_provenance.json"
PRODUCER_ID = "prepare_hawor_frontend.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_image_sources(sequence_root: Path, image_paths: Sequence[Path]) -> str:
    """Hash image names and bytes in the same lexical order used by the consumer."""
    digest = hashlib.sha256()
    for path in image_paths:
        relative = path.relative_to(sequence_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _save_npy_atomic(path: Path, value: Any, *, allow_pickle: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, value, allow_pickle=allow_pickle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _save_npz_atomic(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def image_paths_for(sequence_root: Path) -> list[Path]:
    paths = sorted((sequence_root / "extracted_images").glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(
            f"No extracted JPEG frames under {sequence_root / 'extracted_images'}")
    return paths


def load_sequence_metadata(sequence_root: Path, image_paths: Sequence[Path]) -> dict[str, Any]:
    focal_path = sequence_root / "focal.txt"
    timestamp_path = sequence_root / "tt_list.pkl"
    for path in (focal_path, timestamp_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required sequence input: {path}")

    try:
        focal = float(focal_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid focal length in {focal_path}") from exc
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError(f"Focal length in {focal_path} must be positive and finite")

    with timestamp_path.open("rb") as handle:
        timestamps = np.asarray(pickle.load(handle))
    if timestamps.ndim != 1 or len(timestamps) != len(image_paths):
        raise ValueError(
            f"Expected one timestamp per image in {timestamp_path}, got "
            f"{timestamps.shape} for {len(image_paths)} images")
    if not np.issubdtype(timestamps.dtype, np.integer):
        raise ValueError(f"Expected integer nanosecond timestamps in {timestamp_path}")
    if np.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError(f"Timestamps in {timestamp_path} must be strictly increasing")

    import cv2

    image = cv2.imread(str(image_paths[0]))
    if image is None:
        raise ValueError(f"Cannot read extracted image {image_paths[0]}")
    height, width = image.shape[:2]
    return {
        "frame_count": len(image_paths),
        "height": int(height),
        "width": int(width),
        "focal": focal,
        "image_manifest_sha256": hash_image_sources(sequence_root, image_paths),
        "focal_sha256": sha256_file(focal_path),
        "timestamp_sha256": sha256_file(timestamp_path),
    }


def part_ranges(
    frame_count: int,
    interval: int,
    selected_parts: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if interval < 1:
        raise ValueError("--interval must be positive")
    all_ranges = [
        (start, min(start + interval, frame_count))
        for start in range(0, frame_count, interval)
    ]
    if not selected_parts:
        return all_ranges
    unique_parts = list(dict.fromkeys(int(part) for part in selected_parts))
    invalid = [part for part in unique_parts if part < 0 or part >= len(all_ranges)]
    if invalid:
        raise ValueError(
            f"Part index {invalid[0]} is outside 0..{len(all_ranges) - 1}")
    return [all_ranges[part] for part in unique_parts]


def build_frame_aligned_tracks(
    frame_detections: Sequence[Sequence[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Select at most one detector box per hand and retain every video frame."""
    tracks: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for frame_id, detections in enumerate(frame_detections):
        best: dict[int, dict[str, Any]] = {}
        for detection in detections:
            try:
                side = int(detection["class_id"])
                score = float(detection["confidence"])
                box = np.asarray(detection["xyxy"], dtype=np.float32).reshape(-1)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed detector result at frame {frame_id}") from exc
            if side not in (0, 1) or box.size < 4 or not np.isfinite(box[:4]).all():
                continue
            if not np.isfinite(score):
                continue
            if side not in best or score > float(best[side]["confidence"]):
                best[side] = detection

        for side in (0, 1):
            chosen = best.get(side)
            if chosen is None:
                box = np.zeros((1, 5), dtype=np.float32)
                detected = False
                track_id = -1
            else:
                xyxy = np.asarray(chosen["xyxy"], dtype=np.float32).reshape(-1)[:4]
                score = float(chosen["confidence"])
                box = np.concatenate([xyxy, np.asarray([score], dtype=np.float32)])[None]
                detected = True
                raw_track_id = chosen.get("track_id")
                track_id = -1 if raw_track_id is None else int(raw_track_id)
            tracks[side].append({
                "frame": frame_id,
                "det": detected,
                "det_box": box,
                "det_handedness": np.asarray([side], dtype=np.int8),
                "det_track_id": track_id,
            })
    return tracks


def _detect_tracks(
    sequence_root: Path,
    image_paths: Sequence[Path],
    detector_weights: Path,
    confidence: float,
    device: str,
    image_hash: str,
) -> tuple[Path, dict[str, Any]]:
    track_path = sequence_root / TRACK_RELATIVE_PATH
    if track_path.is_file() and (track_path.parent / TRACK_PROVENANCE_NAME).is_file():
        try:
            _, existing = load_and_verify_detected_tracks(
                sequence_root, image_paths, detector_weights, image_hash)
            return track_path, existing
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass

    import cv2
    import torch
    from ultralytics import YOLO

    if not detector_weights.is_file():
        raise FileNotFoundError(f"Missing YOLO detector weights: {detector_weights}")
    detector = YOLO(str(detector_weights))
    frame_detections: list[list[dict[str, Any]]] = []
    for frame_id, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read extracted image {image_path}")
        with torch.inference_mode():
            result = detector.track(
                image,
                conf=confidence,
                persist=True,
                verbose=False,
                device=device,
            )[0]
        rows: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
            ids = boxes.id
            track_ids = (
                ids.detach().cpu().numpy().astype(np.int64)
                if ids is not None else np.full(len(classes), -1, dtype=np.int64))
            for box, score, class_id, track_id in zip(xyxy, scores, classes, track_ids):
                rows.append({
                    "xyxy": box,
                    "confidence": float(score),
                    "class_id": int(class_id),
                    "track_id": int(track_id),
                })
        frame_detections.append(rows)

    tracks = build_frame_aligned_tracks(frame_detections)
    _save_npy_atomic(track_path, np.asarray(tracks, dtype=object))
    metadata = {
        "schema_version": 1,
        "frontend": "YOLO",
        "producer": PRODUCER_ID,
        "track_file": TRACK_RELATIVE_PATH.as_posix(),
        "track_sha256": sha256_file(track_path),
        "detector_weights": str(detector_weights.resolve()),
        "detector_weights_sha256": sha256_file(detector_weights),
        "image_manifest_sha256": image_hash,
        "oracle_assisted": False,
        "class_map": {"0": "left", "1": "right"},
        "frame_count": len(image_paths),
    }
    _write_json_atomic(track_path.parent / TRACK_PROVENANCE_NAME, metadata)
    return track_path, metadata


def load_and_verify_detected_tracks(
    sequence_root: Path,
    image_paths: Sequence[Path],
    detector_weights: Path,
    image_manifest_sha256: str,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    track_path = sequence_root / TRACK_RELATIVE_PATH
    provenance_path = track_path.parent / TRACK_PROVENANCE_NAME
    if not track_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            f"Run --stage detect first; expected {track_path} and {provenance_path}")
    if not detector_weights.is_file():
        raise FileNotFoundError(f"Missing detector weights used for provenance: {detector_weights}")
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    expected = {
        "schema_version": 1,
        "frontend": "YOLO",
        "producer": PRODUCER_ID,
        "track_file": TRACK_RELATIVE_PATH.as_posix(),
        "track_sha256": sha256_file(track_path),
        "detector_weights_sha256": sha256_file(detector_weights),
        "image_manifest_sha256": image_manifest_sha256,
        "oracle_assisted": False,
        "frame_count": len(image_paths),
    }
    mismatches = [
        key for key, value in expected.items() if provenance.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Detected track provenance does not match current inputs: "
            + ", ".join(mismatches))
    loaded = np.load(track_path, allow_pickle=True)
    try:
        tracks = loaded.item()
    except ValueError as exc:
        raise ValueError(f"Track file must contain a {0}/{1} hand mapping: {track_path}") from exc
    if not isinstance(tracks, dict) or not all(side in tracks for side in (0, 1)):
        raise ValueError(f"Expected detector tracks with keys 0 and 1 in {track_path}")
    for side in (0, 1):
        entries = tracks[side]
        if not isinstance(entries, (list, tuple, np.ndarray)) or len(entries) != len(image_paths):
            raise ValueError(
                f"Hand {side} track must have one entry per image in {track_path}")
        for frame_id, entry in enumerate(entries):
            if not isinstance(entry, dict) or int(entry.get("frame", -1)) != frame_id:
                raise ValueError(
                    f"Hand {side} track is not frame-aligned at frame {frame_id}")
    return tracks, provenance


def frontend_chunk_hashes(frontend_root: Path) -> dict[str, str]:
    prediction_dir = frontend_root / "1"
    chunks: dict[str, str] = {}
    for path in prediction_dir.glob("*.json"):
        if not path.stem.isdecimal():
            raise ValueError(
                f"HaWoR chunk filename must be an integer start frame: {path}")
        chunks[path.relative_to(frontend_root).as_posix()] = sha256_file(path)
    return dict(sorted(chunks.items(), key=lambda item: int(Path(item[0]).stem)))


def build_frontend_manifest(
    sequence_root: Path,
    frontend_root: Path,
    *,
    extra_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    track_path = sequence_root / TRACK_RELATIVE_PATH
    if not track_path.is_file():
        raise FileNotFoundError(f"Missing detector tracks: {track_path}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "frontend": "HaWoR",
        "track_file": TRACK_RELATIVE_PATH.as_posix(),
        "track_sha256": sha256_file(track_path),
        "oracle_assisted": False,
        "chunk_sha256": frontend_chunk_hashes(frontend_root),
    }
    if extra_sources:
        manifest["source_sha256"] = extra_sources
    return manifest


def verify_frontend_manifest(sequence_root: Path, frontend_root: Path) -> dict[str, Any]:
    manifest_path = frontend_root / FRONTEND_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing HaWoR frontend manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = build_frontend_manifest(sequence_root, frontend_root)
    for key in ("schema_version", "frontend", "track_file", "track_sha256", "oracle_assisted", "chunk_sha256"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"HaWoR frontend manifest is stale or invalid at {manifest_path}: {key}")
    if manifest.get("oracle_assisted") is not False:
        raise ValueError(f"Oracle-assisted frontends cannot be used for formal training: {manifest_path}")
    if not expected["chunk_sha256"]:
        raise ValueError(f"No right-hand HaWoR chunks in {frontend_root / '1'}")
    return manifest


def _load_hawor_model(checkpoint: Path, device: str):
    _patch_legacy_mano_pickle_compat()
    from hawor.configs import get_config
    from lib.models.hawor import HAWOR

    config_path = checkpoint.parent.parent / "model_config.yaml"
    model_cfg = get_config(str(config_path), update_cachedir=True)
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        if model_cfg.MODEL.IMAGE_SIZE != 256:
            raise ValueError("HaWoR ViT checkpoint expects MODEL.IMAGE_SIZE == 256")
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    model = HAWOR.load_from_checkpoint(str(checkpoint), strict=False, cfg=model_cfg)
    model = model.to(device).eval()
    return model, config_path


def _interpolated_hand_segment(
    entries: Sequence[dict[str, Any]], start: int, stop: int,
) -> tuple[int, np.ndarray] | None:
    detected_frames: list[int] = []
    detected_boxes: list[np.ndarray] = []
    for frame_id in range(start, stop):
        item = entries[frame_id]
        if not bool(item.get("det", False)):
            continue
        raw = np.asarray(item.get("det_box", []), dtype=np.float32)
        if raw.size == 0 or raw.shape[-1] < 4:
            continue
        box = raw.reshape(-1, raw.shape[-1])[0]
        if not np.isfinite(box[:4]).all() or box[2] <= box[0] or box[3] <= box[1]:
            continue
        confidence = float(box[4]) if len(box) > 4 and np.isfinite(box[4]) else 1.0
        detected_frames.append(frame_id)
        detected_boxes.append(np.concatenate([box[:4], [confidence]]).astype(np.float32))
    if not detected_frames:
        return None
    first, last = detected_frames[0], detected_frames[-1]
    target_frames = np.arange(first, last + 1, dtype=np.int32)
    box_array = np.stack(detected_boxes)
    interpolated = np.stack([
        np.interp(target_frames, detected_frames, box_array[:, column])
        for column in range(box_array.shape[1])
    ], axis=1).astype(np.float32)
    return first, interpolated


def _rotation_postprocess(data_out: dict[str, Any], do_flip: bool) -> dict[str, Any]:
    from hawor.utils.rotation import (
        angle_axis_to_rotation_matrix,
        rotation_matrix_to_angle_axis,
    )

    root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    hand = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
    if do_flip:
        root[..., 1] *= -1
        root[..., 2] *= -1
        hand[..., 1] *= -1
        hand[..., 2] *= -1
    data_out["init_root_orient"] = angle_axis_to_rotation_matrix(root)
    data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(hand)
    return data_out


def _patch_legacy_mano_pickle_compat() -> None:
    """Apply the same aliases used by prepare_mano_assets for legacy MANO pickles."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _predict_hand(
    model: Any,
    image_paths: Sequence[Path],
    focal: float,
    width: int,
    height: int,
    side: int,
    start: int,
    boxes: np.ndarray,
    device: str,
) -> dict[str, Any]:
    import torch

    frames = np.arange(start, start + len(boxes), dtype=np.int32)
    results = model.inference(
        np.asarray([str(image_paths[frame]) for frame in frames]),
        boxes,
        img_focal=focal,
        img_center=[width / 2.0, height / 2.0],
        device=device,
        do_flip=(side == 0),
    )
    data_out = {
        "init_root_orient": results["pred_rotmat"][None, :, 0],
        "init_hand_pose": results["pred_rotmat"][None, :, 1:],
        "init_trans": results["pred_trans"][None, :, 0],
        "init_betas": results["pred_shape"][None, :],
    }
    data_out = _rotation_postprocess(data_out, do_flip=(side == 0))
    return {key: value.detach().cpu().numpy().tolist() for key, value in data_out.items()}


def _part_cache_path(frontend_root: Path, side: int, start: int, stop: int) -> Path:
    return frontend_root / "cache" / str(side) / f"{start}_{stop}.json"


def _read_cache(path: Path, *, track_hash: str, checkpoint_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("track_sha256") != track_hash or payload.get("hawor_checkpoint_sha256") != checkpoint_hash:
        return None
    return payload


def _write_cache(
    path: Path,
    *,
    track_hash: str,
    checkpoint_hash: str,
    prediction_start: int | None,
    prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "track_sha256": track_hash,
        "hawor_checkpoint_sha256": checkpoint_hash,
        "prediction_start": prediction_start,
        "prediction": prediction,
    }
    _write_json_atomic(path, payload)
    return payload


def _empty_rle(mask_tool: Any, height: int, width: int) -> dict[str, Any]:
    zero = np.zeros((height, width), dtype=np.uint8, order="F")
    return mask_tool.encode(zero)


def _render_prediction_masks(
    predictions: dict[int, dict[str, Any]],
    part_start: int,
    part_stop: int,
    height: int,
    width: int,
    focal: float,
    device: str,
    mask_tool: Any,
) -> list[Any]:
    """Render both MANO meshes into per-frame COCO RLE without dense clip masks."""
    import torch
    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
    from hawor.utils.rotation import rotation_matrix_to_angle_axis
    from lib.vis.renderer import Renderer

    empty_mask = _empty_rle(mask_tool, height, width)
    masks = [empty_mask] * (part_stop - part_start)
    if not predictions:
        return masks

    _patch_legacy_mano_pickle_compat()
    original_cwd = Path.cwd()
    os.chdir(REPOSITORY_ROOT)
    try:
        faces_right = get_mano_faces()
    finally:
        os.chdir(original_cwd)
    faces_new = np.asarray([
        [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
        [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
        [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
        [120, 108, 78], [78, 108, 79],
    ], dtype=np.int64)
    faces_right = np.concatenate([faces_right, faces_new], axis=0)
    faces_by_side = {
        0: torch.as_tensor(faces_right[:, [0, 2, 1]], dtype=torch.int64, device=device),
        1: torch.as_tensor(faces_right, dtype=torch.int64, device=device),
    }
    renderer = Renderer(
        width,
        height,
        focal,
        device,
        bin_size=128,
        max_faces_per_bin=20000,
    )
    camera_rotation = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
    camera_translation = torch.zeros((1, 3), dtype=torch.float32, device=device)
    cameras, lights = renderer.create_camera_from_cv(camera_rotation, camera_translation)

    for side in (0, 1):
        cached = predictions.get(side)
        if cached is None:
            continue
        prediction_start = int(cached["start"])
        data = cached["payload"]
        root_matrix = torch.as_tensor(data["init_root_orient"], dtype=torch.float32, device=device)
        hand_matrix = torch.as_tensor(data["init_hand_pose"], dtype=torch.float32, device=device)
        translation = torch.as_tensor(data["init_trans"], dtype=torch.float32, device=device)
        betas = torch.as_tensor(data["init_betas"], dtype=torch.float32, device=device)
        root_axis_angle = rotation_matrix_to_angle_axis(root_matrix)
        hand_axis_angle = rotation_matrix_to_angle_axis(hand_matrix)
        mano = run_mano_left if side == 0 else run_mano
        original_cwd = Path.cwd()
        os.chdir(REPOSITORY_ROOT)
        try:
            with torch.inference_mode():
                outputs = mano(
                    translation,
                    root_axis_angle,
                    hand_axis_angle,
                    betas=betas,
                    use_cuda=str(device).startswith("cuda"),
                )
        finally:
            os.chdir(original_cwd)
        vertices = outputs["vertices"][0]
        for batch_start in range(0, len(vertices), 8):
            batch_end = min(batch_start + 8, len(vertices))
            rendered = renderer.render_silhouettes_batch(
                vertices[batch_start:batch_end].to(device),
                faces_by_side[side],
                cameras,
                lights,
            )
            for offset, dense_mask in enumerate(rendered):
                frame = prediction_start + batch_start + offset
                if not part_start <= frame < part_stop:
                    continue
                encoded = mask_tool.encode(
                    np.asfortranarray(np.asarray(dense_mask, dtype=np.uint8)))
                local_frame = frame - part_start
                masks[local_frame] = mask_tool.merge([masks[local_frame], encoded])
        del outputs, vertices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return masks


def _read_frontend_state(frontend_root: Path) -> dict[str, Any] | None:
    path = frontend_root / FRONTEND_STATE_NAME
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if state.get("producer") != PRODUCER_ID:
        return None
    return state


def _prepare_frontend_state(
    frontend_root: Path,
    source_hashes: dict[str, str],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    state_path = frontend_root / FRONTEND_STATE_NAME
    state = _read_frontend_state(frontend_root)
    generated_dirs = [frontend_root / "1", frontend_root / "cache", frontend_root / "masks"]
    if state is not None and state.get("source_sha256") != source_hashes:
        for path in generated_dirs:
            if path.exists():
                shutil.rmtree(path)
        for name in (FRONTEND_MANIFEST_NAME, FRONTEND_STATE_NAME):
            path = frontend_root / name
            if path.exists():
                path.unlink()
        state = None
    elif state is None and (
        (frontend_root / FRONTEND_MANIFEST_NAME).is_file()
        or any(path.exists() and any(path.iterdir()) for path in generated_dirs)
    ):
        if not overwrite:
            raise FileExistsError(
                f"Untracked HaWoR outputs already exist under {frontend_root}; "
                "pass --overwrite to replace them")
        for path in generated_dirs:
            if path.exists():
                shutil.rmtree(path)
        manifest_path = frontend_root / FRONTEND_MANIFEST_NAME
        if manifest_path.exists():
            manifest_path.unlink()
    frontend_root.mkdir(parents=True, exist_ok=True)
    if state is None:
        state = {
            "schema_version": 1,
            "producer": PRODUCER_ID,
            "source_sha256": source_hashes,
            "parts": {},
        }
        _write_json_atomic(state_path, state)
    return state


def _load_or_initialize_global_masks(
    sequence_root: Path,
    frame_count: int,
    height: int,
    width: int,
    source_hashes: dict[str, str],
    mask_tool: Any,
    *,
    overwrite: bool,
) -> np.ndarray:
    path = sequence_root / MASKS_RELATIVE_PATH
    provenance_path = path.parent / MASKS_PROVENANCE_NAME
    if path.exists() and provenance_path.exists() and not overwrite:
        with provenance_path.open("r", encoding="utf-8") as handle:
            provenance = json.load(handle)
        if provenance.get("source_sha256") == source_hashes:
            if provenance.get("model_masks_sha256") != sha256_file(path):
                raise ValueError(f"Model masks changed after provenance was written: {path}")
            masks = np.load(path, allow_pickle=True)
            if masks.ndim == 1 and len(masks) == frame_count:
                return masks
            raise ValueError(f"Unexpected model mask array shape in {path}: {masks.shape}")
        # A matching producer sidecar proves this is our output; input changes
        # invalidate all prior masks, so reset them to the empty-mask RLE.
    elif path.exists() and not overwrite:
        raise FileExistsError(
            f"Unverified model masks exist at {path}; pass --overwrite to replace them")
    zero_rle = _empty_rle(mask_tool, height, width)
    return np.asarray([zero_rle] * frame_count, dtype=object)


def _write_global_masks(
    sequence_root: Path,
    masks: np.ndarray,
    source_hashes: dict[str, str],
    *,
    completed_parts: dict[str, str],
) -> str:
    path = sequence_root / MASKS_RELATIVE_PATH
    _save_npy_atomic(path, np.asarray(masks, dtype=object))
    digest = sha256_file(path)
    provenance = {
        "schema_version": 1,
        "producer": PRODUCER_ID,
        "frame_count": len(masks),
        "track_sha256": source_hashes["detected_tracks"],
        "hawor_checkpoint_sha256": source_hashes["hawor_checkpoint"],
        "source_sha256": source_hashes,
        "completed_part_rle_sha256": completed_parts,
        "model_masks_sha256": digest,
    }
    _write_json_atomic(path.parent / MASKS_PROVENANCE_NAME, provenance)
    return digest


def _prepare_hawor_parts(
    sequence_root: Path,
    frontend_root: Path,
    image_paths: Sequence[Path],
    metadata: dict[str, Any],
    tracks: dict[int, list[dict[str, Any]]],
    detector_weights: Path,
    checkpoint: Path,
    model_config_path: Path,
    model: Any,
    ranges: Sequence[tuple[int, int]],
    device: str,
    *,
    overwrite: bool,
    full_coverage: bool,
) -> dict[str, Any]:
    import torch
    from pycocotools import mask as mask_tool

    checkpoint_hash = sha256_file(checkpoint)
    model_config_hash = sha256_file(model_config_path)
    track_hash = sha256_file(sequence_root / TRACK_RELATIVE_PATH)
    detector_hash = sha256_file(detector_weights)
    source_hashes = {
        "detector_weights": detector_hash,
        "hawor_checkpoint": checkpoint_hash,
        "hawor_model_config": model_config_hash,
        "image_manifest": metadata["image_manifest_sha256"],
        "focal": metadata["focal_sha256"],
        "timestamps": metadata["timestamp_sha256"],
        "detected_tracks": track_hash,
    }
    state = _prepare_frontend_state(frontend_root, source_hashes, overwrite=overwrite)

    masks_path = sequence_root / MASKS_RELATIVE_PATH
    masks = _load_or_initialize_global_masks(
        sequence_root,
        len(image_paths),
        metadata["height"],
        metadata["width"],
        source_hashes,
        mask_tool,
        overwrite=overwrite,
    )
    mask_provenance_path = masks_path.parent / MASKS_PROVENANCE_NAME
    existing_mask_provenance: dict[str, Any] = {}
    if mask_provenance_path.is_file() and not overwrite:
        try:
            existing_mask_provenance = json.loads(
                mask_provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_mask_provenance = {}
    if (
        existing_mask_provenance.get("source_sha256") == source_hashes
        and existing_mask_provenance.get("model_masks_sha256") == sha256_file(masks_path)
    ):
        completed_rle_hashes = dict(
            existing_mask_provenance.get("completed_part_rle_sha256", {}))
    else:
        completed_rle_hashes = {}

    for part_start, part_stop in ranges:
        part_key = f"{part_start}:{part_stop}"
        prediction_by_side: dict[int, dict[str, Any]] = {}
        part_state = state["parts"].get(part_key, {})
        for side in (0, 1):
            cache_path = _part_cache_path(frontend_root, side, part_start, part_stop)
            cached = _read_cache(
                cache_path,
                track_hash=track_hash,
                checkpoint_hash=checkpoint_hash,
            )
            if cached is None:
                segment = _interpolated_hand_segment(tracks[side], part_start, part_stop)
                if segment is None:
                    cached = _write_cache(
                        cache_path,
                        track_hash=track_hash,
                        checkpoint_hash=checkpoint_hash,
                        prediction_start=None,
                        prediction=None,
                    )
                else:
                    prediction_start, boxes = segment
                    prediction = _predict_hand(
                        model,
                        image_paths,
                        metadata["focal"],
                        metadata["width"],
                        metadata["height"],
                        side,
                        prediction_start,
                        boxes,
                        device,
                    )
                    cached = _write_cache(
                        cache_path,
                        track_hash=track_hash,
                        checkpoint_hash=checkpoint_hash,
                        prediction_start=prediction_start,
                        prediction=prediction,
                    )
            if cached.get("prediction") is not None:
                prediction_by_side[side] = {
                    "start": int(cached["prediction_start"]),
                    "payload": cached["prediction"],
                }

        right = prediction_by_side.get(1)
        old_public = part_state.get("right_chunk")
        if old_public:
            previous_path = frontend_root / old_public
            if previous_path.is_file():
                previous_path.unlink()
        right_relative: str | None = None
        if right is not None:
            right_path = frontend_root / "1" / f"{right['start']}.json"
            _write_json_atomic(right_path, right["payload"])
            right_relative = right_path.relative_to(frontend_root).as_posix()

        part_mask_path = frontend_root / "masks" / f"{part_start}_{part_stop}.npy"
        expected_part_state = (
            part_state.get("track_sha256") == track_hash
            and part_state.get("hawor_checkpoint_sha256") == checkpoint_hash
        )
        if (
            expected_part_state
            and part_state.get("part_mask_sha256")
            and part_mask_path.is_file()
            and sha256_file(part_mask_path) == part_state["part_mask_sha256"]
        ):
            part_rles = np.load(part_mask_path, allow_pickle=True).tolist()
        else:
            part_rles = _render_prediction_masks(
                prediction_by_side,
                part_start,
                part_stop,
                metadata["height"],
                metadata["width"],
                metadata["focal"],
                device,
                mask_tool,
            )
            _save_npy_atomic(part_mask_path, np.asarray(part_rles, dtype=object))
        part_mask_hash = sha256_file(part_mask_path)
        masks[part_start:part_stop] = part_rles
        completed_rle_hashes[part_mask_path.relative_to(frontend_root).as_posix()] = part_mask_hash
        global_mask_hash = _write_global_masks(
            sequence_root,
            masks,
            source_hashes,
            completed_parts=completed_rle_hashes,
        )

        state["parts"][part_key] = {
            "track_sha256": track_hash,
            "hawor_checkpoint_sha256": checkpoint_hash,
            "right_chunk": right_relative,
            "right_chunk_sha256": (
                sha256_file(frontend_root / right_relative) if right_relative else None),
            "left_cache_sha256": sha256_file(_part_cache_path(frontend_root, 0, part_start, part_stop)),
            "right_cache_sha256": sha256_file(_part_cache_path(frontend_root, 1, part_start, part_stop)),
            "part_mask_sha256": part_mask_hash,
            "global_model_masks_sha256": global_mask_hash,
        }
        _write_json_atomic(frontend_root / FRONTEND_STATE_NAME, state)

    owned_chunks = {
        part["right_chunk"]: part["right_chunk_sha256"]
        for part in state["parts"].values()
        if part.get("right_chunk")
    }
    actual_chunks = frontend_chunk_hashes(frontend_root)
    if set(owned_chunks) != set(actual_chunks):
        raise ValueError(
            "Generated HaWoR chunks do not match producer state; rerun the missing "
            "parts before writing a verified manifest")
    for relative_path, digest in actual_chunks.items():
        if owned_chunks[relative_path] != digest:
            raise ValueError(
                f"HaWoR chunk changed after it was recorded in producer state: {relative_path}")

    manifest = build_frontend_manifest(
        sequence_root,
        frontend_root,
        extra_sources={
            **source_hashes,
            "model_masks": sha256_file(masks_path),
        },
    )
    _write_json_atomic(frontend_root / FRONTEND_MANIFEST_NAME, manifest)
    if full_coverage and not manifest["chunk_sha256"]:
        raise ValueError(
            f"HaWoR produced no right-hand detections for {sequence_root}; "
            "formal training requires at least one verified right-hand chunk")
    return state


def _resize_shape(height: int, width: int) -> tuple[int, int]:
    scale = math.sqrt((384 * 512) / float(height * width))
    resized_height = max(8, int(height * scale))
    resized_width = max(8, int(width * scale))
    return resized_height - resized_height % 8, resized_width - resized_width % 8


def _droid_image_stream(
    image_paths: Sequence[Path],
    part_start: int,
    part_stop: int,
    calibration: np.ndarray,
    original_height: int,
    original_width: int,
    resized_height: int,
    resized_width: int,
):
    import cv2
    import torch

    for local_id, frame_id in enumerate(range(part_start, part_stop)):
        image = cv2.imread(str(image_paths[frame_id]))
        if image is None:
            raise ValueError(f"Cannot read image {image_paths[frame_id]}")
        image = cv2.resize(image, (resized_width, resized_height))
        tensor = torch.as_tensor(image).permute(2, 0, 1)
        intrinsics = torch.as_tensor(calibration.copy(), dtype=torch.float32)
        intrinsics[0::2] *= resized_width / original_width
        intrinsics[1::2] *= resized_height / original_height
        yield local_id, tensor[None], intrinsics


def _direct_droid_args(weights: Path, image_size: tuple[int, int], buffer_size: int):
    return SimpleNamespace(
        weights=str(weights),
        image_size=list(image_size),
        buffer=buffer_size,
        disable_vis=True,
        stereo=False,
        filter_thresh=2.4,
        upsample=True,
        beta=0.3,
        warmup=8,
        frontend_nms=1,
        keyframe_thresh=4.0,
        frontend_window=25,
        frontend_thresh=16.0,
        frontend_radius=2,
        backend_thresh=22.0,
        backend_radius=2,
        backend_nms=3,
    )


def slam_tracking_start(part_start: int, part_stop: int) -> int:
    """Give a short final chunk past-image context for SLAM initialization."""
    if not 0 <= part_start < part_stop:
        raise ValueError("Invalid SLAM frame range")
    return max(0, part_start - 1000) if part_stop - part_start < 32 else part_start


def crop_slam_context(trajectory, timestamps, disparities, offset, frame_count):
    if offset < 0 or len(trajectory) != offset + frame_count:
        raise ValueError("SLAM context trajectory length mismatch")
    keep = (timestamps >= offset) & (timestamps < offset + frame_count)
    return trajectory[offset:], timestamps[keep] - offset, disparities[keep]


def validate_context_mask_coverage(completed_parts, start, stop):
    ranges = []
    for key, digest in completed_parts.items():
        if not digest:
            continue
        try:
            left, right = map(int, Path(key).stem.split("_"))
        except ValueError:
            continue
        ranges.append((left, right))
    covered = start
    for left, right in sorted(ranges):
        if left <= covered < right:
            covered = right
        if covered >= stop:
            return
    raise ValueError(f"Missing completed HaWoR masks for SLAM context {start}:{stop}")


def _run_masked_slam_part(
    sequence_root: Path,
    image_paths: Sequence[Path],
    metadata: dict[str, Any],
    part_start: int,
    part_stop: int,
    droid_weights: Path,
    metric_weights: Path,
    device: str,
) -> Path:
    if not device.startswith("cuda"):
        raise RuntimeError("Masked DROID-SLAM and this Metric3D checkpoint require CUDA")
    import cv2
    import torch
    from pycocotools import mask as mask_tool

    sys.path.insert(0, str(REPOSITORY_ROOT / "thirdparty" / "DROID-SLAM" / "droid_slam"))
    sys.path.insert(0, str(REPOSITORY_ROOT / "thirdparty" / "DROID-SLAM"))
    from droid import Droid

    metric3d_root = REPOSITORY_ROOT / "thirdparty" / "Metric3D"
    sys.path.insert(0, str(metric3d_root))
    from metric import Metric3D
    from lib.pipeline.est_scale import est_scale_hybrid

    requested_start = part_start
    part_start = slam_tracking_start(part_start, part_stop)
    if part_start != requested_start:
        print(f"[slam] short_tail={requested_start}:{part_stop} "
              f"tracking_context={part_start}:{part_stop}; past images only", flush=True)
    first_image = cv2.imread(str(image_paths[part_start]))
    if first_image is None:
        raise ValueError(f"Cannot read image {image_paths[part_start]}")
    height, width = first_image.shape[:2]
    resized_height, resized_width = _resize_shape(height, width)
    frame_count = part_stop - part_start
    mask_archive = np.load(sequence_root / MASKS_RELATIVE_PATH, allow_pickle=True)
    rle_part = mask_archive[part_start:part_stop]
    if len(rle_part) != frame_count:
        raise ValueError(f"RLE mask/frame mismatch in {sequence_root}")
    calibration = np.asarray([
        metadata["focal"], metadata["focal"], width / 2.0, height / 2.0,
    ], dtype=np.float32)
    args = _direct_droid_args(
        droid_weights,
        (resized_height, resized_width),
        max(512, frame_count + 16),
    )
    droid = Droid(args)

    for local_id, frame_id in enumerate(range(part_start, part_stop)):
        image = cv2.imread(str(image_paths[frame_id]))
        if image is None:
            raise ValueError(f"Cannot read image {image_paths[frame_id]}")
        resized = cv2.resize(image, (resized_width, resized_height))
        dense_mask = mask_tool.decode(rle_part[local_id]).astype(np.float32)
        resized_mask = cv2.resize(
            dense_mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        confidence_mask = cv2.resize(
            dense_mask,
            (resized_width // 8, resized_height // 8),
            interpolation=cv2.INTER_LINEAR,
        )
        image_tensor = torch.as_tensor(resized).permute(2, 0, 1)
        image_tensor = image_tensor * torch.from_numpy((resized_mask < 0.5).astype(np.uint8))[None]
        intrinsics = torch.as_tensor(calibration.copy(), dtype=torch.float32)
        intrinsics[0::2] *= resized_width / width
        intrinsics[1::2] *= resized_height / height
        mask_tensor = torch.from_numpy(confidence_mask.astype(np.float32))
        droid.track(local_id, image_tensor[None], intrinsics=intrinsics, mask=mask_tensor)
        del dense_mask, resized_mask, confidence_mask, mask_tensor
        if (local_id + 1) % 100 == 0 or local_id + 1 == frame_count:
            print(f"[slam] {sequence_root.name} part={part_start}:{part_stop} "
                  f"tracked={local_id + 1}/{frame_count}", flush=True)

    stream = _droid_image_stream(
        image_paths,
        part_start,
        part_stop,
        calibration,
        height,
        width,
        resized_height,
        resized_width,
    )
    trajectory = droid.terminate(stream)
    used_frames = int(droid.video.counter.value)
    tstamp = droid.video.tstamp[:used_frames].cpu().int().numpy()
    disparities = droid.video.disps_up[:used_frames].cpu().numpy()
    del droid
    torch.cuda.empty_cache()
    if used_frames == 0 or len(trajectory) == 0:
        raise RuntimeError(f"DROID-SLAM produced no poses for {sequence_root} part {part_start}")

    metric = Metric3D(str(metric_weights))
    print(f"[slam] {sequence_root.name} part={part_start}:{part_stop} "
          f"trajectory_frames={len(trajectory)} scale_keyframes={len(tstamp)}", flush=True)
    scales: list[float] = []
    near_threshold, far_threshold = 0.4, 0.7
    for index, local_frame in enumerate(tstamp):
        local_frame = int(local_frame)
        image_index = part_start + local_frame
        predicted_depth = metric(str(image_paths[image_index]), calibration)
        predicted_depth = cv2.resize(predicted_depth, (resized_width, resized_height))
        slam_depth = 1.0 / np.maximum(disparities[index], 1e-8)
        frame_mask = mask_tool.decode(rle_part[local_frame]).astype(np.float32)
        scale = est_scale_hybrid(
            slam_depth,
            predicted_depth,
            sigma=0.5,
            msk=frame_mask,
            near_thresh=near_threshold,
            far_thresh=far_threshold,
        )
        attempts = 0
        while not np.isfinite(scale) and attempts < 12:
            near_threshold = max(0.0, near_threshold - 0.1)
            far_threshold += 0.1
            scale = est_scale_hybrid(
                slam_depth,
                predicted_depth,
                sigma=0.5,
                msk=frame_mask,
                near_thresh=near_threshold,
                far_thresh=far_threshold,
            )
            attempts += 1
        if np.isfinite(scale) and scale > 0:
            scales.append(float(scale))
        del predicted_depth, frame_mask
        if (index + 1) % 10 == 0 or index + 1 == len(tstamp):
            print(f"[slam] {sequence_root.name} part={part_start}:{part_stop} "
                  f"scale_progress={index + 1}/{len(tstamp)} valid_scales={len(scales)}", flush=True)
    if not scales:
        raise RuntimeError(
            f"Metric3D and masked DROID-SLAM found no finite scale samples for "
            f"{sequence_root} part {part_start}")
    median_scale = float(np.median(scales))
    trajectory, tstamp, disparities = crop_slam_context(
        trajectory, tstamp, disparities, requested_start - part_start, part_stop - requested_start)
    output = sequence_root / "preprocess" / "SLAM" / f"hawor_slam_w_scale_{requested_start}_{part_stop}.npz"
    _save_npz_atomic(
        output,
        tstamp=tstamp,
        disps=disparities,
        traj=trajectory,
        img_focal=np.asarray(metadata["focal"], dtype=np.float32),
        img_center=np.asarray([width / 2.0, height / 2.0], dtype=np.float32),
        scale=np.asarray(median_scale, dtype=np.float64),
        tracking_context_start_frame=np.asarray(part_start, dtype=np.int32),
        requested_start_frame=np.asarray(requested_start, dtype=np.int32),
        scale_source=np.asarray("masked DROID + Metric3D over tracking context"),
    )
    print(f"[slam] completed {sequence_root.name} part={part_start}:{part_stop} "
          f"scale={median_scale:.6g} output={output}", flush=True)
    return output


def _verify_mask_provenance(
    sequence_root: Path,
    frame_count: int,
    track_hash: str,
    hawor_checkpoint_hash: str,
) -> tuple[Path, dict[str, Any]]:
    path = sequence_root / MASKS_RELATIVE_PATH
    provenance_path = path.parent / MASKS_PROVENANCE_NAME
    if not path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            f"Run --stage hawor first; missing generated masks at {path}")
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if (
        provenance.get("producer") != PRODUCER_ID
        or provenance.get("frame_count") != frame_count
        or provenance.get("track_sha256") != track_hash
        or provenance.get("hawor_checkpoint_sha256") != hawor_checkpoint_hash
        or provenance.get("model_masks_sha256") != sha256_file(path)
    ):
        raise ValueError(f"SLAM mask provenance does not match current frontend inputs: {path}")
    return path, provenance


def _run_slam_parts(
    sequence_root: Path,
    frontend_root: Path,
    image_paths: Sequence[Path],
    metadata: dict[str, Any],
    checkpoint: Path,
    droid_weights: Path,
    metric_weights: Path,
    ranges: Sequence[tuple[int, int]],
    device: str,
) -> None:
    checkpoint_hash = sha256_file(checkpoint)
    track_hash = sha256_file(sequence_root / TRACK_RELATIVE_PATH)
    mask_path, mask_provenance = _verify_mask_provenance(
        sequence_root, len(image_paths), track_hash, checkpoint_hash)
    droid_hash = sha256_file(droid_weights)
    metric_hash = sha256_file(metric_weights)
    for part_start, part_stop in ranges:
        rle_key = f"masks/{part_start}_{part_stop}.npy"
        rle_hash = mask_provenance.get("completed_part_rle_sha256", {}).get(rle_key)
        if not rle_hash:
            raise FileNotFoundError(
                f"No generated RLE masks for part {part_start}:{part_stop}; "
                "run --stage hawor for this part first")
        output = sequence_root / "preprocess" / "SLAM" / f"hawor_slam_w_scale_{part_start}_{part_stop}.npz"
        provenance_path = output.with_suffix(".provenance.json")
        part_sources = {
            "track_sha256": track_hash,
            "hawor_checkpoint_sha256": checkpoint_hash,
            "model_masks_sha256": sha256_file(mask_path),
            "part_rle_sha256": rle_hash,
            "droid_weights_sha256": droid_hash,
            "metric3d_weights_sha256": metric_hash,
            "image_manifest_sha256": metadata["image_manifest_sha256"],
            "focal_sha256": metadata["focal_sha256"],
        }
        if part_stop - part_start < 32:
            part_sources["short_tail_context_start_frame"] = slam_tracking_start(part_start, part_stop)
            validate_context_mask_coverage(
                mask_provenance.get("completed_part_rle_sha256", {}),
                part_sources["short_tail_context_start_frame"], part_stop)
        if output.is_file() and provenance_path.is_file():
            try:
                old = json.loads(provenance_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                old = {}
            if old.get("source_sha256") == part_sources and old.get("output_sha256") == sha256_file(output):
                continue
        output = _run_masked_slam_part(
            sequence_root,
            image_paths,
            metadata,
            part_start,
            part_stop,
            droid_weights,
            metric_weights,
            device,
        )
        _write_json_atomic(provenance_path, {
            "schema_version": 1,
            "producer": PRODUCER_ID,
            "part": [part_start, part_stop],
            "source_sha256": part_sources,
            "oracle_assisted": False,
            "output_sha256": sha256_file(output),
        })


def _resolve_sequence(value: str, video_root: Path) -> tuple[Path, str]:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        rooted = video_root / candidate
        candidate = rooted if rooted.exists() else candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Sequence directory does not exist: {candidate}")
    candidate = candidate.resolve()
    try:
        key = candidate.relative_to(video_root.resolve()).as_posix()
    except ValueError:
        key = candidate.name
    return candidate, key


def _check_runtime_and_weights(args: argparse.Namespace) -> list[str]:
    stages = {"detect", "hawor", "slam"} if args.stage == "all" else {args.stage}
    checks: list[tuple[str, str]] = [("numpy", "numpy")]
    required_files: list[tuple[Path, str]] = []
    if "detect" in stages:
        checks.extend([("torch", "torch"), ("cv2", "cv2"), ("ultralytics", "ultralytics")])
        required_files.append((args.detector_weights, "YOLO detector weights"))
    if "hawor" in stages:
        checks.append(("torch", "torch"))
        checks.extend([
            ("cv2", "cv2"), ("pytorch_lightning", "pytorch-lightning"),
            ("pytorch3d", "pytorch3d"), ("smplx", "smplx"),
            ("pycocotools", "pycocotools"),
        ])
        config = args.hawor_checkpoint.parent.parent / "model_config.yaml"
        required_files.extend([
            (args.detector_weights, "YOLO detector weights for track provenance"),
            (args.hawor_checkpoint, "HaWoR checkpoint"),
            (config, "HaWoR model config"),
            (REPOSITORY_ROOT / "_DATA/data/mano/MANO_RIGHT.pkl", "right MANO asset"),
            (REPOSITORY_ROOT / "_DATA/data_left/mano_left/MANO_LEFT.pkl", "left MANO asset"),
        ])
    if "slam" in stages:
        required_files.append((args.detector_weights, "YOLO detector weights for track provenance"))
        required_files.append((args.hawor_checkpoint, "HaWoR checkpoint for mask provenance"))
        checks.extend([
            ("torch", "torch"), ("cv2", "cv2"), ("pycocotools", "pycocotools"),
            ("lietorch", "lietorch"), ("mmengine", "mmengine"), ("torchmin", "torchmin"),
            ("torch_scatter", "torch_scatter"),
        ])
        required_files.extend([
            (args.droid_weights, "DROID-SLAM checkpoint"),
            (args.metric3d_weights, "Metric3D checkpoint"),
        ])
    missing: list[str] = []
    seen: set[str] = set()
    for module, package in checks:
        if module in seen:
            continue
        seen.add(module)
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(f"missing Python dependency {package}")
    for path, label in required_files:
        if not path.is_file():
            missing.append(f"missing {label}: {path}")
    if "slam" in stages:
        droid_dir = str(REPOSITORY_ROOT / "thirdparty/DROID-SLAM/droid_slam")
        sys.path.insert(0, droid_dir)
        try:
            try:
                extension_available = importlib.util.find_spec("droid_backends") is not None
            except (ImportError, ValueError):
                extension_available = False
        finally:
            sys.path.remove(droid_dir)
        if not extension_available:
            missing.append("missing compiled Python dependency droid_backends")
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", help="Sequence directory or name under --video_root; repeatable")
    parser.add_argument("--video_root", type=Path, default=REPOSITORY_ROOT / "datasets/hot3d_train100")
    parser.add_argument("--frontend_root", type=Path, default=REPOSITORY_ROOT / "eval_log_train")
    parser.add_argument("--stage", choices=("detect", "hawor", "slam", "all"), default="all")
    parser.add_argument("--part", action="append", type=int, help="0-based part index using --interval; repeatable")
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--detector_weights", type=Path, default=REPOSITORY_ROOT / "weights/external/detector.pt")
    parser.add_argument("--hawor_checkpoint", type=Path, default=REPOSITORY_ROOT / "weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--droid_weights", type=Path, default=REPOSITORY_ROOT / "weights/external/droid.pth")
    parser.add_argument(
        "--metric3d_weights",
        type=Path,
        default=REPOSITORY_ROOT / "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth",
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true", help="Replace untracked outputs at the selected destinations")
    parser.add_argument("--check-only", action="store_true", help="Report required dependencies and model files")
    return parser


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    original_cwd = Path.cwd()
    for name in (
        "video_root", "frontend_root", "detector_weights", "hawor_checkpoint",
        "droid_weights", "metric3d_weights",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (original_cwd / value).resolve())
    os.chdir(REPOSITORY_ROOT)
    if args.interval < 1:
        parser.error("--interval must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    issues = _check_runtime_and_weights(args)
    if args.check_only:
        if issues:
            print("Frontend preflight found missing requirements:")
            for issue in issues:
                print(f"- {issue}")
            return 2
        print(f"Frontend preflight passed for stage={args.stage}")
        return 0
    if not args.video:
        parser.error("at least one --video is required unless --check-only is used")
    if issues:
        raise RuntimeError("Frontend preflight failed:\n- " + "\n- ".join(issues))

    device = _resolve_device(args.device)
    config_path = args.hawor_checkpoint.parent.parent / "model_config.yaml"
    hawor_model = None
    if args.stage in ("hawor", "all"):
        hawor_model, config_path = _load_hawor_model(args.hawor_checkpoint, device)
    for video in args.video:
        sequence_root, sequence_key = _resolve_sequence(video, args.video_root)
        images = image_paths_for(sequence_root)
        metadata = load_sequence_metadata(sequence_root, images)
        frontend_root = args.frontend_root / sequence_key
        ranges = part_ranges(metadata["frame_count"], args.interval, args.part)
        stages = {"detect", "hawor", "slam"} if args.stage == "all" else {args.stage}

        if "detect" in stages:
            track_path, _ = _detect_tracks(
                sequence_root,
                images,
                args.detector_weights,
                args.confidence,
                device,
                metadata["image_manifest_sha256"],
            )
            print(f"detected tracks: {track_path}")
        if "hawor" in stages or "slam" in stages:
            tracks, _ = load_and_verify_detected_tracks(
                sequence_root,
                images,
                args.detector_weights,
                metadata["image_manifest_sha256"],
            )
            if "hawor" in stages:
                _prepare_hawor_parts(
                    sequence_root,
                    frontend_root,
                    images,
                    metadata,
                    tracks,
                    args.detector_weights,
                    args.hawor_checkpoint,
                    config_path,
                    hawor_model,
                    ranges,
                    device,
                    overwrite=args.overwrite,
                    full_coverage=(ranges == part_ranges(metadata["frame_count"], args.interval)),
                )
                print(f"verified HaWoR frontend: {frontend_root / FRONTEND_MANIFEST_NAME}")
            if "slam" in stages:
                _run_slam_parts(
                    sequence_root,
                    frontend_root,
                    images,
                    metadata,
                    args.hawor_checkpoint,
                    args.droid_weights,
                    args.metric3d_weights,
                    ranges,
                    device,
                )
                print(f"masked SLAM complete: {sequence_root / 'preprocess/SLAM'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
