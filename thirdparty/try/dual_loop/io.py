"""I/O adapters for the shared HOT3D prediction and Physical State Trace formats."""

from __future__ import annotations

import json
import hashlib
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch


PREDICTION_KEYS = (
    "frame_id",
    "valid",
    "handedness",
    "bbox",
    "global_orient",
    "hand_pose",
    "betas",
    "pred_cam",
    "camera_translation",
    "focal_length",
    "image_size",
)


def _empty_predictions(total_frames: int, image_size_wh: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "frame_id": np.arange(total_frames, dtype=np.int32),
        "valid": np.zeros(total_frames, dtype=bool),
        "handedness": np.ones(total_frames, dtype=np.int8),
        "bbox": np.full((total_frames, 4), np.nan, dtype=np.float32),
        "global_orient": np.full((total_frames, 1, 3, 3), np.nan, dtype=np.float32),
        "hand_pose": np.full((total_frames, 15, 3, 3), np.nan, dtype=np.float32),
        "betas": np.full((total_frames, 10), np.nan, dtype=np.float32),
        "pred_cam": np.full((total_frames, 3), np.nan, dtype=np.float32),
        "camera_translation": np.full((total_frames, 3), np.nan, dtype=np.float32),
        "focal_length": np.full(total_frames, np.nan, dtype=np.float32),
        "image_size": image_size_wh.astype(np.int32),
    }


def _resolve_tracks_path(sequence_root: Path, allow_gt_boxes: bool) -> tuple[Path, bool]:
    """Choose detected tracks by default; legacy GT-projected boxes are opt-in."""
    detected_path = sequence_root / "preprocess" / "preprocess_tracks.npy"
    if detected_path.is_file():
        return detected_path, False

    legacy_gt_path = sequence_root / "preprocess" / "preprocess_gtdet_tracks.npy"
    if allow_gt_boxes and legacy_gt_path.is_file():
        return legacy_gt_path, True

    if legacy_gt_path.is_file():
        raise FileNotFoundError(
            f"Only oracle-assisted HOT3D tracks are available at {legacy_gt_path}. "
            "Runtime defaults to detected preprocess/preprocess_tracks.npy; "
            "pass allow_gt_boxes=True only for an explicitly oracle-assisted run.")
    raise FileNotFoundError(
        f"Missing detected tracks: {detected_path}. The repository's legacy "
        "preprocess_gtdet_tracks.npy is GT-projected and is accepted only when "
        "allow_gt_boxes=True.")


def _load_right_hand_tracks(path: Path) -> list[dict]:
    loaded = np.load(path, allow_pickle=True)
    try:
        tracks = loaded.item()
    except ValueError:
        tracks = loaded.tolist()
    if isinstance(tracks, dict):
        if 1 in tracks:
            tracks = tracks[1]
        elif "1" in tracks:
            tracks = tracks["1"]
        elif "right" in tracks:
            tracks = tracks["right"]
        elif "right_hand" in tracks:
            tracks = tracks["right_hand"]
        else:
            raise ValueError(
                f"Track file {path} has no unambiguous right-hand track; "
                f"available keys: {list(tracks)}")
    if not isinstance(tracks, (list, tuple, np.ndarray)):
        raise ValueError(f"Unsupported track container in {path}: {type(tracks).__name__}")
    return list(tracks)


def load_timestamps(sequence_root: Path, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Load HOT3D synchronization timestamps and convert nanoseconds to seconds."""
    path = sequence_root / "tt_list.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing HOT3D timestamp source: {path}")
    with path.open("rb") as handle:
        values = pickle.load(handle)
    timestamp_ns = np.asarray(values)
    if timestamp_ns.ndim != 1 or len(timestamp_ns) != frame_count:
        raise ValueError(
            f"Expected {frame_count} timestamps in {path}, got shape {timestamp_ns.shape}")
    if not np.issubdtype(timestamp_ns.dtype, np.integer):
        raise ValueError(f"Expected integer nanosecond timestamps in {path}")
    timestamp_ns = timestamp_ns.astype(np.int64, copy=False)
    if np.any(timestamp_ns[1:] <= timestamp_ns[:-1]):
        raise ValueError(f"Timestamps in {path} must be strictly increasing")
    return timestamp_ns, timestamp_ns.astype(np.float64) / 1_000_000_000.0


def load_slam_c2w(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an interval's raw SLAM camera poses without NumPy scalar promotion."""
    with np.load(path, allow_pickle=True) as archive:
        trajectory = np.asarray(archive["traj"], dtype=np.float32)
        scale = float(np.asarray(archive["scale"]).item())
    if trajectory.ndim != 2 or trajectory.shape[1] < 7:
        raise ValueError(f"Invalid SLAM trajectory shape in {path}: {trajectory.shape}")
    translation_c2w = torch.from_numpy(trajectory[:, :3]) * scale
    quaternion = torch.from_numpy(trajectory[:, 3:7])[:, [3, 0, 1, 2]]
    real, i, j, k = torch.unbind(quaternion, -1)
    two_s = 2.0 / (quaternion * quaternion).sum(-1)
    rotation_c2w = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * real),
            two_s * (i * k + j * real),
            two_s * (i * j + k * real),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * real),
            two_s * (i * k - j * real),
            two_s * (j * k + i * real),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    return rotation_c2w, translation_c2w


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_frontend_provenance(
    sequence_root: Path,
    frontend_result_dir: Path,
    track_path: Path,
    track_sha256: str,
    chunk_paths: list[Path],
    allow_gt_boxes: bool,
    legacy_gt_boxes: bool,
) -> dict:
    """Verify the manifest binds the exact track and HaWoR chunk files consumed."""
    manifest_path = frontend_result_dir / "frontend_provenance.json"
    track_relative = track_path.relative_to(sequence_root).as_posix()
    actual_chunks = {
        path.relative_to(frontend_result_dir).as_posix(): _sha256(path)
        for path in chunk_paths
    }
    if not manifest_path.is_file():
        if not allow_gt_boxes:
            raise ValueError(
                f"Missing frontend provenance manifest: {manifest_path}. "
                "HaWoR chunk origin is unknown; provide a verified manifest or "
                "pass allow_gt_boxes=True for an explicitly oracle-assisted/unknown run.")
        return {
            "manifest": str(manifest_path),
            "manifest_status": "missing_unknown",
            "oracle_assisted": True,
            "oracle_assisted_status": "unknown",
            "chunk_sha256": actual_chunks,
        }

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        if not allow_gt_boxes:
            raise ValueError(f"Invalid frontend provenance manifest: {manifest_path}") from exc
        return {
            "manifest": str(manifest_path),
            "manifest_status": "invalid_unknown",
            "oracle_assisted": True,
            "oracle_assisted_status": "unknown",
            "chunk_sha256": actual_chunks,
        }

    if not isinstance(manifest, dict):
        raise ValueError(f"Frontend provenance manifest must be a JSON object: {manifest_path}")
    expected_chunks = manifest.get("chunk_sha256")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("frontend") != "HaWoR"
        or manifest.get("track_file") != track_relative
        or manifest.get("track_sha256") != track_sha256
        or not isinstance(expected_chunks, dict)
        or expected_chunks != actual_chunks
        or not isinstance(manifest.get("oracle_assisted"), bool)
    ):
        raise ValueError(
            f"Frontend provenance does not match the exact track/chunk inputs in "
            f"{manifest_path}; regenerate the frontend with a matching manifest.")

    oracle_assisted = bool(manifest["oracle_assisted"] or legacy_gt_boxes)
    if oracle_assisted and not allow_gt_boxes:
        raise ValueError(
            f"Frontend provenance marks {frontend_result_dir} as oracle-assisted; "
            "pass allow_gt_boxes=True only for an explicitly assisted run.")
    return {
        "manifest": str(manifest_path),
        "manifest_status": "verified",
        "oracle_assisted": oracle_assisted,
        "oracle_assisted_status": (
            "known_oracle_assisted" if oracle_assisted else "verified_no_oracle"),
        "chunk_sha256": actual_chunks,
    }


def export_hawor_frontend(
    sequence_root: Path,
    frontend_result_dir: Path,
    allow_gt_boxes: bool = False,
    return_provenance: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict]:
    """Convert HaWoR chunk JSON files into the WiLoR/HMP compact NPZ schema."""
    image_paths = sorted((sequence_root / "extracted_images").glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No extracted images under {sequence_root}")
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise ValueError(f"Cannot read {image_paths[0]}")
    height, width = first_image.shape[:2]
    arrays = _empty_predictions(
        len(image_paths), np.asarray([width, height], dtype=np.int32))
    with (sequence_root / "focal.txt").open("r", encoding="utf-8") as handle:
        focal = float(handle.read())

    track_path, oracle_assisted = _resolve_tracks_path(sequence_root, allow_gt_boxes)
    tracks = _load_right_hand_tracks(track_path)
    if len(tracks) != len(image_paths):
        raise ValueError(
            f"Track/image frame mismatch in {sequence_root}: "
            f"{len(tracks)} tracks vs {len(image_paths)} images")
    for frame_id, item in enumerate(tracks):
        if not isinstance(item, dict) or not item.get("det", False):
            continue
        if "det_box" not in item:
            raise ValueError(f"Track entry {frame_id} in {track_path} has no det_box")
        raw_box = np.asarray(item["det_box"], dtype=np.float32)
        if raw_box.size == 0 or raw_box.shape[-1] < 4:
            continue
        box = raw_box.reshape(-1, raw_box.shape[-1])[0, :4]
        if not np.isfinite(box).all():
            continue
        arrays["bbox"][frame_id] = box

    prediction_dir = frontend_result_dir / "1"
    chunk_paths = sorted(prediction_dir.glob("*.json"), key=lambda item: int(item.stem))
    if not chunk_paths:
        raise FileNotFoundError(
            f"No HaWoR right-hand chunk JSON files under {prediction_dir}")
    track_sha256 = _sha256(track_path)
    frontend_provenance = _validate_frontend_provenance(
        sequence_root,
        frontend_result_dir,
        track_path,
        track_sha256,
        chunk_paths,
        allow_gt_boxes,
        oracle_assisted,
    )
    for path in chunk_paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = {key: np.asarray(value, dtype=np.float32)
                       for key, value in json.load(handle).items()}
        start = int(path.stem)
        length = payload["init_trans"].shape[1]
        stop = start + length
        if stop > len(image_paths):
            raise ValueError(f"Prediction chunk exceeds video length: {path}")
        frame_slice = slice(start, stop)
        arrays["valid"][frame_slice] = True
        arrays["global_orient"][frame_slice] = payload["init_root_orient"][0, :, None]
        arrays["hand_pose"][frame_slice] = payload["init_hand_pose"][0]
        arrays["betas"][frame_slice] = payload["init_betas"][0]
        arrays["camera_translation"][frame_slice] = payload["init_trans"][0]
        arrays["focal_length"][frame_slice] = focal
    arrays["valid"] &= np.isfinite(arrays["bbox"]).all(axis=1)
    provenance = {
        "track_file": track_path.relative_to(sequence_root).as_posix(),
        "track_sha256": track_sha256,
        "bbox_source": "legacy_GT_projection_boxes" if oracle_assisted else "detected_tracks",
        **frontend_provenance,
    }
    if return_provenance:
        return arrays, provenance
    return arrays


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = sorted(set(PREDICTION_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"Missing prediction fields in {path}: {missing}")
        return {key: archive[key].copy() for key in PREDICTION_KEYS}


def save_predictions(path: Path, arrays: dict[str, np.ndarray]) -> None:
    missing = sorted(set(PREDICTION_KEYS) - set(arrays))
    if missing:
        raise ValueError(f"Cannot save predictions; missing fields: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: arrays[key] for key in PREDICTION_KEYS})


def save_trace(
    path: Path,
    frame_id: np.ndarray,
    timestamp_ns: np.ndarray,
    timestamp_sec: np.ndarray,
    segment_id: np.ndarray,
    state_world: np.ndarray,
    joints_world_m: np.ndarray,
    visual_valid: np.ndarray,
    quality: np.ndarray,
    source: np.ndarray,
    input_provenance: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_id=frame_id.astype(np.int32),
        timestamp_ns=timestamp_ns.astype(np.int64),
        timestamp_sec=timestamp_sec.astype(np.float64),
        segment_id=segment_id.astype(np.int32),
        state_world=state_world.astype(np.float32),
        joints_world_m=joints_world_m.astype(np.float32),
        visual_valid=visual_valid.astype(bool),
        quality=quality.astype(np.float32),
        source=source.astype("U16"),
        coordinate_frame=np.asarray("per-segment local raw HaWoR SLAM world"),
        position_unit=np.asarray("meter"),
        time_unit=np.asarray("second"),
        timestamp_source=np.asarray("tt_list.pkl nanoseconds"),
        segment_semantics=np.asarray(
            "SLAM interval id; coordinates from different segments are not directly comparable"),
        bbox_source=np.asarray(input_provenance.get("bbox_source", "unknown")),
        oracle_assisted=np.asarray(
            bool(input_provenance.get("oracle_assisted", False)), dtype=np.bool_),
        input_provenance_json=np.asarray(json.dumps(input_provenance, sort_keys=True)),
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

