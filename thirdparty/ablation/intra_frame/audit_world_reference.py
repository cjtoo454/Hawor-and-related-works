#!/usr/bin/env python3
"""Quantify the world-frame floor induced by HOT3D camera alignment.

This audit reads the frozen v2 calibration cache, prepared raw-SLAM cameras,
and the exported HOT3D camera poses. It does not load MANO or write labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRY = ROOT / "thirdparty" / "try"
sys.path[:0] = [str(ROOT), str(TRY), str(TRY / "scripts")]

from lib.eval_utils.custom_utils import (  # noqa: E402
    algin_cam_traj_wo_scale,
    load_gt_cam,
    load_slam_cam,
)
from lib.eval_utils.eval_utils import (  # noqa: E402
    compute_jpe,
    first_align_joints,
)


CACHE_DIR = HERE / "runs" / "feedback_v2_train" / "cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
PREPARED_DIR = TRY / "data" / "prepared_train100"
VIDEO_ROOT = ROOT / "datasets" / "hot3d_train100_export"
CALIBRATION_GT_DIR = HERE / "runs" / "calibration_events_v3" / "gt"
OUTPUT_DIR = HERE / "runs" / "world_reference_audit_v2"
ALIGNMENT_INTERVAL = 1000
MIN_CONTIGUOUS_FRAMES = 16
W_SIMILARITY_BLOCK = 100
FRAME20_PROBE = 20


class AuditError(RuntimeError):
    """Raised when a source, frame mapping, or coordinate transform is unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def torch_load_cpu(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise AuditError(f"Expected a dictionary in {path}")
    return value


def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def summarize(values: np.ndarray | list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    percentiles = np.percentile(array, [50, 75, 90, 95, 99])
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p50": float(percentiles[0]),
        "p75": float(percentiles[1]),
        "p90": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "max": float(array.max()),
    }


def finite_max(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    return float(np.max(array)) if array.size else None


def canonical_raw_world_target_from_cache(
    cached_target_joints_raw_slam: np.ndarray,
    raw_rotation_w2c: np.ndarray,
    raw_translation_w2c: np.ndarray,
    gt_rotation_c2w: np.ndarray,
    gt_translation_c2w: np.ndarray,
    align_rotation_raw_to_gt: np.ndarray,
    align_translation_raw_to_gt: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reconstruct H_G and return the fixed-alignment raw-world target A^-1 H_G.

    All arrays are CPU NumPy values. Joint arrays use (T,J,3); camera arrays
    use (T,3,3)/(T,3); A is one fixed SE(3) transform for the 1000-frame
    segment. This is reusable for supervised calibration while retaining the
    original camera-relative label S_t^-1 J_S in the returned dictionary.
    """
    joints = np.asarray(cached_target_joints_raw_slam, dtype=np.float64)
    raw_R = np.asarray(raw_rotation_w2c, dtype=np.float64)
    raw_t = np.asarray(raw_translation_w2c, dtype=np.float64)
    gt_R = np.asarray(gt_rotation_c2w, dtype=np.float64)
    gt_t = np.asarray(gt_translation_c2w, dtype=np.float64)
    A_R = np.asarray(align_rotation_raw_to_gt, dtype=np.float64)
    A_t = np.asarray(align_translation_raw_to_gt, dtype=np.float64)
    count = len(joints)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"cached joints must have shape (T,J,3), got {joints.shape}")
    if (raw_R.shape != (count, 3, 3) or gt_R.shape != (count, 3, 3)
            or raw_t.shape != (count, 3) or gt_t.shape != (count, 3)
            or A_R.shape != (3, 3) or A_t.shape != (3,)):
        raise ValueError("camera arrays must align with cached joints and one fixed SE(3) A")
    camera_relative = np.einsum("tij,tkj->tki", raw_R, joints) + raw_t[:, None, :]
    H_G = np.einsum("tij,tkj->tki", gt_R, camera_relative) + gt_t[:, None, :]
    canonical_raw = np.einsum("ij,tkj->tki", A_R.T, H_G - A_t[None, None, :])
    return {
        "camera_relative_joints_from_cache": camera_relative,
        "gt_world_joints_reconstructed": H_G,
        "canonical_raw_world_target": canonical_raw,
    }


def matrix_rotation_checks(rotation: np.ndarray, label: str) -> dict[str, float]:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.ndim != 3 or rotation.shape[-2:] != (3, 3):
        raise AuditError(f"{label}: expected rotations shaped (T,3,3), got {rotation.shape}")
    eye = np.eye(3, dtype=np.float64)
    orthogonality = np.max(np.abs(rotation @ np.swapaxes(rotation, -1, -2) - eye))
    determinants = np.linalg.det(rotation)
    determinant_error = np.max(np.abs(determinants - 1.0))
    if orthogonality > 2e-3 or determinant_error > 2e-3:
        raise AuditError(
            f"{label}: invalid camera rotation (orthogonality={orthogonality:.3g}, "
            f"determinant_error={determinant_error:.3g})")
    return {
        "max_orthogonality_abs_error": float(orthogonality),
        "max_determinant_abs_error": float(determinant_error),
    }


def load_camera_sequence_source(sequence: str) -> dict[str, Any]:
    """Load and validate one train/calibration sequence's camera/index sources.

    This labels-free helper is intended for downstream calibration tools. It
    verifies zero-based archive/image/tt_list/GT-camera indexing and the
    official 1000-frame raw-SLAM segment convention.
    """
    prepared_path = PREPARED_DIR / f"{sequence}.npz"
    sequence_root = VIDEO_ROOT / sequence
    head_path = sequence_root / "head_pose.pkl"
    extrinsics_path = sequence_root / "ego_extrinsics.pkl"
    timestamps_path = sequence_root / "tt_list.pkl"
    required = (prepared_path, head_path, extrinsics_path, timestamps_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AuditError(f"{sequence}: missing camera-map sources: {missing}")
    with np.load(prepared_path, allow_pickle=False) as archive:
        required_keys = {
            "timestamp_sec", "segment_id", "rotation_w2c", "translation_w2c", "image_paths",
        }
        missing_keys = sorted(required_keys - set(archive.files))
        if missing_keys:
            raise AuditError(f"{sequence}: prepared archive is missing {missing_keys}")
        data = {key: archive[key].copy() for key in required_keys}
    timestamps = np.asarray(data["timestamp_sec"], dtype=np.float64)
    segment_ids = np.asarray(data["segment_id"], dtype=np.int64)
    rotation_w2c = np.asarray(data["rotation_w2c"], dtype=np.float64)
    translation_w2c = np.asarray(data["translation_w2c"], dtype=np.float64)
    image_paths = np.asarray(data["image_paths"]).astype(str)
    count = len(timestamps)
    if (count < 1 or segment_ids.shape != (count,) or rotation_w2c.shape != (count, 3, 3)
            or translation_w2c.shape != (count, 3) or image_paths.shape != (count,)):
        raise AuditError(f"{sequence}: malformed prepared camera/time/index arrays")
    if not np.array_equal(segment_ids, np.arange(count, dtype=np.int64) // ALIGNMENT_INTERVAL):
        raise AuditError(f"{sequence}: segment ids do not match frame_index // 1000")
    matrix_rotation_checks(np.swapaxes(rotation_w2c, -1, -2), f"{sequence} prepared SLAM")
    if not np.isfinite(translation_w2c).all() or not np.isfinite(timestamps).all():
        raise AuditError(f"{sequence}: non-finite prepared camera/time arrays")
    with head_path.open("rb") as handle:
        head_pose = pickle.load(handle)
    with extrinsics_path.open("rb") as handle:
        ego_extrinsics = pickle.load(handle)
    if len(head_pose) != count or len(ego_extrinsics) != count:
        raise AuditError(f"{sequence}: GT camera sources do not match archive length {count}")
    with timestamps_path.open("rb") as handle:
        timestamp_ns = np.asarray(pickle.load(handle))
    if timestamp_ns.shape != (count,) or not np.issubdtype(timestamp_ns.dtype, np.integer):
        raise AuditError(f"{sequence}: tt_list shape/dtype does not match archive length {count}")
    source_times_sec = timestamp_ns.astype(np.float64) / 1_000_000_000.0
    timestamp_delta = np.abs(timestamps - source_times_sec)
    if float(timestamp_delta.max(initial=0.0)) > 1e-7:
        raise AuditError(f"{sequence}: archive frame/timestamp mapping differs from tt_list")
    probe = min(FRAME20_PROBE, count - 1)
    image_probe = Path(image_paths[probe])
    if image_probe.stem != f"{probe:06d}" or not image_probe.is_file():
        raise AuditError(f"{sequence}: frame-{probe} image does not match zero-based frame id")
    gt_probe = tuple(as_numpy(value)[0] for value in load_gt_cam(
        str(VIDEO_ROOT), sequence, probe, probe + 1))
    gt_full = tuple(as_numpy(value) for value in load_gt_cam(
        str(VIDEO_ROOT), sequence, 0, count))
    probe_delta = max(float(np.max(np.abs(part - full[probe])))
                      for part, full in zip(gt_probe, gt_full))
    if probe_delta > 1e-7:
        raise AuditError(f"{sequence}: load_gt_cam(start={probe}) does not select frame {probe}")
    source_paths = {
        "head_pose.pkl": str(head_path.relative_to(ROOT)),
        "ego_extrinsics.pkl": str(extrinsics_path.relative_to(ROOT)),
        "tt_list.pkl": str(timestamps_path.relative_to(ROOT)),
        "prepared_archive.npz": str(prepared_path.relative_to(ROOT)),
    }
    source_hashes = {key: sha256_file(ROOT / value) for key, value in source_paths.items()}
    return {
        "sequence": sequence,
        "count": count,
        "timestamps": timestamps,
        "segment_ids": segment_ids,
        "rotation_w2c": rotation_w2c,
        "translation_w2c": translation_w2c,
        "image_paths": image_paths,
        "gt_camera_full": gt_full,
        "source_paths": source_paths,
        "source_hashes": source_hashes,
        "indexing_check": {
            "probe_frame_index": int(probe),
            "probe_image": str(image_probe.relative_to(ROOT)),
            "prepared_to_tt_list_max_abs_sec": float(timestamp_delta.max(initial=0.0)),
            "load_gt_cam_single_frame_vs_full_max_abs": probe_delta,
            "frame_index_rule": "archive frame i == extracted_images/i.jpg == camera array i",
            "alignment_segment_rule": "segment_id == frame_index // 1000",
        },
    }


def _read_sequence_sources(sequence: str) -> dict[str, Any]:
    prepared_path = PREPARED_DIR / f"{sequence}.npz"
    sequence_root = VIDEO_ROOT / sequence
    head_path = sequence_root / "head_pose.pkl"
    extrinsics_path = sequence_root / "ego_extrinsics.pkl"
    timestamps_path = sequence_root / "tt_list.pkl"
    required = (prepared_path, head_path, extrinsics_path, timestamps_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AuditError(f"{sequence}: missing required calibration sources: {missing}")

    with np.load(prepared_path, allow_pickle=False) as archive:
        required_keys = {
            "timestamp_sec", "segment_id", "rotation_w2c", "translation_w2c", "image_paths",
            "gt_valid", "visual_valid",
        }
        missing_keys = sorted(required_keys - set(archive.files))
        if missing_keys:
            raise AuditError(f"{sequence}: prepared archive is missing {missing_keys}")
        data = {key: archive[key].copy() for key in required_keys}

    timestamps = np.asarray(data["timestamp_sec"], dtype=np.float64)
    segment_ids = np.asarray(data["segment_id"], dtype=np.int64)
    rotation_w2c = np.asarray(data["rotation_w2c"], dtype=np.float64)
    translation_w2c = np.asarray(data["translation_w2c"], dtype=np.float64)
    image_paths = np.asarray(data["image_paths"]).astype(str)
    archive_gt_valid = np.asarray(data["gt_valid"], dtype=bool)
    archive_visual_valid = np.asarray(data["visual_valid"], dtype=bool)
    count = len(timestamps)
    if (count < 1 or segment_ids.shape != (count,) or rotation_w2c.shape != (count, 3, 3)
            or translation_w2c.shape != (count, 3) or image_paths.shape != (count,)
            or archive_gt_valid.shape != (count,) or archive_visual_valid.shape != (count,)):
        raise AuditError(f"{sequence}: malformed prepared camera/time/index arrays")
    expected_segment_ids = np.arange(count, dtype=np.int64) // ALIGNMENT_INTERVAL
    if not np.array_equal(segment_ids, expected_segment_ids):
        mismatch = np.flatnonzero(segment_ids != expected_segment_ids)[:8].tolist()
        raise AuditError(
            f"{sequence}: prepared segments do not match official 1000-frame indexing; "
            f"first mismatches={mismatch}")
    matrix_rotation_checks(np.swapaxes(rotation_w2c, -1, -2), f"{sequence} prepared SLAM")
    if not np.isfinite(translation_w2c).all() or not np.isfinite(timestamps).all():
        raise AuditError(f"{sequence}: non-finite prepared translations or timestamps")

    with head_path.open("rb") as handle:
        head_pose = pickle.load(handle)
    with extrinsics_path.open("rb") as handle:
        ego_extrinsics = pickle.load(handle)
    camera_lengths = [len(head_pose), len(ego_extrinsics)]
    if camera_lengths != [count, count]:
        raise AuditError(
            f"{sequence}: GT camera source length mismatch: archive={count}, "
            f"head_pose/ego_extrinsics={camera_lengths}")

    reference_path = CALIBRATION_GT_DIR / f"{sequence}.npz"
    if not reference_path.is_file():
        raise AuditError(f"{sequence}: missing independent calibration GT reference {reference_path}")
    with np.load(reference_path, allow_pickle=False) as reference:
        reference_keys = {
            "sequence", "split", "coordinate_frame", "frame_ids", "timestamp_sec",
            "segment_id", "gt_valid", "gt_joints_world_m",
        }
        missing_reference = sorted(reference_keys - set(reference.files))
        if missing_reference:
            raise AuditError(f"{sequence}: independent GT reference is missing {missing_reference}")
        reference_data = {key: reference[key].copy() for key in reference_keys}
    if str(reference_data["sequence"].item()) != sequence or str(reference_data["split"].item()) != "calibration":
        raise AuditError(f"{sequence}: independent GT reference identity/split mismatch")
    if str(reference_data["coordinate_frame"].item()) != "per-interval local raw HaWoR SLAM world":
        raise AuditError(f"{sequence}: independent GT reference has an unexpected coordinate frame")
    reference_frames = np.asarray(reference_data["frame_ids"], dtype=np.int64)
    reference_timestamps = np.asarray(reference_data["timestamp_sec"], dtype=np.float64)
    reference_segments = np.asarray(reference_data["segment_id"], dtype=np.int64)
    reference_gt_valid = np.asarray(reference_data["gt_valid"], dtype=bool)
    gt_joints_world = np.asarray(reference_data["gt_joints_world_m"], dtype=np.float64)
    if (reference_frames.shape != (count,) or not np.array_equal(reference_frames, np.arange(count))
            or reference_timestamps.shape != (count,) or reference_segments.shape != (count,)
            or reference_gt_valid.shape != (count,) or gt_joints_world.shape != (count, 21, 3)):
        raise AuditError(f"{sequence}: independent GT frame array shapes/indices are not aligned")
    if (not np.array_equal(reference_segments, segment_ids)
            or not np.array_equal(reference_gt_valid, archive_gt_valid)
            or not np.allclose(reference_timestamps, timestamps, rtol=0.0, atol=1e-7)):
        raise AuditError(f"{sequence}: independent GT reference is misindexed against prepared archive")
    if not np.isfinite(gt_joints_world[reference_gt_valid]).all():
        raise AuditError(f"{sequence}: reference marks non-finite world joints as GT-valid")
    with timestamps_path.open("rb") as handle:
        timestamp_ns = np.asarray(pickle.load(handle))
    if timestamp_ns.shape != (count,) or not np.issubdtype(timestamp_ns.dtype, np.integer):
        raise AuditError(
            f"{sequence}: tt_list must provide {count} integer timestamps; got "
            f"{timestamp_ns.shape} {timestamp_ns.dtype}")
    source_times_sec = timestamp_ns.astype(np.float64) / 1_000_000_000.0
    timestamp_delta = np.abs(timestamps - source_times_sec)
    if float(timestamp_delta.max(initial=0.0)) > 1e-7:
        bad = np.flatnonzero(timestamp_delta > 1e-7)[:8].tolist()
        raise AuditError(
            f"{sequence}: prepared frame/time mapping disagrees with tt_list at {bad}")

    probe = min(FRAME20_PROBE, count - 1)
    image_probe = Path(image_paths[probe])
    if image_probe.stem != f"{probe:06d}" or not image_probe.is_file():
        raise AuditError(
            f"{sequence}: frame-{probe} image path does not map to the same zero-based id: "
            f"{image_probe}")
    gt_probe = load_gt_cam(str(VIDEO_ROOT), sequence, probe, probe + 1)
    gt_full = load_gt_cam(str(VIDEO_ROOT), sequence, 0, count)
    gt_probe_values = [as_numpy(value)[0] for value in gt_probe]
    gt_full_values = [as_numpy(value)[probe] for value in gt_full]
    probe_camera_delta = max(
        float(np.max(np.abs(first - second)))
        for first, second in zip(gt_probe_values, gt_full_values)
    )
    if probe_camera_delta > 1e-7:
        raise AuditError(
            f"{sequence}: load_gt_cam(start={probe}) does not select archive frame {probe}; "
            f"max delta={probe_camera_delta:.3g}")

    return {
        "sequence": sequence,
        "count": count,
        "timestamps": timestamps,
        "segment_ids": segment_ids,
        "rotation_w2c": rotation_w2c,
        "translation_w2c": translation_w2c,
        "image_paths": image_paths,
        "gt_valid": archive_gt_valid,
        "visual_valid": archive_visual_valid,
        "independent_joints_raw_slam": gt_joints_world,
        "gt_camera_full": tuple(as_numpy(value) for value in gt_full),
        "indexing_check": {
            "probe_frame_index": int(probe),
            "probe_image": str(image_probe.relative_to(ROOT)),
            "probe_timestamp_sec": float(timestamps[probe]),
            "tt_list_timestamp_sec": float(source_times_sec[probe]),
            "prepared_to_tt_list_max_abs_sec": float(timestamp_delta.max(initial=0.0)),
            "load_gt_cam_single_frame_vs_full_max_abs": probe_camera_delta,
            "head_pose_length": int(camera_lengths[0]),
            "ego_extrinsics_length": int(camera_lengths[1]),
            "frame_index_rule": "archive frame i == extracted_images/i.jpg == camera array i",
            "alignment_segment_rule": "segment_id == frame_index // 1000",
        },
        "source_hashes": {
            "head_pose.pkl": sha256_file(head_path),
            "ego_extrinsics.pkl": sha256_file(extrinsics_path),
            "tt_list.pkl": sha256_file(timestamps_path),
            "prepared_archive.npz": sha256_file(prepared_path),
            "independent_gt_reference.npz": sha256_file(reference_path),
        },
        "source_paths": {
            "head_pose.pkl": str(head_path.relative_to(ROOT)),
            "ego_extrinsics.pkl": str(extrinsics_path.relative_to(ROOT)),
            "tt_list.pkl": str(timestamps_path.relative_to(ROOT)),
            "prepared_archive.npz": str(prepared_path.relative_to(ROOT)),
            "independent_gt_reference.npz": str(reference_path.relative_to(ROOT)),
        },
    }


def _load_segment_alignment(sequence_data: dict[str, Any], segment_id: int) -> dict[str, Any]:
    sequence = sequence_data["sequence"]
    count = int(sequence_data["count"])
    start = int(segment_id) * ALIGNMENT_INTERVAL
    stop = min(start + ALIGNMENT_INTERVAL, count)
    if start >= stop:
        raise AuditError(f"{sequence}: segment {segment_id} has no archive frames")
    expected = np.arange(start, stop, dtype=np.int64)
    if not np.all(sequence_data["segment_ids"][start:stop] == segment_id):
        raise AuditError(f"{sequence}: frame range [{start},{stop}) crosses a segment boundary")

    slam_path = (
        VIDEO_ROOT / sequence / "preprocess" / "SLAM" /
        f"hawor_slam_w_scale_{start}_{stop}.npz"
    )
    if not slam_path.is_file():
        raise AuditError(f"{sequence}: missing official raw SLAM camera chunk {slam_path}")
    R_w2c_slam, t_w2c_slam, R_c2w_slam, t_c2w_slam = load_slam_cam(str(slam_path))
    R_w2c_slam = as_numpy(R_w2c_slam).astype(np.float64)
    t_w2c_slam = as_numpy(t_w2c_slam).astype(np.float64)
    R_c2w_slam = as_numpy(R_c2w_slam).astype(np.float64)
    t_c2w_slam = as_numpy(t_c2w_slam).astype(np.float64)
    expected_length = stop - start
    if any(len(value) != expected_length for value in (
            R_w2c_slam, t_w2c_slam, R_c2w_slam, t_c2w_slam)):
        raise AuditError(
            f"{sequence}: raw SLAM chunk {slam_path.name} length is not {expected_length}")

    archive_R_w2c = sequence_data["rotation_w2c"][start:stop]
    archive_t_w2c = sequence_data["translation_w2c"][start:stop]
    prepared_camera_delta = max(
        float(np.max(np.abs(R_w2c_slam - archive_R_w2c))),
        float(np.max(np.abs(t_w2c_slam - archive_t_w2c))),
    )
    if prepared_camera_delta > 2e-4:
        raise AuditError(
            f"{sequence}: cache/prepared camera indexing differs from official SLAM file "
            f"on [{start},{stop}); max abs delta={prepared_camera_delta:.3g}")

    R_w2c_gt_t, t_w2c_gt_t, R_c2w_gt_t, t_c2w_gt_t = load_gt_cam(
        str(VIDEO_ROOT), sequence, start, stop)
    R_w2c_gt = as_numpy(R_w2c_gt_t).astype(np.float64)
    t_w2c_gt = as_numpy(t_w2c_gt_t).astype(np.float64)
    R_c2w_gt = as_numpy(R_c2w_gt_t).astype(np.float64)
    t_c2w_gt = as_numpy(t_c2w_gt_t).astype(np.float64)
    if len(R_c2w_gt) != expected_length or len(t_c2w_gt) != expected_length:
        raise AuditError(
            f"{sequence}: GT camera slice [{start},{stop}) has wrong length")

    # This is the same no-scale alignment used by the formal HOT3D evaluator.
    _, _, R_c2w_aligned_t, t_c2w_aligned_t = algin_cam_traj_wo_scale(
        torch.as_tensor(R_c2w_slam, dtype=torch.float32),
        torch.as_tensor(t_c2w_slam, dtype=torch.float32),
        torch.as_tensor(R_c2w_gt, dtype=torch.float32),
        torch.as_tensor(t_c2w_gt, dtype=torch.float32),
    )
    R_c2w_aligned = as_numpy(R_c2w_aligned_t).astype(np.float64)
    t_c2w_aligned = as_numpy(t_c2w_aligned_t).astype(np.float64)
    # Recover and verify the fixed A=(R,t) applied by algin_cam_traj_wo_scale.
    A_R = R_c2w_aligned[0] @ R_c2w_slam[0].T
    A_t = t_c2w_aligned[0] - A_R @ t_c2w_slam[0]
    rotation_residual = float(np.max(np.abs(R_c2w_aligned - A_R[None] @ R_c2w_slam)))
    translation_residual = float(np.max(np.abs(
        t_c2w_aligned - (A_R[None] @ t_c2w_slam[..., None])[..., 0] - A_t[None]
    )))
    if rotation_residual > 2e-5 or translation_residual > 2e-5:
        raise AuditError(
            f"{sequence} segment {segment_id}: alignment is not one fixed SE(3); "
            f"rotation residual={rotation_residual:.3g}, "
            f"translation residual={translation_residual:.3g}")
    matrix_rotation_checks(A_R[None], f"{sequence} segment {segment_id} alignment A")
    matrix_rotation_checks(R_c2w_gt, f"{sequence} segment {segment_id} GT camera")

    aligned_camera_center_error = np.linalg.norm(t_c2w_aligned - t_c2w_gt, axis=-1) * 1000.0
    relative_rotation = R_c2w_gt @ np.swapaxes(R_c2w_aligned, -1, -2)
    cos_angle = np.clip((np.trace(relative_rotation, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    camera_rotation_error = np.rad2deg(np.arccos(cos_angle))

    return {
        "sequence": sequence,
        "segment_id": int(segment_id),
        "frame_start": start,
        "frame_stop_exclusive": stop,
        "frame_indices": expected,
        "R_w2c_slam": R_w2c_slam,
        "t_w2c_slam": t_w2c_slam,
        "R_c2w_slam": R_c2w_slam,
        "t_c2w_slam": t_c2w_slam,
        "R_c2w_gt": R_c2w_gt,
        "t_c2w_gt": t_c2w_gt,
        "R_w2c_gt": R_w2c_gt,
        "t_w2c_gt": t_w2c_gt,
        "R_c2w_aligned": R_c2w_aligned,
        "t_c2w_aligned": t_c2w_aligned,
        "A_R": A_R,
        "A_t": A_t,
        "record": {
            "sequence": sequence,
            "segment_id": int(segment_id),
            "frame_start": start,
            "frame_stop_exclusive": stop,
            "alignment_frames": int(expected_length),
            "A_convention": "x_gt_world = A_R @ x_raw_slam_world + A_t; scale=1",
            "A_R": A_R.tolist(),
            "A_t_m": A_t.tolist(),
            "raw_slam_source": str(slam_path.relative_to(ROOT)),
            "raw_slam_sha256": sha256_file(slam_path),
            "gt_camera_source_mapping": {
                "head_pose_path": sequence_data["source_paths"]["head_pose.pkl"],
                "head_pose_sha256": sequence_data["source_hashes"]["head_pose.pkl"],
                "ego_extrinsics_path": sequence_data["source_paths"]["ego_extrinsics.pkl"],
                "ego_extrinsics_sha256": sequence_data["source_hashes"]["ego_extrinsics.pkl"],
                "custom_utils_loader": "lib.eval_utils.custom_utils.load_gt_cam",
                "frame_slice_start": start,
                "frame_slice_stop_exclusive": stop,
                "rotation_c2w_shape": list(R_c2w_gt.shape),
                "translation_c2w_shape": list(t_c2w_gt.shape),
                "resolved_rotation_c2w_sha256": sha256_array(R_c2w_gt),
                "resolved_translation_c2w_sha256": sha256_array(t_c2w_gt),
                "axis_convention": "custom_utils.load_gt_cam including its HOT3D R_90 correction",
            },
            "cache_prepared_vs_official_raw_camera_max_abs": prepared_camera_delta,
            "fixed_A_rotation_reconstruction_max_abs": rotation_residual,
            "fixed_A_translation_reconstruction_max_abs": translation_residual,
            "aligned_camera_center_error_mm": summarize(aligned_camera_center_error),
            "aligned_camera_rotation_error_deg": summarize(camera_rotation_error),
        },
    }


def load_formal_segment_alignment(sequence: str, segment_id: int) -> dict[str, Any]:
    """Public CPU helper returning official GT camera slices and fixed segment A.

    Use this with :func:`canonical_raw_world_target_from_cache` for any
    train/calibration archive row. The returned `A_R`/`A_t` map raw SLAM world
    into the formal GT world; all camera arrays and provenance are included.
    """
    source = load_camera_sequence_source(sequence)
    alignment = _load_segment_alignment(source, int(segment_id))
    return {
        "sequence": sequence,
        "segment_id": int(segment_id),
        "frame_start": alignment["frame_start"],
        "frame_stop_exclusive": alignment["frame_stop_exclusive"],
        "R_w2c_raw_slam": alignment["R_w2c_slam"],
        "t_w2c_raw_slam": alignment["t_w2c_slam"],
        "R_c2w_gt": alignment["R_c2w_gt"],
        "t_c2w_gt": alignment["t_c2w_gt"],
        "R_w2c_gt": alignment["R_w2c_gt"],
        "t_w2c_gt": alignment["t_w2c_gt"],
        "R_c2w_aligned_slam": alignment["R_c2w_aligned"],
        "t_c2w_aligned_slam": alignment["t_c2w_aligned"],
        "A_R": alignment["A_R"],
        "A_t": alignment["A_t"],
        "provenance": alignment["record"],
        "indexing_check": source["indexing_check"],
        "source_paths": source["source_paths"],
        "source_hashes": source["source_hashes"],
    }


def _continuous_valid_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(start), int(stop)) for start, stop in zip(edges[::2], edges[1::2])]


def _first2_blocks(run_positions: np.ndarray) -> list[np.ndarray]:
    blocks = [run_positions[start:start + W_SIMILARITY_BLOCK]
              for start in range(0, len(run_positions), W_SIMILARITY_BLOCK)]
    if len(blocks) > 1 and len(blocks[-1]) == 1:
        blocks[-1] = np.r_[blocks[-2][-1:], blocks[-1]]
        blocks[-2] = blocks[-2][:-1]
    if any(len(block) < 2 for block in blocks):
        raise AuditError("first2 similarity alignment requires at least two frames per block")
    return blocks


def _score_window(
    window: dict[str, Any],
    manifest_items: dict[str, dict[str, Any]],
    sequence_data: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    sequence = str(window["sequence"])
    frame_ids = np.asarray(window["frame_indices"], dtype=np.int64)
    if len(frame_ids) != 128 or not np.array_equal(frame_ids, np.arange(frame_ids[0], frame_ids[-1] + 1)):
        raise AuditError(f"{sequence} window {window['window_index']}: not a fixed 128-frame run")
    expected_segment = int(window["segment_id"])
    if not np.all(sequence_data["segment_ids"][frame_ids] == expected_segment):
        raise AuditError(f"{sequence}: window frame/segment id mismatch")

    rows: dict[int, dict[str, Any]] = {}
    used_cache_hashes: dict[str, dict[str, str]] = {}
    for item_id in window["context_item_ids"]:
        item = manifest_items.get(str(item_id))
        if item is None:
            raise AuditError(f"{sequence}: window references unknown cache row {item_id}")
        if item["split"] != "calibration" or item["sequence"] != sequence:
            raise AuditError(f"{sequence}: non-calibration or cross-sequence row {item_id}")
        input_path = CACHE_DIR / item["inputs_path"]
        label_path = CACHE_DIR / item["labels_path"]
        if sha256_file(input_path) != item["inputs_sha256"]:
            raise AuditError(f"Cache input SHA mismatch: {item_id}")
        if sha256_file(label_path) != item["labels_sha256"]:
            raise AuditError(f"Cache label SHA mismatch: {item_id}")
        inputs = torch_load_cpu(input_path)
        labels = torch_load_cpu(label_path)
        labels_frame_ids = as_numpy(labels["frame_indices"]).astype(np.int64)
        if labels_frame_ids.tolist() != item["frame_indices"]:
            raise AuditError(f"{item_id}: label frame indices differ from cache manifest")
        label_keys = (
            "target_joints", "gt_valid", "visual_valid", "timestamp_sec", "segment_id",
        )
        if any(key not in labels for key in label_keys):
            raise AuditError(f"{item_id}: cached labels are missing a required field")
        rotation_rows = as_numpy(inputs["rotation_w2c"])
        translation_rows = as_numpy(inputs["translation_w2c"])
        if len(rotation_rows) != len(labels_frame_ids) or len(translation_rows) != len(labels_frame_ids):
            raise AuditError(
                f"{item_id}: cached input and label row counts are not one-to-one")
        for label_row, frame in enumerate(labels_frame_ids):
            if int(frame) not in set(frame_ids.tolist()):
                continue
            if int(labels["segment_id"][label_row]) != expected_segment:
                raise AuditError(f"{item_id}: frame {frame} has a mismatched label segment")
            target = as_numpy(labels["target_joints"])[label_row].astype(np.float64)
            gt_valid = bool(labels["gt_valid"][label_row])
            visual_valid = bool(labels["visual_valid"][label_row])
            timestamp = float(labels["timestamp_sec"][label_row])
            if gt_valid != bool(sequence_data["gt_valid"][int(frame)]):
                raise AuditError(f"{item_id}: frame {frame} cache/archive GT validity differs")
            if visual_valid != bool(sequence_data["visual_valid"][int(frame)]):
                raise AuditError(f"{item_id}: frame {frame} cache/archive visual validity differs")
            if gt_valid and not np.isfinite(target).all():
                raise AuditError(f"{item_id}: GT-valid frame {frame} has non-finite target joints")
            archive_R = sequence_data["rotation_w2c"][int(frame)]
            archive_t = sequence_data["translation_w2c"][int(frame)]
            # v2_cache stacks one model-input row for each sorted label frame;
            # rows are not the full 16-frame context dimension.
            input_row = int(label_row)
            input_R = np.asarray(rotation_rows[input_row], dtype=np.float64)
            input_t = np.asarray(translation_rows[input_row], dtype=np.float64)
            cache_camera_delta = max(
                float(np.max(np.abs(input_R - archive_R))),
                float(np.max(np.abs(input_t - archive_t))),
            )
            if cache_camera_delta > 2e-4:
                raise AuditError(
                    f"{item_id}: frame {frame} cached camera differs from prepared archive "
                    f"by {cache_camera_delta:.3g}")
            if abs(timestamp - sequence_data["timestamps"][int(frame)]) > 1e-7:
                raise AuditError(f"{item_id}: frame {frame} cached timestamp has an index mismatch")
            image_path = Path(sequence_data["image_paths"][int(frame)])
            if image_path.stem != f"{int(frame):06d}" or not image_path.is_file():
                raise AuditError(f"{item_id}: frame {frame} does not map to its exported image")
            if int(frame) in rows:
                raise AuditError(f"{sequence}: duplicated cached label row for frame {frame}")
            rows[int(frame)] = {
                "target_joints": target,
                "gt_valid": gt_valid,
                "visual_valid": visual_valid,
                "timestamp_sec": timestamp,
                "rotation_w2c": input_R,
                "translation_w2c": input_t,
                "cache_camera_max_abs_delta": cache_camera_delta,
                "image_path": str(image_path.relative_to(ROOT)),
                "row_id": {
                    "frame_index": int(frame),
                    "cache_item_id": str(item_id),
                    "cache_context_start": int(item["context_start"]),
                    "cache_input_row": input_row,
                    "cache_label_row": int(label_row),
                },
            }
        used_cache_hashes[str(item_id)] = {
            "inputs_sha256": item["inputs_sha256"],
            "labels_sha256": item["labels_sha256"],
        }

    if sorted(rows) != frame_ids.tolist():
        missing = sorted(set(frame_ids.tolist()) - set(rows))[:16]
        raise AuditError(f"{sequence}: cache rows do not cover every fixed window frame: {missing}")

    # J_S is in the per-interval raw SLAM world. Reconstruct H_G with the
    # requested H_G = G_t S_t^-1 J_S formula, then form G^-1 H_G and evaluate
    # it with the aligned SLAM camera. The independent raw-world reference is
    # used only to verify that the cached J_S values themselves match.
    R_s = np.stack([rows[int(frame)]["rotation_w2c"] for frame in frame_ids])
    t_s = np.stack([rows[int(frame)]["translation_w2c"] for frame in frame_ids])
    J_s = np.stack([rows[int(frame)]["target_joints"] for frame in frame_ids])
    valid = np.asarray([
        rows[int(frame)]["gt_valid"] and rows[int(frame)]["visual_valid"]
        for frame in frame_ids
    ], dtype=bool)
    independent_J_s = sequence_data["independent_joints_raw_slam"][frame_ids]
    valid &= np.isfinite(J_s).all(axis=(1, 2))

    local_offset = frame_ids - alignment["frame_start"]
    if (local_offset.min() < 0 or local_offset.max() >= len(alignment["R_c2w_gt"])):
        raise AuditError(f"{sequence}: calibration window falls outside aligned camera segment")
    R_g = alignment["R_c2w_gt"][local_offset]
    t_g = alignment["t_c2w_gt"][local_offset]
    R_g_w2c = alignment["R_w2c_gt"][local_offset]
    t_g_w2c = alignment["t_w2c_gt"][local_offset]
    R_a = alignment["R_c2w_aligned"][local_offset]
    t_a = alignment["t_c2w_aligned"][local_offset]
    R_s_c2w = np.swapaxes(R_s, -1, -2)
    t_s_c2w = -np.einsum("tij,tj->ti", R_s_c2w, t_s)

    cached_camera_joints = np.full_like(J_s, np.nan, dtype=np.float64)
    perfect_camera_joints = np.full_like(J_s, np.nan, dtype=np.float64)
    reconstructed_J_s = np.full_like(J_s, np.nan, dtype=np.float64)
    reconstructed_H_g_from_cached_label = np.full_like(J_s, np.nan, dtype=np.float64)
    formal_world = np.full_like(J_s, np.nan, dtype=np.float64)
    fixed_world_label = np.full_like(J_s, np.nan, dtype=np.float64)
    R_A = alignment["A_R"]
    t_A = alignment["A_t"]
    if valid.any():
        reconstructed = canonical_raw_world_target_from_cache(
            J_s[valid], R_s[valid], t_s[valid], R_g[valid], t_g[valid], R_A, t_A)
        cached_camera_joints[valid] = reconstructed["camera_relative_joints_from_cache"]
        reconstructed_H_g_from_cached_label[valid] = reconstructed["gt_world_joints_reconstructed"]
        fixed_world_label[valid] = reconstructed["canonical_raw_world_target"]
    perfect_camera_joints[valid] = (
        np.einsum("tij,tkj->tki", R_g_w2c[valid], reconstructed_H_g_from_cached_label[valid])
        + t_g_w2c[valid, None, :]
    )
    reconstructed_J_s[valid] = (
        np.einsum("tij,tkj->tki", R_s_c2w[valid], cached_camera_joints[valid])
        + t_s_c2w[valid, None, :]
    )
    formal_world[valid] = (
        np.einsum("tij,tkj->tki", R_a[valid], perfect_camera_joints[valid])
        + t_a[valid, None, :]
    )

    roundtrip_camera_error = np.linalg.norm(
        reconstructed_J_s[valid] - J_s[valid], axis=-1)
    cached_camera_label_error = np.linalg.norm(
        cached_camera_joints[valid] - perfect_camera_joints[valid], axis=-1) * 1000.0
    independent_rawworld_reference_error = np.linalg.norm(
        J_s[valid] - independent_J_s[valid], axis=-1) * 1000.0
    formal_A_error = np.linalg.norm(
        formal_world[valid] - (
            np.einsum("ij,tkj->tki", R_A, J_s[valid]) + t_A[None, None, :]
        ), axis=-1)
    direct_joint_mm = np.linalg.norm(
        formal_world[valid] - reconstructed_H_g_from_cached_label[valid], axis=-1) * 1000.0
    fixed_label_delta_mm = np.linalg.norm(J_s[valid] - fixed_world_label[valid], axis=-1) * 1000.0

    score_mask = np.zeros(len(frame_ids), dtype=bool)
    for run_start, run_stop in _continuous_valid_runs(valid):
        if run_stop - run_start >= MIN_CONTIGUOUS_FRAMES:
            score_mask[run_start:run_stop] = True
    score_positions = np.flatnonzero(score_mask)
    scored_runs = [np.arange(start, stop, dtype=np.int64)
                   for start, stop in _continuous_valid_runs(score_mask)]
    first2_frame_error = np.full(len(frame_ids), np.nan, dtype=np.float64)
    first2_blocks: list[dict[str, Any]] = []
    gt_valid_world = torch.as_tensor(reconstructed_H_g_from_cached_label, dtype=torch.float32)
    pred_world_tensor = torch.as_tensor(formal_world, dtype=torch.float32)
    for run in scored_runs:
        for block in _first2_blocks(run):
            gt_block = gt_valid_world[block]
            pred_block = pred_world_tensor[block]
            aligned_prediction = first_align_joints(gt_block, pred_block)
            errors = compute_jpe(gt_block, aligned_prediction) * 1000.0
            first2_frame_error[block] = errors
            first2_blocks.append({
                "start_frame": int(frame_ids[block[0]]),
                "stop_frame_exclusive": int(frame_ids[block[-1]] + 1),
                "frame_count": int(len(block)),
                "first_two_alignment_frames": frame_ids[block[:2]].tolist(),
                "alignment": "official first_align_joints similarity transform",
            })

    # Summaries use exactly the same contiguous-valid mask. Joint summaries
    # retain all 21 joints; frame summaries expose how the mismatch changes.
    direct_frame_mean = np.full(len(frame_ids), np.nan, dtype=np.float64)
    direct_frame_p95 = np.full(len(frame_ids), np.nan, dtype=np.float64)
    label_delta_frame_mean = np.full(len(frame_ids), np.nan, dtype=np.float64)
    label_delta_frame_p95 = np.full(len(frame_ids), np.nan, dtype=np.float64)
    if valid.any():
        direct_frame_mean[valid] = direct_joint_mm.mean(axis=1)
        direct_frame_p95[valid] = np.percentile(direct_joint_mm, 95, axis=1)
        label_delta_frame_mean[valid] = fixed_label_delta_mm.mean(axis=1)
        label_delta_frame_p95[valid] = np.percentile(fixed_label_delta_mm, 95, axis=1)
    scored_joint_selector = np.repeat(score_mask[valid, None], J_s.shape[1], axis=1).reshape(-1)
    scored_direct_joint_mm = direct_joint_mm.reshape(-1)[scored_joint_selector]
    scored_delta_joint_mm = fixed_label_delta_mm.reshape(-1)[scored_joint_selector]
    scored_direct_frames = direct_frame_mean[score_mask]
    scored_delta_frames = label_delta_frame_mean[score_mask]
    scored_first2_frames = first2_frame_error[score_mask]
    if not np.isfinite(scored_first2_frames).all():
        raise AuditError(f"{sequence}: first2 W metric did not score the common mask")

    relative_time = np.asarray([
        rows[int(frame)]["timestamp_sec"] - rows[int(frame_ids[0])]["timestamp_sec"]
        for frame in frame_ids
    ], dtype=np.float64)
    scored_positions = np.flatnonzero(score_mask)
    quarter = max(1, int(np.ceil(len(scored_positions) / 4.0)))

    time_series = []
    frame_row_ids = []
    for index, frame in enumerate(frame_ids):
        row = rows[int(frame)]
        frame_row_ids.append(row["row_id"])
        time_series.append({
            "frame_index": int(frame),
            "elapsed_sec_from_window_start": float(relative_time[index]),
            "gt_valid": bool(row["gt_valid"]),
            "visual_valid": bool(row["visual_valid"]),
            "scored_after_contiguous_min16": bool(score_mask[index]),
            "direct_world_frame_mean_mm": (
                float(direct_frame_mean[index]) if np.isfinite(direct_frame_mean[index]) else None
            ),
            "direct_world_frame_p95_mm": (
                float(direct_frame_p95[index]) if np.isfinite(direct_frame_p95[index]) else None
            ),
            "J_S_vs_fixed_A_inverse_H_G_frame_mean_mm": (
                float(label_delta_frame_mean[index])
                if np.isfinite(label_delta_frame_mean[index]) else None
            ),
            "J_S_vs_fixed_A_inverse_H_G_frame_p95_mm": (
                float(label_delta_frame_p95[index])
                if np.isfinite(label_delta_frame_p95[index]) else None
            ),
            "first2_similarity_align_W_frame_mean_mm": (
                float(first2_frame_error[index])
                if np.isfinite(first2_frame_error[index]) else None
            ),
        })

    return {
        "sequence": sequence,
        "window_index": int(window["window_index"]),
        "segment_id": expected_segment,
        "start_frame": int(window["start_frame"]),
        "stop_frame_exclusive": int(window["stop_frame_exclusive"]),
        "fixed_window_frame_count": int(len(frame_ids)),
        "gt_valid_and_visual_valid_count": int(valid.sum()),
        "scored_frame_count_after_contiguous_min16": int(score_mask.sum()),
        "scoring_status": (
            "scored" if score_mask.any()
            else "no gt_valid & visual_valid run reached 16 consecutive frames"
        ),
        "scored_frame_indices": frame_ids[score_mask].tolist(),
        "context_item_ids": list(window["context_item_ids"]),
        "cache_row_ids": frame_row_ids,
        "cache_item_hashes": used_cache_hashes,
        "same_mask_metrics": {
            "perfect_camera_relative_prediction_in_formal_world_direct_joint_error_mm":
                summarize(scored_direct_joint_mm),
            "perfect_camera_relative_prediction_in_formal_world_direct_frame_mean_mm":
                summarize(scored_direct_frames),
            "perfect_camera_relative_prediction_first2_similarity_align_W_mpjpe_mm":
                summarize(scored_first2_frames),
            "current_J_S_vs_fixed_A_inverse_H_G_joint_delta_mm":
                summarize(scored_delta_joint_mm),
            "current_J_S_vs_fixed_A_inverse_H_G_frame_mean_mm":
                summarize(scored_delta_frames),
            "cached_S_inverse_J_S_vs_G_inverse_H_G_joint_error_mm": summarize(
                cached_camera_label_error.reshape(-1)[scored_joint_selector]),
            "cached_J_S_vs_independent_raw_SLM_world_GT_joint_error_mm": summarize(
                independent_rawworld_reference_error.reshape(-1)[scored_joint_selector]),
            "formula_roundtrip_J_S_camera_world_max_mm":
                (finite_max(roundtrip_camera_error) * 1000.0
                 if roundtrip_camera_error.size else None),
            "formal_world_vs_fixed_A_times_cached_J_S_max_mm":
                (finite_max(formal_A_error) * 1000.0 if formal_A_error.size else None),
            "first2_similarity_blocks": first2_blocks,
        },
        "time_trend_same_mask": {
            "direct_world_first_quarter_frame_mean_mm": summarize(
                direct_frame_mean[scored_positions[:quarter]]),
            "direct_world_last_quarter_frame_mean_mm": summarize(
                direct_frame_mean[scored_positions[-quarter:]]),
            "direct_world_last_minus_first_quarter_mean_mm": (
                float(direct_frame_mean[scored_positions[-quarter:]].mean()
                      - direct_frame_mean[scored_positions[:quarter]].mean())
                if len(scored_positions) else None),
            "fixed_label_delta_first_quarter_frame_mean_mm": summarize(
                label_delta_frame_mean[scored_positions[:quarter]]),
            "fixed_label_delta_last_quarter_frame_mean_mm": summarize(
                label_delta_frame_mean[scored_positions[-quarter:]]),
            "fixed_label_delta_last_minus_first_quarter_mean_mm": (
                float(label_delta_frame_mean[scored_positions[-quarter:]].mean()
                      - label_delta_frame_mean[scored_positions[:quarter]].mean())
                if len(scored_positions) else None),
            "frame_series": time_series,
        },
    }


def run_camera_map(output_path: Path) -> dict[str, Any]:
    """Export labels-free formal camera/A provenance for the frozen 53 videos."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    split_by_sequence: dict[str, str] = {}
    for item in manifest.get("items", []):
        split = str(item["split"])
        if split not in {"train", "calibration"}:
            raise AuditError(f"Camera map refuses a non-train/calibration cache split: {split}")
        sequence = str(item["sequence"])
        previous = split_by_sequence.setdefault(sequence, split)
        if previous != split:
            raise AuditError(f"{sequence}: appears in multiple cache splits")
    calibration_sequences = {str(item["sequence"]) for item in manifest["calibration_windows"]}
    if calibration_sequences != {key for key, split in split_by_sequence.items() if split == "calibration"}:
        raise AuditError("Calibration windows and cache sequence splits differ")
    train_sequences = {key for key, split in split_by_sequence.items() if split == "train"}
    if len(train_sequences) != 33 or len(calibration_sequences) != 20:
        raise AuditError(
            f"Expected 33 train and 20 calibration sequences, got "
            f"{len(train_sequences)} and {len(calibration_sequences)}")

    sequence_sources: dict[str, dict[str, Any]] = {}
    alignments = []
    npz_directory = output_path.parent / "camera_map_train_calibration_53seq"
    npz_directory.mkdir(parents=True, exist_ok=True)
    camera_archives: dict[str, dict[str, Any]] = {}
    for sequence in sorted(split_by_sequence):
        source = load_camera_sequence_source(sequence)
        sequence_sources[sequence] = source
        sequence_segment_ids = np.unique(source["segment_ids"]).astype(np.int64)
        sequence_alignment_rows = []
        for segment_id in np.unique(source["segment_ids"]):
            alignment = _load_segment_alignment(source, int(segment_id))
            record = dict(alignment["record"])
            record["split"] = split_by_sequence[sequence]
            alignments.append(record)
            sequence_alignment_rows.append(alignment)
        gt_full = source["gt_camera_full"]
        npz_path = npz_directory / f"{sequence}.npz"
        tmp_path = npz_path.with_suffix(npz_path.suffix + f".{os.getpid()}.tmp")
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                sequence=np.asarray(sequence),
                split=np.asarray(split_by_sequence[sequence]),
                frame_ids=np.arange(source["count"], dtype=np.int64),
                segment_ids=source["segment_ids"].astype(np.int32),
                raw_rotation_w2c=source["rotation_w2c"].astype(np.float32),
                raw_translation_w2c=source["translation_w2c"].astype(np.float32),
                gt_rotation_c2w=gt_full[2].astype(np.float32),
                gt_translation_c2w=gt_full[3].astype(np.float32),
                gt_rotation_w2c=gt_full[0].astype(np.float32),
                gt_translation_w2c=gt_full[1].astype(np.float32),
                alignment_segment_ids=sequence_segment_ids,
                align_rotation_raw_to_gt_by_segment=np.stack(
                    [row["A_R"] for row in sequence_alignment_rows]).astype(np.float64),
                align_translation_raw_to_gt_by_segment=np.stack(
                    [row["A_t"] for row in sequence_alignment_rows]).astype(np.float64),
                alignment_frame_start=np.asarray(
                    [row["frame_start"] for row in sequence_alignment_rows], dtype=np.int64),
                alignment_frame_stop_exclusive=np.asarray(
                    [row["frame_stop_exclusive"] for row in sequence_alignment_rows], dtype=np.int64),
            )
        os.replace(tmp_path, npz_path)
        camera_archives[sequence] = {
            "path": str(npz_path.relative_to(ROOT)),
            "sha256": sha256_file(npz_path),
            "keys": [
                "frame_ids", "segment_ids", "raw_rotation_w2c", "raw_translation_w2c",
                "gt_rotation_c2w", "gt_translation_c2w", "gt_rotation_w2c",
                "gt_translation_w2c", "alignment_segment_ids",
                "align_rotation_raw_to_gt_by_segment", "align_translation_raw_to_gt_by_segment",
                "alignment_frame_start", "alignment_frame_stop_exclusive",
            ],
        }

    result = {
        "schema": "hot3d_formal_camera_map_v2",
        "status": "complete",
        "device": "cpu",
        "cache_manifest": {
            "path": str(MANIFEST_PATH.relative_to(ROOT)),
            "sha256": sha256_file(MANIFEST_PATH),
            "schema": manifest.get("schema"),
        },
        "scope": {
            "splits": ["train", "calibration"],
            "test_labels_or_test_camera_sources_read": False,
            "sequence_count": len(split_by_sequence),
            "train_sequence_count": len(train_sequences),
            "calibration_sequence_count": len(calibration_sequences),
            "alignment_interval_frames": ALIGNMENT_INTERVAL,
            "alignment_function": "lib.eval_utils.custom_utils.algin_cam_traj_wo_scale",
            "gt_camera_function": "lib.eval_utils.custom_utils.load_gt_cam",
            "per_segment_transform": "x_gt_world = A_R @ x_raw_slam_world + A_t; scale=1",
        },
        "sequences": {
            sequence: {
                "split": split_by_sequence[sequence],
                "frame_count": source["count"],
                "indexing_check": source["indexing_check"],
                "source_paths": source["source_paths"],
                "source_hashes": source["source_hashes"],
                "camera_archive": camera_archives[sequence],
                "segment_ids": np.unique(source["segment_ids"]).astype(int).tolist(),
            }
            for sequence, source in sequence_sources.items()
        },
        "segment_alignments": alignments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return result


def run_audit(mode: str, output_path: Path, sequence_filter: str | None = None) -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise AuditError(f"Missing frozen cache manifest: {MANIFEST_PATH}")
    cache_manifest_sha = sha256_file(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema") != "handloop_intra_v2_feature_cache":
        raise AuditError(f"Unexpected cache schema: {manifest.get('schema')}")
    if manifest.get("counts", {}).get("calibration_window_frames") != 5120:
        raise AuditError("Cache manifest no longer describes the frozen 5120-frame calibration set")
    windows = list(manifest["calibration_windows"])
    if len(windows) != 40 or len({item["sequence"] for item in windows}) != 20:
        raise AuditError("Expected the frozen 40 windows from exactly 20 calibration sequences")
    if mode == "quick":
        sequence_filter = sequence_filter or "P0015_179e1b84"
        matches = [window for window in windows
                   if window["sequence"] == sequence_filter and int(window["window_index"]) == 0]
        if len(matches) != 1:
            raise AuditError(f"No unique quick P0015 calibration window for {sequence_filter}")
        selected_windows = matches
    elif mode == "full":
        if sequence_filter:
            raise AuditError("--sequence is supported only in quick mode")
        selected_windows = windows
    else:
        raise AuditError(f"Unknown audit mode: {mode}")

    covered_frames = [
        (str(window["sequence"]), int(frame))
        for window in selected_windows for frame in window["frame_indices"]
    ]
    if len(covered_frames) != len(set(covered_frames)):
        raise AuditError("The selected fixed calibration windows overlap in frame ids")
    if mode == "full" and len(covered_frames) != 5120:
        raise AuditError(f"Expected 5120 fixed calibration frames, found {len(covered_frames)}")

    item_records: dict[str, dict[str, Any]] = {}
    for item in manifest["items"]:
        item_id = str(item["item_id"])
        if item_id in item_records:
            raise AuditError(f"Duplicate cache manifest item id: {item_id}")
        item_records[item_id] = item

    sequences = sorted({str(window["sequence"]) for window in selected_windows})
    sequence_sources = {sequence: _read_sequence_sources(sequence) for sequence in sequences}
    # Keep A for every 1000-frame segment in each selected calibration video,
    # including segments with no fixed window, so the JSON is reusable for
    # later calibration-window expansion without repeating camera alignment.
    alignment_keys = sorted({
        (sequence, int(segment_id))
        for sequence, source in sequence_sources.items()
        for segment_id in np.unique(source["segment_ids"])
    })
    alignments: dict[tuple[str, int], dict[str, Any]] = {}
    for sequence, segment_id in alignment_keys:
        alignments[(sequence, segment_id)] = _load_segment_alignment(
            sequence_sources[sequence], segment_id)

    window_results = []
    for window in selected_windows:
        sequence = str(window["sequence"])
        segment_id = int(window["segment_id"])
        window_results.append(_score_window(
            window,
            item_records,
            sequence_sources[sequence],
            alignments[(sequence, segment_id)],
        ))

    direct_window_means = np.asarray([
        item["same_mask_metrics"][
            "perfect_camera_relative_prediction_in_formal_world_direct_frame_mean_mm"].get("mean", np.nan)
        for item in window_results
    ], dtype=np.float64)
    first2_window_means = np.asarray([
        item["same_mask_metrics"][
            "perfect_camera_relative_prediction_first2_similarity_align_W_mpjpe_mm"].get("mean", np.nan)
        for item in window_results
    ], dtype=np.float64)
    delta_window_means = np.asarray([
        item["same_mask_metrics"]["current_J_S_vs_fixed_A_inverse_H_G_frame_mean_mm"].get("mean", np.nan)
        for item in window_results
    ], dtype=np.float64)
    scored_frames = sum(item["scored_frame_count_after_contiguous_min16"]
                        for item in window_results)
    valid_frames = sum(item["gt_valid_and_visual_valid_count"] for item in window_results)
    result = {
        "schema": "hot3d_world_reference_audit_v2",
        "status": "complete",
        "mode": mode,
        "device": "cpu",
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "cache_manifest": {
            "path": str(MANIFEST_PATH.relative_to(ROOT)),
            "sha256": cache_manifest_sha,
            "schema": manifest["schema"],
            "calibration_sequences": 20,
            "fixed_windows_in_source_manifest": 40,
            "fixed_window_frames_in_source_manifest": 5120,
        },
        "protocol": {
            "source_split": "calibration only",
            "test_labels_read": False,
            "fixed_windows_selected": len(selected_windows),
            "fixed_window_frame_count": len(covered_frames),
            "frame_indexing": "local archive/image/camera frame ids, zero-based",
            "segment_alignment_interval_frames": ALIGNMENT_INTERVAL,
            "mask": "cached gt_valid & visual_valid, then contiguous runs >= 16 frames",
            "same_scored_mask_for_all_metrics": True,
            "J_S_definition": "cached target_joints in per-interval raw SLAM world",
            "camera_relative_perfect_prediction": "G_t^-1 @ (G_t @ S_t^-1 @ J_S), algebraically checked against S_t^-1 @ J_S",
            "H_G_reconstruction": "G_t @ S_t^-1 @ J_S; independent raw-world GT file is used only to check J_S",
            "formal_prediction_world": "A @ S_t @ J_camera, with A from custom_utils.algin_cam_traj_wo_scale",
            "direct_error": "Euclidean joint distance between formal_prediction_world and reconstructed H_G",
            "first2_similarity_align_W": (
                "official first_align_joints similarity alignment from first two frames, "
                "independently per <=100-frame contiguous scored block; singleton tail is "
                "paired with the preceding frame"
            ),
            "fixed_target_comparison": "J_S versus A^-1 @ H_G, in the same segment raw SLAM world",
            "mano_loaded": False,
            "labels_or_features_modified": False,
        },
        "aggregate": {
            "sequence_count": len(sequences),
            "window_count": len(window_results),
            "fixed_frame_count": len(covered_frames),
            "gt_valid_and_visual_valid_frames": int(valid_frames),
            "scored_frames_after_contiguous_min16": int(scored_frames),
            "window_mean_direct_world_frame_mean_mm": summarize(direct_window_means),
            "window_mean_first2_similarity_align_W_mpjpe_mm": summarize(first2_window_means),
            "window_mean_current_label_delta_frame_mean_mm": summarize(delta_window_means),
            "pooled_same_mask_frame_distributions": {
                "direct_world_frame_mean_mm": summarize(np.concatenate([
                    np.asarray([row["direct_world_frame_mean_mm"] for row in item["time_trend_same_mask"]["frame_series"]
                                if row["scored_after_contiguous_min16"]], dtype=np.float64)
                    for item in window_results
                ])),
                "first2_similarity_align_W_frame_mean_mm": summarize(np.concatenate([
                    np.asarray([row["first2_similarity_align_W_frame_mean_mm"]
                                for row in item["time_trend_same_mask"]["frame_series"]
                                if row["scored_after_contiguous_min16"]], dtype=np.float64)
                    for item in window_results
                ])),
                "current_label_delta_frame_mean_mm": summarize(np.concatenate([
                    np.asarray([row["J_S_vs_fixed_A_inverse_H_G_frame_mean_mm"]
                                for row in item["time_trend_same_mask"]["frame_series"]
                                if row["scored_after_contiguous_min16"]], dtype=np.float64)
                    for item in window_results
                ])),
            },
        },
        "gt_camera_sources": {
            sequence: {
                "head_pose_path": source["source_paths"]["head_pose.pkl"],
                "head_pose.pkl": source["source_hashes"]["head_pose.pkl"],
                "ego_extrinsics_path": source["source_paths"]["ego_extrinsics.pkl"],
                "ego_extrinsics.pkl": source["source_hashes"]["ego_extrinsics.pkl"],
                "tt_list_path": source["source_paths"]["tt_list.pkl"],
                "tt_list.pkl": source["source_hashes"]["tt_list.pkl"],
                "frame_count": source["count"],
                "indexing_check": source["indexing_check"],
            }
            for sequence, source in sequence_sources.items()
        },
        "prepared_source_hashes": {
            sequence: {
                "prepared_archive.npz": source["source_hashes"]["prepared_archive.npz"],
                "independent_gt_reference.npz": source["source_hashes"]["independent_gt_reference.npz"],
                "independent_gt_reference_path": source["source_paths"]["independent_gt_reference.npz"],
            }
            for sequence, source in sequence_sources.items()
        },
        "segment_alignments": [alignments[key]["record"] for key in alignment_keys],
        "windows": window_results,
        "invalid_attempts": [
            {
                "mode": "quick P0015 window0 intermediate run",
                "invalid_direct_world_frame_mean_mm": 12.773389513445847,
                "reason": "An intermediate run misread the independent reference's raw-SLAM-world J_S as global GT H_G; that coordinate-frame mix was corrected and this value is excluded from every final metric.",
                "artifact": "quick_P0015_179e1b84_window0.json is overwritten by the corrected result",
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full", "camera-map"), default="quick")
    parser.add_argument("--sequence", help="Quick mode only; defaults to P0015_179e1b84")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "camera-map":
        if args.sequence:
            raise AuditError("--sequence is only supported in quick mode")
        output_path = args.output or OUTPUT_DIR / "camera_map_train_calibration_53seq.json"
        result = run_camera_map(output_path)
        print(json.dumps({
            "status": result["status"],
            "mode": "camera-map",
            "output": str(output_path),
            "sequences": result["scope"]["sequence_count"],
            "train_sequences": result["scope"]["train_sequence_count"],
            "calibration_sequences": result["scope"]["calibration_sequence_count"],
            "segments": len(result["segment_alignments"]),
        }, indent=2))
        return
    if args.output:
        output_path = args.output
    elif args.mode == "quick":
        sequence = args.sequence or "P0015_179e1b84"
        output_path = OUTPUT_DIR / f"quick_{sequence}_window0.json"
    else:
        output_path = OUTPUT_DIR / "full_calibration_20seq_5120frames.json"
    result = run_audit(args.mode, output_path, args.sequence)
    aggregate = result["aggregate"]
    print(json.dumps({
        "status": result["status"],
        "mode": result["mode"],
        "output": str(output_path),
        "windows": aggregate["window_count"],
        "fixed_frames": aggregate["fixed_frame_count"],
        "scored_frames": aggregate["scored_frames_after_contiguous_min16"],
        "direct_world_frame_mean_mm": aggregate[
            "pooled_same_mask_frame_distributions"]["direct_world_frame_mean_mm"],
        "first2_similarity_align_W_frame_mean_mm": aggregate[
            "pooled_same_mask_frame_distributions"]["first2_similarity_align_W_frame_mean_mm"],
        "current_label_delta_frame_mean_mm": aggregate[
            "pooled_same_mask_frame_distributions"]["current_label_delta_frame_mean_mm"],
    }, indent=2))


if __name__ == "__main__":
    main()
