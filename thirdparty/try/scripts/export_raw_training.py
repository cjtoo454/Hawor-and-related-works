#!/usr/bin/env python3
"""Export raw HOT3D preview video and sensor ZIPs to HaWoR training inputs.

The raw Aria preview has device-time metadata while HOT3D labels use timecode.
This exporter preserves that mapping, creates the fixed-focal pinhole images
expected by HaWoR, and writes the per-frame camera and MANO files consumed by
``prepare_training_data.py``. It never uses labels to generate image inputs.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from prepare_raw_trajectories import (  # noqa: E402
    EmptyHandTrajectory,
    MANO_MEMBER,
    _hand_record,
    load_mano_pca,
)


DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "datasets" / "hot3d_train100"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "datasets" / "hot3d_train100_export"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "data" / "raw_trajectories" / "split_manifest.json"
DEFAULT_MANO_ASSET = PROJECT_ROOT / "assets" / "mano_right.npz"
EXPORT_SCHEMA = "hawor_hot3d_raw_training_export_v1"
DEVICE_TIME_TOLERANCE_NS = 2
IMAGE_SIZE = 1408
FPS = 30
GT_MAPPING_MEMBER = "timecode_devicetime_mapping.csv"
HEAD_POSE_MEMBER = "headset_trajectory.csv"
CAMERA_MODELS_MEMBER = "camera_models.json"
ONLINE_CALIBRATION_MEMBER = "online_calibration.jsonl"


class ExportError(ValueError):
    """Raised when raw inputs cannot be aligned without losing frame integrity."""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split_manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--mano_asset", type=Path, default=DEFAULT_MANO_ASSET)
    parser.add_argument(
        "--rotation_evidence",
        type=Path,
        help="optional pixel-verification JSON to bind by path and SHA-256 in provenance",
    )
    parser.add_argument(
        "--preview_rotation",
        choices=("native", "clockwise90"),
        required=True,
        help=("orientation already present in preview_rgb.mp4 relative to the "
              "native VRS image; required because previews differ by source"),
    )
    parser.add_argument(
        "--device_time_tolerance_ns",
        type=int,
        default=DEVICE_TIME_TOLERANCE_NS,
        help="maximum absolute nearest mapping error (default: 2 ns)",
    )
    parser.add_argument(
        "--pose_tolerance_ns",
        type=int,
        default=0,
        help="maximum nearest headset-pose error; defaults to exact timecode only",
    )
    parser.add_argument(
        "--video", action="append", help="export only this sequence (diagnostic mode; repeatable)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        help="export at most this many decoded source frames per sequence (diagnostic only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only outputs whose source/config manifest and output hashes still match",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip_member(archive: zipfile.ZipFile, member_name: str) -> str:
    """Resolve a unique ZIP member by basename and reject ambiguous archives."""
    members = [name for name in archive.namelist() if Path(name).name == member_name]
    if len(members) != 1:
        if not members:
            raise ExportError(f"{member_name} is missing from {archive.filename}")
        raise ExportError(f"Expected one {member_name} in {archive.filename}, got {members}")
    return members[0]


def _read_csv_rows(archive: zipfile.ZipFile, member_name: str) -> list[dict[str, str]]:
    member = _zip_member(archive, member_name)
    with archive.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        return list(csv.DictReader(text))


def parse_device_timestamps(description: str, expected_frames: int) -> np.ndarray:
    """Read the MP4's full per-frame device-time array from container metadata."""
    try:
        values = json.loads(description)
    except (TypeError, json.JSONDecodeError) as error:
        raise ExportError("preview MP4 has no valid JSON device-time description") from error
    if not isinstance(values, list) or len(values) != expected_frames:
        actual = len(values) if isinstance(values, list) else type(values).__name__
        raise ExportError(
            f"preview device-time count does not match video frames: "
            f"metadata={actual}, frames={expected_frames}"
        )
    try:
        timestamps = np.asarray([int(value) for value in values], dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ExportError("preview device-time metadata contains a non-integer value") from error
    if timestamps.ndim != 1 or np.any(timestamps < 0):
        raise ExportError("preview device-time metadata must be nonnegative integers")
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ExportError("preview device timestamps are not monotonic")
    return timestamps


def map_device_frames_to_timecode(
    device_timestamps_ns: Iterable[int],
    mapping_rows: Iterable[tuple[int, int]],
    tolerance_ns: int = DEVICE_TIME_TOLERANCE_NS,
) -> dict[str, Any]:
    """Nearest-map device timestamps to timecode and drop repeated video times.

    ``mapping_rows`` are ``(timecode_ns, device_time_ns)`` pairs. Duplicate
    timestamps keep the first source frame, matching HOT3D's first-observation
    behavior; no replacement timestamp is synthesized.
    """
    if tolerance_ns < 0:
        raise ValueError("device-time tolerance must be nonnegative")
    rows = sorted((int(device), int(timecode)) for timecode, device in mapping_rows)
    if not rows:
        raise ExportError("timecode/device mapping is empty")
    device_keys = [row[0] for row in rows]
    if any(value < 0 for pair in rows for value in pair):
        raise ExportError("timecode/device mapping contains negative timestamps")

    retained_indices: list[int] = []
    retained_device: list[int] = []
    retained_timecode: list[int] = []
    retained_errors: list[int] = []
    dropped_duplicates: list[dict[str, int]] = []
    seen_timecode: set[int] = set()
    previous_timecode: int | None = None

    for source_index, source_device in enumerate(device_timestamps_ns):
        source_device = int(source_device)
        insertion = bisect.bisect_left(device_keys, source_device)
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(rows)]
        # Earlier mapping timestamp wins exact ties, like HOT3D's CLOSEST lookup.
        mapping_index = min(candidates, key=lambda index: (abs(device_keys[index] - source_device), device_keys[index]))
        mapped_device, mapped_timecode = rows[mapping_index]
        delta = mapped_device - source_device
        if abs(delta) > tolerance_ns:
            raise ExportError(
                f"source video frame {source_index} device timestamp misses nearest "
                f"mapping by {abs(delta)} ns (limit {tolerance_ns})"
            )
        if previous_timecode is not None and mapped_timecode < previous_timecode:
            raise ExportError("mapped preview timecodes are not monotonic")
        previous_timecode = mapped_timecode
        if mapped_timecode in seen_timecode:
            dropped_duplicates.append({
                "source_frame_index": source_index,
                "device_timestamp_ns": source_device,
                "timecode_timestamp_ns": mapped_timecode,
            })
            continue
        seen_timecode.add(mapped_timecode)
        retained_indices.append(source_index)
        retained_device.append(source_device)
        retained_timecode.append(mapped_timecode)
        retained_errors.append(delta)

    if any(right <= left for left, right in zip(retained_timecode, retained_timecode[1:])):
        raise ExportError("deduplicated timecodes are not strictly increasing")
    return {
        "source_frame_indices": np.asarray(retained_indices, dtype=np.int64),
        "device_timestamps_ns": np.asarray(retained_device, dtype=np.int64),
        "timecode_timestamps_ns": np.asarray(retained_timecode, dtype=np.int64),
        "device_time_mapping_delta_ns": np.asarray(retained_errors, dtype=np.int64),
        "dropped_duplicate_frames": dropped_duplicates,
    }


def orient_preview_to_native(image_rgb: np.ndarray, preview_rotation: str) -> np.ndarray:
    """Undo a known clockwise preview rotation before applying native calibration."""
    if preview_rotation == "native":
        return image_rgb
    if preview_rotation == "clockwise90":
        return np.rot90(image_rgb, k=1).copy()
    raise ValueError(f"unsupported preview rotation: {preview_rotation}")


def standardize_linear_image(image_rgb: np.ndarray) -> np.ndarray:
    """Match the repository exporter: RGB linear image rotated clockwise, as BGR."""
    return np.rot90(image_rgb[:, :, ::-1], k=-1).copy()


def quaternion_wxyz_to_matrix(quaternion: Iterable[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ExportError(f"invalid scalar-first quaternion: shape={q.shape}")
    norm = float(np.linalg.norm(q))
    if norm < 1e-10:
        raise ExportError("zero-norm quaternion in HOT3D input")
    w, x, y, z = q / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def quaternion_wxyz_to_axis_angle(quaternion: Iterable[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ExportError(f"invalid scalar-first quaternion: shape={q.shape}")
    norm = float(np.linalg.norm(q))
    if norm < 1e-10:
        raise ExportError("zero-norm quaternion in HOT3D input")
    q = q / norm
    if q[0] < 0:
        q = -q
    sine_half_angle = float(np.linalg.norm(q[1:]))
    if sine_half_angle < 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(sine_half_angle, float(q[0]))
    return (q[1:] * (angle / sine_half_angle)).astype(np.float32)


def _read_hand_samples(path: Path, sequence: str) -> dict[int, dict[str, Any]]:
    samples: dict[int, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        try:
            member = _zip_member(archive, MANO_MEMBER)
        except ExportError as error:
            raise EmptyHandTrajectory(str(error)) from error
        if archive.getinfo(member).file_size == 0:
            raise EmptyHandTrajectory(f"empty {MANO_MEMBER}")
        with archive.open(member) as source:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ExportError(f"invalid JSON in {path}:{member}:{line_number}") from error
                sample = _hand_record(record, sequence)
                if sample is not None:
                    # The official HOT3D trajectory loader preserves the first row per timestamp.
                    samples.setdefault(sample["timestamp_ns"], sample)
    if not samples:
        raise EmptyHandTrajectory("no right-hand (hand_poses['1']) records")
    return samples


def make_right_hand_annotations(
    timecode_timestamps_ns: np.ndarray,
    hand_samples: dict[int, dict[str, Any]],
    components: np.ndarray,
    pose_mean: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Expand MANO PCA labels on the video timeline; missing labels stay zero/masked."""
    count = len(timecode_timestamps_ns)
    rot = np.zeros((count, 3), dtype=np.float32)
    trans = np.zeros((count, 3), dtype=np.float32)
    pose = np.zeros((count, 45), dtype=np.float32)
    betas = np.zeros((count, 10), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    picked: list[tuple[int, dict[str, Any]]] = []
    for frame_id, timestamp_ns in enumerate(timecode_timestamps_ns):
        sample = hand_samples.get(int(timestamp_ns))
        if sample is not None:
            picked.append((frame_id, sample))
            valid[frame_id] = True
    if picked:
        frame_ids = np.asarray([frame_id for frame_id, _ in picked], dtype=np.int64)
        samples = [sample for _, sample in picked]
        pca = np.stack([sample["pose_pca"] for sample in samples]).astype(np.float32)
        pose[frame_ids] = pca @ components + pose_mean[None]
        trans[frame_ids] = np.stack([sample["t_xyz"] for sample in samples]).astype(np.float32)
        betas[frame_ids] = np.stack([sample["betas"] for sample in samples]).astype(np.float32)
        rot[frame_ids] = np.stack([
            quaternion_wxyz_to_axis_angle(sample["q_wxyz"]) for sample in samples
        ])
    return {
        "rot_r": rot,
        "trans_r": trans,
        "pose_r": pose,
        "betas_r": betas,
        "valid_r": valid,
    }, valid


def _single_file(sequence_root: Path, suffix: str) -> Path:
    candidates = sorted(sequence_root.glob(f"*{suffix}"))
    if len(candidates) != 1:
        raise ExportError(
            f"expected one *{suffix} under {sequence_root}, found {[p.name for p in candidates]}"
        )
    return candidates[0]


def _source_paths(data_root: Path, sequence: str) -> dict[str, Path]:
    sequence_root = data_root / sequence
    if not sequence_root.is_dir():
        raise FileNotFoundError(sequence_root)
    return {
        "preview_rgb_mp4": _single_file(sequence_root, "_preview_rgb.mp4"),
        "ground_truth_zip": _single_file(sequence_root, "_ground_truth.zip"),
        "hand_data_zip": _single_file(sequence_root, "_hand_data.zip"),
        "mps_slam_calibration_zip": _single_file(sequence_root, "_mps_slam_calibration.zip"),
    }


def _input_hashes(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
               "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _read_mapping_and_poses(
    ground_truth_zip: Path,
) -> tuple[list[tuple[int, int]], dict[int, np.ndarray], dict[str, Any]]:
    with zipfile.ZipFile(ground_truth_zip) as archive:
        mapping_rows = _read_csv_rows(archive, GT_MAPPING_MEMBER)
        mapping: list[tuple[int, int]] = []
        for row in mapping_rows:
            mapping.append((int(row["timecode_ns"]), int(row["devicetime_ns"])))
        if not mapping:
            raise ExportError(f"empty {GT_MAPPING_MEMBER} in {ground_truth_zip}")

        pose_rows = _read_csv_rows(archive, HEAD_POSE_MEMBER)
        required = {
            "timestamp[ns]", "t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]",
            "q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z",
        }
        if not pose_rows or not required.issubset(pose_rows[0]):
            raise ExportError(f"{HEAD_POSE_MEMBER} has no valid pose rows")
        poses: dict[int, np.ndarray] = {}
        for row in pose_rows:
            timestamp = int(row["timestamp[ns]"])
            quaternion = [float(row[key]) for key in ("q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z")]
            translation = [float(row[key]) for key in ("t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]")]
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = quaternion_wxyz_to_matrix(quaternion)
            matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
            poses[timestamp] = matrix  # Match the official loader's last-row behavior.

        camera_models = json.loads(archive.read(_zip_member(archive, CAMERA_MODELS_MEMBER)))
    rgb_models = [item for item in camera_models if item.get("label") == "camera-rgb"]
    if len(rgb_models) != 1:
        raise ExportError(f"expected one factory camera-rgb calibration, got {len(rgb_models)}")
    factory = rgb_models[0]
    image_width = int(factory["imageWidth"])
    image_height = int(factory["imageHeight"])
    params = np.asarray(factory.get("projectionParams", []), dtype=np.float64)
    if (image_width, image_height) != (IMAGE_SIZE, IMAGE_SIZE) or params.shape != (15,):
        raise ExportError(
            f"expected 1408x1408 FISHEYE624 factory camera, got "
            f"{image_width}x{image_height}, params={params.shape}"
        )
    factory_info = {
        "label": factory["label"],
        "image_width": image_width,
        "image_height": image_height,
        "projection_model": factory.get("projectionModelType"),
        "projection_params": params.tolist(),
        "target_focal_px": float(params[0]),
        "max_solid_angle": float(factory.get("maxSolidAngle", 1.0)),
        "serial_number": factory.get("serialNumber", ""),
    }
    return mapping, poses, factory_info


def _read_online_calibrations(path: Path) -> list[dict[str, Any]]:
    try:
        from projectaria_tools.core import mps  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "projectaria-tools is required to read online camera calibrations"
        ) from error
    calibrations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hot3d-online-calibration-") as temp_dir:
        with zipfile.ZipFile(path) as archive:
            member = _zip_member(archive, ONLINE_CALIBRATION_MEMBER)
            calibration_path = Path(temp_dir) / ONLINE_CALIBRATION_MEMBER
            with archive.open(member) as source, calibration_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        online_records = mps.read_online_calibration(str(calibration_path))
    for online in online_records:
        camera = online.get_camera_calib("camera-rgb")
        if camera is None:
            continue
        tracking = online.tracking_timestamp
        tracking_us = (
            tracking.days * 86_400_000_000
            + tracking.seconds * 1_000_000
            + tracking.microseconds
        )
        params = np.asarray(camera.get_projection_params(), dtype=np.float64)
        if params.shape != (15,) or not np.isfinite(params).all():
            continue
        transform = camera.get_transform_device_camera()
        matrix = np.asarray(transform.to_matrix(), dtype=np.float32)
        calibrations.append({
            "tracking_timestamp_us": int(tracking_us),
            "projection_name": str(camera.get_model_name()),
            "projection_params": params.astype(np.float32),
            "serial_number": camera.get_serial_number(),
            "T_device_camera": matrix,
            "valid_radius": float(camera.get_valid_radius()),
            "max_solid_angle": float(camera.get_max_solid_angle()),
            "source_camera_calibration": camera,
        })
    calibrations.sort(key=lambda item: item["tracking_timestamp_us"])
    if not calibrations:
        raise ExportError(f"no valid online camera-rgb calibration in {path}")
    times = [item["tracking_timestamp_us"] for item in calibrations]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ExportError(f"online camera calibration timestamps are not strictly increasing in {path}")
    return calibrations


def _nearest_calibration(
    device_timestamp_ns: int,
    calibrations: list[dict[str, Any]],
    tracking_times_ns: list[int],
) -> tuple[int, dict[str, Any], int]:
    target_ns = int(device_timestamp_ns)
    insertion = bisect.bisect_left(tracking_times_ns, target_ns)
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(calibrations)]
    index = min(
        candidates,
        key=lambda item: (abs(tracking_times_ns[item] - target_ns),
                          calibrations[item]["tracking_timestamp_us"]),
    )
    return index, calibrations[index], tracking_times_ns[index] - target_ns


def _projectaria_calibration_tools():
    try:
        from projectaria_tools.core.calibration import (  # type: ignore[import-not-found]
            CameraCalibration,
            FISHEYE624,
            distort_by_calibration,
            get_linear_camera_calibration,
        )
        from projectaria_tools.core.sophus import SE3  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "projectaria-tools is required for FISHEYE624 undistortion; "
            "install it into the active .env/try Python environment"
        ) from error
    return CameraCalibration, FISHEYE624, distort_by_calibration, get_linear_camera_calibration, SE3


def _make_online_camera_calibration(
    record: dict[str, Any], factory: dict[str, Any], tools: tuple[Any, ...]
):
    """Build the Project Aria camera object while fixing image dimensions to the MP4."""
    CameraCalibration, fisheye_model, _, _, _ = tools
    online = record["source_camera_calibration"]
    return CameraCalibration(
        "camera-rgb",
        fisheye_model,
        online.get_projection_params(),
        online.get_transform_device_camera(),
        int(factory["image_width"]),
        int(factory["image_height"]),
        online.get_valid_radius(),
        online.get_max_solid_angle(),
        online.get_serial_number(),
    )


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to scalar-first quaternion for Sophus SE3."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray([
            0.25 * scale,
            (m[2, 1] - m[1, 2]) / scale,
            (m[0, 2] - m[2, 0]) / scale,
            (m[1, 0] - m[0, 1]) / scale,
        ])
    else:
        diagonal = np.diag(m)
        i = int(np.argmax(diagonal))
        if i == 0:
            scale = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
            q = np.asarray([
                (m[2, 1] - m[1, 2]) / scale,
                0.25 * scale,
                (m[0, 1] + m[1, 0]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
            ])
        elif i == 1:
            scale = math.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
            q = np.asarray([
                (m[0, 2] - m[2, 0]) / scale,
                (m[0, 1] + m[1, 0]) / scale,
                0.25 * scale,
                (m[1, 2] + m[2, 1]) / scale,
            ])
        else:
            scale = math.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
            q = np.asarray([
                (m[1, 0] - m[0, 1]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
                (m[1, 2] + m[2, 1]) / scale,
                0.25 * scale,
            ])
    return q / np.linalg.norm(q)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _output_hashes(sequence_root: Path, relative_paths: list[str]) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = sequence_root / relative
        outputs[relative] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return outputs


def _resume_sequence(
    sequence_root: Path,
    source_hashes: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = sequence_root / "raw_export_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (manifest.get("schema") != EXPORT_SCHEMA or manifest.get("status") != "exported"
            or manifest.get("source_files") != source_hashes
            or manifest.get("config") != config):
        return None
    for relative, expected in manifest.get("outputs", {}).items():
        path = sequence_root / relative
        if not path.is_file() or path.stat().st_size != expected.get("size_bytes"):
            return None
        if sha256_file(path) != expected.get("sha256"):
            return None
    return manifest


def _read_preview_metadata(video_path: Path):
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required to read preview_rgb.mp4 metadata and frames") from error
    container = av.open(str(video_path))
    if not container.streams.video:
        container.close()
        raise ExportError(f"no video stream in {video_path}")
    stream = container.streams.video[0]
    frame_count = int(stream.frames)
    # The timestamp list is stored as ISO BMFF container-level description metadata.
    description = container.metadata.get("description")
    if not description:
        container.close()
        raise ExportError(f"preview MP4 is missing its device-time description: {video_path}")
    timestamps = parse_device_timestamps(description, frame_count)
    if (int(stream.width), int(stream.height)) != (IMAGE_SIZE, IMAGE_SIZE):
        container.close()
        raise ExportError(
            f"expected {IMAGE_SIZE}x{IMAGE_SIZE} preview, got {stream.width}x{stream.height}"
        )
    return container, stream, timestamps


def _read_expected_split(split_manifest_path: Path, source_sequences: list[str]):
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    train = list(manifest.get("train_sequences", []))
    calibration = list(manifest.get("calibration_sequences", []))
    selected_entries = manifest.get("selected_train_json_entries")
    if not train or not calibration:
        raise ExportError(f"invalid trajectory split manifest: {split_manifest_path}")
    selected_matches = (
        selected_entries == source_sequences
        if isinstance(selected_entries, list)
        else selected_entries == len(source_sequences)
        if isinstance(selected_entries, int)
        else False
    )
    if not selected_matches:
        raise ExportError(
            "trajectory split manifest was prepared from a different train.json; "
            "refusing to change participant separation"
        )
    prepared_sequences = {
        record.get("sequence") for record in manifest.get("sequence_records", [])
        if record.get("sequence")
    }
    if prepared_sequences and prepared_sequences != set(source_sequences):
        raise ExportError("trajectory split manifest sequence audit differs from source train.json")
    if set(train) & set(calibration):
        raise ExportError("trajectory train and calibration sequences overlap")
    if not (set(train) | set(calibration)).issubset(source_sequences):
        raise ExportError("trajectory split contains a sequence absent from source train.json")
    return manifest, train, calibration


def _scan_hand_samples(sequence: str, paths: dict[str, Path]):
    try:
        return _read_hand_samples(paths["hand_data_zip"], sequence)
    except EmptyHandTrajectory:
        raise


def _dump_annotations(path: Path, annotation: dict[str, np.ndarray]) -> None:
    try:
        import joblib
        import torch
    except ImportError as error:
        raise RuntimeError("joblib and PyTorch are required to write HaWoR anno.pth") from error
    payload: dict[str, Any] = {
        key: torch.from_numpy(np.asarray(value, dtype=np.float32))
        for key, value in annotation.items() if key != "valid_r"
    }
    payload["valid_r"] = torch.from_numpy(np.asarray(annotation["valid_r"], dtype=bool))
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(payload, temporary)
    temporary.replace(path)


def _export_sequence(
    sequence: str,
    paths: dict[str, Path],
    destination: Path,
    source_hashes: dict[str, dict[str, Any]],
    config: dict[str, Any],
    components: np.ndarray,
    pose_mean: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # Empty right-hand trajectories are audited but never exported as fake zeros.
    hand_samples = _scan_hand_samples(sequence, paths)
    if args.resume:
        resumed = _resume_sequence(destination, source_hashes, config)
        if resumed is not None:
            return {"sequence": sequence, "status": "exported", "resumed": True,
                    "frame_count": resumed["frame_count"], "output": str(destination)}

    mapping, headset_poses, factory = _read_mapping_and_poses(paths["ground_truth_zip"])
    calibrations = _read_online_calibrations(paths["mps_slam_calibration_zip"])
    calibration_times = [item["tracking_timestamp_us"] for item in calibrations]
    calibration_times_ns = [value * 1000 for value in calibration_times]
    container, stream, all_device_times = _read_preview_metadata(paths["preview_rgb_mp4"])
    mapped = map_device_frames_to_timecode(
        all_device_times, mapping, tolerance_ns=args.device_time_tolerance_ns)
    source_indices = mapped["source_frame_indices"]
    if args.max_frames is not None:
        source_indices = source_indices[:args.max_frames]
        for key in ("device_timestamps_ns", "timecode_timestamps_ns", "device_time_mapping_delta_ns"):
            mapped[key] = mapped[key][:args.max_frames]
        mapped["dropped_duplicate_frames"] = [
            item for item in mapped["dropped_duplicate_frames"]
            if item["source_frame_index"] < (int(source_indices[-1]) + 1 if len(source_indices) else 0)
        ]
    timecodes = mapped["timecode_timestamps_ns"]
    device_times = mapped["device_timestamps_ns"]
    headset_timestamps = sorted(headset_poses)
    matched_head_poses: list[np.ndarray] = []
    head_pose_deltas_ns: list[int] = []
    for timestamp in timecodes:
        insertion = bisect.bisect_left(headset_timestamps, int(timestamp))
        candidates = [index for index in (insertion - 1, insertion)
                      if 0 <= index < len(headset_timestamps)]
        nearest = min(candidates, key=lambda index: (
            abs(headset_timestamps[index] - int(timestamp)), headset_timestamps[index]))
        pose_timestamp = headset_timestamps[nearest]
        delta_ns = pose_timestamp - int(timestamp)
        if abs(delta_ns) > args.pose_tolerance_ns:
            container.close()
            raise ExportError(
                f"video timecode {int(timestamp)} has no headset pose within "
                f"{args.pose_tolerance_ns} ns (nearest delta {delta_ns} ns)"
            )
        matched_head_poses.append(headset_poses[pose_timestamp])
        head_pose_deltas_ns.append(delta_ns)

    frame_to_metadata = {
        int(source_index): index for index, source_index in enumerate(source_indices)
    }
    # Calibration matching and camera transforms are retained per exported frame.
    matched_calibrations: list[dict[str, Any]] = []
    calibration_errors_ns: list[int] = []
    for device_timestamp in device_times:
        _, calibration, error_ns = _nearest_calibration(
            int(device_timestamp), calibrations, calibration_times_ns)
        if calibration["projection_name"] not in (
            "FisheyeRadTanThinPrism", "CameraModelType.FISHEYE624"
        ):
            container.close()
            raise ExportError(
                f"unsupported online RGB projection model: {calibration['projection_name']}"
            )
        matched_calibrations.append(calibration)
        calibration_errors_ns.append(error_ns)

    output_tmp = Path(tempfile.mkdtemp(prefix=f".{sequence}.export-", dir=destination.parent))
    image_root = output_tmp / "extracted_images"
    image_root.mkdir(parents=True)
    tools = _projectaria_calibration_tools()
    _, _, _, get_linear_camera_calibration, _ = tools
    pinhole = get_linear_camera_calibration(IMAGE_SIZE, IMAGE_SIZE, factory["target_focal_px"])
    calibration_cache: dict[int, Any] = {}
    calibration_record_ids: dict[int, int] = {}
    unique_calibrations: list[dict[str, Any]] = []
    head_pose_list: list[np.ndarray] = []
    camera_extrinsics: list[np.ndarray] = []
    online_params: list[np.ndarray] = []
    online_tracking_times: list[int] = []
    output_images: list[str] = []
    frame_metadata: dict[str, Any] = {
        "source_frame_indices": [],
        "device_timestamps_ns": [],
        "timecode_timestamps_ns": [],
        "online_calibration_indices": [],
        "head_pose_time_delta_ns": [],
    }
    video_output_path = output_tmp / f"{sequence}.mp4"
    try:
        import av
        import cv2
        video_container = av.open(str(video_output_path), mode="w")
        video_stream = video_container.add_stream("libx264", rate=FPS)
        video_stream.width = IMAGE_SIZE
        video_stream.height = IMAGE_SIZE
        video_stream.pix_fmt = "yuv420p"

        target_lookup = frame_to_metadata
        output_frame = 0
        decoded_frames = 0
        for frame in container.decode(stream):
            original_index = decoded_frames
            decoded_frames += 1
            if args.max_frames is not None and output_frame >= args.max_frames:
                break
            mapped_index = target_lookup.get(original_index)
            if mapped_index is None:
                continue
            raw_rgb = frame.to_ndarray(format="rgb24")
            raw_native = orient_preview_to_native(raw_rgb, args.preview_rotation)
            calibration = matched_calibrations[mapped_index]
            calibration_key = id(calibration)
            if calibration_key not in calibration_cache:
                calibration_cache[calibration_key] = _make_online_camera_calibration(
                    calibration, factory, tools)
                calibration_record_ids[calibration_key] = len(unique_calibrations)
                unique_calibrations.append({
                    "tracking_timestamp_us": calibration["tracking_timestamp_us"],
                    "projection_name": calibration["projection_name"],
                    "projection_params": calibration["projection_params"].tolist(),
                    "serial_number": calibration["serial_number"],
                    "T_device_camera": calibration["T_device_camera"].tolist(),
                })
            native_calibration = calibration_cache[calibration_key]
            undistorted = tools[2](raw_native, pinhole, native_calibration)
            if undistorted is None or undistorted.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
                raise ExportError(
                    f"invalid undistorted frame shape at source frame {original_index}: "
                    f"{getattr(undistorted, 'shape', None)}"
                )
            bgr = standardize_linear_image(undistorted)
            image_path = image_root / f"{output_frame:06d}.jpg"
            if not cv2.imwrite(str(image_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise ExportError(f"failed to write {image_path}")
            video_frame = av.VideoFrame.from_ndarray(bgr[:, :, ::-1], format="rgb24")
            video_frame.pts = output_frame
            video_frame.time_base = Fraction(1, FPS)
            for packet in video_stream.encode(video_frame):
                video_container.mux(packet)

            source_calibration_index = calibration_record_ids[calibration_key]
            head_pose_list.append(matched_head_poses[mapped_index])
            camera_extrinsics.append(calibration["T_device_camera"])
            online_params.append(calibration["projection_params"])
            online_tracking_times.append(calibration["tracking_timestamp_us"])
            output_images.append(f"extracted_images/{image_path.name}")
            frame_metadata["source_frame_indices"].append(original_index)
            frame_metadata["device_timestamps_ns"].append(int(device_times[mapped_index]))
            frame_metadata["timecode_timestamps_ns"].append(int(timecodes[mapped_index]))
            frame_metadata["online_calibration_indices"].append(source_calibration_index)
            frame_metadata["head_pose_time_delta_ns"].append(head_pose_deltas_ns[mapped_index])
            output_frame += 1
        for packet in video_stream.encode(None):
            video_container.mux(packet)
        video_container.close()
        container.close()
        if output_frame != len(timecodes):
            raise ExportError(
                f"decoded/exported frame mismatch for {sequence}: "
                f"expected {len(timecodes)}, got {output_frame}"
            )

        annotation, hand_valid = make_right_hand_annotations(
            timecodes, hand_samples, components, pose_mean)
        _atomic_pickle(output_tmp / "tt_list.pkl", [int(value) for value in timecodes])
        _atomic_pickle(output_tmp / "head_pose.pkl", np.stack(head_pose_list).astype(np.float32))
        _atomic_pickle(output_tmp / "ego_extrinsics.pkl", np.stack(camera_extrinsics).astype(np.float32))
        _dump_annotations(output_tmp / "anno.pth", annotation)
        (output_tmp / "focal.txt").write_text(
            f"{factory['target_focal_px']:.9f}\n", encoding="utf-8")
        np.savez_compressed(
            output_tmp / "online_camera_calibration.npz",
            T_device_camera=np.stack(camera_extrinsics).astype(np.float32),
            projection_params=np.stack(online_params).astype(np.float32),
            tracking_timestamp_us=np.asarray(online_tracking_times, dtype=np.int64),
            source_device_timestamp_ns=device_times,
            timecode_timestamp_ns=timecodes,
            calibration_index=np.asarray(frame_metadata["online_calibration_indices"], dtype=np.int32),
        )

        relative_outputs = [
            f"{sequence}.mp4", "tt_list.pkl", "head_pose.pkl", "ego_extrinsics.pkl",
            "anno.pth", "focal.txt", "online_camera_calibration.npz", *output_images,
        ]
        output_hashes = _output_hashes(output_tmp, relative_outputs)
        source_indices_list = [int(value) for value in frame_metadata["source_frame_indices"]]
        output_manifest = {
            "schema": EXPORT_SCHEMA,
            "status": "exported",
            "sequence": sequence,
            "diagnostic": bool(args.max_frames is not None or args.video),
            "max_frames": args.max_frames,
            "formal_training_eligible": not bool(args.max_frames is not None or args.video),
            "frame_count": output_frame,
            "source_files": source_hashes,
            "config": config,
            "timestamp_mapping": {
                "source": "preview MP4 container description device ns -> nearest timecode_devicetime_mapping.csv device ns",
                "device_time_tolerance_ns": args.device_time_tolerance_ns,
                "max_abs_mapping_delta_ns": int(np.max(np.abs(mapped["device_time_mapping_delta_ns"]))) if len(mapped["device_time_mapping_delta_ns"]) else 0,
                **frame_metadata,
                "dropped_duplicate_frames": mapped["dropped_duplicate_frames"],
            },
            "head_pose": {
                "source": "ground_truth.zip:headset_trajectory.csv, nearest timecode lookup",
                "max_abs_time_delta_ns": int(max(map(abs, head_pose_deltas_ns), default=0)),
            },
            "online_calibration": {
                "source": "mps_slam_calibration.zip:online_calibration.jsonl, nearest tracking_timestamp_us to MP4 device ns",
                "max_abs_device_time_delta_ns": int(max(map(abs, calibration_errors_ns), default=0)),
                "factory_target_camera": factory,
                "unique_source_calibrations": unique_calibrations,
                "fixed_target_focal_px": factory["target_focal_px"],
                "source_online_focal_px_by_frame": [float(params[0]) for params in online_params],
                "output_extrinsics_file": "ego_extrinsics.pkl; device-to-RGB-camera T_device_camera",
            },
            "image_transform": {
                "input_orientation": args.preview_rotation,
                "source_array_format": "RGB",
                "undistortion": "Project Aria FISHEYE624 online calibration to fixed-focal 1408x1408 LINEAR",
                "output_orientation": "clockwise90, matching hot3d/Hot3DVisualizer.py",
                "saved_jpeg_channel_order": "BGR passed to OpenCV",
            },
            "mano_labels": {
                "source": f"hand_data.zip:{MANO_MEMBER}; exact timecode lookup",
                "handedness": "right hand_poses['1']",
                "pca_expansion": "pose_pca @ hands_components[:15] + hands_mean",
                "missing_frame_behavior": "zero parameters plus valid_r=false; consumers retain their zero-rotation mask convention",
                "valid_frames": int(hand_valid.sum()),
                "runtime_input": False,
            },
            "outputs": output_hashes,
        }
        _write_json(output_tmp / "raw_export_manifest.json", output_manifest)
    except BaseException:
        try:
            if "video_container" in locals():
                video_container.close()
        except Exception:
            pass
        try:
            container.close()
        except Exception:
            pass
        shutil.rmtree(output_tmp, ignore_errors=True)
        raise

    # Replace only a directory carrying this export's own manifest. Unknown data is preserved.
    if destination.exists():
        existing_manifest = destination / "raw_export_manifest.json"
        if not existing_manifest.is_file():
            shutil.rmtree(output_tmp, ignore_errors=True)
            raise ExportError(
                f"refusing to replace existing non-export directory {destination}"
            )
        try:
            old = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            shutil.rmtree(output_tmp, ignore_errors=True)
            raise ExportError(f"refusing to replace unrecognized export directory {destination}")
        if old.get("schema") != EXPORT_SCHEMA:
            shutil.rmtree(output_tmp, ignore_errors=True)
            raise ExportError(f"refusing to replace unrelated data in {destination}")
        backup = destination.with_name(destination.name + ".old-export")
        if backup.exists():
            raise ExportError(f"stale export backup exists at {backup}; inspect before retrying")
        destination.rename(backup)
        try:
            output_tmp.rename(destination)
        except BaseException:
            backup.rename(destination)
            raise
        shutil.rmtree(backup)
    else:
        output_tmp.rename(destination)

    return {"sequence": sequence, "status": "exported", "resumed": False,
            "frame_count": output_frame, "valid_mano_frames": int(hand_valid.sum()),
            "output": str(destination)}


def _write_root_outputs(
    output_root: Path,
    source_sequences: list[str],
    expected_train: list[str],
    expected_calibration: list[str],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    split_manifest_path: Path,
) -> dict[str, Any]:
    by_sequence = {record["sequence"]: record for record in records}
    diagnostic = bool(args.max_frames is not None or args.video)
    exported = {
        sequence for sequence, record in by_sequence.items()
        if record.get("status") == "exported"
    }
    if diagnostic:
        training_sequences: list[str] = []
        train_sequences: list[str] = []
        calibration_sequences: list[str] = []
    else:
        train_sequences = [sequence for sequence in expected_train if sequence in exported]
        calibration_sequences = [sequence for sequence in expected_calibration if sequence in exported]
        allowed = set(expected_train) | set(expected_calibration)
        training_sequences = [
            sequence for sequence in source_sequences
            if sequence in exported and sequence in allowed
        ]
    full_split_complete = (
        not diagnostic
        and train_sequences == expected_train
        and calibration_sequences == expected_calibration
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "train.json").write_text(
        json.dumps(training_sequences, indent=2), encoding="utf-8")
    (output_root / "calibration.json").write_text(
        json.dumps(calibration_sequences, indent=2), encoding="utf-8")
    split = {
        "schema": "hot3d_raw_training_export_split_v1",
        "diagnostic": diagnostic,
        "formal_training_eligible": full_split_complete,
        "train": train_sequences,
        "calibration": calibration_sequences,
        "train_and_calibration_union": training_sequences,
        "source_train_json_entries": len(source_sequences),
        "source_trajectory_split_manifest": str(split_manifest_path.resolve()),
        "skipped_sequences": [record for record in records if record.get("status") != "exported"],
    }
    (output_root / "training_split.json").write_text(
        json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    aggregate = {
        "schema": EXPORT_SCHEMA,
        "diagnostic": diagnostic,
        "max_frames": args.max_frames,
        "selected_sequences": [record["sequence"] for record in records],
        "source_sequence_count": len(source_sequences),
        "exported_sequence_count": len(exported),
        "formal_training_eligible": split["formal_training_eligible"],
        "preview_rotation": args.preview_rotation,
        "output_root": str(output_root.resolve()),
        "training_split": split,
        "sequence_records": records,
    }
    _write_json(output_root / "export_provenance.json", aggregate)
    return aggregate


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max_frames must be positive")
    if args.device_time_tolerance_ns < 0 or args.pose_tolerance_ns < 0:
        raise ValueError("timestamp tolerances must be nonnegative")
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == data_root or data_root in output_root.parents or output_root in data_root.parents:
        raise ValueError("output_root must be separate from the raw data_root")
    source_sequences = json.loads((data_root / "train.json").read_text(encoding="utf-8"))
    split_manifest, train_sequences, calibration_sequences = _read_expected_split(
        args.split_manifest, source_sequences)
    selected = list(source_sequences)
    diagnostic = bool(args.max_frames is not None or args.video)
    if args.video:
        unknown = sorted(set(args.video) - set(source_sequences))
        if unknown:
            raise ValueError(f"unknown sequence(s) requested by --video: {unknown}")
        selected = [sequence for sequence in source_sequences if sequence in set(args.video)]

    rotation_evidence = None
    if args.rotation_evidence is not None:
        evidence_path = args.rotation_evidence.resolve()
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        rows = evidence.get("frames", [])
        if (evidence.get("verified_raw_fisheye_preview") is not True or not rows
                or any(row.get("raw_to_preview_clockwise_degrees") != 90 for row in rows)):
            raise ExportError(f"rotation evidence does not verify clockwise90: {evidence_path}")
        rotation_evidence = {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
            "verified_preview_sha256": evidence.get("preview_sha256"),
            "verified_vrs_sha256": evidence.get("vrs_sha256"),
            "verified_frames": rows,
            "scope": "pixel-verified sample sequence; applied as preview orientation convention",
        }

    if diagnostic and (output_root / "export_provenance.json").is_file():
        existing = json.loads((output_root / "export_provenance.json").read_text(encoding="utf-8"))
        if existing.get("formal_training_eligible"):
            raise ExportError(
                "diagnostic export would overwrite a complete formal export; "
                "choose a separate --output_root"
            )

    components, pose_mean = load_mano_pca(args.mano_asset)
    config = {
        "mano_asset": str(args.mano_asset.resolve()),
        "mano_asset_sha256": sha256_file(args.mano_asset),
        "preview_rotation": args.preview_rotation,
        "device_time_tolerance_ns": args.device_time_tolerance_ns,
        "pose_tolerance_ns": args.pose_tolerance_ns,
        "max_frames": args.max_frames,
        "diagnostic": diagnostic,
        "target_image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "target_fps": FPS,
        "calibration": "online FISHEYE624 -> fixed factory focal LINEAR",
        "rotation_evidence_sha256": (
            rotation_evidence["sha256"] if rotation_evidence is not None else None
        ),
    }
    records: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for sequence in selected:
        destination = output_root / sequence
        record: dict[str, Any] = {"sequence": sequence, "status": "skipped"}
        try:
            paths = _source_paths(data_root, sequence)
            source_hashes = _input_hashes(paths)
            try:
                result = _export_sequence(
                    sequence, paths, destination, source_hashes, config,
                    components, pose_mean, args)
                record.update(result)
                record["source_files"] = source_hashes
            except EmptyHandTrajectory as error:
                record.update({"status": "skipped", "reason": str(error),
                               "source_files": source_hashes,
                               "formal_training_eligible": False})
        except (FileNotFoundError, ExportError, zipfile.BadZipFile) as error:
            record.update({"status": "skipped", "reason": str(error),
                           "formal_training_eligible": False})
        records.append(record)
        print(f"[{record['status']}] {sequence}: {record.get('reason', record.get('frame_count'))}")

    aggregate = _write_root_outputs(
        output_root, source_sequences, train_sequences, calibration_sequences,
        records, args, args.split_manifest)
    # The source stage-1 manifest itself is bound into the export-level provenance.
    aggregate["source_trajectory_split_sha256"] = sha256_file(args.split_manifest)
    aggregate["rotation_evidence"] = rotation_evidence
    _write_json(output_root / "export_provenance.json", aggregate)
    print(
        f"Exported {aggregate['exported_sequence_count']}/{len(selected)} sequences; "
        f"formal_training_eligible={aggregate['formal_training_eligible']}"
    )
    return aggregate


def main() -> None:
    args = build_argument_parser().parse_args()
    export_dataset(args)


if __name__ == "__main__":
    main()
