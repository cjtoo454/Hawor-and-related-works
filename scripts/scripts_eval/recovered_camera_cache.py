"""Validation and loading for the derived aligned WiLoR camera cache."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
COORDINATE_FRAME = "HOT3D evaluation world coordinates, metres"


def install_mano_legacy_compat() -> None:
    """Let the historical chumpy MANO pickle load with current NumPy/Python."""
    for name, value in {"bool": bool, "int": int, "float": float, "complex": complex,
                        "object": object, "unicode": str, "str": str}.items():
        np.__dict__.setdefault(name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


def cache_sources(root: Path, video: str) -> dict[str, Path]:
    return {
        "wilor_predictions": root / "eval_log_wilor_slam" / video / "wilor_predictions.npz",
        "wilor_evaluation_details": root / "eval_log_wilor_slam" / video / "evaluation_details.npz",
        "hmp_predictions": root / "eval_log_hmp_slam" / video / "hmp_predictions.npz",
        "hmp_evaluation_details": root / "eval_log_hmp_slam" / video / "evaluation_details.npz",
        "mano_right_model": root / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_recovered_camera_cache(
    cache_path: Path, frame_count: int, video: str,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Reject stale, tampered, malformed, or partly finite derived caches."""
    expected = ROOT / "datasets" / "hot3d_valset_export" / video / "preprocess" / "SLAM_WiLoR" / "recovered_aligned_cameras.npz"
    if cache_path.resolve() != expected.resolve():
        raise ValueError(f"Unexpected recovered camera cache path: {cache_path}")
    sidecar = cache_path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("format") != "recovered_aligned_cameras_v1" or metadata.get("video") != video:
        raise ValueError("Recovered camera metadata format or video mismatch")
    if metadata.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError("Recovered camera coordinate frame mismatch")
    if metadata.get("cache_sha256") != sha256_file(cache_path):
        raise ValueError("Recovered camera cache SHA-256 mismatch")
    paths = cache_sources(ROOT, video)
    recorded = metadata.get("source_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(paths):
        raise ValueError("Recovered camera source SHA-256 records are incomplete")
    for name, path in paths.items():
        if sha256_file(path) != recorded[name]:
            raise ValueError(f"Recovered camera source SHA-256 mismatch: {name}")
    with np.load(cache_path, allow_pickle=False) as data:
        required = {"frame_ids", "rotation_c2w", "translation_c2w", "valid",
                    "coordinate_frame", "provenance", "source_sha256"}
        if set(data.files) != required:
            raise ValueError("Recovered camera cache fields differ from expected schema")
        ids = data["frame_ids"].copy()
        r = data["rotation_c2w"].copy()
        t = data["translation_c2w"].copy()
        valid = data["valid"].copy()
        coordinate_frame = str(data["coordinate_frame"])
        provenance = str(data["provenance"])
        embedded_hashes = json.loads(str(data["source_sha256"]))
    if (ids.shape != (frame_count,) or not np.array_equal(ids, np.arange(frame_count))
            or r.shape != (frame_count, 3, 3) or t.shape != (frame_count, 3)
            or valid.shape != (frame_count,) or valid.dtype != np.bool_):
        raise ValueError("Recovered camera frame count, indexing, or shapes mismatch")
    if coordinate_frame != COORDINATE_FRAME or provenance != metadata.get("provenance") or embedded_hashes != recorded:
        raise ValueError("Recovered camera embedded provenance mismatch")
    if ("not original raw DROID-SLAM" not in provenance
            or "not identifiable" not in provenance):
        raise ValueError("Recovered camera limitations are missing")
    for log_dir in ("eval_log_wilor_slam", "eval_log_hmp_slam"):
        with np.load(ROOT / log_dir / video / "evaluation_details.npz", allow_pickle=False) as details:
            expected_valid = np.zeros(frame_count, dtype=bool)
            frame_ids = details["frame_ids"]
            if np.any(frame_ids < 0) or np.any(frame_ids >= frame_count):
                raise ValueError(f"{log_dir} evaluation frame IDs invalid")
            expected_valid[frame_ids] = True
        if not np.array_equal(valid, expected_valid):
            raise ValueError(f"Recovered camera valid mask differs from {log_dir}")
    if (not np.isfinite(r[valid]).all() or not np.isfinite(t[valid]).all()
            or not np.isnan(r[~valid]).all() or not np.isnan(t[~valid]).all()):
        raise ValueError("Recovered camera finite/NaN mask mismatch")
    orthogonality = np.max(np.abs(r[valid] @ np.swapaxes(r[valid], -1, -2) - np.eye(3)))
    determinant = np.linalg.det(r[valid])
    if orthogonality > 1e-5 or not np.allclose(determinant, 1.0, atol=1e-5, rtol=0):
        raise ValueError("Recovered camera rotations are not proper SO(3)")
    return torch.from_numpy(r).float(), torch.from_numpy(t).float(), valid
