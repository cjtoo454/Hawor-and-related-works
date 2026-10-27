"""Recover *aligned* WiLoR-SLAM camera poses from saved MANO/world joints.

This is a derived cache for rendering historical evaluations.  It cannot
recover the original DROID trajectory, scale, disparity, or camera gauge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recovered_camera_cache import (  # noqa: E402
    cache_sources, install_mano_legacy_compat, sha256_file,
    validate_recovered_camera_cache,
)


def fit_rigid_camera(camera_joints: np.ndarray, world_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Find SE(3) mapping J_world = R @ J_camera + t, without scale."""
    x = np.asarray(camera_joints, dtype=np.float64)
    y = np.asarray(world_joints, dtype=np.float64)
    if x.shape != (21, 3) or y.shape != (21, 3) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Rigid fit requires two finite (21, 3) joint arrays")
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    centered = x - xc
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-5:
        raise ValueError("Camera joints have insufficient 3D rank")
    u, _, vh = np.linalg.svd(centered.T @ (y - yc))
    reflection = np.eye(3)
    reflection[2, 2] = np.linalg.det(vh.T @ u.T)
    r = vh.T @ reflection @ u.T
    t = yc - r @ xc
    residual = float(np.max(np.abs(x @ r.T + t - y)))
    if not np.isfinite(residual) or abs(np.linalg.det(r) - 1) > 1e-8:
        raise ValueError("Rigid fit produced an invalid proper rotation")
    return r, t, residual


def prediction_batch(prediction: dict[str, np.ndarray], ids: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "init_root_orient": torch.from_numpy(prediction["global_orient"][ids, 0]).float().unsqueeze(0),
        "init_hand_pose": torch.from_numpy(prediction["hand_pose"][ids]).float().unsqueeze(0),
        "init_trans": torch.from_numpy(prediction["camera_translation"][ids]).float().unsqueeze(0),
        "init_betas": torch.from_numpy(prediction["betas"][ids]).float().unsqueeze(0),
    }


@torch.no_grad()
def camera_joints(prediction: dict[str, np.ndarray], ids: np.ndarray, use_cuda: bool) -> np.ndarray:
    from hawor.utils.process import run_mano
    from hawor.utils.rotation import rotation_matrix_to_angle_axis

    batch = prediction_batch(prediction, ids)
    orient = rotation_matrix_to_angle_axis(batch["init_root_orient"])
    pose = rotation_matrix_to_angle_axis(batch["init_hand_pose"])
    result = run_mano(batch["init_trans"], orient, pose, betas=batch["init_betas"], use_cuda=use_cuda)
    return result["joints"][0].detach().cpu().numpy()


@torch.no_grad()
def validate_method(
    name: str,
    prediction: dict[str, np.ndarray],
    saved_world: np.ndarray,
    ids: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    batch_size: int,
    use_cuda: bool,
) -> dict[str, float]:
    from hawor.utils.process import run_mano
    from lib.eval_utils.custom_utils import cam2world_convert

    errors = []
    for offset in range(0, len(ids), batch_size):
        selected = ids[offset:offset + batch_size]
        world = cam2world_convert(
            torch.from_numpy(rotation[selected]).float(),
            torch.from_numpy(translation[selected]).float(),
            prediction_batch(prediction, selected),
            "right",
        )
        result = run_mano(
            world["init_trans"], world["init_root_orient"],
            world["init_hand_pose"], betas=world["init_betas"], use_cuda=use_cuda,
        )
        actual = result["joints"][0].detach().cpu().numpy()
        errors.extend(np.max(np.abs(actual - saved_world[selected]), axis=(1, 2)).tolist())
    errors_arr = np.asarray(errors)
    stats = {"count": int(len(ids)), "max_m": float(errors_arr.max()),
             "p50_m": float(np.percentile(errors_arr, 50)),
             "p95_m": float(np.percentile(errors_arr, 95)),
             "p99_m": float(np.percentile(errors_arr, 99))}
    print(f"{name} reconstructed-world errors: {stats}", flush=True)
    if not np.isfinite(errors_arr).all() or stats["max_m"] > 2e-5:
        raise ValueError(f"{name} reconstructed joints exceed 2e-5 m")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="P0001_8d136980")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0",
                        help="CUDA device required by the historical cam2world_convert path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--single-frame", type=int,
                        help="Check one WiLoR frame and exit without writing")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; historical cam2world_convert requires CUDA")
    install_mano_legacy_compat()
    use_cuda = args.device == "cuda:0"
    paths = cache_sources(ROOT, args.video)
    with np.load(paths["wilor_predictions"]) as data:
        wilor = {key: data[key].copy() for key in ("frame_id", "valid", "global_orient", "hand_pose", "betas", "camera_translation")}
    with np.load(paths["hmp_predictions"]) as data:
        hmp = {key: data[key].copy() for key in ("frame_id", "valid", "global_orient", "hand_pose", "betas", "camera_translation")}
    with np.load(paths["wilor_evaluation_details"]) as data:
        wid = data["frame_ids"].copy()
        wworld = data["pred_world_joints_m"].copy()
    with np.load(paths["hmp_evaluation_details"]) as data:
        hid = data["frame_ids"].copy()
        hworld = data["pred_world_joints_m"].copy()
    n = len(wworld)
    if (len(hworld) != n or not np.array_equal(wid, hid)
            or not np.array_equal(wilor["frame_id"], np.arange(n))
            or not np.array_equal(hmp["frame_id"], np.arange(n))
            or not np.array_equal(wilor["valid"], hmp["valid"])):
        raise ValueError("Prediction frame indexing or method valid masks differ")
    valid = np.zeros(n, dtype=bool)
    valid[wid] = True
    if (len(np.unique(wid)) != len(wid) or np.any(wid < 0) or np.any(wid >= n)
            or np.any(valid & ~wilor["valid"])
            or not np.array_equal(valid, np.isfinite(wworld).all(axis=(1, 2)))
            or not np.array_equal(valid, np.isfinite(hworld).all(axis=(1, 2)))):
        raise ValueError("Evaluation valid masks or finite world joints disagree")
    ids = np.flatnonzero(valid)
    if args.single_frame is not None:
        frame = args.single_frame
        if frame < 0 or frame >= n or not valid[frame]:
            raise ValueError(f"Frame {frame} is outside valid frames")
        joints = camera_joints(wilor, np.array([frame]), use_cuda)[0]
        r, t, residual = fit_rigid_camera(joints, wworld[frame])
        print(f"single_frame={frame} rigid_fit_max_abs_error_m={residual:.9g} det={np.linalg.det(r):.9g}", flush=True)
        if residual > 2e-5:
            raise ValueError("Frame is not rigid to within 2e-5 m")
        rotation = np.full((n, 3, 3), np.nan, dtype=np.float64)
        translation = np.full((n, 3), np.nan, dtype=np.float64)
        rotation[frame], translation[frame] = r, t
        validate_method("WiLoR frame", wilor, wworld, np.array([frame]), rotation,
                        translation, 1, use_cuda)
        return

    rotation = np.full((n, 3, 3), np.nan, dtype=np.float64)
    translation = np.full((n, 3), np.nan, dtype=np.float64)
    rigid_errors = np.full(n, np.nan, dtype=np.float64)
    for offset in range(0, len(ids), args.batch_size):
        selected = ids[offset:offset + args.batch_size]
        joints = camera_joints(wilor, selected, use_cuda)
        for index, frame in enumerate(selected):
            rotation[frame], translation[frame], rigid_errors[frame] = fit_rigid_camera(joints[index], wworld[frame])
        print(f"fitted {min(offset + args.batch_size, len(ids))}/{len(ids)} valid frames", flush=True)
    if np.nanmax(rigid_errors) > 2e-5:
        raise ValueError(f"Maximum rigid fit error {np.nanmax(rigid_errors):.9g} m exceeds tolerance")
    validation = {
        "valid_frame_count": int(len(ids)), "invalid_frame_count": int(n - len(ids)),
        "wilor_rigid_fit_max_m": float(np.nanmax(rigid_errors)),
        "wilor_rigid_fit_p50_m": float(np.nanpercentile(rigid_errors, 50)),
        "wilor_rigid_fit_p95_m": float(np.nanpercentile(rigid_errors, 95)),
        "wilor_rigid_fit_p99_m": float(np.nanpercentile(rigid_errors, 99)),
        "wilor_cam2world_run_mano": validate_method(
            "WiLoR", wilor, wworld, ids, rotation, translation, args.batch_size, use_cuda),
        "hmp_cam2world_run_mano": validate_method(
            "HMP", hmp, hworld, ids, rotation, translation, args.batch_size, use_cuda),
    }
    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    output = ROOT / "datasets" / "hot3d_valset_export" / args.video / "preprocess" / "SLAM_WiLoR"
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "recovered_aligned_cameras.npz"
    sidecar = output / "recovered_aligned_cameras.json"
    provenance = (
        "Derived aligned SE3 camera poses fitted to saved WiLoR MANO camera joints and "
        "world evaluation joints; not original raw DROID-SLAM trajectory. Original scale, "
        "disparity, and pre-alignment gauge are not identifiable from these artifacts."
    )
    np.savez_compressed(
        cache, frame_ids=np.arange(n, dtype=np.int64),
        rotation_c2w=rotation, translation_c2w=translation, valid=valid,
        coordinate_frame=np.array("HOT3D evaluation world coordinates, metres"),
        provenance=np.array(provenance), source_sha256=np.array(json.dumps(source_hashes, sort_keys=True)),
    )
    metadata = {
        "format": "recovered_aligned_cameras_v1", "video": args.video,
        "coordinate_frame": "HOT3D evaluation world coordinates, metres",
        "provenance": provenance, "source_sha256": source_hashes,
        "cache_sha256": sha256_file(cache),
        "validation": validation,
    }
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_recovered_camera_cache(cache, n, args.video)
    print(f"cache={cache} sha256={metadata['cache_sha256']}", flush=True)
    print(f"metadata={sidecar}", flush=True)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
