#!/usr/bin/env python3
"""Prepare aligned HaWoR observations and HOT3D sensor trajectories for training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from dual_loop.geometry import ManoGeometry  # noqa: E402
from dual_loop.io import (  # noqa: E402
    export_hawor_frontend,
    load_slam_c2w,
    load_timestamps,
)
from dual_loop.state import (  # noqa: E402
    STATE_DIM,
    axis_angle_to_matrix,
    pack_state,
)
from lib.eval_utils.custom_utils import (  # noqa: E402
    load_gt_cam,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dual-loop HOT3D training data")
    parser.add_argument(
        "--video_root", type=Path,
        default=REPOSITORY_ROOT / "datasets" / "hot3d_valset_export")
    parser.add_argument("--set_file", default="train.json")
    parser.add_argument("--video", action="append")
    parser.add_argument(
        "--frontend_log_root", type=Path, default=REPOSITORY_ROOT / "eval_log")
    parser.add_argument(
        "--output_root", type=Path, default=PROJECT_ROOT / "data" / "prepared")
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_diagnostic_export", action="store_true",
                        help="Prepare truncated raw exports for smoke checks only; never formal training")
    parser.add_argument(
        "--allow_gt_boxes", action="store_true",
        help=("Allow legacy GT-projected boxes or unverified frontend provenance; "
              "the prepared data is marked oracle-assisted/unknown"))
    return parser.parse_args()


def load_slam_cameras(args: argparse.Namespace, sequence: str, frame_count: int):
    if args.interval < 1:
        raise ValueError("--interval must be positive")
    rotations_c2w, translations_c2w = [], []
    for start in range(0, frame_count, args.interval):
        stop = min(start + args.interval, frame_count)
        path = (
            args.video_root / sequence / "preprocess" / "SLAM" /
            f"hawor_slam_w_scale_{start}_{stop}.npz")
        if not path.is_file():
            raise FileNotFoundError(f"Missing HaWoR SLAM trajectory: {path}")
        rotation, translation = load_slam_c2w(path)
        expected = stop - start
        if len(rotation) != expected or len(translation) != expected:
            raise ValueError(
                f"SLAM frame count mismatch in {path}: expected {expected}, "
                f"got rotations={len(rotation)}, translations={len(translation)}")
        rotations_c2w.append(rotation)
        translations_c2w.append(translation)
    rotation_c2w = torch.cat(rotations_c2w)
    translation_c2w = torch.cat(translations_c2w)
    rotation_w2c = rotation_c2w.transpose(-1, -2)
    translation_w2c = -torch.einsum(
        "bij,bj->bi", rotation_w2c, translation_c2w)
    return rotation_c2w, translation_c2w, rotation_w2c, translation_w2c


def transform_gt_state_to_slam(
    geometry: ManoGeometry,
    gt_state: torch.Tensor,
    rotation_gt_w2c: torch.Tensor,
    translation_gt_w2c: torch.Tensor,
    rotation_slam_c2w: torch.Tensor,
    translation_slam_c2w: torch.Tensor,
) -> torch.Tensor:
    """Apply T_slam_c * T_c_gt to HOT3D world-space MANO labels per frame."""
    rotation_gt_to_slam = torch.einsum(
        "tij,tjk->tik", rotation_slam_c2w, rotation_gt_w2c)
    translation_gt_to_slam = (
        torch.einsum("tij,tj->ti", rotation_slam_c2w, translation_gt_w2c)
        + translation_slam_c2w)
    # MANO's translation parameter is offset from the wrist pivot. Transform the
    # actual pivot through the shared geometry helper to preserve joint positions.
    return geometry.camera_to_world_state(
        gt_state, rotation_gt_to_slam, translation_gt_to_slam)


def ground_truth_valid(annotations):
    """Use explicit availability; zero axis-angle is a valid identity rotation."""
    if "valid_r" in annotations:
        valid = torch.as_tensor(annotations["valid_r"]).bool().cpu().numpy()
        if valid.shape != (len(annotations["rot_r"]),):
            raise ValueError("Right-hand validity mask has the wrong frame count")
        return valid
    return torch.any(annotations["rot_r"] != 0, dim=-1).cpu().numpy()


def main() -> None:
    args = parse_args()
    with (args.video_root / args.set_file).open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    if args.video:
        selected = set(args.video)
        sequences = [sequence for sequence in sequences if sequence in selected]
    device = torch.device(args.device)
    geometry = ManoGeometry(REPOSITORY_ROOT).to(device).eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        sequence_root = args.video_root / sequence
        raw_export_path = sequence_root / "raw_export_manifest.json"
        raw_export = json.loads(raw_export_path.read_text()) if raw_export_path.exists() else {}
        diagnostic_export = bool(raw_export.get("diagnostic", False))
        if diagnostic_export and not args.allow_diagnostic_export:
            raise ValueError(f"{sequence}: diagnostic raw export is not a formal training input")
        images = sorted((sequence_root / "extracted_images").glob("*.jpg"))
        frontend, track_provenance = export_hawor_frontend(
            sequence_root,
            args.frontend_log_root / sequence,
            allow_gt_boxes=args.allow_gt_boxes,
            return_provenance=True,
        )
        frame_count = len(images)
        rotation_c2w, translation_c2w, rotation_w2c, translation_w2c = [
            value.to(device) for value in
            load_slam_cameras(args, sequence, frame_count)]
        visual_state = np.full((frame_count, STATE_DIM), np.nan, dtype=np.float32)
        visual_ids = np.flatnonzero(frontend["valid"])
        with torch.inference_mode():
            for offset in range(0, len(visual_ids), 256):
                ids = visual_ids[offset:offset + 256]
                camera_state = pack_state(
                    torch.from_numpy(frontend["camera_translation"][ids]).to(device),
                    torch.from_numpy(frontend["global_orient"][ids]).to(device),
                    torch.from_numpy(frontend["hand_pose"][ids]).to(device),
                    torch.from_numpy(frontend["betas"][ids]).to(device),
                ).float()
                world_state = geometry.camera_to_world_state(
                    camera_state, rotation_c2w[ids], translation_c2w[ids])
                visual_state[ids] = world_state.cpu().numpy()

        annotations = joblib.load(sequence_root / "anno.pth")
        gt_valid = ground_truth_valid(annotations)
        gt_root = axis_angle_to_matrix(annotations["rot_r"]).unsqueeze(1)
        gt_hand = axis_angle_to_matrix(
            annotations["pose_r"].reshape(-1, 15, 3))
        gt_state = pack_state(
            annotations["trans_r"], gt_root, gt_hand, annotations["betas_r"])
        gt_rotation_w2c, gt_translation_w2c, _, _ = load_gt_cam(
            str(args.video_root), sequence)
        if len(gt_state) != frame_count or len(gt_rotation_w2c) != frame_count:
            raise ValueError(
                f"Ground-truth frame count mismatch for {sequence}: "
                f"state={len(gt_state)}, camera={len(gt_rotation_w2c)}, "
                f"images={frame_count}")
        gt_state = transform_gt_state_to_slam(
            geometry,
            gt_state.to(device),
            gt_rotation_w2c.to(device),
            gt_translation_w2c.to(device),
            rotation_c2w,
            translation_c2w,
        ).cpu().numpy().astype(np.float32)
        gt_state[~gt_valid] = np.nan
        timestamp_ns, timestamp_sec = load_timestamps(sequence_root, frame_count)
        segment_id = np.arange(frame_count, dtype=np.int32) // args.interval
        input_provenance = {
            "raw_export_manifest": str(raw_export_path) if raw_export else None,
            "raw_export_diagnostic": diagnostic_export,
            "visual_frontend": "HaWoR chunk JSON",
            "frontend_result_dir": str(args.frontend_log_root / sequence),
            **track_provenance,
            "gt_labels": "HOT3D anno.pth; training supervision only",
            "runtime_uses_gt_labels": False,
            "slam_camera_source": (
                "raw preprocess/SLAM/hawor_slam_w_scale_<start>_<stop>.npz"),
            "coordinate_frame": "per-interval local raw HaWoR SLAM world",
            "timestamp_source": "tt_list.pkl integer nanoseconds converted to seconds",
            "segment_id_semantics": (
                f"frame_id // {args.interval}; different SLAM segments are not directly comparable"),
        }
        np.savez_compressed(
            args.output_root / f"{sequence}.npz",
            sequence=np.asarray(sequence),
            frame_id=np.arange(frame_count, dtype=np.int32),
            timestamp_ns=timestamp_ns,
            timestamp_sec=timestamp_sec,
            segment_id=segment_id,
            visual_state=visual_state,
            gt_state=gt_state,
            visual_valid=frontend["valid"],
            gt_valid=gt_valid,
            rotation_w2c=rotation_w2c.cpu().numpy(),
            translation_w2c=translation_w2c.cpu().numpy(),
            focal_length=frontend["focal_length"],
            image_size=frontend["image_size"],
            image_paths=np.asarray([str(path) for path in images]),
            coordinate_frame=np.asarray("per-interval local raw HaWoR SLAM world"),
            position_unit=np.asarray("meter"),
            timestamp_source=np.asarray("tt_list.pkl integer nanoseconds"),
            segment_interval_frames=np.asarray(args.interval, dtype=np.int32),
            oracle_assisted=np.asarray(
                track_provenance["oracle_assisted"], dtype=np.bool_),
            oracle_assisted_status=np.asarray(
                track_provenance["oracle_assisted_status"]),
            input_provenance_json=np.asarray(
                json.dumps(input_provenance, sort_keys=True)),
        )
        print(f"[done] {sequence}: {int(frontend['valid'].sum())}/{frame_count} visual frames")


if __name__ == "__main__":
    main()

