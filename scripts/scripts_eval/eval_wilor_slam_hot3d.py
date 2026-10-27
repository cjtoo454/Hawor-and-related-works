"""Evaluate WiLoR + vanilla DROID-SLAM on HOT3D using HaWoR's metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hawor.utils.process import run_mano  # noqa: E402
from lib.eval_utils.custom_utils import (  # noqa: E402
    algin_cam_traj_wo_scale,
    cam2world_convert,
    load_gt_cam,
    load_slam_cam,
    split_list_by_interval,
)
from lib.eval_utils.eval_utils import (  # noqa: E402
    batch_compute_similarity_transform_torch,
    compute_error_accel,
    compute_jpe,
    compute_rte,
    first_align_joints,
    global_align_joints,
)
from lib.pipeline.tools import parse_chunks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WiLoR-SLAM on HOT3D")
    parser.add_argument("--video_root", type=Path,
                        default=ROOT / "datasets" / "hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--prediction_root", type=Path,
                        default=ROOT / "eval_log_wilor_slam")
    parser.add_argument("--eval_log_dir", type=Path,
                        default=ROOT / "eval_log_wilor_slam")
    parser.add_argument("--video", action="append")
    parser.add_argument("--interval", type=int, default=1000)
    return parser.parse_args()


def load_sequences(args: argparse.Namespace) -> list[str]:
    with (args.video_root / args.set_file).open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    if args.video:
        selected = set(args.video)
        unknown = sorted(selected - set(sequences))
        if unknown:
            raise ValueError(f"Sequences not in {args.set_file}: {unknown}")
        sequences = [name for name in sequences if name in selected]
    return sequences


def save_metrics_over_time(
    output_dir: Path,
    sequence: str,
    frame_ids: np.ndarray,
    accel_frame_ids: np.ndarray,
    values: dict[str, np.ndarray],
) -> Path:
    """Save per-frame WiLoR-SLAM errors using the HaWoR evaluation layout."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(
        frame_ids, values["w_mpjpe_mm"], label="W-MPJPE", linewidth=0.8)
    axes[0].plot(
        frame_ids, values["wa_mpjpe_mm"], label="WA-MPJPE", linewidth=0.8)
    axes[0].plot(
        frame_ids, values["pa_mpjpe_mm"], label="PA-MPJPE", linewidth=0.8)
    axes[0].set_ylabel("Joint error (mm)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        accel_frame_ids, values["accel_m_s2"], color="tab:orange", linewidth=0.8)
    axes[1].set_ylabel("Accel. error (m/s²)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        frame_ids, values["rte_percent"], color="tab:green", linewidth=0.8)
    axes[2].set_ylabel("RTE (%)")
    axes[2].set_xlabel("Frame")
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"WiLoR-SLAM HOT3D evaluation: {sequence}")
    fig.tight_layout()
    output_path = output_dir / "metrics_over_time.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def load_sequence_cameras(args: argparse.Namespace, sequence: str,
                          image_paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    c2w_rotations, c2w_translations = [], []
    starts, ends, parts = split_list_by_interval(image_paths, interval=args.interval)
    for start, end, _ in zip(starts, ends, parts):
        slam_path = (
            args.video_root / sequence / "preprocess" / "SLAM_WiLoR" /
            f"wilor_droid_w_scale_{start}_{end}.npz"
        )
        if not slam_path.exists():
            raise FileNotFoundError(f"Missing WiLoR-SLAM camera file: {slam_path}")
        _, _, rotation_c2w, translation_c2w = load_slam_cam(str(slam_path))
        _, _, gt_rotation_c2w, gt_translation_c2w = load_gt_cam(
            str(args.video_root), sequence, start, end)
        _, _, rotation_c2w, translation_c2w = algin_cam_traj_wo_scale(
            rotation_c2w, translation_c2w, gt_rotation_c2w, gt_translation_c2w)
        c2w_rotations.append(rotation_c2w)
        c2w_translations.append(translation_c2w)
    return torch.cat(c2w_rotations), torch.cat(c2w_translations)


def evaluate_sequence(args: argparse.Namespace, sequence: str) -> dict[str, object]:
    sequence_root = args.video_root / sequence
    output_dir = args.eval_log_dir / sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted((sequence_root / "extracted_images").glob("*.jpg"))
    prediction_path = args.prediction_root / sequence / "wilor_predictions.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing WiLoR predictions: {prediction_path}")
    predictions = np.load(prediction_path)
    if len(predictions["valid"]) != len(image_paths):
        raise ValueError(f"Prediction/image length mismatch for {sequence}")

    tracks = np.load(
        sequence_root / "preprocess" / "preprocess_gtdet_tracks.npy",
        allow_pickle=True,
    ).item()
    annotations = joblib.load(sequence_root / "anno.pth")
    mano_valid = torch.any(annotations["rot_r"] != 0, dim=-1).cpu().numpy()
    right_track = tracks[1]
    track_valid = np.asarray([item["det"] for item in right_track], dtype=bool) & mano_valid
    if not np.array_equal(track_valid, predictions["valid"]):
        raise ValueError(f"Evaluation validity mask differs from inference for {sequence}")

    camera_rotation, camera_translation = load_sequence_cameras(args, sequence, image_paths)
    if len(camera_rotation) != len(image_paths):
        raise ValueError(f"SLAM/image length mismatch for {sequence}")

    world_trans = annotations["trans_r"].unsqueeze(0)
    world_rot = annotations["rot_r"].unsqueeze(0)
    world_pose = annotations["pose_r"].unsqueeze(0)
    world_betas = annotations["betas_r"].unsqueeze(0)

    frame_ids = np.flatnonzero(track_valid)
    boxes = np.concatenate([right_track[i]["det_box"] for i in frame_ids], axis=0)
    frame_chunks, _ = parse_chunks(frame_ids, boxes, min_len=16)

    all_w, all_wa, all_pa, all_accel, all_rte = [], [], [], [], []
    all_eval_ids, all_accel_ids = [], []
    pred_world_joints = np.full((len(image_paths), 21, 3), np.nan, dtype=np.float32)
    gt_world_joints = np.full((len(image_paths), 21, 3), np.nan, dtype=np.float32)

    for chunk in frame_chunks:
        root_orient = torch.from_numpy(predictions["global_orient"][chunk]).float()
        root_orient = root_orient[:, 0]  # (T, 3, 3)
        hand_pose = torch.from_numpy(predictions["hand_pose"][chunk]).float()
        betas = torch.from_numpy(predictions["betas"][chunk]).float()
        hand_translation = torch.from_numpy(predictions["camera_translation"][chunk]).float()
        data_camera = {
            "init_root_orient": root_orient.unsqueeze(0),
            "init_hand_pose": hand_pose.unsqueeze(0),
            "init_trans": hand_translation.unsqueeze(0),
            "init_betas": betas.unsqueeze(0),
        }
        data_world = cam2world_convert(
            camera_rotation[chunk], camera_translation[chunk], data_camera, "right")
        pred_output = run_mano(
            data_world["init_trans"],
            data_world["init_root_orient"],
            data_world["init_hand_pose"],
            betas=data_world["init_betas"],
        )
        predicted = pred_output["joints"][0]

        gt_output = run_mano(
            world_trans[:, chunk],
            world_rot[:, chunk],
            world_pose[:, chunk],
            betas=world_betas[:, chunk],
        )
        target = gt_output["joints"][0]
        pred_world_joints[chunk] = predicted.detach().cpu().numpy()
        gt_world_joints[chunk] = target.detach().cpu().numpy()

        predicted_pa = batch_compute_similarity_transform_torch(predicted, target)
        chunk_w, chunk_wa = [], []
        for start in range(0, len(chunk), 100):
            end = min(start + 100, len(chunk))
            target_part = target[start:end].cpu()
            predicted_part = predicted[start:end].cpu()
            chunk_w.append(compute_jpe(
                target_part, first_align_joints(target_part, predicted_part)))
            chunk_wa.append(compute_jpe(
                target_part, global_align_joints(target_part, predicted_part)))
        all_w.append(np.concatenate(chunk_w) * 1000.0)
        all_wa.append(np.concatenate(chunk_wa) * 1000.0)
        all_pa.append(compute_jpe(target.cpu(), predicted_pa.cpu()) * 1000.0)

        acceleration = compute_error_accel(
            joints_pred=predicted.cpu(), joints_gt=target.cpu())[1:-1] * (30 ** 2)
        all_accel.append(acceleration)
        all_eval_ids.append(chunk)
        all_accel_ids.append(chunk[2:-2])
        all_rte.append(compute_rte(
            world_trans[:, chunk].squeeze(0),
            data_world["init_trans"].squeeze(0),
        ) * 100.0)

    if not all_eval_ids:
        raise RuntimeError(f"No valid evaluation chunks for {sequence}")
    eval_ids = np.concatenate(all_eval_ids)
    accel_ids = np.concatenate(all_accel_ids)
    values = {
        "w_mpjpe_mm": np.concatenate(all_w),
        "wa_mpjpe_mm": np.concatenate(all_wa),
        "pa_mpjpe_mm": np.concatenate(all_pa),
        "accel_m_s2": np.concatenate(all_accel),
        "rte_percent": np.concatenate(all_rte),
    }
    np.savez(
        output_dir / "evaluation_details.npz",
        frame_ids=eval_ids,
        accel_frame_ids=accel_ids,
        pred_world_joints_m=pred_world_joints,
        gt_world_joints_m=gt_world_joints,
        **values,
    )
    summary: dict[str, object] = {
        "video": sequence,
        "method": "WiLoR-SLAM (WiLoR + vanilla DROID-SLAM + Metric3D)",
        "total_frames": len(image_paths),
        "evaluated_right_hand_frames": len(eval_ids),
        **{key: float(value.mean()) for key, value in values.items()},
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    save_metrics_over_time(output_dir, sequence, eval_ids, accel_ids, values)

    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(*gt_world_joints[:, 0].T, label="HOT3D GT wrist", linewidth=1.2)
    axis.plot(*pred_world_joints[:, 0].T, label="WiLoR-SLAM wrist", linewidth=1.0)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(f"Right-hand world trajectory: {sequence}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "world_trajectory.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return {**summary, "_values": values}


def main() -> None:
    args = parse_args()
    sequences = load_sequences(args)
    args.eval_log_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_sequence(args, sequence) for sequence in sequences]

    accumulator: dict[str, list[np.ndarray]] = defaultdict(list)
    for result in results:
        for key, value in result["_values"].items():
            accumulator[key].append(value)
    aggregate = {
        "num_videos": len(results),
        "total_frames": sum(int(result["total_frames"]) for result in results),
        "evaluated_right_hand_frames": sum(
            int(result["evaluated_right_hand_frames"]) for result in results),
    }
    for key, arrays in accumulator.items():
        aggregate[key] = float(np.concatenate(arrays).mean())
    public_results = [{k: v for k, v in result.items() if k != "_values"}
                      for result in results]
    summary = {"aggregate": aggregate, "videos": public_results}
    summary_path = args.eval_log_dir / f"evaluation_summary_{len(results)}videos.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (args.eval_log_dir / "eval_log.txt").open("w", encoding="utf-8") as handle:
        handle.write(str(sequences) + "\n")
        for result in public_results:
            handle.write(
                f"{result['video']} {result['w_mpjpe_mm']} {result['wa_mpjpe_mm']}\n")
        handle.write(
            "Evaluation on hot3d WiLoR-SLAM, W_MPJPE, WA_MPJPE, PA_MPJPE, "
            "ACCEL, RTE, "
            f"{aggregate['w_mpjpe_mm']:.2f} {aggregate['wa_mpjpe_mm']:.2f} "
            f"{aggregate['pa_mpjpe_mm']:.2f} {aggregate['accel_m_s2']:.2f} "
            f"{aggregate['rte_percent']:.2f}\n"
        )
    print(f"[done] aggregate summary: {summary_path}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
