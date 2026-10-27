#!/usr/bin/env python3
"""Evaluate dual-loop predictions with the same HOT3D protocol as all baselines."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.geometry import ManoGeometry  # noqa: E402
from dual_loop.state import pack_state, axis_angle_to_matrix  # noqa: E402
from dual_loop.io import load_slam_c2w  # noqa: E402
from lib.eval_utils.custom_utils import (  # noqa: E402
    algin_cam_traj_wo_scale,
    load_gt_cam,
)
from lib.eval_utils.eval_utils import (  # noqa: E402
    batch_compute_similarity_transform_torch,
    compute_error_accel,
    compute_jpe,
    compute_rte,
    first_align_joints,
    global_align_joints,
)

def parse_chunks(frame, boxes, min_len=16):
    """Same contiguous-frame/minimum-length policy as the baseline evaluator."""
    split = np.flatnonzero(np.diff(frame) != 1) + 1
    frame_chunks = np.split(frame, split)
    box_chunks = np.split(boxes, split)
    keep = [i for i, item in enumerate(frame_chunks) if len(item) >= min_len]
    return [frame_chunks[i] for i in keep], [box_chunks[i] for i in keep]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dual-loop observer on HOT3D")
    parser.add_argument(
        "--video_root", type=Path,
        default=REPOSITORY_ROOT / "datasets" / "hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--video", action="append")
    parser.add_argument(
        "--prediction_root", type=Path, default=REPOSITORY_ROOT / "eval_log_try")
    parser.add_argument(
        "--eval_log_dir", type=Path, default=REPOSITORY_ROOT / "eval_log_try")
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--common_frame_mask", type=Path,
                        help="Optional NPZ with one boolean array per sequence, shared by all methods")
    parser.add_argument("--allow_untrained", action="store_true",
                        help="Diagnostic only: permit evaluation of a format smoke test")
    return parser.parse_args()


def load_sequences(args: argparse.Namespace) -> list[str]:
    with (args.video_root / args.set_file).open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    if args.video:
        selected = set(args.video)
        sequences = [sequence for sequence in sequences if sequence in selected]
    return sequences


def load_aligned_cameras(args: argparse.Namespace, sequence: str, frame_count: int):
    rotations, translations = [], []
    for start in range(0, frame_count, args.interval):
        stop = min(start + args.interval, frame_count)
        slam_path = (
            args.video_root / sequence / "preprocess" / "SLAM" /
            f"hawor_slam_w_scale_{start}_{stop}.npz")
        rotation, translation = load_slam_c2w(slam_path)
        _, _, gt_rotation, gt_translation = load_gt_cam(
            str(args.video_root), sequence, start, stop)
        _, _, rotation, translation = algin_cam_traj_wo_scale(
            rotation, translation, gt_rotation, gt_translation)
        rotations.append(rotation)
        translations.append(translation)
    return torch.cat(rotations), torch.cat(translations)


def save_metric_plot(output_dir: Path, sequence: str, ids, accel_ids, values) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for key, label in (
        ("w_mpjpe_mm", "W-MPJPE"),
        ("wa_mpjpe_mm", "WA-MPJPE"),
        ("pa_mpjpe_mm", "PA-MPJPE"),
    ):
        axes[0].plot(ids, values[key], label=label, linewidth=0.8)
    axes[0].set_ylabel("Joint error (mm)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(accel_ids, values["accel_m_s2"], color="tab:orange", linewidth=0.8)
    axes[1].set_ylabel("Accel. error (m/s²)")
    axes[1].grid(alpha=0.25)
    axes[2].plot(ids, values["rte_percent"], color="tab:green", linewidth=0.8)
    axes[2].set_ylabel("RTE (%)")
    axes[2].set_xlabel("Frame")
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"Dual-loop HOT3D evaluation: {sequence}")
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_over_time.png", dpi=160)
    plt.close(fig)


def evaluate_sequence(args: argparse.Namespace, sequence: str) -> dict[str, object]:
    sequence_root = args.video_root / sequence
    output_dir = args.eval_log_dir / sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.prediction_root / sequence / "try_predictions.npz"
    metadata_path = prediction_path.parent / "try_inference.json"
    if not metadata_path.exists():
        raise ValueError(f"Missing inference provenance: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if not metadata.get("trained_checkpoint_loaded") and not args.allow_untrained:
        raise ValueError("Untrained observer output cannot be evaluated as proposed-method results")
    with np.load(prediction_path) as archive:
        prediction = {key: archive[key].copy() for key in archive.files}
    frame_count = len(prediction["frame_id"])
    annotations = joblib.load(sequence_root / "anno.pth")
    mano_valid = torch.any(annotations["rot_r"] != 0, dim=-1).cpu().numpy()
    valid = prediction["valid"].astype(bool) & mano_valid
    if args.common_frame_mask:
        with np.load(args.common_frame_mask, allow_pickle=False) as masks:
            common = masks[sequence].astype(bool)
        if common.shape != valid.shape:
            raise ValueError(f"Common-frame mask shape mismatch: {sequence}")
        valid &= common
    frame_ids = np.flatnonzero(valid)
    boxes = prediction["bbox"][frame_ids]
    chunks, _ = parse_chunks(frame_ids, boxes, min_len=16)
    rotation_c2w, translation_c2w = load_aligned_cameras(
        args, sequence, frame_count)
    geometry = ManoGeometry(REPOSITORY_ROOT).to(args.device).eval()

    all_w, all_wa, all_pa, all_accel, all_rte = [], [], [], [], []
    all_ids, all_accel_ids = [], []
    pred_world = np.full((frame_count, 21, 3), np.nan, dtype=np.float32)
    gt_world = np.full((frame_count, 21, 3), np.nan, dtype=np.float32)
    for chunk in chunks:
        tensor = lambda value: torch.as_tensor(value, device=args.device, dtype=torch.float32)
        with torch.inference_mode():
            camera_state = pack_state(
                tensor(prediction["camera_translation"][chunk]),
                tensor(prediction["global_orient"][chunk]),
                tensor(prediction["hand_pose"][chunk]), tensor(prediction["betas"][chunk]))
            world_state = geometry.camera_to_world_state(
                camera_state, rotation_c2w[chunk].to(args.device), translation_c2w[chunk].to(args.device))
            gt_state = pack_state(
                tensor(annotations["trans_r"][chunk]),
                axis_angle_to_matrix(tensor(annotations["rot_r"][chunk]))[:, None],
                axis_angle_to_matrix(tensor(annotations["pose_r"][chunk]).reshape(-1,15,3)),
                tensor(annotations["betas_r"][chunk]))
            predicted = geometry.joints(world_state).cpu()
            target = geometry.joints(gt_state).cpu()
        pred_world[chunk] = predicted.detach().cpu().numpy()
        gt_world[chunk] = target.detach().cpu().numpy()
        predicted_pa = batch_compute_similarity_transform_torch(predicted, target)
        w_parts, wa_parts = [], []
        for start in range(0, len(chunk), 100):
            stop = min(start + 100, len(chunk))
            gt_part = target[start:stop].cpu()
            pred_part = predicted[start:stop].cpu()
            w_parts.append(compute_jpe(
                gt_part, first_align_joints(gt_part, pred_part)))
            wa_parts.append(compute_jpe(
                gt_part, global_align_joints(gt_part, pred_part)))
        all_w.append(np.concatenate(w_parts) * 1000.0)
        all_wa.append(np.concatenate(wa_parts) * 1000.0)
        all_pa.append(compute_jpe(target.cpu(), predicted_pa.cpu()) * 1000.0)
        all_accel.append(compute_error_accel(
            joints_pred=predicted.cpu(), joints_gt=target.cpu())[1:-1] * (30 ** 2))
        all_rte.append(compute_rte(
            annotations["trans_r"][chunk].cpu(), world_state[:, :3].cpu()) * 100.0)
        all_ids.append(chunk)
        all_accel_ids.append(chunk[2:-2])
    if not all_ids:
        raise RuntimeError(f"No valid evaluation chunks for {sequence}")
    ids = np.concatenate(all_ids)
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
        frame_ids=ids,
        accel_frame_ids=accel_ids,
        pred_world_joints_m=pred_world,
        gt_world_joints_m=gt_world,
        **values,
    )
    summary = {
        "video": sequence,
        "method": "Dual-Loop State Reconciliation (HaWoR front-end)",
        "total_frames": frame_count,
        "evaluated_right_hand_frames": len(ids),
        "gt_valid_right_hand_frames": int(mano_valid.sum()),
        "evaluated_coverage_of_gt": float(len(ids) / max(1, mano_valid.sum())),
        "prediction_protocol": "visual-valid frames only; prior-only gap frames excluded",
        "oracle_assisted": bool(metadata.get("oracle_assisted", False)),
        "untrained_diagnostic": not bool(metadata.get("trained_checkpoint_loaded")),
        **{key: float(value.mean()) for key, value in values.items()},
        "w_mpjpe_p95_mm": float(np.percentile(values["w_mpjpe_mm"], 95)),
        "accel_p95_m_s2": float(np.percentile(values["accel_m_s2"], 95)),
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    save_metric_plot(output_dir, sequence, ids, accel_ids, values)
    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(*gt_world[:, 0].T, label="HOT3D GT wrist", linewidth=1.2)
    axis.plot(*pred_world[:, 0].T, label="Ours wrist", linewidth=1.0)
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
    results = [evaluate_sequence(args, sequence) for sequence in load_sequences(args)]
    aggregate_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for result in results:
        for key, value in result["_values"].items():
            aggregate_arrays[key].append(value)
    aggregate = {
        "num_videos": len(results),
        "total_frames": sum(int(result["total_frames"]) for result in results),
        "evaluated_right_hand_frames": sum(
            int(result["evaluated_right_hand_frames"]) for result in results),
    }
    for key, arrays in aggregate_arrays.items():
        merged = np.concatenate(arrays)
        aggregate[key] = float(merged.mean())
        if key == "w_mpjpe_mm":
            aggregate["w_mpjpe_p95_mm"] = float(np.percentile(merged, 95))
        elif key == "accel_m_s2":
            aggregate["accel_p95_m_s2"] = float(np.percentile(merged, 95))
    public = [{key: value for key, value in result.items() if key != "_values"}
              for result in results]
    args.eval_log_dir.mkdir(parents=True, exist_ok=True)
    path = args.eval_log_dir / f"evaluation_summary_{len(results)}videos.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"aggregate": aggregate, "videos": public}, handle, indent=2)
    with (args.eval_log_dir / "eval_log.txt").open("w", encoding="utf-8") as handle:
        handle.write(str([result["video"] for result in public]) + "\n")
        for result in public:
            handle.write(
                f"{result['video']} {result['w_mpjpe_mm']} "
                f"{result['wa_mpjpe_mm']}\n")
        handle.write(
            "Evaluation on hot3d Dual-Loop, W_MPJPE, WA_MPJPE, PA_MPJPE, "
            "ACCEL, RTE, "
            f"{aggregate['w_mpjpe_mm']:.2f} {aggregate['wa_mpjpe_mm']:.2f} "
            f"{aggregate['pa_mpjpe_mm']:.2f} {aggregate['accel_m_s2']:.2f} "
            f"{aggregate['rte_percent']:.2f}\n")

    metric_specs = (
        ("w_mpjpe_mm", "W-MPJPE (mm)"),
        ("wa_mpjpe_mm", "WA-MPJPE (mm)"),
        ("pa_mpjpe_mm", "PA-MPJPE (mm)"),
        ("accel_m_s2", "Acceleration error (m/s²)"),
        ("rte_percent", "RTE (%)"),
    )
    labels = [result["video"] for result in public]
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(14, 15), sharex=True)
    for axis, (key, label) in zip(axes, metric_specs):
        axis.bar(np.arange(len(public)), [result[key] for result in public])
        axis.axhline(
            aggregate[key], color="tab:red", linestyle="--", linewidth=1,
            label=f"aggregate: {aggregate[key]:.2f}")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xticks(
        np.arange(len(public)), labels, rotation=30, ha="right")
    fig.suptitle(f"Dual-loop HOT3D evaluation across {len(public)} videos")
    fig.tight_layout()
    fig.savefig(
        args.eval_log_dir / f"metrics_{len(public)}videos.png", dpi=160)
    plt.close(fig)
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
