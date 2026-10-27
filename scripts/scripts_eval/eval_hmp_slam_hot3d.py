"""Evaluate HMP + vanilla DROID-SLAM with the HaWoR HOT3D protocol."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import matplotlib
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

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
    parser = argparse.ArgumentParser(description="Evaluate HMP-SLAM on HOT3D")
    parser.add_argument("--video_root", type=Path,
                        default=ROOT / "datasets" / "hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--prediction_root", type=Path,
                        default=ROOT / "eval_log_hmp_slam")
    parser.add_argument("--eval_log_dir", type=Path,
                        default=ROOT / "eval_log_hmp_slam")
    parser.add_argument("--video", action="append")
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--virtual_focal", type=float, default=5000.0,
                        help="Virtual focal length used by official HMP/PyMAF-X")
    parser.add_argument("--no_focal_correction", action="store_true",
                        help="Keep HMP's virtual-focal z translation (diagnostic only)")
    parser.add_argument("--no_frame_npz", action="store_true")
    parser.add_argument("--overwrite_conversion", action="store_true")
    return parser.parse_args()


def load_sequences(args: argparse.Namespace) -> list[str]:
    with (args.video_root / args.set_file).open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    if args.video:
        selected = set(args.video)
        unknown = sorted(selected - set(sequences))
        if unknown:
            raise ValueError(f"Sequences not in {args.set_file}: {unknown}")
        sequences = [sequence for sequence in sequences if sequence in selected]
    return sequences


def save_metrics_over_time(
    output_dir: Path,
    sequence: str,
    frame_ids: np.ndarray,
    accel_frame_ids: np.ndarray,
    values: dict[str, np.ndarray],
) -> Path:
    """Save per-frame HMP-SLAM errors using the HaWoR evaluation layout."""
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

    fig.suptitle(f"HMP-SLAM HOT3D evaluation: {sequence}")
    fig.tight_layout()
    output_path = output_dir / "metrics_over_time.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def evaluation_mask(sequence_root: Path) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.load(
        sequence_root / "preprocess" / "preprocess_gtdet_tracks.npy",
        allow_pickle=True,
    ).item()
    annotations = joblib.load(sequence_root / "anno.pth")
    mano_valid = torch.any(annotations["rot_r"] != 0, dim=-1).cpu().numpy()
    right_track = tracks[1]
    valid = np.asarray([item["det"] for item in right_track], dtype=bool) & mano_valid
    frame_ids = np.flatnonzero(valid)
    boxes = np.concatenate([right_track[i]["det_box"][:, :4] for i in frame_ids], axis=0)
    return valid, boxes.astype(np.float32)


def expand_betas(raw_betas: np.ndarray, length: int) -> np.ndarray:
    betas = np.asarray(raw_betas, dtype=np.float32).reshape(-1, 10)
    if len(betas) == 1:
        return np.repeat(betas, length, axis=0)
    if len(betas) == length:
        return betas
    # Official HMP optimizes one shape per subject.  Older checkpoints can
    # serialize one copy per temporal chunk; their mean is the same subject.
    return np.repeat(betas.mean(axis=0, keepdims=True), length, axis=0)


def empty_frame_payload(frame_id: int, image_size: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "frame_id": np.asarray(frame_id, dtype=np.int32),
        "valid": np.asarray(False),
        "handedness": np.empty((0,), dtype=np.int8),
        "bbox": np.empty((0, 4), dtype=np.float32),
        "vertices_3d": np.empty((0, 778, 3), dtype=np.float32),
        "joints_3d": np.empty((0, 21, 3), dtype=np.float32),
        "global_orient": np.empty((0, 1, 3, 3), dtype=np.float32),
        "hand_pose": np.empty((0, 15, 3, 3), dtype=np.float32),
        "betas": np.empty((0, 10), dtype=np.float32),
        "pred_cam": np.empty((0, 3), dtype=np.float32),
        "camera_translation": np.empty((0, 3), dtype=np.float32),
        "focal_length": np.empty((0,), dtype=np.float32),
        "image_size": image_size.astype(np.int32),
    }


def convert_hmp_prediction(args: argparse.Namespace, sequence: str,
                           image_paths: list[Path]) -> Path:
    sequence_root = args.video_root / sequence
    output_root = args.prediction_root / sequence
    compact_path = output_root / "hmp_predictions.npz"
    if compact_path.exists() and not args.overwrite_conversion:
        return compact_path

    raw_path = output_root / "hmp_optim" / "recon_000_30fps.npz"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing official HMP output: {raw_path}")
    raw = np.load(raw_path, allow_pickle=True)
    total_frames = len(image_paths)
    poses = np.asarray(raw["poses"], dtype=np.float32).reshape(-1, 16, 3)
    translation_raw = np.asarray(raw["trans"], dtype=np.float32).reshape(-1, 3)
    vertices = np.asarray(raw["vertices"], dtype=np.float32).reshape(-1, 778, 3)
    joints = np.asarray(raw["joints_3d"], dtype=np.float32).reshape(-1, 21, 3)
    lengths = {len(poses), len(translation_raw), len(vertices), len(joints), total_frames}
    if len(lengths) != 1:
        raise ValueError(f"HMP/image length mismatch for {sequence}: {sorted(lengths)}")

    valid, valid_boxes = evaluation_mask(sequence_root)
    boxes = np.full((total_frames, 4), np.nan, dtype=np.float32)
    boxes[np.flatnonzero(valid)] = valid_boxes
    with (sequence_root / "focal.txt").open("r", encoding="utf-8") as handle:
        focal = float(handle.read())
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise OSError(f"Cannot read {image_paths[0]}")
    image_size = np.asarray([first_image.shape[1], first_image.shape[0]], dtype=np.int32)

    # PyMAF-X/HMP reconstructs z using a virtual 5000 px focal length.  The
    # full-image x/y conversion is focal-independent; only z must be converted
    # to the calibrated HOT3D focal before metric world-space evaluation.
    translation = translation_raw.copy()
    if not args.no_focal_correction:
        translation[:, 2] *= focal / args.virtual_focal
    delta = translation - translation_raw
    vertices = vertices + delta[:, None, :]
    joints = joints + delta[:, None, :]

    rotations = axis_angle_to_matrix(torch.from_numpy(poses)).numpy().astype(np.float32)
    global_orient = rotations[:, :1]
    hand_pose = rotations[:, 1:]
    betas = expand_betas(raw["betas"], total_frames)
    pred_cam = np.full((total_frames, 3), np.nan, dtype=np.float32)
    focal_lengths = np.full((total_frames,), focal, dtype=np.float32)
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        compact_path,
        frame_id=np.arange(total_frames, dtype=np.int32),
        valid=valid,
        handedness=np.ones((total_frames,), dtype=np.int8),
        bbox=boxes,
        vertices_3d=vertices,
        joints_3d=joints,
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=betas,
        pred_cam=pred_cam,
        camera_translation=translation,
        focal_length=focal_lengths,
        image_size=image_size,
    )

    if not args.no_frame_npz:
        frame_root = output_root / "hmp_frames"
        frame_root.mkdir(parents=True, exist_ok=True)
        for frame_id in range(total_frames):
            destination = frame_root / f"{frame_id:04d}.npz"
            if not valid[frame_id]:
                np.savez_compressed(destination, **empty_frame_payload(frame_id, image_size))
                continue
            np.savez_compressed(
                destination,
                frame_id=np.asarray(frame_id, dtype=np.int32),
                valid=np.asarray(True),
                handedness=np.asarray([1], dtype=np.int8),
                bbox=boxes[frame_id][None],
                vertices_3d=vertices[frame_id][None],
                joints_3d=joints[frame_id][None],
                global_orient=global_orient[frame_id][None],
                hand_pose=hand_pose[frame_id][None],
                betas=betas[frame_id][None],
                # HMP has no final weak-perspective camera after optimization.
                pred_cam=pred_cam[frame_id][None],
                camera_translation=translation[frame_id][None],
                focal_length=focal_lengths[frame_id:frame_id + 1],
                image_size=image_size,
            )
    metadata = {
        "sequence": sequence,
        "source": str(raw_path),
        "total_frames": total_frames,
        "valid_right_hand_frames": int(valid.sum()),
        "virtual_focal_px": args.virtual_focal,
        "calibrated_focal_px": focal,
        "translation_z_focal_corrected": not args.no_focal_correction,
        "pred_cam": "NaN: HMP optimizes perspective translation and has no final weak-perspective camera",
    }
    with (output_root / "hmp_inference.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return compact_path


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
            raise FileNotFoundError(f"Missing shared vanilla DROID-SLAM camera: {slam_path}")
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
    prediction_path = convert_hmp_prediction(args, sequence, image_paths)
    predictions = np.load(prediction_path)
    valid, valid_boxes = evaluation_mask(sequence_root)
    if not np.array_equal(valid, predictions["valid"]):
        raise ValueError(f"Evaluation validity mask differs from HMP input for {sequence}")

    camera_rotation, camera_translation = load_sequence_cameras(args, sequence, image_paths)
    annotations = joblib.load(sequence_root / "anno.pth")
    world_trans = annotations["trans_r"].unsqueeze(0)
    world_rot = annotations["rot_r"].unsqueeze(0)
    world_pose = annotations["pose_r"].unsqueeze(0)
    world_betas = annotations["betas_r"].unsqueeze(0)
    frame_ids = np.flatnonzero(valid)
    frame_chunks, _ = parse_chunks(frame_ids, valid_boxes, min_len=16)

    all_w, all_wa, all_pa, all_accel, all_rte = [], [], [], [], []
    all_eval_ids, all_accel_ids = [], []
    pred_world_joints = np.full((len(image_paths), 21, 3), np.nan, dtype=np.float32)
    gt_world_joints = np.full((len(image_paths), 21, 3), np.nan, dtype=np.float32)

    for chunk in frame_chunks:
        data_camera = {
            "init_root_orient": torch.from_numpy(predictions["global_orient"][chunk, 0]).float().unsqueeze(0),
            "init_hand_pose": torch.from_numpy(predictions["hand_pose"][chunk]).float().unsqueeze(0),
            "init_trans": torch.from_numpy(predictions["camera_translation"][chunk]).float().unsqueeze(0),
            "init_betas": torch.from_numpy(predictions["betas"][chunk]).float().unsqueeze(0),
        }
        data_world = cam2world_convert(
            camera_rotation[chunk], camera_translation[chunk], data_camera, "right")
        predicted = run_mano(
            data_world["init_trans"], data_world["init_root_orient"],
            data_world["init_hand_pose"], betas=data_world["init_betas"])["joints"][0]
        target = run_mano(
            world_trans[:, chunk], world_rot[:, chunk], world_pose[:, chunk],
            betas=world_betas[:, chunk])["joints"][0]
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
        all_accel.append(compute_error_accel(
            joints_pred=predicted.cpu(), joints_gt=target.cpu())[1:-1] * (30 ** 2))
        all_eval_ids.append(chunk)
        all_accel_ids.append(chunk[2:-2])
        all_rte.append(compute_rte(
            world_trans[:, chunk].squeeze(0),
            data_world["init_trans"].squeeze(0)) * 100.0)

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
        "method": "HMP-SLAM (HMP + vanilla DROID-SLAM + Metric3D)",
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
    axis.plot(*pred_world_joints[:, 0].T, label="HMP-SLAM wrist", linewidth=1.0)
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
    public_results = [{key: value for key, value in result.items() if key != "_values"}
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
            "Evaluation on hot3d HMP-SLAM, W_MPJPE, WA_MPJPE, PA_MPJPE, "
            "ACCEL, RTE, "
            f"{aggregate['w_mpjpe_mm']:.2f} {aggregate['wa_mpjpe_mm']:.2f} "
            f"{aggregate['pa_mpjpe_mm']:.2f} {aggregate['accel_m_s2']:.2f} "
            f"{aggregate['rte_percent']:.2f}\n"
        )
    print(f"[done] aggregate summary: {summary_path}")


if __name__ == "__main__":
    main()
