"""Render sampled GT and predicted MANO hands together in world space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GT_COLOR = "#20BFE3"
PRED_COLOR = "#D45AC5"
GT_DARK = "#087F9D"
PRED_DARK = "#8F267F"
METHODS = {
    "hawor": {
        "label": "HaWoR",
        "log_dir": "eval_log",
        "prediction_file": None,
        "slam_dir": "SLAM",
        "slam_prefix": "hawor_slam_w_scale",
    },
    "hmp_slam": {
        "label": "HMP-SLAM",
        "log_dir": "eval_log_hmp_slam",
        "prediction_file": "hmp_predictions.npz",
        "slam_dir": "SLAM_WiLoR",
        "slam_prefix": "wilor_droid_w_scale",
    },
    "wilor_slam": {
        "label": "WiLoR-SLAM",
        "log_dir": "eval_log_wilor_slam",
        "prediction_file": "wilor_predictions.npz",
        "slam_dir": "SLAM_WiLoR",
        "slam_prefix": "wilor_droid_w_scale",
    },
}
HAND_BONES = tuple(
    (0, finger_start) if offset == 0
    else (finger_start + offset - 1, finger_start + offset)
    for finger_start in (1, 5, 9, 13, 17)
    for offset in range(4)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render predicted and HOT3D GT MANO meshes")
    parser.add_argument("--video", required=True)
    parser.add_argument("--method", choices=METHODS, default="hawor")
    parser.add_argument("--segment", type=int, default=1,
                        help="One-based segment index")
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for the new PNG (default: method result directory)")
    return parser.parse_args()


def equal_limits(points: np.ndarray, padding: float = 0.10) -> tuple[tuple[float, float], ...]:
    finite = points[np.isfinite(points).all(axis=1)]
    minimum = finite.min(axis=0)
    maximum = finite.max(axis=0)
    center = (minimum + maximum) / 2.0
    span = max(float((maximum - minimum).max()), 0.08)
    half_span = span * (0.5 + padding)
    return tuple((float(value - half_span), float(value + half_span))
                 for value in center)


def apply_axes(axis, limits: tuple[tuple[float, float], ...]) -> None:
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=22, azim=-58)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_facecolor("#F3F3F3")
    axis.grid(alpha=0.24)


def draw_hand(
    axis,
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    face_color: str,
    edge_color: str,
    alpha: float,
    label_joints: bool = False,
) -> None:
    mesh = Poly3DCollection(
        vertices[faces],
        facecolor=face_color,
        edgecolor="none",
        alpha=alpha,
        rasterized=True,
    )
    axis.add_collection3d(mesh)
    for first, second in HAND_BONES:
        bone = joints[[first, second]]
        axis.plot(*bone.T, color=edge_color, linewidth=1.5, alpha=0.95)
    axis.scatter(*joints.T, color=edge_color, s=13, depthshade=False, alpha=0.98)
    if label_joints:
        for joint_id, point in enumerate(joints):
            axis.text(*point, str(joint_id), color="#1A1A1A", fontsize=7)


def load_aligned_slam_cameras(
    video_root: Path,
    video: str,
    frame_count: int,
    slam_dir: str,
    slam_prefix: str,
):
    from scripts.scripts_eval.recovered_camera_cache import validate_recovered_camera_cache
    from lib.eval_utils.custom_utils import (
        algin_cam_traj_wo_scale,
        load_gt_cam,
        load_slam_cam,
    )

    segments = []
    for start in range(0, frame_count, 1000):
        stop = min(start + 1000, frame_count)
        path = (
            video_root / video / "preprocess" / slam_dir /
            f"{slam_prefix}_{start}_{stop}.npz"
        )
        segments.append((start, stop, path))
    if not all(path.is_file() for _, _, path in segments):
        if slam_dir == "SLAM_WiLoR":
            cache = video_root / video / "preprocess" / slam_dir / "recovered_aligned_cameras.npz"
            if cache.is_file():
                rotation, translation, _ = validate_recovered_camera_cache(cache, frame_count, video)
                print(f"derived_aligned_camera_cache={cache}")
                return rotation, translation
        missing = [str(path) for _, _, path in segments if not path.is_file()]
        raise FileNotFoundError(f"Missing original SLAM camera segment(s): {missing}")

    rotations, translations = [], []
    for start, stop, slam_path in segments:
        _, _, rotation, translation = load_slam_cam(str(slam_path))
        _, _, gt_rotation, gt_translation = load_gt_cam(
            str(video_root), video, start, stop)
        _, _, rotation, translation = algin_cam_traj_wo_scale(
            rotation, translation, gt_rotation, gt_translation)
        rotations.append(rotation)
        translations.append(translation)
    return torch.cat(rotations), torch.cat(translations)


def load_sampled_prediction(
    prediction_dir: Path,
    frame_ids: np.ndarray,
) -> dict[str, torch.Tensor]:
    chunks = []
    for path in sorted(prediction_dir.glob("*.json"), key=lambda item: int(item.stem)):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        length = np.asarray(payload["init_trans"]).shape[1]
        chunks.append((int(path.stem), int(path.stem) + length, payload))

    selected: dict[str, list[np.ndarray]] = {}
    for frame_id in frame_ids:
        matches = [chunk for chunk in chunks if chunk[0] <= frame_id < chunk[1]]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely locate prediction for frame {frame_id}")
        start, _, payload = matches[0]
        offset = int(frame_id) - start
        for key, value in payload.items():
            selected.setdefault(key, []).append(np.asarray(value)[:, offset:offset + 1])
    return {
        key: torch.from_numpy(np.concatenate(value, axis=1)).float()
        for key, value in selected.items()
    }


def load_compact_prediction(
    prediction_path: Path,
    frame_ids: np.ndarray,
) -> dict[str, torch.Tensor]:
    with np.load(prediction_path) as prediction:
        return {
            "init_root_orient": torch.from_numpy(
                prediction["global_orient"][frame_ids, 0]).float().unsqueeze(0),
            "init_hand_pose": torch.from_numpy(
                prediction["hand_pose"][frame_ids]).float().unsqueeze(0),
            "init_trans": torch.from_numpy(
                prediction["camera_translation"][frame_ids]).float().unsqueeze(0),
            "init_betas": torch.from_numpy(
                prediction["betas"][frame_ids]).float().unsqueeze(0),
        }


def main() -> None:
    args = parse_args()
    if not 1 <= args.segment <= args.segments:
        raise ValueError("--segment must be in [1, --segments]")
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    from scripts.scripts_eval.recovered_camera_cache import install_mano_legacy_compat
    install_mano_legacy_compat()
    from hawor.utils.process import get_mano_faces, run_mano
    from lib.eval_utils.custom_utils import cam2world_convert

    method = METHODS[args.method]
    video_root = ROOT / "datasets" / "hot3d_valset_export"
    result_dir = ROOT / method["log_dir"] / args.video
    details_path = result_dir / "evaluation_details.npz"
    with np.load(details_path) as details:
        all_frame_ids = details["frame_ids"].copy()
        saved_predicted_joints = details["pred_world_joints_m"].copy()
        saved_gt_joints = details["gt_world_joints_m"].copy()
    total_frames = len(saved_predicted_joints)
    boundaries = np.linspace(0, total_frames, args.segments + 1, dtype=int)
    start = int(boundaries[args.segment - 1])
    stop = int(boundaries[args.segment])
    valid = all_frame_ids[(all_frame_ids >= start) & (all_frame_ids < stop)]
    if len(valid) < args.samples:
        raise ValueError("Not enough valid frames in the selected segment")
    sample_positions = np.linspace(0, len(valid) - 1, args.samples, dtype=int)
    sample_frames = valid[sample_positions]

    if method["prediction_file"] is None:
        prediction = load_sampled_prediction(result_dir / "1", sample_frames)
    else:
        prediction = load_compact_prediction(
            result_dir / method["prediction_file"], sample_frames)
    camera_rotation, camera_translation = load_aligned_slam_cameras(
        video_root,
        args.video,
        total_frames,
        method["slam_dir"],
        method["slam_prefix"],
    )
    if (not torch.isfinite(camera_rotation[sample_frames]).all()
            or not torch.isfinite(camera_translation[sample_frames]).all()):
        raise ValueError(f"Sample frame(s) lack a valid aligned camera: {sample_frames.tolist()}")
    prediction_world = cam2world_convert(
        camera_rotation[sample_frames],
        camera_translation[sample_frames],
        prediction,
        "right",
    )
    predicted_mano = run_mano(
        prediction_world["init_trans"],
        prediction_world["init_root_orient"],
        prediction_world["init_hand_pose"],
        betas=prediction_world["init_betas"],
    )

    annotations = joblib.load(video_root / args.video / "anno.pth")
    gt_mano = run_mano(
        annotations["trans_r"][sample_frames].unsqueeze(0),
        annotations["rot_r"][sample_frames].unsqueeze(0),
        annotations["pose_r"][sample_frames].unsqueeze(0),
        betas=annotations["betas_r"][sample_frames].unsqueeze(0),
    )
    predicted_vertices = predicted_mano["vertices"][0].detach().cpu().numpy()
    predicted_joints = predicted_mano["joints"][0].detach().cpu().numpy()
    gt_vertices = gt_mano["vertices"][0].detach().cpu().numpy()
    gt_joints = gt_mano["joints"][0].detach().cpu().numpy()
    faces = np.asarray(get_mano_faces(), dtype=np.int32)

    pred_check = np.nanmax(np.abs(
        predicted_joints - saved_predicted_joints[sample_frames]))
    gt_check = np.nanmax(np.abs(gt_joints - saved_gt_joints[sample_frames]))
    tolerance = 2e-5
    if pred_check > tolerance or gt_check > tolerance:
        raise ValueError(
            "Reconstructed joints do not match evaluation data: "
            f"prediction={pred_check:.3e}, GT={gt_check:.3e}")

    fig = plt.figure(figsize=(18, 9), facecolor="white")
    overview = fig.add_subplot(121, projection="3d")
    detail = fig.add_subplot(122, projection="3d")

    segment_gt_wrist = saved_gt_joints[start:stop, 0]
    segment_pred_wrist = saved_predicted_joints[start:stop, 0]
    overview.plot(*segment_gt_wrist.T, color=GT_DARK, linewidth=1.0, alpha=0.48)
    overview.plot(*segment_pred_wrist.T, color=PRED_DARK, linewidth=0.9, alpha=0.42)
    for index, frame_id in enumerate(sample_frames):
        draw_hand(
            overview, gt_vertices[index], gt_joints[index], faces,
            GT_COLOR, GT_DARK, alpha=0.24)
        draw_hand(
            overview, predicted_vertices[index], predicted_joints[index], faces,
            PRED_COLOR, PRED_DARK, alpha=0.22)
        overview.text(
            *gt_joints[index, 0], f"  f{frame_id}", fontsize=7, color="#303030")

    overview_points = np.concatenate(
        [gt_vertices.reshape(-1, 3), predicted_vertices.reshape(-1, 3)], axis=0)
    apply_axes(overview, equal_limits(overview_points, padding=0.08))
    overview.set_title(
        f"World-space hand sequence, segment {args.segment}/{args.segments}\n"
        f"{method['label']}, frames {start}-{stop - 1} "
        f"({start / args.fps:.2f}-{stop / args.fps:.2f} s)")

    detail_index = len(sample_frames) // 2
    draw_hand(
        detail,
        gt_vertices[detail_index],
        gt_joints[detail_index],
        faces,
        GT_COLOR,
        GT_DARK,
        alpha=0.34,
        label_joints=True,
    )
    draw_hand(
        detail,
        predicted_vertices[detail_index],
        predicted_joints[detail_index],
        faces,
        PRED_COLOR,
        PRED_DARK,
        alpha=0.30,
    )
    detail_points = np.concatenate(
        [gt_vertices[detail_index], predicted_vertices[detail_index]], axis=0)
    apply_axes(detail, equal_limits(detail_points, padding=0.14))
    detail.set_title(
        f"21-joint pose detail, frame {sample_frames[detail_index]}\n"
        "joint numbers are shown on GT")

    legend = [
        Line2D([0], [0], color=GT_COLOR, linewidth=7, label="HOT3D GT"),
        Line2D(
            [0], [0], color=PRED_COLOR, linewidth=7,
            label=f"{method['label']} prediction"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=True,
               bbox_to_anchor=(0.5, 0.98), fontsize=12)
    fig.suptitle(
        f"GT vs {method['label']} right-hand pose: {args.video}",
        fontsize=17,
        y=1.00,
    )
    fig.text(
        0.5,
        0.015,
        "Joint order: 0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; "
        "13-16 ring; 17-20 little finger",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    output_dir = args.output_dir if args.output_dir is not None else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"gt_pred_hand_mesh_{args.segment:02d}_of_{args.segments:02d}_"
        f"frames_{start:04d}_{stop - 1:04d}.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output_path)
    print(f"sample_frames={sample_frames.tolist()}")
    print(f"max_pred_joint_reconstruction_error_m={pred_check:.9g}")
    print(f"max_gt_joint_reconstruction_error_m={gt_check:.9g}")


if __name__ == "__main__":
    main()
