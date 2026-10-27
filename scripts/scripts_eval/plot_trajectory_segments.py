"""Plot synchronized trajectory segments for HaWoR, HMP-SLAM, and WiLoR-SLAM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Method:
    label: str
    log_dir: str


METHODS = (
    Method("HaWoR", "eval_log"),
    Method("HMP-SLAM", "eval_log_hmp_slam"),
    Method("WiLoR-SLAM", "eval_log_wilor_slam"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot equal-duration wrist trajectory segments with shared axes")
    parser.add_argument("--video", required=True)
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def shared_limits(points: list[np.ndarray]) -> tuple[tuple[float, float], ...]:
    finite = [point[np.isfinite(point).all(axis=1)] for point in points]
    finite = [point for point in finite if len(point)]
    if not finite:
        raise ValueError("Trajectory segment contains no finite wrist positions")
    combined = np.concatenate(finite, axis=0)
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    center = (minimum + maximum) / 2.0
    span = max(float((maximum - minimum).max()), 0.10)
    half_span = span * 0.58
    return tuple((float(value - half_span), float(value + half_span))
                 for value in center)


def main() -> None:
    args = parse_args()
    if args.segments < 1:
        raise ValueError("--segments must be at least 1")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    trajectories: dict[str, dict[str, object]] = {}
    total_frames: int | None = None
    canonical_gt: np.ndarray | None = None
    for method in METHODS:
        output_dir = ROOT / method.log_dir / args.video
        details_path = output_dir / "evaluation_details.npz"
        if not details_path.exists():
            raise FileNotFoundError(f"Missing evaluation details: {details_path}")
        with np.load(details_path) as details:
            predicted = details["pred_world_joints_m"][:, 0].copy()
            ground_truth = details["gt_world_joints_m"][:, 0].copy()
        if predicted.shape != ground_truth.shape:
            raise ValueError(f"Prediction/GT shape mismatch for {method.label}")
        if total_frames is None:
            total_frames = len(predicted)
            canonical_gt = ground_truth
        elif len(predicted) != total_frames:
            raise ValueError(f"Frame count mismatch for {method.label}")
        elif not np.allclose(canonical_gt, ground_truth, equal_nan=True):
            raise ValueError(f"GT trajectory mismatch for {method.label}")
        trajectories[method.label] = {
            "output_dir": output_dir,
            "predicted": predicted,
        }

    assert total_frames is not None
    assert canonical_gt is not None
    boundaries = np.linspace(0, total_frames, args.segments + 1, dtype=int)
    for segment_index, (start, stop) in enumerate(
            zip(boundaries[:-1], boundaries[1:]), start=1):
        segment_points = [canonical_gt[start:stop]]
        segment_points.extend(
            value["predicted"][start:stop] for value in trajectories.values())
        limits = shared_limits(segment_points)
        filename = (
            f"world_trajectory_{segment_index:02d}_of_{args.segments:02d}_"
            f"frames_{start:04d}_{stop - 1:04d}.png"
        )

        for method in METHODS:
            output_dir = trajectories[method.label]["output_dir"]
            predicted = trajectories[method.label]["predicted"]
            fig = plt.figure(figsize=(10, 8))
            axis = fig.add_subplot(111, projection="3d")
            axis.plot(
                *canonical_gt[start:stop].T,
                label="HOT3D GT wrist",
                linewidth=1.5,
            )
            axis.plot(
                *predicted[start:stop].T,
                label=f"{method.label} wrist",
                linewidth=1.2,
            )
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
            axis.set_zlim(*limits[2])
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=22, azim=-58)
            axis.set_xlabel("X (m)")
            axis.set_ylabel("Y (m)")
            axis.set_zlabel("Z (m)")
            axis.set_title(
                f"Right-hand world trajectory: {args.video}\n"
                f"{method.label}, segment {segment_index}/{args.segments}, "
                f"frames {start}-{stop - 1} "
                f"({start / args.fps:.2f}-{stop / args.fps:.2f} s)"
            )
            axis.legend()
            fig.tight_layout()
            output_path = output_dir / filename
            fig.savefig(output_path, dpi=160)
            plt.close(fig)
            print(output_path)


if __name__ == "__main__":
    main()
