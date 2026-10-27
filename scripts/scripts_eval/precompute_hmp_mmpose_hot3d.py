"""Precompute HMP's official MMPose observations on prepared HOT3D boxes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HMP_ROOT = ROOT / "thirdparty" / "HMP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=Path,
                        default=ROOT / "datasets" / "hot3d_valset_export")
    parser.add_argument("--input_root", type=Path,
                        default=ROOT / "eval_log_hmp_slam" / "_inputs" / "HOT3D")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--video", action="append")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--worker_gpu", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_videos", nargs="*", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_sequences(args: argparse.Namespace) -> list[str]:
    with (args.video_root / args.set_file).open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    if args.video:
        selected = set(args.video)
        sequences = [sequence for sequence in sequences if sequence in selected]
    return sequences


def worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.worker_gpu
    os.environ["PYTHONNOUSERSITE"] = "1"
    import glob

    import numpy as np

    sys.path.insert(0, str(HMP_ROOT / "src"))
    os.chdir(HMP_ROOT)
    from keypoints2d.top_down_pose_tracking_demo_with_mmdet import MMPOSEJson

    model = MMPOSEJson(use_detector=False, render_output=False)
    for sequence in args.worker_videos:
        sequence_root = args.input_root / sequence
        rgb_path = sequence_root / "rgb"
        keypoint_path = sequence_root / "mmpose_keypoints2d"
        total_frames = len(glob.glob(str(rgb_path / "*.jpg")))
        if total_frames == len(glob.glob(str(keypoint_path / "*.json"))):
            print(f"[skip] {sequence}: {total_frames} MMPose frames exist", flush=True)
            continue
        sidecar = np.load(sequence_root / "hot3d_input.npz", allow_pickle=True)
        gt_dict = {
            "frame_id": sidecar["frame_id"].astype(np.int64).tolist(),
            "bbox": sidecar["bbox"].astype(np.float32),
            "handedness": "right",
        }
        print(f"[mmpose] {sequence} on GPU {args.worker_gpu}", flush=True)
        model.main(
            video_path=str(sequence_root / "rgb_raw.mp4"),
            out_folder_path=str(sequence_root / "rgb_mmpose"),
            json_folder=str(keypoint_path),
            gt_dict=gt_dict,
        )


def main() -> None:
    args = parse_args()
    if args.worker_gpu is not None:
        worker(args)
        return
    sequences = load_sequences(args)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    groups = [sequences[index::len(gpus)] for index in range(len(gpus))]
    processes = []
    for gpu, group in zip(gpus, groups):
        if not group:
            continue
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--video_root", str(args.video_root),
            "--input_root", str(args.input_root),
            "--set_file", args.set_file,
            "--worker_gpu", gpu,
            "--worker_videos", *group,
        ]
        processes.append(subprocess.Popen(command, cwd=ROOT))
    failures = [process.wait() for process in processes]
    if any(failures):
        raise subprocess.CalledProcessError(max(failures), "MMPose workers")


if __name__ == "__main__":
    main()
