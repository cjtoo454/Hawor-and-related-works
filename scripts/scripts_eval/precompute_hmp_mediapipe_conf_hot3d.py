"""Precompute the MediaPipe bbox confidence used by official HMP losses."""

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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker_videos", nargs="*", help=argparse.SUPPRESS)
    return parser.parse_args()


def worker(args: argparse.Namespace) -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    import numpy as np

    sys.path.insert(0, str(HMP_ROOT / "src"))
    from keypoints2d.mediapipe_detect import MPJson

    for sequence in args.worker_videos:
        sequence_root = args.input_root / sequence
        destination = sequence_root / "confidences" / "mediapipe_bbox_conf.npz"
        total_frames = len(list((sequence_root / "rgb").glob("*.jpg")))
        if destination.exists() and len(np.load(destination)["arr_0"]) == total_frames:
            print(f"[skip] {sequence}: MediaPipe confidence exists", flush=True)
            continue
        sidecar = np.load(sequence_root / "hot3d_input.npz", allow_pickle=True)
        gt_dict = {
            "frame_id": sidecar["frame_id"].astype(np.int64).tolist(),
            "bbox": sidecar["bbox"].astype(np.float32),
            "handedness": "right",
        }
        print(f"[mediapipe] {sequence}", flush=True)
        MPJson(gt_dict=gt_dict, expand_coef=1.4).mp_bbox_conf(
            img_dir=str(sequence_root / "rgb"), out_dir=str(destination))


def main() -> None:
    args = parse_args()
    if args.worker_videos is not None:
        worker(args)
        return
    with (args.video_root / args.set_file).open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    worker_count = min(args.workers, len(sequences))
    groups = [sequences[index::worker_count] for index in range(worker_count)]
    processes = []
    for group in groups:
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--video_root", str(args.video_root),
            "--input_root", str(args.input_root),
            "--set_file", args.set_file,
            "--worker_videos", *group,
        ]
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        processes.append(subprocess.Popen(command, cwd=ROOT, env=environment))
    failures = [process.wait() for process in processes]
    if any(failures):
        raise subprocess.CalledProcessError(max(failures), "MediaPipe workers")


if __name__ == "__main__":
    main()
