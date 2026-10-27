#!/usr/bin/env python3
"""Restore the frozen HOT3D validation run after the ten-video export.

The runner validates the archived evaluation frame counts, makes the legacy GT
box tracks, renders masks from archived HaWoR predictions through read-only
symlinks, runs masked SLAM, and then delegates the TWM-only metrics pipeline.
All runner logs and status records live in this directory's ``runs`` folder.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Apply the requested execution settings before NumPy/Torch load their runtimes.
os.environ.update({
    "CUDA_VISIBLE_DEVICES": "1",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
})

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN_LEGACY = HERE / "run_legacy.py"
PREPROCESS = ROOT / "lib/datasets/hot3d_dataset_preprocess.py"
HAWOR_EVAL = ROOT / "scripts/scripts_eval/eval_hawor_hot3d.py"
SLAM = ROOT / "scripts/scripts_eval/test_mdslam_hot3d.py"
TWM_PIPELINE = ROOT / "thirdparty/ablation/twm_only/run_pipeline.py"
DEFAULT_PROTOCOL = ROOT / "diagnostics/pinch_hawor_vs_ema_10videos_20261018_final/protocol_frozen.json"
DEFAULT_VIDEO_ROOT = ROOT / "datasets/hot3d_valset_export"
DEFAULT_BASELINE_ROOT = ROOT / "eval_log"
DEFAULT_CHECKPOINT = ROOT / "thirdparty/ablation/training_v2/checkpoints/latest.pt"
DEFAULT_HAWOR_CHECKPOINT = ROOT / "weights/hawor/checkpoints/hawor.ckpt"
DEFAULT_PYTHON = ROOT / ".env/hawor/bin/python"
INTERVAL = 1000
MIN_CHUNK_FRAMES = 16


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _np_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_export_and_archive(
    video_root: Path, set_file: str, protocol_path: Path, baseline_root: Path,
    video_filter: set[str] | None = None,
) -> tuple[list[str], dict[str, int], dict[str, dict[str, int]]]:
    set_path = video_root / set_file
    if not set_path.is_file():
        raise FileNotFoundError(f"Exported validation set is missing: {set_path}")
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Frozen protocol is missing: {protocol_path}")
    if not baseline_root.is_dir():
        raise FileNotFoundError(f"Archived HaWoR evaluation directory is missing: {baseline_root}")

    selected = _read_json(set_path)
    frozen = _read_json(protocol_path).get("test_sequences")
    if not isinstance(selected, list) or not selected or not all(
        isinstance(item, str) and item for item in selected
    ):
        raise ValueError(f"{set_path} must contain a non-empty list of video names")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{set_path} contains duplicate video names")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError(f"Frozen protocol has no test_sequences: {protocol_path}")
    if len(frozen) != len(set(frozen)) or set(selected) != set(frozen):
        raise ValueError(
            "Exported val.json does not match the frozen protocol test set: "
            f"val.json={sorted(selected)}, frozen={sorted(frozen)}")
    if video_filter is not None:
        unknown = sorted(video_filter - set(frozen))
        if unknown:
            raise ValueError(f"Requested videos are outside the frozen validation set: {unknown}")
        videos_to_check = [video for video in selected if video in video_filter]
    else:
        videos_to_check = selected

    frame_counts: dict[str, int] = {}
    archive_checks: dict[str, dict[str, int]] = {}
    for video in videos_to_check:
        sequence_root = video_root / video
        image_root = sequence_root / "extracted_images"
        images = sorted(image_root.glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No exported JPEG frames for {video}: {image_root}")
        frame_count = len(images)
        details_path = baseline_root / video / "evaluation_details.npz"
        summary_path = baseline_root / video / "evaluation_summary.json"
        if not details_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                f"Archived evaluation is incomplete for {video}: {details_path} / {summary_path}")
        with np.load(details_path, allow_pickle=False) as details:
            required = ("pred_world_joints_m", "gt_world_joints_m", "frame_ids")
            missing = [name for name in required if name not in details]
            if missing:
                raise ValueError(f"{details_path} lacks required arrays: {missing}")
            predicted_frames = int(len(details["pred_world_joints_m"]))
            target_frames = int(len(details["gt_world_joints_m"]))
            evaluated_frames = int(len(details["frame_ids"]))
        summary = _read_json(summary_path)
        summary_frames = int(summary.get("total_frames", -1))
        if (frame_count != predicted_frames or frame_count != target_frames
                or frame_count != summary_frames):
            raise ValueError(
                f"Export/archive frame-count mismatch for {video}: "
                f"exported={frame_count}, prediction_details={predicted_frames}, "
                f"GT_details={target_frames}, archived_summary={summary_frames}")
        if not 0 < evaluated_frames <= frame_count:
            raise ValueError(
                f"Invalid archived evaluation frame count for {video}: "
                f"evaluated={evaluated_frames}, total={frame_count}")
        frame_counts[video] = frame_count
        archive_checks[video] = {
            "exported_frames": frame_count,
            "details_prediction_frames": predicted_frames,
            "details_gt_frames": target_frames,
            "details_evaluated_frames": evaluated_frames,
            "summary_total_frames": summary_frames,
        }
    return selected, frame_counts, archive_checks


def _validate_gt_tracks(video_root: Path, videos: list[str], frame_counts: dict[str, int]) -> list[str]:
    missing_or_invalid: list[str] = []
    for video in videos:
        path = video_root / video / "preprocess" / "preprocess_gtdet_tracks.npy"
        if not path.is_file():
            missing_or_invalid.append(video)
            continue
        try:
            loaded = np.load(path, allow_pickle=True)
            tracks = loaded.item()
            if not isinstance(tracks, dict) or not {0, 1}.issubset(tracks):
                raise ValueError("expected left/right tracks under integer keys 0 and 1")
            for side in (0, 1):
                sequence = tracks[side]
                if len(sequence) != frame_counts[video]:
                    raise ValueError(
                        f"track {side} has {len(sequence)} frames, expected {frame_counts[video]}")
                frame_ids = [int(row["frame"]) for row in sequence]
                if frame_ids != list(range(frame_counts[video])):
                    raise ValueError(f"track {side} frame indices are not a complete zero-based range")
        except Exception:
            missing_or_invalid.append(video)
    return missing_or_invalid


def _frame_chunks(frame_ids: np.ndarray) -> list[np.ndarray]:
    if len(frame_ids) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frame_ids) != 1) + 1
    chunks = np.split(frame_ids, breaks)
    return [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_FRAMES]


def _needed_prediction_starts(video_root: Path, video: str, frame_count: int) -> dict[int, list[int]]:
    sequence_root = video_root / video
    tracks_path = sequence_root / "preprocess" / "preprocess_gtdet_tracks.npy"
    tracks = np.load(tracks_path, allow_pickle=True).item()
    annotations = __import__("joblib").load(sequence_root / "anno.pth")
    side_valid = {
        0: np.any(_np_value(annotations["rot_l"]) != 0, axis=-1),
        1: np.any(_np_value(annotations["rot_r"]) != 0, axis=-1),
    }
    result: dict[int, list[int]] = {}
    for side in (0, 1):
        track = tracks[side]
        detected = np.asarray([bool(row["det"]) for row in track], dtype=bool)
        if len(detected) != frame_count or len(side_valid[side]) != frame_count:
            raise ValueError(f"GT-track/annotation frame count mismatch for {video}, side {side}")
        frame_ids = np.asarray([int(row["frame"]) for row in track], dtype=np.int64)
        valid_ids = frame_ids[detected & side_valid[side]]
        result[side] = [int(chunk[0]) for chunk in _frame_chunks(valid_ids)]
    return result


def _stage_prediction_links(
    video_root: Path, baseline_root: Path, mask_root: Path,
    videos: list[str], frame_counts: dict[str, int],
) -> int:
    mask_root.mkdir(parents=True, exist_ok=False)
    total_links = 0
    for video in videos:
        starts_by_side = _needed_prediction_starts(video_root, video, frame_counts[video])
        for side, starts in starts_by_side.items():
            for start in starts:
                source = baseline_root / video / str(side) / f"{start}.json"
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Archived HaWoR prediction needed for mask rendering is missing: {source}")
                # The old mask-only script reads these links and writes its own eval_log.txt
                # beside them. It never writes through a prediction-file symlink.
                destination = mask_root / video / str(side) / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source.resolve())
                total_links += 1
    return total_links


def _valid_mask_archive(path: Path, expected_frames: int) -> bool:
    if not path.is_file():
        return False
    try:
        masks = np.load(path, allow_pickle=True)
        return len(masks) == expected_frames
    except Exception:
        return False


def _valid_slam_archive(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not {"tstamp", "disps", "traj", "scale"}.issubset(archive.files):
                return False
            if len(archive["tstamp"]) < 1 or len(archive["traj"]) < 1:
                return False
            return bool(np.isfinite(archive["scale"]).all())
    except Exception:
        return False


def _write_selection_file(run_dir: Path, stage: str, videos: list[str]) -> Path:
    path = run_dir / f"{stage}_sequences.json"
    path.write_text(json.dumps(videos, indent=2) + "\n", encoding="utf-8")
    return path


def _execution_env(physical_gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": physical_gpu,
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    })
    return env


def _parse_inference_gpus(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",")]
    if not gpu_ids or any(not item.isascii() or not item.isdigit() for item in gpu_ids):
        raise argparse.ArgumentTypeError(
            "GPU IDs must be a comma-separated list of non-negative integers")
    gpu_ids = [str(int(item)) for item in gpu_ids]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("GPU IDs must be unique")
    return gpu_ids


def _run_command(
    command: list[str], label: str, log_path: Path, status: dict[str, Any],
    status_path: Path, env: dict[str, str], *, cwd: Path = ROOT,
) -> None:
    status["current_stage"] = label
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(status_path, status)
    rendered = " ".join(command)
    status.setdefault("commands", []).append({
        "stage": label,
        "argv": command,
        "cwd": str(cwd),
        "environment": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS"),
        },
    })
    _atomic_json(status_path, status)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n## {label}\n$ {rendered}\n")
        log.flush()
        print(f"[{label}] starting; output: {log_path}", flush=True)
        started = time.monotonic()
        with subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            return_code = process.wait()
        log.write(f"\n[{label}] exit={return_code}; elapsed_sec={time.monotonic() - started:.1f}\n")
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    status.setdefault("completed_stages", []).append(label)
    status["current_stage"] = None
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(status_path, status)


def _import_and_single_frame_check(
    videos: list[str], video_root: Path, frame_counts: dict[str, int],
) -> dict[str, Any]:
    """Import just the needed modules and decode one frame; no model is loaded."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import cv2
    import joblib
    import torch
    from PIL import Image  # noqa: F401
    from pycocotools import mask as mask_tool  # noqa: F401

    # Exercise the imports used by old GT preprocessing and the TWM runner.
    from lib.datasets.detect_track_video_hot3d import detect_track_process_gt  # noqa: F401

    try_project = ROOT / "thirdparty/try"
    try_scripts = try_project / "scripts"
    for location in (try_scripts, try_project):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    from dual_loop.geometry import ManoGeometry  # noqa: F401
    from dual_loop.handloop_frontend import HaWoRFrontend, load_hawor  # noqa: F401
    from dual_loop.handloop_twm import TWMConfig, TrajectoryRestorer  # noqa: F401
    from run_handloop import context_batch, load_inputs  # noqa: F401

    video = videos[0]
    image_path = sorted((video_root / video / "extracted_images").glob("*.jpg"))[0]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"Single-frame OpenCV decode failed: {image_path}")
    annotations = joblib.load(video_root / video / "anno.pth")
    annotation_frames = int(len(_np_value(annotations["rot_r"])))
    if annotation_frames != frame_counts[video]:
        raise ValueError(
            f"Single-video annotation frame count mismatch for {video}: "
            f"annotations={annotation_frames}, exported={frame_counts[video]}")
    return {
        "status": "ok",
        "imports": [
            "cv2", "joblib", "torch", "PIL", "pycocotools.mask",
            "lib.datasets.detect_track_video_hot3d", "dual_loop", "run_handloop",
        ],
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "visible_cuda_devices": int(torch.cuda.device_count()),
        "decoded_image": str(image_path),
        "image_shape_hwc": [int(value) for value in image.shape],
        "annotation_frames": annotation_frames,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--baseline_root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hawor_checkpoint", type=Path, default=DEFAULT_HAWOR_CHECKPOINT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physical_gpu", default="1",
                        help="Physical GPU exposed to child processes via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--inference_gpus", type=_parse_inference_gpus, default=None,
                        help="Optional physical GPUs for parallel TWM inference, e.g. 1,2,3")
    parser.add_argument("--interval", type=int, default=INTERVAL)
    parser.add_argument("--video", action="append",
                        help="Optional subset for a per-video smoke run; repeat as needed")
    parser.add_argument("--run_dir", type=Path,
                        help="Unique runner directory; must be under restore_validation/runs")
    parser.add_argument("--check_only", action="store_true",
                        help="Run imports, one-frame decode, and archive checks, then stop")
    parser.add_argument("--prepare_only", action="store_true",
                        help="Complete GT tracks, masks, and SLAM, then stop before TWM metrics")
    return parser.parse_args()


def _resolve_run_dir(value: Path | None) -> tuple[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = value or (HERE / "runs" / run_id)
    run_dir = run_dir.expanduser().resolve()
    runs_root = (HERE / "runs").resolve()
    if not run_dir.is_relative_to(runs_root):
        raise ValueError(f"run_dir must stay under {runs_root}")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse runner directory: {run_dir}")
    if value is not None:
        run_id = run_dir.name
    return run_id, run_dir


def main() -> int:
    args = _parse_args()
    args.video_root = args.video_root.expanduser().resolve()
    args.protocol = args.protocol.expanduser().resolve()
    args.baseline_root = args.baseline_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.hawor_checkpoint = args.hawor_checkpoint.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    if args.interval != INTERVAL:
        raise ValueError(f"Frozen HOT3D validation requires interval={INTERVAL}, got {args.interval}")
    if (args.inference_gpus and args.device.startswith("cuda:")
            and args.device != "cuda:0"):
        raise ValueError(
            "With --inference_gpus each worker sees its physical card as cuda:0; "
            "use --device cuda or --device cuda:0")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise FileNotFoundError(f"HaWoR environment Python is not executable: {args.python}")

    run_id, run_dir = _resolve_run_dir(args.run_dir)
    run_dir.mkdir(parents=True)
    status_path = run_dir / "status.json"
    log_path = run_dir / "run.log"
    status: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_stage": "preflight",
        "completed_stages": [],
        "commands": [],
        "outputs": {},
    }
    _atomic_json(status_path, status)
    env = _execution_env(args.physical_gpu)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
        "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": env["MKL_NUM_THREADS"],
    })

    try:
        for label, path in (
            ("TWM checkpoint", args.checkpoint),
            ("HaWoR checkpoint", args.hawor_checkpoint),
            ("DROID-SLAM checkpoint", ROOT / "weights/external/droid.pth"),
            ("Metric3D checkpoint", ROOT / "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth"),
            ("MANO right-hand asset", ROOT / "_DATA/data/mano/MANO_RIGHT.pkl"),
            ("MANO left-hand asset", ROOT / "_DATA/data_left/mano_left/MANO_LEFT.pkl"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Required {label} is missing: {path}")
        requested = set(args.video) if args.video else None
        if args.video and len(args.video) != len(requested):
            raise ValueError("--video may be repeated for distinct sequences only")
        frozen_videos, all_frame_counts, archive_checks = _validate_export_and_archive(
            args.video_root, args.set_file, args.protocol, args.baseline_root,
            video_filter=requested)
        if args.video:
            assert requested is not None
            videos = [video for video in frozen_videos if video in requested]
        else:
            videos = list(frozen_videos)
        frame_counts = {video: all_frame_counts[video] for video in videos}
        smoke = _import_and_single_frame_check(videos, args.video_root, frame_counts)
        if args.device.startswith("cuda") and not smoke["torch_cuda_available"]:
            raise RuntimeError(
                f"CUDA is unavailable after setting CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}; "
                "cannot run mask rendering, SLAM, or TWM inference")
        status["preflight"] = {
            "frozen_test_videos": frozen_videos,
            "selected_videos": videos,
            "frame_count_check_scope": "selected pilot videos" if args.video else "all frozen videos",
            "frozen_test_set_matches": True,
            "frame_counts": frame_counts,
            "archived_evaluation_frame_counts": {
                video: archive_checks[video] for video in videos
            },
            "environment_python": str(args.python),
            "requested_cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "inference_gpus": args.inference_gpus,
            "omp_threads": env["OMP_NUM_THREADS"],
            "mkl_threads": env["MKL_NUM_THREADS"],
            "single_frame_import_check": smoke,
        }
        _atomic_json(status_path, status)
        print(json.dumps(status["preflight"], indent=2), flush=True)
        if args.check_only:
            status.update({
                "status": "ready",
                "current_stage": None,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            _atomic_json(status_path, status)
            return 0

        invalid_tracks = _validate_gt_tracks(args.video_root, videos, frame_counts)
        if invalid_tracks:
            preprocess_set = run_dir / "gt_preprocess_sequences.json"
            preprocess_set.write_text(json.dumps(invalid_tracks, indent=2) + "\n", encoding="utf-8")
            preprocess_command = [
                str(args.python), str(RUN_LEGACY), str(PREPROCESS),
                "--video_root", str(args.video_root),
                "--set_file", str(preprocess_set),
                "--for_eval",
            ]
            _run_command(preprocess_command, "gt_box_preprocess", log_path, status,
                          status_path, env)
        else:
            status.setdefault("reused_outputs", []).append(
                f"Verified GT-projected tracks for selected videos: {videos}")
            status.setdefault("completed_stages", []).append("gt_box_preprocess (verified existing)")
            _atomic_json(status_path, status)
        remaining_invalid = _validate_gt_tracks(args.video_root, videos, frame_counts)
        if remaining_invalid:
            raise ValueError(f"GT projected-track generation did not complete: {remaining_invalid}")

        mask_missing = [
            video for video in videos
            if not _valid_mask_archive(
                args.video_root / video / "preprocess" / "model_masks.npy",
                frame_counts[video],
            )
        ]
        if mask_missing:
            mask_root = run_dir / "mask_predictions"
            link_count = _stage_prediction_links(
                args.video_root, args.baseline_root, mask_root, mask_missing, frame_counts)
            mask_set = _write_selection_file(run_dir, "mask_only", mask_missing)
            status["outputs"]["mask_prediction_links"] = {
                "root": str(mask_root),
                "linked_json_count": link_count,
                "selected_videos": mask_missing,
                "baseline_root_read_only": str(args.baseline_root),
            }
            _atomic_json(status_path, status)
            mask_command = [
                str(args.python), str(RUN_LEGACY), str(HAWOR_EVAL),
                "--video_root", str(args.video_root),
                "--set_file", str(mask_set),
                "--gen_hand_mask",
                "--eval_log_dir", str(mask_root),
            ]
            _run_command(mask_command, "mask_only", log_path, status, status_path, env)
            for video in mask_missing:
                mask_path = args.video_root / video / "preprocess" / "model_masks.npy"
                if not _valid_mask_archive(mask_path, frame_counts[video]):
                    raise ValueError(
                        f"Mask-only output is missing or has wrong frame count: {mask_path}")
        else:
            status.setdefault("reused_outputs", []).append(
                f"Verified full-length model_masks.npy for selected videos: {videos}")
            status.setdefault("completed_stages", []).append("mask_only (verified existing)")
            _atomic_json(status_path, status)

        slam_missing = []
        for video in videos:
            sequence_missing = False
            slam_dir = args.video_root / video / "preprocess" / "SLAM"
            for start in range(0, frame_counts[video], args.interval):
                stop = min(start + args.interval, frame_counts[video])
                path = slam_dir / f"hawor_slam_w_scale_{start}_{stop}.npz"
                if path.is_file() and not _valid_slam_archive(path):
                    raise ValueError(f"Existing SLAM archive is unreadable/incomplete: {path}")
                if not path.is_file():
                    sequence_missing = True
            if sequence_missing:
                slam_missing.append(video)
        if slam_missing:
            slam_set = _write_selection_file(run_dir, "masked_slam", slam_missing)
            slam_command = [
                str(args.python), str(RUN_LEGACY), str(SLAM),
                "--video_root", str(args.video_root),
                "--set_file", str(slam_set),
                "--device", args.device,
            ]
            _run_command(slam_command, "masked_slam", log_path, status, status_path, env)
        else:
            status.setdefault("reused_outputs", []).append(
                f"Verified all interval SLAM archives for selected videos: {videos}")
            status.setdefault("completed_stages", []).append("masked_slam (verified existing)")
            _atomic_json(status_path, status)
        slam_outputs: list[str] = []
        for video in videos:
            slam_dir = args.video_root / video / "preprocess" / "SLAM"
            for start in range(0, frame_counts[video], args.interval):
                stop = min(start + args.interval, frame_counts[video])
                path = slam_dir / f"hawor_slam_w_scale_{start}_{stop}.npz"
                if not path.is_file():
                    raise FileNotFoundError(f"Masked SLAM stage did not create {path}")
                slam_outputs.append(str(path))
        status["outputs"]["slam_archives"] = slam_outputs
        _atomic_json(status_path, status)

        if args.prepare_only:
            status.update({
                "status": "prepared",
                "current_stage": None,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            _atomic_json(status_path, status)
            print(f"[prepared] GT tracks, masks, and SLAM are ready; see {status_path}", flush=True)
            return 0

        pipeline_run_dir = ROOT / "thirdparty/ablation/twm_only/runs" / f"restore_{run_id}"
        if pipeline_run_dir.exists():
            raise FileExistsError(f"TWM pipeline run path already exists: {pipeline_run_dir}")
        pipeline_command = [
            str(args.python), str(TWM_PIPELINE),
            "--checkpoint", str(args.checkpoint),
            "--hawor_checkpoint", str(args.hawor_checkpoint),
            "--video_root", str(args.video_root),
            "--set_file", args.set_file,
            "--protocol", str(args.protocol),
            "--device", args.device,
            "--interval", str(args.interval),
            "--allow_gt_boxes",
            "--run_dir", str(pipeline_run_dir),
        ]
        if args.video:
            pipeline_command.extend(item for video in videos for item in ("--video", video))
        if args.inference_gpus:
            pipeline_command.extend(("--inference_gpus", ",".join(args.inference_gpus)))
        status["outputs"]["twm_pipeline_run_dir"] = str(pipeline_run_dir)
        _atomic_json(status_path, status)
        _run_command(pipeline_command, "twm_only_wmpjpe_onset", log_path,
                     status, status_path, env)
        pipeline_status = next((
            path for name in ("status.json", "diagnostic_status.json", "blocked_status.json")
            if (path := pipeline_run_dir / name).is_file()
        ), None)
        if pipeline_status is None:
            raise FileNotFoundError(f"TWM pipeline did not write a final status in {pipeline_run_dir}")
        status["outputs"]["twm_pipeline_status"] = str(pipeline_status)
        status["pipeline_status"] = _read_json(pipeline_status)
        if status["pipeline_status"].get("status") not in {
            "complete", "complete_diagnostic_no_formal_row",
        }:
            raise RuntimeError(f"TWM pipeline did not finish metrics: {status['pipeline_status']}")
        status.update({
            "status": status["pipeline_status"]["status"],
            "current_stage": None,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        _atomic_json(status_path, status)
        return 0
    except Exception as error:
        status.update({
            "status": "failed",
            "current_stage": status.get("current_stage"),
            "error": f"{type(error).__name__}: {error}",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        _atomic_json(status_path, status)
        print(f"[failed] {status['error']}; see {log_path} and {status_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
