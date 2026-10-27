#!/usr/bin/env python3
"""Run TWM-only inference, W-MPJPE, and frozen-protocol onset scoring."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INFER = HERE / "run_twm_only.py"
ONSET = HERE / "eval_twm_onset.py"
WMPJPE = ROOT / "thirdparty/try/scripts/eval_hot3d.py"
DEFAULT_PROTOCOL = ROOT / "diagnostics/pinch_hawor_vs_ema_10videos_20261018_final/protocol_frozen.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_evaluation_sources(prediction_root: Path, videos: list[str]) -> None:
    """Bind each W-MPJPE details archive to the predictions that produced it."""
    for video in videos:
        sequence = prediction_root / video
        prediction = sequence / "try_predictions.npz"
        details = sequence / "evaluation_details.npz"
        metadata_path = sequence / "try_inference.json"
        if not all(path.is_file() for path in (prediction, details, metadata_path)):
            raise FileNotFoundError(f"Cannot bind evaluation inputs for {video}: {sequence}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest = {
            "schema": "twm_only_evaluation_source_v1",
            "sequence": video,
            "method": metadata.get("method"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "prediction_sha256": sha256(prediction),
            "evaluation_details_sha256": sha256(details),
            "inference_metadata_sha256": sha256(metadata_path),
        }
        (sequence / "evaluation_source.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_table_row(expected_videos: list[str], inference: dict[str, dict],
                    wmpjpe: dict, onset: dict, *, frozen_videos: list[str]) -> dict:
    expected = set(expected_videos)
    frozen = set(frozen_videos)
    if expected != frozen:
        raise ValueError(
            "Formal table row requires the complete frozen protocol test_sequences set: "
            f"selected={sorted(expected)}, frozen={sorted(frozen)}")
    inferred = set(inference)
    evaluated = {row["video"] for row in wmpjpe.get("videos", [])}
    onset_videos = {row["sequence"] for row in onset.get("per_video", [])}
    if not (expected == inferred == evaluated == onset_videos):
        raise ValueError(
            "Video-set mismatch before table-row generation: "
            f"expected={sorted(expected)}, inference={sorted(inferred)}, "
            f"W-MPJPE={sorted(evaluated)}, onset={sorted(onset_videos)}")
    incomplete = {
        video: (metadata.get("processed_frames"), metadata.get("total_frames"))
        for video, metadata in inference.items()
        if metadata.get("processed_frames") != metadata.get("total_frames")
    }
    if incomplete:
        raise ValueError(f"Incomplete inference cannot produce a result row: {incomplete}")
    methods = {metadata.get("method") for metadata in inference.values()}
    if methods != {"TWM only; raw HaWoR feedback"}:
        raise ValueError(f"Unexpected inference methods in table row: {sorted(str(x) for x in methods)}")
    stages = {metadata.get("checkpoint_stage") for metadata in inference.values()}
    checkpoints = {metadata.get("checkpoint_sha256") for metadata in inference.values()}
    frontends = {metadata.get("frontend_source_sha256") for metadata in inference.values()}
    frontend_sources = {metadata.get("frontend_weights_source") for metadata in inference.values()}
    steps = {metadata.get("checkpoint_global_step") for metadata in inference.values()}
    if any(len(values) != 1 for values in (stages, checkpoints, frontends, frontend_sources, steps)):
        raise ValueError("Inference videos do not share one TWM/frontend checkpoint and training step")
    if next(iter(steps)) is None or int(next(iter(steps))) <= 0:
        raise ValueError("Formal table row requires a trained checkpoint with global_step > 0")
    if not all(bool(metadata.get("oracle_assisted"))
               and bool(metadata.get("baseline_frontend_scope_matches_frozen_pinch_protocol"))
               for metadata in inference.values()):
        raise ValueError(
            "Protocol mismatch: the frozen baseline uses GT-projected boxes; all inference metadata "
            "must confirm actual oracle_assisted input before a formal row is written")
    w = wmpjpe.get("aggregate", {})
    o = onset.get("aggregate", {})
    for name, value in (("w_mpjpe_mm", w.get("w_mpjpe_mm")),
                        ("onset_mae_ms", o.get("onset_mae_ms"))):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Formal table row requires finite {name}; no matched events remain diagnostic")
    row = {
        "method": "TWM only; raw HaWoR feedback",
        "num_videos": len(expected),
        "checkpoint_stage": next(iter(inference.values())).get("checkpoint_stage"),
        "checkpoint_sha256": next(iter(inference.values())).get("checkpoint_sha256"),
        "hawor_checkpoint_sha256": next(iter(inference.values())).get("hawor_checkpoint_sha256"),
        "frontend_source_sha256": next(iter(inference.values())).get("frontend_source_sha256"),
        "frontend_weights_source": next(iter(inference.values())).get("frontend_weights_source"),
        "checkpoint_global_step": next(iter(inference.values())).get("checkpoint_global_step"),
        "oracle_assisted": True,
        "w_mpjpe_mm": w.get("w_mpjpe_mm"),
        "wa_mpjpe_mm": w.get("wa_mpjpe_mm"),
        "pa_mpjpe_mm": w.get("pa_mpjpe_mm"),
        "accel_m_s2": w.get("accel_m_s2"),
        "rte_percent": w.get("rte_percent"),
        "onset_mae_ms": o.get("onset_mae_ms"),
        "event_f1": o.get("f1"),
        "reference_events": o.get("reference_events"),
        "prediction_threshold": onset.get("prediction_threshold"),
        "reference_type": onset.get("reference_type"),
    }
    return row


def render_table_row(row: dict) -> str:
    columns = ("method", "num_videos", "w_mpjpe_mm", "wa_mpjpe_mm", "pa_mpjpe_mm",
               "accel_m_s2", "rte_percent", "onset_mae_ms", "event_f1",
               "checkpoint_stage", "oracle_assisted")
    headers = ("Method", "Videos", "W-MPJPE (mm)", "WA-MPJPE (mm)", "PA-MPJPE (mm)",
               "Accel (m/s²)", "RTE (%)", "Onset MAE (ms)", "Event F1",
               "Checkpoint", "Oracle-assisted")
    values = [row.get(column) for column in columns]

    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) \
        + " |\n| " + " | ".join(cell(value) for value in values) + " |\n"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hawor_checkpoint", type=Path,
                        default=ROOT / "weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--video_root", type=Path,
                        default=ROOT / "datasets/hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--video", action="append",
                        help="Optional repeatable video selection; default is frozen protocol test set")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument(
        "--inference_gpus", type=_parse_inference_gpus, default=None,
        help="Optional comma-separated physical GPU IDs for parallel per-video inference, e.g. 1,2,3",
    )
    parser.add_argument("--allow_gt_boxes", action="store_true")
    parser.add_argument("--run_dir", type=Path,
                        help="Result directory; must remain under thirdparty/ablation/twm_only")
    return parser.parse_args()


def _parse_inference_gpus(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",")]
    if not gpu_ids or any(not item.isascii() or not item.isdigit() for item in gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs must be a comma-separated list of non-negative integers")
    gpu_ids = [str(int(item)) for item in gpu_ids]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("GPU IDs must be unique")
    return gpu_ids


def _run(command: list[str], log, label: str) -> None:
    log.write(f"\n## {label}\n$ {' '.join(command)}\n")
    log.flush()
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write(result.stdout)
    if result.stdout and not result.stdout.endswith("\n"):
        log.write("\n")
    log.flush()
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)


def _group_videos_by_gpu(
    videos: list[str], gpu_ids: list[str], video_root: Path,
) -> list[tuple[str, list[str]]]:
    """Greedily balance sequences by exported frame count without sharing a video."""
    if len(videos) != len(set(videos)):
        raise ValueError("Inference video assignments must be unique")
    if not videos or not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("Inference scheduling requires videos and unique GPU IDs")
    active_gpus = gpu_ids[:min(len(gpu_ids), len(videos))]
    groups = {gpu: [] for gpu in active_gpus}
    loads = {gpu: 0 for gpu in active_gpus}
    weighted_videos = []
    for order, video in enumerate(videos):
        image_root = video_root / video / "extracted_images"
        frame_count = sum(1 for _ in image_root.glob("*.jpg"))
        weighted_videos.append((video, frame_count, order))
    for video, frame_count, _ in sorted(weighted_videos, key=lambda item: (-item[1], item[2])):
        gpu = min(active_gpus, key=lambda candidate: (loads[candidate], active_gpus.index(candidate)))
        groups[gpu].append(video)
        loads[gpu] += frame_count
    return [(gpu, groups[gpu]) for gpu in active_gpus if groups[gpu]]


def _run_inference_workers(
    command_prefix: list[str], videos: list[str], gpu_ids: list[str],
    video_root: Path, prediction_root: Path, run_dir: Path, log,
    *, groups: list[tuple[str, list[str]]] | None = None,
) -> None:
    worker_root = run_dir / "inference_workers"
    worker_root.mkdir()
    groups = groups or _group_videos_by_gpu(videos, gpu_ids, video_root)
    workers = []
    for gpu_id, assigned_videos in groups:
        worker_log = worker_root / f"gpu_{gpu_id}.log"
        command = [*command_prefix]
        command.extend(item for video in assigned_videos for item in ("--video", video))
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        workers.append({
            "gpu_id": gpu_id,
            "videos": assigned_videos,
            "command": command,
            "log": worker_log,
            "env": worker_env,
        })

    plan = [{
        "gpu_id": worker["gpu_id"],
        "videos": worker["videos"],
        "command": worker["command"],
        "cuda_visible_devices": worker["env"]["CUDA_VISIBLE_DEVICES"],
        "log": str(worker["log"]),
    } for worker in workers]
    (worker_root / "worker_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    log.write("\n## Inference worker plan\n")
    log.write(json.dumps(plan, indent=2) + "\n")
    log.flush()

    def launch(worker: dict) -> int:
        with worker["log"].open("w", encoding="utf-8") as output:
            output.write(f"$ {' '.join(worker['command'])}\n")
            output.write(f"CUDA_VISIBLE_DEVICES={worker['gpu_id']}\n\n")
            output.flush()
            result = subprocess.run(
                worker["command"], cwd=ROOT, env=worker["env"],
                stdout=output, stderr=subprocess.STDOUT, check=False,
            )
            output.write(f"\nworker_exit={result.returncode}\n")
        return result.returncode

    worker_status = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = {
            executor.submit(launch, worker): worker
            for worker in workers
        }
        for future in concurrent.futures.as_completed(futures):
            worker = futures[future]
            try:
                return_code = future.result()
                error = None
            except Exception as exception:
                return_code = None
                error = f"{type(exception).__name__}: {exception}"
            worker_status.append({
                "gpu_id": worker["gpu_id"],
                "videos": worker["videos"],
                "return_code": return_code,
                "error": error,
                "log": str(worker["log"]),
            })
    gpu_order = {gpu_id: index for index, (gpu_id, _) in enumerate(groups)}
    worker_status.sort(key=lambda item: gpu_order[item["gpu_id"]])
    status_path = worker_root / "worker_status.json"
    status_path.write_text(json.dumps(worker_status, indent=2) + "\n", encoding="utf-8")
    for worker in worker_status:
        log.write(
            f"[inference worker GPU {worker['gpu_id']}] return_code={worker['return_code']} "
            f"videos={worker['videos']} log={worker['log']}\n")
    log.flush()
    failures = [worker for worker in worker_status if worker["return_code"] != 0]
    if failures:
        raise RuntimeError(
            f"TWM inference worker(s) failed; see {status_path}: {failures}")
    missing = [
        video for video in videos
        if not (prediction_root / video / "try_predictions.npz").is_file()
        or not (prediction_root / video / "try_inference.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Inference workers did not produce predictions for: {missing}")


def _resolve_videos(args, protocol: dict) -> list[str]:
    selected = args.video if args.video else protocol.get("test_sequences", [])
    if not selected:
        raise ValueError("Provide --video or a protocol with test_sequences")
    if len(selected) != len(set(selected)):
        raise ValueError("Selected video list contains duplicates")
    set_path = args.video_root / args.set_file
    if not set_path.is_file():
        # Keep the requested/frozen list for a useful blocked preflight record.
        return list(selected)
    available = json.loads(set_path.read_text(encoding="utf-8"))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Selected videos are absent from {args.set_file}: {unknown}")
    return list(selected)


def _run_dir(args) -> Path:
    value = args.run_dir or HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    value = value.expanduser().resolve()
    if not value.is_relative_to(HERE.resolve()):
        raise ValueError("run_dir must be inside thirdparty/ablation/twm_only")
    if value.exists():
        raise FileExistsError(f"Refusing to overwrite pipeline run: {value}")
    return value


def _load_inference_metadata(prediction_root: Path, videos: list[str]) -> dict[str, dict]:
    result = {}
    for video in videos:
        path = prediction_root / video / "try_inference.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing inference metadata: {path}")
        result[video] = json.loads(path.read_text(encoding="utf-8"))
    return result


def main() -> None:
    args = _parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    videos = _resolve_videos(args, protocol)
    if (args.inference_gpus and args.device.startswith("cuda:")
            and args.device != "cuda:0"):
        raise ValueError(
            "With --inference_gpus each worker sees its physical card as cuda:0; "
            "use --device cuda or --device cuda:0")
    inference_groups = (
        _group_videos_by_gpu(videos, args.inference_gpus, args.video_root)
        if args.inference_gpus else None
    )
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True)
    prediction_root = run_dir / "predictions_and_wmpjpe"
    onset_root = run_dir / "onset"
    preflight_command = [sys.executable, str(INFER), "--checkpoint", str(args.checkpoint),
                         "--hawor_checkpoint", str(args.hawor_checkpoint),
                         "--video_root", str(args.video_root), "--set_file", args.set_file,
                         "--device", args.device, "--interval", str(args.interval), "--preflight"]
    preflight_command.extend(item for video in videos for item in ("--video", video))
    if args.allow_gt_boxes:
        preflight_command.append("--allow_gt_boxes")
    preflight_gpu_ids = (
        [gpu_id for gpu_id, _ in inference_groups]
        if inference_groups else [None]
    )
    preflight_results = []
    for gpu_id in preflight_gpu_ids:
        preflight_env = None
        if gpu_id is not None:
            preflight_env = os.environ.copy()
            preflight_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        preflight_result = subprocess.run(
            preflight_command, cwd=ROOT, env=preflight_env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        try:
            checks = json.loads(preflight_result.stdout)
        except json.JSONDecodeError:
            checks = {"ready": False, "checks": [], "preflight_output": preflight_result.stdout}
        preflight_results.append({
            "gpu_id": gpu_id,
            "return_code": preflight_result.returncode,
            "checks": checks,
        })
    failed_preflights = [
        item for item in preflight_results
        if item["return_code"] or not item["checks"].get("ready")
    ]
    if failed_preflights:
        multi_gpu_preflight = args.inference_gpus is not None
        blocked = {
            "status": "blocked",
            "reason": (
                "preflight found missing or invalid resources on a requested GPU; no inference/evaluation was started"
                if multi_gpu_preflight else
                "preflight found missing or invalid resources; no inference/evaluation was started"
            ),
            "videos": videos,
            "protocol": str(args.protocol.resolve()),
            "preflight_command": preflight_command,
            "preflight": (
                preflight_results if multi_gpu_preflight else preflight_results[0]["checks"]
            ),
            "completed_stages": [],
            "table_row_generated": False,
        }
        if multi_gpu_preflight:
            blocked["failed_gpu_preflights"] = failed_preflights
        (run_dir / "blocked_status.json").write_text(
            json.dumps(blocked, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(blocked, indent=2))
        raise SystemExit(2)

    inference_command = [sys.executable, str(INFER), "--checkpoint", str(args.checkpoint),
                         "--hawor_checkpoint", str(args.hawor_checkpoint),
                         "--video_root", str(args.video_root), "--set_file", args.set_file,
                         "--output_root", str(prediction_root), "--device", args.device,
                         "--interval", str(args.interval)]
    wmpjpe_command = [sys.executable, str(WMPJPE), "--video_root", str(args.video_root),
                      "--set_file", args.set_file, "--prediction_root", str(prediction_root),
                      "--eval_log_dir", str(prediction_root), "--device", args.device,
                      "--interval", str(args.interval)]
    wmpjpe_command.extend(item for video in videos for item in ("--video", video))
    onset_command = [sys.executable, str(ONSET), "--video_root", str(args.video_root),
                     "--set_file", args.set_file, "--eval_log_dir", str(prediction_root),
                     "--protocol", str(args.protocol), "--output", str(onset_root),
                     "--device", args.device]
    onset_command.extend(item for video in videos for item in ("--video", video))

    completed = []
    try:
        with (run_dir / "pipeline.log").open("w", encoding="utf-8") as log:
            if args.inference_gpus:
                worker_command_prefix = list(inference_command)
                if args.allow_gt_boxes:
                    worker_command_prefix.append("--allow_gt_boxes")
                _run_inference_workers(
                    worker_command_prefix, videos, args.inference_gpus, args.video_root,
                    prediction_root, run_dir, log, groups=inference_groups)
            else:
                single_gpu_command = list(inference_command)
                single_gpu_command.extend(
                    item for video in videos for item in ("--video", video))
                if args.allow_gt_boxes:
                    single_gpu_command.append("--allow_gt_boxes")
                _run(single_gpu_command, log, "TWM-only inference")
            completed.append("inference")
            _run(wmpjpe_command, log, "W-MPJPE evaluation")
            completed.append("w_mpjpe")
            bind_evaluation_sources(prediction_root, videos)
            _run(onset_command, log, "Frozen-protocol onset evaluation")
            completed.append("onset")

        inference = _load_inference_metadata(prediction_root, videos)
        wmpjpe_path = prediction_root / f"evaluation_summary_{len(videos)}videos.json"
        wmpjpe = json.loads(wmpjpe_path.read_text(encoding="utf-8"))
        onset = json.loads((onset_root / "summary.json").read_text(encoding="utf-8"))
        try:
            row = build_table_row(videos, inference, wmpjpe, onset,
                                  frozen_videos=list(protocol.get("test_sequences", [])))
        except ValueError as error:
            diagnostic = {
                "status": "complete_diagnostic_no_formal_row",
                "reason": str(error),
                "videos": videos,
                "frozen_protocol_videos": protocol.get("test_sequences", []),
                "completed_stages": completed,
                "table_row_generated": False,
                "run_dir": str(run_dir),
                "frontend_scope_matches_frozen_pinch_protocol": all(
                    bool(metadata.get("oracle_assisted"))
                    and bool(metadata.get("baseline_frontend_scope_matches_frozen_pinch_protocol"))
                    for metadata in inference.values()),
            }
            (run_dir / "diagnostic_status.json").write_text(
                json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(diagnostic, indent=2))
            return
        (run_dir / "table_row.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        (run_dir / "table_row.md").write_text(render_table_row(row), encoding="utf-8")
        status = {
            "status": "complete",
            "videos": videos,
            "completed_stages": completed,
            "table_row_generated": True,
            "run_dir": str(run_dir),
            "frontend_scope_matches_frozen_pinch_protocol": True,
        }
        (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(status, indent=2))
    except Exception as error:
        failed = {
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "videos": videos,
            "completed_stages": completed,
            "table_row_generated": False,
            "run_dir": str(run_dir),
        }
        (run_dir / "failed_status.json").write_text(json.dumps(failed, indent=2) + "\n",
                                                    encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
