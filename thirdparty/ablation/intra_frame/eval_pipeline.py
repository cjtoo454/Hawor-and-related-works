#!/usr/bin/env python3
"""Run frozen-protocol W-MPJPE and pinch evaluation for Intra-frame Feedback."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT / "thirdparty" / "try"
INFER = HERE / "run_intra.py"
WMPJPE = PROJECT / "scripts" / "eval_hot3d.py"
TWM_ONLY = HERE.parent / "twm_only"
BASELINE_ROOT = (
    TWM_ONLY / "runs/restore_full_evaluation_20261020/predictions_and_wmpjpe"
)
PROTOCOL = (
    ROOT / "diagnostics/pinch_hawor_vs_ema_10videos_20261018_final/protocol_frozen.json"
)
PROTOCOL_RECORD = HERE / "records" / "protocol.json"
EXPECTED_PROTOCOL_SHA256 = "f62cba1337a0af9acf2e07ab878eaa6e1356539e522e61d7556b724d02e482fd"
FROZEN_EVALUATED_FRAMES = 34504
EXPECTED_METHOD = "TWM + Intra-frame Feedback; raw HaWoR history"
sys.path[:0] = [str(ROOT), str(PROJECT), str(PROJECT / "scripts"), str(TWM_ONLY)]

from dual_loop.geometry import ManoGeometry  # noqa: E402
from eval_twm_onset import _aggregate, _relative_timestamps, decode_gt_joints, score_twm_onsets  # noqa: E402
from runtime import sha256  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual_checkpoint", type=Path, required=True)
    parser.add_argument("--twm_checkpoint", type=Path, required=True)
    parser.add_argument("--hawor_checkpoint", type=Path,
                        default=ROOT / "weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--video_root", type=Path, default=ROOT / "datasets/hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--baseline_root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--history_batch_size", type=int, default=16)
    return parser.parse_args()


def _run_dir(value: Path | None) -> Path:
    result = value or (HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    result = result.expanduser().resolve()
    if not result.is_relative_to((HERE / "runs").resolve()):
        raise ValueError("run_dir must remain under thirdparty/ablation/intra_frame/runs")
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite Intra-frame evaluation: {result}")
    return result


def _validate_protocol(protocol: dict, protocol_path: Path) -> list[str]:
    actual_hash = sha256(protocol_path)
    if actual_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            f"Frozen protocol SHA256 mismatch: {actual_hash} != {EXPECTED_PROTOCOL_SHA256}")
    if not PROTOCOL_RECORD.is_file():
        raise FileNotFoundError(f"Missing pre-frozen protocol record: {PROTOCOL_RECORD}")
    record = json.loads(PROTOCOL_RECORD.read_text(encoding="utf-8"))
    if record.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Pre-frozen record does not contain the approved protocol SHA256")
    recorded_path = Path(record.get("protocol", "")).expanduser().resolve()
    if protocol_path.expanduser().resolve() != recorded_path:
        raise ValueError("Protocol path differs from the pre-frozen experiment record")
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = ("prediction_threshold", "reference_threshold", "min_frames",
                "onset_tolerance_sec", "test_sequences")
    missing = [field for field in required if field not in protocol]
    if missing:
        raise ValueError(f"Frozen protocol is missing fields: {missing}")
    if protocol["test_sequences"] != frozen["test_sequences"]:
        raise ValueError("Frozen test sequence list has changed")
    if protocol["test_sequences"] != record.get("test_sequences"):
        raise ValueError("Protocol sequence list differs from the pre-frozen experiment record")
    if (protocol["prediction_threshold"] != frozen["prediction_threshold"]
            or protocol["reference_threshold"] != frozen["reference_threshold"]
            or protocol["min_frames"] != frozen["min_frames"]
            or protocol["onset_tolerance_sec"] != frozen["onset_tolerance_sec"]):
        raise ValueError("Frozen event thresholds or matching settings have changed")
    return list(frozen["test_sequences"])


def _validate_baseline_protocol(summary_path: Path, protocol: dict) -> None:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing first-row onset summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("First-row onset summary used a different frozen protocol")
    for key in ("prediction_threshold", "reference_threshold"):
        if summary.get(key) != protocol.get(key):
            raise ValueError(f"First-row onset summary disagrees on frozen {key}")
    if summary.get("num_videos") != 10:
        raise ValueError("First-row onset summary does not cover all ten frozen videos")


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


def _bind_sources(prediction_root: Path, sequences: list[str], protocol_path: Path) -> None:
    for sequence in sequences:
        directory = prediction_root / sequence
        prediction = directory / "try_predictions.npz"
        details = directory / "evaluation_details.npz"
        metadata_path = directory / "try_inference.json"
        trace_path = directory / "intra_frame_trace.npz"
        paths = (prediction, details, metadata_path, trace_path)
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Cannot bind evaluation inputs for {sequence}: {directory}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("method") != EXPECTED_METHOD or not metadata.get("trained_checkpoint_loaded"):
            raise ValueError(f"Inference provenance is not trained Intra-frame Feedback: {metadata_path}")
        baseline_dir = Path(metadata["baseline_sequence_dir"])
        baseline_trace = baseline_dir / "twm_only_trace.npz"
        manifest = {
            "schema": "intra_frame_evaluation_source_v1",
            "sequence": sequence,
            "method": metadata["method"],
            "visual_checkpoint_sha256": metadata.get("visual_checkpoint_sha256"),
            "twm_checkpoint_sha256": metadata.get("twm_checkpoint_sha256"),
            "hawor_checkpoint_sha256": metadata.get("hawor_checkpoint_sha256"),
            "prediction_sha256": sha256(prediction),
            "evaluation_details_sha256": sha256(details),
            "inference_metadata_sha256": sha256(metadata_path),
            "trace_sha256": sha256(trace_path),
            "baseline_trace_sha256": sha256(baseline_trace),
            "track_sha256": metadata.get("track_sha256"),
            "slam_source_chunks": metadata.get("slam_source_chunks"),
            "protocol_sha256": sha256(protocol_path),
        }
        (directory / "evaluation_source.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _verify_shared_frame_ids(prediction_root: Path, baseline_root: Path,
                             sequences: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sequence in sequences:
        current_path = prediction_root / sequence / "evaluation_details.npz"
        baseline_path = baseline_root / sequence / "evaluation_details.npz"
        if not current_path.is_file() or not baseline_path.is_file():
            raise FileNotFoundError(f"Missing W-MPJPE details for shared-frame check: {sequence}")
        with np.load(current_path, allow_pickle=False) as current, \
                np.load(baseline_path, allow_pickle=False) as baseline:
            current_ids = current["frame_ids"].copy()
            baseline_ids = baseline["frame_ids"].copy()
        if not np.array_equal(current_ids, baseline_ids):
            raise ValueError(
                f"Intra-frame and TWM-only W-MPJPE frame_ids differ for {sequence}: "
                f"{len(current_ids)} vs {len(baseline_ids)}")
        counts[sequence] = int(len(current_ids))
    total = sum(counts.values())
    if total != FROZEN_EVALUATED_FRAMES:
        raise ValueError(
            f"Shared evaluation covered {total} frames, expected frozen first-row count "
            f"{FROZEN_EVALUATED_FRAMES}")
    return counts


def _normalize_wmpjpe_method(prediction_root: Path, sequences: list[str]) -> None:
    """Keep evaluator provenance while labelling outputs with the actual method."""
    aggregate_path = prediction_root / f"evaluation_summary_{len(sequences)}videos.json"
    paths = [aggregate_path, *(prediction_root / sequence / "evaluation_summary.json"
                               for sequence in sequences)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing W-MPJPE summary to normalize: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        evaluator_method = summary.get("method")
        if evaluator_method and evaluator_method != EXPECTED_METHOD:
            summary["evaluator_method"] = evaluator_method
        summary["method"] = EXPECTED_METHOD
        for row in summary.get("videos", []):
            row_method = row.get("method")
            if row_method and row_method != EXPECTED_METHOD:
                row["evaluator_method"] = row_method
            row["method"] = EXPECTED_METHOD
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _score_onsets(
    *, video_root: Path, prediction_root: Path, output_dir: Path,
    sequences: list[str], protocol: dict, device: str, protocol_path: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite onset results: {output_dir}")
    output_dir.mkdir(parents=True)
    geometry = ManoGeometry(ROOT).to(device).eval()
    per_video = []
    aggregate_rows = []
    sources = []
    for sequence in sequences:
        sequence_root = video_root / sequence
        directory = prediction_root / sequence
        details_path = directory / "evaluation_details.npz"
        prediction_path = directory / "try_predictions.npz"
        metadata_path = directory / "try_inference.json"
        manifest_path = directory / "evaluation_source.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("method") != EXPECTED_METHOD
                or not metadata.get("trained_checkpoint_loaded")
                or int(metadata.get("visual_global_step", 0)) <= 0
                or int(metadata.get("twm_global_step", 0)) <= 0):
            raise ValueError(f"Invalid or untrained Intra-frame provenance: {metadata_path}")
        if metadata.get("processed_frames") != metadata.get("total_frames"):
            raise ValueError(f"Incomplete inference cannot be scored: {metadata_path}")
        with np.load(details_path, allow_pickle=False) as archive:
            predicted_joints = archive["pred_world_joints_m"].copy()
            frame_ids = archive["frame_ids"].copy()
        predicted_valid = np.isfinite(predicted_joints).all((1, 2))
        if not np.array_equal(np.flatnonzero(predicted_valid), frame_ids):
            raise ValueError(f"W-MPJPE detail mask does not match frame_ids for {sequence}")
        annotations_path = sequence_root / "anno.pth"
        annotations = joblib.load(annotations_path)
        reference_valid = torch.any(annotations["rot_r"] != 0, dim=-1).cpu().numpy()
        if len(reference_valid) != len(predicted_joints):
            raise ValueError(f"GT/prediction frame count mismatch: {sequence}")
        reference_joints = decode_gt_joints(geometry, annotations, reference_valid, device)
        times = _relative_timestamps(sequence_root)
        if len(times) != len(predicted_joints):
            raise ValueError(f"Timestamp/prediction frame count mismatch: {sequence}")
        scored = score_twm_onsets(
            predicted_joints, reference_joints, times, predicted_valid, reference_valid,
            prediction_threshold=float(protocol["prediction_threshold"]),
            reference_threshold=float(protocol["reference_threshold"]),
            min_frames=int(protocol["min_frames"]),
            onset_tolerance_sec=float(protocol["onset_tolerance_sec"]),
        )
        metrics = scored["metrics"]
        video_row = {"sequence": sequence,
                     **{key: value for key, value in metrics.items() if not isinstance(value, list)}}
        per_video.append(video_row)
        aggregate_rows.append(metrics)
        events_path = output_dir / f"{sequence}_events.json"
        events_path.write_text(json.dumps({
            "method": EXPECTED_METHOD,
            "reference_events": [asdict(event) for event in scored["reference_events"]],
            "predicted_events": [asdict(event) for event in scored["predicted_events"]],
            **metrics,
        }, indent=2) + "\n", encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources.append({
            "sequence": sequence,
            "prediction_sha256": sha256(prediction_path),
            "evaluation_details_sha256": sha256(details_path),
            "inference_metadata_sha256": sha256(metadata_path),
            "evaluation_source_sha256": sha256(manifest_path),
            "annotation_sha256": sha256(annotations_path),
            "timestamp_sha256": sha256(sequence_root / "tt_list.pkl"),
            "track_sha256": manifest["track_sha256"],
            "slam_source_chunks": manifest["slam_source_chunks"],
            "protocol_sha256": manifest["protocol_sha256"],
        })
    summary = {
        "method": EXPECTED_METHOD,
        "num_videos": len(per_video),
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "prediction_threshold": float(protocol["prediction_threshold"]),
        "reference_threshold": float(protocol["reference_threshold"]),
        "reference_type": protocol.get("reference_type"),
        "per_video": per_video,
        "aggregate": _aggregate(aggregate_rows),
        "sources": sources,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if per_video:
        fields = list(dict.fromkeys(key for row in per_video for key in row))
        with (output_dir / "per_video.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_video)
    return summary


def build_table_row(
    sequences: list[str], inference: dict[str, dict], wmpjpe: dict,
    onset: dict, common_frame_counts: dict[str, int],
) -> dict:
    if sequences != json.loads(PROTOCOL.read_text(encoding="utf-8"))["test_sequences"]:
        raise ValueError("Formal row requires the exact frozen ten-video sequence order")
    inferred = set(inference)
    evaluated = {row["video"] for row in wmpjpe.get("videos", [])}
    onset_videos = {row["sequence"] for row in onset.get("per_video", [])}
    if inferred != set(sequences) or evaluated != set(sequences) or onset_videos != set(sequences):
        raise ValueError("Inference, W-MPJPE, and event scoring do not contain the same ten videos")
    incomplete = {
        name: (meta.get("processed_frames"), meta.get("total_frames"))
        for name, meta in inference.items()
        if meta.get("processed_frames") != meta.get("total_frames")
    }
    if incomplete:
        raise ValueError(f"Incomplete videos cannot produce a result row: {incomplete}")
    if sum(common_frame_counts.values()) != FROZEN_EVALUATED_FRAMES:
        raise ValueError("Formal row does not use the first-row 34,504 common frames")
    methods = {meta.get("method") for meta in inference.values()}
    if methods != {EXPECTED_METHOD}:
        raise ValueError(f"Unexpected methods: {methods}")
    if wmpjpe.get("method") != EXPECTED_METHOD:
        raise ValueError(f"W-MPJPE aggregate method label is incorrect: {wmpjpe.get('method')}")
    wmpjpe_methods = {row.get("method") for row in wmpjpe.get("videos", [])}
    if wmpjpe_methods != {EXPECTED_METHOD}:
        raise ValueError(f"W-MPJPE summary method labels are incorrect: {wmpjpe_methods}")
    frozen_protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (onset.get("method") != EXPECTED_METHOD
            or onset.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
            or onset.get("prediction_threshold") != frozen_protocol["prediction_threshold"]):
        raise ValueError("Frozen onset summary method or threshold does not match the protocol")
    if not all(meta.get("baseline_raw_input_verification") for meta in inference.values()):
        raise ValueError("Formal row requires successful first-row raw input verification")
    base_hashes = {
        (meta.get("twm_checkpoint_sha256"), meta.get("hawor_checkpoint_sha256"),
         meta.get("visual_checkpoint_sha256"), meta.get("visual_global_step"))
        for meta in inference.values()
    }
    if len(base_hashes) != 1:
        raise ValueError("Videos do not share one visual/TWM/HaWoR checkpoint set")
    twm_hash, hawor_hash, visual_hash, visual_step = next(iter(base_hashes))
    if not visual_step or int(visual_step) <= 0:
        raise ValueError("Formal row requires trained visual weights")
    w = wmpjpe.get("aggregate", {})
    o = onset.get("aggregate", {})
    for name, value in (("w_mpjpe_mm", w.get("w_mpjpe_mm")),
                        ("onset_mae_ms", o.get("onset_mae_ms"))):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Formal row requires finite {name}")
    return {
        "method": EXPECTED_METHOD,
        "num_videos": len(sequences),
        "evaluated_right_hand_frames": sum(common_frame_counts.values()),
        "visual_checkpoint_sha256": visual_hash,
        "visual_global_step": int(visual_step),
        "twm_checkpoint_sha256": twm_hash,
        "hawor_checkpoint_sha256": hawor_hash,
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
        "common_frame_counts": common_frame_counts,
    }


def _render_row(row: dict) -> str:
    columns = ("method", "num_videos", "evaluated_right_hand_frames", "w_mpjpe_mm",
               "wa_mpjpe_mm", "pa_mpjpe_mm", "accel_m_s2", "rte_percent",
               "onset_mae_ms", "event_f1", "visual_global_step", "oracle_assisted")
    headers = ("Method", "Videos", "Common frames", "W-MPJPE (mm)", "WA-MPJPE (mm)",
               "PA-MPJPE (mm)", "Accel (m/s²)", "RTE (%)", "Onset MAE (ms)",
               "Event F1", "Visual step", "Oracle-assisted")

    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) \
        + " |\n| " + " | ".join(cell(row.get(column)) for column in columns) + " |\n"


def main() -> None:
    args = _parse_args()
    if args.history_batch_size < 1:
        raise ValueError("history_batch_size must be positive")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sequences = _validate_protocol(protocol, PROTOCOL)
    baseline_summary_path = args.baseline_root / "evaluation_summary_10videos.json"
    if not baseline_summary_path.is_file():
        raise FileNotFoundError(f"Missing frozen first-row W-MPJPE summary: {baseline_summary_path}")
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    if (baseline_summary.get("aggregate", {}).get("num_videos") != 10
            or baseline_summary.get("aggregate", {}).get("evaluated_right_hand_frames")
            != FROZEN_EVALUATED_FRAMES):
        raise ValueError("Archived first-row summary does not match the frozen ten-video/34,504-frame baseline")
    baseline_videos = {row.get("video") for row in baseline_summary.get("videos", [])}
    if baseline_videos != set(sequences):
        raise ValueError("Archived first-row W-MPJPE video set differs from the frozen protocol")
    _validate_baseline_protocol(args.baseline_root.parent / "onset" / "summary.json", protocol)
    available = json.loads((args.video_root / args.set_file).read_text(encoding="utf-8"))
    missing = sorted(set(sequences) - set(available))
    if missing:
        raise ValueError(f"Frozen protocol videos are absent from {args.set_file}: {missing}")
    run_dir = _run_dir(args.run_dir)
    run_dir.mkdir(parents=True)
    prediction_root = run_dir / "predictions_and_wmpjpe"
    onset_root = run_dir / "onset"
    interpreter = sys.executable
    inference_command = [
        interpreter, str(INFER),
        "--visual_checkpoint", str(args.visual_checkpoint),
        "--twm_checkpoint", str(args.twm_checkpoint),
        "--hawor_checkpoint", str(args.hawor_checkpoint),
        "--video_root", str(args.video_root),
        "--set_file", args.set_file,
        "--baseline_root", str(args.baseline_root),
        "--protocol", str(PROTOCOL),
        "--output_root", str(prediction_root),
        "--device", args.device,
        "--interval", "1000",
        "--history_batch_size", str(args.history_batch_size),
    ]
    inference_command.extend(item for sequence in sequences for item in ("--video", sequence))
    wmpjpe_command = [
        interpreter, str(WMPJPE),
        "--video_root", str(args.video_root),
        "--set_file", args.set_file,
        "--prediction_root", str(prediction_root),
        "--eval_log_dir", str(prediction_root),
        "--device", args.device,
        "--interval", "1000",
    ]
    wmpjpe_command.extend(item for sequence in sequences for item in ("--video", sequence))
    completed = []
    try:
        with (run_dir / "pipeline.log").open("w", encoding="utf-8") as log:
            _run(inference_command, log, "Intra-frame inference")
            completed.append("inference")
            _run(wmpjpe_command, log, "W-MPJPE evaluation")
            completed.append("w_mpjpe")
            _normalize_wmpjpe_method(prediction_root, sequences)
            completed.append("wmpjpe_method_normalization")
        commands = {
            "interpreter": interpreter,
            "pipeline_invocation": [interpreter, str(Path(__file__).resolve()), *sys.argv[1:]],
            "inference": inference_command,
            "w_mpjpe": wmpjpe_command,
            "pipeline_script_sha256": sha256(Path(__file__)),
            "inference_script_sha256": sha256(INFER),
            "runtime_script_sha256": sha256(HERE / "runtime.py"),
            "evaluation_script_sha256": sha256(WMPJPE),
        }
        (run_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n",
                                                encoding="utf-8")
        common_counts = _verify_shared_frame_ids(prediction_root, args.baseline_root, sequences)
        _bind_sources(prediction_root, sequences, PROTOCOL)
        completed.append("shared_frame_verification")
        inference = {
            sequence: json.loads((prediction_root / sequence / "try_inference.json").read_text())
            for sequence in sequences
        }
        wmpjpe_path = prediction_root / f"evaluation_summary_{len(sequences)}videos.json"
        wmpjpe = json.loads(wmpjpe_path.read_text(encoding="utf-8"))
        wmpjpe_frames = int(wmpjpe.get("aggregate", {}).get("evaluated_right_hand_frames", -1))
        if wmpjpe_frames != FROZEN_EVALUATED_FRAMES:
            raise ValueError(f"W-MPJPE evaluated {wmpjpe_frames} frames, expected 34504")
        onset = _score_onsets(
            video_root=args.video_root, prediction_root=prediction_root,
            output_dir=onset_root, sequences=sequences, protocol=protocol,
            device=args.device, protocol_path=PROTOCOL,
        )
        completed.append("frozen_pinch_events")
        row = build_table_row(sequences, inference, wmpjpe, onset, common_counts)
        (run_dir / "table_row.json").write_text(json.dumps(row, indent=2) + "\n",
                                                encoding="utf-8")
        (run_dir / "table_row.md").write_text(_render_row(row), encoding="utf-8")
        input_manifest = {
            "protocol": str(PROTOCOL.resolve()),
            "protocol_sha256": sha256(PROTOCOL),
            "baseline_root": str(args.baseline_root.resolve()),
            "baseline_summary_sha256": sha256(
                args.baseline_root / "evaluation_summary_10videos.json"),
            "baseline_onset_summary_sha256": sha256(
                args.baseline_root.parent / "onset" / "summary.json"),
            "visual_checkpoint": str(args.visual_checkpoint.resolve()),
            "visual_checkpoint_sha256": sha256(args.visual_checkpoint),
            "twm_checkpoint": str(args.twm_checkpoint.resolve()),
            "twm_checkpoint_sha256": sha256(args.twm_checkpoint),
            "hawor_checkpoint": str(args.hawor_checkpoint.resolve()),
            "hawor_checkpoint_sha256": sha256(args.hawor_checkpoint),
            "videos": sequences,
            "shared_frame_counts": common_counts,
            "shared_frame_total": sum(common_counts.values()),
            "interval": 1000,
            "oracle_assisted_gt_boxes": True,
        }
        (run_dir / "input_manifest.json").write_text(
            json.dumps(input_manifest, indent=2) + "\n", encoding="utf-8")
        status = {
            "status": "complete",
            "videos": sequences,
            "completed_stages": completed,
            "table_row_generated": True,
            "run_dir": str(run_dir),
            "shared_evaluated_frames": sum(common_counts.values()),
        }
        (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n",
                                             encoding="utf-8")
        print(json.dumps(status, indent=2))
    except Exception as error:
        failure = {
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "videos": sequences,
            "completed_stages": completed,
            "table_row_generated": False,
            "run_dir": str(run_dir),
        }
        (run_dir / "failed_status.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
