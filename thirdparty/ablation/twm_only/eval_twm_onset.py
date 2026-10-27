#!/usr/bin/env python3
"""Score TWM-only pinch onsets with the repository's frozen HOT3D protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "try"
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(PROJECT)]

from dual_loop.geometry import ManoGeometry  # noqa: E402
from dual_loop.pinch_evaluation import extract_events, match_events  # noqa: E402
from dual_loop.state import axis_angle_to_matrix, pack_state  # noqa: E402
from dual_loop.oracle import normalized_pinch_distance  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "diagnostics/pinch_hawor_vs_ema_10videos_20261018_final/protocol_frozen.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_evaluation_source(details: Path, prediction: Path, metadata_path: Path) -> tuple[dict, Path]:
    manifest_path = details.parent / "evaluation_source.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing pipeline source binding: {manifest_path}; run run_pipeline.py to bind the current "
            "prediction archive to these W-MPJPE details")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "twm_only_evaluation_source_v1"
            or manifest.get("sequence") != details.parent.name):
        raise ValueError(f"Invalid TWM evaluation source manifest: {manifest_path}")
    expected = {
        "prediction_sha256": sha256(prediction),
        "evaluation_details_sha256": sha256(details),
        "inference_metadata_sha256": sha256(metadata_path),
        "method": metadata.get("method"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items()
                  if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"Prediction/details source binding mismatch: {mismatches}")
    return metadata, manifest_path


def score_twm_onsets(
    predicted_joints: np.ndarray,
    reference_joints: np.ndarray,
    timestamps: np.ndarray,
    predicted_valid: np.ndarray,
    reference_valid: np.ndarray,
    *,
    prediction_threshold: float,
    reference_threshold: float = 0.2,
    min_frames: int = 3,
    onset_tolerance_sec: float = 0.2,
) -> dict:
    """Apply the shared event extraction and one-to-one onset matcher."""
    predicted_joints = np.asarray(predicted_joints)
    reference_joints = np.asarray(reference_joints)
    if predicted_joints.shape != reference_joints.shape or predicted_joints.ndim != 3:
        raise ValueError("predicted/reference joints must have matching [N,21,3] shapes")
    pv = np.asarray(predicted_valid, dtype=bool) & np.isfinite(predicted_joints).all((1, 2))
    gv = np.asarray(reference_valid, dtype=bool) & np.isfinite(reference_joints).all((1, 2))
    pred_distance = normalized_pinch_distance(predicted_joints)
    ref_distance = normalized_pinch_distance(reference_joints)
    references = extract_events(ref_distance, timestamps, gv, reference_threshold,
                                min_frames=min_frames)
    predictions = extract_events(pred_distance, timestamps, pv, prediction_threshold,
                                 min_frames=min_frames)
    metrics = match_events(references, predictions,
                           onset_tolerance_sec=onset_tolerance_sec)
    return {
        "reference_events": references,
        "predicted_events": predictions,
        "prediction_distance": pred_distance,
        "reference_distance": ref_distance,
        "metrics": metrics,
        "prediction_valid": pv,
        "reference_valid": gv,
    }


def decode_gt_joints(geometry: ManoGeometry, annotations: dict,
                     valid: np.ndarray, device: str) -> np.ndarray:
    count = len(valid)
    result = np.full((count, 21, 3), np.nan, np.float32)
    ids = np.flatnonzero(valid)
    with torch.inference_mode():
        for start in range(0, len(ids), 256):
            chosen = ids[start:start + 256]
            get = lambda name: annotations[name][chosen].to(device=device, dtype=torch.float32)
            state = pack_state(
                get("trans_r"),
                axis_angle_to_matrix(get("rot_r"))[:, None],
                axis_angle_to_matrix(get("pose_r").reshape(-1, 15, 3)),
                get("betas_r"),
            )
            result[chosen] = geometry.joints(state).cpu().numpy()
    return result


def _relative_timestamps(sequence_root: Path) -> np.ndarray:
    with (sequence_root / "tt_list.pkl").open("rb") as stream:
        timestamp_ns = np.asarray(pickle.load(stream), dtype=np.int64)
    if timestamp_ns.ndim != 1 or not len(timestamp_ns):
        raise ValueError(f"Invalid HOT3D timestamp file under {sequence_root}")
    times = (timestamp_ns - timestamp_ns[0]).astype(np.float64) / 1e9
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Timestamps must be finite and strictly increasing: {sequence_root}")
    return times


def _aggregate(rows: list[dict]) -> dict:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    errors = [match["onset_error_ms"] for row in rows for match in row["matches"]]
    short = sum(row["short_reference_events"] for row in rows)
    short_tp = sum(row["short_matched_events"] for row in rows)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "reference_events": tp + fn,
        "predicted_events": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "onset_mae_ms": float(np.mean(np.abs(errors))) if errors else None,
        "onset_bias_ms": float(np.mean(errors)) if errors else None,
        "onset_p95_absolute_error_ms": float(np.percentile(np.abs(errors), 95)) if errors else None,
        "short_reference_events": short,
        "short_matched_events": short_tp,
        "short_event_recall": short_tp / short if short else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_root", type=Path, default=ROOT / "datasets/hot3d_valset_export")
    parser.add_argument("--eval_log_dir", type=Path, required=True,
                        help="eval_hot3d.py output directory containing per-sequence evaluation_details.npz")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--video", action="append")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite onset results: {args.output}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    required = ("prediction_threshold", "reference_threshold", "min_frames", "onset_tolerance_sec")
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ValueError(f"Frozen pinch protocol missing fields: {missing}")
    sequences = json.loads((args.video_root / args.set_file).read_text(encoding="utf-8"))
    if args.video:
        unknown = set(args.video) - set(sequences)
        if unknown:
            raise ValueError(f"Unknown sequences: {sorted(unknown)}")
        sequences = [sequence for sequence in sequences if sequence in args.video]
    if not sequences:
        raise ValueError("No sequences selected")
    geometry = ManoGeometry(ROOT).to(args.device).eval()
    args.output.mkdir(parents=True)
    video_rows = []
    source_rows = []
    for sequence in sequences:
        sequence_root = args.video_root / sequence
        details = args.eval_log_dir / sequence / "evaluation_details.npz"
        if not details.is_file():
            raise FileNotFoundError(f"Missing W-MPJPE evaluator details: {details}")
        with np.load(details, allow_pickle=False) as archive:
            predicted_joints = archive["pred_world_joints_m"].copy()
            archived_ids = archive["frame_ids"].copy()
        predicted_valid = np.isfinite(predicted_joints).all((1, 2))
        if not np.array_equal(np.flatnonzero(predicted_valid), archived_ids):
            raise ValueError(f"Prediction details validity does not match frame_ids: {sequence}")
        annotation_path = sequence_root / "anno.pth"
        annotations = joblib.load(annotation_path)
        reference_valid = torch.any(annotations["rot_r"] != 0, dim=-1).cpu().numpy()
        if len(reference_valid) != len(predicted_joints):
            raise ValueError(f"GT/prediction frame count mismatch: {sequence}")
        reference_joints = decode_gt_joints(geometry, annotations, reference_valid, args.device)
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
        row = {"sequence": sequence, **{key: value for key, value in metrics.items()
                                       if not isinstance(value, list)}}
        video_rows.append(row)
        predicted_path = args.eval_log_dir / sequence / "try_predictions.npz"
        metadata_path = predicted_path.parent / "try_inference.json"
        if not predicted_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Onset scoring requires TWM predictions and provenance beside evaluator details: {predicted_path}")
        metadata, source_manifest = verify_evaluation_source(details, predicted_path, metadata_path)
        if metadata.get("method") != "TWM only; raw HaWoR feedback":
            raise ValueError(f"Prediction provenance is not TWM-only: {metadata_path}")
        if not metadata.get("trained_checkpoint_loaded"):
            raise ValueError(f"Untrained output cannot be scored as TWM-only: {metadata_path}")
        if int(metadata.get("checkpoint_global_step", 0)) <= 0:
            raise ValueError(f"Inference checkpoint has no completed training steps: {metadata_path}")
        if metadata.get("processed_frames") != metadata.get("total_frames"):
            raise ValueError(f"Incomplete inference cannot produce onset results: {metadata_path}")
        if metadata.get("total_frames") != len(predicted_joints):
            raise ValueError(f"Inference/evaluator frame count mismatch: {sequence}")
        source_rows.append({
            "sequence": sequence,
            "prediction_details": str(details.resolve()),
            "prediction_details_sha256": sha256(details),
            "annotations": str(annotation_path.resolve()),
            "annotations_sha256": sha256(annotation_path),
            "timestamps": str((sequence_root / "tt_list.pkl").resolve()),
            "timestamps_sha256": sha256(sequence_root / "tt_list.pkl"),
            "inference_metadata": str(metadata_path.resolve()),
            "inference_metadata_sha256": sha256(metadata_path),
            "prediction": str(predicted_path.resolve()),
            "prediction_sha256": sha256(predicted_path),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "frontend_weights_source": metadata.get("frontend_weights_source"),
            "frontend_source_sha256": metadata.get("frontend_source_sha256"),
            "oracle_assisted": bool(metadata.get("oracle_assisted", False)),
            "evaluation_source_manifest": str(source_manifest.resolve()),
            "evaluation_source_manifest_sha256": sha256(source_manifest),
        })
        (args.output / f"{sequence}_events.json").write_text(json.dumps({
            "method": "TWM only; raw HaWoR feedback",
            "reference_events": [asdict(event) for event in scored["reference_events"]],
            "predicted_events": [asdict(event) for event in scored["predicted_events"]],
            **metrics,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"[{sequence}] events: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}; "
              f"onset MAE={metrics['onset_mae_ms']}", flush=True)

    checkpoint_hashes = {row.get("checkpoint_sha256") for row in source_rows}
    frontend_hashes = {row.get("frontend_source_sha256") for row in source_rows}
    if len(checkpoint_hashes) != 1 or len(frontend_hashes) != 1:
        raise ValueError("Onset inputs mix TWM checkpoints or effective frontend sources")
    summary = {
        "method": "TWM only; raw HaWoR feedback",
        "num_videos": len(video_rows),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "prediction_threshold": float(protocol["prediction_threshold"]),
        "reference_threshold": float(protocol["reference_threshold"]),
        "reference_type": protocol.get("reference_type"),
        "per_video": video_rows,
        "aggregate": _aggregate([row | {"matches": json.loads(
            (args.output / f"{row['sequence']}_events.json").read_text())["matches"]}
                                  for row in video_rows]),
        "sources": source_rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if video_rows:
        fields = list(dict.fromkeys(key for row in video_rows for key in row))
        with (args.output / "per_video.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(video_rows)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
