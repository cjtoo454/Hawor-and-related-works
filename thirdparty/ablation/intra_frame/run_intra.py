#!/usr/bin/env python3
"""Run trained Intra-frame Feedback with the fixed TWM and HaWoR frontend."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TWM_ONLY = HERE.parent / "twm_only"
sys.path[:0] = [str(ROOT), str(HERE), str(TWM_ONLY)]

import run_twm_only as twm_only  # noqa: E402
from runtime import (  # noqa: E402
    INTRA_ARCHITECTURE,
    load_intra_system,
    sha256,
    validate_intra_checkpoint,
    infer_sequence,
)

DEFAULT_BASELINE_ROOT = (
    TWM_ONLY / "runs/restore_full_evaluation_20261020/predictions_and_wmpjpe"
)
DEFAULT_PROTOCOL = (
    ROOT / "diagnostics/pinch_hawor_vs_ema_10videos_20261018_final/protocol_frozen.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual_checkpoint", type=Path, required=True,
                        help="Trained handloop_intra_v1 checkpoint with visual weights only")
    parser.add_argument("--twm_checkpoint", type=Path, required=True,
                        help="Exact motion_pretrain checkpoint used by the first-row TWM-only run")
    parser.add_argument("--hawor_checkpoint", type=Path,
                        default=ROOT / "weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--video_root", type=Path, default=ROOT / "datasets/hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--video", action="append",
                        help="Optional sequence selection; repeat for multiple sequences")
    parser.add_argument("--baseline_root", type=Path, default=DEFAULT_BASELINE_ROOT,
                        help="Archived TWM-only predictions/traces to verify common raw inputs")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output_root", type=Path,
                        default=HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interval", type=int, default=1000,
                        help="Must match the frozen first-row HOT3D SLAM interval (1000)")
    parser.add_argument("--history_batch_size", type=int, default=16)
    parser.add_argument("--preflight", action="store_true",
                        help="Validate paths/checkpoint hashes without initializing HaWoR or MANO")
    return parser.parse_args()


def _selected_sequences(args: argparse.Namespace, protocol: dict) -> list[str]:
    requested = list(args.video) if args.video else list(protocol.get("test_sequences", []))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Sequence selection must be non-empty and contain no duplicates")
    available_path = args.video_root / args.set_file
    if not available_path.is_file():
        raise FileNotFoundError(f"Missing sequence set: {available_path}")
    available = json.loads(available_path.read_text(encoding="utf-8"))
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Sequences are absent from {args.set_file}: {unknown}")
    return [sequence for sequence in requested if sequence in set(available)]


def _preflight(args: argparse.Namespace, sequences: list[str]) -> dict:
    checks: list[dict[str, str]] = []
    for name, path in (
        ("visual checkpoint", args.visual_checkpoint),
        ("fixed TWM checkpoint", args.twm_checkpoint),
        ("HaWoR checkpoint", args.hawor_checkpoint),
        ("frozen protocol", args.protocol),
        ("HOT3D sequence root", args.video_root),
        ("first-row baseline root", args.baseline_root),
        ("right-hand MANO asset", ROOT / "_DATA/data/mano/MANO_RIGHT.pkl"),
    ):
        checks.append({"name": name, "path": str(path),
                       "status": "ok" if path.exists() else "missing"})
    if args.interval != 1000:
        checks.append({"name": "SLAM interval", "path": str(args.interval),
                       "status": "invalid: frozen protocol requires 1000"})
    if args.history_batch_size < 1:
        checks.append({"name": "history batch size", "path": str(args.history_batch_size),
                       "status": "invalid: must be positive"})
    if str(args.device).startswith("cuda"):
        valid = torch.cuda.is_available()
        index_text = str(args.device).partition(":")[2]
        if index_text:
            valid = valid and index_text.isdigit() and int(index_text) < torch.cuda.device_count()
        checks.append({"name": "CUDA device", "path": str(args.device),
                       "status": "ok" if valid else "unavailable"})
    if args.visual_checkpoint.is_file() and args.twm_checkpoint.is_file() and args.hawor_checkpoint.is_file():
        try:
            visual = twm_only.load_checkpoint_restricted(args.visual_checkpoint)
            step, _, _ = validate_intra_checkpoint(visual)
            twm_hash, hawor_hash = sha256(args.twm_checkpoint), sha256(args.hawor_checkpoint)
            checks.append({
                "name": "trained visual checkpoint and base hashes",
                "path": str(args.visual_checkpoint),
                "status": ("ok" if step > 0
                           and visual["base_twm_sha256"] == twm_hash
                           and visual["base_hawor_sha256"] == hawor_hash else "invalid"),
            })
            base = twm_only.load_checkpoint_restricted(args.twm_checkpoint)
            twm_only.validate_trained_checkpoint(base)
            status = "ok" if base.get("stage") == "motion_pretrain" else "invalid: expected motion_pretrain"
            checks.append({"name": "fixed TWM checkpoint stage", "path": str(args.twm_checkpoint),
                           "status": status})
        except Exception as error:
            checks.append({"name": "checkpoint validation", "path": str(args.visual_checkpoint),
                           "status": f"invalid: {type(error).__name__}: {error}"})
    for sequence in sequences:
        root = args.video_root / sequence
        baseline = args.baseline_root / sequence
        image_count = len(list((root / "extracted_images").glob("*.jpg")))
        checks.append({"name": f"{sequence} images", "path": str(root / "extracted_images"),
                       "status": "ok" if image_count else "missing"})
        for name in ("try_inference.json", "twm_only_trace.npz", "evaluation_details.npz"):
            path = baseline / name
            checks.append({"name": f"{sequence} baseline {name}", "path": str(path),
                           "status": "ok" if path.is_file() else "missing"})
    return {"architecture": INTRA_ARCHITECTURE,
            "ready": all(item["status"] == "ok" for item in checks),
            "sequences": sequences, "checks": checks}


def main() -> None:
    args = _parse_args()
    if args.interval != 1000:
        raise ValueError("The frozen first-row evaluation requires --interval 1000")
    if args.history_batch_size < 1:
        raise ValueError("history_batch_size must be positive")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    sequences = _selected_sequences(args, protocol)
    report = _preflight(args, sequences)
    if args.preflight:
        print(json.dumps(report, indent=2))
        if not report["ready"]:
            raise SystemExit(2)
        return
    if not report["ready"]:
        print(json.dumps(report, indent=2), file=sys.stderr)
        raise SystemExit("Intra-frame inference preflight failed")

    system, base_metadata = load_intra_system(
        visual_checkpoint_path=args.visual_checkpoint,
        twm_checkpoint_path=args.twm_checkpoint,
        hawor_checkpoint_path=args.hawor_checkpoint,
        device=args.device,
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        metadata = infer_sequence(
            system,
            args.video_root / sequence,
            sequence=sequence,
            baseline_sequence_dir=args.baseline_root / sequence,
            output_dir=output_root / sequence,
            device=args.device,
            interval=args.interval,
            max_frames=None,
            base_metadata=base_metadata,
            visual_checkpoint_path=args.visual_checkpoint,
            allow_gt_boxes=True,
            history_batch_size=args.history_batch_size,
        )
        print(json.dumps({
            "sequence": sequence,
            "processed_frames": metadata["processed_frames"],
            "total_frames": metadata["total_frames"],
            "method": metadata["method"],
            "baseline_raw_input_verified": True,
        }), flush=True)


if __name__ == "__main__":
    main()
