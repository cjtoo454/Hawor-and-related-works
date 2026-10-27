"""Run the vanilla DROID-SLAM + naive Metric3D branch of WiLoR-SLAM."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "thirdparty" / "Metric3D"))

from lib.eval_utils.custom_utils import split_list_by_interval  # noqa: E402
from lib.pipeline.est_scale import est_scale_wo_mask  # noqa: E402
from lib.pipeline.masked_droid_slam import (  # noqa: E402
    est_calib,
    get_dimention,
    run_droid_slam,
)
from metric import Metric3D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vanilla DROID-SLAM for WiLoR-SLAM")
    parser.add_argument("--video_root", type=Path,
                        default=ROOT / "datasets" / "hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--video", action="append")
    parser.add_argument("--metric_checkpoint", type=Path,
                        default=ROOT / "thirdparty" / "Metric3D" / "weights" /
                        "metric_depth_vit_large_800k.pth")
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--part", type=int, action="append",
                        help="Zero-based interval index; repeat to select parts")
    parser.add_argument("--stage", choices=("all", "droid", "scale"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_sequences(args: argparse.Namespace) -> list[str]:
    with (args.video_root / args.set_file).open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    if args.video:
        selected = set(args.video)
        unknown = sorted(selected - set(sequences))
        if unknown:
            raise ValueError(f"Sequences not in {args.set_file}: {unknown}")
        sequences = [name for name in sequences if name in selected]
    return sequences


def run_droid_parts(args: argparse.Namespace, sequence: str) -> list[tuple[int, int, Path]]:
    sequence_root = args.video_root / sequence
    images = sorted((sequence_root / "extracted_images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No extracted HOT3D frames for {sequence}")
    with (sequence_root / "focal.txt").open("r", encoding="utf-8") as handle:
        focal = float(handle.read())

    starts, ends, parts = split_list_by_interval(images, interval=args.interval)
    slam_dir = sequence_root / "preprocess" / "SLAM_WiLoR"
    slam_dir.mkdir(parents=True, exist_ok=True)
    records: list[tuple[int, int, Path]] = []
    for part_index, (start, end, part) in enumerate(zip(starts, ends, parts)):
        if args.part and part_index not in set(args.part):
            continue
        output_path = slam_dir / f"wilor_droid_w_scale_{start}_{end}.npz"
        raw_path = slam_dir / f"wilor_droid_raw_{start}_{end}.npz"
        records.append((start, end, raw_path))
        if output_path.exists() and not args.overwrite:
            print(f"[skip] scaled SLAM exists: {output_path}")
            continue
        if raw_path.exists() and not args.overwrite:
            print(f"[skip] raw SLAM exists: {raw_path}")
            continue

        image_names = [str(path) for path in part]
        calib = np.asarray(est_calib(image_names), dtype=np.float64)
        calib[:2] = focal
        print(f"[DROID] {sequence} frames {start}:{end} (vanilla, no hand mask)")
        droid, trajectory = run_droid_slam(image_names, calib=calib)
        keyframe_count = droid.video.counter.value
        timestamps = droid.video.tstamp.cpu().int().numpy()[:keyframe_count]
        disparities = droid.video.disps_up.cpu().numpy()[:keyframe_count]
        np.savez(
            raw_path,
            tstamp=timestamps,
            disps=disparities,
            traj=trajectory,
            img_focal=focal,
            img_center=calib[-2:],
        )
        del droid
        torch.cuda.empty_cache()
        gc.collect()
    return records


def add_metric_scale(args: argparse.Namespace, sequence: str,
                     records: list[tuple[int, int, Path]]) -> None:
    pending = []
    sequence_root = args.video_root / sequence
    all_images = sorted((sequence_root / "extracted_images").glob("*.jpg"))
    slam_dir = sequence_root / "preprocess" / "SLAM_WiLoR"
    for start, end, raw_path in records:
        output_path = slam_dir / f"wilor_droid_w_scale_{start}_{end}.npz"
        if output_path.exists() and not args.overwrite:
            continue
        pending.append((start, end, raw_path, output_path))
    if not pending:
        return

    print(f"[Metric3D] loading {args.metric_checkpoint}")
    metric = Metric3D(str(args.metric_checkpoint))
    for start, end, raw_path, output_path in pending:
        raw = np.load(raw_path)
        part = all_images[start:end]
        calib = np.asarray(est_calib([str(path) for path in part]), dtype=np.float64)
        calib[:2] = float(raw["img_focal"])
        height, width = get_dimention([str(path) for path in part])
        scales = []
        print(f"[Metric3D] {sequence} frames {start}:{end}, {len(raw['tstamp'])} keyframes")
        for keyframe_index, timestamp in enumerate(tqdm(raw["tstamp"], unit="keyframe")):
            pred_depth = metric(str(part[int(timestamp)]), calib.copy())
            pred_depth = cv2.resize(pred_depth, (width, height))
            slam_depth = 1.0 / np.clip(raw["disps"][keyframe_index], 1e-8, None)
            scale = est_scale_wo_mask(slam_depth, pred_depth, sigma=0.5)
            if np.isfinite(scale) and scale > 0:
                scales.append(scale)
        if not scales:
            raise RuntimeError(f"No finite Metric3D scale estimates for {sequence} {start}:{end}")
        median_scale = float(np.median(scales))
        np.savez(
            output_path,
            tstamp=raw["tstamp"],
            disps=raw["disps"],
            traj=raw["traj"],
            img_focal=raw["img_focal"],
            img_center=raw["img_center"],
            scale=median_scale,
            scale_samples=np.asarray(scales, dtype=np.float32),
            scale_method=np.asarray("Metric3D est_scale_wo_mask"),
        )
        print(f"[done] {output_path}: scale={median_scale:.6f}")

    del metric
    torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    args = parse_args()
    for sequence in load_sequences(args):
        if args.stage in ("all", "droid"):
            records = run_droid_parts(args, sequence)
        else:
            sequence_root = args.video_root / sequence
            images = sorted((sequence_root / "extracted_images").glob("*.jpg"))
            starts, ends, _ = split_list_by_interval(images, interval=args.interval)
            records = []
            for part_index, (start, end) in enumerate(zip(starts, ends)):
                if args.part and part_index not in set(args.part):
                    continue
                records.append((
                    start,
                    end,
                    sequence_root / "preprocess" / "SLAM_WiLoR" /
                    f"wilor_droid_raw_{start}_{end}.npz",
                ))
        if args.stage in ("all", "scale"):
            add_metric_scale(args, sequence, records)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
