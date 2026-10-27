"""Prepare and run the official HMP pipeline on the HOT3D evaluation split.

The adapter exposes only the same right-hand boxes and valid-frame mask used
by the HaWoR/WiLoR runs.  HMP still uses its official PyMAF-X initialization,
MMPose 2D keypoints, motion prior, and optimization configuration.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HMP_ROOT = ROOT / "thirdparty" / "HMP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HMP on the HOT3D baseline split")
    parser.add_argument("--video_root", type=Path,
                        default=ROOT / "datasets" / "hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--output_root", type=Path,
                        default=ROOT / "eval_log_hmp_slam")
    parser.add_argument("--video", action="append",
                        help="Sequence name; repeat to select multiple sequences")
    parser.add_argument("--config", default="in_the_wild_sample_config.yaml")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable from the HMP environment")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPUs; sequences run in parallel")
    parser.add_argument("--prepare_only", action="store_true")
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
        sequences = [sequence for sequence in sequences if sequence in selected]
    return sequences


def require_assets() -> None:
    required = {
        "HMP motion-prior checkpoint": HMP_ROOT / "outputs" / "generative" / "results" / "model",
        "HMP normalization mean": HMP_ROOT / "data" / "amass" / "generative" / "mean-neutral-128-30fps.pt",
        "HMP normalization std": HMP_ROOT / "data" / "amass" / "generative" / "std-neutral-128-30fps.pt",
        "PyMAF-X checkpoint": HMP_ROOT / "external" / "PyMAF-X" / "data" / "pretrained_model" / "PyMAF-X_model_checkpoint_v1.1.pt",
        "MMPose detector": HMP_ROOT / "data" / "mmpose_models" / "cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth",
        "MMPose pose estimator": HMP_ROOT / "data" / "mmpose_models" / "res50_onehand10k_256x256-e67998f6_20200813.pth",
        "MANO model": HMP_ROOT / "data" / "body_models" / "mano" / "MANO_RIGHT.pkl",
    }
    missing = [f"- {label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        message = "Required official HMP assets are missing:\n" + "\n".join(missing)
        message += ("\nDownload model.zip, data.zip, and body_models.zip from the HMP site "
                    "into thirdparty/HMP/downloads, then install them with "
                    "scripts/scripts_eval/install_hmp_assets.py.")
        raise FileNotFoundError(message)


def prepare_sequence(args: argparse.Namespace, sequence: str) -> Path:
    source_root = args.video_root / sequence
    image_root = source_root / "extracted_images"
    image_paths = sorted(image_root.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No HOT3D frames found for {sequence}")

    tracks = np.load(
        source_root / "preprocess" / "preprocess_gtdet_tracks.npy",
        allow_pickle=True,
    ).item()
    annotation = joblib.load(source_root / "anno.pth")
    mano_valid = torch.any(annotation["rot_r"] != 0, dim=-1).cpu().numpy()
    right_track = tracks[1]
    detected = np.asarray([item["det"] for item in right_track], dtype=bool)
    valid = detected & mano_valid
    frame_ids = np.flatnonzero(valid)
    boxes = np.concatenate(
        [right_track[frame_id]["det_box"][:, :4] for frame_id in frame_ids], axis=0)
    # HMP/PyMAF-X expects x1,y1,x2,y2 plus a detection confidence.
    boxes = np.concatenate(
        [boxes.astype(np.float32), np.ones((len(boxes), 1), dtype=np.float32)], axis=1)

    input_root = args.output_root / "_inputs" / "HOT3D" / sequence
    input_root.mkdir(parents=True, exist_ok=True)
    rgb_link = input_root / "rgb"
    if rgb_link.is_symlink() and rgb_link.resolve() != image_root.resolve():
        rgb_link.unlink()
    if not rgb_link.exists():
        rgb_link.symlink_to(image_root.resolve(), target_is_directory=True)
    np.savez_compressed(
        input_root / "hot3d_input.npz",
        frame_id=frame_ids.astype(np.int32),
        bbox=boxes,
        handedness=np.asarray("right"),
        total_frames=np.asarray(len(image_paths), dtype=np.int32),
    )
    metadata = {
        "sequence": sequence,
        "total_frames": len(image_paths),
        "valid_right_hand_frames": int(valid.sum()),
        "bbox_source": "HOT3D GT MANO projection (preprocess_gtdet_tracks.npy)",
        "joint_ground_truth_exposed_to_hmp": False,
    }
    with (input_root / "input_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return rgb_link


def run_sequence(args: argparse.Namespace, sequence: str, rgb_path: Path,
                 gpu: str) -> None:
    output_path = args.output_root / sequence / "hmp_optim"
    result_path = output_path / "recon_000_30fps.npz"
    if result_path.exists() and not args.overwrite:
        print(f"[skip] {sequence}: {result_path} exists")
        return
    command = [
        args.python,
        "src/fitting_app.py",
        "--vid-path", str(rgb_path),
        "--config", args.config,
        "--init-method", "pymafx",
        "--dataname", "HOT3D",
        "--save-path", str(output_path),
        "--skip-visualization",
        "--no-plot-loss",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONNOUSERSITE"] = "1"
    print(f"[run] {sequence} on GPU {gpu}")
    subprocess.run(command, cwd=HMP_ROOT, env=environment, check=True)


def run_group(args: argparse.Namespace, gpu: str,
              prepared: list[tuple[str, Path]]) -> None:
    for sequence, rgb_path in prepared:
        run_sequence(args, sequence, rgb_path, gpu)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sequences = load_sequences(args)
    prepared = [(sequence, prepare_sequence(args, sequence)) for sequence in sequences]
    print(f"[prepared] {len(prepared)} HOT3D sequences")
    if args.prepare_only:
        return
    require_assets()
    gpus = [item.strip() for item in (args.gpus or args.gpu).split(",") if item.strip()]
    groups = [prepared[index::len(gpus)] for index in range(len(gpus))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_group, args, gpu, group)
                   for gpu, group in zip(gpus, groups) if group]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
