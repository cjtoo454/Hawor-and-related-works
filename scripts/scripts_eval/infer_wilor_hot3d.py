"""Run frame-wise WiLoR inference on the HOT3D evaluation split.

The script uses the existing HOT3D evaluation boxes so that WiLoR-SLAM and
HaWoR are scored on exactly the same right-hand frames.  This is only an input
protocol choice: hand reconstruction itself is the official, frame-wise WiLoR
model with no temporal smoothing or HaWoR components.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from skimage.filters import gaussian
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
WILOR_ROOT = ROOT / "thirdparty" / "WiLoR"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WILOR_ROOT))

from wilor.datasets.utils import (  # noqa: E402
    convert_cvimg_to_tensor,
    expand_to_aspect_ratio,
    generate_image_patch_cv2,
)
from wilor.models import load_wilor  # noqa: E402
from wilor.utils import recursive_to  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WiLoR inference on HOT3D")
    parser.add_argument("--video_root", type=Path,
                        default=ROOT / "datasets" / "hot3d_valset_export")
    parser.add_argument("--set_file", default="val.json")
    parser.add_argument("--output_root", type=Path,
                        default=ROOT / "eval_log_wilor_slam")
    parser.add_argument("--video", action="append",
                        help="Sequence name; repeat to select multiple sequences")
    parser.add_argument("--checkpoint", type=Path,
                        default=WILOR_ROOT / "pretrained_models" / "wilor_final.ckpt")
    parser.add_argument("--config", type=Path,
                        default=WILOR_ROOT / "pretrained_models" / "model_config.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_frame_npz", action="store_true",
                        help="Only save the compact sequence prediction archive")
    return parser.parse_args()


def cam_crop_to_full(cam_bbox: torch.Tensor, box_center: torch.Tensor,
                     box_size: torch.Tensor, img_size: torch.Tensor,
                     focal_length: torch.Tensor) -> torch.Tensor:
    """Official WiLoR weak-perspective crop to full-image conversion."""
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy = box_center[:, 0], box_center[:, 1]
    scaled_bbox = box_size * cam_bbox[:, 0] + 1e-9
    tz = 2.0 * focal_length / scaled_bbox
    tx = 2.0 * (cx - img_w / 2.0) / scaled_bbox + cam_bbox[:, 1]
    ty = 2.0 * (cy - img_h / 2.0) / scaled_bbox + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


class HOT3DWiLoRDataset(Dataset):
    """Sequence version of WiLoR's official ViTDetDataset."""

    def __init__(self, cfg, image_paths: list[Path], frame_ids: np.ndarray,
                 boxes: np.ndarray, rescale_factor: float) -> None:
        self.cfg = cfg
        self.image_paths = image_paths
        self.frame_ids = frame_ids.astype(np.int64)
        self.boxes = boxes.astype(np.float32)
        self.img_size = int(cfg.MODEL.IMAGE_SIZE)
        self.mean = 255.0 * np.asarray(cfg.MODEL.IMAGE_MEAN)
        self.std = 255.0 * np.asarray(cfg.MODEL.IMAGE_STD)
        self.center = (self.boxes[:, 2:4] + self.boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (self.boxes[:, 2:4] - self.boxes[:, 0:2]) / 200.0

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int | float]:
        frame_id = int(self.frame_ids[index])
        image = cv2.imread(str(self.image_paths[frame_id]))
        if image is None:
            raise OSError(f"Cannot read {self.image_paths[frame_id]}")

        center = self.center[index].copy()
        bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        bbox_size = expand_to_aspect_ratio(
            self.scale[index] * 200.0, target_aspect_ratio=bbox_shape).max()

        downsampling_factor = bbox_size / self.img_size / 2.0
        if downsampling_factor > 1.1:
            image = gaussian(
                image,
                sigma=(downsampling_factor - 1.0) / 2.0,
                channel_axis=2,
                preserve_range=True,
            )

        patch, _ = generate_image_patch_cv2(
            image,
            center[0],
            center[1],
            bbox_size,
            bbox_size,
            self.img_size,
            self.img_size,
            False,  # right hand: no horizontal flip
            1.0,
            0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        patch = convert_cvimg_to_tensor(patch[:, :, ::-1])
        for channel in range(3):
            patch[channel] = (patch[channel] - self.mean[channel]) / self.std[channel]

        return {
            "img": patch,
            "frame_id": frame_id,
            "box_center": center,
            "box_size": np.float32(bbox_size),
            "img_size": np.asarray([image.shape[1], image.shape[0]], dtype=np.float32),
            "right": np.float32(1.0),
        }


def empty_frame_payload(frame_id: int, image_size: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "frame_id": np.asarray(frame_id, dtype=np.int32),
        "valid": np.asarray(False),
        "handedness": np.empty((0,), dtype=np.int8),
        "bbox": np.empty((0, 4), dtype=np.float32),
        "vertices_3d": np.empty((0, 778, 3), dtype=np.float32),
        "joints_3d": np.empty((0, 21, 3), dtype=np.float32),
        "global_orient": np.empty((0, 1, 3, 3), dtype=np.float32),
        "hand_pose": np.empty((0, 15, 3, 3), dtype=np.float32),
        "betas": np.empty((0, 10), dtype=np.float32),
        "pred_cam": np.empty((0, 3), dtype=np.float32),
        "camera_translation": np.empty((0, 3), dtype=np.float32),
        "focal_length": np.empty((0,), dtype=np.float32),
        "image_size": image_size.astype(np.int32),
    }


def load_sequences(args: argparse.Namespace) -> list[str]:
    with (args.video_root / args.set_file).open("r", encoding="utf-8") as handle:
        sequences = json.load(handle)
    if args.video:
        unknown = sorted(set(args.video) - set(sequences))
        if unknown:
            raise ValueError(f"Sequences not in {args.set_file}: {unknown}")
        sequences = [name for name in sequences if name in set(args.video)]
    return sequences


def infer_sequence(model, cfg, args: argparse.Namespace, sequence: str,
                   device: torch.device) -> None:
    sequence_root = args.video_root / sequence
    output_dir = args.output_root / sequence / "wilor_frames"
    compact_path = args.output_root / sequence / "wilor_predictions.npz"
    if compact_path.exists() and not args.overwrite:
        print(f"[skip] {sequence}: {compact_path} exists")
        return

    image_paths = sorted((sequence_root / "extracted_images").glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No extracted HOT3D frames for {sequence}")

    tracks = np.load(
        sequence_root / "preprocess" / "preprocess_gtdet_tracks.npy",
        allow_pickle=True,
    ).item()
    annotation = joblib.load(sequence_root / "anno.pth")
    mano_valid = torch.any(annotation["rot_r"] != 0, dim=-1).cpu().numpy()
    right_track = tracks[1]
    detected = np.asarray([item["det"] for item in right_track], dtype=bool)
    valid = detected & mano_valid
    frame_ids = np.flatnonzero(valid)
    boxes = np.concatenate([right_track[i]["det_box"][:, :4] for i in frame_ids], axis=0)

    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise OSError(f"Cannot read {image_paths[0]}")
    image_size_wh = np.asarray([first_image.shape[1], first_image.shape[0]], dtype=np.int32)
    total_frames = len(image_paths)
    # WiLoR's demo scales its virtual 5000 px crop focal length to the full
    # image.  That is appropriate when the real intrinsics are unknown, but it
    # leaves translation depth proportional to an arbitrary virtual focal.
    # HOT3D provides calibrated intrinsics and the HaWoR world-space evaluator
    # expects metric camera translation, so use the sequence's real focal here.
    with (sequence_root / "focal.txt").open("r", encoding="utf-8") as handle:
        calibrated_focal_length = float(handle.read())

    arrays = {
        "valid": valid.copy(),
        "bbox": np.full((total_frames, 4), np.nan, dtype=np.float32),
        "global_orient": np.full((total_frames, 1, 3, 3), np.nan, dtype=np.float32),
        "hand_pose": np.full((total_frames, 15, 3, 3), np.nan, dtype=np.float32),
        "betas": np.full((total_frames, 10), np.nan, dtype=np.float32),
        "pred_cam": np.full((total_frames, 3), np.nan, dtype=np.float32),
        "camera_translation": np.full((total_frames, 3), np.nan, dtype=np.float32),
        "focal_length": np.full((total_frames,), np.nan, dtype=np.float32),
    }
    arrays["bbox"][frame_ids] = boxes

    dataset = HOT3DWiLoRDataset(
        cfg, image_paths, frame_ids, boxes, args.rescale_factor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    if not args.no_frame_npz:
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in np.flatnonzero(~valid):
            np.savez_compressed(
                output_dir / f"{frame_id:04d}.npz",
                **empty_frame_payload(int(frame_id), image_size_wh),
            )

    progress = tqdm(total=len(dataset), desc=f"WiLoR {sequence}", unit="frame")
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.inference_mode():
            output = model(batch)

        pred_cam = output["pred_cam"].clone()
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        batch_image_size = batch["img_size"].float()
        focal_length = torch.full(
            (len(batch["frame_id"]),),
            calibrated_focal_length,
            dtype=pred_cam.dtype,
            device=pred_cam.device,
        )
        camera_translation = cam_crop_to_full(
            pred_cam, box_center, box_size, batch_image_size, focal_length)

        batch_frame_ids = batch["frame_id"].detach().cpu().numpy().astype(np.int64)
        vertices = output["pred_vertices"].detach().float().cpu().numpy()
        joints = output["pred_keypoints_3d"].detach().float().cpu().numpy()
        global_orient = output["pred_mano_params"]["global_orient"].detach().float().cpu().numpy()
        hand_pose = output["pred_mano_params"]["hand_pose"].detach().float().cpu().numpy()
        betas = output["pred_mano_params"]["betas"].detach().float().cpu().numpy()
        pred_cam_np = pred_cam.detach().float().cpu().numpy()
        camera_np = camera_translation.detach().float().cpu().numpy()
        focal_np = focal_length.detach().float().cpu().numpy()

        arrays["global_orient"][batch_frame_ids] = global_orient
        arrays["hand_pose"][batch_frame_ids] = hand_pose
        arrays["betas"][batch_frame_ids] = betas
        arrays["pred_cam"][batch_frame_ids] = pred_cam_np
        arrays["camera_translation"][batch_frame_ids] = camera_np
        arrays["focal_length"][batch_frame_ids] = focal_np

        if not args.no_frame_npz:
            for offset, frame_id in enumerate(batch_frame_ids):
                np.savez_compressed(
                    output_dir / f"{frame_id:04d}.npz",
                    frame_id=np.asarray(frame_id, dtype=np.int32),
                    valid=np.asarray(True),
                    handedness=np.asarray([1], dtype=np.int8),
                    bbox=arrays["bbox"][frame_id][None],
                    vertices_3d=vertices[offset][None],
                    joints_3d=joints[offset][None],
                    global_orient=global_orient[offset][None],
                    hand_pose=hand_pose[offset][None],
                    betas=betas[offset][None],
                    pred_cam=pred_cam_np[offset][None],
                    camera_translation=camera_np[offset][None],
                    focal_length=focal_np[offset:offset + 1],
                    image_size=image_size_wh,
                )
        progress.update(len(batch_frame_ids))
    progress.close()

    sequence_output = args.output_root / sequence
    sequence_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        compact_path,
        frame_id=np.arange(total_frames, dtype=np.int32),
        handedness=np.ones((total_frames,), dtype=np.int8),
        image_size=image_size_wh,
        **arrays,
    )
    metadata = {
        "sequence": sequence,
        "total_frames": total_frames,
        "valid_right_hand_frames": int(valid.sum()),
        "bbox_source": "HOT3D GT MANO projection (preprocess_gtdet_tracks.npy)",
        "rescale_factor": args.rescale_factor,
        "focal_length_source": "HOT3D calibrated focal.txt",
        "focal_length_px": calibrated_focal_length,
        "checkpoint": str(args.checkpoint),
    }
    with (sequence_output / "wilor_inference.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[done] {sequence}: {int(valid.sum())}/{total_frames} valid right-hand frames")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, cfg = load_wilor(
        checkpoint_path=str(args.checkpoint),
        cfg_path=str(args.config),
        init_renderer=False,
    )
    model = model.to(device).eval()
    for sequence in load_sequences(args):
        infer_sequence(model, cfg, args, sequence, device)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
