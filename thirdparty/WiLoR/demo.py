"""Headless WiLoR inference demo.

This entry point intentionally does not import pyrender, trimesh, PyOpenGL, or
create an EGL context. Predictions are saved as compressed NumPy archives and
an OpenCV preview is written for each successfully processed image.
"""

from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset


SCRIPT_DIR = Path(__file__).resolve().parent


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
    """Convert weak-perspective crop cameras to full-image translation."""
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, bbox_extent = box_center[:, 0], box_center[:, 1], box_size
    half_w, half_h = img_w / 2.0, img_h / 2.0
    scaled_bbox = bbox_extent * cam_bbox[:, 0] + 1e-9
    tz = 2.0 * focal_length / scaled_bbox
    tx = (2.0 * (cx - half_w) / scaled_bbox) + cam_bbox[:, 1]
    ty = (2.0 * (cy - half_h) / scaled_bbox) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def project_full_img(points, camera_translation, focal_length, image_size):
    """Project 3D points into full-image pixel coordinates using NumPy only."""
    camera_points = points + camera_translation[None, :]
    xy = camera_points[:, :2] / np.clip(camera_points[:, 2:3], 1e-9, None)
    center = np.asarray(image_size, dtype=np.float32) / 2.0
    return xy * float(focal_length) + center[None, :]


def save_obj(path, vertices, faces, camera_translation, is_right):
    """Write the optional mesh without importing a rendering library."""
    vertices = vertices + camera_translation[None, :]
    vertices[:, 1:] *= -1.0  # Match the original renderer's 180-degree X flip.
    if not is_right:
        faces = faces[:, [0, 2, 1]]
    with path.open('w', encoding='utf-8') as obj_file:
        for vertex in vertices:
            obj_file.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            obj_file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def save_preview(path, image, boxes, handedness, vertices_2d, joints_2d):
    """Draw lightweight 2D projections without requiring an OpenGL context."""
    preview = image.copy()
    height, width = preview.shape[:2]
    for hand_index, (box, is_right, vertices, joints) in enumerate(
            zip(boxes, handedness, vertices_2d, joints_2d)):
        color = (40, 190, 255) if is_right else (255, 120, 40)
        x1, y1, x2, y2 = np.rint(box).astype(np.int32)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f'{hand_index}: {"right" if is_right else "left"}'
        cv2.putText(preview, label, (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # A sparse vertex cloud makes the reconstructed hand shape visible.
        for x, y in vertices[::4]:
            if np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height:
                cv2.circle(preview, (int(round(x)), int(round(y))), 1, color, -1,
                           cv2.LINE_AA)
        for x, y in joints:
            if np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height:
                cv2.circle(preview, (int(round(x)), int(round(y))), 3,
                           (30, 30, 255), -1, cv2.LINE_AA)

    if not cv2.imwrite(str(path), preview):
        raise OSError(f'Failed to write preview image: {path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Headless WiLoR inference demo')
    parser.add_argument('--img_folder', type=Path, default=SCRIPT_DIR / 'demo_img',
                        help='Folder containing input images')
    parser.add_argument('--out_folder', type=Path, default=SCRIPT_DIR / 'out_demo',
        help='Folder for .npz predictions, preview images, and summary.json')
    parser.add_argument('--checkpoint', type=Path,
                        default=SCRIPT_DIR / 'pretrained_models' / 'wilor_final.ckpt')
    parser.add_argument('--config', type=Path,
                        default=SCRIPT_DIR / 'pretrained_models' / 'model_config.yaml')
    parser.add_argument('--detector', type=Path,
                        default=SCRIPT_DIR / 'pretrained_models' / 'detector.pt')
    parser.add_argument('--save_mesh', action='store_true',
                        help='Also save OBJ meshes without rendering them')
    parser.add_argument('--rescale_factor', type=float, default=2.0,
                        help='Padding factor applied to each detected hand box')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'],
                        help='Input filename patterns')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--conf', type=float, default=0.3,
                        help='YOLO hand detection confidence threshold')
    parser.add_argument('--fast', action='store_true',
                        help='Use FP16 and backbone block skipping (CUDA only)')
    parser.add_argument('--device', default=None,
                        help='Torch device, for example cuda, cuda:0, or cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_fast = args.fast and device.type == 'cuda'
    if args.fast and not use_fast:
        print('Warning: --fast requires CUDA; using FP32 inference instead.')

    model, model_cfg = load_wilor(
        checkpoint_path=str(args.checkpoint),
        cfg_path=str(args.config),
        init_renderer=False,
    )
    if use_fast:
        torch.set_float32_matmul_precision('high')
        model = model.half()
        if hasattr(torch, 'compile'):
            model.backbone = torch.compile(model.backbone)
        model.backbone.skip_blocks = True

    detector = YOLO(str(args.detector))
    model = model.to(device).eval()
    detector = detector.to(device)

    args.out_folder.mkdir(parents=True, exist_ok=True)
    mesh_folder = args.out_folder / 'meshes'
    if args.save_mesh:
        mesh_folder.mkdir(parents=True, exist_ok=True)

    image_paths = sorted({path for pattern in args.file_type
                          for path in args.img_folder.glob(pattern)})
    if not image_paths:
        raise FileNotFoundError(f'No input images found in {args.img_folder}')

    summary = []
    faces = np.asarray(model.mano.faces)
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Skipping unreadable image: {image_path}')
            summary.append({'image': image_path.name, 'status': 'unreadable', 'hands': 0})
            continue

        detections = detector(image, conf=args.conf, verbose=False)[0]
        boxes, handedness = [], []
        for detection in detections:
            box_data = detection.boxes.data.detach().cpu().reshape(-1).numpy()
            boxes.append(box_data[:4])
            handedness.append(float(detection.boxes.cls.detach().cpu().item()))

        if not boxes:
            print(f'{image_path.name}: no hands detected')
            summary.append({'image': image_path.name, 'status': 'no_detection', 'hands': 0})
            continue

        boxes = np.asarray(boxes, dtype=np.float32)
        handedness = np.asarray(handedness, dtype=np.float32)
        dataset = ViTDetDataset(
            model_cfg,
            image,
            boxes,
            handedness,
            rescale_factor=args.rescale_factor,
            fp16=use_fast,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        vertices_all, vertices_2d_all = [], []
        joints_all, joints_2d_all = [], []
        root_orient_all, hand_pose_all, betas_all = [], [], []
        pred_cam_all, camera_all, right_all, focal_all = [], [], [], []
        hand_index = 0
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.inference_mode():
                output = model(batch)

            multiplier = 2 * batch['right'] - 1
            pred_cam = output['pred_cam'].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch['box_center'].float()
            box_size = batch['box_size'].float()
            image_size = batch['img_size'].float()
            focal_length = (model_cfg.EXTRA.FOCAL_LENGTH /
                            model_cfg.MODEL.IMAGE_SIZE * image_size.max())
            camera_full = cam_crop_to_full(
                pred_cam, box_center, box_size, image_size, focal_length)

            for index in range(batch['img'].shape[0]):
                vertices = output['pred_vertices'][index].detach().float().cpu().numpy()
                joints = output['pred_keypoints_3d'][index].detach().float().cpu().numpy()
                is_right = bool(batch['right'][index].detach().cpu().item())
                side_multiplier = 1.0 if is_right else -1.0
                vertices[:, 0] *= side_multiplier
                joints[:, 0] *= side_multiplier
                camera_translation = camera_full[index].detach().float().cpu().numpy()
                image_wh = image_size[index].detach().float().cpu().numpy()
                focal = float(focal_length.detach().float().cpu().item())
                vertices_2d = project_full_img(
                    vertices, camera_translation, focal, image_wh)
                joints_2d = project_full_img(
                    joints, camera_translation, focal, image_wh)

                vertices_all.append(vertices)
                vertices_2d_all.append(vertices_2d)
                joints_all.append(joints)
                joints_2d_all.append(joints_2d)
                root_orient_all.append(
                    output['pred_mano_params']['global_orient'][index]
                    .detach().float().cpu().numpy())
                hand_pose_all.append(
                    output['pred_mano_params']['hand_pose'][index]
                    .detach().float().cpu().numpy())
                betas_all.append(
                    output['pred_mano_params']['betas'][index]
                    .detach().float().cpu().numpy())
                pred_cam_all.append(
                    pred_cam[index].detach().float().cpu().numpy())
                camera_all.append(camera_translation)
                right_all.append(is_right)
                focal_all.append(focal)

                if args.save_mesh:
                    save_obj(
                        mesh_folder / f'{image_path.stem}_hand{hand_index:02d}.obj',
                        vertices.copy(),
                        faces.copy(),
                        camera_translation,
                        is_right,
                    )
                hand_index += 1

        output_path = args.out_folder / f'{image_path.stem}.npz'
        np.savez_compressed(
            output_path,
            handedness=np.asarray(right_all, dtype=np.int8),
            bbox=boxes.astype(np.float32, copy=False),
            vertices_3d=np.stack(vertices_all).astype(np.float32, copy=False),
            joints_3d=np.stack(joints_all),
            global_orient=np.stack(root_orient_all),
            hand_pose=np.stack(hand_pose_all),
            betas=np.stack(betas_all),
            pred_cam=np.stack(pred_cam_all),
            camera_translation=np.stack(camera_all),
            focal_length=np.asarray(focal_all, dtype=np.float32),
            image_size=np.asarray([image.shape[1], image.shape[0]], dtype=np.int32),
        )
        preview_path = args.out_folder / f'{image_path.stem}_wilor.jpg'
        save_preview(
            preview_path,
            image,
            boxes,
            right_all,
            vertices_2d_all,
            joints_2d_all,
        )
        print(f'{image_path.name}: saved {len(right_all)} hand(s) to {output_path}')
        summary.append({
            'image': image_path.name,
            'status': 'ok',
            'hands': len(right_all),
            'prediction': output_path.name,
            'preview': preview_path.name,
        })

    with (args.out_folder / 'summary.json').open('w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f'Finished {len(image_paths)} image(s); summary: {args.out_folder / "summary.json"}')


if __name__ == '__main__':
    main()
