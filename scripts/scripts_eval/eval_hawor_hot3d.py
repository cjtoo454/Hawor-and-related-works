from collections import defaultdict
import copy
from PIL import Image

import json
import sys
import os

import joblib

sys.path.insert(0, os.path.dirname(__file__) + '/../..')


import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from glob import glob
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from lib.pipeline.tools import parse_chunks
from lib.models.hawor import HAWOR
from lib.eval_utils.custom_utils import cam2world_convert, load_gt_cam, split_list_by_interval
from hawor.utils.process import run_mano
from lib.eval_utils.eval_utils import batch_compute_similarity_transform_torch, compute_error_accel, compute_jpe, compute_rte, first_align_joints, global_align_joints
from lib.eval_utils.custom_utils import load_slam_cam, algin_cam_traj_wo_scale
from lib.datasets.hot3d_dataset import load_test_set
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.vis.renderer import Renderer
from pycocotools import mask as masktool


def load_hawor(checkpoint_path):
    from pathlib import Path
    from hawor.configs import get_config
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


def umeyama_alignment(x, y):
    """
    Umeyama algorithm for aligning two sets of points
    :param x: Nx3 tensor of points
    :param y: Nx3 tensor of points
    :return: s, R, t such that s * R * x + t = y
    """
    assert x.shape == y.shape

    m = x.shape[1]
    mean_x = torch.mean(x, dim=0)
    mean_y = torch.mean(y, dim=0)

    sigma_x = torch.mean(torch.sum((x - mean_x) ** 2, dim=1))
    cov_xy = (y - mean_y).T @ (x - mean_x) / x.shape[0]

    U, D, V = torch.svd(cov_xy)
    S = torch.eye(m)
    if torch.det(cov_xy) < 0:
        S[m - 1, m - 1] = -1

    R = U @ S @ V.T
    c = 1 / sigma_x * torch.sum(D * S.diag())
    t = mean_y - c * R @ mean_x

    return c, R, t


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",  type=str, default='weights/hawor/checkpoints/hawor.ckpt')
parser.add_argument("--video_root", type=str, default='datasets/hot3d_valset_export')
parser.add_argument("--set_file", type=str, default='val.json')
parser.add_argument("--inference_stage", action="store_true")
parser.add_argument("--gen_hand_mask", action="store_true")
parser.add_argument("--eval_stage", action="store_true")
parser.add_argument("--skip_existing_eval", action="store_true",
                    help="Reuse complete per-video evaluation artifacts when present")
parser.add_argument("--subset", action="store_true")
parser.add_argument('--eval_log_dir', type=str, default="eval_log")
args = parser.parse_args()

if not args.eval_stage and not (args.gen_hand_mask and not args.inference_stage):
    model, model_cfg = load_hawor(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

video_root = args.video_root
testset = load_test_set(os.path.join(video_root, args.set_file))
if args.subset:
    testset = testset[:1]
if not os.path.exists(args.eval_log_dir):
    os.makedirs(args.eval_log_dir)
eval_log = open(os.path.join(args.eval_log_dir, "eval_log.txt"), "w")
eval_log.write(str(testset))
eval_log.write("\n")
print(testset)
m2mm = 1e3
accumulator = defaultdict(list)
for video in testset:
    result_dir = os.path.join(args.eval_log_dir, video)
    summary_path = os.path.join(result_dir, 'evaluation_summary.json')
    details_path = os.path.join(result_dir, 'evaluation_details.npz')
    if args.eval_stage and args.skip_existing_eval \
            and os.path.exists(summary_path) and os.path.exists(details_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        details = np.load(details_path)
        accumulator['w_mpjpe'].append(details['w_mpjpe_mm'])
        accumulator['wa_mpjpe'].append(details['wa_mpjpe_mm'])
        accumulator['pa_mpjpe'].append(details['pa_mpjpe_mm'])
        accumulator['accel'].append(details['accel_m_s2'])
        accumulator['RTE'].append(details['rte_percent'])
        video_log = f"{video} {summary['w_mpjpe_mm']} {summary['wa_mpjpe_mm']}"
        print(f"Reuse existing evaluation for {video}")
        eval_log.write(video_log + "\n")
        continue

    video_path = os.path.join(video_root, video, video+'.mp4')
    seq_folder = os.path.join(video_root, video, 'preprocess')
    img_folder = f"{video_root}/{video}/extracted_images"
    hps_folder = f'{seq_folder}/hps'
    os.makedirs(hps_folder, exist_ok=True)

    # Previous steps
    imgfiles = np.array(sorted(glob(f'{img_folder}/*.jpg')))
    tracks = np.load(f'{seq_folder}/preprocess_gtdet_tracks.npy', allow_pickle=True).item()
    with open(os.path.join(video_root, video, 'focal.txt'), 'r') as file:
        focal_length = file.read()
        img_focal = float(focal_length)
    img = cv2.imread(imgfiles[0])
    img_center = [img.shape[0] / 2, img.shape[1] / 2]        

    # load SLAM camera  
    if not args.inference_stage and not args.gen_hand_mask:
        R_c2w_sla_all = []
        t_c2w_sla_all = []
        R_w2c_sla_all = []
        t_w2c_sla_all = []
        R_c2w_gt_all = []
        t_c2w_gt_all = []
        R_w2c_gt_all = []
        t_w2c_gt_all = []
        start_idxs, end_idxs, imgfiles_list = split_list_by_interval(imgfiles)
        for part in range(len(imgfiles_list)):
            fpath = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idxs[part]}_{end_idxs[part]}.npz")
            R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla = load_slam_cam(fpath)

            # load GT camera
            R_w2c_gt, t_w2c_gt, R_c2w_gt, t_c2w_gt = load_gt_cam(video_root, video, start_idxs[part], end_idxs[part])

            # align
            R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla = algin_cam_traj_wo_scale(R_c2w_sla, t_c2w_sla, R_c2w_gt, t_c2w_gt)

            R_c2w_sla_all.append(R_c2w_sla)
            t_c2w_sla_all.append(t_c2w_sla)
            R_w2c_sla_all.append(R_w2c_sla)
            t_w2c_sla_all.append(t_w2c_sla)
            R_c2w_gt_all.append(R_c2w_gt)
            t_c2w_gt_all.append(t_c2w_gt)
            R_w2c_gt_all.append(R_w2c_gt)
            t_w2c_gt_all.append(t_w2c_gt)
        R_c2w_sla_all = torch.cat(R_c2w_sla_all, dim=0)
        t_c2w_sla_all = torch.cat(t_c2w_sla_all, dim=0)
        R_w2c_sla_all = torch.cat(R_w2c_sla_all, dim=0)
        t_w2c_sla_all = torch.cat(t_w2c_sla_all, dim=0)
        R_c2w_gt_all = torch.cat(R_c2w_gt_all, dim=0)
        t_c2w_gt_all = torch.cat(t_c2w_gt_all, dim=0)
        R_w2c_gt_all = torch.cat(R_w2c_gt_all, dim=0)
        t_w2c_gt_all = torch.cat(t_w2c_gt_all, dim=0)

    # load gt annotations
    gt_pth = os.path.join(os.path.dirname(video_path), 'anno.pth')
    datasets = joblib.load(gt_pth)
    world_rot = torch.stack([datasets['rot_l'], datasets['rot_r']])
    mano_valid = torch.any(world_rot != 0, dim=-1).numpy()
    world_trans = torch.stack([datasets['trans_l'], datasets['trans_r']])
    world_hand_pose = torch.stack([datasets['pose_l'], datasets['pose_r']])
    world_betas = torch.stack([datasets['betas_l'], datasets['betas_r']])

    # --- Tracks: sort by length  ---
    tid = np.array([tr for tr in tracks])
    tlen = np.array([len(tracks[tr]) for tr in tracks])
    sort = np.argsort(tlen)[::-1]
    tid = tid[sort]

    H, W = img.shape[:2]
    model_masks = np.zeros((len(imgfiles), H, W), dtype=np.uint8)

    bin_size = 128
    max_faces_per_bin = 20000
    renderer = Renderer(img.shape[1], img.shape[0], img_focal, 'cuda', 
                    bin_size=bin_size, max_faces_per_bin=max_faces_per_bin)
    
    # get faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:,[0,2,1]]
    faces_right_cuda = torch.from_numpy(faces_right).cuda()
    faces_left_cuda = torch.from_numpy(faces_left).cuda()
    cam_R = torch.eye(3).unsqueeze(0).cuda()
    cam_T = torch.zeros(1, 3).cuda()
    mask_cameras, mask_lights = renderer.create_camera_from_cv(cam_R, cam_T)

    print(f'Running on {video} ...')
    video_w_mpjpe = []
    video_wa_mpjpe = []
    video_pa_mpjpe = []
    video_accel = []
    video_rte = []
    eval_frame_ids = []
    accel_frame_ids = []
    pred_world_joints = np.full((len(imgfiles), 21, 3), np.nan, dtype=np.float32)
    gt_world_joints = np.full((len(imgfiles), 21, 3), np.nan, dtype=np.float32)

    # --- Run VIMO on each track ---
    for k, idx in enumerate(tid):
        if args.eval_stage and idx == 0:
            continue # we evalute right hand
        if idx == 0:
            do_flip = True
        else:
            do_flip = False
        trk = tracks[idx]
        valid = np.array([t['det'] for t in trk])

        trk_mano_valid = mano_valid[idx]
        valid = valid & trk_mano_valid

        boxes = np.concatenate([t['det_box'] for t in trk])[valid]
        frame = np.array([t['frame'] for t in trk])[valid]
        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=16)

        if len(frame_chunks) == 0:
            continue

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            print(f"from frame {frame_ck[0]} to {frame_ck[-1]}")
            img_ck = imgfiles[frame_ck]
            pred_path = os.path.join(args.eval_log_dir, video, str(idx), f"{frame_ck[0]}.json")
            if args.inference_stage and not args.gen_hand_mask and os.path.exists(pred_path):
                print(f"skip existing prediction {pred_path}")
                continue
            loaded_prediction = False
            if not args.eval_stage:
                if args.gen_hand_mask and not args.inference_stage:
                    pred_path = os.path.join(args.eval_log_dir, video, str(idx), f"{frame_ck[0]}.json")
                    with open(pred_path, "r") as f:
                        pred_dict = json.load(f)
                    data_out = {k: torch.tensor(v) for k, v in pred_dict.items()}
                    loaded_prediction = True
                else:
                    results = model.inference(img_ck, boxes_ck, img_focal=img_focal, img_center=img_center, do_flip=do_flip)

                    data_out = {
                        "init_root_orient": results["pred_rotmat"][None, :, 0], # (B, T, 3, 3)
                        "init_hand_pose": results["pred_rotmat"][None, :, 1:], # (B, T, 15, 3, 3)
                        "init_trans": results["pred_trans"][None, :, 0],  # (B, T, 3)
                        "init_betas": results["pred_shape"][None, :]  # (B, T, 10)
                    }

                # flip left hand
                if not loaded_prediction:
                    init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
                    init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
                    if do_flip:
                        init_root[..., 1] *= -1
                        init_root[..., 2] *= -1
                        init_hand_pose[..., 1] *= -1
                        init_hand_pose[..., 2] *= -1
                    data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
                    data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand_pose)

            if args.inference_stage:
                pred_dict={
                    k:v.tolist() for k, v in data_out.items()
                }
                if not os.path.exists(os.path.join(args.eval_log_dir, video, str(idx))):
                    os.makedirs(os.path.join(args.eval_log_dir, video, str(idx)))
                with open(pred_path, "w") as f:
                    json.dump(pred_dict, f, indent=1)
                if not args.gen_hand_mask:
                    continue

            if args.eval_stage:
                pred_path = os.path.join(args.eval_log_dir, video, str(idx), f"{frame_ck[0]}.json")
                with open(pred_path, "r") as f:
                    pred_dict = json.load(f)
                data_out = {
                    k:torch.tensor(v) for k, v in pred_dict.items()
                }
            
            if args.gen_hand_mask:
                # get hand mask

                data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
                data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
                if do_flip: # left
                    outputs = run_mano_left(data_out["init_trans"], data_out["init_root_orient"], data_out["init_hand_pose"], betas=data_out["init_betas"])
                else: # right
                    outputs = run_mano(data_out["init_trans"], data_out["init_root_orient"], data_out["init_hand_pose"], betas=data_out["init_betas"])
                
                vertices = outputs["vertices"][0]  # (T, N, 3)
                mask_faces = faces_left_cuda if do_flip else faces_right_cuda
                mask_batch_size = 8
                for mask_start in range(0, len(img_ck), mask_batch_size):
                    mask_end = min(mask_start + mask_batch_size, len(img_ck))
                    masks = renderer.render_silhouettes_batch(
                        vertices[mask_start:mask_end].cuda(),
                        mask_faces,
                        mask_cameras,
                        mask_lights,
                    )
                    for mask_offset, mask in enumerate(masks):
                        model_masks[frame_ck[mask_start + mask_offset]] += mask

                continue

            # load this chunk of camera
            R_c2w_sla = R_c2w_sla_all[frame_ck]
            t_c2w_sla = t_c2w_sla_all[frame_ck]
            R_w2c_sla = R_w2c_sla_all[frame_ck]
            t_w2c_sla = t_w2c_sla_all[frame_ck]

            R_c2w_gt = R_c2w_gt_all[frame_ck]
            t_c2w_gt = t_c2w_gt_all[frame_ck]
            R_w2c_gt = R_w2c_gt_all[frame_ck]
            t_w2c_gt = t_w2c_gt_all[frame_ck]


            data_world = cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, 'right')

            pred_glob_r = run_mano(data_world["init_trans"], data_world["init_root_orient"], data_world["init_hand_pose"], betas=data_world["init_betas"])
            
            pred_j3d_glob = {}
            pred_j3d_glob['right'] = pred_glob_r['joints'][0]

            # load GT joints
            gt_trans_r = world_trans[1:2, frame_ck]
            gt_rot_r = world_rot[1:2, frame_ck]
            gt_pose_r = world_hand_pose[1:2, frame_ck]
            gt_betas_r = world_betas[1:2, frame_ck]
            target_glob_r = run_mano(gt_trans_r, gt_rot_r, gt_pose_r, betas=gt_betas_r)
            target_j3d_glob = {}
            target_j3d_glob['right'] = target_glob_r['joints'][0] # T, 21, 3 
            pred_world_joints[frame_ck] = pred_j3d_glob['right'].detach().cpu().numpy()
            gt_world_joints[frame_ck] = target_j3d_glob['right'].detach().cpu().numpy()

            chunk_length = 100
            w_mpjpe, wa_mpjpe = [], []
            T = target_j3d_glob['right'].shape[0]
            pred_j3d_glob['right_pa'] = batch_compute_similarity_transform_torch(pred_j3d_glob['right'], target_j3d_glob['right']) 
            pa_mpjpe = []
            for start in range(0, T, chunk_length):
                end = min(T, start + chunk_length)

                # TODO: for hand in ['left', 'right']:
                for hand in ['right']:

                    target_j3d = target_j3d_glob[hand][start:end].clone().cpu()
                    pred_j3d = pred_j3d_glob[hand][start:end].clone().cpu()
                    pred_j3d_pa = pred_j3d_glob[hand + "_pa"][start:end].clone().cpu()
                    
                    w_j3d = first_align_joints(target_j3d, pred_j3d)
                    wa_j3d = global_align_joints(target_j3d, pred_j3d)
                    
                    w_jpe = compute_jpe(target_j3d, w_j3d)
                    wa_jpe = compute_jpe(target_j3d, wa_j3d)
                    w_mpjpe.append(w_jpe)
                    wa_mpjpe.append(wa_jpe)

                    pa_jpe = compute_jpe(target_j3d, pred_j3d_pa)
                    pa_mpjpe.append(pa_jpe)
            accel = compute_error_accel(joints_pred=pred_j3d_glob['right'].cpu(), joints_gt=target_j3d_glob['right'].cpu())[1:-1]
            accel = accel * (30 ** 2)
            accumulator['accel'].append(accel)
            
            w_mpjpe = np.concatenate(w_mpjpe) * m2mm
            wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm
            pa_mpjpe = np.concatenate(pa_mpjpe) * m2mm

            accumulator['pa_mpjpe'].append(pa_mpjpe)

            accumulator['w_mpjpe'].append(w_mpjpe)
            accumulator['wa_mpjpe'].append(wa_mpjpe)
            video_w_mpjpe.append(w_mpjpe)
            video_wa_mpjpe.append(wa_mpjpe)
            video_pa_mpjpe.append(pa_mpjpe)
            video_accel.append(accel)
            eval_frame_ids.append(frame_ck)
            accel_frame_ids.append(frame_ck[2:-2])

            rte = compute_rte(gt_trans_r.squeeze(0), data_world["init_trans"].squeeze(0)) * 1e2
            accumulator['RTE'].append(rte)
            video_rte.append(rte)
            print(rte.mean())

    if args.gen_hand_mask:
        model_masks = model_masks > 0 # bool
        masks_ = []
        for mask in model_masks:
            mask_bit = masktool.encode(np.asfortranarray(mask > 0))
            masks_.append(mask_bit)
        del model_masks
        masks_ = np.array(masks_, dtype=object)
        np.save(f'{seq_folder}/model_masks.npy', masks_)
    
    if not args.inference_stage and not args.gen_hand_mask:
        frame_ids = np.concatenate(eval_frame_ids)
        w_values = np.concatenate(video_w_mpjpe)
        wa_values = np.concatenate(video_wa_mpjpe)
        pa_values = np.concatenate(video_pa_mpjpe)
        accel_ids = np.concatenate(accel_frame_ids)
        accel_values = np.concatenate(video_accel)
        rte_values = np.concatenate(video_rte)
        result_dir = os.path.join(args.eval_log_dir, video)
        os.makedirs(result_dir, exist_ok=True)
        np.savez(
            os.path.join(result_dir, 'evaluation_details.npz'),
            frame_ids=frame_ids,
            w_mpjpe_mm=w_values,
            wa_mpjpe_mm=wa_values,
            pa_mpjpe_mm=pa_values,
            accel_frame_ids=accel_ids,
            accel_m_s2=accel_values,
            rte_percent=rte_values,
            pred_world_joints_m=pred_world_joints,
            gt_world_joints_m=gt_world_joints,
        )
        summary = {
            'video': video,
            'total_frames': int(len(imgfiles)),
            'evaluated_right_hand_frames': int(len(frame_ids)),
            'w_mpjpe_mm': float(w_values.mean()),
            'wa_mpjpe_mm': float(wa_values.mean()),
            'pa_mpjpe_mm': float(pa_values.mean()),
            'accel_m_s2': float(accel_values.mean()),
            'rte_percent': float(rte_values.mean()),
        }
        with open(os.path.join(result_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        axes[0].plot(frame_ids, w_values, label='W-MPJPE', linewidth=0.8)
        axes[0].plot(frame_ids, wa_values, label='WA-MPJPE', linewidth=0.8)
        axes[0].plot(frame_ids, pa_values, label='PA-MPJPE', linewidth=0.8)
        axes[0].set_ylabel('Joint error (mm)')
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(accel_ids, accel_values, color='tab:orange', linewidth=0.8)
        axes[1].set_ylabel('Accel. error (m/s²)')
        axes[1].grid(alpha=0.25)
        axes[2].plot(frame_ids, rte_values, color='tab:green', linewidth=0.8)
        axes[2].set_ylabel('RTE (%)')
        axes[2].set_xlabel('Frame')
        axes[2].grid(alpha=0.25)
        fig.suptitle(f'HaWoR HOT3D evaluation: {video}')
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, 'metrics_over_time.png'), dpi=160)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*gt_world_joints[:, 0].T, label='HOT3D GT wrist', linewidth=1.2)
        ax.plot(*pred_world_joints[:, 0].T, label='HaWoR wrist', linewidth=1.0)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Right-hand world trajectory: {video}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, 'world_trajectory.png'), dpi=160)
        plt.close(fig)

        video_log = f"{video} {w_values.mean()} {wa_values.mean()}"
        print(video_log)
        eval_log.write(video_log + "\n")
        video_w_mpjpe = []
        video_wa_mpjpe = []

if not args.inference_stage and not args.gen_hand_mask:
    for k, v in accumulator.items():
        accumulator[k] = np.concatenate(v).mean()

    log_str = f'Evaluation on hot3d {args.video_root}, '
    log_str += ' '.join([f'{k.upper()},'for k,v in accumulator.items()])
    log_str += ' '.join([f'{v:.2f}'for k,v in accumulator.items()])
    print(log_str)
    eval_log.write(log_str + "\n")
    eval_log.close()

    per_video = []
    for video in testset:
        with open(os.path.join(args.eval_log_dir, video, 'evaluation_summary.json'), 'r') as f:
            per_video.append(json.load(f))
    aggregate = {
        'num_videos': len(testset),
        'total_frames': sum(item['total_frames'] for item in per_video),
        'evaluated_right_hand_frames': sum(
            item['evaluated_right_hand_frames'] for item in per_video),
        'w_mpjpe_mm': float(accumulator['w_mpjpe']),
        'wa_mpjpe_mm': float(accumulator['wa_mpjpe']),
        'pa_mpjpe_mm': float(accumulator['pa_mpjpe']),
        'accel_m_s2': float(accumulator['accel']),
        'rte_percent': float(accumulator['RTE']),
    }
    summary_all = {'aggregate': aggregate, 'videos': per_video}
    summary_all_path = os.path.join(
        args.eval_log_dir, f'evaluation_summary_{len(testset)}videos.json')
    with open(summary_all_path, 'w') as f:
        json.dump(summary_all, f, indent=2)

    metric_specs = [
        ('w_mpjpe_mm', 'W-MPJPE (mm)'),
        ('wa_mpjpe_mm', 'WA-MPJPE (mm)'),
        ('pa_mpjpe_mm', 'PA-MPJPE (mm)'),
        ('accel_m_s2', 'Acceleration error (m/s²)'),
        ('rte_percent', 'RTE (%)'),
    ]
    labels = [item['video'] for item in per_video]
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(14, 15), sharex=True)
    for ax, (key, ylabel) in zip(axes, metric_specs):
        ax.bar(np.arange(len(per_video)), [item[key] for item in per_video])
        ax.axhline(aggregate[key], color='tab:red', linestyle='--', linewidth=1,
                   label=f"aggregate: {aggregate[key]:.2f}")
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.25)
        ax.legend(loc='upper right')
    axes[-1].set_xticks(np.arange(len(per_video)), labels, rotation=30, ha='right')
    fig.suptitle(f'HaWoR HOT3D evaluation across {len(testset)} videos')
    fig.tight_layout()
    fig.savefig(os.path.join(
        args.eval_log_dir, f'metrics_{len(testset)}videos.png'), dpi=160)
    plt.close(fig)

        




