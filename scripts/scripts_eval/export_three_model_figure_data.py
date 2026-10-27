"""Export all numerical inputs for the three-model segment-1/frame-245 figure."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import joblib
from scripts.scripts_eval import plot_gt_pred_hand_mesh as plot
from scripts.scripts_eval.recovered_camera_cache import install_mano_legacy_compat


def digest(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            d.update(block)
    return d.hexdigest()


def main():
    install_mano_legacy_compat()
    from hawor.utils.process import run_mano, get_mano_faces
    from lib.eval_utils.custom_utils import cam2world_convert
    from hawor.utils.rotation import rotation_matrix_to_angle_axis
    sys.path.insert(0, str(ROOT / 'scripts/scripts_eval'))
    from recover_wilor_aligned_cameras import fit_rigid_camera
    torch.set_num_threads(1)
    video, frame = 'P0001_8d136980', 245
    start, stop = 0, 379
    output = ROOT / 'figure_data' / f'{video}_segment01_frame245_three_models.npz'
    if output.exists():
        raise FileExistsError(output)
    video_root = ROOT / 'datasets/hot3d_valset_export'
    annotation_path = video_root / video / 'anno.pth'
    annotation = joblib.load(annotation_path)
    source_hashes = {str(annotation_path.relative_to(ROOT)): digest(annotation_path)}
    model_path = ROOT / '_DATA/data/mano/MANO_RIGHT.pkl'
    source_hashes[str(model_path.relative_to(ROOT))] = digest(model_path)
    gt = run_mano(annotation['trans_r'][[frame]].unsqueeze(0),
                  annotation['rot_r'][[frame]].unsqueeze(0),
                  annotation['pose_r'][[frame]].unsqueeze(0),
                  betas=annotation['betas_r'][[frame]].unsqueeze(0))
    gt_joints = gt['joints'][0, 0].detach().cpu().numpy()
    gt_vertices = gt['vertices'][0, 0].detach().cpu().numpy()
    payload = {
        'trajectory_frame_ids': np.arange(start, stop, dtype=np.int64),
        'trajectory_plot_time_s': np.arange(start, stop, dtype=np.float64) / 30.,
        'pose_frame_id': np.asarray(frame, dtype=np.int64),
        'mesh_faces': np.asarray(get_mano_faces(), dtype=np.int32),
        'hand_bone_edges': np.asarray(plot.HAND_BONES, dtype=np.int32),
    }
    results, common_gt = {}, None
    for name in ['wilor_slam', 'hmp_slam', 'hawor']:
        method = plot.METHODS[name]
        folder = ROOT / method['log_dir'] / video
        details_path = folder / 'evaluation_details.npz'
        with np.load(details_path, allow_pickle=False) as data:
            total_frames = len(data['pred_world_joints_m'])
            predicted_joints = data['pred_world_joints_m'][start:stop].copy()
            target_joints = data['gt_world_joints_m'][start:stop].copy()
        assert total_frames == 3797
        if common_gt is None:
            common_gt = target_joints
        else:
            assert np.allclose(common_gt, target_joints, equal_nan=True)
        ids = np.asarray([frame], dtype=np.int64)
        if method['prediction_file'] is None:
            prediction = plot.load_sampled_prediction(folder / '1', ids)
            prediction_paths = []
            for path in (folder / '1').glob('*.json'):
                obj = json.loads(path.read_text())
                if int(path.stem) <= frame < int(path.stem) + len(obj['init_trans'][0]):
                    prediction_paths.append(path)
            assert len(prediction_paths) == 1
        else:
            prediction_paths = [folder / method['prediction_file']]
            prediction = plot.load_compact_prediction(prediction_paths[0], ids)
        rotation, translation = plot.load_aligned_slam_cameras(
            video_root, video, total_frames, method['slam_dir'], method['slam_prefix'])
        assert torch.isfinite(rotation[frame]).all() and torch.isfinite(translation[frame]).all()
        world = cam2world_convert(rotation[ids], translation[ids], prediction, 'right')
        mesh = run_mano(world['init_trans'], world['init_root_orient'],
                        world['init_hand_pose'], betas=world['init_betas'])
        joints = mesh['joints'][0, 0].detach().cpu().numpy()
        vertices = mesh['vertices'][0, 0].detach().cpu().numpy()
        pred_error = float(np.max(np.abs(joints - predicted_joints[frame-start])))
        gt_error = float(np.max(np.abs(gt_joints - target_joints[frame-start])))
        original_camera_error = pred_error
        camera_source = 'original aligned SLAM' if name == 'hawor' else 'validated recovered aligned SLAM cache'
        rigid_residual = None
        if name == 'hawor' and pred_error > 2e-5:
            camera_mesh = run_mano(prediction['init_trans'],
                                   rotation_matrix_to_angle_axis(prediction['init_root_orient']),
                                   rotation_matrix_to_angle_axis(prediction['init_hand_pose']),
                                   betas=prediction['init_betas'])
            local_joints = camera_mesh['joints'][0, 0].detach().cpu().numpy()
            r, t, rigid_residual = fit_rigid_camera(local_joints, predicted_joints[frame-start])
            assert rigid_residual <= 2e-5, rigid_residual
            rotation[frame] = torch.as_tensor(r, dtype=rotation.dtype)
            translation[frame] = torch.as_tensor(t, dtype=translation.dtype)
            world = cam2world_convert(rotation[ids], translation[ids], prediction, 'right')
            mesh = run_mano(world['init_trans'], world['init_root_orient'], world['init_hand_pose'], betas=world['init_betas'])
            joints = mesh['joints'][0, 0].detach().cpu().numpy()
            vertices = mesh['vertices'][0, 0].detach().cpu().numpy()
            pred_error = float(np.max(np.abs(joints - predicted_joints[frame-start])))
            camera_source = 'single-frame aligned SE3 recovered from original saved world joints and MANO camera parameters'
        assert pred_error <= 2e-5 and gt_error <= 2e-5, (name, pred_error, gt_error)
        prefix = name + '__'
        payload.update({prefix+'trajectory_gt_xyz_m': target_joints[:, 0].copy(),
                        prefix+'trajectory_prediction_xyz_m': predicted_joints[:, 0].copy(),
                        prefix+'trajectory_valid': np.isfinite(predicted_joints).all((1, 2)) & np.isfinite(target_joints).all((1, 2)),
                        prefix+'pose_gt_joints_m': gt_joints.copy(),
                        prefix+'pose_prediction_joints_m': joints,
                        prefix+'pose_gt_vertices_m': gt_vertices.copy(),
                        prefix+'pose_prediction_vertices_m': vertices,
                        prefix+'camera_rotation_c2w_frame245': rotation[frame].numpy(),
                        prefix+'camera_translation_c2w_frame245_m': translation[frame].numpy(),
                        prefix+'mano_prediction_betas': world['init_betas'][0, 0].detach().cpu().numpy(),
                        prefix+'mano_prediction_root_axis_angle_world': world['init_root_orient'][0, 0].detach().cpu().numpy(),
                        prefix+'mano_prediction_hand_axis_angle': world['init_hand_pose'][0, 0].detach().cpu().numpy(),
                        prefix+'mano_prediction_translation_world_m': world['init_trans'][0, 0].detach().cpu().numpy()})
        for path in [details_path, *prediction_paths]:
            source_hashes[str(path.relative_to(ROOT))] = digest(path)
        if name != 'hawor':
            cache = video_root / video / 'preprocess/SLAM_WiLoR/recovered_aligned_cameras.npz'
            for path in [cache, cache.with_suffix('.json')]:
                source_hashes[str(path.relative_to(ROOT))] = digest(path)
        else:
            for path in (video_root / video / 'preprocess/SLAM').glob('hawor_slam_w_scale_*.npz'):
                source_hashes[str(path.relative_to(ROOT))] = digest(path)
            for filename in ['head_pose.pkl', 'ego_extrinsics.pkl']:
                path = video_root / video / filename
                source_hashes[str(path.relative_to(ROOT))] = digest(path)
        results[name] = {'label': method['label'], 'prediction_max_abs_error_m': pred_error,
                         'GT_max_abs_error_m': gt_error,
                         'valid_trajectory_frames': int(payload[prefix+'trajectory_valid'].sum()),
                         'camera_source': camera_source, 'original_camera_reconstruction_error_m': original_camera_error,
                         'single_frame_rigid_fit_residual_m': rigid_residual}
    metadata = {
        'format': 'three_model_figure_data_v1', 'video': video,
        'purpose': 'numerical source of the three-panel evaluation figure; not training-set data',
        'segment_index': 1, 'segment_total': 10, 'frame_range_inclusive': [0, 378],
        'pose_frame_id': frame, 'coordinate_frame': 'HOT3D evaluation world, camera-aligned; metres',
        'trajectory_time_base': 'frame_id / 30 fps, matching plot labels; not hardware timestamps',
        'models': results, 'joint_order': '0 wrist;1-4 thumb;5-8 index;9-12 middle;13-16 ring;17-20 little',
        'field_layout': 'method__field, methods=wilor_slam,hmp_slam,hawor; GT duplicated per method for convenience',
        'trajectory_fields': 'trajectory_gt_xyz_m/prediction_xyz_m:379x3, NaN gaps preserved; trajectory_valid:379 bool',
        'pose_fields': 'pose_*_joints_m:21x3; pose_*_vertices_m:778x3; mesh_faces:Fx3 vertex indices',
        'marker': 'yellow marker is trajectory_prediction_xyz_m[245] for each model',
        'camera_note': 'HMP/WiLoR use recovered aligned poses on verified evaluated frames, not original raw DROID outputs; HaWoR frame245 is also recovered from saved evaluation joints if current SLAM does not reproduce the original figure',
        'source_sha256': source_hashes,
    }
    payload['metadata_json'] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **payload)
    with np.load(temporary, allow_pickle=False) as saved:
        assert len(saved.files) == len(payload)
        assert saved['mesh_faces'].min() >= 0 and saved['mesh_faces'].max() < 778
        for model in results:
            pre = model + '__'
            assert np.array_equal(saved[pre+'trajectory_prediction_xyz_m'][245], payload[pre+'trajectory_prediction_xyz_m'][245])
            assert np.allclose(saved[pre+'pose_prediction_joints_m'][0], saved[pre+'trajectory_prediction_xyz_m'][245], atol=2e-5)
            assert saved[pre+'pose_prediction_vertices_m'].shape == (778, 3)
    temporary.replace(output)
    print(json.dumps({'output': str(output), 'bytes': output.stat().st_size, 'validation': results}, indent=2), flush=True)


if __name__ == '__main__':
    main()
