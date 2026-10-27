#!/usr/bin/env python3
"""Run a trained handloop_v2 system with live HaWoR IAM, without reading GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(ROOT)]

from dual_loop.handloop_data import crop_affine
from dual_loop.handloop_runtime import build_system, move_batch, rollout, FeedbackHistory
from dual_loop.handloop_state import state_to_legacy
from dual_loop.io import (_empty_predictions, _resolve_tracks_path, _load_right_hand_tracks,
                          load_timestamps, save_predictions)
from run_hot3d import load_slam_cameras
from prepare_handloop_data import fill_boxes


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def load_inputs(sequence_root, allow_gt_boxes=False, interval=1000):
    images = sorted((sequence_root / 'extracted_images').glob('*.jpg'))
    if not images:
        raise ValueError(f'No JPEG frames in {sequence_root}')
    image = cv2.imread(str(images[0]))
    if image is None:
        raise ValueError(f'Cannot read {images[0]}')
    size = np.array([image.shape[1], image.shape[0]])
    track_path, assisted = _resolve_tracks_path(sequence_root, allow_gt_boxes)
    tracks = _load_right_hand_tracks(track_path)
    if len(tracks) != len(images):
        raise ValueError('Track/image length mismatch')
    boxes = np.full((len(images), 4), np.nan, np.float32)
    for index, track in enumerate(tracks):
        if isinstance(track, dict) and track.get('det', False):
            box = np.asarray(track.get('det_box', []), np.float32)
            if box.size and box.shape[-1] >= 4:
                boxes[index] = box.reshape(-1, box.shape[-1])[0, :4]
    valid = np.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
    segments = np.arange(len(images)) // interval
    boxes = fill_boxes(boxes, size, segments)
    ns, seconds = load_timestamps(sequence_root, len(images))
    # This loader opens only SLAM files, not camera truth or hand annotations.
    rcw, tcw, rwc, twc = load_slam_cameras(sequence_root, len(images), interval, len(images))
    return dict(images=images, boxes=boxes, valid=valid, size=size, segments=segments,
                ns=ns, seconds=seconds, rotation_w2c=rwc, translation_w2c=twc,
                focal=float((sequence_root / 'focal.txt').read_text()),
                track_file=str(track_path), track_sha256=hash_file(track_path), oracle_assisted=assisted)


def context_batch(inputs, ids):
    from lib.datasets.track_dataset import TrackDatasetEval
    db = TrackDatasetEval([str(inputs['images'][i]) for i in ids], inputs['boxes'][ids].copy(),
                          img_focal=inputs['focal'], img_center=inputs['size'] / 2,
                          normalization=True, dilate=1.2)
    items = [db[i] for i in range(len(ids))]
    hawor = {key: torch.stack([item[key] for item in items])[None] for key in
             ('img', 'center', 'scale', 'img_focal', 'img_center')}
    return dict(hawor=hawor, pixel_to_patch=torch.from_numpy(np.stack([
        crop_affine(item['center'].numpy(), float(item['scale'])) for item in items]))[None],
        rotation_w2c=inputs['rotation_w2c'][ids][None], translation_w2c=inputs['translation_w2c'][ids][None],
        timestamp_sec=torch.from_numpy(inputs['seconds'][ids])[None],
        segment_id=torch.from_numpy(inputs['segments'][ids])[None],
        handedness=torch.ones(1, len(ids), dtype=torch.long),
        raw_available=torch.from_numpy(np.repeat(inputs['valid'][ids, None], 17, axis=1))[None],
        image_available=torch.from_numpy(inputs['valid'][ids])[None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--hawor_checkpoint', type=Path, default=ROOT / 'weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument('--video_root', type=Path, default=ROOT / 'datasets/hot3d_valset_export')
    parser.add_argument('--set_file', default='val.json')
    parser.add_argument('--video', action='append')
    parser.add_argument('--output_root', type=Path, default=ROOT / 'eval_log_handloop_v2')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--allow_gt_boxes', action='store_true')
    parser.add_argument('--max_frames', type=int)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture') != 'handloop_v2' or checkpoint.get('stage') != 'joint_visual':
        raise ValueError('A handloop_v2 joint_visual checkpoint is required; legacy/prior-only weights are incompatible')
    config = checkpoint['config']
    system = build_system(args.hawor_checkpoint, config, args.device).eval()
    system.load_state_dict(checkpoint['model'], strict=True)
    sequences = json.loads((args.video_root / args.set_file).read_text())
    if args.video:
        unknown = set(args.video) - set(sequences)
        if unknown:
            raise ValueError(f'Unknown sequences: {sorted(unknown)}')
        sequences = [seq for seq in sequences if seq in args.video]
    checkpoint_hash = hash_file(args.checkpoint)
    for sequence in sequences:
        output = args.output_root / sequence
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite inference results: {output}')
        inputs = load_inputs(args.video_root / sequence, args.allow_gt_boxes)
        n = len(inputs['images'])
        frames = min(n, args.max_frames) if args.max_frames else n
        if frames < 1:
            raise ValueError('max_frames must be positive')
        prediction = _empty_predictions(n, inputs['size'])
        prediction['bbox'][:] = inputs['boxes']
        prediction['focal_length'][:] = inputs['focal']
        trace = np.full((n, 99), np.nan, np.float32)
        dispersion = np.full((n, 17), np.nan, np.float32)
        sigma = np.full((n, 3), np.nan, np.float32)
        kappa = np.full((n, 16), np.nan, np.float32)
        anchor_rotation = np.full((n, 3, 3), np.nan, np.float32)
        trace_betas = np.full((n, 10), np.nan, np.float32)
        state_available = np.zeros((n, 17), dtype=bool)
        history = FeedbackHistory(system.observer.config.twm.history_length)
        with torch.inference_mode():
            start = 0
            while start < frames:
                # Never let bidirectional HaWoR contexts cross an SLAM reset.
                end = min(start + 16, ((start // 1000) + 1) * 1000, frames)
                real = end - start
                ids = list(range(start, end)) + [end - 1] * (16 - real)
                batch = move_batch(context_batch(inputs, ids), args.device)
                outputs, frontend, history = rollout(system, batch, history, real_frames=real)
                for local, result in enumerate(outputs):
                    frame = start + local
                    trace[frame] = result['state'][0].cpu().numpy()
                    dispersion[frame] = result['dispersion'][0].cpu().numpy()
                    sigma[frame] = result['current']['sigma'][0].cpu().numpy()
                    kappa[frame] = result['current']['kappa'][0].cpu().numpy()
                    anchor_rotation[frame] = result['anchor_rotation'][0].cpu().numpy()
                    trace_betas[frame] = result['betas'][0].cpu().numpy()
                    state_available[frame] = result['available'][0].cpu().numpy()
                    if inputs['valid'][frame]:
                        legacy = state_to_legacy(result['state'], frontend['betas'][:, local], system.geometry)
                        trans, root, hand, beta = system.geometry.world_to_camera_parameters(
                            legacy, batch['rotation_w2c'][:, local], batch['translation_w2c'][:, local])
                        for key, value in [('camera_translation', trans), ('global_orient', root),
                                           ('hand_pose', hand), ('betas', beta)]:
                            prediction[key][frame] = value[0].cpu().numpy()
                        prediction['pred_cam'][frame] = frontend['pred_cam'][0, local].cpu().numpy()
                        prediction['valid'][frame] = True
                start = end
                print(f'[{sequence}] {end}/{frames}', flush=True)
        output.mkdir(parents=True)
        save_predictions(output / 'try_predictions.npz', prediction)
        np.savez_compressed(output / 'handloop_trace.npz', frame_id=np.arange(n), timestamp_ns=inputs['ns'],
                            timestamp_sec=inputs['seconds'], segment_id=inputs['segments'], state_world=trace,
                            dispersion=dispersion, sigma_anchor=sigma, kappa=kappa, anchor_rotation=anchor_rotation,
                            betas=trace_betas, state_available=state_available,
                            visual_valid=prediction['valid'], architecture=np.asarray('handloop_v2'),
                            coordinate_frame=np.asarray('per-interval raw SLAM world; not registered across segments'))
        metadata = dict(sequence=sequence, method='HandLoop v2', architecture='handloop_v2',
                        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                        trained_checkpoint_loaded=True, total_frames=n, processed_frames=frames,
                        oracle_assisted=inputs['oracle_assisted'], track_file=inputs['track_file'],
                        track_sha256=inputs['track_sha256'], hawor_bidirectional_context=16,
                        numerical_feedback_causal=True, uncertainty_is_calibrated_correctness=False,
                        state_position='anatomical_wrist', beta_is_dynamic_state=False)
        (output / 'try_inference.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
