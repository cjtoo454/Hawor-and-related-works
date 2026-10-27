#!/usr/bin/env python3
"""Verify metric parity, then compare a frozen dual-loop run with archived HaWoR."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import psutil

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(ROOT), str(PROJECT / 'scripts')]
from dual_loop.io import export_hawor_frontend, save_predictions
import eval_hot3d as evaluator

KEYS = ('w_mpjpe_mm', 'wa_mpjpe_mm', 'pa_mpjpe_mm', 'accel_m_s2', 'rte_percent')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metric_frame_ids(valid):
    frames = np.flatnonzero(valid)
    chunks, _ = evaluator.parse_chunks(frames, np.zeros((len(frames), 4)), min_len=16)
    if not chunks:
        raise ValueError('No evaluation chunks')
    return np.concatenate(chunks), np.concatenate([chunk[2:-2] for chunk in chunks])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--video_root', type=Path, default=ROOT / 'datasets/hot3d_valset_export')
    parser.add_argument('--baseline_root', type=Path, default=ROOT / 'eval_log')
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.video_root = args.video_root.resolve()
    args.baseline_root = args.baseline_root.resolve()
    launch = json.loads((args.output_root / 'inference_launch.json').read_text())
    checkpoint = Path(launch['checkpoint'])
    if sha256(checkpoint) != launch['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after inference launch')
    sequences = json.loads((args.video_root / 'val.json').read_text())
    control_root = args.output_root / 'protocol_check_baseline'
    protocol, masks, baseline_arrays = [], {}, {key: [] for key in KEYS}
    for sequence in sequences:
        frontend, provenance = export_hawor_frontend(
            args.video_root / sequence, args.baseline_root / sequence,
            allow_gt_boxes=True, return_provenance=True)
        with np.load(args.baseline_root / sequence / 'evaluation_details.npz') as archive:
            old = {key: archive[key].copy() for key in archive.files}
        annotations = joblib.load(args.video_root / sequence / 'anno.pth')
        gt_valid = annotations['rot_r'].ne(0).any(dim=-1).cpu().numpy()
        ids, accel_ids = metric_frame_ids(frontend['valid'] & gt_valid)
        np.testing.assert_array_equal(ids, old['frame_ids'])
        np.testing.assert_array_equal(accel_ids, old['accel_frame_ids'])
        common = np.zeros(len(frontend['valid']), dtype=bool)
        common[ids] = True
        masks[sequence] = common
        directory = control_root / sequence
        save_predictions(directory / 'try_predictions.npz', frontend)
        # This is a metric-reproduction diagnostic, not proposed-model output.
        write_json(directory / 'try_inference.json', {
            'trained_checkpoint_loaded': False, 'oracle_assisted': True,
            'purpose': 'Protocol parity diagnostic using archived HaWoR; no dual-loop applied',
        })
        config = SimpleNamespace(video_root=args.video_root, prediction_root=control_root,
            eval_log_dir=control_root, interval=1000, device='cpu',
            common_frame_mask=None, allow_untrained=True)
        result = evaluator.evaluate_sequence(config, sequence)
        differences = {}
        for key in KEYS:
            np.testing.assert_allclose(result['_values'][key], old[key], rtol=1e-4, atol=0.01,
                err_msg=f'{sequence}: metric implementation differs from archived HaWoR: {key}')
            differences[key] = float(np.max(np.abs(result['_values'][key] - old[key])))
            baseline_arrays[key].append(old[key])
        protocol.append({'sequence': sequence, 'evaluated_frames': len(ids),
            'accel_frames': len(accel_ids), 'max_absolute_metric_differences': differences,
            'track_sha256': provenance['track_sha256'],
            'frontend_chunk_sha256': provenance['chunk_sha256'],
            'baseline_details_sha256': sha256(args.baseline_root / sequence / 'evaluation_details.npz')})
        write_json(args.output_root / 'protocol_parity.json', {'passed_sequences': protocol})
        print(f'[protocol verified] {sequence}: {len(ids)} frames', flush=True)
    mask_path = args.output_root / 'common_frame_mask.npz'
    np.savez_compressed(mask_path, **masks)

    while True:
        active = []
        for job in launch['jobs']:
            try:
                process = psutil.Process(job['pid'])
                if process.status() != psutil.STATUS_ZOMBIE and 'run_hot3d.py' in ' '.join(process.cmdline()):
                    active.append(job['pid'])
            except psutil.NoSuchProcess:
                pass
        if not active:
            break
        print(f'[waiting for inference] {active}', flush=True)
        time.sleep(30)

    for sequence in sequences:
        directory = args.output_root / sequence
        metadata = json.loads((directory / 'try_inference.json').read_text())
        if not metadata.get('trained_checkpoint_loaded') or Path(metadata['checkpoint']) != checkpoint:
            raise ValueError(f'{sequence}: wrong model provenance')
        if metadata['processed_frames'] != metadata['total_frames']:
            raise ValueError(f'{sequence}: truncated inference is not a complete benchmark')
        with np.load(directory / 'try_predictions.npz') as predictions:
            annotations = joblib.load(args.video_root / sequence / 'anno.pth')
            ids, accel_ids = metric_frame_ids(predictions['valid'] & annotations['rot_r'].ne(0).any(dim=-1).numpy())
            np.testing.assert_array_equal(ids, np.flatnonzero(masks[sequence]))
            for key in ('global_orient', 'hand_pose', 'betas', 'camera_translation'):
                if not np.isfinite(predictions[key][ids]).all():
                    raise ValueError(f'{sequence}: nonfinite proposed-model predictions')
    sys.argv = [str(PROJECT / 'scripts/eval_hot3d.py'), '--video_root', str(args.video_root),
                '--prediction_root', str(args.output_root), '--eval_log_dir', str(args.output_root),
                '--common_frame_mask', str(mask_path), '--device', 'cpu']
    evaluator.main()
    summary = json.loads((args.output_root / f'evaluation_summary_{len(sequences)}videos.json').read_text())
    baseline = {key: float(np.concatenate(arrays).mean()) for key, arrays in baseline_arrays.items()}
    for sequence in sequences:
        with np.load(args.output_root / sequence / 'evaluation_details.npz') as new:
            with np.load(args.baseline_root / sequence / 'evaluation_details.npz') as old:
                for key in ('frame_ids', 'accel_frame_ids'):
                    np.testing.assert_array_equal(new[key], old[key])
    comparison = {'protocol': 'Matched HaWoR controlled benchmark; GT-projected boxes; identical visual-valid frames',
        'checkpoint': str(checkpoint), 'checkpoint_sha256': launch['checkpoint_sha256'],
        'checkpoint_epoch': 29, 'num_videos': len(sequences),
        'evaluated_frames': summary['aggregate']['evaluated_right_hand_frames'],
        'baseline': baseline, 'dual_loop': {key: summary['aggregate'][key] for key in KEYS},
        'relative_error_reduction_percent': {key: 100 * (baseline[key] - summary['aggregate'][key]) / baseline[key] for key in KEYS},
        'protocol_parity_verified': True, 'evaluation_frames_identical': True,
        'note': 'Positive reduction is improvement; negative reduction is degradation. No checkpoint selected using test results.'}
    write_json(args.output_root / 'comparison_with_hawor.json', comparison)
    with (args.output_root / 'comparison_with_hawor.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'HaWoR', 'HaWoR+dual_loop_epoch29', 'error_reduction_percent'])
        for key in KEYS:
            writer.writerow([key, baseline[key], comparison['dual_loop'][key], comparison['relative_error_reduction_percent'][key]])
    print('[comparison complete]\n' + json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
