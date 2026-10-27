#!/usr/bin/env python3
"""Calibrate and evaluate HaWoR pinch events against HOT3D geometry proxies.

This experiment does not manufacture human event annotations or software logs.
Optional simulated expected windows are explicitly labeled as such.
"""
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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(ROOT)]
from dual_loop.geometry import ManoGeometry
from dual_loop.state import pack_state, axis_angle_to_matrix
from dual_loop.oracle import normalized_pinch_distance, PinchOracleConfig, judge_time_window
from dual_loop.pinch_evaluation import extract_events, causal_ema, match_events

METHODS = ('HaWoR', 'HaWoR + Temporal Filter')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def decode(geometry, states, valid, device):
    result = np.full((len(states), 21, 3), np.nan, np.float32)
    ids = np.flatnonzero(valid)
    with torch.inference_mode():
        for start in range(0, len(ids), 256):
            chosen = ids[start:start + 256]
            result[chosen] = geometry.joints(torch.as_tensor(states[chosen], device=device).float()).cpu().numpy()
    return result


def relative_time(seconds):
    seconds = np.asarray(seconds, dtype=np.float64)
    if not np.isfinite(seconds).all() or (np.diff(seconds) <= 0).any():
        raise ValueError('Timestamps must be finite and strictly increasing')
    return seconds - seconds[0]


def calibration_sequence(path, geometry, device):
    with np.load(path, allow_pickle=False) as z:
        pred, target = z['visual_state'], z['gt_state']
        pv, gv = z['visual_valid'].astype(bool), z['gt_valid'].astype(bool)
        times, segments = relative_time(z['timestamp_sec']), z['segment_id'].copy()
        if bool(z['oracle_assisted']) or str(z['oracle_assisted_status'].item()) != 'verified_no_oracle':
            raise ValueError(f'Calibration requires verified independent observations: {path}')
    pv &= np.isfinite(pred).all(-1)
    gv &= np.isfinite(target).all(-1)
    return decode(geometry, pred, pv, device), decode(geometry, target, gv, device), times, segments, pv, gv


def test_sequence(root, baseline, sequence, geometry, device):
    details = baseline / sequence / 'evaluation_details.npz'
    with np.load(details, allow_pickle=False) as z:
        pred = z['pred_world_joints_m'].copy()
        archived_ids = z['frame_ids'].copy()
    pv = np.isfinite(pred).all((1, 2))
    np.testing.assert_array_equal(np.flatnonzero(pv), archived_ids)
    annotation_path = root / sequence / 'anno.pth'
    annotations = joblib.load(annotation_path)
    # Preserve the baseline's declared reference-availability convention.
    gv = torch.any(annotations['rot_r'] != 0, dim=-1).numpy()
    states = pack_state(annotations['trans_r'].float(),
                        axis_angle_to_matrix(annotations['rot_r'].float())[:, None],
                        axis_angle_to_matrix(annotations['pose_r'].float().reshape(-1, 15, 3)),
                        annotations['betas_r'].float()).numpy()
    gt = decode(geometry, states, gv, device)
    with (root / sequence / 'tt_list.pkl').open('rb') as handle:
        timestamps_ns = np.asarray(pickle.load(handle), dtype=np.int64)
    if len(gt) != len(pred) or len(gt) != len(timestamps_ns):
        raise ValueError(f'Frame count mismatch: {sequence}')
    times = relative_time((timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1e9)
    return pred, gt, times, np.arange(len(pred)) // 1000, pv, gv, {
        'prediction_details': str(details.resolve()), 'prediction_sha256': file_hash(details),
        'annotation_sha256': file_hash(annotation_path),
        'timestamps_sha256': file_hash(root / sequence / 'tt_list.pkl')}


def signals(pred, gt, times, segments, pv, gv, tau):
    filtered = causal_ema(pred, times, pv, segments, tau_sec=tau)
    return [normalized_pinch_distance(pred), normalized_pinch_distance(filtered)], normalized_pinch_distance(gt), filtered


def aggregate(results):
    tp, fp, fn = (sum(r[k] for r in results) for k in ('tp', 'fp', 'fn'))
    short_total = sum(r['short_reference_events'] for r in results)
    short_tp = sum(r['short_matched_events'] for r in results)
    errors = [m['onset_error_ms'] for r in results for m in r['matches']]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return dict(tp=tp, fp=fp, fn=fn, reference_events=tp + fn, predicted_events=tp + fp,
                precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                onset_mae_ms=float(np.abs(errors).mean()) if errors else None,
                onset_bias_ms=float(np.mean(errors)) if errors else None,
                onset_p95_absolute_error_ms=float(np.percentile(np.abs(errors), 95)) if errors else None,
                short_reference_events=short_total, short_matched_events=short_tp,
                short_event_recall=short_tp / short_total if short_total else None)


def acceleration_errors(predicted, reference, times, valid, segments):
    dt = np.diff(times)
    velocity_pred = np.diff(predicted, axis=0) / dt[:, None, None]
    velocity_ref = np.diff(reference, axis=0) / dt[:, None, None]
    center_dt = (dt[:-1] + dt[1:]) / 2
    accel_pred = np.diff(velocity_pred, axis=0) / center_dt[:, None, None]
    accel_ref = np.diff(velocity_ref, axis=0) / center_dt[:, None, None]
    keep = (valid[:-2] & valid[1:-1] & valid[2:] & (dt[:-1] <= .25) & (dt[1:] <= .25)
            & (segments[:-2] == segments[1:-1]) & (segments[1:-1] == segments[2:]))
    return np.linalg.norm(accel_pred[keep] - accel_ref[keep], axis=-1).mean(-1)


def evaluate_distances(distances, reference, times, pv, gv, threshold, reference_threshold, min_frames, tolerance):
    refs = extract_events(reference, times, gv, reference_threshold, min_frames=min_frames)
    predictions = [extract_events(d, times, pv, threshold, min_frames=min_frames) for d in distances]
    metrics = [match_events(refs, pred, onset_tolerance_sec=tolerance) for pred in predictions]
    return refs, predictions, metrics


def simulated_windows(refs, reference_distance, times, gv, reference_threshold, tolerance, min_frames):
    windows = []
    for i, event in enumerate(refs):
        windows.append(dict(id=f'positive_{i}', start=max(float(times[0]), event.onset_sec - tolerance),
                            stop=min(float(times[-1]), event.onset_sec + tolerance), expected='pass',
                            source='simulated_from_GT_geometry_reference_onset'))
    # Deterministic negative controls, not software-generated expectations.
    for center in np.arange(1., times[-1] - tolerance, 2.):
        lo, hi = center - tolerance, center + tolerance
        a, b = np.searchsorted(times, [lo, hi], side='left')
        a, b = max(0, a - 1), min(len(times), b + min_frames)
        if b > a and gv[a:b].all() and (reference_distance[a:b] >= reference_threshold).all():
            windows.append(dict(id=f'negative_{len(windows)}', start=float(lo), stop=float(hi), expected='fail',
                                source='simulated_from_GT_geometry_open_intervals'))
    return windows


def judge_expected_window(joints, valid, times, start, stop, oracle, max_gap_s=.25):
    """Fail closed when the window/confirmation interval crosses a time gap."""
    first = int(np.searchsorted(times, start, side='left'))
    last = int(np.searchsorted(times, stop, side='right') - 1)
    if first <= last:
        a = max(0, first - 1)
        b = min(len(times) - 1, last + oracle.consecutive_frames - 1)
        if (np.diff(times[a:b + 1]) > max_gap_s).any():
            return 'unknown'
    return judge_time_window(joints, valid.astype(float), times, (start, stop), oracle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video_root', type=Path, default=ROOT / 'datasets/hot3d_valset_export')
    p.add_argument('--baseline_root', type=Path, default=ROOT / 'eval_log')
    p.add_argument('--calibration_root', type=Path, default=PROJECT / 'data/prepared_train100')
    p.add_argument('--split_file', type=Path, default=PROJECT / 'runs/hot3d_train100_full_dualloop/full_model/split_manifest.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--reference_threshold', type=float, default=.2)
    p.add_argument('--prediction_thresholds', type=float, nargs='+', default=[.15, .175, .2, .225, .25, .275, .3])
    p.add_argument('--min_frames', type=int, default=3)
    p.add_argument('--filter_tau_sec', type=float, default=.1)
    p.add_argument('--onset_tolerance_sec', type=float, default=.2)
    p.add_argument('--software_windows', type=Path,
                   help='Optional JSON {sequence:[{id,start,stop,expected:pass|fail}]}; seconds relative sequence start')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite experiment: {args.output}')
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    sequences = json.loads((args.video_root / 'val.json').read_text())
    calibration = json.loads(args.split_file.read_text())['calibration']
    if {s.split('_')[0] for s in sequences} & {s.split('_')[0] for s in calibration}:
        raise ValueError('Calibration/test participant overlap')
    geometry = ManoGeometry(ROOT).to(args.device).eval()
    # Lock the reference label definition before evaluating any candidate.
    protocol = dict(reference_type='GT-geometry-derived pinch proxy; not human annotation or software ground truth',
        distance='||thumb_tip4-index_tip8|| / ||wrist0-middle_MCP9||',
        reference_threshold=args.reference_threshold, min_frames=args.min_frames, short_event_frames=[3, 10],
        filter='causal EMA of 21 world XYZ joints, reset at missing evidence/SLAM interval/time gap',
        filter_tau_sec=args.filter_tau_sec, filter_alpha='1-exp(-actual_dt/tau)',
        onset_tolerance_sec=args.onset_tolerance_sec, matching='maximum-cardinality one-to-one; minimum absolute onset error; positive interval overlap',
        censored_events='exclude runs without observed preceding OPEN or following OPEN, for both reference and prediction',
        onset='first below-threshold frame, backdated after min_frames confirmation',
        confidence='binary finite-observation mask only; not a learned reliability estimate', quality_threshold=.5,
        calibration_sequences=calibration, test_sequences=sequences, prediction_threshold_candidates=args.prediction_thresholds,
        selection='maximize mean of two methods pooled calibration event F1; ties choose closest to reference threshold',
        missing_prediction='count recoverable reference events as false negatives, not silently remove them',
        baseline_scope='archived HaWoR visual-valid evaluation frames; GT-projected boxes; no missing-gap recovery',
        software_windows='external logs' if args.software_windows else 'synthetic positive/negative controls derived from GT geometry; not real software logs',
        code_sha256=file_hash(Path(__file__)),
        scoring_code_sha256=file_hash(PROJECT/'dual_loop/pinch_evaluation.py'),
        oracle_code_sha256=file_hash(PROJECT/'dual_loop/oracle.py'))
    write_json(args.output / 'protocol_initial.json', protocol)
    grid = {threshold: [[], []] for threshold in args.prediction_thresholds}
    calibration_sources = []
    for sequence in calibration:
        path = args.calibration_root / f'{sequence}.npz'
        pred, gt, times, segments, pv, gv = calibration_sequence(path, geometry, args.device)
        distances, ref, _ = signals(pred, gt, times, segments, pv, gv, args.filter_tau_sec)
        for threshold in grid:
            _, _, scores = evaluate_distances(distances, ref, times, pv, gv, threshold,
                                              args.reference_threshold, args.min_frames, args.onset_tolerance_sec)
            for index in (0, 1):
                grid[threshold][index].append(scores[index])
        calibration_sources.append(dict(sequence=sequence, archive_sha256=file_hash(path), frames=len(times)))
        print(f'[calibration] {sequence}: {len(times)} frames', flush=True)
    candidates = []
    for threshold, methods in grid.items():
        summaries = [aggregate(method) for method in methods]
        values = [r['f1'] for r in summaries]
        if any(v is None for v in values):
            raise ValueError('No calibration reference events for threshold selection')
        candidates.append(dict(threshold=threshold, score=float(np.mean(values)), methods=dict(zip(METHODS, summaries))))
    selected = max(candidates, key=lambda c: (c['score'], -abs(c['threshold'] - args.reference_threshold), -c['threshold']))
    threshold = selected['threshold']
    protocol.update(prediction_threshold=threshold, calibration_selection=selected, calibration_sources=calibration_sources)
    write_json(args.output / 'calibration_grid.json', candidates)
    write_json(args.output / 'protocol_frozen.json', protocol)
    print(f'[frozen] prediction_threshold={threshold}', flush=True)
    external = json.loads(args.software_windows.read_text()) if args.software_windows else None
    results, rows, window_rows, event_rows = [[], []], [], [], []
    coverage, references, sources = [], {}, []
    acceleration = [[], []]
    for sequence in sequences:
        pred, gt, times, segments, pv, gv, provenance = test_sequence(args.video_root, args.baseline_root, sequence, geometry, args.device)
        distances, ref, filtered = signals(pred, gt, times, segments, pv, gv, args.filter_tau_sec)
        refs, predictions, scores = evaluate_distances(distances, ref, times, pv, gv, threshold,
                                                      args.reference_threshold, args.min_frames, args.onset_tolerance_sec)
        sequence_dir = args.output / sequence
        sequence_dir.mkdir()
        references[sequence] = [asdict(e) for e in refs]
        np.savez_compressed(sequence_dir / 'signals.npz', timestamp_sec=times, prediction_valid=pv, reference_valid=gv,
                            reference_distance=ref, raw_distance=distances[0], filtered_distance=distances[1], segment_id=segments)
        cov = dict(sequence=sequence, frames=len(times), reference_valid_frames=int(gv.sum()),
                   predicted_valid_frames=int(pv.sum()), common_frames=int((pv & gv).sum()), reference_events=len(refs),
                   reference_onsets_without_prediction=sum(not pv[event.start] for event in refs))
        coverage.append(cov)
        sources.append(dict(sequence=sequence, **provenance))
        windows = external.get(sequence, []) if external is not None else simulated_windows(
            refs, ref, times, gv, args.reference_threshold, args.onset_tolerance_sec, args.min_frames)
        write_json(sequence_dir / 'expected_windows.json', windows)
        for index, method in enumerate(METHODS):
            accel = acceleration_errors(pred if index == 0 else filtered, gt, times, pv & gv, segments)
            acceleration[index].append(accel)
            results[index].append(scores[index])
            rows.append(dict(sequence=sequence, method=method, **{k:v for k,v in scores[index].items() if not isinstance(v,list)}))
            write_json(sequence_dir / f'{index}_events_and_matches.json', dict(method=method,
                reference_events=[asdict(e) for e in refs], predicted_events=[asdict(e) for e in predictions[index]], **scores[index]))
            for event in predictions[index]:
                event_rows.append(dict(sequence=sequence, method=method, **asdict(event)))
            joints = pred if index == 0 else filtered
            oracle = PinchOracleConfig(threshold, args.min_frames, .5)
            for window in windows:
                if window['expected'] not in ('pass','fail'):
                    raise ValueError('Expected window labels must be pass or fail')
                decision = judge_expected_window(joints, pv, times, window['start'],window['stop'], oracle)
                window_rows.append(dict(sequence=sequence, method=method, **window, decision=decision))
        fig, axes = plt.subplots(2, 1, figsize=(15,6), sharex=True)
        axes[0].plot(times, ref, label='GT geometry proxy', linewidth=.7)
        axes[0].plot(times, distances[0], label='HaWoR', linewidth=.6, alpha=.7)
        axes[0].plot(times, distances[1], label='HaWoR + causal EMA', linewidth=.6)
        axes[0].axhline(threshold, color='black', linestyle='--', label=f'prediction threshold={threshold}')
        axes[0].axhline(args.reference_threshold, color='gray', linestyle=':', label='reference threshold')
        axes[0].set_ylabel('Normalized tip distance'); axes[0].legend(ncol=3)
        for lane, events in enumerate([refs, *predictions]):
            for event in events:
                axes[1].plot([event.onset_sec,event.offset_sec], [lane,lane], linewidth=4)
        axes[1].set_yticks([0,1,2],['GT proxy','HaWoR','HaWoR + EMA']);axes[1].set_xlabel('Time from sequence start (s)')
        fig.suptitle(sequence);fig.tight_layout();fig.savefig(sequence_dir / 'pinch_timeline.png',dpi=150);plt.close(fig)
        print(f'[test] {sequence}: ref={len(refs)}, rawTP={scores[0]["tp"]}, filterTP={scores[1]["tp"]}',flush=True)
    summary = dict(methods=dict(zip(METHODS,[aggregate(r) for r in results])), num_videos=len(sequences),
                   coverage=coverage, reference_type=protocol['reference_type'], protocol_sha256=file_hash(args.output/'protocol_frozen.json'),
                   source_files=sources, software_windows_source=protocol['software_windows'])
    for method in METHODS:
        selected_rows = [r for r in window_rows if r['method']==method]
        summary['methods'][method]['window_decisions'] = {
            label: {decision:sum(r['expected']==label and r['decision']==decision for r in selected_rows)
                    for decision in ('pass','fail','unknown')} for label in ('pass','fail')}
    for index, method in enumerate(METHODS):
        errors = np.concatenate(acceleration[index])
        summary['methods'][method]['auxiliary_accel_error_m_s2'] = float(errors.mean()) if len(errors) else None
        summary['methods'][method]['auxiliary_accel_samples'] = int(len(errors))
    write_json(args.output/'summary.json',summary)
    write_json(args.output/'reference_events.json',references)
    for name,data in [('per_video.csv',rows),('predicted_events.csv',event_rows),('window_decisions.csv',window_rows),
                      ('results.csv',[dict(method=k,**{a:b for a,b in v.items() if not isinstance(b,dict)}) for k,v in summary['methods'].items()])]:
        if data:
            fields=list(dict.fromkeys(k for row in data for k in row))
            with (args.output/name).open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(data)
    print(json.dumps(summary['methods'],indent=2),flush=True)


if __name__=='__main__':
    main()
