#!/usr/bin/env python3
"""Append live filesystem export progress without restarting the exporter."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def snapshot(export_root, expected):
    completed = {}
    for path in export_root.glob('*/raw_export_manifest.json'):
        data = read_json(path)
        if data.get('status') == 'exported' and path.parent.name in expected:
            completed[path.parent.name] = int(data.get('frame_count', 0))
    active = []
    for folder in sorted(export_root.glob('.*.export-*')):
        name = folder.name[1:].split('.export-', 1)[0]
        if name not in expected or name in completed:
            continue
        try:
            frames = sum(item.name.endswith('.jpg') for item in os.scandir(folder / 'extracted_images'))
        except OSError:
            continue
        active.append((name, frames))
    frames = sum(completed.values()) + sum(count for _, count in active)
    return completed, active, frames


def append(path, message):
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    line = f'[export_progress] {stamp} {message}\n'
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, line.encode())
    finally:
        os.close(descriptor)
    print(line, end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    run = args.run_dir.resolve()
    launch = read_json(run / 'pipeline_launch.json').get('args', {})
    raw_root = resolve(launch.get('raw_root', 'datasets/hot3d_train100'))
    export_root = resolve(launch.get('export_root', 'datasets/hot3d_train100_export'))
    split = read_json(ROOT / 'thirdparty/try/data/raw_trajectories/split_manifest.json')
    names = split.get('train', []) + split.get('calibration', [])
    if not names:
        raise ValueError('Missing supervised-sequence split')
    import av
    expected = {}
    for name in names:
        paths = list((raw_root / name).glob('*preview_rgb.mp4'))
        if len(paths) != 1:
            raise ValueError(f'Missing or ambiguous preview: {name}')
        with av.open(str(paths[0])) as container:
            expected[name] = int(container.streams.video[0].frames)
    total_frames = sum(expected.values())
    with (run / 'export_progress.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Export progress watcher already running')
        if not args.once:
            (run / 'export_progress.pid').write_text(str(os.getpid()) + '\n')
        while True:
            state = read_json(run / 'status.json')
            if state.get('stage') != 'export':
                append(run / 'export.log', f'monitor_finished stage={state.get("stage")} status={state.get("status")}')
                break
            completed, active, frames = snapshot(export_root, expected)
            elapsed = max(1, time.time() - state.get('started_at', time.time()))
            rate = frames / elapsed
            remaining = max(0, total_frames - frames)
            eta = f'{remaining / rate / 3600:.2f}h' if rate > 0 else 'unknown'
            details = ','.join(f'{name}:{count}/{expected[name]}' for name, count in active) or 'finalizing/scanning'
            message = (f'completed={len(completed)}/{len(expected)} supervised_videos '
                       f'frames={frames}/{total_frames} ({100 * frames / total_frames:.2f}%) '
                       f'active={details} avg_fps={rate:.2f} export_eta={eta} '
                       'eta_scope=export_only frame_count=filesystem_estimate')
            append(run / 'export.log', message)
            if args.once:
                break
            import psutil
            try:
                process = psutil.Process(state['pid'])
                if process.status() == psutil.STATUS_ZOMBIE:
                    append(run / 'export.log', 'monitor_finished pipeline_process_exited')
                    break
            except (psutil.NoSuchProcess, KeyError):
                append(run / 'export.log', 'monitor_finished pipeline_process_exited')
                break
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
