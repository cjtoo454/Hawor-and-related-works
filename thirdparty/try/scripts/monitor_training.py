#!/usr/bin/env python3
"""Read-only terminal dashboard for the HaWoR dual-loop training pipeline."""
import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def tail(path, count=40):
    try:
        with path.open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 32768))
            return handle.read().decode(errors='replace').splitlines()[-count:]
    except OSError:
        return []


def alive(pid):
    try:
        import psutil
        process = psutil.Process(int(pid))
        return process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_STOPPED)
    except (ImportError, ValueError, TypeError, OSError):
        return False
    except Exception:
        return False


def dashboard(run):
    state = read_json(run / 'status.json')
    launch = read_json(run / 'pipeline_launch.json').get('args', {})
    stage = state.get('stage', '尚未启动')
    model_run = Path(state.get('training_run', str(run / 'full_model')))
    records = []
    for line in tail(model_run / 'train.log', 100):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and 'train_loss' in value:
                records.append(value)
        except ValueError:
            pass
    metrics = []
    metrics_path = model_run / 'checkpoints/dual_loop.metrics.jsonl'
    for line in tail(metrics_path, 10):
        try:
            metrics.append(json.loads(line))
        except ValueError:
            pass
    current = records[-1] if records else (metrics[-1] if metrics else {})
    calibration = metrics[-1].get('calibration_loss') if metrics else None
    config = read_json(model_run / 'checkpoints/dual_loop.run.json').get('args', {})
    pid = state.get('training_pid') if stage == 'full_training' else state.get('pid')
    lines = [f'HaWoR + TRY 双环训练监控 | {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}',
             '=' * 76,
             f'阶段: {stage} | 状态: {state.get("status", "未知")} | 进程: '
             + (f'{pid} 活动' if alive(pid) else f'{pid or "无"} 未活动/已暂停'),
             f'计划: {launch.get("epochs", "?")} epochs × {launch.get("steps_per_epoch", "?")} steps/epoch'
             f' | LR: {config.get("learning_rate", "等待训练配置")}']
    if current:
        lines += [f'当前 Epoch: {current.get("epoch", "?")} | Step: {current.get("step", "本轮结束")}',
                  f'Train loss（本轮累计平均）: {current["train_loss"]:.6f}',
                  'Calibration loss（最近完成轮）: ' + (f'{calibration:.6f}' if calibration is not None else '等待首轮校准')]
    else:
        lines += ['Train loss: 尚未产生（完整双环优化尚未开始）', 'Calibration loss: 尚未产生']
    if stage == 'export':
        export_root = Path(launch.get('export_root', ROOT / 'datasets/hot3d_train100_export'))
        if not export_root.is_absolute():
            export_root = ROOT / export_root
        completed = list(export_root.glob('*/raw_export_manifest.json'))
        active = sorted(export_root.glob('.*.export-*'))
        lines.append(f'完整导出视频: {len(completed)} | 原始视频: 100（47 个缺少监督标签）')
        for folder in active:
            images = folder / 'extracted_images'
            try:
                count = sum(entry.name.endswith('.jpg') for entry in os.scandir(images))
            except OSError:
                count = 0
            lines.append(f'正在导出: {folder.name.split(".export-")[0].lstrip(".")} | {count} 帧')
    lines.append('-' * 76)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        for row in result.stdout.splitlines():
            gpu, used, total, utilization = [item.strip() for item in row.split(',')]
            lines.append(f'GPU {gpu}: {used}/{total} MiB | 利用率 {utilization}%')
    except (OSError, subprocess.TimeoutExpired, ValueError):
        lines.append('GPU 信息暂不可用')
    lines.append('-' * 76)
    if state.get('error'):
        lines.append('错误: ' + str(state['error']))
    stage_logs = sorted(run.glob(f'{stage}*.log'))
    if stage == 'align_training_labels':
        stage_logs = [run / 'prepare.log']
    if stage == 'full_training':
        stage_logs = [model_run / 'train.log']
    for path in stage_logs[:2]:
        lines.append(f'最新日志 [{path.name}]:')
        lines.extend(tail(path, 3) or ['  暂无新日志'])
    lines += ['', '每 5 秒刷新；Ctrl+C 仅退出监控，不停止训练。', f'目录: {run}']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path,
                        default=ROOT / 'thirdparty/try/runs/hot3d_train100_full_dualloop')
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        while True:
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(dashboard(args.run_dir.resolve()), flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n监控已退出，训练进程未受影响。')


if __name__ == '__main__':
    main()
