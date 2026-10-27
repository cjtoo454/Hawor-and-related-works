#!/usr/bin/env python3
"""Run raw export, independent HaWoR observations, and full dual-loop training."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'thirdparty/try'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--raw_root', type=Path, default=ROOT / 'datasets/hot3d_train100')
    parser.add_argument('--export_root', type=Path, default=ROOT / 'datasets/hot3d_train100_export')
    parser.add_argument('--frontend_root', type=Path, default=ROOT / 'eval_log_train100_hawor')
    parser.add_argument('--prepared_root', type=Path, default=PROJECT / 'data/prepared_train100')
    parser.add_argument('--hawor_python', type=Path, default=ROOT / '.env/hawor/bin/python')
    parser.add_argument('--training_python', type=Path, default=ROOT / '.env/try/bin/python')
    parser.add_argument('--frontend_gpus', nargs='+', default=['1', '3'])
    parser.add_argument('--training_gpu', default='1')
    parser.add_argument('--prior_run', type=Path)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=300)
    parser.add_argument('--validation_steps', type=int, default=30)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start_stage', choices=('export', 'slam', 'align_training_labels'), default='export',
                        help='Resume after completed preprocessing stages')
    parser.add_argument('--foreground', action='store_true')
    return parser.parse_args()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_running(pid, command_fragment):
    import psutil
    try:
        process = psutil.Process(pid)
        return (process.status() != psutil.STATUS_ZOMBIE
                and command_fragment in ' '.join(process.cmdline()))
    except psutil.NoSuchProcess:
        return False


def execute(args, name, command, gpu=None):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PATH=str(args.hawor_python.parent) + os.pathsep + env.get('PATH', ''))
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu
    write_json(args.run_dir / f'{name}.command.json', {'command': command, 'physical_gpu': gpu})
    with (args.run_dir / f'{name}.log').open('a', buffering=1) as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def worker(args):
    previous_status_path = args.run_dir / 'status.json'
    previous_status = json.loads(previous_status_path.read_text()) if previous_status_path.exists() else {}
    status = {'status': 'running', 'pid': os.getpid(), 'stage': 'export', 'started_at': time.time()}
    for key in ('training_run', 'training_pid'):
        if key in previous_status:
            status[key] = previous_status[key]
    status_path = args.run_dir / 'status.json'
    def stage(name):
        status.update(stage=name, updated_at=time.time())
        write_json(status_path, status)
        print(json.dumps(status), flush=True)
    try:
        stage('export' if args.start_stage == 'export' else 'verify_prepared_inputs')
        if args.start_stage == 'export':
            execute(args, 'export', [str(args.hawor_python), '-u',
                str(PROJECT / 'scripts/export_raw_training.py'), '--data_root', str(args.raw_root),
                '--output_root', str(args.export_root), '--preview_rotation', 'clockwise90',
                '--rotation_evidence', str(PROJECT / 'data/preview_pixel_verification.json'), '--resume'])
        provenance = json.loads((args.export_root / 'export_provenance.json').read_text())
        if provenance.get('diagnostic') or not provenance.get('formal_training_eligible'):
            raise ValueError('Raw export is not eligible for formal full-model training')
        split = json.loads((args.export_root / 'training_split.json').read_text())
        sequences = split['train'] + split['calibration']
        if not sequences or len(sequences) != len(set(sequences)):
            raise ValueError('Invalid exported train/calibration split')
        status.update(selected_raw_videos=len(json.loads((args.raw_root / 'train.json').read_text())),
                      supervised_videos=len(sequences), train_videos=len(split['train']),
                      calibration_videos=len(split['calibration']))
        frontend_stages = {
            'export': ('detect', 'hawor', 'slam'),
            'slam': ('slam',),
            'align_training_labels': (),
        }[args.start_stage]
        for frontend_stage in frontend_stages:
            stage(frontend_stage)
            jobs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.frontend_gpus)) as pool:
                for group, gpu in enumerate(args.frontend_gpus):
                    selected = sequences[group::len(args.frontend_gpus)]
                    if not selected:
                        continue
                    command = [str(args.hawor_python), '-u', str(PROJECT / 'scripts/prepare_hawor_frontend.py'),
                               '--video_root', str(args.export_root), '--frontend_root', str(args.frontend_root),
                               '--stage', frontend_stage, '--device', 'cuda:0']
                    for sequence in selected:
                        command.extend(['--video', sequence])
                    jobs.append(pool.submit(execute, args, f'{frontend_stage}_gpu{gpu}', command, gpu))
                errors = []
                for future in concurrent.futures.as_completed(jobs):
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(str(error))
                if errors:
                    raise RuntimeError('; '.join(errors))
        stage('align_training_labels')
        execute(args, 'prepare', [str(args.training_python), '-u',
                str(PROJECT / 'scripts/prepare_training_data.py'), '--video_root', str(args.export_root),
                '--set_file', 'train.json', '--frontend_log_root', str(args.frontend_root),
                '--output_root', str(args.prepared_root), '--device', 'cuda'], args.training_gpu)
        prior = None
        if args.prior_run:
            stage('wait_for_prior')
            launch = json.loads((args.prior_run / 'launch.json').read_text())
            prior = args.prior_run / 'checkpoints/best.pt'
            metrics = args.prior_run / 'checkpoints/metrics.jsonl'
            prior_config = json.loads((args.prior_run / 'checkpoints/run.json').read_text())
            prior_epochs = int(prior_config['epochs'])
            while process_running(launch['pid'], str(args.prior_run.resolve())):
                rows = metrics.read_text().splitlines() if metrics.exists() else []
                if rows and json.loads(rows[-1])['epoch'] >= prior_epochs:
                    break
                time.sleep(10)
            if not metrics.exists() or json.loads(metrics.read_text().splitlines()[-1])['epoch'] < prior_epochs:
                raise RuntimeError('Prior training stopped before its configured final epoch')
        stage('full_training')
        training_dir = args.run_dir / 'full_model'
        resume_checkpoint = None
        if training_dir.exists():
            previous_dir = Path(previous_status.get('training_run', str(training_dir)))
            resume_checkpoint = previous_dir / 'checkpoints/dual_loop.pt'
            if not args.resume or not resume_checkpoint.is_file():
                raise FileExistsError(f'Existing full-model run requires a valid resume checkpoint: {previous_dir}')
            training_dir = args.run_dir / time.strftime('full_model_resume_%Y%m%d_%H%M%S')
        command = [str(args.training_python), str(PROJECT / 'scripts/start_full_training.py'),
                   '--run_dir', str(training_dir), '--data_root', str(args.prepared_root),
                   '--split_file', str(args.export_root / 'training_split.json'),
                   '--gpu', args.training_gpu, '--epochs', str(args.epochs),
                   '--steps_per_epoch', str(args.steps_per_epoch),
                   '--validation_steps', str(args.validation_steps)]
        if resume_checkpoint:
            command.extend(['--resume', str(resume_checkpoint)])
        if prior:
            command.extend(['--world_checkpoint', str(prior)])
        execute(args, 'launch_full', command)
        launch = json.loads((training_dir / 'launch.json').read_text())
        status.update(status='training', training_pid=launch['pid'], training_run=str(training_dir))
        stage('full_training')
        metrics = training_dir / 'checkpoints/dual_loop.metrics.jsonl'
        while process_running(launch['pid'], str(training_dir)):
            if metrics.exists():
                lines = metrics.read_text().splitlines()
                if lines:
                    status['last_epoch_metrics'] = json.loads(lines[-1])
                    stage('full_training')
            time.sleep(30)
        if not metrics.exists() or json.loads(metrics.read_text().splitlines()[-1])['epoch'] < args.epochs:
            raise RuntimeError(f'Full training stopped early; inspect {training_dir / "train.log"}')
        status.update(status='complete', completed_at=time.time())
        stage('complete')
    except Exception as error:
        status.update(status='failed', error=str(error))
        stage(status['stage'])
        raise


def main():
    args = parse_args()
    for key in ('run_dir', 'raw_root', 'export_root', 'frontend_root', 'prepared_root',
                'hawor_python', 'training_python', 'prior_run'):
        value = getattr(args, key)
        if value is not None:
            # A venv's python is often a symlink to the base interpreter.
            # Resolving it discards the venv's site-packages and installed deps.
            setattr(args, key, value.absolute() if key.endswith('_python') else value.resolve())
    if args.foreground:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        worker(args)
        return
    if args.run_dir.exists() and not args.resume:
        raise FileExistsError('Choose a new run directory or --resume')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    state = args.run_dir / 'status.json'
    if args.resume and state.exists():
        old = json.loads(state.read_text())
        if old.get('status') in ('running', 'training'):
            pid = old.get('training_pid', old.get('pid'))
            if pid and process_running(pid, str(args.run_dir)):
                raise RuntimeError(f'This pipeline already has a live process: {pid}')
    snapshot = args.run_dir / 'pipeline_source.zip'
    if not snapshot.exists():
        with zipfile.ZipFile(snapshot, 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(PROJECT.rglob('*')):
                if (path.is_file() and path.suffix in ('.py', '.yaml', '.md')
                        and path.relative_to(PROJECT).parts[0] in ('scripts', 'dual_loop', 'configs', 'tests')):
                    archive.write(path, path.relative_to(PROJECT))
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--foreground']
    with (args.run_dir / 'pipeline.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(args.run_dir / 'pipeline_launch.json', {'pid': process.pid, 'command': command,
               'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}})
    print(json.dumps({'pipeline_pid': process.pid, 'run_dir': str(args.run_dir)}))


if __name__ == '__main__':
    main()
