#!/usr/bin/env python3
"""Start a detached, reproducible raw-trajectory pretraining run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    root = Path(__file__).resolve().parents[3]
    project = root/'thirdparty/try'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--gpu', default='1', help='Physical GPU exposed to this run')
    parser.add_argument('--data_root', type=Path, default=project/'data/raw_trajectories')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_windows', type=int, default=512)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--variant', choices=['legacy', 'cv_residual'], default='legacy')
    args = parser.parse_args()
    manifest = args.data_root.resolve()/'split_manifest.json'
    if not manifest.is_file():
        parser.error('Prepare raw trajectories before starting training')
    run_dir=args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    command=[sys.executable,'-u',str(project/'scripts/train_world_model.py'),
             '--data_root',str(args.data_root.resolve()),'--output_dir',str(run_dir/'checkpoints'),
             '--epochs',str(args.epochs),'--steps_per_epoch',str(args.steps_per_epoch),
             '--batch_size',str(args.batch_size),'--eval_windows',str(args.eval_windows),
             '--seed',str(args.seed),'--variant',args.variant,'--device','cuda:0']
    files=[p for folder in ['dual_loop','scripts','configs','tests']
           for p in (project/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.yaml'}]
    files += [project/name for name in ['README.md','DESIGN.md','pyproject.toml','requirements.txt']]
    with zipfile.ZipFile(run_dir/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path,path.relative_to(project))
    (run_dir/'split_manifest.json').write_bytes(manifest.read_bytes())
    packages=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (run_dir/'environment.txt').write_text(packages.stdout)
    with (run_dir/'train.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=root,env=environment,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (run_dir/'train.pid').write_text(str(process.pid)+'\n')
    launch={'stage':'world_model_pretrain','variant':args.variant,'pid':process.pid,'physical_gpu':args.gpu,
            'command':command,'source_sha256':hashlib.sha256((run_dir/'source.zip').read_bytes()).hexdigest(),
            'training_log':str(run_dir/'train.log'),'split_manifest':str(manifest)}
    (run_dir/'launch.json').write_text(json.dumps(launch,indent=2))
    print(json.dumps(launch,indent=2))


if __name__=='__main__':main()
