"""Run the reviewed v4 training and one calibration-qualified formal evaluation."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE / 'runs/feedback_v4_train'
EVAL = HERE / 'runs/full_eval_v4'
READY = HERE / 'records/v4_training_ready.json'
STATUS = HERE / 'runs/investigation_status_v2.json'
state = dict(protocol_revision=4, objective='both W-MPJPE and Onset strictly better than frozen TWM',
             training_dir=str(RUN), evaluation_dir=str(EVAL), gpu=1, success_claimed=False)


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            d.update(block)
    return d.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def status(stage, **details):
    state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    write_json(STATUS, state)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split()
        return fields[21], fields[2]
    except FileNotFoundError:
        return None


def run(command, name):
    with (RUN / name).open('x') as log:
        subprocess.run(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'},
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-pid', type=int)
    args = parser.parse_args()
    manifest = RUN / 'cache/manifest.json'
    identity = process_identity(args.cache_pid) if args.cache_pid else None
    status('preparing_long_training_cache', cache_pid=args.cache_pid)
    while not manifest.is_file():
        if args.cache_pid:
            current = process_identity(args.cache_pid)
            if identity is None or current is None or current[0] != identity[0] or current[1] == 'Z':
                raise RuntimeError('Training-cache extraction ended before publishing a complete manifest')
        time.sleep(15)
    status('waiting_for_v4_review', cache_manifest_sha256=sha(manifest))
    while not READY.is_file():
        time.sleep(15)
    ready = json.loads(READY.read_text())
    if ready.get('review_passed') is not True or ready.get('protocol_revision') != 4:
        raise RuntimeError('Missing independent v4 review and integration evidence')
    for name, expected in ready['source_sha256'].items():
        if sha(HERE / name) != expected:
            raise RuntimeError('Reviewed source changed: ' + name)
    for name, expected in ready['data_sha256'].items():
        if sha(Path(name)) != expected:
            raise RuntimeError('Reviewed data changed: ' + name)
    command = [sys.executable, str(HERE / 'train_feedback_v4.py'), '--device', 'cuda:0',
               '--steps', '400', '--run-dir', str(RUN), '--cache-dir', str(RUN / 'cache')]
    status('training', training_command=command)
    run(command, 'train.log')
    records = [json.loads(line) for line in (RUN / 'training.jsonl').read_text().splitlines() if line]
    if records[-1]['step'] != 400 or records[-1].get('frozen_weights_unchanged') is not True:
        raise RuntimeError('Training did not complete 400 steps with unchanged frozen bases')
    selection = json.loads((RUN / 'selection.json').read_text())
    if selection.get('selection_qualified') is not True:
        status('calibration_not_qualified', selection=selection,
               reason='No v4 candidate improved both calibration metrics with preserved F1/recall; no automatic fallback test')
        return
    import train_feedback_v4 as trainer
    import train_feedback_v2 as base
    import run_feedback_v4 as inference
    inference.configure_runtime()
    source, locked = RUN / 'qualified_best.pt', RUN / 'selected_final.pt'
    if locked.exists():
        raise FileExistsError('A final test checkpoint is already locked; refusing replacement')
    expected = sha(source)
    shutil.copyfile(source, locked)
    if sha(locked) != expected:
        raise RuntimeError('Checkpoint changed while locking')
    checkpoint = inference.runtime.twm_only.load_checkpoint_restricted(locked)
    inference.validate_feedback_checkpoint(checkpoint)
    evidence = checkpoint['trainer']
    if not trainer._score_is_qualified(evidence['last_calibration'], evidence['baseline_calibration']):
        raise RuntimeError('Selected checkpoint calibration evidence is not qualified')
    if evidence['cache_manifest_sha256'] != sha(manifest):
        raise RuntimeError('Checkpoint cache differs from the reviewed training cache')
    world = HERE / 'runs/world_reference_audit_v2/camera_map_train_calibration_53seq.json'
    events = HERE / 'runs/calibration_events_v3/calibration_manifest.json'
    if evidence['world_reference_manifest_sha256'] != sha(world) or evidence['event_manifest_sha256'] != sha(events):
        raise RuntimeError('World/event sources changed after selection')
    record = dict(protocol_revision=4, selection_qualified=True, visual_checkpoint=str(locked),
                  visual_checkpoint_sha256=expected, selected_step=evidence['global_step'],
                  bounds=checkpoint['bounds'], calibration=evidence['last_calibration'],
                  baseline_calibration=evidence['baseline_calibration'],
                  locked_utc=datetime.now(timezone.utc).isoformat(), prior_test_result_observed=True,
                  **{key: evidence[key] for key in inference.PROTOCOL_HASH_KEYS})
    write_json(HERE / 'records/v4_final_selection.json', record)
    command = [sys.executable, str(HERE / 'eval_feedback_v4.py'), '--visual_checkpoint', str(locked),
               '--twm_checkpoint', str(base.DEFAULT_TWM), '--hawor_checkpoint', str(base.DEFAULT_HAWOR),
               '--device', 'cuda:0', '--run_dir', str(EVAL)]
    status('formal_evaluation', selected_step=evidence['global_step'])
    run(command, 'evaluation_driver.log')
    checked = json.loads((EVAL / 'status.json').read_text())
    row = json.loads((EVAL / 'table_row.json').read_text())
    if checked['status'] != 'complete' or row['evaluated_right_hand_frames'] != 34504:
        raise RuntimeError('Formal evaluation failed its fixed coverage checks')
    baseline = json.loads((HERE.parent / 'twm_only/runs/restore_full_evaluation_20261020/table_row.json').read_text())
    metrics = ('w_mpjpe_mm', 'onset_mae_ms', 'event_f1')
    if any(not math.isfinite(row[key]) for key in metrics):
        raise RuntimeError('Final metrics are not finite')
    achieved = row['w_mpjpe_mm'] < baseline['w_mpjpe_mm'] and row['onset_mae_ms'] < baseline['onset_mae_ms']
    f1_preserved = row['event_f1'] >= baseline['event_f1']
    report = dict(new_result=row, twm_only=baseline, both_requested_metrics_improved=achieved,
                  event_f1_preserved=f1_preserved, selection_qualified=True)
    write_json(RUN / 'final_comparison.json', report)
    if not achieved or not f1_preserved:
        status('evaluated_target_not_met', **report)
        return
    doc = ROOT / 'experiment-design.md'
    with (HERE / 'records/experiment_design.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = doc.read_bytes()
        lines = old.splitlines(keepends=True)
        indices = [i for i, line in enumerate(lines) if line.startswith(b'> | + Intra-frame Feedback |')]
        if len(indices) != 1:
            raise RuntimeError('Ambiguous document row; results preserved without updating document')
        index = indices[0]
        before = lines[index]
        if before.strip() != b'> | + Intra-frame Feedback | 51.46 mm | 79.58 ms |':
            raise RuntimeError('Document row changed independently; refusing overwrite')
        end = b'\r\n' if before.endswith(b'\r\n') else b'\n'
        lines[index] = (f"> | + Intra-frame Feedback | {row['w_mpjpe_mm']:.2f} mm | {row['onset_mae_ms']:.2f} ms |").encode() + end
        tmp = doc.with_name('.experiment-design.v4.tmp')
        tmp.write_bytes(b''.join(lines))
        tmp.chmod(doc.stat().st_mode & 0o777)
        if doc.read_bytes() != old:
            tmp.unlink()
            raise RuntimeError('Document changed concurrently')
        tmp.replace(doc)
        write_json(HERE / 'records/v4_experiment_design_update.json', dict(old_row=before.decode().strip(), new_row=lines[index].decode().strip(), result_source=str(EVAL / 'table_row.json')))
    status('complete', success_claimed=True, **report)


if __name__ == '__main__':
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / '.driver.lock').open('a') as singleton:
        try:
            fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A v4 experiment driver is already active')
        try:
            main()
        except BaseException as error:
            status('failed', error=f'{type(error).__name__}: {error}')
            raise
