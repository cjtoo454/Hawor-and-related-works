"""User-requested formal test of a fixed, calibration-unqualified v3 checkpoint."""
import json
import sys
from pathlib import Path

import eval_feedback_v3 as checked
import run_feedback_v3 as inference

HERE = Path(__file__).resolve().parent
RECORD = HERE / 'records/v3_requested_test_selection.json'


def build_row(*args, **kwargs):
    row = checked.checked_build_row(*args, **kwargs)
    row.update(selection_qualified=False,
               test_authorization='Explicit user request to test the newly trained model',
               checkpoint_selection='best_composite.pt selected on calibration only before this test',
               selected_step=checked.locked_selection['selected_step'])
    return row


def main():
    record = json.loads(RECORD.read_text())
    if record.get('protocol_revision') != 3 or record.get('user_requested_test') is not True:
        raise ValueError('Missing explicit user-requested test record')
    if record.get('selection_qualified') is not False:
        raise ValueError('This experiment must preserve the failed calibration qualification')
    checkpoint = Path(sys.argv[sys.argv.index('--visual_checkpoint') + 1]).resolve()
    if str(checkpoint) != record['visual_checkpoint'] or checked.eval_pipeline.sha256(checkpoint) != record['visual_checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the pre-test locked selection')
    inference.configure_runtime()
    payload = inference.runtime.twm_only.load_checkpoint_restricted(checkpoint)
    trainer = inference.validate_protocol(payload)
    for key in inference.PROTOCOL_HASH_KEYS:
        if trainer[key] != record[key]:
            raise ValueError('Checkpoint source binding changed: ' + key)
    checked.locked_selection = record
    checked.eval_pipeline.INFER = HERE / 'run_feedback_v3.py'
    checked.eval_pipeline.EXPECTED_METHOD = inference.METHOD
    checked.eval_pipeline.build_table_row = build_row
    checked.eval_pipeline.main()
    run_dir = Path(sys.argv[sys.argv.index('--run_dir') + 1])
    row = json.loads((run_dir / 'table_row.json').read_text())
    baseline = json.loads((HERE.parent / 'twm_only/runs/restore_full_evaluation_20261020/table_row.json').read_text())
    comparison = {'new_result': row, 'twm_only': baseline,
                  'w_mpjpe_improved': row['w_mpjpe_mm'] < baseline['w_mpjpe_mm'],
                  'onset_mae_improved': row['onset_mae_ms'] < baseline['onset_mae_ms'],
                  'event_f1_preserved': row['event_f1'] >= baseline['event_f1'],
                  'selection_qualified': False, 'user_requested_test': True}
    (run_dir / 'comparison_with_twm.json').write_text(json.dumps(comparison, indent=2) + '\n')


if __name__ == '__main__':
    main()
