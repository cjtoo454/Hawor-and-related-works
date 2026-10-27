"""One calibration-qualified, pre-test locked v4 model on the frozen test set."""
import json
import sys
from pathlib import Path

import eval_pipeline
from run_feedback_v4 import ARCHITECTURE, METHOD, BOUND_KEYS, PROTOCOL_HASH_KEYS

HERE = Path(__file__).resolve().parent
_original_build_row = eval_pipeline.build_table_row
locked_selection = None


def checked_build_row(sequences, inference, wmpjpe, onset, common_frame_counts):
    if locked_selection is None:
        raise ValueError('Lock calibration selection before scoring the test')
    for meta in inference.values():
        if meta.get('visual_checkpoint_sha256') != locked_selection['visual_checkpoint_sha256']:
            raise ValueError('Inference weights differ from the locked selection')
        if meta.get('architecture') != ARCHITECTURE or meta.get('protocol_revision') != 4:
            raise ValueError('Inference did not use the centered v4 architecture and protocol')
        for key in PROTOCOL_HASH_KEYS:
            if meta.get(key) != locked_selection.get(key):
                raise ValueError('Inference source binding differs from selected checkpoint: ' + key)
        if meta.get('bounds') != locked_selection['bounds']:
            raise ValueError('Inference residual bounds differ from selected checkpoint')
        if meta.get('feedback_implementation_sha256') != eval_pipeline.sha256(HERE / 'feedback_v4.py'):
            raise ValueError('Feedback code changed during evaluation')
    row = _original_build_row(sequences, inference, wmpjpe, onset, common_frame_counts)
    row.update(protocol_revision=4, architecture=ARCHITECTURE, bounds=locked_selection['bounds'],
               selection_qualified=True, prior_test_result_observed=True,
               checkpoint_selection='calibration only; checkpoint locked before test',
               interpretation='development on a previously observed test set; independent held-out replication remains desirable')
    return row


def main():
    global locked_selection
    record = json.loads((HERE / 'records/v4_final_selection.json').read_text())
    if record.get('protocol_revision') != 4 or record.get('selection_qualified') is not True:
        raise ValueError('No protocol-4 checkpoint qualified on calibration')
    checkpoint = Path(sys.argv[sys.argv.index('--visual_checkpoint') + 1]).resolve()
    if str(checkpoint) != record['visual_checkpoint'] or eval_pipeline.sha256(checkpoint) != record['visual_checkpoint_sha256']:
        raise ValueError('Test checkpoint differs from the immutable pre-test selection')
    locked_selection = record
    eval_pipeline.INFER = HERE / 'run_feedback_v4.py'
    eval_pipeline.EXPECTED_METHOD = METHOD
    eval_pipeline.build_table_row = checked_build_row
    eval_pipeline.main()


if __name__ == '__main__':
    main()
