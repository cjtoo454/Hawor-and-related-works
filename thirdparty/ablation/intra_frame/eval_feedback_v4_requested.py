"""Explicitly requested evaluation of v4, including calibration-unqualified weights."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import eval_feedback_v4 as checked
import eval_pipeline
import run_feedback_v4 as inference


def main():
    checkpoint = Path(sys.argv[sys.argv.index('--visual_checkpoint') + 1]).resolve()
    run_dir = Path(sys.argv[sys.argv.index('--run_dir') + 1]).resolve()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    inference.configure_runtime()
    weights = inference.runtime.twm_only.load_checkpoint_restricted(checkpoint)
    inference.validate_feedback_checkpoint(weights)
    trainer = weights['trainer']
    record = dict(protocol_revision=4, selection_qualified=False,
                  visual_checkpoint=str(checkpoint),
                  visual_checkpoint_sha256=eval_pipeline.sha256(checkpoint),
                  selected_step=trainer['global_step'], bounds=weights['bounds'],
                  test_authorization='Explicit user request: evaluate retrained Intra-frame Feedback using the same standard as TWM only',
                  checkpoint_selection='step 400; lowest calibration W among four candidates; calibration qualification failed',
                  locked_utc=datetime.now(timezone.utc).isoformat(),
                  **{key: trainer[key] for key in inference.PROTOCOL_HASH_KEYS})
    lock_path = run_dir.with_name(run_dir.name + '_selection.json')
    with lock_path.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    checked.locked_selection = record

    def build_row(*args, **kwargs):
        row = checked.checked_build_row(*args, **kwargs)
        row.update(selection_qualified=False,
                   test_authorization=record['test_authorization'],
                   checkpoint_selection=record['checkpoint_selection'],
                   selected_step=record['selected_step'])
        return row

    eval_pipeline.INFER = checked.HERE / 'run_feedback_v4.py'
    eval_pipeline.EXPECTED_METHOD = inference.METHOD
    eval_pipeline.build_table_row = build_row
    eval_pipeline.main()


if __name__ == '__main__':
    main()
