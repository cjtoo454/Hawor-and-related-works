"""Evaluate one calibration-selected v2 checkpoint with the frozen test protocol."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import eval_pipeline
from run_feedback_v2 import ARCHITECTURE, METHOD, BOUND_KEYS

HERE = Path(__file__).resolve().parent
original_build_row = eval_pipeline.build_table_row
locked_selection = None


def checked_build_row(sequences, inference, wmpjpe, onset, common_frame_counts):
    if locked_selection is None:
        raise ValueError("Calibration selection must be locked before test scoring")
    if any(meta.get("visual_checkpoint_sha256") != locked_selection["visual_checkpoint_sha256"]
           for meta in inference.values()):
        raise ValueError("Actual inference weights differ from the calibration-locked selection")
    if any(meta.get("architecture") != ARCHITECTURE for meta in inference.values()):
        raise ValueError("All videos must actually run the v2 architecture")
    bounds = {tuple(meta.get("bounds", {}).get(k) for k in BOUND_KEYS)
              for meta in inference.values()}
    if len(bounds) != 1 or any(x is None for x in next(iter(bounds))):
        raise ValueError("All videos must use one recorded v2 bounds configuration")
    hashes = {meta.get("feedback_implementation_sha256") for meta in inference.values()}
    if hashes != {eval_pipeline.sha256(HERE / "feedback_v2.py")}:
        raise ValueError("Feedback implementation changed during evaluation")
    row = original_build_row(sequences, inference, wmpjpe, onset, common_frame_counts)
    row.update(architecture=ARCHITECTURE, bounds=dict(zip(BOUND_KEYS, next(iter(bounds)))),
               feedback_implementation_sha256=next(iter(hashes)),
               prior_test_result_observed=True,
               interpretation="development on a previously observed test set; independent replication remains desirable")
    return row


def main():
    global locked_selection
    # A test checkpoint must be locked from calibration evidence before testing.
    record_path = HERE / "records/v2_final_selection.json"
    record = json.loads(record_path.read_text())
    if record.get("selection_qualified") is not True:
        raise ValueError("No checkpoint met the predeclared calibration selection criteria")
    try:
        checkpoint = Path(sys.argv[sys.argv.index("--visual_checkpoint") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("Provide --visual_checkpoint explicitly") from error
    if eval_pipeline.sha256(checkpoint) != record.get("visual_checkpoint_sha256"):
        raise ValueError("Test checkpoint differs from the calibration-locked selection")
    locked_selection = record
    eval_pipeline.INFER = HERE / "run_feedback_v2.py"
    eval_pipeline.EXPECTED_METHOD = METHOD
    eval_pipeline.build_table_row = checked_build_row
    eval_pipeline.main()


if __name__ == "__main__":
    main()
