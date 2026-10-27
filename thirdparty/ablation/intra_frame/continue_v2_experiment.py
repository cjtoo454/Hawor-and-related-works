"""Run the authorized v2 experiment after cache extraction and code review.

Model selection uses calibration only. A failed final test never triggers an
automatic checkpoint search or a rewrite of the original failed experiment.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE / "runs/feedback_v2_train"
EVAL = HERE / "runs/full_eval_v2"
READY = HERE / "records/v2_training_ready.json"
STATUS = HERE / "runs/investigation_status_v2.json"
CACHE_PID = 3251962
state = {"objective": "improve W-MPJPE and Onset MAE over frozen TWM-only",
         "training_dir": str(RUN), "evaluation_dir": str(EVAL), "gpu": 1,
         "success_claimed": False}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def status(stage, **details):
    state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    write_json(STATUS, state)


def process_identity(pid):
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().split()
        return parts[21], parts[2]
    except FileNotFoundError:
        return None


def run(command, log_name):
    with (RUN / log_name).open("w") as log:
        subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    identity = process_identity(CACHE_PID)
    status("extracting_features", cache_pid=CACHE_PID)
    while identity is not None:
        current = process_identity(CACHE_PID)
        if current is None or current[0] != identity[0] or current[1] == "Z":
            break
        time.sleep(15)
    manifest = RUN / "cache/manifest.json"
    if not manifest.is_file():
        raise RuntimeError("Feature extraction ended without a complete cache manifest; inspect cache.log")
    status("waiting_for_review", cache_manifest_sha256=sha(manifest))
    while not READY.is_file():
        time.sleep(15)
    ready = json.loads(READY.read_text())
    if ready.get("review_passed") is not True:
        raise RuntimeError("Training implementation did not pass its review")
    for name, expected in ready["source_sha256"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError(f"Reviewed implementation changed: {name}")
    command = [sys.executable, str(HERE / "train_feedback_v2.py"),
               "--device", "cuda:0", "--steps", "1500", "--run-dir", str(RUN),
               "--cache-dir", str(RUN / "cache")]
    status("training", training_command=command)
    run(command, "train.log")
    records = [json.loads(line) for line in (RUN / "training.jsonl").read_text().splitlines() if line]
    if records[-1]["step"] != 1500 or records[-1].get("frozen_weights_unchanged") is not True:
        raise RuntimeError("Training did not complete 1500 steps with frozen bases")
    selection = json.loads((RUN / "selection.json").read_text())
    if selection.get("selection_qualified") is not True:
        status("calibration_not_qualified", selection=selection,
               reason="No calibration-selected checkpoint met W-MPJPE, Onset and event-F1 criteria; no final test run")
        return

    import train_feedback_v2 as trainer
    import run_feedback_v2 as inference
    inference.configure_runtime()
    source = RUN / "qualified_best.pt"
    locked = RUN / "selected_final.pt"
    if locked.exists():
        raise FileExistsError("Refusing to replace a previously test-locked checkpoint")
    expected = sha(source)
    shutil.copyfile(source, locked)
    if sha(locked) != expected:
        raise RuntimeError("Checkpoint changed during locking")
    checkpoint = inference.runtime.twm_only.load_checkpoint_restricted(locked)
    inference.validate_feedback_checkpoint(checkpoint)
    selected = checkpoint["trainer"]
    if selected["cache_manifest_sha256"] != sha(manifest):
        raise RuntimeError("Selected checkpoint used a different calibration cache")
    if not trainer._score_is_qualified(selected["last_calibration"], selected["baseline_calibration"]):
        raise RuntimeError("Selected checkpoint's own calibration evidence is not qualified")
    selection_record = {
        "selection_qualified": True, "visual_checkpoint_sha256": expected,
        "visual_checkpoint": str(locked), "selected_step": selected["global_step"],
        "cache_manifest_sha256": sha(manifest), "bounds": checkpoint["bounds"],
        "calibration": selected["last_calibration"],
        "baseline_calibration": selected["baseline_calibration"],
        "selection_rule": selected["selection_rule"],
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "prior_test_result_observed": True,
    }
    write_json(HERE / "records/v2_final_selection.json", selection_record)
    command = [sys.executable, str(HERE / "eval_feedback_v2.py"),
               "--visual_checkpoint", str(locked),
               "--twm_checkpoint", str(trainer.DEFAULT_TWM),
               "--hawor_checkpoint", str(trainer.DEFAULT_HAWOR),
               "--device", "cuda:0", "--run_dir", str(EVAL)]
    status("evaluation", selected_step=selected["global_step"], evaluation_command=command)
    run(command, "evaluation_driver.log")
    checked = json.loads((EVAL / "status.json").read_text())
    row = json.loads((EVAL / "table_row.json").read_text())
    if checked["status"] != "complete" or row["evaluated_right_hand_frames"] != 34504:
        raise RuntimeError("Final evaluation did not pass frozen coverage checks")
    baseline_path = HERE.parent / "twm_only/runs/restore_full_evaluation_20261020/table_row.json"
    baseline = json.loads(baseline_path.read_text())
    achieved = (row["w_mpjpe_mm"] < baseline["w_mpjpe_mm"]
                and row["onset_mae_ms"] < baseline["onset_mae_ms"])
    event_quality_preserved = row["event_f1"] >= baseline["event_f1"]
    report = {"metrics": row, "baseline": baseline, "both_requested_metrics_improved": achieved,
              "event_f1_preserved": event_quality_preserved, "selected_on_calibration_only": True}
    write_json(RUN / "final_comparison.json", report)
    if not achieved or not event_quality_preserved:
        status("evaluated_target_not_met", **report,
               reason="Preserved measured results; no test-based checkpoint search or automatic table rewrite")
        return
    doc = ROOT / "experiment-design.md"
    with (HERE / "records/experiment_design.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = doc.read_bytes()
        lines = old.splitlines(keepends=True)
        matches = [i for i, line in enumerate(lines) if line.startswith(b"> | + Intra-frame Feedback |")]
        if len(matches) != 1:
            raise RuntimeError("Ambiguous document row; refusing update")
        index = matches[0]
        before = lines[index]
        if before.strip() != b"> | + Intra-frame Feedback | 51.46 mm | 79.58 ms |":
            raise RuntimeError("Second row changed independently; refusing overwrite")
        ending = b"\r\n" if before.endswith(b"\r\n") else b"\n"
        lines[index] = (f"> | + Intra-frame Feedback | {row['w_mpjpe_mm']:.2f} mm | "
                        f"{row['onset_mae_ms']:.2f} ms |").encode() + ending
        temporary = doc.with_name(".experiment-design.v2.tmp")
        temporary.write_bytes(b"".join(lines))
        temporary.chmod(doc.stat().st_mode & 0o777)
        if doc.read_bytes() != old:
            temporary.unlink()
            raise RuntimeError("Document changed concurrently")
        temporary.replace(doc)
        write_json(HERE / "records/v2_experiment_design_update.json", {
            "old_row": before.decode().strip(), "new_row": lines[index].decode().strip(),
            "before_sha256": hashlib.sha256(old).hexdigest(), "after_sha256": sha(doc),
            "result_source": str(EVAL / "table_row.json"), **report})
    status("complete", success_claimed=True, **report)


if __name__ == "__main__":
    # Hold the lock for cache wait, training, checkpoint locking and final test.
    # Refuse duplicates before touching the active run's status or log files.
    with (RUN / ".driver.lock").open("a") as singleton:
        try:
            fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A v2 experiment driver is already running")
        try:
            main()
        except BaseException as error:
            status("failed", error=f"{type(error).__name__}: {error}")
            raise
