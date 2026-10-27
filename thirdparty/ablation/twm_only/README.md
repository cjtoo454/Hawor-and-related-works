# TWM-only HOT3D ablation

This entry point runs one trained HandLoop v2 TWM step for each frame. The input is the original HaWoR 99D state transformed into the SLAM world frame. The same raw state is passed as the current candidate and committed to causal history after the call; the restored state is never fed back. The script reuses `make_anchor`, `to_local`, and `to_world` from `dual_loop.handloop_state` and bypasses the visual correction network.

Both HandLoop v2 checkpoint stages are supported:

- `motion_pretrain`: load the TWM state dictionary directly.
- `joint_visual`: extract `observer.twm.*` for the single TWM and load the matching `hawor.*` frontend weights. This uses the frontend jointly trained with that TWM while bypassing the visual correction network. The separately supplied HaWoR checkpoint initializes the frontend architecture before those weights are applied.

Checkpoint loading uses `weights_only=True` and allows only the NumPy types needed by the training checkpoint's saved RNG state. Legacy checkpoints are rejected. No unrestricted pickle fallback is used.

Each inference record stores the HandLoop stage, positive training step, checkpoint hash, HaWoR initialization checkpoint hash, and effective frontend source. `frontend_source_sha256` hashes the source checkpoint file (the joint checkpoint for `joint_visual`, or the explicit HaWoR checkpoint for `motion_pretrain`); it is not a hash of extracted tensors.

## Requirements and command

Run from the repository root with the HaWoR runtime dependencies installed, `thirdparty/try/requirements.txt` available, the right-hand MANO asset at `_DATA/data/mano/MANO_RIGHT.pkl`, a HaWoR checkpoint, a HandLoop v2 checkpoint, and a prepared HOT3D export containing images, detected tracks, timestamps, and SLAM trajectories.

The single pipeline command runs preflight, TWM-only inference, W-MPJPE, and Onset MAE, then writes one Markdown/JSON table row. With no `--video` argument it selects the frozen protocol's test sequences. All run data and logs stay under `thirdparty/ablation/twm_only/runs/<UTC timestamp>/`.

```bash
.env/hawor/bin/python thirdparty/ablation/twm_only/run_pipeline.py \
  --checkpoint /path/to/handloop_v2.pt \
  --video_root /path/to/hot3d_valset_export
```

If preflight finds missing or invalid resources, the pipeline writes only `blocked_status.json` for that run and stops before inference, metric calculation, or table-row creation. To request a deliberate subset, repeat `--video SEQUENCE` for each sequence.

For individual stages, first check resources without initializing HaWoR or MANO:

```bash
.env/hawor/bin/python thirdparty/ablation/twm_only/run_twm_only.py \
  --checkpoint /path/to/handloop_v2.pt \
  --video_root /path/to/hot3d_valset_export \
  --video P0001_example \
  --preflight
```

The inference output contains `try_predictions.npz`, which the pipeline passes to the existing W-MPJPE evaluator, plus `twm_only_trace.npz` and `try_inference.json` provenance. After W-MPJPE, the pipeline writes `evaluation_source.json` with hashes binding the prediction archive, evaluator details, and inference metadata. The Onset MAE adapter requires this binding and rejects details from another baseline or an older run. It uses the frozen thresholds and shared `extract_events` / `match_events` functions; it does not recalibrate on the test set.

## Frontend scope

Default inference uses detected tracks and marks `oracle_assisted` false when that is the track file actually loaded. The archived frozen pinch comparison was made with GT-projected boxes. `--allow_gt_boxes` permits the repository's legacy GT-projected track file only when detected tracks are absent; detected tracks retain priority. The metadata reports the actual track file and `oracle_assisted` value. The pipeline writes a formal table row only when every sequence confirms the frozen run's GT-box scope, the full frozen test video set was evaluated, and all frames were processed. Otherwise it writes a diagnostic status and does not call the result a reproduction of that baseline.

The existing HOT3D evaluator aligns SLAM camera trajectories for W-MPJPE. Pinch scoring uses its frozen prediction/reference thresholds and existing annotation-validity convention.

## Small fixture checks

```bash
.env/hawor/bin/python -m unittest thirdparty/ablation/twm_only/test_twm_only.py
.env/hawor/bin/python thirdparty/ablation/twm_only/run_twm_only.py --help
.env/hawor/bin/python thirdparty/ablation/twm_only/eval_twm_onset.py --help
.env/hawor/bin/python thirdparty/ablation/twm_only/run_pipeline.py --help
```

The fixture creates actual `save_checkpoint` motion and joint-stage files with a tiny TWM, loads them through the restricted loader, runs a short synthetic sequence, verifies one TWM call per frame and raw-only history commits, checks the zero-length history case, and scores a synthetic onset with the shared matcher. It does not load HaWoR, run MANO inference on HOT3D, or report experiment metrics.
