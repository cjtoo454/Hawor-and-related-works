# TRY: HandLoop v2 and legacy Dual-Loop

The new `handloop_v2` implementation follows
[`method（1）_公式修复.md`](../../method（1）_公式修复.md). It is a new
architecture; the trained legacy `dual_loop.best.pt` and prior checkpoints
cannot initialize it. Existing runs, legacy scripts and results are retained.

## HandLoop v2

The numerical state contains the **anatomical wrist** and 16 rotations (99D),
with shape outside the dynamic state. Both rounds share a detached wrist
anchor. Each round reads real HaWoR IAM patches, constructs a numerical
candidate, then restores it with the numerical TWM. Completed states are
committed once with detached gradients. TWM inputs exclude image features and
history uncertainty; historical dispersion goes only to the visual reader.

The TWM implements joint spatial attention, causal selective Mamba, dense
experts, current Gaussian/matrix-Fisher density, and a shared-mode distribution
over future trajectories. The selective scan is a pure PyTorch reference
implementation; it does not require `mamba_ssm` or silently replace Mamba with
a GRU. Benchmark speed and memory before choosing a production training batch.

Visual adaptation uses live, differentiable HaWoR, including its original
16-frame **bidirectional** context. This is not a strictly online video model.
The numerical feedback itself uses only completed past states. Crop projection
matches the actual 256px crop and its 32px horizontal removal for the 16x12 IAM
grid. The repository's geometry/data adapter currently handles right hands;
left-hand dataset adapters must explicitly supply a verified mirror/geometry
mapping rather than relabel right-hand states.

Use `.env/hawor/bin/python` for live visual training/inference; `.env/try`
does not include all HaWoR dependencies. Numerical pretraining can use either
environment. Prepare the existing local HOT3D sources without modifying them:

```bash
.env/hawor/bin/python thirdparty/try/scripts/prepare_handloop_data.py \
  --split_file datasets/hot3d_train100_export/training_split.json \
  --mode both

.env/try/bin/python thirdparty/try/scripts/train_handloop.py --help
.env/hawor/bin/python thirdparty/try/scripts/train_handloop.py --help
```

`configs/handloop_v2.yaml` records the document's architecture. The new trainer
has motion-pretraining and joint-visual stages; see its help for manifest,
checkpoint, resume and batch parameters. Motion replay and visual updates use
the same TWM. A full joint checkpoint is required for inference:

```bash
.env/hawor/bin/python thirdparty/try/scripts/run_handloop.py \
  --checkpoint PATH_TO_HANDLOOP_V2_JOINT_CHECKPOINT \
  --video_root datasets/hot3d_valset_export \
  --output_root eval_log_handloop_v2

.env/try/bin/python thirdparty/try/scripts/eval_hot3d.py \
  --prediction_root eval_log_handloop_v2 --eval_log_dir eval_log_handloop_v2
```

GT-projected validation boxes require the explicit `--allow_gt_boxes` option
and are marked oracle-assisted. Inference does not open hand annotations or
GT cameras. `try_predictions.npz` preserves the shared evaluator format;
`handloop_trace.npz` separately stores actual-wrist state and dispersion.
Dispersion is not a calibrated probability of correctness. Evaluation retains
the existing visual-valid-frame scope; it does not score prior-only gaps.

### Versioned data contract

Manifests have `schema: handloop_v2` and `train`, `calibration`, optional `test`
entry lists. Each entry supplies `path`, `participant`, `source`, `kind`
(`motion`/`visual`). Participants must be disjoint within each source across
splits. The converter also excludes the held-out HOT3D participants. New
archives are written to a separate directory and existing archives are not
overwritten.

Motion NPZ fields: `state[N,99]`, `timestamp_sec[N]`, `available[N,17]`,
`handedness[N]`, `segment_id[N]`. Positions are meters, timestamps seconds,
root rotations share the sequence frame, finger rotations are parent-local.
Only actual-wrist positions are accepted: HOT3D's raw `wrist_xform.t_xyz`
is used as MANO translation by its own FK adapter, so both raw and legacy
visual archives are decoded to the actual joint0 before numerical training.

Visual NPZ fields additionally include RGB `image_paths`, independent `bbox`,
`image_size[2]`, `focal_length[N]`, `rotation_w2c`, `translation_w2c`,
`raw_available[N,17]`, `image_available[N]`, `target_state[N,99]`,
`target_available[N,17]`, `target_joints[N,21,3]`, `joint_valid[N,21]`, and
`oracle_assisted`. Training regenerates crops and IAM features from RGB; it
does not use cached features in place of joint adaptation. Labels enter only
losses. The HOT3D converter retains verified detector provenance.

OakInk2, UmeTrack and DexYCB can be added only after converting their coordinate,
shape, handedness and joint conventions into this contract with explicit
participant splits. This change does not download those datasets or claim
their conversion has been validated. Dataset-specific partial keypoint maps
must be supplied by those adapters; the bundled visual adapter uses the
existing HOT3D/OpenPose21 ordering.

## Legacy implementation

### Small legacy feedback ablation

`scripts/run_mini_ablation.py` runs the three fixed variants in
`diagnostics/mini_ablation_plan.json` with one legacy epoch-29 checkpoint:
fixed image evidence/raw history, renewed image evidence/raw history, and
renewed evidence/corrected history. It does not retrain or substitute the new
99D HandLoop architecture. All variants use two corrections and fresh,
identical 32-frame warmup before each 900-frame target clip.

```bash
PYTHONPATH=thirdparty/try .env/try/bin/python \
  thirdparty/try/scripts/run_mini_ablation.py --preflight \
  --video_root PATH_TO_HOT3D_VALSET_EXPORT \
  --output_root diagnostics/mini_ablation_preflight

PYTHONPATH=thirdparty/try .env/try/bin/python \
  thirdparty/try/scripts/run_mini_ablation.py \
  --video_root PATH_TO_HOT3D_VALSET_EXPORT --device cuda:1 \
  --output_root diagnostics/mini_ablation_results
```

Preflight reports missing original RGB, camera and SLAM assets before model
inference. Trajectory caches alone cannot run candidate-guided re-observation.
Use separate fresh output directories for preflight and inference. The cached
HaWoR control in `diagnostics/mini_ablation_hawor_control.json` is a baseline
check, not a result for any of the three ablated variants.

### Pinch-event baseline experiment

Run the existing HaWoR trajectories against a simple causal EMA without
retraining or rerunning HaWoR:

```bash
PYTHONPATH=thirdparty/try .env/try/bin/python \
  thirdparty/try/scripts/eval_pinch_events.py --output NEW_EXPERIMENT_DIRECTORY
```

This reads the ten `val.json` sequences and archived `eval_log` predictions.
It uses the independent P0015 calibration sequences to choose one common
prediction threshold, then freezes the protocol before testing. The reference
is a **GT-joint-derived fingertip-proximity proxy**, not human contact labels.
Reference events at prediction gaps remain eligible false negatives. Short
events contain 3–10 frames. Matching is one-to-one within a fixed onset
tolerance; MAE uses matched events only. The causal EMA resets at observation
gaps and SLAM boundaries, and cannot use future samples.

Without `--software_windows`, expected windows are explicitly simulated from
reference events and open negative controls. They do not establish correctness
on real software logs. External windows use JSON
`{sequence: [{id, start, stop, expected}]}`, with seconds relative to the first
video timestamp and `expected` equal to `pass` or `fail`. External clocks must
be independently aligned first. Windows whose confirmation interval crosses
a time gap over 0.25 seconds return `unknown`.

The output includes frozen protocol/calibration selection, pooled and per-video
event metrics, all matches, raw/filtered distance signals, timeline plots and
separate software-window decisions. Existing oracle unit-test thresholds are
not treated as previously validated physical-contact thresholds.

Implementation of [method.md](../../method.md). The data, training, ablation and
evaluation rules are in [DESIGN.md](DESIGN.md). The observer uses a 109D MANO
state, a causal trajectory GRU, two image re-observation corrections, and final
corrected-state feedback. Confidence includes final projected visibility.

## Environment

The project environment `.env/try` reuses the installed CUDA PyTorch environment
and installs video/MANO dependencies locally. From the repository root:

```bash
PYTHON=.env/try/bin/python
$PYTHON thirdparty/try/scripts/prepare_mano_assets.py
```

The asset converter reads the existing licensed local MANO pickle. Generated
assets, prepared data, checkpoints and run logs are ignored by this package's
`.gitignore`; no MANO model is redistributed.

## Train from the downloaded raw data

First prepare real sensor trajectories, excluding empty labels and holding out
calibration participants:

```bash
.env/try/bin/python thirdparty/try/scripts/prepare_raw_trajectories.py \
  --data_root datasets/hot3d_train100
```

Then use `scripts/train_world_model.py` to pretrain the trajectory prior. The
default `--variant legacy` preserves the original model; pass
`--variant cv_residual` to start from constant-velocity predictions and train a
bounded residual. The run records its epoch-0 CV baseline and keeps it as the
best checkpoint until calibration shows an improvement. See `--help` and the
run's saved arguments for the exact settings. This is real sensor-trajectory
training; it does not train the visual corrector. A prior-only checkpoint is
deliberately rejected by `run_hot3d.py`.

The raw training set has no precomputed HaWoR/SLAM observation caches. To train
the full observer, first generate independent video detections, HaWoR states,
and SLAM, preserving the actual timestamp/camera mapping. Do not substitute
sensor truth for visual observations.

```bash
.env/try/bin/python thirdparty/try/scripts/prepare_training_data.py \
  --video_root PATH_TO_EXPORTED_TRAINING \
  --set_file train.json --frontend_log_root eval_log_train

.env/try/bin/python thirdparty/try/scripts/train.py \
  --data_root thirdparty/try/data/prepared \
  --split_file PATH_TO_TRAIN_CALIBRATION_SPLIT_JSON \
  --world_checkpoint PATH_TO_PRIOR_CHECKPOINT \
  --output thirdparty/try/checkpoints/dual_loop.pt
```

The split JSON contains `train` and `calibration` sequence lists. Full training
rejects participant leakage and GT-assisted observations. Checkpoints include
optimizer state, configuration and split provenance; `--resume` restores them.

## Inference and ablation

```bash
.env/try/bin/python thirdparty/try/scripts/run_hot3d.py \
  --checkpoint thirdparty/try/checkpoints/dual_loop.pt --video SEQUENCE
```

The default detector track input is `preprocess/preprocess_tracks.npy`. Legacy
GT-projected tracks require explicit `--allow_gt_boxes` and are labeled as
oracle-assisted. Raw trace coordinates have segment identifiers and are not
claimed to be registered across runs.

All four ablations use the same checkpoint and two corrections. Toggle
`--intra_frame_reobserve` / `--no-intra_frame_reobserve` and
`--corrected_state_feedback` / `--no-corrected_state_feedback`, writing each
variant to a separate `--output_root`. A missing checkpoint requires the explicit
`--allow_untrained` flag for format diagnostics only.

## Evaluation and checks

```bash
.env/try/bin/python thirdparty/try/scripts/eval_hot3d.py \
  --prediction_root eval_log_try --eval_log_dir eval_log_try --video SEQUENCE

PYTHONPATH=thirdparty/try .env/try/bin/python -m pytest -q thirdparty/try/tests
```

The geometric evaluator rejects untrained output by default, reports evaluated
coverage, and accepts `--common_frame_mask` for a shared baseline-frame NPZ.
Its visual-valid-frame metrics do not measure performance during missing visual
frames. `dual_loop.oracle.judge_time_window` provides quality-aware pinch
judgments after external software timestamps have been aligned to the trace.
