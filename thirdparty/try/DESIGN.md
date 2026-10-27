# Training and experimental contract

`../../method.md` defines the observer. `../../experiment-design.md` defines the
four experiments. This implementation distinguishes learning a trajectory prior
from training and evaluating the complete visual observer.

## Data and separation

The downloaded `datasets/hot3d_train100/train.json` is the only raw-training input
list. A sequence with an empty MANO trajectory is recorded as skipped, never
replaced with zeros. Only right-hand MANO trajectories are used. The 15 PCA
coefficients are expanded with the locally licensed right-hand MANO components
and mean; wrist poses and shape are retained. Source timestamps are resampled to
30 Hz, keeping nearest samples within half a frame and explicit invalid ticks.
Training windows cannot cross a missing state or a segment boundary.

Train and calibration participants are disjoint. Both must be disjoint from the
participants in `datasets/hot3d_valset_export/val.json`. Calibration participants
and the seed are written before optimization. Calibration data select checkpoints
and reliability thresholds; they are not an untouched final test set. Test data
are never used to select epochs or thresholds.

## Stage 1: trajectory prior

Train `TrajectoryWorldModel` on real sensor-state histories, with one-step and
multi-step autoregressive supervision and an uncertainty loss. The exact same
forward function, first-history-frame local gauge, and finite rolling history
are used during training and inference. A checkpoint has
`stage=world_model_pretrain`; inference explicitly refuses to treat it as a full
observer checkpoint. No images, artificial visual observations, or software
logs enter this stage.

The default `legacy` variant retains the original delta predictor and accepts
its existing checkpoints. The opt-in `cv_residual` variant predicts a bounded,
scaled residual over a constant-velocity step and starts with a zero residual.
Its direct translation regression is independent of predicted variance; small
detached-mean NLL trains uncertainty, while rotation and shape terms are
balanced and supervised at every rollout step. Multi-step CV residual rollouts
reuse the same recursive projection and finite-history update as inference.
Evaluation keeps the original anchored `constant_velocity` row and also reports
`recursive_constant_velocity` for that deployed projection path. Since
translation is linear, zero-initialized `cv_residual` matches the original CV
translation baseline; rotation projections can differ after repeated SO(3)
projection. The CV residual run records this initialization as epoch 0 and
retains it as `best.pt` unless calibration improves its horizon-10 translation
error.

Experiment 2 compares last state, constant velocity, and the learned prior at
1/5/10 steps. Report translation and articulated-state errors separately. A
fast-motion cutoff must be fixed using training data or calibration and then
frozen for the held-out test. Calibration results are labeled as calibration,
not reported as final test performance.

## Stage 2: full observer

Generate independent video detections, HaWoR camera-frame states, and visual
SLAM trajectories for the same training sequences. Raw preview MP4s and sensor
ZIP files alone are not these inputs; their timestamp and camera projection
mapping must be established before image supervision.

`prepare_training_data.py` transforms sensor labels into the same SLAM coordinate
frame as visual states. It does not align visual inputs to sensor truth. GT
boxes require an explicit oracle-assisted option, are labeled in provenance,
and are rejected by the full training script. The default visual input contract
is `preprocess/preprocess_tracks.npy`, generated from a video detector rather
than GT projection. Software event logs are never observer inputs.

Each training window begins with an observed visual state. Subsequent missing
GT masks only the loss; missing visual states use model history. The corrector
reprojects each candidate into the current image and samples its feature map.
Final confidence is constrained by final projected visibility, and receives a
supervised target based on actual joint error. Blur hints are computed the same
way in training and inference. Confidence calibration is required before using
it for software pass/fail decisions.

Both loops use the same checkpoint, visual frontend and two correction steps
for the 2x2 experiment. Disable only re-observation or corrected-state feedback.
Do not compare different random heads or independently tuned test thresholds.

## Trace and evaluation

Trace timestamps come from the exported device timestamps, with explicit time
origin and segment identifiers. Independent SLAM gauges are not a globally
registered coordinate system; absolute positions cannot be compared across
segments/runs without a declared external registration. Relative pinch distances
remain invariant to rigid frame changes.

The geometric evaluator follows the shared baseline protocol and explicitly
reports coverage. Its visual-valid-frame metrics exclude prior-only gap states;
they cannot alone demonstrate occlusion recovery. Use a common frame mask for
fair baseline tables, plus separately report missing-state coverage and recovery
on trace states. Never call a GT-assisted controlled benchmark a video-only
end-to-end evaluation.

The pinch oracle requires the event onset inside the expected timestamp window,
followed by m reliable frames. Evidence gaps that could hide a qualifying event
return unknown. Pinch threshold, m and confidence threshold must be fixed on
calibration. Software logs are aligned only after trace generation. Existing
format smoke tests and untrained checkpoints are not method results.

## Frontend provenance manifest

A new HaWoR inference producer must write `frontend_provenance.json` in its
per-sequence result directory at generation time, binding the exact detector
tracks and chunk files it used. The consumer never fabricates a clean manifest
for old results. Required schema:

```json
{
  "schema_version": 1,
  "frontend": "HaWoR",
  "track_file": "preprocess/preprocess_tracks.npy",
  "track_sha256": "SHA256_OF_ACTUAL_TRACK_FILE",
  "oracle_assisted": false,
  "chunk_sha256": {"1/0.json": "SHA256_OF_EXACT_PREDICTION_CHUNK"}
}
```

The chunk map must include every consumed chunk. Paths are relative to the
sequence input or frontend result directory respectively. Unknown provenance
requires explicit oracle-assisted diagnostic mode; it is rejected by formal
full-observer training. Renaming a GT-derived tracks file is not an independent
detector or a valid way to regenerate provenance.
