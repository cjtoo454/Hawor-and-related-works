# Intra-frame Feedback ablation

This directory implements the second row of the B ablation: a separately
trained HaWoR visual adapter performs two same-frame observation rounds while
the first-row motion-pretrained TWM and HaWoR checkpoint stay fixed.

The runtime accepts a visual checkpoint with this schema:

~~~python
{
    "architecture": "handloop_intra_v1",
    "format": "handloop_intra_training",
    "format_version": 1,
    "visual": observer.visual.state_dict(),
    "config": original_handloop_v2_config,
    "trainer": {"global_step": positive_integer, ...},
    "base_twm_sha256": sha256_of_motion_pretrain_checkpoint,
    "base_hawor_sha256": sha256_of_hawor_checkpoint,
}
~~~

The visual state must be trained (trainer.global_step > 0). Inference rejects
either base checkpoint when its file hash differs from the one recorded during
visual training. TWM and HaWoR weights are loaded only from their explicit base
checkpoint paths. The observer configuration is fixed to two rounds with
re-observation enabled.

Each current frame starts from HaWoR's real IAM patches. The observer projects
each updated candidate into that same frame's crop before reading the next
round. Every TWM call receives the original raw HaWoR frame and a causal history
of earlier raw world states. Restored states are not committed to history;
history dispersion is zeroed and marked unknown.

Run the full frozen ten-video comparison with the repository Python
environment:

~~~bash
.env/hawor/bin/python thirdparty/ablation/intra_frame/eval_pipeline.py \
  --visual_checkpoint <trained-intra-visual.pt> \
  --twm_checkpoint thirdparty/ablation/training_v2/checkpoints/latest.pt \
  --hawor_checkpoint weights/hawor/checkpoints/hawor.ckpt
~~~

The default pipeline uses the frozen HOT3D test list and pinch protocol, 1000
frame SLAM intervals, the archived TWM-only row at
thirdparty/ablation/twm_only/runs/restore_full_evaluation_20261020/predictions_and_wmpjpe,
and its GT-projected boxes. It verifies track hashes and compares each live
HaWoR raw_state_world and betas array against the corresponding TWM-only trace
before saving predictions. It also requires exact agreement with the first
row's per-video W-MPJPE frame_ids and the shared 34,504-frame total.
The output directory contains try_predictions.npz, intra_frame_trace.npz,
try_inference.json, W-MPJPE details, frozen-protocol event summaries,
source manifests, recorded commands, and a table row only after full coverage.

For model-free checkpoint and data preflight, use the inference entry point:

~~~bash
.env/hawor/bin/python thirdparty/ablation/intra_frame/run_intra.py \
  --visual_checkpoint <trained-intra-visual.pt> \
  --twm_checkpoint thirdparty/ablation/training_v2/checkpoints/latest.pt \
  --preflight
~~~
