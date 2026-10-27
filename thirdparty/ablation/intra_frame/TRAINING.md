# Intra-frame visual training

This trainer implements the second row of experiment B. It starts with the
byte-identified 5000-step motion-pretrained TWM and the existing HaWoR
checkpoint, freezes both, and optimizes only `HandLoopObserver.visual`.

Each training item is one current frame. HaWoR runs live on that frame's
inference-aligned 16-image context under `no_grad`; the observer then performs
two rounds on the same IAM patch features and reprojects the updated candidate
before the second read. The current frame's preceding history contains only
prepared raw HaWoR states from the same SLAM segment, with unknown dispersion.
GT states enter only the final-state MANO joint and current-NLL loss. No IAM
feature cache is written.

The input split is checked against both `split_manifest.json` files and the
frozen experiment protocol. Training samples every GT-valid current frame in
the train split; raw visual validity stays as an observation mask, and missing
raw states do not add labels to history. Calibration uses one fixed middle
frame per video from `gt_valid & visual_valid` (20 samples total), matching the
frames scored by the held-out evaluator. Calibration is the only basis for
selecting `best.pt`. The held-out sequence list is checked for participant
isolation; training never loads held-out frames or labels.

The default settings are 5000 optimizer steps, seed 7, batch size 4, and visual
learning rate `1e-4`. The loss is final-output joint MPJPE divided by `0.05 m`
plus `1e-4` times the current-state NLL. The frozen-backbone raw-wrist term is
not used. `latest.pt` stores optimizer and RNG state for resuming; `best.pt` is
updated only when fixed calibration loss improves. Both follow the
`handloop_intra_v1` checkpoint schema consumed by `run_intra.py`.

Check prepared inputs and frozen protocol hashes:

```bash
.env/hawor/bin/python3.10 thirdparty/ablation/intra_frame/prepare_training.py --check-only
```

Run the one-step integration smoke on GPU 1, exposed as `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 .env/hawor/bin/python3.10 thirdparty/ablation/intra_frame/train_visual.py \
  --steps 1 --batch-size 4 --calibration-every 1 --save-every 1 \
  --device cuda:0 \
  --output-dir thirdparty/ablation/intra_frame/runs/smoke_gpu1
```

Run the full training after the smoke result is reviewed:

```bash
CUDA_VISIBLE_DEVICES=1 .env/hawor/bin/python3.10 thirdparty/ablation/intra_frame/train_visual.py \
  --steps 5000 --batch-size 4 --device cuda:0 \
  --output-dir thirdparty/ablation/intra_frame/runs/visual_train_gpu1
```

The trainer writes JSONL metrics, calibration samples, frozen in-memory
parameter hashes before and after the final step, and the independent visual
checkpoints under the selected output directory. It rejects non-finite losses
or gradients before calling `optimizer.step()`.
