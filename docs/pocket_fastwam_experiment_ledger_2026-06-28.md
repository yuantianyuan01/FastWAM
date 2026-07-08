# Pocket-FastWAM Experiment Ledger

Date: 2026-06-28  
Project: `FastWAM`  
Main related design note: `docs/pocket_fastwam_dim_design_2026-06-25.md`  
Earlier research note: `docs/pocket_fastwam_research_plan_2026-06-22.md`

This ledger preserves the experimental state of the Pocket-FastWAM direction so the result history does not depend on the chat context.

2026-07-08 update: the strongest current Pocket-FastWAM result is now the
Language-Tapped DiM effective-bs128 run recorded in
`docs/pocket_fastwam_languagetap_dim_bs128_record_2026-07-08.md`. It reaches
`98.75%` Overall on full LIBERO at `step_021700`, beating the reproduced
FastWAM `step_043400` baseline by `+2.10` points.

## 1. Executive Conclusion

The current evidence supports three separate conclusions:

1. FastWAM's action side is overparameterized, especially late ActionDiT FFNs.
2. Replacing the MoT-coupled action path with a standalone Pocket-DiM conditioned only on first-frame Wan taps is not yet enough to match FastWAM.
3. The full-video-objective Language-Tapped DiM route can exceed both the reproduced FastWAM baseline and the previously recorded LT-DiM full run when trained with effective batch `128`.

Latest positive result:

| Model | Step | Effective batch | Spatial | Object | Goal | Libero-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| FastWAM baseline | 043400 | 64 in local reproduction | 96.00 | 99.40 | 97.80 | 93.40 | 96.65 |
| Language-Tapped DiM | 021700 | 128 | 99.00 | 99.80 | 98.40 | 97.80 | 98.75 |

The earlier notes below remain useful for historical context but are no longer the strongest positive result.

The strongest positive result before Language-Tapped DiM was the late-FFN screening result:

| Model | Step | ActionDiT params | Spatial | Object | Goal | Libero-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| FastWAM baseline | 022000 | 1,020,886,023 | 95.40 | 99.60 | 96.60 | 88.80 | 95.10 |
| Late-FFN-2048 | 022000 | 957,940,743 | 96.40 | 99.80 | 96.20 | 92.60 | 96.25 |
| Late-FFN-1024 | 022000 | 926,468,103 | 97.20 | 99.60 | 96.00 | 92.60 | 96.35 |

The strongest negative result is the full-data Pocket-DiM rollout:

| Model | Training mode | Spatial | Object | Goal | Libero-10 | Overall |
|---|---|---:|---:|---:|---:|---:|
| Frozen-Wan Pocket-DiM | train only aggregator + DiM | 91.60 | 95.80 | 74.80 | 60.20 | 80.60 |
| Train-Wan Pocket-DiM | action-only loss also updates Wan/MoT-side params | 74.20 | 65.60 | 56.20 | 22.00 | 54.50 |
| FastWAM baseline | original full objective | 96.00 | 99.40 | 97.80 | 93.40 | 96.65 |

The `train-wan` Pocket-DiM run should not be treated as a real FastWAM full-objective run. It trains Wan-related parameters only through a first-frame feature path and a Pocket action loss. It does not train the original video diffusion loss or the full video/action MoT denoising objective.

## 2. Baselines

Source:

```text
docs/fastwam_reproduction_record_2026-06-16.md
docs/pocket_fastwam_dim_design_2026-06-25.md
```

Full LIBERO evaluation baselines:

| Model | Step | Spatial | Object | Goal | Libero-10 | Overall | Result dir |
|---|---:|---:|---:|---:|---:|---:|---|
| FastWAM baseline | 022000 | 95.40 | 99.60 | 96.60 | 88.80 | 95.10 | `evaluate_results/libero_uncond_2cam224_step022000_eval_20260616_224500` |
| FastWAM baseline | 043400 | 96.00 | 99.40 | 97.80 | 93.40 | 96.65 | `evaluate_results/libero_uncond_2cam224_step043400_eval_20260616_135804` |

Baseline interpretation:

- `step_022000` is a useful fast screening gate because the original model is already at `95.10%`.
- Final full FastWAM reaches `96.65%` at `step_043400`, which is the high bar for any lightweight variant.

## 3. Inference-Step Reduction and Latency Profiling

Purpose: test whether the original FastWAM action generator actually needs many denoising steps.

Result directories:

```text
evaluate_results/libero_uncond_2cam224_step043400_infer20_eval_full_20260622_191055
evaluate_results/libero_uncond_2cam224_step043400_infer10_eval_full_20260622_191055
evaluate_results/libero_uncond_2cam224_step043400_infer5_eval_full_20260622_191055
evaluate_results/libero_uncond_2cam224_step043400_infer2_eval_full_20260622_191055
evaluate_results/libero_uncond_2cam224_step043400_infer1_eval_full_20260622_225000
```

Full LIBERO results:

| Inference steps | Spatial | Object | Goal | Libero-10 | Overall | Avg task time |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 96.40 | 99.80 | 97.20 | 91.80 | 96.30 | 1173s |
| 10 | 96.00 | 99.40 | 97.80 | 93.40 | 96.65 | 837s |
| 5 | 96.40 | 99.40 | 98.00 | 91.40 | 96.30 | 693s |
| 2 | 96.60 | 99.40 | 97.40 | 93.00 | 96.60 | 669s |
| 1 | 97.40 | 98.60 | 96.00 | 92.00 | 96.00 | 738s |

Pure `infer_action` microbenchmark:

| Inference steps | Mean latency | Pure model speed vs 10-step | Peak allocated memory |
|---:|---:|---:|---:|
| 1 | 73.67 ms | 4.41x | 23.26 GiB |
| 2 | 102.74 ms | 3.16x | 23.26 GiB |
| 5 | 188.20 ms | 1.72x | 23.26 GiB |
| 10 | 324.58 ms | 1.00x | 23.26 GiB |
| 20 | 604.89 ms | 0.54x | 23.26 GiB |

Component breakdown:

| Inference steps | Wall total | One-time visual condition | VAE first-frame encode | Video KV-cache prefill | Action loop total | Action MoT/cache core |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 80.83 ms | 39.92 ms | 6.79 ms | 32.43 ms | 32.78 ms | 32.24 ms |
| 2 | 110.92 ms | 38.76 ms | 6.81 ms | 31.23 ms | 63.88 ms | 62.81 ms |
| 5 | 207.51 ms | 39.96 ms | 6.86 ms | 32.37 ms | 158.35 ms | 155.33 ms |
| 10 | 366.08 ms | 39.92 ms | 7.14 ms | 32.07 ms | 317.00 ms | 311.83 ms |
| 20 | 669.81 ms | 39.18 ms | 6.83 ms | 31.62 ms | 620.78 ms | 610.43 ms |

Conclusion:

- FastWAM rollout remains strong even at 2 steps.
- The repeated action MoT loop is the main model-side latency target.
- Fewer diffusion steps reduce latency but do not reduce model-resident memory.

## 4. Early PocketFlow Route

This route used pooled FastWAM/Wan cache features and a small external action decoder. It was useful diagnostically but failed in closed-loop rollout.

### 4.1 Offline Frozen-Feature Regression

1k regression pilot:

```text
run: runs/pocket_action_regression_pilot_1k/20260623_124841
samples: 1024
feature shape: [1024, 10248]
action shape: [1024, 32, 7]
head params: 5.65M
```

| Model / baseline | Val MSE | Val L1 |
|---|---:|---:|
| Zero action baseline | 0.2463 | - |
| Train-set mean action baseline | 0.2360 | - |
| Tiny action regressor, final epoch 50 | 0.0667 | 0.1648 |
| Tiny action regressor, best epoch 48 | 0.0593 | 0.1551 |

Conclusion: frozen FastWAM/Wan features contain action information, but open-loop MSE is not enough to predict closed-loop success.

### 4.2 Offline Flow/DiM Decoder

Feature table:

```text
runs/pocket_action_regression_pilot_1k/20260623_124841/feature_table.pt
```

Best validation by `val_flow_2step_mse`:

| Variant | Hidden / blocks | Params | Best 2-step MSE | Best 2-step L1 |
|---|---:|---:|---:|---:|
| tiny | 128 / 1 | 1.46M | 0.0837 | 0.1759 |
| small | 256 / 2 | 3.39M | 0.0559 | 0.1361 |
| medium | 384 / 3 | 6.21M | 0.0586 | 0.1371 |
| full | 512 / 4 | 10.32M | 0.0549 | 0.1297 |

Decision: `small` was the best efficiency point and was used for rollout integration.

### 4.3 PocketFlow Rollout Integration

Pocket checkpoint:

```text
runs/pocket_action_flow_pilot_small_seed2/20260623_145455/flow_decoder.pt
```

Subset rollout:

```text
evaluate_results/pocket_flow_subset_20260623_155224
```

| Suite / task | PocketFlow small | FastWAM 2-step baseline |
|---|---:|---:|
| libero_spatial task0 | 5/5 | 5/5 |
| libero_object task0 | 2/5 | 4/5 |
| libero_goal task0 | 0/5 | 4/5 |
| libero_10 task0 | 0/5 | 5/5 |
| Overall subset | 7/20 = 35% | 18/20 = 90% |

Conclusion: direct external PocketFlow replacement failed on closed-loop control despite reasonable offline MSE.

### 4.4 Teacher-Action Distillation Attempt

Run:

```text
runs/pocket_action_flow_teacher_pilot/20260623_172248
evaluate_results/pocket_flow_teacher_subset_20260623_182320
```

Offline teacher-action result:

| Model | Sampling | Val MSE | Val L1 |
|---|---:|---:|---:|
| Teacher-distilled small Flow/DiM | 1 step | 0.1593 | 0.2369 |
| Teacher-distilled small Flow/DiM | 2 steps | 0.1774 | 0.2373 |
| Teacher-distilled small Flow/DiM | 5 steps | 0.1968 | 0.2482 |

Closed-loop subset:

| Suite / task | Dataset-action PocketFlow | Teacher-action PocketFlow | FastWAM 2-step baseline |
|---|---:|---:|---:|
| libero_spatial task0 | 5/5 | 0/5 | 5/5 |
| libero_object task0 | 2/5 | 0/5 | 4/5 |
| libero_goal task0 | 0/5 | 0/5 | 4/5 |
| libero_10 task0 | 0/5 | 0/5 | 5/5 |
| Overall subset | 7/20 = 35% | 0/20 = 0% | 18/20 = 90% |

Conclusion: single-sample teacher-action distillation did not fix the closed-loop gap.

## 5. ActionDiT Internal Ablation and Late-FFN Route

### 5.1 ActionDiT FFN Sensitivity

Protocol:

```text
checkpoint: FastWAM step_043400
num_inference_steps: 2
subset: libero_spatial/object/goal/libero_10 task0
trials: 5 per task
baseline: FastWAM 2-step subset = 18/20
```

| Run | Variant | Result | Baseline | Output |
|---|---|---:|---:|---|
| A1 | action FFN x0.5, all 30 layers | 5/20 | 18/20 | `evaluate_results/actiondit_ablation_ffn05_subset_20260623_222854` |
| A1b | action FFN x0.75, all 30 layers | 12/20 | 18/20 | `evaluate_results/actiondit_ablation_ffn075_subset_20260623_225218` |
| A3 | action FFN x0.5, last 15 layers | 19/20 | 18/20 | `evaluate_results/actiondit_ablation_ffn05_last15_subset_20260623_230553` |
| A3b | action FFN x0.0, last 15 layers | 12/20 | 18/20 | `evaluate_results/actiondit_ablation_ffn00_last15_subset_20260624_094253` |

Conclusion:

- Do not compress all ActionDiT FFNs uniformly.
- Late layers 15-29 are much more compressible than the first half.
- Hard deletion is too aggressive; trainable width reduction is a better first structural baseline.

### 5.2 Late-FFN-2048 Structural Pilot

Variant:

```text
ActionDiT layers 0-14 FFN dim = 4096
ActionDiT layers 15-29 FFN dim = 2048
```

Parameter effect:

| Model | ActionDiT params | Delta |
|---|---:|---:|
| Default | 1,020,886,023 | - |
| Late-FFN-2048 | 957,940,743 | -62,945,280 |

22k result:

```text
run: runs/libero_uncond_2cam224_1e-4/libero_uncond_lateffn2048_step022000_20260624_121424
eval: evaluate_results/libero_uncond_lateffn2048_step022000_eval_full_20260625_185113
```

| Model | Step | Spatial | Object | Goal | Libero-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| FastWAM baseline | 022000 | 95.40 | 99.60 | 96.60 | 88.80 | 95.10 |
| Late-FFN-2048 | 022000 | 96.40 | 99.80 | 96.20 | 92.60 | 96.25 |

Conclusion: passed the 22k gate and supports ActionDiT late-FFN overcapacity.

### 5.3 Late-FFN Full-Step and 1024 Probe

Results:

| Model | Step | Late FFN dim | ActionDiT params | Spatial | Object | Goal | Libero-10 | Overall | Result dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Late-FFN-2048 | 043400 | 2048 | 957,940,743 | 91.80 | 98.80 | 89.00 | 82.60 | 90.55 | `evaluate_results/libero_uncond_lateffn2048_step043400_eval_full_20260627_101221` |
| Late-FFN-1024 | 022000 | 1024 | 926,468,103 | 97.20 | 99.60 | 96.00 | 92.60 | 96.35 | `evaluate_results/libero_uncond_lateffn1024_step022000_eval_full_20260627_101221` |

Interpretation:

- Late-FFN-1024 is surprisingly strong at the 22k screening point, saving `94,417,920` ActionDiT parameters.
- Late-FFN-2048 degraded at full 43.4k, so longer training is not automatically better for this structural variant.
- FFN-only compression is useful evidence, but it is too conservative as the final lightweight direction because the action tower still has about `926M-958M` parameters.

## 6. Pocket-FastWAM DiM Architecture Experiments

The stronger route was to decouple action prediction from MoT and condition a compact action denoiser on Wan first-frame features.

### 6.1 R0: First-Frame Wan Feature-Tap Probe

Source:

```text
profile_output/wan_first_frame_feature_taps/20260627_132500/summary.json
docs/pocket_fastwam_dim_design_2026-06-25.md
```

Feature package:

```text
tapped layers: [5, 11, 17, 23, 29]
per-layer tap: [B, 98, 3072]
text_context: [B, 129, 3072]
proprio: [B, 8]
action: [B, 32, 7]
```

Conclusion: the first-frame Wan feature path has stable geometry and can feed a compact external action decoder.

### 6.2 R1/R2: Aggregator and Decoder Unit Modules

Implemented modules:

```text
TextGuidedSlotAggregator
ActionDiMDecoder
ActionSmallDiTDecoder
```

Parameter counts:

| Module | Params |
|---|---:|
| TextGuidedSlotAggregator | 6,013,712 |
| ActionDiMDecoder | 9,661,959 |
| Aggregator + DiM | 15,675,671 |
| Original ActionDiT expert | 1,020,886,023 |

Conclusion: DiM and Small-DiT can share the same condition interface. The DiM branch is about two orders of magnitude smaller than ActionDiT.

### 6.3 R3: Synthetic Overfit

Runs:

```text
runs/pocket_synthetic_overfit/20260627_134651/summary.json
runs/pocket_synthetic_overfit/20260627_134714/summary.json
```

Purpose: verify optimizer, forward pass, loss, backward pass, and saving for fake Wan taps plus action targets.

Conclusion: synthetic overfit passed; the next risk moved from wiring to real frozen-Wan feature usefulness.

### 6.4 R4: Frozen-Wan 16-Sample Real-Data Sanity

Run:

```text
runs/pocket_frozen_wan_sanity/20260627_140255
```

Setup:

```text
samples: 16 real LIBERO samples
steps: 200
batch size: 2
trainable: TextGuidedSlotAggregator + ActionDiMDecoder
frozen: VAE, text path/cache, Wan video expert, MoT/ActionDiT baseline modules
```

| Metric | Value |
|---|---:|
| initial loss | 1.689560 |
| final loss | 0.317619 |
| final / initial | 0.187989 |
| best loss | 0.063478 |
| best step | 129 |

Conclusion: real dataset path and frozen first-frame Wan taps can train the DiM branch on a tiny real-data sanity set.

### 6.5 R4b: Frozen-Wan 128-Sample Real-Data Sanity

Run:

```text
runs/pocket_frozen_wan_sanity_r4b/20260627_141021
```

Setup:

```text
samples: 128
steps: 1000
batch size: 8
trainable params: 15,675,671
```

| Metric | Value |
|---|---:|
| initial loss | 1.644306 |
| final loss | 0.048403 |
| final / initial | 0.029437 |
| best loss | 0.011720 |
| best step | 953 |

Conclusion: larger cached-feature sanity also passed. It still measures train loss only, not rollout.

### 6.6 1024-Sample Cached-Feature 22k Sanity

Run:

```text
runs/pocket_frozen_wan_sanity_dim1024_22k/20260627_142311
```

Setup:

```text
samples: 1024
steps: 22000
batch size: 8
trainable: aggregator + ActionDiMDecoder
feature extraction: frozen Wan first-frame taps
peak memory: 12.68 GiB allocated
```

| Metric | Value |
|---|---:|
| initial loss | 1.678006 |
| final loss | 0.018598 |
| final / initial | 0.011083 |
| best loss | 0.003160 |
| best step | 21655 |
| extract time | 600.97s |
| train time | 1188.24s |

Conclusion:

- Cached-feature training can drive loss very low on 1024 samples.
- This does not imply rollout success; it is still a memorization/generalization-limited offline sanity.
- The next meaningful gate is online full-data training plus LIBERO evaluation.

## 7. Full-Data Online Frozen-Wan Pocket-DiM

Training run:

```text
script: scripts/train_pocket_full_data_dim.py
run: runs/pocket_full_data_dim/pocket_dim_full_data_22k_gpu4567_20260627_163341
checkpoint: runs/pocket_full_data_dim/pocket_dim_full_data_22k_gpu4567_20260627_163341/checkpoints/step_022000.pt
log: logs/pocket_dim_full_data_22k_gpu4567_20260627_163341.log
```

Training setup:

```text
world_size: 4
batch_per_gpu: 1
dataset_len: 277713
max_steps: 22000
base FastWAM checkpoint: step_043400
trainable: aggregator + ActionDiMDecoder = 15,675,671 params
frozen: FastWAM/Wan/VAE/text/MoT/ActionDiT
feature path: online first-frame Wan taps
loss: Pocket action flow loss only
```

Final training log:

```text
step 22000 loss 0.039216
speed about 1.475 step/s
peak allocated about 13.01 GiB
```

Full LIBERO evaluation:

```text
evaluate_results/pocket_dim_full_data_step022000_on_fastwam_step043400_eval_20260627_213422
```

| Model | Spatial | Object | Goal | Libero-10 | Overall |
|---|---:|---:|---:|---:|---:|
| Frozen-Wan Pocket-DiM | 91.60 | 95.80 | 74.80 | 60.20 | 80.60 |
| FastWAM baseline step043400 | 96.00 | 99.40 | 97.80 | 93.40 | 96.65 |

Conclusion:

- Frozen-Wan Pocket-DiM is much better than the earlier pooled-feature PocketFlow subset failures, but still far below FastWAM.
- The largest gap is on `libero_goal` and `libero_10`, suggesting the standalone first-frame action decoder lacks semantic/long-horizon robustness.
- This is the best evidence so far for the current Pocket-DiM design, but it is not paper-ready.

## 8. Action-Only Train-Wan Pocket-DiM

This was launched because we wanted to test whether updating Wan/MoT-side parameters with the Pocket-DiM action loss could close the gap.

Code changes:

```text
scripts/train_pocket_full_data_dim.py
  added --train-wan
  added --wan-lr
  added --no-save-optimizer
  added --no-save-latest
  when --train-wan:
    VAE and text encoder frozen
    video_expert/action_expert/mot/dit/proprio_encoder trainable
    first-frame Wan taps are not detached
    optimizer has pocket group lr=1e-4 and FastWAM group wan_lr=1e-5

experiments/libero/pocket_dim_adapter.py
  eval can load fastwam.mot and proprio_encoder from Pocket joint checkpoint
```

Important objective caveat:

```text
This is not FastWAM full training.
It does not call FastWAM.training_loss(sample).
It only uses video[:, :, 0], extracts first-frame Wan taps, and trains Pocket-DiM action flow loss.
It does not compute video diffusion loss.
It does not train the full future-video latent sequence.
It does not run full video+action MoT joint denoising over the training horizon.
```

Smoke runs:

```text
runs/pocket_full_data_dim/pocket_dim_trainwan_smoke_gpu4_20260628
  2 steps, checkpoint with optimizer/latest was about 31G each

runs/pocket_full_data_dim/pocket_dim_trainwan_noopt_smoke_gpu4_20260628
  1 step, --no-save-optimizer --no-save-latest
  checkpoint about 12.1G
```

Full train-Wan action-only run:

```text
script: scripts/train_pocket_full_data_dim.py
run: runs/pocket_full_data_dim/pocket_dim_trainwan_full_22k_gpu0123_20260627_2338
checkpoint: runs/pocket_full_data_dim/pocket_dim_trainwan_full_22k_gpu0123_20260627_2338/checkpoints/step_022000.pt
log: logs/pocket_dim_trainwan_full_22k_gpu0123_20260627_2338.log
```

Training setup:

```text
world_size: 4
batch_per_gpu: 1
max_steps: 22000
lr: 1e-4
wan_lr: 1e-5
train_wan: true
save_every: 22000
no_save_optimizer: true
no_save_latest: true
```

Trainable parameters:

| Module | Trainable params |
|---|---:|
| aggregator | 6,013,712 |
| decoder | 9,661,959 |
| fastwam total trainable | 6,020,710,599 |
| video_expert trainable | 4,999,787,712 |
| action_expert trainable | 1,020,886,023 |
| proprio_encoder trainable | 36,864 |
| vae trainable | 0 |
| text_encoder | not loaded |

Final training log:

```text
step 22000 loss 0.071622
speed about 1.439 step/s
peak allocated about 50.14 GiB
```

Evaluation:

```text
evaluate_results/pocket_dim_trainwan_step022000_on_fastwam_step043400_eval_20260628_102535
log: logs/pocket_dim_trainwan_step022000_eval_gpu4567_20260628_102535.log
failed_tasks.txt: empty
```

| Model | Spatial | Object | Goal | Libero-10 | Overall |
|---|---:|---:|---:|---:|---:|
| Train-Wan Pocket-DiM action-only | 74.20 | 65.60 | 56.20 | 22.00 | 54.50 |
| Frozen-Wan Pocket-DiM | 91.60 | 95.80 | 74.80 | 60.20 | 80.60 |
| FastWAM baseline step043400 | 96.00 | 99.40 | 97.80 | 93.40 | 96.65 |

Task-level failures:

```text
libero_10_3: 0.00
libero_10_4: 0.00
libero_10_8: 0.00
libero_goal_0: 0.00
libero_object_8: 0.00
libero_spatial_5: 2.00
```

Conclusion:

- Action-only train-Wan made the model much worse than frozen-Wan Pocket-DiM.
- Most likely reason: the action-only loss updates the Wan/MoT representation without the stabilizing original video diffusion objective and without the full FastWAM joint denoising objective.
- This result argues against continuing this exact `--train-wan` script.

## 9. Verification and Tests

Relevant tests added or used:

```text
tests/test_pocket_full_data_dim_train.py
tests/test_pocket_dim_eval_adapter.py
tests/test_pocket_frozen_wan_sanity.py
tests/test_pocket_synthetic_overfit.py
tests/test_probe_wan_first_frame_feature_taps.py
tests/models/test_pocket_fastwam_aggregator.py
tests/models/test_pocket_fastwam_decoders.py
tests/models/test_action_dit_late_ffn.py
tests/models/test_action_dit_dim_mixer.py
```

Known verification snapshots:

```text
R4/R4b test bundle: 18 passed
train-wan/eval adapter tests: 21 passed
ActionDiT LateDiM unit tests: 8 passed
py_compile passed for:
  scripts/train_pocket_full_data_dim.py
  experiments/libero/pocket_dim_adapter.py
  experiments/libero/eval_libero_pocket_dim_single.py
```

## 10. True FastWAM-Objective ActionDiT-DiM

Date: 2026-06-28

Purpose:

```text
Correct the previous direction away from action-only train-Wan / standalone Pocket-DiM.
Keep the original FastWAM trainer and FastWAM.training_loss(sample), including video diffusion loss, action diffusion loss, full video latents, and MoT joint denoising.
Compress only ActionDiT late-layer FFN/refinement internals while keeping MoT attention geometry unchanged.
```

Implemented variant:

```text
name: ActionDiT-LateDiM / dim_ffn_lite
file: src/fastwam/models/wan22/action_dit.py
launcher: scripts/launch_libero_latedim_train4.sh
test: tests/models/test_action_dit_dim_mixer.py
default behavior: unchanged unless late_dim_mixer_* config is explicitly set
```

Architecture:

```text
layers 0-14:
  original ActionDiT block and FFN, ffn_dim=4096

layers 15-29:
  attention / q-k-v-o / cross-attn / hidden_dim / num_heads / head_dim unchanged
  FFN replaced by DiMFFNLite:
    channel_mixer: Linear(1024, 512) -> GELU -> Linear(512, 1024)
    token branch: LayerNorm -> Linear(1024, 64) -> Conv1d temporal mixer -> Linear(64, 1024)
    token_up zero-initialized for warm-start safety
```

Config overrides for the first smoke:

```text
model.action_dit_config.late_dim_mixer_start_layer=15
model.action_dit_config.late_dim_mixer_type=dim_ffn_lite
model.action_dit_config.late_dim_mixer_hidden_dim=512
model.action_dit_config.late_dim_mixer_token_mlp_dim=64
model.action_dit_config.late_dim_mixer_channel_mlp_dim=512
```

Parameter count:

| Model | ActionDiT params | Delta vs default | Delta vs Late-FFN-1024 |
|---|---:|---:|---:|
| Default FastWAM ActionDiT | 1,020,886,023 | - | - |
| Late-FFN-1024 | 926,468,103 | -94,417,920 | - |
| ActionDiT-LateDiM ch512/tok64 | 914,688,903 | -106,197,120 | -11,779,200 |

E0/E1 verification:

```text
/tmp/fastwam_torch210/bin/python -m pytest \
  tests/models/test_action_dit_late_ffn.py \
  tests/models/test_action_dit_dim_mixer.py -q

result: 8 passed
```

ActionDiT-only real-backbone smoke:

```text
checkpoint: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
loaded_params: 914,688,903
dim_mixer_layers: 15-29
token_up_abs_sum: 0.000000
channel0_shape: (512, 1024)
q_shape: (3072, 1024)
```

E2 200-step four-GPU training sanity:

```text
run id: libero_uncond_latedim_dim_ffn_lite_h512_tok64_ch512_step200_20260628_143817
launcher: scripts/launch_libero_latedim_train4.sh
GPUs: 4,5,6,7
MAX_STEPS=200
SAVE_EVERY=100
EVAL_EVERY=100
log: logs/libero_uncond_latedim_dim_ffn_lite_h512_tok64_ch512_step200_20260628_143817.log
weights:
  runs/libero_uncond_2cam224_1e-4/libero_uncond_latedim_dim_ffn_lite_h512_tok64_ch512_step200_20260628_143817/checkpoints/weights/step_000100.pt
  runs/libero_uncond_2cam224_1e-4/libero_uncond_latedim_dim_ffn_lite_h512_tok64_ch512_step200_20260628_143817/checkpoints/weights/step_000200.pt
state:
  runs/libero_uncond_2cam224_1e-4/libero_uncond_latedim_dim_ffn_lite_h512_tok64_ch512_step200_20260628_143817/checkpoints/state/step_000100
  runs/libero_uncond_2cam224_1e-4/libero_uncond_latedim_dim_ffn_lite_h512_tok64_ch512_step200_20260628_143817/checkpoints/state/step_000200
```

Training sanity metrics:

| Step | loss | loss_action | loss_video | Notes |
|---:|---:|---:|---:|---|
| 10 | 0.8680 | 0.5821 | 0.2859 | first logged train point |
| 100 | 0.3833 | 0.1686 | 0.2147 | eval and checkpoint succeeded |
| 200 | 0.3097 | 0.1186 | 0.1911 | final eval and checkpoint succeeded |

Validation snapshots:

| Step | val_loss | infer_psnr | infer_ssim | action_l2 | action_l1 |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.3926 | 25.6034 | 0.7987 | 0.1160 | 0.2139 |
| 200 | 0.4440 | 24.6395 | 0.7921 | 0.0789 | 0.1653 |

Interpretation:

```text
This is the first implemented ActionDiT-internal DiM-style compression that keeps the original FastWAM full training path.
The 200-step run confirms no shape mismatch, no NaN, successful eval, successful checkpoint saves, and logs both loss_video and loss_action.
It is not a LIBERO success-rate result; the next decision gate is a 22k screening run and full 40-task evaluation at step_022000.
```

Next action:

```text
Run E3 22k screening with the same overrides:
MAX_STEPS=22000 SAVE_EVERY=2000 EVAL_EVERY=200 bash scripts/launch_libero_latedim_train4.sh
Then evaluate step_022000 over all 40 LIBERO tasks and compare against FastWAM baseline 95.10 and Late-FFN-1024 96.35.
```

E3 launch status:

```text
started: 2026-06-28T15:24:14+08:00
screen: fastwam_latedim_22k_20260628
run id: libero_uncond_latedim_dim_ffn_lite_h512_tok64_ch512_step22000_20260628_151900
GPUs: 4,5,6,7
log: logs/libero_uncond_latedim_dim_ffn_lite_h512_tok64_ch512_step22000_20260628_151900.log
command:
  RUN_TS=20260628_151900 \
  RUN_ID=libero_uncond_latedim_dim_ffn_lite_h512_tok64_ch512_step22000_20260628_151900 \
  MAX_STEPS=22000 SAVE_EVERY=2000 EVAL_EVERY=200 \
  GPU_IDS=4,5,6,7 NPROC=4 PYTHON_BIN=/tmp/fastwam_torch210/bin/python \
  screen -dmS fastwam_latedim_22k_20260628 bash scripts/launch_libero_latedim_train4.sh
```

Initial E3 health check:

| Step | loss | loss_action | loss_video | Notes |
|---:|---:|---:|---:|---|
| 10 | 1.7933 | 1.3294 | 0.4639 | full objective confirmed |
| 20 | 1.4202 | 1.0745 | 0.3457 | loss decreasing |
| 30 | 1.0761 | 0.7461 | 0.3300 | loss decreasing |

The E3 run is still in progress and is not yet a result. The next checkpoint/eval milestone is step 200; first persistent weight checkpoint is step 002000.

## 11. Current Research State

What is supported:

1. FastWAM can use very few action inference steps with minimal performance loss.
2. The repeated ActionDiT/MoT action loop is the main model-side latency target.
3. ActionDiT late FFNs are overbuilt: `late_ffn_dim=1024` at layers 15-29 still gives `96.35%` at 22k.
4. Wan first-frame features are action-informative and can train compact DiM decoders.
5. Standalone Pocket-DiM with frozen Wan reaches `80.60%`, which is a real signal but far from FastWAM.
6. ActionDiT-LateDiM keeps the original FastWAM full objective and passes 200-step 4-GPU sanity with both video and action losses logged.

What is not supported:

1. The pooled PocketFlow head is not a viable direct replacement.
2. Teacher-action-only distillation with the small pooled-feature head does not fix rollout.
3. Full-data frozen-Wan Pocket-DiM is not close enough to FastWAM.
4. Action-only train-Wan Pocket-DiM is worse than frozen Wan.
5. The current `--train-wan` path should not be called FastWAM-equivalent full training.
6. ActionDiT-LateDiM has not yet been validated by 22k training or full LIBERO rollout.

## 12. Recommended Next Direction

Do not continue:

```text
1. Do not extend the current action-only --train-wan Pocket-DiM run.
2. Do not claim the current train-Wan result is a full FastWAM objective.
3. Do not scale the old pooled PocketFlow/teacher-action-only route.
```

Most defensible next experiments:

1. Scale ActionDiT-LateDiM from E2 to E3:
   - `MAX_STEPS=22000`;
   - same `dim_ffn_lite` overrides;
   - evaluate `step_022000` on all 40 LIBERO tasks.
2. Use Late-FFN-1024 as the strong structural baseline:
   - it is simple, MoT-compatible, and already gets `96.35%` at 22k.
3. If ActionDiT-LateDiM underperforms:
   - increase `late_dim_mixer_channel_mlp_dim` to `1024`;
   - or keep a residual low-rank FFN branch in late layers;
   - diagnose whether `goal/libero_10` degrade more than `spatial/object`.
4. If continuing standalone Pocket-DiM:
   - use frozen-Wan Pocket-DiM as the control baseline (`80.60%`);
   - add stronger conditioning rather than simply unfreezing Wan with action-only loss;
   - consider distilling intermediate ActionDiT/MoT hidden states, not only final actions.
5. Treat `libero_goal` and `libero_10` as the main diagnostic suites because they expose the gap most clearly.
