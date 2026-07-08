# Pocket-FastWAM Language-Tapped DiM bs128 Record

Date: 2026-07-08

This document records the first full LIBERO result for **Pocket-FastWAM Language-Tapped DiM** trained with the effective-batch setting intended to match the likely official Fast-WAM LIBERO training regime more closely:

```text
4 GPUs * batch_size 16 * gradient_accumulation_steps 2 = effective batch 128
max_steps = 21700
```

The run is a fresh training run from step 0. It is not a resume from the previous 22k checkpoint and it is not a constant-LR tail continuation.

## 1. Main Result

Full LIBERO evaluation, `40` tasks, `50` trials per task, `2000` total trials:

| Model / run | Step | Spatial | Object | Goal | Libero-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| FastWAM reproduced baseline | 043400 | 96.00 | 99.40 | 97.80 | 93.40 | 96.65 |
| Language-Tapped DiM, previous 22k screening | 022000 | 97.20 | 99.80 | 97.00 | 96.20 | 97.55 |
| Language-Tapped DiM, previous 43.4k full schedule | 043400 | 98.40 | 99.20 | 98.20 | 95.80 | 97.90 |
| **Language-Tapped DiM, effective bs128** | **021700** | **99.00** | **99.80** | **98.40** | **97.80** | **98.75** |

Delta from reproduced FastWAM `step_043400`:

| Suite | Delta |
|---|---:|
| Spatial | +3.00 |
| Object | +0.40 |
| Goal | +0.60 |
| Libero-10 | +4.40 |
| Overall | +2.10 |

Delta from the previous best Language-Tapped DiM `step_043400` full-schedule run:

| Suite | Delta |
|---|---:|
| Spatial | +0.60 |
| Object | +0.60 |
| Goal | +0.20 |
| Libero-10 | +2.00 |
| Overall | +0.85 |

The official Fast-WAM project page reports vanilla Fast-WAM at `97.6` average LIBERO success. This run is `+1.15` points above that vanilla reported number, while still being a single local run and single seed.

## 2. Claim Boundary

Supported:

- This is the strongest local LIBERO result currently recorded for Pocket-FastWAM Language-Tapped DiM.
- Under the local full-LIBERO protocol, the effective-bs128 run beats the reproduced FastWAM baseline by `+2.10` overall points.
- The biggest improvement is on `libero_10`, where it reaches `97.80%`.
- The previous gap between the local reproduced FastWAM baseline and the official vanilla Fast-WAM result is plausibly explained at least in part by effective-batch / schedule-horizon differences.

Not yet supported:

- A general robotics SOTA claim beyond LIBERO.
- A multi-seed stability claim.
- A claim against all official Fast-WAM variants. The official project page also reports stronger variants such as Fast-WAM-Joint.
- A final paper claim without artifact backup and an external integrity review.

Result-to-claim status:

```text
local verdict: supported for "local LIBERO SOTA among recorded runs"
integrity status: provisional, no external reviewer audit in this session
recommended next evidence: repeat seed or official 8-GPU run; artifact upload
```

## 3. Training Run

Run:

```text
run_id:
  libero_uncond_languagetap_dim_step021700_fromscratch_officialbs128_accum2_20260706_211913

run_dir:
  runs/libero_uncond_2cam224_1e-4/libero_uncond_languagetap_dim_step021700_fromscratch_officialbs128_accum2_20260706_211913

log:
  logs/libero_uncond_languagetap_dim_step021700_fromscratch_officialbs128_accum2_20260706_211913.log
```

Launch configuration:

```text
task = libero_uncond_2cam224_1e-4
model = pocket_fastwam_languagetap
gpu_ids = 0,1,2,3
nproc = 4
python = /tmp/fastwam_torch210/bin/python
resume = null
resume_scheduler_policy = load
batch_size = 16
gradient_accumulation_steps = 2
effective_batch = 128
max_steps = 21700
save_every = 2000
eval_every = 1000
lr_scheduler_type = cosine
learning_rate = 1e-4
eta_min = 1e-6
```

Completion evidence:

```text
started_at = 2026-07-06T21:19:35+08:00
finished_at = 2026-07-08T10:18:16+08:00
final log = max_steps reached step=21700 max_steps=21700
```

Trainer state:

```json
{
  "global_step": 21700,
  "epoch": 9,
  "batch_in_epoch": 4340,
  "max_steps": 21700,
  "stop_steps": null,
  "train_stop_steps": 21700,
  "scheduler": {
    "lr_scheduler_type": "cosine",
    "max_steps": 21700,
    "warmup_steps": 1085,
    "learning_rate": 0.0001,
    "eta_min": 1e-06
  }
}
```

LR trace from the training log:

| Step | LR |
|---:|---:|
| 10 | 1.01e-06 |
| 1000 | 9.22e-05 |
| 5000 | 9.14e-05 |
| 10000 | 6.09e-05 |
| 20000 | 2.65e-06 |
| 21700 | 1.00e-06 |

This confirms the run used the intended warmup+cosine schedule over the `21700`-step horizon, not the earlier constant-LR continuation path.

## 4. Evaluation Run

Run:

```text
eval_run_id:
  libero_uncond_languagetap_dim_step021700_officialbs128_accum2_eval_full_20260708_192749

output_dir:
  evaluate_results/libero_uncond_languagetap_dim_step021700_officialbs128_accum2_eval_full_20260708_192749

log:
  logs/libero_uncond_languagetap_dim_step021700_officialbs128_accum2_eval_full_gpu0123_20260708_192749.log
```

Evaluation configuration:

```text
GPUs = 0,1,2,3
NUM_TRIALS = 50
INFER_STEPS = 10
MAX_TASKS_PER_GPU = 2
model = pocket_fastwam_languagetap
tasks = 40
total trials = 2000
failed tasks = 0
completed_at = 2026-07-08T20:12:53+08:00
```

Summary:

```text
Spatial   99.00
Object    99.80
Goal      98.40
Libero-10 97.80
Overall   98.75
```

Task-level weak spots:

| Task | Success |
|---|---:|
| libero_spatial_5 | 94.00 |
| libero_10_4 | 94.00 |
| libero_10_6 | 94.00 |
| libero_goal_2 | 94.00 |
| libero_goal_3 | 94.00 |

## 5. Artifact Manifest

The main model artifact is intentionally not tracked in normal Git because the repository ignores `runs/`, `evaluate_results/`, and `*.pt`. Preserve the model through a dedicated artifact store such as Git LFS, Hugging Face, S3/R2, or a lab filesystem snapshot.

Files:

| Artifact | Size | SHA256 |
|---|---:|---|
| `runs/libero_uncond_2cam224_1e-4/libero_uncond_languagetap_dim_step021700_fromscratch_officialbs128_accum2_20260706_211913/checkpoints/weights/step_021700.pt` | 9.4G | `2693fff675de27dcaef5c4a00757f11717c5880826c0e0cd8ff9ee2b00f76a23` |
| `runs/libero_uncond_2cam224_1e-4/libero_uncond_languagetap_dim_step021700_fromscratch_officialbs128_accum2_20260706_211913/dataset_stats.json` | 40K | `008a6baab470c64c42b2b8060535897628528ff2a2c0178cc4c66a6bb7727ad8` |
| `evaluate_results/libero_uncond_languagetap_dim_step021700_officialbs128_accum2_eval_full_20260708_192749/summary.csv` | 222B | `c9904da9fdacbbe6f05a75c2648c796c6351c85dbfb7ac268e477908795a7d5e` |
| `evaluate_results/libero_uncond_languagetap_dim_step021700_officialbs128_accum2_eval_full_20260708_192749/summary.json` | 12K | `d321d08dddb41656044d3cb59cd4ade20ffe9860dbdef8bcf627dba1aeb0f50d` |

State checkpoint:

```text
runs/libero_uncond_2cam224_1e-4/libero_uncond_languagetap_dim_step021700_fromscratch_officialbs128_accum2_20260706_211913/checkpoints/state/step_021700
```

The state directory contains full resume metadata, including optimizer shards, `scheduler.bin`, sampler state, random states, and `trainer_state.json`.

## 6. Preservation Recommendation

Recommended Git commit contents:

- this record file;
- the implementation and config files for `pocket_fastwam_languagetap`;
- the resume/scheduler safety fixes;
- evaluation and launcher scripts needed to reproduce the run;
- no `runs/`, `evaluate_results/`, logs, or `*.pt` files.

Recommended artifact upload:

```text
checkpoint:
  step_021700.pt
dataset stats:
  dataset_stats.json
full state:
  checkpoints/state/step_021700
eval result bundle:
  evaluate_results/libero_uncond_languagetap_dim_step021700_officialbs128_accum2_eval_full_20260708_192749
```

If using Hugging Face, create a private model repository first and upload the checkpoint/state/result bundle with the SHA256 values above in the model card. If using GitHub only, use Git LFS for `*.pt` and preferably avoid uploading the DeepSpeed optimizer state unless the storage quota is explicitly acceptable.
