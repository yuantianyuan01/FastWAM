# RoboDojo evaluation

This directory provides a FastWAM policy adapter for RoboDojo evaluation through
the XPolicyLab websocket policy-server interface.

The adapter intentionally lives under `experiments/robodojo`. The official
RoboDojo code is vendored under `third_party/RoboDojo`, following the same
layout style as `third_party/RoboTwin`. Assets, generated datasets, evaluation
outputs, and checkpoints are kept in the regular FastWAM top-level directories.

## Layout

```text
experiments/robodojo/
├── fastwam_policy/
│   ├── model.py
│   ├── deploy.py
│   ├── deploy.yml
│   ├── eval.sh
│   ├── setup_eval_policy_server.sh
│   ├── setup_eval_env_client.sh
│   └── install.sh
└── run_robodojo_manager.py
```

Recommended repository layout:

```text
FastWAM/
├── configs/
│   ├── data/robodojo.yaml
│   ├── data/robodojo_sim.yaml
│   ├── data/robodojo_real.yaml
│   ├── task/robodojo_sim_uncond_3cam_384_1e-4.yaml
│   ├── task/robodojo_real_uncond_3cam_384_1e-4.yaml
│   ├── sim_robodojo.yaml
│   └── real_robodojo.yaml
├── data/
│   └── robodojo/
│       ├── assets/
│       ├── sim/
│       │   ├── lerobot/
│       │   └── dataset_stats.json
│       └── real/
│           ├── lerobot/
│           └── dataset_stats.json
├── checkpoints/
│   └── robodojo/
│       ├── sim/<run_name>/
│       └── real/<run_name>/
├── evaluate_results/
│   └── robodojo/
├── experiments/
│   └── robodojo/
└── third_party/
    └── RoboDojo/
        └── XPolicyLab/
```

## Install

Install FastWAM and the XPolicyLab package in the same policy environment:

```bash
cd <FastWAM>
bash experiments/robodojo/fastwam_policy/install.sh fastwam
conda activate fastwam
pip install -e third_party/RoboDojo/XPolicyLab
```

## Required roots

Set these paths before running RoboDojo evaluation. Keep them outside the
repository if they contain large datasets, assets, or checkpoints.

| Variable / config | Points to | Expected content |
|---|---|---|
| `FASTWAM_ROOT` | FastWAM repository root | This repository, with `src/`, `configs/`, and `experiments/`. |
| `EVALUATION.xpolicylab_root` | XPolicyLab repository root | Defaults to `third_party/RoboDojo/XPolicyLab`. |
| `EVALUATION.robodojo_root` / `ROBODOJO_ROOT` | RoboDojo simulator repository root | Defaults to `third_party/RoboDojo`. |
| `EVALUATION.robodojo_assets_root` / `ROBODOJO_ASSETS_ROOT` | RoboDojo assets root | Defaults to `data/robodojo/assets`. |
| `FASTWAM_CKPT_ROOT` | FastWAM RoboDojo checkpoint root | Defaults to `checkpoints/robodojo/sim` for sim and `checkpoints/robodojo/real` for real. |
| `FASTWAM_CHECKPOINT_PATH` | Optional explicit checkpoint file | A `.pt` checkpoint file. Overrides checkpoint discovery. |
| `FASTWAM_DATASET_STATS_PATH` | Optional explicit stats file | Matching `dataset_stats.json`. Overrides checkpoint-dir discovery. |

Example environment:

```bash
export FASTWAM_ROOT="$(pwd)"
export ROBODOJO_ROOT="$(pwd)/third_party/RoboDojo"
export ROBODOJO_ASSETS_ROOT="$(pwd)/data/robodojo/assets"
export FASTWAM_CKPT_ROOT="$(pwd)/checkpoints/robodojo/sim"
```

RoboDojo and XPolicyLab may use their own variable names for simulator code and
assets in a given release. If so, keep the official variables set as documented
by RoboDojo, and use the variables above for this FastWAM adapter.

Create the XPolicyLab policy link:

```bash
ln -sfn "$(pwd)/experiments/robodojo/fastwam_policy" \
  "$(pwd)/third_party/RoboDojo/XPolicyLab/policy/FastWAM"
```

Prepare a checkpoint directory with:

```text
checkpoints/robodojo/sim/<run_name>/
├── dataset_stats.json
└── checkpoints/
    └── weights/
        └── step_*.pt
```

You can also pass explicit files with `FASTWAM_CHECKPOINT_PATH` and
`FASTWAM_DATASET_STATS_PATH`.

## Same-machine smoke evaluation

```bash
cd third_party/RoboDojo/XPolicyLab/policy/FastWAM
FASTWAM_ROOT="$(git -C ../../../../.. rev-parse --show-toplevel)" \
ROBODOJO_ROOT="../../.." \
ROBODOJO_ASSETS_ROOT="../../../../../data/robodojo/assets" \
FASTWAM_CKPT_ROOT="../../../../../checkpoints/robodojo/sim" \
bash eval.sh RoboDojo stack_bowls <run_name> arx_x5 joint 0 0 1 fastwam RoboDojo
```

Arguments after `<run_name>` are:

```text
env_cfg_type action_type seed policy_gpu_id env_gpu_id policy_conda_env eval_env_conda_env
```

## Split policy server and simulator client

Policy machine:

```bash
cd third_party/RoboDojo/XPolicyLab/policy/FastWAM
FASTWAM_ROOT="$(git -C ../../../../.. rev-parse --show-toplevel)" \
ROBODOJO_ROOT="../../.." \
ROBODOJO_ASSETS_ROOT="../../../../../data/robodojo/assets" \
FASTWAM_CKPT_ROOT="../../../../../checkpoints/robodojo/sim" \
bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls <run_name> arx_x5 joint 0 \
  0 fastwam 6061 0.0.0.0
```

Simulator machine:

```bash
cd third_party/RoboDojo/XPolicyLab/policy/FastWAM
ROBODOJO_ROOT="../../.." \
ROBODOJO_ASSETS_ROOT="../../../../../data/robodojo/assets" \
bash setup_eval_env_client.sh \
  RoboDojo stack_bowls <run_name> arx_x5 joint 0 \
  1 RoboDojo "ckpt_name=<run_name>,action_type=joint" \
  6061 <policy_server_ip>
```

## Manager

The manager follows the same pattern as the LIBERO/RoboTwin managers in this
repository. It launches one policy server and one RoboDojo client per task.

```bash
cd <FastWAM>
export ROBODOJO_ROOT="$(pwd)/third_party/RoboDojo"
export ROBODOJO_ASSETS_ROOT="$(pwd)/data/robodojo/assets"
export FASTWAM_CKPT_ROOT="$(pwd)/checkpoints/robodojo/sim"

python experiments/robodojo/run_robodojo_manager.py \
  ckpt=<run_name> \
  EVALUATION.xpolicylab_root=third_party/RoboDojo/XPolicyLab \
  EVALUATION.policy_conda_env=fastwam \
  EVALUATION.eval_env_conda_env=RoboDojo \
  EVALUATION.dataset_stats_path=checkpoints/robodojo/sim/<run_name>/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

For real-robot deployment/evaluation, use the real config and real checkpoint
root. The same FastWAM policy server is used; XPolicyLab selects the real-world
client through `EVAL_ENV_TYPE=real_world`.

```bash
cd <FastWAM>
export FASTWAM_CKPT_ROOT="$(pwd)/checkpoints/robodojo/real"

python experiments/robodojo/run_robodojo_manager.py \
  --config-name real_robodojo \
  ckpt=<run_name> \
  EVALUATION.xpolicylab_root=third_party/RoboDojo/XPolicyLab \
  EVALUATION.policy_conda_env=fastwam \
  EVALUATION.eval_env_conda_env=RoboDojo \
  EVALUATION.dataset_stats_path=checkpoints/robodojo/real/<run_name>/dataset_stats.json
```

For released checkpoints or custom checkpoint layouts:

```bash
FASTWAM_CHECKPOINT_PATH=<checkpoint.pt> \
FASTWAM_DATASET_STATS_PATH=<dataset_stats.json> \
python experiments/robodojo/run_robodojo_manager.py \
  ckpt=<run_name> \
  EVALUATION.xpolicylab_root=third_party/RoboDojo/XPolicyLab \
  EVALUATION.policy_conda_env=fastwam \
  EVALUATION.eval_env_conda_env=RoboDojo
```
