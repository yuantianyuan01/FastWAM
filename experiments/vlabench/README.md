# VLABench support

This directory documents the FastWAM data/training contract for VLABench and
the expected roots for official VLABench evaluation.

## Repository layout

```text
FastWAM/
├── configs/data/vlabench.yaml
├── configs/task/vlabench_uncond_3cam_384_1e-4.yaml
├── configs/sim_vlabench.yaml
├── data/vlabench/
│   ├── assets/
│   ├── primitive_ft_lerobot_video/
│   └── dataset_stats.json
├── checkpoints/vlabench/
└── third_party/VLABench/
```

## Data contract

The default config expects the original LeRobot-format VLABench primitive
fine-tuning dataset:

- cameras: `image`, `second_image`, `wrist_image`
- FastWAM view order: front, side, wrist
- image resolution in the original data: 480 x 480
- action/state dimension: 7
- action layout: absolute EEF position (3), Euler rotation (3), gripper (1)
- action horizon: 32

Large files should live under `data/vlabench/` or be symlinked there. Do not
commit datasets, simulator assets, checkpoints, logs, or rendered videos.

If the original dataset already exists elsewhere, keep it in place and create a
relative symlink:

```bash
mkdir -p data/vlabench
ln -sfn <vlabench_primitive_ft_lerobot_video> \
  data/vlabench/primitive_ft_lerobot_video
ln -sfn <vlabench_assets> data/vlabench/assets
```

## Training

Precompute text embeddings:

```bash
python scripts/precompute_text_embeds.py task=vlabench_uncond_3cam_384_1e-4
```

Train with the same FastWAM launcher used by the existing LIBERO/RoboTwin
configs:

```bash
bash scripts/train_zero1.sh 8 task=vlabench_uncond_3cam_384_1e-4
```

For a new dataset conversion, set `data.train.pretrained_norm_stats=null` and
`data.val.pretrained_norm_stats=null` for the first run so FastWAM can generate
`dataset_stats.json`; then point both fields to the generated stats file for
reproducible subsequent runs.

## Evaluation

Install VLABench from `third_party/VLABench` following its official README. The
FastWAM policy adapter is provided in
`experiments/vlabench/fastwam_policy.py`; benchmark assets are expected under
`data/vlabench/assets`.

The evaluation interface should pass normalized observations to FastWAM and
execute denormalized absolute EEF actions with the same 7-D layout documented
above.

Example adapter construction inside a VLABench evaluation script:

```python
from experiments.vlabench.fastwam_policy import FastWAMVLABenchPolicy

policy = FastWAMVLABenchPolicy(
    checkpoint_path="checkpoints/vlabench/checkpoint.pt",
    dataset_stats_path="data/vlabench/dataset_stats.json",
)
```

Full evaluation can be sharded across GPUs with:

```bash
bash experiments/vlabench/run_vlabench_parallel.sh \
  checkpoints/vlabench/checkpoint.pt \
  data/vlabench/dataset_stats.json
```
