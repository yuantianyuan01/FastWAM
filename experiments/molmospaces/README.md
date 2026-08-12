# MolmoSpaces support

This directory documents the FastWAM data/training contract for MolmoSpaces and
the expected roots for official MolmoSpaces evaluation.

## Repository layout

```text
FastWAM/
├── configs/data/molmospaces.yaml
├── configs/task/molmospaces_uncond_3cam_384_1e-4.yaml
├── configs/sim_molmospaces.yaml
├── data/molmospaces/
│   ├── assets/
│   ├── cache/
│   ├── droid_molmoact2/
│   └── dataset_stats.json
├── checkpoints/molmospaces/
└── third_party/MolmoSpaces/
```

## Data contract

The default config expects a LeRobot-format DROID/MolmoAct-style dataset:

- cameras: `observation.images.exterior_1_left`,
  `observation.images.exterior_2_left`, `observation.images.wrist_left`
- FastWAM view order: exterior, second exterior, wrist
- action/state dimension: 8
- action layout: relative arm joint position (7), gripper (1)
- action horizon: 10

Large files should live under `data/molmospaces/` or be symlinked there. Do not
commit datasets, simulator assets, checkpoints, logs, or rendered videos.

If the original dataset already exists elsewhere, keep it in place and create a
relative symlink:

```bash
mkdir -p data/molmospaces
ln -sfn <MolmoAct2-DROID-Dataset> data/molmospaces/droid_molmoact2
ln -sfn <molmospaces_assets> data/molmospaces/assets
ln -sfn <molmospaces_cache> data/molmospaces/cache
```

## Training

Precompute text embeddings:

```bash
python scripts/precompute_text_embeds.py task=molmospaces_uncond_3cam_384_1e-4
```

Train with the standard FastWAM launcher:

```bash
bash scripts/train_zero1.sh 8 task=molmospaces_uncond_3cam_384_1e-4
```

For a new dataset conversion, set `data.train.pretrained_norm_stats=null` and
`data.val.pretrained_norm_stats=null` for the first run so FastWAM can generate
`dataset_stats.json`; then point both fields to the generated stats file.

## Evaluation

Install MolmoSpaces from `third_party/MolmoSpaces` following its official
README. The default FastWAM config targets the MolmoBot combined manipulation
set, with assets under `data/molmospaces/assets` and runtime caches under
`data/molmospaces/cache`.

Use the official MolmoSpaces leaderboard runner with the FastWAM adapter in
`experiments/molmospaces/fastwam_policy.py`.

Example command:

```bash
export FASTWAM_ROOT="$(pwd)"
export MLSPACES_ASSETS_DIR="$(pwd)/data/molmospaces/assets"
export MLSPACES_CACHE_DIR="$(pwd)/data/molmospaces/cache"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python third_party/MolmoSpaces/molmo_spaces/evaluation/eval_main.py \
  experiments.molmospaces.fastwam_policy:FastWAMMolmoSpacesEvalConfig \
  --benchmark_dir data/molmospaces/assets/benchmarks/<benchmark_name> \
  --checkpoint_path checkpoints/molmospaces/checkpoint.pt \
  --output_dir evaluate_results/molmospaces/<run_name> \
  --num_workers 1
```

The evaluation interface passes the three camera observations above to the
policy and executes denormalized 8-D relative joint actions.
