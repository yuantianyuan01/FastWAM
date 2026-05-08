## FastWAM（Action-only + VFaR）跨服务器训练速查

本文是为了在当前机器无空闲时，方便在其它服务器 `git clone` 后快速复现实验环境与训练流程。

---

## 1. 环境与路径约定

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

建议统一设置（按需调整路径）：

```bash
export REPO=/path/to/FastWAM
export ASSET_ROOT=/data-ex/home/$USER/fastwam_assets
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/models
```

仓库内用软链接把大文件目录指过去：

```bash
cd "$REPO"
mkdir -p "$ASSET_ROOT"/{data,checkpoints,evaluate_results,raw,cache}
ln -sfn "$ASSET_ROOT/data" data
ln -sfn "$ASSET_ROOT/checkpoints" checkpoints
ln -sfn "$ASSET_ROOT/evaluate_results" evaluate_results
```

---

## 2. Wan2.2 5B 权重复用（推荐用软链接）

如果服务器上已有完整权重目录（示例）：
`/home/sxy/Wan2.2/models/Wan2.2-TI2V-5B`

则链接到 DiffSynth/loader 期望位置：

```bash
mkdir -p "$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI"
ln -sfn /home/sxy/Wan2.2/models/Wan2.2-TI2V-5B \
       "$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.2-TI2V-5B"
```

---

## 3. 数据集下载

### 3.1 LIBERO-fastwam

```bash
cd "$REPO"
export HF_HOME="$ASSET_ROOT/cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$ASSET_ROOT/raw/libero"
huggingface-cli download yuanty/LIBERO-fastwam --repo-type dataset \
  --local-dir "$ASSET_ROOT/raw/libero"

mkdir -p "$ASSET_ROOT/data/libero_mujoco3.3.2"
cd "$ASSET_ROOT/data/libero_mujoco3.3.2"
for f in "$ASSET_ROOT/raw/libero"/*.tar.gz; do tar -xzf "$f"; done
```

### 3.2 RoboTwin（robotwin2.0-fastwam）

```bash
cd "$REPO"
export HF_HOME="$ASSET_ROOT/cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$ASSET_ROOT/raw/robotwin2.0"
huggingface-cli download yuanty/robotwin2.0-fastwam --repo-type dataset \
  --local-dir "$ASSET_ROOT/raw/robotwin2.0"

mkdir -p "$ASSET_ROOT/data/robotwin2.0"
cd "$ASSET_ROOT/data/robotwin2.0"
cat "$ASSET_ROOT/raw/robotwin2.0"/robotwin2.0.tar.gz.part-* | tar -xzf -
```

---

## 4. 文本 embedding cache（必须）

LIBERO：

```bash
cd "$REPO"
python scripts/precompute_text_embeds.py \
  data=libero_2cam \
  model=fastwam \
  +overwrite=true
```

RoboTwin：

```bash
cd "$REPO"
python scripts/precompute_text_embeds.py \
  data=robotwin \
  model=fastwam \
  +overwrite=true
```

---

## 5. Model Preparation（对齐官方 FastWAM：ActionDiT backbone 预生成）

需要可用 CUDA：

如果缺少该文件，训练会在模型创建阶段直接报错：

- `FileNotFoundError: action_dit_pretrained_path does not exist: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`

`--output checkpoints/...` 是相对路径：如果仓库里的 `checkpoints` 是软链接，则会写到软链接指向的盘上目录。

```bash
cd "$REPO"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

---

## 6. 训练 sanity check（验证冻结 Wan，只训 decoder + ActionDiT）

```bash
cd "$REPO"
python scripts/train.py \
  data=libero_2cam \
  model=fastwam \
  batch_size=1 num_workers=0 \
  max_steps=1 log_every=1 \
  wandb.enabled=false
```

验证点：
- `train_mode=action_only_vfar` 时，optimizer 只包含 `vfar_decoder + action_expert (+ proprio_encoder 可选)`
- `video_expert/vae/text_encoder` 的 `requires_grad=False`

---

## 7. 实际训练启动输出（示例）

本段用于留档一次真实训练启动时的关键日志，便于复现实验与快速核对“冻结/可训练参数/分布式配置”是否符合预期。

### 7.1 关键结论（从日志可直接读出）

- 数据集归一化统计：完成一次遍历（`1712/1712`），用于生成/更新 `dataset_stats.json`。
- 分布式训练：`distributed_type=MULTI_GPU`，`world_size=2`，`mixed_precision=bf16`，`gradient_accumulation_steps=4`。
- 训练模式：`action_only_vfar` 已生效（只训 action 相关模块，冻结其它组件）。
- 可训练参数汇总：
  - `total_numel=6,727,259,363`
  - `trainable_numel=1,022,782,983`（与 `optimizer_numel` 一致，表示 optimizer 只接收了这些参数）
  - `trainable_by_top_level={action_expert=1,020,886,023, vfar_decoder=1,860,096, proprio_encoder=36,864}`
- 训练/验证集大小：`Train/val dataset size: 277713/277713`（当前配置未单独构建 val，val 与 train 指向同一数据集实例）。

### 7.2 日志片段（原样摘录）

```text
INFO  >> Calculating dataset stats for normalization...
Iterating dataset to get normalization: 100%|█████| 1712/1712 [00:18<00:00, 90.46it/s]
INFO  >> Accelerate training: distributed_type=MULTI_GPU zero_stage=none world_size=2 process_index=0 cfg_mixed_precision=bf16 accelerator_mixed_precision=bf16 grad_accum=4 grad_clip=1.0000
INFO  >> using accelerator.device=cuda:0
INFO  >> Setting action-only VFaR train mode and freezing other model components.
INFO  >> Trainable params summary: total_numel=6,727,259,363 trainable_numel=1,022,782,983 optimizer_numel=1,022,782,983 trainable_by_top_level={action_expert=1,020,886,023, vfar_decoder=1,860,096, proprio_encoder=36,864}
INFO  >> Trainable param names (first 20): action_expert.action_encoder.weight, action_expert.action_encoder.bias, ...
INFO  >> Train/val dataset size: 277713/277713
INFO  >> Starting training with max_steps=100000.
```

---

## 8. 长训练：W&B 监控 + tmux 启动/续训（2GPU 示例）

本段给出“可直接复制执行”的长训练命令（包含 W&B 同步、tmux 会话管理、以及从 checkpoint state 续训）。

> 注意：不要把 W&B API key 写进仓库文件/命令历史。推荐用交互方式登录，或在 shell 里临时导出环境变量。

### 8.1 一次性准备：W&B 登录（两选一）

交互登录（推荐，不会把 key 留在 shell history）：

```bash
conda activate fastwam
wandb login
```

或导出环境变量（自行替换 `<YOUR_WANDB_API_KEY>`）：

```bash
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
```

### 8.2 tmux 新开长训练（从 0 开始跑）

```bash
tmux new -s fastwam_libero_2gpu
```

在 tmux 里执行：

```bash
cd /home/sxy/embodied-ai/FastWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam
export DIFFSYNTH_MODEL_BASE_PATH=/home/sxy/embodied-ai/models

export RUN_DIR=/home/sxy/embodied-ai/FastWAM/runs/libero_2cam_fastwam_a100_2gpu

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  data=libero_2cam \
  model=fastwam \
  output_dir="$RUN_DIR" \
  batch_size=4 num_workers=8 \
  max_steps=100000 log_every=50 \
  save_every=1000 eval_every=2000 eval_num_inference_steps=10 \
  gradient_accumulation_steps=4 mixed_precision=bf16 \
  wandb.enabled=true wandb.project=fast-wam \
  wandb.name=libero_2cam_fastwam_a100_2gpu \
  wandb.group=libero_2cam_fastwam_a100 \
  wandb.mode=online
```

tmux 常用操作：

```bash
# 退出但不停止训练
# Ctrl-b, d

# 重新进入
tmux attach -t fastwam_libero_2gpu
```

### 8.3 tmux 续训（从最新 state 恢复）

```bash
cd /home/sxy/embodied-ai/FastWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam
export DIFFSYNTH_MODEL_BASE_PATH=/home/sxy/embodied-ai/models

export RUN_DIR=/home/sxy/embodied-ai/FastWAM/runs/libero_2cam_fastwam_a100_2gpu
export RESUME_DIR="$(ls -d "$RUN_DIR"/checkpoints/state/step_* 2>/dev/null | sort | tail -n 1)"
echo "RESUME_DIR=$RESUME_DIR"

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  data=libero_2cam \
  model=fastwam \
  output_dir="$RUN_DIR" \
  resume="$RESUME_DIR" \
  max_steps=100000 log_every=50 \
  save_every=1000 eval_every=2000 eval_num_inference_steps=10 \
  gradient_accumulation_steps=4 mixed_precision=bf16 \
  wandb.enabled=true wandb.project=fast-wam \
  wandb.name=libero_2cam_fastwam_a100_2gpu_resume \
  wandb.group=libero_2cam_fastwam_a100 \
  wandb.mode=online
```
