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
