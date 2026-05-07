# FastWAM 数据流与改造笔记（Action-only + VFaR tokens + 冻结 Wan）

本文面向后续重构与实验迭代，整理当前 FastWAM（基于 Wan2.2-TI2V-5B + ActionDiT + MoT）的训练数据流、关键张量尺寸，以及推荐的最小可行改造路径（MVP）。

代码入口参考：
- 训练主路径：[FastWAM.training_loss](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/fastwam.py#L448-L568)
- Video expert（Wan2.2 DiT）tokenize 接口：[WanVideoDiT.pre_dit/post_dit](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/wan_video_dit.py#L509-L626)
- Action expert（ActionDiT）tokenize 接口：[ActionDiT.pre_dit/post_dit](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/action_dit.py#L226-L337)
- MoT 混合注意力实现：[MoT.forward](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/mot.py#L447-L556)
- Dataset 输出结构（LIBERO/RoboTwin 通用）：[RobotVideoDataset._get](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py#L115-L234)

---

## 1. 当前训练数据流（FastWAM 原生：video+action joint via MoT）

以一次 `training_loss(sample)` 为例，核心步骤如下：

1) **数据集输出（RobotVideoDataset）**
- `video`: `[B, 3, T_video, H, W]`，范围 `[-1,1]`
- `action`: `[B, T_action, action_dim]`
- `proprio`: `[B, T_action, proprio_dim]`
- `context`: `[B, L_ctx, 4096]`（来自缓存的 T5 embedding）
- `context_mask`: `[B, L_ctx]`（当前实现会强制变成全 1）

2) **VAE 编码视频为 latent**
- `input_latents = vae.encode(video)` -> `[B, 48, F, H_lat, W_lat]`
  - WanVideoVAE38 的参数：`z_dim=48`，`upsampling_factor=16`，`temporal_downsample_factor=4`
  - 因此：`H_lat = H/16`，`W_lat = W/16`，`F = 1 + (T_video-1)/4`
  - 参考：[WanVideoVAE38](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/wan_video_vae.py#L1355-L1384)

3) **可选：把 proprio 融入文本 context（1 个 token）**
- `proprio[:,0,:] -> Linear -> proprio_token: [B, 1, 4096]`
- `context = cat([context, proprio_token], dim=1)` -> `[B, L_ctx+1, 4096]`
- 参考：[FastWAM._append_proprio_to_context](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/fastwam.py#L219-L240)

4) **视频分支 tokenization（WanVideoDiT.pre_dit）**
- 输入 `x=input_latents: [B, 48, F, H_lat, W_lat]`
- patch 参数来自配置（默认 `patch_size=[1,2,2]`，`hidden_dim=3072`）
- `tokens_per_frame = (H_lat/2)*(W_lat/2)`
- `video_tokens = [B, Sv, 3072]`，其中 `Sv = F * tokens_per_frame`
- `video_freqs = [Sv, 1, rope_dim]`（3D RoPE，rope_dim 与实现相关）
- `video_t_mod = [B, Sv, 6, 3072]`（`seperated_timestep=true` 分支）
- `video_context = [B, L', 3072]`（`text_dim=4096 -> hidden_dim=3072` 的线性映射）
- `video_context_mask = [B, Sv, L']`
- 参考：[WanVideoDiT.pre_dit](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/wan_video_dit.py#L509-L620)

5) **动作分支 tokenization（ActionDiT.pre_dit）**
- `noisy_action: [B, Sa, action_dim]`（训练时对 action 加噪后得到）
- `action_tokens = [B, Sa, 1024]`（`action_encoder` 映射到 hidden_dim=1024）
- `action_freqs = [Sa, 1, 128]`（1D RoPE，attn_head_dim=128）
- `action_t_mod = [B, 6, 1024]`
- `action_context = [B, L', 1024]`（`text_dim=4096 -> hidden_dim=1024`）
- `action_context_mask = [B, Sa, L']`
- 参考：[ActionDiT.pre_dit](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/action_dit.py#L226-L299)

6) **MoT 混合自注意力（video/action 共同参与每一层 self-attn）**
- 将两路 tokens 在 sequence 维拼接后做一次 attention：
  - `S_total = Sv + Sa`
  - `attention_mask: [S_total, S_total]` 由 `_build_mot_attention_mask` 构建
- 输出再按各自长度切回去，并在各自 block 内继续做 cross-attn + MLP
- 参考：
  - [FastWAM._build_mot_attention_mask](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/fastwam.py#L385-L408)
  - [MoT.forward](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/mot.py#L447-L556)

7) **各 expert 的 post_dit 得到噪声预测**
- `pred_video = video_expert.post_dit(tokens_out["video"], video_pre)` -> `[B, 48, F, H_lat, W_lat]`
- `pred_action = action_expert.post_dit(tokens_out["action"], action_pre)` -> `[B, Sa, action_dim]`
- 参考：
  - [WanVideoDiT.post_dit](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/wan_video_dit.py#L622-L626)
  - [ActionDiT.post_dit](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/action_dit.py#L301-L302)

---

## 2. 关键张量尺寸（以默认 configs 为例）

默认数据配置（LIBERO/RoboTwin）：
- `num_frames=33`，`action_video_freq_ratio=4`，因此 `T_video = (33-1)/4 + 1 = 9`，`T_action = 32`
  - 参考：[libero_2cam.yaml](file:///home/sxy/embodied-ai/FastWAM/configs/data/libero_2cam.yaml#L28-L31)
  - 参考：[robotwin.yaml](file:///home/sxy/embodied-ai/FastWAM/configs/data/robotwin.yaml)

VAE（WanVideoVAE38）：
- `upsampling_factor=16`，`temporal_downsample_factor=4`
  - `H_lat = H/16`，`W_lat = W/16`，`F = 1 + (T_video-1)/4 = 3`
  - 参考：[WanVideoVAE38](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/wan_video_vae.py#L1355-L1384)

模型配置（关键超参）：
- video DiT：`hidden_dim=3072`，`patch_size=[1,2,2]`，`num_heads=24`，`attn_head_dim=128`，`num_layers=30`
- action DiT：`hidden_dim=1024`，`num_heads=24`，`attn_head_dim=128`，`num_layers=30`
- 参考：[configs/model/fastwam.yaml](file:///home/sxy/embodied-ai/FastWAM/configs/model/fastwam.yaml#L12-L46)

### 2.1 LIBERO（2 cam concat horizontal）

数据（来自 dataset 最终 video_size=[224,448]）：
- `video`: `[B, 3, 9, 224, 448]`
- `action`: `[B, 32, 7]`
- `proprio`: `[B, 32, 8]`
- `context`: `[B, 128, 4096]`

VAE latent：
- `input_latents`: `[B, 48, 3, 14, 28]`

video tokens（patch_size=[1,2,2]）：
- `tokens_per_frame = (14/2)*(28/2) = 7*14 = 98`
- `Sv = F*tokens_per_frame = 3*98 = 294`
- `video_tokens`: `[B, 294, 3072]`

action tokens：
- `Sa = 32`
- `action_tokens`: `[B, 32, 1024]`

MoT joint：
- `S_total = 294 + 32 = 326`
- `attention_mask`: `[326, 326]`

### 2.2 RoboTwin（3 cam -> robotwin layout）

数据（video_size=[384,320]）：
- `video`: `[B, 3, 9, 384, 320]`
- `action`: `[B, 32, 7]`
- `proprio`: `[B, 32, 8]`
- `context`: `[B, 128, 4096]`

VAE latent：
- `input_latents`: `[B, 48, 3, 24, 20]`

video tokens：
- `tokens_per_frame = (24/2)*(20/2) = 12*10 = 120`
- `Sv = 3*120 = 360`
- `video_tokens`: `[B, 360, 3072]`

action tokens：
- `Sa = 32`
- `action_tokens`: `[B, 32, 1024]`

MoT joint：
- `S_total = 360 + 32 = 392`
- `attention_mask`: `[392, 392]`

---

## 3. 是否绕开 MoT？以及训练成本差异（定量直觉）

结论建议：**你当前目标是“action-only + vfar tokens + 冻结 Wan”，绕开 MoT 更合适。**

原因不是“纯计算量立刻下降几个数量级”，而是：
- **实现复杂度/风险更低**：不用处理 MoT 内部“video K/V 无梯度、action Q 有梯度”的混合梯度边界与 checkpoint 逻辑。
- **显存更省、更稳定**：训练只保留 action expert + decoder 的反向图；Wan 的 forward 可以 `no_grad`。

### 3.1 纯 attention 计算量的差异（按序列长度估算）

MoT 的混合 attention（每层）复杂度近似与 `S_total^2` 成正比，其中 `S_total = Sv + Sa`。

如果你仍然需要跑一次 Wan forward 来产生 VFaR，那么你无论是否用 MoT，都会承担 video 分支的 token 序列长度 `Sv` 带来的主要计算。

以 LIBERO 为例（`Sv=294`，`Sa=32`）：
- separate（video self-attn + action self-attn）：`Sv^2 + Sa^2 = 294^2 + 32^2 = 86,436 + 1,024`
- MoT（混合 self-attn）：`(Sv+Sa)^2 = 326^2 = 106,276`
- MoT 相比 separate 的额外项约为 `2*Sv*Sa = 18,816`，占比约 `~21%`

以 RoboTwin 为例（`Sv=360`，`Sa=32`）：
- separate：`360^2 + 32^2 = 129,600 + 1,024`
- MoT：`392^2 = 153,664`
- 额外项约 `2*360*32=23,040`，占比约 `~18%`

因此：如果两条路线都要做 Wan forward，**绕开 MoT 对“纯 forward 算力”的收益大约在 10%~30% 量级**（取决于 `Sa/Sv`）。

### 3.2 训练显存/反向的差异（更关键）

真正的大差异来自“哪些模块需要梯度 + 反向图规模”：
- **MoT 路线（原生训练）**：trainer 在 `train_mode=dit_only` 时会训练 `model.dit`（FastWAM 中 `dit` 默认指向 `mot`），等价于训练整个 MoT（包含 video/action 两路 block），反向图很大（尤其 video hidden_dim=3072）。
  - 参考：[Wan22Trainer._apply_configured_train_mode](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/trainer.py#L287-L319)
  - 参考：[FastWAM.__init__ self.dit=self.mot](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/fastwam.py#L46-L47)
- **action-only + VFaR tokens**：Wan 只做 `no_grad` 前向取 VFaR；训练只对 decoder + action expert 反向，显存与梯度同步成本显著下降，稳定性更好。

---

## 4. 推荐的最小可行改造路径（MVP）

目标：**冻结 Wan（video expert / VAE / text encoder），只训练你定义的 decoder + action expert（ActionDiT）。**  
输出形式：将 Wan 在固定时间步（例如“t=O1”）的噪声预测（速度场先验，VFaR）经 decoder 变成 `8` 个 tokens，作为 action diffusion 的条件输入。

### 4.1 需要新增/明确的配置项（建议）

在 `configs/model/fastwam.yaml` 或新建一个分支配置里增加类似字段（名称可调整）：
- `train_mode: action_only_vfar`（用于路由 forward/training_loss）
- `freeze_wan: true`（冻结 video_expert/vae/text_encoder）
- `vfar:`
  - `enabled: true`
  - `timestep_strategy: fixed`（或 `from_scheduler_step0`）
  - `fixed_timestep: <float>`（你定义的 O1 映射到连续时间的值）
  - `num_tokens: 8`
  - `token_dim: 4096`（建议对齐 text_dim，方便拼到 context 上）
- `loss:`
  - `lambda_video: 0.0`（彻底关闭 video loss）

### 4.2 FastWAM 模型侧改动（建议拆分为子任务）

1) **新增模块：VFaR decoder**
- 新增 `self.vfar_decoder`，输入建议用 `pred_video` 或 `video_tokens`（二选一）：
  - `pred_video`: `[B,48,3,H_lat,W_lat]`
  - 或 `video_tokens`: `[B,Sv,3072]`
- 输出：`vfar_tokens: [B,8,4096]`（用于拼接到 `context`）

2) **新增 action-only 的训练入口**
- 新增 `training_loss_action_only_vfar(sample)`：
  - `no_grad` 跑 Wan（固定 timestep）得到 `pred_video`（或其 token）
  - `vfar_tokens = decoder(pred_video)` -> `[B,8,4096]`
  - `context' = cat([context, proprio_token?, vfar_tokens], dim=1)`
  - 用 ActionDiT 走扩散训练（保留现有 `train_action_scheduler` 逻辑即可）

3) **checkpoint 兼容**
- 目前 `save_checkpoint` 只保存 `mot`（见 [save_checkpoint](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/fastwam.py#L1088-L1099)）。
- action-only 后，至少要保存：
  - `action_expert`（或 action 分支相关权重）
  - `vfar_decoder`
  - `proprio_encoder`（如果启用）
- 同时保持能加载旧 checkpoint（例如 key 不存在时 warning）。

### 4.3 Trainer / Optimizer 改动（必要）

当前 trainer 写死只训练 `model.dit`（即 MoT），这与 action-only 目标冲突：
- 参考：[Wan22Trainer.__init__ optimizer 构建](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/trainer.py#L82-L95)

你需要把“可训练参数集合”改为：
- decoder 参数
- action expert 参数
- （可选）proprio_encoder 参数

并确保 Wan 的模块 `requires_grad=False`（并尽量 `eval()`），避免无意义的反向图与 dropout 行为。

---

## 4.4 当前已实现的版本（与论文/草案的差异点）

本仓库当前代码已经落地了一个可跑通的最小版本，目标是“action-only + vfar tokens + 冻结 Wan”，但 decoder 形态与论文草案中的 Perceiver 压缩器并不一致；这里把“实际实现是什么”明确写清楚，方便你后续逐步替换为论文版本。

### 4.4.1 训练模式与冻结策略（已实现）

- 配置入口：`configs/model/fastwam.yaml` 已新增 `train_mode/freeze_wan/vfar`。
- Runtime 会把这些配置传给模型并生效：`create_fastwam(...) -> model.configure_training(...)`。
  - 参考：[runtime.py](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/runtime.py#L76-L167)
- 当 `freeze_wan=true`：
  - `video_expert/vae/text_encoder` 会 `eval()` 并 `requires_grad_(False)`。
  - 参考：[FastWAM._freeze_wan_modules](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/fastwam.py)

### 4.4.2 VFaR 的定义（当前实现）

- 当前实现的 VFaR 输入是 Wan video expert 的 **输出噪声预测张量**（在 latent 空间），即 `pred_vfar`：
  - `pred_vfar` 形状为 `[B, 48, F, H_lat, W_lat]`。
- 当前实现的 timestep 取值语义：使用 `WanContinuousFlowMatchScheduler` 的 **timestep 标度**，其范围是 `[0, num_train_timesteps]`（例如 0~1000），不是论文公式里的 `[0,1]`。
  - 已对齐论文中的 “t=1.0（纯噪声端点）”：使用 `vfar.timestep_strategy: t1`，其效果等价于 `timestep = num_train_timesteps`（默认一般为 1000）。
  - 参考：[scheduler_continuous.py](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/schedulers/scheduler_continuous.py#L31-L56)

### 4.4.3 实际使用的 decoder（当前实现：VFaRPerceiverEncoder）

当前仓库使用的 decoder 已更新为 **VFaRPerceiverEncoder**（Perceiver 风格 learnable queries + cross-attn 压缩器）。

- 输入：`pred_vfar: [B,48,F,H_lat,W_lat]`
- 压缩方式：
  - `Conv3D(1x1x1)` 通道投影：`48 -> latent_dim(默认256)`
  - `AdaptiveAvgPool3D` 到固定 `(pool.t, pool.h, pool.w)`（默认 `3x2x2`）
  - flatten 为 `M = pool.t*pool.h*pool.w` 个 KV tokens，加上可学习位置编码 `pos_embed: [1,M,latent_dim]`
  - `num_tokens(=K)` 个 learnable queries（默认 8）对 KV tokens 做 cross-attn，得到 `K` 个 latent tokens
  - `Linear(latent_dim -> token_dim)` 映射到与 `context` 对齐的维度（默认 4096）
- 输出：`vfar_tokens: [B,8,4096]`
- 参考：[VFaRPerceiverEncoder](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/fastwam.py)

这意味着当前版本的 decoder **完全丢弃了 3D 空间拓扑**（与论文草案中“保留体素拓扑 + cross-attn 压缩”的设计不同）。它的主要价值在于先把训练/冻结/接口打通。

### 4.4.4 action-only 数据流（已实现）

当 `train_mode=action_only_vfar` 时：

1) `build_inputs()` 仍然会：
   - `vae.encode(video)` 得到 `input_latents`
   - `context/proprio` 处理得到 `context/context_mask`
2) `training_loss_action_only_vfar()` 会：
   - `no_grad` 调 Wan video expert 得到 `pred_vfar`
   - `vfar_decoder(pred_vfar)` 得到 `vfar_tokens: [B,8,4096]`
   - `context_aug = cat([context, vfar_tokens], dim=1)`
   - 仅用 `ActionDiT` 做 action diffusion 的噪声回归 loss
3) 不再执行 MoT，也不计算 video loss（当 `loss.lambda_video=0` 时）

参考：
- [FastWAM.training_loss_action_only_vfar](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/fastwam.py)
- [ActionDiT.forward](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/action_dit.py#L304-L337)

### 4.4.5 Trainer/Checkpoint 变更（已实现）

- Trainer 会根据 `model.train_mode` 选择可训练参数集合：
  - `action_only_vfar`：`action_expert + vfar_decoder + (optional) proprio_encoder`
  - 默认：`model.dit (+ optional proprio_encoder)`
  - 参考：[Wan22Trainer._collect_trainable_params](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/trainer.py#L320-L343)
- Checkpoint 现在额外保存：
  - `action_expert`、`vfar_decoder`、`vfar`、`train_mode`（并保持旧 ckpt 兼容）
  - 参考：[FastWAM.save_checkpoint/load_checkpoint](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/fastwam.py#L1188-L1222)

### 4.4.6 与论文草案中 decoder 的关系

当前实现已经是“Conv3D 投影 + AdaptivePool3D 保留体素布局 + learnable queries cross-attn 压缩为 K tokens”的形式；与草案的差异主要体现在：
- 当前默认 `pool=(3,2,2)` 与 `latent_dim=256` 为工程默认值，可按实验调整
- 当前实现输出是 `K` 个 tokens（`[B,K,4096]`）直接拼到 ActionDiT 的 `context`，暂未引入 ResNet-18 视觉分支与 FiLM 注入

## 5. 备注：当前 configs 下的一个关键事实（会影响你对 VFaR timestep 的定义）

在默认数据配置里：
- `T_video=9`，`T_action=32`
- VAE `temporal_downsample_factor=4`，因此 latent 帧数 `F=3`

也就是说：Wan video DiT 实际处理的是 `3` 帧 latent 的短序列（而不是 9 帧像素序列），这会显著影响你对“固定 timestep O1 的信息量”的直觉，以及 decoder 需要处理的时空结构规模。
