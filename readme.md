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

### 4.4.4 VFaR 提取时对“首帧硬条件”的处理（已实现）

当 `video_expert.fuse_vae_embedding_in_latents=true` 时，Wan 的 I2V 接口语义是“首帧作为硬条件锚点”。  
因此在提取 `t=1.0` 纯噪声端点的 VFaR 时，当前实现会：

- 先按 scheduler 把 `input_latents` 加噪到 `x_t`
- 然后将 `x_t[:,:,0:1]` 强行替换回 `first_frame_latents`（如果存在）

这样可以保证“纯噪声端点 + 首帧锚点”同时成立，避免首帧也被噪声破坏。

- 参考：[FastWAM._compute_vfar_tokens](file:///home/sxy/embodied-ai/FastWAM/src/fastwam/models/wan22/fastwam.py#L341-L370)

---

## 5. 当前需要确认的两项实验性检查（已跑通 Dummy）

### 5.1 Dummy forward：确认新增结构与数据流 I/O

在不依赖真实 Wan 权重/真实 dataset 的前提下，用 Dummy VAE + Dummy video expert 验证 action-only 路径：

- `train_mode=action_only_vfar` 能正常走通
- `vfar_tokens.shape == [B, 8, 4096]`（与 context 对齐）
- `loss.backward()` 后只有 `vfar_decoder + action_expert` 有梯度

一次运行结果示例（B=2）：
- `vfar_tokens.shape: (2, 8, 4096)`
- `requires_grad(video_expert)=False, requires_grad(vae)=False`
- `requires_grad(action_expert)=True, requires_grad(vfar_decoder)=True`

### 5.2 训练时是否“冻结 Wan 只训 decoder+action DiT”

结论：当前实现满足“冻结 Wan，只训练 decoder + action expert”的梯度边界：
- `freeze_wan=true` 会对 `video_expert/vae/text_encoder` 执行 `eval()` + `requires_grad_(False)`
- trainer 在 `train_mode=action_only_vfar` 时只将 `action_expert + vfar_decoder (+ proprio_encoder 可选)` 加入 optimizer

注意：要在 LIBERO 上真正启动训练进程，需要先把 `data/libero_mujoco3.3.2/...` 数据放到 `FastWAM/data` 指向的目录中（当前 data 目录为空）。

---

## 6. 改进想法：Adapter Decoder（从 Wan DiT 多层中间特征提取 VFaR）

当前 VFaR 直接使用 `pred_vfar`（最终输出速度场/噪声预测）。它的优点是物理意义明确、与 Rectified Flow 目标一致；但也可能存在两个局限：
- 输出空间受 VAE latent 表达约束，可能缺少更高层的语义/结构信息
- 单一输出可能在某些任务上不足以提供多尺度的“可操作性”线索

### 6.1 可行的 Adapter 设计空间

- **Layer feature adapter（取若干层的 token 激活）**
  - 从 Wan video DiT 的若干 block 输出中取 `h_l: [B,Sv,3072]`
  - 对每层做 `proj_l: 3072 -> d`，再做 pooling/perceiver 压缩到 `K` 个 token

- **Learned layer mixing（跨层加权/注意力融合）**
  - 学习一组层权重 `α_l`（softmax 归一）做 `h_mix = Σ α_l * h_l`
  - 或者对每个 token 做 gating：`α_l(token)`（更强但更贵）

- **Multi-scale tokens（浅层+深层同时保留）**
  - 浅层更偏纹理/局部几何，深层更偏语义/动作相关部位
  - 可以输出 `K1+K2` 个 token：例如 `K=8` 由 4 个浅层 token + 4 个深层 token 组成

- **Prediction + feature hybrid（输出+中间特征结合）**
  - 将 `pred_vfar`（物理速度场）作为“几何/动力学先验”
  - 同时附加中间层 `h_l`（语义/注意力聚合）作为“任务意图先验”

### 6.2 在本代码库里怎么落地（接口与改动点）

要实现中间层提取，需要让 Wan video expert 能“返回中间层特征”。建议路径：
- 在 `WanVideoDiT.forward(...)` 增加可选参数，例如：
  - `return_intermediates: bool`
  - `intermediate_layers: list[int]`
  - 返回 `pred` 的同时返回 `hidden_states: dict[layer_idx] = [B,Sv,3072]`
- 在 `FastWAM._compute_vfar_tokens` 中增加分支：
  - `source: pred`（现状）
  - `source: hidden_states`（走 adapter decoder）
  - `source: mixed`（pred + hidden 融合）

### 6.3 风险与成本（建议提前规避）

- **显存风险**：如果要返回多个层的 `hidden_states`，即使 Wan forward 用 `no_grad`，保存大 tensor 也会占显存/内存；建议只取 1~4 个层，并尽量在线聚合（例如只保存 `h_mix`）。
- **时序/分辨率对齐**：`Sv` 取决于 `F,H_lat,W_lat`，adapter 应优先使用 `AdaptivePool` 让输出 token 数固定，避免写死魔数。
- **与“首帧硬条件”的一致性**：如果输入是 `t=1` 端点，仍需保持首帧锚点（当前已实现），否则中间特征会失去空间 anchor。

### 6.4 推荐的最小 Adapter MVP（低风险）

先做一个最小实验版本验证价值：
- 只取 2 个层：`l_low`（如 5）+ `l_high`（如 25）
- 各自 `proj: 3072 -> 256`，各自走同一个 `VFaRPerceiverEncoder` 压缩成 `4` 个 token
- 拼成 `8` 个 token（维度仍映射到 4096），保持下游 ActionDiT 接口不变

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

---

## 7. Step-2：LIBERO 训练前的准备与一次最小训练验证（建议照做）

下面目标是跑通两件事：
- dataset + text embedding cache 完整可用（RobotVideoDataset 不会因为缺 cache 报错）
- `train_mode=action_only_vfar` 时确实冻结 Wan，只训练 `vfar_decoder + action_expert`

### 7.1 数据集下载与解压（LIBERO-fastwam）

按 FastWAM 的 dataset 约定，目录最终需要是：

`data/libero_mujoco3.3.2/{libero_10_no_noops_lerobot,libero_goal_no_noops_lerobot,libero_object_no_noops_lerobot,libero_spatial_no_noops_lerobot}`

推荐把大文件放到盘上目录，并让仓库的 `data/` 软链接指过去（例如你使用 `/data-ex/home/sxy/fastwam_assets/data`）。

```bash
cd /home/sxy/embodied-ai/FastWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam

ASSET_ROOT=/data-ex/home/sxy/fastwam_assets
mkdir -p "$ASSET_ROOT"/{raw,cache,data}

export HF_HOME="$ASSET_ROOT/cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

# 下载 4 个 tar.gz 到 raw/libero
mkdir -p "$ASSET_ROOT/raw/libero"
huggingface-cli download yuanty/LIBERO-fastwam --repo-type dataset \
  --local-dir "$ASSET_ROOT/raw/libero"

# 解压到 data/libero_mujoco3.3.2
mkdir -p "$ASSET_ROOT/data/libero_mujoco3.3.2"
cd "$ASSET_ROOT/data/libero_mujoco3.3.2"
for f in "$ASSET_ROOT/raw/libero"/*.tar.gz; do tar -xzf "$f"; done
```

### 7.1.1 RoboTwin 数据集下载与解压（robotwin2.0-fastwam）

按 FastWAM 的 dataset 约定，目录最终需要是：

`data/robotwin2.0/robotwin2.0/{data,meta,videos}`

下载与解压：

```bash
cd /home/sxy/embodied-ai/FastWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam

ASSET_ROOT=/data-ex/home/sxy/fastwam_assets
mkdir -p "$ASSET_ROOT"/{raw,cache,data}

export HF_HOME="$ASSET_ROOT/cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$ASSET_ROOT/raw/robotwin2.0"
huggingface-cli download yuanty/robotwin2.0-fastwam --repo-type dataset \
  --local-dir "$ASSET_ROOT/raw/robotwin2.0"

mkdir -p "$ASSET_ROOT/data/robotwin2.0"
cd "$ASSET_ROOT/data/robotwin2.0"
cat "$ASSET_ROOT/raw/robotwin2.0"/robotwin2.0.tar.gz.part-* | tar -xzf -
```

另外，`configs/data/robotwin.yaml` 默认使用：
`./data/robotwin2.0/dataset_stats.json`
作为 normalization 统计文件；可直接下载官方 release 的 stats 放到该路径（下一节给命令）。

### 7.2 文本 embedding cache 预计算（必须，否则 dataset 直接报错）

RobotVideoDataset 在取样时会读取：
`data/text_embeds_cache/libero/*.t5_len128.wan22ti2v5b.pt`
如果不存在会直接抛 `FileNotFoundError`。

```bash
cd /home/sxy/embodied-ai/FastWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam

export DIFFSYNTH_MODEL_BASE_PATH=/home/sxy/embodied-ai/models

python scripts/precompute_text_embeds.py \
  data=libero_2cam \
  model=fastwam \
  +overwrite=true
```

#### 7.2.1 如果本机已有 Wan2.2 5B 权重：用软链接复用（推荐）

如果你已经在本地有完整的 `Wan2.2-TI2V-5B` 权重目录（例如：`/home/sxy/Wan2.2/models/Wan2.2-TI2V-5B`），建议直接把它软链接到 `DIFFSYNTH_MODEL_BASE_PATH` 期望的位置，避免重复下载与缓存冲突：

```bash
mkdir -p /home/sxy/embodied-ai/models/Wan-AI

# 若目标已存在且不是软链接，先备份再链接
if [ -e /home/sxy/embodied-ai/models/Wan-AI/Wan2.2-TI2V-5B ] && [ ! -L /home/sxy/embodied-ai/models/Wan-AI/Wan2.2-TI2V-5B ]; then
  mv /home/sxy/embodied-ai/models/Wan-AI/Wan2.2-TI2V-5B \
     /home/sxy/embodied-ai/models/Wan-AI/Wan2.2-TI2V-5B.bak_$(date +%Y%m%d_%H%M%S)
fi

ln -sfn /home/sxy/Wan2.2/models/Wan2.2-TI2V-5B \
       /home/sxy/embodied-ai/models/Wan-AI/Wan2.2-TI2V-5B
```

如果你遇到 huggingface 的提示：
`Still waiting to acquire lock ... .lock`
通常原因是**另一个 huggingface-cli 进程正在下载同一个文件**；确认没有相关进程后，可以删除对应 `.lock` 文件并重试。

#### 7.2.2 RoboTwin 的 text embedding cache

RoboTwin 对应的 cache 默认目录是：
`data/text_embeds_cache/robotwin`

```bash
cd /home/sxy/embodied-ai/FastWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam

export DIFFSYNTH_MODEL_BASE_PATH=/home/sxy/embodied-ai/models

python scripts/precompute_text_embeds.py \
  data=robotwin \
  model=fastwam \
  +overwrite=true
```

### 7.3 Model Preparation：预生成 ActionDiT backbone（与 FastWAM 对齐建议做）

这一步会从 Wan22 DiT 插值得到 ActionDiT backbone 初始化权重（用于训练更稳定、对齐官方设置）。

```bash
cd /home/sxy/embodied-ai/FastWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam

export DIFFSYNTH_MODEL_BASE_PATH=/home/sxy/embodied-ai/models

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

注意：该脚本需要 CUDA 驱动与可用 GPU；若机器上 `torch.cuda.is_available()` 为 False，需要先把 NVIDIA driver / CUDA 环境修好再跑。

### 7.3.1 可选：下载官方 release 的权重（LIBERO / RoboTwin）

如果你想先对齐官方 inference/eval（或复用 stats），可以下载 release checkpoints：

```bash
cd /home/sxy/embodied-ai/FastWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam

ASSET_ROOT=/data-ex/home/sxy/fastwam_assets
mkdir -p "$ASSET_ROOT/checkpoints/fastwam_release"

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir "$ASSET_ROOT/checkpoints/fastwam_release"
```

其中 RoboTwin stats 建议额外放一份到 `data/robotwin2.0/dataset_stats.json`（以满足默认 config）：

```bash
cp "$ASSET_ROOT/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json" \
   "$ASSET_ROOT/data/robotwin2.0/dataset_stats.json"
```

### 7.4 启动一次最小训练（max_steps=1）验证“冻结 Wan 只训 decoder+ActionDiT”

```bash
cd /home/sxy/embodied-ai/FastWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam

export DIFFSYNTH_MODEL_BASE_PATH=/home/sxy/embodied-ai/models

python scripts/train.py \
  data=libero_2cam \
  model=fastwam \
  batch_size=1 num_workers=0 \
  max_steps=1 log_every=1 \
  wandb.enabled=false
```

验证点（观察日志/或直接在 trainer 里打断点看 optimizer 参数）：
- `video_expert/vae/text_encoder`：`requires_grad=False`
- optimizer param group 只包含：`vfar_decoder + action_expert (+ proprio_encoder 可选)`

---

## 8. 单模型多任务 vs 单任务多模型：当前版本的结论

你在 1D U-Net 的 DP 版本里“每个 task 训练一个模型”的根因是：**策略缺少语言条件**，所以只能靠“数据集选择/单任务训练”来指定目标。

本仓库当前的 action-only + VFaR 版本结论是：
- **支持训练一个模型覆盖多个任务（结论：是）**：数据集每条 sample 都带 `instruction`，并且会读取对应的 `context/context_mask`（T5 embedding cache）。ActionDiT 的去噪预测显式以 `context` 为条件，因此同一套权重可以在训练时学习“语言条件下的多任务映射”。
- **是否“真的能在 LIBERO/RoboTwin 上用一个模型覆盖所有任务”取决于训练数据与采样策略**：
  - 如果训练数据目录包含多个任务（例如 LIBERO 的四个 suite 目录同时喂进来），训练出的就是多任务模型。
  - 如果你只想做单任务训练，既可以只喂某个 task 的子集，也可以使用数据侧的过滤/override（例如 `override_instruction`）来退化成“单任务多模型”。
