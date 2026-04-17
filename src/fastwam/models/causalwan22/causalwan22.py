"""
Teacher-forcing causal training for Wan22 video diffusion.

During training, the sequence is doubled into [conditioning_tokens, noisy_tokens].
Conditioning tokens are the (optionally noised) ground-truth latents that provide
context for denoising.  The teacher-forcing attention mask ensures:

  - cond  -> cond:  frame-wise causal   (frame i attends to frames <= i)
  - cond  -> noisy: blocked
  - noisy -> cond:  frame-wise strictly causal (frame i attends to cond frames < i)
  - noisy -> noisy: frame-wise diagonal  (frame i attends to same frame i only)

With probability ``noisy_cond_prob``, the conditioning latents are noised with
timesteps drawn uniformly from [cond_t_min, cond_t_max] * num_train_timesteps.
This makes the model robust to imperfect conditioning during autoregressive
inference where previous predictions (not GT) are used as context.

"""

import functools
from typing import Any

import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

from fastwam.models.wan22.wan22 import Wan22Core
from fastwam.models.wan22.wan_video_dit import (
    WanVideoDiT,
    sinusoidal_embedding_1d,
    create_block_mask,
)
from fastwam.models.wan22.helpers.gradient import gradient_checkpoint_forward
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalWanVideoDiT(WanVideoDiT):
    """WanVideoDiT extended with a teacher-forcing forward pass.

    The regular ``forward()`` is unchanged and used for inference.
    ``forward_teacher_forcing()`` doubles the sequence into conditioning +
    noisy tokens and applies the teacher-forcing attention mask.
    ``forward_with_per_frame_timesteps()`` supports per-frame timestep
    control for autoregressive generation.

    Args:
        use_flex_attention: If True (default), self-attention in the causal
            forward paths uses ``flex_attention`` with a block-sparse mask.
            If False, falls back to SDPA with the equivalent dense mask.
    """

    def __init__(self, *args, use_flex_attention: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_flex_attention = bool(use_flex_attention)

    @functools.lru_cache(maxsize=16)
    def _rope_freqs(self, f: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        return torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(device)

    # ------- dense self-attention masks (for SDPA fallback paths) ----------

    @functools.lru_cache(maxsize=16)
    def _tf_self_attn_mask(
        self, num_frames: int, tokens_per_frame: int, device: torch.device,
    ) -> torch.Tensor:
        """Dense ``[2N, 2N]`` teacher-forcing self-attention mask."""
        N = num_frames * tokens_per_frame
        total = 2 * N

        frame_idx = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)
        q_frame = frame_idx.unsqueeze(1)  # [N, 1]
        k_frame = frame_idx.unsqueeze(0)  # [1, N]

        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:N, :N] = q_frame >= k_frame   # cond  -> cond:  causal
        # mask[:N, N:] stays False           # cond  -> noisy: blocked
        mask[N:, :N] = q_frame > k_frame    # noisy -> cond:  strictly causal
        mask[N:, N:] = q_frame == k_frame   # noisy -> noisy: same frame only

        return mask

    @functools.lru_cache(maxsize=16)
    def _per_frame_causal_self_attn_mask(
        self, num_frames: int, tokens_per_frame: int, device: torch.device,
    ) -> torch.Tensor:
        """Dense ``[N, N]`` per-frame-causal self-attention mask.

        Cached wrapper around ``build_video_to_video_mask`` (inherited from
        ``WanVideoDiT``), kept for naming symmetry with the block-mask
        helpers below.
        """
        return self.build_video_to_video_mask(
            video_seq_len=num_frames * tokens_per_frame,
            video_tokens_per_frame=tokens_per_frame,
            device=device,
        )

    # ------- flex_attention block masks (fast self-attention path) ---------

    @functools.lru_cache(maxsize=16)
    def _tf_block_mask(
        self, num_frames: int, tokens_per_frame: int, device: torch.device,
    ):
        """flex_attention BlockMask for the ``[2N, 2N]`` teacher-forcing layout."""
        N = num_frames * tokens_per_frame
        TPF = tokens_per_frame

        def mask_mod(b, h, q_idx, kv_idx):
            q_half = q_idx // N               # 0 = cond, 1 = noisy
            k_half = kv_idx // N
            q_frame = (q_idx % N) // TPF
            k_frame = (kv_idx % N) // TPF
            cond_cond = (q_half == 0) & (k_half == 0) & (q_frame >= k_frame)
            noisy_cond = (q_half == 1) & (k_half == 0) & (q_frame > k_frame)
            noisy_noisy = (q_half == 1) & (k_half == 1) & (q_frame == k_frame)
            return cond_cond | noisy_cond | noisy_noisy

        return create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=2 * N, KV_LEN=2 * N, device=device,
        )

    @functools.lru_cache(maxsize=16)
    def _per_frame_causal_block_mask(
        self, num_frames: int, tokens_per_frame: int, device: torch.device,
    ):
        """flex_attention BlockMask for a per-frame-causal ``[N, N]`` self-attn."""
        N = num_frames * tokens_per_frame
        TPF = tokens_per_frame

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx // TPF) >= (kv_idx // TPF)

        return create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=device,
        )

    def forward_with_per_frame_timesteps(
        self,
        latents: torch.Tensor,              # [B, C, F, H, W]
        per_frame_timesteps: torch.Tensor,   # [B, F]
        context: torch.Tensor,              # [B, L, D]
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DiT forward with per-frame timestep control.

        For autoregressive inference: build ``[clean_0, ..., clean_{k-1},
        noisy_k]`` and use the ``per_frame_causal`` attention mask so that
        frame k only attends to frames 0..k.  Clean conditioning frames get
        timestep 0; the target frame gets the denoising timestep.

        Args:
            latents: Video latents ``[B, C, F, H, W]``.
            per_frame_timesteps: Per-frame timestep values ``[B, F]``.
            context: Text embeddings ``[B, L, D]``.
            context_mask: Boolean mask ``[B, L]`` for text tokens.

        Returns:
            Model prediction ``[B, C, F, H, W]``.
        """
        assert self.video_attention_mask_mode == "per_frame_causal"
        
        dummy_t = torch.zeros(
            latents.shape[0], dtype=latents.dtype, device=latents.device,
        )
        latents, _, context_mask = self._validate_forward_inputs(
            x=latents, timestep=dummy_t, context=context,
            context_mask=context_mask, action=None,
        )

        batch_size = latents.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        tokens_per_frame = (latents.shape[3] // patch_h) * (latents.shape[4] // patch_w)

        x = self.patchify(latents)
        f, h, w = x.shape[2:]
        seq_len = f * h * w

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        # Per-token timestep embeddings: broadcast [B, F] -> [B, F*tokens]
        token_timesteps = (
            per_frame_timesteps
            .unsqueeze(-1)
            .expand(batch_size, f, tokens_per_frame)
            .contiguous()
            .reshape(batch_size, -1)
        )
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        context = self.text_embedding(context)
        context_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)

        freqs = self._rope_freqs(f, h, w, x_tokens.device)

        if self.use_flex_attention:
            block_mask = self._per_frame_causal_block_mask(f, tokens_per_frame, x_tokens.device)
            self_attn_mask = None
        else:
            block_mask = None
            self_attn_mask = self._per_frame_causal_self_attn_mask(
                f, tokens_per_frame, x_tokens.device,
            )

        for block in self.blocks:
            x_tokens = block(
                x_tokens, context, t_mod, freqs,
                context_mask=context_mask,
                self_attn_mask=self_attn_mask,
                block_mask=block_mask,
            )

        x_out = self.head(x_tokens, t)
        return self.unpatchify(x_out, (f, h, w))

    def forward_teacher_forcing(
        self,
        noisy_latents: torch.Tensor,    # [B, C, F, H, W]
        cond_latents: torch.Tensor,     # [B, C, F, H, W]
        timestep: torch.Tensor,         # [B, F]
        cond_timestep: torch.Tensor,    # [B, F]
        context: torch.Tensor,          # [B, L, D]
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with teacher forcing.

        Args:
            noisy_latents: Noised video latents (frame 0 already fixed to
                the clean input image).
            cond_latents: Conditioning latents (possibly noised).
            timestep: Per-frame diffusion timestep ``[B, F]`` for the noisy
                branch. All tokens within one frame share a single t.
            cond_timestep: Per-frame diffusion timestep ``[B, F]`` for the
                conditioning branch (0 means fully clean).
            context: Text embeddings ``[B, L, D]``.
            context_mask: Boolean mask ``[B, L]`` for text tokens.

        Returns:
            Model prediction for the noisy frames, ``[B, C, F, H, W]``.
        """
        # Reuse the parent's x/context/context_mask validation with a 1D
        # placeholder timestep, since the parent validator hard-requires 1D.
        dummy_t = torch.zeros(
            noisy_latents.shape[0],
            dtype=noisy_latents.dtype,
            device=noisy_latents.device,
        )
        noisy_latents, _, context_mask = self._validate_forward_inputs(
            x=noisy_latents,
            timestep=dummy_t,
            context=context,
            context_mask=context_mask,
            action=None,
        )
        assert self.seperated_timestep and self.fuse_vae_embedding_in_latents, (
            "Teacher forcing requires seperated_timestep=True and "
            "fuse_vae_embedding_in_latents=True"
        )

        batch_size = noisy_latents.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        tokens_per_frame = (
            (noisy_latents.shape[3] // patch_h)
            * (noisy_latents.shape[4] // patch_w)
        )

        # --- patchify both branches ---
        noisy_patch = self.patchify(noisy_latents)  # [B, D, f, h, w]
        cond_patch = self.patchify(cond_latents)
        num_frames, h, w = noisy_patch.shape[2:]
        tokens_per_block = num_frames * h * w

        noisy_tokens = rearrange(noisy_patch, "b c f h w -> b (f h w) c").contiguous()
        cond_tokens = rearrange(cond_patch, "b c f h w -> b (f h w) c").contiguous()
        x_tokens = torch.cat([cond_tokens, noisy_tokens], dim=1)  # [B, 2N, D]

        # --- per-token timestep embeddings ---
        def _broadcast_ts(ts: torch.Tensor) -> torch.Tensor:
            """Broadcast a [B, F] timestep into [B, F, tokens_per_frame] tokens."""
            if ts.shape != (batch_size, num_frames):
                raise ValueError(
                    f"timestep must have shape [B, F]=[{batch_size}, {num_frames}], "
                    f"got {tuple(ts.shape)}"
                )
            return ts.unsqueeze(-1).expand(batch_size, num_frames, tokens_per_frame).contiguous()

        # Frame 0 is the clean input image on both branches, so its
        # per-token timestep is forced to 0.
        cond_ts = _broadcast_ts(cond_timestep)
        cond_ts[:, 0, :] = 0
        noisy_ts = _broadcast_ts(timestep)
        noisy_ts[:, 0, :] = 0
        token_timesteps = torch.cat(
            [cond_ts.reshape(batch_size, -1), noisy_ts.reshape(batch_size, -1)],
            dim=1,
        )  # [B, 2N]

        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        # --- text context ---
        context = self.text_embedding(context)  # [B, L, D]
        context_mask = context_mask.unsqueeze(1).expand(-1, 2 * tokens_per_block, -1)

        # --- RoPE: identical positions for cond and noisy copies ---
        freqs = self._rope_freqs(num_frames, h, w, x_tokens.device)
        freqs = torch.cat([freqs, freqs], dim=0)  # [2N, 1, D]

        # --- self-attention mask: prefer flex_attention block-sparse path ---
        if self.use_flex_attention:
            block_mask = self._tf_block_mask(
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                device=x_tokens.device,
            )
            self_attn_mask = None
        else:
            block_mask = None
            self_attn_mask = self._tf_self_attn_mask(
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                device=x_tokens.device,
            )

        # --- DiT blocks ---
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, context, t_mod, freqs,
                    context_mask=context_mask,
                    self_attn_mask=self_attn_mask,
                    block_mask=block_mask,
                )
            else:
                x_tokens = block(
                    x_tokens, context, t_mod, freqs,
                    context_mask=context_mask,
                    self_attn_mask=self_attn_mask,
                    block_mask=block_mask,
                )

        # --- extract noisy output, apply head, unpatchify ---
        noisy_out = x_tokens[:, tokens_per_block:]
        noisy_t = t[:, tokens_per_block:]
        x = self.head(noisy_out, noisy_t)
        return self.unpatchify(x, (num_frames, h, w))


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

class CausalWan22Core(Wan22Core):
    """Wan22Core with teacher-forcing causal training.

    During training each noisy frame is conditioned on (optionally noised)
    GT latents of all previous frames.  Inference uses the standard
    autoregressive forward inherited from ``Wan22Core``.

    Args:
        noisy_cond_prob: Probability of noising the conditioning latents.
        cond_t_min: Lower bound of the uniform conditioning timestep range
            (as a fraction of ``num_train_timesteps``).
        cond_t_max: Upper bound of the uniform conditioning timestep range.
    """

    def __init__(
        self,
        *args,
        noisy_cond_prob: float = 0.5,
        cond_t_min: float = 0.5,
        cond_t_max: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noisy_cond_prob = float(noisy_cond_prob)
        self.cond_t_min = float(cond_t_min)
        self.cond_t_max = float(cond_t_max)
        # Causal training/inference requires per_frame_causal masking in the DiT;
        # otherwise the default bidirectional mask leaks future frames into the
        # conditioning prefix during autoregressive rollout.
        dit_mask_mode = getattr(self.dit, "video_attention_mask_mode", None)
        if dit_mask_mode != "per_frame_causal":
            raise ValueError(
                "CausalWan22Core requires dit.video_attention_mask_mode == "
                f"'per_frame_causal', got '{dit_mask_mode}'."
            )

    @classmethod
    def from_causalwan22_pretrained(
        cls,
        device="cuda",
        torch_dtype=torch.bfloat16,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        redirect_common_files=True,
        dit_config: dict[str, Any] | None = None,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        skip_dit_load_from_pretrain: bool = False,
        load_text_encoder: bool = True,
        noisy_cond_prob: float = 0.5,
        cond_t_min: float = 0.5,
        cond_t_max: float = 1.0,
    ):
        if dit_config is None:
            raise ValueError("`dit_config` is required for CausalWan22Core.from_causalwan22_pretrained().")
        from fastwam.models.causalwan22.helpers.loader import load_causalwan22_ti2v_5b_components
        components = load_causalwan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        model = cls(
            dit=components.dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            train_shift=train_shift,
            infer_shift=infer_shift,
            num_train_timesteps=num_train_timesteps,
            noisy_cond_prob=noisy_cond_prob,
            cond_t_min=cond_t_min,
            cond_t_max=cond_t_max,
        )
        model.model_paths = {
            "dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
        }
        return model

    def _resolve_text_context(self, prompt, context, context_mask):
        """Return (ctx[1,L,D], cmask[1,L]) from either pre-computed tensors or a prompt."""
        device, dtype = self.device, self.torch_dtype
        if context is not None and context_mask is not None:
            ctx = context.to(device=device, dtype=dtype)
            cmask = context_mask.to(device=device)
            if ctx.ndim == 2:
                ctx = ctx.unsqueeze(0)
            if cmask.ndim == 1:
                cmask = cmask.unsqueeze(0)
            return ctx, cmask
        if prompt is not None:
            return self.encode_prompt(prompt)
        raise ValueError(
            "Either `prompt` or (`context`, `context_mask`) must be provided."
        )

    def build_inputs(self, sample, tiled: bool = False):
        """Build inputs for teacher-forcing training.

        Same as ``Wan22Core.build_inputs`` but uses pre-computed
        ``sample['context']`` / ``sample['context_mask']`` when present,
        instead of re-encoding the prompt. This allows debug/pretraining runs
        to skip loading the (heavy) T5 text encoder entirely.
        """
        if "context" not in sample or "context_mask" not in sample:
            # Fall back to the parent implementation (which calls encode_prompt).
            return super().build_inputs(sample, tiled=tiled)

        video = sample["video"]
        if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(
                f"`sample['video']` must be [B, 3, T, H, W], got {type(video)} "
                f"shape={tuple(video.shape) if isinstance(video, torch.Tensor) else 'N/A'}"
            )
        _, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")

        input_video = video.to(device=self.device, dtype=self.torch_dtype)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.dit, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        context = sample["context"].to(device=self.device, dtype=self.torch_dtype)
        context_mask = sample["context_mask"].to(device=self.device)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": None,
        }

    @torch.no_grad()
    def infer(
        self,
        input_image: torch.Tensor,
        num_frames: int,
        *,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        **kwargs,
    ):
        """Autoregressive inference, one latent frame at a time.

        For each latent frame *k* (k = 1, ..., F-1):

        1. Build ``[clean_0, ..., clean_{k-1}, noisy_k]``.
        2. Set per-frame timesteps: conditioning frames get t=0, the target
           frame gets the denoising timestep.
        3. Run the DiT forward with ``per_frame_causal`` attention.
        4. Extract prediction for frame k and apply the scheduler step.
        5. Append the denoised frame to the clean context.

        Args:
            input_image: First frame ``[1, 3, H, W]`` or ``[3, H, W]``
                in ``[-1, 1]``.
            num_frames: Number of **video** frames to generate (the method
                computes the corresponding number of latent frames
                internally via the VAE temporal factor).
            prompt: Text prompt (used when the T5 encoder is loaded).
            context: Pre-computed text embeddings ``[L, D]`` or
                ``[1, L, D]``.  Mutually exclusive with ``prompt``.
            context_mask: Boolean mask ``[L]`` or ``[1, L]``.
            num_inference_steps: Diffusion denoising steps per latent frame.
            seed: Random seed for reproducibility.
            rand_device: Device for the random generator.
            tiled: Whether to use tiled VAE encoding / decoding.

        Returns:
            Dict with key ``"video"`` containing a list of PIL Images.
        """
        self.eval()
        device = self.device
        dtype = self.torch_dtype
        ctx, cmask = self._resolve_text_context(prompt, context, context_mask)

        # --- encode first frame ---
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        input_image = input_image.to(device=device, dtype=dtype)
        first_frame_latent = self._encode_input_image_latents_tensor(
            input_image, tiled=tiled,
        )  # [1, C, 1, H, W]

        latent_h = first_frame_latent.shape[3]
        latent_w = first_frame_latent.shape[4]
        z_dim = first_frame_latent.shape[1]
        latent_t = (num_frames - 1) // self.vae.temporal_downsample_factor + 1

        logger.info(
            "Autoregressive generation: %d video frames -> %d latent frames "
            "(latent shape: [%d, %d, %d])",
            num_frames, latent_t, z_dim, latent_h, latent_w,
        )

        generated_latents = first_frame_latent  # [1, C, 1, H, W]

        for k in range(1, latent_t):
            logger.info("Generating latent frame %d / %d ...", k, latent_t - 1)

            timesteps, deltas = self.infer_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=dtype,
            )

            generator = (
                None if seed is None
                else torch.Generator(device=rand_device).manual_seed(
                    int(seed) * 1_000_003 + int(k)
                )
            )
            noisy_frame = torch.randn(
                (1, z_dim, 1, latent_h, latent_w),
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=device, dtype=dtype)

            for step_t, step_delta in zip(timesteps, deltas):
                latent_seq = torch.cat(
                    [generated_latents, noisy_frame], dim=2,
                )  # [1, C, k+1, H, W]

                per_frame_t = torch.zeros(1, k + 1, device=device, dtype=dtype)
                per_frame_t[:, k] = step_t

                pred = self.dit.forward_with_per_frame_timesteps(
                    latents=latent_seq,
                    per_frame_timesteps=per_frame_t,
                    context=ctx,
                    context_mask=cmask,
                )

                pred_k = pred[:, :, k : k + 1]
                noisy_frame = self.infer_scheduler.step(
                    pred_k, step_delta, noisy_frame,
                )

            generated_latents = torch.cat(
                [generated_latents, noisy_frame], dim=2,
            )

        frames = self._decode_latents(generated_latents, tiled=tiled)
        return {"video": frames}

    @torch.no_grad()
    def infer_teacher_forcing(
        self,
        video: torch.Tensor,
        *,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        """Teacher-forcing inference: denoise each frame conditioned on GT prefix.

        All latent frames are denoised in parallel using
        ``forward_teacher_forcing``.  The conditioning branch holds the
        clean GT latents throughout, so each noisy frame k is denoised
        conditioned on the ground-truth frames 0..k-1 (strictly causal).
        The resulting frames are independently generated and may not be
        temporally coherent.

        Args:
            video: Ground-truth video ``[1, 3, T, H, W]`` in ``[-1, 1]``.
            prompt: Text prompt (used when the T5 encoder is loaded).
            context: Pre-computed text embeddings ``[L, D]`` or
                ``[1, L, D]``.  Mutually exclusive with ``prompt``.
            context_mask: Boolean mask ``[L]`` or ``[1, L]``.
            num_inference_steps: Diffusion denoising steps.
            seed: Random seed for reproducibility.
            rand_device: Device for the random generator.
            tiled: Whether to use tiled VAE encoding / decoding.

        Returns:
            Dict with key ``"video"`` containing a list of PIL Images.
        """
        self.eval()
        device = self.device
        dtype = self.torch_dtype
        ctx, cmask = self._resolve_text_context(prompt, context, context_mask)

        # --- encode full GT video to latents ---
        if video.ndim == 4:
            video = video.unsqueeze(0)
        video = video.to(device=device, dtype=dtype)
        gt_latents = self._encode_video_latents(video, tiled=tiled)  # [1, C, F, H, W]
        first_frame_latents = gt_latents[:, :, 0:1]

        _, z_dim, num_latent_frames, latent_h, latent_w = gt_latents.shape

        logger.info(
            "Teacher-forcing inference: %d latent frames "
            "(latent shape: [%d, %d, %d])",
            num_latent_frames, z_dim, latent_h, latent_w,
        )

        # --- conditioning branch: clean GT throughout ---
        cond_latents = gt_latents
        cond_timestep = torch.zeros(
            1, num_latent_frames, device=device, dtype=dtype,
        )

        # --- noisy branch: start from pure noise ---
        generator = (
            None if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        noisy_latents = torch.randn(
            gt_latents.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        noisy_latents[:, :, 0:1] = first_frame_latents  # frame 0 always clean

        # --- denoising loop (all frames in parallel) ---
        timesteps, deltas = self.infer_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
        )

        for step_t, step_delta in zip(timesteps, deltas):
            timestep_bf = torch.full(
                (1, num_latent_frames), float(step_t),
                device=device, dtype=dtype,
            )
            timestep_bf[:, 0] = 0  # frame 0 always clean

            pred = self.dit.forward_teacher_forcing(
                noisy_latents=noisy_latents,
                cond_latents=cond_latents,
                timestep=timestep_bf,
                cond_timestep=cond_timestep,
                context=ctx,
                context_mask=cmask,
            )

            noisy_latents = self.infer_scheduler.step(
                pred, step_delta, noisy_latents,
            )
            noisy_latents[:, :, 0:1] = first_frame_latents  # fix frame 0

        frames = self._decode_latents(noisy_latents, tiled=tiled)
        return {"video": frames}

    def _sample_cond_timestep(
        self, batch_size: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample conditioning timestep uniformly from [cond_t_min, cond_t_max]."""
        u = torch.rand((batch_size,), device=device, dtype=dtype)
        u = u * (self.cond_t_max - self.cond_t_min) + self.cond_t_min
        return u * self.train_scheduler.num_train_timesteps

    def training_loss(self, sample, tiled=False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]  # [B, C, F, H, W] clean GT
        first_frame_latents = inputs["first_frame_latents"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]

        batch_size, _, num_frames = input_latents.shape[:3]
        device = self.device
        dtype = input_latents.dtype
        num_train_t = float(self.train_scheduler.num_train_timesteps)

        # 1. Per-frame diffusion timestep [B, F]. Each frame draws its own
        # t; tokens inside a frame share it.
        noise = torch.randn_like(input_latents)
        timestep = self.train_scheduler.sample_training_t(
            batch_size=batch_size * num_frames,
            device=device,
            dtype=dtype,
        ).reshape(batch_size, num_frames)

        sigma = (timestep.float() / num_train_t).to(dtype).view(batch_size, 1, num_frames, 1, 1)
        noisy_latents = (1.0 - sigma) * input_latents + sigma * noise
        target = noise - input_latents  # flow-matching target (timestep-independent)

        # 2. Frame 0 is fixed to the clean input image.
        if first_frame_latents is not None:
            noisy_latents[:, :, 0:1] = first_frame_latents

        # 3. Per-sequence independent Bernoulli(noisy_cond_prob) decides
        # whether each batch element's conditioning latents get noised.
        noise_mask = torch.rand(batch_size, device=device) < float(self.noisy_cond_prob)
        cond_timestep = self._sample_cond_timestep(batch_size, device, dtype) * noise_mask.to(dtype)  # [B]
        cond_sigma = (cond_timestep.float() / num_train_t).to(dtype).view(batch_size, 1, 1, 1, 1)
        cond_noise = torch.randn_like(input_latents)
        cond_latents = (1.0 - cond_sigma) * input_latents + cond_sigma * cond_noise

        if first_frame_latents is not None:
            cond_latents[:, :, 0:1] = first_frame_latents

        # forward_teacher_forcing expects [B, F] for both timesteps.
        cond_timestep_bf = cond_timestep.view(batch_size, 1).expand(batch_size, num_frames)

        # 4. Teacher-forcing forward.
        pred = self.dit.forward_teacher_forcing(
            noisy_latents=noisy_latents,
            cond_latents=cond_latents,
            timestep=timestep,
            cond_timestep=cond_timestep_bf,
            context=context,
            context_mask=context_mask,
        )

        # 5. Loss — drop frame 0 (it's the fixed conditioning image).
        if first_frame_latents is not None:
            pred = pred[:, :, 1:]
            target = target[:, :, 1:]
            loss_timestep = timestep[:, 1:]
        else:
            loss_timestep = timestep

        # Per-(batch, frame) MSE: reduce in compute dtype first, then cast
        # the small [B, F] tensor to fp32 for the weighted mean. Avoids
        # materializing a full [B, C, F, H, W] fp32 squared-error tensor
        # under bf16/fp16 autocast.
        loss_per_bf = (pred - target).pow(2).mean(dim=(1, 3, 4)).float()
        sample_weight = self.train_scheduler.training_weight(loss_timestep).to(
            loss_per_bf.device, dtype=loss_per_bf.dtype
        )
        loss_total = (loss_per_bf * sample_weight).mean()
        return loss_total, {"loss_video": loss_total.detach()}
