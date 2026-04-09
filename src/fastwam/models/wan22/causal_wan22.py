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

Reference: /scratch/cgao304/dev/lingbot-va/wan_va/modules/model.py
"""

import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

from .wan22 import Wan22Core
from .wan_video_dit import WanVideoDiT, sinusoidal_embedding_1d
from .helpers.gradient import gradient_checkpoint_forward
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Attention mask
# ---------------------------------------------------------------------------

def build_teacher_forcing_self_attn_mask(
    num_frames: int,
    tokens_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    """Build self-attention mask for teacher-forcing causal training.

    Token layout::

        [cond_frame_0 ... cond_frame_{F-1}  noisy_frame_0 ... noisy_frame_{F-1}]

    Each block (cond / noisy) has ``N = num_frames * tokens_per_frame`` tokens.

    Returns:
        Boolean [2N, 2N] mask (True = attend).
    """
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalWanVideoDiT(WanVideoDiT):
    """WanVideoDiT extended with a teacher-forcing forward pass.

    The regular ``forward()`` is unchanged and used for inference.
    ``forward_teacher_forcing()`` doubles the sequence into conditioning +
    noisy tokens and applies the teacher-forcing attention mask.
    """

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
        freqs_single = torch.cat(
            [
                self.freqs[0][:num_frames].view(num_frames, 1, 1, -1).expand(num_frames, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(num_frames, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(num_frames, h, w, -1),
            ],
            dim=-1,
        ).reshape(tokens_per_block, 1, -1).to(x_tokens.device)
        freqs = torch.cat([freqs_single, freqs_single], dim=0)  # [2N, 1, D]

        # --- teacher-forcing self-attention mask ---
        self_attn_mask = build_teacher_forcing_self_attn_mask(
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
                )
            else:
                x_tokens = block(
                    x_tokens, context, t_mod, freqs,
                    context_mask=context_mask,
                    self_attn_mask=self_attn_mask,
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

    def infer(
        self,
        prompt: Optional[str] = None,
        input_image: Optional[torch.Tensor] = None,
        num_frames: int = 1,
        *,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        **kwargs,
    ):
        """Inference with cached text-context support.

        When the dataset already provides pre-computed text embeddings
        (``sample['context']`` / ``sample['context_mask']``) there is no
        reason to spin up the T5 text encoder just to re-encode the prompt
        — and in our CausalWan22 pretraining setup the encoder isn't even
        loaded in the debug config. If ``context`` / ``context_mask`` are
        supplied, we monkey-patch ``self.encode_prompt`` for the duration
        of the parent ``infer`` call so it returns the cached tensors.
        ``text_cfg_scale`` is forced to 1.0 in that path because we have
        no cached negative-prompt embedding.
        """
        if context is None or context_mask is None:
            return super().infer(
                prompt=prompt,
                input_image=input_image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                **kwargs,
            )

        # Normalize cached context to 2D/1D as encode_prompt returns them.
        ctx = context.to(device=self.device, dtype=self.torch_dtype)
        mask = context_mask.to(device=self.device)
        if ctx.ndim == 3:
            ctx = ctx[0]
        if mask.ndim == 2:
            mask = mask[0]

        original_encode_prompt = self.encode_prompt
        self.encode_prompt = lambda _prompt: (ctx.unsqueeze(0), mask.unsqueeze(0))
        try:
            return super().infer(
                prompt="",  # ignored — patched encode_prompt returns cached tensors
                input_image=input_image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                text_cfg_scale=1.0,  # no negative embedding available
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                **kwargs,
            )
        finally:
            self.encode_prompt = original_encode_prompt

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

        # Per-(batch, frame) MSE then per-frame flow-matching weighting.
        loss_per_bf = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 3, 4))
        sample_weight = self.train_scheduler.training_weight(loss_timestep).to(
            loss_per_bf.device, dtype=loss_per_bf.dtype
        )
        loss_total = (loss_per_bf * sample_weight).mean()
        return loss_total, {"loss_video": float(loss_total.detach().item())}
