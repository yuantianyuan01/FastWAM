from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from fastwam.utils.logging_config import get_logger

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.fastwam import FastWAM

from fastwam.models.causalwan22.helpers.loader import load_causalwan22_ti2v_5b_components
from fastwam.models.causalwan22.causal_action_dit import CausalActionDiT

logger = get_logger(__name__)


class CausalWAM(FastWAM):
    """MoT causal world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        # specific to CausalWAM
        video_cond_noise_prob: float = 0.0,
        video_cond_t_min: float = 0.0,
        video_cond_t_max: float = 1.0,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        self.video_cond_noise_prob = video_cond_noise_prob
        self.video_cond_t_min = video_cond_t_min
        self.video_cond_t_max = video_cond_t_max
        
    @classmethod
    def from_causalwan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        video_dit_pretrained_path: str | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        video_cond_noise_prob: float = 0.0,
        video_cond_t_min: float = 0.0,
        video_cond_t_max: float = 1.0,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for CausalWAM.from_causalwan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for CausalWAM.")

        components = load_causalwan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        
        if video_dit_pretrained_path in (None, "", "None", "null"):
            logger.info("No checkpoint specified for video expert, using Pretrained WanVideoDiT weights.")
        elif Path(video_dit_pretrained_path).exists():
            logger.info("Loading checkpoint for video expert: %s", video_dit_pretrained_path)
            components.dit.load_checkpoint(video_dit_pretrained_path)
        else:
            raise FileNotFoundError(f"Checkpoint for video expert not found: {video_dit_pretrained_path}")

        video_expert = components.dit
        action_expert = CausalActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            video_cond_noise_prob=video_cond_noise_prob,
            video_cond_t_min=video_cond_t_min,
            video_cond_t_max=video_cond_t_max,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model
    
    def _build_tf_mot_attention_mask(
        self, 
        cond_video_seq_len: int,
        noisy_video_seq_len: int,
        noisy_action_seq_len: int,
        video_tokens_per_frame: int,
        action_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if noisy_video_seq_len != cond_video_seq_len:
            raise ValueError(
                "Teacher-forcing requires identical `seq_len` for noisy and cond video branches, "
                f"got {noisy_video_seq_len} and {cond_video_seq_len}."
            )
        if noisy_video_seq_len % video_tokens_per_frame != 0:
            raise ValueError(
                "`noisy_video_seq_len` must be divisible by `noisy_video_tokens_per_frame` in "
                f"`teacher_forcing` mode, got {noisy_video_seq_len} and {video_tokens_per_frame}."
            )
        num_frames = noisy_video_seq_len // video_tokens_per_frame
        if num_frames < 2:
            raise ValueError(
                "Teacher-forcing with action tokens requires at least 2 video frames."
            )

        V = noisy_video_seq_len
        A = noisy_action_seq_len
        total = 2 * V + A

        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        frame_idx = torch.arange(num_frames, device=device).repeat_interleave(video_tokens_per_frame)
        q_frame = frame_idx.unsqueeze(1)
        k_frame = frame_idx.unsqueeze(0)

        action_idx = torch.arange(num_frames - 1, device=device).repeat_interleave(action_tokens_per_frame)
        q_action = action_idx.unsqueeze(1)
        k_action = action_idx.unsqueeze(0)

        cond_start, cond_end = 0, V
        noisy_start, noisy_end = V, 2 * V
        action_start, action_end = 2 * V, 2 * V + A

        # cond -> cond: causal by frame
        mask[cond_start:cond_end, cond_start:cond_end] = (q_frame >= k_frame)
        # noisy -> cond: strictly past frames only
        mask[noisy_start:noisy_end, cond_start:cond_end] = (q_frame > k_frame)
        # noisy -> noisy: same frame only
        mask[noisy_start:noisy_end, noisy_start:noisy_end] = (q_frame == k_frame)
        # action -> cond: action_t can see cond_t and cond_{t+1}
        mask[action_start:action_end, cond_start:cond_end] = (
            (q_action <= k_frame) & ((q_action + 1) >= k_frame)
        )
        # action -> action: same transition block only
        mask[action_start:action_end, action_start:action_end] = (q_action == k_action)

        return mask
    
    def _sample_cond_timestep(
        self, 
        batch_size: int, 
        device: torch.device, 
        dtype: torch.dtype,
        cond_t_min: float = 0.0,
        cond_t_max: float = 1.0,
        num_train_timesteps: int = 1000,
    ) -> torch.Tensor:
        """Sample conditioning timestep uniformly from [cond_t_min, cond_t_max]."""
        u = torch.rand((batch_size,), device=device, dtype=dtype)
        u = u * (cond_t_max - cond_t_min) + cond_t_min
        return u * num_train_timesteps
    
    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample flow-matching video loss with per-frame weighting.

        Unlike ``FastWAM._compute_video_loss_per_sample``, every video
        frame here has its own diffusion timestep, so the scheduler
        weight must be applied per-frame *before* reducing to per-sample.
        Padded frames are excluded from both numerator and denominator.

        Args:
            pred_video: ``[B, C, F, H, W]`` model output.
            target_video: ``[B, C, F, H, W]`` flow-matching target.
            image_is_pad: ``[B, raw_frames]`` bool mask at raw-frame
                resolution, or ``None`` if every frame is valid.
            include_initial_video_step: whether the first latent frame
                corresponds to the first raw image frame. Set ``False``
                when the first frame has been sliced off upstream (e.g.
                when using ``first_frame_latents`` as a clean prefix).
            timestep: ``[B, F]`` per-frame diffusion timestep used to
                look up scheduler weights.

        Returns:
            loss_per_sample: ``[B]`` mean over valid frames of ``w_i * L_i``.
        """
        # MSE averaged over channel/spatial dims -> [B, F].
        video_loss_per_frame = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none",
        ).mean(dim=(1, 3, 4))

        if timestep.shape != video_loss_per_frame.shape:
            raise ValueError(
                "`timestep` must have shape [B, F] matching the per-frame loss, "
                f"got {tuple(timestep.shape)} vs {tuple(video_loss_per_frame.shape)}."
            )

        # Valid-frame mask aligned to latent resolution.
        if image_is_pad is None:
            valid = torch.ones_like(video_loss_per_frame)
        else:
            temporal_factor = int(self.vae.temporal_downsample_factor)
            if temporal_factor <= 0:
                raise ValueError(
                    f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}."
                )
            if image_is_pad.shape[1] < 1:
                raise ValueError("`image_is_pad` must contain at least one frame.")
            if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
                raise ValueError(
                    "Cannot align `image_is_pad` with video latent steps: "
                    f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
                )

            tail_is_pad = image_is_pad[:, 1:]
            latent_tail_is_pad = tail_is_pad.view(
                image_is_pad.shape[0], -1, temporal_factor,
            ).all(dim=2)
            if include_initial_video_step:
                video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
            else:
                video_is_pad = latent_tail_is_pad

            if video_is_pad.shape[1] != video_loss_per_frame.shape[1]:
                raise ValueError(
                    "Video-loss mask shape mismatch: "
                    f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_per_frame.shape[1]}."
                )

            valid = (~video_is_pad).to(
                device=video_loss_per_frame.device, dtype=video_loss_per_frame.dtype,
            )

        # Per-frame flow-matching weight, then reduce to [B] as a
        # valid-frame-count-normalized mean of (weight * loss).
        video_weight = self.train_video_scheduler.training_weight(timestep).to(
            device=video_loss_per_frame.device, dtype=video_loss_per_frame.dtype,
        )
        weighted = video_loss_per_frame * video_weight * valid
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return weighted.sum(dim=1) / valid_sum

    def _compute_action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample flow-matching action loss with per-frame weighting.

        Unlike the video helper — which collapses padding to per-frame and
        averages chunk-first — here the valid mask stays at the per-
        action-token resolution of ``action_is_pad``. A partially-padded
        frame still contributes its unpadded tokens. The per-frame
        scheduler weight is broadcast across the ``K`` action tokens of
        each frame, then applied to the per-token loss; the per-sample
        reduction is a mean over the (frame, token) positions that are
        not padded.

        Args:
            pred_action: ``[B, F, K, A]`` model output, where ``F`` is the
                number of action frames and ``K`` is the number of action
                tokens per frame. A flat ``[B, F*K, A]`` shape is also
                accepted and reshaped internally using
                ``F = timestep.shape[1]``.
            target_action: ``[B, F, K, A]`` flow-matching target.
            action_is_pad: ``[B, F*K]`` bool mask at action-token
                resolution (flat across frames), or ``None``. Padded
                tokens are excluded from both numerator and denominator.
            timestep: ``[B, F]`` per-frame diffusion timestep used to
                look up scheduler weights.

        Returns:
            loss_per_sample: ``[B]`` mean over valid action tokens of
                ``w_{f} * L_{f,k}``.
        """
        if target_action.ndim != 4:
            raise ValueError(
                "`target_action` must be 4D [B, F, K, A], "
                f"got shape {tuple(target_action.shape)}"
            )
        if timestep.ndim != 2 or timestep.shape[0] != target_action.shape[0] \
                or timestep.shape[1] != target_action.shape[1]:
            raise ValueError(
                "`timestep` must be [B, F] matching the target, "
                f"got {tuple(timestep.shape)} vs target {tuple(target_action.shape)}"
            )
        batch_size, num_frames, action_num_per_frame, action_dim = target_action.shape

        # Accept either [B, F, K, A] or [B, F*K, A] for pred_action.
        if pred_action.ndim == 3:
            if pred_action.shape != (batch_size, num_frames * action_num_per_frame, action_dim):
                raise ValueError(
                    "Flat `pred_action` shape does not match target: "
                    f"got {tuple(pred_action.shape)}, expected "
                    f"{(batch_size, num_frames * action_num_per_frame, action_dim)}."
                )
            pred_action = pred_action.reshape(
                batch_size, num_frames, action_num_per_frame, action_dim,
            )
        elif pred_action.shape != target_action.shape:
            raise ValueError(
                "`pred_action` shape must match `target_action`, "
                f"got {tuple(pred_action.shape)} vs {tuple(target_action.shape)}."
            )

        # Per-token MSE reduced over action_dim only -> [B, F, K].
        action_loss_per_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none",
        ).mean(dim=3)

        # Per-frame flow-matching weight, broadcast to per-token [B, F, K].
        action_weight = self.train_action_scheduler.training_weight(timestep).to(
            device=action_loss_per_token.device, dtype=action_loss_per_token.dtype,
        )  # [B, F]
        action_weight_per_token = action_weight.unsqueeze(-1).expand(
            batch_size, num_frames, action_num_per_frame,
        )

        # Per-token valid mask from flat `action_is_pad` -> [B, F, K].
        if action_is_pad is None:
            valid = torch.ones_like(action_loss_per_token)
        else:
            expected_flat = (batch_size, num_frames * action_num_per_frame)
            if action_is_pad.shape != expected_flat:
                raise ValueError(
                    "`action_is_pad` must be [B, F*K] aligned with flat action tokens, "
                    f"got {tuple(action_is_pad.shape)}, expected {expected_flat}."
                )
            valid = (~action_is_pad).view(
                batch_size, num_frames, action_num_per_frame,
            ).to(device=action_loss_per_token.device, dtype=action_loss_per_token.dtype)

        # Weight + mask + mean over valid (frame, token) positions.
        weighted = action_loss_per_token * action_weight_per_token * valid
        valid_sum = valid.sum(dim=(1, 2)).clamp(min=1.0)
        return weighted.sum(dim=(1, 2)) / valid_sum
    
    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]
        
        device, dtype = self.device, input_latents.dtype
        batch_size, _, num_frames = input_latents.shape[:3]
        
        # Branch A: cond video
        cond_noise_mask = torch.rand((batch_size,), device=self.device) < float(self.video_cond_noise_prob)
        cond_timestep = self._sample_cond_timestep(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
            cond_t_min=self.video_cond_t_min,
            cond_t_max=self.video_cond_t_max,
            num_train_timesteps=self.train_video_scheduler.num_train_timesteps,
        )
        cond_timestep = torch.where(
            cond_noise_mask, cond_timestep, 0.0
        )
        cond_timestep = cond_timestep.view(batch_size, 1).expand(batch_size, num_frames).contiguous()
        cond_noise = torch.randn_like(input_latents)
        sigma_cond = (cond_timestep.float() / self.train_video_scheduler.num_train_timesteps).to(dtype).view(batch_size, 1, num_frames, 1, 1)
        cond_latents = (1.0 - sigma_cond) * input_latents + sigma_cond * cond_noise

        if inputs["first_frame_latents"] is not None:
            cond_timestep[:, 0:1] = 0.0
            cond_latents[:, :, 0:1] = inputs["first_frame_latents"]

        # Branch B: noisy video
        noisy_video_timestep = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size * num_frames, 
            device=device, 
            dtype=dtype, 
        ).reshape(batch_size, num_frames)
        # did not use add_noise function here because we want to sample the noise independently for each frame
        noisy_video_noise = torch.randn_like(input_latents)
        sigma_video = (noisy_video_timestep.float() / self.train_video_scheduler.num_train_timesteps).to(dtype).view(batch_size, 1, num_frames, 1, 1)
        noisy_video_latents = (1.0 - sigma_video) * input_latents + sigma_video * noisy_video_noise
        target_video = noisy_video_noise - input_latents

        if inputs["first_frame_latents"] is not None:
            noisy_video_timestep[:, 0:1] = 0.0
            noisy_video_latents[:, :, 0:1] = inputs["first_frame_latents"]

        # Branch C: noisy action
        num_chunks = num_frames - 1
        action = action.view(batch_size, num_chunks, -1, action.shape[-1]) # [B, num_chunk, chunk_size, A]
        noisy_action_timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size * num_chunks,
            device=device,
            dtype=action.dtype,
        ).reshape(batch_size, num_chunks)
        noisy_action_noise = torch.randn_like(action)
        sigma_action = (noisy_action_timestep.float() / self.train_action_scheduler.num_train_timesteps).to(dtype).view(batch_size, num_chunks, 1, 1)
        noisy_action_latents = (1.0 - sigma_action) * action + sigma_action * noisy_action_noise
        target_action = noisy_action_noise - action
        
        video_pre = self.video_expert.pre_dit_tf(
            cond_latents=cond_latents, 
            noisy_latents=noisy_video_latents, 
            cond_timestep=cond_timestep, 
            noisy_timestep=noisy_video_timestep, 
            context=context, 
            context_mask=context_mask, 
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action_latents,
            timestep=noisy_action_timestep,
            context=context,
            context_mask=context_mask,
        )

        if video_pre["t_mod"].ndim != 4:
            raise ValueError(
                "Teacher-forcing requires token-wise `t_mod`; "
                "ensure `seperated_timestep=true` and `fuse_vae_embedding_in_latents=true`."
            )

        cond_video_seq_len = int(video_pre["cond_tokens"].shape[1])
        noisy_video_seq_len = int(video_pre["noisy_tokens"].shape[1])
        noisy_action_seq_len = int(action_pre["tokens"].shape[1])
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        action_tokens_per_frame = int(action_pre["meta"]["tokens_per_frame"])

        attention_mask = self._build_tf_mot_attention_mask(
            cond_video_seq_len=cond_video_seq_len,
            noisy_video_seq_len=noisy_video_seq_len,
            noisy_action_seq_len=noisy_action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame, 
            action_tokens_per_frame=action_tokens_per_frame, 
            device=device, 
        )
        
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            }, 
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            attention_mask=attention_mask,
        )

        # extract the predictions
        pred_video_tokens = tokens_out["video"][:, cond_video_seq_len:]
        pred_video = self.video_expert.post_dit_tf(pred_video_tokens, video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
            video_loss_timestep = noisy_video_timestep[:, 1:]

        # Per-frame flow-matching weight + padding are applied inside the
        # helper because each frame carries its own diffusion timestep.
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=inputs["first_frame_latents"] is None,
            timestep=video_loss_timestep,
        )
        loss_video = loss_video_per_sample.mean()

        # Per-chunk flow-matching weight + padding are applied inside the
        # helper because each chunk carries its own diffusion timestep.
        loss_action_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            timestep=noisy_action_timestep,
        )
        loss_action = loss_action_per_sample.mean()

        # total loss
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict