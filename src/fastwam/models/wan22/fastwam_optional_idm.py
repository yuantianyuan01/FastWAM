from typing import Any, Optional

import torch

from .fastwam import FastWAM
from .fastwam_idm import FastWAMIDM


class FastWAMOptionalIDM(FastWAMIDM):
    """FastWAM variant where full-video IDM conditioning is optional."""

    action_idm_prob: float

    @classmethod
    def from_wan22_pretrained(cls, *, action_idm_prob: float, **kwargs):
        prob = float(action_idm_prob)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"`action_idm_prob` must be in [0, 1], got {prob}.")
        model = super().from_wan22_pretrained(**kwargs)
        model.action_idm_prob = prob
        return model

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        full_cond_mask = super()._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_seq_len,
            noisy_video_tokens_per_frame=noisy_video_tokens_per_frame,
            cond_video_tokens_per_frame=cond_video_tokens_per_frame,
            batch_size=batch_size,
            device=device,
        )

        noisy_end = noisy_video_seq_len
        cond_end = noisy_video_seq_len + cond_video_seq_len
        first_frame_tokens = min(cond_video_tokens_per_frame, cond_video_seq_len)
        first_frame_cond_mask = full_cond_mask.clone()
        first_frame_cond_mask[cond_end:, noisy_end:cond_end] = False
        first_frame_cond_mask[
            cond_end:,
            noisy_end : noisy_end + first_frame_tokens,
        ] = True

        use_idm_mask = (
            torch.rand((batch_size, 1, 1, 1), device=device) < self.action_idm_prob
        )
        return torch.where(use_idm_mask, full_cond_mask, first_frame_cond_mask)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        action_infer_mode: str = "idm",
    ) -> dict[str, Any]:
        if action_infer_mode == "idm":
            if num_video_frames is None:
                raise ValueError("`num_video_frames` is required for `action_infer_mode='idm'`.")
            return FastWAMIDM.infer_action(
                self,
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                num_video_frames=num_video_frames,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                compile_action_infer=compile_action_infer,
            )

        if action_infer_mode == "first_frame":
            return FastWAM.infer_action(
                self,
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                compile_action_infer=compile_action_infer,
            )

        raise ValueError("`action_infer_mode` must be one of: idm, first_frame.")
