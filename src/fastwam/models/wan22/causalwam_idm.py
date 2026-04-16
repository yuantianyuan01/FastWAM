from typing import Any, Optional

import functools
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from fastwam.utils.logging_config import get_logger

from .fastwam_joint import FastWAMJoint

logger = get_logger(__name__)


class CausalWAMIDM(FastWAMJoint):
    """IDM variant with teacher-forcing video conditioning for action denoising."""
    video_cond_noise_prob = 0.5
    def __init__(self, *args, use_flex_attention: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_flex_attention = bool(use_flex_attention)
        
    @functools.lru_cache(maxsize=16)
    def _tf_self_attn_mask(
        self,
        num_frames: int,
        video_tokens_per_frame: int,
        action_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        
        V = num_frames * video_tokens_per_frame
        A = (num_frames - 1) * action_tokens_per_frame
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
        action_start, action_end = 2 * V, total

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

    @functools.lru_cache(maxsize=16)
    def _tf_block_mask(
        self,
        num_frames: int,
        video_tokens_per_frame: int,
        action_tokens_per_frame: int,
        device: torch.device,
    ):
       
        V = num_frames * video_tokens_per_frame
        A = (num_frames - 1) * action_tokens_per_frame
        total = 2 * V + A

        def mask_mod(b, h, q_idx, kv_idx):
            q_is_cond = q_idx < V
            q_is_noisy = (q_idx >= V) & (q_idx < 2 * V)
            q_is_action = q_idx >= 2 * V

            k_is_cond = kv_idx < V
            k_is_noisy = (kv_idx >= V) & (kv_idx < 2 * V)
            k_is_action = kv_idx >= 2 * V

            q_video_frame = (q_idx % V) // video_tokens_per_frame
            k_video_frame = (kv_idx % V) // video_tokens_per_frame

            q_action_frame = (q_idx - 2 * V) // action_tokens_per_frame
            k_action_frame = (kv_idx - 2 * V) // action_tokens_per_frame

            cond_cond = q_is_cond & k_is_cond & (q_video_frame >= k_video_frame)
            noisy_cond = q_is_noisy & k_is_cond & (q_video_frame > k_video_frame)
            noisy_noisy = q_is_noisy & k_is_noisy & (q_video_frame == k_video_frame)

            action_cond = q_is_action & k_is_cond & (
                (q_action_frame <= k_video_frame) & ((q_action_frame + 1) >= k_video_frame)
            )

            action_action = q_is_action & k_is_action & (q_action_frame == k_action_frame)

            return cond_cond | noisy_cond | noisy_noisy | action_cond | action_action

        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=total,
            KV_LEN=total,
            device=device,
        )
        
    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        device: torch.device,
    ):
        if noisy_video_tokens_per_frame != cond_video_tokens_per_frame:
            raise ValueError(
                "Teacher-forcing requires identical `tokens_per_frame` for noisy and cond video branches, "
                f"got {noisy_video_tokens_per_frame} and {cond_video_tokens_per_frame}."
            )

        if noisy_video_seq_len != cond_video_seq_len:
            raise ValueError(
                "Teacher-forcing requires identical `seq_len` for noisy and cond video branches, "
                f"got {noisy_video_seq_len} and {cond_video_seq_len}."
            )

        if noisy_video_seq_len % noisy_video_tokens_per_frame != 0:
            raise ValueError(
                "`noisy_video_seq_len` must be divisible by `noisy_video_tokens_per_frame` in "
                f"`teacher_forcing` mode, got {noisy_video_seq_len} and {noisy_video_tokens_per_frame}."
            )

        num_frames = noisy_video_seq_len // noisy_video_tokens_per_frame

        if num_frames < 2:
            raise ValueError(
                "Teacher-forcing with action tokens requires at least 2 video frames."
            )

        if action_seq_len % (num_frames - 1) != 0:
            raise ValueError(
                "`action_seq_len` must be divisible by `num_video_frames - 1` in "
                f"`teacher_forcing` mode, got {action_seq_len} and {num_frames - 1}."
            )

        action_tokens_per_frame = action_seq_len // (num_frames - 1)

        if self.use_flex_attention:
            return self._tf_block_mask(
                num_frames=num_frames,
                video_tokens_per_frame=noisy_video_tokens_per_frame,
                action_tokens_per_frame=action_tokens_per_frame,
                device=device,
            )
        else:
            return self._tf_self_attn_mask(
                num_frames=num_frames,
                video_tokens_per_frame=noisy_video_tokens_per_frame,
                action_tokens_per_frame=action_tokens_per_frame,
                device=device,
            )
    
    @torch.no_grad()
    def _causal_self_attention_mask(
        self,
        num_frames: int,
        video_tokens_per_frame: int,
        action_tokens_per_frame: int,
        device: torch.device,
    ):

        V = num_frames * video_tokens_per_frame
        A = (num_frames - 1) * action_tokens_per_frame
        total = V + A

        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        frame_idx = torch.arange(num_frames, device=device).repeat_interleave(video_tokens_per_frame)
        q_frame = frame_idx.unsqueeze(1)  
        k_frame = frame_idx.unsqueeze(0)  

        action_idx = torch.arange(num_frames - 1, device=device).repeat_interleave(action_tokens_per_frame)
        q_action = action_idx.unsqueeze(1)  
        k_action = action_idx.unsqueeze(0) 

        video_start, video_end = 0, V
        action_start, action_end = V, total

        # video -> video: causal by frame
        mask[video_start:video_end, video_start:video_end] = (q_frame >= k_frame)

        # action -> video: action_t can see video_t and video_{t+1}
        mask[action_start:action_end, video_start:video_end] = (
            (q_action <= k_frame) & ((q_action + 1) >= k_frame)
        )

        # action -> action: same transition block only
        mask[action_start:action_end, action_start:action_end] = (q_action == k_action)

        return mask

    @torch.no_grad()
    def _causal_block_mask(
        self,
        num_frames: int,
        video_tokens_per_frame: int,
        action_tokens_per_frame: int,
        device: torch.device,
    ):

        V = num_frames * video_tokens_per_frame
        A = (num_frames - 1) * action_tokens_per_frame
        total = V + A

        def mask_mod(b, h, q_idx, kv_idx):
            q_is_video = q_idx < V
            q_is_action = q_idx >= V

            k_is_video = kv_idx < V
            k_is_action = kv_idx >= V

            q_video_frame = q_idx // video_tokens_per_frame
            k_video_frame = kv_idx // video_tokens_per_frame

            q_action_frame = (q_idx - V) // action_tokens_per_frame
            k_action_frame = (kv_idx - V) // action_tokens_per_frame

            video_video = q_is_video & k_is_video & (q_video_frame >= k_video_frame)

            action_video = q_is_action & k_is_video & (
                (q_action_frame <= k_video_frame) & ((q_action_frame + 1) >= k_video_frame)
            )

            action_action = q_is_action & k_is_action & (q_action_frame == k_action_frame)

            return video_video | action_video | action_action

        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=total,
            KV_LEN=total,
            device=device,
        )

    @functools.lru_cache(maxsize=16)
    def _causal_video_only_block_mask(
        self,
        num_frames: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ):
        V = num_frames * video_tokens_per_frame

        def mask_mod(b, h, q_idx, kv_idx):
            q_video_frame = q_idx // video_tokens_per_frame
            k_video_frame = kv_idx // video_tokens_per_frame
            return q_video_frame >= k_video_frame

        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=V,
            KV_LEN=V,
            device=device,
        )

    @functools.lru_cache(maxsize=16)
    def _causal_action_query_block_mask(
        self,
        num_frames: int,
        video_tokens_per_frame: int,
        action_tokens_per_frame: int,
        device: torch.device,
    ):
        V = num_frames * video_tokens_per_frame
        A = (num_frames - 1) * action_tokens_per_frame

        def mask_mod(b, h, q_idx, kv_idx):
            q_action_frame = q_idx // action_tokens_per_frame

            k_is_video = kv_idx < V
            k_is_action = kv_idx >= V
            k_video_frame = kv_idx // video_tokens_per_frame
            k_action_frame = (kv_idx - V) // action_tokens_per_frame

            action_video = k_is_video & (
                (q_action_frame <= k_video_frame) & ((q_action_frame + 1) >= k_video_frame)
            )
            action_action = k_is_action & (q_action_frame == k_action_frame)

            return action_video | action_action

        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=A,
            KV_LEN=V + A,
            device=device,
        )

    @torch.no_grad()
    def _resolve_inference_token_layout(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
    ) -> tuple[int, int]:

        if video_seq_len % video_tokens_per_frame != 0:
            raise ValueError(
                "`video_seq_len` must be divisible by `video_tokens_per_frame` in "
                f"`inference` mode, got {video_seq_len} and {video_tokens_per_frame}."
            )

        num_frames = video_seq_len // video_tokens_per_frame

        if num_frames < 2:
            raise ValueError(
                "Inference with action tokens requires at least 2 video frames."
            )

        if action_seq_len % (num_frames - 1) != 0:
            raise ValueError(
                "`action_seq_len` must be divisible by `num_video_frames - 1` in "
                f"`inference` mode, got {action_seq_len} and {num_frames - 1}."
            )

        action_tokens_per_frame = action_seq_len // (num_frames - 1)
        return num_frames, action_tokens_per_frame

    @torch.no_grad()
    def _build_inference_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ):
        num_frames, action_tokens_per_frame = self._resolve_inference_token_layout(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
        )

        if self.use_flex_attention:
            return self._causal_block_mask(
                num_frames=num_frames,
                video_tokens_per_frame=video_tokens_per_frame,
                action_tokens_per_frame=action_tokens_per_frame,
                device=device,
            )
        else:
            return self._causal_self_attention_mask(
                num_frames=num_frames,
                video_tokens_per_frame=video_tokens_per_frame,
                action_tokens_per_frame=action_tokens_per_frame,
                device=device,
            )
            
    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        # Branch A: cond video
        cond_noise_mask = torch.rand((batch_size,), device=self.device) < float(self.video_cond_noise_prob)
        timestep_video_cond = torch.zeros((batch_size,), dtype=input_latents.dtype, device=self.device)
        latents_cond = input_latents
        if bool(cond_noise_mask.any()):
            timestep_video_cond_sampled = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            timestep_video_cond = torch.where(cond_noise_mask, timestep_video_cond_sampled, timestep_video_cond)
            noise_video_cond = torch.randn_like(input_latents)
            latents_cond_noisy = self.train_video_scheduler.add_noise(
                input_latents, noise_video_cond, timestep_video_cond_sampled
            )
            cond_noise_selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
            latents_cond = torch.where(cond_noise_selector, latents_cond_noisy, input_latents)
        if inputs["first_frame_latents"] is not None:
            latents_cond = latents_cond.clone()
            latents_cond[:, :, 0:1] = inputs["first_frame_latents"]
            
        # Branch B: noisy video 
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents_noisy = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]
        
        # Branch C: noisy action.
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre_cond = self.video_expert.pre_dit(
            x=latents_cond,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_pre_noisy = self.video_expert.pre_dit(
            x=latents_noisy,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        
        if video_pre_noisy["t_mod"].ndim != 4 or video_pre_cond["t_mod"].ndim != 4:
            raise ValueError(
                "Teacher-forcing requires token-wise `t_mod`; "
                "ensure `seperated_timestep=true` and `fuse_vae_embedding_in_latents=true`."
            )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        
        cond_video_seq_len = int(video_pre_cond["tokens"].shape[1])
        noisy_video_seq_len = int(video_pre_noisy["tokens"].shape[1])
        cond_video_tokens_per_frame = int(video_pre_cond["meta"]["tokens_per_frame"])
        noisy_video_tokens_per_frame = int(video_pre_noisy["meta"]["tokens_per_frame"])

        # Concatenate [cond_video, noisy_video] as the video expert sequence.
        merged_video_tokens = torch.cat([video_pre_cond["tokens"], video_pre_noisy["tokens"]], dim=1)
        merged_video_freqs = torch.cat([video_pre_cond["freqs"], video_pre_noisy["freqs"]], dim=0)
        merged_video_t_mod = torch.cat([video_pre_cond["t_mod"], video_pre_noisy["t_mod"]], dim=1)
        merged_video_context_mask = torch.cat([video_pre_cond["context_mask"], video_pre_noisy["context_mask"]], dim=1)

        attention_mask = self._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_pre["tokens"].shape[1],
            noisy_video_tokens_per_frame=noisy_video_tokens_per_frame,
            cond_video_tokens_per_frame=cond_video_tokens_per_frame,
            device=merged_video_tokens.device,
        )

        mot_kwargs = {
            "embeds_all": {
                "video": merged_video_tokens,
                "action": action_pre["tokens"],
            },
            "freqs_all": {
                "video": merged_video_freqs,
                "action": action_pre["freqs"],
            },
            "context_all": {
                "video": {
                    "context": video_pre_noisy["context"],
                    "mask": merged_video_context_mask,
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            "t_mod_all": {
                "video": merged_video_t_mod,
                "action": action_pre["t_mod"],
            },
        }
        if self.use_flex_attention:
            mot_kwargs["attention_mask"] = None
            mot_kwargs["block_mask"] = attention_mask
        else:
            mot_kwargs["attention_mask"] = attention_mask

        tokens_out = self.mot(**mot_kwargs)

        # Only the noisy-video half contributes to video denoising loss.
        pred_video_tokens = tokens_out["video"][:, noisy_video_seq_len:]
        pred_video = self.video_expert.post_dit(pred_video_tokens, video_pre_noisy)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)

        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
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
    ) -> dict[str, Any]:
        # Reuse infer_joint pipeline and keep infer_action output contract.
        out = self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=None,
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
            test_action_with_infer_action=False,
        )
        return {"action": out["action"]}

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
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
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale, test_action_with_infer_action
        self.eval()

        if action is not None:
            logger.warning(
                "`FastWAMIDM.infer_joint` ignores `action` input; "
                "video is denoised in a standalone first stage."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        
        num_transitions = (num_video_frames - 1) // self.vae.temporal_downsample_factor
        if num_transitions <= 0:
            raise ValueError(
                "`num_video_frames` is too small for IDM inference with actions: "
                f"got {num_video_frames} with temporal_downsample_factor={self.vae.temporal_downsample_factor}. "
                "Need at least one latent transition."
            )

        if action_horizon % num_transitions != 0:
            raise ValueError(
                f"`action_horizon` must be divisible by the number of action steps per video frame, "
                f"which is {num_transitions}, "
                f"got {action_horizon} and {num_video_frames} video frames."
            )
            
        latent_t = num_transitions + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        noisy_action_horizon = action_horizon // num_transitions

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, 1, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        actions = []
    
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        if not callable(getattr(self.video_expert, "forward_with_per_frame_timesteps", None)):
            raise ValueError(
                "`CausalWAMIDM.infer_joint` requires `video_expert.forward_with_per_frame_timesteps`, "
                "for example `CausalWanVideoDiT`."
            )

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        
        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )

        inference_layout: Optional[dict[str, int]] = None
        dense_joint_attention_mask = None
        video_self_block_mask = None
        action_query_block_mask = None

        for k in range(1, latent_t):
            # Stage 1: denoise only the new video frame autoregressively.
            target_video = torch.randn(
                1,
                self.vae.model.z_dim,
                1,
                latent_h,
                latent_w,
                generator=video_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)

            for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
                latents_seq = torch.cat([latents_video, target_video], dim=2)

                timestep_video = torch.zeros(
                    (1, k + 1),
                    device=self.device,
                    dtype=latents_video.dtype,
                )
                timestep_video[:, -1] = step_t_video

                pred_video = self.video_expert.forward_with_per_frame_timesteps(
                    latents=latents_seq,
                    per_frame_timesteps=timestep_video,
                    context=context,
                    context_mask=context_mask,
                )

                pred_video_target = pred_video[:, :, -1:].contiguous()
                target_video = self.infer_video_scheduler.step(
                    pred_video_target,
                    step_delta_video,
                    target_video,
                )

            latents_video = torch.cat([latents_video, target_video], dim=2)

            # Stage 2: infer only the action chunk for the new transition.
            latents_video_cond = latents_video[:, :, -2:]
            latents_action = torch.randn(
                (1, noisy_action_horizon, self.action_expert.action_dim),
                generator=action_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)

            timestep_video_cond = torch.zeros(
                (latents_video_cond.shape[0],),
                dtype=latents_video.dtype,
                device=self.device,
            )
            video_pre_cond = self.video_expert.pre_dit(
                x=latents_video_cond,
                timestep=timestep_video_cond,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )

            video_seq_len = int(video_pre_cond["tokens"].shape[1])
            video_tokens_per_frame = int(video_pre_cond["meta"]["tokens_per_frame"])
            action_seq_len = int(latents_action.shape[1])

            if inference_layout is None:
                num_frames, action_tokens_per_frame = self._resolve_inference_token_layout(
                    video_seq_len=video_seq_len,
                    action_seq_len=action_seq_len,
                    video_tokens_per_frame=video_tokens_per_frame,
                )
                inference_layout = {
                    "video_seq_len": video_seq_len,
                    "video_tokens_per_frame": video_tokens_per_frame,
                    "action_seq_len": action_seq_len,
                    "num_frames": num_frames,
                    "action_tokens_per_frame": action_tokens_per_frame,
                }

                if self.use_flex_attention:
                    video_self_block_mask = self._causal_video_only_block_mask(
                        num_frames=num_frames,
                        video_tokens_per_frame=video_tokens_per_frame,
                        device=video_pre_cond["tokens"].device,
                    )
                    action_query_block_mask = self._causal_action_query_block_mask(
                        num_frames=num_frames,
                        video_tokens_per_frame=video_tokens_per_frame,
                        action_tokens_per_frame=action_tokens_per_frame,
                        device=video_pre_cond["tokens"].device,
                    )
                else:
                    dense_joint_attention_mask = self._build_inference_attention_mask(
                        video_seq_len=video_seq_len,
                        action_seq_len=action_seq_len,
                        video_tokens_per_frame=video_tokens_per_frame,
                        device=video_pre_cond["tokens"].device,
                    )

            if self.use_flex_attention:
                if video_self_block_mask is None or action_query_block_mask is None or inference_layout is None:
                    raise RuntimeError("Missing cached block masks for flex-attention inference.")

                video_kv_cache = self.mot.prefill_video_cache(
                    video_tokens=video_pre_cond["tokens"],
                    video_freqs=video_pre_cond["freqs"],
                    video_t_mod=video_pre_cond["t_mod"],
                    video_context_payload={
                        "context": video_pre_cond["context"],
                        "mask": video_pre_cond["context_mask"],
                    },
                    video_attention_mask=None,
                    video_block_mask=video_self_block_mask,
                )

                for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
                    timestep_action = step_t_action.unsqueeze(0).to(
                        dtype=latents_action.dtype,
                        device=self.device,
                    )
                    action_pre = self.action_expert.pre_dit(
                        action_tokens=latents_action,
                        timestep=timestep_action,
                        context=context,
                        context_mask=context_mask,
                    )
                    action_tokens = self.mot.forward_action_with_video_cache(
                        action_tokens=action_pre["tokens"],
                        action_freqs=action_pre["freqs"],
                        action_t_mod=action_pre["t_mod"],
                        action_context_payload={
                            "context": action_pre["context"],
                            "mask": action_pre["context_mask"],
                        },
                        video_kv_cache=video_kv_cache,
                        attention_mask=None,
                        block_mask=action_query_block_mask,
                        video_seq_len=inference_layout["video_seq_len"],
                    )
                    pred_action = self.action_expert.post_dit(action_tokens, action_pre)
                    latents_action = self.infer_action_scheduler.step(
                        pred_action,
                        step_delta_action,
                        latents_action,
                    )
            else:
                if dense_joint_attention_mask is None or inference_layout is None:
                    raise RuntimeError("Missing cached dense attention mask for inference.")

                video_kv_cache = self.mot.prefill_video_cache(
                    video_tokens=video_pre_cond["tokens"],
                    video_freqs=video_pre_cond["freqs"],
                    video_t_mod=video_pre_cond["t_mod"],
                    video_context_payload={
                        "context": video_pre_cond["context"],
                        "mask": video_pre_cond["context_mask"],
                    },
                    video_attention_mask=dense_joint_attention_mask[
                        : inference_layout["video_seq_len"],
                        : inference_layout["video_seq_len"],
                    ],
                )

                for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
                    timestep_action = step_t_action.unsqueeze(0).to(
                        dtype=latents_action.dtype,
                        device=self.device,
                    )
                    pred_action = self._predict_action_noise_with_cache(
                        latents_action=latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_kv_cache=video_kv_cache,
                        attention_mask=dense_joint_attention_mask,
                        video_seq_len=inference_layout["video_seq_len"],
                    )
                    latents_action = self.infer_action_scheduler.step(
                        pred_action,
                        step_delta_action,
                        latents_action,
                    )

            actions.append(latents_action)

        actions_out = torch.cat(actions, dim=1)

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": actions_out[0].detach().to(device="cpu", dtype=torch.float32),
        }
