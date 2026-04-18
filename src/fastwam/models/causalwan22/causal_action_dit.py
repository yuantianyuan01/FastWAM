import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from fastwam.utils.logging_config import get_logger

from fastwam.models.wan22.helpers.gradient import gradient_checkpoint_forward
from fastwam.models.wan22.wan_video_dit import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
)
from fastwam.models.wan22.action_dit import ActionDiT, ActionHead

logger = get_logger(__name__)


class CausalActionDiT(ActionDiT):

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Prepare per-chunk-timestep inputs for the action DiT blocks.

        Shape contract (distinct from `ActionDiT.pre_dit`):
            action_tokens: [B, C, CT, A]  (batch, chunk_id, chunk_size, action_dim)
            timestep:      [B, C] or [1, C]  (one noise level per chunk per sample)

        Tokens within the same chunk share a single timestep; tokens are
        flattened to a single sequence of length `C * CT` before the DiT
        blocks, in row-major order `(c0_t0, ..., c0_t{CT-1}, c1_t0, ...)`.

        Returns:
            tokens:       [B, C*CT, hidden_dim]
            t:            [B, C*CT, hidden_dim]       per-token time embedding
            t_mod:        [B, C*CT, 6, hidden_dim]    per-token AdaLN modulation
            freqs:        [C*CT, 1, attn_head_dim]    RoPE freqs along the flat sequence
            context:      [B, L, hidden_dim]
            context_mask: [B, C*CT, L]
            meta:         {batch_size, seq_len, num_chunks, chunk_size}
        """
        if action_tokens.ndim != 4:
            raise ValueError(
                "`action_tokens` must be 4D [B, C, CT, action_dim], "
                f"got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[3] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[3]}"
            )
        if timestep.ndim != 2:
            raise ValueError(
                f"`timestep` must be 2D [B, C] or [1, C], got shape {tuple(timestep.shape)}"
            )
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size, num_chunks, chunk_size, _ = action_tokens.shape
        seq_len = num_chunks * chunk_size

        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` dim-0 must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[1] != num_chunks:
            raise ValueError(
                f"`timestep` dim-1 must equal num_chunks({num_chunks}), got {timestep.shape[1]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep batch dim must match batch_size.")
            timestep = timestep.expand(batch_size, num_chunks)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        # Per-chunk -> per-token timesteps: [B, C] -> [B, C, CT] -> [B, C*CT].
        token_timesteps = (
            timestep.unsqueeze(-1)
            .expand(batch_size, num_chunks, chunk_size)
            .contiguous()
            .reshape(batch_size, seq_len)
        )
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, seq_len, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        tokens = self.action_encoder(
            action_tokens.reshape(batch_size, seq_len, self.action_dim)
        )
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "tokens_per_frame": chunk_size, 
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)
