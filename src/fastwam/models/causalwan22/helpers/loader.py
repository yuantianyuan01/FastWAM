from dataclasses import dataclass
import inspect
from typing import Any

import torch
import time

from ...wan22.helpers.io import load_state_dict
from ...wan22.helpers.loader import (
    SKIPPED_PRETRAIN_SENTINEL,
    _load_registered_model,
    _resolve_configs,
    _validate_dit_config as _validate_wan_video_dit_config,
)
from ...wan22.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ...wan22.wan_video_vae import WanVideoVAE38
from ..causalwan22 import CausalWanVideoDiT
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "CausalWan22LoadedComponents",
    "SKIPPED_PRETRAIN_SENTINEL",
    "load_causalwan22_ti2v_5b_components",
    "_validate_dit_config",
]


@dataclass
class CausalWan22LoadedComponents:
    dit: CausalWanVideoDiT
    vae: WanVideoVAE38
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    """Like wan22's validator, but also allows the causal-only `use_flex_attention` key.

    `CausalWanVideoDiT.__init__` uses `*args, use_flex_attention=False, **kwargs`,
    so `inspect.signature` there cannot derive the concrete arg set — we defer to
    wan22's validator (which inspects `WanVideoDiT.__init__`) and pop the causal
    flag before/after.
    """
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")

    base = dict(dit_config)
    use_flex_attention = base.pop("use_flex_attention", None)
    validated = _validate_wan_video_dit_config(base)
    if use_flex_attention is not None:
        validated["use_flex_attention"] = use_flex_attention
    return validated


def _instantiate_causal_dit_from_pretrained(
    dit_path,
    torch_dtype: torch.dtype,
    device: str,
    dit_kwargs: dict[str, Any],
) -> CausalWanVideoDiT:
    """Load pretrained Wan2.2 DiT weights directly into a CausalWanVideoDiT.

    `CausalWanVideoDiT` inherits from `WanVideoDiT` with no new learnable
    parameters, so the original safetensors load cleanly (strict=False to
    tolerate the trivial buffer/flag differences).
    """
    model = CausalWanVideoDiT(**dit_kwargs)
    state_dict = load_state_dict(dit_path, torch_dtype=torch_dtype, device="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model.to(device=device, dtype=torch_dtype)


def load_causalwan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
):
    """Load Causal Wan2.2-TI2V-5B components.

    Mirrors `wan22.helpers.loader.load_causalwan22_ti2v_5b_components`, except the
    returned DiT is a `CausalWanVideoDiT` (teacher-forcing + per-frame
    timestep support). VAE, text encoder, tokenizer, path resolution, and
    the hash-based registry loader are reused unchanged from `wan22.helpers`.
    """
    logger.info("Loading Causal Wan2.2-TI2V-5B components...")
    start = time.time()

    if dit_config is None:
        raise ValueError("`dit_config` is required for Causal Wan2.2-TI2V-5B loading.")
    validated_dit_config = _validate_dit_config(dit_config)

    dit_model_config, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )

    vae_config.download_if_necessary()
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()

    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained causal video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing causal video expert randomly and expecting checkpoint override."
        )
        dit: CausalWanVideoDiT = CausalWanVideoDiT(**validated_dit_config).to(
            device=device, dtype=torch_dtype,
        )
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_config.download_if_necessary()
        dit = _instantiate_causal_dit_from_pretrained(
            dit_path=dit_model_config.path,
            torch_dtype=torch_dtype,
            device=device,
            dit_kwargs=validated_dit_config,
        )
        dit_path = str(dit_model_config.path)

    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )
    vae: WanVideoVAE38 = _load_registered_model(
        vae_config.path, "wan_video_vae", torch_dtype=torch_dtype, device=device,
    )
    logger.info("Finished loading Causal Wan2.2-TI2V-5B components in %.2f seconds.", time.time() - start)
    return CausalWan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=str(vae_config.path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
