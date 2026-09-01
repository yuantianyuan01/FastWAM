import torch

_SEED_MODULUS = 1 << 64


def make_inference_generator(
    seed: int | None,
    device: str | torch.device,
    *,
    stream_offset: int = 0,
) -> torch.Generator | None:
    """Create a reproducible generator for one inference noise stream."""
    if seed is None:
        return None
    if stream_offset < 0:
        raise ValueError(f"`stream_offset` must be non-negative, got {stream_offset}.")

    stream_seed = (int(seed) + stream_offset) % _SEED_MODULUS
    return torch.Generator(device=device).manual_seed(stream_seed)
