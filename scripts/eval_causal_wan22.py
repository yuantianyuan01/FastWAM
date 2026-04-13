"""Autoregressive evaluation for CausalWan22.

Samples from the training or validation dataset, keeps the first frame and
language instruction, and generates the remaining frames autoregressively
(one latent frame at a time) via ``CausalWan22Core.infer()``.

Usage::

    python scripts/eval_causal_wan22.py \
        --config-path ../runs/causal_wan22_pretrain/2026-04-11_00-53-41 \
        --config-name config \
        +eval.checkpoint_path=runs/.../step_00010000.pt \
        +eval.num_samples=8 \
        +eval.num_inference_steps=20 \
        +eval.output_dir=eval_outputs \
        +eval.source=val \
        +eval.seed=42
"""

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
)
from fastwam.utils.fs import ensure_dir
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.utils.video_io import save_mp4

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def video_tensor_to_pil(video: torch.Tensor):
    """Convert a [C, T, H, W] tensor in [-1, 1] to a list of PIL Images."""
    v = video.detach().float().clamp(-1, 1)
    v = ((v + 1.0) * 127.5).to(torch.uint8).cpu()
    return [
        Image.fromarray(v[:, t].permute(1, 2, 0).numpy())
        for t in range(v.shape[1])
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    # --- eval config (merged via Hydra +eval.xxx overrides) ---
    eval_cfg = cfg.get("eval", {})
    checkpoint_path = eval_cfg.get("checkpoint_path", None)
    num_samples = int(eval_cfg.get("num_samples", 4))
    num_inference_steps = int(
        eval_cfg.get("num_inference_steps", cfg.get("eval_num_inference_steps", 20))
    )
    output_dir = str(eval_cfg.get("output_dir", "./eval_outputs"))
    source = str(eval_cfg.get("source", "val"))
    seed = int(eval_cfg.get("seed", 42))
    device = str(eval_cfg.get("device", "cuda:0"))

    ensure_dir(output_dir)
    set_global_seed(seed)
    logger.info("Eval config: samples=%d steps=%d source=%s seed=%d device=%s",
                num_samples, num_inference_steps, source, seed, device)

    # --- compute actual video frame count ---
    data_cfg = cfg.data.get(source, cfg.data.train)
    raw_num_frames = int(data_cfg.num_frames)
    action_video_freq_ratio = int(data_cfg.get("action_video_freq_ratio", 1))
    num_video_frames = (raw_num_frames - 1) // action_video_freq_ratio + 1
    logger.info(
        "Video frames: raw=%d, action_video_freq_ratio=%d, actual=%d",
        raw_num_frames, action_video_freq_ratio, num_video_frames,
    )

    # --- build model ---
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    logger.info("Building model on %s (dtype=%s) ...", device, model_dtype)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)

    if checkpoint_path not in (None, "null", "None", ""):
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading checkpoint: %s", checkpoint_path)
            model.load_checkpoint(str(ckpt))
        else:
            logger.warning("Checkpoint not found, using random weights: %s", checkpoint_path)
    else:
        logger.info("No checkpoint specified, using current model weights.")

    model.eval()

    # --- build dataset ---
    logger.info("Building %s dataset ...", source)
    dataset = instantiate(data_cfg, seed=seed, rank=0)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # --- generate ---
    data_iter = iter(loader)
    for i in range(num_samples):
        try:
            sample = next(data_iter)
        except StopIteration:
            logger.warning("Dataset exhausted after %d samples.", i)
            break

        video = sample["video"]       # [1, C, T, H, W]
        context = sample["context"]   # [1, L, D]
        context_mask = sample["context_mask"]  # [1, L]

        # First frame as input image [1, C, H, W]
        first_frame = video[0, :, 0:1].permute(1, 0, 2, 3)  # [1, C, H, W]

        logger.info(
            "[sample %d/%d] video shape=%s, generating %d video frames ...",
            i + 1, num_samples, list(video.shape), num_video_frames,
        )

        # Autoregressive generation via model.infer()
        output = model.infer(
            input_image=first_frame,
            num_frames=num_video_frames,
            context=context[0],       # [L, D]
            context_mask=context_mask[0],  # [L]
            num_inference_steps=num_inference_steps,
            seed=seed + i,
        )
        gen_frames = output["video"]

        # Save generated video
        gen_path = str(Path(output_dir) / f"sample_{i:04d}_gen.mp4")
        save_mp4(gen_frames, gen_path, fps=4)
        logger.info("  Saved generated video: %s (%d frames)", gen_path, len(gen_frames))

        # Save ground truth video
        gt_frames = video_tensor_to_pil(video[0])  # [C, T, H, W]
        gt_path = str(Path(output_dir) / f"sample_{i:04d}_gt.mp4")
        save_mp4(gt_frames, gt_path, fps=4)
        logger.info("  Saved ground truth video: %s (%d frames)", gt_path, len(gt_frames))

        # Save first frame as PNG
        first_frame_pil = gt_frames[0]
        png_path = str(Path(output_dir) / f"sample_{i:04d}_first_frame.png")
        first_frame_pil.save(png_path)

    logger.info("Evaluation complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
