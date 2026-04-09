"""
Minimal single-GPU training entry point for VSCode debugger.

Bypasses accelerate/DeepSpeed so you can set breakpoints anywhere.
Launch via VSCode "Python: Current File" or the launch.json config below.

Usage (CLI):
    python scripts/debug_train.py task=video_pretrain_1e-4
    python scripts/debug_train.py task=libero_uncond_2cam224_1e-4 batch_size=1
"""

import hydra
from omegaconf import DictConfig

from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    import torch
    from pathlib import Path
    from omegaconf import OmegaConf
    from fastwam.runtime import (
        build_datasets,
        setup_logging,
        _normalize_mixed_precision,
        _mixed_precision_to_model_dtype,
    )
    from fastwam.utils import misc
    from hydra.utils import instantiate
    import logging

    setup_logging(log_level=logging.INFO, is_main_process=True)
    misc.register_work_dir(cfg.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    print(f"[debug] Building model on {device} with dtype={model_dtype}")
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)

    print("[debug] Building datasets")
    train_ds, val_ds = build_datasets(cfg.data)

    # Freeze VAE + text encoder, train DiT only (same as Wan22Trainer)
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        proprio_encoder.train()
        proprio_encoder.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # 0 workers for easier debugging
        pin_memory=False,
    )

    print(f"[debug] Starting training loop — {len(train_ds)} samples, batch_size={cfg.batch_size}")
    autocast_ctx = torch.amp.autocast("cuda", dtype=model_dtype) if device.startswith("cuda") else torch.amp.autocast("cpu")

    for epoch in range(cfg.num_epochs):
        for step, sample in enumerate(loader):
            with autocast_ctx:
                loss, loss_dict = model.training_loss(sample)

            optimizer.zero_grad()
            loss.backward()
            if cfg.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.dit.parameters(), cfg.max_grad_norm)
            optimizer.step()

            if step % cfg.log_every == 0:
                detail = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                print(f"[debug] epoch={epoch} step={step} loss={loss.item():.4f} {detail}")

            max_steps = cfg.get("max_steps")
            if max_steps and step >= max_steps:
                break

    print("[debug] Done.")


if __name__ == "__main__":
    main()
