"""Training entry point for CausalWan22 + OXE iterable dataset.

This script is separate from ``scripts/train.py`` because ``Wan22Trainer`` is
built around map-style datasets with ``ResumableEpochSampler`` (random access
+ len-based shuffled epochs), which is incompatible with the tf.data-backed
``OXERobotVideoDataset`` (``IterableDataset``).

Design:
  * **Step-based loop**: training is driven purely by optimizer-step count.
    ``cfg.max_steps`` is required and sets both the lr-scheduler horizon and
    the hard loop cap; there is no ``num_epochs`` concept because the
    underlying tf.data stream is infinite (``.repeat()`` in oxe_utils).
  * **Accelerate + DeepSpeed ZeRO1** launch path (see
    ``scripts/train_causal_wan22.sh``). Each rank independently iterates its
    own tf.data pipeline, seeded with ``cfg.seed + rank`` so different ranks
    draw different shuffle / interleave / augmentation orderings.
  * **Plain DataLoader with no sampler** — relies on the dataset's own tf.data
    shuffle buffer. ``num_workers`` must be ``0`` because tf.data graphs are
    not pickle/fork-safe for PyTorch worker subprocesses.
  * **DiT-only training**: ``_freeze_except_dit`` freezes VAE + text encoder
    and only trains ``model.dit`` (matches
    ``Wan22Trainer._apply_dit_only_train_mode``).
  * **Checkpointing** is step-based: every ``cfg.save_every`` optimizer
    steps we write ``checkpoints/weights/step_{N:08d}.pt`` plus an
    accelerator state directory and a ``trainer_state.json`` recording
    ``global_step``. A final checkpoint is also written at exit unless
    ``save_every == 0`` (which disables all saves).
  * **Resume** via ``cfg.resume=<state_dir>``: restores model / optimizer /
    scheduler / RNG state and the ``global_step`` counter. The tf.data
    stream is *not* rewound — it restarts from the beginning of each rank's
    stream, which is acceptable because the sampler is approximately IID.
"""

import json
import logging
import time
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
)
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.fs import ensure_dir
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.pytorch_utils import set_global_seed

register_default_resolvers()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Keys in the OXE sample dict that CausalWan22Core.training_loss actually
# needs. Everything else (notably `prompt`, a python str) is dropped at
# collate time because accelerate's device-transfer helpers choke on
# non-tensor fields inside the batch dict.
_KEEP_SAMPLE_KEYS = ("video", "context", "context_mask")


def _tensor_only_collate(batch):
    filtered = [{k: b[k] for k in _KEEP_SAMPLE_KEYS if k in b} for b in batch]
    return default_collate(filtered)


def _freeze_except_dit(model):
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)


def _build_scheduler(
    optimizer,
    scheduler_type: str,
    total_train_steps: int,
    warmup_steps: int,
    base_lr: float,
):
    scheduler_type = str(scheduler_type).strip().lower()
    total_train_steps = max(int(total_train_steps), 1)
    warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)
    remaining = max(total_train_steps - warmup_steps, 1)

    if scheduler_type == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=base_lr * 0.01)
    elif scheduler_type == "constant":
        main = ConstantLR(optimizer, factor=1.0, total_iters=remaining)
    else:
        raise ValueError(
            f"Unsupported lr_scheduler_type: {scheduler_type}. "
            "Expected one of ['cosine', 'constant']."
        )
    if warmup_steps <= 0:
        return main
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[warmup_steps],
    )


def _save_step_checkpoint(
    accelerator: Accelerator,
    model,
    output_dir: str,
    global_step: int,
):
    """Save DiT weights, accelerator state, and trainer_state.json at a step.

    Layout::

        {output_dir}/checkpoints/weights/step_{N:08d}.pt
        {output_dir}/checkpoints/state/step_{N:08d}/
            <accelerator state files...>
            trainer_state.json  -> {"global_step": N}
    """
    tag = f"step_{global_step:08d}"
    ckpt_root = Path(output_dir) / "checkpoints"
    weights_dir = ckpt_root / "weights"
    state_dir = ckpt_root / "state" / tag

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ensure_dir(str(weights_dir))
        ensure_dir(str(state_dir))
    accelerator.wait_for_everyone()

    weights_path = None
    if accelerator.is_main_process:
        weights_path = str(weights_dir / f"{tag}.pt")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_checkpoint(weights_path, optimizer=None, step=global_step)

    accelerator.save_state(output_dir=str(state_dir))

    if accelerator.is_main_process:
        trainer_state = {"global_step": int(global_step)}
        with open(state_dir / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, ensure_ascii=True, indent=2)

    accelerator.wait_for_everyone()
    return {"weights_path": weights_path, "state_path": str(state_dir)}


def _load_resume(accelerator: Accelerator, resume: str) -> int:
    """Load accelerator state and return the restored ``global_step``.

    ``resume`` must be a directory containing ``trainer_state.json`` plus the
    files written by ``accelerator.save_state``. Training continues from
    ``global_step + 1``.

    Note: the tf.data pipeline is **not** rewound to the resumed step — the
    underlying stream is infinite and approximately IID, so replaying early
    samples is harmless in expectation. Only the model / optimizer /
    scheduler / RNG state are restored.
    """
    resume_path = Path(resume)
    if not resume_path.exists() or not resume_path.is_dir():
        raise FileNotFoundError(
            f"resume path must be an existing directory, got: {resume}"
        )
    accelerator.load_state(input_dir=str(resume_path))
    state_file = resume_path / "trainer_state.json"
    if not state_file.exists():
        raise FileNotFoundError(
            f"Missing trainer_state.json under {resume_path}; cannot resume."
        )
    with open(state_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    global_step = int(payload.get("global_step", 0))
    logger.info(
        "Resumed from %s: global_step=%d -> continuing from step %d",
        resume_path,
        global_step,
        global_step + 1,
    )
    return global_step


def _init_wandb(cfg: DictConfig, accelerator: Accelerator, output_dir: str):
    if not bool(cfg.wandb.enabled) or not accelerator.is_main_process:
        return None
    import wandb

    return wandb.init(
        entity=cfg.wandb.workspace,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        group=None if cfg.wandb.group in (None, "null", "") else str(cfg.wandb.group),
        mode=cfg.wandb.mode,
        dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        mixed_precision=mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    setup_logging(
        log_level=logging.INFO,
        is_main_process=accelerator.is_main_process,
    )
    misc.register_work_dir(cfg.output_dir)
    if accelerator.is_main_process:
        ensure_dir(cfg.output_dir)
        with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
            OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), f)
    set_global_seed(int(cfg.seed))

    # --- model ---
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_device = str(accelerator.device)
    logger.info(
        "Building CausalWan22 model on %s (dtype=%s)", model_device, model_dtype
    )
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _freeze_except_dit(model)

    # --- dataset (iterable, rank-seeded) ---
    logger.info(
        "Building OXE train dataset (rank=%d, seed=%d)",
        accelerator.process_index,
        int(cfg.seed),
    )
    train_dataset = instantiate(
        cfg.data.train,
        seed=int(cfg.seed),
        rank=int(accelerator.process_index),
    )

    if int(cfg.num_workers) != 0:
        raise ValueError(
            "`num_workers` must be 0 for OXERobotVideoDataset — tf.data graphs "
            "are not pickle/fork-safe for DataLoader worker subprocesses."
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,  # IterableDataset — the underlying tf.data pipeline shuffles
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=_tensor_only_collate,
    )

    # --- optimizer ---
    trainable_params = [p for p in model.dit.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )

    # --- scheduler: horizon is set by max_steps (required) ---
    if cfg.get("max_steps") in (None, 0):
        raise ValueError(
            "`max_steps` must be set to a positive integer for step-based training."
        )
    total_train_steps = int(cfg.max_steps)
    warmup_steps = max(1, int(total_train_steps * 0.05))
    scheduler = _build_scheduler(
        optimizer,
        scheduler_type=str(cfg.lr_scheduler_type),
        total_train_steps=total_train_steps,
        warmup_steps=warmup_steps,
        base_lr=float(cfg.learning_rate),
    )
    logger.info(
        "Schedule: max_steps=%d warmup=%d lr=%.2e (%s)",
        total_train_steps,
        warmup_steps,
        float(cfg.learning_rate),
        str(cfg.lr_scheduler_type),
    )

    # --- accelerator prepare ---
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    optimizer.zero_grad(set_to_none=True)

    wandb_run = _init_wandb(cfg, accelerator, cfg.output_dir)

    def _wandb_log(payload: dict, step: int):
        if wandb_run is not None:
            wandb_run.log(payload, step=step)

    # --- resume (step-based) ---
    global_step = 0
    if cfg.get("resume") not in (None, "", "null"):
        global_step = _load_resume(accelerator, str(cfg.resume))

    if global_step >= total_train_steps:
        logger.info(
            "global_step (%d) >= max_steps (%d); nothing to do.",
            global_step,
            total_train_steps,
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    # --- train loop (step-based) ---
    logger.info(
        "Starting training: step %d -> %d", global_step, total_train_steps
    )
    log_every = int(cfg.log_every)
    save_every = int(cfg.save_every)
    max_grad_norm = float(cfg.max_grad_norm)

    run_start = time.perf_counter()
    run_start_step = global_step

    data_iter = iter(train_loader)
    last_saved_step = global_step  # avoid double-saving at a boundary

    while global_step < total_train_steps:
        try:
            sample = next(data_iter)
        except StopIteration:
            # The underlying tf.data stream is infinite; this is a safety
            # net in case that ever changes.
            data_iter = iter(train_loader)
            sample = next(data_iter)

        with accelerator.accumulate(model):
            train_model = (
                model
                if hasattr(model, "training_loss")
                else accelerator.unwrap_model(model)
            )
            with accelerator.autocast():
                loss, loss_dict = train_model.training_loss(sample)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if log_every > 0 and global_step % log_every == 0:
                    global_loss = float(
                        accelerator.gather(
                            loss.detach().float().reshape(1)
                        ).mean().item()
                    )
                    # Gather each per-key loss entry across ranks so the
                    # logged detail string reflects the global mean rather
                    # than rank-0's local value.
                    global_loss_dict = {}
                    for k, v in loss_dict.items():
                        metric_tensor = torch.tensor(
                            float(v), device=loss.device, dtype=torch.float32
                        ).reshape(1)
                        global_loss_dict[k] = float(
                            accelerator.gather(metric_tensor).mean().item()
                        )
                    current_lr = float(optimizer.param_groups[0]["lr"])
                    elapsed = max(time.perf_counter() - run_start, 1e-6)
                    sps = max(global_step - run_start_step, 1) / elapsed

                    if accelerator.is_main_process:
                        detail = " ".join(
                            f"{k}={v:.4f}" for k, v in sorted(global_loss_dict.items())
                        )
                        logger.info(
                            "[train] step=%d/%d loss=%.4f %s "
                            "lr=%.2e speed=%.2f step/s grad_norm=%.3f",
                            global_step,
                            total_train_steps,
                            global_loss,
                            detail,
                            current_lr,
                            sps,
                            float(grad_norm),
                        )
                        _wandb_log(
                            {
                                "train/loss": global_loss,
                                "train/lr": current_lr,
                                "train/grad_norm": float(grad_norm),
                                "performance/steps_per_sec": sps,
                                **{f"train/{k}": v for k, v in global_loss_dict.items()},
                            },
                            step=global_step,
                        )

                if save_every > 0 and global_step % save_every == 0:
                    ckpt_info = _save_step_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        output_dir=cfg.output_dir,
                        global_step=global_step,
                    )
                    last_saved_step = global_step
                    if accelerator.is_main_process:
                        logger.info(
                            "[ckpt] step=%d weights=%s state=%s",
                            global_step,
                            ckpt_info["weights_path"],
                            ckpt_info["state_path"],
                        )

    # Final checkpoint on exit (unless we just saved one at the boundary
    # or checkpointing is disabled entirely).
    if save_every > 0 and global_step != last_saved_step:
        ckpt_info = _save_step_checkpoint(
            accelerator=accelerator,
            model=model,
            output_dir=cfg.output_dir,
            global_step=global_step,
        )
        if accelerator.is_main_process:
            logger.info(
                "[ckpt-final] step=%d weights=%s state=%s",
                global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )

    logger.info("Training finished at global_step=%d", global_step)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
