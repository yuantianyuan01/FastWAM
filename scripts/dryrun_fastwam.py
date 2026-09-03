"""Benchmark FastWAM action inference with synthetic task-shaped inputs.

Example:
    python scripts/dryrun_fastwam.py \
      task=libero_uncond_2cam224_1e-4 \
      ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
      +DRYRUN.compile_action_infer=true \
      +DRYRUN.warmup=2 +DRYRUN.iters=10
"""

from __future__ import annotations

import logging
import inspect
import statistics
import time

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def _dryrun_config(cfg: DictConfig) -> DictConfig:
    defaults = OmegaConf.create(
        {
            "warmup": 2,
            "iters": 10,
            "seed": 42,
            "action_horizon": None,
            "num_inference_steps": None,
            "compile_action_infer": False,
            "benchmark_both": False,
            "use_random_context": False,
            "action_infer_mode": "idm",
        }
    )
    return OmegaConf.merge(defaults, cfg.get("DRYRUN", {}))


def _model_dtype(mixed_precision: str) -> torch.dtype:
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[str(mixed_precision)]


def _timed_infer_with_output(
    model,
    kwargs: dict,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model.infer_action(**kwargs)
    torch.cuda.synchronize(device)
    logger.info("action shape=%s", tuple(output["action"].shape))
    return time.perf_counter() - start, output["action"].detach().clone()


def _timed_infer(model, kwargs: dict, device: torch.device) -> float:
    elapsed, _ = _timed_infer_with_output(model, kwargs, device)
    return elapsed


def _benchmark_mode(
    model,
    infer_kwargs: dict,
    device: torch.device,
    *,
    label: str,
    warmup: int,
    iters: int,
) -> tuple[list[float], float]:
    for index in range(warmup):
        elapsed = _timed_infer(model, infer_kwargs, device)
        logger.info("%s warmup %d/%d: %.4fs", label, index + 1, warmup, elapsed)

    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for index in range(iters):
        elapsed = _timed_infer(model, infer_kwargs, device)
        timings.append(elapsed)
        logger.info("%s iteration %d/%d: %.4fs", label, index + 1, iters, elapsed)
    return timings, torch.cuda.max_memory_allocated(device) / 1024**3


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    dryrun = _dryrun_config(cfg)
    device = torch.device(str(cfg.EVALUATION.device))
    model = instantiate(
        cfg.model,
        model_dtype=_model_dtype(cfg.mixed_precision),
        device=str(device),
    )
    if cfg.ckpt is not None:
        model.load_checkpoint(str(cfg.ckpt))
    model.eval()
    logger.info(
        "device=%s gpu=%s dtype=%s",
        device,
        torch.cuda.get_device_name(device),
        next(model.parameters()).dtype,
    )

    height, width = (int(value) for value in cfg.data.train.video_size)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if dryrun.action_horizon is None
        else int(dryrun.action_horizon)
    )
    num_inference_steps = (
        int(cfg.eval_num_inference_steps)
        if dryrun.num_inference_steps is None
        else int(dryrun.num_inference_steps)
    )
    logger.info(
        "action_horizon=%d num_inference_steps=%d warmup=%d iters=%d",
        action_horizon,
        num_inference_steps,
        int(dryrun.warmup),
        int(dryrun.iters),
    )
    torch.manual_seed(int(dryrun.seed))
    input_image = torch.rand(1, 3, height, width) * 2 - 1
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    proprio = torch.randn(1, proprio_dim)
    if bool(dryrun.use_random_context):
        context = torch.randn(1, int(cfg.data.train.context_len), model.text_dim)
        context_mask = torch.ones(context.shape[:2], dtype=torch.bool)
        prompt = None
    else:
        context = None
        context_mask = None
        prompt = "pick up the object"

    infer_kwargs = {
        "prompt": prompt,
        "input_image": input_image,
        "action_horizon": action_horizon,
        "proprio": proprio,
        "context": context,
        "context_mask": context_mask,
        "num_inference_steps": num_inference_steps,
        "seed": int(dryrun.seed),
        "rand_device": str(cfg.EVALUATION.rand_device),
        "compile_action_infer": bool(dryrun.compile_action_infer),
    }
    infer_parameters = inspect.signature(model.infer_action).parameters
    if "num_video_frames" in infer_parameters:
        infer_kwargs["num_video_frames"] = (
            (int(cfg.data.train.num_frames) - 1)
            // int(cfg.data.train.action_video_freq_ratio)
            + 1
        )
    if "action_infer_mode" in infer_parameters:
        infer_kwargs["action_infer_mode"] = str(dryrun.action_infer_mode)

    warmup = int(dryrun.warmup)
    iters = int(dryrun.iters)
    if bool(dryrun.benchmark_both):
        infer_kwargs["compile_action_infer"] = False
        eager_times, eager_peak = _benchmark_mode(
            model,
            infer_kwargs,
            device,
            label="eager",
            warmup=warmup,
            iters=iters,
        )

        infer_kwargs["compile_action_infer"] = True
        cold_compile_time = _timed_infer(model, infer_kwargs, device)
        logger.info("compile cold start: %.4fs", cold_compile_time)
        compile_times, compile_peak = _benchmark_mode(
            model,
            infer_kwargs,
            device,
            label="compile",
            warmup=warmup,
            iters=iters,
        )
        eager_mean = statistics.mean(eager_times)
        compile_mean = statistics.mean(compile_times)
        logger.info(
            "comparison | eager_mean=%.4fs compile_mean=%.4fs speedup=%.3fx "
            "compile_cold_start=%.4fs eager_peak=%.3fGiB compile_peak=%.3fGiB",
            eager_mean,
            compile_mean,
            eager_mean / compile_mean,
            cold_compile_time,
            eager_peak,
            compile_peak,
        )
        return

    torch.cuda.reset_peak_memory_stats(device)
    for index in range(warmup):
        elapsed = _timed_infer(model, infer_kwargs, device)
        logger.info("warmup %d/%d: %.4fs", index + 1, warmup, elapsed)

    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for index in range(iters):
        elapsed = _timed_infer(model, infer_kwargs, device)
        timings.append(elapsed)
        logger.info("iteration %d/%d: %.4fs", index + 1, iters, elapsed)
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    logger.info(
        "mean=%.4fs median=%.4fs min=%.4fs max=%.4fs peak_allocated=%.3fGiB",
        statistics.mean(timings),
        statistics.median(timings),
        min(timings),
        max(timings),
        peak_gib,
    )


if __name__ == "__main__":
    main()
