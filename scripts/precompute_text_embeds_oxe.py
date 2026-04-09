"""Precompute Wan2.2 text embeddings for OXE (RLDS/TFDS) datasets.

This script mirrors `scripts/precompute_text_embeds.py` but collects prompts
from OXE RLDS datasets instead of `{dataset_dir}/meta/tasks.jsonl`.

Usage (single GPU):

    python scripts/precompute_text_embeds_oxe.py \
        --dataset_dirs /scratch/cgao304/dev/datasets/ \
        --data_mix debug_bridge \
        --text_embedding_cache_dir /path/to/cache_dir

Usage (multi-GPU via torchrun):

    torchrun --standalone --nproc_per_node=8 \
        scripts/precompute_text_embeds_oxe.py \
            --dataset_dirs /scratch/cgao304/dev/datasets/ \
            --data_mix debug_bridge \
            --text_embedding_cache_dir /path/to/cache_dir
"""

import argparse
import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from tqdm import tqdm

from fastwam.datasets.oxe.oxe_robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _resolve_mixture_spec(data_mix: str):
    # Imported lazily to avoid importing TF until we actually need it.
    from oxe_utils.rlds.oxe import OXE_NAMED_MIXTURES

    if data_mix in OXE_NAMED_MIXTURES:
        return list(OXE_NAMED_MIXTURES[data_mix])
    return [(data_mix, 1.0)]


def _read_unique_prompts_oxe(dataset_dirs: str, data_mix: str) -> list[str]:
    """Iterate OXE RLDS trajectories and collect unique `DEFAULT_PROMPT`-wrapped
    language instructions. Images are not decoded because we only access the
    `language_instruction` field of each trajectory.
    """
    # Imported lazily: TF / TFDS startup is slow and not needed when the caller
    # relies on `--override_instruction`.
    import oxe_utils.dlimp as dl
    import tensorflow_datasets as tfds

    mixture_spec = _resolve_mixture_spec(data_mix)

    prompts: list[str] = []
    seen = set()
    total_task_rows = 0

    for dataset_name, _weight in mixture_spec:
        logger.info("Scanning OXE dataset `%s` under `%s` for unique instructions.", dataset_name, dataset_dirs)
        builder = tfds.builder(dataset_name, data_dir=str(dataset_dirs))
        ds = dl.DLataset.from_rlds(builder, split="all", shuffle=False)

        if "language_instruction" not in ds.element_spec:
            raise KeyError(
                f"OXE dataset `{dataset_name}` has no top-level `language_instruction` "
                f"field; got keys {list(ds.element_spec.keys())}. This script assumes "
                "the RLDS dataset exposes language at the trajectory level (as is the "
                "case for bridge_orig-like datasets)."
            )

        ds_local_unique = 0
        for traj in ds.as_numpy_iterator():
            langs = traj["language_instruction"]
            arr = langs.tolist() if hasattr(langs, "tolist") else [langs]
            for raw in arr:
                if isinstance(raw, (bytes, bytearray)):
                    task = raw.decode("utf-8")
                else:
                    task = str(raw)
                task = task.strip()
                if task == "":
                    continue
                total_task_rows += 1
                prompt = DEFAULT_PROMPT.format(task=task)
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)
                    ds_local_unique += 1

        logger.info("  → dataset `%s`: contributed %d new unique prompts.", dataset_name, ds_local_unique)

    logger.info(
        "Loaded %d task rows from %d OXE datasets, deduplicated to %d prompts.",
        total_task_rows,
        len(mixture_spec),
        len(prompts),
    )
    return prompts


def _get_override_prompt(override_instruction: Any) -> str | None:
    if override_instruction is None:
        return None
    task = str(override_instruction).strip()
    if task == "":
        return None
    return DEFAULT_PROMPT.format(task=task)


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute Wan2.2 text embeddings for OXE datasets.",
    )
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        required=True,
        help="Root data directory containing RLDS/TFDS OXE datasets.",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        required=True,
        help="OXE mixture name (from OXE_NAMED_MIXTURES) or a single dataset name.",
    )
    parser.add_argument(
        "--text_embedding_cache_dir",
        type=str,
        action="append",
        required=True,
        help="Output cache dir for text embeddings. Pass multiple times to mirror to several dirs.",
    )
    parser.add_argument("--context_len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer_model_id", type=str, default=DEFAULT_TOKENIZER_MODEL_ID)
    parser.add_argument("--redirect_common_files", action=argparse.BooleanOptionalAction, default=False) # resolve from hugging face
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--override_instruction",
        type=str,
        default=None,
        help="If set, skip the dataset scan and encode exactly this single instruction.",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main():
    setup_logging(log_level=logging.INFO)
    args = _parse_args()

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone --nproc_per_node=%d scripts/precompute_text_embeds_oxe.py ...",
            torch.cuda.device_count(),
        )

    overwrite = bool(args.overwrite)
    context_len = int(args.context_len)
    cache_dirs: list[Path] = []
    for raw in args.text_embedding_cache_dir:
        p = Path(str(raw)).expanduser()
        if p not in cache_dirs:
            cache_dirs.append(p)
    if not cache_dirs:
        raise ValueError("At least one --text_embedding_cache_dir is required.")

    override_prompt = _get_override_prompt(args.override_instruction)
    if override_prompt is not None:
        prompts = [override_prompt]
        logger.info("Using --override_instruction; skipping dataset scan and encoding exactly 1 prompt.")
    else:
        prompts = _read_unique_prompts_oxe(args.dataset_dirs, args.data_mix)
    if not prompts:
        logger.warning("No prompts collected from OXE dataset; nothing to do.")
        return

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16
    model_id = str(args.model_id)
    tokenizer_model_id = str(args.tokenizer_model_id)
    redirect_common_files = bool(args.redirect_common_files)
    enc_id = _model_id_to_enc_id(model_id)

    logger.info(
        "Preparing text encoder with model_id=%s tokenizer_model_id=%s device=%s dtype=%s context_len=%d overwrite=%s",
        model_id,
        tokenizer_model_id,
        device,
        torch_dtype,
        context_len,
        overwrite,
    )

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    stats = {
        str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0}
        for cache_dir in cache_dirs
    }

    prompts.sort()  # this is in prevention of the random shuffling of the as_numpy_iterator()
    prompts = prompts[rank::world_size] if is_distributed else prompts

    if not overwrite:
        fully_cached_local = 0
        prompts_to_encode: list[str] = []
        for prompt in prompts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            fully_cached = True
            for cache_dir in cache_dirs:
                cache_path = cache_dir / filename
                if not cache_path.exists():
                    fully_cached = False
                    break
            if fully_cached:
                fully_cached_local += 1
                for cache_dir in cache_dirs:
                    stats[str(cache_dir)]["skip"] += 1
            else:
                prompts_to_encode.append(prompt)

        prompts = prompts_to_encode

        fully_cached_global = fully_cached_local
        to_encode_global = len(prompts)
        if is_distributed:
            reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
            count_tensor = torch.tensor([fully_cached_local, len(prompts)], device=reduce_device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            fully_cached_global = int(count_tensor[0].item())
            to_encode_global = int(count_tensor[1].item())

        if (not is_distributed) or rank == 0:
            logger.info(
                "overwrite=false: fully cached prompts=%d, prompts to encode=%d",
                fully_cached_global,
                to_encode_global,
            )

    logger.info("Writing caches to %d directories.", len(cache_dirs))
    prompts_encoded_local = len(prompts)
    prompts_encoded_global = prompts_encoded_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([prompts_encoded_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor.item())

    batch_size = int(args.batch_size)
    over_length_prompts = 0
    with tqdm(
        total=len(prompts),
        desc=f"Encoding prompts (rank {rank}/{world_size})" if is_distributed else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                over_length_prompts += int(mask.all(dim=1).sum().item())
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    payload = {
                        "context": context_i,
                        "mask": mask_i,
                    }

                    for cache_dir in cache_dirs:
                        cache_path = cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                        key = str(cache_dir)
                        if cache_path.exists() and not overwrite:
                            stats[key]["skip"] += 1
                            continue

                        if cache_path.exists():
                            stats[key]["overwrite"] += 1
                        else:
                            stats[key]["new"] += 1

                        _atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    over_length_global = over_length_prompts
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        over_tensor = torch.tensor([over_length_prompts], device=reduce_device, dtype=torch.long)
        dist.all_reduce(over_tensor, op=dist.ReduceOp.SUM)
        over_length_global = int(over_tensor.item())

        counts_tensor = torch.tensor(
            [
                [stats[str(cache_dir)]["new"], stats[str(cache_dir)]["overwrite"], stats[str(cache_dir)]["skip"]]
                for cache_dir in cache_dirs
            ],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            for idx, cache_dir in enumerate(cache_dirs):
                key = str(cache_dir)
                stats[key]["new"] = int(counts_tensor[idx, 0].item())
                stats[key]["overwrite"] = int(counts_tensor[idx, 1].item())
                stats[key]["skip"] = int(counts_tensor[idx, 2].item())

    if (not is_distributed) or rank == 0:
        logger.info("Finished precomputing text embeddings.")
        logger.info(
            "Over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
            context_len,
            over_length_global,
            prompts_encoded_global,
        )
        for cache_dir in cache_dirs:
            key = str(cache_dir)
            logger.info(
                "Cache dir: %s | new=%d overwrite=%d skip=%d",
                key,
                stats[key]["new"],
                stats[key]["overwrite"],
                stats[key]["skip"],
            )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
