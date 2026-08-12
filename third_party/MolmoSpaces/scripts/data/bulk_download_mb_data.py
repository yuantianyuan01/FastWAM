"""Download and extract molmobot-data.

Downloads whole shard tars from allenai/molmobot-data, decompresses each
inner .tar.zst archive, and extracts its contents into:

    <target_dir>/<task_config>/part<X>/<split>/…

A local JSON manifest tracks progress per-shard so that interrupted
downloads can be resumed without re-extracting already-completed shards.
"""

import argparse
import io
import json
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

try:
    import datasets
    import zstandard as zstd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
except ImportError:
    print(
        "Please install dependencies with e.g.\n"
        "  pip install zstandard datasets huggingface_hub tqdm"
    )
    raise

datasets.logging.set_verbosity_error()
datasets.disable_progress_bars()

REPO = "allenai/molmobot-data"
REPO_TYPE = "dataset"

TASK_CONFIGS = [
    "DoorOpeningDataGenConfig",
    "FrankaPickAndPlaceColorOmniCamConfig",
    "FrankaPickAndPlaceNextToOmniCamConfig",
    "FrankaPickAndPlaceOmniCamConfig",
    "FrankaPickOmniCamConfig",
    "RBY1OpenDataGenConfig",
    "RBY1PickAndPlaceDataGenConfig",
    "RBY1PickDataGenConfig",
    "FrankaPickAndPlaceOmniCamConfig_ObjectBackfill",
]

MANIFEST_FILENAME = "download_manifest.json"


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**4:
        return f"{size_bytes / 1024**4:.2f} TiB"
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GiB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KiB"
    return f"{size_bytes} B"


def _manifest_path(target_dir: str) -> Path:
    return Path(target_dir) / MANIFEST_FILENAME


def _load_manifest(target_dir: str) -> dict:
    """Load the local progress manifest.

    Structure::

        {
            "<config_name>": {
                "<split>": {
                    "num_completed_entries": [120, 124, ...],
                    "completed_shards": [0, 1, ...],
                }
            },
            ...
        }
    """
    if target_dir is None:
        return {}

    p = _manifest_path(target_dir)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    return {}


def _save_manifest(target_dir: str, manifest: dict) -> None:
    p = _manifest_path(target_dir)
    os.makedirs(p.parent, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(p)


def load_entries(config_name: str, split: str) -> list[dict]:
    """Load the parquet arrow table for a task config as a list of dicts."""
    try:
        ds = load_dataset(REPO, name=config_name, split=f"{split}_pkgs")
        return [row for row in ds]
    except ValueError as e:
        print(f"No {split} split for {config_name}")
        return []


def summarize_config(config_name: str, split: str, entries: list[dict]) -> dict:
    """Return a summary dict with sizes, shard/part counts."""
    compressed = sum(e["size"] for e in entries)
    inflated = sum(e.get("inflated_size", 0) for e in entries)
    has_inflated = entries and "inflated_size" in entries[0]
    parts = sorted(set(e["part"] for e in entries))
    shards = sorted(set(e["shard_id"] for e in entries))

    per_part: dict[int, dict] = {}
    for p in parts:
        part_entries = [e for e in entries if e["part"] == p]
        part_compressed = sum(e["size"] for e in part_entries)
        part_inflated = sum(e.get("inflated_size", 0) for e in part_entries)
        per_part[p] = {
            "entries": len(part_entries),
            "compressed": part_compressed,
            "inflated": part_inflated if has_inflated else None,
        }

    return {
        "config": config_name,
        "split": split,
        "entries": len(entries),
        "compressed": compressed,
        "inflated": inflated if has_inflated else None,
        "parts": parts,
        "shards": shards,
        "per_part": per_part,
    }


def print_summary(summary: dict) -> None:
    print(f"  Task config:   {summary['config']}")
    print(f"  Entries:       {summary['entries']}")
    print(f"  Shards:        {len(summary['shards'])}")
    print(f"  Parts:         {len(summary['parts'])}")
    print(f"  Download size: {_format_size(summary['compressed'])}")
    if summary["inflated"] is not None:
        print(f"  Extracted size: {_format_size(summary['inflated'])}")
    else:
        print(f"  Extracted size: (unknown)")
    for p, info in sorted(summary["per_part"].items()):
        dl = _format_size(info["compressed"])
        ex = (
            _format_size(info["inflated"])
            if info["inflated"] is not None
            else "unknown"
        )
        print(f"    part {p}: {info['entries']} entries, dl {dl}, extracted {ex}")


def _extract_tar_zst_bytes(data: bytes, output_dir: str) -> None:
    """Decompress a .tar.zst payload and stream-extract into *output_dir*."""
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(io.BytesIO(data)) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(path=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download molmobot training data from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available task configs:\n" + "\n".join(f"  - {c}" for c in TASK_CONFIGS)
        ),
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=None,
        help="Local root directory to extract data into.",
    )
    parser.add_argument(
        "--config",
        choices=TASK_CONFIGS,
        default=None,
        help="Download a single task config. Omit to see --list or use --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="Download all task configs.",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=None,
        help="Download only a specific part (e.g. --part 0).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="List available task configs and their sizes, then exit.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Extract only the given split (`train`, `val`, or `all`). Default is `all`.",
    )
    parser.add_argument(
        "--max_part_shards",
        type=int,
        default=None,
        help="Extract only the given number of shards per part.",
    )
    args = parser.parse_args()

    if not args.list_only and args.target_dir is None:
        parser.error("target_dir is required.")

    configs_to_process: list[str] = []
    if args.config:
        configs_to_process = [args.config]
    elif args.download_all or args.list_only:
        configs_to_process = list(TASK_CONFIGS)
    else:
        parser.error("Specify --config <NAME>, --all, or --list.")

    split_choices = dict(
        train=["train"],
        val=["val"],
        all=["val", "train"],
    )
    splits = split_choices[args.split]

    # Load entries and compute summaries
    all_entries: dict[str, dict[str, list[dict]]] = {split: {} for split in splits}
    total_compressed = 0
    total_inflated = 0
    has_all_inflated = True

    print(f"Loading metadata for {len(configs_to_process)} config(s)...\n")

    manifest = _load_manifest(args.target_dir)

    configs_to_skip = {split: set() for split in splits}
    for config_name in configs_to_process:
        for split in splits:
            part_to_shard_to_entries: dict[int, dict[int, list[dict]]] = {}
            entries = load_entries(config_name, split)

            if not entries:
                configs_to_skip[split].add(config_name)
                continue

            part_entries = defaultdict(list)
            for e in entries:
                part_entries[e["part"]].append(e)

            for part in part_entries:
                shard_to_entries = defaultdict(list)
                for e in part_entries[part]:
                    shard_to_entries[e["shard_id"]].append(e)
                part_to_shard_to_entries[part] = {**shard_to_entries}

            if args.part is not None:
                if args.part not in part_to_shard_to_entries:
                    print(f"  {config_name} has no part {args.part}. Skipping")
                    configs_to_skip[split].add(config_name)
                    continue
                if args.max_part_shards is not None:
                    shard_to_entries = part_to_shard_to_entries[args.part]
                    shards = sorted(shard_to_entries.keys())[: args.max_part_shards]
                    part_to_shard_to_entries = {
                        args.part: {s: shard_to_entries[s] for s in shards}
                    }
            else:
                if args.max_part_shards is not None:
                    for part in part_to_shard_to_entries:
                        shard_to_entries = part_to_shard_to_entries[part]
                        shards = sorted(shard_to_entries.keys())[: args.max_part_shards]
                        part_to_shard_to_entries[part] = {
                            s: shard_to_entries[s] for s in shards
                        }

            # Show resume status and filter out fully-completed configs
            cfg_m = manifest.get(config_name, {}).get(split, {})
            done_shards = set(cfg_m.get("completed_shards", []))
            if done_shards:
                required_shards = set(
                    sum(
                        [
                            list(shard_to_entries.keys())
                            for part, shard_to_entries in part_to_shard_to_entries.items()
                        ],
                        [],
                    )
                )
                missing_shards = required_shards - done_shards
                if not missing_shards:
                    print(f"  {config_name}: already fully extracted, skipping")
                    configs_to_skip[split].add(config_name)
                    continue
                else:
                    print(
                        f"  {config_name}: {len(missing_shards)}/{len(required_shards)} "
                        f"shards pending extraction (will resume)"
                    )

            entries = [
                e
                for part, shard_to_entries in part_to_shard_to_entries.items()
                for shard, entries in shard_to_entries.items()
                for e in entries
                if shard not in done_shards
            ]

            all_entries[split][config_name] = entries
            summary = summarize_config(config_name, split, entries)

            print_summary(summary)
            total_compressed += summary["compressed"]
            if summary["inflated"] is not None:
                total_inflated += summary["inflated"]
            else:
                has_all_inflated = False
            print()

    configs_to_process = {
        split: [
            config_name
            for config_name in configs_to_process
            if config_name not in configs_to_skip[split]
        ]
        for split in splits
    }

    all_configs_to_process = sum(
        len(split_cfg) for split_cfg in configs_to_process.values()
    )

    if all_configs_to_process > 1:
        print(f"Total across {all_configs_to_process} configs:")
        print(f"  Download size:  {_format_size(total_compressed)}")
        if has_all_inflated:
            print(f"  Extracted size: {_format_size(total_inflated)}")
        else:
            print(f"  Extracted size: (partially unknown)")
        print()

    if args.list_only:
        return

    if not configs_to_process:
        print("\nNothing to download — all configs are fully extracted.")
        return

    if not args.yes:
        answer = input("\nProceed with download? [Y/n] ").strip().lower()
        if answer and answer not in ("y", "yes"):
            print("Aborted.")
            return

    for split in splits:
        for config_name in configs_to_process[split]:
            print(f"\n{'=' * 60}")
            print(f"  {config_name} {split}")
            print(f"{'=' * 60}")
            try:
                download_and_extract_config(
                    config_name,
                    args.target_dir,
                    split=split,
                    entries=all_entries[split][config_name],
                )
            except KeyboardInterrupt:
                print("\nInterrupted. Progress has been saved; re-run to resume.")
                raise SystemExit(1)
            except Exception as e:
                print(f"FAILED: {config_name} {split} — {e}")

    print("\nAll done.")


def download_and_extract_config(
    config_name: str,
    target_dir: str,
    split: str,
    entries: list[dict] | None = None,
) -> None:
    """Download and extract all shards for a single task config.

    Whole shard tars are downloaded via ``hf_hub_download`` (which caches
    them).  Each inner ``.tar.zst`` member is decompressed and extracted
    into ``<target_dir>/<config_name>/part<X>/<split>/``.

    Progress is tracked per-shard in a local manifest so that interrupted
    runs can be resumed.
    """
    if entries is None:
        print(f"Loading entry table for {config_name}...")
        entries = load_entries(config_name, split)

    # Build lookup: path -> entry (for part resolution)
    path_to_entry = {e["path"]: e for e in entries}

    # Group entries by shard
    shard_to_paths: dict[int, list[str]] = defaultdict(list)
    for e in entries:
        shard_to_paths[e["shard_id"]].append(e["path"])
    all_shard_ids = sorted(shard_to_paths.keys())

    # Load manifest for resume
    manifest = _load_manifest(target_dir)
    cfg_manifest = manifest.setdefault(config_name, {}).setdefault(split, {})
    completed_shards: set[int] = set(cfg_manifest.get("completed_shards", []))
    num_completed_entries: list[int] = cfg_manifest.get("num_completed_entries", [])

    remaining_shards = [s for s in all_shard_ids if s not in completed_shards]
    if completed_shards:
        print(
            f"Resuming: {len(completed_shards)} shard(s) already done, "
            f"{len(remaining_shards)} remaining"
        )

    if not remaining_shards:
        print(f"All shards for {config_name} already extracted.")
        return

    print(
        f"Downloading & extracting {len(remaining_shards)} shard(s) for {config_name} {split}..."
    )

    # Use a temporary cache dir so shard downloads don't accumulate
    # in the default HF cache (~/.cache/huggingface/hub/).
    shard_cache_dir = os.path.join(target_dir, ".shard_cache")
    os.makedirs(shard_cache_dir, exist_ok=True)

    try:
        for shard_id in tqdm(remaining_shards, desc=f"{config_name} shards"):
            entries_count = 0

            shard_filename = f"{config_name}/{split}_shards/{shard_id:05d}.tar"

            shard_local = hf_hub_download(
                repo_id=REPO,
                filename=shard_filename,
                repo_type=REPO_TYPE,
                revision="main",
                cache_dir=shard_cache_dir,
            )

            with tarfile.open(shard_local, "r:") as shard_tar:
                members = [m for m in shard_tar.getmembers() if m.isfile()]
                for member in tqdm(
                    members,
                    desc=f"  extracting shard {shard_id:05d}",
                    leave=False,
                ):
                    archive_path = member.name

                    entry = path_to_entry.get(archive_path)
                    if entry is None:
                        continue

                    part = entry["part"]
                    extract_dir = os.path.join(
                        target_dir, config_name, f"part{part}", split
                    )
                    os.makedirs(extract_dir, exist_ok=True)

                    fobj = shard_tar.extractfile(member)
                    if fobj is None:
                        continue
                    data = fobj.read()
                    _extract_tar_zst_bytes(data, extract_dir)
                    entries_count += 1

            # Wipe the temporary cache after each shard to reclaim disk
            shutil.rmtree(shard_cache_dir, ignore_errors=True)
            os.makedirs(shard_cache_dir, exist_ok=True)

            completed_shards.add(shard_id)
            num_completed_entries.append(entries_count)

            # Persist progress after each shard
            cfg_manifest["completed_shards"] = sorted(completed_shards)
            cfg_manifest["num_completed_entries"] = num_completed_entries
            _save_manifest(target_dir, manifest)
    finally:
        shutil.rmtree(shard_cache_dir, ignore_errors=True)

    print(
        f"Done extracting {config_name} ({sum(num_completed_entries)} entries, {len(completed_shards)} shards)."
    )


if __name__ == "__main__":
    main()
