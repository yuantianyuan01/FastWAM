"""
Postprocess data by calculating statistics for each trajectory and saving them to a stats group within the same file.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, CancelledError, Future, as_completed
from threading import Semaphore, Lock
import os
import json
import glob
import traceback
from typing import TypeAlias

import numpy as np
from tqdm import tqdm
import h5py


StatsDict: TypeAlias = dict[str, np.ndarray | float | int]

STATS_GROUP_NAME = "stats"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")

    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate trajectory stats but do not write anything",
    )
    dry_run_group.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only calculate and save aggregate stats, do not write trajectory stats to data files",
    )
    parser.add_argument("--num-workers", type=int, default=32)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keys", nargs="+", help="List of data keys to calculate statistics for")
    group.add_argument(
        "--keys-file",
        help="File containing a newline-delimited list of data keys to calculate statistics for",
    )
    return parser.parse_args()


def load_dicts_data(key: str, data: h5py.Dataset) -> list[dict]:
    ret = []
    for i in range(data.shape[0]):
        data_str = data[i].tobytes().decode("utf-8").rstrip("\x00")
        try:
            d = json.loads(data_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding key {key} index {i} as JSON: {data_str}") from e
        ret.append(d)
    return ret


def calculate_data_stats(
    traj_group: h5py.Group, traj_stats_group: h5py.Group | None, data_key: str
) -> dict[str, StatsDict]:
    data = traj_group[data_key]
    if np.issubdtype(data.dtype, np.integer):
        # assume byte-encoded json-encoded dicts
        assert data.ndim == 2
        dicts = load_dicts_data(data_key, data)

        # special case for actions: remove the first (dummmy) and last (done sentinel) actions
        if data_key.startswith("actions/"):
            dicts = dicts[1:-1]

        # collect all keys across all dicts (some timesteps may have different keys)
        all_keys: set[str] = set()
        for d in dicts:
            all_keys.update(d.keys())

        stats_dict = {}
        for k in all_keys:
            # only include values from dicts where this key exists
            values = [d[k] for d in dicts if k in d]
            if len(values) == 0:
                continue
            data = np.array(values)
            stats_dict[k] = {
                "mean": np.mean(data, axis=0).tolist(),
                "std": np.std(data, axis=0).tolist(),
                "min": np.min(data, axis=0).tolist(),
                "max": np.max(data, axis=0).tolist(),
                "count": len(data),
                "sum": np.sum(data, axis=0).tolist(),
                "sum_sq": np.sum(data**2, axis=0).tolist(),
            }
        stats_dict_str = json.dumps(stats_dict)
        if traj_stats_group is not None:
            traj_stats_group.create_dataset(f"{data_key}", data=stats_dict_str)
        return stats_dict
    elif np.issubdtype(data.dtype, np.floating):
        assert data.ndim == 2, f"Data for key {data_key} should be 2D, but is {data.ndim}D"
        stats_dict = {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
            "min": np.min(data, axis=0),
            "max": np.max(data, axis=0),
            "count": len(data),
            "sum": np.sum(data, axis=0),
            "sum_sq": np.sum(data**2, axis=0),
        }
        if traj_stats_group is not None:
            for k, v in stats_dict.items():
                traj_stats_group.create_dataset(f"{data_key}/{k}", data=v)
        # key by sentinel value "" since there's only one value
        return {"": {k: v.tolist() for k, v in stats_dict.items()}}
    else:
        raise ValueError(
            f"Unsupported data type for {data_key}: {data.dtype} (shape: {data.shape})"
        )


def calculate_group_stats(
    traj_group: h5py.Group, traj_stats_group: h5py.Group | None, group_key: str
) -> dict[str, StatsDict]:
    ret = {}
    for key in traj_group[group_key].keys():
        full_key = f"{group_key}/{key}"
        if isinstance(traj_group[full_key], h5py.Group):
            ret.update(calculate_group_stats(traj_group, traj_stats_group, full_key))
        elif isinstance(traj_group[full_key], h5py.Dataset):
            stats = calculate_data_stats(traj_group, traj_stats_group, full_key)
            if "" in stats:
                assert len(stats) == 1, "Expected only one key in stats dict"
                ret[full_key] = stats[""]
            else:
                for k, v in stats.items():
                    ret[f"{full_key}/{k}"] = v

    return ret


def calculate_traj_stats(
    traj_group: h5py.Group, traj_stats_group: h5py.Group | None, data_keys: list[str]
):
    ret: dict[str, StatsDict] = {}
    for data_key in data_keys:
        if isinstance(traj_group[data_key], h5py.Group):
            ret.update(calculate_group_stats(traj_group, traj_stats_group, data_key))
        elif isinstance(traj_group[data_key], h5py.Dataset):
            stats = calculate_data_stats(traj_group, traj_stats_group, data_key)
            if "" in stats:
                assert len(stats) == 1, "Expected only one key in stats dict"
                ret[data_key] = stats[""]
            else:
                for k, v in stats.items():
                    ret[f"{data_key}/{k}"] = v
    return ret


def merge_stats_dicts(stats_dicts: list[StatsDict]) -> StatsDict:
    summed = np.sum([d["sum"] for d in stats_dicts], axis=0)
    summed_sq = np.sum([d["sum_sq"] for d in stats_dicts], axis=0)
    count = sum(d["count"] for d in stats_dicts)
    mean = summed / count
    merged_stats = {
        "mean": mean,
        "std": np.sqrt(summed_sq / count - mean**2),
        "min": np.min([d["min"] for d in stats_dicts], axis=0),
        "max": np.max([d["max"] for d in stats_dicts], axis=0),
        "count": count,
        "sum": summed,
        "sum_sq": summed_sq,
    }
    return merged_stats


def sanitize_dict(d: dict) -> dict:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = sanitize_dict(v)
        elif isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer, np.bool_)):
            d[k] = v.item()
    return d


def process_data_file(args, data_file: str, traj_keys: list[str], keys: list[str]):
    stats_dicts: dict[str, list[StatsDict]] = defaultdict(list)
    with h5py.File(data_file, "r" if args.dry_run else "r+") as f:
        if STATS_GROUP_NAME in f and not args.dry_run:
            del f[STATS_GROUP_NAME]

        if not args.dry_run:
            stats_group = f.create_group(STATS_GROUP_NAME)
        else:
            stats_group = None

        try:
            for traj_key in traj_keys:
                if not args.dry_run:
                    traj_stats_group = stats_group.create_group(traj_key)
                else:
                    traj_stats_group = None

                try:
                    stats_dict_dict = calculate_traj_stats(f[traj_key], traj_stats_group, keys)
                    for key, stats_dict in stats_dict_dict.items():
                        stats_dicts[key].append(stats_dict)
                except Exception as e:
                    # if an error occurred, delete the stats group that errored
                    if not args.dry_run:
                        del stats_group[traj_key]
                    raise ValueError(
                        f"Error calculating stats for trajectory {traj_key} in file {data_file}: {e}"
                    ) from e
        except KeyboardInterrupt:
            # if we're interrupted, delete the whole stats group to avoid partial updates
            if not args.dry_run:
                del f[STATS_GROUP_NAME]
            raise

    merged_stats: dict[str, StatsDict] = {}
    for key in stats_dicts:
        merged_stats[key] = merge_stats_dicts(stats_dicts[key])

    return merged_stats


def main():
    args = get_args()
    if args.keys_file is not None:
        with open(args.keys_file, "r") as f:
            keys = f.read().splitlines()
    else:
        keys = args.keys

    dry_run = args.dry_run
    if args.aggregate_only:
        args.dry_run = True

    with open(os.path.join(args.data_root, "valid_trajectory_index.json"), "r") as f:
        traj_index = json.load(f)
    data_file_trajs: dict[str, list[str]] = {}
    for data_file_dict in traj_index.values():
        for data_file_subpath, trajs in data_file_dict.items():
            data_file_path = os.path.join(args.data_root, data_file_subpath)
            data_file_trajs[data_file_path] = list(trajs.keys())

    stats_dicts: dict[str, list[StatsDict]] = defaultdict(list)

    if args.num_workers > 1:
        # use a semaphore to limit the number of queued jobs, helps with large quantity of datafiles
        submit_semaphore = Semaphore(args.num_workers * 4)
        lock = Lock()
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            with tqdm(total=len(data_file_trajs), desc="Processing files...") as pbar:

                def on_done(future: Future):
                    try:
                        future.result()
                    except CancelledError:
                        pass
                    except:
                        traceback.print_exc()
                        executor.shutdown(wait=False, cancel_futures=True)

                    with lock:
                        pbar.update(1)
                    submit_semaphore.release()

                futures: list[Future] = []
                for data_file, traj_keys in data_file_trajs.items():
                    submit_semaphore.acquire()
                    future = executor.submit(process_data_file, args, data_file, traj_keys, keys)
                    future.add_done_callback(on_done)
                    futures.append(future)

                for future in as_completed(futures):
                    stats = future.result()
                    if stats is not None:
                        for key, stats_dict in stats.items():
                            stats_dicts[key].append(stats_dict)
    else:
        for data_file, traj_keys in tqdm(data_file_trajs.items()):
            stats = process_data_file(args, data_file, traj_keys, keys)
            if stats is not None:
                for key, stats_dict in stats.items():
                    stats_dicts[key].append(stats_dict)

    merged_stats = {}
    for key in stats_dicts:
        merged_stats[key] = merge_stats_dicts(stats_dicts[key])

    if args.aggregate_only or not dry_run:
        with open(os.path.join(args.data_root, "aggregated_stats.json"), "w") as f:
            sanitized_merged_stats = sanitize_dict(merged_stats)
            json.dump(sanitized_merged_stats, f, indent=2)


if __name__ == "__main__":
    main()
