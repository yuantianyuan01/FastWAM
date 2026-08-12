#!/usr/bin/env python3
import os
import pickle
import argparse
from pprint import pprint
from molmo_spaces.configs.abstract_config import Config
import logging
from create_benchmark import collect_all_episode_stats
import json, hashlib
from pathlib import Path

DEFAULT_ATTRS = ["scene_dataset", "policy_config", "camera_config", "data_split"]

import logging
log = logging.getLogger(__name__)


def extract_attrs(obj, attrs):
    """Extract attributes or dict keys from an object."""
    out = {}
    for attr in attrs:
        if isinstance(obj, dict):
            res = obj.get(attr, "NOT FOUND")
        else:
            res = getattr(obj, attr, "NOT FOUND")

        if isinstance(res, Config):
            res = str(res.__class__.__name__)
        out[attr] = res

    return out

def process_pkl(dirpath, fname, attrs=None):
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(logging.ERROR)

    stats_by_split = collect_all_episode_stats(Path(dirpath), disable_tqdm=True)

    logger.setLevel(old_level)
    hashes = []
    for key in stats_by_split:
        num_episodes = sum(len(eps) for eps in stats_by_split[key].values())
        #print(f"[{key}] houses", len(stats_by_split[key].keys()), "episodes:", num_episodes)

        # print
        keys = sorted(stats_by_split[key].keys())
        house_ids = sorted([int(x.replace("house_","")) for x in keys])
        #print(house_ids)
        payload = json.dumps(house_ids, separators=(",", ":"))
        house_hash = hashlib.md5(payload.encode()).hexdigest()
        print(f"{Path(dirpath).name} houses hash", house_hash, len(house_ids))

        hashes.append(house_hash)

    pkl_path = os.path.join(dirpath, fname)
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        #print(f"\nFile: {pkl_path}")
        if attrs is not None:
            extracted = extract_attrs(data, args.attrs)
            pprint(extracted, sort_dicts=False)

    except Exception as e:
        #print(f"\nFile: {pkl_path}")
        #print("ERROR loading file:", e)
        pass
    return hashes

def process_json(dirpath, fname):
    #print(Path(dirpath).name)
    with open(Path(dirpath) / fname) as f_obj:
        data = json.load(f_obj)
    house_ids = []
    for house in data:
        #for ep in house:
        house_ids.append(house["house_index"])
    house_ids = sorted(house_ids)
    house_ids = sorted(set(house_ids))
    #print(house_ids)
    payload = json.dumps(house_ids, separators=(",", ":"))
    house_hash = hashlib.md5(payload.encode()).hexdigest()
    print(Path(dirpath).name, "houses hash", house_hash, len(house_ids))
    return house_hash


def check_configs():
    a_base = "/weka/prior/datasets/robomolmo/bench"
    b_base = "/weka/prior/datasets/robomolmo/bench_v2"
    pairs = [
            ("FrankaPickDroidBench_20251219_benchmark", "FrankaPickDroidBench_2000ep_json_benchmark"),
            ("FrankaPickHardBench_20251222_benchmark", "FrankaPickHardBench_2000ep_json_benchmark"),
            ("FrankaPickHolodeckEasyBench_20260116_benchmark", "FrankaPickHolodeckEasyBench_2000ep_json_benchmark"),
            ("FrankaPickHolodeckHardBench_20260116_benchmark", "FrankaPickHolodeckHardBench_2000ep_json_benchmark"),
            ]
    for a,b in pairs:
        process_pkl(Path(a_base) / a, "")
        process_json(Path(b_base) / b, "benchmark.json")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Recursively load .pkl files and print selected attributes."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory to recursively search for .pkl files",
    )
    parser.add_argument(
        "--attrs",
        nargs="+",
        default=DEFAULT_ATTRS,
        help=(
            "List of attributes / dict keys to print "
            f"(default: {', '.join(DEFAULT_ATTRS)})"
        ),
    )

    parser.add_argument("--check",type=bool, default=0)

    args = parser.parse_args()

    if args.check:
        check_configs()

    for dirpath, _, filenames in os.walk(args.root_dir):
        for fname in filenames:
            if fname.endswith(".pkl"):
                if "_benchmark" not in dirpath:
                    continue
                process_pkl(dirpath, fname, args)
            elif fname.endswith("benchmark.json"):
                process_json(dirpath, fname)



if __name__ == "__main__":
    check_configs()
    #main()
