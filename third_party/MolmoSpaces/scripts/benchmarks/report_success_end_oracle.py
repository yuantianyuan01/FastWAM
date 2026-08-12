#!/usr/bin/env python3
"""
Report success rate for an eval run using two criteria:
  - Last frame only (same as current reporting): success if success[-1] is True.
  - Any of last 5 frames: success if any(success[-5:]) is True.

Takes the path to an eval directory that contains episode data in HDF5 form
(house_*/trajectories*.h5 with traj_* groups, or episode_*/**/*.h5, etc.).
Finds all "success" arrays and aggregates.

Usage:
  python launch_scripts/synthvla/report_success_last5.py /path/to/eval_dir
  python launch_scripts/synthvla/report_success_last5.py /path/to/eval_dir --last-n 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np


def find_h5_trajectories(eval_dir: Path):
    """Yield (h5_path, traj_key, episode_idx) for every traj_* group under eval_dir."""
    eval_dir = Path(eval_dir).resolve()
    if not eval_dir.is_dir():
        raise NotADirectoryError(f"Eval dir is not a directory: {eval_dir}")

    for h5_path in sorted(eval_dir.rglob("*.h5")):
        try:
            with h5py.File(h5_path, "r") as f:
                for key in sorted(f.keys()):
                    if not key.startswith("traj_"):
                        continue
                    try:
                        episode_idx = int(key.split("_")[1])
                    except (IndexError, ValueError):
                        episode_idx = -1
                    yield h5_path, key, episode_idx
        except OSError as e:
            print(f"Warning: could not open {h5_path}: {e}", flush=True)


def get_canonical_episode_id(eval_dir: Path, h5_path: Path, traj_key: str):
    """Return a stable id for this episode so duplicates (same episode run twice) can be skipped.

    When the HDF5 lies under a directory named episode_XXXXXXXX, we use that dir name
    so only one instance per episode is counted. Otherwise we use (relative path, traj_key).
    When multiple instances exist, we pick one at random to include.
    """
    try:
        rel = h5_path.relative_to(eval_dir)
        parts = rel.parts
        if parts and parts[0].startswith("episode_") and parts[0][len("episode_") :].isdigit():
            return parts[0]
        return (rel.as_posix(), traj_key)
    except ValueError:
        return (h5_path.name, traj_key)


def get_success_any(success_array: np.ndarray) -> bool:
    """True if any of the elements of success_array are True."""
    if success_array is None or len(success_array) == 0:
        return False
    return bool(np.any(success_array))


def get_success_last_frame(success_array: np.ndarray) -> bool:
    """True iff the last element of success_array is True (current metric)."""
    if success_array is None or len(success_array) == 0:
        return False
    return bool(success_array[-1])


def load_success_array(h5_path: Path, traj_key: str) -> np.ndarray | None:
    """Load the 'success' dataset from the given traj group, or None if missing."""
    with h5py.File(h5_path, "r") as f:
        group = f.get(traj_key)
        if group is None or "success" not in group:
            return None
        return np.asarray(group["success"])


def main():
    parser = argparse.ArgumentParser(
        description="Report success rate (last frame vs any of last N frames) for an eval dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "eval_dir",
        type=str,
        help="Path to the evaluation output directory (contains episode HDF5 data)",
    )
    parser.add_argument(
        "--show-per-episode",
        action="store_true",
        help="Print per-episode last-frame vs last-N success",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Collect all (canonical_id, h5_path, traj_key); group by canonical_id
    id_to_instances: dict = {}  # canonical_id -> list of (h5_path, traj_key)
    for h5_path, traj_key, episode_idx in find_h5_trajectories(eval_dir):
        canonical_id = get_canonical_episode_id(eval_dir, h5_path, traj_key)
        id_to_instances.setdefault(canonical_id, []).append((h5_path, traj_key))

    # For each episode, pick one instance at random when there are duplicates
    chosen = [
        (canonical_id, random.choice(instances))
        for canonical_id, instances in sorted(id_to_instances.items())
    ]

    results: list[tuple[bool, bool]] = []  # (success_last_frame, success_any)
    episode_labels: list[str] = []

    for canonical_id, (h5_path, traj_key) in chosen:
        success_arr = load_success_array(h5_path, traj_key)
        if success_arr is None:
            continue
        s_last = get_success_last_frame(success_arr)
        s_any = get_success_any(success_arr)
        results.append((s_last, s_any))
        if isinstance(canonical_id, str):
            episode_labels.append(canonical_id)
        else:
            try:
                rel = h5_path.relative_to(eval_dir)
                parts = rel.parts
                if parts and parts[0].startswith("episode_"):
                    episode_labels.append(parts[0])
                else:
                    episode_labels.append(f"{h5_path.stem}/{traj_key}")
            except ValueError:
                episode_labels.append(f"{h5_path.name}/{traj_key}")

    if not results:
        print(f"No trajectory data found under {eval_dir}")
        print("Expected HDF5 files with groups named traj_* containing a 'success' dataset.")
        return

    n = len(results)
    last_frame_successes = sum(1 for s_last, _ in results if s_last)
    any_successes = sum(1 for _, s_any in results if s_any)

    print(f"Eval dir: {eval_dir}")
    print(f"Episodes: {n}")
    print()
    print("Success rate (last frame only, current metric):")
    print(f"  {last_frame_successes} / {n} = {last_frame_successes / n:.2%}")
    print()
    print(f"Success rate (any):")
    print(f"  {any_successes} / {n} = {any_successes / n:.2%}")
    print()
    extra = any_successes - last_frame_successes
    if extra > 0:
        #print(f"  ({extra} additional episodes count as success under last-{last_n} criterion)")
        pass

    if args.show_per_episode and results:
        print()
        print("Per-episode (last_frame -> last_N):")
        for i, (label, (s_last, s_last_n)) in enumerate(zip(episode_labels, results)):
            a = "pass" if s_last else "fail"
            b = "pass" if s_last_n else "fail"
            print(f"  {label}: {a} -> {b}")


if __name__ == "__main__":
    main()
