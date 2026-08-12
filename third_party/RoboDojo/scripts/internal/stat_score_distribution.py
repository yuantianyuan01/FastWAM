#!/usr/bin/env python3
"""Count per-score episode frequencies for selected policies under eval_result.

Example output (conceptually):
    stack_bowls / Pi_05: 5 x 0.5, 10 x 0.0, ...

Directory layout (same as summarize_result.py):
    <task>/<policy>/<embodiment>/<seed>_ckpt_name=...,action_type=.../<timestamp>/_result.json

Rules:
  * Only the latest timestamp folder is read for each (task, policy, embodiment, seed).
  * Standalone tasks use the first 50 episodes.
  * Paired tasks (task + task_random) use the first 25 from each half.
  * ``*_random`` tasks are not reported on their own.
  * Scores are aggregated across all seeds by default; use ``--per-seed`` for a breakdown.

Usage:
    python stat_score_distribution.py
    python stat_score_distribution.py --policies Pi_05 X_VLA Xiaomi_Robotics_0
    python scripts/internal/stat_score_distribution.py --task stack_bowls --json-out score_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "eval_result", "RoboDojo")
)

DEFAULT_POLICIES = ("Xiaomi_Robotics_0", "X_VLA", "Pi_05")
POLICY_ALIASES = {
    "xiaomi_robotics_0": "Xiaomi_Robotics_0",
}

SEED_RE = re.compile(r"^(\d+)_")
STANDALONE_EPISODES = 50
PAIRED_HALF_EPISODES = 25


def normalize_policy_name(name: str) -> str:
    return POLICY_ALIASES.get(name, name)


def list_subdirs(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []
    return sorted(
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    )


def latest_timestamp_dir(run_dir: str) -> str | None:
    candidates = []
    for name in list_subdirs(run_dir):
        if os.path.isfile(os.path.join(run_dir, name, "_result.json")):
            candidates.append(name)
    if not candidates:
        return None
    return os.path.join(run_dir, sorted(candidates)[-1])


def load_scores(ts_dir: str) -> list[float] | None:
    try:
        with open(os.path.join(ts_dir, "_result.json")) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    details = data.get("details", {})
    items: list[tuple[int, float]] = []
    for key, entry in details.items():
        try:
            layout = int(key)
        except (TypeError, ValueError):
            continue
        score = float(entry.get("score", 0.0) or 0.0)
        items.append((layout, score))
    items.sort(key=lambda x: x[0])
    return [score for _, score in items]


def scan_task(root: str, task: str) -> dict[tuple[str, str, int], list[float]]:
    """Return {(policy, embodiment, seed): [scores...]} for one task."""
    task_path = os.path.join(root, task)
    out: dict[tuple[str, str, int], list[float]] = {}
    latest_ts: dict[tuple[str, str, int], str] = {}

    for policy in list_subdirs(task_path):
        policy_path = os.path.join(task_path, policy)
        for embodiment in list_subdirs(policy_path):
            emb_path = os.path.join(policy_path, embodiment)
            for run in list_subdirs(emb_path):
                match = SEED_RE.match(run)
                if not match:
                    continue
                seed = int(match.group(1))
                ts_dir = latest_timestamp_dir(os.path.join(emb_path, run))
                if ts_dir is None:
                    continue
                scores = load_scores(ts_dir)
                if scores is None:
                    continue
                key = (policy, embodiment, seed)
                ts_name = os.path.basename(ts_dir)
                if key not in out or ts_name > latest_ts[key]:
                    out[key] = scores
                    latest_ts[key] = ts_name
    return out


def discover_tasks(root: str) -> list[str]:
    return sorted(
        name
        for name in list_subdirs(root)
        if not name.endswith("_random") and not name.startswith("_")
    )


def discover_random_tasks(root: str, tasks: list[str]) -> dict[str, str]:
    all_task_dirs = set(list_subdirs(root))
    random_of: dict[str, str] = {}
    for task in tasks:
        random_name = task + "_random"
        if random_name in all_task_dirs:
            random_of[task] = random_name
    return random_of


def scores_for_run(
    base_scan: dict[tuple[str, str, int], list[float]],
    rand_scan: dict[tuple[str, str, int], list[float]] | None,
    key: tuple[str, str, int],
    paired: bool,
) -> list[float] | None:
    base_scores = base_scan.get(key)
    if base_scores is None:
        return None

    if paired:
        if rand_scan is None:
            return None
        rand_scores = rand_scan.get(key)
        if rand_scores is None:
            return None
        if (
            len(base_scores) < PAIRED_HALF_EPISODES
            or len(rand_scores) < PAIRED_HALF_EPISODES
        ):
            return None
        return base_scores[:PAIRED_HALF_EPISODES] + rand_scores[:PAIRED_HALF_EPISODES]

    if len(base_scores) < STANDALONE_EPISODES:
        return None
    return base_scores[:STANDALONE_EPISODES]


def format_score(score: float) -> str:
    rounded = round(score, 4)
    if rounded == int(rounded):
        return str(int(rounded))
    text = f"{rounded:.4f}".rstrip("0").rstrip(".")
    return text


def counter_to_dict(counter: Counter[float]) -> dict[str, int]:
    return {format_score(score): count for score, count in sorted(counter.items())}


def format_distribution(counter: Counter[float]) -> str:
    if not counter:
        return "(no data)"
    parts = [
        f"{count} x {format_score(score)}"
        for score, count in sorted(counter.items())
    ]
    return ", ".join(parts)


def collect_distributions(
    root: str,
    policies: list[str],
    tasks: list[str] | None = None,
    per_seed: bool = False,
) -> dict:
    all_tasks = discover_tasks(root)
    if tasks:
        unknown = sorted(set(tasks) - set(all_tasks))
        if unknown:
            raise SystemExit(f"Unknown task(s): {', '.join(unknown)}")
        selected_tasks = sorted(tasks)
    else:
        selected_tasks = all_tasks

    random_of = discover_random_tasks(root, selected_tasks)
    policy_set = set(policies)

    # aggregated[task][policy] -> Counter
    aggregated: dict[str, dict[str, Counter[float]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    # per_seed[task][policy][seed] -> Counter
    by_seed: dict[str, dict[str, dict[int, Counter[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )

    for task in selected_tasks:
        base_scan = scan_task(root, task)
        rand_scan = scan_task(root, random_of[task]) if task in random_of else None
        paired = task in random_of

        keys = set(base_scan)
        if rand_scan is not None:
            keys &= set(rand_scan)

        for key in sorted(keys):
            policy, _embodiment, seed = key
            if policy not in policy_set:
                continue
            scores = scores_for_run(base_scan, rand_scan, key, paired)
            if scores is None:
                continue
            aggregated[task][policy].update(scores)
            if per_seed:
                by_seed[task][policy][seed].update(scores)

    result = {
        "root": root,
        "policies": policies,
        "tasks": selected_tasks,
        "aggregated": {
            task: {
                policy: counter_to_dict(counter)
                for policy, counter in sorted(task_data.items())
            }
            for task, task_data in sorted(aggregated.items())
        },
    }
    if per_seed:
        result["per_seed"] = {
            task: {
                policy: {
                    str(seed): counter_to_dict(counter)
                    for seed, counter in sorted(seed_data.items())
                }
                for policy, seed_data in sorted(task_data.items())
            }
            for task, task_data in sorted(by_seed.items())
        }
    return result


def print_report(result: dict, per_seed: bool) -> None:
    policies = result["policies"]
    aggregated = result["aggregated"]

    print(f"Eval root: {result['root']}")
    print(f"Policies: {', '.join(policies)}")
    print()

    for task in result["tasks"]:
        task_data = aggregated.get(task, {})
        if not any(task_data.get(policy) for policy in policies):
            continue

        print(f"## {task}")
        for policy in policies:
            dist = task_data.get(policy, {})
            if not dist:
                print(f"  {policy}: (no completed eval)")
                continue
            total = sum(dist.values())
            detail = ", ".join(f"{count} x {score}" for score, count in dist.items())
            print(f"  {policy} ({total} episodes): {detail}")

            if per_seed:
                seed_data = result.get("per_seed", {}).get(task, {}).get(policy, {})
                for seed, seed_dist in sorted(seed_data.items(), key=lambda x: int(x[0])):
                    seed_total = sum(seed_dist.values())
                    seed_detail = ", ".join(
                        f"{count} x {score}" for score, count in seed_dist.items()
                    )
                    print(f"    seed {seed} ({seed_total}): {seed_detail}")
        print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count how many episodes received each score for selected policies."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("ROBODOJO_EVAL_ROOT", DEFAULT_ROOT),
        help="Path to eval_result/RoboDojo (default: %(default)s)",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_POLICIES),
        help="Policy names to include (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Only include this task; repeatable",
    )
    parser.add_argument(
        "--per-seed",
        action="store_true",
        help="Also show score counts per seed",
    )
    parser.add_argument(
        "--json-out",
        help="Write the full result dict to this JSON file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policies = [normalize_policy_name(name) for name in args.policies]

    if not os.path.isdir(args.root):
        print(f"Eval result directory not found: {args.root}", file=sys.stderr)
        return 1

    result = collect_distributions(
        root=args.root,
        policies=policies,
        tasks=args.tasks,
        per_seed=args.per_seed,
    )

    print_report(result, per_seed=args.per_seed)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"Wrote JSON to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
