#!/usr/bin/env python3
"""Aggregate RoboDojo eval results into a markdown table.

Layout on disk:
    <task>/<policy>/<embodiment>/<seed>_ckpt_name=...,action_type=.../<timestamp>/_result.json

Rules:
  * For every (task, policy, embodiment, seed) we only read the *latest*
    timestamp folder.
  * `_result.json` -> `details` holds per-layout entries with `success` (bool)
    and `score` (float).
  * A task `X` that also has a sibling task `X_random` is a "paired" task:
    we take the first 25 entries of `X` and the first 25 of `X_random`
    (matched by policy + embodiment + seed) and merge them into 50.
  * Every other (non `_random`) task is "standalone": we take the first 50.
  * `_random` tasks are not reported on their own; they only feed their base.
  * A cell is filled only when the required episode count is present in JSON
    (50 for standalone, 25+25 for paired); otherwise the cell stays blank.
  * Each policy table lists seeds 0, 1, 2; incomplete cells are left blank.
  * The overview table lists every policy with at least one completed
    (task, seed) cell. Progress = completed cells / 126 (42 tasks x 3 seeds).
    Generalization-dimension tasks require 25 base + 25 random episodes;
    all other tasks require 50 episodes. Policies still in progress are
    marked with dagger in Progress. Dimension scores and overall averages use
    completed cells so far (partial averages when incomplete).
  * success_rate = successes / count * 100
    score        = sum(scores) / count * 100
"""

import json
import os
import re
import statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get(
    "ROBODOJO_EVAL_ROOT",
    os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "eval_result", "RoboDojo")),
)
OUTPUT_MD = os.path.join(ROOT, "_summary.md")

SEED_RE = re.compile(r"^(\d+)_")
STANDALONE_EPISODES = 50
PAIRED_HALF_EPISODES = 25
EXPECTED_SEEDS = [0, 1, 2]

# Capability dimension -> tasks belonging to it. Used for the overview table.
# Order here controls the column order in the overview table.
DIMENSIONS = {
    "Generalization": [
        "stack_bowls", "push_T", "pack_objects_into_box", "fold_clothes",
        "hang_mugs", "sweep_blocks", "pour_liquid_into_cup", "make_toast",
        "arrange_largest_number", "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones", "stack_blocks",
    ],
    "Precision": [
        "fasten_screws", "plug_in_charger", "insert_tubes",
        "pour_balls_into_vase", "play_Xylophone", "deposit_coin",
        "insert_key", "build_tower",
    ],
    "Long-Horizon": [
        "put_bottles_into_dustbin", "fill_pen_holder", "classify_objects",
        "play_tic_tac_toe", "fill_egg_holder", "organize_table",
        "make_kong", "play_stacking_toy",
    ],
    "Memory": [
        "cover_blocks", "match_and_pick_from_conveyor", "swap_blocks",
        "swap_T", "press_by_number", "imitate_sorting_sequence",
    ],
    "Open": [
        "align_blocks", "general_pickup", "stack_blocks_by_language",
        "solve_equation", "classify_objects_by_language",
        "pick_from_conveyor_by_image", "store_tools_in_toolbox",
        "pour_by_language",
    ],
}

ALL_TASKS = sorted({task for tasks in DIMENSIONS.values() for task in tasks})
GENERALIZATION_TASKS = set(DIMENSIONS["Generalization"])
TOTAL_REQUIRED = len(ALL_TASKS) * len(EXPECTED_SEEDS)  # 42 x 3 = 126


def list_subdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    )


def latest_timestamp_dir(run_dir):
    """Return the latest timestamp subdir that contains a `_result.json`."""
    candidates = []
    for name in list_subdirs(run_dir):
        if os.path.isfile(os.path.join(run_dir, name, "_result.json")):
            candidates.append(name)
    if not candidates:
        return None
    # Timestamp format `YYYY-MM-DD_HH-MM-SS` sorts chronologically as a string.
    return os.path.join(run_dir, sorted(candidates)[-1])


def load_entries(ts_dir):
    """Load (success, score) entries sorted by layout id from a timestamp dir."""
    try:
        with open(os.path.join(ts_dir, "_result.json")) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    details = data.get("details", {})
    items = []
    for key, entry in details.items():
        try:
            layout = int(key)
        except (TypeError, ValueError):
            continue
        items.append((layout, bool(entry.get("success", False)),
                      float(entry.get("score", 0.0) or 0.0)))
    items.sort(key=lambda x: x[0])
    return [(s, sc) for _, s, sc in items]


def scan_task(task):
    """Return {(policy, embodiment, seed): [(success, score), ...]} for a task."""
    task_path = os.path.join(ROOT, task)
    out = {}
    latest_ts = {}
    for policy in list_subdirs(task_path):
        policy_path = os.path.join(task_path, policy)
        for embodiment in list_subdirs(policy_path):
            emb_path = os.path.join(policy_path, embodiment)
            for run in list_subdirs(emb_path):
                m = SEED_RE.match(run)
                if not m:
                    continue
                seed = int(m.group(1))
                ts_dir = latest_timestamp_dir(os.path.join(emb_path, run))
                if ts_dir is None:
                    continue
                ts_name = os.path.basename(ts_dir)
                entries = load_entries(ts_dir)
                if entries is None:
                    continue
                key = (policy, embodiment, seed)
                if key not in out or ts_name > latest_ts[key]:
                    out[key] = entries
                    latest_ts[key] = ts_name
    return out


def collect_policy_seeds():
    """Return {policy: sorted seeds} seen anywhere under ROOT."""
    seeds = {}
    for name in list_subdirs(ROOT):
        if name.endswith("_random"):
            continue
        task_path = os.path.join(ROOT, name)
        for policy in list_subdirs(task_path):
            policy_path = os.path.join(task_path, policy)
            for embodiment in list_subdirs(policy_path):
                emb_path = os.path.join(policy_path, embodiment)
                for run in list_subdirs(emb_path):
                    m = SEED_RE.match(run)
                    if not m:
                        continue
                    seeds.setdefault(policy, set()).add(int(m.group(1)))
    return {policy: sorted(values) for policy, values in seeds.items()}


def stats(entries):
    count = len(entries)
    if count == 0:
        return 0, 0.0, 0.0
    successes = sum(1 for s, _ in entries if s)
    score_sum = sum(sc for _, sc in entries)
    return count, successes / count * 100.0, score_sum / count * 100.0


def discover_random_tasks():
    """Return {base task -> random task name} for Generalization tasks on disk."""
    all_task_dirs = set(list_subdirs(ROOT))
    random_of = {}
    for task in GENERALIZATION_TASKS:
        random_name = task + "_random"
        if random_name in all_task_dirs:
            random_of[task] = random_name
    return random_of


def is_paired_task(task, random_of):
    """Generalization tasks always use 25 base + 25 random episodes."""
    return task in GENERALIZATION_TASKS and task in random_of


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(
            f"Eval result directory not found: {ROOT}\n"
            "Set ROBODOJO_EVAL_ROOT or run from the RoboDojo repo with "
            "eval_result/RoboDojo present."
        )

    random_of = discover_random_tasks()
    policy_seeds = collect_policy_seeds()

    # data[policy][task][seed] = (success_rate, score)
    data = {policy: {} for policy in policy_seeds}
    # gen_split[policy][task][seed] = {"base": (sr, score), "random": (sr, score)}
    gen_split = {policy: {} for policy in policy_seeds}

    def record(policy, task, seed, sr, score):
        data.setdefault(policy, {}).setdefault(task, {})[seed] = (sr, score)

    def record_gen_half(policy, task, seed, half, sr, score):
        gen_split.setdefault(policy, {}).setdefault(task, {}).setdefault(seed, {})[half] = (
            sr, score
        )

    for task in ALL_TASKS:
        base_scan = scan_task(task)

        if is_paired_task(task, random_of):
            rand_scan = scan_task(random_of[task])
            for key in sorted(set(base_scan) | set(rand_scan)):
                policy, _embodiment, seed = key
                base_e = base_scan.get(key, [])
                rand_e = rand_scan.get(key, [])
                if len(base_e) >= PAIRED_HALF_EPISODES:
                    _, base_sr, base_score = stats(base_e[:PAIRED_HALF_EPISODES])
                    record_gen_half(policy, task, seed, "base", base_sr, base_score)
                if len(rand_e) >= PAIRED_HALF_EPISODES:
                    _, rand_sr, rand_score = stats(rand_e[:PAIRED_HALF_EPISODES])
                    record_gen_half(policy, task, seed, "random", rand_sr, rand_score)
                if (len(base_e) < PAIRED_HALF_EPISODES
                        or len(rand_e) < PAIRED_HALF_EPISODES):
                    continue
                merged = (base_e[:PAIRED_HALF_EPISODES]
                          + rand_e[:PAIRED_HALF_EPISODES])
                _, sr, score = stats(merged)
                record(policy, task, seed, sr, score)
        else:
            for key in sorted(base_scan):
                entries = base_scan[key]
                if len(entries) < STANDALONE_EPISODES:
                    continue
                _, sr, score = stats(entries[:STANDALONE_EPISODES])
                policy, _embodiment, seed = key
                record(policy, task, seed, sr, score)

    write_markdown(data, gen_split)
    n_cells = sum(len(seeds) for tasks in data.values() for seeds in tasks.values())
    complete = sum(1 for p in data if policy_is_complete(p, data))
    tested = sum(1 for p in data if policy_progress(p, data)[0] > 0)
    print(
        f"Wrote {len(data)} policy tables ({n_cells} filled cells, "
        f"{tested} in overview, {complete} complete) to {OUTPUT_MD}"
    )


def mean_with_spread(values, integer=False):
    """Format `mean ±amp` where amp is half the (max - min) range across seeds.

    Returns "" when there is no data so the cell stays blank.
    """
    if not values:
        return ""
    mean = sum(values) / len(values)
    amp = (max(values) - min(values)) / 2.0
    fmt = "{:.0f}" if integer else "{:.2f}"
    return f"{fmt.format(mean)} ±{fmt.format(amp)}"


def task_seed_means(task_data, seeds):
    """Average a task's per-seed (sr, score) over the given seed list.

    Returns None unless every seed has a completed result.
    """
    if not seeds:
        return None
    try:
        values = [task_data[seed] for seed in seeds]
    except KeyError:
        return None
    srs = [sr for sr, _ in values]
    scores = [sc for _, sc in values]
    return sum(srs) / len(srs), sum(scores) / len(scores)


def task_partial_means(task_data, seeds):
    """Average (sr, score) over seeds that have data; None if none."""
    values = [task_data[seed] for seed in seeds if seed in task_data]
    if not values:
        return None
    srs = [sr for sr, _ in values]
    scores = [sc for _, sc in values]
    return sum(srs) / len(srs), sum(scores) / len(scores)


def dimension_partial_means(policy_data, tasks, seeds):
    """Return (dim_sr, dim_score) averaged over tasks with any completed seed."""
    task_srs = []
    task_scores = []
    for task in tasks:
        means = task_partial_means(policy_data.get(task, {}), seeds)
        if means is None:
            continue
        task_srs.append(means[0])
        task_scores.append(means[1])
    if not task_srs:
        return None
    return (sum(task_srs) / len(task_srs), sum(task_scores) / len(task_scores))


def dimension_means(policy_data, tasks, seeds):
    """Return (dim_sr, dim_score) averaged over tasks with full seed coverage."""
    task_srs = []
    task_scores = []
    for task in tasks:
        means = task_seed_means(policy_data.get(task, {}), seeds)
        if means is None:
            return None
        task_srs.append(means[0])
        task_scores.append(means[1])
    if not task_srs:
        return None
    return (sum(task_srs) / len(task_srs), sum(task_scores) / len(task_scores))


def policy_progress(policy, data):
    """Return (filled_cells, total_cells, percent) for a policy."""
    policy_data = data.get(policy, {})
    filled = sum(
        1 for task in ALL_TASKS
        for seed in EXPECTED_SEEDS
        if seed in policy_data.get(task, {})
    )
    pct = filled / TOTAL_REQUIRED * 100.0
    return filled, TOTAL_REQUIRED, pct


def format_progress(filled, total, pct, complete):
    """Format progress for the overview table."""
    if complete:
        return f"{pct:.0f}%"
    return f"{pct:.1f}% † ({filled}/{total})"


def policy_is_complete(policy, data):
    """True when all 126 (42 tasks x 3 seeds) cells are filled."""
    policy_data = data.get(policy, {})
    for task in ALL_TASKS:
        if task_seed_means(policy_data.get(task, {}), EXPECTED_SEEDS) is None:
            return False
    return True


def dimension_seed_value(policy_data, tasks, seed):
    """Return (sr, score) for one dimension at one seed.

    Averaged over the dimension tasks that have data at this seed; None when
    none of them do.
    """
    srs = []
    scores = []
    for task in tasks:
        task_data = policy_data.get(task, {})
        if seed in task_data:
            sr, score = task_data[seed]
            srs.append(sr)
            scores.append(score)
    if not srs:
        return None
    return sum(srs) / len(srs), sum(scores) / len(scores)


def overall_seed_value(policy_data, seed):
    """Return (sr, score) at one seed as the mean of the per-dimension values."""
    dim_srs = []
    dim_scores = []
    for tasks in DIMENSIONS.values():
        value = dimension_seed_value(policy_data, tasks, seed)
        if value is None:
            continue
        dim_srs.append(value[0])
        dim_scores.append(value[1])
    if not dim_srs:
        return None
    return sum(dim_srs) / len(dim_srs), sum(dim_scores) / len(dim_scores)


def mean_std(values):
    """Return (mean, population std) over values, or None when empty."""
    if not values:
        return None
    mean = sum(values) / len(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def format_mean_std(stat):
    """Format `mean ±std` to two decimals; blank when no data."""
    if stat is None:
        return ""
    mean, std = stat
    return f"{mean:.2f} ±{std:.2f}"


def policy_seed_summary(policy_data):
    """Return (per_dim_stats, overall_stats) of mean±std across seeds.

    per_dim_stats[dim] = ((sr_mean, sr_std), (score_mean, score_std)) or None.
    overall_stats has the same shape, computed from the per-seed overall means.
    """
    per_dim = {}
    for dim, tasks in DIMENSIONS.items():
        sr_vals = []
        score_vals = []
        for seed in EXPECTED_SEEDS:
            value = dimension_seed_value(policy_data, tasks, seed)
            if value is None:
                continue
            sr_vals.append(value[0])
            score_vals.append(value[1])
        sr_stat = mean_std(sr_vals)
        score_stat = mean_std(score_vals)
        per_dim[dim] = (sr_stat, score_stat) if sr_stat else None

    overall_srs = []
    overall_scores = []
    for seed in EXPECTED_SEEDS:
        value = overall_seed_value(policy_data, seed)
        if value is None:
            continue
        overall_srs.append(value[0])
        overall_scores.append(value[1])
    overall = None
    if overall_srs:
        overall = (mean_std(overall_srs), mean_std(overall_scores))
    return per_dim, overall


def gen_half_seed_value(policy_gen_data, half, seed):
    """Average SR/Score for one Generalization half at one seed."""
    srs = []
    scores = []
    for task in DIMENSIONS["Generalization"]:
        seed_data = policy_gen_data.get(task, {}).get(seed, {})
        if half not in seed_data:
            continue
        sr, score = seed_data[half]
        srs.append(sr)
        scores.append(score)
    if not srs:
        return None
    return sum(srs) / len(srs), sum(scores) / len(scores)


def policy_gen_half_summary(policy_gen_data):
    """Return (base_sr, base_score, random_sr, random_score) as mean±std stats."""
    halves = {}
    for half in ("base", "random"):
        sr_vals = []
        score_vals = []
        for seed in EXPECTED_SEEDS:
            value = gen_half_seed_value(policy_gen_data, half, seed)
            if value is None:
                continue
            sr_vals.append(value[0])
            score_vals.append(value[1])
        sr_stat = mean_std(sr_vals)
        score_stat = mean_std(score_vals)
        halves[half] = (sr_stat, score_stat) if sr_stat else None
    return halves.get("base"), halves.get("random")


def build_generalization_split(gen_split):
    """Compare Generalization standard (25) vs random (25) per policy."""
    lines = [
        "## Generalization: Standard vs Random (25 episodes each)",
        "",
        "Each cell shows `mean ±std` across seeds, averaged over the 12 "
        "Generalization tasks with completed data for that half. "
        "Δ = Random − Standard (mean SR). Sorted by Standard SR.",
        "",
    ]

    header = [
        "Policy",
        "Standard SR (%)", "Standard Score",
        "Random SR (%)", "Random Score",
        "Δ SR", "Δ Score",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * (len(header) - 1)) + " |")

    rows = []
    for policy, policy_gen_data in gen_split.items():
        base_stats, random_stats = policy_gen_half_summary(policy_gen_data)
        if base_stats is None and random_stats is None:
            continue
        sort_key = base_stats[0][0] if base_stats else float("-inf")
        cells = [policy]
        for stats_pair in (base_stats, random_stats):
            if stats_pair is None:
                cells.extend(["", ""])
            else:
                sr_stat, score_stat = stats_pair
                cells.append(format_mean_std(sr_stat))
                cells.append(format_mean_std(score_stat))
        if base_stats and random_stats:
            base_sr, base_score = base_stats[0][0], base_stats[1][0]
            random_sr, random_score = random_stats[0][0], random_stats[1][0]
            delta_sr = random_sr - base_sr
            delta_score = random_score - base_score
            sign_sr = "+" if delta_sr >= 0 else ""
            sign_score = "+" if delta_score >= 0 else ""
            cells.append(f"{sign_sr}{delta_sr:.2f}")
            cells.append(f"{sign_score}{delta_score:.2f}")
        else:
            cells.extend(["", ""])
        rows.append((sort_key, policy, cells))

    rows.sort(key=lambda r: (-r[0], r[1]))
    for _, _, cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def build_overview(data):
    """Overview of tested policies, sorted by overall SR (descending).

    Each dimension and the overall Avg show `mean ±std`, where the mean and
    std are taken across the (up to three) seeds. Scores are blank until a
    cell has data; Generalization cells need 25 base + 25 random episodes.
    """
    lines = [
        "## Overview (by capability dimension)",
        "",
        "Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). "
        "Cells show `mean ±std` across seeds; scores reflect completed cells "
        "only. Policies are sorted by overall Avg SR.",
        "",
    ]

    header = ["Policy", "Progress"]
    for dim in DIMENSIONS:
        header.append(f"{dim} SR (%)")
        header.append(f"{dim} Score")
    header.append("Avg SR (%)")
    header.append("Avg Score")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * (len(header) - 1)) + " |")

    rows = []
    for policy in data:
        filled, total, pct = policy_progress(policy, data)
        if filled == 0:
            continue
        complete = policy_is_complete(policy, data)
        per_dim, overall = policy_seed_summary(data[policy])
        sort_key = overall[0][0] if overall else float("-inf")

        cells = [policy, format_progress(filled, total, pct, complete)]
        for dim in DIMENSIONS:
            stat = per_dim.get(dim)
            if stat is None:
                cells.extend(["", ""])
            else:
                sr_stat, score_stat = stat
                cells.append(format_mean_std(sr_stat))
                cells.append(format_mean_std(score_stat))
        if overall is None:
            cells.extend(["", ""])
        else:
            cells.append(format_mean_std(overall[0]))
            cells.append(format_mean_std(overall[1]))
        rows.append((sort_key, policy, cells))

    rows.sort(key=lambda r: (-r[0], r[1]))
    for _, _, cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def seed_column_means(policy_data, seed):
    """Average SR/Score for one seed across all tasks with data."""
    srs = []
    scores = []
    for task in ALL_TASKS:
        task_data = policy_data.get(task, {})
        if seed in task_data:
            sr, score = task_data[seed]
            srs.append(sr)
            scores.append(score)
    if not srs:
        return None
    return sum(srs) / len(srs), sum(scores) / len(scores)


def format_task_row(task, task_data):
    """Build one task row: per-seed cells plus cross-seed average."""
    cells = [f"<nobr>{task}</nobr>"]
    srs = []
    scores = []
    for seed in EXPECTED_SEEDS:
        if seed in task_data:
            sr, score = task_data[seed]
            cells.append(f"{sr:.0f}")
            cells.append(f"{score:.2f}")
            srs.append(sr)
            scores.append(score)
        else:
            cells.append("")
            cells.append("")
    if len(srs) == len(EXPECTED_SEEDS):
        cells.append(mean_with_spread(srs, integer=True))
        cells.append(mean_with_spread(scores))
    else:
        cells.extend(["", ""])
    return cells


def format_seed_avg_row(policy_data):
    """Summary row: per-seed average across all tasks."""
    cells = ["**Avg (per seed)**"]
    seed_srs = []
    seed_scores = []
    for seed in EXPECTED_SEEDS:
        means = seed_column_means(policy_data, seed)
        if means is None:
            cells.extend(["", ""])
        else:
            sr, score = means
            cells.append(f"{sr:.1f}")
            cells.append(f"{score:.2f}")
            seed_srs.append(sr)
            seed_scores.append(score)
    if seed_srs:
        cells.append(mean_with_spread(seed_srs, integer=True))
        cells.append(mean_with_spread(seed_scores))
    else:
        cells.extend(["", ""])
    return cells


def write_markdown(data, gen_split):
    lines = ["# RoboDojo Evaluation Summary", ""]
    lines += build_overview(data)
    lines += build_generalization_split(gen_split)

    for policy in sorted(data):
        lines.append(f"## {policy}")
        lines.append("")

        header = ["Task"]
        for seed in EXPECTED_SEEDS:
            header.append(f"Seed {seed} SR (%)")
            header.append(f"Seed {seed} Score")
        header.append("Avg SR (%)")
        header.append("Avg Score")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| --- | " + " | ".join(["---:"] * (len(header) - 1)) + " |")

        policy_data = data[policy]
        for task in ALL_TASKS:
            cells = format_task_row(task, policy_data.get(task, {}))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("| " + " | ".join(format_seed_avg_row(policy_data)) + " |")
        lines.append("")

    with open(OUTPUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
