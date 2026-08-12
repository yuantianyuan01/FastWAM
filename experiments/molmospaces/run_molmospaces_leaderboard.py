"""FastWAM MolmoSpaces leaderboard wrapper.

This wrapper keeps the official MolmoSpaces evaluation pipeline intact and only
adds FastWAM-specific task grouping, camera selection, and CSV aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "third_party" / "MolmoSpaces"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@dataclass(frozen=True)
class Task:
    key: str
    name: str
    bench_subpath: str
    exterior_camera: str
    wrist_camera: str
    filament: bool = False


V1 = "molmospaces-bench-v1"
V2 = "molmospaces-bench-v2"
EXO, WR = "exo_camera_1", "wrist_camera"
SHO, ZED = "droid_shoulder_light_randomization", "wrist_camera_zed_mini"
RND = "randomized_zed2_analogue_1"

TASKS = [
    Task("pick_msproc", "Pick-MSProc", f"{V2}/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231", EXO, WR),
    Task("pick_classic", "Pick-Classic", f"{V2}/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark", SHO, ZED),
    Task("pick_filament", "Pick-Filament", f"{V2}/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark", SHO, ZED, True),
    Task("pick_randcam", "Pick-RandCam", f"{V2}/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark", RND, ZED, True),
    Task("pnp_v2", "PnP-v2", f"{V2}/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark", SHO, ZED, True),
    Task("pnp_nextto", "PnP-NextTo-v2", f"{V2}/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark", SHO, ZED, True),
    Task("pnp_color", "PnP-Color-v2", f"{V2}/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark", SHO, ZED, True),
]
TASKS_BY_KEY = {task.key: task for task in TASKS}
SETS = {"molmobot": [task.key for task in TASKS]}


def _assets_dir() -> Path:
    return Path(os.environ.get("MLSPACES_ASSETS_DIR", PROJECT_ROOT / "data" / "molmospaces" / "assets")).resolve()


def _bench_dir(task: Task) -> Path:
    return _assets_dir() / "benchmarks" / task.bench_subpath


def _read_horizon_sec(bench: Path) -> float | None:
    path = bench / "benchmark.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    episode = data[0] if isinstance(data, list) else data
    return episode.get("task_horizon_sec") or episode.get("task", {}).get("task_horizon_sec")


def _has_filament() -> bool:
    try:
        from molmo_spaces.env.env import HAS_FILAMENT

        return bool(HAS_FILAMENT)
    except Exception:
        return False


def _disable_video() -> None:
    import molmo_spaces.data_generation.pipeline as pipeline

    original = pipeline.save_trajectories

    def no_video(*args, **kwargs):
        kwargs["save_mp4s"] = False
        return original(*args, **kwargs)

    pipeline.save_trajectories = no_video


def run_task(task: Task, args: argparse.Namespace) -> dict:
    from molmo_spaces.evaluation.eval_main import run_evaluation
    from scripts.benchmarks.eval_to_csv import eval_to_csv
    from experiments.molmospaces.fastwam_policy import FastWAMMolmoSpacesEvalConfig

    bench = args.benchmark_dir or _bench_dir(task)
    csv_path = args.output_root / "csv" / f"{task.key}_{args.policy_name}.csv"
    run_dir = args.output_root / "runs" / task.key
    if csv_path.exists() and not args.force:
        return {"task": task.name, "status": "SKIP", "csv": str(csv_path)}
    if not (bench / "benchmark.json").exists():
        return {"task": task.name, "status": "MISSING", "path": str(bench)}
    if task.filament and not _has_filament():
        return {"task": task.name, "status": "NO_FILAMENT"}
    horizon_sec = _read_horizon_sec(bench)
    if horizon_sec is None:
        return {"task": task.name, "status": "NO_HORIZON", "path": str(bench)}

    os.environ["FASTWAM_MOLMOSPACES_EXTERIOR_CAMERA"] = task.exterior_camera
    os.environ["FASTWAM_MOLMOSPACES_WRIST_CAMERA"] = task.wrist_camera
    t0 = time.time()
    results = run_evaluation(
        eval_config_cls=FastWAMMolmoSpacesEvalConfig,
        benchmark_dir=bench,
        checkpoint_path=args.checkpoint_path,
        task_horizon_sec=float(horizon_sec),
        num_workers=args.num_workers,
        use_wandb=False,
        max_episodes=args.max_episodes,
        camera_names_override=[task.exterior_camera, task.wrist_camera],
        output_dir=run_dir,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    eval_to_csv(str(results.output_dir), args.policy_name, success_condition="both", output_csv=str(csv_path), dt=0.066)
    return {
        "task": task.name,
        "status": "OK",
        "episodes": results.total_count,
        "successes": results.success_count,
        "success_rate_pct": round(100.0 * results.success_rate, 2),
        "wall_sec": round(time.time() - t0, 1),
        "csv": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--set", dest="task_set", choices=list(SETS), default="molmobot")
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "evaluate_results" / "molmospaces" / "fastwam")
    parser.add_argument("--policy-name", default="fastwam")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--keep-video", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ["FASTWAM_MOLMOSPACES_DATASET_STATS_PATH"] = args.dataset_stats_path
    os.environ["FASTWAM_MOLMOSPACES_ACTION_HORIZON"] = str(args.action_horizon)
    os.environ["FASTWAM_MOLMOSPACES_REPLAN_STEPS"] = str(args.replan_steps)
    os.environ["FASTWAM_MOLMOSPACES_NUM_INFERENCE_STEPS"] = str(args.num_inference_steps)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_CACHE_DIR", str(PROJECT_ROOT / "data" / "molmospaces" / "cache"))
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(PROJECT_ROOT / "data" / "molmospaces" / "assets"))
    if not args.keep_video:
        _disable_video()

    tasks = [TASKS_BY_KEY[key] for key in args.tasks] if args.tasks else [TASKS_BY_KEY[key] for key in SETS[args.task_set]]
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [run_task(task, args) for task in tasks]
    summary = args.output_root / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        cols = ["task", "status", "episodes", "successes", "success_rate_pct", "wall_sec", "csv", "path"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in cols})
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
