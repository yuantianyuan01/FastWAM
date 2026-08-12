"""
Evaluation entrypoint for learned policies on JSON-based benchmarks.

This module evaluates policies on JSON benchmark files where each episode is fully
self-contained. Unlike the pickle-based frozen config approach, JSON benchmarks are
human-readable, version-independent, and support mixed task types.

Key differences from run_benchmark_with_learned_policy.py:
- Uses JsonEvalRunner instead of PatchyRunner
- No patch_config needed - JSON episode specs are authoritative
- Timing parameters (policy_dt_ms, ctrl_dt_ms, sim_dt_ms) come from the eval config,
  NOT from individual episodes. This allows the same benchmark to be run at different
  control rates.
- Supports mixed task types in the same benchmark

Programmatic usage (from external repo):
    from molmo_spaces.evaluation import run_evaluation

    results = run_evaluation(
        eval_config_cls=MyEvalConfig,
        benchmark_dir="/path/to/benchmark",
        checkpoint_path="/path/to/checkpoint",
    )
    print(f"Success rate: {results.success_count}/{results.total_count}")


Environment setup (MacOS):
    export PYTHONPATH="${PYTHONPATH}:."
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.robot_configs import ActionNoiseConfig
from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.evaluation.benchmark_schema import (
    EpisodeSpec,
    load_all_episodes,
)
from molmo_spaces.evaluation.json_eval_runner import JsonEvalRunner
from molmo_spaces.evaluation.robot_eval_overrides import OverrideFn
from molmo_spaces.molmo_spaces_constants import DATA_TYPE_TO_SOURCE_TO_VERSION
from molmo_spaces.utils.eval_utils import (
    EpisodeResult,
    collect_episode_results,
    compose_episode_videos,
    log_eval_results_to_wandb,
)

# TODO: Temporary workaround for data version pinning.
# For now we make the strong assumption that benchmarks are created with these exact data
# versions, and they should only be evaluated with these same versions. In the future, we need
# to save the data versions during datagen so they can be preserved and re-read with benchmarks.
# This assertion ensures we don't accidentally evaluate with mismatched asset versions.
_EXPECTED_DATA_VERSIONS = {
    "robots": {
        "rby1": "20251224",
        "rby1m": "20251224",
        "franka_droid": "20260127",
        "floating_rum": "20251110",
        "g1": "20260802",
    },
    "scenes": {
        "ithor": ["20251217", "20251217_with_occupancy"],
        "refs": "20250923",
        "procthor-10k-train": ["20251122", "20251122_with_occupancy"],
        "procthor-10k-val": [
            "20251121",
            "20251217",
            "20251217_with_occupancy",
        ],  # "20251121" is for nav_to_obj bench
        "procthor-10k-test": ["20251121", "20251121_with_occupancy"],
        "holodeck-objaverse-train": ["20251217", "20251217_with_occupancy"],
        "holodeck-objaverse-val": ["20251217", "20251217_with_occupancy"],
        "procthor-objaverse-train": ["20251205", "20251205_with_occupancy"],
        "procthor-objaverse-val": ["20251205", "20251205_with_occupancy"],
    },
    "objects": {
        "thor": "20251117",
        "objaverse": ["20251016_from_20250610", "20260131"],
        "objathor_metadata": ["20251117", "20260129"],
    },
    "grasps": {
        "droid": "20251116",
        "droid_objaverse": "20251218",
    },
}


def _assert_data_versions_match():
    """Assert that current data versions match expected versions for evaluation."""

    for data_type, expected_sources in _EXPECTED_DATA_VERSIONS.items():
        actual_sources = DATA_TYPE_TO_SOURCE_TO_VERSION.get(data_type, {})
        for source, expected_version in expected_sources.items():
            actual_version = actual_sources.get(source)

            # Admit multiple data versions
            if isinstance(expected_version, list):
                if actual_version not in expected_version:
                    expected_version = expected_version[0]
                else:
                    expected_version = actual_version

            assert actual_version == expected_version, (
                f"Data version mismatch for {data_type}/{source}: "
                f"expected {expected_version}, got {actual_version}. "
                f"Benchmarks must be evaluated with the same data versions they were created with."
            )

            # Warn about newer objaverse version that may have different assets
            if data_type == "objects" and source == "objaverse" and actual_version == "20260131":
                warnings.warn(
                    "Using objaverse data version 20260131. This is a newer version that may "
                    "contain different assets than the original benchmark version (20251016_from_20250610).",
                    UserWarning,
                    stacklevel=2,
                )


_assert_data_versions_match()

if TYPE_CHECKING:
    from molmo_spaces.policy.base_policy import BasePolicy

log = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Results from running an evaluation on a benchmark.

    Attributes:
        success_count: Number of successful episodes
        total_count: Total number of episodes evaluated
        output_dir: Path where evaluation outputs were saved
        episode_results: Per-episode results with details
        exp_config: The experiment config used for evaluation
    """

    success_count: int
    total_count: int
    output_dir: Path
    episode_results: list[EpisodeResult] = field(default_factory=list)
    exp_config: MlSpacesExpConfig | None = None

    @property
    def success_rate(self) -> float:
        """Compute success rate as a fraction."""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for learned policies on JSON benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "exp_config_cls",
        type=str,
        help="Name of the eval config class. Can include module path with colon.",
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        required=True,
        help="Path to JSON benchmark directory containing benchmark.json or house_*/episode_*.json files.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a model checkpoint. Overrides the checkpoint in the policy_config.",
    )
    horizon_group = parser.add_mutually_exclusive_group()
    horizon_group.add_argument(
        "--task_horizon_steps",
        type=int,
        default=None,
        help="Override task horizon (max steps per episode). If None, uses value from episode specs. Cannot be used with --task_horizon_sec.",
    )
    horizon_group.add_argument(
        "--task_horizon_sec",
        type=float,
        default=None,
        help="Override task horizon (max seconds per episode). If None, uses value from episode specs. Cannot be used with --task_horizon_steps.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mlspaces-json-eval",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results. Defaults to eval_output/<config>/<timestamp>.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging.",
    )
    parser.add_argument(
        "--use-filament",
        action="store_true",
        help="Whether or not to use the filament renderer instead of the legacy opengl one (requires installing custom wheel for now)",
    )
    parser.add_argument(
        "--environment-light-intensity",
        type=float,
        default=None,
        help="The default value for the intensity of the default environmental light (when using filament)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit number of episodes to evaluate from benchmark. If None, evaluates all episodes, else, evaluates only the episodes for the houses used in the first `max_episodes`. Note that the final number of episodes can differ from `max_episodes` if more than one episode is sampled for any of the houses among the first `max_episodes` episodes.",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=None,
        help="Override policy_config.camera_names (e.g. --camera_names randomized_zed2_analogue_1 wrist_camera).",
    )

    # Eval camera randomization flags (shared across all JSON eval entry points)
    from molmo_spaces.utils.eval_camera_randomization_utils import add_eval_camera_args

    add_eval_camera_args(parser)

    parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="The index of the episode to evaluate. If None, evaluates all episodes.",
    )
    parser.add_argument(
        "--add_custom_object",
        action="store_true",
        help="Add a custom object to the episode.",
    )
    parser.add_argument(
        "--custom_object_path",
        type=str,
        default=None,
        help="The path to the custom object to add to the episode.",
    )
    parser.add_argument(
        "--custom_object_name",
        type=str,
        default=None,
        help="The natural language name for the custom object (e.g., 'lemon', 'cup'). "
        "If not provided, will attempt to extract from the object path but could be incorrect.",
    )
    return parser.parse_args()


def build_success_status_map(results: list[EpisodeResult]) -> dict[str, bool]:
    """Build a map of episode keys to success status for video naming.

    Args:
        results: List of episode results

    Returns:
        Dict mapping episode keys (e.g., "house_5/episode_00000000") to success status
    """
    return {f"{r.house_id}/episode_{r.episode_idx:08d}": r.success for r in results}


def determine_task_horizon(
    episodes: list[EpisodeSpec],
    task_horizon_override: int | None,
    policy_dt_ms: float | None = None,
) -> int:
    """Determine task horizon from command line override or benchmark.

    Priority:
    1. Explicit override (from CLI --task_horizon_steps or --task_horizon_sec)
    2. Benchmark's per-episode task_horizon_sec (converted to steps via policy_dt_ms)

    Fails loudly if the benchmark does not contain task_horizon_sec and no
    explicit override was provided.

    Args:
        episodes: List of episode specs from the benchmark
        task_horizon_override: Optional override from command line
        policy_dt_ms: Policy timestep in milliseconds, required when reading
            task_horizon_sec from the benchmark.

    Returns:
        Task horizon (in steps) to use for all episodes
    """
    if task_horizon_override is not None:
        log.info(f"Using explicit task_horizon override: {task_horizon_override} steps")
        return task_horizon_override

    # Read task_horizon_sec from the benchmark episodes
    horizon_sec_values = {ep.task.get("task_horizon_sec") for ep in episodes}

    if None in horizon_sec_values:
        missing_count = sum(1 for ep in episodes if ep.task.get("task_horizon_sec") is None)
        raise ValueError(
            f"No explicit task_horizon override provided and {missing_count}/{len(episodes)} "
            f"benchmark episodes are missing 'task_horizon_sec' in their task dict. "
            f"Either add task_horizon_sec to the benchmark JSON or pass "
            f"--task_horizon_steps / --task_horizon_sec explicitly."
        )

    if len(horizon_sec_values) > 1:
        raise ValueError(
            f"Benchmark has inconsistent task_horizon_sec values: {horizon_sec_values}. "
            f"All episodes must have the same task_horizon_sec, or pass an explicit override."
        )

    task_horizon_sec = horizon_sec_values.pop()

    if policy_dt_ms is None:
        raise ValueError(
            "policy_dt_ms is required to convert benchmark task_horizon_sec to steps. "
            "This is a bug in the caller."
        )

    task_horizon_steps = round(task_horizon_sec * 1000.0 / policy_dt_ms)

    log.info("=" * 70)
    log.info("TASK HORIZON RESOLVED FROM BENCHMARK")
    log.info(f"  task_horizon_sec (from benchmark JSON): {task_horizon_sec}")
    log.info(f"  policy_dt_ms (from eval config): {policy_dt_ms}")
    log.info(
        f"  task_horizon_steps = {task_horizon_sec} * 1000 / {policy_dt_ms} = {task_horizon_steps}"
    )
    log.info("=" * 70)

    return task_horizon_steps


@dataclass
class EvalRuntimeParams:
    """Runtime parameters for evaluation that are not part of the base config schema.

    These parameters are set during evaluation initialization and used by the
    evaluation runner to customize episode processing.
    """

    episode_idx: int | None = None
    max_episodes: int | None = None
    add_custom_object: bool = False
    custom_object_path: str | Path | None = None
    custom_object_name: str | None = None
    robot_override_fn: OverrideFn | None = None
    """
    Hook that mutates the experiment config with robot-specific overrides.
    """


def create_eval_config(
    eval_config_cls: type[MlSpacesExpConfig],
    benchmark_dir: Path,
    output_dir: Path,
    checkpoint_path: str | None,
    task_horizon: int,
    num_workers: int,
    camera_config_override: Any | None = None,
) -> MlSpacesExpConfig:
    """Create an MlSpacesExpConfig experiment config from a JSON benchmark for evaluation.

    The eval config class provides:
    - policy_config: Policy configuration (checkpoint, camera names, etc.)
    - robot_config: Robot configuration
    - Timing parameters: policy_dt_ms, ctrl_dt_ms, sim_dt_ms

    The benchmark provides:
    - Scene/task configuration (per-episode)

    Args:
        eval_config_cls: The eval config class to instantiate
        benchmark_dir: Path to JSON benchmark directory
        output_dir: Output directory for results
        checkpoint_path: Optional override for checkpoint path
        task_horizon: Task horizon (already resolved from defaults or override)
        num_workers: Number of worker processes

    Returns:
        Configured MlSpacesExpConfig
    """
    # Instantiate the eval config
    exp_config = eval_config_cls()

    # Override checkpoint if provided
    if checkpoint_path is not None:
        exp_config.policy_config.checkpoint_path = checkpoint_path

    # Set output directory
    exp_config.output_dir = output_dir

    # Set number of workers
    exp_config.num_workers = num_workers

    # Disable action noise for evaluation
    exp_config.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)

    # Disable profiling for cleaner output
    exp_config.datagen_profiler = False
    exp_config.profile = False

    # Don't filter - we want to save all trajectories for analysis
    exp_config.filter_for_successful_trajectories = False

    # Set eval mode seed
    exp_config.seed = 42

    # Set task_horizon (already determined from defaults or override)
    exp_config.task_horizon = task_horizon

    # Apply eval camera config override if provided
    if camera_config_override is not None:
        exp_config.camera_config = camera_config_override
    # Initialize eval_runtime_params with defaults so it always exists
    # This is now a proper field in MlSpacesExpConfig, so normal assignment works
    if exp_config.eval_runtime_params is None:
        exp_config.eval_runtime_params = EvalRuntimeParams()

    return exp_config


def run_evaluation(
    eval_config_cls: type[MlSpacesExpConfig] | str,
    benchmark_dir: Path,
    checkpoint_path: str | None = None,
    task_horizon_steps: int | None = None,
    task_horizon_sec: float | None = None,
    output_dir: str | Path | None = None,
    num_workers: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "mlspaces-online-eval",
    preloaded_policy: BasePolicy | None = None,
    max_episodes: int | None = None,
    camera_config_override: Any | None = None,
    camera_names_override: list[str] | None = None,
    environment_light_intensity: float | None = None,
    episode_idx: int | None = None,
    add_custom_object: bool = False,
    custom_object_path: str | Path | None = None,
    custom_object_name: str | None = None,
) -> EvaluationResults:
    """Run evaluation on a JSON benchmark programmatically.

    This is the primary entry point for running evaluations from external code.
    It can be imported and called directly without using command-line arguments.

    Args:
        eval_config_cls: Either an MlSpacesExpConfig subclass, or a string in the format
            "module.path:ClassName" (e.g., "myrepo.configs:MyEvalConfig").
        benchmark_dir: Path to JSON benchmark directory containing benchmark.json.
        checkpoint_path: Path to model checkpoint. Overrides the checkpoint in policy_config.
        task_horizon_steps: Max steps per episode. If None, uses default for the task class.
        task_horizon_sec: Max seconds per episode, used to calculate horizon in steps. Cannot be used with task_horizon_steps.
        output_dir: Output directory for results. Defaults to eval_output/<config>/<timestamp>.
        num_workers: Number of parallel worker processes.
        use_wandb: Whether to log results to Weights & Biases.
        wandb_project: W&B project name (only used if use_wandb=True).
        preloaded_policy: Optional pre-initialized policy instance. If provided, skips
            policy creation from config.
        max_episodes: Maximum number of episodes to evaluate from benchmark. If None, evaluates all episodes.
        camera_config_override: Optional camera system config (of type FrankaEvalCameraSystem) to
            replace the default camera_config on the experiment config. Other types are ignored.
        camera_names_override: Optional list of camera names to override
            policy_config.camera_names (e.g. ["randomized_zed2_analogue_1", "wrist_camera"]).
        episode_idx: Index of a specific episode to evaluate. If None, evaluates all episodes.
        add_custom_object: Whether to replace the target object with a custom object.
        custom_object_path: Path to the custom object XML file. Required if add_custom_object is True.
        custom_object_name: Natural language name for the custom object (e.g., 'lemon', 'cup').
            If not provided, will attempt to extract from the object path.

    Returns:
        EvaluationResults containing success counts, output paths, and per-episode details.

    Raises:
        FileNotFoundError: If benchmark_dir doesn't exist.
        ValueError: If no episodes found in benchmark or config class not found.

    Example:
        from molmo_spaces.evaluation import run_evaluation
        from my_repo.configs import MyEvalConfig

        results = run_evaluation(
            eval_config_cls=MyEvalConfig,
            benchmark_dir="/path/to/benchmark",
            checkpoint_path="/path/to/checkpoint.pt",
            task_horizon_steps=500,
        )
        print(f"Success rate: {results.success_rate:.1%}")
    """
    # Resolve config class if provided as string
    # Preserve the original string for config_name in case the registered name
    # differs from the class __name__ (e.g., a custom registry name)
    config_name_from_str: str | None = None
    if isinstance(eval_config_cls, str):
        config_name_from_str = eval_config_cls
        if ":" in eval_config_cls:
            # Full module path provided - import and get class directly
            module_path, class_name = eval_config_cls.split(":")
            module = importlib.import_module(module_path)
            eval_config_cls = getattr(module, class_name)
        else:
            # Just a class name - look up in registry
            class_name = eval_config_cls
            eval_config_cls = get_config_class(class_name)

    # Validate benchmark directory
    benchmark_dir = benchmark_dir.resolve()
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    # Load benchmark episodes (for summary info and validation)
    episodes = load_all_episodes(benchmark_dir)

    # Validate episode index if specified
    if episode_idx is not None:
        if episode_idx < 0 or episode_idx >= len(episodes):
            raise ValueError(
                f"Episode index {episode_idx} is out of range. "
                f"Benchmark has {len(episodes)} episodes (indices 0-{len(episodes) - 1})"
            )
        log.info(f"Will evaluate single episode at index {episode_idx}")

    # Validate custom object path if requested
    if add_custom_object:
        if custom_object_path is None:
            raise ValueError(
                "--custom_object_path must be provided when --add_custom_object is set"
            )
        custom_object_path = Path(custom_object_path)
        if not custom_object_path.exists():
            raise FileNotFoundError(f"Custom object path does not exist: {custom_object_path}")
        log.info(f"Will replace target objects with custom object: {custom_object_path}")
        if custom_object_name is None:
            custom_object_name = custom_object_path.stem
            log.warning(f"No custom object name provided, using path stem: {custom_object_name}")
        else:
            log.info(f"Using provided custom object name: {custom_object_name}")

    if max_episodes is not None and len(episodes) > max_episodes:
        log.info(f"Evaluating the first {max_episodes} episodes of {len(episodes)} total episodes")
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError(
            f"No episodes found in benchmark at {benchmark_dir}. "
            f"Expected benchmark.json file with list of episode specs."
        )

    total_episodes = len(episodes)
    num_houses = len(set(ep.house_index for ep in episodes))

    # Create timestamp and output directory
    # Use the original string if eval_config_cls was passed as a string, otherwise use __name__.
    # This handles cases where the registry name differs from the class name.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if config_name_from_str:
        # If "module:ClassName" format, use just the class name for the output dir
        config_name = (
            config_name_from_str.split(":")[-1]
            if ":" in config_name_from_str
            else config_name_from_str
        )
    else:
        config_name = eval_config_cls.__name__

    if output_dir is not None:
        resolved_output_dir = Path(output_dir) / config_name / timestamp
    else:
        resolved_output_dir = Path("eval_output") / config_name / timestamp
    os.makedirs(resolved_output_dir, exist_ok=True)

    # Determine task horizon
    assert not (task_horizon_steps is not None and task_horizon_sec is not None), (
        "Cannot use both task_horizon_steps and task_horizon_sec"
    )
    task_horizon: int | None = None
    if task_horizon_steps is not None:
        task_horizon = task_horizon_steps
    elif task_horizon_sec is not None:
        policy_dt_ms = eval_config_cls.model_fields["policy_dt_ms"].get_default()
        assert isinstance(policy_dt_ms, float | int), (
            f"policy_dt_ms must be a float or int, got {type(policy_dt_ms)}"
        )
        task_horizon = round(task_horizon_sec * 1000.0 / policy_dt_ms)
    config_policy_dt_ms = eval_config_cls.model_fields["policy_dt_ms"].get_default()
    resolved_task_horizon = determine_task_horizon(
        episodes, task_horizon, policy_dt_ms=config_policy_dt_ms
    )

    # Create experiment config
    exp_config = create_eval_config(
        eval_config_cls=eval_config_cls,
        benchmark_dir=benchmark_dir,
        output_dir=resolved_output_dir,
        checkpoint_path=checkpoint_path,
        task_horizon=resolved_task_horizon,
        num_workers=num_workers,
        camera_config_override=camera_config_override,
    )

    # Custom filament settings to overwrite by the user
    exp_config.environment_light_intensity = (
        environment_light_intensity or exp_config.environment_light_intensity
    )

    # Override policy camera names if requested
    if camera_names_override is not None:
        log.info(f"Overriding policy_config.camera_names: {camera_names_override}")
        exp_config.policy_config.camera_names = camera_names_override

    # Patch config with evaluation-specific runtime parameters
    exp_config = JsonEvalRunner.patch_config(
        exp_config=exp_config,
        episode_idx=episode_idx,
        max_episodes=max_episodes,
        add_custom_object=add_custom_object,
        custom_object_path=custom_object_path,
        custom_object_name=custom_object_name,
    )
    JsonEvalRunner.adjust_robot(exp_config)

    # Resolve checkpoint path for logging
    resolved_checkpoint = checkpoint_path or getattr(
        exp_config.policy_config, "checkpoint_path", None
    )

    # Initialize wandb if requested
    if use_wandb:
        import wandb

        if resolved_checkpoint:
            path_parts = Path(resolved_checkpoint).parts
            ckpt_name_parts = [p for p in path_parts[-2:] if p and p != "/"]
            ckpt_name = "_".join(ckpt_name_parts)
        else:
            ckpt_name = "no_ckpt"

        wandb_run_name = f"{ckpt_name}_{timestamp}"
        wandb.init(project=wandb_project, name=wandb_run_name)
        wandb.config.update(
            {
                "checkpoint_path": resolved_checkpoint,
                "benchmark_dir": str(benchmark_dir),
                "task_horizon_steps": exp_config.task_horizon,
                "task_horizon_sec": exp_config.task_horizon / exp_config.fps,
                "exp_config_cls": config_name,
                "num_episodes": total_episodes,
                "num_houses": num_houses,
            }
        )

    # # Run evaluation
    # runner = JsonEvalRunner(exp_config, benchmark_dir)
    # success_count, total_count = runner.run(preloaded_policy=policy)

    # Run evaluation
    # Only pass preloaded policy for single-worker mode. With multiple workers,
    # each worker must create its own connection (WebSocket/msgpack can't be pickled).
    runner = JsonEvalRunner(exp_config, benchmark_dir)
    success_count, total_count = runner.run(preloaded_policy=preloaded_policy)

    # Collect per-episode results
    episode_results = collect_episode_results(resolved_output_dir)

    # Log to wandb if enabled
    if use_wandb:
        import wandb

        camera_names = getattr(exp_config.policy_config, "camera_names", [])
        if camera_names:
            success_status = build_success_status_map(episode_results)
            composed_videos = compose_episode_videos(
                eval_dir=resolved_output_dir,
                camera_names=camera_names,
                success_status=success_status,
            )
        else:
            composed_videos = {}

        log_eval_results_to_wandb(
            results=episode_results,
            composed_videos=composed_videos,
        )
        wandb.finish()

    return EvaluationResults(
        success_count=success_count,
        total_count=total_count,
        output_dir=resolved_output_dir,
        episode_results=episode_results,
        exp_config=exp_config,
    )


def main() -> None:
    """Command-line entry point for evaluation."""
    args = get_args()

    # Build eval camera config from CLI flags (None if --use_eval_cameras not passed)
    from molmo_spaces.utils.eval_camera_randomization_utils import (
        build_eval_camera_config_from_args,
    )

    eval_camera_config = build_eval_camera_config_from_args(args)

    # Load benchmark to log summary info
    benchmark_dir = Path(args.benchmark_dir).resolve()
    episodes = load_all_episodes(benchmark_dir)
    if episodes:
        log.info(f"Loaded benchmark: {benchmark_dir}")
        log.info(f"  Houses: {len(set(ep.house_index for ep in episodes))}")
        log.info(f"  Total episodes: {len(episodes)}")
        log.info(f"  First episode task_cls: {episodes[0].get_task_cls()}")

    # Run evaluation using the programmatic API
    results = run_evaluation(
        eval_config_cls=args.exp_config_cls,
        benchmark_dir=benchmark_dir,
        checkpoint_path=args.checkpoint_path,
        task_horizon_steps=args.task_horizon_steps,
        task_horizon_sec=args.task_horizon_sec,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        max_episodes=args.max_episodes,
        environment_light_intensity=args.environment_light_intensity,
        camera_config_override=eval_camera_config,
        camera_names_override=args.camera_names,
        episode_idx=args.idx,
        add_custom_object=args.add_custom_object,
        custom_object_path=args.custom_object_path,
        custom_object_name=args.custom_object_name,
    )

    log.info(f"Evaluation complete: {results.success_count}/{results.total_count} successful")
    log.info(f"Success rate: {results.success_rate:.1%}")
    log.info(f"Output directory: {results.output_dir}")


if __name__ == "__main__":
    main()
