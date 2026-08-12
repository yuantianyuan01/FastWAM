"""
JSON-based benchmark evaluation runner.

This runner loads episode specifications from JSON benchmark files and runs
policy evaluations against them. Unlike the pickle-based frozen config approach,
JSON specs are fully self-contained and human-readable.

Key design principles:
- Each episode is fully self-contained in JSON (no external config dependencies)
- Timing parameters (policy_dt_ms, ctrl_dt_ms, sim_dt_ms) come from the eval config,
  NOT from individual episodes. This allows the same benchmark to be run at different
  control rates without modifying the benchmark files.
- Task type can vary per episode (mixed task types in same benchmark)
- No patch_config needed - JSON is authoritative

Usage:
    from molmo_spaces.evaluation import JsonEvalRunner, load_benchmark

    # Load benchmark and create config
    metadata, episodes_by_house = load_benchmark(benchmark_dir)
    runner = JsonEvalRunner(exp_config, benchmark_dir)
    success_count, total_count = runner.run(preloaded_policy=policy)
"""

import logging
from collections import defaultdict
from pathlib import Path

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec, load_all_episodes
from molmo_spaces.evaluation.robot_eval_overrides import get_robot_override
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.profiler_utils import DatagenProfiler

log = logging.getLogger(__name__)


class JsonEvalRunner(ParallelRolloutRunner):
    """
    Evaluation runner for JSON-based benchmarks.

    This runner differs from the standard ParallelRolloutRunner in several ways:
    1. Episodes are loaded from JSON files, not from H5 frozen configs
    2. Each episode is fully self-contained (timing, cameras, task config)
    3. Task samplers are created per-episode to support mixed task types
    4. Uses patch_config to add evaluation-specific runtime parameters

    The runner inherits process_single_house from ParallelRolloutRunner and
    customizes behavior by overriding hook methods.
    """

    @staticmethod
    def patch_config(
        exp_config: MlSpacesExpConfig,
        episode_idx: int | None = None,
        max_episodes: int | None = None,
        add_custom_object: bool = False,
        custom_object_path: str | Path | None = None,
        custom_object_name: str | None = None,
    ) -> MlSpacesExpConfig:
        """Patch evaluation config with runtime evaluation-specific parameters.

        This method modifies the config object to store evaluation-specific runtime
        parameters that are not part of the base config schema. These parameters are
        used by the evaluation runner to customize episode processing.

        Args:
            exp_config: The experiment config to patch
            episode_idx: Optional index of a specific episode to evaluate. If provided,
                only that episode will be evaluated and the process will stop after it.
            max_episodes: Optional maximum number of episodes to evaluate. If provided,
                only the episodes for the houses used in the first N episodes will be
                evaluated. Note that the final number of episodes can differ from N
                if more than one episode is sampled for any of the houses among the
                first N episodes.
            add_custom_object: Whether to replace the target object with a custom object.
            custom_object_path: Path to the custom object XML file. Required if
                add_custom_object is True.
            custom_object_name: Natural language name for the custom object (e.g., 'lemon', 'cup').

        Returns:
            The patched config (same object, modified in place)
        """
        # Import here to avoid circular dependency
        from molmo_spaces.evaluation.eval_main import EvalRuntimeParams

        # eval_runtime_params is now a proper field in MlSpacesExpConfig, so normal assignment works
        exp_config.eval_runtime_params = EvalRuntimeParams(
            episode_idx=episode_idx,
            max_episodes=max_episodes,
            add_custom_object=add_custom_object,
            custom_object_path=custom_object_path,
            custom_object_name=custom_object_name,
        )

        return exp_config

    @staticmethod
    def adjust_robot(exp_config: MlSpacesExpConfig) -> None:
        """Apply robot-specific evaluation overrides if configured."""
        robot_override = get_robot_override(exp_config.robot_config)
        if exp_config.eval_runtime_params is None:
            from molmo_spaces.evaluation.eval_main import EvalRuntimeParams

            exp_config.eval_runtime_params = EvalRuntimeParams()
        exp_config.eval_runtime_params.robot_override_fn = robot_override

    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
        benchmark_dir: Path,
    ) -> None:
        """
        Initialize the JSON eval runner.

        The benchmark is authoritative - all episode data comes from the JSON files.
        No fallbacks or defaults; missing data is an error.

        Args:
            exp_config: Base experiment config (provides robot_config, policy_config)
            benchmark_dir: Path to benchmark directory containing benchmark.json
        """
        self.benchmark_dir = benchmark_dir.resolve()

        all_episodes = load_all_episodes(self.benchmark_dir)
        if not all_episodes:
            raise ValueError(
                f"No episodes found in benchmark at {self.benchmark_dir}. "
                f"Expected benchmark.json file with list of episode specs."
            )

        eval_params = exp_config.eval_runtime_params
        if eval_params.max_episodes is not None and len(all_episodes) > eval_params.max_episodes:
            log.info(
                f"Limiting to first {eval_params.max_episodes} of {len(all_episodes)} episodes"
            )
            all_episodes = all_episodes[: eval_params.max_episodes]

        self._episodes_by_house: dict[int, list[EpisodeSpec]] = defaultdict(list)
        for ep in all_episodes:
            self._episodes_by_house[ep.house_index].append(ep)
        self._episodes_by_house = dict(self._episodes_by_house)

        # If episode_idx is specified, only process the house containing that episode
        episode_idx = eval_params.episode_idx
        if episode_idx is not None:
            if episode_idx < 0 or episode_idx >= len(all_episodes):
                raise ValueError(
                    f"Episode index {episode_idx} is out of range. "
                    f"Benchmark has {len(all_episodes)} episodes (indices 0-{len(all_episodes) - 1})"
                )
            target_episode = all_episodes[episode_idx]
            # Only process the house containing the target episode
            exp_config.task_sampler_config.house_inds = [target_episode.house_index]
            exp_config.task_sampler_config.samples_per_house = 1
        else:
            exp_config.task_sampler_config.house_inds = sorted(self._episodes_by_house.keys())
            max_episodes = max(len(eps) for eps in self._episodes_by_house.values())
            exp_config.task_sampler_config.samples_per_house = max_episodes
        exp_config.benchmark_path = self.benchmark_dir

        super().__init__(exp_config)

        total_episodes = sum(len(eps) for eps in self._episodes_by_house.values())
        log.info(
            f"JsonEvalRunner initialized: {len(self._episodes_by_house)} houses, "
            f"{total_episodes} episodes from {self.benchmark_dir}"
        )

    def get_episodes_for_house(self, house_id: int) -> list[EpisodeSpec]:
        """Get all episode specs for a given house."""
        if house_id not in self._episodes_by_house:
            raise KeyError(
                f"House {house_id} not found in benchmark. "
                f"Available houses: {sorted(self._episodes_by_house.keys())}"
            )
        return self._episodes_by_house[house_id]

    # =========================================================================
    # Hook Overrides - Customize episode processing for JSON benchmarks
    # =========================================================================

    @staticmethod
    def load_episodes_for_house(
        exp_config: MlSpacesExpConfig,
        house_id: int,
        batch_suffix: str,
        worker_task_sampler,
        worker_logger,
    ) -> tuple[list[EpisodeSpec], None]:
        """Load episode specifications from JSON benchmark."""
        benchmark_path = exp_config.benchmark_path
        all_episodes = load_all_episodes(benchmark_path)

        if not all_episodes:
            worker_logger.error(
                f"No episodes found in benchmark at {benchmark_path}. Expected benchmark.json file."
            )
            return [], None

        eval_params = exp_config.eval_runtime_params

        # Truncate to max_episodes before any filtering
        if eval_params.max_episodes is not None and len(all_episodes) > eval_params.max_episodes:
            all_episodes = all_episodes[: eval_params.max_episodes]

        # Filter by episode index if specified
        episode_idx = eval_params.episode_idx
        if episode_idx is not None:
            if episode_idx < 0 or episode_idx >= len(all_episodes):
                worker_logger.error(
                    f"Episode index {episode_idx} is out of range. "
                    f"Benchmark has {len(all_episodes)} episodes (indices 0-{len(all_episodes) - 1})"
                )
                return [], None
            # Filter to only the specified episode, but still need to check house_id
            target_episode = all_episodes[episode_idx]
            if target_episode.house_index != house_id:
                # This house doesn't contain the target episode, return empty list
                return [], None
            all_episodes = [target_episode]

        house_episodes = [ep for ep in all_episodes if ep.house_index == house_id]

        if not house_episodes:
            available_houses = sorted(set(ep.house_index for ep in all_episodes))
            worker_logger.error(
                f"House {house_id} not found in benchmark. Available houses: {available_houses}"
            )
            return [], None

        # Apply custom object replacement if requested
        add_custom_object = eval_params.add_custom_object
        custom_object_path = eval_params.custom_object_path
        custom_object_name = eval_params.custom_object_name
        if add_custom_object and custom_object_path is not None:
            from pathlib import Path

            from molmo_spaces.evaluation.benchmark_schema import replace_target_object_with_custom

            custom_object_path = Path(custom_object_path)
            worker_logger.info(f"Replacing target objects with custom object: {custom_object_path}")
            if custom_object_name:
                worker_logger.info(f"Using custom object name: '{custom_object_name}'")
            house_episodes = [
                replace_target_object_with_custom(ep, custom_object_path, custom_object_name)
                for ep in house_episodes
            ]

        worker_logger.info(
            f"Loaded {len(house_episodes)} episodes for house {house_id} from {benchmark_path}"
        )
        return house_episodes, None

    @staticmethod
    def get_max_episode_attempts(
        episode_specs: list[EpisodeSpec],
        samples_per_house: int,
        exp_config: MlSpacesExpConfig,
    ) -> int:
        """Process all episodes in the benchmark - no retry multiplier."""
        return len(episode_specs)

    @staticmethod
    def should_stop_early(
        num_collected: int, samples_per_house: int, exp_config: MlSpacesExpConfig | None = None
    ) -> bool:
        """Stop early if evaluating a single episode (--idx provided) and it's been collected."""
        if exp_config is not None:
            eval_params = exp_config.eval_runtime_params
            if eval_params.episode_idx is not None:
                # Stop after collecting the single requested episode
                return num_collected >= 1
        return False

    @staticmethod
    def get_episode_spec_at_index(episode_specs: list[EpisodeSpec], idx: int) -> EpisodeSpec:
        """Get episode specification at given index."""
        return episode_specs[idx]

    @staticmethod
    def prepare_episode_config(
        exp_config: MlSpacesExpConfig,
        episode_spec: EpisodeSpec,
        episode_idx: int,
    ) -> MlSpacesExpConfig:
        """Prepare episode-specific config from JSON spec.

        Note: task_horizon is NOT read from episode_spec. It's an evaluation
        parameter that comes from exp_config (set via command line or defaults).
        """
        episode_config = exp_config.model_copy(deep=True)
        episode_config.scene_dataset = episode_spec.scene_dataset
        episode_config.data_split = episode_spec.data_split
        # task_horizon comes from exp_config, not episode_spec
        return episode_config

    @staticmethod
    def get_episode_task_sampler(
        exp_config: MlSpacesExpConfig,
        episode_spec: EpisodeSpec,
        shared_task_sampler,
        datagen_profiler: DatagenProfiler | None,
    ) -> JsonEvalTaskSampler:
        """Create per-episode JsonEvalTaskSampler."""
        sampler = JsonEvalTaskSampler(exp_config, episode_spec)
        if datagen_profiler is not None:
            sampler.set_datagen_profiler(datagen_profiler)
        return sampler

    @staticmethod
    def sample_task_from_spec(
        task_sampler: JsonEvalTaskSampler,
        house_id: int,
        episode_spec: EpisodeSpec,
        episode_idx: int,
    ) -> BaseMujocoTask | None:
        """Sample task - episode spec is already in the JsonEvalTaskSampler."""
        return task_sampler.sample_task(house_index=house_id)

    @staticmethod
    def get_episode_seed(
        episode_idx: int,
        episode_spec: EpisodeSpec,
        task_sampler: JsonEvalTaskSampler,
    ) -> int:
        """Get seed from episode spec, falling back to index."""
        return episode_spec.seed if episode_spec.seed is not None else episode_idx

    @staticmethod
    def should_close_episode_task_sampler() -> bool:
        """Close task sampler after each episode - we create per-episode."""
        return True
