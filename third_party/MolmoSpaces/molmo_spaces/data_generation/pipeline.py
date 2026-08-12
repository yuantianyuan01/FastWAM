import gc
import logging
import multiprocessing as mp
import os
import pprint
import random
import signal
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mujoco

# import mujoco.viewer
import psutil
import torch

import wandb
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.policy.base_policy import BasePolicy
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
from molmo_spaces.utils.mp_logging import (
    get_logger,
    get_worker_logger,
    init_logging,
    worker_stdout_context,
)
from molmo_spaces.utils.profiler_utils import DatagenProfiler, Profiler
from molmo_spaces.utils.save_utils import prepare_episode_for_saving, save_trajectories

# Set multiprocessing context based on CUDA availability
# forkserver is safer for CUDA, spawn for fallback
mp_context = mp.get_context("forkserver") if torch.cuda.is_available() else mp.get_context("spawn")

# Silence verbose third-party libraries
logging.getLogger("curobo").setLevel(logging.WARNING)
logging.getLogger("trimesh").setLevel(logging.WARNING)


def get_process_memory():
    """Get current memory usage of the process in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def log_memory_usage(logger, prefix="") -> None:
    """Log current memory usage"""
    mem_usage = get_process_memory()
    logger.info(f"{prefix}Memory usage: {mem_usage:.2f} MB")


def get_detailed_memory_info():
    """Get detailed memory usage information"""
    process = psutil.Process()
    mem = process.memory_info()
    return {
        "rss": mem.rss / 1024 / 1024,  # RSS in MB
        "vms": mem.vms / 1024 / 1024,  # Virtual Memory in MB
        "percent": process.memory_percent(),
    }


# =============================================================================
# Shared helper functions for house processing
# These are used by both ParallelRolloutRunner and JsonEvalRunner to reduce
# code duplication in process_single_house implementations.
# =============================================================================


def setup_house_dirs(
    exp_config: "MlSpacesExpConfig",
    house_id: int,
    batch_num: int | None = None,
    total_batches: int | None = None,
) -> tuple[Path, Path, str, bool]:
    """
    Setup output directories and check for existing output.

    Args:
        exp_config: Experiment configuration
        house_id: House index
        batch_num: Batch number (1-indexed)
        total_batches: Total number of batches

    Returns:
        tuple: (house_output_dir, house_debug_dir, batch_suffix, should_skip)
    """
    house_output_dir = exp_config.output_dir / f"house_{house_id}"
    debug_base_dir = exp_config.output_dir.parent / "debug" / exp_config.output_dir.name
    house_debug_dir = debug_base_dir / f"house_{house_id}"

    if batch_num is None or total_batches is None:
        batch_num = 1
        total_batches = 1
    batch_suffix = f"_batch_{batch_num}_of_{total_batches}"

    batch_file = house_output_dir / f"trajectories{batch_suffix}.h5"
    should_skip = batch_file.exists()

    return house_output_dir, house_debug_dir, batch_suffix, should_skip


def setup_policy(
    exp_config: "MlSpacesExpConfig",
    task: "BaseMujocoTask",
    preloaded_policy: "BasePolicy | None",
    datagen_profiler: "DatagenProfiler | None",
) -> "BasePolicy":
    """
    Create or return policy for episode.

    Args:
        exp_config: Experiment configuration
        task: The task instance
        preloaded_policy: Pre-loaded policy or None
        datagen_profiler: Profiler for timing

    Returns:
        Policy instance
    """
    if datagen_profiler is not None:
        datagen_profiler.start("policy_setup")

    if preloaded_policy is not None:
        policy = preloaded_policy
    else:
        policy = exp_config.policy_config.policy_factory(exp_config, task)

    task.register_policy(policy)

    if datagen_profiler is not None:
        datagen_profiler.end("policy_setup")

    return policy


def setup_viewer(
    exp_config: "MlSpacesExpConfig",
    task: "BaseMujocoTask",
    policy: "BasePolicy",
    current_viewer,
):
    """
    Setup passive viewer if configured.

    Args:
        exp_config: Experiment configuration
        task: The task instance
        policy: The policy instance
        current_viewer: Existing viewer or None

    Returns:
        Viewer instance or None
    """
    viewer = current_viewer
    if exp_config.use_passive_viewer:
        if viewer is not None:
            viewer.close()
            viewer = None
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(
            task.env.mj_datas[task.env.current_batch_index].model,
            task.env.mj_datas[task.env.current_batch_index],
            key_callback=getattr(policy, "get_key_callback", lambda: None)(),
        )
        if exp_config.viewer_cam_dict is not None and "camera" in exp_config.viewer_cam_dict:
            viewer.cam.fixedcamid = (
                task.env.mj_datas[0].camera(exp_config.viewer_cam_dict["camera"]).id
            )
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.opt.sitegroup[0] = False
    task.viewer = viewer
    return viewer


def save_house_trajectories(
    worker_logger,
    house_raw_histories: list,
    house_output_dir: Path,
    exp_config: "MlSpacesExpConfig",
    batch_suffix: str,
    datagen_profiler: "DatagenProfiler | None" = None,
    batch_num: int | None = None,
    total_batches: int | None = None,
) -> None:
    """
    Batch and save trajectory data for a house.

    Args:
        worker_logger: Logger instance
        house_raw_histories: List of episode info dicts with 'history' and 'sensor_suite'
        house_output_dir: Output directory path
        exp_config: Experiment configuration
        batch_suffix: Suffix for batch file naming
        datagen_profiler: Profiler for timing
        batch_num: Batch number for logging
        total_batches: Total batches for logging
    """
    if not house_raw_histories:
        worker_logger.warning(f"No trajectory data to save for {house_output_dir.name}")
        return

    batch_info = f" batch {batch_num}/{total_batches}" if batch_num is not None else ""
    worker_logger.info(
        f"Batching and saving trajectory data for {house_output_dir.name}{batch_info}: "
        f"{len(house_raw_histories)} episodes"
    )

    os.makedirs(house_output_dir, exist_ok=True)

    try:
        t_start = time.perf_counter()
        if datagen_profiler is not None:
            datagen_profiler.start("save_batch_prep")

        house_trajectory_data = []
        for idx, episode_info in enumerate(house_raw_histories):
            prepared_episode = prepare_episode_for_saving(
                episode_info["history"],
                episode_info["sensor_suite"],
                fps=exp_config.fps,
                save_dir=house_output_dir,
                episode_idx=idx,
                save_file_suffix=batch_suffix,
            )
            if prepared_episode is not None:
                house_trajectory_data.append(prepared_episode)
            del episode_info["history"]

        t_batch = time.perf_counter() - t_start
        if datagen_profiler is not None:
            datagen_profiler.end("save_batch_prep")
        worker_logger.debug(f"Batched {len(house_trajectory_data)} episodes in {t_batch:.2f}s")

        t_save_start = time.perf_counter()
        if datagen_profiler is not None:
            datagen_profiler.start("save_trajectories")
        save_trajectories(
            house_trajectory_data,
            save_dir=house_output_dir,
            fps=exp_config.fps,
            save_file_suffix=batch_suffix,
            save_mp4s=True,
            logger=worker_logger,
        )
        if datagen_profiler is not None:
            datagen_profiler.end("save_trajectories")
        t_save = time.perf_counter() - t_save_start

        total_time = time.perf_counter() - t_start
        worker_logger.info(
            f"Successfully saved trajectory data for {house_output_dir.name} in {total_time:.2f}s "
            f"(batch: {t_batch:.2f}s, save: {t_save:.2f}s)"
        )

        del house_trajectory_data
        gc.collect()

    except Exception as e:
        worker_logger.error(f"Failed to save trajectory data for {house_output_dir.name}: {e}")
        traceback.print_exc()


def cleanup_episode_resources(
    task,
    policy,
    task_sampler,
    preloaded_policy: "BasePolicy | None",
    close_task_sampler: bool = False,
) -> None:
    """
    Cleanup resources after an episode.

    Args:
        task: Task instance to cleanup
        policy: Policy instance to cleanup
        task_sampler: Task sampler instance (optional cleanup)
        preloaded_policy: If not None, policy won't be deleted
        close_task_sampler: Whether to close the task sampler
    """
    if task is not None:
        if hasattr(task, "close"):
            task.close()

    if policy is not None and preloaded_policy is None:
        if hasattr(policy, "close"):
            policy.close()

    if close_task_sampler and task_sampler is not None:
        task_sampler.close()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_detailed_memory(logger, prefix="") -> None:
    """Log detailed memory usage information"""
    mem_info = get_detailed_memory_info()
    logger.info(f"{prefix}Memory Usage:")
    logger.info(f"  RSS: {mem_info['rss']:.1f} MB")
    logger.info(f"  Virtual: {mem_info['vms']:.1f} MB")
    logger.info(f"  Percent: {mem_info['percent']:.1f}%")

    # Log system memory
    system_mem = psutil.virtual_memory()
    logger.info("System Memory:")
    logger.info(f"  Total: {system_mem.total / 1024 / 1024:.1f} MB")
    logger.info(f"  Available: {system_mem.available / 1024 / 1024:.1f} MB")
    logger.info(f"  Used: {system_mem.used / 1024 / 1024:.1f} MB")
    logger.info(f"  Percent: {system_mem.percent:.1f}%")


@contextmanager
def cleanup_context():
    """Context manager to ensure proper cleanup of MuJoCo resources"""
    try:
        yield
    finally:
        # Force garbage collection multiple times (CPython sometimes needs this)
        gc.collect()
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def house_processing_worker(
    worker_id: int,
    exp_config: MlSpacesExpConfig,
    work_items: list[tuple[int, int, int, int]],
    shutdown_event,
    counter_lock,
    house_counter,
    success_count,
    total_count,
    completed_houses,
    skipped_houses,
    max_allowed_sequential_task_sampler_failures: int = 10,
    max_allowed_sequential_rollout_failures: int = 10,
    max_allowed_sequential_irrecoverable_failures: int = 5,
    preloaded_policy: BasePolicy | None = None,
    filter_for_successful_trajectories: bool = False,
    runner_class=None,
):
    """
    Standalone worker function that processes work items sequentially from a shared counter.

    Each work item is a (house_id, batch_samples, batch_num, total_batches) tuple.
    This function can be run in either a thread or a process. It continually fetches
    the next work item from a shared counter and processes it.

    Args:
        worker_id: Unique ID for this worker
        exp_config: Experiment configuration
        work_items: List of (house_id, batch_samples, batch_num, total_batches) tuples
        shutdown_event: Event to signal shutdown
        counter_lock: Lock for thread-safe counter access
        house_counter: Shared counter for next work item to process
        success_count: Shared counter for successful episodes
        total_count: Shared counter for total episodes
        completed_houses: Shared counter for completed work items
        skipped_houses: Shared counter for skipped work items
        max_allowed_sequential_task_sampler_failures: Max consecutive task sampling failures
        max_allowed_sequential_rollout_failures: Max consecutive rollout failures
        max_allowed_sequential_irrecoverable_failures: Max consecutive irrecoverable failures
        preloaded_policy: Optional pre-initialized policy instance
        filter_for_successful_trajectories: Whether to filter for successful trajectories only
        runner_class: Runner class with run_single_rollout and process_single_house static methods
    """
    # Create worker-specific logger
    worker_logger = get_worker_logger(worker_id)

    # Create per-worker profiler for timing analysis
    if hasattr(exp_config, "datagen_profiler") and exp_config.datagen_profiler:
        datagen_profiler = DatagenProfiler(logger=worker_logger, enabled=True)
    else:
        datagen_profiler = None

    # Track sequential irrecoverable failures at worker level
    num_sequential_irrecoverable_failures = 0

    # Normal datagen: create task sampler once for this worker (persists across all houses)
    # This allows the worker to track object diversity and other state across houses
    task_sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)
    # Set profiler on task sampler for sub-timing within sample_task
    task_sampler.set_datagen_profiler(datagen_profiler)

    # Use context manager for worker-specific stdout redirection
    with worker_stdout_context(worker_logger, worker_id):
        try:
            while True:
                # Check for shutdown signal
                if shutdown_event.is_set():
                    worker_logger.info(
                        f"Worker {worker_id} received shutdown signal, cleaning up..."
                    )
                    break

                # Get next work item to process atomically
                with counter_lock:
                    if house_counter.value >= len(work_items):
                        break  # No more work items to process
                    item_idx = house_counter.value
                    house_counter.value += 1
                current_house_id, batch_samples, batch_num, total_batches = work_items[item_idx]

                worker_logger.info(
                    f"Worker {worker_id} starting house {current_house_id} "
                    f"batch {batch_num}/{total_batches} ({batch_samples} episodes) "
                    f"(item {item_idx}/{len(work_items)})"
                )

                # Process this work item
                house_success_count, house_total_count, irrecoverable = (
                    runner_class.process_single_house(
                        worker_id,
                        worker_logger,
                        current_house_id,
                        exp_config,
                        batch_samples,
                        shutdown_event,
                        task_sampler,
                        preloaded_policy,
                        max_allowed_sequential_task_sampler_failures,
                        max_allowed_sequential_rollout_failures,
                        filter_for_successful_trajectories=filter_for_successful_trajectories,
                        runner_class=runner_class,
                        batch_num=batch_num,
                        total_batches=total_batches,
                        datagen_profiler=datagen_profiler,
                    )
                )

                # Update global counters
                with counter_lock:
                    success_count.value += house_success_count
                    total_count.value += house_total_count
                    if house_total_count > 0:
                        completed_houses.value += 1
                    else:
                        skipped_houses.value += 1

                # Track sequential irrecoverable failures
                if irrecoverable:
                    num_sequential_irrecoverable_failures += 1
                    if (
                        num_sequential_irrecoverable_failures
                        >= max_allowed_sequential_irrecoverable_failures
                    ):
                        worker_logger.error(
                            f"Worker {worker_id} encountered {num_sequential_irrecoverable_failures} "
                            "sequential irrecoverable failures. This suggests something is seriously wrong. Exiting worker."
                        )
                        break
                else:
                    # Reset counter on success
                    num_sequential_irrecoverable_failures = 0

            worker_logger.info(f"Worker {worker_id} completed processing assigned work items")
        finally:
            # Log final profiling summary for this worker
            if datagen_profiler is not None:
                datagen_profiler.log_worker_summary()

            # Clean up task sampler at end of worker lifecycle
            if task_sampler is not None:
                task_sampler.close()


class ParallelRolloutRunner:
    """
    Orchestrates parallel house processing for offline data generation using multiprocessing.

    This class is designed for data generation workloads where:
    - Each worker processes complete houses independently
    - Task sampling involves heavy Python operations (scene setup, randomization)
    - Workers can run in separate processes for true parallelism

    Note: This is would be an inefficient runner for RL training, and is only intended for
    data generation or online evaluation tasks. For RL with vectorized environments,
    consider implementing a separate RLRolloutRunner that works without the multiprocessing
    overhead, and move threading into env.py with a MJXMuJoCoEnv class or similar.

    Customization via Subclassing:
        You can customize rollout behavior by subclassing this runner and overriding the
        static methods `run_single_rollout` and/or `process_single_house`.

        Example:
            class CustomRolloutRunner(ParallelRolloutRunner):
                @staticmethod
                def run_single_rollout(episode_seed, task, policy, **kwargs):
                    # Add custom logging or behavior
                    print(f"Starting rollout with seed {episode_seed}")
                    return ParallelRolloutRunner.run_single_rollout(
                        episode_seed, task, policy, **kwargs
                    )

            # Use your custom runner
            runner = CustomRolloutRunner(exp_config)
            runner.run()  # Workers will use CustomRolloutRunner.run_single_rollout!
    """

    def __init__(self, exp_config: MlSpacesExpConfig) -> None:
        """
        Initialize the parallel rollout runner.

        Args:
            exp_config: Experiment configuration
        """
        self.config = exp_config

        # House-based processing setup
        self.house_indices = exp_config.task_sampler_config.house_inds
        self.samples_per_house = exp_config.task_sampler_config.samples_per_house

        # don't have houses specified, use all, need to find out which ones those are
        if self.house_indices is None:
            # if user-specified scene xml paths, use all of them
            if exp_config.task_sampler_config.scene_xml_paths:
                assert exp_config.scene_dataset == "user"
                self.house_indices = list(
                    range(len(exp_config.task_sampler_config.scene_xml_paths))
                )
            else:
                # For normal datagen: use all houses from scene mapping
                mapping = get_scenes(exp_config.scene_dataset, exp_config.data_split)
                self.house_indices = [
                    k for k, v in mapping[exp_config.data_split].items() if v is not None
                ]

        self.total_houses = len(self.house_indices)

        # Build (house_id, batch_samples, batch_num, total_batches) work items
        episodes_per_batch = exp_config.task_sampler_config.episodes_per_batch
        self.work_items: list[tuple[int, int, int, int]] = []
        for house_id in self.house_indices:
            total_batches = max(1, round(self.samples_per_house / episodes_per_batch))
            base = self.samples_per_house // total_batches
            extra = self.samples_per_house % total_batches
            for b in range(total_batches):
                batch_samples = base + (1 if b < extra else 0)
                self.work_items.append((house_id, batch_samples, b + 1, total_batches))

        # Failure tracking limits
        self.max_allowed_sequential_task_sampler_failures = (
            exp_config.task_sampler_config.max_allowed_sequential_task_sampler_failures
        )
        self.max_allowed_sequential_rollout_failures = (
            exp_config.task_sampler_config.max_allowed_sequential_rollout_failures
        )
        self.max_allowed_sequential_irrecoverable_failures = (
            exp_config.task_sampler_config.max_allowed_sequential_irrecoverable_failures
        )

        # Setup shared state for multiprocessing
        self.counter_lock = mp_context.Lock()
        self.shutdown_event = mp_context.Event()
        self.house_counter = mp_context.Value("i", 0)
        self.success_count = mp_context.Value("i", 0)
        self.total_count = mp_context.Value("i", 0)
        self.completed_houses = mp_context.Value("i", 0)
        self.skipped_houses = mp_context.Value("i", 0)

        # SIGTERM handling (only register in main thread)
        # Signal handlers can only be registered in the main thread of the main interpreter
        # This can fail when ParallelRolloutRunner is created from worker threads (e.g., DDP, PyTorch Lightning)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._handle_sigterm)
        else:
            # If not in main thread, skip signal handler registration
            # This is safe - the runner will still work, just without SIGTERM handling
            pass

        # Logging & Profiling
        init_logging(
            human_log_level=exp_config.log_level, log_file=exp_config.output_dir / "running_log.log"
        )
        self.logger = get_logger()
        self.profiler = exp_config.profiler

        # WandB initialization (optional, based on environment variables)
        self.wandb_enabled = False
        if "WANDB_RUN_NAME" in os.environ and "WANDB_PROJECT_NAME" in os.environ:
            self.logger.info("Initializing WandB logging...")
            try:
                wandb.init(
                    project=os.environ["WANDB_PROJECT_NAME"],
                    entity=os.environ.get("WANDB_ENTITY", None),
                    name=os.environ["WANDB_RUN_NAME"],
                    config={
                        "num_workers": exp_config.num_workers,
                        "total_houses": self.total_houses,
                        "samples_per_house": self.samples_per_house,
                        "total_work_items": len(self.work_items),
                        "total_expected_episodes": sum(wi[1] for wi in self.work_items),
                        "output_dir": str(exp_config.output_dir),
                        "filter_for_successful_trajectories": exp_config.filter_for_successful_trajectories,
                    },
                )
                self.wandb_enabled = True
                self.logger.info("WandB initialized successfully")
            except Exception as e:
                self.logger.warning(f"WandB initialization failed: {e}. Continuing without WandB.")
                self.wandb_enabled = False
        else:
            self.logger.info("WandB logging disabled (environment variables not set)")

    # =========================================================================
    # Episode Processing Hooks
    # Override these in subclasses to customize episode iteration behavior.
    # All hooks are static methods called via runner_class in process_single_house.
    # =========================================================================

    @staticmethod
    def load_episodes_for_house(
        exp_config: MlSpacesExpConfig,
        house_id: int,
        batch_suffix: str,
        worker_task_sampler,
        worker_logger,
    ) -> tuple[list, Any]:
        """
        Load episode specifications for a house.

        Returns:
            tuple: (episode_specs, task_sampler_to_use)
                - episode_specs: List of saved configs or None values
                - task_sampler_to_use: Task sampler for sampling tasks
        """
        # Datagen mode: generate None list for fresh sampling
        max_multiplier = exp_config.task_sampler_config.max_total_attempts_multiplier
        num_samples = exp_config.task_sampler_config.samples_per_house * max_multiplier
        return [None] * num_samples, worker_task_sampler

    @staticmethod
    def get_max_episode_attempts(
        episode_specs: list,
        samples_per_house: int,
        exp_config: MlSpacesExpConfig,
    ) -> int:
        """Get maximum number of episode attempts for this house."""
        return len(episode_specs)

    @staticmethod
    def should_stop_early(
        num_collected: int, samples_per_house: int, exp_config: "MlSpacesExpConfig | None" = None
    ) -> bool:
        """Whether to stop before processing all episodes (e.g., enough successes)."""
        return num_collected >= samples_per_house

    @staticmethod
    def get_episode_spec_at_index(episode_specs: list, idx: int) -> Any:
        """Get episode specification at given index."""
        return episode_specs[idx]

    @staticmethod
    def prepare_episode_config(
        exp_config: MlSpacesExpConfig,
        episode_spec,
        episode_idx: int,
    ) -> MlSpacesExpConfig:
        """Prepare config for a specific episode. Override to modify config per-episode."""
        return exp_config  # Base class doesn't modify config

    @staticmethod
    def get_episode_task_sampler(
        exp_config: MlSpacesExpConfig,
        episode_spec,
        shared_task_sampler,
        datagen_profiler: DatagenProfiler | None,
    ) -> Any:
        """Get task sampler for episode. Override to create per-episode samplers."""
        return shared_task_sampler

    @staticmethod
    def sample_task_from_spec(
        task_sampler, house_id: int, episode_spec, episode_idx: int
    ) -> BaseMujocoTask | None:
        """Sample task from specification."""
        return task_sampler.sample_task(house_index=house_id)

    @staticmethod
    def get_episode_seed(episode_idx: int, episode_spec, task_sampler) -> int:
        """Get seed for episode."""
        return task_sampler.current_seed

    @staticmethod
    def should_close_episode_task_sampler() -> bool:
        """Whether to close task sampler after each episode."""
        return False

    @staticmethod
    def run_single_rollout(
        episode_seed: int,
        task: BaseMujocoTask,
        policy: Any,
        profiler: Profiler | None = None,
        viewer=None,
        shutdown_event=None,
        datagen_profiler: DatagenProfiler | None = None,
        end_on_success: bool = False,
    ) -> bool:
        """Execute a single rollout with the given task and policy.

        Args:
            episode_seed: Seed for this episode
            task: The task to run
            policy: Policy to use for action selection
            profiler: Legacy Profiler instance (optional)
            viewer: MuJoCo viewer for visualization (optional)
            shutdown_event: Event to signal shutdown (optional)
            datagen_profiler: DatagenProfiler for per-worker timing (optional)

        Returns:
            bool: Whether the episode was successful
        """
        if profiler is not None:
            profiler.start("rollout")
        if datagen_profiler is not None:
            datagen_profiler.start("rollout_total")
            datagen_profiler.start("rollout_reset")

        observation, _info = task.reset()

        if datagen_profiler is not None:
            datagen_profiler.end("rollout_reset")

        if viewer is not None:
            viewer.sync()

        try:
            task.env.current_model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
            # mujoco 3.11 raised the default sleep_tolerance from 1e-4 to 1e-3
            task.env.current_model.opt.sleep_tolerance = 1e-3
        except AttributeError:
            print("Not setting mujoco sleep. Needs version >=mujoco-3.8")

        step_count = 0
        while not task.is_done():
            # Check for shutdown signal
            if shutdown_event is not None and shutdown_event.is_set():
                if datagen_profiler is not None:
                    datagen_profiler.end("rollout_total")
                return False

            # Step with policy
            if profiler is not None:
                profiler.start("policy_get_action")
            if datagen_profiler is not None:
                datagen_profiler.start("policy_get_action")
            # An action chunk is a list of actions to be applied open-loop before a
            # new observation is needed.
            action_chunk = policy.get_action_chunk(observation) or [policy.get_action(observation)]
            if profiler is not None:
                profiler.end("policy_get_action")
            if datagen_profiler is not None:
                datagen_profiler.end("policy_get_action")

            # Step the task
            if profiler is not None:
                profiler.start("task_step")
            if datagen_profiler is not None:
                datagen_profiler.start("task_step")
            if action_chunk[0] is None:
                print("Policy returned None action, ending episode")
                break
            observation, reward, terminal, truncated, infos = task.step_chunk(
                action_chunk, stop_on_success=end_on_success
            )
            step_count += len(action_chunk)
            if profiler is not None:
                profiler.end("task_step")
            if datagen_profiler is not None:
                datagen_profiler.end("task_step")
            # Add termination if succ
            if end_on_success and "success" in infos[0] and infos[0]["success"]:
                success = True
                break

            if viewer is not None:
                viewer.sync()

        try:
            task.env.current_model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        except AttributeError:
            print("Not setting mujoco sleep. Needs version >=mujoco-3.8")

        # Save profiler summary
        if profiler is not None:
            profiler.end("rollout")
        if datagen_profiler is not None:
            datagen_profiler.end("rollout_total")
            # Record step count for reference
            datagen_profiler.record(
                "step_count_indicator", step_count / 1000.0
            )  # Scale down to avoid confusion

        # Check success if method exists
        success = task.judge_success() if hasattr(task, "judge_success") else False

        return success

    @staticmethod
    def process_single_house(
        worker_id: int,
        worker_logger,
        house_id: int,
        exp_config: MlSpacesExpConfig,
        samples_per_house: int,
        shutdown_event,
        task_sampler,
        preloaded_policy: BasePolicy | None = None,
        max_allowed_sequential_task_sampler_failures: int = 10,
        max_allowed_sequential_rollout_failures: int = 10,
        filter_for_successful_trajectories: bool = False,
        runner_class=None,
        batch_num: int | None = None,
        total_batches: int | None = None,
        datagen_profiler: DatagenProfiler | None = None,
    ) -> tuple[int, int, bool]:
        """
        Process all episodes for a single house using customizable hooks.

        This method uses a while loop to iterate over episodes, calling hook methods
        via runner_class to allow subclasses to customize behavior without duplicating
        the entire method.

        Hooks called (override in subclass to customize):
        - load_episodes_for_house: Load episode specs from source (JSON, etc.)
        - get_max_episode_attempts: Maximum iterations of the episode loop
        - should_stop_early: Whether to stop before max attempts (e.g., enough successes)
        - prepare_episode_config: Modify config per-episode
        - get_episode_task_sampler: Get/create task sampler for episode
        - sample_task_from_spec: Sample task from specification
        - get_episode_seed: Get seed for episode
        - should_close_episode_task_sampler: Whether to close sampler per-episode

        Args:
            worker_id: ID of the worker thread/process
            worker_logger: Logger instance for this worker
            house_id: Index of the house to process
            exp_config: Experiment configuration
            samples_per_house: Number of episodes to collect for this house
            shutdown_event: Event to signal shutdown
            task_sampler: Task sampler instance (shared across houses for this worker)
            preloaded_policy: Optional pre-initialized policy instance
            max_allowed_sequential_task_sampler_failures: Max consecutive task sampling failures
            max_allowed_sequential_rollout_failures: Max consecutive rollout failures
            filter_for_successful_trajectories: Whether to filter for successful trajectories only
            runner_class: Runner class with hook methods to call
            batch_num: Batch number for this house (for batched processing)
            total_batches: Total number of batches for this house
            datagen_profiler: DatagenProfiler for per-worker timing (optional)

        Returns:
            tuple: (house_success_count, house_total_count, irrecoverable_failure_flag)
        """
        house_success_count = 0
        house_total_count = 0
        irrecoverable_failure_in_house = False

        # Setup directories and check for existing output
        house_output_dir, house_debug_dir, batch_suffix, should_skip = setup_house_dirs(
            exp_config, house_id, batch_num, total_batches
        )
        if should_skip:
            worker_logger.info(
                f"SKIPPING HOUSE {house_id} BATCH {batch_num}/{total_batches}: "
                f"Output already exists at {house_output_dir / f'trajectories{batch_suffix}.h5'}"
            )
            return 0, 0, False

        # Load episodes using hook - allows subclasses to load from different scene datasets
        episode_specs, shared_task_sampler = runner_class.load_episodes_for_house(
            exp_config, house_id, batch_suffix, task_sampler, worker_logger
        )

        if not episode_specs:
            worker_logger.warning(f"No episodes to process for house {house_id}")
            return 0, 0, False

        max_attempts = runner_class.get_max_episode_attempts(
            episode_specs, samples_per_house, exp_config
        )

        # Collect raw history data for this house
        house_raw_histories = []
        house_debug_raw_histories = []

        # Sequential failure tracking
        num_sequential_task_sampler_failures = 0
        num_sequential_rollout_failures = 0
        viewer = None

        # While loop with explicit index - allows subclasses to customize iteration
        episode_idx = 0
        while episode_idx < max_attempts:
            # Check early stop condition (e.g., enough successes for datagen)
            should_stop = runner_class.should_stop_early(
                len(house_raw_histories), samples_per_house, exp_config=exp_config
            )
            if should_stop:
                break

            # Check for shutdown signal
            if shutdown_event.is_set():
                worker_logger.info(f"Worker {worker_id} house {house_id} received shutdown signal")
                irrecoverable_failure_in_house = True
                break

            # Check for too many consecutive task sampling failures
            if num_sequential_task_sampler_failures >= max_allowed_sequential_task_sampler_failures:
                worker_logger.error(
                    f"Worker {worker_id} house {house_id} encountered "
                    f"{num_sequential_task_sampler_failures} consecutive task sampling failures. "
                    "This is unrecoverable."
                )
                irrecoverable_failure_in_house = True
                break

            # Check for too many consecutive rollout failures
            if num_sequential_rollout_failures >= max_allowed_sequential_rollout_failures:
                worker_logger.error(
                    f"Worker {worker_id} house {house_id} rollout failed across "
                    f"{num_sequential_rollout_failures} retries. This is irrecoverable."
                )
                irrecoverable_failure_in_house = True
                break

            # Get episode spec for this iteration
            episode_spec = runner_class.get_episode_spec_at_index(episode_specs, episode_idx)

            # Track state for this episode
            task = None
            policy = None
            episode_task_sampler = None
            success = False
            task_sampling_failed = False
            house_invalid = False

            if datagen_profiler is not None:
                datagen_profiler.start("episode_total")

            # Prepare episode-specific config
            episode_config = runner_class.prepare_episode_config(
                exp_config, episode_spec, episode_idx
            )

            with cleanup_context():
                if viewer is not None:
                    viewer.close()
                    viewer = None

                # Task sampling phase
                task_sampling_start = time.perf_counter()

                try:
                    # Get task sampler for this episode (shared or per-episode)
                    episode_task_sampler = runner_class.get_episode_task_sampler(
                        episode_config, episode_spec, shared_task_sampler, datagen_profiler
                    )

                    # Sample task
                    task = runner_class.sample_task_from_spec(
                        episode_task_sampler, house_id, episode_spec, episode_idx
                    )

                    if task is None:
                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx}: "
                            "task sampling returned None"
                        )
                        house_invalid = True
                    else:
                        # Record successful sampling time
                        if datagen_profiler is not None:
                            datagen_profiler.record(
                                "task_sampling", time.perf_counter() - task_sampling_start
                            )
                            task.set_datagen_profiler(datagen_profiler)

                        num_sequential_task_sampler_failures = 0

                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx}/{max_attempts} "
                            f"collected={len(house_raw_histories)}/{samples_per_house}"
                        )

                except HouseInvalidForTask as e:
                    traceback.print_exc()
                    worker_logger.warning(
                        f"Worker {worker_id} house {house_id} episode {episode_idx} "
                        f"HouseInvalidForTask: {e.reason}"
                    )
                    house_invalid = True
                    if datagen_profiler is not None:
                        datagen_profiler.record(
                            "task_sampling_failed", time.perf_counter() - task_sampling_start
                        )

                except Exception as e:
                    traceback.print_exc()
                    worker_logger.error(
                        f"Worker {worker_id} house {house_id} episode {episode_idx} "
                        f"task sampling error: {str(e)}"
                    )
                    num_sequential_task_sampler_failures += 1
                    task_sampling_failed = True
                    if datagen_profiler is not None:
                        datagen_profiler.record(
                            "task_sampling_failed", time.perf_counter() - task_sampling_start
                        )

                # Rollout phase (only if task sampling succeeded)
                if task is not None and not house_invalid and not task_sampling_failed:
                    try:
                        # Setup policy and viewer
                        policy = setup_policy(
                            episode_config, task, preloaded_policy, datagen_profiler
                        )
                        viewer = setup_viewer(episode_config, task, policy, viewer)

                        # Get episode seed
                        episode_seed = runner_class.get_episode_seed(
                            episode_idx, episode_spec, episode_task_sampler
                        )

                        # Run the rollout
                        success = runner_class.run_single_rollout(
                            episode_seed=episode_seed,
                            task=task,
                            policy=policy,
                            profiler=episode_config.profiler,
                            viewer=viewer,
                            shutdown_event=shutdown_event,
                            datagen_profiler=datagen_profiler,
                            end_on_success=exp_config.end_on_success,
                        )

                        num_sequential_rollout_failures = 0

                        # Extract object name for logging if available
                        object_name = "unknown"
                        if hasattr(task, "config") and hasattr(task.config, "task_config"):
                            if hasattr(task.config.task_config, "pickup_obj_name"):
                                object_name = task.config.task_config.pickup_obj_name

                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx} "
                            f"object {object_name} completed with success={success}"
                        )

                        # Collect trajectory
                        should_save = success or not filter_for_successful_trajectories
                        history = task.get_history()
                        should_save_debug = not should_save and random.random() < 0.01

                        if should_save or should_save_debug:
                            episode_info = {
                                "history": history,
                                "sensor_suite": task.sensor_suite,
                                "success": success,
                                "seed": episode_seed,
                            }
                            if should_save:
                                house_raw_histories.append(episode_info)
                            elif should_save_debug:
                                house_debug_raw_histories.append(episode_info)
                                worker_logger.info(
                                    f"Queueing failed trajectory for debug (seed: {episode_seed})"
                                )
                        else:
                            del history

                        # Update house counters
                        house_total_count += 1
                        if success:
                            house_success_count += 1
                        else:
                            # Report failure for this asset (may lead to dynamic blacklisting)
                            asset_uid = task_sampler.get_asset_uid_from_object(
                                task.env, object_name
                            )
                            if asset_uid:
                                task_sampler.report_asset_failure(
                                    asset_uid, "rollout failed (e.g., IK failure)"
                                )

                        if datagen_profiler is not None:
                            datagen_profiler.end("episode_total")
                            datagen_profiler.log_episode_summary(
                                episode_idx=episode_idx,
                                house_id=house_id,
                                success=success,
                            )

                    except Exception as e:
                        worker_logger.error(
                            f"Worker {worker_id} house {house_id} episode {episode_idx} rollout error: {str(e)}"
                        )
                        traceback.print_exc()
                        num_sequential_rollout_failures += 1

                        # Report failure for this asset (may lead to dynamic blacklisting)
                        try:
                            asset_uid = task_sampler.get_asset_uid_from_object(
                                task.env, object_name
                            )
                            if asset_uid:
                                task_sampler.report_asset_failure(
                                    asset_uid, f"rollout exception: {e}"
                                )
                        except Exception:
                            pass  # Don't let failure tracking break the error handling

                        if datagen_profiler is not None:
                            datagen_profiler.end("episode_total")

                else:
                    # Task sampling failed or house invalid
                    if datagen_profiler is not None:
                        datagen_profiler.end("episode_total")

                # Cleanup resources
                cleanup_episode_resources(
                    task=task,
                    policy=policy,
                    task_sampler=episode_task_sampler,
                    preloaded_policy=preloaded_policy,
                    close_task_sampler=runner_class.should_close_episode_task_sampler(),
                )

            # Handle house invalid - break after cleanup
            if house_invalid:
                irrecoverable_failure_in_house = True
                break

            # Always increment episode index
            episode_idx += 1

        # Cleanup viewer
        if viewer is not None:
            viewer.close()
            viewer = None

        # Check shutdown signal before saving
        if shutdown_event.is_set():
            worker_logger.info(
                f"Worker {worker_id} house {house_id} shutdown requested, skipping save"
            )
            return house_success_count, house_total_count, True

        # Save trajectories
        save_house_trajectories(
            worker_logger,
            house_raw_histories,
            house_output_dir,
            exp_config,
            batch_suffix,
            datagen_profiler,
            batch_num,
            total_batches,
        )

        # Save debug trajectories
        save_house_trajectories(
            worker_logger,
            house_debug_raw_histories,
            house_debug_dir,
            exp_config,
            batch_suffix,
            datagen_profiler=None,
            batch_num=batch_num,
            total_batches=total_batches,
        )

        worker_logger.info(
            f"Worker {worker_id} completed house {house_id}: "
            f"{house_success_count}/{house_total_count} successful episodes"
        )

        if datagen_profiler is not None:
            datagen_profiler.log_house_summary(
                house_id=house_id,
                success_count=house_success_count,
                total_count=house_total_count,
            )

        return house_success_count, house_total_count, irrecoverable_failure_in_house

    def _handle_sigterm(self, signum, frame):
        """Handle SIGTERM signal gracefully."""
        self.shutdown_event.set()
        self.logger.info("Received SIGTERM. Initiating graceful shutdown...")
        if self.wandb_enabled:
            try:
                wandb.finish()
            except Exception as e:
                self.logger.warning(f"WandB cleanup on SIGTERM failed: {e}")

    def run(self, preloaded_policy: BasePolicy | None = None) -> tuple[int, int]:
        """
        Run house-by-house rollouts using multiprocessing workers.

        Args:
            preloaded_policy: Optional pre-initialized policy instance to use for rollouts.
                If None, a new policy will be created for each rollout.

        Returns:
            tuple: (success_count, total_count)
        """
        total_expected_episodes = sum(wi[1] for wi in self.work_items)
        self.logger.info(
            f"Starting rollout of {self.total_houses} houses "
            f"split into {len(self.work_items)} work items ({total_expected_episodes} total episodes) "
            f"using {self.config.num_workers} worker processes"
        )

        # make a copy of the config in the output directory
        self.logger.info("Evaluation configuration:")
        self.logger.info(pprint.pformat(self.config.model_dump()))
        self.config.save_config(output_dir=Path(self.config.output_dir))

        # Start timing for WandB metrics
        start_time = time.time()

        # Launch worker processes
        if self.config.num_workers > 1:
            processes = []
            for worker_id in range(self.config.num_workers):
                p = mp_context.Process(
                    target=house_processing_worker,
                    args=(
                        worker_id,
                        self.config,
                        self.work_items,
                        self.shutdown_event,
                        self.counter_lock,
                        self.house_counter,
                        self.success_count,
                        self.total_count,
                        self.completed_houses,
                        self.skipped_houses,
                        self.max_allowed_sequential_task_sampler_failures,
                        self.max_allowed_sequential_rollout_failures,
                        self.max_allowed_sequential_irrecoverable_failures,
                        preloaded_policy,
                        self.config.filter_for_successful_trajectories,
                        type(self),  # Pass the runner class to enable customization via subclassing
                    ),
                )
                p.start()
                processes.append(p)

            # Periodic logging loop that monitors progress while workers run
            last_log_time = start_time
            log_interval = 60  # Log every 60 seconds

            while any(p.is_alive() for p in processes):
                # Check if it's time to log
                current_time = time.time()
                if self.wandb_enabled and (current_time - last_log_time) >= log_interval:
                    try:
                        # Read current progress from shared counters
                        elapsed_time = current_time - start_time
                        completed = self.completed_houses.value
                        skipped = self.skipped_houses.value
                        success = self.success_count.value
                        total = self.total_count.value
                        active = sum(1 for p in processes if p.is_alive())

                        # Calculate metrics
                        total_work_items = len(self.work_items)
                        success_rate = success / total if total > 0 else 0.0
                        episodes_per_second = total / elapsed_time if elapsed_time > 0 else 0.0
                        completion_percentage = (completed + skipped) / total_work_items * 100

                        # Log to WandB
                        wandb.log(
                            {
                                "elapsed_time_seconds": elapsed_time,
                                "elapsed_time_hours": elapsed_time / 3600,
                                "completed_houses": completed,
                                "skipped_houses": skipped,
                                "success_count": success,
                                "total_count": total,
                                "success_rate": success_rate,
                                "episodes_per_second": episodes_per_second,
                                "active_workers": active,
                                "completion_percentage": completion_percentage,
                            }
                        )
                        self.logger.info(
                            f"Progress: {completed}/{total_work_items} work items completed "
                            f"({completion_percentage:.1f}%), {success}/{total} successful episodes "
                            f"({success_rate * 100:.1f}%), {active} workers active"
                        )
                        last_log_time = current_time
                    except Exception as e:
                        self.logger.warning(f"WandB periodic logging failed: {e}")

                # Sleep briefly before checking again

                time.sleep(5)

            # Wait for all processes to complete
            for p in processes:
                p.join()
                p.close()

        else:
            # Single-worker mode runs in the main process
            house_processing_worker(
                worker_id=0,
                exp_config=self.config,
                work_items=self.work_items,
                shutdown_event=self.shutdown_event,
                counter_lock=self.counter_lock,
                house_counter=self.house_counter,
                success_count=self.success_count,
                total_count=self.total_count,
                completed_houses=self.completed_houses,
                skipped_houses=self.skipped_houses,
                max_allowed_sequential_task_sampler_failures=self.max_allowed_sequential_task_sampler_failures,
                max_allowed_sequential_rollout_failures=self.max_allowed_sequential_rollout_failures,
                max_allowed_sequential_irrecoverable_failures=self.max_allowed_sequential_irrecoverable_failures,
                preloaded_policy=preloaded_policy,
                filter_for_successful_trajectories=self.config.filter_for_successful_trajectories,
                runner_class=type(self),
            )

        # Extract final values from shared multiprocessing state
        success_count_val = self.success_count.value
        total_count_val = self.total_count.value
        completed_houses_val = self.completed_houses.value
        skipped_houses_val = self.skipped_houses.value

        success_rate = success_count_val / total_count_val if total_count_val > 0 else 0.0
        self.logger.info(
            f"Completed {completed_houses_val} work items, skipped {skipped_houses_val} work items"
        )
        self.logger.info(f"Success count: {success_count_val}, Total count: {total_count_val}")
        self.logger.info(f"Success rate: {success_rate * 100:.2f}%")

        # Log final metrics to WandB
        if self.wandb_enabled:
            try:
                final_elapsed_time = time.time() - start_time
                wandb.log(
                    {
                        "final_success_count": success_count_val,
                        "final_total_count": total_count_val,
                        "final_success_rate": success_rate,
                        "final_completed_houses": completed_houses_val,
                        "final_skipped_houses": skipped_houses_val,
                        "final_elapsed_time_seconds": final_elapsed_time,
                        "final_elapsed_time_hours": final_elapsed_time / 3600,
                    }
                )
                wandb.finish()
                self.logger.info("WandB logging finished")
            except Exception as e:
                self.logger.warning(f"WandB final logging failed: {e}")

        return success_count_val, total_count_val
