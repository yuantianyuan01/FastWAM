import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MutableFloat:
    value: float | None = None


@contextmanager
def Timer():
    time_taken = MutableFloat()
    start = time.perf_counter()
    yield time_taken
    time_taken.value = time.perf_counter() - start


class Profiler:
    def __init__(self, log_realtime: bool = False, save_path: str = None) -> None:
        self._start_time = {}
        self._end_time = {}
        self._avg_time = {}
        self._n = {}
        self.start_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_realtime = log_realtime
        self.save_path = None
        if self.log_realtime:
            assert save_path is not None, "save_path must be provided if log_realtime is True"
            os.makedirs(save_path, exist_ok=True)
            self.save_path = Path(save_path) / f"profiling_summary_{self.start_timestamp}.txt"

    @contextmanager
    def profile(self, key):
        self.start(key)
        try:
            yield
        finally:
            self.end(key)

    def start(self, key) -> None:
        self._start_time[key] = time.perf_counter()

    def end(self, key) -> None:
        self._end_time[key] = time.perf_counter()
        _time = self._end_time[key] - self._start_time[key]
        self._avg_time[key] = (self.get_avg_time(key) * self.get_n(key) + _time) / (
            self.get_n(key) + 1
        )
        self._n[key] = self.get_n(key) + 1

        if self.log_realtime:
            self._write_summary()

    def _write_summary(self) -> None:
        """Write/rewrite the entire summary to file"""
        with open(self.save_path, "a") as f:
            f.write(f"=== Profiling Summary (Started at {self.start_timestamp}) ===\n\n")

            # Sort keys by total time for consistent ordering
            sorted_keys = sorted(
                self._avg_time.keys(), key=lambda k: self._avg_time[k] * self._n[k], reverse=True
            )

            for key in sorted_keys:
                f.write(f"Operation: {key}\n")
                f.write(f"Average time: {self._avg_time[key]:.4f} seconds\n")
                f.write(f"Number of calls: {self._n[key]}\n")
                f.write(f"Total time: {self._avg_time[key] * self._n[key]:.4f} seconds\n")
                f.write("-" * 50 + "\n")

    def get_avg_time(self, key):
        return self._avg_time.get(key, 0)

    def get_n(self, key):
        return self._n.get(key, 0)

    def print_all(self) -> None:
        for key in self._avg_time:
            print(f"{key}: {self._avg_time[key]}")

    def save_summary(self, save_path: str) -> None:
        assert save_path is not None, (
            "save_path must be provided if profiler summary is being saved"
        )
        os.makedirs(save_path, exist_ok=True)
        self.save_path = Path(save_path) / f"profiling_summary_{self.start_timestamp}.txt"
        self._write_summary()


class DatagenProfiler:
    """
    Per-worker profiler for distributed data generation that accumulates timing stats
    across episodes and houses, logging summaries to the worker logger.

    Tracks operations like:
    - task_sampling: Time to sample a task from the task sampler
    - policy_setup: Time to create/setup the policy
    - rollout_total: Total time for a rollout (reset + all steps)
    - rollout_reset: Time for task.reset()
    - policy_get_action: Time for policy.get_action() calls (per step, accumulated)
    - task_step: Time for task.step() calls (per step, accumulated)
    - episode_total: Total time for one episode (sampling + policy setup + rollout)
    - save_batch_prep: Time to prepare episode for saving
    - save_trajectories: Time to save trajectory data

    Usage:
        profiler = DatagenProfiler(logger)

        # For each episode:
        with profiler.profile("task_sampling"):
            task = task_sampler.sample_task(...)

        # After each episode:
        profiler.log_episode_summary(episode_idx, house_id)

        # After each house:
        profiler.log_house_summary(house_id)
    """

    def __init__(self, logger=None, enabled: bool = True) -> None:
        """
        Initialize the datagen profiler.

        Args:
            logger: Logger instance to output summaries to. If None, uses get_logger().
            enabled: Whether profiling is enabled. If False, all operations are no-ops.
        """
        self.logger = logger
        self.enabled = enabled

        # Timing storage: key -> list of durations (in seconds)
        # Episode-level: cleared after each episode summary
        self._episode_times: dict[str, list[float]] = defaultdict(list)
        # House-level: cleared after each house summary
        self._house_times: dict[str, list[float]] = defaultdict(list)
        # Worker-level (cumulative across all houses): never cleared
        self._worker_times: dict[str, list[float]] = defaultdict(list)

        # Active timers (key -> start time)
        self._active_timers: dict[str, float] = {}

        # Episode counters
        self._episode_count_in_house = 0
        self._total_episode_count = 0
        self._house_count = 0

    def start(self, key: str) -> None:
        """Start timing an operation."""
        if not self.enabled:
            return
        self._active_timers[key] = time.perf_counter()

    def end(self, key: str) -> None:
        """End timing an operation and record the duration."""
        if not self.enabled:
            return
        if key not in self._active_timers:
            return
        duration = time.perf_counter() - self._active_timers.pop(key)
        self._episode_times[key].append(duration)
        self._house_times[key].append(duration)
        self._worker_times[key].append(duration)

    @contextmanager
    def profile(self, key: str):
        """Context manager for profiling a block of code."""
        self.start(key)
        try:
            yield
        finally:
            self.end(key)

    def record(self, key: str, duration: float) -> None:
        """Directly record a duration for an operation (useful when timing is external)."""
        if not self.enabled:
            return
        self._episode_times[key].append(duration)
        self._house_times[key].append(duration)
        self._worker_times[key].append(duration)

    def _format_stats(self, times_dict: dict[str, list[float]], prefix: str = "") -> str:
        """Format timing stats into a readable string."""
        if not times_dict:
            return f"{prefix}No timing data recorded"

        lines = []
        # Sort by total time descending
        sorted_keys = sorted(
            times_dict.keys(),
            key=lambda k: sum(times_dict[k]),
            reverse=True,
        )

        for key in sorted_keys:
            values = times_dict[key]
            if not values:
                continue
            total = sum(values)
            count = len(values)
            mean = total / count
            min_val = min(values)
            max_val = max(values)

            # Format based on magnitude
            if mean < 0.001:
                mean_str = f"{mean * 1000000:.1f}us"
            elif mean < 1.0:
                mean_str = f"{mean * 1000:.1f}ms"
            else:
                mean_str = f"{mean:.2f}s"

            if total < 1.0:
                total_str = f"{total * 1000:.1f}ms"
            else:
                total_str = f"{total:.2f}s"

            lines.append(
                f"  {key}: mean={mean_str}, total={total_str}, count={count}, "
                f"min={min_val * 1000:.1f}ms, max={max_val * 1000:.1f}ms"
            )

        return prefix + "\n".join(lines)

    def log_episode_summary(
        self, episode_idx: int, house_id: int, success: bool | None = None
    ) -> None:
        """
        Log a summary of timing for the current episode.

        Args:
            episode_idx: Index of the episode within the house
            house_id: ID of the house being processed
            success: Whether the episode was successful (optional)
        """
        if not self.enabled:
            return
        if self.logger is None:
            return

        self._episode_count_in_house += 1
        self._total_episode_count += 1

        success_str = ""
        if success is not None:
            success_str = f" success={success}"

        # Calculate episode total if we have individual components
        episode_total = 0.0
        for key in ["task_sampling", "policy_setup", "rollout_total"]:
            if key in self._episode_times:
                episode_total += sum(self._episode_times[key])

        total_str = f" episode_total={episode_total:.2f}s" if episode_total > 0 else ""

        self.logger.info(
            f"[PROFILE] Episode {episode_idx} house {house_id}{success_str}{total_str}:\n"
            + self._format_stats(self._episode_times)
        )

        # Clear episode-level times
        self._episode_times.clear()

    def log_house_summary(self, house_id: int, success_count: int, total_count: int) -> None:
        """
        Log a summary of timing for the current house (accumulated across all episodes).

        Args:
            house_id: ID of the house that was processed
            success_count: Number of successful episodes in this house
            total_count: Total number of episodes attempted in this house
        """
        if not self.enabled:
            return
        if self.logger is None:
            return

        self._house_count += 1

        # Calculate some high-level stats
        house_total = sum(sum(v) for v in self._house_times.values())

        self.logger.info(
            f"[PROFILE] House {house_id} complete: {success_count}/{total_count} successful, "
            f"{self._episode_count_in_house} episodes, total_time={house_total:.2f}s\n"
            + self._format_stats(self._house_times, prefix="  House averages:\n")
        )

        # Clear house-level times and reset episode counter
        self._house_times.clear()
        self._episode_count_in_house = 0

    def log_worker_summary(self) -> None:
        """
        Log a summary of timing for the entire worker (accumulated across all houses).
        Call this when the worker is shutting down.
        """
        if not self.enabled:
            return
        if self.logger is None:
            return

        worker_total = sum(sum(v) for v in self._worker_times.values())

        self.logger.info(
            f"[PROFILE] Worker complete: {self._house_count} houses, "
            f"{self._total_episode_count} episodes, total_time={worker_total:.2f}s\n"
            + self._format_stats(self._worker_times, prefix="  Worker averages:\n")
        )

    def get_episode_stats(self) -> dict[str, dict[str, float]]:
        """Get current episode timing stats as a dict."""
        stats = {}
        for key, values in self._episode_times.items():
            if values:
                stats[key] = {
                    "total": sum(values),
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        return stats

    def get_house_stats(self) -> dict[str, dict[str, float]]:
        """Get current house timing stats as a dict."""
        stats = {}
        for key, values in self._house_times.items():
            if values:
                stats[key] = {
                    "total": sum(values),
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        return stats
