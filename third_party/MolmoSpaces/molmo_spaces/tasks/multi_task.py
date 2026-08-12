from typing import Any

import numpy as np

from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.sensors import get_core_sensors
from molmo_spaces.tasks.task import BaseMujocoTask


class MultiTask(BaseMujocoTask):
    def __init__(self, tasks: list[BaseMujocoTask], prompt) -> None:
        if not tasks:
            raise ValueError("MultiTask requires at least one task")

        self.tasks = tasks
        self.prompt = prompt

        self.max_rewards = [0.0] * len(tasks)

        self._env = tasks[0]._env
        self._ctrl_dt_ms = tasks[0]._ctrl_dt_ms
        self._n_sim_steps_per_ctrl = tasks[0]._n_sim_steps_per_ctrl
        self._n_ctrl_steps_per_policy = tasks[0]._n_ctrl_steps_per_policy
        self._task_horizon = tasks[0]._task_horizon
        self._cumulative_reward = np.zeros(self._env.n_batch)
        self._num_steps_taken = np.zeros(self._env.n_batch, dtype=int)
        self.config = tasks[0].config
        self.episode_step_count = 0
        self.viewer = None
        self.frozen_config = None
        self._sensor_suite = tasks[0]._sensor_suite
        self.last_action = None
        self.action_cache = []
        self.observation_cache = []
        self.reward_cache = []
        self.terminal_cache = []
        self.truncated_cache = []
        self.success_cache = []
        self._policy_done = False
        self._registered_policy = None
        self._done_action_received = False
        self._datagen_profiler = None

    def _create_sensor_suite_from_config(self, exp_config):
        return SensorSuite(get_core_sensors(exp_config))

    def get_task_description(self) -> str:
        descriptions = [task.get_task_description() for task in self.tasks]
        return " OR ".join(descriptions)

    def register_policy(self, policy) -> None:
        self._registered_policy = policy
        for task in self.tasks:
            task.register_policy(policy)

    def judge_success(self) -> bool:
        return any(task.judge_success() for task in self.tasks)

    def get_reward(self) -> np.ndarray:
        all_rewards = []
        completed_tasks = []

        print(f"Task Progress for: {self.prompt}")

        for i, task in enumerate(self.tasks):
            task_reward = task.get_reward()
            all_rewards.append(task_reward)

            task.get_task_description()
            reward_value = task_reward[0] if isinstance(task_reward, np.ndarray) else task_reward

            if reward_value > self.max_rewards[i]:
                self.max_rewards[i] = reward_value

            task_type = task.config.task_type
            if task_type == "pick_and_place":
                threshold = 0.9
            elif task_type == "open" or task_type == "close":
                threshold = 0.5
            elif task_type == "pick":
                threshold = 0.02
            else:
                threshold = 0.9

            if self.max_rewards[i] > threshold:
                completed_tasks.append(i)

        num_completed = len(completed_tasks)
        total_tasks = len(self.tasks)
        progress_bar = "█" * (num_completed * 3) + "░" * ((total_tasks - num_completed) * 3)
        print(f"\nOverall: [{progress_bar}] {num_completed}/{total_tasks} completed")

        return all_rewards

    def get_info(self) -> list[dict[str, Any]]:
        all_infos = []
        return all_infos

    def get_obs_scene(self):
        obs_scene = self.tasks[0].get_obs_scene()
        obs_scene["text"] = self.get_task_description()
        obs_scene["num_tasks"] = len(self.tasks)
        return obs_scene

    def reset(self) -> dict[str, Any]:
        for task in self.tasks:
            task.reset()
        return self.tasks[0].reset()

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs, _, terminal, truncated, info = self.tasks[0].step(action)

        multi_reward = self.get_reward()
        multi_info = self.get_info()
        print(f"Prompt: {self.prompt}")

        reward = np.max(multi_reward)

        return obs, reward, terminal, truncated, multi_info
