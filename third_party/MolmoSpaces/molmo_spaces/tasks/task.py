"""
Task abstraction for MuJoCo-based robotic tasks.

The step() method accepts a single action dict for single-env mode, or a list of
action dicts (one per env) for batched mode. Action chunking (if needed) is the
responsibility of the policy.

Action Noise:
    Action noise is applied per-robot via Robot.apply_action_noise(). Configure via
    robot_config.action_noise_config. Each robot implementation specifies which move
    groups receive TCP-bounded noise (e.g., Franka applies to "arm", RBY1 applies
    independently to "left_arm" and "right_arm").
"""

import contextlib
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import MlSpacesObjectAbstract
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.env.object_manager import ObjectManager

if TYPE_CHECKING:
    from molmo_spaces.configs import BaseMujocoTaskConfig
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.policy.base_policy import BasePolicy


log = logging.getLogger(__name__)


class BaseMujocoTask(ABC):
    def __init__(
        self,
        env: BaseMujocoEnv,
        exp_config: "MlSpacesExpConfig",
    ) -> None:
        self._env = env
        self._ctrl_dt_ms = exp_config.ctrl_dt_ms
        sim_dt_ms = round(self._env.mj_model.opt.timestep * 1000)
        if self._ctrl_dt_ms % sim_dt_ms != 0:
            raise ValueError(
                f"Control dt {self._ctrl_dt_ms}ms is not divisible by sim dt {sim_dt_ms}ms"
            )
        self._n_sim_steps_per_ctrl = int(self._ctrl_dt_ms // sim_dt_ms)
        self._n_ctrl_steps_per_policy = int(exp_config.policy_dt_ms // self._ctrl_dt_ms)
        self._task_horizon = (
            exp_config.task_horizon if exp_config.task_horizon is not None else np.inf
        )
        self._cumulative_reward = np.zeros(self._env.n_batch)
        self._num_steps_taken = np.zeros(self._env.n_batch, dtype=int)
        self.config = exp_config
        self.episode_step_count = 0
        self.viewer = None  # placeholder to attach interactive viewer
        self.frozen_config = None

        if exp_config.task_config.use_sensors:
            self._sensor_suite = self._create_sensor_suite_from_config(exp_config)
            self._sensor_suite.extend(self._env.current_robot.create_robot_sensors())
        else:
            self._sensor_suite = None

        # Action tracking for ActionSensors - the most recent action dict
        self.last_action: dict[str, Any] | None = None

        # Caches env's input and outputs. Placed in env after discussion with Rose, placed in env instead
        # of wrapper (Max's preference) with reasoning of preventing the chance of env steps without
        # caching outputs.
        self.action_cache: list[dict[str, Any]] = []
        self.observation_cache: list[list[dict[str, Any]]] = []
        self.reward_cache: list[list[float]] = []
        self.terminal_cache: list[list[bool]] = []
        self.truncated_cache: list[list[bool]] = []
        self.success_cache: list[list[bool]] = []

        # Policy completion tracking
        self._policy_done = False
        self._registered_policy = None  # Reference to the active policy for phase tracking
        self._done_action_received = False  # Flag for when done action is received

        # Optional profiler for granular timing (set via set_datagen_profiler)
        self._datagen_profiler = None

        # Please don't call self.reset() here. reset should return the first observation, if we do it in
        # __init__ it will end up in the cache, but not being returned to the user.

    @property
    def sensor_suite(self) -> SensorSuite | None:
        """Get the sensor suite for this task."""
        return self._sensor_suite

    def set_datagen_profiler(self, profiler) -> None:
        """Set the datagen profiler for granular step timing (physics_step vs sensor_polling)."""
        self._datagen_profiler = profiler

    @abstractmethod
    def get_task_description(self) -> str:
        """Get the task description for this task."""
        raise NotImplementedError

    @staticmethod
    def deduplicate_task_objects_name(
        task_config: "BaseMujocoTaskConfig",
        input_key: str,
        task_objects: dict[str, str],
        output_key: str,
    ) -> None:
        task_object_name = getattr(task_config, input_key, None)
        if task_object_name is None:
            return

        to_pop = [key for key, name in task_objects.items() if name == task_object_name]

        for key in to_pop:
            del task_objects[key]

        task_objects[output_key] = task_object_name

    def get_task_objects(self, batch_index: int = 0) -> dict[str, str]:
        """Return a dict mapping object keys to body names for sensor tracking.

        By default, includes robot gripper bodies (handles both single-gripper robots
        like Franka and dual-gripper robots like RBY1) and all dynamically added
        objects from ``task_config.added_objects``.  Task samplers are expected
        to trim ``added_objects`` to only the episode-relevant subset before
        creating the task, so everything present is task-relevant by construction.
        Override in subclasses to add further task-specific objects by calling
        ``super().get_task_objects()`` and updating the dict.

        Keys should be descriptive (e.g., 'pickup_obj', 'door_handle', 'gripper').
        Values should be valid MuJoCo body names.

        Args:
            batch_index: Batch index for the environment (default 0).

        Returns:
            Dict mapping object role to body name.
        """
        objects = {}

        # Add robot gripper bodies (handles Franka's single gripper and RBY1's left/right grippers)
        robot_view = self._env.robots[batch_index].robot_view
        for gripper_mg_id in robot_view.get_gripper_movegroup_ids():
            gripper_mg = robot_view.get_move_group(gripper_mg_id)
            root_body_id = gripper_mg.root_body_id
            # Handle both int IDs and body view objects (some robot views store the view directly)
            if hasattr(root_body_id, "name"):
                gripper_body_name = root_body_id.name
            else:
                gripper_body_name = robot_view.mj_model.body(root_body_id).name
            objects[gripper_mg_id] = gripper_body_name

        added = getattr(self.config.task_config, "added_objects", {})
        for it, (name, _path) in enumerate(added.items()):
            objects[f"added_{it}"] = name

        return objects

    @abstractmethod
    def _create_sensor_suite_from_config(self, exp_config) -> SensorSuite:
        """
        Create a sensor suite with task-specific sensors.
        Robot-specific sensors should not be added here.
        """
        raise NotImplementedError

    def register_policy(self, policy: "BasePolicy") -> None:
        """
        Register a policy with the task for completion tracking and phase sensing.
        This should only be called once in the task's lifetime.
        """
        if self._registered_policy is not None:
            raise ValueError("Policy already registered")
        self._registered_policy = policy
        policy.task = self
        if self._sensor_suite is not None:
            self._sensor_suite.extend(policy.create_policy_sensors())

    def num_steps_taken(self) -> int:
        """Get the number of steps taken in the current episode."""
        return self.episode_step_count

    def get_observations(self) -> list[dict[str, Any]]:
        """Get observations using the sensor suite and accumulate all other information."""
        observations = []
        for i in range(self._env.n_batch):
            if self._sensor_suite is not None:
                env_obs = self._sensor_suite.get_observations(
                    env=self._env, task=self, batch_index=i
                )
            else:  # allow use_sensors to be False in exp_config
                env_obs = {}
            observations.append(env_obs)
        return observations

    def get_and_cache_all_step_information(
        self,
    ) -> tuple[
        list[dict[str, Any]], NDArray[float], NDArray[bool], NDArray[bool], list[dict[str, Any]]
    ]:
        """Get observations, reward, done, info and cache them."""
        observation = self.get_observations()
        reward = self.get_reward()
        terminated = self.is_terminal()
        truncated = self.is_timed_out()
        info = self.get_info()
        # TODO: do per-environment success tracking, this only does for index 0
        success = np.full(terminated.shape, fill_value=self.judge_success())

        # cache the inputs and outputs
        self.observation_cache.append(observation)
        self.reward_cache.append(reward)
        self.terminal_cache.append(terminated)
        self.truncated_cache.append(truncated)
        self.success_cache.append(success)

        return observation, reward, terminated, truncated, info

    def reset(self):
        """Reset the task and record initial observations."""
        # TODO(rose): Something like this should be done here to be compatible with gym API
        # consider placing settle_scene here.
        # self._env.reset()

        self.episode_step_count = 0
        self._cumulative_reward = np.zeros(self._env.n_batch)
        self._num_steps_taken = np.zeros(self._env.n_batch, dtype=int)

        # Action tracking for ActionSensors
        self.last_action = None
        self.action_cache = []
        self.observation_cache = []
        self.reward_cache = []
        self.terminal_cache = []
        self.truncated_cache = []
        self.success_cache = []
        self._policy_done = False
        self._done_action_received = False

        # Reset sensors that maintain state
        if self.sensor_suite:
            for sensor in self.sensor_suite.sensors.values():
                sensor.reset()

        # Why do we reset the policy here? I'ts because task.reset should return
        # the first observation, this includes sensors from the policy. So we both need to
        # have the env set up correctly, and not have recorded any observation yet.
        if self._registered_policy:
            self._registered_policy.reset()

        # get the current obs and return them, to align with the gymnasium API
        # TODO(max) - possibly this should include padding values for reward/terminal/truncated. Prefer to have everything be the same length for alignment, even if padding values are needed
        observation, reward, terminated, truncated, info = self.get_and_cache_all_step_information()

        self.frozen_config = self.config.freeze_task_config(observation, task=self)
        return observation, info

    def step(
        self,
        action: dict[str, Any] | list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]], NDArray[float], NDArray[bool], NDArray[bool], list[dict[str, Any]]
    ]:
        """Step the environment with a single action.

        Args:
            action: Single action dict for single-env mode, or list of action dicts
                (one per env) for batched mode.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        # TODO: how do we handle when one environment is done but others are not?
        # gym.Env handles it by automatically resetting environments that are done, but probably not ideal here.

        # Normalize action to list of dicts (one per env)
        if isinstance(action, dict):
            if self._env.n_batch != 1:
                raise ValueError(
                    f"Single dict action only supported for n_batch=1, got n_batch={self._env.n_batch}. "
                    f"For multiple environments, provide a list of {self._env.n_batch} action dicts."
                )
            actions = [action]
        else:
            if len(action) != self._env.n_batch:
                raise ValueError(
                    f"Action list length {len(action)} does not match n_batch={self._env.n_batch}."
                )
            actions = action

        # Verify the 0th observation if this is the first step - action sensor will be a padding value
        if self.num_steps_taken() == 0:
            obs = self.get_observations()
            # verify that the current obs are the same as what is in the cache from the reset
            # this would be violated if eg the task was initialized and then additional scene settling steps were taken before the first step

            # Check first camera from config to verify observations match
            if len(self.observation_cache) > 0:
                cached_obs = self.observation_cache[0]

                # Get camera name from camera config (observation key = camera_spec.name)
                camera_name = None
                if (
                    self.config.camera_config is not None
                    and len(self.config.camera_config.cameras) > 0
                ):
                    camera_name = self.config.camera_config.cameras[0].name

                if camera_name is not None and isinstance(cached_obs, list) and len(cached_obs) > 0:
                    if camera_name in obs[0] and camera_name in cached_obs[0]:
                        if not np.array_equal(obs[0][camera_name], cached_obs[0][camera_name]):
                            # Mismatch can occur due to mj_fwdPosition() being called during policy.reset()
                            # inside task.reset(). This happens during grasp collision checking in
                            # get_noncolliding_grasp_mask(). Overwrite cached obs with current state.
                            log.warning(
                                "Camera sensor '%s' observation mismatch between reset and first step. "
                                "Overwriting cached observation with current state.",
                                camera_name,
                            )
                            # Replace the cached observation with the current one
                            self.observation_cache[0] = obs

        # Check if all environments are done
        if np.all(self.is_done()):
            print("Warning: step() called on task where all environments are already done")
            # Return current state without stepping
            return self.get_and_cache_all_step_information()

        self._apply_action(actions)
        return self._observe_and_cache()

    def _apply_action(self, action: dict[str, Any] | list[dict[str, Any]]) -> None:
        """Apply one action and advance the simulation, without polling sensors."""
        actions = [action] if isinstance(action, dict) else action

        # Check if any action contains a "done" signal
        for act in actions:
            if isinstance(act, dict) and act.get("done", False):
                act.pop("done")
                self._done_action_received = True

        # Update episode step count
        self.episode_step_count += 1

        for robot, act in zip(self._env.robots, actions, strict=True):
            robot.update_control(act)

        # Physics step (MuJoCo simulation)
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("physics_step")
        for _ in range(self._n_ctrl_steps_per_policy):
            for robot in self._env.robots:
                robot.compute_control()
            self._env.step(self._n_sim_steps_per_ctrl)
        if self._datagen_profiler is not None:
            self._datagen_profiler.end("physics_step")

        # Store the action for env 0 for ActionSensors
        self.last_action = actions[0] if actions else None

    def _observe_and_cache(
        self,
    ) -> tuple[
        list[dict[str, Any]], NDArray[float], NDArray[bool], NDArray[bool], list[dict[str, Any]]
    ]:
        """Poll the sensor suite and record the resulting step."""
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("sensor_polling")
        observation, reward, terminated, truncated, info = self.get_and_cache_all_step_information()
        if self._datagen_profiler is not None:
            self._datagen_profiler.end("sensor_polling")

        done = np.logical_or(terminated, truncated)
        self._cumulative_reward += np.where(done, 0, reward)
        self._num_steps_taken += np.where(done, 0, 1)

        # Cache the action for history tracking
        self.action_cache.append(self.last_action)

        return observation, reward, terminated, truncated, info

    def step_chunk(
        self,
        action_chunk: list[dict[str, Any] | list[dict[str, Any]]],
        stop_on_success: bool = False,
    ) -> tuple[
        list[dict[str, Any]], NDArray[float], NDArray[bool], NDArray[bool], list[dict[str, Any]]
    ]:
        """Step a chunk of actions, polling sensors only after the last one.

        An approximation of real-time action chunking: the chunk runs open-loop.
        Could take ``obs_on_action: int = None`` to also return an intermediate
        observation.

        Args:
            action_chunk: Actions to apply in order. Each one is a single action
                dict for single-env mode, or a list of action dicts for batched mode.
            stop_on_success: End the chunk once the success criterion is met.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        if not action_chunk:
            raise ValueError("step_chunk requires at least one action")

        for action in action_chunk[:-1]:
            self._apply_action(action)
            if np.all(self.is_done()) or (stop_on_success and self.judge_success()):
                return self._observe_and_cache()

        return self.step(action_chunk[-1])

    def is_done(self) -> NDArray[bool]:
        return np.logical_or(self.is_terminal(), self.is_timed_out())

    @property
    def env(self) -> BaseMujocoEnv:
        return self._env

    @abstractmethod
    def get_reward(self) -> NDArray[float]:
        raise NotImplementedError

    def is_timed_out(self) -> NDArray[bool]:
        return np.array([self.episode_step_count >= self._task_horizon])

    def is_terminal(self) -> np.ndarray:
        """Check if task is terminal for each environment.

        Terminal if a done action was received, or — when ``terminate_upon_success``
        is enabled in the experiment config — the success criterion is met.
        """
        assert self._env.n_batch == 1, (
            f"Only single-task batches supported. Got env.n_batch={self._env.n_batch}"
        )

        terminal = np.zeros(self._env.n_batch, dtype=bool)

        is_success = False
        if hasattr(self.config, "terminate_upon_success") and self.config.terminate_upon_success:
            is_success = self.judge_success()

        terminal[0] = is_success or self._done_action_received

        return terminal

    @abstractmethod
    def judge_success(self) -> bool:
        raise NotImplementedError

    def get_referral_expressions(self):
        filtered_exprs = {
            k: ObjectManager.thresholded_expression_priority(v)
            for k, v in self.config.task_config.referral_expressions_priority.items()
        }
        expr_probs: dict[str, list[float]] = {
            k: ObjectManager.expression_probs(v).tolist() for k, v in filtered_exprs.items()
        }
        return {
            k: [(expr, prob) for (_, _, expr), prob in zip(filtered_exprs[k], expr_probs[k])]
            for k in filtered_exprs
        }

    def get_info(self) -> list[dict[str, Any]]:
        """
        Override this to add custom metrics.
        In the overriden method, you should still call super().get_metrics() and update it to add your custom metrics.
        """
        return [
            {
                "cumulative_reward": self._cumulative_reward[i],
                "num_steps_taken": self._num_steps_taken[i],
            }
            for i in range(self._env.n_batch)
        ]

    def get_obs_scene(self) -> dict[str, Any]:
        """Get scene-related observations that are constant over the entire trajectory."""

        try:
            task_description = self.get_task_description()
        except KeyError as e:
            log.warning(f"Unable to get task description: {e}")
            task_description = "NOT-SAMPLED"

        obs_scene = {
            "task_type": self.config.task_type,
            "task_description": task_description,
            "policy_dt_ms": self.config.policy_dt_ms,
            "referral_expressions": self.get_referral_expressions(),
        }
        if self._registered_policy is not None:
            from molmo_spaces.policy.base_policy import PlannerPolicy

            # A bit of a hack - why do we need phases in the obs_scene?
            if isinstance(self._registered_policy, PlannerPolicy):
                phases_dict = self._registered_policy.get_all_phases()
                obs_scene["policy_phases"] = phases_dict
            obs_scene.update(self._registered_policy.get_info())

        if self.frozen_config is not None:
            obs_scene["frozen_config"] = self.frozen_config
        else:
            log.warning("Warning: please don't call get_obs_scene before reset()")

        return obs_scene

    def get_history(self) -> dict:
        history = dict(
            observations=self.observation_cache,
            rewards=self.reward_cache,
            terminals=self.terminal_cache,
            truncateds=self.truncated_cache,
            successes=self.success_cache,
            actions=self.action_cache,
        )

        history["obs_scene"] = self.get_obs_scene()

        return history

    def close(self):
        # Clear any MlSpacesObject references
        for attr in list(vars(self).keys()):
            obj = getattr(self, attr, None)
            if isinstance(obj, MlSpacesObjectAbstract):
                setattr(self, attr, None)

        # Clear sensor suite
        if hasattr(self, "_sensor_suite"):
            self._sensor_suite = None

        # Clear environment reference (not closing it as it is owned by the task sampler)
        self._env = None

        if hasattr(self, "renderer") and self.renderer is not None:
            with contextlib.suppress(AttributeError):
                self.renderer.close()

    def __del__(self) -> None:
        """Clean up resources when the task is destroyed."""
        # TODO(all): cleanup?
        self.close()
