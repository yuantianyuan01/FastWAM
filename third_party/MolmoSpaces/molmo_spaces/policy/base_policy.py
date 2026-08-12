"""Abstract Policy class to run a policy in the environment for data collection or evaluation.
This class is designed to be subclassed and implemented with specific policies.
For example, the policy can be a planner, a human teleop interface, a reinforcement learning agent, or any other type of agent
that can interact with the environment to collect data.
"""

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

from mujoco import MjSpec

from molmo_spaces.tasks.task import BaseMujocoTask

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.env.abstract_sensors import Sensor

NONE_PHASE = -1


class BasePolicy(ABC):
    """
    Abstract base class for policies.

    This class provides a template for policies that can be used to interact with an environment
    for the purpose of data collection or evaluation. It declares methods that should be implemented by any concrete policy class.
    """

    def __init__(self, config: "MlSpacesExpConfig", task: BaseMujocoTask | None = None) -> None:
        self.config = config
        self.task = task

        if self.config.policy_config.force_enable_depth:
            for camera in self.config.camera_config.cameras:
                if not camera.record_depth:
                    raise ValueError(
                        f"Camera {camera.name} must record depth when force_enable_depth is True"
                    )

    @abstractmethod
    def reset(self):
        """
        Reset the policy's internal state.

        This method should be implemented by each subclass to reset the policy's internal state.
        It is typically called at the beginning of each episode or task.
        """
        pass

    @abstractmethod
    def get_action(self, observation):
        """
        Decide on the action to take based on the current observation of the or environment.
        Information could be observations, goals in the case of an rl_agent, or it could be the full
        environment state in the case of a planner.

        Args:
            observation: The current information about the task or environment.

        Returns:
            The action to take in response to the information.
        """
        pass

    def get_action_chunk(self, observation: Any) -> list[dict[str, Any]] | None:
        """Return the actions to run before the next observation is needed, or None.

        A single action is a ``dict[str, Any]`` in single-env mode and a
        ``list[dict[str, Any]]`` (one entry per env) in batched mode. A batched
        action and a chunk of actions are therefore both lists and cannot be told
        apart by type, so any variable holding a chunk must say so in its name
        rather than leaving the distinction to be inferred.

        Returns None if this policy has no chunk to offer, which keeps the caller
        on its per-step path.
        """
        return None

    @staticmethod
    def add_auxiliary_objects(config: "MlSpacesExpConfig", spec: MjSpec) -> None:
        """
        Add auxiliary objects to the scene that might be required for the policy.

        Args:
            config: The configuration for the policy.
            spec: The experiment configuration.
        """
        return None

    def get_info(self) -> dict:
        """
        Get additional information from the policy. Called after episode ended. This method can be
        overridden by subclasses to provide extra information about the policy's state.
        Must be json serializable.

        Returns:
            A dictionary containing additional information about the policy.
        """
        return {}

    def create_policy_sensors(self) -> list["Sensor"]:
        """
        Create a list of policy-specific sensors.
        """
        return []


PolicyFactory: TypeAlias = Callable[..., BasePolicy]
"""
Factory function with signature ``Callable[[MlSpacesExpConfig, BaseMujocoTask | None], BasePolicy]``.
To avoid forward-reference resolution issues with Pydantic, the type is relaxed to Callable[..., BasePolicy].
"""


class PlannerPolicy(BasePolicy):
    @abstractmethod
    def planners(self) -> None:
        """Abstract property representing the list or dict of planner instances."""
        pass

    @abstractmethod
    def get_phase(self) -> str:
        """
        Returns:
            The current policy phase
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_phases(self) -> dict[str | int]:
        """
        Returns:
            A dictionary of all possible policy phases
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def retry_count(self) -> int:
        """
        The number of retries the policy has taken.
        """
        raise NotImplementedError

    def create_policy_sensors(self) -> list["Sensor"]:
        from molmo_spaces.env.sensors import PolicyNumRetriesSensor, PolicyPhaseSensor

        return super().create_policy_sensors() + [
            PolicyPhaseSensor(uuid="policy_phase"),
            PolicyNumRetriesSensor(uuid="policy_num_retries"),
        ]


class InferencePolicy(BasePolicy):
    def get_action(self, observation):
        model_input = self.obs_to_model_input(observation)
        model_output = self.inference_model(model_input)
        action = self.model_output_to_action(model_output)
        return action

    @abstractmethod
    def prepare_model(self, model_name: str):
        raise NotImplementedError

    @abstractmethod
    def obs_to_model_input(self, obs):
        raise NotImplementedError

    @abstractmethod
    def model_output_to_action(self, model_output):
        raise NotImplementedError

    @abstractmethod
    def inference_model(self, model_input):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def render(self, obs):
        """
        Not required to be implemented by subclasses.
        Defaults to noop.
        """
        pass

    def get_info(self) -> dict:
        info = super().get_info()
        info["task_type"] = self.config.task_type
        info["timestamp"] = time.time()
        return info


class StatefulPolicy(BasePolicy):
    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state):
        raise NotImplementedError
