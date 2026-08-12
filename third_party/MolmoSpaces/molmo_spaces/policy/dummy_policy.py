from typing import TYPE_CHECKING

import numpy as np

from molmo_spaces.policy.base_policy import BasePolicy

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig


class DummyPolicy(BasePolicy):
    """
    A Dummy Policy that return null actions.
    """

    @property
    def type(self):
        return "dummy"

    def reset(self):
        """
        Reset the policy state. No state to reset for DummyPolicy.
        """
        pass

    def get_action(self, obervation):
        """
        Dummy action to take based on the action space.

        Args:
            info: The current information about the task or environment (not used).

        Returns:
        """
        return dict()


class BrownianMotionPolicy(DummyPolicy):
    """Policy that applies Gaussian noise increments over noop control, resulting in Brownian motion."""

    def __init__(self, config: "MlSpacesExpConfig") -> None:
        super().__init__(config)
        self.std = config.policy_config.std

    def get_action(self, observation) -> dict:
        robot_view = self.task.env.current_robot.robot_view
        action = robot_view.get_noop_ctrl_dict()
        for mg_id, ctrl in action.items():
            action[mg_id] = ctrl + np.random.normal(loc=0.0, scale=self.std, size=ctrl.shape)
        return action
