from typing import TYPE_CHECKING

from molmo_spaces.policy.base_policy import BasePolicy

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig


class RandomPolicy(BasePolicy):
    """
    A Random Policy that selects actions randomly.
    """

    @property
    def type(self) -> str:
        return "random"

    def __init__(self, config: "MlSpacesExpConfig") -> None:
        raise NotImplementedError  # TODO(snehal): please fix
        self.config = config
        self.action_space = action_space

    def reset(self) -> None:
        """
        Reset the policy state. No state to reset for RandomPolicy.
        """
        pass

    def get_action(self, obervation):
        """
        Decide on a random action to take based on the action space.

        Args:
            observation: The current observation about the task or environment (not used).

        Returns:
            A random action from the action space.
        """
        return self.action_space.sample()
