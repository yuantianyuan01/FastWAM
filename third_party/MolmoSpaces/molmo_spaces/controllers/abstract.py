import abc

import numpy as np

from molmo_spaces.robots.robot_views.abstract import MoveGroup


class Controller:
    """
    Implementation of a simple, non-blocking controller for the MolmoSpaces environments.
    This controller is meant to be simple, provide a control law based on stateful targets set by the user,
    and return the control inputs needed to achieve those targets.

    The controller DOES NOT step the simulation, it only computes the control inputs based on the
    current state and the targets set by the user.
    """

    def __init__(self, robot_move_group: MoveGroup) -> None:
        """
        Args:
            robot_move_group: The move group of the robot to be controlled
        """
        self.robot_move_group = robot_move_group

    @property
    @abc.abstractmethod
    def target(self):
        """The target state of the controller, set by the user of the controller."""
        pass

    @property
    @abc.abstractmethod
    def stationary(self):
        """Whether the controller is set to hold the robot stationary, for e.g.
        when the robot has to be stopped at a certain position and not drift."""

    @abc.abstractmethod
    def set_target(self, target):
        """
        Set the target state of the controller.

        Args:
            target: The target state to be set, e.g. a joint position, base velocity, or any other state.
        """
        pass

    def set_to_stationary(self) -> None:
        """
        This method sets the robot to stationary mode and computes the targets to hold the
        robot stationary.
        This is useful when the robot needs to be stopped at a certain position and not drift.
        NOTE: If the controller is already stationary, it does not change the target.
        """
        pass

    @abc.abstractmethod
    def compute_ctrl_inputs(self):
        """
        Compute the control inputs based on the current state and the target set by the user.

        Returns:
            The control inputs to be applied to the robot actuators, eg. positions, torques etc.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset the controller to its initial state, clearing any internal state or targets."""
        pass


class AbstractPositionController(Controller):
    """
    Base class for (absolute or relative) position controllers.
    """

    @property
    @abc.abstractmethod
    def target_pos(self) -> np.ndarray:
        """
        The target absolute position of the controller, set by the user of the controller.
        """
        pass
