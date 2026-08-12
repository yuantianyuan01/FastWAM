import numpy as np

from molmo_spaces.controllers.abstract import AbstractPositionController
from molmo_spaces.robots.robot_views.abstract import MoveGroup


class JointRelPosController(AbstractPositionController):
    """
    Generic joint position controller for a robot.
    Computes delta joint position targets for the joint's actuators, and clips to control limits.
    """

    def __init__(self, robot_move_group: MoveGroup) -> None:
        super().__init__(robot_move_group)

        self.ctrl_dim = robot_move_group.n_actuators
        self.ctrl_range = robot_move_group.ctrl_limits

        self._stationary = True
        self.target = np.zeros_like(
            self.robot_move_group.noop_ctrl
        )  # Initial pos target is 0 relative to current position
        self.desired_joint_positions = self.robot_move_group.noop_ctrl.copy()
        self.reset()

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value: np.ndarray) -> None:
        self._target = value

    @property
    def target_pos(self) -> np.ndarray:
        return self.desired_joint_positions.copy()

    @property
    def stationary(self):
        """Returns whether the controller is in stationary mode"""
        return self._stationary

    def set_target(self, target_rel_joint_positions) -> None:
        """Set the target joint positions for the controller.
        NOTE: For this controller we assume that the controller will be set to
        stationary every time before the relative target has to be computed"""

        # Case to avoid drift: if target is close to zero, stay stationary
        if np.linalg.norm(target_rel_joint_positions) < 1e-6:
            if not self._stationary:  # important to avoid re-setting to noop ctrl and drifting
                self.set_to_stationary()
        else:
            self._stationary = False
            self.target = target_rel_joint_positions
            self.desired_joint_positions = (
                self.robot_move_group.noop_ctrl + target_rel_joint_positions
            )
            self.desired_joint_positions = np.clip(
                self.desired_joint_positions, self.ctrl_range[:, 0], self.ctrl_range[:, 1]
            )

    def set_to_stationary(self) -> None:
        """
        This method sets the robot to stationary mode and computes the targets to hold the
        robot stationary.
        This is useful when the robot needs to be stopped at a certain position and not drift.
        """
        # Set to stationary mode and set target to current joint positions to hold them stationary
        self._stationary = True
        self.target = np.zeros_like(self.robot_move_group.noop_ctrl)
        self.desired_joint_positions = self.robot_move_group.noop_ctrl.copy()

    def compute_ctrl_inputs(self):
        """
        Compute the control inputs based on the current state and the target set by the user.

        Returns:
            The control inputs to be applied to the robot actuators, in this case: positions
        """
        return self.desired_joint_positions.copy()

    def reset(self) -> None:
        """Reset the controller to its initial state, clearing any internal state or targets"""
        self.set_to_stationary()  # Explicit reset to stationary mode
