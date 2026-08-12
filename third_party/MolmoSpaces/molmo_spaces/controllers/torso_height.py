import numpy as np

from molmo_spaces.controllers.abstract import AbstractPositionController
from molmo_spaces.robots.robot_views.abstract import MoveGroup


class TorsoHeightJointPosController(AbstractPositionController):
    """Controls the RBY1 torso via a single height scalar.

    Accepts a 1-element command ``[h]``; ``compute_ctrl_inputs()`` returns the
    full 6-DOF joint position array produced by the constraint
    torso_1 = torso_3 = h, torso_2 = -2*h.
    """

    @staticmethod
    def height_to_joints(height: float) -> np.ndarray:
        """Return the 6-DOF torso joint positions for a given height scalar.

        torso[1] = height, torso[2] = -2*height, torso[3] = height.
        Indices 0, 4, 5 are left at zero.
        """
        joints = np.zeros(6)
        joints[1] = height
        joints[2] = -2.0 * height
        joints[3] = height
        return joints

    @staticmethod
    def joints_to_height(joints: np.ndarray) -> float:
        """Recover the height scalar from a 6-DOF torso joint position array."""
        return float(joints[1])

    def __init__(self, robot_move_group: MoveGroup, max_height: float = 0.738) -> None:
        super().__init__(robot_move_group)
        self.max_height = max_height
        self._stationary = True
        self._target_height: float = self.joints_to_height(robot_move_group.joint_pos)

    @property
    def target(self) -> np.ndarray:
        return np.array([self._target_height])

    @property
    def target_pos(self) -> np.ndarray:
        return self.target.copy()

    @property
    def stationary(self) -> bool:
        return self._stationary

    def set_target(self, target: np.ndarray) -> None:
        self._stationary = False
        self._target_height = float(np.clip(target[0], 0.0, self.max_height))

    def set_to_stationary(self) -> None:
        self._stationary = True
        self._target_height = self.joints_to_height(self.robot_move_group.joint_pos)

    def hold_at_height(self, height: float) -> None:
        """Set the controller to hold at a specific height without reading current joints.

        Use this to lock the torso at a desired target after a height adjustment is
        complete, so that subsequent steps without an explicit torso command will
        maintain this height rather than drifting to the current joint position.
        """
        self._stationary = True
        self._target_height = float(np.clip(height, 0.0, self.max_height))

    def compute_ctrl_inputs(self) -> np.ndarray:
        """Return the 6-DOF torso joint positions for the current target height."""
        joints = self.height_to_joints(self._target_height)
        return joints

    def reset(self) -> None:
        self.set_to_stationary()
