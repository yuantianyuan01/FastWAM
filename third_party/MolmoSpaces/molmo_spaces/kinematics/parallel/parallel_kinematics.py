from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig


class ParallelKinematics(ABC):
    def __init__(self, robot_config: "BaseRobotConfig"):
        self._robot_config = robot_config

    @abstractmethod
    def warmup_ik(self, batch_size: int):
        """
        Incur startup costs up-front to speed up subsequent calls to IK.
        The warmed up batch size may need to match that of the subsequent calls to IK.

        Args:
            batch_size: The batch size to warmup.
        """
        pass

    def _batchify(
        self,
        qpos_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        poses: np.ndarray | None = None,
    ):
        if poses is not None and poses.ndim > 2:
            batch_size = poses.shape[0]
            is_batch = True
        elif isinstance(qpos_dicts, list):
            batch_size = len(qpos_dicts)
            is_batch = True
        elif base_poses.ndim > 2:
            batch_size = base_poses.shape[0]
            is_batch = True
        else:
            batch_size = 1
            is_batch = False

        if poses is not None:
            poses = (
                poses
                if poses.ndim == 3
                else np.broadcast_to(poses[None], (batch_size, *poses.shape))
            )
        qpos_dicts = qpos_dicts if isinstance(qpos_dicts, list) else [qpos_dicts] * batch_size
        base_poses = (
            base_poses
            if base_poses.ndim == 3
            else np.broadcast_to(base_poses[None], (batch_size, *base_poses.shape))
        )

        if poses is not None:
            return is_batch, batch_size, qpos_dicts, base_poses, poses
        else:
            return is_batch, batch_size, qpos_dicts, base_poses

    @abstractmethod
    def fk(
        self,
        qpos_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
    ) -> list[dict[str, np.ndarray]] | dict[str, np.ndarray]:
        """
        Compute forward kinematics for all move groups.

        Args:
            qpos_dicts: The joint positions.
            base_pose: The base pose(s) of the robots. Shape: (batch_size, 4, 4) or (4, 4)
            rel_to_base: Whether the returned pose(s) should be relative to the base frame.

        Returns:
            A list of qpos dictionaries for each robot in the batch, or a single qpos dictionary if unbatched.
        """
        raise NotImplementedError

    @abstractmethod
    def ik(
        self,
        move_group_id: str,
        poses: np.ndarray,
        unlocked_move_group_ids: list[str] | None,
        q0_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, np.ndarray] | None] | dict[str, np.ndarray] | None:
        """
        Finds joint positions that would place the end-effector at the target pose.

        Args:
            pose: The target pose(s) to reach. Shape: (batch_size, 4, 4) or (4, 4)
            q0_dicts: The initial joint positions.
            base_pose: The base pose(s) of the robots. Shape: (batch_size, 4, 4) or (4, 4)
            rel_to_base: Whether the pose(s) are relative to the base frame.
            **kwargs: Additional keyword arguments for the IK solver, defined by the concrete implementation.

        Returns:
            A list of qpos dictionaries for each robot in the batch, or a single qpos dictionary if unbatched.
                If the solver fails to converge for a given robot, the corresponding qpos dictionary is None.
        """
        raise NotImplementedError
