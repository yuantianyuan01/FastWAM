import logging
from typing import TYPE_CHECKING

import numpy as np

from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.parallel_kinematics import ParallelKinematics

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig

logger = logging.getLogger(__name__)


class DummyParallelKinematics(ParallelKinematics):
    """
    Dummy parallel kinematics wrapper that sequentially calls :class:`MlSpacesKinematics` internally.
    """

    def __init__(
        self,
        robot_config: "BaseRobotConfig",
        kinematics: MlSpacesKinematics,
    ):
        super().__init__(robot_config)
        self._kinematics = kinematics

    def warmup_ik(self, batch_size: int):
        pass

    def fk(
        self,
        qpos_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
    ) -> list[dict[str, np.ndarray]] | dict[str, np.ndarray]:
        is_batch, batch_size, qpos_dicts, base_poses = self._batchify(qpos_dicts, base_poses)

        ret = []
        for qpos_dict, base_pose in zip(qpos_dicts, base_poses):
            # Ensure base_pose is writable (create copy if read-only)
            base_pose = np.array(base_pose, copy=True)
            ret.append(self._kinematics.fk(qpos_dict, base_pose, rel_to_base))
        return ret if is_batch else ret[0]

    def ik(
        self,
        move_group_id: str,
        poses: np.ndarray,
        unlocked_move_group_ids: list[str] | None,
        q0_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
        dt: float = 1.0,
        max_iter: int = 100,
        success_eps: float = 1e-4,
        **kwargs,
    ):
        is_batch, batch_size, q0_dicts, base_poses, poses = self._batchify(
            q0_dicts, base_poses, poses
        )

        ret = []
        for q0_dict, base_pose, pose in zip(q0_dicts, base_poses, poses):
            # Ensure arrays are writable (create copies if read-only)
            # This fixes "ValueError: buffer source array is read-only" when setting base.pose
            # Use copy=True to ensure we have a writable array
            base_pose = np.array(base_pose, copy=True)
            pose = np.array(pose, copy=True)

            ret.append(
                self._kinematics.ik(
                    move_group_id,
                    pose,
                    unlocked_move_group_ids,
                    q0_dict,
                    base_pose,
                    rel_to_base,
                    eps=success_eps,
                    max_iter=max_iter,
                    dt=dt,
                )
            )
        success = np.array([jp_dict is not None for jp_dict in ret])
        if not np.all(success):
            logger.debug(f"[DummyParallelKinematics] IK failed for indices {np.where(~success)[0]}")
        return ret if is_batch else ret[0]
