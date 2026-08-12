import logging
from typing import Any

import gymnasium.spaces as gyms
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.task_configs import PickTaskConfig
from molmo_spaces.env.abstract_sensors import Sensor, SensorSuite
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.sensors import (
    GraspStateSensor,
    ObjectStartPoseSensor,
    get_core_sensors,
)
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom

log = logging.getLogger(__name__)


class PickupObjGoalPoseSensor(Sensor):
    """Sensor for target/end object pose in 7D format (x, y, z, qw, qx, qy, qz)."""

    def __init__(self, uuid: str) -> None:
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get target object pose."""
        assert isinstance(task.config.task_config, PickTaskConfig), (
            "PickupObjGoalPoseSensor requires a PickTaskConfig"
        )
        goal_pose = np.array(task.config.task_config.pickup_obj_goal_pose, dtype=np.float32)
        return goal_pose


class PickTask(BaseMujocoTask):
    """Pick task implementation."""

    def get_task_description(self) -> str:
        pickup_obj_name = self.config.task_config.referral_expressions["pickup_obj_name"]
        return f"Pick up the {pickup_obj_name}"

    def get_task_objects(self, batch_index: int = 0) -> dict[str, str]:
        """Return task objects for pick task."""
        task_objects = super().get_task_objects(batch_index)
        task_config = self.config.task_config

        self.deduplicate_task_objects_name(
            task_config, "pickup_obj_name", task_objects, "pickup_obj"
        )
        return task_objects

    def _create_sensor_suite_from_config(self, config: MlSpacesExpConfig) -> SensorSuite:
        sensors = get_core_sensors(config)

        sensors.extend(
            [
                ObjectStartPoseSensor(
                    object_name=config.task_config.pickup_obj_name, uuid="obj_start"
                ),
                GraspStateSensor(
                    object_name=config.task_config.pickup_obj_name,
                    uuid="grasp_state_pickup_obj",
                ),
                PickupObjGoalPoseSensor(
                    uuid="obj_end",
                ),
            ]
        )

        return SensorSuite(sensors)

    def judge_success(self) -> bool:
        """Judge if the task was successful (for data generation)."""

        if self.config.task_type == "pick":
            return self.get_info()[0]["success"]
        else:
            raise ValueError(f"Invalid action_type {self.config.task_type}")

    def get_reward(self) -> np.ndarray:
        """Calculate reward for each environment in the batch."""
        rewards = np.zeros(self._env.n_batch)

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]

            # Get pickup object using Object class for proper positioning
            pickup_obj = MlSpacesObject(
                data=data, object_name=self.config.task_config.pickup_obj_name
            )

            # reward is height above starting positions
            # consider judge_success threshold when chaninging this
            lift_height = pickup_obj.position[2] - self.config.task_config.pickup_obj_start_pose[2]
            pickup_obj_supporting_geom = get_supporting_geom(data, pickup_obj.body_id)
            robot_geoms = descendant_geoms(
                self.env._mj_model, self.env.current_robot.robot_view.base.root_body_id
            )
            object_lifted = (
                pickup_obj_supporting_geom is None or pickup_obj_supporting_geom in robot_geoms
            )
            reward = int(object_lifted) * lift_height
            reward = np.clip(reward, 0.0, 1000.0)
            rewards[i] = reward

        return rewards

    def get_info(self) -> list[dict[str, Any]]:
        """Get additional metrics for each environment."""
        metrics = []

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]

            # Get pickup object using Object class for proper positioning
            pickup_obj = MlSpacesObject(
                data=data, object_name=self.config.task_config.pickup_obj_name
            )

            place_target_pos = self.config.task_config.pickup_obj_goal_pose[:3]
            place_target_quat = self.config.task_config.pickup_obj_goal_pose[3:7]

            # Calculate errors
            pos_error = np.linalg.norm(pickup_obj.position - place_target_pos)
            pickup_rot = R.from_quat(pickup_obj.quat, scalar_first=True)
            target_rot = R.from_quat(place_target_quat, scalar_first=True)
            rot_error = (pickup_rot.inv() * target_rot).magnitude()

            # Would like to cache this, but no easy way atm
            lift_height = pickup_obj.position[2] - self.config.task_config.pickup_obj_start_pose[2]

            # Option 1: go via descendant geoms and check if all contacts are with robot geoms (didn't work)
            # obj_geoms = get_supporting_geom(data, pickup_obj.body_id)
            # robot_geoms = descendant_geoms(self.env._mj_model, self.env.current_robot.robot_view.base.root_body_id)
            # gripper_root_body_id = self.env.current_robot.robot_view.get_gripper("gripper").root_body_id
            # gripper_geoms = descendant_geoms(self.env._mj_model, gripper_root_body_id)
            # for c in data.contact:
            #     if (c.geom[0] in obj_geoms) ^ (c.geom[1] in obj_geoms):
            #         other_geom_id = c.geom[1] if c.geom[0] in obj_geoms else c.geom[0]
            #         if other_geom_id in robot_geoms:
            #             only_robot_collision = True
            #         else:
            #             only_robot_collision = False
            #             break

            # Option 2: go via root body and check if all contacts are with robot geoms
            # Check if object collides only with robot geoms
            robot_collision = False
            non_robot_collision = False
            for c in data.contact:
                root_body1 = data.model.body_rootid[data.model.geom_bodyid[c.geom1]]
                root_body2 = data.model.body_rootid[data.model.geom_bodyid[c.geom2]]
                if (root_body1 == pickup_obj.body_id) ^ (root_body2 == pickup_obj.body_id):
                    other_root_body = root_body1 if root_body1 != pickup_obj.body_id else root_body2
                    if other_root_body == self.env.current_robot.robot_view.base.root_body_id:
                        robot_collision = True
                    else:
                        non_robot_collision = True
                        break  # no need to keep checking if we already know there's a non-robot collision

            only_robot_collision = robot_collision and not non_robot_collision

            # Success check
            success = (
                only_robot_collision and lift_height >= self.config.task_config.succ_pos_threshold
                # and rot_error < self.config.task_config.succ_rot_threshold
            )

            metrics.append(
                {
                    "position_error": pos_error,
                    "rotation_error": rot_error,
                    "success": success,
                    "episode_step": self.episode_step_count,
                }
            )

        return metrics

    def get_obs_scene(self):
        """
        This is for observations that are constant over all time steps of an env.
        """
        obs_scene = super().get_obs_scene()
        text = self.config.task_type + " " + self.config.task_config.pickup_obj_name
        obs_extra = dict(text=text, object_name=self.config.task_config.pickup_obj_name)
        obs_scene.update(obs_extra)

        return obs_scene
