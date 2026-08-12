from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.sensors import (
    GraspStateSensor,
    ObjectStartPoseSensor,
    get_core_sensors,
)
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.mujoco_scene_utils import is_object_supported_by_body
from molmo_spaces.utils.pose import pos_quat_to_pose_mat


class PickAndPlaceTask(BaseMujocoTask):
    """Franka pick-and-place task implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._supported_rel_poses: dict[int, list[np.ndarray]] = {}

    def get_task_description(self) -> str:
        pickup_name = self.config.task_config.referral_expressions["pickup_name"]
        place_name = self.config.task_config.referral_expressions["place_name"]
        return f"Pick up the {pickup_name} and place it in or on the {place_name}"

    def get_task_objects(self, batch_index: int = 0) -> dict[str, str]:
        """Return task objects for pick-and-place task."""
        task_objects = super().get_task_objects(batch_index)
        task_config = self.config.task_config

        self.deduplicate_task_objects_name(
            task_config, "pickup_obj_name", task_objects, "pickup_obj"
        )
        self.deduplicate_task_objects_name(
            task_config, "place_receptacle_name", task_objects, "place_receptacle"
        )

        return task_objects

    def _create_sensor_suite_from_config(self, config: MlSpacesExpConfig) -> SensorSuite:
        sensors = get_core_sensors(config)
        assert config.task_config.place_receptacle_name, "No place receptacle name provided"

        sensors.extend(
            [
                ObjectStartPoseSensor(
                    object_name=config.task_config.pickup_obj_name, uuid="obj_start"
                ),
                GraspStateSensor(
                    object_name=config.task_config.pickup_obj_name,
                    uuid="grasp_state_pickup_obj",
                ),
                GraspStateSensor(
                    object_name=config.task_config.place_receptacle_name,
                    uuid="grasp_state_place_receptacle",
                ),
            ]
        )

        return SensorSuite(sensors)

    def reset(self):
        result = super().reset()
        self._supported_rel_poses.clear()
        return result

    def judge_success(self) -> bool:
        """Judge if the task was successful (for data generation)."""
        return self.get_info()[0]["success"]

    def get_reward(self) -> np.ndarray:
        """Calculate reward for each environment in the batch."""
        rewards = np.zeros(self._env.n_batch)

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]
            task_config = self.config.task_config
            assert isinstance(task_config, PickAndPlaceTaskConfig)
            pickup_obj = create_mlspaces_body(data, task_config.pickup_obj_name)
            place_receptacle = create_mlspaces_body(data, task_config.place_receptacle_name)

            place_receptacle_aabb_center, _ = body_aabb(data.model, data, place_receptacle.body_id)

            pos_err = np.linalg.norm(pickup_obj.position - place_receptacle_aabb_center)
            rewards[i] = np.exp(-pos_err)

        return rewards

    def get_info(self) -> list[dict[str, Any]]:
        """Get additional metrics for each environment."""
        metrics = []

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]
            task_config = self.config.task_config
            assert isinstance(task_config, PickAndPlaceTaskConfig)
            pickup_obj = create_mlspaces_body(data, task_config.pickup_obj_name)
            place_receptacle = create_mlspaces_body(data, task_config.place_receptacle_name)

            pickup_obj_aabb = body_aabb(data.model, data, pickup_obj.body_id)
            pickup_obj_aabb_min = pickup_obj_aabb[0] - pickup_obj_aabb[1] / 2
            pickup_obj_aabb_max = pickup_obj_aabb[0] + pickup_obj_aabb[1] / 2
            place_receptacle_aabb_center, _ = body_aabb(data.model, data, place_receptacle.body_id)

            # the pos err is the distance between the receptacle center and the pickup object aabb
            # pos err is 0 when the pickup object contains the receptacle center point
            pos_err = np.linalg.norm(
                np.maximum(0, pickup_obj_aabb_min - place_receptacle_aabb_center)
                + np.maximum(0, place_receptacle_aabb_center - pickup_obj_aabb_max)
            )

            # Success check
            # is the pickup object supported by the receptacle?

            # based on contact force only:
            supported_by_receptacle = is_object_supported_by_body(
                data,
                pickup_obj.body_id,
                place_receptacle.body_id,
                frac_weight_threshold=task_config.receptacle_supported_weight_frac,
            )

            if not supported_by_receptacle:
                # Use heuristic
                om = self._env.object_managers[i]
                objects_on_receptacle = om.objects_on_receptacle(
                    [om.get_object_by_name(task_config.pickup_obj_name)],
                    om.get_object_by_name(task_config.place_receptacle_name).geom_ids,
                )
                names_on_receptacle = {obj.name for obj in objects_on_receptacle}
                supported_by_receptacle = task_config.pickup_obj_name in names_on_receptacle

            # Relative-pose carry-forward: track the object's pose in the
            # receptacle's frame. When contact-based checks detect support,
            # snapshot this relative pose. When they fail (e.g. lightweight
            # objects at equilibrium losing contact forces), compare against
            # the snapshot — if the relative pose hasn't drifted, the object
            # is still where it was when support was last confirmed.
            rel_pose = np.linalg.solve(place_receptacle.pose, pickup_obj.pose)

            carried_forward = False
            carry_forward_pos_diff = float("inf")
            carry_forward_rot_diff = float("inf")
            if supported_by_receptacle:
                self._supported_rel_poses.setdefault(i, []).append(rel_pose.copy())
                carry_forward_pos_diff = 0.0
                carry_forward_rot_diff = 0.0
            if i in self._supported_rel_poses:
                # Track nearest-by-position for diagnostics, and separately
                # check if any cached pose satisfies both thresholds jointly
                # (greedy nearest-by-position can miss a slightly farther pose
                # that satisfies rotation while the nearest one doesn't).
                best_pos_diff = float("inf")
                best_rot_diff = float("inf")
                match_found = False
                for stored in self._supported_rel_poses[i]:
                    pd = float(np.linalg.norm(rel_pose[:3, 3] - stored[:3, 3]))
                    rd = float(R.from_matrix(rel_pose[:3, :3] @ stored[:3, :3].T).magnitude())
                    if pd < best_pos_diff:
                        best_pos_diff = pd
                        best_rot_diff = rd
                    if (
                        pd <= task_config.carry_forward_rel_pos_threshold
                        and rd <= task_config.carry_forward_rel_rot_threshold
                    ):
                        match_found = True
                carry_forward_pos_diff = best_pos_diff
                carry_forward_rot_diff = best_rot_diff
                if not supported_by_receptacle and match_found:
                    supported_by_receptacle = True
                    carried_forward = True

            # has the place receptacle moved too much?
            start_pose = pos_quat_to_pose_mat(
                task_config.place_receptacle_start_pose[0:3],
                task_config.place_receptacle_start_pose[3:7],
            )
            curr_pose = place_receptacle.pose
            displacement = np.linalg.inv(start_pose) @ curr_pose
            pos_displacement = displacement[:3, 3]
            rot_displacement = R.from_matrix(displacement[:3, :3]).magnitude()

            # Tilt-only rotation displacement: compute the start→current
            # rotation in world frame and see how much it rotates [0,0,1].
            r_diff_world = curr_pose[:3, :3] @ start_pose[:3, :3].T
            diff_up = r_diff_world @ np.array([0.0, 0.0, 1.0])
            cos_tilt = np.clip(diff_up[2], -1.0, 1.0)
            tilt_displacement = np.arccos(cos_tilt)

            # Check that the robot has released the object (no robot-object contact).
            robot_root_body_id = self._env.current_robot.robot_view.base.root_body_id
            robot_contact = False
            for c in data.contact:
                root_body1 = data.model.body_rootid[data.model.geom_bodyid[c.geom1]]
                root_body2 = data.model.body_rootid[data.model.geom_bodyid[c.geom2]]
                if (root_body1 == pickup_obj.body_id) ^ (root_body2 == pickup_obj.body_id):
                    other_root_body = root_body1 if root_body1 != pickup_obj.body_id else root_body2
                    if other_root_body == robot_root_body_id:
                        robot_contact = True
                        break

            # Momentary check: re-evaluated from current poses each step, so a
            # past transient displacement does not permanently prevent success.
            # max_place_receptacle_rot_displacement is used as the tilt threshold
            # (yaw-excluded); see tilt_displacement computation above.
            success = (
                supported_by_receptacle
                and not robot_contact
                and np.linalg.norm(pos_displacement)
                <= task_config.max_place_receptacle_pos_displacement
                and tilt_displacement <= task_config.max_place_receptacle_rot_displacement
            )

            metrics.append(
                {
                    "position_error": pos_err,
                    "success": success,
                    "supported_by_receptacle": supported_by_receptacle,
                    "supported_by_carry_forward": carried_forward,
                    "carry_forward_pos_diff": carry_forward_pos_diff,
                    "carry_forward_rot_diff": carry_forward_rot_diff,
                    "robot_contact": robot_contact,
                    "receptacle_pos_displacement": pos_displacement,
                    "receptacle_rot_displacement": rot_displacement,
                    "receptacle_tilt_displacement": tilt_displacement,
                    "episode_step": self.episode_step_count,
                }
            )

        return metrics

    def get_obs_scene(self):
        """
        This is for observations that are constant over all time steps of an env.
        """
        obs_scene = super().get_obs_scene()
        text = f"{self.config.task_type} {self.config.task_config.pickup_obj_name} {self.config.task_config.place_receptacle_name}"
        obs_extra = dict(
            text=text,
            object_name=self.config.task_config.pickup_obj_name,
            place_receptacle_name=self.config.task_config.place_receptacle_name,
        )
        obs_scene.update(obs_extra)
        return obs_scene
