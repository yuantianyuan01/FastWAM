import logging

import numpy as np

from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
    JointMoveSegment,
    JointMoveSequence,
    NoopAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.utils.grasp_sample import select_grasp_pose
from molmo_spaces.utils.grasps import get_pickup_grasps
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

log = logging.getLogger(__name__)


class PickAndPlacePlannerPolicy(BaseObjectManipulationPlannerPolicy):
    def _compute_trajectory(self) -> list[ActionPrimitive]:
        robot_view = self.task.env.current_robot.robot_view
        target_poses = self._compute_target_poses()

        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        start_ee_pose = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world

        return [
            GripperAction(robot_view, True, 0.0),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                move_segments=[
                    TCPMoveSegment(
                        name="pregrasp",
                        start_pose=start_ee_pose,
                        end_pose=target_poses["pregrasp"],
                        speed=self.policy_config.speed_fast,
                    ),
                    TCPMoveSegment(
                        name="grasp",
                        start_pose=target_poses["pregrasp"],
                        end_pose=target_poses["grasp"],
                        speed=self.policy_config.speed_slow,
                    ),
                ],
            ),
            GripperAction(robot_view, False, self.policy_config.gripper_close_duration),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                is_holding_object=True,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                move_segments=[
                    TCPMoveSegment(
                        name="lift",
                        start_pose=target_poses["grasp"],
                        end_pose=target_poses["lift"],
                        speed=self.policy_config.speed_slow,
                    ),
                    TCPMoveSegment(
                        name="preplace",
                        start_pose=target_poses["lift"],
                        end_pose=target_poses["preplace"],
                        speed=self.policy_config.speed_fast,
                    ),
                    TCPMoveSegment(
                        name="place",
                        start_pose=target_poses["preplace"],
                        end_pose=target_poses["place"],
                        speed=self.policy_config.speed_slow,
                    ),
                ],
            ),
            GripperAction(robot_view, True, self.policy_config.gripper_open_duration),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                move_segments=[
                    TCPMoveSegment(
                        name="retreat",
                        start_pose=target_poses["place"],
                        end_pose=target_poses["postplace"],
                        speed=self.policy_config.speed_fast,
                    )
                ],
            ),
            JointMoveSequence(
                robot_view,
                self.policy_config.move_settle_time,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                move_segments=[
                    JointMoveSegment(
                        name="go_home",
                        start_qpos=None,
                        end_qpos=self.config.robot_config.init_qpos,
                        duration_s=4.0,
                    )
                ],
            ),
            NoopAction(robot_view, 2.0),
        ]

    def _get_grasp_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
        robot_view,
        task_config: PickAndPlaceTaskConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        randomize_pregrasp = False
        if randomize_pregrasp:
            # Random height variations
            pregrasp_height_offset = np.random.uniform(
                -self.policy_config.pregrasp_height_noise,
                self.policy_config.pregrasp_height_noise,
            )
            postgrasp_height_offset = np.random.uniform(
                -self.policy_config.postgrasp_height_noise,
                self.policy_config.postgrasp_height_noise,
            )
        else:
            pregrasp_height_offset = 0.0
            postgrasp_height_offset = 0.0

        pregrasp_pose = grasp_pose_world.copy()
        # Pregrasp pose - above the pickup object with randomization
        pregrasp_pose[:3, 3] -= (
            self.policy_config.pregrasp_z_offset + pregrasp_height_offset
        ) * pregrasp_pose[:3, 2]

        log.debug(f"  - obj_start (p): {pickup_obj.position}")
        log.debug(f"  - obj_start (t): {task_config.pickup_obj_start_pose}")
        log.debug(f"  - obj_end (t): {task_config.pickup_obj_goal_pose}")
        log.debug(f"  - Pregrasp position: {pregrasp_pose[:3, 3]}")

        if not self.check_feasible_ik(pregrasp_pose):
            log.debug("  - ❌ IK FAILED for pregrasp pose!")
            log.debug(f"  - Pregrasp position: {pregrasp_pose[:3, 3]}")
            log.debug(f"  - Robot base: {robot_view.base.pose[:3, 3]}")
            log.debug(
                f"  - Height difference: {pregrasp_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
            )
            raise ValueError("IK failed for pregrasp pose")

        log.debug(f"  - Grasp pose position: {grasp_pose_world[:3, 3]}")
        log.debug(
            f"  - Grasp height above robot base: {grasp_pose_world[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
        )
        if not self.check_feasible_ik(grasp_pose_world):
            log.debug("  - ❌ IK FAILED for grasp pose!")
            log.debug(f"  - Grasp position: {grasp_pose_world[:3, 3]}")
            log.debug(f"  - Robot base: {robot_view.base.pose[:3, 3]}")
            log.debug(
                f"  - Height difference: {grasp_pose_world[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
            )
            raise ValueError("IK failed for grasp pose")

        # Lift pose - above grasp position
        place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, place_receptacle.object_id
        )
        receptacle_top_z = place_receptacle_aabb_center[2] + place_receptacle_aabb_size[2] / 2
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
        )
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - pickup_obj_bottom_z, 0.0)
        lift_pose = grasp_pose_world.copy()
        lift_pose[2, 3] = (
            receptacle_top_z
            + pickup_obj_clearance_offset
            + self.policy_config.place_z_offset
            + postgrasp_height_offset
        )

        if not self.check_feasible_ik(lift_pose):
            log.debug("  - ❌ IK FAILED for lift pose!")
            raise ValueError("IK failed for lift pose")

        return pregrasp_pose, grasp_pose_world, lift_pose

    def _get_placement_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, place_receptacle.object_id
        )
        receptacle_top_z = place_receptacle_aabb_center[2] + place_receptacle_aabb_size[2] / 2
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
        )
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - pickup_obj_bottom_z, 0.0)

        preplace_pose = grasp_pose_world.copy()
        preplace_pose[:2, 3] = place_receptacle.position[:2]
        preplace_pose[2, 3] = (
            receptacle_top_z + pickup_obj_clearance_offset + self.policy_config.place_z_offset
        )
        # offset the EE to ensure the pickup object is in the middle of the receptacle
        preplace_pose[:3, 3] += grasp_pose_world[:3, 3] - pickup_obj.position
        if not self.check_feasible_ik(preplace_pose):
            log.debug("  - ❌ IK FAILED for preplace pose!")
            raise ValueError("IK failed for preplace pose")

        place_pose = preplace_pose.copy()
        place_pose[2, 3] = receptacle_top_z + pickup_obj_clearance_offset
        if not self.check_feasible_ik(place_pose):
            log.debug("  - ❌ IK FAILED for place pose!")
            raise ValueError("IK failed for place pose")

        postplace_pose = place_pose.copy()
        postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]

        return preplace_pose, place_pose, postplace_pose

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        task_config = self.config.task_config
        assert isinstance(task_config, PickAndPlaceTaskConfig)
        target_poses = {}

        robot_view = self.task.env.current_robot.robot_view
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj: MlSpacesObject = om.get_object_by_name(task_config.pickup_obj_name)
        place_receptacle: MlSpacesObject = om.get_object_by_name(task_config.place_receptacle_name)

        candidate_grasps = get_pickup_grasps(
            self.task.env, pickup_obj, grasp_libraries=self.policy_config.grasp_libraries
        )
        grasp_pose_world = select_grasp_pose(
            self.task.env,
            candidate_grasps,
            pickup_obj.pose,
            check_collision=self.policy_config.filter_colliding_grasps,
            n_collision_checks=self.policy_config.grasp_collision_max_grasps,
            collision_batch_size=self.policy_config.grasp_collision_batch_size,
            check_ik=self.policy_config.filter_feasible_grasps,
            n_ik_checks=self.policy_config.grasp_feasibility_max_grasps,
            ik_batch_size=self.policy_config.grasp_feasibility_batch_size,
            pos_cost_weight=self.policy_config.grasp_pos_cost_weight,
            rot_cost_weight=self.policy_config.grasp_rot_cost_weight,
            vertical_cost_weight=self.policy_config.grasp_vertical_cost_weight,
            com_dist_cost_weight=self.policy_config.grasp_com_dist_cost_weight,
        )

        pregrasp_pose, grasp_pose, lift_pose = self._get_grasp_poses(
            grasp_pose_world=grasp_pose_world,
            pickup_obj=pickup_obj,
            place_receptacle=place_receptacle,
            robot_view=robot_view,
            task_config=task_config,
        )
        target_poses["pregrasp"] = pregrasp_pose
        target_poses["grasp"] = grasp_pose
        target_poses["lift"] = lift_pose

        preplace_pose, place_pose, postplace_pose = self._get_placement_poses(
            grasp_pose_world=grasp_pose_world,
            pickup_obj=pickup_obj,
            place_receptacle=place_receptacle,
        )
        target_poses["preplace"] = preplace_pose
        target_poses["place"] = place_pose
        target_poses["postplace"] = postplace_pose

        # debug
        visualize_poses = True
        if visualize_poses and self.task.viewer is not None:
            self._show_poses(np.stack(list(target_poses.values()), axis=0), style="tcp")
            if self.task.viewer:
                self.task.viewer.sync()

        return target_poses
