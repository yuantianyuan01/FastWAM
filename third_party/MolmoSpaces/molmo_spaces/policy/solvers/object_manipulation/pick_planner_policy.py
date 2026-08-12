import logging

import numpy as np

from molmo_spaces.configs import PickTaskConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.utils.grasp_sample import select_grasp_pose
from molmo_spaces.utils.grasps import get_pickup_grasps

log = logging.getLogger(__name__)


class PickPlannerPolicy(BaseObjectManipulationPlannerPolicy):
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
                    )
                ],
            ),
        ]

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        """Compute all target poses for the pick-and-place sequence."""
        task_config = self.config.task_config
        assert isinstance(task_config, PickTaskConfig)
        robot_view = self.task.env.current_robot.robot_view
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj: MlSpacesObject = om.get_object_by_name(task_config.pickup_obj_name)

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

        target_poses = {}

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
        pregrasp_pose[:3, 3] += np.array(
            [0, 0, self.policy_config.pregrasp_z_offset + pregrasp_height_offset]
        )

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

        target_poses["pregrasp"] = pregrasp_pose

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

        target_poses["grasp"] = grasp_pose_world

        # Lift pose - above grasp position
        lift_pose = grasp_pose_world.copy()
        lift_pose[:3, 3] += np.array(
            [0, 0, self.policy_config.postgrasp_z_offset + postgrasp_height_offset]
        )

        # debug
        visualize_poses = False
        if visualize_poses:
            self._show_poses(np.array([pregrasp_pose]), style="tcp", color=(0, 1, 0, 1))  # green
            self._show_poses(np.array([grasp_pose_world]), style="tcp")  # red
            self._show_poses(np.array([lift_pose]), style="tcp", color=(0, 0, 1, 1))  # blue
            if self.task.viewer:
                self.task.viewer.sync()

        if not self.check_feasible_ik(lift_pose):
            log.debug("  - ❌ IK FAILED for lift pose!")
            raise ValueError("IK failed for lift pose")

        target_poses["lift"] = lift_pose

        log.info(f"Planning completed. w/ {len(target_poses)} steps\n")

        return target_poses
