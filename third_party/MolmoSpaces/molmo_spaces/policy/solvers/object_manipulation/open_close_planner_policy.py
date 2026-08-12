import logging

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.utils.articulation_utils import (
    gather_joint_info,
    step_circular_path,
    step_linear_path,
)
from molmo_spaces.utils.grasp_sample import select_grasp_pose
from molmo_spaces.utils.grasps import get_joint_grasps

log = logging.getLogger(__name__)


class OpenClosePlannerPolicy(BaseObjectManipulationPlannerPolicy):
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

    def _compute_target_poses(self) -> None:
        """Compute all target poses for the pick-and-place sequence."""
        self.task._env.mj_datas[0]
        robot_view = self.task._env.robots[0].robot_view
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)

        candidate_grasps, object_pose = get_joint_grasps(
            self.task.env,
            pickup_obj,
            self.config.task_config.joint_index,
            grasp_libraries=self.policy_config.grasp_libraries,
        )
        grasp_pose_world = select_grasp_pose(
            self.task.env,
            candidate_grasps,
            object_pose,
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
            horizontal_cost_weight=self.policy_config.grasp_horizontal_cost_weight,
        )

        target_poses = {}

        pregrasp_pose = grasp_pose_world.copy()
        grasp_pos = grasp_pose_world[:3, 3]
        rotation = R.from_matrix(grasp_pose_world[:3, :3])
        distance = self.policy_config.pregrasp_z_offset
        pregrasp_pose[:3, 3] = grasp_pos + rotation.apply(np.array([0, 0, -distance]))

        log.debug(f"  - obj_start (p): {pickup_obj.position}")
        log.debug(f"  - obj_start (t): {self.config.task_config.pickup_obj_start_pose}")
        log.debug(f"  - obj_end (t): {self.config.task_config.pickup_obj_goal_pose}")
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

        joint_info = gather_joint_info(
            self.task._env.mj_model,
            self.task._env.mj_datas[0],
            pickup_obj.joint_ids[self.config.task_config.joint_index],
        )

        if joint_info["joint_type"] == mujoco.mjtJoint.mjJNT_HINGE:
            grasp_pos = grasp_pose_world[:3, 3]
            grasp_quat = R.from_matrix(grasp_pose_world[:3, :3]).as_quat(scalar_first=True)

            joint_range = joint_info["joint_range"]
            nonzero_index = np.nonzero(joint_range)
            assert len(nonzero_index) == 1, (
                f"Joint range has multiple non-zero indices for {pickup_obj.joint_names[self.config.task_config.joint_index]}"
            )
            if self.config.task_type == "open":
                max_joint_angle = joint_range[nonzero_index[0]]
            elif self.config.task_type == "close":
                max_joint_angle = 0

            circluar_path_dict = step_circular_path(
                grasp_pos, grasp_quat, joint_info, max_joint_angle, n_waypoints=10
            )

            lift_pose = pregrasp_pose.copy()

            # open half way
            n_waypoints = len(circluar_path_dict["mocap_pos"])
            some_waypoint = n_waypoints // 3
            lift_pose[:3, 3] = circluar_path_dict["mocap_pos"][some_waypoint]  # -1
            lift_pose[:3, :3] = R.from_quat(
                circluar_path_dict["mocap_quat"][some_waypoint], scalar_first=True
            ).as_matrix()

        elif joint_info["joint_type"] == mujoco.mjtJoint.mjJNT_SLIDE:
            grasp_pos = grasp_pose_world[:3, 3]
            grasp_quat = R.from_matrix(grasp_pose_world[:3, :3]).as_quat(scalar_first=True)

            """ Z Axis should be the same"""
            # Transform joint axis from local body frame to world frame
            joint_axis_world = joint_info["joint_body_orientation"] @ joint_info["joint_axis"]
            joint_direction = -joint_axis_world
            normalize_dir_axis = joint_direction / np.linalg.norm(joint_direction)

            # joint_body_position = joint_info["joint_body_position"]
            # joint_body_position[2] = grasp_pos[2]
            # normalize_dir_axis = (joint_body_position - grasp_pos) / np.linalg.norm(
            #    joint_body_position - grasp_pos
            # )

            current_joint_pos = joint_info["joint_pos"]

            if self.config.task_type == "open":
                max_joint_angle = joint_info["max_range"]
            elif self.config.task_type == "close":
                max_joint_angle = 0

            linear_path_dict = step_linear_path(
                to_handle_dist=normalize_dir_axis * (max_joint_angle - current_joint_pos),
                current_pos=grasp_pos,
                current_quat=grasp_quat,
                step_size=0.01,  # 0.005,
                is_reverse=True,
            )

            lift_pose = pregrasp_pose.copy()
            lift_pose[:3, 3] = linear_path_dict["mocap_pos"][-1]
            lift_pose[:3, :3] = R.from_quat(
                linear_path_dict["mocap_quat"][-1], scalar_first=True
            ).as_matrix()

        else:
            raise ValueError(f"Unknown joint type: {joint_info['joint_type']}")

        # debug
        visualize_poses = False
        if visualize_poses:
            self._show_poses(np.array([pregrasp_pose]), style="tcp", color=(0, 1, 0, 1))  # green
            self._show_poses(np.array([grasp_pose_world]), style="tcp")  # red
            self._show_poses(np.array([lift_pose]), style="tcp", color=(0, 0, 1, 1))  # blue
            if self.task.viewer:
                self.task.viewer.sync()

        # if not self.check_feasible_ik(lift_pose):
        #    log.debug("  - ❌ IK FAILED for lift pose!")
        #    raise ValueError("IK failed for lift pose")

        target_poses["lift"] = lift_pose

        log.info(f"Planning completed. w/ {len(target_poses)} steps\n")

        return target_poses
