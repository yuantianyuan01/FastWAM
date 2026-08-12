import io
import logging
import sys
from enum import IntEnum
from typing import Any

import mujoco
import numpy as np
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.motion_gen import MotionGenStatus
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.planner.curobo_planner import CuroboPlanner
from molmo_spaces.policy.solvers.curobo_planner_policy import CuroboPlannerPolicy
from molmo_spaces.tasks.opening_tasks import DoorOpeningTask
from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms, geom_aabb
from molmo_spaces.utils.pose import pose_mat_to_pos_quat

log = logging.getLogger(__name__)


class DoorOpeningPhase(IntEnum):
    """Enumeration of door opening phases."""

    NAVIGATE_TO_DOOR = 0
    REACH_PRE_GRASP = 1
    REACH_GRASP = 2
    GRASP_HANDLE = 3
    OPEN_DOOR = 4
    RELEASE_HANDLE = 5
    COMPLETE = 6
    RECOVERY = 7


class DoorOpeningPlannerPolicy(CuroboPlannerPolicy):
    """Planner policy for RBY1 door opening tasks using motion planning.

    Inherits common motion planning functionality from CuroboPlannerPolicy.
    """

    def __init__(self, config: MlSpacesExpConfig, task: DoorOpeningTask | None = None) -> None:
        super().__init__(config, task)

        # This policy uses both planners (unlike pick-and-place which lazy-loads one)
        self.left_motion_planner = CuroboPlanner(
            config=config.policy_config.left_curobo_planner_config
        )
        self.right_motion_planner = CuroboPlanner(
            config=config.policy_config.right_curobo_planner_config
        )

        # Planner policy state
        self.arm_side = None
        self.current_phase = DoorOpeningPhase.REACH_PRE_GRASP
        self.planning_failures = 0

        # Trajectory execution parameters
        self.max_steps_per_waypoint = self.config.policy_config.max_steps_per_waypoint
        self.joint_position_tolerance = self.config.policy_config.joint_position_tolerance

        # Door opening parameters
        self.pre_grasp_distance = self.config.policy_config.pre_grasp_distance
        self.articulate_deltas = self.config.policy_config.articulation_deltas
        self.first_pushing_articulation_deltas = (
            self.config.policy_config.first_pushing_articulation_deltas
        )

        # Phase-specific state
        self.grasping_timesteps = 0
        self.recovery_step_count = 0

        # Articulation tracking
        self.curr_articulation_step = 0
        self.num_steps_for_articulation = len(self.articulate_deltas)

    @property
    def planners(self) -> dict[str, CuroboPlanner]:
        return {
            "left_arm_planner": self.left_motion_planner,
            "right_arm_planner": self.right_motion_planner,
        }

    @property
    def is_done(self) -> bool:
        """Property to expose completion state for task checking."""
        return self.current_phase == DoorOpeningPhase.COMPLETE

    @property
    def phase_name(self) -> str:
        """Property to expose current phase name for task tracking."""
        return self.current_phase.name

    @property
    def phase_value(self) -> int:
        """Property to expose current phase name for task tracking."""
        return self.current_phase.value

    def get_phase(self) -> str:
        """Return the current phase name as a string."""
        return self.current_phase.name

    def get_all_phases(self) -> dict[str, int]:
        """Return a dictionary mapping all phase names to their enum values."""
        return {phase.name: phase.value for phase in DoorOpeningPhase}

    def reset(self) -> None:
        """Reset the policy state."""
        super().reset()
        log.debug("[RESET] Resetting DoorOpeningPlannerPolicy state")
        self.arm_side = None
        self.current_phase = DoorOpeningPhase.REACH_PRE_GRASP
        self.grasping_timesteps = 0
        self.planning_failures = 0
        self.curr_articulation_step = 0
        self.recovery_step_count = 0

        # Reset motion planners
        self.left_motion_planner.reset()
        self.right_motion_planner.reset()

    def _get_motion_planner(self) -> CuroboPlanner:
        """Get the motion planner for the currently selected arm."""
        if self.arm_side == "left":
            return self.left_motion_planner
        elif self.arm_side == "right":
            return self.right_motion_planner
        else:
            raise ValueError(f"Invalid arm selection: {self.arm_side}")

    def _get_planner_joint_ranges(self) -> dict[str, tuple]:
        """Get the joint ranges for the currently selected arm."""
        if self.arm_side == "left":
            return self.config.policy_config.left_planner_joint_ranges
        elif self.arm_side == "right":
            return self.config.policy_config.right_planner_joint_ranges
        else:
            raise ValueError(f"Invalid arm selection: {self.arm_side}")

    def _get_gripper_open_command(self) -> dict[str, float]:
        """Get the gripper open command for the currently selected arm."""
        if self.arm_side == "left":
            return self.config.policy_config.left_gripper_open_command
        elif self.arm_side == "right":
            return self.config.policy_config.right_gripper_open_command
        else:
            raise ValueError(f"Invalid arm selection: {self.arm_side}")

    def _get_gripper_close_command(self) -> dict[str, float]:
        """Get the gripper close command for the currently selected arm."""
        if self.arm_side == "left":
            return self.config.policy_config.left_gripper_close_command
        elif self.arm_side == "right":
            return self.config.policy_config.right_gripper_close_command
        else:
            raise ValueError(f"Invalid arm selection: {self.arm_side}")

    def _setup_collision_avoidance_config(self) -> None:
        """Setup collision avoidance for the world."""
        world_config_dict = {"mesh": {}, "cuboid": {}, "capsule": {}, "sphere": {}, "cylinder": {}}

        if self.task is not None and hasattr(self.task, "get_door_joint_position"):
            if not self.task._is_pushing_door:
                cylinder_list = self._get_door_swing_cylinder_config()
                world_config_dict["cylinder"] = {
                    item["name"]: {k: v for k, v in item.items() if k != "name"}
                    for item in cylinder_list
                }
                log.info("[COLLISION AVOIDANCE] Added door swing cylinder")

        cuboid_list = self._get_cuboid_configs()
        world_config_dict["cuboid"] = {
            item["name"]: {k: v for k, v in item.items() if k != "name"} for item in cuboid_list
        }
        log.info(
            f"[COLLISION AVOIDANCE] Added {len(world_config_dict['cuboid'])} nearby and structural objects"
        )

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            world_cfg = WorldConfig.from_dict(world_config_dict)
            world_cfg = WorldConfig.create_collision_support_world(world_cfg)
        finally:
            sys.stdout = old_stdout
        if self.config.policy_config.verbose:
            log.info(
                f"[COLLISION AVOIDANCE] World config created with "
                f"{len(world_cfg.cuboid) if world_cfg.cuboid else 0} cuboids"
            )

        self._get_motion_planner().motion_gen.update_world(world_cfg)

        self._world_cfg = world_cfg
        if self.config.policy_config.verbose:
            self._visualize_world_config_mesh()

    def _get_door_swing_cylinder_config(self) -> list[dict]:
        swing_arc = self.task.door_object.get_swing_arc_circle()
        cylinder_center_pos = swing_arc["center"]
        cylinder_radius = swing_arc["radius"] * 0.75

        cylinder_height = 0.3
        cylinder_pos = cylinder_center_pos.copy()
        cylinder_pos[2] = cylinder_height / 2
        cylinder_quat = [1.0, 0.0, 0.0, 0.0]

        cylinder_pose_world = np.concatenate([cylinder_pos, cylinder_quat])

        if self.config.policy_config.plan_in_robot_frame:
            cylinder_pose = self._transform_to_base_frame(cylinder_pose_world)
            # Convert 4x4 matrix to 7D pose
            from molmo_spaces.utils.pose import pose_mat_to_7d

            cylinder_pose = pose_mat_to_7d(cylinder_pose)
        else:
            cylinder_pose = cylinder_pose_world

        cylinder = {
            "name": "door_swing_region",
            "pose": cylinder_pose.tolist(),
            "radius": cylinder_radius,
            "height": cylinder_height,
        }
        return [cylinder]

    def _get_door_cuboids(self, door_body_names: list[str]) -> list[dict]:
        from molmo_spaces.env.data_views import Door

        door_configs = []

        model = self.task.env.current_model
        data = self.task.env.current_data
        om = self.task.env.object_managers[self.task.env.current_batch_index]

        for door_body_name in door_body_names:
            if door_body_name == self.task.door_object.door_name:
                continue
            try:
                door_object = Door(door_body_name, data)
                door_bboxes = om.get_door_bboxes_array(door_object)

                geom_infos = om.get_geom_infos(door_object, include_descendants=True)

                geom_id_to_bbox = {}
                bbox_idx = 0
                for geom_info in geom_infos:
                    geom_id = geom_info["id"]
                    if model.geom(geom_id).contype != 0 or model.geom(geom_id).conaffinity != 0:
                        if bbox_idx < len(door_bboxes):
                            geom_id_to_bbox[geom_id] = door_bboxes[bbox_idx]
                            bbox_idx += 1

                for geom_id, bbox in geom_id_to_bbox.items():
                    try:
                        geom_pos = data.geom_xpos[geom_id].copy()
                        geom_rot_mat = data.geom_xmat[geom_id].reshape(3, 3).copy()

                        geom_quat = R.from_matrix(geom_rot_mat).as_quat(scalar_first=True)

                        bbox_size = bbox[3:6]

                        pose = np.concatenate([geom_pos, geom_quat])

                        if self.config.policy_config.plan_in_robot_frame:
                            pose_mat = self._transform_to_base_frame(pose)
                            from molmo_spaces.utils.pose import pose_mat_to_7d

                            pose = pose_mat_to_7d(pose_mat)

                        door_config = {
                            "name": f"door_geom_{geom_id}",
                            "pose": pose.tolist(),
                            "dims": bbox_size.tolist(),
                        }
                        door_configs.append(door_config)
                    except Exception as e:
                        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                        log.warning(f"Failed to extract pose for door geom {geom_name}: {e}")
                        continue
            except Exception as e:
                log.warning(f"Failed to process door body {door_body_name}: {e}")
                continue
        return door_configs

    def _get_cuboid_configs(self) -> list[dict]:
        """Create cuboid configuration for the structural objects."""
        model = self.task.env.current_model
        data = self.task.env.current_data
        om = self.task.env.object_managers[self.task.env.current_batch_index]

        robot_pos = self.task.env.robots[0].get_world_pose_tf_mat()[:3, 3]
        cuboid_configs = []

        door_cuboids = self._get_door_cuboids(om.find_door_names())
        cuboid_configs.extend(door_cuboids)

        for b in om.top_level_bodies():
            add_this_obj = False
            name = om.get_object_name(b)
            if om.is_structural(b):
                if name.startswith("wall"):
                    add_this_obj = True
            else:
                obj = om.get_object_by_name(name)
                obj_pos = obj.position
                distance = np.linalg.norm(obj_pos[:2] - robot_pos[:2])
                if distance <= self.config.policy_config.relevant_collision_objects_radius:
                    add_this_obj = True

            if add_this_obj:
                geoms = descendant_geoms(model, b, visible_only=False)
                for geom in geoms:
                    aabb_center, aabb_size = geom_aabb(model, data, [geom])
                    obj_center = np.concatenate([aabb_center, np.array([1.0, 0.0, 0.0, 0.0])])
                    if self.config.policy_config.plan_in_robot_frame:
                        obj_center_mat = self._transform_to_base_frame(obj_center)
                        from molmo_spaces.utils.pose import pose_mat_to_7d

                        obj_center_7d = pose_mat_to_7d(obj_center_mat)
                    else:
                        obj_center_7d = np.concatenate(pose_mat_to_pos_quat(obj_center))
                    cuboid = {
                        "name": name,
                        "pose": obj_center_7d.tolist(),
                        "dims": aabb_size.tolist(),
                    }
                    cuboid_configs.append(cuboid)

        return cuboid_configs

    def _visualize_world_config_mesh(self) -> None:
        if not hasattr(self, "_world_cfg") or self._world_cfg is None:
            log.warning("World config not available. Call _setup_collision_avoidance_config first.")
            return

        world_cfg = self._world_cfg
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"world_config_mesh_{timestamp}"
        obj_path = f"{file_name}.obj"
        world_cfg.save_world_as_mesh(obj_path)

        log.info(f"Successfully saved world config mesh to {obj_path}")

    def select_arm_for_opening(self) -> str:
        """Select the arm for opening the door.

        Also sets self.arm_side and self.planner_joint_ranges for compatibility
        with base class methods.
        """
        robot_base_pose_tf = self.task.env.robots[0].get_world_pose_tf_mat()
        door_joint_pos_world = self.task.get_door_joint_position()
        door_handle_pos_world = self.task.get_door_handle_position()

        log.debug("\n[ARM SELECTION] Analyzing door geometry:")
        log.debug(f"  - Robot base position: {robot_base_pose_tf[:3, 3]}")
        log.debug(f"  - Door joint position: {door_joint_pos_world}")
        log.debug(f"  - Door handle position: {door_handle_pos_world}")

        door_joint_pos_robot = np.linalg.inv(robot_base_pose_tf) @ np.array(
            [*door_joint_pos_world, 1.0]
        )
        door_handle_pos_robot = np.linalg.inv(robot_base_pose_tf) @ np.array(
            [*door_handle_pos_world, 1.0]
        )

        selected_arm = "left" if door_joint_pos_robot[1] > door_handle_pos_robot[1] else "right"

        log.debug(f"  - Door joint in robot frame: {door_joint_pos_robot[:3]}")
        log.debug(f"  - Door handle in robot frame: {door_handle_pos_robot[:3]}")
        log.debug(
            f"  - Selected arm: {selected_arm} (joint Y: {door_joint_pos_robot[1]:.3f}, "
            f"handle Y: {door_handle_pos_robot[1]:.3f})"
        )

        # Set base class properties for compatibility with shared methods
        self.arm_side = selected_arm
        self.planner_joint_ranges = self._get_planner_joint_ranges()

        return selected_arm

    def get_look_at_action(self) -> dict[str, Any]:
        target_pos = self.task.get_door_handle_position()
        return {"head": self.task.get_target_head_pose(target_pos)}

    def get_action(self, info: dict[str, Any]) -> dict[str, Any]:
        """Get the next action based on current task state and phase."""
        self._print_distance_to_handle()

        if self.arm_side is None:
            self.arm_side = self.select_arm_for_opening()
            if self.config.policy_config.verbose:
                log.info(f"[ARM] Selected {self.arm_side} arm for door opening")

        action_cmd = {}
        if self.current_phase == DoorOpeningPhase.NAVIGATE_TO_DOOR:
            action_cmd = self._execute_navigate_phase()
        elif self.current_phase == DoorOpeningPhase.REACH_PRE_GRASP:
            action_cmd = self._execute_reach_pre_grasp_phase()
        elif self.current_phase == DoorOpeningPhase.REACH_GRASP:
            action_cmd = self._execute_reach_grasp_phase()
        elif self.current_phase == DoorOpeningPhase.GRASP_HANDLE:
            action_cmd = self._execute_grasp_handle_phase()
        elif self.current_phase == DoorOpeningPhase.OPEN_DOOR:
            action_cmd = self._execute_open_phase()
        elif self.current_phase == DoorOpeningPhase.RELEASE_HANDLE:
            action_cmd = self._execute_release_phase()
        elif self.current_phase == DoorOpeningPhase.COMPLETE:
            action_cmd = self._execute_complete_phase()
        elif self.current_phase == DoorOpeningPhase.RECOVERY:
            action_cmd = self._execute_recovery_phase()
        else:
            raise ValueError(f"Unknown phase: {self.current_phase}")

        action_cmd.update(self.get_look_at_action())
        action_cmd = self.clip_to_velocity_constraint(action_cmd)

        return action_cmd

    def _execute_navigate_phase(self) -> dict[str, Any]:
        """Execute navigation phase - move robot to door."""
        # TODO: Implement navigation phase if you want idk
        pass

    def _execute_reach_pre_grasp_phase(self) -> dict[str, Any]:
        """Execute reach pre-grasp phase."""
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory(gripper_command=self._get_gripper_open_command())
            else:
                log.info("[PHASE] Pre-grasp position reached, transitioning to reach grasp phase")
                self.current_phase = DoorOpeningPhase.REACH_GRASP
                self.planned_trajectory = None
                self.trajectory_index = 0
                self.steps_spent_in_waypoint = 0
                self.current_gripper_command = self._get_gripper_open_command()
                return self._execute_reach_grasp_phase()

        log.info(
            f"[PLANNING] Planning trajectory to pre-grasp pose (offset: {self.pre_grasp_distance}m)"
        )
        target_ee_poses = self.task.get_target_ee_pose(
            current_arm=self.arm_side,
            offset_distance=self.pre_grasp_distance,
            return_both_poses=True,
        )
        planned_trajectories = []

        last_failure_action = None
        for target_ee_pose in target_ee_poses:
            planning_result = self._plan_trajectory(target_ee_pose)
            if planning_result is not None:
                last_failure_action = planning_result
            planned_trajectories.append(self.planned_trajectory)
        best_trajectory = self._select_best_trajectory(planned_trajectories)
        if best_trajectory is None:
            if last_failure_action is not None:
                return last_failure_action
            log.warning("[PLANNING] All planning attempts failed but no failure action captured")
            return self._get_stationary_action()
        self.planned_trajectory = best_trajectory
        return self._execute_trajectory(gripper_command=self._get_gripper_open_command())

    def _execute_reach_grasp_phase(self) -> dict[str, Any]:
        """Execute reach grasp phase."""
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory(gripper_command=self._get_gripper_open_command())
            else:
                log.info("[PHASE] Grasp position reached, transitioning to grasp handle phase")
                self.current_phase = DoorOpeningPhase.GRASP_HANDLE
                self.planned_trajectory = None
                self.trajectory_index = 0
                self.steps_spent_in_waypoint = 0
                self.current_gripper_command = self._get_gripper_close_command()
                return self._execute_grasp_handle_phase()

        target_ee_pose = self.task.get_target_ee_pose(
            current_arm=self.arm_side,
            return_both_poses=False,
            maintain_orientation=True,
        )
        log.info("[PLANNING] Planning trajectory to grasp pose")
        log.info(f"  - Target EE pose: {target_ee_pose[:3]} (position)")
        return self._plan_and_execute_trajectory(
            target_ee_pose, gripper_command=self._get_gripper_open_command()
        )

    def _execute_grasp_handle_phase(self) -> dict[str, Any]:
        """Execute grasp handle phase."""
        self.current_gripper_command = self._get_gripper_close_command()
        action = self._get_stationary_action()
        if self.grasping_timesteps >= self.config.policy_config.max_grasping_timesteps:
            if self._grasping_something():
                log.info(
                    f"[GRASP] Handle successfully grasped after {self.grasping_timesteps} timesteps"
                )
                self.current_phase = DoorOpeningPhase.OPEN_DOOR
                self.grasping_timesteps = 0
            else:
                log.info(
                    f"[GRASP] Grasping failed after {self.grasping_timesteps} timesteps, "
                    "returning to pre-grasp"
                )
                self.current_phase = DoorOpeningPhase.REACH_PRE_GRASP
                self.grasping_timesteps = 0
        else:
            self.grasping_timesteps += 1
            if self.grasping_timesteps % 2 == 0:
                log.debug(
                    f"[GRASP] Closing gripper... "
                    f"({self.grasping_timesteps}/{self.config.policy_config.max_grasping_timesteps})"
                )
        return action

    def _execute_open_phase(self) -> dict[str, Any]:
        """Execute door opening phase."""
        door_joint_pos = self.task.door_object.get_joint_position(
            self.task.door_object.get_hinge_joint_index()
        )
        joint_min, joint_max = self.task.exp_config.task_config.articulated_joint_range
        if abs(door_joint_pos - joint_max) <= 0.02:
            self.current_phase = DoorOpeningPhase.RELEASE_HANDLE
            return self._execute_release_phase()
        elif (
            self.task._is_pushing_door
            and abs(door_joint_pos - joint_min) <= self.first_pushing_articulation_deltas[0]
        ):
            self.articulate_deltas = self.first_pushing_articulation_deltas
        else:
            self.articulate_deltas = self.config.policy_config.articulation_deltas

        if not self._grasping_something():
            log.warning("Handle lost during opening phase, returning to pre-grasp")
            self.current_phase = DoorOpeningPhase.REACH_PRE_GRASP
            self.planned_trajectory = None
            self.trajectory_index = 0
            self.steps_spent_in_waypoint = 0
            return self._execute_reach_pre_grasp_phase()

        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory(gripper_command=self._get_gripper_close_command())
            else:
                self.planned_trajectory = None
                self.trajectory_index = 0
                self.steps_spent_in_waypoint = 0

                if self.curr_articulation_step >= self.num_steps_for_articulation:
                    log.info(
                        "[ARTICULATION] All articulation steps completed, "
                        "transitioning to release phase"
                    )
                    self.current_phase = DoorOpeningPhase.RELEASE_HANDLE
                    return self._execute_release_phase()

        target_ee_pose = self.task.get_target_ee_pose(
            current_arm=self.arm_side,
            articulate_deltas=self.articulate_deltas,
            maintain_orientation=False,
        )
        action = self._plan_and_execute_trajectory(
            target_ee_pose, gripper_command=self._get_gripper_close_command()
        )

        return action

    def _execute_release_phase(self) -> dict[str, Any]:
        """Execute release phase."""
        log.info("[RELEASE] Releasing door handle")
        self.current_gripper_command = self._get_gripper_open_command()
        action = self._get_stationary_action()

        self.current_phase = DoorOpeningPhase.COMPLETE
        return action

    def _execute_complete_phase(self) -> dict[str, Any]:
        """Execute complete phase."""
        log.info("[COMPLETE] Door opening sequence completed! Emitting done action.")
        return {"done": True}

    def _execute_recovery_phase(self) -> dict[str, Any]:
        """Execute recovery phase."""
        if self.recovery_step_count < self.config.policy_config.num_recovery_steps:
            self.recovery_step_count += 1
        else:
            self.current_phase = DoorOpeningPhase.REACH_PRE_GRASP
            self.recovery_step_count = 0
        return self._get_recovery_action()

    def _plan_trajectory(self, target_ee_pose: np.ndarray) -> dict[str, Any] | None:
        current_joint_pos = self._get_current_joint_positions()

        if self.config.policy_config.plan_in_robot_frame:
            if self.config.policy_config.enable_collision_avoidance:
                self._setup_collision_avoidance_config()

            target_ee_pose_mat = self._transform_to_base_frame(target_ee_pose)
            from molmo_spaces.utils.pose import pose_mat_to_7d

            target_ee_pose = pose_mat_to_7d(target_ee_pose_mat)
            planner_joint_ranges = self._get_planner_joint_ranges()
            current_joint_pos[planner_joint_ranges["base"][0] : planner_joint_ranges["base"][1]] = (
                np.array([0.0, 0.0, 0.0])
            )

        self.planned_trajectory, planning_result = self._get_motion_planner().motion_plan(
            current_joint_pos, [target_ee_pose.tolist()], verbose=False
        )
        self.trajectory_index = 0
        self.steps_spent_in_waypoint = 0

        if planning_result.success.item() is True:
            log.debug(
                f"[PLANNING] Motion planning successful with trajectory length: "
                f"{len(self.planned_trajectory)}"
            )
            if self.config.policy_config.plan_in_robot_frame:
                self._transform_traj_to_world_frame(self.planned_trajectory)
        else:
            log.warning(f"Planning failed with status: {planning_result.status}")
            self.planning_failures += 1
            self.planned_trajectory = None
            failure_action = self._get_stationary_action()
            if self.planning_failures >= self.config.policy_config.max_planning_failures:
                log.warning("Stopping planning. Switching to complete state")
                self.current_phase = DoorOpeningPhase.COMPLETE
            elif planning_result.status == MotionGenStatus.IK_FAIL:
                log.warning("Will re-plan...")
            elif planning_result.status == MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION:
                log.warning("Start state is in collision. Switching to recovery motion...")
                self.current_phase = DoorOpeningPhase.RECOVERY
                failure_action = self._get_recovery_action()
            return failure_action

    def _plan_and_execute_trajectory(
        self, target_ee_pose: np.ndarray, gripper_command: dict[str, float]
    ) -> dict[str, Any]:
        """Helper method to plan and execute trajectory."""
        planning_result = self._plan_trajectory(target_ee_pose)
        if planning_result is not None:
            return planning_result
        return self._execute_trajectory(gripper_command=gripper_command)

    def _get_stationary_action(self) -> dict[str, Any]:
        """Get action to maintain current robot position."""
        action = {}
        action.update(self.current_gripper_command)
        action.update(self.get_look_at_action())
        return action

    def _get_recovery_action(self) -> dict[str, Any]:
        """Get action to recover from a planning failure."""
        recovery_action = {}
        current_robot_base_pose_tf = self.task.env.robots[0].get_world_pose_tf_mat()
        current_robot_base_theta = R.from_matrix(current_robot_base_pose_tf[:3, :3]).as_euler(
            "XYZ", degrees=False
        )[2]
        new_position_base_frame = np.array(
            [-self.config.policy_config.recovery_motion_backward_distance, 0.0, 0.0]
        )
        new_position_world_frame = (
            current_robot_base_pose_tf[:3, 3]
            + current_robot_base_pose_tf[:3, :3] @ new_position_base_frame
        )
        base_action = np.array(
            [new_position_world_frame[0], new_position_world_frame[1], current_robot_base_theta]
        )
        recovery_action.update({"base": base_action})
        recovery_action.update(self.config.policy_config.left_gripper_open_command)
        recovery_action.update(self.config.policy_config.right_gripper_open_command)
        recovery_action.update(self.get_look_at_action())
        return recovery_action

    def _print_distance_to_handle(self) -> None:
        """Print current distance from robot to door handle for debugging."""
        try:
            robot_base_pos = self.task.env.robots[0].get_world_pose_tf_mat()[:3, 3]
            door_handle_pos = self.task.get_door_handle_position()

            np.linalg.norm(door_handle_pos - robot_base_pos)
            np.linalg.norm(door_handle_pos[:2] - robot_base_pos[:2])

            if self.arm_side is not None:
                try:
                    arm_move_group = self.task.env.robots[0].robot_view.get_move_group(
                        f"{self.arm_side}_arm"
                    )
                    ee_pose_rel_to_base = arm_move_group.leaf_frame_to_robot
                    robot_base_pose = self.task.env.robots[0].get_world_pose_tf_mat()
                    ee_world_pose = robot_base_pose @ ee_pose_rel_to_base
                    ee_pos = ee_world_pose[:3, 3]
                    np.linalg.norm(door_handle_pos - ee_pos)
                except Exception as e:
                    log.error(
                        f"[DISTANCE] Failed to get end-effector position for EE distance calculation: {e}"
                    )
                    pass

        except Exception as e:
            log.error(f"[DISTANCE] Failed to calculate distance: {e}")
