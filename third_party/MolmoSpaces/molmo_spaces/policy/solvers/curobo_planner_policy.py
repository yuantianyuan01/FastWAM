import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from curobo.geom.types import WorldConfig
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.planner.curobo_planner import CuroboPlanner
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.profiler_utils import Profiler

log = logging.getLogger(__name__)


class CuroboPlannerPolicy(PlannerPolicy):
    """Base class for Curobo-based planner policies.

    This class provides common functionality for motion planning using Curobo,
    including trajectory execution, coordinate frame transformations, and
    gripper control. Subclasses should implement task-specific planning logic.
    """

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask | None = None) -> None:
        super().__init__(config, task)
        self.config = config

        # Planner instances (to be set by subclasses)
        self.planner: CuroboPlanner | None = None
        self.planner_joint_ranges: dict[str, tuple[int, int]] = {}

        # Trajectory state
        self.planned_trajectory: list[list[float]] | None = None
        self.trajectory_index: int = 0
        self.steps_spent_in_waypoint: int = 0
        self._retry_count: int = 0

        # Arm selection
        self.arm_side: str | None = None
        self.arm_start_idx: int = 0
        self.arm_end_idx: int = 0

        # Gripper state
        self.current_gripper_command: dict[str, float] = {}

        # Profiler
        self.profiler = Profiler()

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    @abstractmethod
    def planners(self) -> dict[str, CuroboPlanner]:
        """Return dictionary of planner instances."""
        pass

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """Property to expose completion state for task checking."""
        pass

    def reset(self) -> None:
        """Reset the policy state."""
        self.planned_trajectory = None
        self.trajectory_index = 0
        self.steps_spent_in_waypoint = 0
        self._retry_count = 0
        self.current_gripper_command = {}

    # ========== Joint Position Methods ==========

    def _get_current_joint_positions(self) -> list[float]:
        """Get current joint positions for all move groups.

        Returns:
            List of joint positions for all configured move groups.
        """
        joint_positions = []

        move_groups_for_planning = list(self.planner_joint_ranges.keys())
        for move_group in move_groups_for_planning:
            move_group_view = self.task.env.robots[0].robot_view.get_move_group(move_group)
            move_group_joint_pos = move_group_view.joint_pos.copy()
            # Clip to joint limits (mujoco uses soft constraints)
            move_group_joint_pos = np.clip(
                move_group_joint_pos,
                move_group_view.joint_pos_limits[:, 0],
                move_group_view.joint_pos_limits[:, 1],
            )
            joint_positions.extend(move_group_joint_pos.tolist())

        return joint_positions

    # ========== Coordinate Frame Transformations ==========
    def _transform_to_base_frame(self, target_ee_pose: np.ndarray) -> np.ndarray:
        """Transform target end-effector pose from world frame to robot base frame.

        Args:
            target_ee_pose: 7D pose [x, y, z, qw, qx, qy, qz] in world frame.

        Returns:
            4x4 transformation matrix in robot base frame.
        """
        robot_base_pose_tf = self.task.env.robots[0].get_world_pose_tf_mat()
        target_ee_pose_tf_world_f = np.eye(4)
        target_ee_pose_tf_world_f[:3, 3] = target_ee_pose[:3]
        target_ee_pose_tf_world_f[:3, :3] = R.from_quat(
            target_ee_pose[3:], scalar_first=True
        ).as_matrix()
        return np.linalg.inv(robot_base_pose_tf) @ target_ee_pose_tf_world_f

    def _transform_traj_to_world_frame(self, trajectory: list[list[float]]) -> None:
        """Transform planned trajectory from robot base frame to world frame.

        Modifies the trajectory in-place, converting base joint positions
        from robot-relative to world coordinates.

        Args:
            trajectory: List of waypoints to transform.
        """
        robot_base_pose_tf = self.task.env.robots[0].get_world_pose_tf_mat()
        planner_joint_ranges = self.planner_joint_ranges
        for waypoint in trajectory:
            base_joint_x, base_joint_y, base_joint_yaw = waypoint[
                planner_joint_ranges["base"][0] : planner_joint_ranges["base"][1]
            ]
            base_joint_tf = np.eye(4)
            base_joint_tf[:3, 3] = np.array([base_joint_x, base_joint_y, 0.0])
            base_joint_tf[:3, :3] = R.from_euler("Z", base_joint_yaw, degrees=False).as_matrix()
            new_base_joint_tf = robot_base_pose_tf @ base_joint_tf
            waypoint[planner_joint_ranges["base"][0] : planner_joint_ranges["base"][1] - 1] = (
                new_base_joint_tf[:3, 3][:2]
            )
            waypoint[planner_joint_ranges["base"][1] - 1] = R.from_matrix(
                new_base_joint_tf[:3, :3]
            ).as_euler("XYZ", degrees=False)[2]

    def _target_pose_to_base_frame(self, target_pose: np.ndarray) -> np.ndarray:
        """Convert a 4x4 target pose matrix to 7D pose in robot base frame.

        Args:
            target_pose: 4x4 transformation matrix in world frame.

        Returns:
            7D pose [x, y, z, qw, qx, qy, qz] in robot base frame.
        """
        pose_7d = pose_mat_to_7d(target_pose)
        pose_base_frame_4x4 = self._transform_to_base_frame(pose_7d)
        pose_base_frame_7d = pose_mat_to_7d(pose_base_frame_4x4)
        return pose_base_frame_7d

    def _interpolate_joint_trajectory(
        self, start_config: np.ndarray, end_config: np.ndarray, num_steps: int = 10
    ) -> list[list[float]]:
        """Linearly interpolate between two joint configurations.

        Args:
            start_config: Starting joint configuration.
            end_config: Ending joint configuration.
            num_steps: Number of interpolation steps.

        Returns:
            List of waypoints interpolated between start and end configs.
        """
        trajectory = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            waypoint = (1 - alpha) * start_config + alpha * end_config
            trajectory.append(waypoint.tolist())
        return trajectory

    # ========== Waypoint and Action Methods ==========

    def _waypoint_to_action(self, waypoint: list[float]) -> dict[str, Any]:
        """Convert a waypoint (joint positions) to robot action dictionary.

        Args:
            waypoint: List of joint positions for all move groups.

        Returns:
            Dictionary mapping move group names to joint position arrays.
        """
        action = {}

        for move_group, (start_idx, end_idx) in self.planner_joint_ranges.items():
            if start_idx < len(waypoint) and end_idx <= len(waypoint):
                action[move_group] = np.array(waypoint[start_idx:end_idx])

        return action

    def _is_waypoint_reached(self, waypoint: list[float], tolerance: float = 0.0275) -> bool:
        """Check if the current robot state is close enough to the waypoint.

        Args:
            waypoint: Target joint positions.
            tolerance: Maximum allowed joint position error.

        Returns:
            True if all joints are within tolerance of the waypoint.
        """
        current_joint_pos = self._get_current_joint_positions()
        joint_diff = np.abs(np.array(current_joint_pos) - np.array(waypoint))
        return bool(np.all(joint_diff < tolerance))

    # ========== Trajectory Execution ==========

    def _execute_trajectory(self, gripper_command: dict[str, float]) -> dict[str, Any]:
        """Execute the planned trajectory with gripper control.

        Args:
            gripper_command: Dictionary with gripper control commands.

        Returns:
            Action dictionary for the robot.

        Raises:
            ValueError: If no trajectory is planned or trajectory is exhausted.
        """
        if not self.planned_trajectory or self.trajectory_index >= len(self.planned_trajectory):
            raise ValueError(
                "_execute_trajectory was called with no planned trajectory or "
                "trajectory index >= len(planned_trajectory)"
            )

        # Skip waypoints that are already reached
        while self.trajectory_index < len(self.planned_trajectory):
            waypoint = self.planned_trajectory[self.trajectory_index]
            if self._is_waypoint_reached(waypoint):
                log.debug(f"[WAYPOINT] Skipping already-reached waypoint {self.trajectory_index}")
                self.trajectory_index += 1
                self.steps_spent_in_waypoint = 0
            else:
                break

        # If we've skipped all waypoints, return minimal action
        if self.trajectory_index >= len(self.planned_trajectory):
            action = {}
            action.update(gripper_command)
            self.current_gripper_command = gripper_command
            if hasattr(self, "get_look_at_action"):
                action.update(self.get_look_at_action())
            return action

        waypoint = self.planned_trajectory[self.trajectory_index]
        action = self._waypoint_to_action(waypoint)

        # Add gripper control
        action.update(gripper_command)
        self.current_gripper_command = gripper_command

        self.steps_spent_in_waypoint += 1

        # Check if waypoint reached
        if self._is_waypoint_reached(waypoint):
            log.debug(
                f"[WAYPOINT] Waypoint {self.trajectory_index} reached after "
                f"{self.steps_spent_in_waypoint} steps"
            )
            self.trajectory_index += 1
            self.steps_spent_in_waypoint = 0
        else:
            max_steps = getattr(self.config.policy_config, "max_steps_per_waypoint", 100)
            if self.steps_spent_in_waypoint >= max_steps:
                log.warning(
                    f"[TIMEOUT] Timed out on reaching waypoint {self.trajectory_index} after "
                    f"{self.steps_spent_in_waypoint} steps. Will re-plan..."
                )
                max_reattempts = getattr(self.config.policy_config, "max_planning_reattempts", 3)
                if self.retry_count > max_reattempts:
                    raise ValueError("Max planning reattempts reached.")
                self.planned_trajectory = None
                self.trajectory_index = 0
                self._retry_count += 1
                self.steps_spent_in_waypoint = 0

        return action

    def _select_best_trajectory(
        self, planned_trajectories: list[list[list[float]] | None]
    ) -> list[list[float]] | None:
        """Select the trajectory with the least total joint movement.

        Args:
            planned_trajectories: List of candidate trajectories (may include None for failures).

        Returns:
            Best trajectory or None if all planning attempts failed.
        """
        # Filter out None trajectories (failed planning attempts)
        valid_trajectories = [traj for traj in planned_trajectories if traj is not None]

        if not valid_trajectories:
            log.warning("[TRAJECTORY SELECTION] All planning attempts failed")
            return None

        if len(valid_trajectories) == 1:
            return valid_trajectories[0]

        current_joint_pos = np.array(self._get_current_joint_positions())

        best_trajectory = None
        min_total_movement = float("inf")

        for trajectory in valid_trajectories:
            if not trajectory:
                continue

            total_movement = 0.0
            prev_joint_pos = current_joint_pos

            for waypoint in trajectory:
                waypoint_joint_pos = np.array(waypoint)
                joint_movement = np.linalg.norm(waypoint_joint_pos - prev_joint_pos)
                total_movement += joint_movement
                prev_joint_pos = waypoint_joint_pos

            if total_movement < min_total_movement:
                min_total_movement = total_movement
                best_trajectory = trajectory

        if best_trajectory is None:
            best_trajectory = valid_trajectories[0]

        log.info(
            f"Selected trajectory with total joint movement: {min_total_movement:.4f} "
            f"(out of {len(valid_trajectories)} candidates)"
        )
        return best_trajectory

    # ========== Velocity Constraints ==========

    def clip_to_velocity_constraint(self, action: dict[str, Any]) -> dict[str, Any]:
        """Clip action to respect velocity constraints.

        Args:
            action: Dictionary of commanded joint positions by move group.

        Returns:
            Clipped action dictionary.
        """
        velocity_constraints = getattr(self.config.policy_config, "velocity_constraints", {})
        clipped_action = action.copy()

        for move_group, commanded_action in action.items():
            if move_group in velocity_constraints:
                max_velocity_rad_per_s = velocity_constraints[move_group]
                move_group_view = self.task.env.robots[0].robot_view.get_move_group(move_group)
                move_group_joint_pos = move_group_view.joint_pos.copy()

                # Calculate difference, handling angular wraparound for base theta
                diff = commanded_action - move_group_joint_pos
                if move_group == "base":
                    # Normalize theta difference to [-π, π]
                    diff[2] = np.arctan2(np.sin(diff[2]), np.cos(diff[2]))

                velocity_rad_per_s = diff * 10.0

                # Clip velocity and recalculate commanded action
                clipped_velocity = np.clip(
                    velocity_rad_per_s, -max_velocity_rad_per_s, max_velocity_rad_per_s
                )
                commanded_action = move_group_joint_pos + clipped_velocity / 10.0

                clipped_action[move_group] = commanded_action

        return clipped_action

    # ========== Gripper Methods ==========

    def _grasping_something(self, arm_side: str | None = None) -> bool:
        """Check if the gripper is grasping something.

        Determines grasp by checking if the gripper position deviates from
        the fully closed position by more than a threshold.

        Args:
            arm_side: Which arm to check ('left' or 'right'). Uses self.arm_side if None.

        Returns:
            True if gripper appears to be grasping an object.
        """
        if arm_side is None:
            arm_side = self.arm_side

        gripper_pos = (
            self.task.env.robots[0].robot_view.get_move_group(f"{arm_side}_gripper").joint_pos
        )
        gripper_closed_pos = getattr(self.config.policy_config, "gripper_closed_pos", 0.0)
        gripper_closed_tolerance = getattr(
            self.config.policy_config, "gripper_closed_tolerance", 0.01
        )

        gripper_deviation_from_closed = np.abs(gripper_pos - gripper_closed_pos).sum()
        is_grasping = gripper_deviation_from_closed > gripper_closed_tolerance

        log.debug(
            f"[GRIPPER] Gripper check - position: {gripper_pos}, "
            f"deviation: {gripper_deviation_from_closed:.4f}, grasping: {is_grasping}"
        )

        return is_grasping

    # ========== Arm Selection ==========

    def select_arm(self) -> None:
        """Select which arm to use based on distance to pickup object.

        Also instantiates the motion planner for the selected arm.
        This lazy initialization saves ~11GB of GPU memory by only loading one arm's planner.
        """
        task_config = self.config.task_config
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_config.pickup_obj_name)
        pickup_obj_pos = pickup_obj.position

        left_tcp_pose = self.task.env.current_robot.robot_view.get_move_group(
            "left_gripper"
        ).leaf_frame_to_world
        right_tcp_pose = self.task.env.current_robot.robot_view.get_move_group(
            "right_gripper"
        ).leaf_frame_to_world

        left_tcp_pos = left_tcp_pose[:3, 3]
        right_tcp_pos = right_tcp_pose[:3, 3]

        # Compute distances
        left_dist = np.linalg.norm(left_tcp_pos - pickup_obj_pos)
        right_dist = np.linalg.norm(right_tcp_pos - pickup_obj_pos)

        selected_arm = "left" if left_dist < right_dist else "right"
        log.info(
            f"Selected {selected_arm} arm (left dist: {left_dist:.3f}m, right dist: {right_dist:.3f}m)"
        )

        self.arm_side = selected_arm

        # Instantiate the planner for the selected arm only
        log.info(f"Instantiating motion planner for {selected_arm} arm")
        if selected_arm == "left":
            self.planner = CuroboPlanner(
                config=self.config.policy_config.left_curobo_planner_config
            )
            self.planner_joint_ranges = self.config.policy_config.left_planner_joint_ranges
        else:
            self.planner = CuroboPlanner(
                config=self.config.policy_config.right_curobo_planner_config
            )
            self.planner_joint_ranges = self.config.policy_config.right_planner_joint_ranges

        self.arm_start_idx = self.planner_joint_ranges[f"{self.arm_side}_arm"][0]
        self.arm_end_idx = self.planner_joint_ranges[f"{self.arm_side}_arm"][1]

    # ========== IK and Motion Planning ==========

    def solve_ik(self, target_pose: np.ndarray) -> None:
        """Solve inverse kinematics for a target pose and create interpolated trajectory.

        Args:
            target_pose: 4x4 transformation matrix for target end-effector pose in world frame.

        Raises:
            ValueError: If IK solution cannot be found.
        """
        init_config = np.concatenate(
            [
                np.zeros(3),
                self.task.env.current_robot.robot_view.get_move_group(
                    f"{self.arm_side}_arm"
                ).joint_pos,
            ]
        )

        target_pose_7d = self._target_pose_to_base_frame(target_pose)
        joint_config, _ = self.planner.ik_solve(
            goal_pose=target_pose_7d.tolist(),
            seed_config=init_config.tolist(),
            disable_collision=True,
        )

        current_phase = getattr(self, "current_phase", "unknown")
        if joint_config is None:
            raise ValueError(f"Could not solve {current_phase} phase IK.")

        trajectory = self._interpolate_joint_trajectory(
            init_config,
            np.array(joint_config),
            num_steps=10,
        )
        self._transform_traj_to_world_frame(trajectory)
        self.planned_trajectory = trajectory

    def batch_plan_trajectory(self) -> None:
        """Plan trajectory using batch motion planning.

        Uses the current phase to determine goal poses and plans trajectories
        in batches for efficiency. Sets self.planned_trajectory to the best
        trajectory found.
        """
        init_config = np.concatenate(
            [
                np.zeros(3),
                self.task.env.current_robot.robot_view.get_move_group(
                    f"{self.arm_side}_arm"
                ).joint_pos,
            ]
        )

        # Setup collision avoidance if enabled
        if getattr(self.config.policy_config, "enable_collision_avoidance", False):
            if hasattr(self, "_setup_collision_avoidance_config"):
                self._setup_collision_avoidance_config()

        # Get goal poses based on current phase (to be provided by subclass)
        goal_poses = self._get_batch_goal_poses()
        if goal_poses is None or len(goal_poses) == 0:
            log.warning("[BATCH PLAN] No goal poses available")
            return

        total = goal_poses.shape[0]
        batch_size = getattr(self.config.policy_config, "batch_size", 8)
        max_batches = getattr(self.config.policy_config, "max_batch_plan_attempts", 4)
        num_poses = min(max_batches * batch_size, total)
        num_batches = (num_poses + batch_size - 1) // batch_size

        all_successful_trajectories = []
        current_phase = getattr(self, "current_phase", "unknown")

        for batch_start in range(0, num_poses, batch_size):
            batch_end = min(batch_start + batch_size, num_poses)
            batch = goal_poses[batch_start:batch_end]

            if hasattr(self, "_show_poses"):
                self._show_poses(batch, style="tcp")
            if self.task.viewer:
                self.task.viewer.sync()

            log.info(
                f"Processing batch {batch_start // batch_size + 1}/{num_batches}: "
                f"poses {batch_start}-{batch_end - 1}"
            )

            # Transform poses to base frame and convert to 7D format
            batch_base_frame_7d = []
            for pose in batch:
                pose_base = self._target_pose_to_base_frame(pose)
                batch_base_frame_7d.append(pose_base.tolist())

            # Prepare batch inputs for planner
            batch_len = len(batch_base_frame_7d)
            start_states = [init_config.tolist()] * batch_len

            # Plan batch
            log.info(
                f"Planning batch of {batch_len} {current_phase} poses with {self.arm_side} arm"
            )
            result = self.planner.plan_batch(
                start_states,
                batch_base_frame_7d,
                verbose=False,
            )

            # Check for successes
            successes = result.success.cpu().numpy()
            log.info(
                f"{self.arm_side.capitalize()} arm: {np.sum(successes)}/{batch_len} "
                f"{current_phase} planning successes"
            )

            # Process successful trajectories
            if np.any(successes):
                optimized_plan = result.optimized_plan
                position = optimized_plan.position
                if position.ndim == 2:
                    position = position.unsqueeze(0)

                for i in range(batch_len):
                    if not successes[i]:
                        continue

                    trajectory = []
                    for t in range(position.shape[1]):
                        waypoint = position[i, t].cpu().tolist()
                        trajectory.append(waypoint)

                    self._transform_traj_to_world_frame(trajectory)
                    all_successful_trajectories.append(trajectory)

            if all_successful_trajectories:
                log.info(
                    f"Found {len(all_successful_trajectories)} successful trajectories, "
                    "skipping remaining batches"
                )
                break

        if all_successful_trajectories:
            self.planned_trajectory = self._select_best_trajectory(all_successful_trajectories)
        else:
            log.warning("[BATCH PLAN] No successful trajectories found across all batches")

    def _get_batch_goal_poses(self) -> np.ndarray | None:
        """Get goal poses for batch planning based on current phase.

        Subclasses should override this method to return appropriate goal poses.

        Returns:
            Array of 4x4 pose matrices or None if not applicable.
        """
        return None

    # ========== Visualization ==========

    def visualize_world_config_mesh(self, world_cfg: WorldConfig) -> None:
        """Visualize the world configuration as a mesh file.

        Args:
            world_cfg: Curobo WorldConfig to visualize.
        """
        from datetime import datetime

        import trimesh
        from curobo.geom.types import WorldConfig

        current_phase = getattr(self, "current_phase", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"world_config_mesh_{current_phase}_{timestamp}"
        obj_path = f"{file_name}.obj"
        initial_joint_configuration = np.array(self._get_current_joint_positions())

        try:
            # Create a combined scene with robot and world obstacles
            combined_scene = trimesh.scene.scene.Scene(base_frame="world_origin")

            # Add robot mesh if we have a joint configuration
            if initial_joint_configuration is not None and self.planner is not None:
                try:
                    config_for_mesh = np.array(initial_joint_configuration).copy()
                    # Zero out base joints to place robot at origin
                    config_for_mesh[:3] = 0.0
                    q_tensor = torch.tensor(config_for_mesh).unsqueeze(0).float().cuda()

                    robot_spheres_batch = self.planner.motion_gen.kinematics.get_robot_as_spheres(
                        q_tensor
                    )
                    if robot_spheres_batch and len(robot_spheres_batch) > 0:
                        robot_spheres = robot_spheres_batch[0]
                        for i, sphere in enumerate(robot_spheres):
                            sphere_mesh = trimesh.creation.icosphere(radius=sphere.radius)
                            sphere_transform = np.eye(4)
                            sphere_transform[:3, 3] = sphere.pose[:3]
                            combined_scene.add_geometry(
                                sphere_mesh,
                                geom_name=f"robot_sphere_{i}",
                                parent_node_name="world_origin",
                                transform=sphere_transform,
                            )
                        log.info(
                            f"Added {len(robot_spheres)} robot collision spheres to visualization"
                        )

                except Exception as e:
                    log.warning(f"Could not add robot mesh: {e}")

            # Add world obstacles to the scene
            try:
                world_scene = WorldConfig.get_scene_graph(world_cfg, process_color=True)
                for geom_name, geom in world_scene.geometry.items():
                    transform = world_scene.graph.get(geom_name)[0]
                    combined_scene.add_geometry(
                        geom,
                        geom_name=geom_name,
                        parent_node_name="world_origin",
                        transform=transform,
                    )
            except Exception as e:
                log.warning(f"Could not add world obstacles: {e}")

            # Export combined scene
            if len(combined_scene.geometry) > 0:
                combined_scene.export(obj_path)
                log.info(f"Successfully saved combined robot + world mesh to {obj_path}")
            else:
                log.debug("Skipping mesh export - scene is empty")

        except ValueError as e:
            if "empty scene" in str(e).lower():
                log.debug("Skipping mesh export - world config is empty")
            else:
                raise

    # ========== Look-at Action (can be overridden) ==========

    def get_look_at_action(self) -> dict[str, Any]:
        """Get action to look at a target.

        Subclasses can override this to implement head tracking.

        Returns:
            Dictionary with head control commands, or empty dict.
        """
        return {}
