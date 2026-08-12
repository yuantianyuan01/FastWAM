import logging
import random
from enum import Enum
from typing import Any

import numpy as np
from curobo.geom.types import Cuboid
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.planner.curobo_planner_client import CuroboClient
from molmo_spaces.policy.solvers.curobo_planner_policy import CuroboPlannerPolicy
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.constants.object_constants import RECEPTACLE_TYPES_THOR
from molmo_spaces.utils.grasps import get_pickup_grasps
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.profiler_utils import Timer

log = logging.getLogger(__name__)


class PickAndPlacePhase(str, Enum):
    PREGRASP = "pregrasp"
    GRASP = "grasp"
    LIFT = "lift"
    PLACE = "place"
    POSTPLACE = "postplace"
    DONE = "done"


class CuroboPickAndPlacePlannerPolicy(CuroboPlannerPolicy, PickAndPlacePlannerPolicy):
    """Curobo-based planner policy for pick and place tasks.

    Inherits common motion planning functionality from CuroboPlannerPolicy
    and task-specific functionality from PickAndPlacePlannerPolicy.
    Planning requests are made to a remote CuroboPlanner server via CuroboClient.
    """

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        # Initialize both parent classes
        CuroboPlannerPolicy.__init__(self, config, task)
        PickAndPlacePlannerPolicy.__init__(self, config, task)

        self.client: CuroboClient | None = None

        self.select_arm()
        self.pre_grasp_poses = self._get_pregrasp_poses()
        if isinstance(self.task, PickAndPlaceTask):
            self.place_poses = self._get_place_poses()

        self.current_phase = PickAndPlacePhase.PREGRASP

        self.grasping_timesteps = 0
        self.opening_timesteps = 0
        self.settle_steps = 0

    @property
    def planners(self) -> dict:
        return {}

    @property
    def is_done(self) -> bool:
        """Property to expose completion state for task checking."""
        return self.current_phase == PickAndPlacePhase.DONE

    def get_phase(self) -> PickAndPlacePhase:
        return self.current_phase

    def get_all_phases(self) -> dict[str, int]:
        """Return list of all phase names."""
        return {phase.value: i for i, phase in enumerate(PickAndPlacePhase)}

    def _get_batch_goal_poses(self) -> np.ndarray | None:
        """Get goal poses for batch planning based on current phase."""
        if self.current_phase == PickAndPlacePhase.PREGRASP:
            return self.pre_grasp_poses
        if self.current_phase == PickAndPlacePhase.PLACE:
            return self.place_poses
        return None

    def _get_pregrasp_poses(self) -> np.ndarray:
        from molmo_spaces.utils.grasp_sample import get_noncolliding_grasp_mask

        task_config = self.config.task_config
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_config.pickup_obj_name)
        model = self.task.env.current_model
        data = self.task.env.current_data

        # Get all grasp poses
        grasp_poses_world = get_pickup_grasps(
            self.task.env, pickup_obj, grasp_libraries=self.config.policy_config.grasp_libraries
        )

        # Get current TCP position
        gripper_mg_id = self.task.env.current_robot.robot_view.get_gripper_movegroup_ids()[0]
        tcp_pose_world = self.task.env.current_robot.robot_view.get_move_group(
            gripper_mg_id
        ).leaf_frame_to_world
        tcp_pose_inv = np.linalg.inv(tcp_pose_world)

        dist_tcp = tcp_pose_inv @ grasp_poses_world
        dists_tcp_p = np.linalg.norm(dist_tcp[:, :3, 3], axis=1)
        dist_tcp_o = R.from_matrix(dist_tcp[:, :3, :3]).magnitude() * 180 / np.pi
        dists_up = grasp_poses_world[:, 2, 2]
        dists_com = np.linalg.norm(
            (np.linalg.inv(pickup_obj.pose) @ grasp_poses_world)[:, :3, 3], axis=1
        )

        dist_total = (
            self.config.policy_config.grasp_pos_cost_weight * dists_tcp_p
            + self.config.policy_config.grasp_rot_cost_weight * dist_tcp_o
            + self.config.policy_config.grasp_vertical_cost_weight * dists_up
            + self.config.policy_config.grasp_com_dist_cost_weight * dists_com
        )

        # Sort and take top N for collision checking
        close_grasp_ids = np.argsort(dist_total, kind="stable")
        n_collision_checks = self.config.policy_config.grasp_collision_max_grasps
        close_grasp_ids = close_grasp_ids[:n_collision_checks]

        # Filter out colliding grasps if enabled (NO IK CHECKS)
        if self.config.policy_config.filter_colliding_grasps:
            with Timer() as collision_check_time:
                noncolliding_grasp_mask = get_noncolliding_grasp_mask(
                    model,
                    data,
                    grasp_poses_world[close_grasp_ids],
                    self.config.policy_config.grasp_collision_batch_size,
                )
            log.info(
                f"Collision-checked {len(close_grasp_ids)} grasps in {collision_check_time.value:.3f}s, "
                f"found {np.sum(noncolliding_grasp_mask)} non-colliding grasps"
            )

            noncolliding_close_grasp_ids = close_grasp_ids[noncolliding_grasp_mask]
            if len(noncolliding_close_grasp_ids) == 0:
                log.warning("No non-colliding grasps found, falling back to all grasp poses")
                noncolliding_close_grasp_ids = close_grasp_ids
        else:
            noncolliding_close_grasp_ids = close_grasp_ids

        # Return ALL non-colliding grasps sorted by cost (not just the best one)
        grasp_poses = grasp_poses_world[noncolliding_close_grasp_ids]

        # Offset pregrasp poses 2cm back along z-axis (away from object)
        z_directions = grasp_poses[:, :3, 2]
        grasp_poses[:, :3, 3] = grasp_poses[:, :3, 3] + z_directions * -(
            self.config.policy_config.pregrasp_z_offset
        )

        return grasp_poses

    def _get_place_poses(self) -> np.ndarray:
        grasp_pose_world = self.pre_grasp_poses
        task_config = self.config.task_config
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_config.pickup_obj_name)
        place_receptacle = om.get_object_by_name(task_config.place_receptacle_name)

        # Get receptacle and object properties (these are the same for all grasps)
        place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, place_receptacle.object_id
        )
        receptacle_top_z = place_receptacle_aabb_center[2] + place_receptacle_aabb_size[2] / 2
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
        )
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2

        # Handle both single pose and batch of poses
        is_batch = grasp_pose_world.ndim == 3
        if not is_batch:
            # Convert single pose to batch for uniform processing
            grasp_pose_world = grasp_pose_world[np.newaxis, ...]

        # Compute clearance offsets for all grasps (N,)
        pickup_obj_clearance_offsets = np.maximum(
            grasp_pose_world[:, 2, 3] - pickup_obj_bottom_z, 0.0
        )

        # Initialize place poses as copies of grasp poses (N, 4, 4)
        place_poses = grasp_pose_world.copy()

        # Set XY positions to center of receptacle for all poses
        place_poses[:, :2, 3] = place_receptacle.position[:2]

        # Set Z heights: receptacle top + clearance offset + small margin
        place_poses[:, 2, 3] = receptacle_top_z + pickup_obj_clearance_offsets + 0.02

        # Offset the EE to ensure the pickup object is in the middle of the receptacle
        # (grasp_pose - object_position) gives the offset from object center to grasp
        place_poses[:, :3, 3] += grasp_pose_world[:, :3, 3] - pickup_obj.position

        # Return in original format (single pose or batch)
        if not is_batch:
            return place_poses[0]
        return place_poses

    def _get_collision_cuboids(self) -> list[Cuboid]:
        """Get cuboid collision geometry for nearby objects.

        The server only supports cuboid obstacles, so all objects are represented
        as axis-aligned bounding boxes in the robot base frame.
        """
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        model = self.task.env.current_model
        data = self.task.env.current_data

        target_object_names: set[str] = set()

        pickup_obj_name = self.config.task_config.pickup_obj_name
        target_object_names.add(pickup_obj_name)
        if isinstance(self.task, PickAndPlaceTask):
            target_object_names.add(self.config.task_config.place_receptacle_name)

        support_below = om.get_support_below(pickup_obj_name, RECEPTACLE_TYPES_THOR)
        target_object_names.add(support_below)

        from molmo_spaces.editor.constants import ALL_PICKUP_TYPES_THOR

        objects_on_surface_below = om.get_objects_that_are_on_top_of_object(
            support_below, pickup_types=ALL_PICKUP_TYPES_THOR
        )
        for obj in objects_on_surface_below:
            target_object_names.add(obj.name)

        for b in om.top_level_bodies():
            name = om.get_object_name(b)
            if om.is_structural(b) and name.startswith("wall"):
                target_object_names.add(name)

        cuboids = []
        for name in target_object_names:
            try:
                obj = om.get_object_by_name(name)
            except Exception as e:
                log.debug(f"Could not get object {name}: {e}")
                continue

            # Skip attached pickup object during place phase
            if (
                name == pickup_obj_name
                and self.current_phase == PickAndPlacePhase.PLACE
                and self.config.policy_config.attach_obj
            ):
                continue

            cuboid = self._create_cuboid_for_object(obj, model, data)
            if cuboid is not None:
                cuboids.append(cuboid)

        log.info(f"Collision geometry: {len(cuboids)} cuboids")
        return cuboids

    def _create_cuboid_for_object(self, obj, model, data) -> Cuboid | None:
        try:
            aabb_center, aabb_size = body_aabb(model, data, obj.body_id, visible_only=False)
        except ValueError:
            log.debug(f"Skipping object {obj.name} (body_id={obj.body_id}) - no geoms found")
            return None
        obj_center = np.concatenate([aabb_center, np.array([1.0, 0.0, 0.0, 0.0])])
        obj_center_base_frame = self._transform_to_base_frame(obj_center)
        obj_center_base_frame_7d = pose_mat_to_7d(obj_center_base_frame)
        dims = aabb_size.tolist()
        return Cuboid(name=obj.name, pose=obj_center_base_frame_7d, dims=dims)

    def _cuboids_to_obstacle_list(self, cuboids: list[Cuboid]) -> list[dict]:
        """Convert curobo Cuboid objects to obstacle dicts for the client."""
        obstacles = []
        for c in cuboids:
            pose = np.array(c.pose).tolist()
            dims = np.array(c.dims).tolist()
            obstacles.append({"name": c.name, "pose": pose, "dims": dims})
        return obstacles

    # ========== Overrides: arm selection / IK / motion planning ==========

    @property
    def _use_local_planner(self) -> bool:
        server_urls = getattr(self.config.policy_config, "server_urls", None)
        return not server_urls

    def select_arm(self) -> None:
        """Select which arm to use and set up the planner (local or remote)."""
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

        left_dist = np.linalg.norm(left_tcp_pose[:3, 3] - pickup_obj_pos)
        right_dist = np.linalg.norm(right_tcp_pose[:3, 3] - pickup_obj_pos)

        selected_arm = "left" if left_dist < right_dist else "right"
        log.info(
            f"Selected {selected_arm} arm (left dist: {left_dist:.3f}m, right dist: {right_dist:.3f}m)"
        )

        self.arm_side = selected_arm

        if self._use_local_planner:
            self._select_arm_local(selected_arm)
        else:
            server_url = random.choice(self.config.policy_config.server_urls)
            log.info(f"Connecting to CuroboPlanner server at {server_url} (arm={selected_arm})")
            server_timeout = getattr(self.config.policy_config, "server_timeout", 120.0)
            self.client = CuroboClient(
                base_url=server_url, arm=selected_arm, timeout=server_timeout
            )

        if selected_arm == "left":
            self.planner_joint_ranges = self.config.policy_config.left_planner_joint_ranges
        else:
            self.planner_joint_ranges = self.config.policy_config.right_planner_joint_ranges

        self.arm_start_idx = self.planner_joint_ranges[f"{self.arm_side}_arm"][0]
        self.arm_end_idx = self.planner_joint_ranges[f"{self.arm_side}_arm"][1]

    def _select_arm_local(self, selected_arm: str) -> None:
        """Instantiate a local CuroboPlanner for the selected arm."""
        from molmo_spaces.planner.curobo_planner import CuroboPlanner

        log.info(f"Instantiating local motion planner for {selected_arm} arm")
        if selected_arm == "left":
            self.planner = CuroboPlanner(
                config=self.config.policy_config.left_curobo_planner_config
            )
        else:
            self.planner = CuroboPlanner(
                config=self.config.policy_config.right_curobo_planner_config
            )

    def solve_ik(self, target_pose: np.ndarray) -> None:
        """Solve IK and create an interpolated trajectory.

        Uses local planner or remote server depending on configuration.

        Args:
            target_pose: 4x4 transformation matrix for target end-effector pose in world frame.
        """
        if self._use_local_planner:
            return CuroboPlannerPolicy.solve_ik(self, target_pose)

        init_config = np.concatenate(
            [
                np.zeros(3),
                self.task.env.current_robot.robot_view.get_move_group(
                    f"{self.arm_side}_arm"
                ).joint_pos,
            ]
        )

        target_pose_7d = self._target_pose_to_base_frame(target_pose)
        joint_config, success = self.client.ik(
            goal_pose=target_pose_7d.tolist(),
            seed_config=init_config.tolist(),
            disable_collision=True,
        )

        current_phase = getattr(self, "current_phase", "unknown")
        if not success:
            raise ValueError(f"Could not solve {current_phase} phase IK.")

        trajectory = self._interpolate_joint_trajectory(
            init_config,
            np.array(joint_config),
            num_steps=10,
        )
        self._transform_traj_to_world_frame(trajectory)
        self.planned_trajectory = trajectory

    def batch_plan_trajectory(self) -> None:
        """Plan trajectory in batches.

        Uses local planner or remote server depending on configuration.
        When using the server, obstacles are passed atomically with each planning
        request to avoid races when multiple workers share the same server.
        """
        if self._use_local_planner:
            return CuroboPlannerPolicy.batch_plan_trajectory(self)

        init_config = np.concatenate(
            [
                np.zeros(3),
                self.task.env.current_robot.robot_view.get_move_group(
                    f"{self.arm_side}_arm"
                ).joint_pos,
            ]
        )

        # Build obstacle list for atomic world update during planning
        obstacles: list[dict] | None = None
        if getattr(self.config.policy_config, "enable_collision_avoidance", False):
            cuboids = self._get_collision_cuboids()
            obstacles = self._cuboids_to_obstacle_list(cuboids)
            log.debug(
                f"Adding {len(cuboids)} cuboids to collision avoidance for phase: {self.current_phase}"
            )

        # Attach pickup object to EE link for place phase if configured
        if self.current_phase == PickAndPlacePhase.PLACE and self.config.policy_config.attach_obj:
            pickup_obj_name = self.config.task_config.pickup_obj_name
            joint_pos_for_attach = np.array(self._get_current_joint_positions()).copy()
            base_range = self.planner_joint_ranges["base"]
            joint_pos_for_attach[base_range[0] : base_range[1]] = 0.0
            self.client.attach_object(
                object_names=[pickup_obj_name],
                joint_position=joint_pos_for_attach.tolist(),
                attach_link_names=[f"attached_object_{self.arm_side}"],
            )
            log.info(
                f"Attached object '{pickup_obj_name}' to link 'attached_object_{self.arm_side}'"
            )

        # Get goal poses based on current phase
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

            batch_len = len(batch_base_frame_7d)
            start_states = [init_config.tolist()] * batch_len

            log.info(
                f"Planning batch of {batch_len} {current_phase} poses with {self.arm_side} arm"
            )
            trajectories, successes = self.client.motion_plan_batch(
                joint_positions=start_states,
                goal_poses=batch_base_frame_7d,
                obstacles=obstacles,
            )

            num_successes = sum(successes)
            log.info(
                f"{self.arm_side.capitalize()} arm: {num_successes}/{batch_len} "
                f"{current_phase} planning successes"
            )

            for success, trajectory in zip(successes, trajectories):
                if not success or not trajectory:
                    continue
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

    # ========== Phase execution (unchanged) ==========

    def _execute_pre_grasp_phase(self) -> dict[str, Any]:
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory({f"{self.arm_side}_gripper": -100})
            else:
                self.current_phase = PickAndPlacePhase.GRASP
                self.planned_trajectory = None
                self.trajectory_index = 0
                self.steps_spent_in_waypoint = 0
                return {}
        log.info("ENTERING PREGRASP PHASE")
        self.batch_plan_trajectory()
        return self._execute_trajectory({f"{self.arm_side}_gripper": -100})

    def _execute_grasp_phase(self) -> dict[str, Any]:
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory({f"{self.arm_side}_gripper": -100})
            else:
                if self.grasping_timesteps < self.config.policy_config.max_grasping_timesteps:
                    self.grasping_timesteps += 1
                    return {f"{self.arm_side}_gripper": 100}
                else:
                    if self._grasping_something():
                        log.info(
                            f"Object successfully grasped after {self.grasping_timesteps} timesteps"
                        )
                        self.current_phase = PickAndPlacePhase.LIFT
                        self.planned_trajectory = self.planned_trajectory[::-1]
                        self.trajectory_index = 0
                        self.steps_spent_in_waypoint = 0
                        self.grasping_timesteps = 0
                    else:
                        log.warning(
                            f"Object not grasped after {self.grasping_timesteps} timesteps, "
                            "returning to pre-grasp"
                        )
                        self.pre_grasp_poses = self._get_pregrasp_poses()
                        self.current_phase = PickAndPlacePhase.PREGRASP
                        self.planned_trajectory = None
                        self.trajectory_index = 0
                        self.steps_spent_in_waypoint = 0
                        self.grasping_timesteps = 0
                    return {}
        log.info("ENTERING GRASP PHASE")

        tcp_pose_world = self.task.env.current_robot.robot_view.get_move_group(
            f"{self.arm_side}_gripper"
        ).leaf_frame_to_world

        grasp_pose_world = tcp_pose_world.copy()
        z_direction = grasp_pose_world[:3, 2]
        grasp_pose_world[:3, 3] = grasp_pose_world[:3, 3] + z_direction * (
            self.config.policy_config.pregrasp_z_offset + 0.01
        )
        self.solve_ik(grasp_pose_world)
        return self._execute_trajectory({f"{self.arm_side}_gripper": -100})

    def _execute_lift_phase(self) -> dict[str, Any]:
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory({f"{self.arm_side}_gripper": 100})
            else:
                if not self._grasping_something():
                    # Object not grasped, move back to reach pre-grasp phase
                    log.warning("Object not grasped during lift phase, returning to pre-grasp")
                    self.pre_grasp_poses = self._get_pregrasp_poses()
                    self.current_phase = PickAndPlacePhase.PREGRASP
                    self.planned_trajectory = None
                    self.trajectory_index = 0
                    self.steps_spent_in_waypoint = 0
                    self.grasping_timesteps = 0
                    return {}
                if isinstance(self.task, PickAndPlaceTask):
                    self.current_phase = PickAndPlacePhase.PLACE
                else:
                    self.current_phase = PickAndPlacePhase.DONE
                self.planned_trajectory = None
                self.trajectory_index = 0
                self.steps_spent_in_waypoint = 0
                return {}
        log.info("ENTERING LIFT PHASE")
        tcp_pose_world = self.task.env.current_robot.robot_view.get_move_group(
            f"{self.arm_side}_gripper"
        ).leaf_frame_to_world

        lift_pose_world = tcp_pose_world.copy()
        lift_pose_world[:3, 3] = lift_pose_world[:3, 3] + np.array([0, 0, 1]) * (0.05)
        self.solve_ik(lift_pose_world)
        return self._execute_trajectory({f"{self.arm_side}_gripper": 100})

    def _execute_postplace_phase(self) -> dict[str, Any]:
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory({f"{self.arm_side}_gripper": -100})
            else:
                if self.settle_steps < self.config.policy_config.max_settle_steps:
                    self.settle_steps += 1
                    return {}
                self.current_phase = PickAndPlacePhase.DONE
                self.planned_trajectory = None
                self.trajectory_index = 0
                self.steps_spent_in_waypoint = 0
                return {}
        log.info("ENTERING POST PLACE PHASE")
        tcp_pose_world = self.task.env.current_robot.robot_view.get_move_group(
            f"{self.arm_side}_gripper"
        ).leaf_frame_to_world
        lift_pose_world = tcp_pose_world.copy()
        lift_pose_world[:3, 3] = lift_pose_world[:3, 3] + np.array([0, 0, 1]) * (0.05)
        self.solve_ik(lift_pose_world)
        return self._execute_trajectory({f"{self.arm_side}_gripper": -100})

    def _execute_place_phase(self) -> dict[str, Any]:
        if not self._grasping_something():
            # Object not grasped, move back to reach pre-grasp phase
            log.warning("Object lost during place phase, returning to pre-grasp")
            self.pre_grasp_poses = self._get_pregrasp_poses()
            self.current_phase = PickAndPlacePhase.PREGRASP
            self.planned_trajectory = None
            self.trajectory_index = 0
            self.steps_spent_in_waypoint = 0
            return {}
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory({f"{self.arm_side}_gripper": 100})
            else:
                if self.opening_timesteps < self.config.policy_config.max_opening_timesteps:
                    self.opening_timesteps += 1
                    return {f"{self.arm_side}_gripper": -100}
                elif self.opening_timesteps == self.config.policy_config.max_opening_timesteps:
                    self.current_phase = PickAndPlacePhase.POSTPLACE
                    self.planned_trajectory = None
                    self.trajectory_index = 0
                    self.steps_spent_in_waypoint = 0
                    return {}
        log.info("ENTERING PLACE PHASE")
        self.batch_plan_trajectory()
        return self._execute_trajectory({f"{self.arm_side}_gripper": 100})

    def get_action(self, info: dict[str, Any]) -> dict[str, Any]:
        action_cmd = self.task.env.current_robot.robot_view.get_noop_ctrl_dict()
        if self.current_phase == PickAndPlacePhase.PREGRASP:
            action_cmd.update(self._execute_pre_grasp_phase())
        elif self.current_phase == PickAndPlacePhase.GRASP:
            action_cmd.update(self._execute_grasp_phase())
        elif self.current_phase == PickAndPlacePhase.LIFT:
            action_cmd.update(self._execute_lift_phase())
        elif self.current_phase == PickAndPlacePhase.PLACE:
            action_cmd.update(self._execute_place_phase())
        elif self.current_phase == PickAndPlacePhase.POSTPLACE:
            action_cmd.update(self._execute_postplace_phase())
        elif self.current_phase == PickAndPlacePhase.DONE:
            return {"done": True}
        else:
            raise ValueError(f"Unknown phase: {self.current_phase}")
        action_cmd = self.clip_to_velocity_constraint(action_cmd)
        return action_cmd

    def reset(self, reset_retries: bool = True):
        # Call parent reset
        CuroboPlannerPolicy.reset(self)

        self.current_arm = None
        self.current_phase = PickAndPlacePhase.PREGRASP
        self.grasping_timesteps = 0
        self.opening_timesteps = 0
        dummy_pose = np.eye(4)
        self.target_poses = {
            "grasp": dummy_pose,
            "pregrasp": dummy_pose,
            "place": dummy_pose,
        }
        self.settle_steps = 0
        from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
            NoopAction,
        )

        self.action_primitives = [NoopAction(self.task.env.current_robot.robot_view, 0.0)]

        # if self.client is not None:
        #     self.client.reset()
