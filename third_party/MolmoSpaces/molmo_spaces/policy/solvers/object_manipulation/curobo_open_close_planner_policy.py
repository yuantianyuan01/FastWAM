import logging
import random
from enum import Enum
from typing import Any

import mujoco
import numpy as np
from curobo.geom.types import Cuboid
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.planner.curobo_planner_client import CuroboClient
from molmo_spaces.policy.solvers.curobo_planner_policy import CuroboPlannerPolicy
from molmo_spaces.policy.solvers.object_manipulation.open_close_planner_policy import (
    OpenClosePlannerPolicy,
)
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.articulation_utils import (
    gather_joint_info,
    step_circular_path,
    step_linear_path,
)
from molmo_spaces.utils.constants.object_constants import (
    EXTENDED_ARTICULATION_TYPES_THOR,
    RECEPTACLE_TYPES_THOR,
)
from molmo_spaces.utils.grasps import get_joint_grasps
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb, descendant_geoms, geom_aabb
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.profiler_utils import Timer

log = logging.getLogger(__name__)


class OpenClosePhase(str, Enum):
    HEIGHT_SELECTION = "height_selection"
    PREGRASP = "pregrasp"
    GRASP = "grasp"
    ARTICULATE = "articulate"
    POSTARTICULATE = "place"
    DONE = "done"


class CuroboOpenClosePlannerPolicy(CuroboPlannerPolicy, OpenClosePlannerPolicy):
    """Curobo-based planner policy for open/close articulated object tasks.

    Inherits common motion planning functionality from CuroboPlannerPolicy
    and task-specific functionality from OpenClosePlannerPolicy.
    Planning requests are made to a remote CuroboPlanner server via CuroboClient.
    """

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        # Initialize both parent classes
        CuroboPlannerPolicy.__init__(self, config, task)
        OpenClosePlannerPolicy.__init__(self, config, task)

        self.client: CuroboClient | None = None
        self._base_lock_joints: dict = {}  # fetched from server in select_arm()

        self.select_arm()
        self.pre_grasp_poses = None  # computed after height adjustment

        self.current_phase = OpenClosePhase.HEIGHT_SELECTION

        self.articulation_pose_index = 0
        self.grasping_timesteps = 0
        self.opening_timesteps = 0
        self.settle_steps = 0
        self.height_adjustment_steps = 0
        self._height_initial: float | None = None
        self._height_target: float | None = None

    @property
    def planners(self) -> dict:
        return {}

    @property
    def is_done(self) -> bool:
        """Property to expose completion state for task checking."""
        return self.current_phase == OpenClosePhase.DONE

    def get_phase(self) -> OpenClosePhase:
        return self.current_phase

    def get_all_phases(self) -> dict[str, int]:
        """Return list of all phase names."""
        return {phase.value: i for i, phase in enumerate(OpenClosePhase)}

    def _get_batch_goal_poses(self) -> np.ndarray | None:
        """Get goal poses for batch planning based on current phase."""
        if self.current_phase == OpenClosePhase.PREGRASP:
            return self.pre_grasp_poses
        if self.current_phase == OpenClosePhase.ARTICULATE:
            return np.array([self.articulation_poses[self.articulation_pose_index]])
        return None

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

            # Fetch base lock_joints from server once; merged with live torso values per-request.
            self._base_lock_joints = self.client.get_lock_joints()
            log.info(f"Base lock_joints from server: {self._base_lock_joints}")

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

    def _get_lock_joints(self) -> dict:
        """Build the complete lock_joints dict: base values + current torso positions.

        This is called immediately before each IK/plan request so the torso position
        is sampled at the time of the call and passed atomically to the server,
        avoiding races when multiple workers share the same server.
        """
        torso_joint_pos = self.task.env.current_robot.robot_view.get_move_group("torso").joint_pos
        lock_joints = dict(self._base_lock_joints)
        lock_joints.update(
            {
                "torso_1": float(torso_joint_pos[1]),
                "torso_2": float(torso_joint_pos[2]),
                "torso_3": float(torso_joint_pos[3]),
            }
        )
        return lock_joints

    def _cuboids_to_obstacle_list(self, cuboids: list[Cuboid]) -> list[dict]:
        """Convert curobo Cuboid objects to obstacle dicts for the client."""
        obstacles = []
        for c in cuboids:
            pose = np.array(c.pose).tolist()
            dims = np.array(c.dims).tolist()
            obstacles.append({"name": c.name, "pose": pose, "dims": dims})
        return obstacles

    def solve_ik(self, target_pose: np.ndarray) -> None:
        """Solve IK and create an interpolated trajectory.

        Uses local planner or remote server depending on configuration.
        When using the server, lock joints and obstacles are passed atomically
        with the request to avoid races when multiple workers share the same server.

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
        lock_joints = self._get_lock_joints()
        joint_config, success = self.client.ik(
            goal_pose=target_pose_7d.tolist(),
            seed_config=init_config.tolist(),
            disable_collision=True,
            lock_joints=lock_joints,
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
        When using the server, lock joints and obstacles are passed atomically
        with each planning request to avoid races when multiple workers share
        the same server.
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
            cuboids = self._get_nearby_object_collision_geometry()
            obstacles = self._cuboids_to_obstacle_list(cuboids)
            log.debug(
                f"Adding {len(cuboids)} cuboids to collision avoidance for phase: {self.current_phase}"
            )

        # Sample lock_joints atomically with the planning request
        lock_joints = self._get_lock_joints()

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
                lock_joints=lock_joints,
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

    # ========== Grasp pose computation ==========

    def _get_pregrasp_poses(self) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R

        from molmo_spaces.utils.grasp_sample import get_noncolliding_grasp_mask

        task_config = self.config.task_config
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_config.pickup_obj_name)
        model = self.task.env.current_model
        data = self.task.env.current_data

        # Get all grasp poses
        grasp_poses_world, object_pose = get_joint_grasps(
            self.task.env,
            pickup_obj,
            self.config.task_config.joint_index,
            grasp_libraries=self.config.policy_config.grasp_libraries,
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
            (np.linalg.inv(object_pose) @ grasp_poses_world)[:, :3, 3], axis=1
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

    def _get_articulation_poses(self) -> list[np.ndarray]:
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        grasp_pose_world = self.task.env.current_robot.robot_view.get_move_group(
            f"{self.arm_side}_gripper"
        ).leaf_frame_to_world
        pickup_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
        joint_info = gather_joint_info(
            self.task._env.mj_model,
            self.task._env.mj_datas[0],
            pickup_obj.joint_ids[self.config.task_config.joint_index],
        )
        path_dict = {}
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

            path_dict = step_circular_path(
                grasp_pos, grasp_quat, joint_info, max_joint_angle, n_waypoints=4
            )

        elif joint_info["joint_type"] == mujoco.mjtJoint.mjJNT_SLIDE:
            grasp_pos = grasp_pose_world[:3, 3]
            grasp_quat = R.from_matrix(grasp_pose_world[:3, :3]).as_quat(scalar_first=True)

            """ Z Axis should be the same"""
            # Transform joint axis from local body frame to world frame
            joint_axis_world = joint_info["joint_body_orientation"] @ joint_info["joint_axis"]
            joint_direction = -joint_axis_world
            normalize_dir_axis = joint_direction / np.linalg.norm(joint_direction)

            current_joint_pos = joint_info["joint_pos"]

            if self.config.task_type == "open":
                max_joint_angle = joint_info["max_range"]
            elif self.config.task_type == "close":
                max_joint_angle = 0

            path_dict = step_linear_path(
                to_handle_dist=normalize_dir_axis * (max_joint_angle - current_joint_pos),
                current_pos=grasp_pos,
                current_quat=grasp_quat,
                step_size=0.1,
                is_reverse=True,
            )

        else:
            raise ValueError(f"Unknown joint type: {joint_info['joint_type']}")
        articulation_poses = []
        for mocap_pos, mocap_quat in zip(path_dict["mocap_pos"], path_dict["mocap_quat"]):
            pose = np.eye(4)
            pose[:3, 3] = mocap_pos
            pose[:3, :3] = R.from_quat(mocap_quat, scalar_first=True).as_matrix()
            articulation_poses.append(pose)
        self.articulation_pose_index = 0
        return articulation_poses

    def _select_height(self) -> float:
        """Compute target torso height."""
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
        target_pos_z = pickup_obj.position[2]
        current_gripper_z = self.task.env.current_robot.robot_view.get_move_group(
            f"{self.arm_side}_gripper"
        ).leaf_frame_to_world[2, 3]

        torso_ctrl = self.task.env.current_robot.controllers["torso"]
        # Object below gripper → lower torso to bottom (max_height); otherwise raise to top (0.0).
        target_height = torso_ctrl.max_height if target_pos_z < current_gripper_z else 0.0
        log.info(
            f"Height selection: object_z={target_pos_z:.3f}, gripper_z={current_gripper_z:.3f}, "
            f"target_height={target_height:.3f}"
        )

        return target_height

    def _is_articulation_complete(self) -> bool:
        """Check if the joint has reached the desired open/close threshold.

        Handles both positive [0, max] and negative [-max, 0] joint ranges.
        Closed position is always at 0.
        """
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
        joint_info = gather_joint_info(
            self.task._env.mj_model,
            self.task._env.mj_datas[0],
            pickup_obj.joint_ids[self.config.task_config.joint_index],
        )

        current_joint_pos = joint_info["joint_pos"]
        joint_range = joint_info["joint_range"]
        joint_range_float = np.abs(joint_range[1] - joint_range[0])

        # Calculate percent open (same logic as task's get_reward)
        # abs() handles both positive [0, 1.57] and negative [-1.57, 0] ranges
        percent_open = np.abs(current_joint_pos) / joint_range_float

        threshold = self.config.task_config.task_success_threshold

        if self.config.task_type == "open":
            target_joint_pos = joint_range[np.argmax(np.abs(joint_range))]
            log.debug(
                f"[ARTICULATION CHECK] task=open current_joint_pos={current_joint_pos:.4f}, "
                f"target_joint_pos={target_joint_pos:.4f}, joint_range={joint_range}, "
                f"percent_open={percent_open:.3f}, threshold={threshold:.3f}"
            )
            # For opening, check if we've reached the threshold percentage
            is_complete = percent_open >= threshold
        elif self.config.task_type == "close":
            log.debug(
                f"[ARTICULATION CHECK] task=close current_joint_pos={current_joint_pos:.4f}, "
                f"target_joint_pos=0.0, joint_range={joint_range}, "
                f"percent_open={percent_open:.3f}, threshold={threshold:.3f}"
            )
            # For closing, check if percent_open is below (1 - threshold)
            is_complete = percent_open <= (1 - threshold)
        else:
            return False

        if is_complete:
            log.debug(
                f"Articulation complete: percent_open={percent_open:.3f}, "
                f"threshold={threshold:.3f}, task_type={self.config.task_type}"
            )
        return is_complete

    def _get_nearby_object_collision_geometry(self) -> list[Cuboid]:
        types_to_add = EXTENDED_ARTICULATION_TYPES_THOR + RECEPTACLE_TYPES_THOR
        om = self.task.env.object_managers[self.task.env.current_batch_index]

        model = self.task.env.current_model
        data = self.task.env.current_data

        cuboids = []

        target_object_names = set()

        for obj in om.get_objects_of_type(types_to_add):
            target_object_names.add(obj.name)

        all_scene_objects = om.top_level_bodies()
        scene_obj_names = [om.get_object_name(b) for b in all_scene_objects]
        for obj in scene_obj_names:
            if "counter" in obj:
                target_object_names.add(obj)

        pickup_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
        pickup_aabb_center, pickup_aabb_size = body_aabb(
            model, data, pickup_obj.body_id, visible_only=False
        )

        for name in target_object_names:
            try:
                obj = om.get_object_by_name(name)
            except Exception as e:
                log.debug(f"Could not get object {name}: {e}")
                continue
            if name != self.config.task_config.pickup_obj_name:
                if "counter" in name:
                    cuboid = self._create_cuboid_for_best_overlapping_geom(
                        obj, model, data, pickup_aabb_center, pickup_aabb_size
                    )
                    if cuboid is not None:
                        cuboids.append(cuboid)
                else:
                    cuboid = self._create_cuboid_for_object(obj, model, data)
                    if cuboid is not None:
                        cuboids.append(cuboid)

        log.info(f"Collision geometry: {len(cuboids)} cuboids")
        return cuboids

    def _create_cuboid_for_best_overlapping_geom(
        self, obj, model, data, pickup_aabb_center: np.ndarray, pickup_aabb_size: np.ndarray
    ) -> Cuboid | None:
        """Return a cuboid for the single counter geom with the most XY overlap with the pickup object."""
        all_geom_ids = descendant_geoms(model, obj.body_id, visible_only=False)
        geom_ids = [
            g for g in all_geom_ids if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
        ]
        best_geom_id = None
        best_overlap = -1.0
        best_aabb_center = None
        best_aabb_size = None

        p_min = pickup_aabb_center[:2] - pickup_aabb_size[:2] / 2
        p_max = pickup_aabb_center[:2] + pickup_aabb_size[:2] / 2

        for geom_id in geom_ids:
            try:
                aabb_center, aabb_size = geom_aabb(model, data, [geom_id])
            except Exception as e:
                log.debug(f"Skipping geom {geom_id} of {obj.name}: {e}")
                continue
            if np.all(aabb_size == 0):
                continue
            g_min = aabb_center[:2] - aabb_size[:2] / 2
            g_max = aabb_center[:2] + aabb_size[:2] / 2
            overlap_x = max(0.0, min(g_max[0], p_max[0]) - max(g_min[0], p_min[0]))
            overlap_y = max(0.0, min(g_max[1], p_max[1]) - max(g_min[1], p_min[1]))
            overlap = overlap_x * overlap_y
            if overlap > best_overlap:
                best_overlap = overlap
                best_geom_id = geom_id
                best_aabb_center = aabb_center
                best_aabb_size = aabb_size

        if best_geom_id is None:
            return None

        obj_center = np.concatenate([best_aabb_center, np.array([1.0, 0.0, 0.0, 0.0])])
        obj_center_base_frame = self._transform_to_base_frame(obj_center)
        obj_center_base_frame_7d = pose_mat_to_7d(obj_center_base_frame)
        return Cuboid(
            name=f"{obj.name}_geom_{best_geom_id}",
            pose=obj_center_base_frame_7d,
            dims=best_aabb_size.tolist(),
        )

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

    # ========== Phase execution ==========

    def _execute_height_selection_phase(self) -> dict[str, Any]:
        max_steps = getattr(self.config.policy_config, "max_height_adjustment_steps", 10)

        if self._height_target is None:
            log.info("ENTERING HEIGHT SELECTION PHASE")
            self._height_target = self._select_height()
            torso_joint_pos = self.task.env.current_robot.robot_view.get_move_group(
                "torso"
            ).joint_pos
            self._height_initial = float(torso_joint_pos[1])
            log.info(
                f"Interpolating torso height {self._height_initial:.3f} → {self._height_target:.3f} "
                f"over {max_steps} steps"
            )

        self.height_adjustment_steps += 1
        alpha = self.height_adjustment_steps / max_steps
        self.current_height = self._height_initial + alpha * (
            self._height_target - self._height_initial
        )

        if self.height_adjustment_steps >= max_steps:
            log.info(
                f"Height adjustment complete after {self.height_adjustment_steps} steps, "
                "transitioning to PREGRASP"
            )
            self.current_height = self._height_target
            self.pre_grasp_poses = self._get_pregrasp_poses()
            self.current_phase = OpenClosePhase.PREGRASP
            self.height_adjustment_steps = 0
            self._height_initial = None
            self._height_target = None
            # Return {} here so get_action sends torso=current_height (the final
            # target) as a standalone step.  PREGRASP planning starts on the next
            # call, by which point current_height is already stable at the target
            # and the robot won't experience an additional sink at phase transition.
            return {}

        return {}

    def _execute_pre_grasp_phase(self) -> dict[str, Any]:
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory({f"{self.arm_side}_gripper": -100})
            else:
                self.current_phase = OpenClosePhase.GRASP
                self.planned_trajectory = None
                self.trajectory_index = 0
                self.steps_spent_in_waypoint = 0
                return self._execute_grasp_phase()
        log.info("ENTERING PREGRASP PHASE")
        self.batch_plan_trajectory()
        return self._execute_trajectory({f"{self.arm_side}_gripper": -100})

    def _execute_grasp_phase(self) -> dict[str, Any]:
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory({f"{self.arm_side}_gripper": -100})
            else:
                if self._grasping_something():
                    log.info(
                        f"Object successfully grasped after {self.grasping_timesteps} timesteps"
                    )
                    self.articulation_poses = self._get_articulation_poses()
                    self.current_phase = OpenClosePhase.ARTICULATE
                    self.planned_trajectory = self.planned_trajectory[::-1]
                    self.trajectory_index = 0
                    self.steps_spent_in_waypoint = 0
                    self.grasping_timesteps = 0
                    return self._execute_articulate_phase()
                elif self.grasping_timesteps < self.config.policy_config.max_grasping_timesteps:
                    self.grasping_timesteps += 1
                    return {f"{self.arm_side}_gripper": 100}
                else:
                    log.warning(
                        f"Object not grasped after {self.grasping_timesteps} timesteps, "
                        "returning to pre-grasp"
                    )
                    self.pre_grasp_poses = self._get_pregrasp_poses()
                    self.current_phase = OpenClosePhase.PREGRASP
                    self.planned_trajectory = None
                    self.trajectory_index = 0
                    self.steps_spent_in_waypoint = 0
                    self.grasping_timesteps = 0
                    return self._execute_pre_grasp_phase()
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

    def _execute_articulate_phase(self) -> dict[str, Any]:
        if (
            self.articulation_pose_index == len(self.articulation_poses)
            or self._is_articulation_complete()
        ):
            self.current_phase = OpenClosePhase.POSTARTICULATE
            self.planned_trajectory = None
            self.trajectory_index = 0
            self.steps_spent_in_waypoint = 0
            return self._execute_postarticulate_phase()
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory({f"{self.arm_side}_gripper": 100})
            else:
                if not self._grasping_something():
                    # Object not grasped, move back to reach pre-grasp phase
                    log.warning(
                        "Object not grasped during articulate phase, returning to pre-grasp"
                    )
                    self.pre_grasp_poses = self._get_pregrasp_poses()
                    self.current_phase = OpenClosePhase.PREGRASP
                    self.planned_trajectory = None
                    self.trajectory_index = 0
                    self.steps_spent_in_waypoint = 0
                    return self._execute_pre_grasp_phase()
                else:
                    self.planned_trajectory = None
                    self.trajectory_index = 0
                    self.steps_spent_in_waypoint = 0
                    self.articulation_pose_index += 1
                    return self._execute_articulate_phase()
        log.info(f"EXECUTING ARTICULATE PHASE {self.articulation_pose_index}")

        # Log current TCP position
        tcp_pose_world = self.task.env.current_robot.robot_view.get_move_group(
            f"{self.arm_side}_gripper"
        ).leaf_frame_to_world
        current_tcp_pos = tcp_pose_world[:3, 3]
        current_tcp_quat = R.from_matrix(tcp_pose_world[:3, :3]).as_quat(scalar_first=True)
        log.info(f"[ARTICULATE] Current TCP position: {current_tcp_pos}, quat: {current_tcp_quat}")

        # Log target articulation pose
        target_pose = self.articulation_poses[self.articulation_pose_index]
        target_pos = target_pose[:3, 3]
        target_quat = R.from_matrix(target_pose[:3, :3]).as_quat(scalar_first=True)
        log.info(f"[ARTICULATE] Target position: {target_pos}, quat: {target_quat}")

        # Clear pregrasp pose visualizations and visualize all articulation poses
        if self.task.viewer:
            self.task.viewer.user_scn.ngeom = 0
        self._show_poses(np.array(self.articulation_poses), style="tcp", color=(0, 1, 0, 1))
        if self.task.viewer:
            self.task.viewer.sync()

        self.solve_ik(target_pose)
        return self._execute_trajectory({f"{self.arm_side}_gripper": 100})

    def _execute_postarticulate_phase(self) -> dict[str, Any]:
        if self.planned_trajectory is not None:
            if self.trajectory_index < len(self.planned_trajectory):
                return self._execute_trajectory({f"{self.arm_side}_gripper": -100})
            else:
                if self.settle_steps < self.config.policy_config.max_settle_steps:
                    self.settle_steps += 1
                    return {}
                self.current_phase = OpenClosePhase.DONE
                self.planned_trajectory = None
                self.trajectory_index = 0
                self.steps_spent_in_waypoint = 0
                return {}
        log.info("ENTERING POST ARTICULATE PHASE")
        tcp_pose_world = self.task.env.current_robot.robot_view.get_move_group(
            f"{self.arm_side}_gripper"
        ).leaf_frame_to_world
        lift_pose_world = tcp_pose_world.copy()
        lift_pose_world[:3, 3] = lift_pose_world[:3, 3] + np.array([0, 0, 1]) * (0.05)
        self.solve_ik(lift_pose_world)
        return self._execute_trajectory({f"{self.arm_side}_gripper": -100})

    def get_action(self, info: dict[str, Any]) -> dict[str, Any]:
        action_cmd = self.task.env.current_robot.robot_view.get_noop_ctrl_dict()
        if self.current_phase == OpenClosePhase.HEIGHT_SELECTION:
            action_cmd.update(self._execute_height_selection_phase())
        elif self.current_phase == OpenClosePhase.PREGRASP:
            action_cmd.update(self._execute_pre_grasp_phase())
        elif self.current_phase == OpenClosePhase.GRASP:
            action_cmd.update(self._execute_grasp_phase())
        elif self.current_phase == OpenClosePhase.ARTICULATE:
            action_cmd.update(self._execute_articulate_phase())
        elif self.current_phase == OpenClosePhase.POSTARTICULATE:
            action_cmd.update(self._execute_postarticulate_phase())
        elif self.current_phase == OpenClosePhase.DONE:
            return {"done": True}
        else:
            raise ValueError(f"Unknown phase: {self.current_phase}")

        action_cmd["torso"] = np.array([self.current_height])
        action_cmd = self.clip_to_velocity_constraint(action_cmd)
        return action_cmd

    def reset(self, reset_retries: bool = True):
        # Call parent reset
        CuroboPlannerPolicy.reset(self)

        self.current_arm = None
        self.current_phase = OpenClosePhase.HEIGHT_SELECTION
        self.grasping_timesteps = 0
        self.opening_timesteps = 0
        self.articulation_pose_index = 0
        self.height_adjustment_steps = 0
        self.current_height = 0.0
        self._height_initial = None
        self._height_target = None
        self.pre_grasp_poses = None
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
