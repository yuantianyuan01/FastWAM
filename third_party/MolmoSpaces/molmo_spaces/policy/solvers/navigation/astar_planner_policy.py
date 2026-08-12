import logging

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.planner.astar_planner import AStarPlanner
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.util_samplers.navgoal_sampler import NavGoalSampler
from molmo_spaces.utils.linalg_utils import normalize_ang_error

log = logging.getLogger(__name__)


"""
This planner policy relies on the computation of a sparse set of (x,y) waypoints
from the AStarPlanner. It then builds a navigation plan by interleaving rotation
and translation phases with interpolated waypoints (either by slerp for rotation
phases, or by linear interpolation for translation ones).

Note: Even though rotation phases could be ideally skipped for waypoints that are
colinear (i.e., interpolated between two spatially distant planner waypoints), as
the agent orientation is constant, we still allow a correction, since the controller
can deviate from the ideal plan. We introduce corrections by means of two mechanisms:
  1. Enforcing some intermediate waypoints between spatially distant ones, regardless
     of Euclidean distance (via `path_interpolation_density`, which can be kept low,
     e.g. 1)
  2. Enforcing intermediate waypoint to keep consecutive ones under a limit distance
     (via `path_max_inter_waypoint_dist`, which should be kept low enough to prevent
     too much overshooting by the controller)

Note 2: For rotation, we limit the maximal arc length via `path_max_inter_waypoint_angle`,
which we set by default to 10 degrees. This, combined with a fix in the holonomic
base control to express the current yaw according to the intended motion direction to
prevent wrapping errors, leads to smooth rotations.

Note 3: the current replanning heuristic is brittle, so we can just switch it off by
e.g. setting `plan_fail_after_waypoint_steps` to a value larger than the task horizon.

Some TODOs:
 - Use joint_pos_rel to decide upon failure to complete previous action
 - Select the nearest plannable random location near target instead of first one with valid plan
"""


class AStarPlannerPolicy(PlannerPolicy):
    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)

        self._target_pos_quat = None
        self._nav_goal_sampler = None

        self.config.policy_config.planner_config.agent_radius = (
            self.config.task_sampler_config.robot_safety_radius
        )

        self.nav_planner = AStarPlanner(
            self.config.policy_config.planner_config, self.task.env.current_model_path
        )

        self._current_waypoint = 0
        self._reached_waypoints = 0
        self._nav_plan = None
        self._target_pos_quat = None
        self._dists_to_waypoint = []
        self._retries_left = self.config.policy_config.plan_max_retries
        self._target_object = None
        self._candidate_objs = None
        self._skipped_candidates = set()
        self._replan_after = None

        self.robot_view = task.env.current_robot.robot_view

    def planners(self):
        return self.nav_planner

    def reset(self):
        self._current_waypoint = 0
        self._reached_waypoints = 0
        self._nav_plan = None
        self._target_pos_quat = None
        self._dists_to_waypoint = []
        self._retries_left = self.config.policy_config.plan_max_retries
        self._target_object = None
        self._candidate_objs = None
        self._skipped_candidates = set()
        self._replan_after = None
        self.nav_planner.blacklist.clear()

    @property
    def retry_count(self) -> int:
        return self.config.policy_config.plan_max_retries - self._retries_left

    def get_phase(self) -> str:
        return "unknown"

    def get_all_phases(self) -> dict[str | int]:
        return {
            "unknown": 0,
        }

    @property
    def candidate_objs(self) -> list[MlSpacesObject]:
        if self._candidate_objs is None:
            batch_idx = self.task.env.current_batch_index
            self._candidate_objs = self.task.nav_objs[batch_idx]

        return self._candidate_objs

    def skip_candidate(self, obj_name):
        self._skipped_candidates.add(obj_name)

    @property
    def target_object(self) -> MlSpacesObject:
        """Get the nearest navigation target object for the current batch."""
        if (
            self._target_object is None
            or not self.config.policy_config.plan_stick_to_original_target
            or self._target_object.name in self._skipped_candidates
        ):
            if len(self.candidate_objs) == 1:
                self._target_object = self.candidate_objs[0]
            else:
                # If more than one candidate, return the nearest remaining one
                batch_idx = self.task.env.current_batch_index
                priority = self.task.get_nav_object_priority(batch_idx)
                for obj in priority:
                    if obj.name in self._skipped_candidates:
                        continue
                    else:
                        self._target_object = obj
                        break
                else:
                    # Fallback: just pick the nearest
                    self._target_object = priority[0] if priority else None

        return self._target_object

    @property
    def nav_goal_sampler(self) -> NavGoalSampler:
        if self._nav_goal_sampler is None:
            self._nav_goal_sampler = NavGoalSampler(
                self.nav_planner.map, check_target_in_view=False, camera_name="head_camera"
            )

        return self._nav_goal_sampler

    @property
    def target_pos_quat(self):
        if self._target_pos_quat is None:
            self.nav_goal_sampler.set_target(self.target_object)
            self.nav_goal_sampler.set_robot_view(self.robot_view)
            for attempt in range(5):
                self._target_pos_quat = self.nav_goal_sampler.sample()
                if self._target_pos_quat is not None:
                    log.info(
                        f"[A* PLAN] Target position quaternion: {self._target_pos_quat} after {attempt} attempts"
                    )
                    break

        return self._target_pos_quat

    def stop_plan(self, waypoints: np.ndarray) -> np.ndarray:
        r = self.config.policy_config.path_min_dist_to_target_center

        if r == 0.0:
            return waypoints

        cc = self.target_object.position[:2]

        # 1. find first waypoint entering circle and not leaving again
        last_out = None
        for i in reversed(range(len(waypoints))):
            if np.linalg.norm(waypoints[i] - cc) > r:
                last_out = i
                break
        else:
            # all waypoints were under r, so use the first two waypoints only
            return waypoints[:2]

        # if all waypoints are further than r, keep them all
        if last_out == len(waypoints) - 1:
            return waypoints

        # If not, intersect circumference around object center and last segment
        # (x-c)^T(x-c) = r^2
        # with x = s + alpha d
        # resulting in alpha^2 * (d^Td) + alpha * [2 d^T(s-c)] +[(s-c)^T(s-c) - r^2] = 0
        # for convenience, we make d a unitary direction

        segment = waypoints[last_out : last_out + 2]
        s = segment[0]

        d = segment[1] - s
        d /= np.linalg.norm(d)  # so a == 1 in the 2nd order equation

        sc = s - cc
        b = 2 * d @ sc
        c = np.linalg.norm(sc) ** 2 - r**2

        # Discriminant should always be positive, as the two points differ
        # in their inclusion in the circle with given radius
        # (we enforce it's at least non-negative)
        disc = max(b**2 - 4 * c, 0)

        # We keep the smallest (entering) solution (minus sign)
        # the relative displacement from the last waypoint outside along the
        # direction to the first one inside needs to be positive
        # (we enforce it's at least non-negative)
        alpha = max((-b - np.sqrt(disc)) / 2, 0)

        intersection = s + alpha * d
        return np.concatenate([waypoints[: last_out + 1], intersection[None, :]])

    def max_dist_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        assert waypoints.shape == (2, 2)

        direction = waypoints[-1] - waypoints[0]

        dist = np.linalg.norm(direction)
        num_points = int(np.ceil(dist / self.config.policy_config.path_max_inter_waypoint_dist))
        if num_points <= 1:
            return waypoints[1:]

        stops = np.linspace(0, 1, num_points + 1)[1:]
        return waypoints[:1] + direction[None, :] * stops[:, None]

    def max_angle_waypoints(self, angles: np.ndarray) -> np.ndarray:
        assert angles.shape == (2, 1)

        angle = float(abs(normalize_ang_error(angles[1] - angles[0])).squeeze())
        num_points = int(np.ceil(angle / self.config.policy_config.path_max_inter_waypoint_angle))
        if num_points <= 1:
            # Enofrce always at least one orientation correction
            return angles[1:]

        steps = np.linspace(0, 1, num_points + 1)[1:]
        r0 = R.from_euler("z", angles[0], degrees=False)
        r1 = R.from_euler("z", angles[1], degrees=False)
        rots = Slerp([0, 1], R.concatenate([r0, r1]))(steps)
        new_angles = rots.as_euler("xyz", degrees=False)[:, 2:]

        return new_angles

    def interpolate_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        """
        Interpolate waypoints between each pair of waypoints.

        Args:
            waypoints: original waypoints array, shape (N, 2)

        Returns:
            interpolated waypoints array
        """
        density = self.config.policy_config.path_interpolation_density
        if density <= 0 or waypoints is None or len(waypoints) <= 1:
            return waypoints

        # Use np.linspace for faster vectorized interpolation
        segments = []
        for i in range(len(waypoints) - 1):
            # Create density+2 points from waypoints[i] to waypoints[i+1], excluding endpoint
            t = np.linspace(0, 1, density + 2, endpoint=False)[1:]  # exclude start point
            segment = waypoints[i] + t[:, np.newaxis] * (waypoints[i + 1] - waypoints[i])
            segments.append(segment)
        segments.append(waypoints[-1:])  # add final waypoint

        return np.vstack([waypoints[0:1]] + segments)

    def build_policy_plan(self, world_waypoints):
        world_waypoints = self.stop_plan(world_waypoints)

        # the first difference computes theta from first to second waypoint
        pos_deltas = world_waypoints[1:] - world_waypoints[:-1]
        # Here we have the thetas from waypoint i to i+1
        thetas = np.arctan2(pos_deltas[:, 1], pos_deltas[:, 0])[:, None]

        combined_waypoints = []

        # First, we orient toward the 1st waypoint from the 0-th waypoint
        start_theta = self.robot_view.get_noop_ctrl_dict(["base"])["base"][2]
        for theta in self.max_angle_waypoints(np.stack([[start_theta], thetas[0]])):
            combined_waypoints.append(np.concatenate((world_waypoints[0], theta)))

        for i in range(1, len(world_waypoints) - 1):
            # We move towards i-th waypoint from the (i-1)-th waypoint
            for waypoint in self.max_dist_waypoints(world_waypoints[i - 1 : i + 1]):
                combined_waypoints.append(np.concatenate((waypoint, thetas[i - 1])))

            # We orient towards (i+1)-th waypoint from the i-th waypoint
            for theta in self.max_angle_waypoints(thetas[i - 1 : i + 1]):
                combined_waypoints.append(np.concatenate((world_waypoints[i], theta)))

        # First arrive at final position with last movement direction
        for waypoint in self.max_dist_waypoints(world_waypoints[-2:]):
            combined_waypoints.append(np.concatenate((waypoint, thetas[-1])))

        # Then rotate to face the target
        final_pos = world_waypoints[-1]
        target_pos = self.target_object.position[:2]
        final_theta = np.arctan2(target_pos[1] - final_pos[1], target_pos[0] - final_pos[0])
        for theta in self.max_angle_waypoints(np.stack([thetas[-1], [final_theta]])):
            combined_waypoints.append(np.concatenate((final_pos, theta)))

        return np.array(combined_waypoints)

    @property
    def nav_plan(self):
        if self._nav_plan is None:
            total_attempts = 0
            for candidate_attempt in range(
                max(len(self.candidate_objs) - len(self._skipped_candidates), 1)
            ):
                for pose_attempt in range(5):
                    total_attempts += 1
                    if self.target_pos_quat is None:
                        log.info(
                            "[A* PLAN ATTEMPT FAIL] target_pos_quat is None - NavGoalSampler failed to find valid goal position"
                        )
                        break
                    else:
                        world_waypoints = None
                        try:
                            world_waypoints = self.nav_planner.motion_plan(
                                self.target_pos_quat[0], self.robot_view
                            )
                        except ValueError as e:
                            if "starting position" in str(e):
                                self._nav_plan = None
                                return self._nav_plan

                        if world_waypoints is None:
                            robot_pos = self.robot_view.base.pose[:3, 3]
                            target_pos = self.target_pos_quat[0]
                            log.info(
                                f"[A* PLAN ATTEMPT FAIL] A* pathfinding failed - no valid path found. "
                                f"Robot pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                                f"Target pos: ({target_pos[0]:.2f}, {target_pos[1]:.2f})"
                            )
                            self._target_pos_quat = None
                            continue
                        else:
                            if self.config.policy_config.path_interpolation_density > 0:
                                world_waypoints = self.interpolate_waypoints(world_waypoints)

                            world_waypoints = self.build_policy_plan(world_waypoints)
                            self._nav_plan = world_waypoints
                            self._current_waypoint = 0
                            self._dists_to_waypoint = []

                            log.info(
                                f"[A* PLAN OK] Path planned successfully with {len(world_waypoints)} waypoints"
                                f" after {pose_attempt + 1} pose samples"
                                f" in the {candidate_attempt + 1}-th highest priority candidate"
                                f" (total {total_attempts} plan attempts)"
                            )
                            break

                if self._nav_plan is None:
                    # Ignore current candidate for future plan attempts
                    self.skip_candidate(self.target_object.name)
                else:
                    break
            else:
                log.warning("[A* PLAN FAIL] no valid trajectory found")

        return self._nav_plan

    def current_waypoint(self):
        if self._current_waypoint < len(self.nav_plan):
            cur_distance = self.robot_view.distance_to(
                ["base"], self.nav_plan[self._current_waypoint]
            )
            pose = self.robot_view.base.pose
            angle = R.from_matrix(pose[:3, :3]).as_euler("xyz")[2:3]
            delta = pose[:2, 3]
            pose = np.concatenate((delta, angle))

            log.debug(
                f"Steps {self.task.num_steps_taken()}"
                f" Retries left {self._retries_left}"
                f" Waypoint {self._reached_waypoints}/{len(self.nav_plan) + self._reached_waypoints - self._current_waypoint}"
                f" {np.round(self.nav_plan[self._current_waypoint], 3)}"
                f" Pose {np.round(pose, 3)}"
                f" Dist {cur_distance:3f}"
            )

            if self.robot_view.is_close_to(["base"], self.nav_plan[self._current_waypoint]):
                self._reached_waypoints += 1
                self._dists_to_waypoint = []

                if self._replan_after is None:
                    self._current_waypoint += 1
                else:
                    if self._replan_after <= 1:
                        self._replan_after = None
                        self._nav_plan = None
                        self._target_pos_quat = None
                        if self.nav_plan is None:
                            log.warning("Terminating due to failure to replan.")
                            return None
                    else:
                        self._replan_after -= 1
                        self._current_waypoint -= 1

            elif (
                len(self._dists_to_waypoint)
                > self.config.policy_config.plan_fail_after_waypoint_steps
                and min(self._dists_to_waypoint) - cur_distance
                <= self.config.policy_config.plan_fail_max_dist_delta
            ):
                self.nav_planner.blacklist.append(self.robot_view.base.pose[:3, 3].copy())
                self.nav_planner.apply_black_list()
                if self._retries_left > 0:
                    if self._replan_after is None:

                        def waypoints_until_different_location():
                            back = 0
                            while (
                                np.linalg.norm(
                                    self.nav_plan[self._current_waypoint][:2]
                                    - self.nav_plan[self._current_waypoint - back][:2]
                                )
                                < 0.25  # TODO make this a config param
                            ):
                                if self._current_waypoint - back > 0:
                                    back += 1
                                else:
                                    break

                            return back

                        self._retries_left -= 1
                        self._replan_after = waypoints_until_different_location()
                        self._dists_to_waypoint = []
                        self._current_waypoint = max(self._current_waypoint - 1, 0)
                        log.warning(
                            f"Replanning requested after returning to the previous {self._replan_after} waypoints"
                            f" due to failure to progress with distance {cur_distance:.3f}"
                            f" to waypoint with {self._retries_left} retries left."
                        )
                    else:
                        log.warning(
                            f"Terminating due to failure to return to previous waypoint"
                            f" with {self._replan_after} missing return waypoints"
                        )
                        return None
                else:
                    log.warning(
                        f"Terminating due to failure to progress with distance {cur_distance:.3f} to waypoint"
                        f" and no plan retries left."
                    )
                    return None

            else:
                self._dists_to_waypoint.append(cur_distance)

        if self._current_waypoint < len(self.nav_plan):
            return self.nav_plan[self._current_waypoint]

        return None

    def get_action(self, observation):
        if self.nav_plan is None:
            # No plan possible, finish task immediately
            log.warning(
                f"[A* DONE] Planning failed - terminating episode at step {self.task.num_steps_taken()}"
                f" with {self._reached_waypoints} reached waypoints."
                f" Reason: A* could not find a valid path (see earlier PLAN FAIL logs for details)"
            )
            return self._build_done_action()

        # get next waypoint in the planned trajectory
        waypoint = self.current_waypoint()

        if waypoint is None:
            # All waypoints reached - navigation complete
            log.info(
                f"[A* DONE] Navigation complete - reached {self._reached_waypoints} waypoints"
                f" in {self.task.num_steps_taken()} steps."
            )
            return self._build_done_action()

        # Still navigating - return action to reach next waypoint
        return self._build_navigation_action(waypoint)

    def _build_done_action(self):
        """Build action to signal episode completion."""
        return {**self.robot_view.get_noop_ctrl_dict(["base"]), "done": True}

    def _build_navigation_action(self, waypoint):
        """Build action to navigate toward the given waypoint."""
        return {"done": False, "base": waypoint}


class AStarSmoothPlannerPolicy(AStarPlannerPolicy):
    def build_policy_plan(self, world_waypoints):
        world_waypoints = self.stop_plan(world_waypoints)

        plan_length = sum(
            np.linalg.norm(world_waypoints[it] - world_waypoints[it - 1])
            for it in range(1, len(world_waypoints))
        )
        num_points = 2 * int(
            np.ceil(plan_length / self.config.policy_config.path_max_inter_waypoint_dist)
        )

        tck, u = splprep(world_waypoints.transpose(), s=1e-5)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)

        # First derivatives
        dx_du, dy_du = splev(u_new, tck, der=1)
        # Tangent angle (radians)
        thetas = np.arctan2(dy_du, dx_du)
        # TODO handle large theta deltas

        combined_waypoints = []

        # First, we orient toward the 1st waypoint from the 0-th waypoint
        start_theta = self.robot_view.get_noop_ctrl_dict(["base"])["base"][2]
        for theta in self.max_angle_waypoints(np.stack([start_theta, thetas[0]])[:, None]):
            combined_waypoints.append(np.concatenate((world_waypoints[0], theta)))

        for cur_x, cur_y, cur_theta in zip(x_new, y_new, thetas):
            combined_waypoints.append(np.stack((cur_x, cur_y, cur_theta)))

        # Then rotate to face the target
        final_pos = world_waypoints[-1]
        target_pos = self.target_object.position[:2]
        final_theta = np.arctan2(target_pos[1] - final_pos[1], target_pos[0] - final_pos[0])
        for theta in self.max_angle_waypoints(np.stack([thetas[-1], final_theta])[:, None]):
            combined_waypoints.append(np.concatenate((final_pos, theta)))

        return np.array(combined_waypoints)
