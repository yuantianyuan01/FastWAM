import logging
from typing import NoReturn

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.robots.robot_views.abstract import RobotView
from molmo_spaces.utils.pose import pos_quat_to_pose_mat
from molmo_spaces.utils.sampler_utils import UniformRandomMapSampler
from molmo_spaces.utils.scene_maps import THORMap

log = logging.getLogger(__name__)


class NavGoalSampler:
    def __init__(
        self,
        thormap: THORMap,
        view_range_deg: float = 30.0,
        distance_threshold: float = 0.5,
        open_space_sampler_cls: UniformRandomMapSampler = UniformRandomMapSampler,
        seed: int = 0,
        camera_name: str = None,
        check_target_in_view: bool = False,
        debug: bool = False,
        device_id: int = None,
    ) -> None:
        self.view_range_deg = view_range_deg
        self.distance_threshold = distance_threshold
        self.target = None
        self.map_sampler = open_space_sampler_cls(thormap, seed=seed, debug=debug)
        self.navgoal_pose = None
        # Initialize renderer and camera_name if needed
        self.check_target_in_view = check_target_in_view
        self.camera_name = camera_name
        self._renderer = None
        self.model_bindings = None
        self.robot_view = None
        self.debug = debug
        self.device_id = device_id

    def set_target(self, target: MlSpacesObject) -> None:
        self.target = target

    def set_robot_view(self, robot_view: RobotView) -> None:
        self.robot_view = robot_view

    def _set_target_room(self) -> NoReturn:
        # TODO: Find a room target is in and re-generate map with additional room constraint
        raise NotImplementedError

    def _check_target_in_view(self, sampled_pose: dict, robot_view: RobotView, camera_name: str):
        assert self.camera_name is not None, "camera_name is None"
        assert self.robot_view is not None or robot_view is not None, "robot_view is None"

        robot_view = robot_view or self.robot_view

        # Create a temporary data object instead of modifying the original
        temp_data = MjData(robot_view.mj_model)
        old_data = robot_view.mj_data

        try:
            robot_view.mj_data = temp_data

            robot_view.base.pose = pos_quat_to_pose_mat(
                sampled_pose["position"][0], sampled_pose["quaternion"][0]
            )
            mujoco.mj_forward(robot_view.mj_model, temp_data)

            # TODO render segmentation mask and check if target in view
            raise NotImplementedError
            # # Check target visibility
            # target_in_view = robot_view.check_target_in_view(
            #     self.target.object_id,
            #     temp_data,
            #     camera_name,
            #     visualize=self.debug,
            #     device_id=self.device_id,
            # )
        finally:
            # Clean up
            robot_view.mj_data = old_data
            del temp_data

        return target_in_view

    def _sample_around_target_boundary(
        self,
        distance_threshold: float = 0.2,
        num_samples: int = 10,  # maximum number of attempts (not used here but could be incorporated)
        robot_position: np.ndarray = None,
        **kwargs,
    ):
        """
        Sample a navigation goal using the object's occupied zone boundary.

        Instead of using an annulus, create a mask for the object footprint (using aabb_size),
        then dilate that mask by a given margin. Subtract the original mask to get a boundary ring.
        Intersect this with the free space in the occupancy map and sample a candidate.

        Returns:
            A dict containing a navigation goal pose {'position': candidate_position, 'quaternion': quat}
            or None if no valid candidate is found.
        """
        thormap = self.map_sampler.thormap

        # Ensure we have a target
        assert self.target is not None, "Target is not set"

        # Use object's center as the base position (ground-level)
        target_center_m = np.array(self.target.position)
        target_center_m[2] = 0.0

        # Convert object's center to pixel coordinates
        target_center_px = thormap.pos_m_to_px(target_center_m)
        px_per_m = thormap.px_per_m

        # Create masks only once and reuse
        if not hasattr(self, "_occupancy_mask"):
            self._occupancy_mask = cv2.cvtColor(
                thormap.occupancy_map.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR
            )

        # Use cached occupancy mask
        occupancy_mask = self._occupancy_mask.copy()

        # Create kernel for dilation
        kernel_size = max(3, int(distance_threshold * px_per_m))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Get boundary ring
        dilated_mask = cv2.dilate(occupancy_mask, kernel, iterations=1)
        boundary_mask = cv2.subtract(dilated_mask, occupancy_mask)
        boundary_mask = cv2.cvtColor(boundary_mask, cv2.COLOR_BGR2GRAY)

        # Get free boundary points
        free_boundary = boundary_mask == 255
        free_indices = np.argwhere(free_boundary)

        if free_indices.shape[0] == 0:
            return None

        # Compute distances of free boundary pixels to the object's center in pixel space
        distances = np.linalg.norm(free_indices - target_center_px, axis=1)

        # Get distances that are less than distance_threshold
        top_candidates = free_indices[distances < px_per_m * distance_threshold]
        if len(top_candidates) == 0:
            log.warning("No free candidate positions found on object boundary! Using top closest")
            min_max_distance = np.min(distances) + px_per_m * distance_threshold
            top_candidates = free_indices[distances < min_max_distance]
            # Get the top 5% pixels closest to the object center (ensure at least one candidate)
            # num_top = max(1, int(0.05 * len(free_indices)))
            # sorted_idx = np.argsort(distances)
            # top_candidates = free_indices[sorted_idx[:num_top]]

        # Don't pick to close to the robot
        min_distance_threshold = 3.0
        if robot_position is not None:
            min_distance_threshold_px = px_per_m * min_distance_threshold
            robot_position_px = thormap.pos_m_to_px(robot_position)
            dist_to_robot = np.linalg.norm(top_candidates - robot_position_px, axis=1)
            top_candidates = top_candidates[dist_to_robot > min_distance_threshold_px]

            if len(top_candidates) == 0:
                log.warning(
                    "No free candidate positions found on further than 3m from the robot! Skipping..."
                )
                return None

        # Randomly select one candidate from these top candidates
        chosen_candidate = top_candidates[np.random.choice(len(top_candidates))]
        candidate_px = chosen_candidate

        # Convert the candidate pixel back to world (meter) coordinates.
        candidate_m = thormap.pos_px_to_m(candidate_px)

        # Compute orientation so that the robot faces the target.
        delta = target_center_m - candidate_m
        angle_to_target = np.arctan2(delta[1], delta[0])
        quat = np.array([np.cos(angle_to_target / 2.0), 0.0, 0.0, np.sin(angle_to_target / 2.0)])

        # create a vis map
        vis_map = cv2.cvtColor(
            self.map_sampler.thormap.occupancy_map.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR
        )
        pos_px = candidate_px
        # target/contraint positon
        vis_map = cv2.circle(
            vis_map,
            (int(target_center_px[1]), int(target_center_px[0])),
            radius=15,
            color=(0, 0, 255),  # Red (BGR)
            thickness=-1,
        )
        # sampled position
        vis_map = cv2.circle(
            vis_map,
            (int(pos_px[1]), int(pos_px[0])),
            radius=15,
            color=(255, 0, 0),  # Blue (BGR)
            thickness=-1,
        )
        if robot_position is not None:
            # robot position
            robot_position_px = thormap.pos_m_to_px(robot_position)
            vis_map = cv2.circle(
                vis_map,
                (int(robot_position_px[1]), int(robot_position_px[0])),
                radius=15,
                color=(0, 255, 0),  # Green (BGR)
                thickness=-1,
            )

        self.vis_map = vis_map
        if self.debug:
            log.debug(f"sampled_{pos_px}_for_target_at_{target_center_px}")
            log.debug(f"map size: {vis_map.shape}")
            plt.imshow(vis_map)
            plt.title(f"sampled_{pos_px}_for_target_at_{self.target.name}")
            plt.show()
            # cv2.imshow(f"sampled_{pos_px}_for_target_at_{target_center_px}", vis_map)
            # cv2.waitKey(1)
            # cv2.destroyAllWindows()
            return {"position": [candidate_m], "quaternion": [quat], "debug_vis_map": vis_map}

        return {"position": [candidate_m], "quaternion": [quat]}

    def sample(
        self,
        robot_view: RobotView = None,
        distance_threshold: float = None,
        thormap: THORMap = None,
        n_tries: int = 10,
        robot_position: np.ndarray = None,
        **kwargs,
    ):
        """mode is 'global' or robot 'base'. Always sample in global coordinates.
        Note: for some itetms realy small and low, robot might have to be further than distance thoreshold
        """
        if thormap is not None:
            self.map_sampler.thormap = thormap

        pose_sample = None
        is_in_view = False
        count = 0

        while pose_sample is None and count < n_tries:
            count += 1

            # For now, if no target is set, fall back on map sampler
            if self.target is None:
                pose_sample = self.map_sampler.sample(
                    N=1,
                    z_pos=0.005,  # 0.005
                )
            else:
                # Our sampler can now be switched to use _sample_around_target_boundary if desired.
                if distance_threshold is None:
                    distance_threshold = self.distance_threshold

                pose_sample = self._sample_around_target_boundary(
                    distance_threshold=distance_threshold, robot_position=robot_position, **kwargs
                )
                """
                # Fallback: use uniform sampling based on constraint (if candidate not found)
                pose_sample = self.map_sampler.sample(
                    N=1,
                    constraint_positions=[self.target.position],
                    constraint_distances=[distance_threshold],
                    z_pos=0.005,
                    look_at=True,
                    camera_pose_rel_base=None,  # reshape to 4x4  if not None#camera_pose_rel_base,
                    view_range_deg=self.view_range_deg,  # 30.0,
                )
                """
            if pose_sample is None:
                log.warning("No pose sample found. Trying again...")
            elif self.check_target_in_view:
                is_in_view = self._check_target_in_view(pose_sample, robot_view, self.camera_name)
                if not is_in_view:
                    pose_sample = None
                    distance_threshold += 0.25  # some object seems better when further away can see
                    log.warning("Target is not in view")
        if pose_sample is not None:
            return pose_sample["position"][0], pose_sample["quaternion"][0]
        else:
            log.warning(f"No pose sample found after {n_tries} tries")
            return None  # i feel like this happens too often :(
