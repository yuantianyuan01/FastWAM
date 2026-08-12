import logging
from typing import NoReturn

import cv2
import numpy as np

from molmo_spaces.utils.linalg_utils import relative_to_global_transform
from molmo_spaces.utils.scene_maps import ProcTHORMap

log = logging.getLogger(__name__)


class UniformRandomMapSampler:
    def __init__(self, thormap: ProcTHORMap, seed: int = 0, debug: bool = False) -> None:
        self.thormap = thormap
        self.rng = np.random.default_rng(seed)
        self.debug = debug

        # Cache commonly used values
        self._base_occupancy = cv2.cvtColor(
            self.thormap.occupancy_map.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR
        )
        self._base_mask = np.zeros_like(self._base_occupancy)
        self._free_points = None
        self._cached_constraint_masks = {}  # Cache masks by (pos, distance) tuple

        # Pre-compute view angle constants
        self._deg_to_rad = np.pi / 180.0

    def _make_constraint_mask(self, constraint_positions, constraint_distances):
        """Make constraint masks more efficiently with caching"""
        all_constraint_masks = []

        for pos, d in zip(constraint_positions, constraint_distances):
            # Create cache key from position and distance
            cache_key = (tuple(pos), d)

            if cache_key in self._cached_constraint_masks:
                all_constraint_masks.append(self._cached_constraint_masks[cache_key])
                continue

            constraint_mask = self._base_mask.copy()
            pos_px = self.thormap.pos_m_to_px(pos)
            radius = int(d * self.thormap.px_per_m)

            # Draw circle
            cv2.circle(
                constraint_mask,
                center=(int(pos_px[1]), int(pos_px[0])),
                radius=radius,
                color=(255, 255, 255),  # 0 Black circles (free areas)
                thickness=-1,  # -1 means filled circle
            )

            # Convert to bool mask
            bool_mask = cv2.cvtColor(constraint_mask, cv2.COLOR_BGR2GRAY) == 0

            # Cache the mask
            self._cached_constraint_masks[cache_key] = bool_mask
            all_constraint_masks.append(bool_mask)

        return all_constraint_masks

    def _sample_position(self, N, constraint_positions, constraint_distances) -> NoReturn:
        raise NotImplementedError

    def _sample_quaternion(self, N, look_at, camera_pose_rel_base, view_range_deg) -> NoReturn:
        raise NotImplementedError

    def sample(
        self,
        N=1,
        positions: np.ndarray | None = None,
        quaternions: np.ndarray | None = None,
        constraint_positions: np.ndarray | None = None,
        constraint_distances: np.ndarray | None = None,
        z_pos: float | None = None,
        look_at: bool | None = True,
        camera_pose_rel_base: np.ndarray | None = None,
        view_range_deg: float | None = 30.0,
    ):
        """
        Samples N points from free space on the map.
        If positions and quaternions are not provided, sample uniformly from all free points.
        If constraint_positions and constraint_distances are provided, use them to sample points within the specified distances from the given positions.

        Args:
            N (int, optional): The number of points to sample. Defaults to 1.
            positions (Optional[np.ndarray], optional): An array of positions to force.
            quaternions (Optional[np.ndarray], optional): An array of quaternions to force.
            constraint_positions (Optional[np.ndarray], optional): An array of positions to treat as the center of circular constraints.
            constraint_distances (Optional[np.ndarray], optional): An array of distances to treat as the radius of circular constraints.
            z_pos (Optional[float], optional): If specified, overrides the z-coordinate of the sampled points. Defaults to None.

        Returns:
            np.ndarray: An array of sampled points, of shape (N, 3) if N > 1, or (3,) if N == 1.
        """
        # TODO: assert len N for positions, quaternions, constraint_positions, constraint_distances
        # TODO: positiion and constraint_positions cannot be both specified. throw error if both are provided - was this intended?
        new_positions = np.zeros((N, 3))
        new_quaternions = np.zeros((N, 4))

        # Debug visualization
        if self.debug:
            self._debug_visualize(N, constraint_positions)

        # Sample positions
        if positions is None:
            if constraint_positions is not None:
                new_positions = self._sample_constrained_positions(
                    N, constraint_positions, constraint_distances
                )
                if new_positions is None:
                    return None
            else:
                new_positions = self._sample_free_positions(N)
        else:
            new_positions = positions

        # Sample orientations
        if quaternions is None:
            new_quaternions = self._sample_orientations(
                N,
                new_positions,
                constraint_positions,
                look_at,
                camera_pose_rel_base,
                view_range_deg,
            )
        else:
            new_quaternions = quaternions

        # Prepare return dict
        ret = {"position": new_positions, "quaternion": new_quaternions}
        if z_pos is not None:
            ret["position"][..., 2] = z_pos

        return ret

    def _debug_visualize(self, N, constraint_positions) -> None:
        """Separate debug visualization logic"""
        vis_map = self._base_occupancy.copy()
        for i in range(N):
            if constraint_positions is not None:
                pos_px = self.thormap.pos_m_to_px(constraint_positions[i])
                cv2.circle(
                    vis_map,
                    (int(pos_px[1]), int(pos_px[0])),
                    radius=15,
                    color=(255, 0, 0),
                    thickness=-1,
                )
            cv2.imshow("vis_map", vis_map)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _sample_constrained_positions(self, N, constraint_positions, constraint_distances):
        """Sample positions with constraints"""
        assert len(constraint_positions) == len(constraint_distances)
        all_constraint_masks = self._make_constraint_mask(
            constraint_positions, constraint_distances
        )

        positions = np.zeros((N, 3))
        for i in range(N):
            free_mask = ~self.thormap.occupancy & ~all_constraint_masks[i]
            free_points_px = np.argwhere(free_mask)

            if len(free_points_px) == 0:
                log.warning("No free points found")
                return None

            idx = self.rng.choice(len(free_points_px), 1, replace=False)[0]
            positions[i] = self.thormap.pos_px_to_m(free_points_px[idx])

        return positions

    def _sample_free_positions(self, N):
        """Sample positions from free space"""
        if self._free_points is None:
            self._free_points = self.thormap.get_free_points().copy()
        return self._free_points[self.rng.choice(len(self._free_points), N, replace=False)]

    def _sample_orientations(
        self, N, positions, constraint_positions, look_at, camera_pose_rel_base, view_range_deg
    ):
        """Sample orientation quaternions"""
        if constraint_positions is not None and look_at:
            return self._sample_look_at_orientations(
                N, positions, constraint_positions, camera_pose_rel_base, view_range_deg
            )
        else:
            z_angles = self.rng.uniform(0, 2 * np.pi, N)
            return np.column_stack(
                (np.cos(z_angles / 2), np.zeros(N), np.zeros(N), np.sin(z_angles / 2))
            )

    def _sample_look_at_orientations(
        self, N, positions, constraint_positions, camera_pose_rel_base, view_range_deg
    ):
        """Sample orientations that look at targets"""
        camera_positions = positions.copy()
        if camera_pose_rel_base is not None:
            for i in range(N):
                base_pose = np.eye(4)
                base_pose[:3, 3] = positions[i]
                camera_pose = relative_to_global_transform(camera_pose_rel_base, base_pose)
                camera_positions[i] = camera_pose[:3, 3]

        z_angles = np.zeros(N)
        half_range = (view_range_deg / 2.0) * self._deg_to_rad

        for i in range(N):
            direction = constraint_positions[i] - camera_positions[i]
            direction /= np.linalg.norm(direction)
            angle_of_direction = np.arctan2(direction[1], direction[0])
            z_angles[i] = self.rng.uniform(
                angle_of_direction - half_range,
                angle_of_direction + half_range,
            )

        return np.column_stack(
            (np.cos(z_angles / 2), np.zeros(N), np.zeros(N), np.sin(z_angles / 2))
        )


def furthest_point_sampling(points, k):
    """
    Furthest Point Sampling (FPS)

    Args:
        points (np.ndarray): Array of shape (N, D), N points in D dimensions
        k (int): Number of points to sample

    Returns:
        sampled_indices (np.ndarray): Indices of sampled points in the original array
    """
    N, D = points.shape
    sampled_indices = np.zeros(k, dtype=int)
    distances = np.full(N, np.inf)  # distance to the closest sampled point

    # Initialize with a random point
    sampled_indices[0] = np.random.randint(0, N)

    for i in range(1, k):
        # Compute distances from the last added point
        last_point = points[sampled_indices[i - 1]]
        dist_to_last = np.linalg.norm(points - last_point, axis=1)

        # Update minimum distances
        distances = np.minimum(distances, dist_to_last)

        # Choose the point with the maximum distance to the sampled set
        sampled_indices[i] = np.argmax(distances)

    return sampled_indices
