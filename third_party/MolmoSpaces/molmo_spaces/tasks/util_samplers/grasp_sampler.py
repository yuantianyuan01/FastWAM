import abc
from typing import NoReturn

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import MlSpacesObject

# from molmo_spaces.tasks.atomic.abstract_sampler import TargetPoseSampler
from molmo_spaces.utils.linalg_utils import (
    global_to_relative_transform,
    relative_to_global_transform,
)


class GraspPoseSampler:
    @abc.abstractmethod
    def sample(self, **kwargs) -> NoReturn:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_pregrasp(self) -> NoReturn:
        raise NotImplementedError


class TopDownGraspPoseSampler(GraspPoseSampler):
    def __init__(self, top_down_grasp_orientation: np.ndarray) -> None:
        """Top down EE orientation might be different per embodiment"""
        assert top_down_grasp_orientation.shape == (3, 3)
        self.top_down_grasp_orientation = top_down_grasp_orientation
        self.grasp_position = None
        self.target = None
        self.grasp_pose = np.eye(4)
        self.pregrasp_pose = np.eye(4)

    def set_target(self, target: MlSpacesObject) -> None:
        self.target = target

    def sample(self, agent, mode: str = "global", **kwargs):
        """mode is 'global' or agent 'base'
        I forgot what the mode was for... for returning in this reference frame or....???

        """
        pose = relative_to_global_transform(
            self._heuristics_based_on_center_bbox(agent), agent.base_pose
        )
        position = pose[:3, 3]
        quaternion = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        return position, quaternion

    def sample_pregrasp(self, agent, mode: str = "global"):
        """mode is 'global' or agent 'base'
        I forgot what the mode was for... for returning in this reference frame or....???

        """
        return self._heuristics_based_pregrasp(self.grasp_pose)

    def _heuristics_based_on_center_bbox(self, base_pose):
        """Always from base"""
        grasp_pose = self.grasp_pose.copy()

        # Get target pose from agent base
        target_roll = 0.0
        object_pose_from_base = global_to_relative_transform(self.target.pose, base_pose)

        # Get Position of target object
        grasp_pose[:3, 3] = object_pose_from_base[:3, 3]

        # Adjust agent ee roll based on object dimensions
        if self.target.aabb_size[2] >= 0.2:  # TODO: arbitrary gripper length
            target_roll = R.from_matrix(object_pose_from_base[:3, :3]).as_euler(
                "xyz", degrees=False
            )[0]
            if self.target.aabb_size[0] > self.target.aabb_size[2]:
                # All Thor/Objaverse objects have Y axis as height
                target_roll += np.pi / 2  # 90 degree
            # NOTE: Is the width wide enough to be graspable?
            print(f"Bbox of object: {self.target.aabb_size}")
            print(f"Target roll: {np.rad2deg(target_roll)}")

            # Apply roll rotation to grasp orientation
            roll_rotation = R.from_euler("x", target_roll)
            final_orientation = roll_rotation.as_matrix() @ self.top_down_grasp_orientation
            grasp_pose[:3, :3] = final_orientation
        else:
            grasp_pose[:3, :3] = self.top_down_grasp_orientation

        # get the height of the object using orientation and bounding box
        object_height = self._compute_object_height(self.target.pose, self.target.aabb_size)
        print(f"Computed object height: {object_height}")

        # grasp lower bottom of the object
        grasp_pose[2, 3] += min(0.0025, object_height / 6)

        return grasp_pose

    def _heuristics_based_on_mesh_vertices(self, mode: str = "global") -> NoReturn:
        raise NotImplementedError

    def _heuristics_based_pregrasp(self, grasp_pose, offset: float = 0.1):
        """Raise up"""
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose[:3, 3] += [0.0, 0.0, offset]  # pick up topdown from 10cm above
        return pregrasp_pose

    def get_delta_pregrasp_to_grasp_joint_pos(self) -> NoReturn:
        raise NotImplementedError

    def _compute_object_height(self, pose, aabb_size):
        """
        Compute the object's height using its 4x4 pose and axis-aligned bounding box dimensions.

        Parameters:
            pose (np.ndarray): 4x4 transformation matrix of the object.
            aabb_size (list or np.ndarray): Bounding box dimensions [w, d, h] in the object's local frame.

        Returns:
            float: Height of the object in world coordinates.
        """
        half_sizes = np.array(aabb_size) / 2.0
        # Define the 8 corners of the local bounding box
        corners = np.array(
            [
                [-half_sizes[0], -half_sizes[1], -half_sizes[2]],
                [-half_sizes[0], -half_sizes[1], half_sizes[2]],
                [-half_sizes[0], half_sizes[1], -half_sizes[2]],
                [-half_sizes[0], half_sizes[1], half_sizes[2]],
                [half_sizes[0], -half_sizes[1], -half_sizes[2]],
                [half_sizes[0], -half_sizes[1], half_sizes[2]],
                [half_sizes[0], half_sizes[1], -half_sizes[2]],
                [half_sizes[0], half_sizes[1], half_sizes[2]],
            ]
        )
        # Extract the object's rotation from pose (ignoring translation since height is relative)
        R_obj = pose[:3, :3]
        # Transform the local corners to world-scaled coordinates
        transformed_corners = (R_obj @ corners.T).T
        # Compute height as the difference between the highest and lowest z values among the corners
        z_vals = transformed_corners[:, 2]
        height = z_vals.max() - z_vals.min()
        return height


class SideGraspPoseSampler(GraspPoseSampler):
    def __init__(self, thormap) -> None:
        """Side EE orientation might be different per embodiment"""
        self.thormap = thormap

    def set_target(self, target: MlSpacesObject) -> None:
        self.target = target

    def sample(self, agent, mode: str = "global", agent_position: np.ndarray = None):
        """
        Find nearest reachable pose (x, y, theta) to the target 3D position.

        Args:
            target_pos_m: np.ndarray of shape (3,), world coordinate

        Returns:
            pose: np.ndarray of shape (3,), (x, y, theta)
        """
        if mode == "global":  # sample base location
            return self.sample_base()
        elif mode == "base":  # sample ee location
            return self.sample_ee_pose()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def sample_base(
        self,
    ):
        # Step 1: Get nearest free (x, y)
        target_pos_m = self.target.pose[:3, 3]

        target_px = self.thormap.pos_m_to_px(target_pos_m[None])[0][:2]
        free_points_px = np.argwhere(~self.thormap.occupancy)
        distances = np.linalg.norm(free_points_px - target_px, axis=1)

        # Optional:
        # min_dist_to_target = 0.65 # robot base should be at least this far away from the target (i.e. don't be too close)
        # min_dist_px = self.thormap.px_per_m * min_dist_to_target
        # # Filter out points that are too close to the target
        # distances[distances < min_dist_px] = np.inf

        nearest_px = free_points_px[np.argmin(distances)]
        nearest_pos_m = self.thormap.pos_px_to_m(nearest_px[None])[0]

        # Step 2: Compute θ to face the target
        dx = target_pos_m[0] - nearest_pos_m[0]
        dy = target_pos_m[1] - nearest_pos_m[1]
        theta = np.arctan2(dy, dx)

        position = [nearest_pos_m[0], nearest_pos_m[1], 0.105]
        quaternion = R.from_euler("xyz", [0, 0, theta], degrees=False).as_quat(scalar_first=True)

        return position, quaternion

    def sample_ee_pose(
        self,
    ):
        return NotImplementedError

    def sample_pregrasp(self, agent, mode: str = "global") -> None:
        pass
