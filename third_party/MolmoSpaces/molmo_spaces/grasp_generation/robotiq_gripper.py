import numpy as np
import trimesh
import trimesh.transformations as tra

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR


class RobotiqGripper:
    """
    Robotiq 2F-85 gripper specifically implemented for the Molmo-Spaces grasp generation pipeline.
    Not the same as the Molmo-Spaces Robotiq 2F-85 floating gripper implementation, which is used for the Molmo-Spaces robot.
    """

    tcp_offset = np.array([0, 0, 0.155])

    def __init__(self, q=0.048372, num_contact_points_per_finger=20, root_folder=""):
        self.default_pregrasp_configuration = 0.048372

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        gripping_center = 0.155
        self.tcp_offset = np.array([0, 0, gripping_center])
        fn_base = root_folder + str(ASSETS_DIR / "robots/floating_robotiq/assets/hand.stl")
        fn_finger = root_folder + str(ASSETS_DIR / "robots/floating_robotiq/assets/finger.stl")
        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.13686])
        self.finger_r.apply_translation([-q, 0, 0.13686])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.ray_origins = []
        self.ray_directions = []
        for i in np.linspace(-0.032, 0.037, num_contact_points_per_finger):
            self.ray_origins.append(np.r_[self.finger_l.bounding_box.centroid + [0, 0, i], 1])
            self.ray_origins.append(np.r_[self.finger_r.bounding_box.centroid + [0, 0, i], 1])
            self.ray_directions.append(
                np.r_[-self.finger_l.bounding_box.primitive.transform[:3, 0]]
            )
            self.ray_directions.append(
                np.r_[+self.finger_r.bounding_box.primitive.transform[:3, 0]]
            )

        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

        self.standoff_range = np.array([0.1, 0.174])
        self.standoff_range[0] += 0.001

    def get_base_mesh(self):
        return self.base

    def get_closing_rays(self, transform):
        return transform[:3, :].dot(self.ray_origins.T).T, transform[:3, :3].dot(
            self.ray_directions.T
        ).T

    def get_finger_meshes(self):
        return [self.finger_l, self.finger_r]
