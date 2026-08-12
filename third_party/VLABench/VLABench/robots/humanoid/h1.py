import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from VLABench.robots.humanoid.base import Humanoid
from VLABench.utils.register import register
from VLABench.utils.utils import create_mesh_box, quaternion_to_matrix, compute_rotation_quaternion, normalize

@register.add_robot("h1")
class H1Humanoid(Humanoid):
    def __init__(self, **kwargs):
        super().__init__(name=kwargs.get("name", "h1"), **kwargs)
        self.mask = {
            "right_arm": [0]*11 + [1]*4 + [0]*30,
            "left_arm": [0]*15 + [1]*4 + [0]*26,
        }
    
    @property
    def gripper_geoms(self):
        left_finger_geoms = self.mjcf_model.find("body", "left_hand_link").find_all("geom")
        right_finger_geoms = self.mjcf_model.find("body", "right_hand_link").find_all("geom")
        return left_finger_geoms + right_finger_geoms
    
    @property
    def link_base(self):
        return self._mjcf_model.find("body", "pelvis")
    
    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for i, qpos in enumerate(self.default_qpos):
            physics.bind(self.joints[i]).qpos = qpos
        physics.data.ctrl = self.default_qpos
    
    def get_qpos_from_ee_pos(self, physics, pos, quat, inplace=False, target_hand="right", **kwargs):
        success, qpos = super().get_qpos_from_ee_pos(physics, pos, quat, inplace, target_hand, **kwargs)
        return success, qpos[:45]
    
    def get_whole_body_qpos(self, physics):
        qposes = []
        for joint in self.joints:
            qpos = physics.bind(joint).qpos
            qposes.append(qpos)
        return qposes
    
    def get_arm_qpos(self, physics):
        qposes = super().get_arm_qpos(physics)
        return qposes[11:19]