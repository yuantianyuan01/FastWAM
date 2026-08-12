import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from VLABench.robots.single_arm.base import SingleArm
from VLABench.utils.register import register
from VLABench.utils.utils import create_mesh_box, quaternion_to_matrix, compute_rotation_quaternion, normalize
    
@register.add_robot("widowx")
class Widowx(SingleArm):
    def __init__(self, **kwargs):
        super().__init__(name=kwargs.get("name", "widowx"), 
                         n_dof=kwargs.get("n_dof", 6),
                         **kwargs)
    
    def _add_camera_to_wrist(self, **camera_params):
        name = camera_params.get("name", "wrist_cam")
        pos = camera_params.get("pos", [0, 0, 0])
        euler = camera_params.get("euler", [np.pi, 0, np.pi/2])
        fovy = camera_params.get("fovy", 30)
        gripper_site = self.mjcf_model.find("body", "hand")
        gripper_site.add("camera", name=name, pos=pos, euler=euler, fovy=fovy)
    
    @property
    def link_base(self):
        return self._mjcf_model.find("body", "base_link")

    @property
    def end_effector_site(self):
        return self.mjcf_model.find("site", "end_effector")
    
    @property
    def gripper_geoms(self):
        left_finger_geoms = self.mjcf_model.find("body", "left_finger_link").find_all("geom")
        right_finger_geoms = self.mjcf_model.find("body", "right_finger_link").find_all("geom")
        return left_finger_geoms + right_finger_geoms
    
    def gripper_move_orientation(self, physics):
        ee_site = self.end_effector_site
        ee_site_pos = physics.bind(ee_site).xpos
        ee_move_site = self._mjcf_model.find("site", "end_effector_move")
        ee_move_site_pos = physics.bind(ee_move_site).xpos
        move_quat = compute_rotation_quaternion(ee_site_pos, ee_move_site_pos)
        return move_quat
    
    @property
    def ee2move_transform(self):
        R_ee = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
        R_move = R.from_euler("xyz", [0, 0, -90], degrees=True).as_matrix()
        ee2move_transform = np.dot(R_move, R_ee.T)
        return ee2move_transform
    
    def get_ee_open_state(self, physics=None):
        left_finger_joint = self.mjcf_model.find("joint", "left_finger")
        right_finger_joint = self.mjcf_model.find("joint", "right_finger")
        left_finger_pos = physics.bind(left_finger_joint).qpos
        right_finger_pos = physics.bind(right_finger_joint).qpos
        if left_finger_pos < 0.035 and right_finger_pos < 0.035:
            return False
        else:
            return True
        
    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        assert len(self.default_qpos) == len(self.joints), "default_qpos should be the same length as joints, but got {} and {}".format(len(self.default_qpos), len(self.joints))
        for i, qpos in enumerate(self.default_qpos):
            physics.bind(self.joints[i]).qpos = qpos
        physics.data.ctrl = self.default_qpos

    def get_ee_state(self, physics):
        pos = np.array(self.get_end_effector_pos(physics))
        quat = np.array(self.get_end_effector_quat(physics))
        open = np.array(self.get_ee_open_state(physics)).astype(np.float32).reshape((1,))
        return np.concatenate([pos, quat, open])
    
    def ee_offset(self, physics):
        grasp_pos = self.get_end_effector_pos(physics)
        ee_end_site = self.mjcf_model.find("site", "end_effector_move")
        ee_end_pos = physics.bind(ee_end_site).xpos
        return grasp_pos - ee_end_pos
    
    def gripper_pcd(self, target_pos=None, target_quat=None, color=None):
        """
        get abstract&simple gripper point cloud for collision detection
        """
        raise NotImplementedError