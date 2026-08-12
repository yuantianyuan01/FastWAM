import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from VLABench.robots.single_arm.base import SingleArm
from VLABench.utils.register import register
from VLABench.utils.utils import create_mesh_box, quaternion_to_matrix, compute_rotation_quaternion, normalize
    
@register.add_robot("franka")
class Franka(SingleArm):
    def __init__(self, **kwargs):
        super().__init__(name=kwargs.get("name", "franka"), 
                         n_dof=kwargs.get("n_dof", 7),
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
        return self._mjcf_model.find("body", "link0")

    @property
    def end_effector_site(self):
        return self.mjcf_model.find("site", "end_effector")
    
    @property
    def gripper_geoms(self):
        left_finger_geoms = self.mjcf_model.find("body", "left_finger").find_all("geom")
        right_finger_geoms = self.mjcf_model.find("body", "right_finger").find_all("geom")
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
        left_finger_joint = self.mjcf_model.find("joint", "finger_joint1")
        right_finger_joint = self.mjcf_model.find("joint", "finger_joint2")
        left_finger_pos = physics.bind(left_finger_joint).qpos
        right_finger_pos = physics.bind(right_finger_joint).qpos
        if left_finger_pos < 0.035 and right_finger_pos < 0.035:
            return True # BUG: should be False
        else:
            return False
        
    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for i, qpos in enumerate(self.default_qpos):
            if i < 7: physics.bind(self.mjcf_model.find("joint", f"joint{i + 1}")).qpos = qpos
            else: physics.bind(self.mjcf_model.find("joint", f"finger_joint{i - 6}")).qpos = qpos
        physics.data.ctrl = self.default_qpos
    
    def gripper_pcd(self, target_pos=None, target_quat=None, color=None):
        """
        get abstract&simple gripper point cloud for collision detection
        """
        center = np.asarray(target_pos)
        quat = np.asarray(target_quat)
        rot_matrix = quaternion_to_matrix(quat)
        left_finger = create_mesh_box(width=0.02, height=0.02, depth=0.08, dx=-0.01, dy=-0.055, dz=-0.04)
        right_finger = create_mesh_box(width=0.02, height=0.02, depth=0.08, dx=-0.01, dy=0.04, dz=-0.04)
        gripper_link = create_mesh_box(width=0.05, height=0.18, depth=0.08, dx=-0.025, dy=-0.09, dz=-0.12)
        
        move_point = np.array([np.array([0, 0, 0]), np.array([0, 0, 0.1])])
        move_point = np.dot(rot_matrix, move_point.T).T + center
        move_vector = normalize(move_point[1] - move_point[0])
        move_point_pcd = o3d.geometry.PointCloud()
        move_point_pcd.points = o3d.utility.Vector3dVector(move_point)
        move_point_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0]])
        
        left_points = np.array(left_finger.vertices)
        left_triangles = np.array(left_finger.triangles)
        
        right_points = np.array(right_finger.vertices)
        right_triangles = np.array(right_finger.triangles) + 8
        
        gripper_link_points = np.array(gripper_link.vertices)
        gripper_link_triangles = np.array(gripper_link.triangles) + 16
        
        vertices = np.concatenate([left_points, right_points, gripper_link_points], axis=0)
        vertices = np.dot(rot_matrix, vertices.T).T + center
        triangles = np.concatenate([left_triangles, right_triangles, gripper_link_triangles], axis=0)
        
        colors = np.array([[0, 0, 1] for _ in range(len(vertices))]) if color is None else [color for _ in range(len(vertices))]
        
        gripper = o3d.geometry.TriangleMesh()
        gripper.vertices = o3d.utility.Vector3dVector(vertices)
        gripper.triangles = o3d.utility.Vector3iVector(triangles)
        gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
        return gripper.sample_points_uniformly(number_of_points=1000), move_vector
    
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
        