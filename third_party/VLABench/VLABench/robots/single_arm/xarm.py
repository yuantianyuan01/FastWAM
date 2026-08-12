import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from VLABench.robots.single_arm.base import SingleArm
from VLABench.utils.register import register
from VLABench.utils.utils import create_mesh_box, quaternion_to_matrix, compute_rotation_quaternion, normalize

@register.add_robot("xarm")
class XArm(SingleArm):
    def __init__(self, **kwargs):
        super().__init__(name=kwargs.get("name", "xarm"), **kwargs)
    
    @property
    def link_base(self):
        return self._mjcf_model.find("body", "link_base")
    
    @property
    def end_effector_site(self):
        return self.mjcf_model.find("site", "link_tcp")
    
    @property
    def gripper_geoms(self):
        left_finger_geoms = self.mjcf_model.find("body", "left_finger").find_all("geom")
        right_finger_geoms = self.mjcf_model.find("body", "right_finger").find_all("geom")
        return left_finger_geoms + right_finger_geoms
          
    def get_qpos_from_ee_pos(self, physics, pos, quat=None, inplace=False, **kwargs):
        ik_result = qpos_from_site_pose(physics,
                                        site_name=self.end_effector_site.full_identifier,
                                        target_pos=pos,
                                        target_quat=quat,
                                        inplace=inplace,
                                        **kwargs)
        target_qpos = ik_result.qpos
        return target_qpos
    
    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        for i, qpos in enumerate(self.default_qpos):
            if i < 7: physics.bind(self.mjcf_model.find("joint", f"joint{i + 1}")).qpos = qpos
            else: physics.bind(self.mjcf_model.find("actuator", f"gripper")).ctrl = qpos
        physics.data.ctrl = self.default_qpos
        
    def get_qpos(self, physics):
        qposes = []
        for joint in self.joints[:7]:
            qpos = physics.bind(joint).qpos
            qposes.append(qpos)
        return qposes
    
    def get_qvel(self, physics):
        qvels = []
        for joint in self.joints[:7]:
            qvel = physics.bind(joint).qvel
            qvels.append(qvel)
        return qvels
    
    def get_qacc(self, physics):
        qaccs = []
        for joint in self.joints[:7]:
            qacc = physics.bind(joint).qacc
            qaccs.append(qacc)
        return qaccs
    
    def get_ee_open_state(self, physics=None):
        gripper = self.mjcf_model.find("actuator", "gripper")
        gripper_pos = physics.bind(gripper).ctrl
        if gripper_pos == 0:
            return True
        else:
            return False
    
    def get_ee_state(self, physics):
        pos = np.array(self.get_end_effector_pos(physics))
        quat = np.array(self.get_end_effector_quat(physics))
        open = np.array(self.get_ee_open_state(physics)).astype(np.float32).reshape((1,))
        return np.concatenate([pos, quat, open])
    
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