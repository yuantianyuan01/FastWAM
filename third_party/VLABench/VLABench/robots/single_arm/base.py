import os
import numpy as np
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from VLABench.robots.base import Robot
from VLABench.utils.utils import matrix_to_quaternion

class SingleArm(Robot):
    """
    Base class for single arm robots.
    Params:
        n_dof: number of degrees of freedom for the robot
    """
    def __init__(self, n_dof=7, **kwargs):
        super().__init__(**kwargs)
        self._n_dof = n_dof
    
    @property
    def link_base(self):
        return self.mjcf_model.find("body", "link_base")
    
    @property
    def end_effector_site(self):
        return self.mjcf_model.find("site", "end_effector")
    
    @property
    def position_limits(self):
        raise NotImplementedError
    
    @property
    def velocity_limits(self):
        raise NotImplementedError  
    
    @property
    def gripper_geoms(self):
        raise NotImplementedError
    
    @property
    def n_dof(self):
        return self._n_dof
    
    def get_qpos(self, physics):
        qposes = []
        for joint in self.joints[:self.n_dof]:
            qpos = physics.bind(joint).qpos
            qposes.append(qpos)
        return np.array(qposes).reshape(-1)
    
    def get_qvel(self, physics):
        qvels = []
        for joint in self.joints[:self.n_dof]:
            qvel = physics.bind(joint).qvel
            qvels.append(qvel)
        return np.array(qvels).reshape(-1)
    
    def get_qacc(self, physics):
        qaccs = []
        for joint in self.joints[:self.n_dof]:
            qacc = physics.bind(joint).qacc
            qaccs.append(qacc)
        return np.array(qaccs).reshape(-1)
    
    def set_base_position(self, pos):
        """
        change the base world position for the robot
        """
        assert len(pos) == 3, "pos must be a 3D vector"
        self.link_base.pos = pos
    
    def set_base_orientation(self, ori):
        """
        change the base world orientation for the robot in quaternion
        """
        assert len(ori) == 4 or len(ori) == 3, "quat must be a quaternion or euler"
        if len(ori) == 4:
            self.link_base.quat = ori
        elif len(ori) == 3:
            self.link_base.euler = ori
    
    def get_qpos_from_ee_pos(self, physics, pos, quat=None, inplace=False, **kwargs):
        """
        get the joint angles by inverse kinematics from the end effector pose
        """
        ik_result = qpos_from_site_pose(physics,
                                        site_name=self.end_effector_site.full_identifier,
                                        target_pos=pos,
                                        target_quat=quat,
                                        inplace=inplace,
                                        **kwargs)
        success = ik_result.success
        target_qpos = ik_result.qpos
        return success, target_qpos[:self.n_dof]
    
    def get_end_effector_pos(self, physics=None):
        return physics.bind(self.end_effector_site).xpos
    
    def get_end_effector_quat(self, physics=None):
        matrix = physics.bind(self.end_effector_site).xmat
        return matrix_to_quaternion(matrix)