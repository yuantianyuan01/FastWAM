from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from VLABench.robots.dual_arm import DualArm
from VLABench.utils.utils import matrix_to_quaternion

class Humanoid(DualArm):
    def __init__(self, *args, **kwargs):
        """
        To unifiy the interface of the humanoid robot with standard single arm, we set the main hand as the right hand by default.
        """
        super().__init__(*args, **kwargs)
        if kwargs.get("main_hand", "right") == "left":
            self.main_hand = "left"
        else:
            self.main_hand = "right"
        self.right_hand_is_open = False
        self.left_hand_is_open = False
    
    @property
    def link_base(self):
        return self.mjcf_model.find("body", "link_base")
    
    @property
    def right_hand_site(self):
        return self.mjcf_model.find("site", "right_hand_site")
    
    @property
    def left_hand_site(self):
        return self.mjcf_model.find("site", "left_hand_site")
    
    @property
    def position_limites(self):
        raise NotImplementedError
    
    def get_qpos(self, target="whole"):
        """
        Get the joint angles of the robot
        """
        if target == "whole":
            return self.get_whole_body_qpos()
        elif target == "arm":
            return self.get_arm_qpos()
        elif target == "leg":
            return self.get_leg_qpos()
        else:
            raise ValueError("target must be whole, arm or leg")
    
    def get_whole_body_qpos(self):
        raise NotImplementedError
    
    def get_arm_qpos(self):
        raise NotImplementedError
    
    def get_leg_qpos(self):
        raise NotImplementedError

    def open_hand(self, physics, hand="right"):
        raise NotImplementedError
    
    def close_hand(self, physics, hand="right"):
        raise NotImplementedError
    
    def get_qpos_from_ee_pos(self, physics, pos, quat, inplace=False, target_hand="right", **kwargs):
        target_site = self.left_hand_site if target_hand == "left" else self.right_hand_site
        ik_result = qpos_from_site_pose(physics,
                                        site_name=target_site.full_identifier,
                                        target_pos=pos,
                                        target_quat=quat,
                                        inplace=inplace,
                                        **kwargs)
        success = ik_result.success
        target_qpos = ik_result.qpos
        return success, target_qpos
    
    def set_base_position(self, pos):
        assert len(pos) == 3, "pos must be a 3D vector"
        self.link_base.pos = pos
    
    def set_base_orientation(self, ori):
        assert len(ori) == 4 or len(ori) == 3, "quat must be a quaternion or euler"
        if len(ori) == 4:
            self.link_base.quat = ori
        elif len(ori) == 3:
            self.link_base.euler = ori
    