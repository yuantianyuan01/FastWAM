"""
Register the common entities used in the VLABench
"""
import numpy as np
from VLABench.tasks.components.entity import CommonGraspedEntity
from VLABench.utils.register import register
from VLABench.utils.utils import rotate_point_around_axis, distance

@register.add_entity("Toy")
class Toy(CommonGraspedEntity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register.add_entity("Book")
class Book(CommonGraspedEntity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register.add_entity("Drink")
class Drink(CommonGraspedEntity):
    def get_grasped_keypoints(self, physics):
        grasp_keypoints = []
        if len(self.grasp_sites(physics)) > 0: grasp_keypoints.extend([physics.bind(site).xpos for site in self.grasp_sites(physics)])
        else: grasp_keypoints.append(self.get_xpos(physics) + np.array([0, 0, 0.03])) 
        return grasp_keypoints

@register.add_entity("Snack")
class Snack(CommonGraspedEntity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register.add_entity("Fruit")
class Fruit(CommonGraspedEntity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register.add_entity("Ingredient")
class Ingredient(CommonGraspedEntity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register.add_entity("Mug")
class Mug(CommonGraspedEntity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@register.add_entity("GiftBoxCover")
class GiftBoxCover(CommonGraspedEntity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
@register.add_entity("Bread")
class Bread(CommonGraspedEntity):
    def _build(self, **kwargs):
        super()._build(**kwargs)
        
@register.add_entity("Dessert")
class Dessert(CommonGraspedEntity):
    def _build(self, **kwargs):
        super()._build(**kwargs)

@register.add_entity("Hammer")
class Hammer(CommonGraspedEntity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@register.add_entity("HammerHandle")
class HammerHandle(CommonGraspedEntity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def get_top_sites(self):
        top_sites = [site for site in self.sites if "top" in site.name]
        return top_sites
    
    def get_top_site_pos(self, physics):
        top_sites = self.get_top_sites
        top_site_pos = [physics.bind(site).xpos for site in top_sites]
        return top_site_pos    

@register.add_entity("Nail")
class Nail(CommonGraspedEntity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@register.add_entity("Laptop")
class Laptop(CommonGraspedEntity):
    def _build(self, 
               open_threshold=1.3, 
               close_threshold=0.1,
               **kwargs):
        super()._build(**kwargs)
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
    
    @property
    def screen_joint(self):
        for joint in self.joints:
            if "screen" in joint.name:
                return joint
        return None
    
    def is_open(self, physics):
        joint_pos = physics.bind(self.screen_joint).qpos
        if abs(joint_pos) > self.open_threshold:return True
        else:return False

    def is_closed(self, physics):
        joint_pos = physics.bind(self.screen_joint).qpos
        if abs(joint_pos) < self.close_threshold:return True
        else:return False
    
    def get_grasped_keypoints(self, physics):
        return super().get_grasped_keypoints(physics)
        
    def get_open_trajectory(self, physics):
        trajectory = []
        init_pos = np.array(self.get_grasped_keypoints(physics)[-1])
        rotation_axis = physics.bind(self.screen_joint).xaxis
        rotaion_anchor = physics.bind(self.screen_joint).xanchor
        current_joint_qpos = physics.bind(self.screen_joint).qpos
        target_joint_qpos = physics.bind(self.screen_joint).range[-1]
        delta_angles = np.arange(0.1, target_joint_qpos-current_joint_qpos[0], 0.1)
        for delta_angle in delta_angles:
            new_pos = rotate_point_around_axis(init_pos, rotaion_anchor, rotation_axis, delta_angle)
            trajectory.append(new_pos)
        return trajectory
    
    def get_close_trajectory(self, physics):
        trajectory = []
        init_pos = np.array(self.get_grasped_keypoints(physics)[-1])
        rotation_axis = physics.bind(self.screen_joint).xaxis
        rotaion_anchor = physics.bind(self.screen_joint).xanchor
        current_joint_qpos = physics.bind(self.screen_joint).qpos
        target_joint_qpos = physics.bind(self.screen_joint).range[0]
        delta_angles = np.arange(-0.1, target_joint_qpos-current_joint_qpos[0], -0.1)
        for delta_angle in delta_angles:
            new_pos = rotate_point_around_axis(init_pos, rotaion_anchor, rotation_axis, delta_angle)
            trajectory.append(new_pos)
        return trajectory

@register.add_entity("CardHolder")
class CardHolder(CommonGraspedEntity):
    """
    The naive card holder to place cards such as pokers and nametag
    """
    def _build(self,
               **kwargs):
        super()._build(**kwargs)
        size = kwargs.get("size", [0.03, 0.02, 0.012])
        interval = kwargs.get("interval", 0.005)
        rgba = kwargs.get("rgba", [0.5, 0.5, 0.5, 1])
        self.mjcf_model.worldbody.add("geom", 
                                  type="box",  
                                  size=size,
                                  pos=[0, 0, size[2]/2], 
                                  rgba=rgba,
                                  solref=[0.001, 1],
                                  mass=1,
                                  group="1",
                                  contype="1",
                                  conaffinity="1")

        self.mjcf_model.worldbody.add("geom",
                                type="box",
                                size=[size[0]/2 - interval, size[1], size[2]],
                                pos=[-(size[0]/2 + 2 * interval), 0, 3*size[2]/2],
                                rgba=rgba,
                                solref=[0.001, 1],
                                group="1",
                                contype="1",
                                conaffinity="1")
        self.mjcf_model.worldbody.add("geom",
                                type="box",
                                size=[size[0]/2 - interval, size[1], size[2]],
                                pos=[(size[0]/2 + 2 * interval), 0, 3*size[2]/2],
                                rgba=rgba,
                                solref=[0.001, 1],
                                group="1",
                                contype="1",
                                conaffinity="1")

@register.add_entity("Cord")
class Cord(CommonGraspedEntity):
    def _build(self, **kwargs):
        super()._build(**kwargs)
        