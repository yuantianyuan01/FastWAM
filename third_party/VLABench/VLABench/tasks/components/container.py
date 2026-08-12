import numpy as np
from VLABench.tasks.components.entity import Entity
from VLABench.utils.register import register
from VLABench.utils.utils import rotate_point_around_axis, slide_point_along_axis, distance

class ContainerMiXin:
    """
    Abstract mixin class for Container/Receptacle 
    """
    def get_place_points(self, physics):
        """
        Get the place points of the container
        """
        raise NotImplementedError
    
    def caculate_keypoints(self, **kwargs):
        """
        Caculate the keypoints of the container
        """
        raise NotImplementedError
    
    def contain(self, point, physics):
        """
        Judge whether the target point is in the container
        """
        raise NotImplementedError
    
    def generate_point_in_container(self, physics, margin):
        """
        Generate a random point in the container
        """
        raise NotImplementedError
    
@register.add_entity("CommonContainer")
class CommonContainer(Entity, ContainerMiXin):
    """
    Common 3d container/receptacle, which can be used to contain objects. 
    Usually can be abstractly modeled as boxes.
    Such as boxes, drawers and etc.
    """
    def get_place_point(self, physics):
        placement_sites = self.place_sites(physics)
        place_points = [np.array(physics.bind(site).xpos) for site in placement_sites]
        return place_points
    
    def contain(self, point, physics):
        keysites = self.key_sites(physics)
        self.keypoints = np.array([physics.bind(kp).xpos for kp in keysites])
        minX, maxX, minY, maxY, minZ, maxZ = self.keypoints[:, 0].min(), self.keypoints[:, 0].max(), self.keypoints[:, 1].min(), self.keypoints[:, 1].max(), self.keypoints[:, 2].min(), self.keypoints[:, 2].max()
        if minX <= point[0] <= maxX and \
           minY <= point[1] <= maxY and \
            minZ <=point[2] <= maxZ:
               return True
        return False
        
@register.add_entity("FlatContainer")
class FlatContainer(Entity, ContainerMiXin):
    """
    Flat container/receptacle, which can be used to contain objects.
    Usually can be abstractly modeled as a plane.
    Such as plates, palce mats and etc.
    """
    def _build(self, 
               z_threshold=0.1,
               **kwargs):
        """
        Parameter:
            -z_threshold: the threshold of the height to confirm a point is in the container
        """
        super()._build(**kwargs)
        self.z_threshold = z_threshold
    
    def get_radius(self, physics):
        """
        If the container is as a circle shape, return the radius of the container
        """
        radius_site = self.mjcf_model.worldbody.find("site", "horizontal_radius_site")
        if radius_site:
            center_pos = self.get_xpos(physics)
            radius = distance(physics.bind(radius_site).xpos[:2], center_pos[:2])
            return radius
        return None
    
    def contain(self, point, physics):
        """
        Judge whether the target point is in the plant area
        """
        radius = self.get_radius(physics)
        if radius:
            center_pos = self.get_xpos(physics)
            if distance(point[:2], center_pos[:2]) < radius and \
                (point[-1] - center_pos[-1]) > 0 and (point[-1] - center_pos[-1]) < self.z_threshold:
                return True
        else:
            self.keypoints = np.array([physics.bind(kp).xpos for kp in self.key_sites(physics)])
            minX, maxX, minY, maxY = self.keypoints[:, 0].min(), self.keypoints[:, 0].max(), self.keypoints[:, 1].min(), self.keypoints[:, 1].max()
            platform_z = self.keypoints[:, -1].min()
            if minX <= point[0] <= maxX and \
            minY <= point[1] <= maxY and \
            point[2] > platform_z and point[2] < platform_z + self.z_threshold:
                return True
            return False
    
    def generate_point_in_container(self, physics, margin=[0.1, 0.1, 0.1]):
        pos = physics.bind(self.mjcf_model.worldbody).xpos
        minX, maxX, minY, maxY = self.keypoints[:, 0].min(), self.keypoints[:, 0].max(), self.keypoints[:, 1].min(), self.keypoints[:, 1].max()
        x = np.random.uniform(minX+margin[0], maxX-margin[0])
        y = np.random.uniform(minY+margin[1], maxY-margin[1])
        z = pos[-1] + margin[2]
        return np.array([x, y, z])
    
    def get_place_point(self, physics):
        placement_sites = self.place_sites(physics)
        place_points = [np.array(physics.bind(site).xpos) for site in placement_sites]
        if len(place_points) == 0:  
            place_points.append(self.get_xpos(physics) + np.array([0, 0, self.z_threshold]))
        return place_points
        
@register.add_entity("ContainerWithDoor")
class ContainerWithDoor(CommonContainer):
    """
    Container/Receptacle with door connected hinge joint.
    Default state is closed.
    Such as fridge, safe and etc. 
    """
    def __init__(self, 
                 open_threshold=np.pi/3,
                 close_threshold=np.pi/10,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.open_threshold = open_threshold
        self.closed_threshold = close_threshold
    
    @property
    def door_joint(self):
        for joint in self.joints:
            if "door" in joint.name:
                return joint
            
    def is_open(self, physics):
        joint_pos = physics.bind(self.door_joint).qpos
        if abs(joint_pos) > self.open_threshold: return True
        else: return False
    
    def is_closed(self, physics):
        """
        Note that the open & close state can not be transfered directly. 
        There exists a intermediate state.
        """
        joint_pos = physics.bind(self.door_joint).qpos
        if abs(joint_pos) < self.closed_threshold: return True
        else: return False
    
    def get_handle_pos(self, physics):
        grasp_sites = self.grasp_sites(physics)
        if len(grasp_sites) > 0: return physics.bind(np.random.choice(grasp_sites)).xpos
        else: raise ValueError("No handle site found")
        
    def get_open_trajectory(self, physics):
        trajectory = []
        init_pos = self.get_handle_pos(physics)
        rotation_axis = physics.bind(self.door_joint).xaxis
        rotaion_anchor = physics.bind(self.door_joint).xanchor
        current_joint_qpos = physics.bind(self.door_joint).qpos
        target_joint_qpos = physics.bind(self.door_joint).range[-1]
        delta_angles = np.arange(0.1, target_joint_qpos-current_joint_qpos[0], 0.1)
        for delta_angle in delta_angles:
            new_pos = rotate_point_around_axis(init_pos, rotaion_anchor, rotation_axis, delta_angle)
            trajectory.append(new_pos)
        return trajectory
    
    def get_close_trajectory(self, physics):
        trajectory = []
        init_pos = self.get_handle_pos(physics)
        rotation_axis = physics.bind(self.door_joint).xaxis
        rotaion_anchor = physics.bind(self.door_joint).xanchor
        current_joint_qpos = physics.bind(self.door_joint).qpos
        target_joint_qpos = physics.bind(self.door_joint).range[0]
        delta_angles = np.arange(-0.1, target_joint_qpos-current_joint_qpos[0], -0.1)
        for delta_angle in delta_angles:
            new_pos = rotate_point_around_axis(init_pos, rotaion_anchor, rotation_axis, delta_angle)
            trajectory.append(new_pos)
        return trajectory
    
    def get_grasped_keypoints(self, physics):
        grasp_sites = self.grasp_sites(physics)
        grasp_keypoints = []
        grasp_keypoints.extend([physics.bind(site).xpos for site in grasp_sites])
        return grasp_keypoints
    
    def is_grasped(self, physics, robot):
        # FIXME
        return True
    
@register.add_entity("ContainerWithDrawer")
class ContainerWithDrawer(CommonContainer):
    """
    Container/Receptacle with drawer connected slide joint.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drawers = []
        for body in self.bodies:
            if "drawer" in body.name:
                self.drawers.append(body)
        self.n_drawer = len(self.drawers)    
    
    def get_drawer_open_level(self, physics):
        open_levels = []
        for joint in self.joints:
            qpos = physics.bind(joint).qpos
            range = physics.bind(joint).range
            open_level = (qpos - range[0])/(range[1] - range[0])
            open_levels.append(open_level)
        return open_levels
    
    def get_drawer_handle_pos(self, physics, drawer_id):
        drawer = self.drawers[drawer_id]
        sites = drawer.find_all("site")
        for site in sites:
            if physics.bind(site).group == 4:
                return physics.bind(site).xpos
        raise ValueError("No handle site found")
    
    def get_drawer_open_trajectory(self, physics, drawer_id):
        handle_init_pos = self.get_drawer_handle_pos(physics, drawer_id)
        init_joint_qpos = physics.bind(self.joints[drawer_id]).qpos
        range = physics.bind(self.joints[drawer_id]).range
        axis = physics.bind(self.joints[drawer_id]).xaxis
        max_distance = range[1] if abs(range[1]) > abs(range[0]) else range[0]
        trajectory = []
        step = -0.05 if max_distance < init_joint_qpos else 0.05
        for distance in np.arange(0.05, max_distance - init_joint_qpos, step):
            new_pos = slide_point_along_axis(handle_init_pos, axis, distance)
            trajectory.append(new_pos)
        return trajectory
    
    def get_grasped_keypoints(self, physics):
        """
        This is a special function in container classes.
        """
        grasp_sites = self.grasp_sites(physics)
        grasp_keypoints = []
        grasp_keypoints.extend([physics.bind(site).xpos for site in grasp_sites])
        return grasp_keypoints
    
    def is_grasped(self, physics, robot):
        gripper_geoms = robot.gripper_geoms
        gripper_geom_ids = [physics.bind(geom).element_id for geom in gripper_geoms]
        entity_geom_ids = [physics.bind(geom).element_id for geom in self.geoms]
        contacts = physics.data.contact
        for contact in contacts:
            if (contact.geom1 in gripper_geom_ids and contact.geom2 in entity_geom_ids) or \
                (contact.geom2 in gripper_geom_ids and contact.geom1 in entity_geom_ids):
                return True
        return False
