import os
import yaml
import json
import random
import numpy as np
from VLABench.utils.utils import euler_to_quaternion, quaternion_multiply 
from VLABench.utils.register import register
from dm_control import composer
from dm_control import mjcf

class Entity(composer.Entity):
    def __init__(self, *args, **kwargs):
        self.asset_root = os.path.join(os.getenv("VLABENCH_ROOT"), "assets")
        self.texture_root = os.path.join(os.getenv("VLABENCH_ROOT"), "assets/obj/assets/textures")
        self.entity_asset_root = os.path.join(os.getenv("VLABENCH_ROOT"), "assets/obj/meshes")
        super().__init__(*args, **kwargs)
        
    def _build(self, *args, **kwargs):
        """
        parameter:
            -name: str, the name of the entity
            -xml_path: str, the path of the xml file
            -position: 3d list/np.array, the initial position of the entity in the parent entity frame
            -orientation: 3d/4d list/np.array, the initial orientation of the entity in the parent entity frame, by euler or quanternion
            -randomness: dict, the randomness config for domain randomization, including offset of pose, texture, mesh scale, etc.
            -subentities: list, the subentities of the entity. The subentities will be attached to the entity or initilized in entity's frame.
            -parent_entity: Entity, the parent entity of the entity. If the parent entity is not None, the entity will be attached to the parent entity.
        """
        xml_path = kwargs.get("xml_path", None)
        self.relative_xml_path = xml_path.split(self.asset_root)[-1][1:] if xml_path is not None else None # for save the entity info
        name = kwargs.get("name", "entity")
        if xml_path is not None:
            self._mjcf_model = mjcf.from_path(xml_path)
        else:
            self._mjcf_model = mjcf.RootElement()
        self._mjcf_model.model = name
        self.init_pos = np.array(kwargs.get("position", [0, 0, 0]))
        self.init_quat = np.array(kwargs.get("orientation", [1, 0, 0, 0]))
        self.randomness = kwargs.get("randomness", None)
        self.subentities = kwargs.get("subentities", None)
        self.parent_entity = kwargs.get("parent_entity", None)
        if self.randomness is not None:
            assert isinstance(self.randomness, dict), "randomness config should be a dictionary"
            for k, v in self.randomness.items():
                if isinstance(v, list):
                    self.randomness[k] = np.array(v)
        if len(self.init_quat) == 3:
            self.init_quat = np.array(euler_to_quaternion(self.init_quat[0], self.init_quat[1], self.init_quat[2]))
    
    def initialize_episode(self, physics, random_state):
        """
        Take domain randomization here.
        """
        new_xpos, new_xquat = self.init_pos.copy(), self.init_quat.copy()
        if self.parent_entity is not None:
            if self.parent == self.parent_entity: pass# if the attached parent equals to the value parent_entity, no additional transformaton need to apply
            else: # else, compute the relative pose of the parent entity
                new_xpos += self.parent_entity.init_pos
                new_xquat = quaternion_multiply(self.parent_entity.init_quat, new_xquat)
        if self.randomness is not None:
            if self.randomness.get("pos", None) is not None:
                new_xpos = new_xpos + self.randomness["pos"] * random_state.uniform([-1, -1, -1], [1, 1, 1])
            if self.randomness.get("quat", None) is not None:
                new_xquat = quaternion_multiply(new_xquat, 
                                                euler_to_quaternion(*(self.randomness["quat"] * random_state.uniform([-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]))))
            if self.randomness.get("scale", None) is not None:
                # modify the scale of the mesh, size and relative pos of geom
                self.set_scale(physics, self.randomness.get("scale"))
            if self.randomness.get("texture", None) is not None and self.randomness.get("texture") != False:
                # modify the texture of the mesh
                self.set_texture(physics, self.randomness.get("texture"))
        self.set_pose(physics, new_xpos, new_xquat)
        return super().initialize_episode(physics, random_state)
        
    @property
    def mjcf_model(self):
        return self._mjcf_model
    
    @property
    def name(self):
        return self.mjcf_model.model
    
    @property
    def actuators(self):
        return self._mjcf_model.find_all('actuator')
    
    @property
    def bodies(self):
        return self._mjcf_model.find_all('body')
    
    @property
    def geoms(self):
        return self._mjcf_model.find_all('geom')
    
    @property
    def sites(self):
        return self.mjcf_model.find_all("site")
    
    @property
    def joints(self):
        return self.mjcf_model.find_all("joint")
    
    def place_sites(self, physics):
        """
        Special type of sites, group = 2. 
        Annotating the placement point of the receptacle. 
        E.g.the recommended placement of the point of a plate is the center above the plate.
        """
        place_sites = []
        for site in self.sites:
            if physics.bind(site).group == 2:
                place_sites.append(site)
        return place_sites
    
    def key_sites(self, physics):
        """
        Special type of sites, group = 3.
        Annotating the key points of the receptacle, usually the bounding box points.
        E.g. the two diagonal points of a box.
        """
        key_sites = []
        for site in self.sites:
            if physics.bind(site).group == 3:
                key_sites.append(site)
        return key_sites
        
    def grasp_sites(self, physics):
        """
        Special type of sites, group = 4.
        Annotating the recommonded grasp points of the entity.
        E.g. the grasp point of an apple is the center point
        """
        grasp_sites = []
        for site in self.sites:
            if physics.bind(site).group == 4:
                grasp_sites.append(site)
        return grasp_sites
      
    
    def get_xpos(self, physics):
        return physics.bind(self._mjcf_model.worldbody).xpos
    
    def get_xqaut(self, physics):
        return physics.bind(self._mjcf_model.worldbody).xquat

    def build_from_config(self, config_path, **kwargs):
        config = self.load_config(config_path)
        self._build(**config)
    
    def load_config(self, config_path:str):
        if config_path.endswith(".yaml"):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        elif config_path.endswith(".json"):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            raise ValueError("config file should be a yaml or json file")
        assert isinstance(config, dict), "config should be a dictionary"
        return config
    
    def get_entity_pcd(self, env):
        """
        Get the point cloud of the entity
        """
        geom_ids = [env.physics.bind(geom).element_id for geom in self.geoms]
        obs = env.get_observation()
        rgb = obs["rgb"]
        depth = obs["depth"]
        segmentation = obs["segmentation"]
        masks = np.where((segmentation[..., 0] <= max(geom_ids))&(segmentation[..., 0] >= min(geom_ids)), 1, 0).astype(np.uint8)
        env.pcd_generator.physics = env.physics
        entity_pcd = env.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(env.physics.model.ncam - 1)), 
                                                                rgb=rgb,
                                                                depth=depth,
                                                                mask=masks)
        return entity_pcd

    def set_texture(self, physics, texture):
        """
        Change the texture of the entity's surface
        """
        raise NotImplementedError
    
    def set_scale(self, physics, scale_multiplier):
        """
        Change the scale of the entity's geoms and their relative poses for geometry consistance.
        """
        if not hasattr(self, "scale_ratio_to_recover"): self.scale_ratio_to_recover = 1
        self.reset_scale()
        if isinstance(scale_multiplier, list) or isinstance(scale_multiplier, np.ndarray) or isinstance(scale_multiplier, tuple):
            scale_ratio = random.uniform(scale_multiplier[0], scale_multiplier[1])
        elif isinstance(scale_multiplier, float) or isinstance(scale_multiplier, int):
            scale_ratio = scale_multiplier
        else:
            raise TypeError("scale_multiplier should be a list or tuple to represent a range or a float or int")
        self.scale_ratio_to_recover = 1 / scale_ratio # update the scale ratio to recover
        self.apply_scale(scale_ratio)
        
    def reset_scale(self):
        """
        Reset the scale of the entity to the original scale before apply new scale
        """
        self.apply_scale(self.scale_ratio_to_recover)
    
    def apply_scale(self, scale_ratio):
        """
        Apply scale transform
        """
        meshes = self.mjcf_model.find_all("mesh")
        for mesh in meshes:
            if mesh.scale is not None:
                mesh.scale = mesh.scale * scale_ratio
            else:
                mesh.scale = np.array([scale_ratio, scale_ratio, scale_ratio]) 
        
        element_to_scale = self.geoms + self.sites
        for elem in element_to_scale:
            if elem.size is not None:
                elem.size = elem.size * scale_ratio
            if elem.pos is not None:
                elem.pos = elem.pos * scale_ratio
    
    def save(self, physics):
        info_to_save=dict(
            name=self.mjcf_model.model,
            xml_path=self.relative_xml_path,
            position=self.get_xpos(physics).tolist(),
            orientation=self.get_xqaut(physics).tolist(),
        )
        info_to_save["class"] = self.__class__.__name__
        return info_to_save
    
@register.add_entity("CommonGraspedEntity")
class CommonGraspedEntity(Entity):    
    def is_grasped(self, physics, robot):
        """
        Judge whether the entity is grasped by the robot with the contact force
        """
        gripper_geoms = robot.gripper_geoms
        gripper_geom_ids = [physics.bind(geom).element_id for geom in gripper_geoms]
        entity_geom_ids = [physics.bind(geom).element_id for geom in self.geoms]
        contacts = physics.data.contact
        for contact in contacts:
            if (contact.geom1 in gripper_geom_ids and contact.geom2 in entity_geom_ids) or \
                (contact.geom2 in gripper_geom_ids and contact.geom1 in entity_geom_ids):
                return True
        return False
    
    def get_grasped_keypoints(self, physics):
        """
        Return a valid grasp position and quaternion.
        If no annotated grasp site, return the center of the entity.
        """ 
        grasp_keypoints = []
        if len(self.grasp_sites(physics)) > 0:
            grasp_keypoints.extend([physics.bind(site).xpos for site in self.grasp_sites(physics)])
        else:
            grasp_keypoints.append(self.get_xpos(physics)) 
        return grasp_keypoints
    