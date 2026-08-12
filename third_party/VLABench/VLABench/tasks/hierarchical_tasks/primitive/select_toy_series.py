import numpy as np
import random
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register

@register.add_config_manager("select_toy")
class SelectToyConfigManager(BenchTaskConfigManager):    
    def load_objects(self, target_entity):
        objects = []
        objects.append(target_entity)
        other_objects = self.seen_object.copy() + self.unseen_object.copy()
        for ip_objects in self.seen_object + self.unseen_object:
            if isinstance(ip_objects, list) and target_entity in ip_objects:
                other_objects.remove(ip_objects)
            elif isinstance(ip_objects, str) and target_entity == ip_objects:
                other_objects.remove(ip_objects)
        other_objects_flatten = []
        for ip_objects in other_objects:
            other_objects_flatten.extend(ip_objects)
        objects.extend(random.sample(other_objects_flatten, self.num_object-1))
        for object in objects:
            object_config = self.get_entity_config(object, orientation=random.choice([[np.pi/2, 0, 0], [np.pi/2, 0, np.pi]]))
            self.config["task"]["components"].append(object_config)
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        conditions_config = dict(
            contain=dict(
                container=f"{target_container}",
                entities=[f"{target_entity}"]
            )
        )
        self.config["task"]["conditions"] = conditions_config
        return conditions_config
    
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = [f"Put the {target_entity} into the {target_container}"]
        self.config["task"]["instructions"] = instruction
        return instruction

@register.add_config_manager("select_toy_common_sense")
class SelectToyCommonSenseConfigManager(SelectToyConfigManager):
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = [f"Select the toy model of disney"]
        self.config["task"]["instructions"] = instruction
        return instruction

@register.add_config_manager("select_toy_spatial")
class SelectToySpatialConfigManager(SelectToyConfigManager):
    def load_containers(self, target_container):
        self.target_container = f"target_{target_container}"
        target_container_config = self.get_entity_config(target_container, position=[random.uniform(-0.35, -0.3), random.uniform(-0.2, 0.2), 0.8], specific_name=f"target_{target_container}")
        another_container_config = self.get_entity_config(target_container, position=[random.uniform(0.3, 0.35), random.uniform(-0.2, 0.2), 0.8], specific_name=f"another_{target_container}")
        self.config["task"]["components"].extend([target_container_config, another_container_config])
    
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = [f"Put {target_entity} into TODO {target_container}"]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        super().get_condition_config(target_entity, target_container, **kwargs)
        self.config["task"]["conditions"]["contain"]["container"] = f"target_{target_container}"
        
@register.add_config_manager("select_toy_semantic")
class SelectToySemanticConfigManager(SelectToyConfigManager):
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = [f"It's a conversation"]
        self.config["task"]["instructions"] = instruction

@register.add_task("select_toy")
class SelectToyTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        grasppoint = np.array(self.entities[self.target_entity].get_grasped_keypoints(physics)[-1])
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, target_pos=grasppoint - np.array([0, 0, 0.02]), prior_eulers=[[-np.pi, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.3, gripper_state=np.zeros(2)),
            partial(SkillLib.place, target_container_name=self.target_container)
        ]
        return skill_sequence
    
@register.add_task("select_toy_spatial")
class SelectToySpatialTask(SelectToyTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)
        
    def initialize_episode(self, physics, random_state):
        res = super().initialize_episode(physics, random_state)
        # compute the distance
        target_pos, another_pos = None, None
        for key, entity in self.entities.items():
            if "target" in key: target_pos = entity.init_pos[:2]
            elif "another" in key: another_pos = entity.init_pos[:2]
        assert target_pos is not None and another_pos is not None, "could not find valid target and another container"
        robot_pos = self.robot.robot_config["position"][:2]
        spatial_description = []    
        # left or right
        spatial_description.append("left") if target_pos[0] < another_pos[0] else spatial_description.append("right")
        # distance
        spatial_description.append("nearest") if np.linalg.norm(target_pos - robot_pos) < np.linalg.norm(another_pos - robot_pos) else spatial_description.append("farthest")
        # print(f"target_pos: {target_pos}, another_pos: {another_pos}, robot_pos: {robot_pos}, spatial_description: {spatial_description}")
        if isinstance(self.instructions, list):
            self.instructions[0] = self.instructions[0].replace("TODO", f"{spatial_description}")
        return res

@register.add_task("select_toy_common_sense")
class SelectToyCommonSense(SelectToyTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("select_toy_semantic")
class SelectToySemantic(SelectToyTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)
