import random
import numpy as np
from functools import partial
from VLABench.tasks.hierarchical_tasks.primitive.select_drink_series import SelectDrinkConfigManager, SelectDrinkTask
from VLABench.utils.register import register
from VLABench.utils.utils import flatten_list, euler_to_quaternion
from VLABench.utils.skill_lib import SkillLib

@register.add_config_manager("cool_drink")
class CoolDrinkTaskConfigManager(SelectDrinkConfigManager):
    """
    the target entity is outside the container, put into the fridge to cool down.
    """
    def __init__(self, 
                 task_name,
                 **kwargs):
        super().__init__(task_name, **kwargs)
    
    def load_objects(self, target_entity):
        outside_objects = []
        outside_objects.append(target_entity)
        
        all_objects = []
        for seen_drink, unseen_drink in zip(self.seen_object, self.unseen_object):
            all_objects.append(seen_drink+unseen_drink)
        
        # remove the target type drink (such as sota)
        for similar_drink in all_objects:
            if target_entity in similar_drink:
                all_objects.remove(similar_drink)
                
        other_objects = flatten_list(all_objects)
        
        inside_object = random.sample(other_objects, self.num_object-1)
    
        self.target_entity = f"{target_entity}_outside"
        outside_objects.extend(random.sample(other_objects, self.num_object-1))
        inside_object_configs = []
        
        random.shuffle(inside_object)
        random.shuffle(outside_objects)
        for i, inside_object in enumerate(inside_object):
            object_config = self.get_entity_config(inside_object,
                                                   position=[(i-0.5)*0.12, random.uniform(-0.04, -0.02), 0.15], 
                                                   )
            inside_object_configs.append(object_config) 

        for i, object in enumerate(outside_objects):
            object_config = self.get_entity_config(object,
                                                   position=[random.uniform(0.25, 0.35), random.uniform(i*0.1, (i+1)*0.1), 0.85],
                                                   specific_name=f"{object}_outside",
                                                   )
            self.config["task"]["components"].append(object_config)
        
        return inside_object_configs
    
    def get_condition_config(self, target_entity, init_container, **kwargs):
        conditions_config = dict(
            contain=dict(
                container=init_container,
                entities=[f"{self.target_entity}"]
            )
        )
        self.config["task"]["conditions"] = conditions_config
    
    def get_instruction(self, target_entity, init_container, **kwargs):
        instruction = [f"I am so thirsty after sport, I want to drink something healthy cool"]
        self.config["task"]["instructions"] = instruction
        
@register.add_config_manager("take_out_cool_drink")
class TakeCoolDrinkConfigManager(CoolDrinkTaskConfigManager):
    """
    Target entity is inside the container and there is a same one outside, take it out to drink.
    """
    def load_objects(self, target_entity):
        outside_objects = [target_entity]
        inside_object = [target_entity]
        self.target_entity = target_entity
        
        all_objects = []
        for seen_drink, unseen_drink in zip(self.seen_object, self.unseen_object):
            all_objects.append(seen_drink+unseen_drink)
        
        # remove the target type drink (such as sota)
        for similar_drink in all_objects:
            if target_entity in similar_drink:
                all_objects.remove(similar_drink)
                
        # extend other drinks into outside_objects and inside objects
        other_objects = flatten_list(all_objects)
        inside_object.extend(random.sample(other_objects, self.num_object-1))
        outside_objects.extend(random.sample(other_objects, self.num_object-1))
        random.shuffle(inside_object)
        random.shuffle(outside_objects)
        inside_object_configs = []
        for i, inside_object in enumerate(inside_object):
            object_config = self.get_entity_config(inside_object,
                                                   position=[(i-0.5)*0.12, random.uniform(-0.04, -0.02), 0.15], 
                                                   )
            inside_object_configs.append(object_config) 

        for i, object in enumerate(outside_objects):
            object_config = self.get_entity_config(object,
                                                   position=[random.uniform(0.25, 0.35), random.uniform(i*0.1, (i+1)*0.1), 0.85],
                                                   specific_name=f"{object}_outside",
                                                   )
            self.config["task"]["components"].append(object_config)
        
        return inside_object_configs
    
    def get_condition_config(self, target_entity, init_container, **kwargs):
        conditions_config = dict(
            not_contain=dict(
                container=init_container,
                entities=[f"{self.target_entity}"]
            )
        )
        self.config["task"]["conditions"] = conditions_config

@register.add_task("cool_drink")
class CoolDrinkTask(SelectDrinkTask):
    def __init__(self, task_name, robot, random_init=False, **kwargs):
        super().__init__(task_name, robot=robot, random_init=random_init, **kwargs)

    def should_terminate_episode(self, physics):
        condition_is_met = super().should_terminate_episode(physics)
        for key, entity in self.entities.items():
            if "fridge" in key:
               fridge = entity
               break
        is_close = fridge.is_closed(physics)
        if condition_is_met and is_close:
            return True
        else:
            return False
    
    def get_expert_skill_sequence(self, physics):
        pass
    
@register.add_task("take_out_cool_drink")
class TakeOutCoolDrinkTask(CoolDrinkTask):
    def __init__(self, task_name, robot, random_init=False, **kwargs):
        super().__init__(task_name, robot=robot, random_init=random_init, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.init_container, prior_eulers=[[-np.pi, 0, -np.pi/2]]),
            partial(SkillLib.open_door, target_container_name=self.init_container), 
            partial(SkillLib.lift, gripper_state=np.ones(2)*0.04, lift_height=0.1),
            partial(SkillLib.pull, gripper_state=np.ones(2)*0.04, target_quat=euler_to_quaternion(-np.pi, 0, -np.pi/2), pull_distance=0.15),
            partial(SkillLib.move_offset, offset=[0.35, 0, 0], gripper_state=np.ones(2)*0.04),
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-np.pi, 1.1, -np.pi/2]]),
            partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.1),
            partial(SkillLib.pull, pull_distance=0.3),
            partial(SkillLib.place, target_container_name="table", target_pos=np.array([0.2, 0.1, 0.83])),
            partial(SkillLib.lift, lift_height=0.3),
            partial(SkillLib.pull, pull_distance=0.2),
            partial(SkillLib.move_offset, gripper_state=np.ones(2)*0.04, offset=[-0.4, 0, 0], target_quat=euler_to_quaternion(-np.pi, 0, -np.pi/2)),
            partial(SkillLib.pick, target_entity_name=self.init_container, prior_eulers=[[-np.pi, 0, np.pi]]),
            partial(SkillLib.close_door, target_container_name=self.init_container),
        ]
        return skill_sequence