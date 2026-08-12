import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask
from VLABench.utils.utils import grid_sample

drinks = ["cola", "spirit", "jumpstart", "redbull", "monster"]
fruits = ["apple", "banana", "orange", "pear", "peach", "lemon", "kiwi", "mango"]
snacks = ["bagged_food", "bar", "boxed_food", "canned_food", "chips", "chocolate"]

def get_inside_euler(obj):
    """
    As objects have different orientations when they are inside the container, this function is used to distinguish them.
    """
    if obj in drinks:
        return [np.pi/2, 0, 0]
    elif obj in fruits:
        return [0, 0, 0]
    elif obj in snacks:
        if obj in ["bar", "chocolate", "bagged_food", "chips"]:
            return [0, 0, 0]
        elif obj in ["canned_food"]:
            return [np.pi/2, 0, 0]

@register.add_config_manager("find_unseen_object")
class FindUnseenObjectConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects = [4, 5],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def load_init_containers(self, init_container):
        super().load_init_containers(init_container)
        self.config["task"]["components"][-1]["position"] = [random.uniform(-0.1, 0.1), 
                                                            random.uniform(0.15, 0.2), 
                                                            0.78]
        self.config["task"]["components"][-1]["subentities"] = list()
        self.config["task"]["components"][-1]["randomness"] = None
        self.config["task"]["random_ignored_entities"].append(init_container)
    
    def load_objects(self, target_entity, shuffle=False, **kwargs):
        objects = []
        objects.append(target_entity)
        # other_objects = self.seen_object.copy() + self.unseen_object.copy()
        other_objects = [] 
        for seen_obj, unseen_obj in zip(self.seen_object, self.unseen_object):
            other_objects.append(seen_obj.copy() + unseen_obj.copy())
        for similar_objects in other_objects:
            if target_entity in similar_objects:
                other_objects.remove(similar_objects)
        other_objects_flatten = []
        for similar_objects in other_objects:
            other_objects_flatten.extend(similar_objects)
        objects.extend(random.sample(other_objects_flatten, self.num_object-1))
        
        init_container_config = self.config["task"]["components"][1]
        inside_positions = [[random.uniform(-0.05, 0.05), 
                             random.uniform(-0.05, -0.02), 
                             z] for z in random.sample([0.08, 0.21, 0.34], self.num_object//2)]
        if shuffle:
            random.shuffle(objects)
        for i, (object, pos) in enumerate(zip(objects[:self.num_object//2], inside_positions)):
            if object == target_entity:
               self.target_entity_pos = "inside"
            object_config = self.get_entity_config(object, position=pos, orientation=get_inside_euler(object))
            init_container_config["subentities"].append(object_config)
            self.config["task"]["random_ignored_entities"].append(object)   
        
        positions = grid_sample(workspace=self.config["task"]["workspace"],
                                grid_size=self.config["task"]["ngrid"],
                                n_samples=len(objects[self.num_object//2:]))
        for object, pos in zip(objects[self.num_object//2:], positions):
            pos = [pos[0], pos[1], 0.85]
            if object == target_entity:
                self.target_entity_pos = "outside"
            object_config = self.get_entity_config(object, position=pos)
            self.config["task"]["components"].append(object_config)
    
    def get_instruction(self, target_entity, **kwargs):
        instructions = [f"find the {target_entity} for me"]
        self.config["task"]["instructions"] = instructions
        
    def get_condition_config(self, target_entity, init_container, **kwargs):
        if self.target_entity_pos == "inside":
            condition_config = dict(
                not_contain=dict(
                    container=init_container,
                    entities=[target_entity]
                )
            )
        elif self.target_entity_pos == "outside":
            condition_config = dict(
                is_grasped=dict(
                    entities=[f"{target_entity}"],
                    robot="franka"
                )
            )
        self.config["task"]["conditions"] = condition_config

@register.add_task("find_unseen_object")
class FindUnseenObjectTask(CompositeTask):    
    def __init__(self, task_name, robot, random_init=False, **kwargs):
        super().__init__(task_name, robot=robot, random_init=random_init, **kwargs)
    
    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if "cabinet" in key:
                entity.detach()
                self._arena.attach(entity)