import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask

@register.add_config_manager("store_food")
class StoreFoodConfigManager(BenchTaskConfigManager):
    """
    Store the food into the fridge. 
    The snacks or other objects should not be in the fridge.
    """
    def __init__(self, 
                 task_name,
                 num_objects=[3, 4],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def load_containers(self, target_container):
        super().load_containers(target_container)
        self.config["task"]["components"][-1]["position"] = [0.1, 0.5, 0]
    
    def get_seen_task_config(self):
        target_entities = [random.choice(self.seen_object) for _ in range(random.choice([1, 2]))]
        target_container = random.choice(self.seen_container)
        return self.get_task_config(target_entities, target_container, None)
        
    def get_unseen_task_config(self):
        target_entities = [random.choice(self.unseen_object) for _ in range(random.choice([1, 2]))]
        target_container = random.choice(self.unseen_container)
        return self.get_task_config(target_entities, target_container, None)
        
    def load_objects(self, target_entity):
        objects = []
        objects.extend(target_entity)
    
        for i, object in enumerate(objects):
            plate_config = self.get_entity_config("plate", 
                                                   position=[-1+i*0.3, random.uniform(0.2, 0.3), 0.8])
            plate_config["subentities"] = [self.get_entity_config(object, position=[0, 0, 0.1])]
            self.config["task"]["components"].append(plate_config)
        
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = [f"Please store the food properly."]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        assert isinstance(target_entity, list), "target_entity should be a list in this task"
        condition_config = dict(
            contain=dict(
                entities=target_entity,
                container=target_container
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_task("store_food")
class StoreFoodTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)