import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask

@register.add_config_manager("make_juice")
class MakeJuiceConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects = [3, 4],
                 **kwargs
                 ):
        super().__init__(task_name, num_objects, **kwargs)
        
    def get_condition_config(self, target_entity, target_container, **kwargs):
        condition_config = dict(
            contain=dict(
                container=target_container,
                entities=[target_entity]
            ),
            press_button=dict(
                target_button=target_container
            )
        )
        self.config["task"]["conditions"] = condition_config

    def load_containers(self, target_container):
        super().load_containers(target_container)
        if hasattr(self.config["task"], "random_ignored_entities"):
            self.config["task"]["random_ignored_entities"].append(target_container)
        else:
            self.config["task"]["random_ignored_entities"] = ["table", target_container]
        self.config["task"]["components"][-1]["position"] = [random.uniform(-0.3, -0.1), random.uniform(-0.1, 0.2), 0.8]
        cap_config = self.get_entity_config("juicer_cap", position=[0, 0, 0.5])
        self.config["task"]["components"][-1]["subentities"] = [cap_config]
        
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Make juice by squeezing the {target_entity}."]
        self.config["task"]["instructions"] = instruction

@register.add_task("make_juice")
class MakeJuiceTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        self.config_manager_cls = register.load_config_manager("make_juice")
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("find_fruit_to_make_juice")
class FindFruitToMakeJuiceTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        self.config_manager_cls = register.load_config_manager("find_fruit_to_make_juice")
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("insert_power_cord_to_make_juice")
class InsertPowerCordToMakeJuiceTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        self.config_manager_cls = register.load_config_manager("insert_power_cord_to_make_juice")
        super().__init__(task_name, robot=robot, **kwargs)