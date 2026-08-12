import random
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive.physical_series.base import *

@register.add_config_manager("reflection_qa")
class ReflectionQAConfigManager(PhysicalQAConfigManager):        
    def load_objects(self, *args, **kwargs):
        candidate_reflections = [random.uniform(0.2, 0.4), random.uniform(0.5, 0.7), random.uniform(0.8, 1)]
        target_reflection = random.choice(candidate_reflections)
        self.order_index = candidate_reflections.index(target_reflection)
        
        components_config = self.config["task"]["components"]
        button_configs = []
        for _, config in enumerate(components_config):
            if config["class"] == "Button" or config["class"] == Button:
                button_configs.append(config)
        random.shuffle(candidate_reflections)
        for _, (button_config, reflection) in enumerate(zip(button_configs, candidate_reflections)):
            if reflection == target_reflection:
                self.target_button = button_config["name"]
        mirror_config = self.get_entity_config("mirrors", position=[0, 0.1, 0], reflectance=candidate_reflections)
        button_config["subentities"] = [mirror_config]
    
    def get_instruction(self, **kwargs):
        if self.order_index == 0: # the smallest reflection ratio
            instruction = ["Choose the mirror with smallest reflection ratio."]
        elif self.order_index == 1: # the medium reflection ratio
            instruction = ["Choose the mirror with the medium reflection ratio."]
        elif self.order_index == 2: # the largest reflection ratio
            instruction = ["Choose the mirror with the largest reflection ratio."]
        self.config["task"]["instructions"] = instruction

@register.add_task("reflection_qa")
class ReflectionQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)