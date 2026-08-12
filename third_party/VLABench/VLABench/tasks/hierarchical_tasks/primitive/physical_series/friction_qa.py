import random
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive.physical_series.base import *

@register.add_config_manager("friction_qa")
class FrictionQAConfigManager(PhysicalQAConfigManager):    
    def load_objects(self, materials):
        components_config = self.config["task"]["components"]
        chosen_materials = [random.choice(materials) for _ in range(self.num_object)]
        geom_types = [random.choice(["box", "sphere", "cylinder"]) for _ in range(self.num_object)]
        target_geom_type, geom_types = self.get_target_geom(chosen_materials, geom_types)
        self.target_geom_type = target_geom_type
        random.shuffle(geom_types)
        button_configs = []
        for config in components_config:
            if config["class"] == "Button" or config["class"] == Button:
                button_configs.append(config)
        
        for _, (button_config, geom, material) in enumerate(zip(button_configs, geom_types, chosen_materials)):
            if geom == target_geom_type:
                self.target_button = button_config["name"]
            geom_config = dict(
                name=f"{material}_{geom}",
                geom_type=geom,
                material=material,
                position=[0, 0.1, 0.05]
            )
            geom_config["class"] = RandomGeom
            button_config["subentities"] = [geom_config]
        
    def get_instruction(self, **kwargs):            
        if self.target_order_index == 0:
            instruction = ["We are going to test the friction of the different object. Choose the object that slides the fastest on a steel ramp."]
        elif self.target_order_index == 1:
            instruction = ["We are going to test the friction of the different object. Choose the object that slides the second fastest on a steel ramp."]
        elif self.target_order_index == 2:
            instruction = ["We are going to test the friction of the different object. Choose the object that slides the slowest on a steel ramp."]
        self.config["task"]["instructions"] =instruction
        return self.config
    
    def get_target_geom(self, materials, geom_types:list):
        num_sphere = geom_types.count("sphere")
        num_cylinder = geom_types.count("cylinder")
        num_box = geom_types.count("box")
        if num_sphere == 3 or num_cylinder == 3 or num_box == 3: # shape [3, 0, 0]
            """
            avoid all the shapes are the same
            """
            if num_sphere == 3: 
                geom_types[-1] = random.choice(["cylinder", "box"])
                self.target_order_index = 2
            elif num_cylinder == 3: 
                geom_types[-1] = random.choice(["sphere", "box"])
                if geom_types[-1] == "sphere": self.target_order_index = 0
                elif geom_types[-1] == "box": self.target_order_index = 2
            elif num_box == 3: 
                geom_types[-1] = random.choice(["sphere", "cylinder"])
                self.target_order_index = 0
            return geom_types[-1], geom_types
 
        elif num_sphere == 0 or num_cylinder == 0 or num_box == 0: # shape [2, 1, 0]
            if num_sphere == 1:
                self.target_order_index = 0 
                return "sphere", geom_types
            elif num_cylinder == 1: 
                if num_box == 0: self.target_order_index = 2
                elif num_sphere == 0: self.target_order_index = 0
                return "cylinder", geom_types
            else: 
                self.target_order_index = 2
                return "box", geom_types
        else: # shape [1, 1, 1]
            target_geom = random.choice(geom_types)
            if target_geom == "box": self.target_order_index = 2
            elif target_geom == "sphere": self.target_order_index = 0
            elif target_geom == "cylinder": self.target_order_index = 1
            return target_geom, geom_types

@register.add_task("friction_qa")
class FrictionQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)