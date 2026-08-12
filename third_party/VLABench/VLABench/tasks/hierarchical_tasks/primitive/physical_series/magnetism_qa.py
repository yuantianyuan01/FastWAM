import random
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive.physical_series.base import *

@register.add_config_manager("magnetism_qa")
class MagnetismQAConfigManager(PhysicalQAConfigManager):
    def __init__(self, 
                 task_name, 
                 num_objects=3, 
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.choose_magnetic = random.randint(0, 1) # the number 1 represents to choose magenic object,the number 0 represents to choose non-magenic object.
        self.target_button_id = random.randint(0, self.num_object-1)
        
    def get_seen_task_config(self):
        return self.get_task_config(self.seen_material)
        
    def get_unseen_task_config(self):
        return self.get_task_config(self.unseen_material)
    
    def load_objects(self, materials):
        components_config = self.config["task"]["components"]
        # geom_types = [random.choice(["box", "sphere", "cylinder", "capsule"]) for _ in range(self.num_object)]
        geom_types = [random.choice(["cylinder", "capsule"]) for _ in range(self.num_object)]
        if "metal" not in materials:
            raise ValueError("The material 'metal' is not in the list!")        
        else:
            materials_without_metal = [material for material in materials if material != "metal"]
            if not materials_without_metal:
                raise ValueError("materials_without_metal list is empty!")
        materials_list = []
        if not (0 <= self.target_button_id < self.num_object):
            raise ValueError("target_button_id is out of range!")
        
        if self.choose_magnetic ==0:# No magnetism 
            for i in range(self.num_object):
                if i==self.target_button_id:
                    random_material = random.choice(materials_without_metal)
                    materials_list.append(random_material)
                else:
                    materials_list.append("metal")
        elif self.choose_magnetic ==1:# With magnetism
            for i in range(self.num_object):
                if i==self.target_button_id:
                    materials_list.append("metal")
                else:
                    random_material = random.choice(materials_without_metal)
                    materials_list.append(random_material)
        else:
            raise AttributeError("Random function is wrong!")

        button_configs = []
        for _, config in enumerate(components_config):
            if config["class"] == "Button" or config["class"] == Button:
                button_configs.append(config)
       
        for i, (button_config, material, geom_type) in enumerate(zip(button_configs, materials_list, geom_types)):
            if i == self.target_button_id:
                if material == "metal" and self.choose_magnetic==1:
                    self.target_button = button_config["name"]
                    self.target_entity = f"{material}_{geom_type}"

                elif material != "metal" and self.choose_magnetic==0:
                    self.target_button = button_config["name"]
                    self.target_entity = f"{material}_{geom_type}"

                else:
                    raise ValueError(f"material name dismatchs with target_button_id, material_i is {material}")
                    
            geom_config = dict(
                name=f"{material}_{geom_type}",
                geom_type=geom_type,
                material=material,
                position=[0, 0.1, 0.05]
            )
            geom_config["class"] = RandomGeom
            button_config["subentities"] = [geom_config]

        
    def get_instruction(self, **kwargs):            
        if self.choose_magnetic == 0:
            instruction = ["We are conducting an experiment to identify whether the objects made of various kinds of materials are magnetic. Choose the object that is non-magnetic."]
        elif self.choose_magnetic == 1:
            instruction = ["We are conducting an experiment to identify whether the objects made of various kinds of materials are magnetic. Choose the object that is magnetic."]
        else:
            raise print("Random function is wrong!")

        self.config["task"]["instructions"] =instruction
        return self.config

@register.add_task("magnetism_qa")
class MagnetismQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)