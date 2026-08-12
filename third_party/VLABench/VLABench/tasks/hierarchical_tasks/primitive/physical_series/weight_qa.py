import random
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive.physical_series.base import *
from VLABench.tasks.hierarchical_tasks.primitive.physical_series.density_qa import DENSITY_ORDER

@register.add_config_manager("weight_qa")
class WeightQAConfigManager(PhysicalQAConfigManager):
    def __init__(self, 
                 task_name, 
                 num_objects=3, 
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.mode_experiment = random.randint(0, 1) # 0 same material different shape; 1 different shape, same material
        self.target_index_weight = random.randint(0, 2) # 0: the lightest  1: medium 2: the heaviest
        self.target_button_id = -1
    
    def load_objects(self, materials):
        components_config = self.config["task"]["components"]
        if self.mode_experiment == 0:
            # random geoms
            geom_types = ["box", "sphere", "cylinder"]
            self.order_geom_types = random.sample(geom_types, len(geom_types))
            # choose material
            self.target_material = random.choice(materials)
            # get button_id
            self.target_button_id = self.get_volume2target_button(self.target_index_weight)
            assert self.target_button_id in [0, 1, 2], "Error: target_button_id is wrong!"
            if isinstance(self.target_material, list):
                self.target_material = random.choice(self.target_material)
            # set button_config         
            components_config = self.config["task"]["components"]
            button_configs = []
            for _, config in enumerate(components_config):
                if config["class"] == "Button" or config["class"] == Button:
                        button_configs.append(config)
            # set geom_config
            self.geom_size_uni = random.uniform(0.02, 0.03)
            # print(f"mode_experiment:{self.mode_experiment}, target_index_weight:{self.target_index_weight}, order_geom_types:{self.order_geom_types}, target_material: {self.target_material}, self.target_button_id:{self.target_button_id}, target_geoms:{self.order_geom_types[self.target_button_id]}")
            for i, (button_config, geom_type) in enumerate(zip(button_configs, self.order_geom_types)):
                # print(f"{i}:geom:{geom_type}, material:{self.target_material}")
                if i == self.target_button_id:
                    self.target_button = button_config["name"]
                    self.target_entity = f"{self.target_material}_{geom_type}"
                    # print(f"is_target:{i}, {geom_type}")
                if geom_type == "box":
                    size = [self.geom_size_uni, self.geom_size_uni, self.geom_size_uni]
                elif geom_type == "sphere":
                    size = [self.geom_size_uni]
                elif geom_type == "cylinder":
                    size = [self.geom_size_uni, self.geom_size_uni]
                else:
                    raise ValueError("Random function is wrong!")
                geom_config = dict(
                    name=f"{self.target_material}_{geom_type}",
                    geom_type=geom_type,
                    material=self.target_material,
                    position=[0, 0.1, 0],
                    size=size
                )
                geom_config["class"] = RandomGeom
                # geom_config["class"] = RandomGeom
                button_config["subentities"] = [geom_config]
        
        elif self.mode_experiment == 1: # different material, same shape
            geom_types = ["box", "sphere", "cylinder", "capsule"]
            self.target_geom_type = random.choice(geom_types)
            self.geom_size_uni = random.uniform(0.02, 0.03)
            if self.target_geom_type == "box":
                size = [self.geom_size_uni, self.geom_size_uni, self.geom_size_uni]
            elif self.target_geom_type == "sphere":
                size = [self.geom_size_uni / 2]
            elif self.target_geom_type == "cylinder":
                size = [self.geom_size_uni / 2, self.geom_size_uni]
            elif self.target_geom_type == "capsule":
                size = [self.geom_size_uni / 2, self.geom_size_uni]
            else:
                raise ValueError("Random function is wrong!")
            # random materials
            order_materials = random.sample(materials, self.num_object)
            self.order_materials = []
            for material in order_materials:
                if isinstance(material, list):
                    material = random.choice(material)
                self.order_materials.append(material)
            # get button_id
            density_order = self.get_density_order(self.order_materials)
            self.target_material = list(density_order.keys())[self.target_index_weight]
            self.target_button_id = self.order_materials.index(self.target_material)

            components_config = self.config["task"]["components"]
            button_configs = []     
            for _, config in enumerate(components_config):
                if config["class"] == "Button" or config["class"] == Button:
                    button_configs.append(config)  

            for i, (button_config, material) in enumerate(zip(button_configs, self.order_materials)):
                if material == self.target_material:
                    self.target_button = button_config["name"]
                    self.target_entity = f"{material}_{self.target_geom_type}"
                geom_config = dict(
                    name=f"{material}_{self.target_geom_type}",
                    geom_type=self.target_geom_type,
                    material=material,
                    position=[0, 0.1, 0.05],
                    size=size
                )
                geom_config["class"] = RandomGeom
                button_config["subentities"] = [geom_config]
    
        else:
            raise ValueError("Random function is wrong!")

        
    def get_instruction(self, **kwargs):     
        if self.mode_experiment == 0:
            if self.target_index_weight == 0:
                instruction = ["We are conducting an experiment to identify the weight of the objects made of various kinds of materials. Choose the lightest object."]
            elif self.target_index_weight == 1:
                instruction = ["We are conducting an experiment to identify the weight of the objects made of various kinds of materials. Choose the object that is in the middle in weight."]
            elif self.target_index_weight == 2:
                instruction = ["We are conducting an experiment to identify the weight of the objects made of various kinds of materials. Choose the heaviest object."]
            else:
                raise ValueError("Random function is wrong!")
    
        elif self.mode_experiment == 1:
            if self.target_index_weight == 0:
                instruction = ["We are conducting an experiment to identify the weight of the objects with the same shape but made of different materials. Choose the object with the lightest material."]
            elif self.target_index_weight == 1:
                instruction = ["We are conducting an experiment to identify the weight of the objects with the same shape but made of different materials. Choose the object with the medium material."]
            elif self.target_index_weight == 2:
                instruction = ["We are conducting an experiment to identify the weight of the objects with the same shape but made of different materials. Choose the object with the heaviest material."]
            else:
                raise ValueError("Random function is wrong!")
        else:
            raise ValueError("Random function is wrong!")
               
        self.config["task"]["instructions"] =instruction
        print(f"instruction:{instruction}")
        return self.config
    
    def get_volume2target_button(self, target_index_weight):
        if target_index_weight == 0:
            return self.order_geom_types.index("sphere")
        elif target_index_weight == 1:
            return self.order_geom_types.index("cylinder")
        elif target_index_weight == 2:
            return self.order_geom_types.index("box")
        else:
            raise ValueError("Random function is wrong!")

    def get_density_order(self, materials):
        density_order = dict()
        for material in materials:
            density_order[material] = DENSITY_ORDER.index(material)
        density_order = dict(sorted(density_order.items(), key=lambda item: item[1]))
        return density_order

@register.add_task("weight_qa")
class WeightQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)