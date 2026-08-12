import random
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive.physical_series.base import *

DENSITY_ORDER = ["wood", "rubber", "water", "glass", "stone", "metal"]

@register.add_config_manager("density_qa")
class DensityQAConfigManager(PhysicalQAConfigManager):
    def load_objects(self, materials):
        chosen_materials = random.sample(materials, self.num_object)
        density_order = self.get_density_order(chosen_materials)
        target_material = random.choice(chosen_materials)
        self.target_material_order_idx = list(density_order.keys()).index(target_material)
        geom_types = [random.choice(["box", "sphere", "cylinder", "capsule"]) for _ in range(self.num_object)]
        chosen_mateials = []
        for material in chosen_materials:
            if isinstance(material, list):
                material = random.choice(material)
            chosen_mateials.append(material)
        
        components_config = self.config["task"]["components"]
        button_configs = []
        for _, config in enumerate(components_config):
            if config["class"] == "Button" or config["class"] == Button:
                button_configs.append(config)
        for i, (button_config, material, geom_type) in enumerate(zip(button_configs, chosen_mateials, geom_types)):
            if i == self.target_material_order_idx:
                self.target_button = button_config["name"]
            geom_config = dict(
                name=f"{material}_{geom_type}",
                geom_type=geom_type,
                material=material,
                position=[0, 0.1, 0.05]
            )
            geom_config["class"] = RandomGeom
            button_config["subentities"] = [geom_config]
    
    def get_density_order(self, materials):
        density_order = dict()
        for material in materials:
            density_order[material] = DENSITY_ORDER.index(material)
        density_order = dict(sorted(density_order.items(), key=lambda item: item[1]))
        return density_order
    
    def get_instruction(self, **kwargs):
        if self.target_material_order_idx == 0: # the smallest density
            instruction = ["We are conducting a density experiment. We have three objects with different densities. Choose the object with the smallest density.",
                           "Choose the object that can float in the water."]
        elif self.target_material_order_idx == 1: # the medium density
            instruction = ["We are conducting a density experiment. We have three objects with different densities. Choose the object with the medium density."]
        elif self.target_material_order_idx == 2: # the largest density
            instruction = ["We are conducting a density experiment. We have three objects with different densities. Choose the object with the largest density."]
        self.config["task"]["instructions"] = instruction

@register.add_task("density_qa")
class DensityQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)