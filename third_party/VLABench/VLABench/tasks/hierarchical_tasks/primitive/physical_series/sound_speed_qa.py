import random
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive.physical_series.base import *

SPEED_OF_SOUND_ORDER = ["rubber", "water", "wood", "stone", "glass", "metal"]

@register.add_config_manager("speed_of_sound_qa")
class SpeedofSoundQAConfigManager(PhysicalQAConfigManager):
    def load_objects(self, materials):
        chosen_materials = random.sample(materials, self.num_object)
        speed_of_sound_order = self.get_speed_of_sound_order(chosen_materials)
        target_material = random.choice(chosen_materials)
        self.target_material_order_idx = list(speed_of_sound_order.keys()).index(target_material)
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
            if material == target_material:
                self.target_button = button_config["name"]
                self.target_entity = f"{material}_{geom_type}"
            geom_config = dict(
                name=f"{material}_{geom_type}",
                geom_type=geom_type,
                material=material,
                position=[0, 0.1, 0.05]
            )
            geom_config["class"] = RandomGeom
            button_config["subentities"] = [geom_config]
    
    def get_speed_of_sound_order(self, materials):
        density_order = dict()
        for material in materials:
            density_order[material] = SPEED_OF_SOUND_ORDER.index(material)
        density_order = dict(sorted(density_order.items(), key=lambda item: item[1]))
        return density_order
    
    def get_instruction(self, **kwargs):
        if self.target_material_order_idx == 0: # the smallest thermal_expansion
            instruction = ["At room temperature, we conduct an experiment for three objects, each made of three different materials, and ask which of the objects has sound traveling the slowest?"]
        elif self.target_material_order_idx == 1: # the medium thermal_expansion
            instruction = ["At room temperature, we conduct an experiment for three objects, each made of three different materials, and ask which of the objects has sound traveling moderately?"]
        elif self.target_material_order_idx == 2: # the largest thermal_expansion
            instruction = ["At room temperature, we conduct an experiment for three objects, each made of three different materials, and ask which of the objects has sound traveling the fastest?"]
        
        print(f"instruction:{instruction}, target:{self.target_material_order_idx}")
        self.config["task"]["instructions"] = instruction

@register.add_task("speed_of_sound_qa")
class SoundSpeedQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)