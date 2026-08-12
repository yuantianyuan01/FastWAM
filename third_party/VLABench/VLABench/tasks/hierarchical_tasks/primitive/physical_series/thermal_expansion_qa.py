import random
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive.physical_series.base import *

THERMAL_EXPANSION_ORDER = ["glass", "wood", "stone", "metal", "rubber"]

@register.add_config_manager("thermal_expansion_qa")
class ThermalExpansionQAConfigManager(PhysicalQAConfigManager):
    def load_objects(self, materials):
        chosen_materials = random.sample(materials, self.num_object)
        thermal_expansion_order = self.get_thermal_expansion_order(chosen_materials)
        target_material = random.choice(chosen_materials)
        self.target_material_order_idx = list(thermal_expansion_order.keys()).index(target_material)
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
    
    def get_thermal_expansion_order(self, materials):
        density_order = dict()
        for material in materials:
            density_order[material] = THERMAL_EXPANSION_ORDER.index(material)
        density_order = dict(sorted(density_order.items(), key=lambda item: item[1]))
        return density_order
    
    def get_instruction(self, **kwargs):
        choose_hot_cool = random.randint(0, 1)
        if self.target_material_order_idx == 0: # the smallest thermal_expansion
            if choose_hot_cool ==0:
                instruction = ["At room temperature, we conduct an experiment in which the same amount of energy is used to heat three objects, each made of three different materials, and ask who expands the least? Note that if you see stone, think of it as marble, and if you see glass, think of it as quartz glass."]
            else:
                instruction = ["At room temperature, we conduct an experiment in which the same amount of energy is used to cool three objects, each made of three different materials, and ask who shrinks the least? Note that if you see stone, think of it as marble, and if you see glass, think of it as quartz glass."]
        elif self.target_material_order_idx == 1: # the medium thermal_expansion
            if choose_hot_cool ==0:
                instruction = ["At room temperature, we conduct an experiment in which the same amount of energy is used to heat three objects, each made of three different materials, and ask who expands moderately? Note that if you see stone, think of it as marble, and if you see glass, think of it as quartz glass."]
            else:
                instruction = ["At room temperature, we conduct an experiment in which the same amount of energy is used to cool three objects, each made of three different materials, and ask who shrinks moderately? Note that if you see stone, think of it as marble, and if you see glass, think of it as quartz glass."]
        elif self.target_material_order_idx == 2: # the largest thermal_expansion
            if choose_hot_cool ==0:
                instruction = ["At room temperature, we conduct an experiment in which the same amount of energy is used to heat three objects, each made of three different materials, and ask who expands the most? Note that if you see stone, think of it as marble, and if you see glass, think of it as quartz glass."]
            else:
                instruction = ["At room temperature, we conduct an experiment in which the same amount of energy is used to cool three objects, each made of three different materials, and ask who shrinks the most? Note that if you see stone, think of it as marble, and if you see glass, think of it as quartz glass."]
        print(f"instruction:{instruction}, choose_hot_cool:{choose_hot_cool}, target:{self.target_material_order_idx}")
        self.config["task"]["instructions"] = instruction

@register.add_task("thermal_expansion_qa")
class ThermalExpansionQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)