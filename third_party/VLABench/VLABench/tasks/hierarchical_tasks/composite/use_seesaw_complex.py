import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive import SimpleSeesawUseTask, UseSeeSawSimpleConfigManager
from VLABench.tasks.components import RandomGeom

@register.add_config_manager("complex_seesaw_use")
class UseSeeSawComplexConfigManager(UseSeeSawSimpleConfigManager):
    def __init__(self,
                 task_name,
                 num_objects=[2, 3],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

@register.add_task("complex_seesaw_use")
class UseSeesawComplexTask(SimpleSeesawUseTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
    
    def initialize_episode(self, physics, random_state):
        """
        Apply a weight to the target entity
        """
        res = super().initialize_episode(physics, random_state)
        weight_entities = [self.entities[name] for name in self.entities if "weight" in name]
        weight_masses = []
        for weight_entity in weight_entities:
            weight_masses.append(physics.bind(weight_entity.mjcf_model.worldbody).mass)
        min_weight = np.min(weight_masses)
        sum_weight = np.sum(weight_masses)
        new_mass = random.uniform(min_weight, sum_weight)
        
        target_entity_instance = self.entities[self.target_entity]
        bodies_of_target_entities = target_entity_instance.mjcf_model.find_all("body")
        n_bodies = len(bodies_of_target_entities)
        for body in bodies_of_target_entities:
            physics.bind(body).mass = new_mass / n_bodies
        return res