import random
from VLABench.tasks.components import RandomGeom
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.dm_task import LM4ManipBaseTask
from VLABench.utils.register import register

@register.add_config_manager("simple_seesaw_use")
class UseSeeSawSimpleConfigManager(BenchTaskConfigManager):
    def __init__(self,
                 task_name,
                 num_objects=[2, 3],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def get_task_config(self, target_entity, target_container, init_container, **kwargs):
        self.target_entity, self.target_container, self.init_container = target_entity, "seesaw", init_container
        target_subentity = self.load_objects(target_entity)
        self.load_init_containers(target_subentity)
        self.get_instruction(target_entity)
        self.get_condition_config(target_entity)
        return self.config
    
    def load_init_containers(self, subentity, **kwargs):
        seesaw_machine = self.get_entity_config("seesaw", 
                                        position=[random.uniform(-0.2, -0.1), 0.2, 0.8])
        
        seesaw_machine["subentities"] = [subentity]
        self.config["task"]["components"].append(seesaw_machine)
    
    def load_objects(self, target_entity):
        target_entity = self.get_entity_config(target_entity,
                                               position=[0.0, 0, 0.15],
                                               randomness=None)
        # load boxes as weights
        for i in range(self.num_object):
            weight_config = dict(
                name=f"weight_{i}",
                gemo_type="box",
                position=[(i-2)*random.uniform(0.1, 0.15), random.uniform(-0.1, 0.1), 0.8],
            )
            weight_config["class"] = RandomGeom
            self.config["task"]["components"].append(weight_config)
        return target_entity
    
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Get the {target_entity} out of container"]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, **kwargs):
        condition_config = dict(
            is_grasped=dict(
                entities=[f"{target_entity}"],
                robot="franka"
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_task("simple_seesaw_use")
class SimpleSeesawUseTask(LM4ManipBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)
