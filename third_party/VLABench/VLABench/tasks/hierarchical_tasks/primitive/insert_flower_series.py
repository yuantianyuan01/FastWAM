import random
import numpy as np
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.utils import euler_to_quaternion

@register.add_config_manager("insert_flower")
class InsertFlowerConfigManager(BenchTaskConfigManager):
    def __init__(self,
                 task_name,
                 num_objects=[3],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def load_containers(self, target_container):
        super().load_containers(target_container)
        self.config["task"]["components"][-1].update(
            position=[random.uniform(-0.3, 0.3), random.uniform(0.15, 0.25), 0.78]
        )
        
    def load_objects(self, target_entity):
        super().load_objects(target_entity)
        for i in range(self.num_object):
            self.config["task"]["components"][-i-1]["position"] = [-0.3 + i * 0.3 + random.uniform(-0.05, 0.05), 
                                                                   random.uniform(-0.15, -0.05), 0.9]
        
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = f"Insert the {target_entity} into the {target_container}."
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        condition_config = dict(
            contain=dict(
                container=target_container,
                entities=[target_entity]
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_config_manager("insert_flower_spatial")
class InsertFlowerSpatialConfigManager(InsertFlowerConfigManager):
    pass


@register.add_config_manager("insert_flower_common_sense")
class InsertFlowerCommonSenseConfigManager(InsertFlowerConfigManager):
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = f"Insert the flower with ... into vase."
        self.config["task"]["instructions"] = instruction
        
@register.add_config_manager("insert_bloom_flower")
class InsertBloomFlowerConfigManager(InsertFlowerConfigManager):
    def load_objects(self, target_entity, **kwargs):
        entities = [target_entity]
        entities.extend([f"wilted_flower" for i in range(self.num_object - 1)])
        random.shuffle(entities)
        for i, entity in enumerate(entities):
            wilted_flower_config = self.get_entity_config(entity,
                                                          position=[-0.3 + i * 0.3 + random.uniform(-0.05, 0.05), 
                                                                   random.uniform(-0.15, -0.05), 0.9])
            self.config["task"]["components"].append(wilted_flower_config)

@register.add_config_manager("insert_flower_semantic")
class InsertFlowerFlowerSemanticConfigManager(InsertFlowerConfigManager):
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = f"Insert the flower into the vase."
        self.config["task"]["instructions"] = instruction

@register.add_config_manager("replace_wilted_flower")
class ReplaceWiltedFlowerConfigManager(InsertFlowerConfigManager):
    def load_containers(self, target_container):
        super().load_containers(target_container)
        container_config = self.config["task"]["components"][-1]
        wilter_flower_config = self.get_entity_config("wilted_flower", position=[0., 0., 0.05], orientation=[np.pi/2, 0 ,0])
        container_config["subentities"] = [wilter_flower_config]
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        condition = dict(
            contain=dict(
                container=target_container,
                entities=[target_entity]
            ),
            not_contain=dict(
                container=target_container,
                entities=[f"wilted_flower"]
            )
        )
        self.config["task"]["conditions"] = condition

@register.add_task("insert_flower")
class InsertFlowerTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        
    def build_from_config(self, eval=False, **kwargs):
        """
        Attach the vase for stable interaction.
        """
        # FIXME delete the re-attachment of the vase for more realistic interaction
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if "vase" in key:
                entity.detach()
                self._arena.attach(entity)
    
    def get_expert_skill_sequence(self, physics):
        target_place_point = np.array(self.entities[self.target_container].get_place_point(physics)[-1]) + np.array([0, 0, 0.05])
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity),
            partial(SkillLib.lift, target_quat=euler_to_quaternion(-np.pi/2, np.pi/2, 0)),
            partial(SkillLib.moveto, target_pos=target_place_point, target_quat=euler_to_quaternion(-np.pi/2, np.pi/2, 0)),
            partial(SkillLib.lift, lift_height=-0.2)
        ]
        return skill_sequence
        

@register.add_task("insert_flower_common_sense")
class InsertFlowerCommonSenseTask(InsertFlowerTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("insert_flower_semantic")
class InsertFlowerSemanticTask(InsertFlowerTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("insert_flower_spatial")
class InsertFlowerSpatialTask(InsertFlowerTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("insert_bloom_flower")
class InsertBloomFlowerTask(InsertFlowerTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("replace_wilted_flower")
class ReplaceWiltedFlowerTask(InsertFlowerTask):
    """
    Drop the wilted flower in vase into dustbin and replace it with a blooming flower
    """
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

