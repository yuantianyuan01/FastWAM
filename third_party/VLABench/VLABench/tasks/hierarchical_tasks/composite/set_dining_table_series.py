import random
import numpy as np
from functools import partial
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask
from VLABench.utils.skill_lib import SkillLib 
from VLABench.utils.utils import euler_to_quaternion

@register.add_config_manager("set_dining_table")
class SetDiningTableConfigManager(BenchTaskConfigManager):
    def __init__(self,
                 task_name,
                 num_objects=6,
                 **kwargs
                 ):
        super().__init__(task_name, num_objects, **kwargs)
    
    def load_init_containers(self, init_container):
        for i, entity in enumerate(["knife", "fork", "chopstick", "spoon"]):
            init_container_config = self.get_entity_config(init_container, position=[-0.3 + 0.15*i, random.uniform(0.25, 0.3), 0.8], randomness={"scale":[0.6, 0.7]})
            init_container_config["subentities"] = [self.get_entity_config(entity, position=[0, 0, 0.15], orientation=[-np.pi/2, 0, 0], randomness=None)]
            self.config["task"]["components"].append(init_container_config)
            
    def load_containers(self, target_container):
        super().load_containers(target_container)
        self.config["task"]["components"][-1]["position"] = [random.uniform(-0.1, 0.1), random.uniform(-0.1, -0), 0.8]
        self.config["task"]["components"][-1]["subentities"] = [self.get_entity_config("plate", position=[0, 0, 0.04], randomness=None)]
    
    def load_objects(self, **kwargs):
        pass
    
    def get_condition_config(self, **kwargs):
        """
        normal case is the knife on the left of the plate while the fork on the right of the plate
        """
        self.order_entities = ["knife", "plate", "fork"]
        condition_config = dict(
            order=dict(
                entities=self.order_entities,
                axis=[0],
                offset=0.1
            )
        )
        self.config["task"]["conditions"] = condition_config
        
    def get_instruction(self, **kwargs):
        instructions = ["set the dining table"]
        self.config["task"]["instructions"] = instructions
        
@register.add_config_manager("set_dining_left_hand")
class SetDiningTableLeftHandConfigManager(SetDiningTableConfigManager):
    def get_condition_config(self, **kwargs):
        """
        the knife on the right of the plate while the fork on the left of the plate
        """
        self.order_entities=["fork", "plate", "knife"]
        condition_config = dict(
            order=dict(
                entities=self.order_entities,
                axis=[0],
                offset=0.1
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_config_manager("set_dining_chopstick")
class SetDiningTableChopstickConfigManager(SetDiningTableConfigManager):
    def get_condition_config(self, **kwargs):
        """
        the chopstick on the right of the plate while the fork on the left of the plate
        """
        self.order_entities = ["plate", "chopstick"]
        condition_config = dict(
            order=dict(
                entities=self.order_entities,
                axis=[0],
                offset=0.1
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_config_manager("set_dining_chopstick_left_hand")
class SetDiningTableChopstickLeftHandConfigManager(SetDiningTableConfigManager):
    def get_condition_config(self, **kwargs):
        """
        the chopstick on the left of the plate while the fork on the right of the plate
        """
        self.order_entities = ["chopstick", "plate"]
        condition_config = dict(
            order=dict(
                entities=self.order_entities,
                axis=[0],
                offset=0.1
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_task("set_dining_table")
class DiningSetTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def reset_camera_views(self, index=1):
        return super().reset_camera_views(index)
    
    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if "placemat" in key:
                entity.detach()
                self._arena.attach(entity)
    
    def get_expert_skill_sequence(self, physics):
        order_entities = self.config_manager.order_entities
        plate_pos = np.array(self.entities["plate"].get_xpos(physics))
        # position to place the object
        right_hand_pos = plate_pos + np.array([0.15, -0.05, 0.03])
        left_hand_pos = plate_pos + np.array([-0.15, -0.05, 0.03])
        if len(order_entities) == 3:
            order_entities.remove("plate")
        assert len(order_entities) == 2
        skill_sequence = []
        for i, (entity, pos) in enumerate(zip(order_entities, [left_hand_pos, right_hand_pos])):
            if "plate" in entity:
                continue
            entity_grasppoint = np.array(self.entities[entity].get_grasped_keypoints(physics)[-1])
            skill_sequence.extend([
                partial(SkillLib.pick, target_entity_name=entity, target_pos=entity_grasppoint+np.array([0, 0.02, 0.02]), target_quat=euler_to_quaternion(-np.pi/2, -np.pi/2, 0)),
                partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.2),
                partial(SkillLib.place, target_container_name="plate", target_pos=pos, target_quat=euler_to_quaternion(-np.pi, 0, -np.pi/2)),
                partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.2),
            ])
        return skill_sequence
            
@register.add_task("set_dining_left_hand")
class DiningSetLeftHandTask(DiningSetTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("set_dining_chopstick")
class DiningSetChopsticksTask(DiningSetTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        
@register.add_task("set_dining_chopstick_left_hand")
class DiningSetChopsticksLeftHandTask(DiningSetTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)    