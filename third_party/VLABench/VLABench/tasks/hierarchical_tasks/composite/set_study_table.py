import random
import numpy as np
from functools import partial
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive.select_book_series import SelectSpecificTypeBookConfigManager, SelectBookTask
from VLABench.utils.skill_lib import SkillLib
from VLABench.utils.utils import euler_to_quaternion

@register.add_config_manager("set_study_table")
class SelectBookConfigManager(SelectSpecificTypeBookConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=[3, 4],
                 **kwargs):
        """select the book with specific name"""
        super().__init__(task_name, num_objects, **kwargs)
        
    def load_objects(self, target_entity):
        super().load_objects(target_entity)
        init_container_config = self.config["task"]["components"][-1]
        for i, subentity_config in enumerate(init_container_config["subentities"]):
            subentity_config["position"][1] -= 0.1
        laptop_config = self.get_entity_config("laptop", position=[random.uniform(-0.2, -0.1), random.uniform(-0.1, 0.0), 0.8])
        self.config["task"]["components"].append(laptop_config) 
    
    def load_init_containers(self, init_container):
        if init_container is not None:
            self.config["task"]["components"].append(self.get_entity_config(init_container, 
                                                                            position=[random.uniform(-0.1, 0.1), 
                                                                                      random.uniform(0.35, 0.4), 0.8]))
    
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Today I want to review the content about ... Please help me to manage the study table before I come."]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, init_container, **kwargs):
        condition_config = dict(
            not_contain=dict(
                container=init_container,
                entities=[target_entity]
            ),
            joint_in_range=dict(
                target_pos_range=[1.2, 1.5],
                entities=["laptop"]
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_task("set_study_table")
class SetStudyTableTask(SelectBookTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        laptop_pos = np.array(self.entities["laptop"].get_xpos(physics))
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-np.pi/2, -np.pi/2, 0]]),
            partial(SkillLib.pull, gripper_state=np.zeros(2), pull_distance=0.15),
            partial(SkillLib.place, target_container_name="table", target_pos=laptop_pos + np.array([0.4, 0, 0.1]), target_quat=euler_to_quaternion(-np.pi*3/4,  0, np.pi/2)),
            partial(SkillLib.lift, gripper_state=np.ones(2)*0.04, lift_height=0.1),
            partial(SkillLib.moveto, target_pos=laptop_pos + np.array([0, 0, 0.2]), target_quat=euler_to_quaternion(-np.pi*0.7, 0, 0), gripper_state=np.ones(2)*0.04),
            partial(SkillLib.pick, target_entity_name="laptop", prior_eulers=[[-np.pi*0.7, 0, 0]]),
            partial(SkillLib.open_laptop, target_entity_name="laptop"),
        ]
        return skill_sequence