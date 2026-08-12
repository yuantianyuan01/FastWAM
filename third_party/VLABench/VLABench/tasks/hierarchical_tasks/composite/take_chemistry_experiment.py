import random
import json
import os
import numpy as np
import copy
from functools import partial
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml
from VLABench.tasks.hierarchical_tasks.primitive.select_chemistry_tube_series import (relative_col_pos, relative_row_pos, SelectChemistryTubeConfigManager, SelectChemistryTubeTask)
from VLABench.tasks.condition import ConditionSet, AsynSequenceCondition
from VLABench.utils.utils import euler_to_quaternion
from VLABench.utils.skill_lib import SkillLib

with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task_related/experiment.json"), "r") as f:
    EXPERIMENTS = json.load(f)

SOLUTIONS=["CuCl2", "CuSO4", "FeCl3", "KMnO4", "I2", "K2CrO4", "NaCl", "AgNO3", "BaCl2", "H2SO4", "NaOH", "Ba(NO3)2", "Pb(NO3)2", "Na2CO3", "CaCl2", "HCl", "CaSO4"]

@register.add_config_manager("take_chemistry_experiment")
class TakeChemistryExperimentConfigManager(SelectChemistryTubeConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=[4, 5],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.seen_experiment = list(EXPERIMENTS.keys())[::2]
        self.unseen_experiment = list(EXPERIMENTS.keys())[1::2]

    def get_seen_task_config(self):
        self.target_experiment = random.choice(self.seen_experiment)
        target_entities = EXPERIMENTS[self.target_experiment]["solutions"]
        target_container = random.choice(self.seen_container)
        init_container = random.choice(self.seen_init_container)
        return self.get_task_config(target_entities, target_container, init_container)
    
    def get_unseen_task_config(self):
        self.target_experiment = random.choice(self.unseen_experiment)
        target_entities = EXPERIMENTS[self.target_experiment]["solutions"]
        target_container = random.choice(self.unseen_container)
        init_container = random.choice(self.seen_init_container)
        return self.get_task_config(target_entities, target_container, init_container)
    
    def load_init_containers(self, init_container):
        super().load_init_containers(init_container)
        self.config["task"]["components"][-1]["position"] = [random.uniform(-0.3, -0.2), random.uniform(0, 0.2), 0.78]
        self.config["task"]["components"][-1]["randomness"] = None
    
    def load_containers(self, target_container):
        super().load_containers(target_container)
        self.config["task"]["components"][-1]["position"] = [random.uniform(0.1, 0.3), random.uniform(0, 0.2), 0.78]
        
    def load_objects(self, target_entity):
        tube_stand_pos = self.config["task"]["components"][-1]["position"]
        solutions = []
        solutions.extend(target_entity)
        other_solutions = SOLUTIONS.copy()
        for solution in target_entity:
            other_solutions.remove(solution)
        
        solutions.extend(random.sample(other_solutions, self.num_object-len(target_entity)))
        target_rol_poses = random.sample(relative_col_pos, self.num_object)
        target_poses = [[pos, random.choice(relative_row_pos), 0.05] for pos in target_rol_poses] 
        init_container_config = self.config["task"]["components"][-1]
        init_container_config["subentities"] = []
        self.target_positions = []
        for object, pos in zip(solutions, target_poses):
            if object in target_entity:
                self.target_positions.append(np.array(pos) + np.array(tube_stand_pos))
            object_config = dict(
                name=object,
                solution=object,
                xml_path=name2class_xml["tube"][-1],
                position=pos,
            )
            object_config["class"] = name2class_xml["tube"][0]
            init_container_config["subentities"].append(object_config)
            
            tag_config = dict(
                name=f"{object}_tag",
                content=object,
                xml_path=name2class_xml["nametag"][-1],
                position=[pos[0], pos[1]+0.05, 0.1],
            )
            tag_config["class"] = name2class_xml["nametag"][0]
            init_container_config["subentities"].append(tag_config)
    
    def get_instruction(self, **kwargs):
        instruction = [EXPERIMENTS[self.target_experiment]["instruction"]]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        assert isinstance(target_entity, list), "solution in the experiment should be a list"
        conditions_config = dict(
            substep_satisfy=[]
        )
        for i, solution in enumerate(target_entity):
            conditions_config["substep_satisfy"].extend([
                dict(
                    pour=dict(
                        target_entity=solution,
                    )
                ),
                dict(
                    above=dict(
                        target_entity=solution,
                        platform=target_container,
                    )
                )
              ]
            ) 
            
        self.config["task"]["conditions"] = conditions_config
        return conditions_config

@register.add_task("take_chemistry_experiment")
class TakeChemistryExperimentTask(SelectChemistryTubeTask):
    def __init__(self, task_name, robot, random_init=False, **kwargs):
        super().__init__(task_name, robot=robot, random_init=random_init, **kwargs)

    def init_conditions(self):
        if self.config["task"].get("conditions") is not None:
            condition_config = copy.deepcopy(self.config["task"]["conditions"])
        else:
            self.conditions = None
            return False
        assert "substep_satisfy" in condition_config.keys(), "only support 'substep_satisfy' condition in clustering tasks"
        condition_sets = []
        for condition_items in condition_config["substep_satisfy"]:
            conditions = []
            for condition_key, specific_condition in condition_items.items():
                if "pour" in condition_key:
                    condition_key = "pour"
                elif "above_platform" in condition_key:
                    condition_key = "above_platform"
                condition_cls = register.load_condition(condition_key)
                for k, entities in specific_condition.items():
                    if isinstance(entities, str):
                        specific_condition[k] = self.entities.get(entities, None)
                    elif isinstance(entities, list):
                        specific_condition[k] = [self.entities.get(entity, None) for entity in entities]
                condition = condition_cls(**specific_condition)
                conditions.append(condition)
            condition_set = ConditionSet(conditions)
            condition_sets.append(condition_set)
        self.conditions = AsynSequenceCondition(condition_sets)
    
    def get_expert_skill_sequence(self, physics):
        place_point = np.array(self.entities[self.target_container].get_place_point(physics)[-1])
        target_positions = self.config_manager.target_positions
        skill_sequence = []
        for solution, end_pos in zip(self.target_entity, target_positions):
            grasppoint = np.array(self.entities[solution].get_grasped_keypoints(physics)[-1])
            skill_sequence.extend([
                partial(SkillLib.pick, target_entity_name=solution, target_pos=grasppoint+np.array([0, 0, 0.02]), target_quat=euler_to_quaternion(-np.pi, np.pi/4, -np.pi/2)), 
                partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.25),
                partial(SkillLib.moveto, target_pos=place_point, gripper_state=np.zeros(2)),
                partial(SkillLib.pour, target_delta_qpos=np.pi/2+np.pi/10, n_repeat_step=4),
                partial(SkillLib.wait, wait_time=10),
                partial(SkillLib.pull, pull_distance=0.1),
                partial(SkillLib.pour, target_delta_qpos=-(np.pi/2+np.pi/10), n_repeat_step=4, target_q_velocity=-np.pi/40),
                partial(SkillLib.moveto, target_pos=end_pos+np.array([0, 0, 0.25]), target_quat=euler_to_quaternion(-np.pi/2, -np.pi/2, 0), gripper_state=np.zeros(2)),
                partial(SkillLib.move_offset, offset=[0, 0, -0.1]),
                partial(SkillLib.open_gripper)
            ])
        return skill_sequence
            