import random
import numpy as np
from functools import partial
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.hierarchical_tasks.primitive.select_chemistry_tube_series import SelectChemistryTubeTask, relative_col_pos, relative_row_pos
from VLABench.configs.constant import name2class_xml
from VLABench.utils.skill_lib import SkillLib 

@register.add_config_manager("rearrange_tube")
class RearrangeTubeConfigManager(BenchTaskConfigManager):
    def __init__(self,
                 task_name,
                 num_objects=[2, 3],
                 **kwargs
                 ):
        super().__init__(task_name, num_objects, **kwargs)
        
    def get_task_config(self, target_container, init_container, **kwargs):
        self.target_entity, self.target_container, self.init_container = None, None, None
        objects = random.sample(self.seen_object + self.unseen_object, self.num_object)
        target_entities = []
        for similar_objects in objects:
            target_entities.append(random.choice(similar_objects))
        self.load_containers(target_container, target_entities)
        self.load_init_containers(init_container, target_entities)
        self.get_instruction()
        self.get_condition_config(target_container)
        return self.config
        
    def load_containers(self, target_container, entities, **kwargs):
        super().load_containers(target_container)
        self.config["task"]["components"][-1]["position"] = [random.uniform(-0.1, 0.1),
                                                             random.uniform(-0.1, 0.),
                                                             0.8
                                                             ]
        self.config["task"]["components"][-1]["subentities"] = []
        self.config["task"]["components"][-1]["randomness"] = None
        container_pos = self.config["task"]["components"][-1]["position"]
        targer_rol_poses = random.sample(relative_col_pos, self.num_object)
        target_nametag_poses = [[pos, random.choice(relative_row_pos)-0.05, 0.1] for pos in targer_rol_poses]
        self.target_tube_position = dict()
        for entity, pos in zip(entities, target_nametag_poses):            
            tag_config = dict(
                name=f"{entity}_tag",
                content=entity,
                xml_path=name2class_xml["nametag"][-1],
                position=pos,
            )
            tag_config["class"] = name2class_xml["nametag"][0]
            self.target_tube_position[entity] = [container_pos[0] + pos[0], 
                                                 container_pos[1] + pos[1] +0.05, 0.8]
            self.config["task"]["components"][-1]["subentities"].append(tag_config)
            
    def load_init_containers(self, init_container, entities, **kwargs):
        super().load_init_containers(init_container)
        self.config["task"]["components"][-1]["position"] = [random.uniform(-0.2, 0.2),
                                                             random.uniform(0.2, 0.3),
                                                             0.8
                                                             ]
        self.config["task"]["components"][-1]["subentities"] = []
        self.config["task"]["components"][-1]["randomness"] = None
        targer_rol_poses = random.sample(relative_col_pos, self.num_object)
        target_solution_poses = [[pos, random.choice(relative_row_pos), 0] for pos in targer_rol_poses]
        for entity, pos in zip(entities, target_solution_poses):            
            tube_config = dict(
                name=entity,
                solution=entity,
                xml_path=name2class_xml["tube"][-1],
                position=pos,
                randomness=None
            )
            tube_config["class"] = name2class_xml["tube"][0]
            self.config["task"]["components"][-1]["subentities"].append(tube_config)
            
    def get_instruction(self, **kwargs):
        instruction = [f"Please rearrange the tubes by the nametag"]    
        self.config["task"]["instructions"] = instruction
        
    def get_condition_config(self, target_container, **kwargs):
        entities, entities_pos = [], []
        for entity, pos in self.target_tube_position.items():
            entities.append(entity)
            entities_pos.append(pos)
        condition_config = dict(
            contain=dict(
                entities=entities,
                container=target_container
            ),
            on_position=dict(
                entities=entities,
                positions=entities_pos,
                tolerance_distance=0.05
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_task("rearrange_tube")
class RearrangeTubeTask(SelectChemistryTubeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        target_positions = self.config_manager.target_tube_position
        skill_sequence = []
        for key, pos in target_positions.items():
            skill_sequence.extend([
                partial(SkillLib.pick, target_entity_name=key, prior_eulers=[[-np.pi, 0, 0]]),
                partial(SkillLib.lift, gripper_state=np.ones(2)*0.005, lift_height=0.2),
                partial(SkillLib.moveto, target_pos=np.array(pos)+np.array([0, 0, 0.2]),  gripper_state=np.ones(2)*0.005),
                partial(SkillLib.place, target_container_name="chemistry_tube_stand", target_pos=np.array(pos)+np.array([0, 0, 0.05])),
                partial(SkillLib.lift, gripper_state=np.ones(2)*0.04, lift_height=0.2),
            ])
        return skill_sequence