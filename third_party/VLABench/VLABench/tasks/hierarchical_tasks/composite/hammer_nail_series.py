import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask
from VLABench.tasks.hierarchical_tasks.primitive.select_painting_series import HangPictureConfigManager, HangPictureTask
from VLABench.utils.utils import flatten_list, euler_to_quaternion
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("hammer_loose_nail")
class HammerLooseNailConfigManager(BenchTaskConfigManager):
    def load_objects(self, **kwargs):
        hammer_config = self.get_entity_config("hammer",
                                               position=[random.uniform(-0.3, 0.3),  
                                                              random.uniform(-0.2, 0.3), 
                                                              0.83])
        self.config["task"]["components"].append(hammer_config)
        loose_indice = random.choice(range(self.num_object))
        self.target_entity = f"nail_{loose_indice}"
        for i in range(self.num_object):
            if i==loose_indice: 
                position = [random.uniform(-0.3, 0.3), 0.3, random.uniform(1, 1.3)]
            else: 
                position = [random.uniform(-0.3, 0.3), random.uniform(0.37, 0.38), random.uniform(1, 1.3)]   
            nail_config = self.get_entity_config("nail", position=position, orientation=[np.pi/2, 0, 0], specific_name=f"nail_{i}")
            self.config["task"]["components"].append(nail_config)
    
    def get_seen_task_config(self):
        return self.get_task_config(target_entity=None,
                                    target_container=None,
                                    init_container=None)
    
    def get_unseen_task_config(self):
        return self.get_task_config(target_entity=None,
                                    target_container=None,
                                    init_container=None)
    
    def get_instruction(self, **kwargs):
        instruction = ["There is a loose nail on the wall, strenthen it."]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, **kwargs):
        condition_config=dict(
            joint_in_range=dict(
                entities=[self.target_entity],
                target_pos_range=[0.05, 0.08]
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_config_manager("assemble_hammer")
class AssembleHammerConfigManager(BenchTaskConfigManager):
    def load_objects(self, **kwargs):
        hammer_head_config = self.get_entity_config("hammer_head",
                                                    position=[random.uniform(-0.3, 0.3),  
                                                              random.uniform(-0.2, 0.1), 
                                                              0.83],
                                                    orientation=[np.pi/2, 0, 0])
        hammer_handle_config = self.get_entity_config("hammer_handle")
        
        nail_config = self.get_entity_config("nail",
                                            position=[random.uniform(-0.3, 0.3), 
                                                      0.3, 
                                                      random.uniform(1, 1.3)],
                                            orientation=[np.pi/2, 0, 0])
        
        self.config["task"]["components"].append(hammer_head_config)
        self.config["task"]["components"].append(hammer_handle_config)
        self.config["task"]["components"].append(nail_config)
    
    def get_seen_task_config(self):
        return self.get_task_config(target_entity=None,
                                    target_container=None,
                                    init_container=None)
    
    def get_unseen_task_config(self):
        return self.get_task_config(target_entity=None,
                                    target_container=None,
                                    init_container=None)
    
    def get_instruction(self, **kwargs):
        instruction = ["Assemble the hammer by attaching the hammer head to the hammer handle."]
        self.config["task"]["instructions"] = instruction
        
    def get_condition_config(self, **kwargs):
        # FIXME
        condition_config = dict(
            contain=dict(
                container="hammer_head",
                entities=["hammer_handle"]
            ),
            joint_in_range=dict(
                entities=["nail"],
                target_pos_range=[0.05, 0.08]
            )
        )
        self.config["task"]["conditions"] = condition_config
        
@register.add_config_manager("hammer_nail_and_hang_picture")
class HammerNailandHangPicture(HangPictureConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=[2],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def load_objects(self, target_entity):
        paintings = []
        paintings.append(target_entity)
        other_paintings = flatten_list(self.seen_object) + flatten_list(self.unseen_object)
        other_paintings.remove(target_entity)
        paintings.extend(random.sample(other_paintings, self.num_object-1))
        random.shuffle(paintings)
        for i, painting in enumerate(paintings):
            painting_config = dict(
                name=f"{painting.lower()}_painting",
                specific_painting=painting, 
                xml_path=name2class_xml["painting"][-1],
                position=[random.uniform(-0.02, 0.02) + (i-0.5) * 0.35, 
                          random.uniform(-0.3, -0.29), 0.8],
            )
            painting_config["class"] = name2class_xml["painting"][0]
            self.config["task"]["components"].append(painting_config)
        
        self.target_nail = "nail"
        nail_config =self.get_entity_config("nail",
                                            position=[random.uniform(-0.2, 0.2), 0.3, random.uniform(1.1, 1.3)],
                                            orientation=[np.pi/2, 0, 0])
        self.config["task"]["components"].append(nail_config)
        
        hammer_config = self.get_entity_config("hammer",
                                               position=[random.uniform(-0.1, 0.3), 0.23, 0.8],
                                               orientation=[0, 0, np.pi/2]) 
        self.config["task"]["components"].append(hammer_config)
    
    def get_condition_config(self, **kwargs):
        condition_config = dict(
            joint_in_range = dict(
                entities=["nail"],
                target_pos_range=[0.05, 0.08]
            )
        )
        self.config["task"]["conditions"] = condition_config
    
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Please hang the {target_entity} on the wall safely and steadily."]
        self.config["task"]["instructions"] = instruction

@register.add_task("hammer_loose_nail")
class HammerLooseNailTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        for key in list(self.entities.keys()):
            if "nail" in key:
                nail = self.entities[key]
                nail.detach()
                self._arena.attach(nail)

@register.add_task("assemble_hammer")
class AssembleHammerTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
    
    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        nail = self.entities["nail"]
        nail.detach() # delete the free joint of nail
        self._arena.attach(nail)
    
    def after_step(self, physics, random_state):
        pass
    
    def after_substep(self, physics, random_state):
        """
        judge if the hammer handle is inserted into hammer head
        """
        hammer_handle = self.entities["hammer_handle"]
        hammer_head = self.entities["hammer_head"]
        if hammer_head.parent == hammer_handle:
            return super().after_substep(physics, random_state)
        handle_top_sites_positions = hammer_handle.get_top_site_pos(physics)
        
        for pos in handle_top_sites_positions:    
            if hammer_head.contain(pos, physics):
                hammer_head.detach()
                hammer_handle.attach(hammer_head)
        return super().after_substep(physics, random_state)

@register.add_task("hammer_nail_and_hang_picture")
class HammerNailandHangPictureTask(HangPictureTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        nail_pos = np.array(self.entities["nail"].get_xpos(physics))
        target_entity = f"{self.target_entity.lower()}_painting"
        grasppoint= np.array(self.entities[target_entity].get_grasped_keypoints(physics)[-1])
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="hammer", prior_eulers=[[-np.pi, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=np.zeros(2)),
            partial(SkillLib.pull, pull_distance=0.1, gripper_state=np.zeros(2)),
            partial(SkillLib.moveto, target_pos=nail_pos+np.array([0.03, -0.2, -0.1]), target_quat=euler_to_quaternion(-np.pi/2, np.pi/2, np.pi/2), gripper_state=np.zeros(2)),
            partial(SkillLib.push, push_distance=0.15, gripper_state=np.zeros(2)),
            partial(SkillLib.place, target_container_name="counter", target_pos=np.array([0, 0.25, 0.8]), target_quat=euler_to_quaternion(-np.pi, 0, 0)),
            partial(SkillLib.lift, lift_height=0.1),
            partial(SkillLib.pick, target_entity_name=target_entity, target_pos=grasppoint+np.array([0, 0.01, 0.]), target_quat=euler_to_quaternion(-np.pi, 0, 0)),
            partial(SkillLib.lift, gripper_state=np.zeros(2)),
            partial(SkillLib.moveto, target_pos=nail_pos+np.array([0.01, -0.1, 0.02]), target_quat=euler_to_quaternion(-np.pi/2, 0, 0), gripper_state=np.zeros(2)),
            partial(SkillLib.push, push_distance=0.1, target_quat=euler_to_quaternion(-np.pi/2, 0, 0), gripper_state=np.zeros(2))
        ]
        return skill_sequence
    
    def update_task_progress(self, physics):
        pass
    
    def reset_task_progress(self):
        pass