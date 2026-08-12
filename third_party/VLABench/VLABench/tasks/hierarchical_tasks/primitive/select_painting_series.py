import random
import os
import json
import numpy as np
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PressButtonTask, PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager, PressButtonConfigManager
from VLABench.utils.register import register
from VLABench.utils.utils import flatten_list, euler_to_quaternion
from VLABench.tasks.components import Button, RandomGeom
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("select_painting")
class SelectPaintingConfigManager(PressButtonConfigManager):
    seen_style = ['Romanticism', 'Realism', 'Rococo', 'Baroque', 'Symbolism', 'Minimalism', 'Post-Impressionism', 'Surrealism', 'Ukiyo-e']
    unseen_style = ['Academicism', 'Expressionism', 'Impressionism', 'Neoclassicism', 'Art Nouveau', 'Cubism']
    def __init__(self, 
                 task_name,
                 num_objects = [3],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def get_seen_task_config(self):  # to choose target style and all loaded styles
        target_style = random.choice(self.seen_style)
        if isinstance(target_style, list):  # multitple forms in one style
            target_style = random.choice(target_style)
        return self.get_task_config(target_style)
    
    def get_unseen_task_config(self):  # to choose target style and all loaded styles
        target_style = random.choice(self.unseen_style)
        if isinstance(target_style, list):  # multitple forms in one style
            target_style = random.choice(target_style)
        return self.get_task_config(target_style)
    
    def get_instruction(self, target_style, **kwargs):
        instruction = [f"Please select the painting of style {target_style.lower()}."]
        self.config["task"]["instructions"] = instruction
        return self.config
    
    def get_task_config(self, target_style):
        self.load_buttons()
        self.target_entity, self.target_container, self.init_container = target_style, None, None
        other_styles = flatten_list(self.seen_style) + flatten_list(self.unseen_style)
        other_styles.remove(target_style)
        loaded_styles = random.sample(other_styles, self.num_object-1) + [target_style]
        random.shuffle(loaded_styles)
        buttons_config = [config for config in self.config["task"]["components"] if config["class"] == Button]
        for style, button_config in zip(loaded_styles, buttons_config):
            style_config = self.get_style_config(style)
            button_config["subentities"] = [style_config]
            if style == target_style:
                self.get_condition_config(button_config["name"])
        self.get_instruction(target_style)
        return self.config
        
    def get_style_config(self, style):
        style_painting_config = dict(
            name=f"{style.lower()}_painting",
            style=style,
            xml_path=name2class_xml["painting"][-1],
            position=[random.uniform(-0.05, 0.05), random.uniform(0.05, 0.1), 0.0],
        )
        style_painting_config["class"] = name2class_xml["painting"][0]
        return style_painting_config

@register.add_config_manager("put_box_on_painting")
class PutBoxOnPaintingConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name, 
                 num_objects=3, 
                 **kwargs):
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task_related/painting.json"), "r") as f:
            paintings = json.load(f)
        super().__init__(task_name, num_objects, seen_object=paintings[::2], unseen_object=paintings[1::2], **kwargs)

    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Please put the box on the {target_entity}."]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, **kwargs):
        condition_config = dict(
            contain=dict(
                entities=["target_box"],
                container=target_entity,
            )
        )
        self.config["task"]["conditions"] = condition_config
    
    def get_task_config(self, target_entity, **kwargs):
        self.target_container, self.init_container = target_entity, None
        self.load_objects(target_entity)
        self.get_condition_config(target_entity)
        self.get_instruction(target_entity)
        return self.config
    
    def load_objects(self, target_entity, **kwargs):
        paintings = []
        paintings.append(target_entity)
        other_paintings = flatten_list(self.seen_object) + flatten_list(self.unseen_object)
        other_paintings.remove(target_entity)
        paintings.extend(random.sample(other_paintings, self.num_object-1))
        random.shuffle(paintings)
        for i, painting in enumerate(paintings):
            painting_config = dict(
                name=painting,
                specific_painting=painting,
                xml_path=name2class_xml["painting"][-1],
                position=[random.uniform(-0.05, 0.05) + (i-1) * 0.35, 
                          random.uniform(0., 0.2), 0.8],
            )
            painting_config["class"] = name2class_xml["painting"][0]
            self.config["task"]["components"].append(painting_config)
        
        # load target box
        box_config = dict(
            name="target_box",
            geom_type="box",
            position=[random.uniform(-0.3, 0.3), random.uniform(-0.1, 0), 0.8]
        )
        box_config["class"] = RandomGeom
        self.config["task"]["components"].append(box_config)
        self.target_entity = "target_box"

@register.add_config_manager("hang_picture_on_specific_nail")
class HangPictureConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects = [2],
                 **kwargs):
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task_related/painting.json"), "r") as f:
            paintings = json.load(f)
        super().__init__(task_name, num_objects, seen_object=paintings[::2], unseen_object=paintings[1::2], **kwargs)
    
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
                position=[random.uniform(-0.05, 0.05) + (i-0.5) * 0.35, 
                          random.uniform(-0.3, -0.29), 0.8],
            )
            painting_config["class"] = name2class_xml["painting"][0]
            self.config["task"]["components"].append(painting_config)
        
        candidate_z = [1.2 + random.uniform(0, 0.02), 
                       1.3 + random.uniform(-0.02, 0.02), 
                       1.4 + random.uniform(-0.02, 0.02)]
        random.shuffle(candidate_z)
        ordered_z = sorted(candidate_z.copy())
        candidate_positions = [[-0.2+random.uniform(-0.05, 0.05) + 0.2*i, 
                                0.36, z] for i, z in enumerate(candidate_z)]
        target_index = random.randint(0, 2)
        
        self.x_order = target_index
        self.z_order = ordered_z.index(candidate_z[-1])
            
        self.target_nail = f"nail_{target_index}"
        for i, pos in enumerate(candidate_positions):
            nail_config =self.get_entity_config("nail",
                                                specific_name=f"nail_{i}",
                                                position=pos,
                                                orientation=[np.pi/2, 0, 0])
            self.config["task"]["components"].append(nail_config)
            
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Please hang the {target_entity} on the nail in the x-aixs {self.x_order}/z-aixs {self.z_order}."]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, **kwargs):
        pass

@register.add_task("select_painting")
class SelectPaintingTask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
    
    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if key not in ["table", "button0", "button1", "button2", "target_box"]:
                entity.detach()
                self._arena.attach(entity)
    
    def reset_camera_views(self, index=1): # default index=1
        return super().reset_camera_views(index)
    
@register.add_task("put_box_on_painting")
class PutBoxOnPaintingTask(SelectPaintingTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="target_box"),
            partial(SkillLib.place, target_container_name=self.target_container), 
        ]
        return skill_sequence

@register.add_task("select_painting_by_style")
class SelectPaintingByStyleTask(SelectPaintingTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("put_box_on_painting_semantic")
class PutBoxOnPaintingSemanticTask(PutBoxOnPaintingTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)    

@register.add_task("hang_picture_on_specific_nail")
class HangPictureTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        self.should_terminate = False
        
    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        for key in list(self.entities.keys()):
            if "nail" in key:
                nail = self.entities[key]
                nail.detach()
                self._arena.attach(nail)
    
    def initialize_episode(self, physics, random_state):
        self.should_terminate = False
        return super().initialize_episode(physics, random_state)
    
    def after_step(self, physics, random_state):
        target_entity = self.entities[f"{self.config_manager.target_entity.lower()}_painting"]
        paint_kp = target_entity.get_grasped_keypoints(physics)[-1]
        target_nail = self.entities[self.config_manager.target_nail]
        nail_kp = target_nail.get_grasped_keypoints(physics)[-1]
        distance = np.linalg.norm(paint_kp - nail_kp)
        if distance < 0.05:
            self.should_terminate = True
        return super().after_step(physics, random_state)
    
    def should_terminate_episode(self, physics):
        if hasattr(self, "conditions") and self.conditions is not None:
            condition_met = self.conditions.is_met(physics)
        else:
            condition_met = True
        if self.should_terminate and condition_met:
            terminal = True
        else:
            terminal = False
        return terminal
    
    def get_expert_skill_sequence(self, physics):
        target_entity = f"{self.target_entity.lower()}_painting"
        target_nail = self.config_manager.target_nail
        nail_pos = np.array(self.entities[target_nail].get_xpos(physics))
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=target_entity, prior_eulers=[[-np.pi, 0, 0]]),
            partial(SkillLib.lift, gripper_state=np.zeros(2)),
            partial(SkillLib.moveto, nail_pos+np.array([0.01, -0.1, 0.02]), target_quat=euler_to_quaternion(-np.pi/2, 0, 0), gripper_state=np.zeros(2)),
            partial(SkillLib.moveto, nail_pos+np.array([0.01, -0.1, 0.02]), target_quat=euler_to_quaternion(-np.pi/2, 0, 0), gripper_state=np.zeros(2))
        ]
        return skill_sequence