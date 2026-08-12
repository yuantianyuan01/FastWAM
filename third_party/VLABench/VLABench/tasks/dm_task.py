import os
import numpy as np
import yaml
import json
import random
import itertools
import logging
import re
import copy
from functools import partial
from dm_control import composer
from VLABench.utils.register import register
from VLABench.tasks.condition import ConditionSet, OrCondition
from VLABench.utils.utils import grid_sample, distance
from VLABench.tasks.components.scene import Scene
from VLABench.configs.constant import name2class_xml
from VLABench.tasks.components.entity import Entity
from VLABench.utils.skill_lib import SkillLib
from VLABench.utils.gpt_utils import query_gpt4_v


with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/camera_config.json"), "r") as f:
    CAMERA_VIEWS = json.load(f)

NUM_SUBSTEPS = 100

class LM4ManipBaseTask(composer.Task):
    """
    Base class for task to carry out, derived from dm_control composer.Task.
    The key attribute to manage the task is 'config_manager' in build_from_config method. 
    For evaluation, 'episode_config' will be passed to generate deterministic configurations.
    """
    def __init__(self, 
                 task_name, 
                 robot,
                 eval=False,
                 random_init=True,
                 use_llm=False,
                 episode_config=None,
                 xml_file="base/default.xml",
                 run_mode="efficient",
                 **kwargs):
        """
        Params:
            robot: robot name to use in the task
            eval: whether load unseen objects to evaluate the generalization ability
            random_init: whether to compute grid sampling positions when initializing the entities
            use_llm: whether to use LLMs to generate instructions
            episode_config: deterministic configuration for task building
            xml_file: the file path of root xml for the task
            run_mode: the running mode of the environment, choose 'efficient' for data collection and 'eval' for evaluation
        """
        self.task_name = task_name
        self.config_manager = register.load_config_manager(task_name)(task_name)
        self.asset_path = os.path.join(os.getenv("VLABENCH_ROOT"), "assets") 
        self.use_llm = use_llm       
        self._arena = composer.Arena(xml_path=os.path.join(self.asset_path, xml_file))
        self._robot = robot
        self.attach_entity(robot)
        self._task_observables = {}
        config = self.config_manager.config
        self.control_timestep = self.physics_timestep * NUM_SUBSTEPS
        self.entities = dict()
        self.distractors = dict()
        self.random_init = random_init
        self.run_mode = run_mode
        if config is not None:
            self.random_ignored_entities = config["task"].get("random_ignored_entities", ["table"])
            self.ngrid = config["task"].get("ngrid", None)
            self.workspace = config["task"].get("workspace", [-0.3, 0.3, -0.2, 0.3, 0.75, 1.5]) # minx, maxx, miny, maxy, minz, maxz
        else:
            self.random_ignored_entities = ["table"]
            self.ngrid = None
            self.workspace = [-0.3, 0.3, -0.2, 0.3, 0.75, 1.5]
        
        self.build_from_config(eval, deterministic_config=episode_config)
        self.reset_camera_views()
    
    def reset_camera_views(self, index=2):
        if self.task_name in CAMERA_VIEWS:
            cameras = self._arena.mjcf_model.find_all("camera")
            target_camera = cameras[index]
            for attr, value in CAMERA_VIEWS[self.task_name].items():
                setattr(target_camera, attr, value)
            
    def step(self, action):
        pass
    
    @property
    def root_entity(self):
        return self._arena
    
    @property
    def task_observables(self):
        return self._task_observables
    
    @property
    def robot(self):
        return self._robot
    
    @property
    def name(self):
        return self.task_name
    
    @property
    def target_entity(self): # especially for primitive tasks
        return self.config_manager.target_entity
    
    @property
    def target_entities(self): 
        """
        expoecially for composite tasks
        """
        return self.config_manager.target_entities
    
    @property
    def target_container(self):
        return self.config_manager.target_container
    
    @property
    def init_container(self):
        return self.config_manager.init_container
    
    def initialize_episode(self, physics, random_state):
        if self.run_mode == "eval":
            self.reset_intention_distance()
            self.reset_task_progress()
        # grid sampling
        if self.ngrid is not None and self.random_init:
            entities_to_random = [key for key in self.entities.keys() if key not in self.random_ignored_entities]
            sampled_points = grid_sample(self.workspace, self.ngrid, len(entities_to_random), farthest_sample=True)
            for key, point in zip(entities_to_random, sampled_points):
                entity = self.entities[key]
                entity.init_pos[:2] = point
        return super().initialize_episode(physics, random_state)
    
    def get_reward(self, physics):
        return 0
    
    def before_step(self, physics, action, random_state):
        data = physics.data
        data.ctrl[:] = action
        pass
    
    def before_substep(self, physics, action, random_state):
        pass
    
    def after_step(self, physics, random_state):
        physics.data.ctrl[:] = 0
        if self.run_mode == "eval":
            self.update_intention_distance(physics)
            self.update_task_progress(physics)
    
    def after_substep(self, physics, random_state):
        pass
    
    def add_free_entity(self, entity):
        frame = self._arena.add_free_entity(entity)
        self.entities[entity.mjcf_model.model] = entity
    
    def delete_entity(self, entity):
        entity.detach()
        self.entities.pop(entity.mjcf_model.model)
    
    def build_from_config(self, eval=False, **kwargs):
        """
        Load configurations from the config file and build the task.
        Configuration includes：
            - options and parameters of mujoco physics engine
            - load scene by configuration
            - entity configuration
        """
        if eval: config = self.config_manager.get_unseen_task_config()
        else: config = self.config_manager.get_seen_task_config() 
        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, str):
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)
        # override the config with deterministic config
        deterministic_config = kwargs.get("deterministic_config", None)
        if deterministic_config is not None: 
            for key in ["scene", "components", "instructions", "conditions"]:
                if key in deterministic_config["task"].keys():
                    self.config["task"][key] = deterministic_config["task"][key]
            for key in ["target_entity", "target_container", "target_entities"]:
                if key in deterministic_config["task"].keys() and hasattr(self.config_manager, key):
                    setattr(self.config_manager, key, deterministic_config["task"][key])
        # load engine config
        self.set_engine_config(self.config["engine"])
        # load scene and entities
        if self.config["task"].get("scene", None) is not None:
            self.load_scene_from_config(config["task"]["scene"]) 
        for entity_config in self.config["task"]["components"]:
            self.load_entity_from_config(entity_config)
        # build instrutions
        self.build_instruction()
        # build conditions
        self.init_conditions()
        self.random_ignored_entities.extend(self.config["task"].get("random_ignored_entities", []))

    def init_conditions(self):
        if self.config["task"].get("conditions", None) is not None:
            condition_config = copy.deepcopy(self.config["task"]["conditions"])
        else: # no condition configs, return False. Task build default conditions
            self.conditions = None
            return False
        conditions = list()
        for condition_key, specific_condition in condition_config.items():
            condition_cls = register.load_condition(condition_key)
            for k, entities in specific_condition.items():
                if k in ["robot"]:
                    specific_condition[k] = self.robot
                    continue
                if k in ["positions", "target_pos_range"]: continue
                if isinstance(entities, str):
                    specific_condition[k] = self.entities.get(entities, None)
                elif isinstance(entities, list):
                    specific_condition[k] = [self.entities.get(entity, None) for entity in entities]
            condition = condition_cls(**specific_condition)
            conditions.append(condition)
        self.conditions = ConditionSet(conditions)
        return True
    
    def set_engine_config(self, config):
        """
        Recursively sets attributes from a nested dictionary to mujoco engine attributes.
        """

        def set_recursive_attr(obj, config):
            for key, value in config.items():
                if isinstance(value, dict):
                    if hasattr(obj, key):
                        nested_obj = getattr(obj, key)
                        set_recursive_attr(nested_obj, value)
                    else:
                        setattr(obj, key, type(obj)())
                        set_recursive_attr(getattr(obj, key), value)
                else:
                    setattr(obj, key, value)
                    
        set_recursive_attr(self._arena.mjcf_model, config)
         
    def load_scene_from_config(self, config):
        """
        Build the scene from the configuration dictionary
        """
        scene = Scene(**config)
        self.attach_entity(scene)
        self.scene = scene
        
    def load_entity_from_config(self, config: dict, parent_node=None):
        entity_cls = config.get("class", None)
        assert entity_cls is not None, "entity class must be provided"
        if isinstance(entity_cls, str):
            entity_cls = register.load_entity(entity_cls)
        elif isinstance(entity_cls, Entity):
            entity_cls = entity_cls
        if config.get("xml_path", None) is not None:
            config["xml_path"] = os.path.join(self.asset_path, config["xml_path"])
        if parent_node is not None:
            config["parent_entity"] = parent_node
            self.random_ignored_entities.append(config["name"])
        entity = entity_cls(**config)
        self.add_free_entity(entity)
        if entity.subentities is not None:
            for subentity_config in entity.subentities:
                self.load_entity_from_config(subentity_config, parent_node=entity)
        return entity
    
    def attach_entity(self, entity):
        self._arena.attach(entity)
    
    def _build_observables(self):
        self.robot.observables.joint_positions.enabled = True
        self.robot.observables.joint_velocities.enabled = True
        # for i in range(len(self.robot.observables.gripper_state)):
        #     self.robot.observables.gripper_state[i].enabled = True
        self._task_observables["robot"] = self.robot.observables
        for obs in self._task_observables.values():
            obs.enabled = True
    
    def get_element_by_name(self, name, type):
        return self._arena.mjcf_model.find(f"{type}", name)
    
    def add_disturbance(self, magnitude, random_state):
        """randomly add disturbance to the scene for robustness testing"""
        pass
    
    def get_instruction(self):
        if isinstance(self.instructions, str):
            return self.instructions
        elif isinstance(self.instructions, list):
            if self.instructions:
                return random.sample(self.instructions, 1)[-1]
            return None
    
    def should_terminate_episode(self, physics):
        if hasattr(self, "conditions"):
            terminal = self.conditions.is_met(physics)
        else:
            terminal = False
        return terminal

    def check_collision(self, physics):
        # body1_id = physics.model.body(body1_name+"/").id
        # body2_id = physics.model.body(body2_name+"/").id
        contact_body_pairs = set()
        for contact in physics.data.contact:
            if contact.dist == 0:
                continue
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            contact_body1 = physics.model.geom_bodyid[geom1_id]
            contact_body2 = physics.model.geom_bodyid[geom2_id]
            pair = sorted([contact_body1, contact_body2])
            contact_body_pairs.add(tuple(pair))   

        combinations = itertools.combinations(list(self.entities.keys()), 2)
        for combine in combinations:
            pair = tuple(sorted([physics.model.body(combine[0]+"/").id, 
                                physics.model.body(combine[1]+"/").id]))
            if pair in contact_body_pairs:
                print(f"{pair} in contact")
                # return True
            
    def intialization_valid(self, physics):
        collision = self.check_collision(physics)
        if collision:
            return False
        return True
    
    def reset_distractors(self, n_distractor=1):
        """
        generate n distractor entities randomly for task robustness and diversity
        """
        for name, distractor in self.distractors.items():
            self.delete_entity(distractor)
        self.distractors.clear()    
        for i in range(n_distractor):
            entity_name,  entity_cls_and_xml= random.choice(list(name2class_xml.items()))
            entity_cls, xml_path = entity_cls_and_xml[0], entity_cls_and_xml[1]
            # choose a new entity if the entity name already exists
            while entity_name in self.entities.keys():
                entity_name, entity_cls_and_xml = random.choice(list(name2class_xml.items()))
                entity_cls, xml_path = entity_cls_and_xml[0], entity_cls_and_xml[1]
            xml_path = os.path.join(self.asset_path, "obj/meshes", xml_path)
            distractor = entity_cls(name=f"distractor_{i}_{entity_name}", xml_path=xml_path, position=[0, 0, 0.82], orientation=[0, 0, 0])
            self.add_free_entity(distractor)
            self.distractors[distractor.mjcf_model.model] = distractor
    
    def reset_intention_distance(self):
        self.intention_distance = dict()
        entity_names = list(self.entities.keys())
        for ignore_entity in self.random_ignored_entities:
            if ignore_entity in entity_names:
                entity_names.remove(ignore_entity)
        for entity_name in entity_names: 
            self.intention_distance[entity_name] = np.inf
            
    def reset_task_progress(self):
        self.target_is_grasped = dict()
        if isinstance(self.target_entity, str):
            self.target_is_grasped[self.target_entity] = False
        elif isinstance(self.target_entity, list):
            for entity in self.target_entity:
                self.target_is_grasped[entity] = False
        
    def update_intention_distance(self, physics):
        ee_pos = self.robot.get_end_effector_pos(physics)
        for key, entity in self.entities.items():
            if key in self.random_ignored_entities: continue
            self.intention_distance[key] = min(self.intention_distance[key], distance(ee_pos, entity.get_xpos(physics)))
    
    def update_task_progress(self, physics):
        if isinstance(self.target_entity, list):
            for entity in self.target_entity:
                if self.entities[entity].is_grasped(physics, self.robot):
                    self.target_is_grasped[entity] = True
        elif isinstance(self.target_entity, str):
            if self.entities[self.target_entity].is_grasped(physics, self.robot):
                self.target_is_grasped[self.target_entity] = True
        
    def get_intention_score(self, physics, threshold=0.2, discrete=True):
        if isinstance(self.target_entity, list):
            return self.get_intention_score_to_entity(physics, self.target_entity[-1], threshold, discrete)
        return self.get_intention_score_to_entity(physics, self.target_entity, threshold, discrete)
    
    def get_task_progress(self, physics):
        # FIXME: temporary solution: in primitive tasks, a successful pick often occupies half of the task progress
        _, conditions_met = self.conditions.met_progress(physics)
        n_condition = len(self.conditions)
        n_condition += len(self.target_is_grasped)
        target_entity_met = []
        for value in self.target_is_grasped.values():
            if value: target_entity_met.append(value)
        return (len(conditions_met) + len(target_entity_met)) / n_condition
    
    def get_intention_score_to_entity(self, physics, entity_name, threshold=0.2, discrete=False):
        """
        Get the intention score of the entity during carry out, computed by the min distance to the entity.
        """
        if discrete:
            return int(self.intention_distance[entity_name] < threshold)
        else:
            if threshold - self.intention_distance[entity_name] < 0:
                return 0
            return 1 / (1 + (threshold - self.intention_distance[entity_name]) + 1e-6)
        
    def get_expert_skill_sequence(self, physics):
        """
        Expert trajectory generation for the task. Notice that the success rate is not 100%.
        """
        logging.info(f"Task:{self.task_name} did not implement get_expert_skill_sequence method")
        return None
    
    def build_instruction(self):
        self.instructions = ['']
        if self.config["task"].get("instructions", None) is not None:
            self.instructions = self.config["task"]["instructions"]
        if not self.use_llm:
            return 
        # generate instruction with GPT4
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/prompt/prompt.json"), "r") as f:
            prompts = json.load(f)
        if not self.task_name in prompts.keys():
            assert self.instructions is not None, "instruction must be provided"
            return 
        prompt = prompts[self.task_name]
        sys_prompt_0, sys_prompt_1, sys_prompt_2 = prompt["instruction_0"], prompt["instruction_1"], prompt["instruction_2"]
        target_entity = self.target_entity
        entity_names = list(self.entities.keys())
        query = f"{sys_prompt_0} {entity_names}. {sys_prompt_1} {target_entity}. {sys_prompt_2}"
        try:
            instructions = query_gpt4_v(query)
            instructions = re.findall(r'instruction:\s*"([^"]*)"', instructions)
            self.instructions = instructions
        except:
            self.instructions = []
    
    def save(self, physics):
        """
        Save the task information for deterministic evaluation
        """
        data_to_dump = dict(
            task=dict(
                components=[],
            )   
        )
        for entity in self.entities.values():
            data_to_dump["task"]["components"].append(entity.save(physics))
        data_to_dump["task"]["scene"] = self.scene.save(physics)
        data_to_dump["task"]["instructions"] = self.instructions
        data_to_dump["task"]["conditions"] = self.config["task"].get("conditions", None)
        data_to_dump["robot"] = self.robot.save(physics)
        for key in ["target_entity", "target_container", "target_entities"]:
            if hasattr(self.config_manager, key):
                data_to_dump["task"][key] = getattr(self.config_manager, key)
        return data_to_dump
        
    
class SpatialMixin:
    """
    Base class for task focusing on assessing spatial perception and understanding.
    This class provides methods to generate spatial relations between entities.
    """
    def generate_spatial_relation(self, physics):
        target_entity_name = self.target_entity
        target_entity_pos = self.entities[target_entity_name].get_xpos(physics)
    
    def build_from_config(self, config, eval=False):
        super().build_from_config(config, eval)
        self.generate_spatial_relation()

