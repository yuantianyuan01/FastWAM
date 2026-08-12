import random
import numpy as np
import os
import json
from VLABench.configs import name2config
from VLABench.configs.constant import name2class_xml
from VLABench.utils.utils import flatten_list, grid_sample, find_key_by_value

DEFAULT_RABDOMNESS = dict(
    pos=[0.02, 0.02, 0],
    quat=[0, 0, 0.05],
)

class BenchTaskConfigManager():
    """
    Config manager class for task configuration load and management.
    The items of config should include: robot, task, engine.
    For each children classes, they differs from the task-specific assets, layouts, conditions and instructions.
    Basic assets layout include: 
        (un)seen objects - the objects that can be manipulated.
        (un)seen containers - the cintainer that is empty at the begining but can be used to hold objects.
        (un)seen init_containers - the container that is filled with objects at the begining. 
    """
    def __init__(self, 
                 task_name,
                 num_objects=[3, 4],
                 **kwargs):
        """
        param:
            config: base config from yaml file
        """
        # default config
        self.config = dict(
            task=dict(
                ngrid=[10, 10],
                n_distractor=1,
                workspace=[-0.3, 0.3, -0.2, 0.2, 0.8, 1.5]
            )
        )
        # load additional config from task_config.json
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task_config.json"), "r") as f:
            configs = json.load(f)
        config = configs.get("default", {})
        config.update(configs.get(find_key_by_value(name2config, task_name), None))
        if config is None: 
            raise ValueError(f"Task {task_name} is invalid. Check the valid ones task_config.json file.")
        self.config.update(config)
        if "components" not in self.config["task"]:
            self.config["task"]["components"] = []
        # init assets config
        self.num_objects = num_objects
        if "asset" not in self.config["task"]: self.config["task"]["asset"] = {}
        for attr in["seen_object", "unseen_object", "seen_container", "unseen_container", "seen_init_container", "unseen_init_container"]:
            if attr not in self.config["task"]["asset"]:
                value = kwargs[attr] if attr in kwargs else None
            else:
                value = self.config["task"]["asset"][attr]                     
            setattr(self, attr, value)
        
        self.kwargs = kwargs
        if isinstance(self.num_objects, list):
            self.num_object = random.choice(self.num_objects)
        elif isinstance(self.num_objects, int):
            self.num_object = self.num_objects
        
    def get_seen_task_config(self):
        target_entity = random.choice(self.seen_object)
        if isinstance(target_entity, list):
            target_entity = random.choice(target_entity)
        if self.seen_container is not None:
            container = random.choice(self.seen_container)
        else:
            container = None
        if self.seen_init_container is not None:
            init_container = random.choice(self.seen_init_container)
        else:
            init_container= None
        return self.get_task_config(target_entity=target_entity, 
                                    target_container=container, 
                                    init_container=init_container,
                                    **self.kwargs)
    
    def get_unseen_task_config(self):
        target_entity = random.choice(self.unseen_object)
        if isinstance(target_entity, list):
            target_entity = random.choice(target_entity)
        if self.unseen_container is not None:
            container = random.choice(self.unseen_container)
        else:
            container = None
        if self.unseen_init_container is not None:
            init_container = random.choice(self.unseen_init_container)
        else:
            init_container= None
        return self.get_task_config(target_entity=target_entity, 
                                    target_container=container, 
                                    init_container=init_container,
                                    **self.kwargs)
    
    def get_entity_config(self, target_entity:str, position=[0,0,0.8], orientation=[0, 0, 0], randomness=DEFAULT_RABDOMNESS, **kwargs):
        name = kwargs.get("specific_name", f"{target_entity}")
        xml_path = name2class_xml[target_entity][-1]
        if isinstance(xml_path, list):
            xml_path = random.choice(xml_path)
        entity_config = dict(
            name=name,
            xml_path=xml_path,
            position=position,
            orientation=orientation
        )
        entity_config.update(**kwargs)
        entity_config["class"] = kwargs.get("dclass", name2class_xml[target_entity][0])
        entity_config["randomness"] = randomness.copy() if randomness is not None else None
        return entity_config
    
    def get_instruction(self, **kwargs):
        """
        automatically generate instruction for the task
        """
        raise NotImplementedError
    
    def get_task_config(self, target_entity, target_container, init_container, **kwargs):
        """
        Load task related entity configs.
        param:
            target_entity: target entity to manipulate in most common tasks
            target_container: target container to place the target entity
            init_container: task the target entity from the init containers 
        """
        self.target_entity, self.target_container, self.init_container = target_entity, target_container, init_container
        self.load_containers(target_container=target_container)
        self.load_init_containers(init_container=init_container)
        self.load_objects(target_entity=target_entity)
        self.get_condition_config(target_entity=target_entity, target_container=target_container, init_container=init_container)
        self.get_instruction(target_entity=target_entity, target_container=target_container, init_container=init_container)
        return self.config
        
    def get_condition_config(self, target_entity, target_container, **kwargs):
        """
        Load the task-specific condition config, including condition names and their parameters
        param:
            config: base config from yaml file
        """
        raise NotImplementedError

    def load_containers(self, target_container):
        if target_container is not None:
            container_config = self.get_entity_config(target_container)
            self.config["task"]["components"].append(container_config)
    
    def load_init_containers(self, init_container):
        if init_container is not None:
            init_container_config = self.get_entity_config(init_container)
            self.config["task"]["components"].append(init_container_config)
    
    def load_objects(self, target_entity):
        objects = []
        objects.append(target_entity)
        self.other_objects = flatten_list(self.seen_object) + flatten_list(self.unseen_object)
        self.other_objects.remove(target_entity)
        objects.extend(random.sample(self.other_objects, self.num_object-1))

        for i, object in enumerate(objects):
            object_config = self.get_entity_config(object, position=[-0.1+i*0.1, 0.2, 0.8])
            self.config["task"]["components"].append(object_config)
    

class PressButtonConfigManager(BenchTaskConfigManager):     
    def get_condition_config(self, target_button, **kwargs):
        self.target_button = target_button
        conditions_config = dict(
            press_button=dict(
                target_button=target_button
            )
        )
        self.config["task"]["conditions"] = conditions_config

    def get_instruction(self, **kwargs):
        instruction = ["Press the button."]
        self.config["task"]["instructions"] =instruction
        return self.config
    
    def load_buttons(self, **kwargs):
        for i in range(self.num_object):
            button_config = self.get_entity_config("button", 
                                                   position=[-0.4+i*0.4+random.uniform(-0.05, 0.05), 
                                                             random.uniform(-0.2, 0), 
                                                             0.78],
                                                   specific_name=f"button{i}")
            self.config["task"]["components"].append(button_config)

class ClusterConfigManager(BenchTaskConfigManager):
    def get_seen_task_config(self):
        assert isinstance(self.seen_object, list) and isinstance(self.seen_object[-1], list), "similar objects in a list"
        target_entities = random.sample(self.seen_object, 2)
        containers = [random.choice(self.seen_container) for _ in range(2)]
        return self.get_task_config(target_entity=target_entities, target_container=containers, init_container=None, **self.kwargs)

    def get_unseen_task_config(self):
        assert isinstance(self.unseen_object, list) and isinstance(self.unseen_object[-1], list), "similar objects in a list"
        target_entities = random.sample(self.unseen_object, 2)
        containers = [random.choice(self.unseen_container) for _ in range(2)]
        return self.get_task_config(target_entity=target_entities, target_container=containers, init_container=None, **self.kwargs)
    
    def load_containers(self, target_container):
        assert isinstance(target_container, list), "containers should be more than 2 in clustering tasks"
        for i, container in enumerate(target_container):
            container_config = self.get_entity_config(container, 
                                                      position=[(i-0.5)*0.6, random.uniform(-0.1, 0.1), 0.8], 
                                                      specific_name=f"{container}_{i}")
            self.config["task"]["components"].append(container_config)
    
    def load_objects(self, target_entity, **kwargs):
        assert isinstance(target_entity, list) and isinstance(target_entity[0], list), "target entities should be a list"
        self.entities_to_load = dict(
            cls_1 = [], cls_2 = []
        )
        for i, entities in enumerate(target_entity):
            if len(entities) < self.num_object:
                for _ in range(self.num_object - len(entities)):
                    entities.append(entities[-1])
            entities_in_same_class = random.sample(entities, self.num_object)   
            self.entities_to_load[f"cls_{i+1}"] = entities_in_same_class
        entities_to_load = []
        for ls in self.entities_to_load.values():
            entities_to_load.extend(ls)
        random.shuffle(entities_to_load)
        positions = grid_sample(workspace=self.config["task"]["workspace"], 
                                grid_size=self.config["task"]["ngrid"],
                                n_samples=self.num_object+self.num_object)
        for i, (entity, pos) in enumerate(zip(entities_to_load, positions)):
            pos = [pos[0], pos[1], 0.8]
            object_config = self.get_entity_config(entity, 
                                                   position=pos)
            self.config["task"]["components"].append(object_config)
            
    def get_instruction(self, **kwargs):
        instruction = ["Cluster the objects into two classes."]
        self.config["task"]["instructions"] = instruction
        
    def get_condition_config(self, target_entity, target_container, **kwargs):
        assert isinstance(target_entity[-1], list) and isinstance(target_container, list), "target entities and containers should be in list"
        for cls, objects in self.entities_to_load.items():
            if objects[0] == objects[-1]:
                objects[-1] = objects[-1] + "_1"
        condition_config = dict()
        # or condition, either one of the two conditions is satisfied
        condition_config["or"] = [
        dict(
            contain_1=dict(
                entities=self.entities_to_load["cls_1"],
                container=f"{target_container[0]}_0"
            ),
            contain_2=dict(
                entities=self.entities_to_load["cls_2"],
                container=f"{target_container[1]}_1"
            )
        ),
        dict(
            contain_1=dict(
                entities=self.entities_to_load["cls_1"],
                container=f"{target_container[1]}_1"
            ),
            contain_2=dict(
                entities=self.entities_to_load["cls_2"],
                container=f"{target_container[0]}_0"
            )
        )]        
        
        self.config["task"]["conditions"] = condition_config