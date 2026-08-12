import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.configs.constant import name2class_xml

VALUES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
MAHJONGS = [(value, type) for value in VALUES for type in ["man", "sou", "pin"]] 


@register.add_config_manager("select_mahjong")
class SelectMahjongConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects = [6, 7],
                 **kwargs):
        super().__init__(task_name, num_objects, seen_object=MAHJONGS[::2], unseen_object=MAHJONGS[::2], **kwargs)
    
    def load_containers(self, target_container):
        container_config = self.get_entity_config(target_container, position=[random.uniform(-0.2, 0.2), random.uniform(0.2, 0.3), 0.8])
        self.config["task"]["components"].append(container_config)
        
    def load_objects(self, target_entity):
        objects = [target_entity]
        other_objects = self.seen_object.copy()
        other_objects.remove(target_entity)
        objects.extend(random.sample(other_objects, self.num_object - 1))
        start_point = random.uniform(-0.3, -0.1)
        random.shuffle(objects)
        y_pos = random.uniform(-0.1, 0)
        for i, object in enumerate(objects):
            value, type = object[0], object[1]
            mahjong_config = dict(
                name=f"{value}_{type}",
                xml_path=name2class_xml["mahjong"][-1],
                value=value,
                suite=type,
                position=[start_point + i * 0.07, y_pos, 0.83],
                orientation=[0, 0, np.pi],
            )
            mahjong_config["class"] = name2class_xml["mahjong"][0]
        
            self.config["task"]["components"].append(mahjong_config)
    
    def get_instruction(self, target_entity, **kwargs):
        value, type = target_entity[0], target_entity[1]
        instruction = [f"Pick up the mahjong of {value} {type}"]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        value, type = target_entity[0], target_entity[1]
        condition_config = dict(
            contain=dict(
                entities=[f"{value}_{type}"],
                container=target_container
            )
        )
        self.config["task"]["conditions"] = condition_config
        self.target_entity = f"{target_entity[0]}_{target_entity[1]}"
        
@register.add_config_manager("select_mahjong_spatial")
class SelectMahjongSpatialConfigManager(SelectMahjongConfigManager):
    def load_objects(self, **kwargs):
        objects = random.sample(self.seen_object, self.num_object)
        start_point = random.uniform(-0.2, -0.1)
        self.order = random.choice(list(range(self.num_object)))
        self.target_entity = objects[self.order]
        y_pos = random.uniform(-0.1, 0)
        for i, object in enumerate(objects):
            value, type = object[0], object[1]
            mahjong_config = dict(
                name=f"{value}_{type}",
                xml_path=name2class_xml["mahjong"][-1],
                value=value,
                suite=type,
                position=[start_point + i * 0.07, y_pos, 0.83],
                orientation=[0, 0, np.pi],
            )
            mahjong_config["class"] = name2class_xml["mahjong"][0]
        
            self.config["task"]["components"].append(mahjong_config)
    
    def get_condition_config(self, target_container, **kwargs):
        return super().get_condition_config(self.target_entity, target_container)

    def get_instruction(self, target_container, **kwargs):
        instruction = [f"Put the {self.order+1}th mahjong from left to right on the {target_container}"]
        self.config["task"]["instructions"] = instruction
    
@register.add_config_manager("select_mahjong_semantic")
class SelectMahjongSemanticConfigManager(SelectMahjongConfigManager):
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"You are replacing me to play mahjong, and i think it's time to take out {target_entity} now"]
        self.config["task"]["instructions"] = instruction

@register.add_config_manager("select_unique_type_mahjong")
class SelectUniqueTypeMahjongConfigManager(SelectMahjongConfigManager):
    def load_objects(self, **kwargs):
        unique_type = random.choice(["man", "sou", "pin"])
        mans = [mahjong for mahjong in self.seen_object if "man" in mahjong]
        sous = [mahjong for mahjong in self.seen_object if "sou" in mahjong]
        pins = [mahjong for mahjong in self.seen_object if "pin" in mahjong]
        num_1 = random.randint(2, self.num_object - 3)
        num_2 = self.num_object - num_1 - 1
        if unique_type == "man": 
            objects = random.sample(mans, 1)
            objects.extend(random.sample(sous, num_1))
            objects.extend(random.sample(pins, num_2))
        elif unique_type == "sou":
            objects = random.sample(sous, 1)
            objects.extend(random.sample(mans, num_1))
            objects.extend(random.sample(pins, num_2))
        else:
            objects = random.sample(pins, 1)
            objects.extend(random.sample(mans, num_1))
            objects.extend(random.sample(sous, num_2))
        self.target_entity = objects[0]
        start_point = random.uniform(-0.3, -0.1)
        y_pos = random.uniform(-0.1, 0)
        random.shuffle(objects)
        for i, object in enumerate(objects):
            value, type = object[0], object[1]
            mahjong_config = dict(
                name=f"{value}_{type}",
                xml_path=name2class_xml["mahjong"][-1],
                value=value,
                suite=type,
                position=[start_point + i * 0.07, y_pos, 0.83],
                orientation=[0, 0, np.pi],
            )
            mahjong_config["class"] = name2class_xml["mahjong"][0]
        
            self.config["task"]["components"].append(mahjong_config)
    
    def get_condition_config(self, target_container, **kwargs):
        return super().get_condition_config(self.target_entity, target_container)

    def get_instruction(self, target_container, **kwargs):
        instruction = [f"Put the unique type mahjong on the {target_container}"]
        self.config["task"]["instructions"] = instruction

@register.add_task("select_mahjong")
class SelectMahjongTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-np.pi, 0, np.pi]]),
            partial(SkillLib.place, target_container_name=self.target_container), 
        ]
        return skill_sequence

@register.add_task("select_mahjong_spatial")
class SelectMahjongSpatialTask(SelectMahjongTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_mahjong_semantic")
class SelectMahjongSemanticTask(SelectMahjongTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_unique_type_mahjong")
class SelectUniqueTypeMahjongTask(SelectMahjongTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)