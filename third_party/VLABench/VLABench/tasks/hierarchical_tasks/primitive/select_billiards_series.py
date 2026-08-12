import random
from VLABench.utils.register import register
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager

SOLID = ["solid_1", "solid_2", "solid_3", "solid_4", "solid_5", "solid_6", "solid_7"]
STRIPED = ["striped_9", "striped_10", "striped_11", "striped_12","striped_13", "striped_14", "striped_15"]

BILLIARDS = SOLID + STRIPED + ["white", "black_8"]

SNOOKER = ["snooker_red", "snooker_yellow", "snooker_green", "snooker_brown", "snooker_blue", "snooker_pink", "snooker_black"]
REACHABLE_HOLES = ["container_right_1", "container_right_2", "container_left_1", "container_left_2", "container_mid_1", "container_mid_2"]

reachable_holes2natural_language = {
    "container_right_1": "right bottom pocket",
    "container_right_2": "right top pocket",
    "container_left_1": "left bottom pocket",
    "container_left_2": "left top pocket",
    "container_mid_1": "left middle pocket",
    "container_mid_2": "right middle pocket"
}

@register.add_config_manager("select_billiards")
class SelectBilliardsConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=[4, 5],
                 seen_object=BILLIARDS[::2], 
                 unseen_object=BILLIARDS[1::2],
                 **kwargs):
        super().__init__(task_name, num_objects, seen_object=seen_object, unseen_object=unseen_object,  **kwargs)
    
    def load_containers(self, **kwargs):
        billiards_table = self.get_entity_config("billiards_table",
                                                 position=[0, 0, 0])
        self.target_container = "billiards_table"
        self.config["task"]["components"].append(billiards_table)
    
    def load_objects(self, target_entity):
        objects = [target_entity]
        other_objects = self.seen_object.copy() + self.unseen_object.copy()
        other_objects.remove(target_entity)
        objects.extend(random.sample(other_objects, self.num_object - 1))
        for i, object in enumerate(objects):
            billiard_config = self.get_entity_config("billiards",
                                                     position=[-0.1+i*0.1, 0.2, 0.6],
                                                     specific_name=object,
                                                     value=object)
        
            self.config["task"]["components"].append(billiard_config)
        
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Please put the {target_entity} in any hole."]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, **kwargs):
        condition_config = dict(
            contain=dict(
                container="billiards_table",
                entities=[target_entity]
            )
        )
        self.config["task"]["conditions"] = condition_config
        
@register.add_config_manager("select_billiards_spatial")
class SelectBilliardsSpatialConfigManager(SelectBilliardsConfigManager):
    def load_containers(self, **kwargs):
        self.target_hole = random.choice(REACHABLE_HOLES)
        container_config = self.get_entity_config("billiards_table",
                                                  position=[0, 0, 0],
                                                  target_hole=self.target_hole)
        self.target_container = "billiards_table"
        self.config["task"]["components"].append(container_config)
    
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Please put the {target_entity} in the {reachable_holes2natural_language[self.target_hole]}."]
        self.config["task"]["instructions"] = instruction

@register.add_config_manager("select_billiards_common_sense")
class SelectBilliardsCommonSenseConfigManager(SelectBilliardsConfigManager):
    def __init__(self, 
                 task_name, 
                 num_objects=[4, 5], 
                 **kwargs):
        super().__init__(task_name, num_objects, seen_object=SNOOKER[::2], unseen_object=SNOOKER[1::2], **kwargs)
    
    def load_objects(self, target_entity):
        objects = [target_entity]
        other_objects = SNOOKER + BILLIARDS
        other_objects.remove(target_entity)
        objects.extend(random.sample(other_objects, self.num_object - 1))
        for i, object in enumerate(objects):
            billiard_config = self.get_entity_config("billiards",
                                                     position=[-0.1+i*0.1, 0.2, 0.6],
                                                     specific_name=object,
                                                     value=object)
        
            self.config["task"]["components"].append(billiard_config)
    
    def get_instruction(self, target_entity, **kwargs):
        if "red" in target_entity: score = 1
        elif "yellow" in target_entity: score = 2
        elif "brown" in target_entity: score = 3
        elif "green" in target_entity: score = 4
        elif "blue" in target_entity: score = 5
        elif "pink" in target_entity: score = 6
        elif "black" in target_entity: score = 7
        instruction = [f"Please put the ball worth {score} score in hole"]
        self.config["task"]["instructions"] = instruction   

@register.add_config_manager("select_billiards_semantic")
class SelectBilliardsSemanticConfigManager(SelectBilliardsConfigManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
@register.add_task("select_billiards")
class SelectBilliardsTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, motion_planning_kwargs={"z_threshold": 0.6}),
            partial(SkillLib.place, target_container_name=self.target_container, motion_planning_kwargs={"z_threshold": 0.6}) 
        ]
        return skill_sequence

@register.add_task("select_billiards_spatial")
class SelectBilliardsSpatialTask(SelectBilliardsTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_billiards_common_sense")
class SelectBilliardsCommonSenseTask(SelectBilliardsTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_billiards_semantic")
class SelectBilliardsSemanticTask(SelectBilliardsTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)