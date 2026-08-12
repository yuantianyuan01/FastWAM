import random
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.utils.utils import flatten_list

@register.add_config_manager("select_drink")
class SelectDrinkConfigManager(BenchTaskConfigManager):
    """
    Take out the specific drink from the fridge. 
    """
    def __init__(self, 
                 task_name,
                 num_objects = [2],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def load_objects(self, target_entity):
        objects = []
        objects.append(target_entity)
        other_objects = flatten_list(self.seen_object) + flatten_list(self.unseen_object)
        other_objects.remove(target_entity)
        objects.extend(random.sample(other_objects, self.num_object-1))
        object_configs = []
        random.shuffle(objects)
        for i, object in enumerate(objects):
            object_config = self.get_entity_config(object,
                                                   position=[(i-0.5)*0.12, random.uniform(-0.04, -0.02), 0.15], 
                                                   )
        
            object_configs.append(object_config)
        return object_configs
    
    def get_task_config(self, target_entity, target_container, init_container,**kwargs):
        self.target_entity, self.target_container, self.init_container = target_entity, target_container, init_container
        subentities = self.load_objects(target_entity)
        self.load_init_containers(init_container, subentities)
        self.get_condition_config(target_entity, init_container=init_container)
        self.get_instruction(target_entity, init_container=init_container)
        return self.config
    
    def get_condition_config(self, target_entity, init_container, **kwargs):
        conditions_config = dict(
            not_contain=dict(
                container=f"{init_container}",
                entities=[f"{target_entity}"]
            ),
        )
        self.config["task"]["conditions"] = conditions_config
        return conditions_config
    
    def get_instruction(self, target_entity, init_container, **kwargs):
        instruction = [f"Take out the {target_entity} from the {init_container}"]
        self.config["task"]["instructions"] = instruction
        return instruction
    
    def load_init_containers(self, target_init_container, subentities):
        container_config = self.get_entity_config(target_init_container,
                                                  position=[random.uniform(-0.1, 0.1), random.uniform(0.3, 0.4), 0.74],
                                                  )
        container_config["subentities"] = subentities
        self.config["task"]["random_ignored_entities"].append(target_init_container)
        self.config["task"]["components"].append(container_config)

@register.add_config_manager("select_drink_common_sense")
class SelectDrinkConfigCommonSense(SelectDrinkConfigManager):
    """
    Select a type of drink with common sense
    """
    def load_objects(self, target_entity):
        objects = []
        objects.append(target_entity)
        other_objects = self.seen_object.copy() + self.unseen_object.copy()
        for similar_objects in self.seen_object + self.unseen_object:
            if isinstance(similar_objects, list) and target_entity in similar_objects:
                other_objects.remove(similar_objects)
            elif isinstance(similar_objects, str) and target_entity == similar_objects:
                other_objects.remove(similar_objects)
        other_objects_flatten = []
        for similar_objects in other_objects:
            other_objects_flatten.extend(similar_objects)
        objects.extend(random.sample(other_objects_flatten, self.num_object-1))
        object_configs = []
        random.shuffle(objects)
        for i, object in enumerate(objects):
            object_config = self.get_entity_config(object,
                                                position=[(i-0.5)*0.12, random.uniform(-0.04, -0.02), 0.15], 
                                                   )
        
            object_configs.append(object_config)
        return object_configs
    
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Take me a bottle of {target_entity}"]
        return instruction
    
@register.add_config_manager("select_drink_spatial")
class SelectDrinkConfigSpatial(SelectDrinkConfigManager):
    """
    Select a specific drink from the specific place
    """
    def get_instruction(self, target_entity, init_container, **kwargs):
        if "outside" in self.target_entity:
            instruction = [f"Take the {target_entity} outside on the table"]
        else:
            instruction = [f"Take the {target_entity} in {init_container}"]
        return instruction

    def load_objects(self, target_entity):
        inside_entity_configs = super().load_objects(target_entity)
        outside_entity = [target_entity]
        self.target_entity = target_entity if random.random() > 0.5 else f"{target_entity}_outside"
        other_objects = flatten_list(self.seen_object) + flatten_list(self.unseen_object)
        other_objects.remove(target_entity)
        outside_entity.extend(random.sample(other_objects, self.num_object-1))
        for entity in outside_entity:
            entity_config = self.get_entity_config(entity, 
                                                   specific_name=f"{entity}_outside")
            self.config["task"]["components"].append(entity_config)
        return inside_entity_configs
    
    def get_condition_config(self, target_entity, init_container, **kwargs):
        if "outside" in self.target_entity:
            conditions_config = dict(
                is_grasped=dict(
                    entities=[f"{self.target_entity}"],
                    robot="franka"
                )
            )
        else:
            conditions_config = dict(
                not_contain=dict(
                    container=f"{init_container}",
                    entities=[f"{self.target_entity}"]
                )
            )
        self.config["task"]["conditions"] = conditions_config

@register.add_config_manager("select_drink_semantic")
class SelectDrinkSemanticConfigManager(SelectDrinkConfigManager):
    """
    Select a specific drink with the specific semantically rich information
    """
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Take the {target_entity}"]
        return instruction


@register.add_task("select_drink")
class SelectDrinkTask(PrimitiveTask):
    def __init__(self, task_name, robot, eval=False, **kwargs):
       super().__init__(task_name, robot, eval, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        """
        Attach the fridge to the arena for reasonable and stable visual display.
        """
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if "fridge" in key:
                entity.detach()
                self._arena.attach(entity)
    
    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-np.pi/2, -np.pi/2, 0]]),
            partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.1),
            partial(SkillLib.pull, pull_distance=0.3),
        ]
        return skill_sequence

@register.add_task("select_drink_common_sense")
class SelectDrinkCommonSenseTask(SelectDrinkTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_drink_spatial")
class SelectDrinkSpatialTask(SelectDrinkTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        
@register.add_task("select_drink_semantic")
class SelectDrinkSemanticTask(SelectDrinkTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)