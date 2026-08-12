import random
import numpy as np
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.utils.utils import flatten_list

@register.add_config_manager("select_ingredient")
class SelectIngredientConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects = [3, 4],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.z_position = [0.83, 1.13]

    def load_containers(self, target_container):
        super().load_containers(target_container)
        self.config["task"]["components"][-1]["position"] = [-0.55 + random.uniform(-0.05, 0.02), 
                                                             0.25 + random.uniform(-0.03, 0.03), 0.8]
    
    def load_init_containers(self, init_container):
        if init_container is not None:
            self.init_container_config = self.get_entity_config(init_container,
                                                           position=[0, 0.5, 0],
                                                           randomness=None)
            self.config["task"]["components"].append(self.init_container_config)
            
    def load_objects(self, target_entity):
        self.init_container_config["subentities"] = []
        
        objects = []
        objects.append(target_entity)
        other_objects = flatten_list(self.seen_object) + flatten_list(self.unseen_object)
        other_objects.remove(target_entity)
        objects.extend(random.sample(other_objects, self.num_object-1))
        random.shuffle(objects)
        for i, object in enumerate(objects):
            object_config = self.get_entity_config(object, 
                                                   position=[-0.15+i*0.12+random.uniform(-0.02, 0.02), 
                                                             random.uniform(-0.3, -0.28), random.choice([0.83, 1.13])])
            self.init_container_config["subentities"].append(object_config)
    
    def get_instruction(self, target_entity, init_container, **kwargs):
        self.config["task"]["instructions"] = [f"Please take out {target_entity} from {init_container}."]
        return self.config
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        condition_config = dict(
            contain=dict(
                entities=[target_entity],
                container=target_container
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_config_manager("select_ingredient_spatial")
class SelectIngredientSpatial(SelectIngredientConfigManager):
    def load_objects(self, target_entity):
        self.init_container_config["subentities"] = []
        objects = []
        objects.append(target_entity)
        other_objects = flatten_list(self.seen_object) + flatten_list(self.unseen_object)
        other_objects.remove(target_entity)
        objects.extend(random.sample(other_objects, self.num_object-1))
        random.shuffle(objects)
        # position 
        n_top_layer_object = random.randint(1, self.num_object - 1)
        n_mid_layer_object = self.num_object - n_top_layer_object
        top_layer_x = random.sample(np.linspace(-0.15, 0.15, 4).tolist(), int(n_top_layer_object))
        top_layer_pos = [[x, random.uniform(-0.3, -0.28), self.z_position[1]] for x in top_layer_x]
        mid_layer_x = random.sample(np.linspace(-0.15, 0.15, 4).tolist(), int(n_mid_layer_object))
        mid_layer_pos = [[x, random.uniform(-0.3, -0.28), self.z_position[0]] for x in mid_layer_x]
        positions = top_layer_pos + mid_layer_pos
        for i, (object, pos) in enumerate(zip(objects, positions)):
            if object == target_entity:  # get target entity position and order
                self.layer = "top" if i < n_top_layer_object else "bottom"
                self.order = i + 1 if self.layer == "top" else i - n_top_layer_object + 1
            object_config = self.get_entity_config(object, 
                                                   position=pos)
            self.init_container_config["subentities"].append(object_config)
    
    def get_instruction(self, **kwargs):
        if self.order == 1: order = "first"
        elif self.order == 2: order = "second"
        elif self.order == 3: order = "third"
        else: order = f"{self.order}th"
        instruction = [f"Please take the {order} ingredient on the {self.layer} layer"]
        self.config["task"]["instructions"] = instruction

@register.add_config_manager("select_ingredient_common_sense")
class SelectIngredientCommonSense(SelectIngredientConfigManager):
    def get_instruction(self, target_entity, init_container, **kwargs):
        instruction = [f"Please take out the vegetable with ..."]
        self.config["task"]["instructions"] = instruction

@register.add_config_manager("select_ingredient_semantic")
class SelectIngredientSemantic(SelectIngredientConfigManager):
    def get_instruction(self, target_entity, init_container, **kwargs):
        instruction = [f"Please take out the ingredient during our conversation"]
        self.config["task"]["instructions"] = instruction

@register.add_task("select_ingredient")
class SelectIngredientTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        target_container_pos = np.array(self.entities[self.target_container].get_place_point(physics))
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-np.pi,  1.1, -np.pi/2]]), # prior orientation is 
            partial(SkillLib.lift, gripper_state=np.ones(2) * 0.008, lift_height=0.05),
            partial(SkillLib.pull, gripper_state=np.ones(2) * 0.008),
            partial(SkillLib.moveto, target_pos=target_container_pos - np.array([0, 0.3, 0]), gripper_state=np.zeros(2)),
            partial(SkillLib.place, target_container_name=self.target_container),
        ]
        return skill_sequence

@register.add_task("select_ingredient_spatial")
class SelectIngredientSpatialTask(SelectIngredientTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_ingredient_common_sense")
class SelectIngredientCommonSenseTask(SelectIngredientTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_ingredient_semantic")
class SelectIngredientSemanticTask(SelectIngredientTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)