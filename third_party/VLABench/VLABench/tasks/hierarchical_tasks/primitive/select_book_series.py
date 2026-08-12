import random
import numpy as np
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.utils.utils import flatten_list

BOOK_TYPE = ["biographies", "fiction", "biotechnology", "physics", "law", "computer_science", 
"history"]

@register.add_config_manager("select_book")
class SelectBookConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=[3, 4],
                 **kwargs):
        """select the book with specific name"""
        super().__init__(task_name, num_objects, **kwargs)
    
    def load_init_containers(self, init_container):
        if init_container is not None:
            self.config["task"]["components"].append(self.get_entity_config(init_container, 
                                                                            position=[random.uniform(-0.1, 0.1), 
                                                                                      random.uniform(0.25, 0.35), 0.8]))
    
    def load_objects(self, target_entity):
        self.config["task"]["components"][-1]["subentities"] = []
        subentities = [target_entity]
        other_objects = flatten_list(self.seen_object) + flatten_list(self.unseen_object)
        other_objects.remove(target_entity)
        other_subentities = random.sample(other_objects, self.num_object-1)
        subentities.extend(other_subentities)
        random.shuffle(subentities)
        for i, subentity in enumerate(subentities):
            subentity_config = self.get_entity_config(subentity.lower(),
                                                      position=[(i-1.5)*np.random.uniform(0.1, 0.2), 0, 0.47],
                                                      orientation=[np.pi, 0, np.pi/2], 
                                                      randomness=dict(pos=[0.02, 0.02, 0]))
            self.config["task"]["components"][-1]["subentities"].append(subentity_config)
    
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Please take the book {target_entity}"]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, init_container, **kwargs):
        condition_config = dict(
            not_contain=dict(
                container=init_container,
                entities=[target_entity]
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_config_manager("select_specific_type_book")
class SelectSpecificTypeBookConfigManager(SelectBookConfigManager):
    def load_objects(self, target_entity):
        self.config["task"]["components"][-1]["subentities"] = []
        subentities = [target_entity]
        other_objects = []
        for similar_seen_obj, similar_unseen_obj in zip(self.seen_object, self.unseen_object):
            if target_entity in similar_seen_obj or target_entity in similar_unseen_obj:
                continue
            other_objects.append(similar_seen_obj+similar_unseen_obj)
        other_type_id = random.sample(range(len((other_objects))), self.num_object-1)
        for id in other_type_id:
            subentities.append(random.choice(other_objects[id]))
        random.shuffle(subentities)
        for i, subentity in enumerate(subentities):
            subentity_config = self.get_entity_config(subentity.lower(),
                                                      position=[(i-1.5)*np.random.uniform(0.1, 0.2), 0, 0.46],
                                                      orientation=[np.pi, 0, np.pi/2])
            self.config["task"]["components"][-1]["subentities"].append(subentity_config)
    
    def get_instruction(self, target_entity, **kwargs):
        for i in range(len(BOOK_TYPE)):
            if target_entity in self.seen_object[i] or target_entity in self.unseen_object[i]:
                book_type = BOOK_TYPE[i]
                break
        instruction = [f"Please take out the book of {book_type} on the shelf"]
        self.config["task"]["instructions"] = instruction

@register.add_config_manager("select_book_spatial")
class SelectBookSpatialConfigManager(SelectBookConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects = [5, 6],
                 z_position = [0.15, 0.45],
                 **kwargs):
        self.z_position = z_position
        super().__init__(task_name, num_objects, **kwargs)
    
    def load_objects(self, target_entity):
        self.config["task"]["components"][-1]["subentities"] = []
        subentities = [target_entity]
        other_objects = flatten_list(self.seen_object) + flatten_list(self.unseen_object)
        other_objects.remove(target_entity)
        other_subentities = random.sample(other_objects, self.num_object-1)
        subentities.extend(other_subentities)
        # position generation
        n_top_layer_object = random.randint(1, self.num_object - 1)
        n_bottom_layer_object = self.num_object - n_top_layer_object
        top_layer_x = random.sample(np.linspace(-0.25, 0.25, 6).tolist(), int(n_top_layer_object))
        top_layer_pos = sorted([[x, 0, self.z_position[1]] for x in top_layer_x])
        bottom_layer_x = random.sample(np.linspace(-0.25, 0.25, 6).tolist(), int(n_bottom_layer_object))
        bottom_layer_pos = sorted([[x, 0, self.z_position[0]] for x in bottom_layer_x])
        positions = top_layer_pos + bottom_layer_pos
        assert len(positions) == self.num_object == len(subentities), "number of objects and positions should be the same"
        random.shuffle(subentities)
        for i, (subentity, position) in enumerate(zip(subentities, positions)):
            if subentity == target_entity:  # get target entity position and order
                self.layer = "top" if i < n_top_layer_object else "bottom"
                self.order = i + 1 if self.layer == "top" else i - n_top_layer_object + 1
            subentity_config = self.get_entity_config(subentity.lower(),
                                                      position=position,
                                                      orientation=[np.pi, 0, np.pi/2])
            self.config["task"]["components"][-1]["subentities"].append(subentity_config)
    
    def get_instruction(self, **kwargs):
        if self.order == 1: order = "first"
        elif self.order == 2: order = "second"
        elif self.order == 3: order = "third"
        else: order = f"{self.order}th"
        instruction = [f"Please take the {order} book on the {self.layer} layer"]
        self.config["task"]["instructions"] = instruction
        
@register.add_config_manager("select_book_semantic")
class SelectBookSemanticConfigManager(SelectBookConfigManager):
    def get_instruction(self, **kwargs):
        instruction = [f"I'm ... Could you recommend a book and take it out for me?"]
        self.config["task"]["instructions"] = instruction

@register.add_task("select_book")
class SelectBookTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        """
        Attach the shelf for stable interaction.
        """
        # FIXME delete the re-attachment of the shelf for more realistic interaction
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if "shelf" in key:
                entity.detach()
                self._arena.attach(entity)
    
    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-np.pi/2, -np.pi/2, 0]]),
            partial(SkillLib.pull, )
        ]
        return skill_sequence
    
@register.add_task("select_specific_type_book")
class SelectSpecificTypeBookTask(SelectBookTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_book_spatial")
class SelectBookSpatialTask(SelectBookTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_book_semantic")
class SelectBookSemanticTask(SelectBookTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)