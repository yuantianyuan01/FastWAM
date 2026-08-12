import random
import os
import json
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.utils.utils import grid_sample

INGREDIENTS = ["bell_pepper", "broccoli", "carrot", "cheese", "corn", "egg", "eggplant", "fish", "garlic", "mushroom", "onion", "potato", "steak", "sweet_potato", "tomato"]

@register.add_config_manager("cook_dishes")
class CookDishesConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=[4, 5],
                 **kwargs
                 ):
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task_related/recipe.json"), "r") as f:
            recipes = json.load(f)
        self.seen_recipes = recipes["train"]
        self.unseen_recipes = recipes["eval"]
        super().__init__(task_name, num_objects, **kwargs)

    def get_seen_task_config(self):
        self.target_recipe = random.choice(list(self.seen_recipes.keys()))
        target_entities = self.seen_recipes[self.target_recipe]
        target_container = random.choice(self.seen_container)
        return self.get_task_config(target_entities, target_container)
    
    def get_unseen_task_config(self):
        self.target_recipe = random.choice(list(self.unseen_recipes.keys()))
        target_entities = self.unseen_recipes[self.target_recipe]
        target_container = random.choice(self.unseen_container)
        return self.get_task_config(target_entities, target_container)
    
    def get_task_config(self, target_entities, target_container, **kwargs):
        self.target_entities, self.target_container, self.init_container = target_entities, target_container, None
        self.target_entity = None
        self.load_containers(target_container)
        self.load_objects(target_entities)
        self.get_condition_config(target_entities, target_container)
        self.get_instruction()
        return self.config
    
    def load_objects(self, target_entities, **kwargs):
        other_entities = INGREDIENTS.copy()
        for entity in target_entities:
            other_entities.remove(entity)
        entities = target_entities + random.sample(other_entities, self.num_object - len(target_entities))
        random.shuffle(entities)
        positions = grid_sample(workspace=self.config["task"]["workspace"], 
                                grid_size=self.config["task"]["ngrid"],
                                n_samples=self.num_object)
        for i, (entity, pos) in enumerate(zip(entities, positions)):
            target_pos = [pos[0], pos[1], 0.8]
            entity_config = self.get_entity_config(entity, position=target_pos)
            self.config["task"]["components"].append(entity_config)
        
    def load_containers(self, target_container):
        super().load_containers(target_container)
        self.config["task"]["components"][-1]["position"] = [random.uniform(-0.3, -0.1), random.choice([0., 0.4]), 0.83]
        self.config["task"]["components"][-1]["scale"] = [1.3, 1.5]
        
    def get_condition_config(self, target_entities, target_container, **kwargs):
        condition_config = dict(
            contain=dict(
                container=target_container,
                entities=target_entities
            )
        )
        self.config["task"]["conditions"] = condition_config
    
    def get_instruction(self, **kwargs):
        instruction = [f"Prepare the ingredients of {self.target_recipe} in the plate."]
        self.config["task"]["instructions"] = instruction

@register.add_task("cook_dishes")
class CookDishesTask(CompositeTask):
    def __init__(self, task_name, robot, random_init=False, **kwargs):
        self.config_manager_cls = register.load_config_manager("cook_dishes")
        super().__init__(task_name, robot=robot, random_init=random_init, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        target_place_point = self.entities[self.target_container].get_place_point(physics)[-1]
        target_place_points = [np.array([target_place_point[0]-0.05, target_place_point[1]-0.05, target_place_point[2]]),
                               np.array([target_place_point[0]+0.05, target_place_point[1]-0.05, target_place_point[2]]),
                               np.array([target_place_point[0]-0.05, target_place_point[1]+0.05, target_place_point[2]]),
                               np.array([target_place_point[0]+0.05, target_place_point[1]-0.05, target_place_point[2]])]
        skill_sequences = []
        for i, entity in enumerate(self.target_entities):
            skill_sequences.extend([
                partial(SkillLib.pick, target_entity_name=entity, prior_eulers=[[-np.pi, 0, np.pi/2]]),
                partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.2),
                partial(SkillLib.place, target_container_name=self.target_container, target_pos=target_place_points[i])
            ])
        return skill_sequences