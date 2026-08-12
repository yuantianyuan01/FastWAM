import random
import os
import json
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask
from VLABench.utils.utils import grid_sample
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("play_math_game")
class MathGameConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_obejcts=[5, 6],
                 **kwargs):
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task_related/math.json"), "r") as f:
            self.all_questions = json.load(f)
        self.seen_question_id = list(self.all_questions.keys())[::2]
        self.unseen_question_id = list(self.all_questions.keys())[1::2]
        super().__init__(task_name, num_obejcts, **kwargs)
    
    def compute_target_numbers(self, answer):
        assert isinstance(answer, str)
        self.target_numbers = []
        for digit in answer:
            if digit.isdigit():
                self.target_numbers.append(digit)
        return self.target_numbers.copy()
    
    def get_seen_task_config(self):    
        target_question = random.choice(self.seen_question_id)
        self.target_question = self.all_questions[target_question]
        target_container = random.choice(self.seen_container)
        target_entities = self.compute_target_numbers(self.target_question["answer"])
        return self.get_task_config(target_entities, target_container, None)
    
    def get_unseen_task_config(self):
        target_question = random.choice(self.unseen_question_id)
        self.target_question = self.all_questions[target_question]
        target_container = random.choice(self.unseen_container)
        target_entities = self.compute_target_numbers(self.target_question["answer"])
        return self.get_task_config(target_entities, target_container, None)

    def load_containers(self, target_container):
        super().load_containers(target_container)    
        self.config["task"]["components"][-1]["position"] = [random.uniform(-0.2, 0.2), random.uniform(-0.1, 0.05), 0.78]
    
    def load_objects(self, target_entity):
        assert isinstance(target_entity, list)
        all_numbers = [str(i) for i in range(10)]
        objects = target_entity.copy()
        for number in objects:
            all_numbers.remove(number)
        objects.extend(random.sample(all_numbers, self.num_object-len(objects)))
        positions = grid_sample(workspace=self.config["task"]["workspace"], 
                                grid_size=self.config["task"]["ngrid"],
                                n_samples=self.num_object, 
                                farthest_sample=False)
        random.shuffle(objects)
        for object, pos in zip(objects, positions):
            pos = [pos[0], pos[1], 0.8]
            object_config = dict(
                name=f"{object}",
                number=object,
                xml_path=name2class_xml["number_cube"][-1],
                position=pos,
            )
            object_config["class"] = name2class_xml["number_cube"][0]
            self.config["task"]["components"].append(object_config)
    
    def get_instruction(self, **kwargs):
        instruction = f"Please give the answer of the following question by rearrange the number cube in {self.target_container}:" + self.target_question["question"]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        condition_config = dict(
            contain=dict(
                entities=target_entity,
                container=target_container,
            ),
            order=dict(
                entities=target_entity,
                axis=[0]
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_task("play_math_game") 
class MathGameTask(CompositeTask):
    def __init__(self, task_name, robot, random_init=False, **kwargs):
        super().__init__(task_name, robot=robot, random_init=random_init, **kwargs)
    
    def get_expert_skill_sequence(self, physics):
        center_place_point = self.entities[self.target_container].get_place_point(physics)[-1]
        skill_sequence = []
        for i, entity in enumerate(self.target_entity):
            target_pos = [(i - 1) * 0.05 + center_place_point[0], center_place_point[1], center_place_point[2] - 0.05]
            skill_sequence.extend([
                partial(SkillLib.pick, target_entity_name=entity, prior_eulers=[[-np.pi, 0, 0]]),
                partial(SkillLib.place, target_container_name=self.target_container, target_pos=target_pos),
            ])
        return skill_sequence
    
    def get_intention_score(self, physics, threshold=0.2, discrete=True):
        intention_scores = []
        if isinstance(self.target_entity, list):
            for entity in self.target_entity:
                intention_scores.append(self.get_intention_score_to_entity(physics, entity, threshold, discrete))
        return np.mean(intention_scores)
    
    def get_task_progress(self, physics):
        return super().get_task_progress(physics)