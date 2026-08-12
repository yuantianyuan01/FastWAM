import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask
from VLABench.configs.constant import name2class_xml
from VLABench.tasks.components import CardHolder, Poker
from VLABench.tasks.hierarchical_tasks.poker_utils import *
from VLABench.tasks.condition import ConditionSet, ContainCondition
from VLABench.utils.utils import euler_to_quaternion

@register.add_config_manager("texas_holdem")
class TexasHoldemConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=7,
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def get_seen_task_config(self):
        container = random.choice(self.seen_container)
        return self.get_task_config(None, container, None)

    def get_unseen_task_config(self):
        container = random.choice(self.unseen_container)
        return self.get_task_config(None, container, None)
    
    def load_containers(self, target_container):
        container_config = self.get_entity_config(target_container, 
                                                  position=[random.uniform(-0.2, 0.2), random.uniform(0.3, 0.35), 0.8],
                                                  specific_name="target_container")
        self.config["task"]["components"].append(container_config)
    
    def load_objects(self, target_entity):
        # card holder& poker init
        pokers = random.sample(CARDS, k=self.num_object)
        
        for i in range(self.num_object):
            position = [(i-2)*0.1+random.uniform(0, 0.05), random.uniform(-0.05, 0.05), 0.8] if i < 5 else [(i-5.5)*0.1+random.uniform(-0.03, 0.03), random.uniform(-0.25, -0.15), 0.8]
            card_holder_config = dict(
                name=f"card_holder{i}",
                position=position,
                orientation=[0, 0, np.pi/2]
            )
            card_holder_config["class"] = CardHolder
            card_holder_config["subentities"] = []
            value, suite = pokers[i][0], pokers[i][1]

            name = f"{value}_{suite}" if suite == "joker" else f"{value}_of_{suite}"

            poker_config = dict(
                name=name,
                xml_path=name2class_xml["poker"][-1],
                value=value,
                suite=suite,
                position=[0, 0, 0.07],
                orientation=[0, np.pi/2, 0]
            )
            poker_config["class"] = Poker
            card_holder_config["subentities"].append(poker_config)
            self.config["task"]["components"].append(card_holder_config)
        return self.config  
    
    def get_condition_config(self, **kwargs):
        pass  
    
    def get_instruction(self, **kwargs):
        instruction = ["We're playing Texas hodl'em game! What's your largest cards? Show me on the placemat" ]
        self.config["task"]["instructions"] = instruction
        
@register.add_config_manager("texas_holdem_explore")
class TexasHoldemExploreConfigManager(TexasHoldemConfigManager):
    def load_objects(self, target_entity):
        super().load_objects(target_entity)
        n_face_down = random.choice([1]) # TODO: change to [1, 2]
        poker_id = random.sample(range(self.num_object - 2), n_face_down)
        start_id = 0
        for entity_config in self.config["task"]["components"]:
            if entity_config["class"] == CardHolder:
                if start_id in poker_id:
                    entity_config["subentities"][0]["name"] += "_face_down"
                    entity_config["subentities"][0]["orientation"] = [0, np.pi, 0]
                    entity_config["subentities"][0]["position"] = [0, 0.02, 0.07]
                start_id += 1
                
        return self.config

@register.add_task("texas_holdem")
class PokerPlayTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        self.pokers = list()
        super().__init__(task_name, robot, **kwargs)
        
    def load_entity_from_config(self, config, parent_node=None):
        entity = super().load_entity_from_config(config, parent_node)
        if isinstance(entity, Poker):
            self.pokers.append(entity)
        return entity
    
    @property
    def target_entity(self):
        return list(self._target_entities.keys())
    
    @property
    def target_entities(self):
        return list(self._target_entities.keys())
    
    def init_conditions(self):
        self._target_entities = dict()
        rank, target_cards = get_largest_combination([(poker.name.split('_')[0], poker.name.split('_')[2]) for poker in self.pokers])
        self.max_cardtype = list(RANKING.keys())[-rank]
        assert len(target_cards) > 0
        for target_card in target_cards:
            target_name = "{}_of_{}".format(VALUES[target_card[0] - 2], target_card[1])
            target_entity = self.entities[target_name]
            self._target_entities[target_name] = target_entity
        entities=list(self._target_entities.values())
        if None in entities:
            entities.remove(None)
        on_condition = ContainCondition(entities=entities, 
                                        container=self.entities.get("target_container", None))
        self.conditions = ConditionSet([on_condition])
    
    def initialize_episode(self, physics, random_state):
        return super().initialize_episode(physics, random_state)

    def initialize_episode_mjcf(self, random_state):
        return super().initialize_episode_mjcf(random_state)
    
    def should_terminate_episode(self, physics):
        terminal = self.conditions.is_met(physics)
        return terminal
    
    def get_expert_skill_sequence(self, physics):
        if isinstance(self.target_entities, dict):
            target_entities = list(self.target_entities.keys())
        else:
            target_entities = self.target_entities

        skill_sequence = []
        for entity in target_entities:
            skill_sequence.extend([
                partial(SkillLib.pick, target_entity_name=entity, prior_eulers=[[-np.pi, 0, 0]]),
                partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.1),
                partial(SkillLib.place, target_container_name="target_container"),
            ])
        return skill_sequence

@register.add_task("texas_holdem_explore")
class PokerPlayExploreTask(PokerPlayTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)
    
    def init_conditions(self):
        self._target_entities = dict()
        rank, target_cards = get_largest_combination([(poker.name.split('_')[0], poker.name.split('_')[2]) for poker in self.pokers])
        self.max_cardtype = list(RANKING.keys())[-rank]
        assert len(target_cards) > 0
        for target_card in target_cards:
            target_name = "{}_of_{}".format(VALUES[target_card[0] - 2], target_card[1])
            if target_name not in self.entities:
                target_name += "_face_down"
            target_entity = self.entities[target_name]
            self._target_entities[target_name] = target_entity
        entities=list(self._target_entities.values())
        if None in entities:
            entities.remove(None)
        on_condition = ContainCondition(entities=entities, 
                                        container=self.entities.get("target_container", None))
        self.conditions = ConditionSet([on_condition])
    
    def get_expert_skill_sequence(self, physics):
        skill_sequence = []
        if isinstance(self.target_entities, dict):
            target_entities = list(self.target_entities.keys())
        else:
            target_entities = self.target_entities
        facing_down_entity = [entity for entity in self.entities.keys() if "face_down" in entity]
        for entity in facing_down_entity:
            skill_sequence.extend([
                partial(SkillLib.pick, target_entity_name=entity, prior_eulers=[[np.pi*6/7, 0, 0]]),
                partial(SkillLib.lift, target_quat=euler_to_quaternion(-np.pi*0.8, 0, 0), gripper_state=np.zeros(2), lift_height=0.1),
            ])
            if entity in target_entities: # remove the facing down poker to avoid duplicate actions
                target_entities.remove(entity)
        
        for entity in target_entities:
            skill_sequence.extend([
                partial(SkillLib.pick, target_entity_name=entity, prior_eulers=[[-np.pi, 0, 0]]),
                partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.1),
                partial(SkillLib.place, target_container_name="target_container"),
            ])
        return skill_sequence
        