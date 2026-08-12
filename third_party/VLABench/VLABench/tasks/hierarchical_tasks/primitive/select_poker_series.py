import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.configs.constant import name2class_xml
from VLABench.tasks.components import Poker, CardHolder

SUITES = ["spades", "clubs", "diamonds", "hearts"]
VALUES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace"]

CARDS = [(value, suite) for value in VALUES for suite in SUITES]

value2int = {value: i + 2 for i, value in enumerate(VALUES)}

@register.add_config_manager("select_poker")
class SelectPokerConfigManager(BenchTaskConfigManager):
    def __init__(self,
                 task_name,
                 num_objects=[3, 4],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def get_seen_task_config(self):
        return self.get_task_config(None, None, None)
    
    def get_unseen_task_config(self):
        return self.get_task_config(None, None, None)
    
    def load_objects(self, **kwargs):
        cards = random.sample(CARDS, self.num_object)
        self.target_card = random.choice(cards) # tuple: (value, suite)
        self.target_entity = f"{self.target_card[0]}_of_{self.target_card[1]}"
        for i, card in enumerate(cards):
            card_holder_config = dict(
                name=f"card_holder{i}",
                position=[(i-1.5)*random.uniform(0.1, 0.15), random.uniform(0, 0.2), 0.8],
                orientation=[0, 0, np.pi/2]
            )
            card_holder_config["class"] = CardHolder
            card_holder_config["subentities"] = []
            value, suite = card[0], card[1]
            poker_config = dict(
                name=f"{value}_of_{suite}",
                xml_path=name2class_xml["poker"][-1],
                value=value,
                suite=suite,
                position=[0, 0, 0.1],
                orientation=[0, np.pi/2, 0]
            )
            poker_config["class"] = Poker
            card_holder_config["subentities"].append(poker_config)
            self.config["task"]["components"].append(card_holder_config)
            
    def get_instruction(self, **kwargs):
        instruction = [f"Please pick the poker {self.target_card[0]} of {self.target_card[1]}"]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, **kwargs):
        condition_config = dict(
            is_grasped=dict(
                entities=[f"{self.target_card[0]}_of_{self.target_card[1]}"],
                robot="franka"
            ),
            lift=dict(
                entities=[f"{self.target_card[0]}_of_{self.target_card[1]}"],
                target_height=0.9,
            )
            
        )
        self.config["task"]["conditions"] = condition_config
    
@register.add_config_manager("select_nth_largest_poker")
class SelectNthLargestPokerConfigManager(SelectPokerConfigManager):    
    def load_objects(self, **kwargs):
        values = random.sample(VALUES, self.num_object)
        value2ints = [value2int[value] for value in values]
        self.nth_largest = random.randint(1, self.num_object)
        sorted_values = [value for value, _ in sorted(zip(values, value2ints), key=lambda x: x[1], reverse=True)]
        target_value = sorted_values[self.nth_largest - 1]
        for i, value in enumerate(values):
            suite = random.choice(SUITES)
            if target_value == value:
                self.target_card = (value, suite)
                self.target_entity = f"{value}_of_{suite}"
            card_holder_config = dict(
                name=f"card_holder{value}",
                position=[(i-2)*random.uniform(0.1, 0.15), 0.2, 0.8],
                orientation=[0, 0, np.pi/2]
            )
            card_holder_config["class"] = CardHolder
            card_holder_config["subentities"] = []
            poker_config = dict(
                name=f"{value}_of_{suite}",
                xml_path=name2class_xml["poker"][-1],
                value=value,
                suite=suite,
                position=[0, 0, 0.1],
                orientation=[0, np.pi/2, 0]
            )
            poker_config["class"] = Poker
            card_holder_config["subentities"].append(poker_config)
            self.config["task"]["components"].append(card_holder_config)
    
    def get_instruction(self, **kwargs):
        if self.nth_largest == 1: order="the largest"
        elif self.nth_largest == 2: order="the 2nd largest"
        elif self.nth_largest == 3: order="the 3rd largest"
        elif self.nth_largest == self.num_object: order="the smallest"
        else: order=f"the {self.nth_largest}th largest"
        instruction = [f"Please pick the {order} poker"]
        self.config["task"]["instructions"] = instruction

@register.add_config_manager("select_poker_spatial")
class SelectPokerSpatialConfigManager(SelectPokerConfigManager):    
    def load_objects(self, **kwargs):
        cards = random.sample(CARDS, self.num_object)
        self.target_card = random.choice(cards) # tuple: (value, suite)
        self.target_entity = f"{self.target_card[0]}_of_{self.target_card[1]}"
        for i, card in enumerate(cards):
            if card == self.target_card:
                self.order = i + 1
            card_holder_config = dict(
                name=f"card_holder{i}",
                position=[(i-1.5)*random.uniform(0.1, 0.15), 0.2, 0.8],
                orientation=[0, 0, np.pi/2]
            )
            card_holder_config["class"] = CardHolder
            card_holder_config["subentities"] = []
            value, suite = card[0], card[1]
            poker_config = dict(
                name=f"{value}_of_{suite}",
                xml_path=name2class_xml["poker"][-1],
                value=value,
                suite=suite,
                position=[0, 0, 0.05],
                orientation=[0, np.pi/2, 0]
            )
            poker_config["class"] = Poker
            card_holder_config["subentities"].append(poker_config)
            self.config["task"]["components"].append(card_holder_config)
    
    def get_instruction(self, **kwargs):
        if self.order == 1: order="1st"
        elif self.order == 2: order="2nd"
        elif self.order == 3: order="3rd"
        else: order=f"{self.order}th"
        instruction = [f"Please pick the {order} poker from left to right"]
        self.config["task"]["instructions"] = instruction

@register.add_config_manager("select_poker_semantic")    
class SelectPokerSemanticConfigManager(SelectPokerConfigManager):
    def get_instruction(self, **kwargs):
        instruction = [f"Please pick the poker {self.target_card[0]} of {self.target_card[1]} in the interaction"]
        self.config["task"]["instructions"] = instruction

@register.add_task("select_poker")
class SelectPokerTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-np.pi, 0, 0]]),
            partial(SkillLib.lift, ), 
        ]
        return skill_sequence
    
@register.add_task("select_nth_largest_poker")
class SelectNthLargestPokerTask(SelectPokerTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_poker_spatial")
class SelectpokerSpatialTask(SelectPokerTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_poker_semantic")
class SelectPokerSemanticTask(SelectPokerTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)