import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.hierarchical_tasks.composite.base import CompositeTask
from VLABench.configs.constant import name2class_xml
from VLABench.tasks.hierarchical_tasks.mahjong_utils import *

@register.add_config_manager("play_mahjong")
class PlayMahjongConfigManager(BenchTaskConfigManager):
    def __init__(self,
                 task_name,
                 num_objects=3,
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def get_seen_task_config(self):
        return self.get_task_config(None, None, None)

    def get_unseen_task_config(self):
        return super().get_unseen_task_config(None, None, None)
    
    def load_objects(self, **kwargs):
        all_mahjongs, _ = get_all_mahjongs()
        ready_hands, winning_hands = generate_ready_hand_mahjongs()
        for i, hand in enumerate(ready_hands):
            hand_config = dict(
                name=hand,
                xml_path=name2class_xml["mahjong"][-1],
                value=hand.split('_')[1],
                suite=hand.split('_')[0],
                position=[-0.5+i*0.07, 0.2, 0.82],
            )
            hand_config["class"] = name2class_xml["mahjong"][0]
            self.config["task"]["components"].append(hand_config)
            all_mahjongs.remove(hand)
        
        for hand in winning_hands: all_mahjongs.remove(hand)
        
        target_entity = random.choice(winning_hands)
        self.target_entity = f"candidate_{target_entity}"
        candidate_mahjong = [target_entity] + random.sample(all_mahjongs, self.num_object-1)
        for i, candidate in enumerate(candidate_mahjong):
            candidate_config = dict(
                name=f"candidate_{candidate}",
                xml_path=name2class_xml["mahjong"][-1],
                value=candidate.split('_')[1],
                suite=candidate.split('_')[0],
                position=[-0.1+i*0.07, 0.3, 0.82],
            )
            candidate_config["class"] = name2class_xml["mahjong"][0]
            self.config["task"]["components"].append(candidate_config)
    
    def get_instruction(self, **kwargs):
        instruction = ["Please select the winning hand."]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, **kwargs):
        condition_config=dict(
            is_grasped=dict(
                entities=[self.target_entity],
                robot="franka"
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_task("play_mahjong")
class PlayMahjongTask(CompositeTask):
    def __init__(self, task_name, robot, **kwargs):
        self.config_manager_cls = register.load_config_manager("play_mahjong")
        super().__init__(task_name, robot=robot, **kwargs)