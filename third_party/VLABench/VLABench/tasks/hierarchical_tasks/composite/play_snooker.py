import random
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive.select_billiards_series import SNOOKER, SelectBilliardsConfigManager, SelectBilliardsTask

@register.add_config_manager("play_snooker")
class PlaySnookerConfigManager(SelectBilliardsConfigManager):
    def __init__(self, 
                 task_name,
                 **kwargs):
        super().__init__(task_name,  **kwargs)
    
    def get_task_config(self, **kwargs):
        self.load_objects()
        self.load_containers()
        self.get_instruction()
        self.get_condition_config()
        return self.config     
    
    def load_objects(self, **kwargs):
        start_index = random.randint(1, 6)
        snookers = SNOOKER[start_index:] # the balls to place in order
        self.target_entities = snookers
        if random.random() < 0.5: # add white ball as disturbance
            snookers.append("snooker_white")
        for i, snooker in enumerate(snookers):
            snooker_config = self.get_entity_config("billiards",
                                                    position=[-0.1+i*0.1, 0.2, 0.6],
                                                    specific_name=snooker,
                                                    value=snooker)
            self.config["task"]["components"].append(snooker_config)
    
    def get_instruction(self, **kwargs):
        instruction = ["Please kill the snooker game one time just by putting the ball into the hole."]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, **kwargs):
        condition_config = dict(
            contain=dict(
                container="billiards_table",
                entities=self.target_entities
            )
        )
        self.config["task"]["conditions"] = condition_config
        
    
@register.add_task("play_snooker")
class PlaySnookerTask(SelectBilliardsTask):
    def after_step(self, physics, random_state):
        # if the ball stay in the hole, remove it to the same space of the table for termination condition
        return super().after_step(physics, random_state)
    
    def should_terminate_episode(self, physics):
        return super().should_terminate_episode(physics)  
    