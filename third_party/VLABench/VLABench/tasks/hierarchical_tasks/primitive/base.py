from VLABench.tasks.dm_task import *

class PrimitiveTask(LM4ManipBaseTask):
    def reset_intention_distance(self):
        self.intention_distance = dict()
        entity_names = list(self.entities.keys())
        for ignore_entity in self.random_ignored_entities:
            if ignore_entity in entity_names:
                entity_names.remove(ignore_entity)
        for entity_name in entity_names: 
            self.intention_distance[entity_name] = np.inf
            
    def reset_task_progress(self):
        self.target_is_grasped = dict()
        if isinstance(self.target_entity, str):
            self.target_is_grasped[self.target_entity] = False
        elif isinstance(self.target_entity, list):
            for entity in self.target_entity:
                self.target_is_grasped[entity] = False
        
    def update_intention_distance(self, physics):
        ee_pos = self.robot.get_end_effector_pos(physics)
        for key, entity in self.entities.items():
            if key in self.random_ignored_entities: continue
            self.intention_distance[key] = min(self.intention_distance[key], distance(ee_pos, entity.get_xpos(physics)))
    
    def update_task_progress(self, physics):
        if isinstance(self.target_entity, list):
            for entity in self.target_entity:
                if self.entities[entity].is_grasped(physics, self.robot):
                    self.target_is_grasped[entity] = True
        elif isinstance(self.target_entity, str):
            if self.entities[self.target_entity].is_grasped(physics, self.robot):
                self.target_is_grasped[self.target_entity] = True
        
    def get_intention_score(self, physics, threshold=0.2, discrete=True):
        if isinstance(self.target_entity, list):
            return self.get_intention_score_to_entity(physics, self.target_entity[-1], threshold, discrete)
        return self.get_intention_score_to_entity(physics, self.target_entity, threshold, discrete)
    
    def get_task_progress(self, physics):
        # FIXME: temporary solution: in primitive tasks, a successful pick often occupies half of the task progress
        _, conditions_met = self.conditions.met_progress(physics)
        n_condition = len(self.conditions)
        n_condition += len(self.target_is_grasped)
        target_entity_met = []
        for value in self.target_is_grasped.values():
            if value: target_entity_met.append(value)
        return (len(conditions_met) + len(target_entity_met)) / n_condition
    
    def get_intention_score_to_entity(self, physics, entity_name, threshold=0.2, discrete=False):
        """
        Get the intention score of the entity during carry out, computed by the min distance to the entity.
        """
        if discrete:
            return int(self.intention_distance[entity_name] < threshold)
        else:
            if threshold - self.intention_distance[entity_name] < 0:
                return 0
            return 1 / (1 + (threshold - self.intention_distance[entity_name]) + 1e-6)

class PressButtonTask(PrimitiveTask):
    """
    Base class for task to press button for question-answering.
    This type of tasks can be easily expanded with previous vision-language QA datasets, 
        to evaluate the capability retention of VLA-based VLMs. 
    """
    @property
    def target_button(self):
        return self.config_manager.target_button
    
    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        for key in list(self.entities.keys()):
            if "button" in key:
                button = self.entities[key]
                button.detach()
                self._arena.attach(button)

    def get_expert_skill_sequence(self, physics):
        target_button_pos = self.entities[self.target_button].get_xpos(physics)
        skill_sequence = [
            partial(SkillLib.press, target_pos=target_button_pos)
        ]
        return skill_sequence

    def reset_task_progress(self): # press button task didn't compute the pick progress
        pass
    
    def update_task_progress(self, physics):
        pass
    
    def get_task_progress(self, physics):
        return self.conditions.is_met(physics)
    
    def get_intention_score(self, physics, threshold=0.2, discrete=True):
        target_button = self.conditions.conditions[0].button._mjcf_model.model
        return self.get_intention_score_to_entity(physics, target_button, threshold, discrete)
   