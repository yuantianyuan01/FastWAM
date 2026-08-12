from VLABench.tasks.dm_task import *

class CompositeTask(LM4ManipBaseTask):
    def reset_intention_distance(self):
        pass
    
    def reset_task_progress(self):
        pass
    
    def update_task_progress(self, physics):
        pass
    
    def update_intention_distance(self, physics):
        pass
    
class ClusterTask(CompositeTask):
    def __init__(self, task_name, robot, random_init=False, **kwargs):
        super().__init__(task_name, robot, random_init=random_init, **kwargs)
    
    def init_conditions(self):
        if self.config["task"].get("conditions") is not None:
            condition_config = copy.deepcopy(self.config["task"]["conditions"])
        else:
            self.conditions = None
            return False
        assert "or" in condition_config.keys(), "only support 'or' condition in clustering tasks"
        condition_sets = []
        for condition_config in condition_config["or"]:
            conditions = []
            for condition_key, specific_condition in condition_config.items():
                if "contain_" in condition_key:
                    condition_key = "contain"
                condition_cls = register.load_condition(condition_key)
                for k, entities in specific_condition.items():
                    if isinstance(entities, str):
                        specific_condition[k] = self.entities.get(entities, None)
                    elif isinstance(entities, list):
                        specific_condition[k] = [self.entities.get(entity, None) for entity in entities]
                condition = condition_cls(**specific_condition)
                conditions.append(condition)
            condition_set = ConditionSet(conditions)
            condition_sets.append(condition_set)
        self.conditions = OrCondition(condition_sets)
        return True

    def get_expert_skill_sequence(self, physics, prior_eulers):
        cluster_entities_1 = self.config_manager.entities_to_load["cls_1"]
        cluster_entities_2 = self.config_manager.entities_to_load["cls_2"]
        if cluster_entities_1[0] == cluster_entities_1[-1]:
            cluster_entities_1[-1] = cluster_entities_1[-1] + "_1"
        if cluster_entities_2[0] == cluster_entities_2[-1]:
            cluster_entities_2[-1] = cluster_entities_2[-1] + "_1"
        container_1 = self.target_container[0] + "_0"
        container_2 = self.target_container[1] + "_1"
        skill_sequence = []
        for index, (cluster_entities, container) in enumerate(zip([cluster_entities_1, cluster_entities_2], [container_1, container_2])):
            for i, entity in enumerate(cluster_entities):
                skill_sequence.extend([
                        partial(SkillLib.pick, target_entity_name=entity, prior_eulers=prior_eulers),    
                        partial(SkillLib.lift, gripper_state=np.zeros(2)),
                ]) 
                target_container = self.entities[container]
                target_place_point = np.array(target_container.get_place_point(physics)[-1])
                target_place_point[1] += 0.1 * (i-0.5)
                skill_sequence.append(partial(SkillLib.place, target_container_name=container, target_pos=target_place_point))
                if index == 1 and i == 1: # wait when last placing
                    skill_sequence.append(partial(SkillLib.wait))
        return skill_sequence
    
    def reset_task_progress(self):
        pass
        
    def update_task_progress(self, physics):
        pass