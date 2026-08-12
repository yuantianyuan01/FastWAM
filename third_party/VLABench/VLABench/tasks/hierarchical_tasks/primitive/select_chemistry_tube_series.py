import random
import numpy as np
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml


relative_col_pos = [-0.16, -0.08, 0, 0.08, 0.16]
relative_row_pos = [-0.05, 0.05]

@register.add_config_manager("select_chemistry_tube")
class SelectChemistryTubeConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=[2, 3],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
    
    def load_init_containers(self, init_container):
        init_container_config = self.get_entity_config(init_container, position=[random.uniform(-0.2, 0.2), random.uniform(0, 0.2), 0.8], randomness=None)
        self.config["task"]["components"].append(init_container_config)
        
    def load_objects(self, target_entity):
        objects = []
        objects.append(target_entity)
        other_objects = self.seen_object.copy() + self.unseen_object.copy()
        for similar_objects in self.seen_object + self.unseen_object:
            if isinstance(similar_objects, list) and target_entity in similar_objects:
                other_objects.remove(similar_objects)
            elif isinstance(similar_objects, str) and target_entity == similar_objects:
                other_objects.remove(similar_objects)
        other_objects_flatten = []
        for similar_objects in other_objects:
            other_objects_flatten.extend(similar_objects)
        objects.extend(random.sample(other_objects_flatten, self.num_object-1))
        targer_rol_poses = random.sample(relative_col_pos, self.num_object)
        target_poses = [[pos, random.choice(relative_row_pos), 0.05] for pos in targer_rol_poses] 
        
        init_container_config = self.config["task"]["components"][-1]
        init_container_config["subentities"] = []
        for object, pos in zip(objects, target_poses):
            object_config = dict(
                name=object,
                solution=object,
                xml_path=name2class_xml["tube"][-1],
                position=pos,
            )
            object_config["class"] = name2class_xml["tube"][0]
            init_container_config["subentities"].append(object_config)
            
            tag_config = dict(
                name=f"{object}_tag",
                content=object,
                xml_path=name2class_xml["nametag"][-1],
                position=[pos[0], pos[1]+0.05, 0.1],
            )
            tag_config["class"] = name2class_xml["nametag"][0]
            init_container_config["subentities"].append(tag_config)
        
    def get_condition_config(self, target_entity, init_container, **kwargs):
        conditions_config = dict(
            not_contain=dict(
                container=f"{init_container}",
                entities=[f"{target_entity}"]
            )
        )
        self.config["task"]["conditions"] = conditions_config
        return conditions_config
    
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Take out the {target_entity} solution"]
        self.config["task"]["instructions"] = instruction
        return instruction

@register.add_config_manager("select_chemistry_tube_common_sense")
class SelectChemistryTubeCommonSenseConfigManager(SelectChemistryTubeConfigManager):
        def load_objects(self, target_entity):
            """load objects without the name tag on"""
            objects = []
            objects.append(target_entity)
            other_objects = self.seen_object.copy() + self.unseen_object.copy()
            for similar_objects in self.seen_object + self.unseen_object:
                if isinstance(similar_objects, list) and target_entity in similar_objects:
                    other_objects.remove(similar_objects)
                elif isinstance(similar_objects, str) and target_entity == similar_objects:
                    other_objects.remove(similar_objects)
            other_objects_flatten = []
            for similar_objects in other_objects:
                other_objects_flatten.extend(similar_objects)
            objects.extend(random.sample(other_objects_flatten, self.num_object-1))
            targer_rol_poses = random.sample(relative_col_pos, self.num_object)
            target_poses = [[pos, random.choice(relative_row_pos), 0.05] for pos in targer_rol_poses] 
            
            init_container_config = self.config["task"]["components"][-1]
            init_container_config["subentities"] = []
            for object, pos in zip(objects, target_poses):
                object_config = dict(
                    name=object,
                    solution=object,
                    xml_path=name2class_xml["tube"][-1],
                    position=pos,
                )
                object_config["class"] = name2class_xml["tube"][0]
                init_container_config["subentities"].append(object_config)

@register.add_config_manager("select_chemistry_tube_spatial")
class SelectChemistryTubeSpatialConfigManager(SelectChemistryTubeConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=[4, 5],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        
    def load_objects(self, target_entity):
        objects = []
        objects.append(target_entity)
        other_objects = self.seen_object.copy() + self.unseen_object.copy()
        for similar_objects in self.seen_object + self.unseen_object:
            if isinstance(similar_objects, list) and target_entity in similar_objects:
                other_objects.remove(similar_objects)
            elif isinstance(similar_objects, str) and target_entity == similar_objects:
                other_objects.remove(similar_objects)
        other_objects_flatten = []
        for similar_objects in other_objects:
            other_objects_flatten.extend(similar_objects)
        objects.extend(random.sample(other_objects_flatten, self.num_object-1))
        positions = [[col, row, 0.05] for col in relative_col_pos for row in relative_row_pos]
        poses_to_load = random.sample(positions, self.num_object)
        
        init_container_config = self.config["task"]["components"][-1]
        init_container_config["subentities"] = []
        for object, pos in zip(objects, poses_to_load):
            if object == target_entity:
                n_row = relative_row_pos.index(pos[1])
                n_col = relative_col_pos.index(pos[0])
                self.target_row_col = (n_row, n_col)
            object_config = dict(
                name=object,
                solution=object,
                xml_path=name2class_xml["tube"][-1],
                position=pos,
            )
            object_config["class"] = name2class_xml["tube"][0]
            init_container_config["subentities"].append(object_config)
        
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Take out the solution on the {self.target_row_col[0]+1}th row and {self.target_row_col[1]+1}th column"]
        self.config["task"]["instructions"] = instruction
        
@register.add_config_manager("select_chemistry_tube_semantic")
class SelectChemistryTubeSemanticConfigManager(SelectChemistryTubeConfigManager):
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Take out the {target_entity} solution in a conversation"]
        self.config["task"]["instructions"] = instruction
        return instruction

@register.add_task("select_chemistry_tube")
class SelectChemistryTubeTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        """
        Attach the nametag to the tubestand.
        """
        super().build_from_config(eval, **kwargs)
        self.random_ignored_entities = ["table"]
        for key in list(self.entities.keys()):
            if "tag" in key:
                nametag = self.entities[key]
                if nametag.parent_entity is not None:
                    nametag.detach()
                    nametag.parent_entity.attach(nametag)
            if "tube_stand" in key:
                tube_stand = self.entities[key]
                tube_stand.detach()
                self._arena.attach(tube_stand)
    
    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity),
            partial(SkillLib.lift)
        ]
        return skill_sequence
            
@register.add_task("select_chemistry_tube_common_sense")
class SelectChemistryTubeCommonSenseTask(SelectChemistryTubeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_chemistry_tube_spatial")
class SelectChemistryTubeSpatialTask(SelectChemistryTubeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_chemistry_tube_semantic")
class SelectChemistryTubeSemanticTask(SelectChemistryTubeTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
    