from VLABench.tasks.config_manager import PressButtonConfigManager
from VLABench.tasks.components import Button, RandomGeom
from VLABench.tasks.hierarchical_tasks.primitive.base import PressButtonTask, PrimitiveTask

class PhysicalQAConfigManager(PressButtonConfigManager):
    def __init__(self, 
                 task_name, 
                 num_objects=3, 
                 seen_material=["wood", "metal", "stone"],
                 unseen_material=["rubber", "glass", "metal"],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.seen_material = seen_material
        self.unseen_material = unseen_material
    
    def get_seen_task_config(self):
        return self.get_task_config(self.seen_material)
        
    def get_unseen_task_config(self):
        return self.get_task_config(self.unseen_material)
        
    def get_task_config(self, materials, **kwargs):
        self.target_entity, self.target_container, self.init_container = None, None, None 
        self.load_buttons()
        self.load_objects(materials)
        self.get_instruction()
        self.get_condition_config(self.target_button)
        return self.config