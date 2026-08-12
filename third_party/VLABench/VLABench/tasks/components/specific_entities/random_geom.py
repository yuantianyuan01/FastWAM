import random
import os
from VLABench.tasks.components.entity import Entity, CommonGraspedEntity
from VLABench.utils.register import register

geom_types = ["box", "sphere"]
materials = ["wood", "metal", "rubber", "glass", "stone"]
material2texture = {
    "metal": ["metal0", "metal1", "metal2", "metal3", "metal4"],
    "wood": ["wood0", "wood1", "wood2", "wood3", "wood4"],
    "stone": ["stone0", "stone", "stone2", "stone3", "stone4"],
    "glass": ["glass"],
    "rubber": ["rubber"]
}

@register.add_entity("RandomGeom")
class RandomGeom(CommonGraspedEntity):
    def __init__(self, 
                 geom_type="box",
                 size=None,
                 material=None,
                 **kwargs):
        # initialize the geom_type and material
        if geom_type is None or geom_type not in geom_types:
            geom_type = random.choice(geom_types)
        self.geom_type = geom_type
        
        if material is None or material not in materials:
            material = random.choice(materials)
        self.material = material
        if kwargs.get("texture", None) is None:
            self.texture = random.choice(material2texture[self.material])
        else:
            self.texture = kwargs.get("texture")
        self.size = size
        super().__init__(**kwargs)
    
    def _build(self, **kwargs):
        super()._build(**kwargs)
        if self.size is None:
            if self.geom_type == "box":
                l = random.uniform(0.02, 0.03)
                size = [l, l, l]
            elif self.geom_type == "sphere":
                size = [random.uniform(0.02, 0.03)]
        else:
            size = self.size if self.size is not None else [random.uniform(0.02, 0.03)]
        
        if self.material == "metal": density = random.uniform(5000, 8000)
        elif self.material == "wood": density = random.uniform(400, 900)
        elif self.material == "glass": density = random.uniform(2200, 2600)
        elif self.material == "rubber": density = random.uniform(910, 930)
        elif self.material == "stone": density = random.uniform(2300, 2900)
        
        self.mjcf_model.asset.add("texture", name=f"{self.material}", type="cube", file=os.path.join(self.texture_root, self.texture + ".png"))
        self.mjcf_model.asset.add("material", name=f"{self.material}", texture=f"{self.material}")
        self.mjcf_model.worldbody.add("geom", type=f"{self.geom_type}", material=f"{self.material}", size=size, density=density, contype="0", conaffinity="0")
        self.mjcf_model.worldbody.add("geom", type=f"{self.geom_type}", material=f"{self.material}", size=size, solref="0.001 1")
        
    def save(self, save_path):
        data_to_save = super().save(save_path)
        data_to_save["material"] = self.material
        data_to_save["texture"] = self.texture
        return data_to_save
    

        