"""
Register the entities with flexible textures used in the VLABench
"""
import os
import numpy as np
import random
from VLABench.tasks.components.entity import CommonGraspedEntity
from VLABench.tasks.components.container import FlatContainer
from VLABench.utils.register import register

@register.add_entity("BilliardBall")
class BilliardBall(CommonGraspedEntity):
    """
    Including billiard balls in snooker or classic eight-ball and  with different colors
    """
    def _build(self, value="white", **kwargs):
        super()._build(**kwargs)  
        self.value = value
        body = self.mjcf_model.worldbody.add("body", euler="0 1 0")
        body.add("geom", dclass="visual", mesh="ball_vis", material=value)
        body.add("geom", dclass="collision", mesh="ball_col")
    
    def save(self, physics):
        data_to_save = super().save(physics)
        data_to_save["value"] = self.value
        return data_to_save

@register.add_entity("ChemistryTube")
class ChemistryTube(CommonGraspedEntity):
    """
    Different chemistry solutions in the tube, some of them have diverse colors to distinguish their types and even concentrations
    """
    solution2rgba = {
        "CuCl2":[0.141000, 1.000000, 0.174043, 0.400000],
        "CuSO4":[0, 0.45, 1, 0.400000],
        "FeCl3":[0.6475, 0.5686, 0.023, 0.400000],
        "KMnO4":[0.5, 0, 0.5, 0.4],
        "I2":[0.3, 0.13, 0.0, 0.4],
        "K2CrO4":[0.57, 0.12, 0.013, 0.40000],
        "NaCl": [1, 1, 1, 0.3],
        "AgNO3": [1, 1, 1, 0.3],
        "BaCl2": [1, 1, 1, 0.3],
        "H2SO4": [1, 1, 1, 0.3],
        "NaOH": [1, 1, 1, 0.3],
        "Ba(NO3)2": [1, 1, 1, 0.3],
        "Pb(NO3)2": [1, 1, 1, 0.3],
        "Na2CO3": [1, 1, 1, 0.3],
        "CaCl2": [1, 1, 1, 0.3],
        "HCl": [1, 1, 1, 0.3],
        "CaSO4": [1, 1, 1, 0.7]
    }
    def __init__(self, solution, **kwargs):
        super().__init__(**kwargs)
        self.solution = solution    
        
    def change_texture(self, physics, texture_name):
        physics.bind(self.mjcf_model.worldbody.find("geom", "solution")).rgba = self.solution2rgba[self.solution]
    
    def get_solution(self):
        return self.solution
    
    def initialize_episode(self, physics, random_state):
        self.change_texture(physics, self.solution)
        return super().initialize_episode(physics, random_state)
    
    def save(self, physics):
        data_to_save = super().save(physics)
        data_to_save["solution"] = self.solution
        return data_to_save

@register.add_entity("NameTag")
class NameTag(CommonGraspedEntity):
    """
    Nametag to identify diffent entities with similar apperance such as tubes, codiment shakers.
    """
    def __init__(self, content, **kwargs):
        self.content = content
        super().__init__(**kwargs)
        
    def _build(self, **kwargs):
        super()._build(**kwargs)
        self.body = self.mjcf_model.worldbody.add("body", name="tag", euler=[0, np.pi/2, np.pi/2])
        self.body.add("geom", dclass="visual", type="mesh", mesh="tag", material=self.content)
        self.body.add("geom", dclass="collision", type="mesh", mesh="tag")
    
    def initialize_episode(self, physics, random_state):
        return super().initialize_episode(physics, random_state)
    
    def save(self, physics):
        data_to_save = super().save(physics)
        data_to_save["content"] = self.content
        return data_to_save

@register.add_entity("Mahjong")
class Mahjong(CommonGraspedEntity):
    suites = ["man", "pin", "sou"]
    values = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    def _build(self, value=None, suite=None, **kwargs):
        super()._build(**kwargs)
        self.mahjongs = [f"{value}_{suite}" for value in self.values for suite in self.suites]
        self.mahjongs *= 4
        
        if value is None or suite is None:
            mahjong = random.choice(self.mahjongs)
            str_split = mahjong.split("_")
            value, suite = str_split[0], str_split[-1]
        self.value, self.suite = value, suite
        material = f"{value}_{suite}"
        
        body = self.mjcf_model.worldbody.find("body", "mahjong")
        body.add("geom", dclass="visual", mesh="majong_2", material=material)
        body.add("geom", dclass="collision", mesh="majong_2")
    
    def save(self, physics):
        data_to_save = super().save(physics)
        data_to_save["value"] = self.value
        data_to_save["suite"] = self.suite
        return data_to_save
    
@register.add_entity("NumberCube")
class NumberCube(CommonGraspedEntity):
    digit2str = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    def _build(self, number="zero", **kwargs):
        super()._build(**kwargs)
        if isinstance(number, int) or number.isdigit():
            self.number = self.digit2str[str(number)]
        else:
            self.number = number
        body = self.mjcf_model.worldbody.find("body", "body")
        body.add("geom", mesh="cube", dclass="visual", material=self.number)
        body.add("geom", mesh="cube", dclass="collision")
    
    def save(self, physics):
        data_to_save = super().save(physics)
        data_to_save["number"] = self.number
        return data_to_save

@register.add_entity("Painting")
class Painting(FlatContainer, CommonGraspedEntity):
    """
    Painting is a special class in VLABench. 
    On the one hand, it is a container to place something; 
    on the other hand, it is a specific entity to grasp in tasks such as 'hang the picture'.
    """
    styles = ['Romanticism', 'Realism', 'Rococo', 'Baroque', 'Symbolism', 'Minimalism', 'Post-Impressionism', 'Surrealism', 'Ukiyo-e', 'Academicism', 'Expressionism', 'Impressionism', 'Neoclassicism', 'Art Nouveau', 'Cubism']
    def _build(self, 
               style=None,
               specific_painting=None,
               **kwargs):
        self.painting_root = os.path.join(self.entity_asset_root, "paintings/assets/famous paintings")        
        if style is not None and specific_painting is None: # style
            style = style if style is not None else random.choice(self.styles) 
            texture_file = random.choice(os.listdir(os.path.join(self.painting_root, style)))
            specific_painting = texture_file.split(".png")[0]
            if kwargs.get("name", None) is None:
                kwargs["name"] = f"{style.lower()}_painting"
        elif specific_painting is not None: # specific painting
            subdirs = os.listdir(self.painting_root)
            for subdir in subdirs:
                files = os.listdir(os.path.join(self.painting_root, subdir))
                if specific_painting + ".png" in files:
                    texture_file = specific_painting + ".png"
                    style = subdir
                    break
        self.style = style
        self.specific_painting = specific_painting
        super()._build(**kwargs) 
        self.mjcf_model.asset.add("texture", name=f"{style.lower()}", file=os.path.join(self.painting_root, style, texture_file), type="2d")    
        self.mjcf_model.asset.add("material", name=f"{style.lower()}", texture=f"{style.lower()}")
        target_body = self.mjcf_model.worldbody.find("body", "picture_frame")   
        target_body.add("geom", type="mesh", mesh="picture_frame_0", dclass="visual", material=style.lower())
        target_body.add("geom", type="mesh", mesh="picture_frame_0", dclass="collision")

    def get_grasped_keypoints(self, physics):
        """
        return a valid grasp position and quaternion
        """ 
        sites = self.mjcf_model.worldbody.find_all("site")
        grasp_keypoints = []
        for site in sites:
            if physics.bind(site).group == 4:
                grasppoint = physics.bind(site).xpos
                grasp_keypoints.append(grasppoint)
        if len(grasp_keypoints) == 0:
            grasp_keypoints.append(self.get_xpos(physics)) 
        return grasp_keypoints

    def save(self, physics):
        data_to_save = super().save(physics)
        data_to_save["style"] = self.style
        data_to_save["specific_painting"] = self.specific_painting
        return data_to_save

@register.add_entity("Poker")
class Poker(CommonGraspedEntity):
    suites = ["spades", "clubs", "diamonds", "hearts"]
    values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace"]
    
    def _build(self, 
               position, 
               orientation, 
               value=None, 
               suite=None,
               **kwargs):

        self.cards = [f"{value}_of_{suite}" for value in self.values for suite in self.suites]
        self.cards.extend(["black_joker", "red_joker"])
        kwargs["position"], kwargs["orientation"] = position, orientation
        super()._build(**kwargs)        
        self.build_single_card(position, orientation, value, suite)

            
    def build_single_card(self, pos, ori, value, suite):
        if value is None or suite is None:
            card = random.choice(self.cards)
            str_split = card.split("_")
            value, suite = str_split[0], str_split[-1]
        self.value, self.suite = value, suite
        if suite == "joker":
            material = f"{value}_{suite}"
        else:
            material = f"{value}_of_{suite}"
        body = self.mjcf_model.worldbody.add("body")     
        body.add("geom", dclass="card", material=material, name="visual_geom")
        body.add("geom", dclass="collision", name="collision_geom")
        body.add("site", dclass="grasppoint", pos="-0.02 0 0")
    
    def save(self, physics):
        data_to_save = super().save(physics)
        data_to_save["value"] = self.value
        data_to_save["suite"] = self.suite
        return data_to_save