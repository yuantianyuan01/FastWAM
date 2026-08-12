"""
Register the interactive containers/recaptacles in daily life, such as electrical device
"""
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.components.container import CommonContainer, ContainerWithDoor

@register.add_entity("CoffeeMachine")
class CoffeeMachine(CommonContainer):
    """
    Coffee manchine that can be interactived by the user, press the button and the coffee fluid will be shown
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_pressed = False
    
    @property
    def fluid_sites(self):
        fluid_sites = []
        sites = self.mjcf_model.worldbody.find_all("site")
        for site in sites:
            if hasattr(site, "name") and ("fluid" in site.name or "liquid" in site.name):
                fluid_sites.append(site)
        return fluid_sites
    
    @property
    def start_button(self):
        return self.mjcf_model.worldbody.find("geom", "start_button")
        
    def get_start_button_pos(self, physics):
        return physics.bind(self.start_button).xpos
    
    def show_fluid(self, physics):
        for fluid_site in self.fluid_sites:
            physics.bind(fluid_site).rgba = np.concatenate([physics.bind(fluid_site).rgba[:3], [1]])
    
    def hidden_fluid(self, physics):
        for fluid_site in self.fluid_sites:
            physics.bind(fluid_site).rgba = np.concatenate([physics.bind(fluid_site).rgba[:3], [0]])   
    
    def is_activate(self, physics):
        contacts = physics.data.contact
        contact_goems = [contact.geom1 for contact in contacts] + [contact.geom2 for contact in contacts]
   
        if physics.bind(self.start_button).element_id in contact_goems:
            self._is_pressed = True
            return True
        else:
            self._is_pressed = False
            return False
    
    def after_substep(self, physics, random_state):
        if self.is_activate(physics): self.show_fluid(physics)
        else: self.hidden_fluid(physics)
    
    def initialize_episode(self, physics, random_state):
        if self.is_activate(physics): self.show_fluid(physics)
        else: self.hidden_fluid(physics)
        return super().initialize_episode(physics, random_state)
    
    def is_pressed(self):
        return self._is_pressed

@register.add_entity("Juicer")
class Juicer(CommonContainer):
    """
    Juicer that can be interactived by the user, press the button and the juicer will be activated
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_pressed = False
    
    @property
    def start_button(self):
        return self.mjcf_model.worldbody.find("geom", "start_button")
    
    def is_activate(self, physics):
        contacts = physics.data.contact
        contact_goems = [contact.geom1 for contact in contacts] + [contact.geom2 for contact in contacts]
   
        if physics.bind(self.start_button).element_id in contact_goems:
            self._is_pressed = True
            return True
        else:
            self._is_pressed = False
            return False
    
    def initialize_episode(self, physics, random_state):
        return super().initialize_episode(physics, random_state)
    
    def is_pressed(self):
        return self._is_pressed

@register.add_entity("Microwave")
class Microwave(ContainerWithDoor):
    def _build(self, 
               name:str="microwave",
               target_force_range=(1, 5),
               **kwargs):
        self._is_activated = False
        self._min_force, self._max_force = target_force_range
        super()._build(name=name, **kwargs)
    
    @property
    def start_button(self):
        return self.mjcf_model.worldbody.find("geom", "start_button")
        
    def get_start_button_pos(self, physics):
        return physics.bind(self.start_button).xpos
    
    def is_activate(self, physics):
        contacts = physics.data.contact
        contact_goems = [contact.geom1 for contact in contacts] + [contact.geom2 for contact in contacts]
   
        if physics.bind(self.start_button).element_id in contact_goems:
            self._is_pressed = True
            return True
        else:
            self._is_pressed = False
            return False
    
    def is_pressed(self):
        return self._is_pressed