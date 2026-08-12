from VLABench.tasks.components.entity import Entity
from VLABench.utils.register import register

@register.add_entity("Button")
class Button(Entity):
    def _build(self, 
               target_force_range=(0.01, 100),
               size=[0.04, 0.02], # radius, half height
               rgba=[1, 0, 0, 1], # default color
               **kwargs):
        super()._build(**kwargs)
        self._min_force, self._max_force = target_force_range
        self._geom = self._mjcf_model.worldbody.add(
            "geom", type="cylinder", size=size, rgba=rgba, mass=0.01, solref=[0.001, 2])
        self._site = self._mjcf_model.worldbody.add(
            "site", type="cylinder", size=self._geom.size*1.01, rgba=rgba
        )
        self._sensor = self._mjcf_model.sensor.add('touch', site=self._site)
    
    @property
    def mjcf_model(self):
        return self._mjcf_model
    
    @property
    def num_activated_steps(self):
        return self._num_activated_steps
    
    @property
    def touch_sensor(self):
        return self._sensor
    
    def _update_activation(self, physics):
        current_force = physics.bind(self.touch_sensor).sensordata[0]
        self._is_activated = (current_force >= self._min_force and
                            current_force <= self._max_force)
        if not self._is_activated: # if not activated, reset the num_activated_steps
            self._num_activated_steps = 0
        physics.bind(self._geom).rgba = (
            [0, 1, 0, 1] if self._is_activated else [1, 0, 0, 1])
        self._num_activated_steps += int(self._is_activated)

    def initialize_episode(self, physics, random_state):
        self._reward = 0.0
        self._num_activated_steps = 0
        self._update_activation(physics)
        return super().initialize_episode(physics, random_state)

    def after_substep(self, physics, random_state):
        self._update_activation(physics)
        
    def is_pressed(self):
        return self._is_activated and self.num_activated_steps > 10
        
        