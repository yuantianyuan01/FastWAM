import abc
import json
import os
from typing import Any

from mujoco import MjData

from molmo_spaces.env.mj_extensions import MjModelBindings
from molmo_spaces.renderer.abstract_renderer import MjAbstractRenderer


class BaseDomainRandomizationOfflineRenderer(MjAbstractRenderer, abc.ABC):
    """
    Abstract base class for offline renderers that support domain randomization.

    This class performs the following:
      1. Loads an episode's metadata (e.g., the scene XML path) to create MuJoCo model bindings.
      2. Initializes the MuJoCo data for simulation.
      3. Accepts an externally provided renderer (e.g., OpenGL, Omniverse, Madrona).

    Subclasses must implement:
      - randomize(): to modify the model/renderer properties (lights, shadows, textures, etc.)

    This implementation applies random modifications to the scene:
      - Lights: Randomizes intensity and color.
      - Shadows: Randomizes (if supported by the renderer) a shadow softness parameter.
      - Textures: Chooses a texture from a provided pool.
    """

    def __init__(
        self, episode_path: str, renderer: Any, device_id: int | None = None, **kwargs: Any
    ) -> None:
        """
        Args:
            episode_path (str): Path to the episode folder (should contain task_metadata.json and state_data.npz).
            renderer (Any): Instance of the rendering backend (OpenGL, Omniverse, Madrona, etc.).
            device_id (Optional[int]): Device identifier (e.g., GPU id) if applicable.
            kwargs: Any additional kwargs for the MjAbstractRenderer initialization.
        """
        self.episode_path = episode_path

        # Load metadata (for example, the scene XML location)
        metadata_file = os.path.join(episode_path, "task_metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        scene_path = metadata["scene_path"]

        # Create model bindings from the scene XML.
        model_bindings = MjModelBindings.from_xml_path(scene_path)
        super().__init__(model_bindings, device_id=device_id, **kwargs)

        # Create the MuJoCo simulation data.
        self.data = MjData(self.model)

        # Use the externally provided renderer.
        self.renderer = renderer

    def randomize(self) -> None:
        """
        Apply domain randomizations to the scene.

        This method should perform modifications such as altering
        light intensities/colors, shadow properties, textures, etc.
        """
        if self.randomize_lights:
            self._randomize_lights()
        if self.randomize_shadows:
            self._randomize_shadows()
        if self.randomize_textures:
            self._randomize_textures()

    @abc.abstractmethod
    def _randomize_lights(self):
        """
        Randomize light properties in the simulation model.

        This method is intended to be overridden by subclasses to apply random modifications
        to light-related attributes in the simulation. Typical modifications include adjusting
        light intensity and RGB color values to simulate various lighting scenarios.

        Subclasses should implement this method based on the specific details of their MuJoCo
        model and rendering backend.
        """
        pass

    @abc.abstractmethod
    def _randomize_shadows(self):
        """
        Randomize shadow properties in the simulation.

        This method should be overridden by subclasses to modify shadow-related attributes,
        such as softness or intensity, in order to simulate different environmental lighting
        conditions. The exact properties to randomize depend on the renderer's capabilities and
        the simulation model's configuration.

        Override this method to implement custom shadow randomization logic.
        """
        pass

    @abc.abstractmethod
    def _randomize_textures(self):
        """
        Randomize texture properties for materials in the simulation.

        This method is intended to be overridden by subclasses to apply random alterations
        to textures used in the simulated environment. Randomization can involve selecting
        from a pool of predefined texture files or dynamically modifying texture parameters
        to diversify the scene's appearance.

        Subclasses should provide specific logic depending on how textures are managed in
        their rendering pipeline.
        """
        pass
