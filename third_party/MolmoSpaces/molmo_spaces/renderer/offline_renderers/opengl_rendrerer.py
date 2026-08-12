import json
import os
from typing import Any, Literal, Queue

import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.env.mj_extensions import MjModelBindings
from molmo_spaces.env.vector_env import MuJoCoVectorEnv
from molmo_spaces.renderer.abstract_renderer import MjAbstractRenderer, MultithreadRenderer
from molmo_spaces.renderer.offline_renderers.domain_randomization import (
    BaseDomainRandomizationOfflineRenderer,
)
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer, MultithreadOpenGLRenderer


class MultithreadedDomainRandomizationOfflineOpenGLRenderer(
    MultithreadRenderer, BaseDomainRandomizationOfflineRenderer
):
    """
    For Depth and Segmentation rendering using OpenGL renderer
    For RGB rendering using Default OpenGL renderer
    """

    def __init__(
        self,
        env: "MuJoCoVectorEnv",
        device_id: int = 0,
        renderer_cls: type[MjAbstractRenderer] = MjOpenGLRenderer,
        max_render_contexts: int | None = None,
        namespace: str = "robot_0/",
        width: int = 1280,
        height: int = 720,
        randomize_lights: bool = True,
        randomize_shadows: bool = True,
        randomize_textures: bool = True,
        light_config_range: dict | None = None,
        shadow_config_range: dict | None = None,
        textures_pool: list | None = None,
        **kwargs: Any,
    ) -> None:
        self.multithreaded_render = MultithreadOpenGLRenderer(
            env=env,
            renderer_cls=renderer_cls,
            max_render_contexts=max_render_contexts,
            namespace=namespace,
            width=width,
            height=height,
            **kwargs,
        )
        self.randomize_lights = randomize_lights
        self.randomize_shadows = randomize_shadows
        self.randomize_textures = randomize_textures

        # Default configuration ranges.
        self.light_config_range = light_config_range or {
            "intensity": (0.5, 1.5),
            "color": [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0)],
        }
        self.shadow_config_range = shadow_config_range or {"shadow_softness": (0.0, 1.0)}
        self.textures_pool = textures_pool or ["texture1.png", "texture2.png", "texture3.png"]

    @classmethod
    def from_dataset(cls, data_path: str) -> None:
        pass

    def _load_episode_data(self, episode_path: str):
        assert os.path.exists(episode_path + "/state_data.npz"), (
            f"State path {episode_path + '/state_data.npz'} does not exist"
        )
        mj_state = np.load(episode_path + "/state_data.npz", allow_pickle=True)

        assert "qpos" in mj_state, "qpos not found in state data"
        assert "camera_names" in mj_state, "camera_names not found in state data"
        assert "camera_resolution" in mj_state, "camera_resolution not found in state data"
        return mj_state

    def _get_model_data_renderer(self, episode_path: str):
        with open(episode_path + "/task_metadata.json", "r") as f:
            metadata = json.load(f)
        scene_path = metadata["scene_path"]

        # TODO: create env from scene_path
        model_bindings = MjModelBindings.from_xml_path(scene_path)
        model = model_bindings.model
        data = MjData(model)

        # renderer
        renderer = MjOpenGLRenderer(model_bindings)

        return model, data, renderer

    def process_request(
        self,
        renderer: MjOpenGLRenderer,
        request: Any,
        output_queue: Queue,
        episode_path: str,
        # add_namespace: bool = True,
        **process_request_kwargs: Any,
    ) -> None:
        # load from episode path and metadata
        model, data, renderer = self._load_episode_data(episode_path)
        state_data = self._load_episode_data(episode_path)
        qpos = state_data["qpos"]
        # camera_names = state_data["camera_names"]
        # camera_resolution = state_data["camera_resolution"]

        # get from request
        idx, camera, data, mode = request

        # mode handling
        if mode == "rgb":
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
        elif mode == "depth" or mode == "pointcloud":
            renderer.enable_depth_rendering()
        elif mode == "segmentation":
            renderer.enable_segmentation_rendering()
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # render all frames
        num_frames = qpos.shape[0]
        observations = []
        for i in range(num_frames):
            data.qpos = qpos[i]
            mujoco.mj_forward(model, data)

            renderer.update(data, camera=camera)
            image = renderer.render(**process_request_kwargs)

            if mode == "pointcloud":
                # pointcloud = mujoco_depth_to_pointcloud(depth)
                raise NotImplementedError("Pointcloud rendering not implemented")
            else:
                observations.append(image)

        output_queue.put((idx, observations))

    def render(
        self,
        camera: str = "camera_rgb",
        mode: Literal["rgb", "depth", "segmentation"] = "rgb",
        add_namespace: bool = True,
    ) -> None:
        pass

    def _randomize_lights(self) -> None:
        """
        Randomize light parameters in the MuJoCo model.

        Assumes the model has an attribute `light_rgba` of shape (num_lights, 4)
        where columns 0-2 are the RGB channels and column 3 represents intensity.
        """
        if not hasattr(self.model, "light_rgba"):
            return
        num_lights = self.model.light_rgba.shape[0]
        intensity_min, intensity_max = self.light_config_range.get("intensity", (1.0, 1.0))
        for i in range(num_lights):
            new_intensity = np.random.uniform(intensity_min, intensity_max)
            if "color" in self.light_config_range and len(self.light_config_range["color"]) == 3:
                new_color = [np.random.uniform(*rng) for rng in self.light_config_range["color"]]
            else:
                new_color = self.model.light_rgba[i, :3]
            self.model.light_rgba[i, :3] = new_color
            self.model.light_rgba[i, 3] = new_intensity

    def _randomize_shadows(self) -> None:
        """
        Randomize shadow properties, if supported by the renderer.

        For example, if the renderer has a `shadow_softness` attribute,
        this method randomizes its value.
        """
        if hasattr(self.renderer, "shadow_softness"):
            shadow_min, shadow_max = self.shadow_config_range.get("shadow_softness", (0.0, 0.0))
            self.renderer.shadow_softness = np.random.uniform(shadow_min, shadow_max)

    def _randomize_textures(self) -> None:
        """
        Randomly choose a texture from the textures pool and apply it.

        Assumes that the renderer provides a method `set_texture`.
        """
        chosen_texture = np.random.choice(self.textures_pool)
        if hasattr(self.renderer, "set_texture"):
            self.renderer.set_texture(chosen_texture)
        # Otherwise, implement updating the model/material parameters accordingly.
