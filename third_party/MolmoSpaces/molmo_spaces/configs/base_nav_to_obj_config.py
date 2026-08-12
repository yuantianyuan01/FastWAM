"""
Example configuration for RBY1 navigation to object data generation using the extracted task sampler.
This shows how the scene randomization functionality from the reference script has been
properly integrated into the modular task sampler architecture.
"""

from __future__ import annotations

import numpy as np

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import RBY1MjcfCameraSystem
from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig, BasePolicyConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig, RBY1Config
from molmo_spaces.configs.task_configs import NavToObjTaskConfig
from molmo_spaces.configs.task_sampler_configs import NavToObjTaskSamplerConfig
from molmo_spaces.tasks.nav_task import NavToObjTask
from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler
from molmo_spaces.utils.profiler_utils import Profiler


class NavToObjBaseConfig(MlSpacesExpConfig):
    """Base configuration for navigation to object data generation tasks."""

    # NOTE: will not work if used directly. Subclass examples in data_generation/configs.py

    # --- Experiment-level config parameters ---
    num_envs: int = 1  # Number of environments to run in each thread
    use_passive_viewer: bool = False  # Launch passive viewer for rendering
    viewer_camera: None = None
    viewer_cam_dict: dict = {
        "distance": 5.0,
        "azimuth": 45.0,
        "elevation": -30.0,
        "lookat": np.array([0.0, 0.0, 0.5]),
    }
    policy_dt_ms: float = 200.0  # policy time step
    ctrl_dt_ms: float = 2.0  # control time step
    sim_dt_ms: float = 2.0  # simulation time step
    task_horizon: int = 500  # Maximum steps per episode to prevent infinite runs
    record_videos: bool = False  # Whether to record videos of episodes

    # --- Data generation settings ---
    num_threads: int = 1  # parallel data generation threads
    profile: bool = True  # Whether to profile the data generation pipeline
    profiler: Profiler | None = None
    output_dir: str | None = None  # Directory to save generated data
    use_wandb: bool = False  # Whether to use Weights & Biases logging
    wandb_name: str | None = None  # Weights & Biases run name
    wandb_project: str | None = None  # Weights & Biases project name

    # --- Task type configuration ---
    task_type: str = "nav_to_obj"  # Task type: nav_to_obj

    # --- ProcTHOR dataset configuration ---
    scene_dataset: str = "procthor-10k"  # Name of the scene dataset to load
    data_split: str = "train"  # Data split to use

    robot_config: BaseRobotConfig | None = None

    # Camera configuration - using new unified camera system
    camera_config: RBY1MjcfCameraSystem = RBY1MjcfCameraSystem()

    # Task sampler configuration (imported from task_sampler_configs.py)
    task_sampler_config: NavToObjTaskSamplerConfig = NavToObjTaskSamplerConfig(
        task_sampler_class=NavToObjTaskSampler
    )

    # Task configuration (imported from task_configs.py)
    task_config: NavToObjTaskConfig = NavToObjTaskConfig(task_cls=NavToObjTask)
    task_config_preset: NavToObjTaskConfig | None = None

    # Policy configuration (imported from policy_configs.py)
    policy_config: BasePolicyConfig = AStarNavToObjPolicyConfig()

    def _init_policy_config(self) -> BasePolicyConfig:
        """Initialize policy config. Override in subclasses for dynamic initialization."""
        return self.policy_config

    def model_post_init(self, __context) -> None:
        """Initialize and validate configuration after Pydantic model initialization"""
        super().model_post_init(__context)

        try:
            self.policy_config = self._init_policy_config()
        except RuntimeError as e:
            # Check if this is a CUDA/GPU-related error
            error_msg = str(e)
            if "NVIDIA" in error_msg or "CUDA" in error_msg or "GPU" in error_msg:
                # No GPU available - this is expected on manager nodes that just coordinate jobs
                # Policy config will be initialized later on worker nodes that have GPUs
                print(
                    f"Warning: Skipping policy config initialization due to missing GPU: {error_msg}"
                )
                self.policy_config = None
            else:
                raise

        # Auto-create profiler instance if profiling is enabled
        if self.profile and self.profiler is None:
            self.profiler = Profiler()

    @property
    def tag(self) -> str:
        return "nav_to_obj_datagen"

    class SavedEpisode(Config):
        camera_config: RBY1MjcfCameraSystem | None = None  # Configuration for cameras and sensors
        robot_config: RBY1Config | None = None  # Configuration for the robot
        task_config: NavToObjTaskConfig | None = None  # Configuration for tasks
        task_cls_str: str | None = None
