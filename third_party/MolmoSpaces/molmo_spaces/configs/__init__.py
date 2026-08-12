"""Configuration module for MolmoSpaces experiments.

This module provides configuration classes organized by category:
- abstract_config: Base Config class
- abstract_exp_config: Base experiment configuration
- camera_configs: Camera-related configurations
- robot_configs: Robot-related configurations
- task_configs: Task-related configurations
- task_sampler_configs: Task sampler-related configurations
- policy_configs: Policy-related configurations
"""

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import (
    CameraConfig,
    CameraSystemConfig,
    FixedExocentricCameraConfig,
    FrankaDroidCameraSystem,
    FrankaRandomizedD405D455CameraSystem,
    MjcfCameraConfig,
    RandomizedExocentricCameraConfig,
    RBY1GoProD455CameraSystem,
    RBY1MjcfCameraSystem,
    RobotMountedCameraConfig,
)
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaRobotConfig
from molmo_spaces.configs.task_configs import BaseMujocoTaskConfig, PickTaskConfig
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
    PickTaskSamplerConfig,
)

__all__ = [
    "Config",
    "MlSpacesExpConfig",
    # Camera configs - new unified system
    "CameraSystemConfig",
    "CameraConfig",
    "MjcfCameraConfig",
    "RobotMountedCameraConfig",
    "FixedExocentricCameraConfig",
    "RandomizedExocentricCameraConfig",
    "RBY1MjcfCameraSystem",
    "RBY1GoProD455CameraSystem",
    "FrankaRandomizedD405D455CameraSystem",
    "FrankaDroidCameraSystem",
    # Robot configs
    "BaseRobotConfig",
    "FrankaRobotConfig",
    # Task configs
    "BaseMujocoTaskConfig",
    "PickTaskConfig",
    # Task sampler configs
    "BaseMujocoTaskSamplerConfig",
    "PickTaskSamplerConfig",
    # Policy configs
    "BasePolicyConfig",
    "ObjectManipulationPlannerPolicyConfig",
]
