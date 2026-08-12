import logging
from collections.abc import Callable

from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import MjcfCameraConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaCAPRobotConfig
from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

log = logging.getLogger(__name__)

OverrideFn = Callable[[EpisodeSpec, MlSpacesExpConfig], None]


def cap_robot_eval_override(
    episode_spec: EpisodeSpec,
    exp_config: MlSpacesExpConfig,
) -> None:
    log.info("Applying CAP robot evaluation overrides")

    camera_config = exp_config.camera_config

    camera_config.cameras[0] = MjcfCameraConfig(
        name="wrist_camera",
        mjcf_name="wrist_camera",
        robot_namespace="robot_0/gripper/",
        fov=53.0,
        fov_noise_degrees=(0.0, 0.0),
        pos_noise_range=(0.0, 0.0),
        orientation_noise_degrees=0.0,
        record_depth=True,
    )

    camera_config.cameras[1].record_depth = True
    camera_config.cameras[1].fov = 71

    rot_base = R.from_quat(episode_spec.task["robot_base_pose"][3:7], scalar_first=True).as_matrix()
    episode_spec.task["robot_base_pose"][:3] += 0.05 * rot_base[0:3, 0]
    episode_spec.task["robot_base_pose"][2] -= 0.2

    camera_config.img_resolution = (960, 720)

    episode_spec.robot.init_qpos = {
        "base": [],
        "arm": [[0, -1.5, 0.116, -2.45, 0, 0.842, 0.965]],
        "gripper": [0.00296, 0.00296],
    }


ROBOT_OVERRIDE_REGISTRY: dict[type[BaseRobotConfig], OverrideFn] = {
    FrankaCAPRobotConfig: cap_robot_eval_override,
}


def register_robot_override(robot_config_cls: type[BaseRobotConfig], override_fn: OverrideFn):
    """
    Register a robot override for a given robot config class.

    Args:
        robot_config_cls: The robot config class to register the override for.
        override_fn: The override function to register.
    """
    if robot_config_cls in ROBOT_OVERRIDE_REGISTRY:
        raise ValueError(f"Robot override already registered for {robot_config_cls.__name__}")
    ROBOT_OVERRIDE_REGISTRY[robot_config_cls] = override_fn


def get_robot_override(robot_config: BaseRobotConfig) -> OverrideFn | None:
    """
    Get the robot override for a given robot config. This handles inheritance,
    so if the robot config is a subclass of a registered override, the registered
    override will be returned.

    Args:
        robot_config: The robot config to get the override for.

    Returns:
        The robot override function, or None if no override is found.
    """
    robot_class = type(robot_config)

    # Traverse the MRO to find the first override, handles inheritance
    for cls in robot_class.mro():
        if cls in ROBOT_OVERRIDE_REGISTRY:
            log.info(f"Found robot override for {robot_class.__name__} ({cls.__name__})")
            return ROBOT_OVERRIDE_REGISTRY[cls]

    return None
