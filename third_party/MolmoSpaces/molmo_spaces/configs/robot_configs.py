"""Robot configuration classes for MolmoSpaces experiments.

This module contains:
- ActionNoiseConfig: TCP-bounded noise configuration for arm actions
- BaseRobotConfig: Base configuration for all robots
- Robot-specific configs: FrankaRobotConfig, RBY1Config, FloatingRUMRobotConfig
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from mujoco import MjData

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.molmo_spaces_constants import get_robot_path
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.bimanual_yam import BimanualYamRobot
from molmo_spaces.robots.floating_robotiq import FloatingRobotiqRobot
from molmo_spaces.robots.floating_rum import FloatingRUMRobot
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.robots.i2rt_yam import I2rtYamRobot
from molmo_spaces.robots.mobile_franka import MobileFrankaRobot
from molmo_spaces.robots.rby1 import RBY1
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory
from molmo_spaces.robots.robot_views.bimanual_yam_view import BimanualYamRobotView
from molmo_spaces.robots.robot_views.franka_cap_view import (
    FrankaCAPRobotView,
)
from molmo_spaces.robots.robot_views.franka_droid_view import (
    FloatingRobotiq2f85RobotView,
    FrankaDroidRobotView,
)
from molmo_spaces.robots.robot_views.i2rt_yam_view import I2rtYamRobotView
from molmo_spaces.robots.robot_views.mobile_franka_droid_view import MobileFrankaDroidRobotView
from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView
from molmo_spaces.robots.robot_views.rum_gripper_view import FloatingRUMRobotView


class ActionNoiseConfig(Config):
    """Configuration for action noise injection.

    This noise model supports:
    - Arm noise: TCP-bounded noise that maps through Jacobian to joint space
    - Base noise: Planar noise applied directly to (x, y, theta) commands

    Noise is proportional to the commanded action magnitude:
        noise_std = action_scale_factor * ||delta||

    When the commanded delta is zero, no noise is applied.
    """

    enabled: bool = True  # Whether to apply action noise

    # === Arm noise configuration (TCP-bounded) ===

    # Scale factor for arm noise proportional to TCP delta magnitude
    # noise_std = action_scale_factor * ||tcp_delta||
    # e.g., action_scale_factor=0.1 means noise std is 10% of commanded TCP delta
    action_scale_factor: float = 0.1

    # Rotation noise scale relative to position noise
    rotation_noise_scale: float = 0.1

    # Maximum noise magnitude in TCP space (clipped to this bound)
    max_tcp_position_noise: float = 0.02  # 2cm max position noise
    max_tcp_rotation_noise: float = 0.1  # ~5.7 degrees max rotation noise

    # === Base noise configuration (planar) ===

    # Scale factor for base noise proportional to commanded displacement magnitude
    # position_noise_std = base_action_scale_factor * ||position_delta||
    # rotation_noise_std = base_action_scale_factor * |rotation_delta|
    base_action_scale_factor: float = 0.1

    # Maximum base noise magnitude (clipped to this bound)
    max_base_position_noise: float = 0.02  # 2cm max
    max_base_rotation_noise: float = 0.05  # ~2.8 degrees max


class BaseRobotConfig(Config):
    """Base configuration for robot setup."""

    robot_cls: type[Robot] | None
    robot_factory: (
        Callable[[MjData, Any], Robot] | None
    )  # (MjData, MlSpacesExpConfig) -> Robot. here (and subclasses) we use Any to avoid annotation dependency on MlSpacesExpConfig
    robot_view_factory: RobotViewFactory | None
    robot_namespace: (
        str  # namespace used to differentiate between one or multiple robots and the environment
    )
    command_mode: dict[
        str, str
    ]  # move_group to command_mode e.g., "joint", "cartesian", "velocity"
    init_qpos: dict[str, list[float]]
    init_qpos_noise_range: dict[str, list[float]] | None
    name: str | None
    robot_xml_path: Path  # path to the robot XML file within the robot directory
    robot_dir: Path | None = (
        None  # path to the robot directory, if not using a prepackaged MlSpaces robot
    )

    # configurable control parameters for low-level mujoco controllers
    gravcomp: bool = False  # apply gravity compensation to every body in the robot
    K_stiffness: list[float] | None = None  # if None use values from model
    K_damping: list[float] | None = None  # if None use values from model
    force_limit: list[float] | None = (
        None  # Limit actuator-applied generalized force magnitude, if None use values from model
    )

    # Action noise configuration - applied per-robot in Robot.apply_action_noise()
    action_noise_config: ActionNoiseConfig | None = None

    def model_post_init(self, _context):
        """Ensure action_noise_config is always initialized, even when loading from old configs."""
        if self.action_noise_config is None:
            object.__setattr__(self, "action_noise_config", ActionNoiseConfig())

    def get_robot_dir(self) -> Path:
        """
        Get the path to the robot directory, which may or may not be a prepackaged MlSpaces robot.
        """
        if self.robot_dir is not None:
            return self.robot_dir
        return get_robot_path(self.name)

    def get_robot_xml_path(self) -> Path:
        """
        Get the full path to the robot XML file.
        """
        return self.get_robot_dir() / self.robot_xml_path


# Concrete robot configurations


class FrankaRobotConfig(BaseRobotConfig):
    """Configuration for Franka FR3 robot."""

    robot_cls: type[FrankaRobot] | None = FrankaRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = FrankaRobot
    robot_namespace: str = "robot_0/"
    robot_view_factory: RobotViewFactory | None = FrankaDroidRobotView
    name: str = "franka_droid"
    robot_xml_path: Path = Path("model.xml")
    base_size: list[float] | None = [0.5, 0.5, 0.58]
    init_qpos: dict[str, list[float]] = {
        "arm": [0, -0.7853, 0, -2.35619, 0, 1.57079, 0.0],
        "gripper": [0.00296, 0.00296],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = {
        # selected to allow for more displacement in later joints and keep TCP displacement <=10cm
        # joint_weights = [1, ..., 7] (allow more movement in later joints)
        # J_p is 3x7 Jacobian of TCP position wrt arm joints
        # dq = joint_weights * 0.1 / ||J_p @ joint_weights||
        "arm": [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
    }
    command_mode: dict[str, str | None] = {
        "arm": "joint_position",  # e.g., "joint_position", "joint_velocity", "ee_position", "ee_velocity"
        "gripper": "joint_position",
    }
    gravcomp: bool = True
    # texture randomization parameters, ignored if texture randomization is disabled
    perturb_texture_probability: float = 0.7

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ["joint_position", "joint_rel_position"]


class MobileFrankaRobotConfig(BaseRobotConfig):
    robot_cls: type[MobileFrankaRobot] | None = MobileFrankaRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = MobileFrankaRobot
    robot_namespace: str = "robot_0/"
    robot_view_factory: RobotViewFactory | None = MobileFrankaDroidRobotView
    name: str = "franka_droid"
    robot_xml_path: Path = Path("model.xml")
    base_size: list[float] = [0.5, 0.5, 0.58]
    init_qpos: dict[str, list[float]] = {
        "base": [0, 0, 0],
        "arm": [0, -0.7853, 0, -2.35619, 0, 1.57079, 0.0],
        "gripper": [0.00296, 0.00296],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = {
        "arm": [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
    }
    command_mode: dict[str, str | None] = {
        "base": "holo_joint_planar_position",
        "arm": "joint_position",
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    base_control_params: dict[str, dict[str, float]] = {
        "base_x_act": {
            "kp": 25000,
            "damping_ratio": 1.0,
            "ctrlrange": 25,
        },
        "base_y_act": {
            "kp": 25000,
            "damping_ratio": 1.0,
            "ctrlrange": 25,
        },
        "base_theta_act": {
            "kp": 5000,
            "damping_ratio": 1.0,
        },
    }


class FrankaCAPRobotConfig(BaseRobotConfig):
    """Configuration for Franka FR3 robot."""

    robot_cls: type[FrankaRobot] | None = FrankaRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = FrankaRobot
    robot_namespace: str = "robot_0/"
    robot_view_factory: RobotViewFactory | None = FrankaCAPRobotView
    name: str = "franka_cap"
    robot_xml_path: Path = Path("model.xml")
    base_size: list[float] | None = [0.5, 0.5, 0.58]
    init_qpos: dict[str, list[float]] = {
        "arm": [0, -1.5, 0.116, -2.45, 0, 0.842, 0.965],
        "gripper": [0.00296, 0.00296],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = {
        # selected to allow for more displacement in later joints and keep TCP displacement <=10cm
        # joint_weights = [1, ..., 7] (allow more movement in later joints)
        # J_p is 3x7 Jacobian of TCP position wrt arm joints
        # dq = joint_weights * 0.1 / ||J_p @ joint_weights||
        "arm": [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
    }
    command_mode: dict[str, str | None] = {
        "arm": "joint_position",  # e.g., "joint_position", "joint_velocity", "ee_position", "ee_velocity"
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ["joint_position", "joint_rel_position"]


class RBY1Config(BaseRobotConfig):
    """Configuration for RBY1 robot."""

    robot_cls: type[RBY1] = RBY1
    robot_factory: Callable[[MjData, Any], Robot] | None = RBY1
    robot_view_factory: RobotViewFactory | None = None  # set in model_post_init
    robot_namespace: str = "robot_0/"
    init_qpos: dict[str, np.ndarray] = {
        "base": np.array([0.0, 0.0, 0.0]),  # x, y, theta
        "head": np.array(
            [0.0, 0.6]
        ),  # (pan, tilt) - 0 pan = forward, ~0.4 rad tilt = looking down ~34 degrees
        "left_arm": np.array([0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0]),
        "left_gripper": np.array([-0.05]),  # Open position - coupling handled in RBY1GripperGroup
        "right_arm": np.array([0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0]),
        "right_gripper": np.array([-0.05]),  # Open position - coupling handled in RBY1GripperGroup
        "torso": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    # TODO: Add noise ranges for arms etc
    init_qpos_noise_range: dict[str, np.ndarray] = {
        "base": np.array([0.0, 0.0, 0.0]),
        # "head": np.array([0.15, 0.1]),  # (pan, tilt) noise in radians (~8.5 deg, ~5.7 deg)
        "head": np.array([0.2, 0.2]),  # (pan, tilt) noise in radians (~11.4 deg, ~11.4 deg)
        "left_arm": np.array(
            [0.05, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
        ),  # Graduated noise: more distal = more variation
        "left_gripper": np.array([0.01]),
        "right_arm": np.array(
            [0.05, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
        ),  # Graduated noise: more distal = more variation
        "right_gripper": np.array([0.01]),
        "torso": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }

    use_holo_base: bool = True  # Whether to use virtual holonomic base joints or not
    command_mode: dict[str, str | None] = {
        "arm": "joint_position",  # e.g., "joint_position", "joint_velocity", "ee_position", "ee_velocity"
        "gripper": "joint_position",
        "base": "holo_joint_planar_position",  # e.g., "planar_position", "planar_velocity", "wheel_velocity"
        "head": None,  # Must be None - RBY1 head actuation is disabled
    }
    name: str = "rby1"
    robot_xml_path: Path = Path("rby1_site_control.xml")
    gravcomp: bool = True

    def model_post_init(self, _context):
        super().model_post_init(_context)
        self.robot_view_factory = partial(RBY1RobotView, holo_base=self.use_holo_base)


class RBY1MConfig(RBY1Config):
    """Configuration for RBY1M i.e. mecanum wheel robot."""

    use_holo_base: bool = True  # Whether to use virtual holonomic base joints or not
    name: str = "rby1m"
    robot_xml_path: Path = Path("rby1_v1.2_site_control.xml")
    # NOTE: No wheel control for now so we can re-use this config for both the robot types


class RBY1MOpenCloseConfig(RBY1MConfig):
    """RBY1M config for open/close tasks.

    Uses single-scalar torso height control (torso_1 = torso_3 = h, torso_2 = -2*h)
    instead of commanding all 6 torso joints independently.
    """

    command_mode: dict[str, str | None] = {
        "arm": "joint_rel_position",
        "gripper": "joint_position",
        "base": "holo_joint_rel_planar_position",
        "head": None,
        "torso": "height",
    }


class FloatingRUMRobotConfig(BaseRobotConfig):
    robot_cls: type[FloatingRUMRobot] | None = FloatingRUMRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = FloatingRUMRobot
    robot_view_factory: RobotViewFactory | None = FloatingRUMRobotView
    robot_namespace: str = "robot_0/"
    ctrl_dt_ms: float = 50.0
    command_mode: dict = {}
    name: str = "floating_rum"
    robot_xml_path: Path = Path("model.xml")
    init_qpos: dict[str, list] = {
        "gripper": [0.0, 0.0],
    }
    init_qpos_noise_range: dict[str, list] = {}


class FloatingRobotiq2f85RobotConfig(BaseRobotConfig):
    robot_cls: type[FloatingRobotiqRobot] = FloatingRobotiqRobot
    robot_factory: Callable[[MjData, BaseRobotConfig], Robot] = FloatingRobotiqRobot
    robot_view_factory: RobotViewFactory = FloatingRobotiq2f85RobotView
    robot_namespace: str = "robot_0/"
    ctrl_dt_ms: float = 50.0
    command_mode: dict = {}
    action_spec: dict[str, int] = {"base": 7, "gripper": 2}  # Max lengths for action components
    name: str = "floating_robotiq"
    robot_xml_path: Path = Path("model.xml")
    init_qpos: dict[str, list] = {
        "gripper": [0.00296, 0.00296],
    }
    init_qpos_noise_range: dict[str, list] = {}


class I2rtYamRobotConfig(BaseRobotConfig):
    """Configuration for i2rt YAM 6-DOF robot."""

    robot_cls: type[I2rtYamRobot] | None = I2rtYamRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = I2rtYamRobot
    robot_view_factory: RobotViewFactory | None = I2rtYamRobotView
    robot_namespace: str = "robot_0/"
    name: str = "i2rt_yam"
    robot_xml_path: Path = Path("yam.xml")
    # Base platform size [width, depth, height] - raises robot above ground
    base_size: list[float] | None = [0.3, 0.3, 0.7]
    # Initial joint positions - modified from XML keyframe "home" to avoid wrist singularity
    # Original: "0 1.047 1.047 0 0 0" but joints 4,5,6 at 0 causes wrist singularity
    # Adding small offsets to wrist joints (4,5) to move away from singular configuration
    init_qpos: dict[str, list[float]] = {
        "arm": [0.0, 1.047, 1.047, 0.1, -0.1, 0.0],  # Offset joints 4,5 to avoid singularity
        "gripper": [0.0, 0.0],  # left_finger, right_finger (coupled)
    }
    init_qpos_noise_range: dict[str, list[float]] | None = None
    command_mode: dict[str, str] = {
        "arm": "joint_position",
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ["joint_position", "joint_rel_position"]


class BimanualYamRobotConfig(BaseRobotConfig):
    """Configuration for bimanual YAM robot (two 6-DOF arms with parallel grippers).

    The bimanual YAM consists of two YAM arms positioned 44cm apart,
    both facing forward.
    """

    robot_cls: type[BimanualYamRobot] | None = BimanualYamRobot
    robot_factory: Callable[[MjData, Any], Robot] | None = BimanualYamRobot
    robot_view_factory: RobotViewFactory | None = BimanualYamRobotView
    robot_namespace: str = "robot_0/"
    name: str = "i2rt_yam"  # Use same directory as single-arm YAM
    robot_xml_path: Path = Path("bimanual_yam.xml")
    # Base platform size [x, y, z] - raises robot above ground
    # Wider in Y to accommodate both arms (44cm apart along Y axis)
    base_size: list[float] | None = [0.3, 0.8, 0.7]
    # Initial joint positions for both arms
    # These initializations are taken from observation values that I saw in the dataset
    init_qpos: dict[str, list[float]] = {
        "left_arm": [0.0624, 0.0109, 0.1707, -0.5938, 0.411, 0.3401],
        "right_arm": [0.0006, 0.0147, 0.1669, -0.6407, 0.0746, 0.1516],
        "left_gripper": [0.03914, 0.0],
        "right_gripper": [0.04068, 0.0],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = None
    command_mode: dict[str, str] = {
        "arm": "joint_position",
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ["joint_position", "joint_rel_position"]
