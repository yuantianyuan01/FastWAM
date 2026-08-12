import abc
import logging
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from mujoco import MjData, MjSpec
from scipy.spatial.transform import Rotation as R
from scipy.stats import truncnorm

from molmo_spaces.controllers.abstract import AbstractPositionController, Controller
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.robot_views.abstract import RobotView

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import ActionNoiseConfig, BaseRobotConfig
    from molmo_spaces.env.abstract_sensors import Sensor
    from molmo_spaces.kinematics.parallel.parallel_kinematics import ParallelKinematics


log = logging.getLogger(__name__)


class Robot:
    def __init__(self, mj_data: MjData, exp_config: "MlSpacesExpConfig"):
        """
        Args:
            mj_data: The MuJoCo data structure containing the robot definistion and current simulation state
        """
        self.mj_model = mj_data.model
        self.mj_data = mj_data
        self.exp_config = exp_config
        self._last_unnoised_cmd_joint_pos: dict[str, np.ndarray] | None = None

    @property
    @abc.abstractmethod
    def namespace(self) -> str:
        """robot namespace used to differentiate between one or multiple robots and the environment"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def robot_view(self) -> RobotView:
        """robot view for interfacing with joints / links / actuators"""

    @property
    @abc.abstractmethod
    def kinematics(self) -> MlSpacesKinematics:
        """kinematic solver for the robot"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parallel_kinematics(self) -> "ParallelKinematics":
        """parallel kinematic solver for the robot"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def controllers(self) -> dict[str, Controller]:
        """One or more controllers for the robot joints / wheels / etc."""
        raise NotImplementedError

    def set_stationary(self) -> None:
        """Set all controllers of the robot to stationary mode (if not already set),
        i.e., hold current positions or zero velocities."""
        for _, controller in self.controllers.items():
            if not controller.stationary:
                controller.set_to_stationary()

    def create_robot_sensors(self) -> list["Sensor"]:
        """Get robot-specific sensors that should be registerd with a Task's sensor suite."""
        return []

    def get_arm_move_group_ids(self) -> list[str]:
        """Get the move group IDs that should have TCP-bounded noise applied.

        Override in subclass to specify which move groups are arms.
        Default returns empty list (no noise applied).
        """
        return []

    def _apply_tcp_noise_to_move_group(
        self,
        mg_id: str,
        commanded_joint_pos: np.ndarray,
        noise_config: "ActionNoiseConfig",
        use_truncated_gaussian: bool = True,
    ) -> np.ndarray:
        """Apply TCP-bounded noise to a single arm move group.

        The noise model:
        1. Computes expected TCP delta from the commanded joint action using Jacobian
        2. Scales noise proportionally to TCP delta magnitude
        3. Samples noise in TCP space (bounded by config)
        4. Maps noise back to joint space via Jacobian pseudo-inverse

        Args:
            mg_id: Move group ID (must be an arm with Jacobian available)
            commanded_joint_pos: The commanded joint positions
            noise_config: The noise configuration

        Returns:
            Noisy joint positions
        """
        mg = self.robot_view.get_move_group(mg_id)
        current_joint_pos = mg.joint_pos

        # Compute joint delta (what the action commands)
        joint_delta = commanded_joint_pos - current_joint_pos

        # Get Jacobian for this move group
        # The Jacobian maps joint velocities to TCP twist [linear_vel, angular_vel]
        # For small displacements, we can use it for positions too
        J = self.robot_view.get_jacobian(mg_id, [mg_id])  # 6 x n_joints

        # Compute expected TCP delta from joint action
        tcp_delta = J @ joint_delta  # 6D: [dx, dy, dz, drx, dry, drz]
        tcp_pos_delta = tcp_delta[:3]
        tcp_pos_delta_norm = np.linalg.norm(tcp_pos_delta)

        # Compute noise scale proportional to action magnitude
        # When TCP delta is zero, noise is zero
        scale_factor = noise_config.action_scale_factor
        position_noise_std = scale_factor * tcp_pos_delta_norm
        rotation_noise_std = position_noise_std * noise_config.rotation_noise_scale

        if use_truncated_gaussian:
            # Sample noise in TCP space using truncated Gaussian
            # truncnorm bounds are in units of std devs from mean
            if position_noise_std > 0:
                pos_bound = noise_config.max_tcp_position_noise / position_noise_std
                position_noise = truncnorm.rvs(
                    -pos_bound, pos_bound, scale=position_noise_std, size=3
                )
            else:
                position_noise = np.zeros(3)

            if rotation_noise_std > 0:
                rot_bound = noise_config.max_tcp_rotation_noise / rotation_noise_std
                rotation_noise = truncnorm.rvs(
                    -rot_bound, rot_bound, scale=rotation_noise_std, size=3
                )
            else:
                rotation_noise = np.zeros(3)
        else:
            # Sample noise in TCP space
            position_noise = np.random.randn(3) * position_noise_std
            rotation_noise = np.random.randn(3) * rotation_noise_std

            # Clip to maximum bounds
            position_noise = np.clip(
                position_noise,
                -noise_config.max_tcp_position_noise,
                noise_config.max_tcp_position_noise,
            )
            rotation_noise = np.clip(
                rotation_noise,
                -noise_config.max_tcp_rotation_noise,
                noise_config.max_tcp_rotation_noise,
            )

        # Combine into TCP space noise
        tcp_noise = np.concatenate([position_noise, rotation_noise])

        # Map TCP noise back to joint space via least-squares solve
        joint_noise, _, _, _ = np.linalg.lstsq(J, tcp_noise, rcond=None)

        # Add noise to commanded joint positions
        noisy_joint_pos = commanded_joint_pos + joint_noise

        # Clip to joint limits
        joint_limits = mg.joint_pos_limits
        noisy_joint_pos = np.clip(noisy_joint_pos, joint_limits[:, 0], joint_limits[:, 1])

        return noisy_joint_pos

    def apply_action_noise(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply action noise to the commanded action.

        Each robot subclass can override this to customize noise behavior.
        Default implementation applies TCP-bounded noise to arm move groups
        returned by get_arm_move_group_ids().

        Args:
            action: Action dict with move_group_id -> joint positions

        Returns:
            Modified action dict with noise added
        """
        noise_config = self.exp_config.robot_config.action_noise_config
        if not noise_config.enabled:
            return action

        arm_mg_ids = self.get_arm_move_group_ids()
        if not arm_mg_ids:
            return action

        noisy_action = dict(action)  # shallow copy

        for mg_id in arm_mg_ids:
            if mg_id not in action or action[mg_id] is None:
                continue

            commanded_joint_pos = np.asarray(action[mg_id])
            noisy_joint_pos = self._apply_tcp_noise_to_move_group(
                mg_id, commanded_joint_pos, noise_config
            )
            noisy_action[mg_id] = noisy_joint_pos

        return noisy_action

    def last_unnoised_cmd_joint_pos(self) -> dict[str, np.ndarray] | None:
        """Get the last unnoised joint position commands for the robot.

        Even if the commanded action was partial (due to e.g. noop), this will return the completely filled-in action.
        Done signals are not included.
        Non-position controlled move groups are not included.

        Returns:
            The last unnoised joint position commands for the robot, or None if no action has been set yet.
        """
        return deepcopy(self._last_unnoised_cmd_joint_pos)

    def _apply_action_noise_and_save_unnoised_cmd_jp(
        self, action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Note: this sets the unnoised pos commands as targets for the controllers.
        The controllers are idempotent, so this is fine.
        """
        # before noising, save the unnoised joint pos commands
        unnoised_cmd_jp = {}
        for mg_id, controller in self.controllers.items():
            if isinstance(controller, AbstractPositionController):
                if mg_id in action:
                    controller.set_target(action[mg_id])
                elif not controller.stationary:
                    controller.set_to_stationary()
                unnoised_cmd_jp[mg_id] = controller.target_pos.copy()
        self._last_unnoised_cmd_joint_pos = unnoised_cmd_jp

        return self.apply_action_noise(action)

    def update_control(self, action_command_dict: dict[str, np.ndarray]) -> None:
        """
        Update the control targets to the robot based on the provided action commands.
        Does not set the control inputs to the robot actuators. See compute_control().

        Args:
            action_command_dict: Dictionary containing action commands for the robot
                                 based on the move groups ids to be used.
        """
        action_command_dict = self._apply_action_noise_and_save_unnoised_cmd_jp(action_command_dict)

        for mg_id, controller in self.controllers.items():
            if mg_id in action_command_dict and action_command_dict[mg_id] is not None:
                controller.set_target(action_command_dict[mg_id])
            elif not controller.stationary:
                controller.set_to_stationary()

    def compute_control(self) -> None:
        """
        Compute and set the control inputs to the robot actuators based on the
        current state and the targets set by the user.
        Must be called after update_control().
        """
        for controller in self.controllers.values():
            ctrl_inputs = controller.compute_ctrl_inputs()
            controller.robot_move_group.ctrl = ctrl_inputs

    def set_joint_pos(self, robot_joint_pos_dict: dict[str, np.ndarray]) -> None:
        """Set all the robot's joint positions to the specified values.

        Args:
            robot_joint_pos_dict: Dictionary containing joint positions for the robot
                based on the move groups ids.
        """
        for mg_id, joint_pos in robot_joint_pos_dict.items():
            self.robot_view.get_move_group(mg_id).joint_pos = joint_pos

    def set_world_pose(self, robot_world_pose: np.ndarray | list[float]) -> None:
        """
        Set the robot's world pose to the specified pose in the world frame.

        Args:
            robot_world_pose: The pose of the robot in the world frame.
                If (4, 4) numpy array, it is assumed to be a transformation matrix.
                If (3,) list or numpy array, it is assumed to be (x, y, yaw).
                If (7,) list or numpy array, it is assumed to be (xyz + wxyz).
        """
        robot_world_pose = np.asarray(robot_world_pose)
        if robot_world_pose.shape == (4, 4):
            pose = robot_world_pose
        elif robot_world_pose.shape == (3,):
            pose = np.eye(4)
            pose[:2, 3] = robot_world_pose[:2]
            pose[:3, :3] = R.from_euler("Z", robot_world_pose[2]).as_matrix()
        elif robot_world_pose.shape == (7,):
            pose = np.eye(4)
            pose[:3, 3] = robot_world_pose[:3]
            pose[:3, :3] = R.from_quat(robot_world_pose[3:], scalar_first=True).as_matrix()
        else:
            raise ValueError(f"Invalid robot world pose shape: {robot_world_pose.shape}")

        self.robot_view.base.pose = pose

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the robot to its initial state or a provided state."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def robot_model_root_name() -> str:
        """The name of the root body of the robot model. This is NOT necessarily the root body of the robot after insertion."""
        raise NotImplementedError

    @classmethod
    def apply_control_overrides(cls, spec: MjSpec, robot_config: "BaseRobotConfig"):
        if robot_config.gravcomp:
            log.debug(f"Applying gravity compensation to robot {robot_config.name}")
            body_name = robot_config.robot_namespace + cls.robot_model_root_name()
            robot_body = spec.body(body_name)
            assert robot_body is not None, f"Robot body {body_name} not found in spec"
            for body in robot_body.find_all("body"):
                body: mujoco.MjsBody
                body.gravcomp = 1.0

        # TODO: control overrides will break if non-robot actuators are present
        stiffness, damping = robot_config.K_stiffness, robot_config.K_damping
        if stiffness is not None and damping is not None:
            assert len(stiffness) == len(damping), (
                "K_stiffness and K_damping must have the same length"
            )

        actuators: list[mujoco.MjsActuator] = spec.actuators
        if stiffness is not None:
            log.debug(f"Applying stiffness gains to robot {cls.robot_model_root_name()}")
            assert len(stiffness) <= len(actuators), (
                "number of stiffness gains cannot exceed number of actuators"
            )
            for i, actuator in enumerate(actuators[: len(stiffness)]):
                actuator.gainprm[0] = stiffness[i]
                actuator.biasprm[1] = -stiffness[i]
        if damping is not None:
            log.debug(f"Applying damping gains to robot {cls.robot_model_root_name()}")
            assert len(damping) <= len(actuators), (
                "number of damping gains cannot exceed number of actuators"
            )
            for i, actuator in enumerate(actuators[: len(damping)]):
                actuator.biasprm[2] = -damping[i]

        force_limit = robot_config.force_limit
        if force_limit is not None:
            log.debug(f"Applying force limits to robot {cls.robot_model_root_name()}")
            assert len(force_limit) <= len(actuators), (
                "number of force limits cannot exceed number of actuators"
            )
            for i, actuator in enumerate(actuators[: len(force_limit)]):
                actuator.forcelimited = 1
                actuator.forcerange[:] = [-force_limit[i], force_limit[i]]

    @classmethod
    def _load_robot_spec(
        cls, robot_config: "BaseRobotConfig", strip_meshes: bool = False
    ) -> MjSpec:
        """Load the robot MjSpec from the robot config's XML path.

        Args:
            robot_config: The robot config to load the spec for.
            strip_meshes: If True, remove all mesh geometries from the spec.
                Useful for kinematics-only models that don't need visual/collision meshes.
        """
        robot_spec = MjSpec.from_file(str(robot_config.get_robot_xml_path()))
        if strip_meshes:
            for body in robot_spec.bodies:
                body: mujoco.MjsBody
                for geom in body.geoms:
                    geom: mujoco.MjsGeom
                    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                        robot_spec.delete(geom)
        return robot_spec

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "BaseRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        """
        Add a robot to a scene, taking care of any robot-specific considerations.

        Args:
            robot_config: The robot config, of the corresponding derived class (i.e. FrankaConfig for Franka, etc.)
            spec: The scene to insert the robot into
            prefix: The prefix to use for the robot, i.e. the namespace
            pos: The position to use for the robot, either 2d or 3d. If 2d, the z-coordinate is assumed to be 0.
            quat: The quaternion to use for the robot, in [w, x, y, z] format.
            randomize_textures: Whether to randomize the textures of the robot, if applicable. Not supported by all robots.
            strip_meshes: If True, remove all mesh geometries before insertion (for kinematics-only models).
        """
        pos = pos + [0.0] if len(pos) == 2 else pos
        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        robot_root_name = cls.robot_model_root_name()
        attach_frame = spec.worldbody.add_frame(pos=pos, quat=quat)

        robot_root = robot_spec.body(robot_root_name)
        if robot_root is None:
            raise ValueError(f"Robot {robot_root_name=} not found in {robot_spec}")
        attach_frame.attach_body(robot_root, prefix, "")
