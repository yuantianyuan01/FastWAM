from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from mujoco import MjData, MjSpec

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.env.rby1_sensors import RBY1GraspStateSensor
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.controllers.base_pose import DiffDriveBasePoseController
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.controllers.torso_height import TorsoHeightJointPosController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView


class RBY1(Robot):
    """RBY1 Robot class for the RBY1 robot.
    This class extends the Robot class and provides specific implementations for the RBY1 robot
    including the robot view, controllers, and a kinematic solver.

    The mode in which the robot arm, gripper, or base is commanded (Eg. ee position command or joint
    position command) can be set using the `arm_command_mode`, `gripper_command_mode`, and `base_command_mode`
    parameters in the robot configuration.

    The move_group i.e., the set of robot parts that are to be moved can be dynamically changed,
    e.g., move base for navigation, then move both base and arm for opening doors.

    """

    def __init__(self, mj_data: MjData, exp_config: "MlSpacesExpConfig") -> None:
        """
        Args:
            exp_config: Experiment configuration params
            mj_data: MuJoCo data structure containing the current simulation state
        """
        super().__init__(mj_data, exp_config)

        self._namespace = self.exp_config.robot_config.robot_namespace
        self._use_holo_base = self.exp_config.robot_config.use_holo_base

        # Create the robot view:
        self._robot_view = RBY1RobotView(mj_data, self.namespace, holo_base=self._use_holo_base)

        # Create kinematic solver:
        self._kinematics = MlSpacesKinematics(self.exp_config.robot_config)

        # Create controllers:

        # All the joint actuators are position controlled.
        # However, they can be commanded in other modes like velocity, ee position, etc. and the controller will
        # perform the necessary conversions to joint position.
        self.arm_command_modes = [
            "joint_position",
            "joint_rel_position",
            "joint_velocity",
            "ee_position",
            "ee_velocity",
        ]
        if self.exp_config.robot_config.command_mode["arm"] is not None:
            assert self.exp_config.robot_config.command_mode["arm"] in self.arm_command_modes, (
                f"Arm command mode {self.exp_config.robot_config.command_mode['arm']} not in {self.arm_command_modes}"
            )
        self.arm_command_mode = (
            "joint_position"
            if not self.exp_config.robot_config.command_mode["arm"]
            else self.exp_config.robot_config.command_mode["arm"]
        )
        if self.arm_command_mode == "joint_rel_position":
            left_arm_controller = JointRelPosController(self.robot_view.get_move_group("left_arm"))
            right_arm_controller = JointRelPosController(
                self.robot_view.get_move_group("right_arm")
            )
        elif self.arm_command_mode == "joint_position":
            left_arm_controller = JointPosController(self.robot_view.get_move_group("left_arm"))
            right_arm_controller = JointPosController(self.robot_view.get_move_group("right_arm"))
        else:
            raise NotImplementedError(
                f"Arm command mode {self.arm_command_mode} not implemented yet."
            )

        # Gripper command modes - separate from arm command modes
        self.gripper_command_modes = [
            "joint_position",
            "joint_rel_position",
            "joint_velocity",
        ]
        if self.exp_config.robot_config.command_mode["gripper"] is not None:
            assert (
                self.exp_config.robot_config.command_mode["gripper"] in self.gripper_command_modes
            ), (
                f"Gripper command mode {self.exp_config.robot_config.command_mode['gripper']} not in {self.gripper_command_modes}"
            )
        self.gripper_command_mode = (
            "joint_position"
            if not self.exp_config.robot_config.command_mode["gripper"]
            else self.exp_config.robot_config.command_mode["gripper"]
        )
        if self.gripper_command_mode == "joint_rel_position":
            left_gripper_controller = JointRelPosController(
                self.robot_view.get_move_group("left_gripper")
            )
            right_gripper_controller = JointRelPosController(
                self.robot_view.get_move_group("right_gripper")
            )
        elif self.gripper_command_mode == "joint_position":
            left_gripper_controller = JointPosController(
                self.robot_view.get_move_group("left_gripper")
            )
            right_gripper_controller = JointPosController(
                self.robot_view.get_move_group("right_gripper")
            )
        else:
            raise NotImplementedError(
                f"Gripper command mode {self.gripper_command_mode} not implemented yet."
            )

        # The base actuators (wheels) are usually velocity controlled. If using virtual holonomic joints, they are position controlled.
        # Base actuators can still be commanded in other modes like planar position, planar velocity, etc.
        # and the controller should perform the necessary conversions to wheel velocity or holo joint position.
        self.base_command_modes = [
            "planar_position",
            "planar_velocity",
            "wheel_velocity",
            "holo_joint_planar_position",
            "holo_joint_rel_planar_position",
        ]
        if self.exp_config.robot_config.command_mode["base"] is not None:
            assert self.exp_config.robot_config.command_mode["base"] in self.base_command_modes, (
                f"Base command mode {self.exp_config.robot_config.command_mode['base']} not in {self.base_command_modes}"
            )
        self.base_command_mode = (
            "planar_position"
            if not self.exp_config.robot_config.command_mode["base"]
            else self.exp_config.robot_config.command_mode["base"]
        )
        if self.base_command_mode == "planar_position":
            base_controller = DiffDriveBasePoseController(
                self.exp_config.robot_config, self.robot_view.get_move_group("base")
            )
        elif self.base_command_mode == "holo_joint_rel_planar_position":
            base_controller = JointRelPosController(self.robot_view.get_move_group("base"))
        elif self.base_command_mode == "holo_joint_planar_position":
            base_controller = JointPosController(self.robot_view.get_move_group("base"))
        else:
            raise NotImplementedError(
                f"Base command mode {self.base_command_mode} not implemented yet."
            )

        # Head is fixed - no head actions are supported
        self.head_command_mode = self.exp_config.robot_config.command_mode.get("head")
        assert self.head_command_mode is None, (
            "RBY1 head actuation is disabled. The head is fixed at init_qpos['head'] with optional "
            "randomization via init_qpos_noise_range['head']. "
            "Do not set command_mode['head'] to a non-None value."
        )

        # Torso command modes
        self.torso_command_modes = ["joint_position", "height"]
        self.torso_command_mode = (
            self.exp_config.robot_config.command_mode.get("torso", "joint_position")
            or "joint_position"
        )
        assert self.torso_command_mode in self.torso_command_modes, (
            f"Torso command mode {self.torso_command_mode} not in {self.torso_command_modes}"
        )
        if self.torso_command_mode == "height":
            torso_controller = TorsoHeightJointPosController(
                self.robot_view.get_move_group("torso")
            )
        else:
            torso_controller = JointPosController(self.robot_view.get_move_group("torso"))

        self._controllers = {
            "base": base_controller,
            "torso": torso_controller,
            "left_arm": left_arm_controller,
            "right_arm": right_arm_controller,
            "left_gripper": left_gripper_controller,
            "right_gripper": right_gripper_controller,
        }
        assert set(self._controllers.keys()).issubset(set(self._robot_view.move_group_ids())), (
            "All controller keys must be move group IDs"
        )

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    @property
    def namespace(self):
        return self._namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        raise NotImplementedError("Parallel kinematics not implemented for RBY1")

    def create_robot_sensors(self):
        return super().create_robot_sensors() + [
            RBY1GraspStateSensor(uuid="rby1_left_grasp_state", arm_side="left"),
            RBY1GraspStateSensor(uuid="rby1_right_grasp_state", arm_side="right"),
        ]

    def get_arm_move_group_ids(self) -> list[str]:
        """RBY1 has two independent arms - each gets independent noise."""
        return ["left_arm", "right_arm"]

    def _apply_base_noise(
        self,
        commanded_base_pos: np.ndarray,
    ) -> np.ndarray:
        """Apply planar noise to base commands (x, y, theta).

        The noise model:
        1. Computes displacement from current base pose
        2. Scales noise proportionally to displacement magnitude
        3. Samples noise in planar space (bounded by config)

        Args:
            commanded_base_pos: The commanded base position [x, y, theta]

        Returns:
            Noisy base position [x, y, theta]
        """
        noise_config = self.exp_config.robot_config.action_noise_config

        # Get current base position (x, y, theta)
        base_mg = self.robot_view.get_move_group("base")
        current_base_pos = base_mg.joint_pos  # [x, y, theta] for holo base

        # Compute displacement
        delta = commanded_base_pos - current_base_pos
        position_delta = delta[:2]  # x, y
        rotation_delta = delta[2] if len(delta) > 2 else 0.0  # theta

        position_delta_norm = np.linalg.norm(position_delta)
        rotation_delta_abs = abs(rotation_delta)

        # Compute noise scale proportional to action magnitude
        # When delta is zero, noise is zero
        scale_factor = noise_config.base_action_scale_factor

        position_noise_std = scale_factor * position_delta_norm
        rotation_noise_std = scale_factor * rotation_delta_abs

        # Sample noise
        position_noise = np.random.randn(2) * position_noise_std
        rotation_noise = np.random.randn() * rotation_noise_std

        # Clip to maximum bounds
        position_noise = np.clip(
            position_noise,
            -noise_config.max_base_position_noise,
            noise_config.max_base_position_noise,
        )
        rotation_noise = np.clip(
            rotation_noise,
            -noise_config.max_base_rotation_noise,
            noise_config.max_base_rotation_noise,
        )

        # Apply noise
        noisy_base_pos = commanded_base_pos.copy()
        noisy_base_pos[:2] += position_noise
        if len(noisy_base_pos) > 2:
            noisy_base_pos[2] += rotation_noise

        return noisy_base_pos

    def apply_action_noise(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply action noise to the commanded action.

        Extends the base class implementation to also apply base noise
        for RBY1's holonomic base commands.

        Args:
            action: Action dict with move_group_id -> joint positions

        Returns:
            Modified action dict with noise added
        """
        noise_config = self.exp_config.robot_config.action_noise_config
        if not noise_config.enabled:
            return action

        # Apply arm noise via parent class
        noisy_action = super().apply_action_noise(action)

        # Apply base noise if base command is present and using holonomic planar mode
        if "base" in action and action["base"] is not None:
            if "holo_joint" in self.base_command_mode and "planar" in self.base_command_mode:
                commanded_base_pos = np.asarray(action["base"])
                noisy_base_pos = self._apply_base_noise(commanded_base_pos)
                noisy_action["base"] = noisy_base_pos

        return noisy_action

    def get_world_pose_tf_mat(self):
        """Get the robot's world pose transformation matrix.

        Returns:
            np.ndarray: 4x4 transformation matrix for the robot base pose in world frame
        """
        return self.robot_view.get_move_group("base").pose

    def reset(self) -> None:
        """Reset the robot to its initial state."""

        init_qpos_dict = self.exp_config.robot_config.init_qpos
        self.set_joint_pos(init_qpos_dict)

        # reset controllers
        for _, controller in self._controllers.items():
            controller.reset()

        # Set the head ctrl to match its qpos to prevent jerking at trajectory start.
        # The head has actuators but no controller, so we manually set ctrl = noop_ctrl.
        head_mg = self.robot_view.get_move_group("head")
        head_mg.ctrl = head_mg.noop_ctrl

    def get_joint_position(self, move_group_ids: list[str]) -> np.ndarray:
        """Get the current joint positions of the move groups"""
        return np.concatenate(
            [
                self._robot_view.get_move_group(move_group_id).joint_pos.copy()
                for move_group_id in move_group_ids
            ]
        )

    def get_joint_ranges(self, move_group_ids: list[str]):
        """Get the joint ranges of the move groups"""
        joint_ranges = {}
        count = 0
        for move_group_id in move_group_ids:
            joint_ranges[move_group_id] = (
                count,
                count + self._robot_view.get_move_group(move_group_id).n_joints,
            )
            count += self._robot_view.get_move_group(move_group_id).n_joints
        return joint_ranges

    @staticmethod
    def robot_model_root_name() -> str:
        return "robot_0/base"

    @classmethod
    def apply_control_overrides(cls, spec: MjSpec, robot_config: "BaseRobotConfig"):
        # the model root name already includes the hardcoded namespace
        tmp_robot_config = robot_config.model_copy(deep=True)
        tmp_robot_config.robot_namespace = ""
        super().apply_control_overrides(spec, tmp_robot_config)

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
        assert prefix == "robot_0/", "RBY1 robot namespace must be 'robot_0/'"
        super().add_robot_to_scene(
            robot_config=robot_config,
            spec=spec,
            prefix="",  # elements already have robot_0/ prefix in MJCF
            pos=pos,
            quat=quat,
            randomize_textures=randomize_textures,
            strip_meshes=strip_meshes,
        )

        # RBY1 doesn't support insertion not at the origin or identity rotation
        # This can be fixed if it's worth the effort (see mobile franka for example)
        assert np.allclose(np.array(pos), [0, 0, 0]), "RBY1 insertion position is not zero!"
        assert np.allclose(np.array(quat), [1, 0, 0, 0]), "RBY1 insertion rotation is not identity!"

        def add_slider_act(
            name: str, ctrlrange: float, gainprm: float, biasprm: list[float], gear_idx: int
        ):
            act = spec.add_actuator()
            act.name = f"{prefix}{name}"
            act.target = f"{prefix}base_site"
            act.refsite = f"{prefix}world"
            act.ctrlrange = np.array([-ctrlrange, ctrlrange])
            act.gainprm[0] = gainprm
            act.biasprm[: len(biasprm)] = biasprm
            act.trntype = mujoco.mjtTrn.mjTRN_SITE
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            gear = [0] * 6
            gear[gear_idx] = 1
            act.gear = gear
            return act

        if robot_config.use_holo_base:
            spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0.005], quat=[1, 0, 0, 0])
            add_slider_act("base_x_act", 25, 25000, [0, -25000, 0.5], 0)
            add_slider_act("base_y_act", 25, 25000, [0, -25000, 0.5], 1)
            add_slider_act("base_theta_act", np.pi, 5000, [0, -5000, 0.5], 5)
