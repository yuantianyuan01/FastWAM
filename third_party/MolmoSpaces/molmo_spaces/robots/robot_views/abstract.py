"""
This module defines the core abstractions for representing and controlling robots in MuJoCo.
The architecture is based on a hierarchical structure where a RobotView contains multiple MoveGroups,
each representing an atomic collection of joints and actuators.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Literal, NoReturn, Optional, TypeAlias

import mujoco
import numpy as np
from mujoco import MjData
from scipy.spatial.transform import Rotation as R

from molmo_spaces.utils.linalg_utils import (
    global_to_relative_transform,
    normalize_ang_error,
    twist_to_transform,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose, site_pose
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_pos_quat


class MoveGroup(ABC):
    """Base class for any collection of joints and actuators in a robot.

    A MoveGroup represents an atomic collection of robotic joints and their associated actuators.
    It maintains a kinematic tree between a root frame and a leaf frame, and provides methods
    to control and query the state of its joints and actuators.

    The class handles the low-level details of interfacing with MuJoCo's state vectors (qpos, qvel)
    and control signals (ctrl), providing a clean interface for higher-level control.
    """

    def __init__(
        self,
        mj_data: MjData,
        joint_ids: list[int],
        actuator_ids: list[int],
        root_body_id: int,
        robot_base_group: Optional["RobotBaseGroup"] = None,
    ) -> None:
        """Initialize a MoveGroup.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            joint_ids: List of joint IDs that belong to this move group
            actuator_ids: List of actuator IDs that control the joints
            root_body_id: The ID of the root body of this move group
            robot_base_group: The RobotBaseGroup for the robot. If None, this MoveGroup is assumed to be the base.
        """
        self.mj_model = mj_data.model
        self.mj_data = mj_data
        self._name: str | None = None
        self._joint_ids = joint_ids
        self._robot_base_group = robot_base_group
        self._root_body_id = root_body_id

        self._joint_posadr: list[int] = []
        self._joint_veladr: list[int] = []
        for i in joint_ids:
            n_pos_dim = 1
            n_vel_dim = 1
            if self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                n_pos_dim = 7
                n_vel_dim = 6
            elif self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_BALL:
                n_pos_dim = 4
                n_vel_dim = 3
            self._joint_posadr.extend(
                list(range(self.mj_model.jnt_qposadr[i], self.mj_model.jnt_qposadr[i] + n_pos_dim))
            )
            self._joint_veladr.extend(
                list(range(self.mj_model.jnt_dofadr[i], self.mj_model.jnt_dofadr[i] + n_vel_dim))
            )

        self._actuator_ids = actuator_ids

    @property
    def name(self) -> str:
        """
        The name of this move group, which matches the move group ID in the containing robot view.
        Not safe to call before the move group is added to a robot view.
        """
        assert self._name is not None, "Move group name is not set"
        return self._name

    @cached_property
    def n_joints(self) -> int:
        """Number of joints in this move group."""
        return len(self._joint_ids)

    @cached_property
    def pos_dim(self) -> int:
        """
        Dimension of the ambient space of the manifold of joint positions.
        This may be different from the number of joints, in the case of free or ball joints.
        """
        return len(self._joint_posadr)

    @cached_property
    def vel_dim(self) -> int:
        """
        Dimension of the space of joint velocities.
        This may be different from the number of position dimensions, in the case of free or ball joints.
        """
        return len(self._joint_veladr)

    @cached_property
    def n_actuators(self) -> int:
        """Number of actuators in this move group."""
        return len(self._actuator_ids)

    @property
    def joint_pos(self) -> np.ndarray:
        """Current joint positions."""
        return self.mj_data.qpos[self._joint_posadr]

    @joint_pos.setter
    def joint_pos(self, joint_pos: np.ndarray) -> None:
        """Set joint positions.

        Args:
            joint_pos: Array of joint positions to set
        """
        self.mj_data.qpos[self._joint_posadr] = joint_pos

    @property
    def joint_pos_limits(self) -> np.ndarray:
        """Joint position limits (min, max) for each joint."""
        jnt_range = np.empty((self.pos_dim, 2))
        jnt_range[:, 0] = -np.inf
        jnt_range[:, 1] = np.inf

        i = 0
        for jnt_id in self._joint_ids:
            if self.mj_model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                i += 7
            elif self.mj_model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_BALL:
                i += 4
            else:
                if self.mj_model.jnt_limited[jnt_id]:
                    jnt_range[i] = self.mj_model.jnt_range[jnt_id]
                i += 1
        return jnt_range

    @property
    def joint_vel(self) -> np.ndarray:
        """Current joint velocities."""
        return self.mj_data.qvel[self._joint_veladr]

    @joint_vel.setter
    def joint_vel(self, joint_vel: np.ndarray) -> None:
        """Set joint velocities.

        Args:
            joint_vel: Array of joint velocities to set
        """
        self.mj_data.qvel[self._joint_veladr] = joint_vel

    @property
    def ctrl(self) -> np.ndarray:
        """Current control signals for the actuators."""
        return self.mj_data.ctrl[self._actuator_ids]

    @ctrl.setter
    def ctrl(self, ctrl: np.ndarray) -> None:
        """Set control signals for the actuators.

        Args:
            ctrl: Array of control signals to set
        """
        self.mj_data.ctrl[self._actuator_ids] = ctrl

    @property
    @abstractmethod
    def noop_ctrl(self) -> np.ndarray:
        """Get a control signal that maintains the current state."""
        raise NotImplementedError

    @cached_property
    def ctrl_limits(self) -> np.ndarray:
        """Control limits (min, max) for each actuator."""
        ctrl_range = np.empty((self.n_actuators, 2))
        for i, act_id in enumerate(self._actuator_ids):
            ctrl_range[i] = self.mj_model.actuator(act_id).ctrlrange
        return ctrl_range

    def integrate_joint_vel(self, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        """
        Integrate joint velocities by 1 unit time to get joint positions.
        This does not modify the state.

        Args:
            joint_pos: Joint positions at the start of the integration
            joint_vel: Joint velocities to integrate

        Returns:
            Joint positions at the end of the integration
        """
        new_jp = joint_pos.copy()
        i = 0
        j = 0
        for jnt_id in self._joint_ids:
            if self.mj_model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                trf = pos_quat_to_pose_mat(self.joint_pos[i : i + 3], self.joint_pos[i + 3 : i + 7])
                twist = joint_vel[j : j + 6]
                delta_trf = twist_to_transform(twist[:3], twist[3:])
                trf = trf @ delta_trf
                new_jp[i : i + 3], new_jp[i + 3 : i + 7] = pose_mat_to_pos_quat(trf)
                i += 7
                j += 6
            elif self.mj_model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_BALL:
                rotmat = R.from_quat(self.joint_pos[i : i + 4], scalar_first=True).as_matrix()
                twist = joint_vel[j : j + 3]
                delta_rotmat = R.from_rotvec(twist).as_matrix()
                rotmat = rotmat @ delta_rotmat
                new_jp[i : i + 4] = R.from_matrix(rotmat).as_quat(scalar_first=True)
                i += 4
                j += 3
            else:
                new_jp[i] += joint_vel[j]
                i += 1
                j += 1
        return new_jp

    @property
    def root_body_id(self) -> int:
        """The ID of the root body of this move group."""
        return self._root_body_id

    @property
    @abstractmethod
    def leaf_frame_to_world(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def root_frame_to_world(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def leaf_frame_to_root(self) -> np.ndarray:
        """Returns the pose of the leaf frame relative to the root frame.

        Returns:
            A 4x4 numpy array representing the rigid transformation from root to leaf frame.
        """
        return global_to_relative_transform(self.leaf_frame_to_world, self.root_frame_to_world)

    @property
    def root_frame_to_robot(self) -> np.ndarray:
        """Returns the pose of the root frame relative to the robot frame.

        Returns:
            A 4x4 numpy array representing the rigid transformation from robot to root frame.
        """
        if self._robot_base_group is None:
            return np.eye(4)
        return global_to_relative_transform(self.root_frame_to_world, self._robot_base_group.pose)

    @property
    def leaf_frame_to_robot(self) -> np.ndarray:
        """Returns the pose of the leaf frame relative to the robot frame.

        Returns:
            A 4x4 numpy array representing the rigid transformation from robot to leaf frame.
        """
        if self._robot_base_group is None:
            return self.leaf_frame_to_root
        return global_to_relative_transform(self.leaf_frame_to_world, self._robot_base_group.pose)

    @abstractmethod
    def get_jacobian(self) -> np.ndarray:
        """Returns the (6, model.nv) jacobian of the move group.

        The jacobian maps joint velocities to the spatial velocity of the leaf frame.

        Returns:
            A 6xN numpy array where N is the number of degrees of freedom in the model.

        See: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
        """
        raise NotImplementedError


class MJCFFrameMixin(ABC):
    """
    Mixin for move groups that represent the leaf frame as a body or site in the MJCF model.

    Note: Since this mixin provides a `get_jacobian()` implementation, inheriting classes must
        put this mixin first in the inheritance chain to satisfy the MRO.
    """

    @property
    @abstractmethod
    def leaf_frame_id(self) -> int:
        """The ID of the leaf frame, either a body ID or a site ID."""
        raise NotImplementedError

    @property
    @abstractmethod
    def leaf_frame_type(self) -> Literal["site", "body"]:
        """The type of the leaf frame."""
        raise NotImplementedError

    @property
    def leaf_frame_to_world(self) -> np.ndarray:
        assert isinstance(self, MoveGroup)
        if self.leaf_frame_type == "site":
            return site_pose(self.mj_data, self.leaf_frame_id)
        elif self.leaf_frame_type == "body":
            return body_pose(self.mj_data, self.leaf_frame_id)
        else:
            raise ValueError(f"Invalid leaf frame type: {self.leaf_frame_type}")

    def get_jacobian(self) -> np.ndarray:
        """
        Returns the (6, model.nv) jacobian of the move group's leaf frame.

        The jacobian maps joint velocities to the spatial velocity of the leaf frame.

        Returns:
            A 6xN numpy array where N is the number of degrees of freedom in the model.

        See: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
        """
        assert isinstance(self, MoveGroup), (
            f"{self.__class__.__name__} must be used with a MoveGroup"
        )

        J = np.zeros((6, self.mj_model.nv))
        if self.leaf_frame_type == "site":
            mujoco.mj_jacSite(self.mj_model, self.mj_data, J[:3], J[3:], self.leaf_frame_id)
        elif self.leaf_frame_type == "body":
            mujoco.mj_jacBody(self.mj_model, self.mj_data, J[:3], J[3:], self.leaf_frame_id)
        else:
            raise ValueError(f"Invalid leaf frame type: {self.leaf_frame_type}")
        return J


class SimplyActuatedMoveGroup(MoveGroup):
    """
    A SimplyActuatedMoveGroup is a move group with a 1:1 mapping between joints, actuators, and position/velocity addresses.
    """

    @property
    def joint_ids(self):
        return self._joint_ids

    @property
    def actuator_ids(self):
        return self._actuator_ids

    @property
    def joint_posadr(self):
        return self._joint_posadr

    @property
    def joint_veladr(self):
        return self._joint_veladr

    @property
    def noop_ctrl(self) -> np.ndarray:
        return self.joint_pos.copy()


class GripperGroup(MoveGroup):
    @abstractmethod
    def set_gripper_ctrl_open(self, open: bool) -> None:
        """Set the gripper commanded position to be fully open or closed.

        Args:
            open: True to open the gripper, False to close it
        """
        raise NotImplementedError

    @property
    def is_open(self) -> bool:
        """Whether the gripper is open."""
        return self.inter_finger_dist > np.mean(self.inter_finger_dist_range).item()

    @property
    @abstractmethod
    def inter_finger_dist_range(self) -> tuple[float, float]:
        """The (min, max) of the distance between the two fingers of the gripper."""
        raise NotImplementedError

    @property
    @abstractmethod
    def inter_finger_dist(self) -> float:
        """The distance between the two fingers of the gripper."""
        raise NotImplementedError

    @property
    def noop_ctrl(self) -> np.ndarray:
        old_ctrl = self.ctrl
        self.set_gripper_ctrl_open(self.is_open)
        noop_ctrl = self.ctrl.copy()
        self.ctrl = old_ctrl
        return noop_ctrl


class RobotBaseGroup(MoveGroup):
    """A MoveGroup that controls the overall pose of the robot.

    A RobotBase represents the base of a robot, which can be either mobile (e.g., wheeled)
    or immobile (e.g., fixed to a table). It provides methods to control and query the
    overall pose of the robot in the world frame.
    """

    def __init__(
        self, mj_data: MjData, joint_ids: list[int], actuator_ids: list[int], root_body_id: int
    ):
        """Initialize a RobotBase.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            joint_ids: List of joint IDs that belong to this base
            actuator_ids: List of actuator IDs that control the base
        """
        super().__init__(mj_data, joint_ids, actuator_ids, root_body_id)

    @cached_property
    def is_mobile(self) -> bool:
        """Whether this base is mobile (has actuators)."""
        return self.n_actuators > 0

    @property
    @abstractmethod
    def pose(self) -> np.ndarray:
        """Get the pose of the robot base relative to the world frame.

        Returns:
            A 4x4 numpy array representing the rigid transformation from world to base frame.
        """
        raise NotImplementedError

    @pose.setter
    @abstractmethod
    def pose(self, pose: np.ndarray):
        """Set the pose of the robot base relative to the world frame.

        Args:
            pose: A 4x4 numpy array representing the rigid transformation from world to base frame.
        """
        raise NotImplementedError

    @property
    def leaf_frame_to_world(self) -> np.ndarray:
        return self.pose

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.pose


class FreeJointRobotBaseGroup(RobotBaseGroup):
    """A RobotBase that uses a free joint to represent its pose.

    This implementation uses MuJoCo's free joint type to represent the base pose,
    which allows for full 6-DOF control of the robot's position and orientation.
    """

    def __init__(
        self,
        mj_data: MjData,
        base_joint_id: int,
        joint_ids: list[int],
        actuator_ids: list[int],
        floating=False,
    ):
        """Initialize a FreeJointRobotBase.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            base_joint_id: The ID of the free joint that represents the base pose
            joint_ids: List of additional joint IDs that belong to this base
            actuator_ids: List of actuator IDs that control the base
            floating: Whether the base is floating (concretely, whether IK should not constrain it to be on the ground)
        """
        base_body_id = mj_data.model.jnt_bodyid[base_joint_id]
        super().__init__(mj_data, [base_joint_id] + joint_ids, actuator_ids, base_body_id)
        assert self.mj_model.jnt_type[base_joint_id] == mujoco.mjtJoint.mjJNT_FREE
        self._base_joint_id = base_joint_id
        self._floating = floating

    @property
    def floating(self):
        """Whether the base is floating."""
        return self._floating

    @property
    def pose(self) -> np.ndarray:
        trf = np.eye(4)
        adr = self.mj_model.jnt_qposadr[self._base_joint_id]
        trf[:3, 3] = self.mj_data.qpos[adr : adr + 3]
        trf[:3, :3] = R.from_quat(
            self.mj_data.qpos[adr + 3 : adr + 7], scalar_first=True
        ).as_matrix()
        return trf

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        pos = pose[:3, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        adr = self.mj_model.jnt_qposadr[self._base_joint_id]
        self.mj_data.qpos[adr : adr + 3] = pos
        self.mj_data.qpos[adr + 3 : adr + 7] = quat

    def get_jacobian(self) -> np.ndarray:
        body_id = self.mj_model.jnt_bodyid[self._base_joint_id]
        J = np.zeros((6, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, J[:3], J[3:], body_id)
        return J


class HoloJointsRobotBaseGroup(RobotBaseGroup, SimplyActuatedMoveGroup):
    """A RobotBase that uses virtual holonomic joints to represent its pose.

    Assumes three virtual holonomic joints for x, y, and theta control.
    """

    def __init__(
        self,
        mj_data: MjData,
        world_site_id: int,
        holo_base_site_id: int,
        joint_ids: list[int],
        actuator_ids: list[int],
        root_body_id: int,
    ):
        """Initialize a HoloJointsRobotBase that has virtual holonomic joints and uses site control.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            world_site_id: The ID of the world site
            holo_base_site_id: The ID of the site that represents the holonomic base pose
            joint_ids: List of joint IDs that belong to the virtual holonomic base pose.
                       NOTE: Assumed order is [x, y, theta].
            actuator_ids: List of actuator IDs that control the base
            root_body_id: The ID of the body that represents the robot base
        """
        super().__init__(mj_data, joint_ids, actuator_ids, root_body_id)

        assert len(joint_ids) == 3, "HoloJointsRobotBaseGroup must have 3 joints (x, y, theta)"
        assert all(
            self.mj_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE for jid in joint_ids[:2]
        ), "x, y joints must be slide joints"
        assert self.mj_model.jnt_type[joint_ids[2]] == mujoco.mjtJoint.mjJNT_HINGE, (
            "theta joint must be hinge"
        )
        assert len(actuator_ids) == 3, (
            "HoloJointsRobotBaseGroup must have 3 actuators (x, y, theta)"
        )

        self._world_site_id = world_site_id
        self._holo_base_site_id = holo_base_site_id

    @property
    def pose(self) -> np.ndarray:
        trf = np.eye(4)
        x = self.mj_data.qpos[self.mj_model.jnt_qposadr[self._joint_ids[0]]]
        y = self.mj_data.qpos[self.mj_model.jnt_qposadr[self._joint_ids[1]]]
        theta = self.mj_data.qpos[self.mj_model.jnt_qposadr[self._joint_ids[2]]]
        trf[0, 3] = x
        trf[1, 3] = y
        trf[:2, :2] = R.from_euler("z", theta, degrees=False).as_matrix()[:2, :2]
        return trf

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        pos = pose[:3, 3]
        theta = np.arctan2(pose[1, 0], pose[0, 0])
        self.mj_data.qpos[self.mj_model.jnt_qposadr[self._joint_ids[0]]] = pos[0]  # x
        self.mj_data.qpos[self.mj_model.jnt_qposadr[self._joint_ids[1]]] = pos[1]  # y
        self.mj_data.qpos[self.mj_model.jnt_qposadr[self._joint_ids[2]]] = theta  # theta

    @property
    def ctrl(self) -> np.ndarray:
        """Current control signals for the holonomic base actuators."""
        return self.mj_data.ctrl[self._actuator_ids]

    @ctrl.setter
    def ctrl(self, ctrl: np.ndarray) -> None:
        """Set control signals for the holonomic base actuators.

        Args:
            ctrl: Array of control signals to set
        """
        ctrl = ctrl.copy()

        # Wrap target yaw to be within +-pi of current yaw
        curr_yaw = self.joint_pos[2]
        ctrl[2] = curr_yaw + normalize_ang_error(ctrl[2] - curr_yaw)

        self.mj_data.ctrl[self._actuator_ids] = ctrl

    @property
    def noop_ctrl(self):
        pose = site_pose(self.mj_data, self._holo_base_site_id)
        yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)[2]
        yaw = normalize_ang_error(yaw)
        return np.array([pose[0, 3], pose[1, 3], yaw])

    def get_jacobian(self) -> np.ndarray:
        J = np.zeros((6, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, J[:3], J[3:], self._holo_base_site_id)
        return J


class MocapRobotBaseGroup(RobotBaseGroup):
    """A RobotBase that uses a mocap body to represent its pose."""

    def __init__(self, mj_data: MjData, robot_base_body_id: int):
        """Initialize a MocapRobotBase.

        Args:
            mj_data: The MuJoCo data structure containing the simulation state
            robot_base_body_id: The ID of the mocap body that represents the robot base. Not the Mocap ID!
        """
        robot_base_mocap_id = mj_data.model.body_mocapid[robot_base_body_id]
        super().__init__(mj_data, [], [], robot_base_body_id)
        self._robot_base_mocap_id = robot_base_mocap_id

    @property
    def noop_ctrl(self):
        return np.array([])

    @property
    def pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = self.mj_data.mocap_pos[self._robot_base_mocap_id]
        pose[:3, :3] = R.from_quat(
            self.mj_data.mocap_quat[self._robot_base_mocap_id], scalar_first=True
        ).as_matrix()
        return pose

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        pos = pose[:3, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        self.mj_data.mocap_pos[self._robot_base_mocap_id] = pos
        self.mj_data.mocap_quat[self._robot_base_mocap_id] = quat

    def get_jacobian(self) -> np.ndarray:
        return np.zeros((6, self.mj_model.nv))


class ImmobileRobotBaseGroup(RobotBaseGroup):
    """A RobotBase that is immobile and does not have a mocap body.

    This generally shouldn't be used, and MocapRobotBaseGroup should be preferred.
    If a scene is badly constructed and the robot base is not a mocap body, this might be necessary.
    """

    def __init__(self, mj_data: MjData, robot_base_body_id: int) -> None:
        """Initialize a ImmobileRobotBase.

        Args:
            mj_data: The MuJoCo data structure containing the simulation state
            robot_base_body_id: The ID of the body that represents the robot base
        """
        super().__init__(mj_data, [], [], robot_base_body_id)
        self._robot_base_body_id = robot_base_body_id

    @property
    def noop_ctrl(self):
        return np.array([])

    @property
    def pose(self) -> np.ndarray:
        return body_pose(self.mj_data, self._robot_base_body_id)

    @pose.setter
    def pose(self, pose: np.ndarray) -> NoReturn:
        raise ValueError("Cannot set the pose of an immobile robot base")

    def get_jacobian(self) -> np.ndarray:
        return np.zeros((6, self.mj_model.nv))


class RobotView(ABC):
    """Top-level class that contains and manages multiple MoveGroups.

    A RobotView represents a complete robot, composed of multiple MoveGroups (arms, base, etc.).
    It provides a unified interface to control and query the state of all move groups,
    and handles the coordination between different parts of the robot.
    """

    def __init__(self, mj_data: MjData, move_groups: dict[str, MoveGroup]) -> None:
        """Initialize a RobotView.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            move_groups: Dictionary mapping move group names to their implementations
        """
        self.mj_model = mj_data.model
        self.mj_data = mj_data
        self._move_groups = move_groups
        for mg_name, mg in move_groups.items():
            assert mg._name is None, f"Move group {mg_name} already has a name {mg._name}"
            mg._name = mg_name

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this robot model."""
        raise NotImplementedError

    @cached_property
    def n_grippers(self) -> int:
        """Number of grippers in this robot."""
        return sum(1 for mg in self._move_groups.values() if isinstance(mg, GripperGroup))

    @property
    @abstractmethod
    def base(self) -> RobotBaseGroup:
        """The base of the robot."""
        raise NotImplementedError

    def move_group_ids(self) -> list[str]:
        """Get the IDs of all move groups in this robot."""
        return list(self._move_groups.keys())

    def get_move_group(self, mg_id: str):
        """Get a move group by its ID.

        Args:
            mg_id: The ID of the move group to get
        """
        return self._move_groups[mg_id]

    def get_gripper(self, gripper_group_id: str):
        """Get a gripper by its ID.

        Args:
            gripper_group_id: The ID of the gripper group to get
        """
        gripper = self.get_move_group(gripper_group_id)
        assert isinstance(gripper, GripperGroup)
        return gripper

    def get_qpos_dict(self, move_group_ids: list[str] | None = None):
        """Get the joint positions of all move groups.

        Args:
            move_group_ids: The IDs of the move groups to get the joint positions of.
                            If None, all move groups will be included.

        Returns:
            A dictionary mapping move group IDs to their joint positions.
        """
        if move_group_ids is None:
            move_group_ids = self.move_group_ids()
        return {mg_id: self._move_groups[mg_id].joint_pos for mg_id in move_group_ids}

    def get_qvel_dict(self, move_group_ids: list[str] | None = None):
        """Get the joint velocities of all move groups.

        Args:
            move_group_ids: The IDs of the move groups to get the joint velocities of.
                            If None, all move groups will be included.

        Returns:
            A dictionary mapping move group IDs to their joint velocities.
        """
        if move_group_ids is None:
            move_group_ids = self.move_group_ids()
        return {mg_id: self._move_groups[mg_id].joint_vel for mg_id in move_group_ids}

    def set_qpos_dict(self, qpos_dict: dict[str, np.ndarray]) -> None:
        """Set the joint positions of all move groups.

        Args:
            qpos_dict: A dictionary mapping move group IDs to their joint positions.
        """
        for mg_id, qpos in qpos_dict.items():
            self._move_groups[mg_id].joint_pos = qpos

    def get_ctrl_dict(self, move_group_ids: list[str] | None = None):
        if move_group_ids is None:
            move_group_ids = self.move_group_ids()
        return {mg_id: self._move_groups[mg_id].ctrl for mg_id in move_group_ids}

    def get_noop_ctrl_dict(self, move_group_ids: list[str] | None = None):
        if move_group_ids is None:
            move_group_ids = self.move_group_ids()
        return {mg_id: self._move_groups[mg_id].noop_ctrl for mg_id in move_group_ids}

    def get_gripper_movegroup_ids(self) -> list[str]:
        """Get the IDs of all gripper move groups in this robot."""
        if hasattr(self, "_gripper_movegroup_ids_cache"):
            return self._gripper_movegroup_ids_cache
        self._gripper_movegroup_ids_cache = [
            mg_id
            for mg_id in self.move_group_ids()
            if isinstance(self._move_groups[mg_id], GripperGroup)
        ]
        return self._gripper_movegroup_ids_cache

    def get_jacobian(self, move_group_id: str, input_move_group_ids: list[str]) -> np.ndarray:
        """Calculate the Jacobian of a move group with respect to specific input move groups.

        This allows computing the Jacobian while locking certain joints (by excluding their
        move groups from input_move_group_ids).

        Args:
            move_group_id: The ID of the move group to get the jacobian of
            input_move_group_ids: The IDs of the move groups to use as input

        Returns:
            The (6, N) jacobian of the move group, where N is the total number of degrees
            of freedom of the input move groups.

        See: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
        """
        J = self._move_groups[move_group_id].get_jacobian()
        qveladr: list[int] = []
        for mg_id in input_move_group_ids:
            mg = self._move_groups[mg_id]
            qveladr.extend(mg._joint_veladr)

            # don't allow non-floating robot bases to move vertically, or pitch/roll
            if isinstance(mg, FreeJointRobotBaseGroup) and not mg.floating:
                for jnt_id in mg._joint_ids:
                    if self.mj_model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                        dofadrs = self.mj_model.jnt_dofadr[jnt_id] + np.array([2, 3, 4])
                        J[:, dofadrs] = 0.0
        J = J[:, qveladr]
        return J


RobotViewFactory: TypeAlias = Callable[
    [MjData, str], RobotView
]  # factory function that creates a RobotView from a MjData and a robot namespace
