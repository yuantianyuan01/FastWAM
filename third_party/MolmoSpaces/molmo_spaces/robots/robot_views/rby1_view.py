"""
Implementation of the RBY1 robot model.

The RBY1 is a humanoid robot with:
- Two 7-DOF arms with grippers
- A 6-DOF torso
- A mobile base with two wheels
- A 2-DOF head

Each component is implemented as a MoveGroup, with the overall robot structure
managed by the RBY1 RobotView class.
"""

from typing import Literal

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    FreeJointRobotBaseGroup,
    GripperGroup,
    HoloJointsRobotBaseGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.linalg_utils import normalize_ang_error
from molmo_spaces.utils.mj_model_and_data_utils import body_pose


class RBY1ArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """Implementation of the RBY1's 7-DOF arm, excluding the gripper."""

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base: RobotBaseGroup,
        namespace: str = "",
    ) -> None:
        """Initialize an RBY1 arm.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            side: Which side of the robot this arm is on ("left" or "right")
            namespace: Optional prefix for all joint/body names to support multiple robots
        """
        model = mj_data.model
        self.side = side
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}{side}_arm_{i}").id for i in range(7)]
        act_ids = [model.actuator(f"{namespace}{side}_arm_{i + 1}_act").id for i in range(7)]
        self._ee_site_id = model.site(f"{namespace}ee_site_{side[0]}").id
        self._arm_root_id = model.body(f"{namespace}link_{side}_arm_0").id
        super().__init__(mj_data, joint_ids, act_ids, self._arm_root_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._arm_root_id)


class RBY1GripperGroup(MJCFFrameMixin, GripperGroup):
    """Implementation of the RBY1's gripper.

    The RBY1 gripper has 2 fingers that are mechanically coupled, allowing for open/close motion of
    the two fingers.
    """

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base: RobotBaseGroup,
        namespace: str = "",
    ) -> None:
        """Initialize an RBY1 gripper.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            side: Which side of the robot this gripper is on ("left" or "right")
            namespace: Optional prefix for all joint/body names to support multiple robots
        """
        model = mj_data.model
        self._side = side
        self._namespace = namespace
        joint_ids = [
            model.joint(f"{namespace}gripper_finger_{side[0]}{i + 1}").id for i in range(2)
        ]
        act_ids = [model.actuator(f"{namespace}{side}_finger_act").id]
        root_body_id = model.body(f"{namespace}EE_BODY_{side[0].upper()}")
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base)
        self._ee_site_id = model.site(f"{namespace}ee_site_{side[0]}").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = np.array([-0.05 if open else 0.0])

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return 0.0, 0.1

    @property
    def inter_finger_dist(self) -> float:
        return np.abs(self.joint_pos).sum().item()

    @property
    def ctrl(self) -> np.ndarray:
        """Current control signals for the gripper actuators."""
        return self.mj_data.ctrl[self._actuator_ids]

    @ctrl.setter
    def ctrl(self, ctrl: np.ndarray) -> None:
        """Set control signals for the gripper actuators.

        Args:
            ctrl: Array of control signals to set
        """
        # The gripper actuators are coupled, so we have two joints but one actuator
        # If two control inputs are provided, we assume it is two joint position inputs and we only use the first one
        if len(ctrl) == 2:
            ctrl = ctrl[0]

        self.mj_data.ctrl[self._actuator_ids] = ctrl

    @property
    def joint_pos(self) -> np.ndarray:
        """Current joint positions."""
        return self.mj_data.qpos[self._joint_posadr]

    @joint_pos.setter
    def joint_pos(self, joint_pos: np.ndarray) -> None:
        """Set joint positions with proper coupling for RBY1 gripper.

        Args:
            joint_pos: Array of joint positions to set
        """
        # For RBY1 gripper, we need to handle the coupling constraint
        # The gripper has 2 fingers that are mechanically coupled
        if len(joint_pos) == 1:
            # Single value provided - set both fingers according to coupling
            finger1_pos = joint_pos[0]
            finger2_pos = -finger1_pos  # Coupling constraint: finger2 = -finger1
            coupled_pos = np.array([finger1_pos, finger2_pos])
        else:
            # Two values provided - use them directly
            coupled_pos = joint_pos

        self.mj_data.qpos[self._joint_posadr] = coupled_pos

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class RBY1TorsoGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """Implementation of the RBY1's torso.

    The RBY1 torso has 6 degrees of freedom, allowing for full control of the
    upper body's position and orientation relative to the base.
    """

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        """Initialize the RBY1 torso.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            namespace: Optional prefix for all joint/body names to support multiple robots
        """
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}torso_{i}").id for i in range(6)]
        act_ids = [model.actuator(f"{namespace}link{i + 1}_act").id for i in range(6)]
        self._torso_root_id = model.body(f"{namespace}link_torso_0").id
        self._torso_leaf_id = model.body(f"{namespace}link_torso_5").id
        super().__init__(mj_data, joint_ids, act_ids, self._torso_root_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._torso_leaf_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._torso_root_id)


class RBY1BaseGroup(FreeJointRobotBaseGroup):
    """Implementation of the RBY1's mobile base.

    The RBY1 base uses a free joint for its pose and has two wheels for mobility.
    The wheels are controlled independently, allowing for differential drive motion.
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        """Initialize the RBY1 base.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            namespace: Optional prefix for all joint/body names to support multiple robots
        """
        model = mj_data.model
        base_joint_id = model.joint(f"{namespace}freejoint").id
        joints = [model.joint(f"{namespace}{side}_wheel").id for side in ["left", "right"]]
        act = [model.actuator(f"{namespace}{side}_wheel_act").id for side in ["left", "right"]]
        super().__init__(mj_data, base_joint_id, joints, act)

    @property
    def noop_ctrl(self) -> np.ndarray:
        return np.zeros(2)


class RBY1HoloBaseGroup(HoloJointsRobotBaseGroup):
    """Implementation of a RBY1 mobile base with virtual holonomic joints and site control.

    The RBY1 base uses three virtual holonomic joints for x, y and theta control.
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        """Initialize the RBY1 holo base.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            namespace: Optional prefix for all joint/body names to support multiple robots
        """
        model = mj_data.model
        world_site_id = model.site(f"{namespace}world").id
        holo_base_site_id = model.site(f"{namespace}base_site").id
        joints = [model.joint(f"{namespace}base_{axis}").id for axis in ["x", "y", "theta"]]
        act = [model.actuator(f"{namespace}base_{axis}_act").id for axis in ["x", "y", "theta"]]
        root_body_id = model.body(f"{namespace}base")
        super().__init__(mj_data, world_site_id, holo_base_site_id, joints, act, root_body_id)


class RBY1HeadGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """Implementation of the RBY1's head.

    The RBY1 head has 2 degrees of freedom, allowing for pan and tilt motion
    to look around the environment.
    """

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        """Initialize the RBY1 head.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            namespace: Optional prefix for all joint/body names to support multiple robots
        """
        model = mj_data.model
        joint_ids = [model.joint(f"{namespace}head_{i}").id for i in range(2)]
        act_ids = [model.actuator(f"{namespace}head_{i}_act").id for i in range(2)]
        root_body_id = model.body(f"{namespace}link_head_1").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base)
        self._head_root_id = model.body(f"{namespace}link_head_1").id
        self._head_leaf_id = model.body(f"{namespace}link_head_2").id

    @property
    def leaf_frame_id(self) -> int:
        return self._head_leaf_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._head_root_id)


class RBY1RobotView(RobotView):
    """Implementation of the complete RBY1 robot.

    The RBY1 is a humanoid robot with:
    - Two 7-DOF arms with grippers
    - A 6-DOF torso
    - A mobile base with two wheels (or three holonomic joints)
    - A 2-DOF head

    Each component is implemented as a MoveGroup, with the overall robot structure
    managed by this class.
    """

    def __init__(self, mj_data: MjData, namespace: str = "", holo_base: bool = False) -> None:
        """Initialize the RBY1 robot.

        Args:
            mj_data: The MuJoCo data structure containing the current simulation state
            namespace: Optional prefix for all joint/body names to support multiple robots
        """
        self._namespace = namespace
        base = (
            RBY1BaseGroup(mj_data, namespace=namespace)
            if not holo_base
            else RBY1HoloBaseGroup(mj_data, namespace=namespace)
        )
        move_groups = {
            "base": base,
            "torso": RBY1TorsoGroup(mj_data, base, namespace=namespace),
            "left_arm": RBY1ArmGroup(mj_data, "left", base, namespace=namespace),
            "right_arm": RBY1ArmGroup(mj_data, "right", base, namespace=namespace),
            "left_gripper": RBY1GripperGroup(mj_data, "left", base, namespace=namespace),
            "right_gripper": RBY1GripperGroup(mj_data, "right", base, namespace=namespace),
            "head": RBY1HeadGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(
            mj_data,
            move_groups,
        )

    @property
    def name(self) -> str:
        return "rby1"

    @property
    def base(self):
        base = self.get_move_group("base")
        # assert isinstance(base, RBY1BaseGroup)
        return base

    def get_joint_position(self, move_group_ids: list[str]) -> np.ndarray:
        """Get the current joint positions of the move groups"""
        return np.concatenate(
            [
                self.get_move_group(move_group_id).joint_pos.copy()
                for move_group_id in move_group_ids
            ]
        )

    def is_close_to(
        self, move_group_ids: list[str], target_pose: list, threshold: float = 0.05
    ) -> bool:
        """Check if the current joint positions of the move groups are close to the target pose"""
        return self.distance_to(move_group_ids, target_pose) < threshold

    def distance_to(self, move_group_ids: list[str], target_pose: list) -> float:
        """Calculate the distance between the current joint positions of the move groups and the target pose"""
        assert len(target_pose) == 3, f"Expected [x, y, theta] pose, got {target_pose}"
        current_joint_pos = self.get_joint_position(move_group_ids)
        x_delta = current_joint_pos[0] - target_pose[0]
        y_delta = current_joint_pos[1] - target_pose[1]
        theta_delta = normalize_ang_error(current_joint_pos[2] - target_pose[2])
        return float(np.linalg.norm(np.array([x_delta, y_delta, theta_delta])))
