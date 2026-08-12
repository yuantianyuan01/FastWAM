"""
Implementation of the bimanual YAM robot model.

The bimanual YAM consists of two 6-DOF YAM arms with parallel grippers,
positioned side by side (44cm apart), both facing forward.

Each arm component is implemented as a MoveGroup (left_arm, right_arm,
left_gripper, right_gripper), with the overall robot structure managed
by the BimanualYamRobotView class.
"""

from typing import Literal

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    MJCFFrameMixin,
    MocapRobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose


class BimanualYamBaseGroup(MocapRobotBaseGroup):
    """Base group for the bimanual YAM robot using mocap body control.

    The mocap body is created by add_robot_to_scene and serves as the
    root for both arms.
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        body_id: int = mj_data.model.body(f"{namespace}base").id
        super().__init__(mj_data, body_id)


class BimanualYamArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """6-DOF arm group for one side of the bimanual YAM robot."""

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base_group: BimanualYamBaseGroup,
        namespace: str = "",
        grasp_site_name: str = "grasp_site",
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        self._side = side
        # Prefix for this arm: namespace + side_
        self._arm_prefix = f"{namespace}{side}_"

        # 6 arm joints: joint1 through joint6
        joint_ids = [model.joint(f"{self._arm_prefix}joint{i + 1}").id for i in range(6)]
        # 6 arm actuators with the same names
        act_ids = [model.actuator(f"{self._arm_prefix}joint{i + 1}").id for i in range(6)]
        self._arm_root_id = model.body(f"{self._arm_prefix}arm").id
        self._ee_site_id = model.site(f"{self._arm_prefix}{grasp_site_name}").id
        super().__init__(mj_data, joint_ids, act_ids, self._arm_root_id, base_group)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def side(self) -> str:
        return self._side

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._arm_root_id)


class BimanualYamGripperGroup(MJCFFrameMixin, GripperGroup):
    """Parallel gripper group for one side of the bimanual YAM robot.

    The YAM gripper has two coupled fingers (left_finger and right_finger)
    controlled by a single actuator. The fingers move in opposite directions
    due to an equality constraint.
    """

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base_group: BimanualYamBaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        self._side = side
        # Prefix for this gripper: namespace + side_
        self._gripper_prefix = f"{namespace}{side}_"

        # Two coupled finger joints
        joint_ids = [
            model.joint(f"{self._gripper_prefix}left_finger").id,
            model.joint(f"{self._gripper_prefix}right_finger").id,
        ]
        # Single gripper actuator controls left_finger (right follows via equality constraint)
        act_ids = [model.actuator(f"{self._gripper_prefix}gripper").id]
        root_body_id = model.body(f"{self._gripper_prefix}link_6").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base_group)
        self._ee_site_id = model.site(f"{self._gripper_prefix}grasp_site").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def side(self) -> str:
        return self._side

    def set_gripper_ctrl_open(self, open: bool) -> None:
        """Set gripper to fully open or closed.

        From XML: gripper actuator ctrlrange is 0.0 to 0.041
        - 0.0 = closed
        - 0.041 = open
        """
        self.ctrl = [0.041 if open else 0.0]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        """The (min, max) of the distance between the two fingers.

        From XML:
        - left_finger range: -0.00205 to 0.037524
        - right_finger range: -0.037524 to 0.00205 (mirrored)
        - Closed (both at 0): dist = 0
        - Open (left at 0.037524, right at -0.037524): dist ~= 0.075
        """
        return 0.0, 0.075

    @property
    def inter_finger_dist(self) -> float:
        """Distance between fingers.

        Since fingers are coupled in opposite directions (right = -left),
        the total opening is the sum of absolute positions.
        """
        return np.abs(self.joint_pos[0]) + np.abs(self.joint_pos[1])

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class BimanualYamRobotView(RobotView):
    """Robot view for the bimanual YAM (two 6-DOF arms with parallel grippers).

    Move groups:
    - base: Mocap body controlling both arms
    - left_arm: Left 6-DOF arm
    - right_arm: Right 6-DOF arm
    - left_gripper: Left parallel gripper
    - right_gripper: Right parallel gripper
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = BimanualYamBaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "left_arm": BimanualYamArmGroup(mj_data, "left", base, namespace=namespace),
            "right_arm": BimanualYamArmGroup(mj_data, "right", base, namespace=namespace),
            "left_gripper": BimanualYamGripperGroup(mj_data, "left", base, namespace=namespace),
            "right_gripper": BimanualYamGripperGroup(mj_data, "right", base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}bimanual_yam"

    @property
    def base(self) -> BimanualYamBaseGroup:
        return self._move_groups["base"]
