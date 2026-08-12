"""
Implementation of the i2rt YAM robot model.

The YAM is a 6-DOF arm robot with a parallel gripper that has coupled fingers.

Each component is implemented as a MoveGroup, with the overall robot structure
managed by the I2rtYamRobotView class.
"""

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


class I2rtYamBaseGroup(MocapRobotBaseGroup):
    """Base group for the YAM robot using mocap body control.

    Note: The mocap body is created by add_robot_to_scene and named 'base',
    not 'arm'. The 'arm' body from the XML is attached under this mocap body.
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        body_id: int = mj_data.model.body(f"{namespace}base").id
        super().__init__(mj_data, body_id)


class I2rtYamArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """6-DOF arm group for the YAM robot."""

    def __init__(
        self,
        mj_data: MjData,
        base_group: I2rtYamBaseGroup,
        namespace: str = "",
        grasp_site_name: str = "grasp_site",
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        # 6 arm joints: joint1 through joint6
        joint_ids = [model.joint(f"{namespace}joint{i + 1}").id for i in range(6)]
        # 6 arm actuators with the same names
        act_ids = [model.actuator(f"{namespace}joint{i + 1}").id for i in range(6)]
        self._arm_root_id = model.body(f"{namespace}arm").id
        self._ee_site_id = model.site(f"{namespace}{grasp_site_name}").id
        super().__init__(mj_data, joint_ids, act_ids, self._arm_root_id, base_group)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._arm_root_id)


class I2rtYamGripperGroup(MJCFFrameMixin, GripperGroup):
    """Parallel gripper group for the YAM robot.

    The YAM gripper has two coupled fingers (left_finger and right_finger)
    controlled by a single actuator. The fingers move in opposite directions
    due to an equality constraint: right_finger = -left_finger.
    """

    def __init__(self, mj_data: MjData, base_group: I2rtYamBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        # Two coupled finger joints
        joint_ids = [
            model.joint(f"{namespace}left_finger").id,
            model.joint(f"{namespace}right_finger").id,
        ]
        # Single gripper actuator controls left_finger (right follows via equality constraint)
        act_ids = [model.actuator(f"{namespace}gripper").id]
        root_body_id = model.body(f"{namespace}link_6").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base_group)
        self._ee_site_id = model.site(f"{namespace}grasp_site").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

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


class I2rtYamRobotView(RobotView):
    """Robot view for the i2rt YAM 6-DOF arm with parallel gripper."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = I2rtYamBaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "arm": I2rtYamArmGroup(mj_data, base, namespace=namespace),
            "gripper": I2rtYamGripperGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}i2rt_yam"

    @property
    def base(self) -> I2rtYamBaseGroup:
        return self._move_groups["base"]
