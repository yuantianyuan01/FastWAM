"""
Implementation of the Franka FR3 robot model.

The Franka FR3 is a single-arm 7-DOF robot with a gripper.

Each component is implemented as a MoveGroup, with the overall robot structure
managed by the FrankaFR3RobotView class.
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


class FrankaFR3BaseGroup(MocapRobotBaseGroup):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        body_id: int = mj_data.model.body(f"{namespace}base").id
        super().__init__(mj_data, body_id)


class FrankaFR3ArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    def __init__(
        self,
        mj_data: MjData,
        base_group: FrankaFR3BaseGroup,
        namespace: str = "",
        grasp_site_name: str = "grasp_site",
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}fr3_joint{i + 1}").id for i in range(7)]
        act_ids = [model.actuator(f"{namespace}fr3_joint{i + 1}").id for i in range(7)]
        self._arm_root_id = model.body(f"{namespace}fr3_link0").id
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


class FrankaFR3GripperGroup(MJCFFrameMixin, GripperGroup):
    def __init__(
        self, mj_data: MjData, base_group: FrankaFR3BaseGroup, namespace: str = ""
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [
            model.joint(f"{namespace}finger_joint1").id,
            model.joint(f"{namespace}finger_joint2").id,
        ]
        act_ids = [model.actuator(f"{namespace}panda_hand").id]
        root_body_id = model.body(f"{namespace}hand").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base_group)
        self._ee_site_id = model.site(f"{namespace}grasp_site").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = [255 if open else 0]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return 0.0, 0.08

    @property
    def inter_finger_dist(self) -> float:
        return np.sum(self.joint_pos).item()

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class FrankaFR3RobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = FrankaFR3BaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "arm": FrankaFR3ArmGroup(mj_data, base, namespace=namespace),
            "gripper": FrankaFR3GripperGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}franka_fr3"

    @property
    def base(self) -> FrankaFR3BaseGroup:
        return self._move_groups["base"]
