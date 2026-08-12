import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    MJCFFrameMixin,
    RobotView,
)
from molmo_spaces.robots.robot_views.franka_fr3_view import FrankaFR3ArmGroup, FrankaFR3BaseGroup


class CAPGripperGroup(MJCFFrameMixin, GripperGroup):
    def __init__(
        self, mj_data: MjData, base_group: FrankaFR3BaseGroup, namespace: str = ""
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [
            model.joint(f"{namespace}gripper/finger_left_joint").id,
            model.joint(f"{namespace}gripper/finger_right_joint").id,
        ]
        act_ids = [model.actuator(f"{namespace}gripper/fingers_actuator").id]
        root_body_id = model.body(f"{namespace}gripper/base").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base_group)
        self._ee_site_id = model.site(f"{namespace}gripper/grasp_site").id
        self._finger_1_geom_id = model.geom(f"{namespace}gripper/left_pad2").id
        self._finger_2_geom_id = model.geom(f"{namespace}gripper/right_pad2").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = [0 if open else 255]

    @property
    def inter_finger_dist(self) -> float:
        dist = mujoco.mj_geomDistance(
            self.mj_model, self.mj_data, self._finger_1_geom_id, self._finger_2_geom_id, 0.1, None
        )
        return max(0.0, dist)

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return 0.0, 0.087

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class FrankaCAPRobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = FrankaFR3BaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "arm": FrankaFR3ArmGroup(
                mj_data, base, namespace=namespace, grasp_site_name="gripper/grasp_site"
            ),
            "gripper": CAPGripperGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}franka_droid"

    @property
    def base(self) -> FrankaFR3BaseGroup:
        return self._move_groups["base"]
