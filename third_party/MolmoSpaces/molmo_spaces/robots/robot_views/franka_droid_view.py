from functools import cached_property

import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.robots.robot_views.abstract import (
    FreeJointRobotBaseGroup,
    GripperGroup,
    MJCFFrameMixin,
    RobotView,
)
from molmo_spaces.robots.robot_views.franka_fr3_view import FrankaFR3ArmGroup, FrankaFR3BaseGroup


class RobotIQGripperGroup(MJCFFrameMixin, GripperGroup):
    def __init__(
        self, mj_data: MjData, base_group: FrankaFR3BaseGroup, namespace: str = ""
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [
            model.joint(f"{namespace}gripper/left_driver_joint").id,
            model.joint(f"{namespace}gripper/right_driver_joint").id,
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


class FrankaDroidRobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = FrankaFR3BaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "arm": FrankaFR3ArmGroup(
                mj_data, base, namespace=namespace, grasp_site_name="gripper/grasp_site"
            ),
            "gripper": RobotIQGripperGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}franka_droid"

    @property
    def base(self) -> FrankaFR3BaseGroup:
        return self._move_groups["base"]


class FloatingRobotiqGripperGroup(RobotIQGripperGroup):
    def __init__(
        self, mj_data: MjData, base_group: FrankaFR3BaseGroup, namespace: str = ""
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [
            model.joint(f"{namespace}left_driver_joint").id,
            model.joint(f"{namespace}right_driver_joint").id,
        ]
        act_ids = [model.actuator(f"{namespace}fingers_actuator").id]
        root_body_id = model.body(f"{namespace}base").id
        GripperGroup.__init__(self, mj_data, joint_ids, act_ids, root_body_id, base_group)
        self._ee_site_id = model.site(f"{namespace}grasp_site").id
        self._finger_1_geom_id = model.geom(f"{namespace}left_pad2").id
        self._finger_2_geom_id = model.geom(f"{namespace}right_pad2").id


class FloatingRobotiq2f85BaseGroup(FreeJointRobotBaseGroup):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        base_joint_id = model.joint(f"{namespace}robotiq_2f85_free_joint").id
        self._target_pose_body = create_mlspaces_body(mj_data, f"{namespace}target_ee_pose")
        super().__init__(mj_data, base_joint_id, [], [], floating=True)

    def is_mobile(self):
        return True

    def n_actuators(self):
        return 7

    @property
    def ctrl(self):
        ret = np.zeros(7)
        ret[:3] = self._target_pose_body.position
        ret[3:] = self._target_pose_body.quat
        return ret

    @ctrl.setter
    def ctrl(self, ctrl: np.ndarray):
        self._target_pose_body.position = ctrl[:3]
        self._target_pose_body.quat = ctrl[3:]

    @property
    def noop_ctrl(self):
        return self.joint_pos.copy()

    @cached_property
    def ctrl_limits(self):
        ctrl_range = np.empty((self.n_actuators, 2))
        ctrl_range[:, 0] = -np.inf
        ctrl_range[:, 1] = np.inf
        return ctrl_range


class FloatingRobotiq2f85RobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = FloatingRobotiq2f85BaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "gripper": FloatingRobotiqGripperGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}floating_robotiq"

    @property
    def base(self) -> FloatingRobotiq2f85BaseGroup:
        return self._move_groups["base"]
