from functools import cached_property

import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.robots.robot_views.abstract import (
    FreeJointRobotBaseGroup,
    GripperGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
)


class RUMGripperGroup(MJCFFrameMixin, GripperGroup):
    def __init__(self, mj_data: MjData, base_group: RobotBaseGroup, namespace: str = ""):
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [
            model.joint(f"{namespace}finger_left_joint").id,
            model.joint(f"{namespace}finger_right_joint").id,
        ]
        actuator_ids = [model.actuator(f"{namespace}fingers_actuator").id]
        super().__init__(mj_data, joint_ids, actuator_ids, base_group.root_body_id, base_group)
        self._ee_site_id = model.site(f"{namespace}grasp_site").id
        self._left_fingertip_geom_id = model.geom(f"{namespace}left_fingertip").id
        self._right_fingertip_geom_id = model.geom(f"{namespace}right_fingertip").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool):
        self.ctrl = [0 if open else -255]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return 0.0, 0.1

    @property
    def inter_finger_dist(self) -> float:
        dist = mujoco.mj_geomDistance(
            self.mj_model,
            self.mj_data,
            self._left_fingertip_geom_id,
            self._right_fingertip_geom_id,
            0.1,
            None,
        )
        return max(0.0, dist)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class FloatingRUMBaseGroup(FreeJointRobotBaseGroup):
    """
    A robot base group for the floating RUM gripper.
    The gripper is controlled via a target pose mocap body.
    Since this doesn't correspond to an actuator, we "fake" the actuators by overriding the appropriate methods and properties.
    """

    def __init__(self, mj_data: MjData, namespace: str = ""):
        model = mj_data.model
        base_joint_id = model.joint(f"{namespace}rum_free_joint").id
        self._target_pose_body = create_mlspaces_body(mj_data, f"{namespace}target_ee_pose")
        super().__init__(mj_data, base_joint_id, [], [], floating=True)

    @cached_property
    def is_mobile(self):
        return True

    @cached_property
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


class FloatingRUMRobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = ""):
        self._namespace = namespace
        base = FloatingRUMBaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "gripper": RUMGripperGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self):
        return f"{self._namespace}rum_gripper"

    @property
    def base(self) -> FloatingRUMBaseGroup:
        return self._move_groups["base"]


if __name__ == "__main__":

    def main():
        import time

        import mujoco
        import numpy as np
        from mujoco import MjData, MjModel
        from mujoco.viewer import launch_passive

        from molmo_spaces.configs.robot_configs import FloatingRUMRobotConfig

        np.set_printoptions(linewidth=np.inf)

        robot_config = FloatingRUMRobotConfig()
        xml_path = str(robot_config.get_robot_xml_path())
        model = MjModel.from_xml_path(xml_path)

        data = MjData(model)
        mujoco.mj_forward(model, data)

        ns = ""
        robot_view = FloatingRUMRobotView(data, ns)
        robot_view.base.pose = np.eye(4)
        robot_view.base.ctrl = np.array([0, 0, 0, 1.0, 0, 0, 0])

        with launch_passive(model, data) as viewer:
            while viewer.is_running():
                viewer.sync()
                x = np.sin(data.time)
                y = 1 - np.cos(data.time)
                robot_view.base.ctrl = np.array([x, y, 0, 1.0, 0, 0, 0])
                mujoco.mj_step(model, data)
                time.sleep(0.02)

    main()
