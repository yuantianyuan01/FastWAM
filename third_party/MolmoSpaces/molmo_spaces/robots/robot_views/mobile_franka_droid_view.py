from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    HoloJointsRobotBaseGroup,
    RobotView,
)
from molmo_spaces.robots.robot_views.franka_droid_view import RobotIQGripperGroup
from molmo_spaces.robots.robot_views.franka_fr3_view import FrankaFR3ArmGroup


class MobileFrankaDroidBaseGroup(HoloJointsRobotBaseGroup):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        world_site_id = model.site(f"{namespace}world").id
        holo_base_site_id = model.site(f"{namespace}base_site").id
        joints = [model.joint(f"{namespace}base_{axis}").id for axis in ["x", "y", "theta"]]
        act = [model.actuator(f"{namespace}base_{axis}_act").id for axis in ["x", "y", "theta"]]
        root_body_id = model.body(f"{namespace}base").id
        super().__init__(mj_data, world_site_id, holo_base_site_id, joints, act, root_body_id)


class MobileFrankaDroidRobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = MobileFrankaDroidBaseGroup(mj_data, namespace=namespace)
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
        return f"{self._namespace}mobile_franka_droid"

    @property
    def base(self) -> MobileFrankaDroidBaseGroup:
        return self._move_groups["base"]
