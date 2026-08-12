from copy import deepcopy

import numpy as np
from mujoco import MjData, MjModel
from scipy.spatial.transform import Rotation as R

from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics


class FloatingRUMKinematics(MlSpacesKinematics):
    """
    Kinematics solver that provides a specialized implementation for the Floating RUM robot.
    """

    def ik(
        self,
        move_group_id: str,
        pose: np.ndarray,
        unlocked_move_group_ids: list[str] | None,
        q0: dict[str, np.ndarray],
        base_pose: np.ndarray,
        rel_to_base: bool = False,
        eps: float = 1e-4,
        max_iter: int = 1000,
        damping: float = 1e-12,
        dt: float = 1.0,
    ):
        if move_group_id == "gripper":
            ee_to_base = self._robot_view.get_move_group("gripper").leaf_frame_to_robot
            pose = pose @ np.linalg.inv(ee_to_base)
        else:
            assert move_group_id == "base"

        if unlocked_move_group_ids is None:
            unlocked_move_group_ids = self._robot_view.move_group_ids()

        if "base" not in unlocked_move_group_ids:
            return None

        if rel_to_base:
            pose = base_pose @ pose
        pos = pose[:3, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        ret = deepcopy(q0)
        ret["base"] = np.concatenate([pos, quat])
        return ret


if __name__ == "__main__":
    import time

    import mujoco
    import numpy as np
    from mujoco.viewer import Handle, launch_passive

    from molmo_spaces.configs.robot_configs import FloatingRUMRobotConfig
    from molmo_spaces.robots.robot_views.rum_gripper_view import FloatingRUMRobotView
    from molmo_spaces.utils.pose import pos_quat_to_pose_mat

    def _show_poses(viewer: Handle, poses: np.ndarray, color=(1, 0, 0, 1)) -> None:
        assert poses.ndim == 3 and poses.shape[1:] == (4, 4)
        ngeom = viewer.user_scn.ngeom
        # Define relative parts of the gripper

        gripper_parts = [
            ("sphere", mujoco.mjtGeom.mjGEOM_CYLINDER, [0.003, 0.015, 0], [0.00, 0.00, 0.05]),
            (
                "cylinder_left",
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                [0.003, 0.02, 0],
                [0, -0.046, 0.0835],
            ),
            (
                "cylinder_right",
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                [0.003, 0.02, 0],
                [0, 0.046, 0.0835],
            ),
            ("connecting_bar", mujoco.mjtGeom.mjGEOM_BOX, [0.002, 0.044, 0.002], [0, 0, 0.065]),
        ]

        i = 0
        for T in poses:
            for _part_name, geom_type, size, offset in gripper_parts:
                T_part = pos_quat_to_pose_mat(offset, [1, 0, 0, 0])
                A = T.copy() @ T_part
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[ngeom + i],
                    type=geom_type,
                    size=np.array(size),
                    pos=A[:3, 3],
                    mat=T[:3, :3].flatten(),
                    rgba=color,
                )
                i += 1
        viewer.user_scn.ngeom = ngeom + i

    def main():
        np.set_printoptions(linewidth=np.inf)

        robot_config = FloatingRUMRobotConfig()
        xml_path = robot_config.get_robot_xml_path()
        model = MjModel.from_xml_path(str(xml_path))

        data = MjData(model)
        mujoco.mj_forward(model, data)

        ns = ""
        robot_view = FloatingRUMRobotView(data, ns)
        kinematics = FloatingRUMKinematics(robot_config)

        pose0 = np.array(
            [
                [0.7361, -0.6769, 0.0, 0.5325],
                [0.6769, 0.7361, -0.0, -0.0325],
                [0.0, 0.0, 1.0, 0.0734],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        pose1 = np.array(
            [
                [0.23935042, 0.84872045, 0.47157711, 1.05149006],
                [0.92900645, -0.34137376, 0.14286698, 0.4176159],
                [0.28223818, 0.40390291, -0.87017472, 0.88686037],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        robot_view.base.pose = pose0
        mujoco.mj_forward(model, data)

        groups = [
            "base",
        ]

        with launch_passive(model, data) as viewer:
            _show_poses(viewer, np.array([pose0, pose1]))
            viewer.sync()
            i = 0
            while viewer.is_running():
                if i % 2 == 0:
                    ret = kinematics.ik(
                        "gripper",
                        pose1,
                        groups,
                        robot_view.get_qpos_dict(),
                        robot_view.base.pose,
                        dt=0.1,
                    )
                else:
                    ret = kinematics.ik(
                        "gripper", pose0, groups, robot_view.get_qpos_dict(), robot_view.base.pose
                    )
                print(i % 2, ret)
                i += 1
                if ret is not None:
                    robot_view.set_qpos_dict(ret)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(3)

    main()
