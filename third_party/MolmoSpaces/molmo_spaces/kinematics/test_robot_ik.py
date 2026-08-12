"""
Interactive script to test a robot with the parallel and non-parallel IK solvers.

On mac, run with mjpython.
"""

import argparse
import importlib
import time

import mujoco
from mujoco.viewer import launch_passive

from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_class", help="The class name of the robot config to use, e.g. 'FrankaRobotConfig'"
    )
    parser.add_argument(
        "--config_module",
        default="molmo_spaces.configs.robot_configs",
        help="The module name the robot config is in, defaults to 'molmo_spaces.configs.robot_configs'",
    )
    parser.add_argument(
        "--move-group",
        help="The move group to test IK for, defaults to the first gripper move group",
    )
    parser.add_argument(
        "--unlocked-move-groups",
        nargs="+",
        help="The move groups to unlock for the IK solver, defaults to all move groups",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use the parallel IK solver instead of the non-parallel one",
    )
    args = parser.parse_args()

    config_module = importlib.import_module(args.config_module)
    config_class = getattr(config_module, args.config_class)
    robot_config: BaseRobotConfig = config_class()

    if args.parallel:
        kinematics = SimpleWarpKinematics(robot_config)
    else:
        kinematics = MlSpacesKinematics(robot_config)

    spec = mujoco.MjSpec()
    robot_config.robot_cls.add_robot_to_scene(
        robot_config,
        spec,
        prefix="",
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    robot_view = robot_config.robot_view_factory(data, "")

    robot_view.set_qpos_dict(robot_config.init_qpos)
    mujoco.mj_forward(model, data)

    if args.move_group is None:
        move_group_id = robot_view.get_gripper_movegroup_ids()[0]
        move_group = robot_view.get_move_group(move_group_id)
    else:
        move_group_id = args.move_group
        move_group = robot_view.get_move_group(move_group_id)

    pose0 = move_group.leaf_frame_to_world.copy()
    pose1 = pose0.copy()
    pose0[2, 3] += 0.05  # Move up 5cm
    pose1[2, 3] -= 0.05  # Move down 5cm

    with launch_passive(model, data) as viewer:
        viewer.sync()
        i = 0
        while viewer.is_running():
            # Alternate between two target poses
            target_pose = pose1 if i % 2 == 0 else pose0
            ret = kinematics.ik(
                move_group_id,
                target_pose,
                args.unlocked_move_groups,
                robot_config.init_qpos,
                robot_view.base.pose,
            )
            print(f"IK iteration {i}: {'Success' if ret is not None else 'Failed'}")
            i += 1
            if ret is not None:
                robot_view.set_qpos_dict(ret)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(2)


if __name__ == "__main__":
    main()
