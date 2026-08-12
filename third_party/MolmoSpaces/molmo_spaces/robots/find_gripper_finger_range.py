"""
Script to find the inter finger distance range for all gripper move groups in a robot.
This is useful for adding new robots to MolmoSpaces.
"""

import argparse
import importlib

import mujoco
from mujoco import MjData, MjModel, MjSpec

from molmo_spaces.configs.robot_configs import BaseRobotConfig


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "robot_config_module", type=str, help="Python module which contains the robot config class"
    )
    parser.add_argument("robot_config_class", type=str, help="Class name of the robot config")
    parser.add_argument(
        "--settle-time",
        type=float,
        default=2.0,
        help="Amount of simulation time to let the gripper go to the target (default: 2.0s)",
    )
    return parser.parse_args()


def main():
    args = get_args()
    robot_config_module = importlib.import_module(args.robot_config_module)
    robot_config_class = getattr(robot_config_module, args.robot_config_class)

    robot_config: BaseRobotConfig = robot_config_class()

    spec = MjSpec()
    robot_config.robot_cls.add_robot_to_scene(
        robot_config,
        spec,
        prefix=robot_config.robot_namespace,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )

    model: MjModel = spec.compile()
    data = MjData(model)

    robot_view = robot_config.robot_view_factory(data, robot_config.robot_namespace)

    mujoco.mj_forward(model, data)
    gripper_names = robot_view.get_gripper_movegroup_ids()
    gripper_groups = [robot_view.get_gripper(gripper_name) for gripper_name in gripper_names]

    for gripper_group in gripper_groups:
        gripper_group.set_gripper_ctrl_open(True)

    mujoco.mj_step(model, data, round(1.0 / model.opt.timestep))

    dists_hi = [gripper.inter_finger_dist for gripper in gripper_groups]

    for gripper in gripper_groups:
        gripper.set_gripper_ctrl_open(False)

    mujoco.mj_step(model, data, round(1.0 / model.opt.timestep))

    dists_lo = [gripper.inter_finger_dist for gripper in gripper_groups]

    print("Gripper finger ranges:")
    for gripper_name, dist_lo, dist_hi in zip(gripper_names, dists_lo, dists_hi):
        print(f"\tGripper '{gripper_name}': {dist_lo:.3f} - {dist_hi:.3f}")


if __name__ == "__main__":
    main()
