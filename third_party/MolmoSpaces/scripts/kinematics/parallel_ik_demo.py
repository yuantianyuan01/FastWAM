"""
Demo script that demonstrates the parallel IK solver for multiple robots simultaneously.

On mac, run with mjpython.
"""

from collections import defaultdict

import numpy as np
import mujoco
from mujoco import MjModel, MjData, MjSpec
from mujoco.viewer import launch_passive

from molmo_spaces.configs import BaseRobotConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig, I2rtYamRobotConfig, RBY1MConfig
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics
from molmo_spaces.robots.robot_views.abstract import RobotView

N_ROBOTS = 4
PERIOD = 3.0
RADIUS = 0.1
FPS = 20

MJCF = """
<mujoco model="scene">
    <option timestep="0.002" integrator="implicit" cone="elliptic" impratio="10" />
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="120" elevation="-20" />
    </visual>

    <asset>
        <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="512" height="512" />
        <material name="floor_mat" texture="floor_tex" texrepeat="5 5" texuniform="true" reflectance="0.2" />
    </asset>

    <worldbody>
        <light pos="0 0 5" dir="0 0 -1" directional="true" castshadow="true" diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3" />
        <light pos="0 -3 3" dir="0 1 -1" directional="false" castshadow="false" diffuse="0.3 0.3 0.3" specular="0.1 0.1 0.1" />

        <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat" />
    </worldbody>
</mujoco>
"""


def main():
    spec = MjSpec.from_string(MJCF)

    robot_configs: dict[str, BaseRobotConfig] = {
        str(rc.name): rc for rc in [I2rtYamRobotConfig(), FrankaRobotConfig(), RBY1MConfig()]
    }
    robot_configs["rby1m"].init_qpos["base"] = [0.0, 2.0, 0.0]

    for robot_config in robot_configs.values():
        for i in range(N_ROBOTS):
            robot_config.robot_cls.add_robot_to_scene(
                robot_config,
                spec,
                prefix=f"{robot_config.name}_{i}/",
                pos=[0.0, 0.0, 0.0],
                quat=[1.0, 0.0, 0.0, 0.0],
            )

    model: MjModel = spec.compile()
    data = MjData(model)
    mujoco.mj_forward(model, data)

    robot_views: dict[str, list[RobotView]] = {}
    for robot_idx, (name, robot_config) in enumerate(robot_configs.items()):
        views = []
        for i in range(N_ROBOTS):
            view = robot_config.robot_view_factory(data, f"{robot_config.name}_{i}/")
            views.append(view)
            view.set_qpos_dict(robot_config.init_qpos)
            pose = view.base.pose.copy()
            pose[1, 3] = robot_idx
            view.base.pose = pose
        robot_views[name] = views

    mujoco.mj_forward(model, data)

    robot_grippers = [
        ("franka_droid", "gripper"),
        ("i2rt_yam", "gripper"),
        ("rby1m", "left_gripper"),
        ("rby1m", "right_gripper"),
    ]
    unlocked_move_groups = {
        "rby1m": {"left_gripper": ["left_arm", "base"], "right_gripper": ["right_arm"]},
        "franka_droid": {"gripper": ["arm"]},
        "i2rt_yam": {"gripper": ["arm"]},
    }

    gripper_init_poses: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for name, move_group_id in robot_grippers:
        gripper_init_poses[name][move_group_id] = (
            robot_views[name][0].get_move_group(move_group_id).leaf_frame_to_world
        )

    kinematics = {name: SimpleWarpKinematics(rc) for name, rc in robot_configs.items()}

    t = 0.0
    t_offsets = np.linspace(0, 2 * np.pi, N_ROBOTS, endpoint=False)
    with launch_passive(model, data) as viewer:
        while viewer.is_running():
            for name, views in robot_views.items():
                for view in views:
                    view.set_qpos_dict(robot_configs[name].init_qpos)

            for name in gripper_init_poses:
                kin = kinematics[name]
                for move_group_id, init_pose in gripper_init_poses[name].items():
                    target_poses = np.repeat(init_pose[None], N_ROBOTS, axis=0)
                    target_poses[:, 0, 3] += (
                        RADIUS * 1.5 * np.sin(t * 2 * np.pi / PERIOD / 4 + t_offsets)
                    )
                    target_poses[:, 1, 3] += RADIUS * np.sin(t * 2 * np.pi / PERIOD + t_offsets)
                    target_poses[:, 2, 3] += RADIUS * np.cos(t * 2 * np.pi / PERIOD + t_offsets)

                    move_groups = unlocked_move_groups[name][move_group_id]
                    rets = kin.ik(
                        move_group_id,
                        target_poses,
                        move_groups,
                        [view.get_qpos_dict() for view in robot_views[name]],
                        robot_views[name][0].base.pose,
                        rel_to_base=False,
                    )

                    for i, view in enumerate(robot_views[name]):
                        view.set_qpos_dict({k: v for k, v in rets[i].items() if k in move_groups})

            mujoco.mj_kinematics(model, data)
            viewer.sync()
            t += 1 / FPS


if __name__ == "__main__":
    main()
