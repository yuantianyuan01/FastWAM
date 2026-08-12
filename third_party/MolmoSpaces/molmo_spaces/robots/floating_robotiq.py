from typing import TYPE_CHECKING

import numpy as np
from mujoco import MjSpec, mjtEq, mjtObj

from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.floating_rum import FloatingRUMRobot

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig


class FloatingRobotiqRobot(FloatingRUMRobot):
    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "BaseRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        pos = pos + [0.0] if len(pos) == 2 else pos
        # call grandparent class method (Robot) directly, skipping FloatingRUMRobot
        Robot.add_robot_to_scene.__func__(
            cls,
            robot_config=robot_config,
            spec=spec,
            prefix=prefix,
            pos=pos,
            quat=quat,
            strip_meshes=strip_meshes,
        )

        # add target pose body and weld to base
        target_body_name = f"{prefix}target_ee_pose"
        spec.worldbody.add_body(name=target_body_name, pos=pos, quat=quat, mocap=True)
        eq = spec.add_equality()
        eq.name1 = target_body_name
        eq.name2 = f"{prefix}{cls.robot_model_root_name()}"
        # Stronger weld constraint: lower solref[0] = stiffer (0.001 is much stiffer than 0.02)
        # This makes the mocap body follow the target pose more precisely, allowing better lifting

        # best for lift to pick objects strongly
        eq.solref = np.array([0.001, 1])
        eq.solimp = np.array([0.99, 0.99, 0.001, 1, 2])

        # best for articulation where it's compliant``
        # eq.solref = np.array([0.01, 1])
        # eq.solimp = np.array([0.8, 0.8, 0.05, 1, 2])

        # in between?
        # eq.solref = np.array([0.0025, 1])
        # eq.solimp = np.array([0.99, 0.85, 0.01, 1, 2])

        eq.objtype = mjtObj.mjOBJ_BODY
        eq.type = mjtEq.mjEQ_WELD
