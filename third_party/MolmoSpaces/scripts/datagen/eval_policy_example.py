import argparse
import datetime
import logging
from typing import Any
from pathlib import Path

import numpy as np

import molmo_spaces.configs.policy_configs_baselines  # noqa: F401
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.policy_configs_baselines import PiPolicyConfig
from molmo_spaces.configs.robot_configs import ActionNoiseConfig
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.configs.abstract_config import Config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("websockets").setLevel(logging.WARNING)


class MyRunner(ParallelRolloutRunner):
    @staticmethod
    def patch_config(frozen_config: Config, data: Any = None, exp_config: Config = None) -> Config:
        # Overload this to patch up the saved frozen configs, i.e. if you add varaibles to your config
        # When overwriting this function make sure that it can be pickled for multiprocessing.
        exp_config.policy_config.tcp_pos_err_threshold = 1  # some old policies need these
        exp_config.policy_config.tcp_rot_err_threshold = np.radians(90.0)

        # Currently by default frozen configs don't save robot_cls, robot_factory, robot_view_factory
        # as I wasn't sure these can always be pickled.
        assert frozen_config.robot_config.robot_cls is None
        # Get our current robot from experiment config and populate our "task"
        robot_cls = exp_config.robot_config.robot_cls
        robot_factory = exp_config.robot_config.robot_factory
        robot_view_factory = exp_config.robot_config.robot_view_factory
        assert None not in (robot_cls, robot_factory, robot_view_factory)  # make sure its there
        frozen_config.robot_config.robot_cls = robot_cls
        frozen_config.robot_config.robot_factory = robot_factory
        frozen_config.robot_config.robot_view_factory = robot_view_factory

        # Update the robot
        new_command_mode = {"arm": "joint_velocity", "gripper": "joint_position"}
        frozen_config.robot_config.command_mode = new_command_mode

        # check pydantic
        robot_config_copy = frozen_config.robot_config.model_copy(deep=True)
        assert robot_config_copy.command_mode == new_command_mode
        return frozen_config


def main(args: argparse.ArgumentParser) -> None:
    log.info(f"Loading pre-saved config from {args.eval}. This will override other settings.")
    exp_config = MlSpacesExpConfig.load_config(Path(args.eval))
    exp_config.frozen_config_path = Path(args.eval)
    exp_config.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)  # for eval
    if not hasattr(exp_config, "task_type"):
        object.__setattr__(exp_config, "task_type", "default")
    exp_config.num_workers = 1
    exp_config.output_dir = (
        ASSETS_DIR
        / "datagen"
        / f"{exp_config.task_type}_v1"
        / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    exp_config.save_config()
    runner = MyRunner(exp_config)
    runner.run()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--eval", type=str, help="Load a fixed benchmark")
    args = args.parse_args()
    main(args)
