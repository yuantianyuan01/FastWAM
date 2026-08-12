import argparse
import datetime
import logging
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import molmo_spaces.configs.policy_configs_baselines  # noqa: F401
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
from molmo_spaces.configs.base_open_task_configs import OpeningBaseConfig, ClosingBaseConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.policy_configs_baselines import (
    BimanualYamPiPolicyConfig,
    PiPolicyConfig,
    TeleopPolicyConfig,
)
from robot_conversion_patches import patch_droid_config_for_rum

from molmo_spaces.configs.camera_configs import (
    BimanualYamCameraSystem,
    FrankaDroidCameraSystem,
    FrankaRandomizedD405D455CameraSystem,
    RBY1GoProD455CameraSystem,
    I2rtYamCameraSystem,
)
from molmo_spaces.configs.policy_configs import (
    AStarNavToObjPolicyConfig,
    OpenClosePlannerPolicyConfig,
    PickAndPlaceNextToPlannerPolicyConfig,
    PickAndPlacePlannerPolicyConfig,
    PickPlannerPolicyConfig,
)
from molmo_spaces.configs.robot_configs import (
    FloatingRUMRobotConfig,
    FrankaRobotConfig,
    I2rtYamRobotConfig,
    RBY1Config,
    ActionNoiseConfig,
)
from molmo_spaces.configs.robot_configs import (
    BimanualYamRobotConfig,
    FloatingRUMRobotConfig,
    FrankaRobotConfig,
    I2rtYamRobotConfig,
    RBY1Config,
    ActionNoiseConfig,
)
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.configs.base_packing_configs import PackingDataGenConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaPickAndPlaceDroidDataGenConfig,
    FrankaPickAndPlaceColorDataGenConfig,
    FrankaPickAndPlaceColorDroidDataGenConfig,
    FrankaPickAndPlaceColorOmniCamConfig,
    FrankaPickAndPlaceNextToDroidDataGenConfig,
)
from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.profiler_utils import Profiler

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("websockets").setLevel(logging.WARNING)


class MyRolloutRunner(ParallelRolloutRunner):
    @staticmethod
    def patch_config(frozen_config, data: Any = None, exp_config: Any = None):
        if "FrankaRobotConfig" in str(type(frozen_config.robot_config)):
            if "rum" in str(type(exp_config.robot_config)).lower():
                return patch_droid_config_for_rum(frozen_config, data)

        return frozen_config

    @staticmethod
    def run_single_rollout(
        episode_seed: int,
        task: BaseMujocoTask,
        policy: Any,
        profiler: Profiler | None = None,
        viewer=None,
        shutdown_event=None,
        datagen_profiler: Profiler | None = None,
        end_on_success: bool = False,
    ):
        observation, _info = task.reset()  # don't update env after reset!

        # if observation[0]["object_image_points"]["pickup_obj"]["exo_camera_1"] == []:
        #   raise ValueError("Pickup object not visible in exo_camera_1")

        if policy.__class__.__name__ in ("PI_Policy", "RUM_Policy", "BimanualYamPiPolicy"):
            policy.prepare_model()
        if viewer is not None:
            viewer.sync()

        # Run episode
        success = False
        policy.task = task

        # enable sleep mode before executing the rollout
        try:
            task.env.current_model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        except AttributeError:
            log.warning("Not setting enable sleep")

        while not task.is_done():
            if shutdown_event is not None and shutdown_event.is_set():
                return False

            # Get action from policy and step the task
            action_cmd = policy.get_action(observation)
            if action_cmd is None:
                break
            observation, reward, terminal, truncated, infos = task.step(action_cmd)
            if end_on_success and "success" in infos[0] and infos[0]["success"]:
                success = True
                break

            if viewer is not None:
                viewer.sync()

        # disable sleep mode after the rollout
        try:
            task.env.current_model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        except AttributeError:
            log.warning("Not setting sleep")

        # Check success if method exists
        success = task.judge_success() if hasattr(task, "judge_success") else False

        return success


def setup_config(args: argparse.ArgumentParser) -> MlSpacesExpConfig:
    task_type = args.task_type  # pick or open or nav_to_obj
    robot = args.robot  #  droid or rum
    single_step = args.single_step

    if task_type == "pick":
        datagen_cfg = PickBaseConfig()
        datagen_cfg.policy_config = PickPlannerPolicyConfig()
    elif task_type == "open":
        datagen_cfg = OpeningBaseConfig()
        datagen_cfg.policy_config = OpenClosePlannerPolicyConfig()
    elif task_type == "close":
        datagen_cfg = ClosingBaseConfig()
        datagen_cfg.policy_config = OpenClosePlannerPolicyConfig()
    elif task_type == "pick_and_place":
        datagen_cfg = FrankaPickAndPlaceDroidDataGenConfig()
        datagen_cfg.policy_config = PickAndPlacePlannerPolicyConfig()
    elif task_type == "pick_and_place_color":
        datagen_cfg = FrankaPickAndPlaceColorDroidDataGenConfig()
        datagen_cfg.policy_config = PickAndPlacePlannerPolicyConfig()
    elif task_type == "pick_and_place_next_to":
        datagen_cfg = FrankaPickAndPlaceNextToDroidDataGenConfig()
        datagen_cfg.policy_config = PickAndPlaceNextToPlannerPolicyConfig()
    elif task_type == "packing":
        datagen_cfg = PackingDataGenConfig()
        datagen_cfg.policy_config = PickAndPlacePlannerPolicyConfig()
    elif task_type == "nav_to_obj":
        datagen_cfg = NavToObjBaseConfig()
        datagen_cfg.policy_config = AStarNavToObjPolicyConfig()
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    datagen_cfg.seed = args.seed
    datagen_cfg.scene_dataset = args.scene_dataset  # ithor, procthor-10k, procthor-objaverse
    datagen_cfg.data_split = args.data_split  # train or test
    datagen_cfg.task_type = task_type

    datagen_cfg.task_horizon = 300
    if args.target_types:
        datagen_cfg.task_sampler_config.pickup_types = args.target_types.split(",")
    datagen_cfg.task_sampler_config.samples_per_house = (
        args.samples_per_house
    )  # overwrite with scene samples

    # randomize scene
    datagen_cfg.task_sampler_config.randomize_lighting = (
        args.randomize_lighting or args.randomize_scene
    )
    datagen_cfg.task_sampler_config.randomize_textures = (
        args.randomize_textures or args.randomize_scene
    )
    datagen_cfg.task_sampler_config.randomize_dynamics = (
        args.randomize_dynamics or args.randomize_scene
    )

    # datagen_cfg.num_workers = args.samples_per_house if not args.use_passive_viewer else 1
    datagen_cfg.filter_for_successful_trajectories = (
        True if args.filter_for_successful_trajectories else False
    )  # if False, Save both successful and failed trajectories

    if args.eval is not None:
        datagen_cfg.frozen_config_path = Path(args.eval)
        datagen_cfg.seed = 42

    if args.house_inds is None:
        datagen_cfg.task_sampler_config.house_inds = None  # list(range(0,20))  # default is (0,20)
    elif isinstance(args.house_inds, int):
        datagen_cfg.task_sampler_config.house_inds = [args.house_inds]
    elif isinstance(args.house_inds, (list, tuple)):
        datagen_cfg.task_sampler_config.house_inds = args.house_inds
    else:
        raise ValueError()

    if robot == "franka":
        datagen_cfg.robot_config = FrankaRobotConfig()
        datagen_cfg.camera_config = FrankaRandomizedD405D455CameraSystem()
    elif robot == "droid":
        datagen_cfg.robot_config = FrankaRobotConfig()
        datagen_cfg.camera_config = FrankaDroidCameraSystem()
        # datagen_cfg.camera_config.img_resolution = (1280, 720)  # overrride for human videos
        datagen_cfg.policy_config.phase_timeout = 20.0
    elif robot == "rum":
        datagen_cfg.robot_config = FloatingRUMRobotConfig()
        datagen_cfg.task_sampler_config.robot_object_z_offset = 0
        datagen_cfg.task_sampler_config.base_pose_sampling_radius_range = (0, 0.8)
        datagen_cfg.task_sampler_config.robot_safety_radius = 0.2
        # datagen_cfg.camera_config.img_resolution = (960, 720)  # overrride for human videos
        datagen_cfg.policy_config.phase_timeout = 30.0
    elif robot == "rby1":
        datagen_cfg.robot_config = RBY1Config()
        datagen_cfg.camera_config = RBY1GoProD455CameraSystem()
        datagen_cfg.task_sampler_config.base_pose_sampling_radius_range = (3.0, 10.0)
        datagen_cfg.task_sampler_config.robot_safety_radius = 0.35
    elif robot == "yam":
        datagen_cfg.robot_config = I2rtYamRobotConfig()
        datagen_cfg.camera_config = I2rtYamCameraSystem()
    elif robot == "bimanual_yam":
        datagen_cfg.robot_config = BimanualYamRobotConfig()
        datagen_cfg.camera_config = BimanualYamCameraSystem()
    else:
        raise ValueError

    if single_step:
        datagen_cfg.task_horizon = 0  # first obervationd does not count
        datagen_cfg.use_passive_viewer = False

    return datagen_cfg


def get_policy_config(s: str, robot: str = None):
    # policy_name = ''.join(word.capitalize() for word in s.split('_')) + "PolicyConfig"
    # policy_config = get_config_class(policy_name)
    if s == "pi":
        # Use BimanualYamPiPolicyConfig for bimanual_yam robot
        if robot == "bimanual_yam":
            return BimanualYamPiPolicyConfig()
        return PiPolicyConfig()
    elif s == "rum":
        return RumPolicyConfig()
    elif s == "teleop":
        return TeleopPolicyConfig()
    else:
        raise ValueError(f"Unknown policy name {s}")


def patch_planner_policy(exp_config):
    if exp_config.policy_config is not None:
        try:
            exp_config.policy_config.tcp_pos_err_threshold
        except AttributeError:
            exp_config.policy_config.tcp_pos_err_threshold = 1
            exp_config.policy_config.tcp_rot_err_threshold = np.radians(90.0)


def get_output_dir(args, exp_config):
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name_prefix:
        run_name = f"{args.run_name_prefix}_{run_name}"
    if hasattr(exp_config, "task_type"):
        task_name = f"{exp_config.task_type}_{args.policy}_v1"
    else:
        task_name = f"default_{args.policy}_v1"
    output_dir = ASSETS_DIR / "datagen" / task_name / run_name
    return output_dir


def main(args: argparse.ArgumentParser) -> None:
    if args.eval:  # 1) load an benchmark config
        log.info(f"Loading pre-saved config from {args.eval}. This will override other settings.")
        exp_config = MlSpacesExpConfig.load_config(Path(args.eval))
        exp_config.frozen_config_path = Path(args.eval)
        exp_config.seed = 42  # new seed
        exp_config.filter_for_successful_trajectories = False  # see eval failures
        exp_config.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)  # for eval
    elif args.config:  # 2) load an experiment config
        exp_config = get_config_class(args.config)()
    else:  # 3) create config from arguments
        exp_config = setup_config(args)

    # overload some config values
    exp_config.num_workers = 1
    exp_config.use_passive_viewer = args.viewer

    # Overload robot
    if args.robot == "rum" or args.policy == "rum":
        exp_config.robot_config = FloatingRUMRobotConfig()
        exp_config.robot_config.init_qpos_noise_range = None  # no randomization

    # Overload policy
    policy = None
    if args.policy in ("pi", "rum", "teleop"):
        if args.policy == "rum":
            exp_config.policy_dt_ms = 500
            exp_config.task_horizon = 80
        elif args.policy == "pi":
            exp_config.policy_dt_ms = 500
            exp_config.task_horizon = 300
        elif args.policy == "teleop":
            exp_config.policy_dt_ms = 40  # More responsive for teleoperation
            exp_config.task_horizon = 1000
        exp_config.policy_config = get_policy_config(args.policy, robot=args.robot)
        policy = exp_config.policy_config.policy_factory(exp_config, None)
    elif args.policy == "planner":
        pass
    else:
        raise ValueError(f"Unknown policy option {args.policy}")

    exp_config.output_dir = get_output_dir(args, exp_config)
    exp_config.save_config()
    runner = MyRolloutRunner(exp_config)
    runner.run(preloaded_policy=policy)


if __name__ == "__main__":
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument("--eval", type=str, default=None, help="Load a fixed benchmark")
    args.add_argument("--config", type=str, default=None, help="Load a fixed config")
    args.add_argument("--viewer", action="store_true", help="single step")
    args.add_argument(
        "--robot", type=str, default="droid", help="franka, droid, rum, rby1, yam, or bimanual_yam"
    )
    args.add_argument(
        "--policy",
        type=str,
        default="planner",
        choices=["planner", "pi", "rum", "teleop"],
        help="policy to use: planner, pi, rum, or teleop",
    )

    # Arguments below ONLY used for policy from scratch (no eval or config given)
    args.add_argument("--task_type", type=str, default="pick", help="pick or open")
    args.add_argument("--single_step", action="store_true", help="single step")
    args.add_argument(
        "--scene_dataset",
        type=str,
        default="ithor",
        help="ithor, procthor-10k, procthor-objaverse, procthor-100k-debug",
    )
    args.add_argument("--data_split", type=str, default="train", help="train or test")
    args.add_argument("--house_inds", type=int, default=1, help="house indices")
    args.add_argument(
        "--target_types", type=str, default=None, help="comma separated list of target types"
    )
    args.add_argument(
        "--samples_per_house", type=int, default=4, help="number of samples per house"
    )
    args.add_argument(
        "--filter_for_successful_trajectories",
        action="store_true",
        help="filter for successful trajectories",
    )
    args.add_argument("--randomize_lighting", type=bool, default=False, help="randomize lighting")
    args.add_argument("--randomize_textures", type=bool, default=False, help="randomize textures")
    args.add_argument("--randomize_dynamics", type=bool, default=False, help="randomize dynamics")
    args.add_argument("--randomize_scene", type=bool, default=False, help="randomize scene all")
    args.add_argument("--seed", type=int, default=2, help="random seed")
    args.add_argument("--run_name_prefix", type=str, default="", help="prefix for run name")
    args = args.parse_args()
    main(args)
