"""
Data generation configs for Franka move-to-pose tasks.

These configs subclass from the base_pick_config and are registered
for use in the data generation pipeline.
"""

import math
from functools import cache
from pathlib import Path

from molmo_spaces.configs import BasePolicyConfig, BaseRobotConfig
from molmo_spaces.configs.base_open_task_configs import ClosingBaseConfig, OpeningBaseConfig
from molmo_spaces.configs.base_pick_and_place_color_configs import PickAndPlaceColorDataGenConfig

# This is here so that un-pickling benchmarks works
from molmo_spaces.configs.base_pick_and_place_configs import (
    PickAndPlaceDataGenConfig,
)
from molmo_spaces.configs.base_pick_and_place_next_to_configs import PickAndPlaceNextToDataGenConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
    FrankaEasyRandomizedDroidCameraSystem,
    FrankaGoProD405D455CameraSystem,
    FrankaOmniPurposeCameraSystem,
    FrankaRandomizedD405D455CameraSystem,
    FrankaRandomizedDroidCameraSystem,
    RBY1GoProD455CameraSystem,
)
from molmo_spaces.configs.policy_configs import (
    CuroboOpenClosePlannerPolicyConfig,
    CuroboPickAndPlacePlannerPolicyConfig,
    OpenClosePlannerPolicyConfig,
    PickPlannerPolicyConfig,
)
from molmo_spaces.configs.robot_configs import (
    FloatingRUMRobotConfig,
    FrankaRobotConfig,
    RBY1MConfig,
    RBY1MOpenCloseConfig,
)
from molmo_spaces.configs.task_sampler_configs import (
    OpenTaskSamplerConfig,
    PickAndPlaceColorTaskSamplerConfig,
    PickAndPlaceNextToTaskSamplerConfig,
    PickAndPlaceTaskSamplerConfig,
    PickTaskSamplerConfig,
    RUMPickTaskSamplerConfig,
)

# Oder of configs should be order the code is executed in
# scenes, robots, camera, task_sampler, policy, output
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_robot_path
from molmo_spaces.tasks.opening_task_samplers import OpenTaskSampler
from molmo_spaces.tasks.pick_and_place_color_task_sampler import PickAndPlaceColorTaskSampler
from molmo_spaces.tasks.pick_and_place_next_to_task_sampler import PickAndPlaceNextToTaskSampler
from molmo_spaces.tasks.pick_and_place_task_sampler import (
    PickAndPlaceMultiTaskSampler,
    PickAndPlaceTaskSampler,
)
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.constants.object_constants import PICK_AND_PLACE_OBJECTS
from molmo_spaces.utils.synset_utils import get_valid_pickupable_obja_uids


@register_config("FrankaPickDroidDataGenConfig")
class FrankaPickDroidDataGenConfig(PickBaseConfig):
    """Data generation config for Franka pick task with DROID-style fixed cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_droid_datagen"


@register_config("FrankaPickGoProD405D455DataGenConfig")
class FrankaPickGoProD405D455DataGenConfig(PickBaseConfig):
    """Data generation config for Franka pick task with GoPro D405 cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaGoProD405D455CameraSystem = FrankaGoProD405D455CameraSystem()
    num_workers: int = 4
    task_horizon: int = 150
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_go_pro_d405_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_go_pro_d405_datagen"


@register_config("FrankaPickRandomizedDataGenConfig")
class FrankaPickRandomizedDataGenConfig(PickBaseConfig):
    """Data generation config for Franka pick task with randomized exocentric cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_randomized_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_randomized_datagen"


@register_config("RUMPickDataGenConfig")
class RUMPickDataGenConfig(PickBaseConfig):
    scene_dataset: str = "holodeck-objaverse"
    robot_config: FloatingRUMRobotConfig = FloatingRUMRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaRandomizedD405D455CameraSystem(
        img_resolution=(960, 720)
    )
    task_sampler_config: RUMPickTaskSamplerConfig = RUMPickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler, robot_object_z_offset=0
    )
    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rum_pick_v1"

    @property
    def tag(self) -> str:
        return "rum_pick_datagen"


@register_config("FrankaPickAndPlaceDataGenConfig")
class FrankaPickAndPlaceDataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedDroidCameraSystem = FrankaRandomizedDroidCameraSystem()
    policy_dt_ms: float = 66.0  # ~15hz
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_randomized_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_datagen"


@register_config("FrankaPickAndPlaceEasyDataGenConfig")
class FrankaPickAndPlaceEasyDataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaEasyRandomizedDroidCameraSystem = FrankaEasyRandomizedDroidCameraSystem()
    policy_dt_ms: float = 66.0  # ~15hz
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_randomized_easy_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_easy_datagen"


@register_config("FrankaPickAndPlaceDroidDataGenConfig")
class FrankaPickAndPlaceDroidDataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_droid_datagen"


@register_config("FrankaPickAndPlaceGoProD405D455DataGenConfig")
class FrankaPickAndPlaceGoProD405D455DataGenConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaGoProD405D455CameraSystem = FrankaGoProD405D455CameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_go_pro_d405_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_go_pro_d405_datagen"


@register_config("FrankaPickAndPlaceNextToDataGenConfig")
class FrankaPickAndPlaceNextToDataGenConfig(PickAndPlaceNextToDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_next_to_randomized_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_datagen"


@register_config("FrankaPickAndPlaceNextToDroidDataGenConfig")
class FrankaPickAndPlaceNextToDroidDataGenConfig(PickAndPlaceNextToDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_next_to_droid_v1"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_droid_datagen"


@register_config("FrankaPickAndPlaceColorDataGenConfig")
class FrankaPickAndPlaceColorDataGenConfig(PickAndPlaceColorDataGenConfig):
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_colors_randomized_v1"
    )
    wandb_project: str = "molmo-spaces-data-generation"
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_datagen"


@register_config("FrankaPickAndPlaceColorDroidDataGenConfig")
class FrankaPickAndPlaceColorDroidDataGenConfig(PickAndPlaceColorDataGenConfig):
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_colors_droid_randomized_v1"
    )
    wandb_project: str = "molmo-spaces-data-generation"
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_droid_datagen"


@register_config("FrankaOpenDataGenConfig")
class FrankaOpenDataGenConfig(OpeningBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "train"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0,  # 0.67 for close task, 0 for open task
    )
    policy_config: BasePolicyConfig = OpenClosePlannerPolicyConfig()
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "open_v1"

    @property
    def tag(self) -> str:
        return "franka_open_datagen"


@register_config("RBY1OpenDataGenConfig")
class RBY1OpenDataGenConfig(OpeningBaseConfig):
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rby1_open_v1"
    wandb_project: str = "mujoco-thor-data-generation"
    robot_config: RBY1MOpenCloseConfig = RBY1MOpenCloseConfig()
    policy_config: BasePolicyConfig = CuroboOpenClosePlannerPolicyConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0,  # 0.67 for close task, 0 for open task
        robot_safety_radius=0.2,
        base_pose_sampling_radius_range=(0.3, 1.0),
    )
    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    use_passive_viewer: bool = False
    seed: int = None
    filter_for_successful_trajectories: bool = True
    policy_dt_ms: float = 100.0  # Default policy time step
    ctrl_dt_ms: float = 20.0  # Default control time step
    sim_dt_ms: float = 4.0  # Default simulation time step

    @property
    def tag(self) -> str:
        return "rby1_open_datagen"

    def _init_policy_config(self) -> CuroboPickAndPlacePlannerPolicyConfig:
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig
        from molmo_spaces.policy.solvers.object_manipulation.curobo_open_close_planner_policy import (
            CuroboOpenClosePlannerPolicy,
        )

        rby1_path = get_robot_path("rby1m")

        left_curobo_planner_config = CuroboPlannerConfig(
            curobo_robot_config_path=str(
                rby1_path / "curobo_config" / "rby1m_left_arm_holobase.yml"
            ),
            collision_activation_distance=0.01,
            num_trajopt_seeds=12,
            max_attempts=15,
            num_ik_seeds=128,
            trajopt_tsteps=48,
            interpolation_dt=self.ctrl_dt_ms / 1000.0,  # 1x control dt
            check_start_validity=False,
            enable_finetune_trajopt=True,
        )
        right_curobo_planner_config = left_curobo_planner_config.model_copy(deep=True)
        right_curobo_planner_config.curobo_robot_config_path = str(
            rby1_path / "curobo_config" / "rby1m_right_arm_holobase.yml"
        )
        return CuroboOpenClosePlannerPolicyConfig(
            policy_cls=CuroboOpenClosePlannerPolicy,
            policy_factory=CuroboOpenClosePlannerPolicy,
            left_curobo_planner_config=left_curobo_planner_config,
            right_curobo_planner_config=right_curobo_planner_config,
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.policy_config = self._init_policy_config()
        self.task_config.task_success_threshold = 0.67
        self.task_sampler_config.randomize_textures = True


@register_config("RBY1PickAndPlaceDataGenConfig")
class RBY1PickAndPlaceDataGenConfig(PickAndPlaceDataGenConfig):
    seed: int | None = None  # 75133535  # 4821697
    viewer_cam_dict: dict = {"camera": "robot_0/camera_follower"}
    use_passive_viewer: bool = False
    task_horizon: int | None = 400  # Maximum number of steps per episode (if None, no time limit)
    filter_for_successful_trajectories: bool = True
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rby1_pick_and_place_v1"
    wandb_project: str = "mujoco-thor-data-generation"
    policy_dt_ms: float = 100.0  # Default policy time step
    ctrl_dt_ms: float = 20.0  # Default control time step
    sim_dt_ms: float = 4.0  # Default simulation time step

    robot_config: RBY1MConfig = RBY1MConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    policy_config: CuroboPickAndPlacePlannerPolicyConfig | None = None

    def _init_policy_config(self) -> CuroboPickAndPlacePlannerPolicyConfig:
        from molmo_spaces.policy.solvers.object_manipulation.curobo_pick_and_place_planner_policy import (
            CuroboPickAndPlacePlannerPolicy,
        )

        rby1_path = get_robot_path("rby1m")
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig

        left_curobo_planner_config = CuroboPlannerConfig(
            curobo_robot_config_path=str(
                rby1_path / "curobo_config" / "rby1m_left_arm_holobase.yml"
            ),
            collision_activation_distance=0.01,
            num_trajopt_seeds=12,
            max_attempts=15,
            num_ik_seeds=128,
            trajopt_tsteps=48,
            interpolation_dt=self.ctrl_dt_ms / 1000.0,  # 1x control dt
            check_start_validity=False,
            enable_finetune_trajopt=True,
        )
        right_curobo_planner_config = left_curobo_planner_config.model_copy(deep=True)
        right_curobo_planner_config.curobo_robot_config_path = str(
            rby1_path / "curobo_config" / "rby1m_right_arm_holobase.yml"
        )
        return CuroboPickAndPlacePlannerPolicyConfig(
            policy_cls=CuroboPickAndPlacePlannerPolicy,
            policy_factory=CuroboPickAndPlacePlannerPolicy,
            left_curobo_planner_config=left_curobo_planner_config,
            right_curobo_planner_config=right_curobo_planner_config,
            enable_collision_avoidance=True,
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        try:
            self.policy_config = self._init_policy_config()
        except RuntimeError as e:
            # Check if this is a CUDA/GPU-related error
            error_msg = str(e)
            if "NVIDIA" in error_msg or "CUDA" in error_msg or "GPU" in error_msg:
                # No GPU available - this is expected on manager nodes that just coordinate jobs
                # Policy config will be initialized later on worker nodes that have GPUs
                print(
                    f"Warning: Skipping policy config initialization due to missing GPU: {error_msg}"
                )
                self.policy_config = None
            else:
                raise
        self.robot_config.init_qpos["head"][1] = 0.6
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.max_robot_to_obj_dist = 0.5
        self.task_sampler_config.object_placement_radius_range = (0.1, 0.5)
        self.task_sampler_config.min_object_to_receptacle_dist = 0.05
        self.task_sampler_config.max_robot_to_place_receptacle_dist = 0.5

    @property
    def tag(self) -> str:
        return "rby1_pick_and_place_datagen"


@register_config("RBY1PickDataGenConfig")
class RBY1PickDataGenConfig(PickBaseConfig):
    seed: int | None = None
    viewer_cam_dict: dict = {"camera": "robot_0/camera_follower"}
    use_passive_viewer: bool = False
    task_horizon: int | None = 400  # Maximum number of steps per episode (if None, no time limit)
    filter_for_successful_trajectories: bool = True
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "rby1_pick_v1"
    policy_dt_ms: float = 100.0  # Default policy time step
    ctrl_dt_ms: float = 20.0  # Default control time step
    sim_dt_ms: float = 4.0  # Default simulation time step

    robot_config: RBY1MConfig = RBY1MConfig()
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()
    policy_config: CuroboPickAndPlacePlannerPolicyConfig | None = None

    def _init_policy_config(self) -> CuroboPickAndPlacePlannerPolicyConfig:
        from molmo_spaces.policy.solvers.object_manipulation.curobo_pick_and_place_planner_policy import (
            CuroboPickAndPlacePlannerPolicy,
        )

        rby1_path = get_robot_path("rby1m")
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig

        left_curobo_planner_config = CuroboPlannerConfig(
            curobo_robot_config_path=str(
                rby1_path / "curobo_config" / "rby1m_left_arm_holobase.yml"
            ),
            collision_activation_distance=0.01,
            num_trajopt_seeds=12,
            max_attempts=15,
            num_ik_seeds=128,
            trajopt_tsteps=48,
            interpolation_dt=self.ctrl_dt_ms / 1000.0,  # 1x control dt
            check_start_validity=True,
            enable_finetune_trajopt=True,
        )
        right_curobo_planner_config = left_curobo_planner_config.model_copy(deep=True)
        right_curobo_planner_config.curobo_robot_config_path = str(
            rby1_path / "curobo_config" / "rby1m_right_arm_holobase.yml"
        )
        return CuroboPickAndPlacePlannerPolicyConfig(
            policy_cls=CuroboPickAndPlacePlannerPolicy,
            policy_factory=CuroboPickAndPlacePlannerPolicy,
            left_curobo_planner_config=left_curobo_planner_config,
            right_curobo_planner_config=right_curobo_planner_config,
            enable_collision_avoidance=True,
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        try:
            self.policy_config = self._init_policy_config()
        except RuntimeError as e:
            # Check if this is a CUDA/GPU-related error
            error_msg = str(e)
            if "NVIDIA" in error_msg or "CUDA" in error_msg or "GPU" in error_msg:
                # No GPU available - this is expected on manager nodes that just coordinate jobs
                # Policy config will be initialized later on worker nodes that have GPUs
                print(
                    f"Warning: Skipping policy config initialization due to missing GPU: {error_msg}"
                )
                self.policy_config = None
            else:
                raise
        self.robot_config.init_qpos["head"][1] = 0.6
        self.task_sampler_config.robot_safety_radius = 0.35
        self.task_sampler_config.max_robot_to_obj_dist = 0.5
        self.task_sampler_config.object_placement_radius_range = (0.1, 0.5)

    @property
    def tag(self) -> str:
        return "rby1_pick_datagen"


@register_config("FrankaCloseDataGenConfig")
class FrankaCloseDataGenConfig(ClosingBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "train"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0.5,  # 0.67 for close task, 0 for open task
    )
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "close_v1"

    @property
    def tag(self) -> str:
        return "franka_close_datagen"


@register_config("FrankaPickAndPlaceGoProD405D455DataGenConfigDebug")
class FrankaPickAndPlaceGoProD405D455DataGenConfigDebug(FrankaPickAndPlaceDroidDataGenConfig):
    """Data generation config for Franka pick and place task with GoPro D405 cameras - deterministic version."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaGoProD405D455CameraSystem = FrankaGoProD405D455CameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        samples_per_house=10,
        max_tasks=100,
        # pickup_types=["Mug"],
        house_inds=[2],
    )
    num_workers: int = 1
    task_horizon: int = 100
    use_wandb: bool = False
    log_level: str = "debug"
    filter_for_successful_trajectories: bool = False
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_go_pro_d405_v1_debug"
    )

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_go_pro_d405_d455_datagen_debug"


@register_config("FrankaPickOmniCamConfig")
class FrankaPickOmniCamConfig(PickBaseConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_omni_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_omni_datagen"


@register_config("FrankaPickOmniCamAblationConfig")
class FrankaPickOmniCamAblationConfig(FrankaPickOmniCamConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_omni_v1_cam_ablation"

    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        added_pickup_objects=None,  # will get set after instantiation
        # num_added_pickups=30, these are defaults
        # episodes_per_added_pickup=1,
    )

    @staticmethod
    @cache
    def _get_valid_pickupable_obja_uids() -> list[str]:
        return get_valid_pickupable_obja_uids()

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.task_sampler_config.added_pickup_objects is None:
            self.task_sampler_config.added_pickup_objects = self._get_valid_pickupable_obja_uids()

    @property
    def tag(self) -> str:
        return "franka_pick_omni_v1_cam_ablation"


@register_config("FrankaPickAndPlaceOmniCamConfig")
class FrankaPickAndPlaceOmniCamConfig(PickAndPlaceDataGenConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_omni_v1"
    log_level: str = "info"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_omnicam_datagen"


# PickAndPlaceNextToDataGenConfig
@register_config("FrankaPickAndPlaceNextToOmniCamConfig")
class FrankaPickAndPlaceNextToOmniCamConfig(PickAndPlaceNextToDataGenConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = (
        ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_next_to_omni_v1"
    )
    log_level: str = "info"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_omnicam_datagen"


# PickAndPlaceColorDataGenConfig
@register_config("FrankaPickAndPlaceColorOmniCamConfig")
class FrankaPickAndPlaceColorOmniCamConfig(PickAndPlaceColorDataGenConfig):
    """Data generation config for Franka pick task with Omni-directional cameras."""

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaOmniPurposeCameraSystem()
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_color_omni_v1"
    log_level: str = "info"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_omnicam_datagen"


################################################################################
# Benchmark configs
################################################################################


@register_config("FrankaPickDroidMiniBench")
class FrankaPickDroidMiniBench(PickBaseConfig):
    scene_dataset: str = "procthor-10k"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_minbench"


@register_config("FrankaPickandPlaceMiniBench")
class FrankaPickandPlaceDroidMiniBench(PickAndPlaceDataGenConfig):
    scene_dataset: str = "procthor-10k"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_pickandplace_minbench"


@register_config("FrankaPickDroidBench")
class FrankaPickDroidBench(PickBaseConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_obja_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_bench"


@register_config("FrankaPickandPlaceDroidBench")
class FrankaPickandPlaceDroidBench(PickAndPlaceDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=40,
        house_inds=list(range(101)),
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_obja_v1"

    @property
    def tag(self) -> str:
        return "franka_pickandplace_bench"


@register_config("FrankaPickandPlaceNextToDroidBench")
class FrankaPickandPlaceNextToDroidBench(PickAndPlaceNextToDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_next_to_obja_v1"

    @property
    def tag(self) -> str:
        return "franka_pickandplacenextto_bench"


@register_config("FrankaPickandPlaceColorDroidBench")
class FrankaPickandPlaceColorDroidBench(PickAndPlaceColorDataGenConfig):
    """Data generation config for Franka pick task with DROID-style fixed cameras."""

    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_color_obja_v1"
    scene_dataset: str = "procthor-objaverse"

    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()

    @property
    def tag(self) -> str:
        return "franka_pickandplacecolor_bench"


@register_config("FrankaOpenHardBench")
class FrankaOpenHardBench(OpeningBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "val"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0,  # 0.67 for close task, 0 for open task
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    policy_config: BasePolicyConfig = OpenClosePlannerPolicyConfig()
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "open_bench"

    @property
    def tag(self) -> str:
        return "franka_open_hard_bench"


@register_config("FrankaCloseHardBench")
class FrankaCloseHardBench(ClosingBaseConfig):
    """Data generation config for Franka open task."""

    scene_dataset: str = "ithor"  # Name of the scene dataset to load
    data_split: str = "val"  # Data split to use
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0.5,  # 0.67 for close task, 0 for open task
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    task_horizon: int | None = 200  # Maximum number of steps per episode (if None, no time limit)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "close_bench"

    @property
    def tag(self) -> str:
        return "franka_close_hard_bench"


@register_config("FrankaPickHardBench")
class FrankaPickHardBench(PickBaseConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_hard_bench"


@register_config("FrankaPickandPlaceHardBench")
class FrankaPickandPlaceHardBench(PickAndPlaceDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )

    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_hard_bench"


@register_config("FrankaPickandPlaceNextToHardBench")
class FrankaPickandPlaceNextToHardBench(PickAndPlaceNextToDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )
    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceNextToTaskSamplerConfig(
        task_sampler_class=PickAndPlaceNextToTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_next_to_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_next_to_hard_bench"


@register_config("FrankaPickandPlaceColorHardBench")
class FrankaPickandPlaceColorHardBench(PickAndPlaceColorDataGenConfig):
    scene_dataset: str = "procthor-objaverse"
    data_split: str = "val"
    robot_config: BaseRobotConfig = FrankaRobotConfig(
        init_qpos_noise_range={"arm": [0.26] * 6 + [math.pi / 2]}
    )

    camera_config: FrankaOmniPurposeCameraSystem = FrankaOmniPurposeCameraSystem()
    task_sampler_config: PickAndPlaceColorTaskSamplerConfig = PickAndPlaceColorTaskSamplerConfig(
        task_sampler_class=PickAndPlaceColorTaskSampler,
        robot_object_z_offset_random_min=-0.25,
        robot_object_z_offset_random_max=0.25,
        robot_placement_rotation_range_rad=0.52,  # ±30 degrees
    )
    output_dir: Path = ASSETS_DIR / "benchmark" / "pick_and_place_color_hard_v1"

    @property
    def tag(self) -> str:
        return "franka_pick_and_place_color_hard_bench"


@register_config("MultiPnPTask")
class RUMPickAndPlaceMultiDataGenConfig(PickAndPlaceDataGenConfig):
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pnpmulti_V1"
    wandb_project: str = "mujoco-thor-data-generation"
    robot_config: FloatingRUMRobotConfig = FloatingRUMRobotConfig()
    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceMultiTaskSampler,
        pickup_types=None,
        samples_per_house=20,
        house_inds=[7],  # TODO(Omar): choose one you like.
        robot_object_z_offset=0.2,
        check_robot_placement_visibility=False,
    )

    camera_config: FrankaDroidCameraSystem = FrankaRandomizedD405D455CameraSystem(
        img_resolution=(960, 720),
        visibility_constraints=None,
        allow_relaxed_constraints=True,
    )

    @property
    def tag(self) -> str:
        return "pnpmulti_bench"


@register_config("FrankaPickBatchTestConfig")
class FrankaPickBatchTestConfig(PickBaseConfig):
    """Minimal config to test local batch-based work distribution.

    Single house, 12 episodes split into 3 batches of 4, processed by 2 workers.
    Output: house_7/trajectories_batch_{1,2,3}_of_3.h5
    """

    num_workers: int = 2
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        house_inds=[7],
        samples_per_house=12,
        episodes_per_batch=4,  # 4 is the default value - change to 1 or 10 to test different batch sizes.
    )
    filter_for_successful_trajectories: bool = True
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_batch_test"
    log_level: str = "debug"

    @property
    def tag(self) -> str:
        return "franka_pick_batch_test"
