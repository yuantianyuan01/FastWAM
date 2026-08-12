"""
Data generation configs for RBY1 navigation to object tasks.

These configs subclass from NavToObjBaseConfig and are registered
for use in the data generation pipeline.
"""

from pathlib import Path

from molmo_spaces.configs import BasePolicyConfig, BaseRobotConfig
from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
from molmo_spaces.configs.camera_configs import RBY1MjcfCameraSystem
from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig
from molmo_spaces.configs.robot_configs import RBY1Config
from molmo_spaces.configs.task_sampler_configs import NavToObjTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler


@register_config("NavToObjDataGenConfig")
class NavToObjDataGenConfig(NavToObjBaseConfig):
    """Data generation config for RBY1 navigation to object tasks."""

    task_type: str = "nav_to_obj"
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "nav_to_obj_v1"
    wandb_project: str = "molmo-spaces-data-generation"
    robot_config: BaseRobotConfig = RBY1Config()
    policy_config: BasePolicyConfig = AStarNavToObjPolicyConfig()
    camera_config: RBY1MjcfCameraSystem = RBY1MjcfCameraSystem()
    task_sampler_config: NavToObjTaskSamplerConfig = NavToObjTaskSamplerConfig(
        task_sampler_class=NavToObjTaskSampler,
        pickup_types=None,
        # pickup_types=[
        #     "alarmclock",
        #     "apple",
        #     "basketball",
        #     "bed",
        #     "bowl",
        #     "chair",
        #     "garbagecan",
        #     "houseplant",
        #     "laptop",
        #     "mug",
        #     "sofa",
        #     "spraybottle",
        #     "television",
        #     "toilet",
        #     "vase",
        # ],
        robot_safety_radius=0.35,  # Radius around robot to avoid collisions
        robot_object_z_offset=0.1,  # Offset to place robot base relative to object height
        base_pose_sampling_radius_range=(
            4.0,
            20.0,
        ),  # Radius to sample robot base pose (min and max)
        face_target=False,  # Whether to face the target when placing the robot
        max_robot_placement_attempts=10,  # Maximum number of attempts to place the robot
        filter_for_successful_trajectories=True,
    )

    def model_post_init(self, __context) -> None:
        """Initialize and validate configuration after Pydantic model initialization"""
        super().model_post_init(__context)

        if not self.task_sampler_config.house_inds:
            self.task_sampler_config.house_inds = list(range(4))

    @property
    def tag(self) -> str:
        return "rby1_nav_to_obj_datagen"
