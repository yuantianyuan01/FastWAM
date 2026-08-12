from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.policy_configs import PickAndPlaceNextToPlannerPolicyConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaRobotConfig
from molmo_spaces.configs.task_configs import PickAndPlaceNextToTaskConfig
from molmo_spaces.configs.task_sampler_configs import PickAndPlaceNextToTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.tasks.pick_and_place_next_to_task import PickAndPlaceNextToTask
from molmo_spaces.tasks.pick_and_place_next_to_task_sampler import PickAndPlaceNextToTaskSampler
from molmo_spaces.utils.constants.object_constants import PICK_AND_PLACE_OBJECTS


@register_config("PickAndPlaceNextToDataGenConfig")
class PickAndPlaceNextToDataGenConfig(PickBaseConfig):
    task_type: str = "pick_and_place_next_to"
    num_workers: int = 1
    task_sampler_config: PickAndPlaceNextToTaskSamplerConfig = PickAndPlaceNextToTaskSamplerConfig(
        task_sampler_class=PickAndPlaceNextToTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=20,
    )
    task_config: PickAndPlaceNextToTaskConfig = PickAndPlaceNextToTaskConfig(
        task_cls=PickAndPlaceNextToTask
    )
    policy_config: PickAndPlaceNextToPlannerPolicyConfig = PickAndPlaceNextToPlannerPolicyConfig()
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    use_passive_viewer: bool = False
