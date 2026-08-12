from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.policy_configs import PickAndPlacePlannerPolicyConfig
from molmo_spaces.configs.task_configs import PackingTaskConfig
from molmo_spaces.configs.task_sampler_configs import PackingTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.tasks.packing_task import PackingTask
from molmo_spaces.tasks.packing_task_sampler import PackingTaskSampler
from molmo_spaces.utils.constants.object_constants import PICK_AND_PLACE_OBJECTS


@register_config("PackingDataGenConfig")
class PackingDataGenConfig(PickBaseConfig):
    task_type: str = "packing"
    num_workers: int = 1
    task_sampler_config: PackingTaskSamplerConfig = PackingTaskSamplerConfig(
        task_sampler_class=PackingTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=20,
    )
    task_config: PackingTaskConfig = PackingTaskConfig(task_cls=PackingTask)
    policy_config: PickAndPlacePlannerPolicyConfig = PickAndPlacePlannerPolicyConfig()
