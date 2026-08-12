from pathlib import Path

from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.policy_configs import PickAndPlaceColorPlannerPolicyConfig
from molmo_spaces.configs.task_configs import PickAndPlaceColorTaskConfig
from molmo_spaces.configs.task_sampler_configs import PickAndPlaceColorTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pick_and_place_color_task import PickAndPlaceColorTask
from molmo_spaces.tasks.pick_and_place_color_task_sampler import PickAndPlaceColorTaskSampler


@register_config("PickAndPlaceColorDataGenConfig")
class PickAndPlaceColorDataGenConfig(PickBaseConfig):
    task_type: str = "pick_and_place_color"
    num_workers: int = 1
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_and_place_color_base_v1"
    wandb_project: str = "molmo-spaces-data-generation"
    task_sampler_config: PickAndPlaceColorTaskSamplerConfig = PickAndPlaceColorTaskSamplerConfig(
        task_sampler_class=PickAndPlaceColorTaskSampler,
        samples_per_house=20,
    )
    task_config: PickAndPlaceColorTaskConfig = PickAndPlaceColorTaskConfig(
        task_cls=PickAndPlaceColorTask
    )
    policy_config: PickAndPlaceColorPlannerPolicyConfig = PickAndPlaceColorPlannerPolicyConfig()
    use_passive_viewer: bool = False

    @property
    def tag(self) -> str:
        return "pick_and_place_color_base_datagen"
