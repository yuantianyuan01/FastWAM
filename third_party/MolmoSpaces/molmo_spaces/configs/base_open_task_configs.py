from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.policy_configs import BasePolicyConfig, OpenClosePlannerPolicyConfig
from molmo_spaces.configs.task_configs import OpeningTaskConfig
from molmo_spaces.configs.task_sampler_configs import (
    OpenTaskSamplerConfig,
)
from molmo_spaces.tasks.opening_task_samplers import OpenTaskSampler
from molmo_spaces.tasks.opening_tasks import OpeningTask


class OpeningBaseConfig(PickBaseConfig):
    """Base configuration for opening task data generation."""

    task_type: str = "open"

    # Task sampler configuration (imported from task_sampler_configs.py)
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0,  # 0 for open task
    )

    # Task configuration (imported from task_configs.py)
    task_config: OpeningTaskConfig = OpeningTaskConfig(
        task_cls=OpeningTask,
        task_success_threshold=0.15,  # low for now, due to placement/IK constraints, should be ~0.66
        joint_index=0,
        any_inst_of_category=True,  # open any instance of category
    )
    task_config_preset: OpeningTaskConfig | None = None

    # Policy configuration (imported from policy_configs.py)
    policy_config: BasePolicyConfig = OpenClosePlannerPolicyConfig()


class ClosingBaseConfig(PickBaseConfig):
    """Base configuration for closing task data generation."""

    task_type: str = "close"

    # Task sampler configuration (imported from task_sampler_configs.py)
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler,
        target_initial_state_open_percentage=0.5,  # 0.67 for close task
    )

    # Task configuration (imported from task_configs.py)
    task_config: OpeningTaskConfig = OpeningTaskConfig(
        task_cls=OpeningTask,
        task_success_threshold=0.85,  # For closing Task, 0.33
        joint_index=0,
        any_inst_of_category=False,  # open any instance of category
    )
    task_config_preset: OpeningTaskConfig | None = None

    # Policy configuration (imported from policy_configs.py)
    policy_config: BasePolicyConfig = OpenClosePlannerPolicyConfig()
