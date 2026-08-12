"""

These configs are EXAMPLES of how to set up evaluation configs for use
with JSON benchmarks via molmo_spaces.evaluation.run_evaluation(). The anticipated
pattern is that users will create their own eval configs in their own repositories,
import run_evaluation from molmo_spaces.evaluation, and pass their config to it.

Example usage from an external repo:
    from molmo_spaces.evaluation import run_evaluation
    from my_repo.configs import MyPolicyEvalConfig

    results = run_evaluation(
        eval_config_cls=MyPolicyEvalConfig,
        benchmark_dir="/path/to/benchmark",
        checkpoint_path="/path/to/checkpoint",
    )

Eval configs provide:
- Robot config (factories for instantiation, gravcomp settings)
- Policy config (checkpoint path, camera names, action spec)
- Timing parameters (policy_dt_ms, ctrl_dt_ms, sim_dt_ms)

Episode-specific data (init_qpos, robot_base_pose, cameras, object_poses, task config)
comes from the JSON benchmark files, not from these configs. The benchmark JSON
is strictly authoritative for episode initialization.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.policy_configs import BrownianMotionPolicyConfig, DummyPolicyConfig
from molmo_spaces.configs.policy_configs_baselines import (
    CAPPolicyConfig,
    DreamZeroPolicyConfig,
    PiPolicyConfig,
    TeleopPolicyConfig,
)
from molmo_spaces.configs.robot_configs import (
    ActionNoiseConfig,
    FrankaCAPRobotConfig,
    FrankaRobotConfig,
)
from molmo_spaces.configs.task_configs import (
    BaseMujocoTaskConfig,
    NavToObjTaskConfig,
    PickAndPlaceColorTaskConfig,
    PickAndPlaceTaskConfig,
)
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
    NavToObjTaskSamplerConfig,
    PickAndPlaceColorTaskSamplerConfig,
    PickAndPlaceTaskSamplerConfig,
)
from molmo_spaces.data_generation.config.nav_to_obj_configs import NavToObjDataGenConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaPickAndPlaceDataGenConfig,
)
from molmo_spaces.policy.dummy_policy import BrownianMotionPolicy, DummyPolicy
from molmo_spaces.tasks.nav_task import NavToObjTask
from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler
from molmo_spaces.tasks.pick_and_place_color_task import PickAndPlaceColorTask
from molmo_spaces.tasks.pick_and_place_color_task_sampler import (
    PickAndPlaceColorTaskSampler,
)
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
from molmo_spaces.tasks.pick_and_place_task_sampler import (
    PickAndPlaceTaskSampler,
)
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.utils.function_utils import make_lenient

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class JsonBenchmarkEvalConfig(MlSpacesExpConfig):
    """
    Minimal base config for JSON benchmark evaluation.

    This config is designed for use ONLY with JSON benchmarks. It provides
    the minimal infrastructure needed to run a learned policy against a
    benchmark where all episode-specific data (task type, cameras, robot poses,
    object poses, etc.) comes from the benchmark JSON.

    Subclass this and provide:
    - robot_config: Robot configuration for instantiation
    - policy_config: Your learned policy configuration

    DO NOT provide task_sampler_config or task_config - those are placeholders
    that will be overridden by the benchmark. If you accidentally try to use
    this config for data generation (not evaluation), it will fail because
    the task sampler/config are minimal stubs.

    Example:
        class MyPolicyBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
            robot_config = FrankaRobotConfig()
            policy_config = MyPolicyConfig(checkpoint_path="/path/to/ckpt")
    """

    # Required infrastructure - subclasses must provide robot_config and policy_config

    # Timing parameters - can be overridden per-policy as needed
    num_envs: int = 1
    num_workers: int = 1
    policy_dt_ms: float = 66.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0
    task_horizon: int = 500

    # Viewer config (usually disabled for eval)
    use_passive_viewer: bool = False
    viewer_cam_dict: dict = {
        "distance": 5.0,
        "azimuth": 45.0,
        "elevation": -30.0,
        "lookat": [0.0, 0.0, 0.5],
    }

    # These are overridden by benchmark - provide placeholders to satisfy base class
    # DO NOT rely on these values; the benchmark JSON is authoritative.
    task_type: str = "pick"  # Overridden per-episode from benchmark
    scene_dataset: str = "procthor-10k"  # Overridden per-episode from benchmark
    data_split: str = "val"  # Overridden per-episode from benchmark
    camera_config: None = None  # Overridden per-episode from benchmark

    # Minimal stubs - these exist only to satisfy the base class.
    # JsonEvalTaskSampler replaces these entirely with benchmark data.
    # Note: task_sampler_class must be a valid class (not None) since pipeline.py
    # instantiates a worker-level task sampler. JsonEvalRunner overrides the per-episode
    # task sampler via get_episode_task_sampler, so this worker-level sampler is unused.
    task_sampler_config: BaseMujocoTaskSamplerConfig = BaseMujocoTaskSamplerConfig(
        task_sampler_class=BaseMujocoTaskSampler,
        house_inds=[0],  # Dummy value, overridden by JsonEvalRunner from benchmark
        samples_per_house=1,
        task_batch_size=1,
        max_tasks=10000,
    )
    task_config: BaseMujocoTaskConfig = BaseMujocoTaskConfig(task_cls=None)

    # Output config
    output_dir: Path = Path("eval_output")
    use_wandb: bool = False
    wandb_project: str = "mlspaces-benchmark-eval"
    filter_for_successful_trajectories: bool = False

    # Episode termination
    terminate_upon_success: bool = False

    @property
    def tag(self) -> str:
        return "json_benchmark_eval"


class DummyBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
    """
    Test config that inherits from JsonBenchmarkEvalConfig.

    This tests the recommended pattern from evaluation/README.md:
    external repos should inherit from JsonBenchmarkEvalConfig and provide
    their robot_config and policy_config. The benchmark JSON provides all
    episode-specific data (cameras, poses, task params).

    Note: Prefixed with underscore to avoid pytest collection warning since
    this inherits from a class with __init__.
    """

    # Timing - short horizon for testing
    task_horizon: int = 10
    seed: int = 42
    policy_dt_ms: float = 200.0

    # Robot config - standard Franka
    robot_config: FrankaRobotConfig = FrankaRobotConfig()

    # Policy config - DummyPolicy returns empty dict (no-op)
    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    @property
    def tag(self) -> str:
        return "dummy_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Disable action noise for deterministic testing
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


class PiPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: PiPolicyConfig = PiPolicyConfig()
    # policy_dt_ms: float = 200.0  # Match your model's expected control rate
    policy_dt_ms: float = 66.0  # ~15hz
    end_on_success: bool = True  # End episode immediately upon success, ignoring task_horizon

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class CAPPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaCAPRobotConfig = FrankaCAPRobotConfig()
    policy_config: CAPPolicyConfig = CAPPolicyConfig()
    policy_dt_ms: float = 500.0  # Match your model's expected control rate

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class TeleopPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: TeleopPolicyConfig = TeleopPolicyConfig()
    policy_dt_ms: float = 40

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


# @register_config("DummyPickPlaceEvalConfig")
class DummyPickPlaceEvalConfig(FrankaPickAndPlaceDataGenConfig):
    """Evaluation config for Dummy pick and place."""

    wandb_project: str = "dummy-eval"
    use_wandb: bool = False
    use_passive_viewer: bool = False
    wandb_name: str = f"dummy_pick_place_eval_{TIMESTAMP}"
    filter_for_successful_trajectories: bool = False
    task_type: str = "pick_and_place"
    task_horizon: int = 600
    output_dir: Path = Path("eval_output") / f"dummy_{TIMESTAMP}"

    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        house_inds=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145],
        samples_per_house=3,
    )
    task_config: PickAndPlaceTaskConfig = PickAndPlaceTaskConfig(task_cls=PickAndPlaceTask)

    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    def _init_policy_config(self) -> DummyPolicyConfig:
        self.policy_config.policy_cls = DummyPolicy
        self.policy_config.policy_factory = make_lenient(DummyPolicy)
        return self.policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


# @register_config("BrownianMotionPickPlaceEvalConfig")
class BrownianMotionPickPlaceEvalConfig(FrankaPickAndPlaceDataGenConfig):
    """Evaluation config for Dummy pick and place."""

    wandb_project: str = "brownian-motion-eval"
    use_wandb: bool = False
    use_passive_viewer: bool = False
    wandb_name: str = f"brownian_motion_pick_place_eval_{TIMESTAMP}"
    filter_for_successful_trajectories: bool = False
    task_type: str = "pick_and_place"
    task_horizon: int = 600
    output_dir: Path = Path("eval_output") / f"brownian_motion_{TIMESTAMP}"

    task_sampler_config: PickAndPlaceTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        house_inds=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145],
        samples_per_house=3,
    )
    task_config: PickAndPlaceTaskConfig = PickAndPlaceTaskConfig(task_cls=PickAndPlaceTask)

    policy_config: BrownianMotionPolicyConfig = BrownianMotionPolicyConfig()

    def _init_policy_config(self) -> BrownianMotionPolicyConfig:
        self.policy_config.policy_cls = BrownianMotionPolicy
        self.policy_config.policy_factory = make_lenient(BrownianMotionPolicy)
        return self.policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


# @register_config("BrownianMotionPickPlaceColorEvalConfig")
class BrownianMotionPickPlaceColorEvalConfig(BrownianMotionPickPlaceEvalConfig):
    wandb_name: str = f"brownian_motion_pick_place_color_eval_{TIMESTAMP}"
    task_type: str = "pick_and_place_color"

    task_sampler_config: PickAndPlaceColorTaskSamplerConfig = PickAndPlaceColorTaskSamplerConfig(
        task_sampler_class=PickAndPlaceColorTaskSampler,
        house_inds=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145],
        samples_per_house=3,
    )
    task_config: PickAndPlaceColorTaskConfig = PickAndPlaceColorTaskConfig(
        task_cls=PickAndPlaceColorTask
    )


class DreamZeroPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: DreamZeroPolicyConfig = DreamZeroPolicyConfig()
    policy_dt_ms: float = 66.0

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class DummyNavToObjEvalConfig(NavToObjDataGenConfig):
    """Evaluation config for Dummy pick and place."""

    wandb_project: str = "dummy-eval"
    use_wandb: bool = False
    use_passive_viewer: bool = False
    wandb_name: str = f"dummy_nav_to_obj_eval_{TIMESTAMP}"
    filter_for_successful_trajectories: bool = False
    task_type: str = "nav_to_obj"
    task_horizon: int = 600
    output_dir: Path = Path("eval_output") / f"dummy_{TIMESTAMP}"

    task_sampler_config: NavToObjTaskSamplerConfig = NavToObjTaskSamplerConfig(
        task_sampler_class=NavToObjTaskSampler,
        house_inds=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145],
        samples_per_house=3,
    )
    task_config: NavToObjTaskConfig = NavToObjTaskConfig(task_cls=NavToObjTask)

    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    def _init_policy_config(self) -> DummyPolicyConfig:
        self.policy_config.policy_cls = DummyPolicy
        self.policy_config.policy_factory = make_lenient(DummyPolicy)
        return self.policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class BrownianNavToObjEvalConfig(DummyNavToObjEvalConfig):
    policy_config: BrownianMotionPolicyConfig = BrownianMotionPolicyConfig()

    def _init_policy_config(self) -> BrownianMotionPolicyConfig:
        self.policy_config.policy_cls = BrownianMotionPolicy
        self.policy_config.policy_factory = make_lenient(BrownianMotionPolicy)
        return self.policy_config
