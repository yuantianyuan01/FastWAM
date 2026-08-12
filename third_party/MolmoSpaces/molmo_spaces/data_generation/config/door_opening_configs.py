from __future__ import annotations

from pathlib import Path

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import RBY1GoProD455CameraSystem
from molmo_spaces.configs.policy_configs import (
    DoorOpeningPolicyConfig,
)
from molmo_spaces.configs.robot_configs import RBY1MConfig
from molmo_spaces.configs.task_configs import DoorOpeningTaskConfig
from molmo_spaces.configs.task_sampler_configs import (
    DoorOpeningTaskSamplerConfig,
)
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import (
    ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR,
    get_robot_path,
)
from molmo_spaces.tasks.opening_task_samplers import (
    DoorOpeningTaskSampler,
)
from molmo_spaces.tasks.opening_tasks import DoorOpeningTask
from molmo_spaces.utils.profiler_utils import Profiler


@register_config("DoorOpeningDataGenConfig")
class DoorOpeningDataGenConfig(MlSpacesExpConfig):
    """
    All-ProcTHOR variant for RBY1 door opening dataset generation.

    Iterates through multiple houses from the ProcTHOR dataset for large-scale data generation.
    This is the main config - use DoorOpeningDebugConfig for single-scene testing.
    """

    num_envs: int = 1  # Number of environments to run in each thread
    task_type: str = "door_open"
    use_passive_viewer: bool = False  # Launch passive viewer for rendering
    viewer_cam_dict: dict = {
        "camera": "robot_0/camera_follower"
    }  # Dictionary containing viewer camera parameters
    policy_dt_ms: float = 100.0  # Default policy time step
    ctrl_dt_ms: float = 20.0  # Default control time step
    sim_dt_ms: float = 4.0  # Default simulation time step
    task_horizon: int = 400  # Maximum number of steps per episode

    # --- Data generation settings ---
    num_workers: int = 1  # Number of parallel worker processes for data generation
    profile: bool = False  # Whether to profile the data generation pipeline
    profiler: Profiler | None = None  # Profiler()
    output_dir: Path = (
        ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR / "experiment_output"
    )  # Directory to save generated data
    use_wandb: bool = False  # Whether to use Weights & Biases logging
    wandb_name: str | None = None  # Weights & Biases run name
    wandb_project: str = "molmo-spaces-data-generation"  # Weights & Biases project name

    # --- ProcTHOR dataset configuration ---
    scene_dataset: str = "procthor-10k"  # Name of the scene dataset to load
    data_split: str = "train"  # Data split to use
    # Robot configuration (imported from robot_configs.py)
    robot_config: RBY1MConfig = RBY1MConfig()

    # Camera configuration (imported from camera_configs.py)
    camera_config: RBY1GoProD455CameraSystem = RBY1GoProD455CameraSystem()

    # Task sampler configuration (imported from task_sampler_configs.py)
    task_sampler_config: DoorOpeningTaskSamplerConfig = DoorOpeningTaskSamplerConfig(
        task_sampler_class=DoorOpeningTaskSampler
    )

    # Task configuration (imported from task_configs.py)
    task_config: DoorOpeningTaskConfig = DoorOpeningTaskConfig(task_cls=DoorOpeningTask)

    # Policy configuration (imported from policy_configs.py)
    # Will be initialized in model_post_init
    policy_config: DoorOpeningPolicyConfig | None = None

    def _init_policy_config(self) -> DoorOpeningPolicyConfig:
        """Initialize policy config with dynamically computed planner configs"""
        # Import GPU-requiring modules only when actually creating policy (requires GPU)
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig
        from molmo_spaces.policy.solvers.opening_solver import DoorOpeningPlannerPolicy

        # Setup curobo planner configs with current ctrl_dt_ms
        rby1m_path = get_robot_path("rby1m")

        left_curobo_planner_config = CuroboPlannerConfig(
            curobo_robot_config_path=str(
                rby1m_path / "curobo_config" / "rby1m_left_arm_holobase.yml"
            ),
            urdf_path=str(rby1m_path / "curobo_config" / "urdf" / "model_holobase.urdf"),
            asset_root_path=str(rby1m_path / "curobo_config" / "urdf" / "meshes"),
            usd_robot_root=str(rby1m_path / "curobo_config"),
            collision_spheres_path=str(rby1m_path / "curobo_config" / "rby1m_holobase_spheres.yml"),
            interpolation_dt=self.ctrl_dt_ms / 1000.0,  # 1x control dt
        )
        right_curobo_planner_config = CuroboPlannerConfig(
            curobo_robot_config_path=str(
                rby1m_path / "curobo_config" / "rby1m_right_arm_holobase.yml"
            ),
            urdf_path=str(rby1m_path / "curobo_config" / "urdf" / "model_holobase.urdf"),
            asset_root_path=str(rby1m_path / "curobo_config" / "urdf" / "meshes"),
            usd_robot_root=str(rby1m_path / "curobo_config"),
            collision_spheres_path=str(rby1m_path / "curobo_config" / "rby1m_holobase_spheres.yml"),
            interpolation_dt=self.ctrl_dt_ms / 1000.0,  # 1x control dt
        )

        return DoorOpeningPolicyConfig(
            policy_cls=DoorOpeningPlannerPolicy,
            policy_factory=DoorOpeningPlannerPolicy,
            left_curobo_planner_config=left_curobo_planner_config,
            right_curobo_planner_config=right_curobo_planner_config,
        )

    def model_post_init(self, __context) -> None:
        """Initialize policy config after Pydantic model initialization"""
        super().model_post_init(__context)
        # Set up policy config with dynamically computed planner configs
        # Skip if no GPU available (e.g., when launching jobs from manager)
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

        # Auto-create profiler instance if profiling is enabled
        if self.profile and self.profiler is None:
            self.profiler = Profiler()

    @property
    def tag(self) -> str:
        return "rby1_door_opening_all_procthor"


@register_config("DoorOpeningDebugConfig")
class DoorOpeningDebugConfig(DoorOpeningDataGenConfig):
    """
    Debug config for door opening dataset generation.
    """

    num_workers: int = 1
    use_passive_viewer: bool = True
    filter_for_successful_trajectories: bool = False
    seed: int | None = 83067780
    policy_dt_ms: float = 100.0
    output_dir: Path = (
        ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR / "experiment_output" / "door_opening_debug"
    )
    task_horizon: int = 1000

    task_sampler_config: DoorOpeningTaskSamplerConfig = DoorOpeningTaskSamplerConfig(
        task_sampler_class=DoorOpeningTaskSampler,
        samples_per_house=1,
        house_inds=[22],
    )

    def tag(self) -> str:
        return "rby1_door_opening_debug"
