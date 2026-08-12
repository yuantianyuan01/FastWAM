"""Task configuration classes for MolmoSpaces experiments."""

from pathlib import Path
from typing import TypeAlias

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_config import Config


class BaseMujocoTaskConfig(Config):
    """Base configuration for MuJoCo tasks.

    NOTE:
    If these task config parameters are left to None, they will be sampled by the task sampler.
    If these task config parameters are not None, their value will take precedence over
    any parameters sampled by the task sampler and will remain fixed across all simulation
    tasks sampled by the task sampler.
    """

    task_cls: type | None  # [AbstractMujocoTask]

    # dict of object names to xml locations
    added_objects: dict[str, Path] = {}

    # Object positions (for internal use by eval_task_sampler)
    # dict of object names to world poses
    object_poses: dict[str, list[float]] | None = None

    # Map from task-relevant role (e.g. 'object_name', 'pickup_name',
    # or 'place_name') to the chosen referral expression
    referral_expressions: dict[str, str] = {}

    # Map from task-relevant role (e.g. 'object_name', 'pickup_name',
    # or 'place_name') to prioritized referral expressions.
    # Each represented referral expression is represented as a tuple with
    #  - CLIP score difference between the actual target and the most similar other object in the context,
    #  - CLIP score for the expression and the actual target
    #  - the referral expression/description
    referral_expressions_priority: dict[str, list[tuple[float, float, str]]] = {}

    robot_base_pose: list[float] | None = None  # initial robot base pose, xyz + quat

    # Sensor settings (common to all task types)
    use_sensors: bool = True  # Whether to use the sensor system
    tracked_object_names: list[str] | None = None  # Object names for ObjectPoseSensor
    action_dtype: str = "float32"  # Enforced dtype for all action components


class PickTaskConfig(BaseMujocoTaskConfig):
    """Configuration for Franka move-to-pose task."""

    task_cls: type | None = None  # Will be set by importing module to avoid circular imports

    # Object names
    pickup_obj_start_pose: list[float] | None = None
    pickup_obj_goal_pose: list[float] | None = None
    receptacle_name: str | None = None
    place_target_name: str | None = None
    pickup_obj_name: str | None = None

    # Task parameters
    succ_pos_threshold: float = 0.01  # lower threshold lift height in meters
    # succ_rot_threshold: float = 0.15  # Rotation success threshold in radians

    # Rendering settings
    enable_rendering: bool = True  # Whether to enable environment rendering for visual sensors


class PickAndPlaceTaskConfig(PickTaskConfig):
    place_receptacle_name: str | None = None
    place_receptacle_start_pose: list[float] | None = None
    succ_pos_threshold: float = np.inf  # no position success threshold, we use support instead
    receptacle_supported_weight_frac: float = (
        0.5  # how much of the object weight should be supported by the receptacle
    )
    max_place_receptacle_pos_displacement: float = (
        0.1  # maximum distance the receptacle can be moved
    )
    max_place_receptacle_rot_displacement: float = np.radians(
        45
    )  # maximum rotation the receptacle can be rotated
    carry_forward_rel_pos_threshold: float = 0.005  # meters
    carry_forward_rel_rot_threshold: float = np.radians(10)  # radians


class PickAndPlaceNextToTaskConfig(PickAndPlaceTaskConfig):
    max_place_receptacle_pos_displacement: float = (
        0.05  # maximum distance the receptacle can be moved
    )
    max_place_receptacle_rot_displacement: float = np.radians(45)
    # actually task success
    min_surface_to_surface_gap: float = 0
    max_surface_to_surface_gap: float = 0.05


class PickAndPlaceColorTaskConfig(PickAndPlaceTaskConfig):
    object_colors: dict[str, list[float]] | None = None  # object name -> rgba
    other_receptacle_names: list[str] | None = None
    other_receptacle_start_poses: dict[str, list[float]] | None = None


class PackingTaskConfig(PickAndPlaceTaskConfig):
    pass


class OpeningTaskConfig(PickTaskConfig):
    """Configuration for opening task."""

    # --- Opening-specific task parameters ---
    articulation_object_name: str | None = None  # e.g., "door|2|8_Doorway_Double_7_doorway_door_7"
    joint_name: str | None = None  # e.g., "joint_0"
    joint_index: int | None = 0  # index of the joint to open
    joint_start_position: float | None = None
    joint_goal_position: float | None = None

    # Success criteria
    any_inst_of_category: bool = False  # for open, reward for any instance of category
    task_success_threshold: float = 0.20  # percentage of opening

    # Rendering settings
    enable_rendering: bool = True  # Whether to enable environment rendering for visual sensors


class DoorOpeningTaskConfig(BaseMujocoTaskConfig):
    """Configuration for RBY1 door opening task.

    NOTE:
    If these task config parameters are left to None, they will be sampled by the task sampler.
    If these task config parameters are not None, their value will take precedence over
    any parameters sampled by the task sampler and will remain fixed across all simulation
    tasks sampled by the task sampler.
    """

    task_cls: type = None  # Will be set by importing module to avoid circular imports

    # --- DoorOpening-specific task parameters ---
    door_body_name: str | None = None  # e.g., "door|2|8_Doorway_Double_7_doorway_door_7"
    articulated_joint_range: np.ndarray | None = None
    articulated_joint_reset_state: np.ndarray | None = None
    additional_tcp_rotation_offset_mat: np.ndarray = (
        R.from_euler("Y", -90, degrees=True)
    ).as_matrix()
    additional_tcp_offset_distance: float = 0.03  # optional distance to move tcp closer/farther from the door handle (tune as per gripper / grasping requirements)

    # Reward function
    door_open_reward: float = 1.0

    # Success criteria
    door_openness_threshold: float = 0.67  # percentage of door opening

    # Debug visualizations
    viz_target_ee: bool = True  # Visualize target end-effector pose


class NavToObjTaskConfig(BaseMujocoTaskConfig):
    """Configuration for RBY1 navigation to object task.

    NOTE:
    If these task config parameters are left to None, they will be sampled by the task sampler.
    If these task config parameters are not None, their value will take precedence over
    any parameters sampled by the task sampler and will remain fixed across all simulation
    tasks sampled by the task sampler.
    Uses pickup_obj_name for compatibility with EvalTaskSampler.
    """

    task_cls: type | None = None  # Will be set by importing module to avoid circular imports
    seed: int | None = None

    # Object navigation-specific task parameters (using pickup_obj_* naming for compatibility)
    pickup_obj_name: str | None = None  # Target object instance name
    pickup_obj_candidates: list[str] | None = (
        None  # List of all candidate object instances of this type
    )
    pickup_obj_category: str | None = None  # Semantic category (e.g., "apple")
    pickup_obj_synset: str | None = None  # WordNet synset (e.g., "apple.n.01")

    # For compatibility with EvalTaskSampler (not used in nav tasks, but needed for shared code)
    receptacle_name: str | None = None
    pickup_obj_start_pose: list[float] | None = None

    # Task parameters
    succ_pos_threshold: float = 1.5  # meters  # Success distance threshold in meters

    # Rendering settings
    enable_rendering: bool = True  # Whether to enable environment rendering for visual sensors


AllTaskConfigs: TypeAlias = (
    BaseMujocoTaskConfig
    | PickTaskConfig
    | PickAndPlaceTaskConfig
    | PickAndPlaceColorTaskConfig
    | PickAndPlaceNextToTaskConfig
    | PackingTaskConfig
    | OpeningTaskConfig
    | DoorOpeningTaskConfig
    | NavToObjTaskConfig
)
