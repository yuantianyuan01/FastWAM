"""
JSON-based benchmark schema definitions.

This module defines Pydantic models for JSON benchmark files that fully specify
episode initialization without relying on pickle serialization. Each episode
is fully self-contained and can be loaded/inspected independently.

Design principles:
    - Each episode JSON is fully self-contained (no external config dependencies)
    - A benchmark is simply a list/directory of episode JSONs (can mix task types)
    - All fields needed to recreate exact initial conditions are explicit
    - Task horizon is NOT stored per-episode - it's an evaluation parameter

Benchmark directory structure:
    benchmark_dir/
    ├── house_5/
    │   ├── episode_00000000.json  # Fully self-contained
    │   └── ...
    └── ...

Key fields for robot placement:
    - robot.init_qpos: Initial joint positions per move group
    - task.robot_base_pose: Robot base pose in world frame

The actual robot world placement comes from task.robot_base_pose, which is set
by the task sampler and frozen into the episode.

Task horizons:
    Task horizon (max steps per episode) is an EVALUATION parameter, not a task
    specification. Use DEFAULT_TASK_HORIZONS for sensible defaults per task class,
    and override via command line for specific eval runs.
"""

import copy
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class RobotSpec(BaseModel):
    """Robot initialization specification.

    Note: Robot world placement is in task.robot_base_pose, not here.
    This spec only contains robot-intrinsic state (joint positions).
    """

    # Robot identifier for factory lookup (e.g. "franka_droid", "rby1")
    robot_name: str

    # Initial joint positions per move group (e.g. {"arm": [...], "gripper": [...]})
    init_qpos: dict[str, list[float]]


class RobotMountedCameraSpec(BaseModel):
    """Specification for a camera mounted on the robot."""

    name: str
    type: Literal["robot_mounted"] = "robot_mounted"
    reference_body_names: list[str]
    camera_offset: list[float] = Field(..., min_length=3, max_length=3)
    lookat_offset: list[float] = Field(..., min_length=3, max_length=3)
    camera_quaternion: list[float] = Field(..., min_length=4, max_length=4)
    fov: float
    record_depth: bool = False  # Whether to record depth images for this camera


class ExocentricCameraSpec(BaseModel):
    """Specification for an exocentric (fixed) camera."""

    name: str
    type: Literal["exocentric"] = "exocentric"
    pos: list[float] = Field(..., min_length=3, max_length=3)
    up: list[float] = Field(..., min_length=3, max_length=3)
    forward: list[float] = Field(..., min_length=3, max_length=3)
    fov: float
    record_depth: bool = False  # Whether to record depth images for this camera


CameraSpec = RobotMountedCameraSpec | ExocentricCameraSpec


class SceneModificationsSpec(BaseModel):
    """Scene modifications required for this episode.

    This captures objects that need to be added to the base scene XML
    and their initial poses.
    """

    # Objects to add: name -> path relative to ASSETS_DIR
    # e.g. {"place_receptacle/0/Bowl_27": "objects/thor/Bowl_27.xml"}
    added_objects: dict[str, str] = Field(default_factory=dict)

    # Object poses: name -> [x, y, z, qw, qx, qy, qz]
    # Includes both added objects and existing scene objects that need repositioning
    object_poses: dict[str, list[float]] = Field(default_factory=dict)

    # Objects to remove from the base scene: list of object names
    # These objects will be removed from the scene spec before adding auxiliary objects
    removed_objects: list[str] = Field(default_factory=list)


class BaseTaskSpec(BaseModel):
    """Base task specification with fields common to all task types.

    robot_base_pose is the authoritative field for robot world placement.
    This comes from task_config in the codebase, not robot_config.

    task_cls is the authoritative identifier for the task type. The eval task
    sampler is responsible for interpreting task_cls and creating the appropriate
    task. task_type is optional and for human convenience only.
    """

    # Task identification - task_cls is authoritative
    task_cls: str  # Fully qualified class name, e.g. "molmo_spaces.tasks.pick_task.PickTask"
    task_type: str | None = None  # Optional human-readable type (e.g. "pick", "pick_and_place")

    # Robot world placement [x, y, z, qw, qx, qy, qz]
    # This is the actual field used for robot placement
    robot_base_pose: list[float] = Field(..., min_length=7, max_length=7)


class PickTaskSpec(BaseTaskSpec):
    """Task-specific parameters for pick tasks."""

    pickup_obj_name: str
    pickup_obj_start_pose: list[float] = Field(..., min_length=7, max_length=7)
    pickup_obj_goal_pose: list[float] | None = Field(default=None, min_length=7, max_length=7)
    receptacle_name: str | None = None

    # Success criteria
    succ_pos_threshold: float = 0.01  # lift height threshold in meters


class PickAndPlaceTaskSpec(PickTaskSpec):
    """Task-specific parameters for pick and place tasks."""

    place_receptacle_name: str
    place_receptacle_start_pose: list[float] = Field(..., min_length=7, max_length=7)

    # Success criteria
    receptacle_supported_weight_frac: float = 0.5
    max_place_receptacle_pos_displacement: float = 0.15
    max_place_receptacle_rot_displacement: float = np.deg2rad(60)


class PickAndPlaceNextToTaskSpec(PickAndPlaceTaskSpec):
    """Task-specific parameters for pick and place next-to tasks."""

    min_surface_to_surface_gap: float | None = None
    max_surface_to_surface_gap: float | None = None


class PickAndPlaceColorTaskSpec(PickAndPlaceTaskSpec):
    """Task-specific parameters for pick and place color tasks."""

    object_colors: dict[str, list[float]] | None = None  # object name -> rgba
    other_receptacle_names: list[str] | None = None
    other_receptacle_start_poses: dict[str, list[float]] | None = None


class OpenCloseTaskSpec(BaseTaskSpec):
    """Task-specific parameters for open/close tasks."""

    pickup_obj_name: str  # The articulated object name
    pickup_obj_start_pose: list[float] = Field(..., min_length=7, max_length=7)
    articulation_object_name: str | None = None
    joint_name: str
    joint_index: int = 0
    joint_start_position: float | list[float]
    joint_goal_position: float | None = None

    # Success criteria
    task_success_threshold: float = 0.20  # percentage of opening
    any_inst_of_category: bool = False


class NavToObjTaskSpec(BaseTaskSpec):
    """Task-specific parameters for navigation to object tasks."""

    pickup_obj_name: str  # Target object instance name
    pickup_obj_candidates: list[str] | None = None  # All candidate instances
    pickup_obj_start_pose: list[float] | None = Field(default=None, min_length=7, max_length=7)
    receptacle_name: str | None = None

    # Success criteria
    succ_pos_threshold: float = 1.5  # meters


class DoorOpeningTaskSpec(BaseTaskSpec):
    """Task-specific parameters for door opening tasks."""

    door_body_name: str  # e.g., "door|2|8_Doorway_Double_7_doorway_door_7"
    articulated_joint_range: list[float] | None = None
    articulated_joint_reset_state: list[float] | None = None

    # Success criteria
    door_openness_threshold: float = 0.67  # percentage of door opening


TaskSpec = (
    PickTaskSpec
    | PickAndPlaceTaskSpec
    | PickAndPlaceColorTaskSpec
    | PickAndPlaceNextToTaskSpec
    | OpenCloseTaskSpec
    | NavToObjTaskSpec
    | DoorOpeningTaskSpec
)

# All TaskSpec subclasses for introspection
ALL_TASK_SPEC_CLASSES: list[type[BaseTaskSpec]] = [
    PickTaskSpec,
    PickAndPlaceTaskSpec,
    PickAndPlaceNextToTaskSpec,
    PickAndPlaceColorTaskSpec,
    OpenCloseTaskSpec,
    NavToObjTaskSpec,
    DoorOpeningTaskSpec,
]

# Fields that are metadata about the task, not configuration to copy
_TASK_METADATA_FIELDS = {"task_cls", "task_type"}


def get_task_spec_field_names() -> set[str]:
    """Get all field names from TaskSpec models that should be copied to task_config.

    Returns the union of all fields from all TaskSpec subclasses, excluding
    metadata fields (task_cls, task_type) which identify the task but aren't
    configuration values.

    This is derived from the Pydantic models to stay in sync automatically.
    """
    fields: set[str] = set()
    for spec_cls in ALL_TASK_SPEC_CLASSES:
        fields.update(spec_cls.model_fields.keys())
    return fields - _TASK_METADATA_FIELDS


class LanguageSpec(BaseModel):
    """Natural language task specification."""

    task_description: str

    # Semantic referral expressions for objects
    # e.g. {"pickup_name": "red mug", "place_name": "white bowl"}
    referral_expressions: dict[str, str] = Field(default_factory=dict)

    # Prioritized expressions with CLIP scores (optional, for analysis)
    # Each entry is [clip_score_diff, clip_score, expression_text]
    # Using list instead of tuple for JSON compatibility
    referral_expressions_priority: dict[str, list[list[float | str]]] = Field(default_factory=dict)


class SourceSpec(BaseModel):
    """Provenance information for this episode.

    Tracks where this episode specification came from (which H5 file and trajectory).
    """

    h5_file: str  # Full path to the source H5 file
    traj_key: str  # Trajectory key within the H5 file (e.g. "traj_0")
    episode_length: int | None = None  # Number of steps in original episode (proxy for difficulty)
    camera_system_class: str | None = (
        None  # CameraSystemConfig class used (e.g. "FrankaDroidCameraSystem")
    )
    source_data_date: str | None = None  # Approx date source H5 files were created (YYYY-MM-DD)
    benchmark_created_date: str | None = None  # Date this benchmark JSON was created (YYYY-MM-DD)


class EpisodeSpec(BaseModel):
    """Complete specification for a single benchmark episode.

    This is a FULLY SELF-CONTAINED specification - no external config needed.
    Contains all information needed to recreate the exact initial conditions
    for an episode: scene, robot, cameras, and task parameters.

    NOTE: Timing/execution parameters (policy_dt_ms, ctrl_dt_ms, sim_dt_ms,
    task_horizon) are NOT stored per-episode. They come from the evaluation
    config or command line.

    A benchmark is simply a list of EpisodeSpec objects in a single JSON file.
    """

    # === Source/provenance ===
    source: SourceSpec | None = None  # Where this episode came from (H5 file, traj key)

    # === Scene identification ===
    house_index: int
    scene_dataset: str  # e.g. "procthor-objaverse", "procthor-10k"
    data_split: str = "val"  # e.g. "train", "val", "test"
    seed: int | None = None

    # === Robot setup ===
    robot: RobotSpec

    # === Camera setup ===
    # img_resolution: (width, height) for all cameras in this episode
    img_resolution: tuple[int, int]
    cameras: list[CameraSpec] = Field(default_factory=list)

    # === Scene modifications ===
    scene_modifications: SceneModificationsSpec = Field(default_factory=SceneModificationsSpec)

    # === Task specification ===
    # Contains task_type, task_cls, robot_base_pose, and task-specific fields
    task: dict  # Flexible dict; can be validated against TaskSpec subtypes

    # === Task-relevant objects ===
    # Object body names that must be visible for camera placement (e.g. pickup object,
    # receptacles). Used by the eval camera system for visibility checks and workspace
    # center computation. Computed at benchmark creation time from task config fields.
    # Empty list for legacy benchmarks or tasks without visibility requirements.
    task_relevant_objects: list[str] = Field(default_factory=list)

    # === Language specification ===
    language: LanguageSpec

    class Config:
        # Allow extra fields for forward compatibility
        extra = "allow"

    @classmethod
    def from_json_file(cls, path: str | Path) -> "EpisodeSpec":
        """Load an episode spec from a JSON file."""
        path = Path(path)
        with open(path) as f:
            import json

            data = json.load(f)
        return cls.model_validate(data)

    def to_json_file(self, path: str | Path) -> None:
        """Save the episode spec to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            import json

            json.dump(self.model_dump(), f, indent=2)

    def get_task_type(self) -> str | None:
        """Get optional human-readable task type from task dict."""
        return self.task.get("task_type")

    def get_task_cls(self) -> str:
        """Get fully qualified task class name from task dict (authoritative identifier)."""
        task_cls = self.task.get("task_cls")
        if not task_cls:
            raise ValueError("task dict missing required 'task_cls' field")
        # TODO(max): XXX remove this
        # if "molmo_spaces" in task_cls:  # TODO(rose): forking breanch
        #    print("XXX patching config molmo_spaces->mujoco_thor")
        #    task_cls = task_cls.replace("molmo_spaces", "mujoco_thor")
        return task_cls


class BenchmarkMetadata(BaseModel):
    """Optional metadata for a benchmark directory.

    This is NOT required - each episode is fully self-contained.
    This file provides optional human-readable metadata about the benchmark.
    """

    # Human-readable description
    description: str | None = None
    created_at: str | None = None  # ISO timestamp
    source_datagen_path: str | None = None  # Original datagen output path

    # Summary statistics (computed, not prescriptive)
    num_episodes: int | None = None
    num_houses: int | None = None

    # Task class info - keyed by task_cls string (e.g. "molmo_spaces.tasks.pick_task.PickTask")
    task_cls_counts: dict[str, int] | None = None  # Count per task class

    # Object category info (for manipulation tasks)
    object_category_counts: dict[str, int] | None = None  # Count per object category

    # Robot info
    robot_counts: dict[str, int] | None = None  # Count per robot name

    # Episode length statistics (from source episodes, proxy for difficulty)
    episode_length_stats: dict[str, float] | None = None  # min, max, mean, median

    # House distribution
    house_counts: dict[int, int] | None = None  # Count per house_index

    # Camera system info
    camera_system_class: str | None = None  # CameraSystemConfig class name

    # Provenance dates
    source_data_date: str | None = None  # Approximate date source H5 files were created
    benchmark_created_date: str | None = None  # Date this benchmark JSON was created

    class Config:
        extra = "allow"

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BenchmarkMetadata":
        """Load benchmark metadata from a JSON file."""
        path = Path(path)
        with open(path) as f:
            import json

            data = json.load(f)
        return cls.model_validate(data)

    def to_json_file(self, path: str | Path) -> None:
        """Save the benchmark metadata to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            import json

            json.dump(self.model_dump(), f, indent=2)


def load_benchmark(
    benchmark_dir: Path,
) -> tuple[BenchmarkMetadata | None, dict[int, list[Path]]]:
    """Load a benchmark directory.

    A benchmark is simply a directory of episode JSON files. Each episode is
    fully self-contained. An optional benchmark_metadata.json provides human-readable
    info but is not required.

    Args:
        benchmark_dir: Path to benchmark directory containing house_* subdirectories
            with episode JSON files. May optionally contain benchmark_metadata.json.

    Returns:
        Tuple of (BenchmarkMetadata or None, dict mapping house_id -> list of episode JSON paths)
    """

    # Load optional metadata (not required)
    metadata: BenchmarkMetadata | None = None
    metadata_path = benchmark_dir / "benchmark_metadata.json"
    if metadata_path.exists():
        metadata = BenchmarkMetadata.from_json_file(metadata_path)

    # Discover episode files organized by house
    episodes_by_house: dict[int, list[Path]] = {}
    for house_dir in sorted(benchmark_dir.glob("house_*")):
        if not house_dir.is_dir():
            continue
        house_id = int(house_dir.name.replace("house_", ""))
        episode_files = sorted(house_dir.glob("episode_*.json"))
        if episode_files:
            episodes_by_house[house_id] = episode_files

    return metadata, episodes_by_house


def load_all_episodes(benchmark_dir: Path) -> list[EpisodeSpec]:
    """Load all episodes from a benchmark directory as a flat list.

    Supports two formats:
    1. Single benchmark.json file (preferred): List of EpisodeSpec dicts
    2. Legacy house_*/episode_*.json structure

    Args:
        benchmark_dir: Path to benchmark directory

    Returns:
        List of EpisodeSpec objects
    """
    import json

    # Try new single-file format first
    benchmark_file = benchmark_dir / "benchmark.json"
    if benchmark_file.exists():
        with open(benchmark_file) as f:
            data = json.load(f)
        return [EpisodeSpec.model_validate(ep) for ep in data]

    # Fall back to legacy directory structure
    _, episodes_by_house = load_benchmark(benchmark_dir)
    episodes = []
    for episode_paths in episodes_by_house.values():
        for path in episode_paths:
            episodes.append(EpisodeSpec.from_json_file(path))
    return episodes


def replace_target_object_with_custom(
    episode: EpisodeSpec,
    custom_object_path: str | Path,
    custom_object_name: str | None = None,
) -> EpisodeSpec:
    """Replace the target object in an episode with a custom object.

    This function:
    1. Identifies the target object from the task specification (e.g., pickup_obj_name)
    2. Gets the target object's pose from task or scene_modifications
    3. Removes the target object from scene_modifications if it's an added object
    4. Adds the custom object to scene_modifications with the same pose
    5. Updates the task specification to reference the new custom object

    Args:
        episode: The episode specification to modify
        custom_object_path: Path to the custom object XML file (relative to ASSETS_DIR or absolute)
        custom_object_name: Optional natural language name for the custom object (e.g., 'lemon').
            If not provided, will extract from the XML body name.

    Returns:
        A new EpisodeSpec with the target object replaced by the custom object

    Raises:
        ValueError: If the episode doesn't have a target object or if required fields are missing
    """

    log = logging.getLogger(__name__)

    # Create a deep copy to avoid modifying the original
    modified_episode = copy.deepcopy(episode)

    # Get target object name from task - most tasks use pickup_obj_name
    task = modified_episode.task
    target_obj_name = task.get("pickup_obj_name")

    if not target_obj_name:
        raise ValueError(
            f"Episode task does not have a pickup_obj_name field. "
            f"Task type: {task.get('task_cls', 'unknown')}"
        )

    # Get target object pose - prefer pickup_obj_start_pose from task, fall back to object_poses
    target_obj_pose = None
    if "pickup_obj_start_pose" in task:
        target_obj_pose = task["pickup_obj_start_pose"]
    elif target_obj_name in modified_episode.scene_modifications.object_poses:
        target_obj_pose = modified_episode.scene_modifications.object_poses[target_obj_name]
    else:
        raise ValueError(
            f"Could not find pose for target object '{target_obj_name}'. "
            f"Expected either pickup_obj_start_pose in task or object_poses entry."
        )

    # Ensure we have a valid pose (7 elements: x, y, z, qw, qx, qy, qz)
    if len(target_obj_pose) != 7:
        raise ValueError(
            f"Target object pose must have 7 elements [x, y, z, qw, qx, qy, qz], "
            f"got {len(target_obj_pose)} elements"
        )

    # Convert custom_object_path to string and ensure it's relative to ASSETS_DIR if needed
    custom_obj_path_str = str(custom_object_path)

    # Determine the custom object body name to use
    if custom_object_name:
        # Use the provided custom object name
        custom_obj_body_name = custom_object_name
        log.info(f"Using provided custom object name: '{custom_obj_body_name}'")
    else:
        raise ValueError(
            "No custom object name provided. "
            "Please provide a custom object name using --custom_object_name."
        )

    # Generate a new name for the custom object
    # Use a prefix to avoid conflicts, and use the body name (either provided or from XML)
    custom_obj_name = f"custom_object/{custom_obj_body_name}"

    # Remove the target object from scene_modifications if it's an added object
    # Also add it to removed_objects to ensure it's removed from the base scene if present
    if target_obj_name in modified_episode.scene_modifications.added_objects:
        del modified_episode.scene_modifications.added_objects[target_obj_name]
        log.info(f"Removed target object '{target_obj_name}' from added_objects")

    if target_obj_name in modified_episode.scene_modifications.object_poses:
        del modified_episode.scene_modifications.object_poses[target_obj_name]
        log.info(f"Removed target object '{target_obj_name}' from object_poses")

    # Add to removed_objects to ensure it's removed from base scene if it exists there
    if target_obj_name not in modified_episode.scene_modifications.removed_objects:
        modified_episode.scene_modifications.removed_objects.append(target_obj_name)
        log.info(
            f"Added target object '{target_obj_name}' to removed_objects for base scene removal"
        )

    # Add the custom object to scene_modifications
    modified_episode.scene_modifications.added_objects[custom_obj_name] = custom_obj_path_str
    modified_episode.scene_modifications.object_poses[custom_obj_name] = target_obj_pose.copy()

    # Update task to reference the new custom object
    task["pickup_obj_name"] = custom_obj_name

    # Also update pickup_obj_start_pose to match (in case task uses it)
    task["pickup_obj_start_pose"] = target_obj_pose.copy()

    log.info(
        f"Replaced target object '{target_obj_name}' with custom object '{custom_obj_name}' "
        f"at path '{custom_obj_path_str}'"
    )

    return modified_episode
