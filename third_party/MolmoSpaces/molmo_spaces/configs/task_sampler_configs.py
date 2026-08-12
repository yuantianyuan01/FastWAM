"""Task sampler configuration classes for MolmoSpaces experiments."""

import math

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.utils.constants.object_constants import RECEPTACLE_TYPES_THOR


class BaseMujocoTaskSamplerConfig(Config):
    """Base configuration for task samplers.

    A task is sampled based on this configuration.
    """

    task_sampler_class: type | None = None  # [AbstractMujocoTaskSampler]
    house_inds: list[int] | None  # List of thor house indices to use
    scene_xml_paths: list[str] | None = None
    samples_per_house: int | None  # Number of tasks to sample per house
    episodes_per_batch: int = 4  # Max episodes per work item; houses are split into batches of this size for parallel processing
    task_batch_size: int
    max_tasks: int | None  # Maximum number of tasks to sample
    sim_settle_timesteps: int = 500
    verbose: bool = False  # Whether to print verbose logging
    randomize_lighting: bool = False  # Whether to randomize the lighting of the scene
    randomize_textures: bool = False  # Whether to randomize the textures of the scene
    randomize_textures_all: bool = False  # Whether to randomize the textures of the scene
    randomize_robot_textures: bool = False  # Whether to randomize the textures of the robot
    randomize_dynamics: bool = False  # Whether to randomize the dynamics of the scene

    # Failure recovery parameters (used by ParallelRolloutRunner)
    max_allowed_sequential_task_sampler_failures: int = 10
    max_allowed_sequential_rollout_failures: int = 10
    max_allowed_sequential_irrecoverable_failures: int = 5
    max_total_attempts_multiplier: int = (
        6  # Max attempts = samples_per_house * multiplier. Just to bound this a little
    )

    # Asset blacklist: after this many failures for a single asset, skip it for the rest of the run
    max_asset_failures: int = 10

    # Robot placement visibility checking
    # NOTE: Disabled by default for performance - visibility checking renders segmentation
    # frames which is expensive. Enable only if you have cameras with visibility_constraints.
    check_robot_placement_visibility: bool = True

    robot_placement_exclusion_threshold: float = 0.15

    robot_placement_rotation_range_rad: float = math.radians(45)

    # Scene configuration
    enable_texture_randomization: bool = False
    house_variant: str = "ceiling"


class ObjectCentricTaskSamplerConfig(BaseMujocoTaskSamplerConfig):
    # Note: Abhay's request.

    # Dataset configuration
    dataset_name: str = "procthor-10k"

    # Object names in the scene (will be set dynamically during sampling)
    pickup_obj_name: str | None = None  # Will be selected from existing small objects in scene

    # Using pickup_types for compatibility with EvalTaskSampler
    pickup_types: list[str] | None = (
        None  # List of object types for navigation targets (None means use all pickup objects)
    )

    # Oversample objaverse assets because they are quite rare, to get more balanced
    # final distributions of samples. (Roughly 20% objaverse assets in data)
    objaverse_oversampling_factor: int = 30

    filter_for_grasps: bool = True  # only sample objects with valid grasp files
    # grasp libraries to use for filtering, if None all available libraries will be used
    grasp_libraries: list[str] | None = None


class PickTaskSamplerConfig(ObjectCentricTaskSamplerConfig):
    """Configuration for Franka move-to-pose task sampler."""

    task_sampler_class: type | None = (
        None  # Will be set by importing module to avoid circular imports
    )

    task_batch_size: int = 1

    place_target_name: str | None = None  # Placement target will be added to the scene

    # Distance constraints
    max_robot_to_target_dist: float = 0.6
    max_robot_to_obj_dist: float = 0.6

    # House iteration configuration
    house_inds: list[int] | None = list(range(0, 4))  # order of house indices to iterate over
    samples_per_house: int = 2  # number of tasks to sample per house before advancing
    max_tasks: float = math.inf  # total tasks to sample; inf means unbounded

    # Receptacle selection
    receptacle_types: list[str] = tuple(RECEPTACLE_TYPES_THOR)
    # Resolved at runtime
    receptacle_name: str | None = None
    placement_volume_name: str | None = None

    # Object placement parameters (within robot reach)
    object_placement_radius_range: tuple[float, float] = (0.1, 0.8)
    min_object_separation: float = 0.05  # Minimum distance between pickup object and target
    max_object_placement_attempts: int = 200
    max_robot_placement_attempts: int = 10

    # Receptacle retry parameters
    max_receptacle_attempts: int = 10  # Maximum number of different receptacles to try

    # Robot safety radius (placement based on occupancy map)
    robot_safety_radius: float = 0.15  # Radius around robot to avoid collisions
    robot_object_z_offset: float = -0.75
    robot_object_z_offset_random_min: float = (
        -0.30  # Minimum offset to place robot base relative to object height.
    )
    robot_object_z_offset_random_max: float = (
        0.25  # Maximum offset to place robot base relative to object height.
    )
    base_pose_sampling_radius_range: tuple[float, float] = (
        0.0,
        0.7,
    )  # Radius to sample robot base pose around receptacle

    # -- Added pickup objects (pick-from-set mode) --
    # When not None, external objects matching these synsets/categories/UIDs are added to the
    # scene and used as pickup targets instead of the scene's own objects (which serve only as
    # reference positions for placement on cluttered surfaces).
    added_pickup_objects: list[str] | None = None
    # 1-indexed rank(s) from _pickupable_class_ranking(); overrides added_pickup_objects
    # in model_post_init with the UIDs for those ranked classes.
    added_pickup_class_rank: int | list[int] | None = None
    added_pickup_class_max_uids: int | None = None  # cap UIDs per ranked class (None = all)
    added_pickup_namespace: str = "pickup/"
    num_added_pickups: int = 30  # max no. of added pickupables in the scene
    episodes_per_added_pickup: int = 1
    min_reference_to_added_pickup_dist: float = 0.15
    max_reference_to_added_pickup_dist: float = 0.5
    max_added_pickup_placement_attempts: int = 100
    max_robot_to_added_pickup_dist: float = 0.7

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.added_pickup_class_rank is not None:
            from molmo_spaces.utils.synset_utils import _pickupable_class_ranking

            ranking = _pickupable_class_ranking()
            ranks = (
                self.added_pickup_class_rank
                if isinstance(self.added_pickup_class_rank, list)
                else [self.added_pickup_class_rank]
            )
            all_uids: list[str] = []
            for rank in ranks:
                idx = rank - 1
                if idx < 0 or idx >= len(ranking):
                    raise ValueError(
                        f"added_pickup_class_rank={rank} out of range [1, {len(ranking)}]"
                    )
                uids = ranking[idx][1]
                if self.added_pickup_class_max_uids is not None:
                    uids = uids[: self.added_pickup_class_max_uids]
                all_uids.extend(uids)
            self.added_pickup_objects = all_uids
        if self.added_pickup_objects:
            self.objaverse_oversampling_factor = 1


class OpenTaskSamplerConfig(PickTaskSamplerConfig):
    robot_object_z_offset: float = -1.0
    robot_placement_radius_range: tuple[float, float] = (0.30, 0.8)

    # samples_per_house: int = 3
    # pickup_types: list[str] | None = ["drawer"]

    robot_placement_rotation_range_rad: float = 0.25  # +/- approx 15 degrees

    robot_object_z_offset_random_min: float = (
        0  # Minimum offset to place robot base relative to object height.
    )
    robot_object_z_offset_random_max: float = (
        0  # Maximum offset to place robot base relative to object height.
    )
    target_initial_state_open_percentage: float = (
        0  # Percentage of the target joint at start to open or close the joint
    )
    grasp_libraries: list[str] | None = ["droid"]  # only thor provides articulated grasps


class RUMPickTaskSamplerConfig(PickTaskSamplerConfig):
    robot_object_z_offset: float = 0
    robot_object_z_offset_random_min: float = (
        0  # Minimum offset to place robot base relative to object height.
    )
    robot_object_z_offset_random_max: float = (
        0  # Maximum offset to place robot base relative to object height.
    )


class PickAndPlaceTaskSamplerConfig(PickTaskSamplerConfig):
    # When empty or None, uses synset-based filtering for receptacles/all synsets we have judged to be appropriate for "in or on" placement.
    # Otherwise, uses explicit type list (legacy behavior).
    place_receptacle_types: list[str] = []
    place_receptacle_namespace: str = "place_receptacle/"
    max_robot_to_place_receptacle_dist: float = 0.7
    min_object_to_receptacle_dist: float = 0.15
    max_object_to_receptacle_dist: float = 0.5
    max_place_receptacle_sampling_attempts: int = 100
    robot_placement_rotation_range_rad: float = math.radians(45)
    # Number of receptacles to preload in the scene for fallback when IK fails
    num_place_receptacles: int = 3
    # Auto-advance to next receptacle after this many episodes (0 = disabled)
    episodes_per_receptacle: int = 2


class PickAndPlaceNextToTaskSamplerConfig(PickAndPlaceTaskSamplerConfig):
    place_receptacle_types: list[str] = []  # Empty = any object on bench
    min_object_to_receptacle_dist: float = 0.3  # avoid insta-success by keeping this large
    max_object_to_receptacle_dist: float = 0.5  # don't overdo (hard to reach, maybe)
    max_place_receptacle_sampling_attempts: int = 100
    episodes_per_receptacle: int = 0  # we're not using added receptacles!


class PickAndPlaceColorTaskSamplerConfig(PickAndPlaceTaskSamplerConfig):
    """Configuration for pick and place color task sampler."""

    pass


class PackingTaskSamplerConfig(PickAndPlaceTaskSamplerConfig):
    box_uids: list[str] | None = None  # If None, uses Box_1..Box_30


class DoorOpeningTaskSamplerConfig(BaseMujocoTaskSamplerConfig):
    """Configuration for RBY1 door opening task sampler."""

    task_sampler_class: type = None  # Will be set by importing module to avoid circular imports
    sim_settle_timesteps: int = 500
    verbose: bool = False  # Whether to print verbose debug info
    fixed_door_name: str | None = None  # e.g., "door|2|8_Doorway_Double_7_doorway_door_7"

    # Dataset configuration
    dataset_name: str = "procthor-10k"
    random_seed: int | None = None  # Random seed for deterministic task sampling

    # House iteration configuration
    house_inds: list[int] | None = list(
        range(0, 22)
    )  # List of thor house indices to iterate through (first 20 for demo)
    scene_xml_paths: list[str] | None = None
    samples_per_house: int = 1  # Number of tasks per house
    task_batch_size: int = 1
    max_tasks: float = math.inf  # total tasks to sample; inf means unbounded

    robot_placement_rotation_range_rad: float = 0.25  # +/- approx 15 degrees

    # Door opening specific task sampling parameters
    # Door joint randomization
    enable_door_joint_randomization: bool = True  # Whether to randomize door joint parameters
    door_stiffness_range: tuple = (3, 7)  # Range for door joint stiffness (reduced from ~250)
    door_damping_range: tuple = (8, 12)  # Range for door joint damping (reduced from ~100)
    door_frictionloss_range: tuple = (
        8,
        12,
    )  # Range for door joint frictionloss (reduced from ~50)
    handle_stiffness_range: tuple = (
        200,
        300,
    )  # Range for handle joint stiffness (increased from ~0)
    handle_damping_range: tuple = (
        80,
        120,
    )  # Range for handle joint damping (increased from ~0.1)
    handle_frictionloss_range: tuple = (
        40,
        60,
    )  # Range for handle joint frictionloss (increased from ~0)

    # Either choose a random door from the scene
    choose_random_door_from_scene: bool = True
    base_pose_sampling_radius_range: tuple[float, float] = (1.0, 1.5)
    # Radius of the circle around the door handle to sample the robot base pose

    robot_safety_radius: float = (
        0.7  # Radius of the robot base to avoid collisions with the environment
    )
    robot_base_pose_noise: float = 0.1  # Random noise added to robot base pose when sampling task


class NavToObjTaskSamplerConfig(ObjectCentricTaskSamplerConfig):
    """Configuration for navigation to object task sampler.
    Uses pickup_types/pickup_obj_name for compatibility with EvalTaskSampler.
    """

    task_sampler_class: type | None = (
        None  # Will be set by importing module to avoid circular imports
    )
    task_batch_size: int = 1

    house_inds: (
        list[int] | None
    ) = []  # list(range(0, 20))  # List of thor house indices to iterate through (first 20 for demo)
    samples_per_house: int = 1  # Number of tasks per house
    max_tasks: float = math.inf  # total tasks to sample; inf means unbounded

    robot_safety_radius: float = 0.3  # Radius around robot to avoid collisions
    robot_object_z_offset: float = 0.1  # Offset to place robot base relative to object height
    base_pose_sampling_radius_range: tuple[float, float] = (
        1.0,
        10.0,
    )  # Radius to sample robot base pose (min and max)
    face_target: bool = True  # Whether to face the target when placing the robot
    max_robot_placement_attempts: int = 10  # Maximum number of attempts to place the robot

    verbose: bool = False  # Whether to print verbose debug info

    max_valid_candidates: int = 6  # maximum number of instances of type in scene to accept the task
