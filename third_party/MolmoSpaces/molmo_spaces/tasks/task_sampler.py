"""
This module contains the abstract task sampler class.
Each task sampler is responsible for sampling its corresponding task configuration and creating the task.
Eg. DoorOpeningTaskSampler samples a door opening task configuration and creates a DoorOpeningTask.
If task parameters are explicitly provided, they are fixed. Otherwise they are sampled by the task sampler.

Subclasses of AbstractMujocoTaskSampler should implement the _sample_task method.
"""

import logging
import math
import os
import random
from abc import abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from mujoco import MjData, MjSpec

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.arena.arena_utils import get_all_bodies_with_joints_as_mlspaces_objects
from molmo_spaces.env.arena.randomization.dynamics import DynamicsRandomizer
from molmo_spaces.env.arena.randomization.lighting import LightingRandomizer
from molmo_spaces.env.arena.randomization.texture import TextureRandomizer
from molmo_spaces.env.env import BaseMujocoEnv, CPUMujocoEnv

# Dataset helpers for house index mapping
from molmo_spaces.molmo_spaces_constants import (
    ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR,
    DATA_TYPE_TO_SOURCE_TO_VERSION,
    get_scenes,
    get_scenes_root,
)

MJC_VERSION = tuple(map(int, mujoco.__version__.split(".")))
FILAMENT_ATTR_ENV_LIGHT_INTENSITY = "filament.fallback.environment_light_intensity"


# Asset blacklist for data generation - assets that cause consistent failures
# Use environment variable if set (for distributed jobs with read-only code mounts)
# Otherwise fall back to local path in the codebase
# NOTE: This must be a function (not a constant) because the env var may be set after import
def get_asset_blacklist_path() -> Path:
    """Get the asset blacklist path, preferring environment variable.

    Checks MLSPACES_ASSET_BLACKLIST_PATH env var on each call, since the env var
    may be set after module import (e.g., by Beaker job configuration).
    """
    env_path = os.environ.get("MLSPACES_ASSET_BLACKLIST_PATH")
    if env_path:
        return Path(env_path)
    return (
        ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR
        / "molmo_spaces"
        / "data_generation"
        / "asset_blacklist.txt"
    )


# Default max failures before an asset is dynamically blacklisted during a run
DEFAULT_MAX_ASSET_FAILURES = 10


def load_asset_blacklist() -> set[str]:
    """Load the static asset blacklist from file.

    Returns:
        Set of asset UIDs that should be skipped during data generation.

    Raises:
        FileNotFoundError: If an explicit blacklist path was specified via
            MLSPACES_ASSET_BLACKLIST_PATH but does not exist.
    """
    blacklist_path = get_asset_blacklist_path()
    blacklist = set()
    if not blacklist_path.exists():
        # If an explicit path was specified via env var, it must exist
        if os.environ.get("MLSPACES_ASSET_BLACKLIST_PATH"):
            raise FileNotFoundError(
                f"Asset blacklist path specified via MLSPACES_ASSET_BLACKLIST_PATH "
                f"does not exist: {blacklist_path}"
            )
        return blacklist

    with open(blacklist_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Handle inline comments
            uid = line.split("#")[0].strip()
            if uid:
                blacklist.add(uid)

    return blacklist


# Load blacklist once at module import
_STATIC_ASSET_BLACKLIST: set[str] | None = None


def get_static_asset_blacklist() -> set[str]:
    """Get the static asset blacklist, loading it if necessary."""
    global _STATIC_ASSET_BLACKLIST
    if _STATIC_ASSET_BLACKLIST is None:
        _STATIC_ASSET_BLACKLIST = load_asset_blacklist()
        if _STATIC_ASSET_BLACKLIST:
            log.info(f"Loaded {len(_STATIC_ASSET_BLACKLIST)} assets from static blacklist")
    return _STATIC_ASSET_BLACKLIST


def add_to_static_blacklist(asset_uid: str, reason: str = "") -> bool:
    """Add an asset UID to the static blacklist file with file locking.

    This is safe to call from multiple workers concurrently.
    The blacklist path can be configured via MLSPACES_ASSET_BLACKLIST_PATH env var.

    Args:
        asset_uid: The asset UID to add to the blacklist
        reason: Optional reason for blacklisting (added as comment)

    Returns:
        True if the asset was added, False if it was already in the blacklist
    """
    global _STATIC_ASSET_BLACKLIST
    from filelock import FileLock

    if not asset_uid:
        return False

    # Check in-memory cache first (fast path)
    if _STATIC_ASSET_BLACKLIST is not None and asset_uid in _STATIC_ASSET_BLACKLIST:
        return False

    blacklist_path = get_asset_blacklist_path()

    # Ensure parent directory exists
    blacklist_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = blacklist_path.with_suffix(".txt.lock")
    lock = FileLock(lock_path, timeout=30)

    try:
        with lock:
            # Re-check file contents under lock (another worker may have added it)
            current_blacklist = load_asset_blacklist()
            if asset_uid in current_blacklist:
                # Update in-memory cache
                if _STATIC_ASSET_BLACKLIST is not None:
                    _STATIC_ASSET_BLACKLIST.add(asset_uid)
                return False

            # Append to file
            comment = f"  # {reason}" if reason else ""
            with open(blacklist_path, "a") as f:
                f.write(f"{asset_uid}{comment}\n")

            # Update in-memory cache
            if _STATIC_ASSET_BLACKLIST is not None:
                _STATIC_ASSET_BLACKLIST.add(asset_uid)

            log.warning(
                f"Added asset {asset_uid} to static blacklist at {blacklist_path}: {reason}"
            )
            return True

    except Exception as e:
        log.error(f"Failed to add {asset_uid} to static blacklist at {blacklist_path}: {e}")
        return False


def extract_asset_uid_from_object_name(object_name: str) -> str | None:
    """Extract the asset UID hash from an object name.

    Object names follow the pattern: {category}_{md5_hash}_{instance_ids}
    e.g., 'egg_458c762f78242a821082a457af28fe5c_1_0_2'

    Note: This extracts the MD5 hash of the UID, not the original UID.
    For physics errors where we only have the object name, this is sufficient
    for blacklisting since scenes use the same hash.

    Args:
        object_name: The object name from the scene

    Returns:
        The 32-character MD5 hash portion, or None if pattern doesn't match
    """
    import re

    # Match pattern: category_32charhash_numbers
    match = re.match(r"^[a-z]+_([a-f0-9]{32})_", object_name, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


from molmo_spaces.robots.abstract import Robot
from molmo_spaces.tasks.scene_xml_utils import xml_add_rby1_to_scene
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path
from molmo_spaces.utils.mujoco_scene_utils import randomize_door_joints

log = logging.getLogger(__name__)


class MetadataAdder:
    def __init__(self):
        import threading

        self.semaphore = threading.Semaphore()
        self.pending = True
        self.name_to_meta: dict[str, dict[str, Any]] = {}

    def reset(self):
        """Reset for a new scene. Must be called before add_auxiliary_objects populates new entries."""
        self.semaphore.acquire()
        try:
            self.pending = True
            self.name_to_meta.clear()
        finally:
            self.semaphore.release()

    def update(self, name_to_meta):
        self.semaphore.acquire()
        try:
            assert self.pending, (
                "Attempt to call MetadataAdder.update() with additional metadata already flushed"
            )
            self.name_to_meta.update(name_to_meta)
        finally:
            self.semaphore.release()

    def add_meta(self, metadata):
        if self.pending:
            self.semaphore.acquire()
            try:
                if self.pending:
                    for name, meta in self.name_to_meta.items():
                        if name not in metadata["objects"]:
                            metadata["objects"][name] = meta

                    self.pending = False
            finally:
                self.semaphore.release()


class BaseMujocoTaskSampler:
    """
    Base task sampler that provides common scene loading and randomization functionality.

    This class extracts scene loading logic from CPUMujocoEnv and provides:
    - XML preprocessing with dummy cameras
    - Scene model loading with texture randomization
    - Environment initialization with the loaded scene
    - Hook for scene randomization that subclasses can override

    Subclasses should implement:
    - _sample_task(): Create the specific task instance
    - randomize_scene(): Apply runtime scene randomization (optional)
    """

    def __init__(self, exp_config: MlSpacesExpConfig) -> None:
        self.config = exp_config

        # Environment will be created lazily when first accessed
        self._env = None
        self.current_seed = None

        # Shared metadata adder for all dynamically-added objects
        self._metadata_adder = MetadataAdder()

        # Optional profiler for sub-timing within task sampling (set via set_datagen_profiler)
        self._datagen_profiler = None

        # randomizer
        self.lighting_randomizer = None
        self.texture_randomizer = None
        self.dynamics_randomizer = None

        # Dataset/house iteration state (optional; used when dataset_name/house_inds provided)
        self._dataset_index_map: dict | None = None

        self._house_inds = exp_config.task_sampler_config.house_inds
        if self._house_inds is None or self._house_inds == []:
            mapping = self._get_dataset_index_map()
            self._house_inds = sorted(
                [k for k, v in mapping[exp_config.data_split].items() if v is not None]
            )

        self._house_iterator_index = -1
        self._samples_per_current_house = 0
        self._max_tasks = getattr(self.config.task_sampler_config, "max_tasks", math.inf)
        self._current_tasks_left = self._max_tasks
        self._last_loaded_house_index: int | None = None
        self.object_synset_counter = Counter()
        self.used_robot_positions: dict[str, list[np.ndarray]] = defaultdict(list)

        # Asset blacklist tracking - dynamic failures during this run
        self._asset_failure_counts: Counter = Counter()
        self._dynamic_blacklist: set[str] = set()
        self._max_asset_failures = getattr(
            self.config.task_sampler_config, "max_asset_failures", DEFAULT_MAX_ASSET_FAILURES
        )

        # Seed task sampling once at initialization
        # If no seed provided, generate a random one
        seed = self.config.seed if self.config.seed is not None else np.random.randint(0, 100000000)
        self.seed_task_sampling(seed)

        # Log data versions being used
        log.info(f"Data versions in use: {DATA_TYPE_TO_SOURCE_TO_VERSION}")

    @property
    def env(self) -> BaseMujocoEnv:
        # """Get the environment instance, creating it if necessary."""
        # if self._env is None:
        #     # Create environment without any scene loaded
        #     # Scene will be loaded via load_scene() when needed
        #     self._env = self._create_env()
        return self._env

    def close(self) -> None:
        """Clean up the task sampler and its environment."""
        import gc

        if hasattr(self, "_env") and self._env is not None:
            self._env.close()
            self._env = None

        # Explicit cleanup to prevent memory leaks
        gc.collect()

    def set_datagen_profiler(self, profiler) -> None:
        """Set the datagen profiler for sub-timing within task sampling."""
        self._datagen_profiler = profiler

    def is_asset_blacklisted(self, asset_uid: str) -> bool:
        """Check if an asset UID is blacklisted (static or dynamic).

        Checks both the original UID and its MD5 hash, since blacklist entries
        may come from either scene metadata (original UID) or error messages (MD5 hash).

        Args:
            asset_uid: The asset UID to check

        Returns:
            True if the asset is in the static or dynamic blacklist
        """
        import hashlib

        # Check original UID
        if asset_uid in get_static_asset_blacklist():
            return True
        if asset_uid in self._dynamic_blacklist:
            return True

        # Also check MD5 hash of UID (compile errors store hashes, not original UIDs)
        uid_hash = hashlib.md5(asset_uid.encode()).hexdigest()
        if uid_hash in get_static_asset_blacklist():
            return True
        return uid_hash in self._dynamic_blacklist

    def report_asset_failure(self, asset_uid: str, reason: str = "") -> bool:
        """Report a failure for an asset, potentially adding it to the dynamic blacklist.

        Call this when an asset causes a failure (e.g., IK failure, physics error).
        After max_asset_failures, the asset will be dynamically blacklisted for
        the remainder of this task sampler's lifetime.

        Args:
            asset_uid: The asset UID that failed
            reason: Optional reason for the failure (for logging)

        Returns:
            True if the asset was just added to the dynamic blacklist
        """
        import hashlib

        if not asset_uid:
            return False

        # Use MD5 hash for consistency with blacklist (compile errors use hashes)
        uid_hash = hashlib.md5(asset_uid.encode()).hexdigest()

        self._asset_failure_counts[uid_hash] += 1
        count = self._asset_failure_counts[uid_hash]

        if count >= self._max_asset_failures and uid_hash not in self._dynamic_blacklist:
            self._dynamic_blacklist.add(uid_hash)
            log.warning(
                f"Asset {asset_uid} (hash: {uid_hash}) dynamically blacklisted after {count} failures"
                + (f": {reason}" if reason else "")
            )
            return True

        return False

    def get_asset_uid_from_object(self, env, obj_name: str) -> str | None:
        """Extract the asset UID from an object name using scene metadata.

        Args:
            env: The environment instance
            obj_name: The object name in the scene

        Returns:
            The asset UID, or None if not found
        """
        scene_metadata = env.current_scene_metadata
        if scene_metadata is None:
            return None
        return scene_metadata.get("objects", {}).get(obj_name, {}).get("asset_id", None)

    def seed_task_sampling(self, seed) -> None:
        self.current_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _create_robot(self, mj_data: MjData) -> Robot:
        return self.config.robot_config.robot_factory(mj_data, self.config)

    def setup_cameras(self, env: CPUMujocoEnv, deterministic_only: bool = False) -> None:
        """Set up all cameras defined in the camera system config.

        This is the main entry point for camera setup in task samplers.
        All camera configuration should be done via CameraSystemConfig.

        Subclasses can override get_workspace_center() and resolve_visibility_object()
        to provide runtime information for dynamic camera placement.

        Args:
            env: The environment instance to set up cameras in
            deterministic_only: If True, only set up fixed cameras, skip randomized ones
        """
        camera_config = self.config.camera_config
        if camera_config is None:
            log.warning("[CAMERA SETUP] No camera config provided, skipping camera setup")
            return

        # Compute workspace center and visibility resolver for camera placement
        workspace_center = self.get_workspace_center(env)

        def visibility_resolver(key):
            return self.resolve_visibility_object(env, key)

        env.camera_manager.setup_cameras(
            env,
            camera_config,
            workspace_center,
            visibility_resolver,
            deterministic_only=deterministic_only,
        )

    def get_workspace_center(self, env: CPUMujocoEnv) -> np.ndarray:
        """Get the workspace center for camera placement.

        Default implementation uses robot gripper position. Subclasses can override
        to provide task-specific workspace centers (e.g., average of pickup/place/gripper).

        Args:
            env: The environment instance

        Returns:
            3D position of workspace center
        """
        gripper_positions = env.get_robot_gripper_positions(0)
        if gripper_positions:
            positions = list(gripper_positions.values())
            return np.mean(positions, axis=0)
        return np.array([0.0, 0.0, 0.8])  # Default table height

    def resolve_visibility_object(self, env: CPUMujocoEnv, key: str) -> list[str]:
        """Resolve special visibility object keys to actual body names.

        Default implementation handles common keys like __gripper__. Subclasses can
        override to handle task-specific keys.

        Args:
            env: The environment instance
            key: Special key to resolve (e.g., "__gripper__")

        Returns:
            Resolved body names, or empty list if key cannot be resolved
        """
        if key == "__gripper__":
            # Get gripper body name
            robot_view = env.current_robot.robot_view
            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
            gripper_body_id = robot_view.get_move_group(gripper_mg_id).root_body_id
            return [robot_view.mj_model.body(gripper_body_id).name]
        return []

    def get_visibility_resolver(self, env: CPUMujocoEnv):
        """Get a visibility resolver closure for the current environment.

        This is a helper method to avoid duplicating the closure pattern across task samplers.

        Args:
            env: The environment instance

        Returns:
            A callable that resolves visibility keys to body names
        """

        def visibility_resolver(key):
            return self.resolve_visibility_object(env, key)

        return visibility_resolver

    @abstractmethod
    def _sample_task(self, env: BaseMujocoEnv) -> BaseMujocoTask:
        raise NotImplementedError

    def reset(self) -> None:
        self._house_iterator_index = -1  # -1 so that it doesn't skip the first task
        self._samples_per_current_house = 0
        self._current_tasks_left = self._max_tasks
        self.object_synset_counter = Counter()
        self.used_robot_positions.clear()

        # Re-seed if a seed was originally provided
        if self.config.seed is not None:
            self.seed_task_sampling(self.config.seed)

    def setup_robot_scene(
        self,
        robot_config,
        scene_file_path,
        randomize_textures=False,
        environment_light_intensity: float = 15000.0,
    ):
        """
        Complete function to set up a scene with robot and objects.

        Args:
            scene_file_path: Path to the base scene XML file (should be base variant)
            robot_file_path: Path to the robot XML file

        Returns:
            Compiled MuJoCo model
        """
        # Track XML/spec loading time
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("compile_xml_load")

        robot_file_path = robot_config.get_robot_xml_path()
        use_include = robot_config.name == "rby1" or robot_config.name == "rby1m"

        if use_include:
            # for whatever reason, the rby1 specifically doesn't play nice with MjSpec insertion,
            # so we directly insert the rby1 xml into the scene. For all other robots,
            # we use spec editing.
            spec = xml_add_rby1_to_scene(
                self.config.task_sampler_config, scene_file_path, robot_file_path
            )
        else:
            spec = MjSpec.from_file(str(scene_file_path))

        # Hack(wilbert): only run on 3.5.1 onwards for now (custom mujoco wheel)
        if MJC_VERSION >= (3, 5, 1):
            # This custom arg is used by the filament renderer to set the default intensity of the environment light
            spec.add_numeric(
                FILAMENT_ATTR_ENV_LIGHT_INTENSITY,
                [environment_light_intensity],
                1,
                "Filament default env light intensity",
            )

        if self._datagen_profiler is not None:
            self._datagen_profiler.end("compile_xml_load")

        # Check if door joint randomization is enabled (for RBY1 door opening tasks)
        enable_door_joint_randomization = (
            hasattr(self.config.task_sampler_config, "enable_door_joint_randomization")
            and self.config.task_sampler_config.enable_door_joint_randomization
        )
        if enable_door_joint_randomization:
            # Load scene metadata directly from scene path (env might not exist yet)
            from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

            scene_metadata = get_scene_metadata(scene_file_path)
            if scene_metadata is None:
                log.warning(
                    f"Could not load scene metadata from {scene_file_path}, skipping door joint randomization"
                )
            else:
                randomize_door_joints(spec, scene_metadata)

        # Track robot addition time
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("compile_robot_add")

        if not use_include:
            # Add the robot using a default position
            self.config.robot_config.robot_cls.add_robot_to_scene(
                self.config.robot_config,
                spec,
                prefix="robot_0/",
                pos=[0.0, 0.0],
                quat=[1.0, 0.0, 0.0, 0.0],
                randomize_textures=self.config.task_sampler_config.randomize_robot_textures,
            )

        # apply robot control overrides
        self.config.robot_config.robot_cls.apply_control_overrides(spec, self.config.robot_config)

        if self._datagen_profiler is not None:
            self._datagen_profiler.end("compile_robot_add")

        # Track auxiliary object loading time (task-specific assets)
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("compile_aux_objects")

        # Load task specific assets into the scene, use this to add pickup, placement obj
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("compile_aux_policy_objects")
        self.add_auxiliary_objects(spec)
        if self._datagen_profiler is not None:
            self._datagen_profiler.end("compile_aux_policy_objects")

        # Setup empty materials for texture randomization if enabled
        if randomize_textures:
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("compile_aux_empty_materials")
            self.setup_empty_materials(spec)
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("compile_aux_empty_materials")

        if self._datagen_profiler is not None:
            self._datagen_profiler.end("compile_aux_objects")

        # Track MuJoCo compilation time (spec.compile())
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("compile_mujoco")

        # Delete blacklisted bodies before compilation to prevent mass/inertia errors
        from molmo_spaces.utils.scene_maps import _delete_blacklisted_bodies

        _delete_blacklisted_bodies(spec)

        # Compile and return the model
        try:
            model = spec.compile()
        except ValueError as e:
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("compile_mujoco")

            # Try to blacklist problematic assets from mass/inertia errors
            error_str = str(e)
            if "mass and inertia" in error_str.lower():
                import re

                match = re.search(r"Element name '([^']+)'", error_str)
                if match:
                    element_name = match.group(1)
                    hash_match = re.search(r"[a-f0-9]{32}", element_name, re.IGNORECASE)
                    if hash_match:
                        add_to_static_blacklist(
                            hash_match.group(0), f"mass/inertia error from {element_name}"
                        )

            if "Nan, Inf or huge value" in error_str or "simulation is unstable" in error_str:
                # MuJoCo compilation failed due to unstable physics
                raise HouseInvalidForTask(
                    reason="MuJoCo compilation failed due to unstable physics",
                    house_info=f"Compilation error: {error_str}",
                ) from e
            elif "mass and inertia" in error_str.lower():
                # Asset has invalid mass/inertia - treat as house invalid
                raise HouseInvalidForTask(
                    reason="Asset has invalid mass/inertia (now blacklisted)",
                    house_info=f"Compilation error: {error_str}",
                ) from e
            else:
                # Re-raise other ValueError exceptions
                raise

        if self._datagen_profiler is not None:
            self._datagen_profiler.end("compile_mujoco")

        return model

    def add_auxiliary_objects(self, spec: MjSpec | None) -> None:
        """Add add auxiliary objects to  a scene or make task specific model changes
        This gives access to the MjSpec pre-compilation."""
        pass

    def setup_empty_materials(
        self, spec: mujoco.MjSpec | None = None, num_materials: int = 200
    ) -> None:
        """
        Create a pool of empty materials and textures in the MjSpec that can be assigned to geoms at runtime.
        This allows texture randomization to modify materials and textures without affecting other geoms.

        Args:
            spec: MjSpec to modify
            num_materials: Maximum number of empty materials/textures to create. Actual number is based on
                          visual geom count with a safety buffer.
        """
        from molmo_spaces.env.arena.randomization.texture import (
            setup_empty_materials,
        )

        return setup_empty_materials(spec, num_materials)

    def update_scene(self, scene_path: str | None = None) -> None:
        """Update the environment's scene by loading a new scene model.

        Args:
            scene_path: Path to scene file. If None, uses _current_house_scene_path()
            variant: The scene variant to use when scene_path is None (ceiling", "map", "base", etc.)
        """
        if scene_path is None:
            scene_path = self._current_house_scene_path()

        # If using a MolmoSpaces scene, install it
        if get_scenes_root().resolve() in Path(scene_path).resolve().parents:
            # Track asset installation time (fetching/extracting scene, objects, grasps)
            # Use detailed profiling to identify which asset type is slow
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("scene_asset_install")
                from molmo_spaces.utils.lazy_loading_utils import (
                    install_grasps_for_scene,
                    install_objects_for_scene,
                    install_scene_from_path,
                )

                self._datagen_profiler.start("asset_install_scene")
                install_scene_from_path(scene_path)
                self._datagen_profiler.end("asset_install_scene")

                self._datagen_profiler.start("asset_install_objects")
                install_objects_for_scene(scene_path, exclude_thor=True)
                self._datagen_profiler.end("asset_install_objects")

                self._datagen_profiler.start("asset_install_grasps")
                for grasp_source in ("droid_objaverse",):
                    install_grasps_for_scene(
                        scene_path, grasp_source=grasp_source, exclude_thor=True
                    )
                self._datagen_profiler.end("asset_install_grasps")

                self._datagen_profiler.end("scene_asset_install")
            else:
                install_scene_with_objects_and_grasps_from_path(scene_path)

        # Track scene compilation time (XML processing + MuJoCo spec.compile())
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("scene_compile")
        try:
            enable_door_randomization = getattr(
                self.config.task_sampler_config, "enable_door_joint_randomization", False
            )

            model = self.setup_robot_scene(
                robot_config=self.config.robot_config,
                scene_file_path=scene_path,
                randomize_textures=self.config.task_sampler_config.randomize_textures
                or enable_door_randomization,  # Enable door joint randomization
                environment_light_intensity=self.config.environment_light_intensity,
            )
        except HouseInvalidForTask:
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("scene_compile")
            raise  # Re-raise HouseInvalidForTask exceptions
        except Exception as e:
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("scene_compile")
            # Catch any other compilation errors and convert to HouseInvalidForTask
            raise HouseInvalidForTask(
                reason="Scene setup failed during compilation",
                house_info=f"Setup error: {str(e)}",
                error=e,
            ) from e
        if self._datagen_profiler is not None:
            self._datagen_profiler.end("scene_compile")

        # Create new environment around new model
        if self._env is not None:
            self._env.close()

        # Track environment creation time
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("scene_env_create")
        self._env = CPUMujocoEnv(
            self.config,
            robot_factory=self._create_robot,
            mj_model=model,
            mj_base_scene_path=scene_path,
        )
        if self._datagen_profiler is not None:
            self._datagen_profiler.end("scene_env_create")

        self.used_robot_positions.clear()

        log.info(f"Scene updated: {scene_path}")

        # Track scene initialization time (randomizer setup, etc.)
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("scene_init")
        self.init_scene(self._env)
        if self._datagen_profiler is not None:
            self._datagen_profiler.end("scene_init")

    def init_scene(self, env: BaseMujocoEnv) -> None:
        """
        Initialize a new scene after it is loaded, this is called once after a scene is loaded, not for each task that is generated.
        e.g. set/randomize joint friction values.
        """
        # TODO(max): currently no-oped everywhere - should fill this out or delete it
        # also the function names setup_scene, update_scene, and init_scene are confusing

        # Create seeded random states for randomizers
        # Use the randomization seed if available, otherwise generate a new one
        # Each randomizer gets its own RandomState to ensure independent randomness
        if (
            self.config.task_sampler_config.randomize_lighting
            or self.config.task_sampler_config.randomize_textures
            or self.config.task_sampler_config.randomize_dynamics
        ):
            base_seed = (
                self.current_seed + 1
                if self.current_seed is not None
                else np.random.randint(0, 100000000)
            )

        # Create separate RandomState instances for each randomizer
        # This ensures they have independent random streams even if using the same base seed

        if self.config.task_sampler_config.randomize_lighting:
            lighting_seed = base_seed
            lighting_random_state = (
                np.random.RandomState(lighting_seed) if lighting_seed is not None else None
            )

            self.lighting_randomizer = LightingRandomizer(
                model=env.mj_model,
                random_state=lighting_random_state,
                randomize_position=True,
                randomize_direction=True,
                randomize_specular=True,
                randomize_ambient=True,
                randomize_diffuse=True,
                randomize_active=True,
                position_perturbation_size=0.5,
                direction_perturbation_size=1.0,
                specular_perturbation_size=0.1,
                ambient_perturbation_size=0.1,
                diffuse_perturbation_size=0.1,
            )
        if self.config.task_sampler_config.randomize_textures:
            texture_seed = base_seed + 1 if base_seed is not None else None
            texture_random_state = (
                np.random.RandomState(texture_seed) if texture_seed is not None else None
            )

            self.texture_randomizer = TextureRandomizer(
                model=env.mj_model,
                random_state=texture_random_state,
                randomize_geom_rgba=True,
                randomize_material_rgba=True,
                randomize_material_specular=True,
                randomize_material_shininess=True,
                randomize_texture=True,
                texture_paths=None,  # None = use model textures (default)
                scene_metadata=env.current_scene_metadata,
                rgba_perturbation_size=0.2,
            )
        if self.config.task_sampler_config.randomize_dynamics:
            dynamics_seed = base_seed + 2 if base_seed is not None else None
            dynamics_random_state = (
                np.random.RandomState(dynamics_seed) if dynamics_seed is not None else None
            )
            self.dynamics_randomizer = DynamicsRandomizer(
                random_state=dynamics_random_state,
                randomize_friction=True,
                randomize_mass=True,
                randomize_inertia=True,
                mass_perturbation_ratio=0.2,
                friction_perturbation_ratio=0.2,
                inertia_perturbation_ratio=0.2,
            )

    def _get_dataset_index_map(self) -> dict:
        if self._dataset_index_map is not None:
            return self._dataset_index_map
        name = self.config.scene_dataset

        if name != "user":
            mapping = get_scenes(name, self.config.data_split)
            assert isinstance(mapping, dict)
        else:
            mapping = {
                self.config.data_split: {
                    i: {"base": scene_xml_path}
                    for i, scene_xml_path in enumerate(
                        self.config.task_sampler_config.scene_xml_paths
                    )
                }
            }

        self._dataset_index_map = mapping

        return mapping

    @property
    def current_house_index(self) -> int:
        if self._house_iterator_index < 0:
            return self._house_inds[0]
        return self._house_inds[self._house_iterator_index]

    def _increment_task_and_reset_house(
        self, force_advance_scene: bool, house_index: int | None
    ) -> None:
        if house_index is not None:
            # External override for house selection (e.g., from SQS queue in distributed datagen)
            # Add house to list if not present - this happens when workers receive houses
            # that weren't in the config's default house_inds
            if house_index not in self._house_inds:
                self._house_inds.append(house_index)
            self._house_iterator_index = self._house_inds.index(house_index)
            self._samples_per_current_house = 1
        else:
            if (not force_advance_scene) and (
                self._samples_per_current_house
                < getattr(self.config.task_sampler_config, "samples_per_house", 1)
            ):
                self._samples_per_current_house += 1
            else:
                self._house_iterator_index = (self._house_iterator_index + 1) % len(
                    self._house_inds
                )
                self._samples_per_current_house = 1

    def _current_house_scene_path(self) -> str | None:
        """Get the scene path for the current house index and specified variant.

        Args:
            variant: The scene variant to use ("ceiling", "map", "base", etc.)
                    Defaults to "base" as the foundation scene.

        Returns:
            Path to the scene file for the specified variant, or None if not found.
        """
        mapping = self._get_dataset_index_map()
        split_map = mapping[self.config.data_split]
        idx = self.current_house_index
        house_variants = split_map.get(idx, None)
        variant = self.config.task_sampler_config.house_variant

        if house_variants is None:
            raise RuntimeError(f"No scene file for split '{self.config.data_split}' index {idx}")

        # Handle both old and new format for backward compatibility
        if isinstance(house_variants, (Path | str)):
            # Old format: split_map[idx] is directly a path
            return str(house_variants)
        elif isinstance(house_variants, dict):
            # New format: split_map[idx] is a dict of {variant: path}
            scene_path = house_variants.get(variant, None)
            if scene_path is None:
                available_variants = [v for v, p in house_variants.items() if p is not None]
                raise RuntimeError(
                    f"No scene file for variant '{variant}' at split '{self.config.data_split}' index {idx}. "
                    f"Available variants: {available_variants}"
                )
            return scene_path
        else:
            raise RuntimeError(f"Invalid house variants format: {type(house_variants)}")

    @abstractmethod
    def randomize_scene(self, env: BaseMujocoEnv, robot_view) -> None:
        """
        Randomize a scene, this is called each time a task is generated, not just after loading scene.

        Args:
            robot_view: The robot view for accessing robot state
            env: The environment instance with loaded scene
        """
        # Base implementation does nothing - subclasses should override
        if (
            self.lighting_randomizer is not None
            and self.config.task_sampler_config.randomize_lighting
        ):
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("randomize_lighting")
            self.lighting_randomizer.randomize(env.mj_datas[self.env.current_batch_index])
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("randomize_lighting")
            log.info("Lighting randomization completed.\n")

        if self.texture_randomizer is not None and random.random() >= 0.02:
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("randomize_texture")
            if self.config.task_sampler_config.randomize_textures_all:
                self.texture_randomizer.randomize(env.mj_datas[self.env.current_batch_index])
            elif self.config.task_sampler_config.randomize_textures:
                self.texture_randomizer.randomize_by_category(
                    env.mj_datas[self.env.current_batch_index]
                )

            # Mark textures as dirty so they will be uploaded to GPU on next render
            if hasattr(env, "_renderer") and env._renderer is not None:
                if hasattr(env._renderer, "mark_textures_dirty"):
                    env._renderer.mark_textures_dirty()
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("randomize_texture")
            log.info("Texture randomization completed.\n")

        if (
            self.dynamics_randomizer is not None
            and self.config.task_sampler_config.randomize_dynamics
        ):
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("randomize_dynamics")
            objects_to_randomize = get_all_bodies_with_joints_as_mlspaces_objects(
                env.mj_model, env.mj_datas[self.env.current_batch_index]
            )
            self.dynamics_randomizer.randomize_objects(objects_to_randomize)
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("randomize_dynamics")
            log.info("Dynamics randomization completed.\n")

    def sample_task(
        self,
        force_advance_scene=False,
        house_index=None,
    ) -> None | BaseMujocoTask:
        """Returns a task with batch size task_batch_size.

        Args:
            force_advance_scene: Whether to force advancing to the next scene
            house_index: Specific house index to use, overriding iteration
            variant: Scene variant to use ("ceiling", "map", "base", etc.). Defaults to "ceiling".
        """
        # save the task_config at the beginning of the experiment
        if self.config.task_config_preset_exp is None:
            self.config.task_config_preset_exp = self.config.task_config.model_copy(deep=True)

        # Don't modify self.config.task_config from here on.

        # Stopping condition for dataset workflows
        if not math.isinf(self._current_tasks_left):
            if self._current_tasks_left <= 0:
                return None  # type: ignore[return-value]

        self._increment_task_and_reset_house(
            force_advance_scene=force_advance_scene, house_index=house_index
        )
        assert house_index is None or self.current_house_index == house_index

        scene_path = self._current_house_scene_path()

        need_load = (
            self._env is None
            or self._last_loaded_house_index is None
            or self._last_loaded_house_index != self.current_house_index
        )

        if need_load:
            # We are changing the scene. Start with a clean config containing None values
            # for which we will sample, otherwise valid values stay un-changed
            self.config.task_config = self.config.task_config_preset_exp.model_copy(deep=True)

            # Reset metadata adder so add_auxiliary_objects can register new objects.
            # On scene reuse the old metadata is already in current_scene_metadata.
            self._metadata_adder.reset()

            log.debug(
                f"[HOUSE] Loading scene for house index {self.current_house_index} (prev loaded: {self._last_loaded_house_index})"
            )
            # Time scene loading (expensive - includes asset loading and MuJoCo compilation)
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("scene_load")
            try:
                self.update_scene(scene_path=scene_path)
            finally:
                if self._datagen_profiler is not None:
                    self._datagen_profiler.end("scene_load")
            # Cache this config so that we can initalize with it, in cases where we re-use this scene.
            self.config.task_config_preset_scn = self.config.task_config.model_copy(deep=True)
            self._last_loaded_house_index = self.current_house_index
        else:
            # Re-use scene, initialize with our saved task config.
            self.config.task_config = self.config.task_config_preset_scn.model_copy(deep=True)
            log.info(f"[HOUSE] Reusing loaded scene for house index {self.current_house_index}")
            # Record zero time for scene_load when reusing (for tracking reuse vs load ratio)
            if self._datagen_profiler is not None:
                self._datagen_profiler.record("scene_reuse", 0.0)

        # Now you can modify self.config.task_config again.

        # Call the actual task sampling implementation
        # Apply scene randomization - subclasses can override this
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("scene_randomize")
        try:
            if self.env.robots:
                robot = self.env.robots[0]
                robot_view = robot.robot_view
                self.randomize_scene(self.env, robot_view)
        finally:
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("scene_randomize")

        # Forward to MuJoCo to ensure scene state is consistent
        if self.env.mj_datas:
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("mj_forward_sync")
            data = self.env.mj_datas[0]
            model = self.env.mj_model
            mujoco.mj_forward(model, data)
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("mj_forward_sync")

        # Budget decrement if bounded
        if not math.isinf(self._current_tasks_left):
            self._current_tasks_left -= 1

        # Flush accumulated metadata from add_auxiliary_objects into scene metadata
        self._metadata_adder.add_meta(self.env.current_scene_metadata)

        # Time task-specific sampling (object selection, robot placement, camera setup, etc.)
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("task_specific_sample")
        try:
            task = self._sample_task(self.env)
        finally:
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("task_specific_sample")

        # Update robot-mounted camera poses to ensure they reflect the final robot state.
        # This is needed because camera setup may happen before the final mj_forward call,
        # and env.step() (which normally updates cameras) hasn't been called yet.
        self.env.camera_manager.registry.update_all_cameras(self.env)
        log.info(f"Sampled task '{task.get_task_description()}'")

        return task

    def balance_sample_names(self, candidate_objects):
        # Do oversampling of objects if they are objaverse assets
        if not hasattr(self.config.task_sampler_config, "objaverse_oversampling_factor"):
            return candidate_objects
        objaverse_factor = self.config.task_sampler_config.objaverse_oversampling_factor
        if objaverse_factor > 1:
            result = []
            for obj in candidate_objects:
                if obj.name.startswith("obja"):
                    result.extend([obj] * objaverse_factor)
                else:
                    result.append(obj)
            return result
        else:
            return candidate_objects
