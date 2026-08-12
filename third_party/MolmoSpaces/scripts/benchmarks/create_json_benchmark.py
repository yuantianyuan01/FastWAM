"""
JSON benchmark creation script for MuJoCo-THOR datasets.

Creates a single JSON benchmark file from H5 trajectory data. The benchmark is
a list of EpisodeSpec dicts, each fully self-contained, using the schema
defined in mujoco_thor/evaluation/benchmark_schema.py.

WHAT IT DOES:
- Scans all house_* directories under the provided base_path
- Extracts frozen_config from each trajectory's obs_scene
- Converts frozen_config to EpisodeSpec JSON format
- Samples episodes balanced across object categories and houses
- Outputs a single benchmark.json file (list of episode dicts)

EXPECTED INPUT STRUCTURE:
    /path/to/dataset/
        ConfigName/
            train/
                house_0/
                    trajectories_batch_0_of_1.h5
                    ...
                house_1/
                    ...
            val/
                ...

OUTPUT STRUCTURE:
    /path/to/benchmark/
        benchmark.json           (list of EpisodeSpec dicts)
        benchmark_metadata.json  (optional summary)

Each episode in benchmark.json includes:
- source: {h5_file, traj_key} - full path to source video/trajectory
- house_index, scene_dataset, data_split
- robot, cameras, task, language specs
- timing parameters

USAGE:
    python scripts/datagen/create_json_benchmark.py \\
        --base_path /path/to/dataset \\
        --output_path /path/to/benchmark \\
        --num_episodes 100
"""

import argparse
import base64
import datetime
import json
import logging
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
import importlib
import io

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from molmo_spaces.evaluation.benchmark_schema import (
    BenchmarkMetadata,
    EpisodeSpec,
    ExocentricCameraSpec,
    LanguageSpec,
    RobotMountedCameraSpec,
    RobotSpec,
    SceneModificationsSpec,
    SourceSpec,
)
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.utils.object_metadata import ObjectMeta

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Suppress noisy debug/info logs from dependencies
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("curobo").setLevel(logging.WARNING)

RANDOM_STATE = 42

# THOR object category simplifications for diversity analysis
THOR_CAT_SIMPLIFY = {
    "saltshaker": "s/p shaker",
    "peppershaker": "s/p shaker",
    "tomato": "fruit",
    "apple": "fruit",
    "butterknife": "knife",
    "boiler": "kettle",
    "winebottle": "bottle",
    "atomizer": "spray bottle",
    "remotecontrol": "remote control",
    "soapdispenser": "soap dispenser",
    "tissuepaper": "tissue paper",
}


def parse_obs_scene(obs_scene_data) -> dict:
    """Parse obs_scene from HDF5 dataset."""
    if isinstance(obs_scene_data, bytes):
        obs_scene_str = obs_scene_data.decode("utf-8")
    elif isinstance(obs_scene_data, str):
        obs_scene_str = obs_scene_data
    else:
        raise TypeError(f"obs_scene_data must be bytes or str, got {type(obs_scene_data)}")

    return json.loads(obs_scene_str)


class ConfigUnpickler(pickle.Unpickler):
    moved_name_to_module = {}

    def find_class(self, module, name):
        if module.startswith("mujoco_thor."):
            module = module.replace("mujoco_thor.", "molmo_spaces.", 1)

        if name in self.moved_name_to_module:
            module = self.moved_name_to_module[name]

        try:
            mod = importlib.import_module(module)
            cls = mod
            for part in name.split("."):
                cls = getattr(cls, part)
            return cls
        except (ImportError, AttributeError, TypeError):
            return super().find_class(module, name)


def extract_frozen_config(obs_scene: dict, exp_config=None):
    """
    Extract and decode frozen_config from obs_scene.

    Handles both:
    - New JSON format (frozen_config is JSON string)
    - Old pickle format (frozen_config is base64-encoded pickle)

    Returns the decoded SavedEpisode object or dict.
    Raises KeyError if frozen_config is missing.
    Raises ValueError if frozen_config cannot be decoded.
    """
    if "frozen_config" not in obs_scene:
        raise KeyError("frozen_config not found in obs_scene")

    frozen_config_raw = obs_scene["frozen_config"]

    # Try JSON first (new format)
    try:
        return json.loads(frozen_config_raw)
    except json.JSONDecodeError:
        pass

    # Try pickle (old format)
    try:
        loaded_bytes = base64.b64decode(frozen_config_raw)
        return ConfigUnpickler(io.BytesIO(loaded_bytes)).load()
    except Exception as e:
        raise ValueError(f"Failed to decode frozen_config as JSON or pickle: {e}") from e


def camera_config_to_spec(camera_config):
    """
    Convert a camera config object/dict to CameraSpec (RobotMounted or Exocentric).

    Handles both pydantic objects and dicts.
    Raises ValueError if required fields are missing.
    """
    if hasattr(camera_config, "model_dump"):
        cam_dict = camera_config.model_dump()
    elif isinstance(camera_config, dict):
        cam_dict = camera_config
    else:
        # Extract attributes directly from object
        cam_dict = {}
        for attr in [
            "name",
            "fov",
            "pos",
            "up",
            "forward",
            "reference_body_names",
            "camera_offset",
            "lookat_offset",
            "camera_quaternion",
            "record_depth",
        ]:
            if hasattr(camera_config, attr):
                val = getattr(camera_config, attr)
                if val is not None:
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    cam_dict[attr] = val

    if "name" not in cam_dict:
        raise ValueError("Camera config missing required 'name' field")

    # Determine camera type based on fields present
    # Robot-mounted cameras have reference_body_names
    # Exocentric cameras have pos, up, forward
    if "reference_body_names" in cam_dict and cam_dict["reference_body_names"]:
        for field in ["camera_offset", "lookat_offset", "camera_quaternion", "fov"]:
            if field not in cam_dict:
                raise ValueError(
                    f"Robot-mounted camera '{cam_dict['name']}' missing required field: {field}"
                )
        return RobotMountedCameraSpec(
            name=cam_dict["name"],
            reference_body_names=cam_dict["reference_body_names"],
            camera_offset=cam_dict["camera_offset"],
            lookat_offset=cam_dict["lookat_offset"],
            camera_quaternion=cam_dict["camera_quaternion"],
            fov=cam_dict["fov"],
            record_depth=cam_dict.get("record_depth", False),
        )
    elif "pos" in cam_dict:
        for field in ["up", "forward", "fov"]:
            if field not in cam_dict:
                raise ValueError(
                    f"Exocentric camera '{cam_dict['name']}' missing required field: {field}"
                )
        return ExocentricCameraSpec(
            name=cam_dict["name"],
            pos=cam_dict["pos"],
            up=cam_dict["up"],
            forward=cam_dict["forward"],
            fov=cam_dict["fov"],
            record_depth=cam_dict.get("record_depth", False),
        )
    else:
        raise ValueError(
            f"Camera '{cam_dict['name']}' has neither reference_body_names nor pos - cannot determine type"
        )


def get_nested_attr(obj, path, default=None):
    """Get nested attribute from object or dict using dot notation."""
    parts = path.split(".")
    current = obj
    for part in parts:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
        elif hasattr(current, part):
            current = getattr(current, part, default)
        else:
            return default
    return current


def frozen_config_to_episode_spec(
    frozen_config,
    obs_scene: dict,
    house_id: int,
    scene_dataset: str,
    data_split: str,
    source_h5_file: str,
    source_traj_key: str,
    source_episode_length: int,
    img_resolution: tuple[int, int],
    camera_system_class: str | None = None,
    source_data_date: str | None = None,
    benchmark_created_date: str | None = None,
    task_horizon_sec: int = 30,
) -> EpisodeSpec:
    """
    Convert a frozen_config (SavedEpisode) to EpisodeSpec format.

    Args:
        frozen_config: Decoded frozen_config (pydantic object or dict)
        obs_scene: The full obs_scene dict (contains task_description, text, etc.)
        house_id: House index (from directory name, e.g., house_5 -> 5)
        scene_dataset: Scene dataset name (e.g., "procthor-objaverse")
        data_split: Data split (e.g., "train", "val")
        source_h5_file: Path to source H5 file
        source_traj_key: Trajectory key within H5 file
        source_episode_length: Number of steps in original episode
        img_resolution: Image resolution as (width, height) tuple
        camera_system_class: Name of CameraSystemConfig class used
        source_data_date: Approximate date source H5 was created (YYYY-MM-DD)
        benchmark_created_date: Date this benchmark is being created (YYYY-MM-DD)

    Returns:
        EpisodeSpec object

    Raises:
        ValueError: If required fields are missing from frozen_config or obs_scene

    Note:
        Timing parameters (policy_dt_ms, ctrl_dt_ms, sim_dt_ms) are NOT stored
        per-episode. They should come from the evaluation config.
    """
    if frozen_config is None:
        raise ValueError("frozen_config cannot be None")

    # Helper to get values from either pydantic object or dict
    def get_val(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Extract robot config
    robot_config = get_val(frozen_config, "robot_config")
    if robot_config is None:
        raise ValueError("No robot_config in frozen_config")

    # Note: The field is 'name' in BaseRobotConfig, not 'robot_name'
    robot_name = get_val(robot_config, "name")
    if robot_name is None:
        raise ValueError("robot_config missing required 'name' field")

    # Extract init_qpos - must be a dict mapping move group names to joint positions
    init_qpos_raw = get_val(robot_config, "init_qpos")
    if not isinstance(init_qpos_raw, dict):
        raise ValueError(
            f"init_qpos must be a dict, got {type(init_qpos_raw)}. "
            "This may indicate a corrupted or incompatible frozen_config."
        )
    init_qpos = {k: list(v) if hasattr(v, "__iter__") else v for k, v in init_qpos_raw.items()}

    robot_spec = RobotSpec(robot_name=robot_name, init_qpos=init_qpos)

    # Extract camera configs
    camera_config = get_val(frozen_config, "camera_config")
    cameras_raw = get_val(camera_config, "cameras", [])
    camera_specs = []
    for cam in cameras_raw:
        spec = camera_config_to_spec(cam)
        if spec is not None:
            camera_specs.append(spec)

    # Extract task config
    task_config = get_val(frozen_config, "task_config")
    if task_config is None:
        raise ValueError("No task_config in frozen_config")

    # Use task_cls directly - don't infer task_type, let the eval task sampler handle it
    task_cls_str = get_val(frozen_config, "task_cls_str")
    if not task_cls_str:
        raise ValueError("No task_cls_str in frozen_config")

    # Extract robot_base_pose (this is the authoritative robot world placement)
    robot_base_pose = get_val(task_config, "robot_base_pose")
    if robot_base_pose is None:
        raise ValueError("No robot_base_pose in task_config")
    if hasattr(robot_base_pose, "tolist"):
        robot_base_pose = robot_base_pose.tolist()

    # Build task dict with task_cls as the authoritative task identifier
    # The eval task sampler is responsible for interpreting task_cls
    task_dict = {
        "task_cls": task_cls_str,
        "robot_base_pose": robot_base_pose,
        "task_horizon_sec": task_horizon_sec,
    }

    # Extract all task config fields directly - no defaults, no type-specific filtering
    # The eval task sampler will use what it needs based on task_cls
    def add_if_present(key: str, required: bool = False):
        """Add field to task_dict if present in task_config."""
        val = get_val(task_config, key)
        if val is None:
            if required:
                raise ValueError(f"task_config missing required field: {key}")
            return
        if hasattr(val, "tolist"):
            val = val.tolist()
        task_dict[key] = val

    # Common task fields
    add_if_present("pickup_obj_name")
    add_if_present("pickup_obj_start_pose")
    add_if_present("pickup_obj_goal_pose")
    add_if_present("receptacle_name")
    add_if_present("succ_pos_threshold")

    # Pick and place fields
    add_if_present("place_receptacle_name")
    add_if_present("place_receptacle_start_pose")
    add_if_present("receptacle_supported_weight_frac")
    add_if_present("max_place_receptacle_pos_displacement")
    add_if_present("max_place_receptacle_rot_displacement")

    # Open/close fields
    add_if_present("articulation_object_name")
    add_if_present("joint_name")
    add_if_present("joint_index")
    add_if_present("joint_start_position")
    add_if_present("joint_goal_position")
    add_if_present("task_success_threshold")
    add_if_present("any_inst_of_category")

    # Nav fields
    add_if_present("pickup_obj_candidates")

    # Extract added_objects and object_poses for scene modifications
    added_objects_raw = get_val(task_config, "added_objects", {})
    added_objects = {}
    for name, path in added_objects_raw.items():
        # Convert to relative path if absolute
        path_str = str(path)
        assets_dir_str = str(ASSETS_DIR)
        if path_str.startswith(assets_dir_str):
            path_str = path_str[len(assets_dir_str) :].lstrip("/")
        added_objects[name] = path_str

    object_poses_raw = get_val(task_config, "object_poses", {})
    object_poses = {}
    if object_poses_raw:
        for name, pose in object_poses_raw.items():
            if hasattr(pose, "tolist"):
                pose = pose.tolist()
            object_poses[name] = pose

    scene_modifications = SceneModificationsSpec(
        added_objects=added_objects,
        object_poses=object_poses,
    )

    # Extract language info - require task_description
    task_description = obs_scene.get("task_description")
    if task_description is None:
        raise ValueError("obs_scene missing required 'task_description' field")

    referral_expressions = get_val(task_config, "referral_expressions", {})
    referral_expressions_priority = get_val(task_config, "referral_expressions_priority", {})

    language_spec = LanguageSpec(
        task_description=task_description,
        referral_expressions=referral_expressions,
        referral_expressions_priority=referral_expressions_priority,
    )

    # Create source provenance
    source_spec = SourceSpec(
        h5_file=source_h5_file,
        traj_key=source_traj_key,
        episode_length=source_episode_length,
        camera_system_class=camera_system_class,
        source_data_date=source_data_date,
        benchmark_created_date=benchmark_created_date,
    )

    # Create EpisodeSpec
    # Note: Timing parameters (policy_dt_ms, ctrl_dt_ms, sim_dt_ms) and task_horizon
    # are NOT stored per-episode. They should come from the evaluation config.
    episode_spec = EpisodeSpec(
        source=source_spec,
        house_index=house_id,
        scene_dataset=scene_dataset,
        data_split=data_split,
        seed=None,  # Could extract if available
        robot=robot_spec,
        img_resolution=img_resolution,
        cameras=camera_specs,
        scene_modifications=scene_modifications,
        task=task_dict,
        language=language_spec,
    )

    return episode_spec


def analyze_single_hdf5_for_json(
    hdf5_path: Path,
    scene_dataset: str,
    data_split: str,
) -> list[dict]:
    """
    Analyze a single HDF5 file and extract episode specs for JSON conversion.

    Returns:
        List of episode data dicts, one per trajectory.
        Returns empty list if file cannot be opened.
    """
    episodes_data = []

    try:
        f = h5py.File(hdf5_path, "r")
    except OSError as e:
        log.warning(f"Skipping unopenable H5 file {hdf5_path}: {e}")
        return episodes_data

    try:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]

        for traj_key in traj_keys:
            traj_group = f[traj_key]

            # Parse obs_scene
            if "obs_scene" not in traj_group:
                continue

            obs_scene = parse_obs_scene(traj_group["obs_scene"][()])

            # Extract frozen_config
            frozen_config = extract_frozen_config(obs_scene)

            # Get basic episode info
            episode_data = {
                "traj_key": traj_key,
                "file": str(hdf5_path),
                "obs_scene": obs_scene,
                "frozen_config": frozen_config,
            }

            # Episode length - actions is a group with sub-datasets (arm, gripper, etc.)
            if "actions" not in traj_group:
                raise ValueError(f"Trajectory {traj_key} missing 'actions' field")
            actions_group = traj_group["actions"]
            # Get length from first available sub-dataset
            action_keys = list(actions_group.keys())
            if not action_keys:
                raise ValueError(f"Trajectory {traj_key} has empty 'actions' group")
            episode_data["episode_length"] = len(actions_group[action_keys[0]])

            # Success/failure - require success field
            if "success" not in traj_group:
                raise ValueError(f"Trajectory {traj_key} missing 'success' field")
            success_array = np.array(traj_group["success"])
            if len(success_array) == 0:
                raise ValueError(f"Trajectory {traj_key} has empty 'success' array")
            episode_data["success"] = bool(success_array[-1])
            episode_data["first_success"] = np.argmax(success_array)

            # Extract object info for balancing
            episode_data["object_name"] = obs_scene.get("object_name")

            def parse_object_category(object_name):
                if object_name is None:
                    return None
                return "_".join(object_name.split("_")[0:1])

            def parse_object_instance(object_name):
                return object_name

            episode_data["object_category"] = parse_object_category(episode_data["object_name"])
            episode_data["object_instance"] = parse_object_instance(episode_data["object_name"])

            episodes_data.append(episode_data)
    finally:
        f.close()

    return episodes_data


def collect_all_episodes_for_json(
    dataset_path: Path,
    max_h5_files: int | None = None,
) -> tuple[dict[str, dict[str, list[dict]]], str, str, tuple[int, int], str | None, str | None]:
    """
    Collect all episodes from dataset, organized by split.

    Args:
        dataset_path: Path to dataset root
        max_h5_files: Maximum number of H5 files to process (for testing)

    Returns:
        Tuple of (stats_by_split, scene_dataset, data_split, img_resolution, camera_system_class, source_data_date)
        stats_by_split: Dictionary mapping split_name -> {house_id -> list of episode data}
        img_resolution: Image resolution as (width, height) tuple
        camera_system_class: Name of the CameraSystemConfig class used (e.g. "FrankaDroidCameraSystem")
        source_data_date: Approximate date source H5 files were created (YYYY-MM-DD)
    """
    log.info(f"Scanning dataset at: {dataset_path}")

    # In test mode, don't collect all houses upfront - use iterator and stop early
    if max_h5_files is not None:
        log.info(f"Test mode: will stop after {max_h5_files} H5 files")
        # Use iterator to avoid loading all paths into memory
        house_dirs_iter = dataset_path.glob("**/house_*")
        house_dirs = []
        # Collect just enough houses (estimate ~1 H5 per house, collect 2x to be safe)
        for i, hd in enumerate(house_dirs_iter):
            house_dirs.append(hd)
            if i >= max_h5_files * 2:
                break
        house_dirs = sorted(house_dirs)
        log.info(f"Test mode: scanning {len(house_dirs)} houses (limited)")
    else:
        house_dirs = sorted(dataset_path.glob("**/house_*"))
        log.info(f"Found {len(house_dirs)} houses")

    if not house_dirs:
        log.warning(f"No house_* directories found under {dataset_path}")
        return {}, "", "", (0, 0), None, None

    # Extract scene_dataset and data_split from exp_config
    # Note: task_horizon is NOT extracted - it's an evaluation parameter, not a task spec
    scene_dataset = ""
    data_split = ""

    config_files = list(dataset_path.glob("experiment_config*.pkl"))
    if not config_files:
        raise FileNotFoundError(f"No experiment_config*.pkl files found in {dataset_path}")

    config_file = max(config_files, key=lambda p: p.stat().st_mtime)
    with open(config_file, "rb") as f:
        exp_config = ConfigUnpickler(f).load()

    scene_dataset = getattr(exp_config, "scene_dataset", None)
    if scene_dataset is None:
        raise ValueError(f"exp_config missing 'scene_dataset' field: {config_file}")

    data_split = getattr(exp_config, "data_split", None)
    if data_split is None:
        raise ValueError(f"exp_config missing 'data_split' field: {config_file}")

    # Extract camera system class name and img_resolution
    camera_system_class = None
    img_resolution = None
    if hasattr(exp_config, "camera_config") and exp_config.camera_config is not None:
        camera_system_class = type(exp_config.camera_config).__name__
        img_resolution = getattr(exp_config.camera_config, "img_resolution", None)

    if img_resolution is None:
        raise ValueError(
            f"exp_config.camera_config missing 'img_resolution' field: {config_file}. "
            "Cannot create benchmark without knowing the image resolution."
        )

    log.info(
        f"Loaded from exp_config: scene_dataset={scene_dataset}, data_split={data_split}, img_resolution={img_resolution}, camera_system_class={camera_system_class}"
    )

    stats_by_split = {}
    h5_files_processed = 0

    for house_dir in tqdm(house_dirs, desc="Processing houses", unit="house"):
        # Check if we've hit the H5 file limit
        if max_h5_files is not None and h5_files_processed >= max_h5_files:
            log.info(f"Reached max H5 files limit ({max_h5_files}), stopping scan")
            break

        house_name = house_dir.name
        house_id = int(house_name.replace("house_", ""))
        relative_path = house_dir.relative_to(dataset_path)

        # Determine split from path (used for organizing output, not for data_split field)
        parts = relative_path.parts
        if len(parts) >= 2:
            split_parts = parts[1:-1]
            split_name = "/".join(split_parts) if split_parts else "unknown"
        else:
            split_name = "unknown"

        if split_name not in stats_by_split:
            stats_by_split[split_name] = {}

        hdf5_files = sorted(house_dir.glob("trajectories*.h5"))

        house_episodes = []
        for hdf5_file in hdf5_files:
            if max_h5_files is not None and h5_files_processed >= max_h5_files:
                break
            episodes = analyze_single_hdf5_for_json(
                hdf5_file,
                scene_dataset=scene_dataset,
                data_split=data_split,
            )
            house_episodes.extend(episodes)
            h5_files_processed += 1

        if house_episodes:
            stats_by_split[split_name][house_name] = house_episodes

    # Log summary
    log.info("Dataset summary by split:")
    for split_name in sorted(stats_by_split.keys()):
        num_houses = len(stats_by_split[split_name])
        num_episodes = sum(len(eps) for eps in stats_by_split[split_name].values())
        log.info(f"  {split_name}: {num_houses} houses, {num_episodes} episodes")

    # Sample one H5 file to get approximate source data creation date
    source_data_date = None
    sample_h5_files = list(dataset_path.glob("**/trajectories*.h5"))
    if sample_h5_files:
        sample_h5 = sample_h5_files[0]
        mtime = sample_h5.stat().st_mtime
        source_data_date = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        log.info(f"Source data date (from {sample_h5.name}): {source_data_date}")

    return (
        stats_by_split,
        scene_dataset,
        data_split,
        img_resolution,
        camera_system_class,
        source_data_date,
    )


def batch_from_file(file_path: str) -> tuple[str, str]:
    """Extract batch number from file path."""
    m = re.search(r"batch_(\d+)_of_(\d+)", file_path)
    if m:
        return m.groups()
    return "0", "1"


def main():
    parser = argparse.ArgumentParser(
        description="Create JSON-based benchmark from MuJoCo-THOR datasets"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to dataset root (should contain house_* directories)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Full path to output benchmark directory (overrides --output_dir and naming scheme)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Parent directory to place benchmark in (uses auto-generated name within this dir)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag to add to benchmark name (e.g., 'recreation' -> name_20260123_recreation_json_benchmark)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to sample (default: 100, ignored if --all_episodes)",
    )
    parser.add_argument(
        "--all_episodes",
        action="store_true",
        help="Include all successful episodes without sampling/balancing",
    )
    parser.add_argument(
        "--min_cat_num",
        type=int,
        default=0,
        help="Minimum unique (house, instance) pairs per category to include (default: 0)",
    )
    parser.add_argument(
        "--print_stats",
        action="store_true",
        help="Only print statistics, don't create benchmark",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 20 H5 files and create 5-episode benchmark",
    )
    parser.add_argument(
        "--task_horizon_sec",
        type=int,
        default=30,
        help="Task horizon in seconds (default: 30)",
    )

    args = parser.parse_args()

    # Override settings in test mode
    if args.test:
        args.num_episodes = 5
        args.all_episodes = False
        log.info("Test mode: limiting to 20 H5 files and 5 episodes")

    dataset_path = Path(args.base_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Collect episodes
    max_h5_files = 20 if args.test else None
    (
        stats_by_split,
        scene_dataset,
        data_split,
        img_resolution,
        camera_system_class,
        source_data_date,
    ) = collect_all_episodes_for_json(dataset_path, max_h5_files=max_h5_files)

    # Get today's date for benchmark creation
    benchmark_created_date = datetime.datetime.now().strftime("%Y-%m-%d")

    if not stats_by_split:
        log.error("No data found")
        return

    # Build dataframe of all episodes
    rows = []
    for split_name, split in stats_by_split.items():
        for house_name, house_episodes in split.items():
            for ep in house_episodes:
                ep["split"] = split_name
                ep["house"] = house_name
                ep["house_id"] = int(house_name.replace("house_", ""))
                ep["selected"] = False
                ep["score"] = 0.0
                rows.append(ep)

    df_all = pd.DataFrame(rows)
    df_all["is_obja"] = df_all["object_category"].str.startswith("obja", na=False)

    # Filter for successful episodes
    # df_succ = df_all[df_all["success"] == True].copy()
    df_succ = df_all[(df_all["success"] == True) & (df_all["first_success"] > 0)].copy()
    # df_succ = df_all[(df_all["success"] == True) & (df_all["first_success"] >= 150)].copy()

    # Exclude laptops for open/close tasks (based on current script logic)
    if "Open" in args.base_path or "Close" in args.base_path:
        df_succ = df_succ[df_succ["object_category"] != "laptop"]

    # Simplify object categories
    df_succ["object_category"] = df_succ["object_category"].replace(THOR_CAT_SIMPLIFY)
    df_succ["model_instance"] = (
        df_succ["object_instance"].str.rsplit("_", n=3).str[0]
    )  # expects obja_<uid>_ ...
    # object_instance is unique to a simulation occurance (i.e. includes scene index)
    # model_instance is unique to an objects 3D model

    # Replace objaverse categories with synsets
    mask = df_succ["is_obja"]
    if mask.any():
        obja_uids = df_succ.loc[mask, "object_instance"].str.split("_").str[1]
        synsets = obja_uids.map(
            lambda k: ObjectMeta.annotation(k).get("synset") if ObjectMeta.annotation(k) else None
        )
        df_succ.loc[mask & synsets.notna(), "object_category"] = synsets

    # Filter out infrequent categories
    if args.min_cat_num > 0:
        counts = df_succ.groupby("object_category")[["house", "model_instance"]].apply(
            lambda g: g.drop_duplicates(subset=["house", "model_instance"]).shape[0]
        )
        valid_cats = counts[counts >= args.min_cat_num].index.tolist()
        invalid_cats = counts[counts < args.min_cat_num].index.tolist()
        log.info(
            f"Excluding categories (less than {args.min_cat_num} unique pairs): {invalid_cats}"
        )
        df = df_succ[df_succ["object_category"].isin(valid_cats)].copy()
    else:
        df = df_succ.copy()

    log.info(f"Total episodes: {len(df_all)}")
    log.info(f"Successful episodes after filtering: {len(df)}")
    log.info(f"Percent objaverse: {100.0 * df['is_obja'].sum() / len(df):.2f}%")

    if len(df) == 0:
        log.error("No episodes remaining after filtering")
        return

    # Select episodes: either all or balanced sampling
    if args.all_episodes:
        # Take all successful episodes without balancing
        df["selected"] = True
        log.info(f"All episodes mode: selecting all {len(df)} episodes")
        dfs = df
    else:
        # Greedy balanced sampling
        used_pairs = set()

        def row2used_score(row):
            if (row["house"], row["model_instance"]) in used_pairs:
                return 1
            return 0

        N = args.num_episodes
        num_available = len(df[~df["selected"]])
        if num_available < N:
            log.warning(f"Requested {N} episodes but only {num_available} available")
            N = num_available

        for _ in tqdm(range(N), desc="Selecting episodes"):
            dfs = df[df["selected"] == True]
            cur_house_freq = dfs["house"].value_counts(normalize=True)
            cur_category_freq = dfs["object_category"].value_counts(normalize=True)
            cur_obj_inst_freq = dfs["model_instance"].value_counts(normalize=True)

            # Compute scores (lower is better for selection)
            # Priority: avoid reusing (house, instance) pairs > balance categories > balance houses > balance instances
            df.loc[:, "score"] = (
                df.apply(row2used_score, axis=1) * 1000
                + df["object_category"].map(cur_category_freq).fillna(0) * 100
                + df["house"].map(cur_house_freq).fillna(0) * 10
                + df["model_instance"].map(cur_obj_inst_freq).fillna(0) * 1
            )

            unselected = df[~df["selected"]]
            if len(unselected) == 0:
                log.warning("No more unselected episodes available")
                break

            min_score = unselected["score"].min()
            idx = (
                unselected[unselected["score"] == min_score]
                .sample(1, random_state=RANDOM_STATE)
                .index
            )

            row = df.loc[idx].iloc[0]
            used_pairs.add((row["house"], row["model_instance"]))
            df.loc[idx, "selected"] = True

        dfs = df[df["selected"] == True]

    log.info(f"Selected {len(dfs)} episodes")
    log.info(f"Categories: {dfs['object_category'].nunique()}")
    log.info(f"Houses: {dfs['house'].nunique()}")

    if args.print_stats:
        log.info("Category distribution:")
        log.info(dfs["object_category"].value_counts())
        log.info("House distribution:")
        log.info(dfs["house"].value_counts())
        return

    # Create output directory
    if args.output_path:
        # Full path specified - use directly
        benchmark_path = Path(args.output_path)
    else:
        # Build benchmark name from dataset name
        base_name = dataset_path
        if base_name.name in ("train", "test", "val"):
            base_name = base_name.parent
        base_name = base_name.name

        # Build suffix parts
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        suffix_parts = [timestamp]
        if args.tag:
            suffix_parts.append(args.tag)
        suffix_parts.append("test_json_benchmark" if args.test else "json_benchmark")
        suffix = "_".join(suffix_parts)

        benchmark_name = f"{base_name}_{suffix}"

        # Place in output_dir if specified, otherwise next to dataset
        if args.output_dir:
            benchmark_path = Path(args.output_dir) / benchmark_name
        else:
            benchmark_path = dataset_path.parent / benchmark_name

    benchmark_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Creating benchmark at: {benchmark_path}")

    # Collect all episode specs into a list
    all_episode_specs = []
    houses_seen = set()

    for _, row in tqdm(dfs.iterrows(), desc="Converting episodes", total=len(dfs)):
        house_name = row["house"]
        house_id = row["house_id"]
        frozen_config = row["frozen_config"]
        obs_scene = row["obs_scene"]
        source_h5_file = row["file"]
        source_traj_key = row["traj_key"]
        source_episode_length = row["episode_length"]

        houses_seen.add(house_name)

        # Convert to EpisodeSpec
        episode_spec = frozen_config_to_episode_spec(
            frozen_config=frozen_config,
            obs_scene=obs_scene,
            house_id=house_id,
            scene_dataset=scene_dataset,
            data_split=data_split,
            source_h5_file=source_h5_file,
            source_traj_key=source_traj_key,
            source_episode_length=source_episode_length,
            img_resolution=img_resolution,
            camera_system_class=camera_system_class,
            source_data_date=source_data_date,
            benchmark_created_date=benchmark_created_date,
            task_horizon_sec=args.task_horizon_sec,
        )

        all_episode_specs.append(episode_spec.model_dump())

    num_episodes = len(all_episode_specs)
    num_houses = len(houses_seen)
    log.info(f"Successfully converted: {num_episodes} episodes from {num_houses} houses")

    # Write single benchmark JSON file (list of episode dicts)
    benchmark_file = benchmark_path / "benchmark.json"
    with open(benchmark_file, "w") as f:
        json.dump(all_episode_specs, f, indent=2)
    log.info(f"Wrote benchmark to: {benchmark_file}")

    # In test mode, print out first two episode specs for review
    if args.test and num_episodes > 0:
        log.info("\n" + "=" * 80)
        log.info("TEST MODE: Printing first 2 episode specifications for review")
        log.info("=" * 80)
        for i, spec in enumerate(all_episode_specs[:2]):
            log.info(f"\n--- Episode {i} ---")
            log.info(json.dumps(spec, indent=2))
        log.info("=" * 80 + "\n")

    # Compute enhanced metadata statistics
    # Task class counts
    task_cls_counts = {}
    for spec in all_episode_specs:
        task_cls = spec.get("task", {}).get("task_cls", "unknown")
        task_cls_counts[task_cls] = task_cls_counts.get(task_cls, 0) + 1

    # Object category counts (from the dataframe we already have)
    object_category_counts = dfs["object_category"].value_counts().to_dict()

    # Robot counts
    robot_counts = {}
    for spec in all_episode_specs:
        robot_name = spec.get("robot", {}).get("robot_name", "unknown")
        robot_counts[robot_name] = robot_counts.get(robot_name, 0) + 1

    # Episode length statistics (from source episodes)
    episode_lengths = [
        spec.get("source", {}).get("episode_length")
        for spec in all_episode_specs
        if spec.get("source", {}).get("episode_length") is not None
    ]
    if episode_lengths:
        episode_length_stats = {
            "min": float(min(episode_lengths)),
            "max": float(max(episode_lengths)),
            "mean": float(np.mean(episode_lengths)),
            "median": float(np.median(episode_lengths)),
        }
    else:
        episode_length_stats = None

    # Write optional metadata (as separate file for human readability)
    metadata = BenchmarkMetadata(
        description=f"JSON benchmark from {dataset_path.name}",
        created_at=datetime.datetime.now().isoformat(),
        source_datagen_path=str(dataset_path),
        num_episodes=num_episodes,
        num_houses=num_houses,
        task_cls_counts=task_cls_counts,
        object_category_counts=object_category_counts,
        robot_counts=robot_counts,
        episode_length_stats=episode_length_stats,
        camera_system_class=camera_system_class,
        source_data_date=source_data_date,
        benchmark_created_date=benchmark_created_date,
    )
    metadata.to_json_file(benchmark_path / "benchmark_metadata.json")

    log.info(f"Benchmark created at: {benchmark_path}")
    log.info(f"Total episodes: {num_episodes}")
    log.info(f"Total houses: {num_houses}")


if __name__ == "__main__":
    main()
