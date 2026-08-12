"""
Patch existing JSON benchmarks to add missing fields.

This script scans a directory containing benchmark directories (each with a
benchmark.json file), infers missing fields from the camera_system_class
stored in the metadata, and patches the benchmark.json.

Currently patches:
- img_resolution: (width, height) for all cameras, inferred from camera_system_class
- record_depth: per-camera boolean, inferred from camera_system_class and camera name
- Updates max_place_receptacle_pos_displacement and max_place_receptacle_rot_displacement
  in task dicts to match the benchmark schema defaults (in benchmark_schema.py)

USAGE:
    python scripts/datagen/patch_benchmarks.py \
        --benchmarks_dir /path/to/benchmarks \
        [--dry_run]

The script will:
1. Find all directories containing benchmark.json files
2. Load benchmark_metadata.json to get camera_system_class
3. Look up field values from known camera system configurations
4. Patch each episode spec with the missing fields
5. Write the updated benchmark.json (unless --dry_run)

Requires:
    - Each benchmark directory must have benchmark_metadata.json with camera_system_class
    - camera_system_class must be a known class with defined configuration
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Mapping from camera system class names to their img_resolution (width, height)
# This is derived from molmo_spaces/configs/camera_configs.py
CAMERA_SYSTEM_IMG_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "RBY1MjcfCameraSystem": (640, 480),
    "RBY1GoProD455CameraSystem": (1024, 576),
    "FrankaRandomizedD405D455CameraSystem": (624, 352),
    "FrankaDroidCameraSystem": (624, 352),
    "FrankaEasyRandomizedDroidCameraSystem": (624, 352),
    "FrankaRandomizedDroidCameraSystem": (624, 352),
    "FrankaGoProD405D455CameraSystem": (640, 480),
    "FrankaGoProD405RandomizedCameraSystem": (640, 480),
    "FrankaOmniPurposeCameraSystem": (624, 352),

}

# Mapping from camera system class names to per-camera record_depth settings
# Key is camera_system_class, value is dict mapping camera_name -> record_depth
# If a camera is not listed, defaults to False
CAMERA_SYSTEM_RECORD_DEPTH: dict[str, dict[str, bool]] = {
    "RBY1MjcfCameraSystem": {
        "wrist_camera_l": True,
        "wrist_camera_r": True,
    },
    "RBY1GoProD455CameraSystem": {
        "wrist_camera_l": True,
        "wrist_camera_r": True,
    },
    "FrankaRandomizedD405D455CameraSystem": {},  # wrist_camera has record_depth commented out
    "FrankaDroidCameraSystem": {},  # wrist_camera has record_depth commented out
    "FrankaEasyRandomizedDroidCameraSystem": {
        "wrist_camera": True,
    },
    "FrankaRandomizedDroidCameraSystem": {},  # no record_depth enabled
    "FrankaGoProD405D455CameraSystem": {
        "wrist_camera": True,
    },
    "FrankaGoProD405RandomizedCameraSystem": {
        "wrist_camera": True,
    },
}

DEFAULT_TASK_HORIZONS_SEC = {
    'molmo_spaces.tasks.pick_task.PickTask': 20,
    "molmo_spaces.tasks.opening_tasks.OpeningTask": 30,
    "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask": 40,
    "molmo_spaces.tasks.pick_and_place_next_to_task.PickAndPlaceNextToTask": 40,
    "molmo_spaces.tasks.pick_and_place_color_task.PickAndPlaceColorTask": 40,
    "molmo_spaces.tasks.opening_tasks.DoorOpeningTask": 40,
    "molmo_spaces.tasks.nav_task.NavToObjTask": 100,
}

DEFAULT_TASK_TYPES = {
    'molmo_spaces.tasks.pick_task.PickTask': 'pick',
    "molmo_spaces.tasks.opening_tasks.OpeningTask": 'open',
    "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask": 'pick_and_place',
    "molmo_spaces.tasks.pick_and_place_next_to_task.PickAndPlaceNextToTask": 'pick_and_place_next_to',
    "molmo_spaces.tasks.pick_and_place_color_task.PickAndPlaceColorTask": 'pick_and_place_color',
    "molmo_spaces.tasks.opening_tasks.DoorOpeningTask": 'open_door',
    "molmo_spaces.tasks.nav_task.NavToObjTask": 'nav_to_obj',
}


def patch_benchmark(benchmark_dir: Path, args, dry_run: bool = False) -> bool:
    """
    Patch a single benchmark directory to add missing fields.

    Args:
        benchmark_dir: Path to benchmark directory containing benchmark.json
        dry_run: If True, don't write changes, just report what would be done

    Returns:
        True if patched successfully, False if skipped or failed
    """
    benchmark_file = benchmark_dir / "benchmark.json"
    metadata_file = benchmark_dir / "benchmark_metadata.json"

    if not benchmark_file.exists():
        log.warning(f"No benchmark.json in {benchmark_dir}, skipping")
        return False

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"No benchmark_metadata.json in {benchmark_dir}. "
            "Cannot determine camera_system_class without metadata."
        )

    # Load metadata to get camera_system_class
    with open(metadata_file) as f:
        metadata = json.load(f)

    camera_system_class = metadata.get("camera_system_class")

    # Camera-related patches require knowing the camera system; others don't.
    if camera_system_class and camera_system_class not in CAMERA_SYSTEM_IMG_RESOLUTIONS:
        raise ValueError(
            f"Unknown camera_system_class '{camera_system_class}' in {benchmark_dir}. "
            f"Add it to CAMERA_SYSTEM_IMG_RESOLUTIONS in this script. "
            f"Known classes: {list(CAMERA_SYSTEM_IMG_RESOLUTIONS.keys())}"
        )

    img_resolution = CAMERA_SYSTEM_IMG_RESOLUTIONS.get(camera_system_class) if camera_system_class else None
    record_depth_map = CAMERA_SYSTEM_RECORD_DEPTH.get(camera_system_class, {}) if camera_system_class else {}

    # Load benchmark
    with open(benchmark_file) as f:
        episodes = json.load(f)

    if not isinstance(episodes, list):
        raise ValueError(
            f"benchmark.json in {benchmark_dir} is not a list of episodes"
        )

    if not episodes:
        log.warning(f"Empty benchmark in {benchmark_dir}, skipping")
        return False

    # Task fields that should match the benchmark schema defaults.
    # Maps field name -> authoritative value from the TaskSpec in benchmark_schema.py.
    TASK_FIELD_SCHEMA_VALUES = {
        "max_place_receptacle_pos_displacement": 0.15,
        "max_place_receptacle_rot_displacement": np.deg2rad(60),
    }

    OLD_MODULE_PREFIX = "mujoco_thor."
    NEW_MODULE_PREFIX = "molmo_spaces."

    # Track what we're patching
    needs_img_resolution = img_resolution is not None and "img_resolution" not in episodes[0]
    needs_record_depth = False
    first_task = episodes[0].get("task", {})
    needs_task_cls_rename = first_task.get("task_cls", "").startswith(OLD_MODULE_PREFIX)
    needs_task_field_update = any(
        field in first_task and first_task[field] != expected
        for field, expected in TASK_FIELD_SCHEMA_VALUES.items()
    )

    if camera_system_class and episodes[0].get("cameras"):
        first_cam = episodes[0]["cameras"][0]
        needs_record_depth = "record_depth" not in first_cam

    #if not needs_img_resolution and not needs_record_depth and not needs_time_limit and not needs_task_field_update and not needs_task_cls_rename:
    #    log.info(f"Already patched: {benchmark_dir}")
    #    return True

    # Patch each episode
    for episode in episodes:
        # Rename task_cls from old module path first, so downstream lookups work
        if needs_task_cls_rename:
            old_cls = episode["task"]["task_cls"]
            episode["task"]["task_cls"] = old_cls.replace(OLD_MODULE_PREFIX, NEW_MODULE_PREFIX, 1)

        # Patch img_resolution
        if needs_img_resolution:
            episode["img_resolution"] = list(img_resolution)

        # Patch record_depth for each camera
        if needs_record_depth and "cameras" in episode:
            for camera in episode["cameras"]:
                camera_name = camera.get("name", "")
                camera["record_depth"] = record_depth_map.get(camera_name, False)

        needs_time_limit = "task_horizon_sec" not in episode.get("task", {})
        if needs_time_limit:
            task_cls = episode["task"]["task_cls"]
            if task_cls in DEFAULT_TASK_HORIZONS_SEC:
                time_horizon = DEFAULT_TASK_HORIZONS_SEC[task_cls]
            else:
                log.warning(
                    f"Unknown task_cls '{task_cls}' in {benchmark_dir}. "
                    f"Using default time_limit=30 seconds. "
                )
                time_horizon = 30
            episode.setdefault("task", {})["task_horizon_sec"] = time_horizon
            episode.pop("task_horizon_sec", None)

        #needs_task_type = "task_type" not in episode.get("task", {})
        needs_task_type=True
        if needs_task_type:
            task_cls = episode["task"]["task_cls"]
            if args.task_type is not None:
                task_type = args.task_type

            elif task_cls in DEFAULT_TASK_TYPES:
                task_type = DEFAULT_TASK_TYPES[task_cls]
            else:
                log.warning(
                    f"Unknown task_cls '{task_cls}' in {benchmark_dir}. "
                    f"Using default task_type=pick "
                )
                task_type = "pick"

            episode.setdefault("task", {})["task_type"] = task_type
            episode.pop("task_type", None)


        if needs_task_field_update and "task" in episode:
            for field, expected in TASK_FIELD_SCHEMA_VALUES.items():
                if field in episode["task"]:
                    episode["task"][field] = expected

    patch_desc = []
    if needs_task_cls_rename:
        patch_desc.append(f"rename task_cls {OLD_MODULE_PREFIX}* -> {NEW_MODULE_PREFIX}*")
    if needs_img_resolution:
        patch_desc.append(f"img_resolution={img_resolution}")
    if needs_record_depth:
        patch_desc.append(f"record_depth (per-camera)")
    if needs_task_field_update:
        patch_desc.append(f"update task fields to schema defaults: {TASK_FIELD_SCHEMA_VALUES}")

    log.info(
        f"{'Would patch' if dry_run else 'Patching'}: {benchmark_dir} "
        f"({len(episodes)} episodes, {', '.join(patch_desc)})"
    )

    if not dry_run:
        with open(benchmark_file, "w") as f:
            json.dump(episodes, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Patch existing benchmarks to add missing fields (img_resolution, record_depth)"
    )
    parser.add_argument(
        "--benchmarks_dir",
        type=str,
        required=True,
        help="Directory containing benchmark directories (each with benchmark.json)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't write changes, just report what would be done",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="Overwrite imputed task type",
    )


    args = parser.parse_args()
    benchmarks_dir = Path(args.benchmarks_dir)

    if not benchmarks_dir.exists():
        raise ValueError(f"Benchmarks directory does not exist: {benchmarks_dir}")

    # Find all benchmark directories
    benchmark_dirs = []
    #for path in benchmarks_dir.iterdir():
    if benchmarks_dir.is_dir() and (benchmarks_dir / "benchmark.json").exists():
        benchmark_dirs.append(benchmarks_dir)

    if not benchmark_dirs:
        log.warning(f"No benchmark directories found in {benchmarks_dir}")
        return

    log.info(f"Found {len(benchmark_dirs)} benchmark directories")

    # Patch each benchmark
    patched = 0
    failed = 0
    for benchmark_dir in sorted(benchmark_dirs):
        try:
            if patch_benchmark(benchmark_dir, args, dry_run=args.dry_run):
                patched += 1
        except Exception as e:
            log.error(f"Failed to patch {benchmark_dir}: {e}")
            raise e
            failed += 1

    log.info(f"Done. Patched: {patched}, Failed: {failed}")

    if args.dry_run:
        log.info("(Dry run - no files were modified)")


if __name__ == "__main__":
    main()
