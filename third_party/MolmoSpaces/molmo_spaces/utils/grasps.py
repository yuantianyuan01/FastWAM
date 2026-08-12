"""
This module contains functionality for loading grasps from registered grasp libraries.

Note: this module caches aggressively, so grasp/asset libraries must be registered before the first call into this module.
New registrations will not be visible until the caches are cleared.
"""

import inspect
import logging
import random
import zipfile
from collections.abc import Sequence
from functools import lru_cache, wraps
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import MlSpacesArticulationObject, MlSpacesObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.molmo_spaces_constants import (
    ASSETS_DIR,
    OBJECT_LIBRARY_TO_GRASP_LIBRARIES,
    USER_GRASP_LIBRARIES,
)
from molmo_spaces.utils.lazy_loading_utils import get_user_grasp_library_index, locate_uid_package

log = logging.getLogger(__name__)


@lru_cache(maxsize=10000)
def _locate_uid_package(uid: str):
    return locate_uid_package(uid)


def get_grasp_libraries_for_object(uid: str) -> list[str]:
    package, _, _ = _locate_uid_package(uid)
    if package not in OBJECT_LIBRARY_TO_GRASP_LIBRARIES:
        return []
    return list(OBJECT_LIBRARY_TO_GRASP_LIBRARIES[package])


def _filter_grasp_libraries_for_object(
    uid: str, grasp_libraries: Sequence[str] | None = None
) -> list[str]:
    available_grasp_libraries = get_grasp_libraries_for_object(uid)
    if grasp_libraries is None:
        return available_grasp_libraries
    available_set = set(available_grasp_libraries)
    return [library for library in grasp_libraries if library in available_set]


def get_pickup_grasp_path(uid: str, grasp_libraries: Sequence[str] | None = None) -> Path | None:
    libs = _filter_grasp_libraries_for_object(uid, grasp_libraries)

    for library in libs:
        if library in USER_GRASP_LIBRARIES:
            grasp_library_dir = USER_GRASP_LIBRARIES[library]
            grasp_library_index = get_user_grasp_library_index(grasp_library_dir)
            robot_name = library.split("/", 1)[-1]
            grasp_file = grasp_library_index.grasp_paths.get(robot_name, {}).get(uid, None)
            if grasp_file is not None:
                grasp_file = grasp_library_dir / grasp_file
        else:
            grasp_file = ASSETS_DIR / f"grasps/{library}/{uid}/{uid}_grasps_filtered.npz"

        if grasp_file is not None and grasp_file.exists():
            return grasp_file

    return None


def get_joint_grasp_path(
    uid: str, joint_name: str, grasp_libraries: Sequence[str] | None = None
) -> Path | None:
    # If we only specify one grasp library, just use it and fail later if not found.
    # In general we shouldn't do this, but thor articulated objects can't be looked
    # up by uid (for whatever reason) so this serves as a workaround by skipping the lookup.
    # Client code doing articulated object manipulation with thor should only specify one grasp library.
    if grasp_libraries is not None and len(grasp_libraries) == 1:
        libs = grasp_libraries
    else:
        libs = _filter_grasp_libraries_for_object(uid, grasp_libraries)

    for library in libs:
        if library in USER_GRASP_LIBRARIES:
            grasp_library_dir = USER_GRASP_LIBRARIES[library]
            grasp_library_index = get_user_grasp_library_index(grasp_library_dir)
            robot_name = library.split("/", 1)[-1]
            grasp_file = (
                grasp_library_index.articulated_grasp_paths.get(robot_name, {})
                .get(uid, {})
                .get(joint_name, None)
            )
            if grasp_file is not None:
                grasp_file = grasp_library_dir / grasp_file
        else:
            # droid (thor) is the only builtin grasp library with joint grasps
            grasp_file = ASSETS_DIR / f"grasps/droid/{uid}/{joint_name}_grasps_filtered.npz"

        if grasp_file is not None and grasp_file.exists():
            return grasp_file

    return None


def sanitize_grasp_library_list_and_cache(cache_size: int):
    def decorator(func):
        sig = inspect.signature(func)

        @lru_cache(maxsize=cache_size)
        def cached(*args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            libs = bound.arguments.get("grasp_libraries")
            if libs is not None and not isinstance(libs, tuple):
                bound.arguments["grasp_libraries"] = tuple(libs)
            return cached(*bound.args, **bound.kwargs)

        wrapper.cache_info = cached.cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


@sanitize_grasp_library_list_and_cache(cache_size=10000)
def has_pickup_grasp_path(uid: str, grasp_libraries: Sequence[str] | None = None) -> bool:
    return get_pickup_grasp_path(uid, grasp_libraries) is not None


@sanitize_grasp_library_list_and_cache(cache_size=10000)
def has_joint_grasp_path(
    uid: str, joint_name: str, grasp_libraries: Sequence[str] | None = None
) -> bool:
    return get_joint_grasp_path(uid, joint_name, grasp_libraries) is not None


@sanitize_grasp_library_list_and_cache(cache_size=10000)
def has_valid_pickup_grasps(
    uid: str, num_grasps: int = 1, grasp_libraries: Sequence[str] | None = None
) -> bool:
    grasp_path = get_pickup_grasp_path(uid, grasp_libraries)
    if grasp_path is None:
        return False

    # read the number of grasps from the grasp file without loading the entire file into memory
    with zipfile.ZipFile(grasp_path) as zf:
        with zf.open("transforms.npy") as f:
            version = np.lib.format.read_magic(f)
            if version[0] == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            return shape[0] >= num_grasps


@sanitize_grasp_library_list_and_cache(cache_size=10000)
def has_valid_joint_grasps(
    uid: str,
    joint_name: str,
    num_grasps: int = 1,
    grasp_libraries: Sequence[str] | None = None,
) -> bool:
    grasp_path = get_joint_grasp_path(uid, joint_name, grasp_libraries)
    if grasp_path is None:
        return False

    # read the number of grasps from the grasp file without loading the entire file into memory
    with zipfile.ZipFile(grasp_path) as zf:
        with zf.open("transforms.npy") as f:
            version = np.lib.format.read_magic(f)
            if version[0] == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            return shape[0] >= num_grasps


def load_pickup_grasps(
    uid: str, grasp_libraries: list[str] | None = None, num_grasps: int = 50
) -> np.ndarray:
    """
    Load the first available pickup grasps for a given object in the local frame.

    Args:
        uid: The asset ID of the object
        grasp_libraries: The grasp libraries to use (defaults to all available libraries for the object)
        num_grasps: The maximum number of grasps to load

    Returns:
        A numpy array of shape (N, 4, 4) containing the grasp poses in the local frame
    """
    grasp_path = get_pickup_grasp_path(uid, grasp_libraries)
    if grasp_path is None:
        raise ValueError(f"No grasp file found for {uid}")

    npz_data = np.load(grasp_path)
    transforms: np.ndarray = npz_data["transforms"]
    if len(transforms) <= num_grasps:
        return transforms
    else:
        idxs = random.sample(range(len(transforms)), num_grasps)
        return transforms[idxs]


def load_joint_grasps(
    uid: str, joint_name: str, grasp_libraries: list[str] | None = None, num_grasps: int = 50
) -> np.ndarray:
    """
    Load the first available joint grasps for a given object and joint in the joint's local frame.

    Args:
        uid: The asset ID of the object
        joint_name: The name of the joint
        grasp_libraries: The grasp libraries to use (defaults to all available libraries for the object)
        num_grasps: The maximum number of grasps to load

    Returns:
        A numpy array of shape (N, 4, 4) containing the grasp poses in the joint's local frame.
    """
    grasp_path = get_joint_grasp_path(uid, joint_name, grasp_libraries)
    if grasp_path is None:
        raise ValueError(f"No joint grasp file found for {uid}/{joint_name}")

    npz_data = np.load(grasp_path)
    transforms: np.ndarray = npz_data["transforms"]
    if len(transforms) <= num_grasps:
        return transforms
    else:
        idxs = random.sample(range(len(transforms)), num_grasps)
        return transforms[idxs]


def flip_grasps(grasps: np.ndarray) -> np.ndarray:
    flip = np.eye(4)
    flip[:3, :3] = R.from_euler("z", 180, degrees=True).as_matrix()
    return grasps @ flip


def get_pickup_grasps(
    env: CPUMujocoEnv,
    obj: MlSpacesObject,
    include_flipped: bool = True,
    grasp_libraries: list[str] | None = None,
) -> np.ndarray:
    """
    Load the first available pickup grasps for a given object in the world frame.

    Args:
        env: The environment
        obj: The object
        include_flipped: Whether to include flipped grasps
        grasp_libraries: The grasp libraries to use (defaults to all available libraries for the object)

    Returns:
        A numpy array of shape (N, 4, 4) containing the grasp poses in the world frame.
    """
    scene_metadata = env.current_scene_metadata
    if scene_metadata is None:
        raise ValueError(f"Could not load grasps for object {obj.name}: No scene metadata found!")
    if obj.name not in scene_metadata["objects"]:
        raise ValueError(
            f"Could not load grasps for object {obj.name}: Object not found in scene metadata!"
        )

    asset_id: str = scene_metadata["objects"][obj.name]["asset_id"]
    grasps = load_pickup_grasps(asset_id, grasp_libraries, num_grasps=int(1e6))
    if len(grasps) == 0:
        raise ValueError(f"No grasps found for {obj.name}")

    grasps_world = obj.pose @ grasps
    if include_flipped:
        all_grasp_poses = np.concatenate([grasps_world, flip_grasps(grasps_world)])
    else:
        all_grasp_poses = grasps_world

    log.info(
        f"Loaded {len(all_grasp_poses)} total grasp poses"
        + (" (including flipped versions)" if include_flipped else "")
    )
    return all_grasp_poses


def get_joint_grasps(
    env: CPUMujocoEnv,
    obj: MlSpacesArticulationObject,
    joint_idx: int,
    include_flipped: bool = True,
    grasp_libraries: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the first available joint grasps for a given object and joint in the world frame.

    Args:
        env: The environment
        obj: The object
        joint_idx: The index of the joint
        include_flipped: Whether to include flipped grasps
        grasp_libraries: The grasp libraries to use (defaults to all available libraries for the object)

    Returns:
        Numpy array of shape (N, 4, 4) containing the grasp poses in the world frame.
        Numpy array of shape (4, 4) containing the joint body pose in the world frame.
    """
    scene_metadata = env.current_scene_metadata
    if scene_metadata is None:
        raise ValueError(f"Could not load grasps for object {obj.name}: No scene metadata found!")
    if obj.name not in scene_metadata["objects"]:
        raise ValueError(
            f"Could not load grasps for object {obj.name}: Object not found in scene metadata!"
        )

    joint_name: str = obj.joint_names[joint_idx]
    asset_joint_name = scene_metadata["objects"][obj.name]["name_map"]["joints"][joint_name]
    asset_id = scene_metadata["objects"][obj.name]["asset_id"]

    grasps = load_joint_grasps(asset_id, asset_joint_name, grasp_libraries, num_grasps=int(1e6))
    if len(grasps) == 0:
        raise ValueError(f"No grasps found for {obj.name}/{joint_name}")

    joint_bodyid = env.current_model.joint(joint_name).bodyid.item()
    joint_body_pose = np.eye(4)
    joint_body_pose[:3, 3] = env.current_data.xpos[joint_bodyid]
    joint_body_pose[:3, :3] = env.current_data.xmat[joint_bodyid].reshape(3, 3)

    grasps_world = joint_body_pose @ grasps
    if include_flipped:
        all_grasp_poses = np.concatenate([grasps_world, flip_grasps(grasps_world)])
    else:
        all_grasp_poses = grasps_world
    log.info(
        f"Loaded {len(all_grasp_poses)} total grasp poses"
        + (" (including flipped versions)" if include_flipped else "")
    )
    return all_grasp_poses, joint_body_pose
