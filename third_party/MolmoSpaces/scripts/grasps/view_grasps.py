"""
View grasps for objects in a given scene.

example:
mjpython scripts/grasps/view_grasps.py --scene_index 2 --dataset_name ithor --split train --number_of_grasp_per_object 100 --check_collision

"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import argparse

from mujoco import MjSpec
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.utils.lazy_loading_utils import (
    install_scene_from_path,
    install_scene_from_source_index,
)
from molmo_spaces.utils.grasp_sample import (
    add_grasp_collision_bodies,
    get_noncolliding_grasp_mask,
)
from molmo_spaces.utils.grasps import (
    has_pickup_grasp_path,
    has_joint_grasp_path,
    load_pickup_grasps,
    load_joint_grasps,
)
from molmo_spaces.configs.policy_configs import ObjectManipulationPlannerPolicyConfig
from molmo_spaces.utils.constants.object_constants import (
    ALL_PICKUP_TYPES_THOR,
    EXTENDED_ARTICULATION_TYPES_THOR,
)
from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata
from molmo_spaces.utils.pose import pos_quat_to_pose_mat
from molmo_spaces.utils.sampler_utils import furthest_point_sampling as fps_generic


def furthest_point_sampling(poses, num_samples):
    """
    Perform furthest point sampling on grasp poses.
    Selects points that are maximally distant from each other for better visualization coverage.

    Args:
        poses: Array of 4x4 transformation matrices, shape (N, 4, 4)
        num_samples: Number of samples to select

    Returns:
        Array of indices into poses array, shape (num_samples,)
    """
    if len(poses) <= num_samples:
        return np.arange(len(poses))

    # Extract positions (translation components) from poses
    positions = poses[:, :3, 3]  # shape (N, 3)

    # Use the generic furthest point sampling implementation
    return fps_generic(positions, num_samples)


def _show_grasp_poses(
    viewer,
    poses,
    grasp_width: float,
    grasp_length: float,
    grasp_height: float,
    grasp_base_pos: np.ndarray,
    color=(0, 0, 1, 1),
):
    """Show grasp poses in the viewer using mjv_initGeom (similar to _show_poses).

    Args:
        viewer: MuJoCo viewer handle
        poses: Array of 4x4 transformation matrices, shape (N, 4, 4)
        grasp_width: Width of the grasp
        grasp_length: Length of the grasp
        grasp_height: Height of the grasp
        grasp_base_pos: Base position of the grasp in TCP frame
        color: RGBA color tuple, default blue
    """
    if viewer is None:
        return

    assert poses.ndim == 3 and poses.shape[1:] == (4, 4)
    ngeom = viewer.user_scn.ngeom

    # Define relative parts of the gripper (same as in _show_poses)
    half_length = grasp_length / 2
    half_width = grasp_width / 2
    cylinder_radius = grasp_height / 2

    gripper_parts = [
        ("sphere", mujoco.mjtGeom.mjGEOM_SPHERE, [cylinder_radius, 0.0, 0.0], grasp_base_pos),
        (
            "cylinder_left",
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [cylinder_radius, half_length, 0.0],
            np.array([0.0, half_width, half_length]) + grasp_base_pos,
        ),
        (
            "cylinder_right",
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [cylinder_radius, half_length, 0.0],
            np.array([0.0, -half_width, half_length]) + grasp_base_pos,
        ),
        ("connecting_bar", mujoco.mjtGeom.mjGEOM_BOX, [0.002, 0.044, 0.002], grasp_base_pos),
    ]

    i = 0
    for T in poses:
        for _part_name, geom_type, size, offset in gripper_parts:
            # Transform local offset by the pose
            offset_trf = np.eye(4)
            offset_trf[:3, 3] = offset
            A = T.copy() @ offset_trf

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[ngeom + i],
                type=geom_type,
                size=np.array(size),
                pos=A[:3, 3],
                mat=T[:3, :3].flatten(),
                rgba=np.array(color),
            )
            i += 1

    viewer.user_scn.ngeom = ngeom + i


def extract_objects_from_metadata(model, scene_metadata):
    """Extract objects from scene metadata filtered by pickup and articulation categories.

    Returns:
        tuple: (pickup_objects, jointed_objects) where:
            pickup_objects: {asset_id: body_name} mapping for pickup objects
            jointed_objects: {asset_id: {joint_name: (model_joint_name, joint_body_name)}} mapping for articulated objects
    """
    if scene_metadata is None:
        print("Warning: No scene metadata found, falling back to model-based extraction")
        return {}, {}

    # Separate pickup and articulation types (convert to lowercase for comparison)
    pickup_categories = set(cat.lower() for cat in ALL_PICKUP_TYPES_THOR)
    articulation_categories = set(cat.lower() for cat in EXTENDED_ARTICULATION_TYPES_THOR)

    objects_dict = scene_metadata.get("objects", {})
    pickup_objects = {}  # {asset_id: body_name}
    jointed_objects = {}  # {asset_id: {joint_name: (model_joint_name, joint_body_name)}}

    for object_name, object_data in objects_dict.items():
        category = object_data.get("category", "").lower()
        asset_id = object_data.get("asset_id", None)

        if not asset_id:
            continue

        name_map = object_data.get("name_map", {})
        bodies_map = name_map.get("bodies", {})
        joints_map = name_map.get("joints", {})

        # Find the root body name in the model
        body_name = None
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id >= 0:
            body_name = object_name
        else:
            # Try bodies from name_map
            for hash_name, actual_name in bodies_map.items():
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hash_name)
                if body_id >= 0:
                    body_name = hash_name
                    break
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, actual_name)
                if body_id >= 0:
                    body_name = actual_name
                    break

        if not body_name:
            continue

        # Handle pickup objects
        if category in pickup_categories and has_pickup_grasp_path(asset_id):
            pickup_objects[asset_id] = body_name
            print(
                f"Found pickup object: {object_name} -> asset_id={asset_id}, body={body_name}, category={category}"
            )

        # Handle Objaverse pickup objects
        if "obja" in category and has_pickup_grasp_path(asset_id):
            pickup_objects[asset_id] = body_name
            print(
                f"Found Objaverse pickup object: {object_name} -> asset_id={asset_id}, body={body_name}, category={category}"
            )

        # Handle articulated objects with joints
        if category in articulation_categories and joints_map:
            joint_info = {}
            for model_joint_name, metadata_joint_name in joints_map.items():
                # Check if this joint has grasp files
                if has_joint_grasp_path(asset_id, metadata_joint_name):
                    # Get joint body name
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, model_joint_name)
                    if joint_id >= 0:
                        joint_body_id = model.joint(joint_id).bodyid[0]
                        joint_body_name = mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_BODY, joint_body_id
                        )
                        if joint_body_name:
                            joint_info[metadata_joint_name] = (model_joint_name, joint_body_name)
                            print(
                                f"Found joint: {object_name} -> asset_id={asset_id}, joint={metadata_joint_name} (model: {model_joint_name}), joint_body={joint_body_name}"
                            )

            if joint_info:
                jointed_objects[asset_id] = joint_info

    return pickup_objects, jointed_objects


def load_grasps_for_visualization(object_name, num_grasps=50, load_all_for_collision=False):
    """Load grasps for visualization, keeping as 4x4 transform matrices.

    Args:
        object_name: Asset ID of the object
        num_grasps: Number of grasps to return (after collision filtering if applicable)
        load_all_for_collision: If True, load all available grasps (for collision filtering)
    """
    try:
        if load_all_for_collision:
            # Load all available grasps for collision filtering (use a very large number)
            # load_pickup_grasps samples randomly, so we use a large number to get all
            grasp_transforms = load_pickup_grasps(object_name, num_grasps=10000)
        else:
            grasp_transforms = load_pickup_grasps(object_name, num_grasps=num_grasps)

        # grasp_transforms is a numpy array of shape (N, 4, 4) - transform matrices
        # If not loading all, limit to num_grasps if more were loaded
        if not load_all_for_collision and len(grasp_transforms) > num_grasps:
            grasp_transforms = grasp_transforms[:num_grasps]

        return grasp_transforms
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading grasps for {object_name}: {e}")
        return []


def load_grasps_for_joint_visualization(
    object_name, joint_name, num_grasps=50, load_all_for_collision=False
):
    """Load grasps for a joint, keeping as 4x4 transform matrices.

    Args:
        object_name: Asset ID of the object
        joint_name: Name of the joint
        num_grasps: Number of grasps to return (after collision filtering if applicable)
        load_all_for_collision: If True, load all available grasps (for collision filtering)
    """
    try:
        grasp_transforms = load_joint_grasps(object_name, joint_name, num_grasps=num_grasps)

        if len(grasp_transforms) == 0:
            return []

        # grasp_transforms is a numpy array of shape (N, 4, 4) - transform matrices
        # If not loading all, limit to num_grasps
        if not load_all_for_collision and len(grasp_transforms) > num_grasps:
            # Use furthest point sampling for better distribution
            indices = furthest_point_sampling(grasp_transforms, num_grasps)
            grasp_transforms = grasp_transforms[indices]

        return grasp_transforms
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Error loading grasps for {object_name} joint {joint_name}: {e}")
        return []


def visualize_all_objects_grasps(
    viewer,
    model,
    data,
    pickup_grasps,
    jointed_grasps,
    pickup_body_map,
    jointed_body_map,
    check_collision=False,
    collision_batch_size=10,
    num_grasps_per_object=50,
):
    """Visualize grasps for both pickup objects (blue) and jointed objects (green) using viewer overlay.

    Args:
        viewer: MuJoCo viewer
        model: MuJoCo model
        data: MuJoCo data
        pickup_grasps: Dict mapping asset_id to grasp transforms
        jointed_grasps: Dict mapping asset_id to joint_name -> grasp transforms
        pickup_body_map: Dict mapping asset_id to body name
        jointed_body_map: Dict mapping asset_id to joint_name -> (model_joint_name, joint_body_name)
        check_collision: If True, filter out colliding grasps and sample N from non-colliding
        collision_batch_size: Batch size for collision checking
        num_grasps_per_object: Number of grasps to visualize per object (after collision filtering)
    """
    # Reset the viewer's user scene geometry count
    viewer.user_scn.ngeom = 0

    all_grasp_poses_blue = []
    all_grasp_poses_green = []

    # Collect pickup object grasps (blue) - per object for collision filtering
    pickup_grasps_per_object = {}
    for asset_id, grasp_transforms in pickup_grasps.items():
        obj_body_name = pickup_body_map[asset_id]
        object_grasp_poses = []

        try:
            object_pos = data.body(obj_body_name).xpos.copy()
            object_quat = data.body(obj_body_name).xquat.copy()

            # Convert object pose to 4x4 matrix
            object_pose = pos_quat_to_pose_mat(object_pos, object_quat)

            for grasp_transform in grasp_transforms:
                # Transform grasp to world coordinates: object_pose @ grasp_transform
                grasp_pose_world = object_pose @ grasp_transform
                object_grasp_poses.append(grasp_pose_world)

            pickup_grasps_per_object[asset_id] = np.array(object_grasp_poses)

        except Exception as e:
            print(f"Error positioning grasps for {obj_body_name}: {e}")
            continue

    # Collect jointed object grasps (green) - per object/joint for collision filtering
    jointed_grasps_per_object = {}
    for asset_id, joint_grasps_dict in jointed_grasps.items():
        jointed_grasps_per_object[asset_id] = {}
        for joint_name, grasp_transforms in joint_grasps_dict.items():
            joint_body_name = jointed_body_map[asset_id][joint_name][1]  # Get joint body name
            joint_grasp_poses = []

            try:
                joint_body_pos = data.body(joint_body_name).xpos.copy()
                joint_body_quat = data.body(joint_body_name).xquat.copy()

                # Convert joint body pose to 4x4 matrix
                joint_body_pose = pos_quat_to_pose_mat(joint_body_pos, joint_body_quat)

                for grasp_transform in grasp_transforms:
                    # Transform grasp to world coordinates: joint_body_pose @ grasp_transform
                    grasp_pose_world = joint_body_pose @ grasp_transform
                    joint_grasp_poses.append(grasp_pose_world)

                jointed_grasps_per_object[asset_id][joint_name] = np.array(joint_grasp_poses)

            except Exception as e:
                print(f"Error positioning grasps for joint {joint_name} on {joint_body_name}: {e}")
                continue

    # Filter by collision if requested, then sample N per object from non-colliding
    if check_collision:
        # Filter pickup object grasps per object
        for asset_id, object_grasp_poses in pickup_grasps_per_object.items():
            if len(object_grasp_poses) > 0:
                try:
                    noncolliding_mask = get_noncolliding_grasp_mask(
                        model, data, object_grasp_poses, collision_batch_size
                    )
                    noncolliding_grasps = object_grasp_poses[noncolliding_mask]
                    # Sample N from non-colliding grasps for this object using FPS
                    if len(noncolliding_grasps) > num_grasps_per_object:
                        indices = furthest_point_sampling(
                            noncolliding_grasps, num_grasps_per_object
                        )
                        selected_grasps = noncolliding_grasps[indices]
                    else:
                        selected_grasps = noncolliding_grasps
                    all_grasp_poses_blue.extend(selected_grasps)
                    print(
                        f"Object {asset_id}: {len(selected_grasps)}/{len(object_grasp_poses)} non-colliding pickup grasps (requested {num_grasps_per_object})"
                    )
                except Exception as e:
                    print(f"Error in collision checking for pickup object {asset_id}: {e}")
                    # Fall back to showing N if collision check fails
                    if len(object_grasp_poses) > num_grasps_per_object:
                        indices = furthest_point_sampling(object_grasp_poses, num_grasps_per_object)
                        all_grasp_poses_blue.extend(object_grasp_poses[indices])
                    else:
                        all_grasp_poses_blue.extend(object_grasp_poses)

        # Filter jointed object grasps per object/joint
        for asset_id, joint_grasps_dict in jointed_grasps_per_object.items():
            for joint_name, joint_grasp_poses in joint_grasps_dict.items():
                if len(joint_grasp_poses) > 0:
                    try:
                        noncolliding_mask = get_noncolliding_grasp_mask(
                            model, data, joint_grasp_poses, collision_batch_size
                        )
                        noncolliding_grasps = joint_grasp_poses[noncolliding_mask]
                        # Sample N from non-colliding grasps for this joint using FPS
                        if len(noncolliding_grasps) > num_grasps_per_object:
                            indices = furthest_point_sampling(
                                noncolliding_grasps, num_grasps_per_object
                            )
                            selected_grasps = noncolliding_grasps[indices]
                        else:
                            selected_grasps = noncolliding_grasps
                        all_grasp_poses_green.extend(selected_grasps)
                        print(
                            f"Joint {joint_name} on {asset_id}: {len(selected_grasps)}/{len(joint_grasp_poses)} non-colliding grasps (requested {num_grasps_per_object})"
                        )
                    except Exception as e:
                        print(
                            f"Error in collision checking for joint {joint_name} on {asset_id}: {e}"
                        )
                        # Fall back to showing N if collision check fails
                        if len(joint_grasp_poses) > num_grasps_per_object:
                            indices = furthest_point_sampling(
                                joint_grasp_poses, num_grasps_per_object
                            )
                            all_grasp_poses_green.extend(joint_grasp_poses[indices])
                        else:
                            all_grasp_poses_green.extend(joint_grasp_poses)
    else:
        # No collision checking - just sample N per object using FPS
        for asset_id, object_grasp_poses in pickup_grasps_per_object.items():
            if len(object_grasp_poses) > num_grasps_per_object:
                indices = furthest_point_sampling(object_grasp_poses, num_grasps_per_object)
                all_grasp_poses_blue.extend(object_grasp_poses[indices])
            else:
                all_grasp_poses_blue.extend(object_grasp_poses)

        for asset_id, joint_grasps_dict in jointed_grasps_per_object.items():
            for joint_name, joint_grasp_poses in joint_grasps_dict.items():
                if len(joint_grasp_poses) > num_grasps_per_object:
                    indices = furthest_point_sampling(joint_grasp_poses, num_grasps_per_object)
                    all_grasp_poses_green.extend(joint_grasp_poses[indices])
                else:
                    all_grasp_poses_green.extend(joint_grasp_poses)

    # Show all grasps using viewer overlay
    if len(all_grasp_poses_blue) > 0:
        if isinstance(all_grasp_poses_blue, list):
            all_grasp_poses_blue = np.array(all_grasp_poses_blue)
        _show_grasp_poses(
            viewer,
            all_grasp_poses_blue,
            grasp_width,
            grasp_length,
            grasp_height,
            grasp_base_pos,
            color=(0, 0, 1, 1),
        )  # Blue

    if len(all_grasp_poses_green) > 0:
        if isinstance(all_grasp_poses_green, list):
            all_grasp_poses_green = np.array(all_grasp_poses_green)
        _show_grasp_poses(
            viewer,
            all_grasp_poses_green,
            grasp_width,
            grasp_length,
            grasp_height,
            grasp_base_pos,
            color=(0, 1, 0, 1),
        )  # Green


def get_scene_path(dataset_name: str, scene_index: int, split: str = "train") -> str:
    """Get scene path from dataset name and scene index.

    Automatically installs the scene if it's not found locally.
    Prefers "physics" variant if available, otherwise "base", otherwise first available.
    """
    scenes = get_scenes(dataset_name, split=split)

    # Check if index exists in the map
    if scene_index not in scenes[split]:
        print(
            f"Scene index {scene_index} not in index map for {dataset_name} (split: {split}), attempting to install..."
        )

        # Determine scene source based on dataset
        if dataset_name == "ithor":
            scene_source = "ithor"
        elif dataset_name == "procthor-10k":
            scene_source = f"procthor-10k-{split}"
        elif dataset_name == "procthor-objaverse-debug":
            scene_source = "procthor-objaverse-debug"
        elif dataset_name == "procthor-100k-debug":
            scene_source = "procthor-objaverse-debug"  # Same source
        else:
            raise ValueError(
                f"Unknown dataset {dataset_name}, cannot determine scene source for installation"
            )

        # Install the archive containing this scene index via the current API.
        # archives_with_number / install_scenes were removed in the resource
        # manager rework; install_scene_from_source_index does the equivalent
        # index_lookup + install_packages.
        try:
            install_scene_from_source_index(scene_source, scene_index)

            # Clear cache and refresh the scene index map after installation
            from molmo_spaces.molmo_spaces_constants import _DATASET_INDEX_CACHE

            if dataset_name in _DATASET_INDEX_CACHE:
                del _DATASET_INDEX_CACHE[dataset_name]

            # Refresh the scene index map
            scenes = get_scenes(dataset_name, split=split)
        except Exception as e:
            print(f"Error: Failed to install scene by index: {e}")
            import traceback

            traceback.print_exc()
            raise

    # Now get the scene path (should exist after installation)
    scene_path_or_dict = scenes[split][scene_index]

    if isinstance(scene_path_or_dict, dict):
        # Prefer "physics" variant, then "base", then first available
        scene_path = (
            scene_path_or_dict.get("physics")
            or scene_path_or_dict.get("base")
            or list(scene_path_or_dict.values())[0]
        )
    else:
        scene_path = scene_path_or_dict

    if scene_path is None:
        raise ValueError(
            f"Scene path is None for {dataset_name} scene_index {scene_index} (split: {split})"
        )

    scene_path = Path(scene_path)

    # If scene file doesn't exist, install it using the same pattern as fetch_scene
    if not scene_path.exists():
        print(f"Scene file not found at {scene_path}, attempting to install...")
        # PR #97/#99 reworked the ResourceManager API (archives_for_paths /
        # install_scenes were removed). install_scene_from_path resolves the
        # archive for a scene path and installs it via the current API.
        install_scene_from_path(scene_path)

    # Verify the file exists after installation
    if not scene_path.exists():
        raise ValueError(
            f"Scene file does not exist at {scene_path} even after installation attempt. "
            f"Please check if the scene index {scene_index} is valid for {dataset_name} (split: {split})"
        )

    return str(scene_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize object grasps in MiJoCo scene.")
    parser.add_argument(
        "--scene_index",
        type=int,
        required=True,
        help="Index of the scene to visualize",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'ithor', 'procthor-10k')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "debug"],
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--number_of_grasp_per_object",
        type=int,
        default=50,
        help="Number of grasps per object (default: 50)",
    )
    parser.add_argument(
        "--check_collision",
        action="store_true",
        help="Filter out colliding grasps before visualization",
    )
    parser.add_argument(
        "--collision_batch_size",
        type=int,
        default=10,
        help="Batch size for collision checking (default: 10)",
    )
    args = parser.parse_args()

    scene_index = args.scene_index
    dataset_name = args.dataset_name
    split = args.split
    number_of_grasp_per_object = args.number_of_grasp_per_object

    print(f"Loading scene {scene_index} from dataset '{dataset_name}' (split: {split})...")

    try:
        scene_xml_path = get_scene_path(dataset_name, scene_index, split=split)
        print(f"Scene XML path: {scene_xml_path}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Load model first
    model = mujoco.MjModel.from_xml_path(scene_xml_path)
    data = mujoco.MjData(model)

    # Load scene metadata
    print("Loading scene metadata...")
    scene_metadata = get_scene_metadata(scene_xml_path)

    # Extract objects from metadata filtered by categories
    print("Extracting objects from scene metadata (pickup and articulation types)...")
    pickup_objects, jointed_objects = extract_objects_from_metadata(model, scene_metadata)

    if not pickup_objects and not jointed_objects:
        print("No objects with grasps found in this scene")
        exit(0)

    print(f"Found {len(pickup_objects)} pickup objects with grasp data:")
    for asset_id, body_name in pickup_objects.items():
        print(f"  {asset_id} -> {body_name}")

    print(f"Found {len(jointed_objects)} jointed objects with grasp data:")
    for asset_id, joint_info in jointed_objects.items():
        print(f"  {asset_id} with {len(joint_info)} joints: {list(joint_info.keys())}")

    # Get grasp geometry parameters from policy config defaults
    # Create a temporary config just to get default values, then extract them
    _temp_config = ObjectManipulationPlannerPolicyConfig()
    grasp_width = _temp_config.grasp_width
    grasp_length = _temp_config.grasp_length
    grasp_height = 0.006  # Visualization uses 0.006 instead of 0.01
    grasp_base_pos = np.array([0.0, 0.0, -0.06])  # Visualization uses -0.06 instead of -0.04
    del _temp_config  # Clean up to avoid serialization issues

    # Load model from scene using MjSpec (needed for adding collision bodies)
    spec = MjSpec.from_file(scene_xml_path)

    # Add collision bodies if collision checking is enabled
    if args.check_collision:
        print("Adding grasp collision bodies for collision checking...")
        # Calculate total number of grasps we might need to check
        total_grasps = 0
        for asset_id in pickup_objects.keys():
            total_grasps += number_of_grasp_per_object
        for asset_id, joint_info in jointed_objects.items():
            total_grasps += len(joint_info) * number_of_grasp_per_object

        # Use same parameters as visual geometries
        add_grasp_collision_bodies(
            spec,
            args.collision_batch_size,  # Only need as many as batch size
            grasp_width,
            grasp_length,
            grasp_height,
            grasp_base_pos,
        )

    model = spec.compile()
    data = mujoco.MjData(model)

    pickup_grasps = {}
    jointed_grasps = {}

    # Load grasps for pickup objects
    # If collision checking is enabled, load all available grasps for filtering
    available_pickup_objects = []
    for asset_id, body_name in pickup_objects.items():
        grasps = load_grasps_for_visualization(
            asset_id, number_of_grasp_per_object, load_all_for_collision=args.check_collision
        )
        if grasps is not None and len(grasps) > 0:
            pickup_grasps[asset_id] = grasps
            available_pickup_objects.append(asset_id)
            print(f"Loaded {len(grasps)} grasps for pickup object {asset_id} (body: {body_name})")

    # Load grasps for jointed objects
    # If collision checking is enabled, load all available grasps for filtering
    available_jointed_objects = []
    for asset_id, joint_info in jointed_objects.items():
        joint_grasps_dict = {}
        for joint_name, (model_joint_name, joint_body_name) in joint_info.items():
            grasps = load_grasps_for_joint_visualization(
                asset_id,
                joint_name,
                number_of_grasp_per_object,
                load_all_for_collision=args.check_collision,
            )
            if grasps is not None and len(grasps) > 0:
                joint_grasps_dict[joint_name] = grasps
                print(
                    f"Loaded {len(grasps)} grasps for joint {joint_name} on {asset_id} (joint_body: {joint_body_name})"
                )

        if joint_grasps_dict:
            jointed_grasps[asset_id] = joint_grasps_dict
            available_jointed_objects.append(asset_id)

    if not available_pickup_objects and not available_jointed_objects:
        print("No objects with grasps found")
        exit(0)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Showing grasps for all available objects (pickup and jointed) simultaneously")
        print("Close the viewer window to exit.")

        while viewer.is_running():
            mujoco.mj_step(model, data)

            visualize_all_objects_grasps(
                viewer,
                model,
                data,
                pickup_grasps,
                jointed_grasps,
                pickup_objects,
                jointed_objects,
                check_collision=args.check_collision,
                collision_batch_size=args.collision_batch_size,
                num_grasps_per_object=number_of_grasp_per_object,
            )

            viewer.sync()

    print("Viewer closed. Exiting.")
    exit(0)
