"""
Test grasps for all objects in a scene.
Can test only static (pickup) objects, only jointed (articulated) objects, or both.
Can use a passive viewer to visualize the scene.

The script uses a GraspLoader abstraction to load grasps, making it easy to swap
between different grasp sources (files, prediction models, etc.).

To use a custom grasp loader:
    from molmo_spaces.scripts.grasps.grasp_loaders import GraspLoader

    class MyCustomGraspLoader(GraspLoader):
        def load_grasps_for_object(self, object_name, num_grasps=1000):
            # Your implementation
            return "droid", grasps_array

        def load_grasps_for_joint(self, object_name, joint_name, num_grasps=1000):
            # Your implementation
            return "droid", grasps_array

    # Then pass it to test_grasps_for_scene:
    custom_loader = MyCustomGraspLoader()
    metrics = test_grasps_for_scene(..., grasp_loader=custom_loader)

Example:
    mjpython scripts/grasps/run_grasps_test.py --scene_dataset ithor --house_ind 2 --task_type pick --use_passive_viewer
    mjpython scripts/grasps/run_grasps_test.py --scene_dataset ithor --house_ind 2 --task_type pick --use_passive_viewer
    mjpython scripts/grasps/run_grasps_test.py --scene_dataset ithor --house_ind 410 --task_type open_close --use_passive_viewer

    mjpython scripts/grasps/run_grasps_test.py --scene_dataset procthor-objaverse-debug --house_ind 2 --task_type pick --use_passive_viewer
    mjpython scripts/grasps/run_grasps_test.py --scene_dataset procthor-objaverse-debug --house_ind 2 --task_type open_close --use_passive_viewer


    python scripts/grasps/run_grasps_test.py --scene_dataset ithor --task_type pick

"""

import logging
import datetime
import argparse
import json
import gc
import sys
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import mujoco

from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.base_open_task_configs import OpeningBaseConfig
from molmo_spaces.configs.robot_configs import FloatingRobotiq2f85RobotConfig
from molmo_spaces.configs.policy_configs import (
    PickPlannerPolicyConfig,
    OpenClosePlannerPolicyConfig,
)
from molmo_spaces.configs.camera_configs import FrankaRobotiq2f85CameraSystem

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.utils.profiler_utils import Profiler
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.env.data_views import (
    MlSpacesObject,
    MlSpacesArticulationObject,
)
from molmo_spaces.utils.lazy_loading_utils import (
    install_scene_with_objects_and_grasps_from_path,
    install_scene_from_path,
)
from molmo_spaces.env.arena.scene_tweaks import (
    is_body_within_any_site,
    is_body_within_site_in_freespace,
)
from molmo_spaces.utils.grasp_sample import (
    get_noncolliding_grasp_mask,
    has_grasp_folder,
    has_joint_grasp_file,
)
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    GripperAction,
    TCPMoveSegment,
    TCPMoveSequence,
)

# Import grasp loaders - use relative import since we're in the same package
try:
    from .grasp_loaders import GraspLoader, FileBasedGraspLoader
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from grasp_loaders import GraspLoader, FileBasedGraspLoader
from molmo_spaces.utils.constants.object_constants import (
    ALL_PICKUP_TYPES_THOR,
    EXTENDED_ARTICULATION_TYPES_THOR,
)
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_7d
from molmo_spaces.utils.asset_names import get_thor_name
from scipy.spatial.transform import Rotation as R

# Import from view_grasps for object extraction - need to handle import path
try:
    from scripts.grasps.view_grasps import extract_objects_from_metadata
except ImportError:
    # Fallback: define it here if import fails
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from view_grasps import extract_objects_from_metadata

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable types to JSON-compatible types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def get_object_openness_percentage(data, object_name: str, joint_name: str) -> float:
    """Get the openness percentage of an articulated object (0.0 = closed, 1.0 = fully open).

    Uses MlSpacesArticulationObject to get joint position and range.
    """
    try:
        from molmo_spaces.env.data_views import MlSpacesArticulationObject

        # Create articulation object
        articulation_obj = MlSpacesArticulationObject(object_name=object_name, data=data)

        # Find joint index by name
        if joint_name not in articulation_obj.joint_names:
            log.warning(f"Joint {joint_name} not found in {object_name}")
            return 0.0

        joint_index = articulation_obj.joint_names.index(joint_name)

        # Get current joint position and range
        current_joint_state = articulation_obj.get_joint_position(joint_index)
        joint_range = articulation_obj.get_joint_range(joint_index)

        # Calculate percentage (similar to OpeningTask.get_reward())
        # Closed position is always at 0, so distance from 0 is the opening amount
        # abs() handles both positive [0, 1.57] and negative [-1.57, 0] ranges
        joint_range_float = abs(joint_range[1] - joint_range[0])
        if joint_range_float == 0:
            return 0.0

        percent_open = abs(current_joint_state) / joint_range_float
        return float(np.clip(percent_open, 0.0, 1.0))
    except Exception as e:
        log.warning(f"Error getting openness for {object_name}/{joint_name}: {e}")
        return 0.0


def check_robot_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_view,
    target_object_name: str = None,
) -> tuple[bool, list[str]]:
    """
    Check if robot is in collision with objects in the scene.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        robot_view: Robot view to check collisions for
        target_object_name: Name of target object to exclude from collision check

    Returns:
        Tuple of (has_collision, collision_details)
        - has_collision: True if robot is colliding with non-target objects
        - collision_details: List of strings describing collisions
    """
    has_collision = False
    collision_details = []

    # Get robot root body ID
    robot_root_id = robot_view.base.root_body_id

    # Get target object body ID if provided
    target_body_id = None
    if target_object_name:
        try:
            # Try to get object from environment
            # This might not always work, so we'll catch exceptions
            pass  # We'll get this from the environment if available
        except:
            pass

    # Check all contacts
    for i in range(data.ncon):
        c = data.contact[i]

        # Skip if contact is excluded
        if c.exclude != 0:
            continue

        # Skip if contact distance is positive (not actually in contact)
        if c.dist > 0:
            continue

        # Get body IDs for the two geoms in contact
        geom1_body = model.geom_bodyid[c.geom1]
        geom2_body = model.geom_bodyid[c.geom2]

        # Get root body IDs
        root1 = model.body_rootid[geom1_body]
        root2 = model.body_rootid[geom2_body]

        # Check if either body belongs to the robot
        if root1 == robot_root_id or root2 == robot_root_id:
            # Skip self-collisions (robot parts colliding with each other)
            if root1 == robot_root_id and root2 == robot_root_id:
                continue

            # Get body names for logging (may be None if body has no name)
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, geom1_body)
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, geom2_body)

            # Skip if this is the target object (expected contact during grasping)
            if target_object_name:
                if (body1_name and body1_name == target_object_name) or (
                    body2_name and body2_name == target_object_name
                ):
                    continue

            # Skip floor contacts (expected)
            floor_names = ["floor", "Floor", "ground", "Ground"]
            if (body1_name and any(floor_name in body1_name for floor_name in floor_names)) or (
                body2_name and any(floor_name in body2_name for floor_name in floor_names)
            ):
                continue

            # This is a collision with a non-target object
            has_collision = True
            # Determine which body is the non-robot body
            if root1 == robot_root_id:
                other_body = body2_name if body2_name else f"body_{geom2_body}"
            else:
                other_body = body1_name if body1_name else f"body_{geom1_body}"
            collision_details.append(f"Robot colliding with {other_body}")

    return has_collision, collision_details


def place_robot_near_grasp(
    robot_view,
    grasp_pose: np.ndarray,
    offset_distance: float = 0.15,
    offset_direction: np.ndarray = None,
) -> None:
    """Place the floating robot at an offset from the grasp pose.

    Args:
        robot_view: Robot view to position
        grasp_pose: 4x4 transformation matrix of the TCP/grasp pose in world frame
        offset_distance: Distance to offset from grasp pose (default 0.15m)
        offset_direction: Direction to offset (default: negative Z direction of grasp)
    """
    if offset_direction is None:
        # Default: offset in negative Z direction of grasp (backward from grasp)
        offset_direction = -grasp_pose[:3, 2]
        offset_direction = offset_direction / np.linalg.norm(offset_direction)

    # Calculate offset TCP position along the offset direction
    offset_tcp_pos = grasp_pose[:3, 3] + offset_distance * offset_direction

    # Create offset TCP pose (same orientation as grasp, offset position)
    offset_tcp_pose = grasp_pose.copy()
    offset_tcp_pose[:3, 3] = offset_tcp_pos

    # Transform from TCP pose to base pose
    # TCP pose in world = base pose @ (base to TCP)
    # Therefore: base pose = TCP pose @ inv(base to TCP)
    try:
        gripper_mg = robot_view.get_move_group(robot_view.get_gripper_movegroup_ids()[0])
        base_to_tcp = gripper_mg.leaf_frame_to_robot  # Transformation from base to TCP

        # Check if matrix is valid (rotation part should have determinant close to 1)
        rot_part = base_to_tcp[:3, :3]
        det = np.linalg.det(rot_part)
        if abs(det) < 1e-6:
            raise ValueError(f"Singular transformation matrix (det={det})")
        if abs(abs(det) - 1.0) > 0.1:  # Rotation matrix should have det = ±1
            raise ValueError(f"Invalid rotation matrix (det={det}, expected ±1)")

        tcp_to_base = np.linalg.inv(base_to_tcp)  # Transformation from TCP to base

        # Compute base pose: offset_tcp_pose @ tcp_to_base
        base_pose = offset_tcp_pose @ tcp_to_base

        # Set robot base pose
        robot_view.base.pose = base_pose
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fallback: if transformation fails, use grasp pose directly with offset
        # This assumes base and TCP have same orientation (which may not be true, but better than crashing)
        log.warning(f"Failed to compute base-to-TCP transformation: {e}. Using direct offset.")
        base_pose = offset_tcp_pose.copy()
        robot_view.base.pose = base_pose
        raise  # Re-raise to be caught by outer exception handler


class GraspTestRolloutRunner(ParallelRolloutRunner):
    @staticmethod
    def run_single_rollout(
        episode_seed: int,
        task: BaseMujocoTask,
        policy: Any,
        profiler: Profiler | None = None,
        viewer=None,
        shutdown_event=None,
        use_passive_viewer: bool = False,
        save_failed_video_dir: Optional[Path] = None,
        grasp_idx: int = 0,
        object_name: str = "",
        joint_name: str = "",
    ):
        """Run a single rollout and return (success, failure_modes)."""
        # Register policy with task (needed for sensors and phase tracking)
        task.register_policy(policy)

        observation, _info = task.reset()

        # Collect observations for video saving (always collect, will save if directory is provided)
        # observation is a list[dict] (one per environment), extract first element
        observations_list = []
        if isinstance(observation, list) and len(observation) > 0:
            obs_dict = observation[0]
            if isinstance(obs_dict, dict):
                observations_list.append(obs_dict)
            else:
                log.warning(f"Observation[0] is not a dict: {type(obs_dict)}")
        elif isinstance(observation, dict):
            observations_list.append(observation)
        else:
            log.warning(f"Unexpected observation type: {type(observation)}")

        # Use provided viewer if available, otherwise create one if requested
        if viewer is not None:
            task.viewer = viewer
        elif use_passive_viewer:
            # Only create viewer if one wasn't provided
            viewer = mujoco.viewer.launch_passive(
                task.env.mj_datas[task.env.current_batch_index].model,
                task.env.mj_datas[task.env.current_batch_index],
                key_callback=getattr(policy, "get_key_callback", lambda: None)(),
            )
            viewer.opt.sitegroup[0] = False
            task.viewer = viewer

        if viewer is not None:
            viewer.sync()

        success = False
        policy.task = task

        # Enable sleep mode before executing the rollout
        try:
            task.env.current_model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
        except AttributeError:
            log.warning("Not setting enable sleep")

        while not task.is_done():
            if shutdown_event is not None and shutdown_event.is_set():
                if viewer is not None:
                    viewer.close()
                return False

            # Step with policy
            action_cmd = policy.get_action(observation)
            if (
                isinstance(action_cmd, dict)
                and hasattr(task, "_env")
                and hasattr(task._env, "n_batch")
            ):
                actions_for_task = [action_cmd] * task._env.n_batch
            else:
                actions_for_task = action_cmd
            observation, reward, terminal, truncated, infos = task.step(actions_for_task)

            # Collect observation for video saving (always collect, will save if directory is provided)
            # observation is a list[dict] (one per environment), extract first element
            if isinstance(observation, list) and len(observation) > 0:
                obs_dict = observation[0]
                if (
                    isinstance(obs_dict, dict) and observations_list
                ):  # Only append if we have a valid list
                    observations_list.append(obs_dict)
            elif isinstance(observation, dict) and observations_list:
                observations_list.append(observation)

            # Check success using judge_success() during execution
            if hasattr(task, "judge_success"):
                success_result = task.judge_success()
                # Convert numpy bool/array to Python bool if needed
                if isinstance(success_result, np.ndarray):
                    success_result = bool(
                        success_result.item() if success_result.size == 1 else success_result
                    )
                elif isinstance(success_result, np.bool_):
                    success_result = bool(success_result)
                else:
                    success_result = bool(success_result)
                if success_result:
                    success = True
                    log.debug(f"Task succeeded during execution")
                    break

            if viewer is not None:
                viewer.sync()

        # Disable sleep mode after the rollout
        try:
            task.env.current_model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        except AttributeError:
            log.warning("Not setting sleep")

        # Check success using judge_success() function
        if hasattr(task, "judge_success"):
            success = task.judge_success()
        else:
            log.warning("Task does not have judge_success() method, using default success=False")

        # Detect failure modes
        failure_modes = []

        if not success:
            # Check 1: Robot placement / camera visibility
            try:
                if hasattr(task, "env") and hasattr(
                    task.env, "check_camera_visibility_constraints"
                ):
                    all_satisfied, visibility_results = (
                        task.env.check_camera_visibility_constraints()
                    )
                    if not all_satisfied:
                        failure_modes.append("robot_placement_camera_not_visible")
                        log.warning(f"Camera visibility check failed: {visibility_results}")
            except Exception as e:
                log.debug(f"Could not check camera visibility: {e}")

            # Check 2: Not lifted enough (low reward)
            try:
                if hasattr(task, "get_reward"):
                    final_reward = task.get_reward()
                    if isinstance(final_reward, np.ndarray):
                        final_reward = final_reward[0] if final_reward.size > 0 else 0.0
                    # For pick tasks, reward is based on lift height (typically > 0.03 for success)
                    if final_reward < 0.03:
                        failure_modes.append("not_lifted_enough")
                        log.warning(
                            f"Object not lifted enough: reward={final_reward:.4f} (threshold: 0.03)"
                        )
            except Exception as e:
                log.debug(f"Could not check reward: {e}")

            # Check 3: Not grasped target object
            try:
                model = task.env.current_model
                data = task.env.current_data
                robot_view = task.env.current_robot.robot_view

                # Get gripper finger geom IDs
                gripper_group = None
                for move_group in robot_view.get_move_groups():
                    if hasattr(move_group, "_finger_1_geom_id") and hasattr(
                        move_group, "_finger_2_geom_id"
                    ):
                        gripper_group = move_group
                        break

                if gripper_group is not None:
                    finger_1_geom_id = gripper_group._finger_1_geom_id
                    finger_2_geom_id = gripper_group._finger_2_geom_id
                    finger_geom_ids = [finger_1_geom_id, finger_2_geom_id]

                    # Get target object body ID
                    target_object_name = None
                    if hasattr(task, "config") and hasattr(task.config, "task_config"):
                        target_object_name = task.config.task_config.pickup_obj_name

                    if target_object_name:
                        try:
                            target_obj = MlSpacesObject(object_name=target_object_name, data=data)
                            target_body_id = target_obj.body_id

                            # Check if gripper is in contact with target object
                            grasped_target = False
                            for i in range(data.ncon):
                                c = data.contact[i]
                                geom1_body = model.geom_bodyid[c.geom1]
                                geom2_body = model.geom_bodyid[c.geom2]

                                # Check if contact involves gripper finger and target object
                                if (
                                    c.geom1 in finger_geom_ids and geom2_body == target_body_id
                                ) or (c.geom2 in finger_geom_ids and geom1_body == target_body_id):
                                    grasped_target = True
                                    break

                            if not grasped_target:
                                failure_modes.append("not_grasped_target_object")
                                log.warning(
                                    f"Gripper not in contact with target object: {target_object_name}"
                                )
                        except Exception as e:
                            log.debug(f"Could not check grasp contact: {e}")
            except Exception as e:
                log.debug(f"Could not check grasp contact: {e}")

        # Log failure modes
        if failure_modes:
            log.warning(f"❌ FAILURE modes detected: {', '.join(failure_modes)}")
        elif not success:
            log.warning("❌ FAILURE: Task failed but no specific failure mode detected")

        # Log task completion status with obvious emoji
        if success:
            log.info(f"✅ SUCCESS (judge_success): Task completed successfully")
        else:
            log.error(
                f"❌❌❌ FAILED (judge_success): Task failed"
                + (f" | Failure modes: {', '.join(failure_modes)}" if failure_modes else "")
            )

        # Save video for failed cases using camera sensor observations (always save if enabled)
        if not success and save_failed_video_dir is not None and observations_list:
            try:
                from molmo_spaces.utils.save_utils import save_videos_from_raw_observations
                import os

                # Calculate FPS from task config
                # Clamp FPS to reasonable range (5-120) to avoid 0-second videos
                policy_dt_ms = task.config.policy_dt_ms
                fps = 1000.0 / policy_dt_ms

                log.debug(f"Calculated FPS: {fps:.2f} (policy_dt_ms: {policy_dt_ms})")

                # Create object-specific subdirectory: {timestamp}/{object_name}/
                if object_name:
                    object_video_dir = Path(save_failed_video_dir) / object_name
                    if joint_name:
                        # For articulated objects, include joint name in path
                        object_video_dir = object_video_dir / joint_name
                else:
                    object_video_dir = Path(save_failed_video_dir)

                # Create save directory
                os.makedirs(object_video_dir, exist_ok=True)
                log.info(f"📁 Created/verified video directory: {object_video_dir.absolute()}")

                # Filter observations to only include dicts with string keys (camera sensors)
                # Some sensors return dicts as values (e.g., RobotStateSensor), skip those
                filtered_observations = []
                for obs in observations_list:
                    if isinstance(obs, dict):
                        # Create a filtered dict with only camera-like sensors (string keys, non-dict values)
                        filtered_obs = {}
                        for key, value in obs.items():
                            if isinstance(key, str) and not isinstance(value, dict):
                                filtered_obs[key] = value
                        if filtered_obs:
                            filtered_observations.append(filtered_obs)

                if not filtered_observations:
                    log.warning(
                        f"No valid camera observations found to save video (had {len(observations_list)} observations)"
                    )
                else:
                    # Get sensor suite
                    sensor_suite = None
                    if hasattr(task, "_sensor_suite"):
                        sensor_suite = task._sensor_suite
                    elif hasattr(task, "sensor_suite"):
                        sensor_suite = task.sensor_suite

                    # Save videos from observations
                    save_videos_from_raw_observations(
                        observations_list=filtered_observations,
                        save_dir=str(object_video_dir),
                        fps=fps,
                        episode_idx=grasp_idx,
                        save_file_suffix="_failed",
                        sensor_suite=sensor_suite,
                    )
                log.info(f"Saved failed grasp video to {object_video_dir}")
            except Exception as e:
                log.warning(f"Failed to save video for failed grasp: {e}")

        # Don't close viewer if we created it - keep it open for next grasp
        # Only close at the very end when all grasps are tested
        # if use_passive_viewer and viewer is not None:
        #     viewer.close()

        return success, failure_modes


def test_grasps_for_scene(
    scene_path: str,
    scene_metadata: dict,
    offset_distance: float = 0.15,
    max_grasps_per_object: int = None,
    collision_batch_size: int = 10,
    task_horizon: int = 120,
    use_passive_viewer: bool = False,
    task_type: str = "both",
    grasp_loader: Optional[GraspLoader] = None,
    save_failed_videos_dir: Optional[Path] = None,
    save_metrics_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Test grasps for all objects in a scene.

    Returns:
        Dictionary with metrics:
        - total_objects: Number of objects tested
        - total_grasps_tested: Total number of grasps tested
        - total_noncolliding_grasps: Total number of non-colliding grasps found
        - total_successful_grasps: Total number of successful grasps
        - success_rate: Overall success rate
        - per_object_metrics: Detailed metrics per object
    """
    from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
    from molmo_spaces.tasks.opening_task_samplers import OpenTaskSampler
    from mujoco import MjSpec
    from molmo_spaces.utils.grasp_sample import add_grasp_collision_bodies
    from molmo_spaces.configs.policy_configs import ObjectManipulationPlannerPolicyConfig

    # Get grasp geometry parameters from policy config defaults
    # Create a temporary config just to get default values, then extract them
    _temp_config = ObjectManipulationPlannerPolicyConfig()
    grasp_width = _temp_config.grasp_width
    grasp_length = _temp_config.grasp_length
    grasp_height = _temp_config.grasp_height
    grasp_base_pos = np.array(_temp_config.grasp_base_pos)
    del _temp_config  # Clean up to avoid serialization issues

    # Load scene to get model for collision checking
    spec = MjSpec.from_file(scene_path)

    # Add grasp collision bodies for collision checking
    # Only need as many as batch_size since we check in batches
    add_grasp_collision_bodies(
        spec,
        collision_batch_size,  # Only need as many as batch size
        grasp_width,
        grasp_length,
        grasp_height,
        grasp_base_pos,
    )

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Initialize grasp loader (default to file-based if not provided)
    if grasp_loader is None:
        grasp_loader = FileBasedGraspLoader()

    # Extract objects with grasps
    pickup_objects, jointed_objects = extract_objects_from_metadata(model, scene_metadata)

    log.info(
        f"Found {len(pickup_objects)} pickup objects and {len(jointed_objects)} articulated objects with grasps"
    )

    # Filter based on task_type
    if task_type == "pick":
        jointed_objects = {}  # Skip articulated objects
        log.info("Testing only static (pickup) objects")
    elif task_type == "open_close":
        pickup_objects = {}  # Skip pickup objects
        log.info("Testing only jointed (articulated) objects")
    elif task_type == "both":
        log.info("Testing both static and jointed objects")
    else:
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'pick', 'open_close', or 'both'")

    # Initialize metrics structure
    metrics = {
        "scene_path": str(scene_path),
        "scene_metadata_file": None,  # Will be set if available
        "timestamp": datetime.datetime.now().isoformat(),
        "total_objects": 0,
        "total_grasps_tested": 0,
        "total_noncolliding_grasps": 0,
        "total_successful_grasps": 0,
        "success_rate": 0.0,
        "per_object_metrics": [],
    }

    # Helper function to save metrics to JSON
    def save_metrics_to_json():
        """Save current metrics to JSON file."""
        if save_metrics_json_path is None:
            return
        try:
            serializable_metrics = convert_to_json_serializable(metrics)

            # Ensure directory exists
            save_metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
            log.info(f"📁 Created/verified metrics directory: {save_metrics_json_path.parent}")

            # Write to file
            with open(save_metrics_json_path, "w") as f:
                json.dump(serializable_metrics, f, indent=2)

            log.info(f"💾 Saved metrics JSON to: {save_metrics_json_path.absolute()}")
            log.debug(f"Metrics file size: {save_metrics_json_path.stat().st_size} bytes")
        except Exception as e:
            log.error(f"❌ Failed to save metrics to JSON: {e}", exc_info=True)

    # Test pickup objects
    for asset_id, body_name in pickup_objects.items():
        log.info(f"Testing pickup object: {asset_id} (body: {body_name})")

        # Check if object is in PICKABLE types or is an Objaverse object
        object_data = scene_metadata.get("objects", {}).get(body_name, {})
        category = object_data.get("category", "").lower()
        is_pickable = category in [t.lower() for t in ALL_PICKUP_TYPES_THOR]
        is_objaverse = "obja" in category

        if not (is_pickable or is_objaverse):
            log.info(f"Skipping {asset_id} - not in PICKABLE types and not Objaverse")
            continue

        # Load grasps using the provided loader
        try:
            gripper, grasps = grasp_loader.load_grasps_for_object(asset_id, num_grasps=1000)
        except Exception as e:
            log.warning(f"Failed to load grasps for {asset_id}: {e}")
            continue

        if len(grasps) == 0:
            log.warning(f"No grasps found for {asset_id}")
            continue

        # Get object pose
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            log.warning(f"Body {body_name} not found in model")
            continue

        # Ensure forward kinematics are computed before accessing quaternions
        mujoco.mj_forward(model, data)

        # Check if object is inside an enclosed site (e.g., inside a drawer or cabinet)
        is_within_site, site_id = is_body_within_any_site(model, data, body_id)
        if is_within_site:
            in_free_space, _, _ = is_body_within_site_in_freespace(site_id, body_id, model, data)
            if not in_free_space:
                log.info(
                    f"Skipping {asset_id} (body: {body_name}) - object is inside an enclosed site"
                )
                continue

        object_pos = data.xpos[body_id]
        object_quat = data.xquat[body_id].copy()
        object_pose = pos_quat_to_pose_mat(object_pos, object_quat)

        # Transform grasps to world frame
        GRIP_BASE_TCP = np.eye(4)  # For droid gripper
        grasp_poses_world = object_pose @ grasps @ GRIP_BASE_TCP

        # Filter non-colliding grasps
        noncolliding_mask = get_noncolliding_grasp_mask(
            model, data, grasp_poses_world, collision_batch_size
        )
        noncolliding_grasps = grasp_poses_world[noncolliding_mask]

        log.info(
            f"Found {len(noncolliding_grasps)} non-colliding grasps out of {len(grasps)} total"
        )

        if len(noncolliding_grasps) == 0:
            log.warning(f"No non-colliding grasps for {asset_id}")
            continue

        # Test up to max_grasps_per_object (or all if None)
        if max_grasps_per_object is None:
            num_to_test = len(noncolliding_grasps)
            test_grasps = noncolliding_grasps
        else:
            num_to_test = min(len(noncolliding_grasps), max_grasps_per_object)
            test_grasps = noncolliding_grasps[:num_to_test]

        object_metrics = {
            "object_name": asset_id,
            "object_type": "pickup",
            "body_name": body_name,
            "total_grasps": len(grasps),
            "noncolliding_grasps": len(noncolliding_grasps),
            "grasps_attempted": num_to_test,  # Number of grasps we intended to test
            "grasps_tested": 0,  # Actual number of grasps tested (excluding skipped due to collision)
            "successful_grasps": 0,
            "success_rate": 0.0,
            "total_time_seconds": 0.0,
            "avg_time_per_grasp_seconds": 0.0,
            "grasp_failure_modes": [],
            "skipped_collision": 0,
        }

        # Create profiler for this object
        object_profiler = Profiler(log_realtime=False)

        # Create task config once for this object
        datagen_cfg = PickBaseConfig()
        datagen_cfg.policy_config = PickPlannerPolicyConfig()
        datagen_cfg.robot_config = FloatingRobotiq2f85RobotConfig()
        datagen_cfg.camera_config = FrankaRobotiq2f85CameraSystem()
        datagen_cfg.task_horizon = task_horizon
        datagen_cfg.use_passive_viewer = use_passive_viewer
        datagen_cfg.seed = 42
        datagen_cfg.num_workers = 1
        # Don't restrict pickup_types - let it default to all pickup types
        # This ensures the object we want is found by _get_scene_objects
        # We'll set pickup_obj_name explicitly to target the specific object
        # If we restrict pickup_types to just [category], it might not find the object
        # if the category matching doesn't work exactly as expected
        datagen_cfg.task_sampler_config.pickup_types = None  # Use default (all pickup types)
        datagen_cfg.task_sampler_config.robot_object_z_offset = 0.15
        datagen_cfg.task_sampler_config.base_pose_sampling_radius_range = (0, 0.8)
        datagen_cfg.task_sampler_config.robot_safety_radius = 0
        datagen_cfg.policy_config.phase_timeout = 30.0
        datagen_cfg.policy_config.max_retries = 0  # Disable retries - test each grasp once
        datagen_cfg.scene_dataset = "ithor"  # Will be overridden
        datagen_cfg.data_split = "train"

        # Set the specific object name
        datagen_cfg.task_config.pickup_obj_name = body_name

        # Create task sampler and load scene
        task_sampler = PickTaskSampler(datagen_cfg)
        task_sampler.update_scene(scene_path)

        # Verify that the object we want exists in the environment
        try:
            # Search through candidate_objects to find the object by name
            if task_sampler.candidate_objects is None or len(task_sampler.candidate_objects) == 0:
                raise ValueError("No candidate objects available")
            pickup_obj = None
            for obj in task_sampler.candidate_objects:
                if obj.name == body_name:
                    pickup_obj = obj
                    break
            if pickup_obj is None:
                raise ValueError(f"Object {body_name} not found in candidate_objects")
            log.info(
                f"Verified object {body_name} exists in environment (type: {type(pickup_obj).__name__})"
            )
        except Exception as e:
            log.error(
                f"Object {body_name} not found in environment after scene initialization: {e}"
            )
            if (
                task_sampler.candidate_objects is not None
                and len(task_sampler.candidate_objects) > 0
            ):
                log.error(
                    f"Available objects: {[obj.name for obj in task_sampler.candidate_objects]}"
                )
            else:
                log.error("Available objects: None")
            task_sampler.close()
            continue

        model = task_sampler.env.current_model
        data = task_sampler.env.current_data

        # Create viewer once for all grasps if requested
        shared_viewer = None
        if use_passive_viewer:
            shared_viewer = mujoco.viewer.launch_passive(
                task_sampler.env.mj_datas[task_sampler.env.current_batch_index].model,
                task_sampler.env.mj_datas[task_sampler.env.current_batch_index],
            )
            shared_viewer.opt.sitegroup[0] = False

        # Profile the entire grasp testing loop for this object
        profile_key = f"test_all_grasps_{asset_id}"
        with object_profiler.profile(profile_key):
            # Test each grasp
            for i, grasp_pose in enumerate(test_grasps):
                try:
                    model.eq_solimp[:] = np.array([0.99, 0.99, 0.001, 1, 2])
                    model.eq_solref[:] = np.array([0.001, 1])

                    # Reset mjdata to original state
                    mujoco.mj_resetData(model, data)

                    # Ensure forward kinematics are computed before accessing robot transformations
                    mujoco.mj_forward(task_sampler.env.current_model, task_sampler.env.current_data)

                    # Place robot near grasp (teleport to new pre-grasp pose)
                    robot_view = task_sampler.env.current_robot.robot_view
                    place_robot_near_grasp(robot_view, grasp_pose, offset_distance)

                    # Forward kinematics again after placing robot
                    mujoco.mj_forward(task_sampler.env.current_model, task_sampler.env.current_data)

                    # Check for collisions after robot placement
                    model = task_sampler.env.current_model
                    data = task_sampler.env.current_data
                    task_cfg = datagen_cfg.task_config
                    has_collision, collision_details = check_robot_collision(
                        model, data, robot_view, target_object_name=task_cfg.pickup_obj_name
                    )

                    if has_collision:
                        log.warning(
                            f"Skipping grasp {i + 1}/{num_to_test} for {asset_id} - robot in collision after placement: {', '.join(collision_details)}"
                        )
                        # Track skipped grasps due to collision (don't count as tested)
                        object_metrics["skipped_collision"] += 1
                        continue

                    # Manually set up task config (similar to _sample_and_place_robot but skip placement)
                    pickup_obj = MlSpacesObject(object_name=task_cfg.pickup_obj_name, data=data)
                    task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
                    task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

                    pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
                    pickup_obj_goal_pose[2] += 0.05  # 5 cm
                    task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

                    # Set up cameras
                    task_sampler.setup_cameras(task_sampler.env)

                    # Create task directly
                    from molmo_spaces.tasks.pick_task import PickTask

                    task = PickTask(task_sampler.env, datagen_cfg)
                    policy = datagen_cfg.policy_config.policy_factory(datagen_cfg, task)

                    # Override _compute_target_poses to use our specific grasp pose
                    def compute_target_poses_with_grasp(self):
                        """Override to use the specific grasp pose we want to test."""
                        target_poses = {}
                        robot_view = self.task.env.current_robot.robot_view

                        # Use the specific grasp pose we're testing
                        grasp_pose_world = grasp_pose.copy()

                        # Visualize the grasp pose being tested
                        if hasattr(self, "_show_poses") and self.task.viewer is not None:
                            self._show_poses(
                                np.array([grasp_pose_world]), style="tcp", color=(1, 0, 0, 1)
                            )  # Red for grasp
                            self.task.viewer.sync()

                        log.info(f"Testing grasp {i + 1}/{num_to_test} for {asset_id}")

                        # Compute pregrasp and lift poses using planner policy logic
                        # (matching PickPlannerPolicy._compute_target_poses)
                        randomize_pregrasp = False
                        if randomize_pregrasp:
                            pregrasp_height_offset = np.random.uniform(
                                -self.policy_config.pregrasp_height_noise,
                                self.policy_config.pregrasp_height_noise,
                            )
                            postgrasp_height_offset = np.random.uniform(
                                -self.policy_config.postgrasp_height_noise,
                                self.policy_config.postgrasp_height_noise,
                            )
                        else:
                            pregrasp_height_offset = 0.0
                            postgrasp_height_offset = 0.0

                        pregrasp_pose = grasp_pose_world.copy()
                        # Pregrasp pose - above the pickup object
                        grasp_z_axis = grasp_pose_world[:3, 2]
                        pregrasp_offset = (
                            self.policy_config.pregrasp_z_offset + pregrasp_height_offset
                        )
                        pregrasp_pose[:3, 3] -= pregrasp_offset * grasp_z_axis

                        target_poses["pregrasp"] = pregrasp_pose

                        target_poses["grasp"] = grasp_pose_world

                        # Lift pose - above grasp position (matching PickPlannerPolicy)
                        lift_pose = grasp_pose_world.copy()
                        lift_pose[:3, 3] += np.array(
                            [0, 0, self.policy_config.postgrasp_z_offset + postgrasp_height_offset]
                        )

                        target_poses["lift"] = lift_pose

                        log.info(f"Planning completed. w/ {len(target_poses)} steps\n")
                        return target_poses

                    # Bind the method to the policy instance
                    import types

                    policy._compute_target_poses = types.MethodType(
                        compute_target_poses_with_grasp, policy
                    )

                    # Set viewer on task if we have a shared viewer
                    if shared_viewer is not None:
                        task.viewer = shared_viewer

                    # Run rollout (pass viewer to reuse it)
                    try:
                        success, failure_modes = GraspTestRolloutRunner.run_single_rollout(
                            episode_seed=42 + i,
                            task=task,
                            policy=policy,
                            viewer=shared_viewer,
                            use_passive_viewer=False,  # Don't create new viewer, use shared one
                            save_failed_video_dir=save_failed_videos_dir,
                            grasp_idx=i,
                            object_name=asset_id,
                            joint_name="",  # Empty for pickup objects
                        )

                        # Increment grasps_tested only after actually running the rollout (not skipped)
                        object_metrics["grasps_tested"] += 1
                        metrics["total_grasps_tested"] += 1

                        if success:
                            object_metrics["successful_grasps"] += 1
                        else:
                            # Store failure modes for this grasp
                            if "grasp_failure_modes" not in object_metrics:
                                object_metrics["grasp_failure_modes"] = []
                            object_metrics["grasp_failure_modes"].append(
                                {
                                    "grasp_idx": i,
                                    "failure_modes": failure_modes,
                                }
                            )
                    finally:
                        # Clean up task and policy objects after each rollout to free memory
                        try:
                            if "task" in locals():
                                try:
                                    if hasattr(task, "close"):
                                        task.close()
                                except Exception:
                                    pass
                                del task
                        except NameError:
                            pass
                        try:
                            if "policy" in locals():
                                try:
                                    if hasattr(policy, "close"):
                                        policy.close()
                                except Exception:
                                    pass
                                del policy
                        except NameError:
                            pass
                        # Force garbage collection after cleanup
                        gc.collect()

                except Exception as e:
                    # Extract error message without serializing config objects
                    # Handle Pydantic serialization errors gracefully
                    import traceback
                    import warnings

                    # Suppress Pydantic serialization warnings when logging errors
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", category=UserWarning, message=".*Pydantic.*"
                        )
                        warnings.filterwarnings(
                            "ignore", message=".*serialized value may not be as expected.*"
                        )

                        # Handle Pydantic serialization errors specifically
                        error_type_name = type(e).__name__
                        if (
                            "PydanticSerialization" in error_type_name
                            or "Pydantic" in error_type_name
                        ):
                            error_msg = f"{error_type_name}: Config serialization error (type mismatch in camera_config)"
                        else:
                            error_msg = str(e) if str(e) else repr(e)
                            error_msg = f"{error_type_name}: {error_msg}"

                        traceback_str = traceback.format_exc()
                        log.error(f"Error testing grasp {i} for {asset_id}: {error_msg}")
                        log.error(f"Traceback for grasp {i}:\n{traceback_str}")
                    continue

        # Close shared viewer after all grasps are tested
        if shared_viewer is not None:
            shared_viewer.close()

        # Get profiling results
        if object_profiler.get_n(profile_key) > 0:
            total_time = object_profiler.get_avg_time(profile_key) * object_profiler.get_n(
                profile_key
            )
            object_metrics["total_time_seconds"] = total_time
            if object_metrics["grasps_tested"] > 0:
                object_metrics["avg_time_per_grasp_seconds"] = (
                    total_time / object_metrics["grasps_tested"]
                )

        # Calculate success rate (always calculate, not just when profiling)
        object_metrics["success_rate"] = (
            object_metrics["successful_grasps"] / object_metrics["grasps_tested"]
            if object_metrics["grasps_tested"] > 0
            else 0.0
        )
        metrics["total_noncolliding_grasps"] += object_metrics["noncolliding_grasps"]
        metrics["total_successful_grasps"] += object_metrics["successful_grasps"]
        metrics["per_object_metrics"].append(object_metrics)
        metrics["total_objects"] += 1

        # Update overall success rate
        if metrics["total_grasps_tested"] > 0:
            metrics["success_rate"] = (
                metrics["total_successful_grasps"] / metrics["total_grasps_tested"]
            )

        log.info(
            f"Object {asset_id}: {object_metrics['successful_grasps']}/{object_metrics['grasps_tested']} successful "
            f"({object_metrics['success_rate'] * 100:.1f}%) | "
            f"Total time: {object_metrics['total_time_seconds']:.2f}s | "
            f"Avg per grasp: {object_metrics['avg_time_per_grasp_seconds']:.2f}s"
        )

        # Save metrics to JSON after each object
        save_metrics_to_json()

        # Clean up
        task_sampler.close()

    # Test articulated objects
    for asset_id, joint_info in jointed_objects.items():
        log.info(f"Testing articulated object: {asset_id} with {len(joint_info)} joints")

        do_skip = False
        if do_skip:
            log.info(f"Skipping {asset_id} - do_skip is True")
            continue

        # Check if object is in EXTENDED_ARTICULABLE types
        object_name = None
        for obj_name, obj_data in scene_metadata.get("objects", {}).items():
            if obj_data.get("asset_id") == asset_id:
                object_name = obj_name
                break

        if not object_name:
            log.warning(f"Could not find object name for asset_id {asset_id}")
            continue

        object_data = scene_metadata.get("objects", {}).get(object_name, {})
        category = object_data.get("category", "").lower()
        if category not in [t.lower() for t in EXTENDED_ARTICULATION_TYPES_THOR]:
            log.info(f"Skipping {asset_id} - not in EXTENDED_ARTICULABLE types")
            continue

        # Test each joint
        for joint_name, (model_joint_name, joint_body_name) in joint_info.items():
            log.info(f"Testing joint {joint_name} on {asset_id}")

            # Load grasps for joint
            try:
                gripper, grasps = grasp_loader.load_grasps_for_joint(
                    asset_id, joint_name, num_grasps=1000
                )
            except Exception as e:
                log.warning(f"Failed to load grasps for {asset_id}/{joint_name}: {e}")
                continue

            if len(grasps) == 0:
                log.warning(f"No grasps found for {asset_id}/{joint_name}")
                continue

            # Get joint body pose
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, model_joint_name)
            if joint_id < 0:
                log.warning(f"Joint {model_joint_name} not found in model")
                continue

            joint_body_id = model.joint(joint_id).bodyid[0]

            # Ensure forward kinematics are computed before accessing quaternions
            mujoco.mj_forward(model, data)

            # Check if object is inside an enclosed site (e.g., inside a drawer or cabinet)
            # Check the main object body (not the joint body) for enclosed sites
            object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            if object_body_id >= 0:
                is_within_site, site_id = is_body_within_any_site(model, data, object_body_id)
                if is_within_site:
                    in_free_space, _, _ = is_body_within_site_in_freespace(
                        site_id, object_body_id, model, data
                    )
                    if not in_free_space:
                        log.info(
                            f"Skipping {asset_id}/{joint_name} (body: {object_name}) - object is inside an enclosed site"
                        )
                        continue

            joint_body_pos = data.xpos[joint_body_id]
            joint_body_quat = data.xquat[joint_body_id].copy()
            # Handle zero-norm quaternion (use identity quaternion as fallback)
            if np.linalg.norm(joint_body_quat) < 1e-6:
                joint_body_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
            joint_body_pose = pos_quat_to_pose_mat(joint_body_pos, joint_body_quat)

            # Transform grasps to world frame
            GRIP_BASE_TCP = np.eye(4)  # For droid gripper
            grasp_poses_world = joint_body_pose @ grasps @ GRIP_BASE_TCP

            # Filter non-colliding grasps
            noncolliding_mask = get_noncolliding_grasp_mask(
                model, data, grasp_poses_world, collision_batch_size
            )
            noncolliding_grasps = grasp_poses_world[noncolliding_mask]

            log.info(
                f"Found {len(noncolliding_grasps)} non-colliding grasps out of {len(grasps)} total"
            )

            if len(noncolliding_grasps) == 0:
                log.warning(f"No non-colliding grasps for {asset_id}/{joint_name}")
                continue

            # Test up to max_grasps_per_object (or all if None)
            if max_grasps_per_object is None:
                num_to_test = len(noncolliding_grasps)
                test_grasps = noncolliding_grasps
            else:
                num_to_test = min(len(noncolliding_grasps), max_grasps_per_object)
                test_grasps = noncolliding_grasps[:num_to_test]

            object_metrics = {
                "object_name": asset_id,
                "object_type": "articulated",
                "joint_name": joint_name,
                "body_name": object_name,
                "initial_openness": None,  # Will be set after loading scene
                "task_type": None,  # Will be determined based on openness
                "total_grasps": len(grasps),
                "noncolliding_grasps": len(noncolliding_grasps),
                "grasps_attempted": num_to_test,  # Number of grasps we intended to test
                "grasps_tested": 0,  # Actual number of grasps tested (excluding skipped due to collision)
                "successful_grasps": 0,
                "success_rate": 0.0,
                "total_time_seconds": 0.0,
                "avg_time_per_grasp_seconds": 0.0,
                "grasp_failure_modes": [],
                "skipped_collision": 0,
            }

            # Create profiler for this object/joint
            object_profiler = Profiler(log_realtime=False)

            # Create task config once for this object/joint
            datagen_cfg = OpeningBaseConfig()
            datagen_cfg.policy_config = OpenClosePlannerPolicyConfig()
            datagen_cfg.robot_config = FloatingRobotiq2f85RobotConfig()
            datagen_cfg.camera_config = FrankaRobotiq2f85CameraSystem()
            datagen_cfg.task_horizon = task_horizon
            datagen_cfg.use_passive_viewer = use_passive_viewer
            datagen_cfg.seed = 42
            datagen_cfg.num_workers = 1
            # task_type will be determined after checking openness
            datagen_cfg.task_sampler_config.robot_object_z_offset = 0.15
            datagen_cfg.task_sampler_config.base_pose_sampling_radius_range = (0, 0.8)
            datagen_cfg.task_sampler_config.robot_safety_radius = 0
            datagen_cfg.policy_config.phase_timeout = 30.0
            datagen_cfg.policy_config.max_retries = 0  # Disable retries - test each grasp once
            datagen_cfg.scene_dataset = "ithor"
            datagen_cfg.data_split = "train"

            datagen_cfg.task_config.task_success_threshold = 0.5
            datagen_cfg.policy_config.speed_slow = 0.005
            datagen_cfg.policy_config.move_settle_time = 0.5

            # Set the specific object and joint
            datagen_cfg.task_config.pickup_obj_name = object_name
            datagen_cfg.task_config.joint_name = model_joint_name

            # Create task sampler and load scene
            task_sampler = OpenTaskSampler(datagen_cfg)
            task_sampler.update_scene(scene_path)

            # Check initial openness using MlSpacesArticulationObject
            pickup_obj = MlSpacesArticulationObject(
                object_name=object_name, data=task_sampler.env.current_data
            )
            if not isinstance(pickup_obj, MlSpacesArticulationObject):
                log.warning(f"Object {object_name} is not an articulation object")
                task_sampler.close()
                continue

            # Find joint index
            if model_joint_name not in pickup_obj.joint_names:
                log.warning(f"Joint {model_joint_name} not found in {object_name}")
                task_sampler.close()
                continue

            joint_index = pickup_obj.joint_names.index(model_joint_name)
            openness = get_object_openness_percentage(
                task_sampler.env.current_data, object_name, model_joint_name
            )
            task_type = "open" if openness < 0.5 else "close"
            datagen_cfg.task_type = task_type
            object_metrics["initial_openness"] = openness
            object_metrics["task_type"] = task_type

            # Set target initial state based on task type
            if task_type == "open":
                datagen_cfg.task_sampler_config.target_initial_state_open_percentage = 0.0
            else:
                datagen_cfg.task_sampler_config.target_initial_state_open_percentage = 0.5

            log.info(
                f"Object {asset_id} joint {joint_name} is {openness * 100:.1f}% open, using {task_type} task"
            )

            model = task_sampler.env.current_model
            data = task_sampler.env.current_data

            # Create viewer once for all grasps if requested
            shared_viewer = None
            if use_passive_viewer:
                shared_viewer = mujoco.viewer.launch_passive(
                    task_sampler.env.mj_datas[task_sampler.env.current_batch_index].model,
                    task_sampler.env.mj_datas[task_sampler.env.current_batch_index],
                )
                shared_viewer.opt.sitegroup[0] = False

            # Profile the entire grasp testing loop for this object/joint
            joint_profile_key = f"test_all_grasps_{asset_id}_{joint_name}"
            with object_profiler.profile(joint_profile_key):
                # Test each grasp
                for i, grasp_pose in enumerate(test_grasps):
                    try:
                        model.eq_solimp[:] = np.array([0.8, 0.8, 0.05, 1, 2])
                        model.eq_solref[:] = np.array([0.01, 1])

                        # Reset mjdata to original state
                        mujoco.mj_resetData(model, data)

                        # Ensure forward kinematics are computed before accessing robot transformations
                        mujoco.mj_forward(
                            task_sampler.env.current_model, task_sampler.env.current_data
                        )

                        ### CHECK ROBOT COLLISION AT PREGRASP POSE and GRASP POSE  ####

                        # Place robot near grasp (teleport to new pre-grasp pose)
                        robot_view = task_sampler.env.current_robot.robot_view
                        place_robot_near_grasp(robot_view, grasp_pose, offset_distance)

                        # Forward kinematics again after placing robot
                        mujoco.mj_forward(
                            task_sampler.env.current_model, task_sampler.env.current_data
                        )

                        # Check for collisions after robot placement (at pre-grasp pose)
                        model = task_sampler.env.current_model
                        data = task_sampler.env.current_data
                        task_cfg = datagen_cfg.task_config
                        has_collision, collision_details = check_robot_collision(
                            model, data, robot_view, target_object_name=task_cfg.pickup_obj_name
                        )

                        if has_collision:
                            log.warning(
                                f"Skipping grasp {i + 1}/{num_to_test} for {asset_id}/{joint_name} - robot in collision at pre-grasp pose: {', '.join(collision_details)}"
                            )
                            # Track skipped grasps due to collision (don't count as tested)
                            object_metrics["skipped_collision"] += 1
                            continue

                        # Check for collisions at grasp pose
                        # Get IK solution for grasp pose to check collisions
                        kinematics = task_sampler.env.current_robot.kinematics
                        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
                        grasp_jp_dict = kinematics.ik(
                            gripper_mg_id,
                            grasp_pose,
                            robot_view.move_group_ids(),
                            robot_view.get_qpos_dict(),
                            base_pose=robot_view.base.pose,
                        )

                        if grasp_jp_dict is not None:
                            # Save current joint positions
                            original_qpos_dict = robot_view.get_qpos_dict()

                            # Temporarily set robot to grasp pose
                            robot_view.set_qpos_dict(grasp_jp_dict)
                            mujoco.mj_forward(model, data)

                            # Check for collisions at grasp pose
                            has_collision_grasp, collision_details_grasp = check_robot_collision(
                                model, data, robot_view, target_object_name=task_cfg.pickup_obj_name
                            )

                            # Restore original joint positions
                            robot_view.set_qpos_dict(original_qpos_dict)
                            mujoco.mj_forward(model, data)

                            if has_collision_grasp:
                                log.warning(
                                    f"Skipping grasp {i + 1}/{num_to_test} for {asset_id}/{joint_name} - robot in collision at grasp pose: {', '.join(collision_details_grasp)}"
                                )
                                # Track skipped grasps due to collision (don't count as tested)
                                object_metrics["skipped_collision"] += 1
                                continue
                        else:
                            # IK failed for grasp pose, skip this grasp
                            log.warning(
                                f"Skipping grasp {i + 1}/{num_to_test} for {asset_id}/{joint_name} - IK failed for grasp pose"
                            )
                            object_metrics["skipped_collision"] += 1
                            continue

                        # Manually set up task config (similar to _sample_and_place_robot but skip placement)
                        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
                        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

                        # Set joint info (needed for opening tasks)
                        task_cfg.joint_index = joint_index
                        task_cfg.joint_name = model_joint_name
                        task_cfg.joint_start_position = pickup_obj.get_joint_position(joint_index)

                        # Set up cameras
                        task_sampler.setup_cameras(task_sampler.env)

                        # Create task directly
                        from molmo_spaces.tasks.opening_tasks import OpeningTask

                        task = OpeningTask(task_sampler.env, datagen_cfg)
                        policy = datagen_cfg.policy_config.policy_factory(datagen_cfg, task)

                        # Set postgrasp_z_offset to 0.05
                        policy.policy_config.postgrasp_z_offset = 0.05

                        # Override _compute_trajectory to compute arc AFTER grasping based on actual TCP position
                        def compute_trajectory_with_deferred_arc(self):
                            """Override to compute arc after grasping based on actual TCP position."""
                            from scipy.spatial.transform import Rotation as R
                            from molmo_spaces.utils.articulation_utils import (
                                gather_joint_info,
                                step_circular_path,
                                step_linear_path,
                            )

                            robot_view = self.task.env.current_robot.robot_view
                            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
                            start_ee_pose = robot_view.get_move_group(
                                gripper_mg_id
                            ).leaf_frame_to_world

                            # Use the specific grasp pose we're testing
                            grasp_pose_world = grasp_pose.copy()
                            # Visualize the grasp pose being tested
                            if hasattr(self, "_show_poses") and self.task.viewer is not None:
                                self._show_poses(
                                    np.array([grasp_pose_world]), style="tcp", color=(1, 0, 0, 1)
                                )  # Red for grasp
                                self.task.viewer.sync()

                            log.info(
                                f"Testing grasp {i + 1}/{num_to_test} for {asset_id}/{joint_name}"
                            )

                            # Compute pregrasp pose
                            pregrasp_pose = grasp_pose_world.copy()
                            grasp_pos = grasp_pose_world[:3, 3]
                            rotation = R.from_matrix(grasp_pose_world[:3, :3])
                            distance = self.policy_config.pregrasp_z_offset
                            pregrasp_pose[:3, 3] = grasp_pos + rotation.apply(
                                np.array([0, 0, -distance])
                            )

                            # Deferred arc move sequence - computes arc from actual TCP position after grasping
                            class DeferredArcMoveSequence(TCPMoveSequence):
                                def __init__(
                                    self,
                                    policy_ref,
                                    robot_view,
                                    tcp_to_jp_fn,
                                    settle_time,
                                    **kwargs,
                                ):
                                    super().__init__(
                                        robot_view,
                                        tcp_to_jp_fn,
                                        settle_time,
                                        move_segments=[],
                                        **kwargs,
                                    )
                                    self._policy_ref = policy_ref
                                    self._arc_computed = False
                                    self._settle_wait_time = 0.5
                                    self.duration = self._settle_wait_time + 1.0

                                def get_current_phase(self) -> str:
                                    if not self._arc_computed or len(self._move_segments) == 0:
                                        return "waiting_for_arc"
                                    return super().get_current_phase()

                                def get_current_action(self):
                                    if not self._arc_computed or len(self._move_segments) == 0:
                                        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[
                                            0
                                        ]
                                        current_pose = self.robot_view.get_move_group(
                                            gripper_mg_id
                                        ).leaf_frame_to_world
                                        return self.tcp_to_jp_fn(gripper_mg_id, current_pose)
                                    return super().get_current_action()

                                def execute(self) -> bool:
                                    if not self._arc_computed:
                                        if self.start_time is None:
                                            self.start_time = self.robot_view.mj_data.time
                                            log.info(
                                                "Waiting for gripper to settle before computing arc..."
                                            )
                                            return False

                                        elapsed = self.robot_view.mj_data.time - self.start_time
                                        if elapsed < self._settle_wait_time:
                                            return False

                                        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[
                                            0
                                        ]
                                        actual_grasp_pose = self.robot_view.get_move_group(
                                            gripper_mg_id
                                        ).leaf_frame_to_world

                                        task_cfg = self._policy_ref.config.task_config
                                        pickup_obj = MlSpacesArticulationObject(
                                            object_name=task_cfg.pickup_obj_name,
                                            data=self._policy_ref.task.env.current_data,
                                        )
                                        joint_idx = task_cfg.joint_index
                                        joint_info = gather_joint_info(
                                            self._policy_ref.task.env.current_model,
                                            self._policy_ref.task.env.current_data,
                                            pickup_obj.joint_ids[joint_idx],
                                        )

                                        actual_pos = actual_grasp_pose[:3, 3]
                                        actual_quat = R.from_matrix(
                                            actual_grasp_pose[:3, :3]
                                        ).as_quat(scalar_first=True)

                                        if joint_info["joint_type"] == mujoco.mjtJoint.mjJNT_HINGE:
                                            joint_range = joint_info["joint_range"]
                                            nonzero_index = np.nonzero(joint_range)
                                            task_type = self._policy_ref.config.task_type
                                            if task_type == "open":
                                                max_joint_angle = joint_range[nonzero_index[0]]
                                            else:
                                                max_joint_angle = 0

                                            path_dict = step_circular_path(
                                                actual_pos,
                                                actual_quat,
                                                joint_info,
                                                max_joint_angle,
                                                n_waypoints=500,
                                            )
                                        elif (
                                            joint_info["joint_type"] == mujoco.mjtJoint.mjJNT_SLIDE
                                        ):
                                            joint_axis_world = (
                                                joint_info["joint_body_orientation"]
                                                @ joint_info["joint_axis"]
                                            )
                                            joint_direction = -joint_axis_world
                                            normalize_dir_axis = joint_direction / np.linalg.norm(
                                                joint_direction
                                            )
                                            current_joint_pos = joint_info["joint_pos"]

                                            task_type = self._policy_ref.config.task_type
                                            if task_type == "open":
                                                max_joint_angle = joint_info["max_range"]
                                            else:
                                                max_joint_angle = 0

                                            path_dict = step_linear_path(
                                                to_handle_dist=normalize_dir_axis
                                                * (max_joint_angle - current_joint_pos),
                                                current_pos=actual_pos,
                                                current_quat=actual_quat,
                                                step_size=0.005,
                                                is_reverse=True,
                                            )
                                        else:
                                            raise ValueError(
                                                f"Unknown joint type: {joint_info['joint_type']}"
                                            )
                                        all_lift_poses = []
                                        for c in range(len(path_dict["mocap_pos"])):
                                            lift_pose = np.eye(4)
                                            lift_pose[:3, 3] = path_dict["mocap_pos"][c]
                                            lift_pose[:3, :3] = R.from_quat(
                                                path_dict["mocap_quat"][c], scalar_first=True
                                            ).as_matrix()
                                            all_lift_poses.append(lift_pose)

                                        if len(all_lift_poses) > 0:
                                            self._move_segments = [
                                                TCPMoveSegment(
                                                    name="lift",
                                                    start_pose=actual_grasp_pose,
                                                    end_pose=all_lift_poses[0],
                                                    speed=self._policy_ref.policy_config.speed_slow,
                                                )
                                            ]
                                            for idx in range(len(all_lift_poses) - 1):
                                                self._move_segments.append(
                                                    TCPMoveSegment(
                                                        name="lift",
                                                        start_pose=all_lift_poses[idx],
                                                        end_pose=all_lift_poses[idx + 1],
                                                        speed=self._policy_ref.policy_config.speed_slow,
                                                    )
                                                )
                                            self.duration = sum(
                                                seg.duration for seg in self._move_segments
                                            )
                                        else:
                                            # todo(omar)
                                            self._move_segments = [
                                                TCPMoveSegment(
                                                    name="lift",
                                                    start_pose=actual_grasp_pose,
                                                    end_pose=actual_grasp_pose,
                                                    speed=self._policy_ref.policy_config.speed_slow,
                                                )
                                            ]
                                            self.duration = 0.1

                                        self._arc_computed = True
                                        self.start_time = None
                                        self.move_seg_idx = None
                                        self.move_seg_start_time = None
                                        log.info(
                                            f"Arc computed with {len(self._move_segments)} segments"
                                        )

                                    return super().execute()

                            settle_before_grasp_time = 0.5

                            actions = [
                                GripperAction(robot_view, True, 0.0),
                                TCPMoveSequence(
                                    robot_view,
                                    self._tcp_to_jp_fn,
                                    settle_before_grasp_time,
                                    gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                                    tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                                    tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                                    move_segments=[
                                        TCPMoveSegment(
                                            name="pregrasp",
                                            start_pose=start_ee_pose,
                                            end_pose=pregrasp_pose,
                                            speed=self.policy_config.speed_fast,
                                        ),
                                        TCPMoveSegment(
                                            name="grasp",
                                            start_pose=pregrasp_pose,
                                            end_pose=grasp_pose_world,
                                            speed=self.policy_config.speed_slow,
                                        ),
                                    ],
                                ),
                                GripperAction(
                                    robot_view, False, self.policy_config.gripper_close_duration
                                ),
                                DeferredArcMoveSequence(
                                    self,
                                    robot_view,
                                    self._tcp_to_jp_fn,
                                    self.policy_config.move_settle_time,
                                    is_holding_object=True,
                                    gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                                ),
                            ]

                            return actions

                        # Bind the method to the policy instance
                        import types

                        policy._compute_trajectory = types.MethodType(
                            compute_trajectory_with_deferred_arc, policy
                        )

                        # Set viewer on task if we have a shared viewer
                        if shared_viewer is not None:
                            task.viewer = shared_viewer

                        # Run rollout (pass viewer to reuse it)
                        try:
                            success, failure_modes = GraspTestRolloutRunner.run_single_rollout(
                                episode_seed=42 + i,
                                task=task,
                                policy=policy,
                                viewer=shared_viewer,
                                use_passive_viewer=False,  # Don't create new viewer, use shared one
                                save_failed_video_dir=save_failed_videos_dir,
                                grasp_idx=i,
                                object_name=asset_id,
                                joint_name=joint_name,
                            )

                            # Increment grasps_tested only after actually running the rollout (not skipped)
                            object_metrics["grasps_tested"] += 1
                            metrics["total_grasps_tested"] += 1

                            if success:
                                object_metrics["successful_grasps"] += 1
                            else:
                                # Store failure modes for this grasp
                                if "grasp_failure_modes" not in object_metrics:
                                    object_metrics["grasp_failure_modes"] = []
                                object_metrics["grasp_failure_modes"].append(
                                    {
                                        "grasp_idx": i,
                                        "failure_modes": failure_modes,
                                    }
                                )
                        finally:
                            # Clean up task and policy objects after each rollout to free memory
                            try:
                                if "task" in locals():
                                    try:
                                        if hasattr(task, "close"):
                                            task.close()
                                    except Exception:
                                        pass
                                    del task
                            except NameError:
                                pass
                            try:
                                if "policy" in locals():
                                    try:
                                        if hasattr(policy, "close"):
                                            policy.close()
                                    except Exception:
                                        pass
                                    del policy
                            except NameError:
                                pass
                            # Force garbage collection after cleanup
                            gc.collect()

                    except Exception as e:
                        # Extract error message without serializing config objects
                        import traceback
                        import warnings

                        error_msg = str(e) if str(e) else repr(e)
                        error_type = type(e).__name__
                        traceback_str = traceback.format_exc()

                        log.error(
                            f"Error testing grasp {i} for {asset_id}/{joint_name}: {error_type}: {error_msg}"
                        )
                        log.error(f"Traceback for grasp {i}:\n{traceback_str}")
                        continue

            # Close shared viewer after all grasps are tested for this joint
            if shared_viewer is not None:
                shared_viewer.close()

            # Get profiling results
            if object_profiler.get_n(joint_profile_key) > 0:
                total_time = object_profiler.get_avg_time(
                    joint_profile_key
                ) * object_profiler.get_n(joint_profile_key)
                object_metrics["total_time_seconds"] = total_time
                if object_metrics["grasps_tested"] > 0:
                    object_metrics["avg_time_per_grasp_seconds"] = (
                        total_time / object_metrics["grasps_tested"]
                    )

            # Calculate success rate (always calculate, not just when profiling)
            object_metrics["success_rate"] = (
                object_metrics["successful_grasps"] / object_metrics["grasps_tested"]
                if object_metrics["grasps_tested"] > 0
                else 0.0
            )
            metrics["total_noncolliding_grasps"] += object_metrics["noncolliding_grasps"]
            metrics["total_successful_grasps"] += object_metrics["successful_grasps"]
            metrics["per_object_metrics"].append(object_metrics)
            metrics["total_objects"] += 1

            # Update overall success rate
            if metrics["total_grasps_tested"] > 0:
                metrics["success_rate"] = (
                    metrics["total_successful_grasps"] / metrics["total_grasps_tested"]
                )

            log.info(
                f"Object {asset_id} joint {joint_name}: {object_metrics['successful_grasps']}/{object_metrics['grasps_tested']} successful "
                f"({object_metrics['success_rate'] * 100:.1f}%) | "
                f"Total time: {object_metrics['total_time_seconds']:.2f}s | "
                f"Avg per grasp: {object_metrics['avg_time_per_grasp_seconds']:.2f}s"
            )

            # Save metrics to JSON after each object/joint
            save_metrics_to_json()

            # Clean up
            task_sampler.close()

    # Calculate overall success rate
    if metrics["total_grasps_tested"] > 0:
        metrics["success_rate"] = (
            metrics["total_successful_grasps"] / metrics["total_grasps_tested"]
        )

    # Save final metrics to JSON
    save_metrics_to_json()

    # Clean up memory after scene processing
    # Clear any remaining references to models, data, and other resources
    log.debug("Cleaning up memory after scene processing...")
    try:
        # Clean up model, data, and spec used for collision checking
        # These are defined at function scope, so they should exist
        del model
        del data
        del spec
    except NameError:
        # Variables may not exist if there was an early return
        pass
    except Exception as e:
        log.debug(f"Error during cleanup: {e}")

    # Force garbage collection to free memory
    gc.collect()
    log.debug("Memory cleanup completed")

    return metrics


def main(args: argparse.ArgumentParser) -> None:
    scene_dataset = args.scene_dataset
    data_split = args.data_split
    house_ind = args.house_ind
    offset_distance = args.offset_distance
    max_grasps_per_object = args.max_grasps_per_object
    task_horizon = args.task_horizon
    use_passive_viewer = args.use_passive_viewer

    # Get scene path and metadata
    from molmo_spaces.molmo_spaces_constants import (
        get_scenes,
        get_scenes_root,
        get_resource_manager,
    )

    scenes_root = get_scenes_root()
    log.info(f"SCENES_ROOT: {scenes_root}")

    def get_scene_path_by_index(dataset_name: str, house_index: int) -> Path | None:
        """Get scene path directly by index, checking all splits if needed.
        Searches recursively in subdirectories since files may be organized in folders.
        """
        dataset_dir = scenes_root / dataset_name
        log.info(f"dataset dir: {dataset_dir}, exists: {dataset_dir.exists()}")

        if not dataset_dir.exists():
            log.info(f"dataset dir does not exist: {dataset_dir}")
            return None

        if dataset_name == "ithor":
            # For ithor, look for FloorPlan{index}_physics.xml recursively
            pattern = f"FloorPlan{house_index}_physics.xml"
            path = dataset_dir / pattern
            log.info(f"path: {path} exists: {path.is_file()} is symlink: {path.is_symlink()}")
            return path

        else:
            # For procthor datasets, check all splits recursively
            for split in ["train", "val", "test"]:
                dataset_dir = scenes_root / f"{dataset_name}-{split}"
                log.info(f"Checking dataset dir: {dataset_dir}, exists: {dataset_dir.exists()}")

                if not dataset_dir.exists():
                    continue

                # Try base pattern first
                pattern = f"{split}_{house_index}.xml"
                matches = list(dataset_dir.rglob(pattern))
                if matches:
                    matches.sort(key=lambda p: len(p.parts))
                    return matches[0]
                # Try physics variant
                pattern = f"{split}_{house_index}_physics.xml"
                matches = list(dataset_dir.rglob(pattern))
                if matches:
                    matches.sort(key=lambda p: len(p.parts))
                    return matches[0]

        return None

    # Get scene mapping for reference (used for multi-house mode)
    scene_mapping = get_scenes(scene_dataset, data_split)

    # Determine which houses to process
    validated_scene_path = None  # Only used for single-house mode
    if house_ind is None:
        # Run entire dataset - filter out None values
        print(
            f"Running entire dataset: {len(scene_mapping[data_split])} houses in {data_split} split",
            flush=True,
        )
        house_indices = sorted(
            [
                idx
                for idx in scene_mapping[data_split].keys()
                if scene_mapping[data_split][idx] is not None
            ]
        )
        skipped_count = len(scene_mapping[data_split]) - len(house_indices)
        log.info(f"Running entire dataset: {len(house_indices)} houses in {data_split} split")
        if skipped_count > 0:
            log.info(f"Skipped {skipped_count} houses with None values (not part of the split)")
    else:
        # Run single house - try to get scene path from mapping first, then fallback to direct search
        full_scene_path = None

        # First, try to get path from scene mapping (checks all splits)
        for split in ["train", "val", "test"]:
            if house_ind in scene_mapping.get(split, {}):
                scene_info = scene_mapping[split][house_ind]
                if scene_info is not None:
                    # scene_info could be a dict with variants or a direct path
                    if isinstance(scene_info, dict):
                        # Try to get base or physics variant
                        full_scene_path = scene_info.get("base") or scene_info.get("physics")
                        if full_scene_path:
                            full_scene_path = Path(full_scene_path)
                            break
                    elif isinstance(scene_info, (str, Path)):
                        full_scene_path = Path(scene_info)
                        break

        # Fallback to direct directory search if mapping didn't work
        if full_scene_path is None:
            full_scene_path = get_scene_path_by_index(scene_dataset, house_ind)

        log.info(f"got scene path: {full_scene_path}")

        if full_scene_path is None:
            log.warning(f"House index {house_ind} not found. Could not determine scene path.")
            log.info(
                f"Checked path: {scenes_root / scene_dataset / f'FloorPlan{house_ind}_physics.xml' if scene_dataset == 'ithor' else 'various'}"
            )
            log.info("No houses to process after skipping missing house")
            return 1

        # House exists and path exists - proceed
        log.info(f"House {house_ind} found with valid scene path: {full_scene_path}")

        # Install scene assets (objects and grasps) early, right after we have the scene path
        # This ensures everything is ready before we start processing
        log.info(f"Installing scene assets (objects and grasps) for house {house_ind}...")
        try:
            install_scene_with_objects_and_grasps_from_path(full_scene_path)
            log.info(f"Successfully installed scene assets for house {house_ind}")
        except Exception as e:
            log.error(f"Failed to install scene assets for {full_scene_path}: {e}")
            import traceback

            log.error(traceback.format_exc())
            log.warning(
                f"Failed to install scene assets for house {house_ind}, but continuing anyway..."
            )

        house_indices = [house_ind]
        # Store validated scene path for reuse in loop (avoid redundant extraction/resolution)
        validated_scene_path = full_scene_path
        log.info(f"Running single house: {house_ind}")

    # Set up save directories and timestamp (create once for consistency)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("=" * 80)
    log.info("📂 OUTPUT DIRECTORY CONFIGURATION")
    log.info("=" * 80)
    log.info(f"ASSETS_DIR: {ASSETS_DIR.absolute()}")

    base_save_failed_videos_dir = None
    if args.save_failed_videos_dir and args.save_failed_videos_dir.strip():
        # Non-empty string provided
        if args.save_failed_videos_dir.strip() == "auto":
            # Auto-generate path similar to run_pipeline.py
            base_save_failed_videos_dir = ASSETS_DIR / "datagen" / "grasp_test_v1" / timestamp
            base_save_failed_videos_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"🎥 Failed videos directory (auto): {base_save_failed_videos_dir.absolute()}")
        else:
            # Custom path provided
            base_save_failed_videos_dir = Path(args.save_failed_videos_dir)
            base_save_failed_videos_dir.mkdir(parents=True, exist_ok=True)
            log.info(
                f"🎥 Failed videos directory (custom): {base_save_failed_videos_dir.absolute()}"
            )
    else:
        # Empty string or None - disable video saving
        log.info("🎥 Video saving disabled (save_failed_videos_dir is empty)")

    # Set up JSON metrics save path (similar to run_pipeline.py structure)
    base_save_metrics_json_path = None

    if args.save_metrics_json and args.save_metrics_json.strip():
        if args.save_metrics_json == "auto":
            if house_ind is None:
                # For full dataset, save both per-house metrics (in loop) and aggregated metrics (at end)
                # Per-house metrics will be saved in the loop with auto-generated paths
                # Aggregated metrics will be saved at the end
                base_save_metrics_json_path = (
                    ASSETS_DIR
                    / "datagen"
                    / "grasp_test_v1"
                    / timestamp
                    / f"{scene_dataset}_{data_split}_all_metrics.json"
                )
                base_save_metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
                log.info(
                    f"💾 Aggregated metrics JSON (auto): {base_save_metrics_json_path.absolute()}"
                )
                log.info(
                    f"💾 Per-house metrics directory: {base_save_metrics_json_path.parent.absolute()}/"
                )
            # Note: Per-house metrics are always saved when args.save_metrics_json is set (handled in loop)
        else:
            # Custom path provided
            if house_ind is None:
                # For full dataset, use custom path for aggregated metrics
                # Per-house metrics will use auto-generated paths based on scene names
                base_save_metrics_json_path = Path(args.save_metrics_json)
                base_save_metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
                log.info(
                    f"💾 Aggregated metrics JSON (custom): {base_save_metrics_json_path.absolute()}"
                )
                log.info(f"💾 Per-house metrics will be saved with scene-specific names")
            # Note: Per-house metrics are always saved when args.save_metrics_json is set (handled in loop)
    else:
        log.info("💾 Metrics JSON saving disabled (save_metrics_json is empty)")

    log.info("=" * 80)

    # Aggregate metrics across all houses
    all_metrics = {
        "scene_dataset": scene_dataset,
        "data_split": data_split,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_houses": len(house_indices),
        "total_objects": 0,
        "total_grasps_tested": 0,
        "total_noncolliding_grasps": 0,
        "total_successful_grasps": 0,
        "success_rate": 0.0,
        "per_house_metrics": [],
    }

    # Track success for single house mode
    house_success = False

    # Track metrics file path for single house mode (for final summary)
    single_house_metrics_path = None

    # Refresh scene mapping before processing to ensure we have the latest state
    scene_mapping = get_scenes(scene_dataset, data_split)
    log.info(
        f"Refreshed scene mapping: {len(scene_mapping[data_split])} houses in {data_split} split"
    )
    log.info(f"House indices to process: {house_indices}")

    # Process each house
    for idx, current_house_ind in enumerate(house_indices):
        log.info(f"\n{'=' * 80}")
        log.info(f"Processing house {current_house_ind} ({idx + 1}/{len(house_indices)})")
        log.info(f"{'=' * 80}")

        # For single house mode, we already validated everything upfront, so skip redundant checks
        # For multi-house mode, validate each house in the loop
        if house_ind is None:
            # Multi-house mode: get scene path directly by index (ignore split)
            scene_path = get_scene_path_by_index(scene_dataset, current_house_ind)
            if scene_path is None or not scene_path.exists():
                log.warning(f"Skipping house {current_house_ind} - scene file not found")
                continue
        else:
            # Single house mode: already validated earlier, reuse the validated path
            scene_path = validated_scene_path
            # If somehow validated_scene_path is None, try direct lookup
            if scene_path is None:
                scene_path = get_scene_path_by_index(scene_dataset, current_house_ind)
                if scene_path is None or not scene_path.exists():
                    log.warning(f"Skipping house {current_house_ind} - scene file not found")
                    continue

        log.info(f"House {current_house_ind} found in mapping, proceeding with grasp evaluation...")

        # Install scene assets (objects and grasps) before loading metadata
        # This ensures the scene XML and metadata files exist
        # Note: For single house mode, this was already installed earlier, so skip here
        if house_ind is None:
            # Multi-house mode: install here for each house
            log.info(
                f"Installing scene assets (objects and grasps) for house {current_house_ind}..."
            )
            try:
                install_scene_with_objects_and_grasps_from_path(scene_path)
                log.info(f"Successfully installed scene assets for house {current_house_ind}")
            except Exception as e:
                log.error(f"Failed to install scene assets for {scene_path}: {e}")
                import traceback

                log.error(traceback.format_exc())
                log.warning(
                    f"Failed to install scene assets for house {current_house_ind}, but continuing anyway..."
                )
        else:
            # Single house mode: already installed earlier, skip redundant installation
            log.info(
                f"Scene assets already installed earlier for house {current_house_ind}, skipping..."
            )

        # Load scene metadata using the utility function which handles naming patterns
        from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

        scene_metadata = get_scene_metadata(str(scene_path))
        if scene_metadata is None:
            log.warning(
                f"Metadata file not found for {scene_path}. Tried standard and _physics_metadata.json patterns. Skipping."
            )
            # For single house mode, mark as failed if we skip the house
            if house_ind is not None:
                house_success = False
                log.error(
                    f"House {current_house_ind} skipped - metadata not found. Marking as failed."
                )
            continue

        log.info(f"Testing grasps for scene: {scene_path}")

        # Set up per-house save paths
        save_failed_videos_dir = None
        if base_save_failed_videos_dir is not None:
            # Create subdirectory for this house
            scene_name = scene_path.stem if isinstance(scene_path, Path) else Path(scene_path).stem
            save_failed_videos_dir = (
                base_save_failed_videos_dir / f"house_{current_house_ind}_{scene_name}"
            )
            save_failed_videos_dir.mkdir(parents=True, exist_ok=True)

        # Set up per-house metrics path - saves metrics after each object within this house
        # This works for both single house and multiple houses scenarios
        save_metrics_json_path = None
        if args.save_metrics_json and args.save_metrics_json.strip():
            # Per-house metrics path - always save per house when metrics saving is enabled
            # Metrics will be saved incrementally after each object via save_metrics_to_json()
            if args.save_metrics_json == "auto":
                scene_name = (
                    scene_path.stem if isinstance(scene_path, Path) else Path(scene_path).stem
                )
                save_metrics_json_path = (
                    ASSETS_DIR
                    / "datagen"
                    / "grasp_test_v1"
                    / timestamp
                    / f"{scene_name}_metrics.json"
                )
            else:
                # For custom path, append house/scene identifier to avoid overwriting when processing multiple houses
                scene_name = (
                    scene_path.stem if isinstance(scene_path, Path) else Path(scene_path).stem
                )
                custom_path = Path(args.save_metrics_json)
                # If it's a directory, create file inside it; otherwise use as base filename
                if custom_path.suffix == ".json":
                    # It's a file path, append scene name before extension
                    save_metrics_json_path = (
                        custom_path.parent / f"{custom_path.stem}_{scene_name}{custom_path.suffix}"
                    )
                else:
                    # It's a directory, create file inside it
                    save_metrics_json_path = custom_path / f"{scene_name}_metrics.json"
            save_metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
            log.info(
                f"💾 Per-house metrics JSON for house {current_house_ind}: {save_metrics_json_path.absolute()}"
            )
            log.info(f"   (Updates after each object)")

        # Run grasp tests for this house
        try:
            house_metrics = test_grasps_for_scene(
                scene_path=str(scene_path),
                scene_metadata=scene_metadata,
                offset_distance=offset_distance,
                max_grasps_per_object=max_grasps_per_object,
                task_horizon=task_horizon,
                use_passive_viewer=use_passive_viewer,
                task_type=args.task_type,
                save_failed_videos_dir=save_failed_videos_dir,
                save_metrics_json_path=save_metrics_json_path,
            )

            # Add house index to metrics
            house_metrics["house_ind"] = current_house_ind
            house_metrics["scene_name"] = (
                scene_path.stem if isinstance(scene_path, Path) else Path(scene_path).stem
            )

            # Verify metrics were saved (critical for SQS worker)
            metrics_saved = False
            if save_metrics_json_path is None:
                print(f"ERROR: METRICS_PATH_NONE: save_metrics_json_path is None", flush=True)
                log.info(f"ERROR: METRICS_PATH_NONE: save_metrics_json_path is None")
            elif save_metrics_json_path.exists():
                metrics_saved = True
                single_house_metrics_path = save_metrics_json_path  # Store for final summary
                file_size = save_metrics_json_path.stat().st_size
                # Print to stdout so it shows up in subprocess output
                print(
                    f"SUCCESS: METRICS_SAVED: {save_metrics_json_path.absolute()} (size: {file_size} bytes)",
                    flush=True,
                )
                log.info(
                    f"SUCCESS: METRICS_SAVED: {save_metrics_json_path.absolute()} (size: {file_size} bytes)"
                )
                log.info(f"✅ Verified metrics file exists: {save_metrics_json_path.absolute()}")
                log.info(f"💾 Metrics JSON saved to: {save_metrics_json_path.absolute()}")
                if file_size > 0:
                    log.info(f"   File size: {file_size} bytes")
            else:
                print(
                    f"ERROR: METRICS_NOT_FOUND: Expected path: {save_metrics_json_path.absolute()}",
                    flush=True,
                )
                log.info(
                    f"ERROR: METRICS_NOT_FOUND: Expected path: {save_metrics_json_path.absolute()}"
                )
                log.warning(f"Metrics file not found at expected path: {save_metrics_json_path}")

            # For single house mode, track success
            # Only mark as successful if we actually ran grasp evaluation AND metrics were saved (if required)
            if house_ind is not None:
                grasps_tested = house_metrics.get("total_grasps_tested", 0)
                metrics_required = args.save_metrics_json and args.save_metrics_json.strip()

                # Check all failure conditions - if any fail, house_success stays False
                if grasps_tested == 0:
                    print(
                        f"ERROR: HOUSE_FAILED: House {current_house_ind} processed but no grasps were tested",
                        flush=True,
                    )
                    log.info(
                        f"ERROR: HOUSE_FAILED: House {current_house_ind} processed but no grasps were tested"
                    )
                    house_success = False
                elif metrics_required and not metrics_saved:
                    print(
                        f"ERROR: HOUSE_FAILED: Metrics file was not saved for house {current_house_ind}",
                        flush=True,
                    )
                    log.info(
                        f"ERROR: HOUSE_FAILED: Metrics file was not saved for house {current_house_ind}"
                    )
                    log.info(
                        f"  Expected path: {save_metrics_json_path.absolute() if save_metrics_json_path else 'None'}"
                    )
                    log.info(
                        f"  Metrics required: {metrics_required}, Metrics saved: {metrics_saved}"
                    )
                    house_success = False
                else:
                    # Success: grasps were tested AND (metrics not required OR metrics were saved)
                    house_success = True
                    print(
                        f"SUCCESS: HOUSE_SUCCESS: House {current_house_ind} - {grasps_tested} grasps tested, metrics saved: {metrics_saved}",
                        flush=True,
                    )
                    log.info(
                        f"SUCCESS: HOUSE_SUCCESS: House {current_house_ind} - {grasps_tested} grasps tested, metrics saved: {metrics_saved}"
                    )
                    log.info(
                        f"House {current_house_ind} successfully processed: {grasps_tested} grasps tested"
                    )

            # Aggregate metrics
            all_metrics["total_objects"] += house_metrics["total_objects"]
            all_metrics["total_grasps_tested"] += house_metrics["total_grasps_tested"]
            all_metrics["total_noncolliding_grasps"] += house_metrics["total_noncolliding_grasps"]
            all_metrics["total_successful_grasps"] += house_metrics["total_successful_grasps"]
            all_metrics["per_house_metrics"].append(house_metrics)

            log.info(
                f"House {current_house_ind}: {house_metrics['total_successful_grasps']}/{house_metrics['total_grasps_tested']} successful "
                f"({house_metrics['success_rate'] * 100:.1f}%)"
            )

        except Exception as e:
            log.error(f"Error processing house {current_house_ind}: {e}")
            import traceback

            log.debug(f"Traceback:\n{traceback.format_exc()}")

            # For single house mode, mark as failed and return immediately
            if house_ind is not None:
                house_success = False
                return 1

            continue
        finally:
            # Clean up memory after each scene/house is processed
            # This ensures resources are freed before moving to the next scene
            log.debug(f"Cleaning up memory after house {current_house_ind}...")
            del scene_metadata
            gc.collect()
            log.debug(f"Memory cleanup completed for house {current_house_ind}")

    # Calculate overall success rate
    if all_metrics["total_grasps_tested"] > 0:
        all_metrics["success_rate"] = (
            all_metrics["total_successful_grasps"] / all_metrics["total_grasps_tested"]
        )

    # Save aggregated metrics if running full dataset
    if house_ind is None and base_save_metrics_json_path:
        serializable_metrics = convert_to_json_serializable(all_metrics)
        base_save_metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(base_save_metrics_json_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        log.info(f"💾 Aggregated metrics saved to: {base_save_metrics_json_path.absolute()}")
        log.info(f"   File size: {base_save_metrics_json_path.stat().st_size} bytes")

    # Print final summary with output locations
    log.info(f"\n{'=' * 80}")
    log.info("📊 OUTPUT FILES SUMMARY")
    log.info("=" * 80)
    if base_save_failed_videos_dir:
        log.info(f"🎥 Failed videos directory: {base_save_failed_videos_dir.absolute()}")
        video_count = sum(1 for _ in base_save_failed_videos_dir.rglob("*.mp4"))
        if video_count > 0:
            log.info(f"   Total video files: {video_count}")
    if base_save_metrics_json_path:
        log.info(f"💾 Aggregated metrics: {base_save_metrics_json_path.absolute()}")
    if args.save_metrics_json and args.save_metrics_json.strip():
        metrics_dir = (
            base_save_metrics_json_path.parent
            if base_save_metrics_json_path
            else (ASSETS_DIR / "datagen" / "grasp_test_v1" / timestamp)
        )
        json_count = sum(1 for _ in metrics_dir.glob("*_metrics.json"))
        if json_count > 0:
            log.info(f"💾 Per-house metrics files: {json_count} files in {metrics_dir.absolute()}")
        # For single house mode, also show the specific file path
        if (
            house_ind is not None
            and single_house_metrics_path
            and single_house_metrics_path.exists()
        ):
            log.info(
                f"💾 Metrics JSON for house {house_ind}: {single_house_metrics_path.absolute()}"
            )
            log.info(f"   File size: {single_house_metrics_path.stat().st_size} bytes")
    log.info("=" * 80)
    log.info(f"\n{'=' * 80}")
    log.info("FINAL SUMMARY")
    log.info(f"{'=' * 80}")
    log.info(
        f"Overall: {all_metrics['total_successful_grasps']}/{all_metrics['total_grasps_tested']} successful "
        f"({all_metrics['success_rate'] * 100:.1f}%)"
    )
    if all_metrics["total_noncolliding_grasps"] > 0:
        log.info(
            f"Non-colliding grasps: {all_metrics['total_noncolliding_grasps']}, "
            f"Success rate: {all_metrics['total_successful_grasps']}/{all_metrics['total_noncolliding_grasps']} "
            f"({all_metrics['total_successful_grasps'] / all_metrics['total_noncolliding_grasps'] * 100:.1f}% if all tested)"
        )
    log.info(
        f"Total houses processed: {len(all_metrics['per_house_metrics'])}/{all_metrics['total_houses']}"
    )
    log.info(f"Total objects tested: {all_metrics['total_objects']}")

    # Determine exit code based on success
    # For single house: return 0 if successful, 1 if failed
    # For multiple houses: return 0 if at least one house succeeded, 1 if all failed
    if house_ind is not None:
        # Single house mode - check if it succeeded
        print(
            f"DEBUG: Final check - house_success={house_success}, house_indices={len(house_indices)}, metrics_count={len(all_metrics['per_house_metrics'])}",
            flush=True,
        )
        log.info(
            f"DEBUG: Final check - house_success={house_success}, house_indices={len(house_indices)}, metrics_count={len(all_metrics['per_house_metrics'])}"
        )

        if len(house_indices) == 0:
            print(
                f"ERROR: EXIT_CODE_1: House {house_ind} was not processed - no houses in house_indices",
                flush=True,
            )
            log.info(
                f"ERROR: EXIT_CODE_1: House {house_ind} was not processed - no houses in house_indices"
            )
            return 1
        elif len(all_metrics["per_house_metrics"]) == 0:
            print(
                f"ERROR: EXIT_CODE_1: House {house_ind} was not processed - no metrics collected",
                flush=True,
            )
            log.info(
                f"ERROR: EXIT_CODE_1: House {house_ind} was not processed - no metrics collected"
            )
            return 1
        elif house_success:
            print(f"SUCCESS: EXIT_CODE_0: House {house_ind} processed successfully", flush=True)
            if single_house_metrics_path:
                print(
                    f"SUCCESS: METRICS_LOCATION: {single_house_metrics_path.absolute()}", flush=True
                )
                log.info(f"SUCCESS: METRICS_LOCATION: {single_house_metrics_path.absolute()}")
            else:
                print(
                    f"WARNING: METRICS_LOCATION_NONE: single_house_metrics_path is None", flush=True
                )
                log.info(f"WARNING: METRICS_LOCATION_NONE: single_house_metrics_path is None")
            log.info(f"SUCCESS: EXIT_CODE_0: House {house_ind} processed successfully")
            log.info(f"House {house_ind} processed successfully")
            return 0
        else:
            print(
                f"ERROR: EXIT_CODE_1: House {house_ind} processing failed (house_success=False)",
                flush=True,
            )
            log.info(
                f"ERROR: EXIT_CODE_1: House {house_ind} processing failed (house_success=False)"
            )
            return 1
    else:
        # Multiple houses mode - success if at least one house was processed
        if len(all_metrics["per_house_metrics"]) > 0:
            log.info("At least one house processed successfully")
            return 0
        else:
            log.error("No houses were processed successfully")
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test grasps for objects in a scene")
    parser.add_argument(
        "--scene_dataset",
        type=str,
        default="ithor",
        help="Scene dataset (ithor, procthor-10k, etc.)",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="train",
        help="Data split (train or test)",
    )
    parser.add_argument(
        "--house_ind",
        type=int,
        default=None,
        help="House index to test. If not provided, runs the entire dataset",
    )
    parser.add_argument(
        "--offset_distance",
        type=float,
        default=0.05,
        help="Distance to offset robot from grasp pose (meters)",
    )
    parser.add_argument(
        "--max_grasps_per_object",
        type=int,
        default=None,
        help="Maximum number of grasps to test per object (default: test all valid non-colliding grasps)",
    )
    parser.add_argument(
        "--task_horizon",
        type=int,
        default=1000,
        help="Task horizon (number of steps)",
    )
    parser.add_argument(
        "--use_passive_viewer",
        action="store_true",
        help="Use passive viewer",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["pick", "open_close", "both"],
        default="pick",
        help="Type of task to run: 'pick' for static objects, 'open_close' for jointed objects, 'both' for all (default: pick)",
    )
    parser.add_argument(
        "--save_failed_videos_dir",
        type=str,
        default="auto",
        help="Directory to save videos for failed grasps. Use 'auto' to auto-generate path in assets/datagen/grasp_test_v1/ (default: auto). Set to empty string to disable.",
    )
    parser.add_argument(
        "--save_metrics_json",
        type=str,
        default="auto",
        help="Path to save metrics JSON file. Use 'auto' to auto-generate path in assets/datagen/grasp_test_v1/ (default: auto). Set to empty string to disable.",
    )

    args = parser.parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
