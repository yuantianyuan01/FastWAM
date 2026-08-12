import json
import logging
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import h5py
import imageio
import numpy as np
import torch

from molmo_spaces.env.abstract_sensors import SensorSuite

log = logging.getLogger(__name__)

COMPR = "lzf"  # default H5 compression to use.


def is_camera_sensor(sensor_name: str, sensor_suite: SensorSuite | None = None) -> bool:
    """
    Determine if a sensor corresponds to a camera (RGB or depth) that produces image data.

    Uses sensor type metadata when available (preferred), falls back to naming heuristics
    for backward compatibility when sensor_suite is not provided.

    Args:
        sensor_name: Name of the sensor to check
        sensor_suite: Optional SensorSuite to query for sensor type metadata

    Returns:
        True if the sensor is a camera that produces image data (RGB or depth), False otherwise.
        Returns False for camera parameter sensors (CameraParameterSensor) which contain
        metadata but not image data.
    """
    # Import here to avoid circular dependency
    from molmo_spaces.env.sensors_cameras import CameraParameterSensor, CameraSensor, DepthSensor

    # Preferred approach: Check sensor type directly via SensorSuite
    if sensor_suite is not None and sensor_name in sensor_suite.sensors:
        sensor = sensor_suite.sensors[sensor_name]
        # Check if it's a CameraSensor or DepthSensor (but not CameraParameterSensor)
        return isinstance(sensor, CameraSensor | DepthSensor) and not isinstance(
            sensor, CameraParameterSensor
        )


def byte_array_to_string(bytes_to_decode: np.ndarray):
    return bytes(bytes_to_decode).rstrip(b"\x00").decode("utf-8")


def dict_to_byte_array(target_dict, sensor_name: str, str_max_len: int):
    data_string = json.dumps(target_dict, sort_keys=True, separators=(",", ":"))
    if len(data_string) > str_max_len:
        log.warning(
            f"Warning: Truncated JSON string to {str_max_len} characters for {sensor_name}. Data values may be missing."
        )
    byte_array = np.zeros(str_max_len, dtype=np.uint8)
    encoded = data_string.encode("utf-8")[:str_max_len]
    byte_array[: len(encoded)] = list(encoded)
    return byte_array


def _debug_structure(data, max_depth=2, current_depth=0):
    """Helper function to show the structure of nested dictionaries for debugging."""
    if current_depth >= max_depth:
        return "..."

    if isinstance(data, dict):
        return {
            k: _debug_structure(v, max_depth, current_depth + 1) for k, v in list(data.items())[:3]
        }
    elif isinstance(data, list):
        return f"list[{len(data)}]({type(data[0]).__name__ if data else 'empty'})"
    elif hasattr(data, "shape"):  # tensor or numpy array
        return f"{type(data).__name__}({data.shape})"
    else:
        return f"{type(data).__name__}"


def safe_to_tensor(data):
    """Safely convert data to tensor, handling different dimensionalities."""
    if isinstance(data, np.ndarray):
        # For 1D arrays (sensor data like poses, joint positions), convert directly to tensor
        if data.ndim == 1:
            return torch.from_numpy(data.copy())
        # For 2D/3D arrays, convert directly without torchvision transforms
        elif data.ndim >= 2:
            # Always preserve original format for images and other multi-dimensional data
            # This avoids torchvision's to_tensor which changes HWC->CHW and uint8->float32
            return torch.from_numpy(data.copy())
        else:
            return torch.from_numpy(data.copy())
    else:
        # Try to convert to numpy first, then to tensor
        try:
            return torch.from_numpy(np.array(data))
        except ValueError:
            # Last resort: convert to numpy array and then to tensor
            return torch.from_numpy(np.array(data, dtype=np.float32))


def convert_to_arr(observations: list[dict], sensor_suite: SensorSuite) -> list[dict]:
    for observation in observations:
        for sensor in observation:
            if sensor_suite.sensors[sensor].is_dict:
                observation[sensor] = dict_to_byte_array(
                    observation[sensor], sensor, sensor_suite.sensors[sensor].str_max_len
                )
    return observations


def batch_observations(
    observations: list[dict], sensor_suite: SensorSuite, device: torch.device | None = None
) -> dict[str, dict | torch.Tensor]:
    """Transpose a batch of observation dicts to a dict of batched
    observations.

    # Arguments

    observations :  List of dicts of observations.
    device : The torch.device to put the resulting tensors on.
        Will not move the tensors if None.

    # Returns

    Transposed dict of lists of observations.
    """

    def collect_arrays(observation: dict[str, Any]) -> dict[str, dict | list]:
        """Collect raw numpy arrays/data without converting to tensors yet."""
        if not isinstance(observation, dict):
            raise TypeError(f"Expected dict observation, got {type(observation)}: {observation}")

        batch_dict: defaultdict = defaultdict(list)

        for sensor in observation:
            if isinstance(observation[sensor], dict):
                # For nested dicts, recurse
                batch_dict[sensor] = collect_arrays(observation[sensor])
            else:
                # For leaf values, just add the raw data (don't convert to tensor yet)
                batch_dict[sensor].append(observation[sensor])

        return dict(batch_dict)

    def fill_arrays(input_batch: Any, observation: dict[str, Any]) -> None:
        """Fill batch structure with raw arrays."""
        for sensor in observation:
            if isinstance(observation[sensor], dict):
                fill_arrays(input_batch[sensor], observation[sensor])
            else:
                input_batch[sensor].append(observation[sensor])

    def stack_and_tensorize(input_batch: Any) -> None:
        """Stack numpy arrays first, then convert to tensor once (more efficient)."""
        for sensor in input_batch:
            if isinstance(input_batch[sensor], dict):
                stack_and_tensorize(input_batch[sensor])
            else:
                # Stack numpy arrays first
                data_list = input_batch[sensor]
                try:
                    # Try to stack as numpy first (much faster than stacking tensors)
                    if isinstance(data_list[0], np.ndarray):
                        stacked_numpy = np.stack(data_list, axis=0)
                        input_batch[sensor] = torch.from_numpy(stacked_numpy)
                    else:
                        # Fallback: convert each to numpy, then stack
                        numpy_arrays = [
                            np.array(d) if not isinstance(d, np.ndarray) else d for d in data_list
                        ]
                        stacked_numpy = np.stack(numpy_arrays, axis=0)
                        input_batch[sensor] = torch.from_numpy(stacked_numpy)

                    # Move to device if specified
                    if device is not None:
                        input_batch[sensor] = input_batch[sensor].to(device=device)
                except Exception as e:
                    # Fallback to old method if numpy stacking fails
                    log.warning(
                        f"Fast stacking failed for {sensor}, falling back to tensor stack: {e}"
                    )
                    input_batch[sensor] = torch.stack(
                        [
                            safe_to_tensor(batch).to(device=device)
                            if device
                            else safe_to_tensor(batch)
                            for batch in data_list
                        ],
                        dim=0,
                    )

    if len(observations) == 0:
        return cast(dict[str, dict | torch.Tensor], observations)

    observations = convert_to_arr(observations, sensor_suite)
    batch = collect_arrays(observations[0])

    for obs in observations[1:]:
        fill_arrays(batch, obs)

    stack_and_tensorize(batch)

    return cast(dict[str, dict | torch.Tensor], batch)


def prepare_episode_for_saving(
    history: dict,
    sensor_suite: SensorSuite,
    fps: float,
    save_dir: str | None = None,
    episode_idx: int = 0,
    save_file_suffix: str = "",
    remove_sensors_if_save_dir: bool = True,
) -> dict[str, torch.Tensor] | None:
    """
    Transform raw episode history into batched format ready for save_trajectories().

    Takes the output of task.get_history() and produces a single dict with all data
    batched along the time dimension.

    Args:
        history: Dict from task.get_history() containing:
            - "observations": List[List[Dict]] - [timestep][batch_idx][sensor_name]
            - "rewards": List[List[float]]
            - "terminals": List[List[bool]]
            - "truncateds": List[List[bool]]
            - "actions": List[...] (optional, currently unused)
            - "obs_scene": Dict (optional)
        sensor_suite: SensorSuite for observation processing
        save_dir: Optional directory to save videos immediately (before batching)
        episode_idx: Episode index for video filenames
        save_file_suffix: Optional suffix for video filenames
        remove_sensors_if_save_dir: remove camera-related sensors if video saved

    Returns:
        Dict[str, Tensor] with all data batched along time dimension, or None if no data
        Structure:
        {
            # Sensor observations (from batch_observations)
            "qpos": Tensor(T, ...),
            "tcp_pose": Tensor(T, 7),
            ...
            # Episode metadata (camera data removed if videos saved)
            "rewards": Tensor(T,),
            "terminals": Tensor(T,),
            "truncateds": Tensor(T,),
            "successes": Tensor(T,),
            "obs_scene": str (JSON),
        }
    """
    import gc

    observations_list = history.get("observations", [])

    if not observations_list or len(observations_list) == 0:
        log.info("No observation history to save")
        return None

    # Flatten batch dimension (extract first environment since batch_size=1)
    flattened_obs = [timestep_obs[0] for timestep_obs in observations_list]

    if not flattened_obs:
        log.info("No flattened observations to save")
        return None

    log.info(f"Preparing episode data: {len(flattened_obs)} timesteps")

    # Delete original observations_list to free memory immediately
    # This removes one complete copy of all episode data
    del observations_list
    history.pop("observations", None)
    gc.collect()

    # MEMORY OPTIMIZATION: Save videos BEFORE batching to avoid massive memory spike
    # Camera images are ~80% of episode memory. By saving videos now and removing
    # images from observations, we avoid creating giant tensor copies during batching.
    if save_dir is not None:
        log.debug(f"Saving videos before batching for episode {episode_idx}")
        # Use existing function to extract and save videos from raw observations
        save_videos_from_raw_observations(
            flattened_obs,
            save_dir,
            fps,
            episode_idx=episode_idx,
            save_file_suffix=save_file_suffix,
            sensor_suite=sensor_suite,
        )

        if remove_sensors_if_save_dir:
            # CRITICAL: Delete camera data (RGB and depth) from observations to avoid batching it
            # This is where the massive memory savings come from
            removed_sensors = set()
            for obs in flattened_obs:
                sensors_to_remove = []
                for sensor_name in obs:
                    # Check if this is a camera sensor (RGB or depth)
                    # Skip segmentation sensors as they're not videos
                    if is_camera_sensor(sensor_name, sensor_suite) and not sensor_name.endswith(
                        "_seg"
                    ):
                        sensors_to_remove.append(sensor_name)

                # Remove camera data
                for sensor_name in sensors_to_remove:
                    obs.pop(sensor_name, None)
                    removed_sensors.add(sensor_name)

            if removed_sensors:
                log.debug(
                    f"Removed camera sensors from observations before batching: {removed_sensors}"
                )

        gc.collect()

    # Batch observations: List[Dict] -> Dict[str, Tensor(T, ...)]
    # Note: Camera images already removed if save_dir was provided, so this is much smaller
    batched_data = batch_observations(flattened_obs, sensor_suite)

    # Delete flattened_obs after batching to free memory
    # This removes the intermediate flattened copy
    del flattened_obs
    gc.collect()

    # Add rewards if present
    if "rewards" in history:
        rewards_list = history["rewards"]
        # Flatten batch dimension directly into numpy array (avoid intermediate list)
        rewards_array = np.array(
            [timestep_reward[0] for timestep_reward in rewards_list], dtype=np.float32
        )
        batched_data["rewards"] = torch.from_numpy(rewards_array)
        del rewards_array  # Free numpy array after conversion
        history.pop("rewards", None)

    # Add terminals if present
    if "terminals" in history:
        terminals_list = history["terminals"]
        # Flatten batch dimension directly into numpy array (avoid intermediate list)
        terminals_array = np.array(
            [timestep_terminal[0] for timestep_terminal in terminals_list], dtype=bool
        )
        batched_data["terminals"] = torch.from_numpy(terminals_array)
        del terminals_array
        history.pop("terminals", None)

    # Add truncateds if present
    if "truncateds" in history:
        truncateds_list = history["truncateds"]
        # Flatten batch dimension directly into numpy array (avoid intermediate list)
        truncateds_array = np.array(
            [timestep_truncated[0] for timestep_truncated in truncateds_list], dtype=bool
        )
        batched_data["truncateds"] = torch.from_numpy(truncateds_array)
        del truncateds_array
        history.pop("truncateds", None)

    # Add successes if present
    if "successes" in history:
        successes_list = history["successes"]
        # Flatten batch dimension directly into numpy array (avoid intermediate list)
        successes_array = np.array(
            [timestep_success[0] for timestep_success in successes_list], dtype=bool
        )
        batched_data["successes"] = torch.from_numpy(successes_array)
        del successes_array
        history.pop("successes", None)

    # Add obs_scene if present
    if "obs_scene" in history:
        batched_data["obs_scene"] = json.dumps(history["obs_scene"])

    # Final GC to clean up any remaining references
    gc.collect()

    return batched_data


def _save_sensor_video(
    sensor_name: str,
    sensor_type: str,
    frames: np.ndarray | list[np.ndarray],
    video_path: str,
    fps: float,
    logger: logging.Logger | None = None,
) -> None:
    """
    Save a single sensor's video (RGB or depth).

    Single source of truth for all sensor video saving logic.
    Handles shape validation, dtype conversion, and routing to appropriate save function.

    Args:
        sensor_name: Name of the sensor (for error messages)
        sensor_type: Either "rgb" or "depth"
        frames: Video frames as numpy array or list of arrays
        video_path: Path to save the video
        fps: Frames per second
        logger: Optional logger instance
    """
    from molmo_spaces.utils.depth_utils import save_depth_video

    logger = logger or log

    # Stack frames if needed
    if isinstance(frames, list):
        frames = np.array(frames)

    if sensor_type == "depth":
        # Handle depth sensors
        if frames.dtype != np.float32:
            logger.warning(f"{sensor_name} has dtype {frames.dtype}, expected float32. Converting.")
            frames = frames.astype(np.float32)

        logger.debug(
            f"{sensor_name} (depth): shape={frames.shape}, "
            f"dtype={frames.dtype}, range=[{frames.min():.3f}, {frames.max():.3f}]m"
        )

        # Handle shape: should be (T, H, W) or (T, H, W, 1)
        if frames.ndim == 4 and frames.shape[-1] == 1:
            frames = frames.squeeze(-1)
            logger.debug(f"Squeezed {sensor_name} from 4D to 3D: {frames.shape}")
        elif frames.ndim != 3:
            raise ValueError(
                f"{sensor_name} has unexpected shape {frames.shape} for depth data. "
                f"Expected (T, H, W) or (T, H, W, 1). "
                f"Check the DepthSensor observation_space and data format."
            )

        # Save depth video
        save_depth_video(frames, video_path, fps=fps, logger=logger)

    elif sensor_type == "rgb":
        # Handle RGB sensors
        logger.debug(
            f"{sensor_name} (RGB): shape={frames.shape}, "
            f"dtype={frames.dtype}, range=[{frames.min():.3f}, {frames.max():.3f}]"
        )

        # Validate shape: should be (T, H, W, C)
        if frames.ndim != 4:
            raise ValueError(
                f"{sensor_name} has unexpected shape {frames.shape} for RGB data. "
                f"Expected (T, H, W, C)."
            )

        # Check if we have CHW format instead of HWC
        if frames.shape[-1] not in [1, 3, 4] and frames.shape[1] in [1, 3, 4]:
            # Convert from (T, C, H, W) to (T, H, W, C)
            frames = np.transpose(frames, (0, 2, 3, 1))
            logger.debug(f"Converted {sensor_name} from CHW to HWC: {frames.shape}")

        # Validate channel count
        if frames.shape[-1] not in [1, 3, 4]:
            raise ValueError(
                f"{sensor_name} has invalid channel count: {frames.shape[-1]}. "
                f"Expected 1, 3, or 4 channels."
            )

        # Convert to uint8 if needed
        if frames.dtype == np.float32 or frames.dtype == np.float64:
            # Check range - should be [0,1] for RGB
            min_val = float(frames.min())
            max_val = float(frames.max())
            # Allow small tolerance for floating point errors
            if min_val < -1e-6 or max_val > 1.0 + 1e-6:
                raise ValueError(
                    f"{sensor_name}: RGB values outside valid [0,1] range: [{min_val:.6f}, {max_val:.6f}]. "
                    f"This likely indicates depth data being treated as RGB. "
                    f"Check if sensor name ends with '_depth' and is being routed correctly."
                )
            # Clip to [0, 1] and convert to uint8
            frames = np.clip(frames, 0.0, 1.0)
            frames = (frames * 255).astype(np.uint8)
            logger.debug(f"Converted {sensor_name} from float32 [0,1] to uint8 [0,255]")
        elif frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
            logger.debug(f"Converted {sensor_name} to uint8")

        # Save RGB video
        save_frames_to_mp4(frames, video_path, fps=fps)
    else:
        raise ValueError(f"Unknown sensor_type: {sensor_type}. Expected 'rgb' or 'depth'.")


def save_frames_to_mp4(
    frames: Sequence[np.ndarray],
    file_path: str,
    fps: float,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    """Save RGB frames to MP4 video file.

    Low-level function that assumes frames are already validated and in uint8 format.
    Use _save_sensor_video() for high-level saving with validation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if not isinstance(frames, np.ndarray):
        frames = np.array(frames)

    # Frames should already be uint8 if coming from _save_sensor_video()
    if frames.dtype != np.uint8:
        log.warning(f"save_frames_to_mp4: Expected uint8, got {frames.dtype}. Converting...")
        if frames.dtype in [np.float32, np.float64]:
            frames = np.clip(frames, 0.0, 1.0)
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    # Ensure file path has .mp4 extension
    file_path = Path(file_path)
    if file_path.suffix != ".mp4":
        file_path = file_path.with_suffix(".mp4")

    kwargs = {
        "fps": fps,
        "quality": 5,
        **(extra_kwargs if extra_kwargs is not None else {}),
    }

    # Explicitly use ffmpeg plugin to avoid imageio selecting wrong plugin (e.g., tifffile)
    try:
        # Try using ffmpeg plugin explicitly
        writer = imageio.get_writer(
            file_path, format="ffmpeg", fps=fps, quality=5, **(extra_kwargs if extra_kwargs else {})
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        # Fallback to mimwrite if explicit writer fails
        # Common exceptions:
        # - ImportError: ffmpeg plugin not installed
        # - OSError: file system or ffmpeg execution issues
        # - ValueError: invalid parameters
        # - RuntimeError: ffmpeg runtime errors
        log.debug(f"FFmpeg writer failed ({type(e).__name__}: {e}), falling back to mimwrite")
        # Don't pass macro_block_size parameter as it's not supported in newer imageio versions
        imageio.mimwrite(file_path, frames, format="mp4", **kwargs)


def save_trajectories(
    episodes_data: list[dict[str, torch.Tensor]],
    save_dir: str,
    fps: float,
    save_file_suffix: str = "",
    save_mp4s: bool = True,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Save trajectories in the expected hierarchical HDF5 format.

    Args:
        episodes_data: List of batched observations (output of batch_observations())
                      Each episode is a Dict[str, torch.Tensor] where tensors have shape (T, ...)
        save_dir: Directory to save files
        fps: Frames per second of episode data
        save_file_suffix: Optional suffix for filenames
        save_mp4s: Whether to save MP4 videos
        logger: Optional logger to use (defaults to module logger)

    Expected structure:
    traj_N/
    ├── obs/
    │   ├── agent/
    │   │   ├── qpos (T,str_max_len)
    │   │   └── qvel (T,str_max_len)
    │   ├── extra/
    │   │   ├── obj_start (T,7)
    │   │   ├── obj_end (T,7)
    │   │   ├── tcp_pose (T,7)
    │   │   ├── grasp_pose (T,7)
    │   │   ├── robot_base_pose (T,7)
    │   │   └── door_state (T,str_max_len)
    │   │       ├── joint_angle
    │   │       ├── opening_percentage
    │   │       ├── handle_position
    │   │       ├── handle_extents
    │   │       ├── door_position
    │   │       └── is_open
    │   ├── sensor_param/
    │   │   └── render_camera/
    │   │       ├── extrinsic_cv (T,3,4)
    │   │       ├── cam2world_gl (T,4,4)
    │   │       └── intrinsic_cv (T,3,3)
    │   └── sensor_data/
    │       └── render_camera/
    │           ├── rgb (T,str_max_len) - video path
    │           ├── depth (T,str_max_len) - video path
    │           └── segmentation (T,str_max_len) - video path
    ├── actions (T,str_max_len) - flattened
    ├── extra/ - original formats for reference
    └── episode metadata...
    """
    logger = logger or log

    os.makedirs(save_dir, exist_ok=True)

    # Save HDF5 file
    hdf5_path = os.path.join(save_dir, f"trajectories{save_file_suffix}.h5")

    with h5py.File(hdf5_path, "w") as hdf5_file:
        for episode_idx, episode_data in enumerate(episodes_data):
            episode_group = hdf5_file.create_group(f"traj_{episode_idx}")

            logger.debug(
                f"\n[SAVE_UTILS DEBUG] Processing episode {episode_idx} with batched observations"
            )
            logger.debug(f"[SAVE_UTILS DEBUG] Available sensors: {list(episode_data.keys())}")

            # Create main obs group with expected structure
            obs_group = episode_group.create_group("obs")

            # Save agent data (qpos, qvel)
            _save_agent_data_from_batched(obs_group, episode_data)

            # Save extra data (pose sensors)
            _save_extra_data_from_batched(obs_group, episode_data)

            # Save sensor parameters (camera params)
            _save_sensor_params_from_batched(obs_group, episode_data)

            # Save sensor data (camera structure)
            _save_sensor_data_from_batched(
                obs_group, episode_data, episode_idx, save_dir, save_file_suffix, logger
            )

            # Save actions
            _save_actions_from_batched(episode_group, episode_data, logger)

            # Save environment states
            _save_env_states_from_batched(episode_group, episode_data)

            # Create placeholder episode metadata (since we don't have this in batched format)
            num_timesteps = len(episode_data["qpos"])

            if "terminateds" in episode_data:
                terminated_array = episode_data["terminateds"]
            else:
                logger.warning("No terminated recorded, assuming episode ended")
                # Create dummy terminated/truncated arrays - assume last step is terminal
                terminated_array = np.zeros(num_timesteps, dtype=bool)
                terminated_array[-1] = True  # Assume episode ended
            episode_group.create_dataset("terminated", data=terminated_array, compression=COMPR)

            if "truncateds" in episode_data:
                truncated_array = episode_data["truncateds"]
            else:
                logger.warning("No truncateds recorded, assuming untruncated")
                truncated_array = np.zeros(num_timesteps, dtype=bool)
            episode_group.create_dataset("truncated", data=truncated_array, compression=COMPR)

            # Create dummy rewards (could be made configurable)
            if "rewards" in episode_data:
                rewards_array = episode_data["rewards"]
            else:
                logger.warning("No rewards recorded, assuming success")
                rewards_array = np.zeros(num_timesteps, dtype=np.float32)
                rewards_array[-1] = 1.0  # Reward at end
            episode_group.create_dataset("rewards", data=rewards_array, compression=COMPR)

            # Save obs_scene
            if "obs_scene" in episode_data:
                obs_scene = episode_data["obs_scene"]
            else:
                logger.warning("No obs_scene recorded, using default")
                obs_scene = json.dumps({})
            episode_group.create_dataset(
                "obs_scene", data=obs_scene
            )  # don't compress scalar dataset

            # Success and fail arrays
            if "successes" in episode_data:
                success_array = episode_data["successes"]
            else:
                logger.warning("No successes recorded, assuming success at end")
                success_array = np.zeros(num_timesteps, dtype=bool)
                success_array[-1] = True  # Assume success

            fail_array = ~success_array
            episode_group.create_dataset("success", data=success_array, compression=COMPR)
            episode_group.create_dataset("fail", data=fail_array, compression=COMPR)

    logger.info(f"Saved {len(episodes_data)} episodes to: {os.path.abspath(save_dir)}")
    # logger.info(f"  HDF5 file: {hdf5_path}")

    # Videos should have been saved during prepare_episode_for_saving() before batching
    # This is required for memory optimization - camera data is removed before batching
    if save_mp4s:
        if len(episodes_data) > 0:
            # Verify that camera data was removed (indicates videos were already saved)
            first_episode = episodes_data[0]
            camera_sensors_in_batch = [s for s in first_episode if is_camera_sensor(s)]

            if camera_sensors_in_batch:
                raise RuntimeError(
                    f"Camera data still present in batched episodes: {camera_sensors_in_batch}. "
                    f"Videos must be saved via save_videos_from_raw_observations() before batching. "
                    f"Pass save_dir to prepare_episode_for_saving() to enable this."
                )

        logger.debug("Videos were saved during prepare_episode_for_saving() (before batching)")
    return Path(hdf5_path)


def _save_agent_data_from_batched(obs_group, episode_data) -> None:
    """Save agent qpos/qvel data from batched observations."""
    agent_group = obs_group.create_group("agent")

    if "qpos" in episode_data:
        if "qpos" in episode_data:
            qpos_tensor = episode_data["qpos"]
            qpos_numpy = qpos_tensor.detach().cpu().numpy()
            agent_group.create_dataset("qpos", data=qpos_numpy, compression=COMPR)
    else:
        log.warning("No qpos found in episode_data, cannot save qpos!")

    if "qvel" in episode_data or "qvel_dict" in episode_data:
        if "qvel" in episode_data:
            qvel_tensor = episode_data["qvel"]
            qvel_numpy = qvel_tensor.detach().cpu().numpy()
            agent_group.create_dataset("qvel", data=qvel_numpy, compression=COMPR)
    else:
        log.warning("No qvel found in episode_data, cannot save qvel!")


EXTRA_DATA_BLACKLIST = {
    # Agent data, saved by _save_agent_data_from_batched
    "qpos",
    "qvel",
    # Actions, saved by _save_actions_from_batched (episode_group["actions"])
    "actions",
    # Environment states, saved by _save_env_states_from_batched
    "env_states",
    # Episode-level metadata, saved directly in save_trajectories_from_batched
    "terminals",
    "truncateds",
    "rewards",
    "successes",
    "obs_scene",
}


def _is_reserved_extra_key(sensor_name: str) -> bool:
    """Check whether a key is already saved elsewhere and should be skipped in obs/extra."""
    if sensor_name in EXTRA_DATA_BLACKLIST:
        return True
    # Action sensors (e.g. "actions/joint_pos"), saved by _save_actions_from_batched
    if sensor_name.startswith("actions/"):
        return True
    # Camera parameters, saved by _save_sensor_params_from_batched
    if sensor_name.startswith("sensor_param_"):
        return True
    # Camera frame data (rgb/depth/segmentation), saved by _save_sensor_data_from_batched
    return sensor_name.endswith("_depth") or sensor_name.endswith("_seg") or "camera" in sensor_name


def _save_extra_data_from_batched(obs_group, episode_data) -> None:
    """Save extra task data (pose sensors) from batched observations."""
    extra_group = obs_group.create_group("extra")

    def _save_nested_data(data, group, name_prefix=""):
        """Recursively save nested dictionary data until hitting tensors."""
        if isinstance(data, dict):
            # Create subgroup and recurse
            if name_prefix:
                subgroup = group.create_group(name_prefix)
            else:
                subgroup = group
            for key, value in data.items():
                _save_nested_data(value, subgroup, key)
        elif isinstance(data, torch.Tensor):
            data_numpy = data.detach().cpu().numpy()
            group.create_dataset(name_prefix, data=data_numpy, compression=COMPR)
        else:
            # Handle other data types (lists, primitives, etc.)
            if isinstance(data, np.ndarray):
                group.create_dataset(name_prefix, data=data, compression=COMPR)
            else:
                # Convert to numpy if possible
                try:
                    data_numpy = np.array(data)
                    group.create_dataset(name_prefix, data=data_numpy, compression=COMPR)
                except Exception as e:
                    log.warning(f"Could not save data for {name_prefix}: {type(data)}, error: {e}")

    for sensor_name in episode_data:
        if _is_reserved_extra_key(sensor_name):
            continue
        # Use recursive loop for all sensors - handles both simple tensors and nested dicts
        sensor_data = episode_data[sensor_name]
        _save_nested_data(sensor_data, extra_group, sensor_name)


def _save_sensor_params_from_batched(obs_group, episode_data) -> None:
    """Save sensor parameters (camera params) from batched observations."""
    sensor_param_group = None

    for sensor_name, param_sensors in episode_data.items():
        if not sensor_name.startswith("sensor_param_"):
            continue

        if sensor_param_group is None:
            sensor_param_group = obs_group.create_group("sensor_param")

        camera_name = sensor_name.replace("sensor_param_", "")
        camera_param_group = sensor_param_group.create_group(camera_name)
        for param_type, sensor_tensor in param_sensors.items():
            sensor_numpy = sensor_tensor.detach().cpu().numpy()
            camera_param_group.create_dataset(param_type, data=sensor_numpy, compression=COMPR)


def _save_sensor_data_from_batched(
    obs_group, episode_data, episode_idx=None, save_dir=None, save_file_suffix="", logger=None
) -> None:
    """Save sensor data groups (camera structure) from batched observations.

    Note: RGB data is intentionally skipped to keep HDF5 file sizes manageable.
    Only depth and segmentation data are saved if present.

    Args:
        obs_group: HDF5 group to save sensor data to
        episode_data: Batched episode data
        episode_idx: Episode index for generating video paths
        save_dir: Directory where videos are saved
        save_file_suffix: Suffix for video filenames
        logger: Logger instance
    """
    logger = logger or log
    sensor_data_group = obs_group.create_group("sensor_data")

    # Find camera sensors (depth, segmentation) - SKIP RGB to reduce file size
    camera_sensors = {}
    for sensor_name in episode_data:
        logger.debug(f"Sensor name: {sensor_name}")
        # Skip RGB sensors to reduce HDF5 file size
        # RGB sensors: exo_camera_X, wrist_camera
        # if (sensor_name.startswith("exo_camera") and not ("_depth" in sensor_name or "_seg" in sensor_name)) or \
        #    sensor_name.startswith("wrist_camera"):
        #     camera_sensors[sensor_name] = "rgb"

        # Depth sensors
        if sensor_name.endswith("_depth"):
            camera_sensors[sensor_name] = "depth"
        # Segmentation sensors
        elif sensor_name.endswith("_seg"):
            camera_sensors[sensor_name] = "segmentation"
        elif (
            "camera" in sensor_name and "sensor_param" not in sensor_name
        ):  # TODO: this is a hack to get the rgb camera
            camera_sensors[sensor_name] = "rgb"

    # Group cameras by base name
    camera_groups = {}
    for sensor_name, sensor_type in camera_sensors.items():
        # Extract camera name from sensor name
        if sensor_name.endswith("_depth"):
            camera_name = sensor_name[:-6]  # Remove '_depth' suffix
        elif sensor_name.endswith("_seg"):
            camera_name = sensor_name[:-4]  # Remove '_seg' suffix
        else:
            camera_name = sensor_name  # Use as-is for other sensors

        if camera_name not in camera_groups:
            camera_groups[camera_name] = {}
        camera_groups[camera_name][sensor_type] = sensor_name
    # Create groups for each camera and save the actual frame data
    for camera_name, sensors in camera_groups.items():
        # Save frame data for each sensor type
        for sensor_type, sensor_name in sensors.items():
            if sensor_type == "rgb":
                # RGB: save video filename reference
                video_filename = f"episode_{episode_idx:08d}_{camera_name}{save_file_suffix}.mp4"
                # Convert string to byte array
                filename_bytes = video_filename.encode("utf-8")
                byte_array = np.zeros(100, dtype=np.uint8)  # Fixed size byte array
                byte_array[: len(filename_bytes)] = list(filename_bytes)
                sensor_data_group.create_dataset(
                    camera_name, data=byte_array, dtype=np.uint8, compression=COMPR
                )
            elif sensor_type == "depth":
                # Depth: save video filename reference (depth videos are now saved like RGB)
                video_filename = f"episode_{episode_idx:08d}_{sensor_name}{save_file_suffix}.mp4"
                filename_bytes = video_filename.encode("utf-8")
                byte_array = np.zeros(100, dtype=np.uint8)
                byte_array[: len(filename_bytes)] = list(filename_bytes)
                # Use full sensor name (e.g., "exo_camera_1_depth") as dataset name
                sensor_data_group.create_dataset(
                    sensor_name, data=byte_array, dtype=np.uint8, compression=COMPR
                )
            else:
                # Segmentation or other: save as numpy array
                sensor_tensor = episode_data[sensor_name]
                sensor_numpy = sensor_tensor.detach().cpu().numpy()
                sensor_data_group.create_dataset(sensor_name, data=sensor_numpy, compression=COMPR)


def _save_actions_from_batched(episode_group, episode_data, logger=None) -> None:
    """Save actions from batched observations."""
    logger = logger or log

    action_keys: list[str] = [k for k in episode_data if k.startswith("actions/")]

    actions_group = episode_group.create_group("actions")

    if action_keys:
        for action_key in action_keys:
            actions_tensor = episode_data[action_key]
            action_key_bn = action_key.replace("actions/", "")
            actions_numpy = actions_tensor.detach().cpu().numpy()
            actions_group.create_dataset(action_key_bn, data=actions_numpy, compression=COMPR)
            logger.debug(f"[SAVE_UTILS] Successfully saved {action_key_bn} from batched data")
    else:
        logger.warning("[SAVE_UTILS] No actions found in batched data, leaving actions group empty")


def _save_env_states_from_batched(episode_group, episode_data) -> None:
    """Save environment states from batched observations."""
    env_states_group = episode_group.create_group("env_states")

    if "env_states" not in episode_data:
        # Create empty groups
        env_states_group.create_group("actors")
        articulations_group = env_states_group.create_group("articulations")
        num_timesteps = len(episode_data[list(episode_data.keys())[0]])
        dummy_articulation_state = np.zeros((num_timesteps, 31), dtype=np.float32)
        articulations_group.create_dataset(
            "panda", data=dummy_articulation_state, compression=COMPR
        )
        return

    # Handle env_states tensor - need to decode JSON from each timestep
    env_states_tensor = episode_data["env_states"]
    env_states_numpy = env_states_tensor.detach().cpu().numpy()

    # Decode JSON data from each timestep
    env_states_data = []
    for timestep_data in env_states_numpy:
        # Find the end of the actual data (before null padding)
        actual_length = np.where(timestep_data == 0)[0]
        if len(actual_length) > 0:
            timestep_data = timestep_data[: actual_length[0]]

        json_string = bytes(timestep_data.astype(np.uint8)).decode("utf-8")
        parsed_data = json.loads(json_string)
        env_states_data.append(parsed_data)

    # Create actors group
    actors_group = env_states_group.create_group("actors")

    # Collect all unique actor names across all timesteps
    all_actor_names = set()
    for env_data in env_states_data:
        all_actor_names.update(env_data.get("actors", {}).keys())

    # Create datasets for each discovered actor
    for actor_name in sorted(all_actor_names):
        actor_data = []
        for env_data in env_states_data:
            if actor_name in env_data.get("actors", {}):
                actor_data.append(env_data["actors"][actor_name])
            else:
                actor_data.append([0.0] * 13)  # Fallback zeros if missing

        actors_group.create_dataset(
            actor_name, data=np.array(actor_data, dtype=np.float32), compression=COMPR
        )

    # Create articulations group
    articulations_group = env_states_group.create_group("articulations")
    articulation_data = []
    for env_data in env_states_data:
        if "panda" in env_data.get("articulations", {}):
            articulation_data.append(env_data["articulations"]["panda"])
        else:
            articulation_data.append([0.0] * 31)  # Fallback zeros

    articulations_group.create_dataset(
        "panda", data=np.array(articulation_data, dtype=np.float32), compression=COMPR
    )


def save_videos_from_raw_observations(
    observations_list,
    save_dir,
    fps,
    episode_idx=0,
    save_file_suffix="",
    sensor_suite: SensorSuite | None = None,
) -> None:
    """
    Save videos immediately from raw observations before batch processing.
    This avoids the corruption that happens during batch_observations tensor conversion.

    Args:
        observations_list: List of raw observation dicts from episode steps
        save_dir: Directory to save videos
        fps: Frames per second of episode data
        episode_idx: Episode index for naming
        save_file_suffix: Optional suffix for filenames
        sensor_suite: Optional SensorSuite for proper sensor type detection
    """
    os.makedirs(save_dir, exist_ok=True)

    if not observations_list:
        log.warning("No observations to save videos from")
        return

    # Find all camera sensors in the first observation
    camera_sensors = {}
    for sensor_name in observations_list[0]:
        if not is_camera_sensor(sensor_name, sensor_suite):
            continue

        # Categorize by type
        # Check depth FIRST (before RGB) to correctly identify depth cameras
        # e.g., "wrist_camera_depth" should be "depth" not "rgb"
        if sensor_name.endswith("_depth"):
            camera_sensors[sensor_name] = "depth"
        # Skip segmentation sensors (not saved as videos)
        elif sensor_name.endswith("_seg"):
            continue
        else:
            # All other camera sensors are RGB
            camera_sensors[sensor_name] = "rgb"

    log.debug(
        f"INFO: Saving videos for episode {episode_idx} with {len(camera_sensors)} cameras "
        f"({sum(1 for t in camera_sensors.values() if t == 'rgb')} RGB, "
        f"{sum(1 for t in camera_sensors.values() if t == 'depth')} depth) "
        f"and {len(observations_list)} frames"
    )
    log.debug(f"Camera sensors detected: {camera_sensors}")

    # Extract and save video for each camera
    for sensor_name, sensor_type in camera_sensors.items():
        # Extract frames from all observations for this camera
        frames = []
        for obs in observations_list:
            if sensor_name in obs:
                frame_data = obs[sensor_name]
                if isinstance(frame_data, np.ndarray):
                    # Remove batch dimension if present
                    if frame_data.ndim == 4 and frame_data.shape[0] == 1:
                        frame_data = frame_data[0]
                    frames.append(frame_data)

        if frames:
            # Generate video path
            video_path = os.path.join(
                save_dir, f"episode_{episode_idx:08d}_{sensor_name}{save_file_suffix}.mp4"
            )

            # Use unified video saving function (handles all validation and conversion)
            _save_sensor_video(
                sensor_name=sensor_name,
                sensor_type=sensor_type,
                frames=frames,
                video_path=video_path,
                fps=fps,
                logger=log,
            )

            log.debug(f"SUCCESS: Saved {sensor_type} video: {video_path} ({len(frames)} frames)")
        else:
            log.warning(f"WARNING: No frames found for camera {sensor_name}")
