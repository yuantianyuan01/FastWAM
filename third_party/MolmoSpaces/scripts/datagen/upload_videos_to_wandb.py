#!/usr/bin/env python3
"""Script to upload existing evaluation videos to wandb in the same format as eval_main.py.

This script creates a wandb table with task instructions, success/fail status, and videos,
matching the format used by log_eval_results_to_wandb in eval_main.py.

Usage:
    python upload_eval_videos_to_wandb.py \\
        --eval_dir /path/to/eval_output/.../20260201_211647 \\
        --wandb_project mujoco-thor-pick-and-place-eval \\
        --wandb_run_name my_eval_run
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import wandb

from molmo_spaces.utils.eval_utils import compose_episode_videos



@dataclass
class EpisodeResult:
    """Episode result information."""
    house_id: str
    episode_idx: int
    success: bool
    task_instruction: Optional[str] = None
    failure_category: Optional[str] = None  # One of: "placement_failed", "failed_to_grasp", "never_touched_target", or None if successful



def _extract_goal_from_h5(video_dir: Path, traj_idx: int = 0) -> Optional[str]:
    """Extract goal/task_description from the trajectory h5 file in the same directory as the video."""
    # Look for trajectory h5 files in the video directory
    h5_files = list(video_dir.glob("trajectories*.h5"))
    if not h5_files:
        return None

    h5_path = h5_files[0]
    try:
        with h5py.File(h5_path, "r") as f:
            traj_key = f"traj_{traj_idx}"
            if traj_key in f and "obs_scene" in f[traj_key]:
                obs_scene_data = f[traj_key]["obs_scene"][()]
                if isinstance(obs_scene_data, bytes):
                    obs_scene_str = obs_scene_data.decode("utf-8")
                else:
                    obs_scene_str = str(obs_scene_data)
                obs_scene = json.loads(obs_scene_str)
                return obs_scene.get("task_description", None)
    except Exception as e:
        print(f"Error extracting goal from {h5_path}: {e}")

    return None


def _decode_dict_data(key_data):
    """Decode JSON-encoded dictionary data from HDF5.

    Args:
        key_data: HDF5 dataset containing byte arrays

    Returns:
        List of decoded dictionaries, one per timestep
    """
    decoded_data = []
    for i in range(key_data.shape[0]):
        byte_array = key_data[i]
        json_string = byte_array.tobytes().decode('utf-8').rstrip('\x00')
        try:
            trajectory_dict = json.loads(json_string)
            decoded_data.append(trajectory_dict)
        except json.JSONDecodeError as e:
            decoded_data.append({})
    return decoded_data


def _extract_failure_category_from_h5(video_dir: Path, traj_idx: int = 0) -> Optional[str]:
    """Extract failure category from the trajectory h5 file.

    Returns one of:
    - "placement_failed": Picked right object but placement failed
    - "failed_to_grasp": Touched target object but never held it
    - "never_touched_target": Never touched the target object
    - None: If successful or cannot determine
    """
    h5_files = list(video_dir.glob("trajectories*.h5"))
    if not h5_files:
        return None

    h5_path = h5_files[0]
    try:
        with h5py.File(h5_path, "r") as f:
            traj_key = f"traj_{traj_idx}"
            if traj_key not in f:
                return None

            traj = f[traj_key]

            # Check success status first
            success = False
            if "success" in traj:
                success_data = traj["success"][()]
                if hasattr(success_data, "__len__") and len(success_data) > 0:
                    success = bool(np.any(success_data))
                else:
                    success = bool(success_data)

            # If successful, no failure category
            if success:
                return None

            # Analyze grasp states
            ever_held_pickup_obj = False
            ever_touched_pickup_obj = False

            if "obs" in traj and "extra" in traj["obs"]:
                extra_data = traj["obs"]["extra"]

                # Decode grasp_state_pickup_obj
                if "grasp_state_pickup_obj" in extra_data:
                    grasp_pickup_data = _decode_dict_data(extra_data["grasp_state_pickup_obj"])
                    for step_data in grasp_pickup_data:
                        if isinstance(step_data, dict):
                            # Check all grippers
                            for gripper_id, grasp_info in step_data.items():
                                if isinstance(grasp_info, dict):
                                    held = grasp_info.get("held", False)
                                    touching = grasp_info.get("touching", False)
                                    if held:
                                        ever_held_pickup_obj = True
                                        ever_touched_pickup_obj = True
                                    elif touching:
                                        ever_touched_pickup_obj = True

            # Determine failure category
            if ever_held_pickup_obj:
                return "placement_failed"
            elif ever_touched_pickup_obj:
                return "failed_to_grasp"
            else:
                return "never_touched_target"

    except Exception as e:
        print(f"Error extracting failure category from {h5_path}: {e}")

    return None


def _extract_success_from_h5(video_dir: Path, traj_idx: int = 0) -> Optional[bool]:
    """Extract success status from the trajectory h5 file.

    Checks if success occurred at ANY time during the rollout (not just at the end),
    matching the behavior of collect_info function.
    """
    h5_files = list(video_dir.glob("trajectories*.h5"))
    if not h5_files:
        return None

    h5_path = h5_files[0]
    try:
        with h5py.File(h5_path, "r") as f:
            traj_key = f"traj_{traj_idx}"
            if traj_key not in f:
                return None

            traj = f[traj_key]
            # Check for success-related keys
            if "success" in traj:
                success_data = traj["success"][()]
                # Check if ANY step was successful (matching collect_info behavior)
                if hasattr(success_data, "__len__") and len(success_data) > 0:
                    # Use np.any() to check if success occurred at any time during rollout
                    return bool(np.any(success_data))
                else:
                    return bool(success_data)
    except Exception as e:
        print(f"Error extracting success from {h5_path}: {e}")

    return None


def collect_episode_results(eval_dir: Path) -> list[EpisodeResult]:
    """Collect episode results by scanning the evaluation directory.

    Args:
        eval_dir: Directory containing evaluation outputs (with house_*/ subdirectories)

    Returns:
        List of EpisodeResult objects
    """
    results = []

    # Scan all house directories
    for house_dir in sorted(eval_dir.glob("house_*")):
        if not house_dir.is_dir():
            continue

        house_id = house_dir.name

        # Find all episode videos to determine episode indices
        episode_videos = list(house_dir.glob("episode_*.mp4"))
        if not episode_videos:
            continue

        # Extract unique episode indices
        episode_indices = set()
        for video_path in episode_videos:
            match = re.match(r"episode_(\d+)_", video_path.name)
            if match:
                episode_indices.add(int(match.group(1)))

        # For each episode, extract task instruction, success, and failure category
        for episode_idx in sorted(episode_indices):
            # Extract task instruction from h5 file
            task_instruction = _extract_goal_from_h5(house_dir, traj_idx=episode_idx)

            # Read success status from h5 file only
            success = _extract_success_from_h5(house_dir, traj_idx=episode_idx)

            # If not found in h5, default to False
            if success is None:
                success = False

            # Extract failure category
            failure_category = _extract_failure_category_from_h5(house_dir, traj_idx=episode_idx)

            results.append(EpisodeResult(
                house_id=house_id,
                episode_idx=episode_idx,
                success=success,
                task_instruction=task_instruction or "Unknown task",
                failure_category=failure_category,
            ))

    return results


def log_eval_results_to_wandb(
    results: list[EpisodeResult],
    composed_videos: dict[str, Path],
):
    """Log evaluation results to wandb as a table, matching the format from eval_main.py.

    Args:
        results: List of episode results
        composed_videos: Dict mapping episode keys (e.g., "house_2/episode_00000000") to composed video paths
    """
    # Create wandb table with columns: episode, task_instruction, success, failure_category, video
    table_data = []

    for result in results:
        episode_key = f"{result.house_id}/episode_{result.episode_idx:08d}"

        # Get composed video if available
        video_path = composed_videos.get(episode_key)
        wandb_video = None
        if video_path and video_path.exists():
            wandb_video = wandb.Video(str(video_path), format="mp4")

        # Format failure category for display
        failure_category_display = result.failure_category or ("N/A" if result.success else "Unknown")

        table_data.append([
            episode_key,
            result.task_instruction or "Unknown task",
            "Success" if result.success else "Failed",
            failure_category_display,
            wandb_video,
        ])

    # Create and log the table
    table = wandb.Table(
        columns=["episode", "task_instruction", "success", "failure_category", "video"],
        data=table_data,
    )

    wandb.log({"eval/episode_results": table})

    # Also log summary metrics
    success_count = sum(1 for r in results if r.success)
    total_count = len(results)
    success_rate = success_count / total_count if total_count > 0 else 0.0

    # Count failure categories
    placement_failed_count = sum(1 for r in results if r.failure_category == "placement_failed")
    failed_to_grasp_count = sum(1 for r in results if r.failure_category == "failed_to_grasp")
    never_touched_count = sum(1 for r in results if r.failure_category == "never_touched_target")

    wandb.log({
        "eval/success_count": success_count,
        "eval/total_count": total_count,
        "eval/success_rate": success_rate,
        "eval/failure_placement_failed": placement_failed_count,
        "eval/failure_failed_to_grasp": failed_to_grasp_count,
        "eval/failure_never_touched_target": never_touched_count,
    })


def upload_videos_to_wandb(
    eval_dir: Path,
    wandb_project: str = "json-eval",
    wandb_run_name: Optional[str] = None,
    camera_names: Optional[list[str]] = None,
):
    """Upload evaluation videos to wandb in the same format as eval_main.py.

    This creates a wandb table with task instructions, success/fail status, and videos.

    Args:
        eval_dir: Directory containing evaluation outputs (with house_*/ subdirectories and optionally composed/)
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name (defaults to eval_dir name)
        camera_names: List of camera names (e.g., ["exo_camera_1", "wrist_camera"]) for finding composed videos
    """
    eval_dir = Path(eval_dir).resolve()
    if not eval_dir.exists():
        raise ValueError(f"Evaluation directory does not exist: {eval_dir}")

    # Initialize wandb
    if wandb_run_name is None:
        wandb_run_name = eval_dir.name

    wandb.init(project=wandb_project, name=wandb_run_name)

    # Collect episode results
    print("Collecting episode results...")
    episode_results = collect_episode_results(eval_dir)
    print(f"Found {len(episode_results)} episodes")

    # Build success status map and find composed videos
    success_status = {f"{r.house_id}/episode_{r.episode_idx:08d}": r.success for r in episode_results}

    # Find composed videos
    composed_videos = {}
    composed_dir = eval_dir / "composed"

    # Check which episodes already have composed videos
    missing_episodes = set()
    if composed_dir.exists():
        # Look for composed videos matching the episode keys
        for result in episode_results:
            episode_key = f"{result.house_id}/episode_{result.episode_idx:08d}"
            house_id = result.house_id
            episode_idx = result.episode_idx

            # Prefer success/failed versions if they exist, otherwise use regular composed
            success_video = composed_dir / f"{house_id}_episode_{episode_idx:08d}_composed_success.mp4"
            failed_video = composed_dir / f"{house_id}_episode_{episode_idx:08d}_composed_failed.mp4"
            regular_video = composed_dir / f"{house_id}_episode_{episode_idx:08d}_composed.mp4"

            if success_video.exists():
                composed_videos[episode_key] = success_video
            elif failed_video.exists():
                composed_videos[episode_key] = failed_video
            elif regular_video.exists():
                composed_videos[episode_key] = regular_video
            else:
                missing_episodes.add(episode_key)
    else:
        # No composed directory exists, mark all episodes as missing
        missing_episodes = {f"{r.house_id}/episode_{r.episode_idx:08d}" for r in episode_results}

    # If there are missing composed videos and camera_names are provided, compose them
    if missing_episodes and camera_names and compose_episode_videos is not None:
        print(f"Found {len(composed_videos)} existing composed videos")
        print(f"Composing {len(missing_episodes)} missing videos...")


        # Create composed directory if it doesn't exist
        composed_dir.mkdir(parents=True, exist_ok=True)

        try:
            newly_composed = compose_episode_videos(
                eval_dir=eval_dir,
                camera_names=list(camera_names.split(" ")),  # Only regular cameras, ignoring attention frames
                success_status=success_status,
            )

            # Add newly composed videos to our dict
            for episode_key, video_path in newly_composed.items():
                if episode_key in missing_episodes:
                    composed_videos[episode_key] = video_path
                    print(f"  Composed video for {episode_key}")
        except Exception as e:
            print(f"Warning: Failed to compose some videos: {e}")
            print("  Continuing with existing composed videos only...")


    print(f"Total composed videos available: {len(composed_videos)}")

    # Log to wandb in the same format as eval_main.py
    log_eval_results_to_wandb(
        results=episode_results,
        composed_videos=composed_videos,
    )

    print("Upload complete!")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Upload existing evaluation videos to wandb in the same format as eval_main.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Path to evaluation output directory (e.g., eval_output/.../20260201_211647)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mujoco-thor-pick-and-place-eval",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (defaults to eval_dir name)",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default="exo_camera_1 wrist_camera",
        help="Camera names (e.g., exo_camera_1 wrist_camera). Required for composing videos if composed/ folder doesn't exist.",
    )

    args = parser.parse_args()

    upload_videos_to_wandb(
        eval_dir=Path(args.eval_dir),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        camera_names=args.camera_names,
    )


if __name__ == "__main__":
    main()
