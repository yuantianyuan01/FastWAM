"""Evaluation utilities for logging stats and videos to wandb."""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Result from a single evaluation episode.

    Attributes:
        episode_idx: Index of the episode within its house.
        house_id: House identifier (int or str like "house_5").
        success: Whether the episode was successful (at end of episode).
        num_steps: Number of steps taken in the episode.
        task_description: Natural language task description.
        object_name: Name of the target object (if applicable).
        seed: Random seed used for the episode.
        data_file_path: Path to the HDF5 file containing this episode's data.
            Use this together with episode_idx to uniquely identify an episode,
            especially when there are multiple batches per house.
        oracle_done: Whether success was achieved at ANY point during the episode.
        metadata: Additional metadata about the episode.
    """

    episode_idx: int
    house_id: int | str
    success: bool
    num_steps: int
    task_description: str | None = None
    object_name: str | None = None
    seed: int | None = None
    data_file_path: Path | None = None
    oracle_done: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    """Load frames from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (list of frames as numpy arrays in RGB format, fps)
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames, fps


def compose_videos_side_by_side(
    video_paths: list[Path],
    output_path: Path,
    target_height: int = 368,
    target_width: int = 1280,
) -> Path | None:
    """Compose multiple videos side-by-side into a single video.

    Args:
        video_paths: List of paths to input videos
        output_path: Path for the output composed video
        target_height: Target height for the output video
        target_width: Target width for the output video

    Returns:
        Path to the composed video, or None if failed
    """
    import cv2

    from molmo_spaces.utils.save_utils import save_frames_to_mp4

    if not video_paths:
        return None

    # Load all videos
    all_frames = []
    fps = None
    for vp in video_paths:
        if not vp.exists():
            log.warning(f"Video not found: {vp}")
            return None
        frames, video_fps = load_video_frames(vp)
        if not frames:
            log.warning(f"No frames in video: {vp}")
            return None
        all_frames.append(frames)
        if fps is None:
            fps = video_fps

    # Find the minimum number of frames across all videos
    min_frames = min(len(frames) for frames in all_frames)

    # Truncate all videos to the same length
    all_frames = [frames[:min_frames] for frames in all_frames]

    # Resize each video to fit within the target dimensions split equally
    n_videos = len(all_frames)
    per_video_width = target_width // n_videos
    resized_frames = []
    for frames in all_frames:
        h, w = frames[0].shape[:2]
        # Scale to fit target_height, then clamp width to per_video_width
        scale = min(target_height / h, per_video_width / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        resized = [cv2.resize(f, (new_w, new_h)) for f in frames]
        resized_frames.append(resized)
    all_frames = resized_frames

    # Compose frames side-by-side, padding to exact target dimensions
    composed_frames = []
    for frame_idx in range(min_frames):
        row_frames = [frames[frame_idx] for frames in all_frames]
        composed = np.concatenate(row_frames, axis=1)
        # Pad to exact target size if needed
        ch, cw = composed.shape[:2]
        if ch != target_height or cw != target_width:
            padded = np.zeros((target_height, target_width, 3), dtype=composed.dtype)
            padded[:ch, :cw] = composed
            composed = padded
        composed_frames.append(composed)

    # Save composed video using save_frames_to_mp4 (same as rest of codebase)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stacked_array = np.array(composed_frames)

    # Ensure uint8 format
    if stacked_array.dtype != np.uint8:
        if stacked_array.max() <= 1.0:
            stacked_array = (stacked_array * 255).astype(np.uint8)
        else:
            stacked_array = stacked_array.astype(np.uint8)

    save_frames_to_mp4(stacked_array, str(output_path), fps=fps or 30.0)

    return output_path


def compose_episode_videos(
    eval_dir: Path,
    camera_names: list[str],
    output_dir: Path | None = None,
    success_status: dict[str, bool] | None = None,
) -> dict[str, Path]:
    """Compose videos from multiple cameras for each episode.

    Args:
        eval_dir: Directory containing evaluation videos
        camera_names: List of camera names to compose
        output_dir: Directory to save composed videos (defaults to eval_dir/composed)
        success_status: Optional dict mapping episode keys to success status

    Returns:
        Dict mapping episode keys to composed video paths
    """
    if output_dir is None:
        output_dir = eval_dir / "composed"

    # Find all episodes by scanning for video files
    episode_videos = defaultdict(dict)

    for cam_name in camera_names:
        for video_path in eval_dir.glob(f"**/episode_*_{cam_name}*.mp4"):
            # Extract episode key from path
            house_dir = video_path.parent.name
            # Parse episode index from filename
            match = re.match(r"episode_(\d+)_", video_path.name)
            if match:
                episode_idx = int(match.group(1))
                episode_key = f"{house_dir}/episode_{episode_idx:08d}"
                episode_videos[episode_key][cam_name] = video_path

    composed_paths = {}
    for episode_key, cam_paths in sorted(episode_videos.items()):
        # Only compose if we have all cameras
        if len(cam_paths) < len(camera_names):
            continue

        # Order cameras consistently
        video_paths = [cam_paths[cam] for cam in camera_names if cam in cam_paths]

        # Determine success suffix for filename
        success_suffix = ""
        if success_status and episode_key in success_status:
            success_suffix = "_success" if success_status[episode_key] else "_failed"

        output_path = output_dir / f"{episode_key.replace('/', '_')}_composed{success_suffix}.mp4"
        result = compose_videos_side_by_side(video_paths, output_path)
        if result:
            composed_paths[episode_key] = result

    return composed_paths


def compute_eval_stats(results: list[EpisodeResult]) -> dict[str, Any]:
    """Compute aggregate statistics from evaluation results.

    Args:
        results: List of episode results

    Returns:
        Dict of aggregate statistics
    """
    if not results:
        return {}

    successes = [r.success for r in results]
    oracle_dones = [r.oracle_done for r in results if r.oracle_done is not None]
    num_steps = [r.num_steps for r in results]

    # Per-house stats
    house_results = defaultdict(list)
    for r in results:
        house_results[r.house_id].append(r.success)

    house_success_rates = {h: sum(s) / len(s) for h, s in house_results.items()}

    stats = {
        "total_episodes": len(results),
        "success_count": sum(successes),
        "failure_count": len(successes) - sum(successes),
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "avg_episode_length": sum(num_steps) / len(num_steps) if num_steps else 0.0,
        "min_episode_length": min(num_steps) if num_steps else 0,
        "max_episode_length": max(num_steps) if num_steps else 0,
        "num_houses": len(house_results),
        "house_success_rates": house_success_rates,
    }

    # Oracle done stats (success at any point during episode)
    if oracle_dones:
        stats["oracle_done_count"] = sum(oracle_dones)
        stats["oracle_done_rate"] = sum(oracle_dones) / len(oracle_dones)

    # Successful episode stats
    successful_steps = [r.num_steps for r in results if r.success]
    if successful_steps:
        stats["avg_successful_episode_length"] = sum(successful_steps) / len(successful_steps)

    return stats


def create_video_results_table(
    episode_data: list[dict],
    table_name: str = "eval/video_results",
) -> None:
    """Create and log a WandB table with videos and episode metadata.

    This is a shared utility for both distributed and non-distributed evaluation.
    Each dict in episode_data should contain:
        - video_path: Path to the video file (required)
        - task_description: Natural language task description
        - object_name: Target object name
        - house_id: House identifier
        - episode_idx: Episode index
        - num_steps: Number of steps taken
        - success: Boolean success status (at end of episode)
        - oracle_done: Boolean, success at ANY point during episode (optional)
        - source_episode_path: Original episode path (optional, for provenance)

    Args:
        episode_data: List of dicts with video paths and metadata
        table_name: Name for the WandB table (default: "eval/video_results")
    """
    import wandb

    table_data = []
    for ep in episode_data:
        video_path = ep.get("video_path")
        if video_path is None or not Path(video_path).exists():
            continue

        row = [
            wandb.Video(str(video_path), format="mp4"),
            ep.get("task_description", ""),
            ep.get("object_name", ""),
            str(ep.get("house_id", "")),
            ep.get("episode_idx", 0),
            ep.get("num_steps", 0),
            "Success" if ep.get("success") else "Failed",
            "Yes" if ep.get("oracle_done") else "No",
            ep.get("source_episode_path", ""),
        ]
        table_data.append(row)

    if table_data:
        video_table = wandb.Table(
            data=table_data,
            columns=[
                "video",
                "task_description",
                "object_name",
                "house_id",
                "episode_idx",
                "num_steps",
                "result",
                "oracle_done",
                "source_episode_path",
            ],
        )
        wandb.log({table_name: video_table})
        log.info(f"Uploaded {table_name} with {len(table_data)} episodes")


def log_eval_results_to_wandb(
    results: list[EpisodeResult],
    composed_videos: dict[str, Path] | None = None,
) -> None:
    """Log evaluation results and composed videos to wandb.

    Creates a video table with composed videos in the first column and metadata
    (task description, episode length, success/fail, etc.) in subsequent columns.

    Args:
        results: List of episode results
        composed_videos: Optional dict mapping episode keys to composed video paths
    """
    import wandb

    # Compute and log stats
    stats = compute_eval_stats(results)

    # Log scalar metrics as summary values (not time-series)
    # These are final metrics for a single checkpoint evaluation
    wandb.summary["eval/total_episodes"] = stats.get("total_episodes", 0)
    wandb.summary["eval/success_count"] = stats.get("success_count", 0)
    wandb.summary["eval/failure_count"] = stats.get("failure_count", 0)
    wandb.summary["eval/success_rate"] = stats.get("success_rate", 0.0)
    wandb.summary["eval/avg_episode_length"] = stats.get("avg_episode_length", 0.0)
    wandb.summary["eval/min_episode_length"] = stats.get("min_episode_length", 0)
    wandb.summary["eval/max_episode_length"] = stats.get("max_episode_length", 0)
    wandb.summary["eval/num_houses"] = stats.get("num_houses", 0)

    if stats.get("oracle_done_count") is not None:
        wandb.summary["eval/oracle_done_count"] = stats["oracle_done_count"]
        wandb.summary["eval/oracle_done_rate"] = stats["oracle_done_rate"]

    if stats.get("avg_successful_episode_length"):
        wandb.summary["eval/avg_successful_episode_length"] = stats["avg_successful_episode_length"]

    # Log per-house success rates as a table
    if stats.get("house_success_rates"):
        house_data = [[str(h), rate] for h, rate in sorted(stats["house_success_rates"].items())]
        house_table = wandb.Table(data=house_data, columns=["house_id", "success_rate"])
        wandb.log({"eval/house_success_rates": house_table})

    # Build result lookup by episode key
    result_by_key = {f"{r.house_id}/episode_{r.episode_idx:08d}": r for r in results}

    # Create video table with composed videos and metadata
    if composed_videos:
        # Convert EpisodeResult objects to dicts for the shared utility
        episode_data = []
        for episode_key in sorted(composed_videos.keys()):
            video_path = composed_videos[episode_key]
            result = result_by_key.get(episode_key)
            if result is None:
                continue

            episode_data.append(
                {
                    "video_path": video_path,
                    "task_description": result.task_description or "",
                    "object_name": result.object_name or "",
                    "house_id": result.house_id,
                    "episode_idx": result.episode_idx,
                    "num_steps": result.num_steps,
                    "success": result.success,
                    "oracle_done": result.oracle_done,
                    "source_episode_path": result.metadata.get("source_h5_file", ""),
                }
            )

        create_video_results_table(episode_data)


def log_eval_videos_to_wandb(eval_dir: Path, camera_names: list[str], epoch: int):
    """Find and log evaluation videos to wandb.

    DEPRECATED: Use log_eval_results_to_wandb with compose_episode_videos instead.
    """
    import wandb

    if not eval_dir.exists():
        return

    # Find all matching video files
    video_files = sorted(
        set(path for cam in camera_names for path in eval_dir.glob(f"**/episode_*_{cam}*.mp4"))
    )

    if not video_files:
        return

    wandb_videos = {}
    for video_path in video_files:
        # Find matching camera name
        camera_name = next((cam for cam in camera_names if cam in video_path.name), None)
        if not camera_name:
            continue

        # Extract sub-path (e.g., "house_3") and suffix (e.g., "batch_1_of_1")
        try:
            sub_path = video_path.relative_to(eval_dir).parent.name
            sub_path = sub_path if sub_path != "." else ""
        except ValueError:
            sub_path = ""

        stem = video_path.stem
        suffix = stem[stem.find(camera_name) + len(camera_name) :].lstrip("_")

        # Build wandb key: eval/video_{sub_path}_{camera_name}_{suffix}
        key_parts = filter(None, ["video", sub_path, camera_name, suffix])
        wandb_key = f"eval/{'_'.join(key_parts)}"

        try:
            wandb_videos[wandb_key] = wandb.Video(str(video_path), format="mp4")
        except Exception as e:
            print(f"Error logging {video_path.name}: {e}")

    if wandb_videos:
        wandb.log({**wandb_videos, "epoch": epoch})


def parse_obs_scene(obs_scene_data) -> dict:
    """Parse obs_scene from HDF5 dataset."""
    import json

    if isinstance(obs_scene_data, bytes):
        obs_scene_str = obs_scene_data.decode("utf-8")
    elif isinstance(obs_scene_data, str):
        obs_scene_str = obs_scene_data
    else:
        obs_scene_str = str(obs_scene_data)

    return json.loads(obs_scene_str)


def collect_episode_results(output_dir: Path) -> list[EpisodeResult]:
    """Scan output directory for HDF5 files and extract episode results.

    Args:
        output_dir: Directory containing evaluation output

    Returns:
        List of EpisodeResult objects. Each result includes data_file_path
        to uniquely identify the episode even when there are multiple batches
        per house.
    """
    results = []

    # Find all house directories
    for house_dir in sorted(output_dir.iterdir()):
        if not house_dir.is_dir() or not house_dir.name.startswith("house_"):
            continue

        house_id = house_dir.name

        # Find HDF5 files in this house directory
        for hdf5_path in sorted(house_dir.glob("trajectories*.h5")):
            with h5py.File(hdf5_path, "r") as f:
                for traj_key in sorted(f.keys()):
                    if not traj_key.startswith("traj_"):
                        continue

                    traj_group = f[traj_key]
                    episode_idx = int(traj_key.split("_")[1])

                    # Extract success (at end) and oracle_done (at any point)
                    success = False
                    oracle_done = False
                    if "success" in traj_group:
                        success_array = np.array(traj_group["success"])
                        if len(success_array) > 0:
                            success = bool(success_array[-1])
                            oracle_done = bool(np.any(success_array))

                    # Extract episode length from first action sub-dataset
                    num_steps = 0
                    if "actions" in traj_group:
                        actions_group = traj_group["actions"]
                        for action_key in actions_group:
                            num_steps = len(actions_group[action_key]) - 2
                            break

                    # Extract task description from obs_scene
                    task_description = None
                    object_name = None
                    if "obs_scene" in traj_group:
                        obs_scene = parse_obs_scene(traj_group["obs_scene"][()])
                        task_description = obs_scene.get("task_description") or obs_scene.get(
                            "text"
                        )
                        object_name = obs_scene.get("object_name")

                    results.append(
                        EpisodeResult(
                            episode_idx=episode_idx,
                            house_id=house_id,
                            success=success,
                            num_steps=num_steps,
                            task_description=task_description,
                            object_name=object_name,
                            data_file_path=hdf5_path,
                            oracle_done=oracle_done,
                        )
                    )

    return results
