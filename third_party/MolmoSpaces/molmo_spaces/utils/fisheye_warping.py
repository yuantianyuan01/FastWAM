"""GPU-accelerated fisheye lens distortion warping for camera images.

This module provides functions to apply fisheye distortion to images and videos,
simulating the effect of wide-angle GoPro cameras. The warping is GPU-accelerated
using PyTorch and uses a radial distortion model with parameters k1, k2, k3, k4.
"""

import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from molmo_spaces.utils.constants.camera_constants import (
    DEFAULT_CROP_PERCENT,
    DEFAULT_DISTORTION_PARAMETERS,
    GOPRO_CAMERA_HEIGHT,
    GOPRO_CAMERA_WIDTH,
    GOPRO_VERTICAL_FOV,
)

# Global cache for distortion map
_cached_map: np.ndarray | None = None


def get_default_distortion_map() -> np.ndarray:
    """Get the default distortion map for a camera, loading from disk if necessary."""
    global _cached_map
    if _cached_map is None:
        map_path = "molmo_spaces/utils/constants/default_unity_distortion_map.npy"
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"No default distortion map found at {map_path}.")
        _cached_map = np.load(map_path)
        # Verify map dimensions
        assert (
            _cached_map.shape[0] == GOPRO_CAMERA_HEIGHT
            and _cached_map.shape[1] == GOPRO_CAMERA_WIDTH
        ), (
            f"Default distortion map has wrong size: {_cached_map.shape}, expected: {(GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH)}"
        )
    return _cached_map


def calc_camera_intrinsics(fov_y: float, frame_height: int, frame_width: int) -> np.ndarray:
    """Calculate camera intrinsic matrix from field of view and frame dimensions.

    Args:
        fov_y: Vertical field of view in degrees
        frame_height: Image height in pixels
        frame_width: Image width in pixels

    Returns:
        3x3 camera intrinsic matrix K
    """
    focal_length = 0.5 * frame_height / math.tan(math.radians(fov_y / 2))
    f_x = f_y = focal_length

    c_x = frame_width / 2
    c_y = frame_height / 2
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    return K


def get_randomized_distortion_parameters(
    distortion_parameters: dict | None = None,
    randomization_factor: float = 0.001,
) -> dict:
    """Get distortion parameters with random perturbations.

    Args:
        distortion_parameters: Base distortion parameters (uses DEFAULT if None)
        randomization_factor: Magnitude of random perturbation

    Returns:
        Dictionary of randomized distortion parameters
    """
    if distortion_parameters is None:
        distortion_parameters = DEFAULT_DISTORTION_PARAMETERS
    randomized_distortion_parameters = {}
    for key, value in distortion_parameters.items():
        randomized_distortion_parameters[key] = value + np.random.uniform(
            -randomization_factor, randomization_factor
        )
    return randomized_distortion_parameters


def make_distorted_grid(
    H: int,
    W: int,
    K: np.ndarray,
    distortion_parameters: dict,
    device: torch.device | None = None,
    x_normalized: torch.Tensor | None = None,
    y_normalized: torch.Tensor | None = None,
    r: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create a distorted sampling grid for warping images.

    Args:
        H: Image height
        W: Image width
        K: Camera intrinsic matrix (3x3)
        distortion_parameters: Dict with keys k1, k2, k3, k4
        device: PyTorch device (defaults to CUDA if available)
        x_normalized: Pre-computed normalized x coordinates (optional)
        y_normalized: Pre-computed normalized y coordinates (optional)
        r: Pre-computed radial distances (optional)

    Returns:
        Grid tensor of shape [1, H, W, 2] for use with grid_sample
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if x_normalized is None or y_normalized is None or r is None:
        # Create meshgrid of pixel coordinates
        y, x = torch.meshgrid(
            torch.arange(H, device=device).float(),
            torch.arange(W, device=device).float(),
            indexing="ij",
        )

        # Normalize pixel coordinates using camera intrinsics
        x_normalized = (x - K[0, 2]) / K[0, 0]
        y_normalized = (y - K[1, 2]) / K[1, 1]

        r = torch.sqrt(x_normalized**2 + y_normalized**2)
    else:
        # Ensure the precomputed values are on the correct device
        x_normalized = x_normalized.to(device)
        y_normalized = y_normalized.to(device)
        r = r.to(device)

    # Extract distortion parameters
    k1, k2, k3, k4 = (distortion_parameters[k] for k in ["k1", "k2", "k3", "k4"])

    # Apply radial distortion
    distortion_factor = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8
    x_distorted = x_normalized * distortion_factor
    y_distorted = y_normalized * distortion_factor

    # Transform back to pixel coordinates
    x_distorted = x_distorted * K[0, 0] + K[0, 2]
    y_distorted = y_distorted * K[1, 1] + K[1, 2]

    # Normalize coordinates to [-1, 1] for grid_sample
    x_distorted = 2 * (x_distorted / (W - 1)) - 1
    y_distorted = 2 * (y_distorted / (H - 1)) - 1

    # Stack coordinates
    grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    return grid


def warp_image_gpu(
    image: torch.Tensor,
    K: np.ndarray | None = None,
    distortion_parameters: dict | None = None,
    crop_percent: float = DEFAULT_CROP_PERCENT,
    grid: torch.Tensor | None = None,
    x_normalized: torch.Tensor | None = None,
    y_normalized: torch.Tensor | None = None,
    r: torch.Tensor | None = None,
    output_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Apply fisheye distortion to an image using GPU acceleration.

    Args:
        image: Input image tensor of shape [B, C, H, W]
        K: Camera intrinsic matrix (required if grid is None)
        distortion_parameters: Distortion parameters (required if grid is None)
        crop_percent: Percentage to crop from each edge after warping
        grid: Pre-computed distortion grid (optional)
        x_normalized: Pre-computed normalized x coordinates (optional)
        y_normalized: Pre-computed normalized y coordinates (optional)
        r: Pre-computed radial distances (optional)
        output_shape: Target output size (H, W) for resizing (optional)

    Returns:
        Warped image tensor
    """
    B, C, H, W = image.shape
    assert C == 3, "Input image should have 3 channels (RGB)"

    assert H == GOPRO_CAMERA_HEIGHT and W == GOPRO_CAMERA_WIDTH, (
        f"Image should be raw GoPro format, actually {H}x{W}"
    )

    if grid is None:
        assert distortion_parameters is not None, (
            "distortion_parameters must be provided if grid is not"
        )
        assert K is not None, "K must be provided if grid is not"
        grid = make_distorted_grid(
            H,
            W,
            K,
            distortion_parameters,
            device=image.device,
            x_normalized=x_normalized,
            y_normalized=y_normalized,
            r=r,
        )
    grid = grid.repeat(B, 1, 1, 1)  # [B, H, W, 2]
    distorted_image = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    crop_h = int(H * crop_percent)
    crop_w = int(W * crop_percent)
    cropped_image = distorted_image[
        :, :, crop_h : -crop_h if crop_h > 0 else None, crop_w : -crop_w if crop_w > 0 else None
    ]

    if output_shape is not None:
        cropped_image = F.interpolate(
            cropped_image, size=output_shape, mode="bilinear", align_corners=True
        )

    return cropped_image


def warp_video_gpu(
    video: np.ndarray | torch.Tensor,
    K: np.ndarray | None = None,
    randomize_distortion_parameters: bool = False,
    crop_percent: float = DEFAULT_CROP_PERCENT,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply fisheye distortion to a video using GPU acceleration.

    Args:
        video: Input video as numpy array [T, H, W, C] or tensor
        K: Camera intrinsic matrix (computed from defaults if None)
        randomize_distortion_parameters: Whether to randomize distortion params
        crop_percent: Percentage to crop from each edge after warping
        output_shape: Target output size (H, W) for resizing (optional)

    Returns:
        Warped video as numpy array [T, H, W, C] with uint8 values
    """
    assert video.shape[2] == GOPRO_CAMERA_WIDTH and video.shape[1] == GOPRO_CAMERA_HEIGHT, (
        "Video should be raw GoPro format"
    )

    if randomize_distortion_parameters:
        distortion_parameters = get_randomized_distortion_parameters()
    else:
        distortion_parameters = DEFAULT_DISTORTION_PARAMETERS

    if K is None:
        K = calc_camera_intrinsics(GOPRO_VERTICAL_FOV, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH)

    # Convert to tensor if needed
    if not isinstance(video, torch.Tensor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        video_tensor = torch.from_numpy(video).float().to(device) / 255.0
    else:
        video_tensor = video.float() / 255.0

    # Permute to [B, C, H, W] format
    video_tensor = video_tensor.permute(0, 3, 1, 2)

    warped_video = warp_image_gpu(
        image=video_tensor,
        K=K,
        distortion_parameters=distortion_parameters,
        crop_percent=crop_percent,
        output_shape=output_shape,
    )
    # Convert back to numpy
    warped_video = (warped_video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return warped_video


def warp_point(
    pixel_x: float,
    pixel_y: float,
    K: np.ndarray,
    distortion_parameters: dict,
    crop_percent: float,
    output_shape: tuple[int, int],
) -> tuple[int, int]:
    """Warp a single point through the fisheye distortion.

    Args:
        pixel_x: X coordinate in original image
        pixel_y: Y coordinate in original image
        K: Camera intrinsic matrix
        distortion_parameters: Distortion parameters
        crop_percent: Crop percentage used in warping
        output_shape: Output image size (H, W)

    Returns:
        Tuple of (warped_x, warped_y) coordinates
    """
    # Create a blank frame with the point marked
    blank_frame = torch.zeros((1, 3, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH), dtype=torch.float32)
    blank_frame[0, :, int(pixel_y), int(pixel_x)] = 1.0  # Mark the point as white

    # Warp the frame
    warped_frame = warp_image_gpu(
        blank_frame,
        K=K,
        distortion_parameters=distortion_parameters,
        crop_percent=crop_percent,
        output_shape=output_shape,
    )

    # Find the warped point
    warped_frame_np = warped_frame.squeeze().permute(1, 2, 0).cpu().numpy()
    flat_index = np.argmax(warped_frame_np[:, :, 0])
    warped_y, warped_x = np.unravel_index(flat_index, warped_frame_np.shape[:2])

    return warped_x, warped_y


def load_frames_from_mp4(video_path: Path | str) -> tuple[list[np.ndarray], float]:
    """Load frames from an MP4 video file.

    Args:
        video_path: Path to MP4 video file

    Returns:
        List of frames as numpy arrays (H, W, C) in RGB format
        FPS of the video
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps > 0, f"Error reading FPS from video {video_path}, got {fps}"

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames, fps


def warp_video_frames_batch(
    frames: list[np.ndarray],
    K: np.ndarray,
    distortion_parameters: dict,
    crop_percent: float,
    output_shape: tuple[int, int] | None,
    device: torch.device,
    batch_size: int = 16,
) -> list[np.ndarray]:
    """Apply fisheye warping to a list of video frames in batches.

    Args:
        frames: List of frames as numpy arrays (H, W, C)
        K: Camera intrinsic matrix
        distortion_parameters: Distortion parameters
        crop_percent: Crop percentage after warping
        output_shape: Output size (H, W) or None
        device: PyTorch device
        batch_size: Number of frames to process at once

    Returns:
        List of warped frames as numpy arrays (H, W, C)
    """
    warped_frames = []

    # Convert K to tensor
    K_tensor = torch.tensor(K, dtype=torch.float32, device=device)

    # Process in batches for efficiency
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]

        # Stack frames into batch tensor [B, H, W, C]
        batch_array = np.stack(batch, axis=0)
        batch_tensor = torch.from_numpy(batch_array).float().to(device) / 255.0

        # Permute to [B, C, H, W]
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)

        # Apply warping
        warped_batch = warp_image_gpu(
            image=batch_tensor,
            K=K_tensor,
            distortion_parameters=distortion_parameters,
            crop_percent=crop_percent,
            output_shape=output_shape,
        )

        # Convert back to numpy and uint8
        warped_batch = (warped_batch.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

        # Add to results
        for frame in warped_batch:
            warped_frames.append(frame)

    return warped_frames


def apply_fisheye_warping_to_video_file(
    video_path: Path | str,
    output_path: Path | str,
    K: np.ndarray,
    distortion_parameters: dict,
    crop_percent: float,
    output_shape: tuple[int, int] | None,
    device: torch.device | None = None,
) -> bool:
    """Apply fisheye warping to a video file and save the result.

    Args:
        video_path: Path to input video
        output_path: Path to save warped video
        K: Camera intrinsic matrix
        distortion_parameters: Distortion parameters
        crop_percent: Crop percentage after warping
        output_shape: Output size (H, W) or None
        device: PyTorch device (defaults to CUDA if available)

    Returns:
        True if successful, False otherwise
    """
    from molmo_spaces.utils.video_utils import ffmpeg_save_video

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load frames
        frames, fps = load_frames_from_mp4(video_path)

        # Warp frames
        warped_frames = warp_video_frames_batch(
            frames=frames,
            K=K,
            distortion_parameters=distortion_parameters,
            crop_percent=crop_percent,
            output_shape=output_shape,
            device=device,
        )

        # Save warped video
        ffmpeg_save_video(warped_frames, str(output_path), fps=fps, pix_fmt="rgb24")
        return True
    except Exception:
        return False
