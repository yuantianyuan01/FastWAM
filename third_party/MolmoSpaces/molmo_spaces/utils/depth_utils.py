"""Utilities for depth image encoding and decoding.

Optimized for Intel RealSense D405 camera specs:
- D405 actual spec: 7cm - 50cm range, ±1.4% at 20cm = ±2.8mm
- Encoding range: 5cm - 55cm (extended for margin)
- Resolution: 1280x720
- Baseline: 18mm, Global shutter

Depth images are encoded as 16-bit values across RG channels:
1. High precision: 7.6 microns over 50cm range (65,534 discrete values for valid data)
2. Video compatibility: Standard RGB video codecs (H.264 RGB)
3. Efficient lossy compression: Unused B channel reduces artifacts
4. Smaller file sizes vs 24-bit encoding
5. Invalid data handling: 0 reserved for missing/out-of-range pixels

The encoding range (5-55cm) extends slightly beyond D405's spec (7-50cm) to:
- Provide margin for edge cases and measurement noise
- Still maintain excellent precision (7.6μm vs 15μm with wider ranges)
- Keep compression efficient (tight dynamic range = better lossy codec performance)

Invalid/missing data convention:
- Pixels outside [DEPTH_MIN, DEPTH_MAX] are encoded as 0 (not clipped)
- This allows easy masking: valid_mask = depth > 0
- Common for far-away regions or sensor failures in real-world depth cameras
"""

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Fixed depth range optimized for Intel RealSense D405 camera specs
# D405 actual spec: 7cm-50cm range, ±1.4% accuracy at 20cm, 1280x720, 18mm baseline
# Extended slightly to 5-55cm for margin (50cm range with 16-bit encoding = 7.6μm precision)
# Note: 0.0 is reserved for invalid/missing data (too close, too far, or no measurement)
DEPTH_MIN = 0.05  # 5cm minimum valid depth (D405 spec: 7cm, extended for margin)
DEPTH_MAX = 0.55  # 55cm maximum valid depth (D405 spec: 50cm, extended for margin)

# Video codec configuration for depth videos
# Change these settings in one place to affect all depth video saving
DEPTH_VIDEO_CODEC = (
    "libx264rgb"  # RGB H.264 codec (no YUV conversion - avoids chroma subsampling artifacts)
)
DEPTH_VIDEO_PIXELFORMAT = "rgb24"  # RGB pixel format (preserves RG channel depth encoding)
DEPTH_VIDEO_CRF = "23"  # High quality lossy (18 = visually lossless, 23 = smaller files)


def encode_depth_to_rgb(depth_meters: np.ndarray) -> np.ndarray:
    """Encode metric depth values as 16-bit RG channels for video storage.

    Converts floating-point depth values (in meters) to uint8 RG encoding.
    Provides ~7.6 micron precision over the 50cm range using 16-bit encoding.
    The B channel is set to 0, which helps with lossy video compression.

    Invalid pixels (outside [DEPTH_MIN, DEPTH_MAX]) are encoded as 0, allowing
    downstream processing to use `depth_mask = depth > 0` to identify valid data.

    Args:
        depth_meters: (H, W) float32 array of depth values in meters.
                     Values outside [DEPTH_MIN, DEPTH_MAX] are set to 0 (invalid).

    Returns:
        rgb_frame: (H, W, 3) uint8 array with depth encoded as:
                  - R channel: bits 8-15 (high byte)
                  - G channel: bits 0-7 (low byte)
                  - B channel: 0 (unused, helps compression)
                  - RGB(0,0,0): invalid/missing data

    Example:
        >>> depth = np.array([[0.5, 1.0], [0.1, 1.0]], dtype=np.float32)
        >>> rgb = encode_depth_to_rgb(depth)
        >>> rgb.shape
        (2, 2, 3)
        >>> rgb.dtype
        dtype('uint8')
    """
    # Identify valid pixels (within depth range)
    valid_mask = (depth_meters >= DEPTH_MIN) & (depth_meters <= DEPTH_MAX)

    # Set invalid pixels to 0 (missing data sentinel)
    depth_masked = np.where(valid_mask, depth_meters, 0.0)

    # Normalize valid range to [0, 1]
    depth_normalized = (depth_masked - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)

    # Map to [1, 65535] to reserve 0 for invalid data
    # Valid pixels: [DEPTH_MIN, DEPTH_MAX] → [1, 65535]
    # Invalid pixels: 0.0 → 0 (after normalization becomes negative, handled below)
    depth_16bit = np.where(
        valid_mask,
        (depth_normalized * 65534.0 + 1.0).astype(np.uint16),  # Valid: [1, 65535]
        np.uint16(0),  # Invalid: 0
    )

    # Split into RG channels, leave B as 0
    h, w = depth_meters.shape
    rgb_frame = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_frame[:, :, 0] = (depth_16bit >> 8) & 0xFF  # R: bits 8-15
    rgb_frame[:, :, 1] = depth_16bit & 0xFF  # G: bits 0-7
    # rgb_frame[:, :, 2] remains 0                   # B: unused

    return rgb_frame


def decode_depth_from_rgb(rgb_frame: np.ndarray, validate: bool = True) -> np.ndarray:
    """Decode RG-encoded depth back to metric depth in meters.

    Reverses the encoding from encode_depth_to_rgb() to recover
    floating-point depth values from uint8 RG channels.

    Encoded value of 0 (RGB(0,0,0)) represents invalid/missing data and
    is decoded to 0.0 meters.

    Args:
        rgb_frame: (H, W, 3) uint8 array with depth encoded in RG channels
        validate: If True, warns if B channel is non-zero (indicates wrong pixel format)

    Returns:
        depth_meters: (H, W) float32 array of depth values in meters.
                     Valid pixels in range [DEPTH_MIN, DEPTH_MAX].
                     Invalid pixels are 0.0 (use depth > 0 to mask valid data).

    Example:
        >>> depth_original = np.array([[0.5, 1.0], [0.1, 1.0]], dtype=np.float32)
        >>> rgb = encode_depth_to_rgb(depth_original)
        >>> depth_decoded = decode_depth_from_rgb(rgb)
        >>> np.allclose(depth_original, depth_decoded, atol=0.001)
        True
    """
    # Validate B channel (should be ~0 for properly encoded depth)
    if validate:
        b_mean = float(np.mean(rgb_frame[:, :, 2]))
        if b_mean > 5.0:
            log.warning(
                f"B channel has non-zero values (mean={b_mean:.1f}). "
                "This may indicate YUV pixel format was used instead of RGB, "
                "which causes chroma subsampling artifacts. "
                "Use load_depth_video() or imageio with pixelformat='rgb24' to avoid this."
            )

    # Reconstruct 16-bit integer from RG channels (ignore B channel)
    depth_16bit = (
        rgb_frame[:, :, 0].astype(np.uint32) * np.uint32(256)  # R * 2^8
        + rgb_frame[:, :, 1].astype(np.uint32)  # G * 2^0
    )

    # Identify valid pixels (encoded as non-zero)
    valid_mask = depth_16bit > 0

    # Decode valid pixels: [1, 65535] → [DEPTH_MIN, DEPTH_MAX]
    # Invalid pixels (0) → 0.0
    depth_normalized = (depth_16bit.astype(np.float32) - 1.0) / 65534.0
    metric_depth = depth_normalized * np.float32(DEPTH_MAX - DEPTH_MIN) + np.float32(DEPTH_MIN)

    # Set invalid pixels to 0.0
    metric_depth = np.where(valid_mask, metric_depth, 0.0)

    return metric_depth.astype(np.float32)


def compute_depth_encoding_stats(depth_meters: np.ndarray) -> dict:
    """Compute statistics about depth encoding precision.

    Useful for validating that the depth range and encoding are appropriate
    for your specific use case.

    Args:
        depth_meters: (H, W) float32 array of depth values in meters

    Returns:
        Dictionary with statistics:
        - min_depth: Minimum depth value in the image (excluding zeros)
        - max_depth: Maximum depth value in the image
        - mean_depth: Mean depth value (excluding zeros)
        - invalid_pixels: Number of pixels outside [DEPTH_MIN, DEPTH_MAX] (will be encoded as 0)
        - precision: Theoretical precision in meters (at current encoding)
        - precision_mm: Theoretical precision in millimeters
    """
    depth_range = DEPTH_MAX - DEPTH_MIN
    precision = depth_range / 65534.0  # 16-bit precision (minus 1 for invalid marker)

    # Count invalid pixels (will be set to 0)
    invalid = np.sum((depth_meters < DEPTH_MIN) | (depth_meters > DEPTH_MAX))

    # Compute stats only on valid pixels for meaningful min/mean
    valid_mask = (depth_meters >= DEPTH_MIN) & (depth_meters <= DEPTH_MAX)
    valid_depths = depth_meters[valid_mask]

    return {
        "min_depth": float(np.min(valid_depths)) if valid_depths.size > 0 else 0.0,
        "max_depth": float(np.max(valid_depths)) if valid_depths.size > 0 else 0.0,
        "mean_depth": float(np.mean(valid_depths)) if valid_depths.size > 0 else 0.0,
        "invalid_pixels": int(invalid),
        "invalid_fraction": float(invalid) / depth_meters.size,
        "precision_meters": precision,
        "precision_mm": precision * 1000.0,
        "depth_range": depth_range,
        "encoding_bits": 16,
    }


def validate_roundtrip_accuracy(depth_meters: np.ndarray, tolerance_mm: float = 0.1) -> dict:
    """Validate that depth encoding/decoding roundtrip is accurate.

    Args:
        depth_meters: (H, W) float32 array of depth values in meters
        tolerance_mm: Maximum acceptable error in millimeters

    Returns:
        Dictionary with validation results:
        - max_error_mm: Maximum error in millimeters (valid pixels only)
        - mean_error_mm: Mean error in millimeters (valid pixels only)
        - passed: Whether the roundtrip is within tolerance
        - invalid_preserved: Whether invalid pixels are correctly encoded as 0
        - errors: (H, W) array of absolute errors in meters
    """
    # Encode and decode
    encoded = encode_depth_to_rgb(depth_meters)
    decoded = decode_depth_from_rgb(encoded)

    # Check valid pixels (should be accurate within tolerance)
    valid_mask = (depth_meters >= DEPTH_MIN) & (depth_meters <= DEPTH_MAX)
    errors = np.abs(decoded - depth_meters)

    if np.sum(valid_mask) > 0:
        max_error_mm = float(np.max(errors[valid_mask])) * 1000.0
        mean_error_mm = float(np.mean(errors[valid_mask])) * 1000.0
    else:
        max_error_mm = 0.0
        mean_error_mm = 0.0

    # Check invalid pixels (should be 0.0 after roundtrip)
    invalid_mask = ~valid_mask
    invalid_preserved = np.all(decoded[invalid_mask] == 0.0) if np.sum(invalid_mask) > 0 else True

    return {
        "max_error_mm": max_error_mm,
        "mean_error_mm": mean_error_mm,
        "tolerance_mm": tolerance_mm,
        "passed": max_error_mm <= tolerance_mm,
        "invalid_preserved": invalid_preserved,
        "errors": errors,
    }


# =============================================================================
# Visualization and debugging utilities (require matplotlib/scipy)
# =============================================================================


def visualize_depth_image(depth_meters: np.ndarray, title: str, save_path: Path | None = None):
    """Visualize depth image with statistics and save to debug file.

    Creates a 4-panel visualization showing:
    1. Raw depth with full range (0-2m)
    2. Raw depth with encoding range
    3. Valid/invalid pixel visualization (too close/valid/too far)
    4. Encoded RGB representation

    Args:
        depth_meters: (H, W) float32 array of depth values in meters
        title: Title for the visualization
        save_path: Optional path to save the visualization (PNG)

    Returns:
        Dictionary of depth statistics from compute_depth_encoding_stats()
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Visualization requires matplotlib. Install with: pip install matplotlib"
        ) from e

    # Compute stats
    stats = compute_depth_encoding_stats(depth_meters)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Raw depth with full range (0-2m)
    ax = axes[0, 0]
    im = ax.imshow(depth_meters, cmap="turbo", vmin=0, vmax=2.0)
    ax.set_title("Raw Depth (0-2m range)")
    plt.colorbar(im, ax=ax, label="Depth (m)")

    # 2. Raw depth with encoding range
    ax = axes[0, 1]
    im = ax.imshow(depth_meters, cmap="turbo", vmin=DEPTH_MIN, vmax=DEPTH_MAX)
    ax.set_title(f"Raw Depth ({DEPTH_MIN}m-{DEPTH_MAX}m encoding range)")
    plt.colorbar(im, ax=ax, label="Depth (m)")

    # 3. Valid/invalid visualization
    ax = axes[1, 0]
    validity_vis = np.ones_like(depth_meters)
    validity_vis[depth_meters < DEPTH_MIN] = 0  # Too close (red)
    validity_vis[depth_meters > DEPTH_MAX] = 2  # Too far (blue)
    im = ax.imshow(validity_vis, cmap="RdYlGn", vmin=0, vmax=2)
    ax.set_title("Validity (Red=too close→0, Green=valid, Blue=too far→0)")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Too Close", "Valid", "Too Far"])

    # 4. Encoded RGB representation
    ax = axes[1, 1]
    encoded = encode_depth_to_rgb(depth_meters)
    ax.imshow(encoded)
    ax.set_title("Encoded as RGB (16-bit RG, 0=invalid)")

    # Add statistics text
    stats_text = f"""Statistics:
Min: {stats["min_depth"]:.3f}m  Max: {stats["max_depth"]:.3f}m  Mean: {stats["mean_depth"]:.3f}m
Invalid (→0): {stats["invalid_pixels"]:,} pixels ({stats["invalid_fraction"] * 100:.2f}%)
Precision: {stats["precision_mm"]:.4f}mm  Range: {DEPTH_MIN}m - {DEPTH_MAX}m
"""

    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.debug(f"Saved depth visualization to {save_path}")

    plt.close()

    return stats


def print_depth_stats(depth_meters: np.ndarray, name: str = "Depth"):
    """Print detailed depth statistics to console.

    Args:
        depth_meters: (H, W) float32 array of depth values in meters
        name: Name to display in the output (e.g., "Wrist Camera Depth")
    """
    stats = compute_depth_encoding_stats(depth_meters)

    print(f"\n{name} Statistics:")
    print(f"  Range: [{stats['min_depth']:.3f}m, {stats['max_depth']:.3f}m]")
    print(f"  Mean: {stats['mean_depth']:.3f}m")
    print(f"  Encoding precision: {stats['precision_mm']:.4f}mm ({stats['precision_meters']:.6f}m)")
    print(f"  Valid range: {DEPTH_MIN}m - {DEPTH_MAX}m")

    if stats["invalid_pixels"] > 0:
        invalid_below = np.sum(depth_meters < DEPTH_MIN)
        invalid_above = np.sum(depth_meters > DEPTH_MAX)
        print(
            f"  Invalid pixels (will be set to 0): {stats['invalid_pixels']:,} ({stats['invalid_fraction'] * 100:.2f}%)"
        )
        print(f"    Below {DEPTH_MIN}m: {invalid_below:,}")
        print(f"    Above {DEPTH_MAX}m: {invalid_above:,}")

    else:
        print("  All pixels within valid range")


def detect_depth_edges(depth: np.ndarray, gradient_threshold_mm: float = 50.0) -> np.ndarray:
    """Detect depth discontinuities (edges) where compression artifacts are expected.

    Used for analysis/visualization only - not part of encoding/decoding pipeline.

    Args:
        depth: (H, W) depth array in meters
        gradient_threshold_mm: Depth gradient threshold in mm to classify as edge

    Returns:
        edge_mask: (H, W) boolean array, True at edge pixels
    """
    try:
        import scipy.ndimage
    except ImportError as e:
        raise ImportError("Edge detection requires scipy. Install with: pip install scipy") from e

    # Compute depth gradients in mm
    depth_mm = depth * 1000.0
    grad_x = scipy.ndimage.sobel(depth_mm, axis=1)
    grad_y = scipy.ndimage.sobel(depth_mm, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Classify as edge if gradient exceeds threshold
    edge_mask = gradient_magnitude > gradient_threshold_mm

    # Dilate edge mask to catch nearby compression artifacts (1-2 pixel radius)
    edge_mask = scipy.ndimage.binary_dilation(edge_mask, iterations=2)

    return edge_mask


def visualize_depth_error(
    original_depth: np.ndarray,
    decoded_depth: np.ndarray,
    error: np.ndarray,
    title: str,
    save_path: Path | None = None,
):
    """Visualize the compression error between original and decoded depth.

    Shows where errors occur (smooth regions vs edges) to understand compression behavior.
    Creates a 4-panel visualization:
    1. Original depth
    2. Decoded depth (after compression)
    3. Edge detection (shows discontinuities)
    4. Error heatmap

    Args:
        original_depth: (H, W) float32 array of original depth in meters
        decoded_depth: (H, W) float32 array of decoded depth in meters
        error: (H, W) float32 array of absolute errors in meters
        title: Title for the visualization
        save_path: Optional path to save the visualization (PNG)
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Visualization requires matplotlib. Install with: pip install matplotlib"
        ) from e

    # Create mask for valid pixels (within valid range in original)
    valid_mask = (original_depth >= DEPTH_MIN) & (original_depth <= DEPTH_MAX)
    num_valid = np.sum(valid_mask)
    num_invalid = np.sum(~valid_mask)

    # Detect edges (for analysis - shows where high errors are expected)
    edge_mask = detect_depth_edges(original_depth, gradient_threshold_mm=50.0)
    smooth_mask = valid_mask & ~edge_mask
    edge_valid_mask = valid_mask & edge_mask

    num_smooth = np.sum(smooth_mask)
    num_edges = np.sum(edge_valid_mask)

    # Convert error to mm
    error_mm = error * 1000.0

    # Create masked error array (NaN for invalid regions)
    error_mm_masked = error_mm.copy()
    error_mm_masked[~valid_mask] = np.nan

    # Create 4-panel visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 1. Original depth
    ax = axes[0]
    im = ax.imshow(original_depth, cmap="turbo", vmin=DEPTH_MIN, vmax=DEPTH_MAX)
    ax.set_title("Original Depth")
    plt.colorbar(im, ax=ax, label="Depth (m)")

    # 2. Decoded depth
    ax = axes[1]
    im = ax.imshow(decoded_depth, cmap="turbo", vmin=DEPTH_MIN, vmax=DEPTH_MAX)
    ax.set_title("Decoded Depth (After MP4)")
    plt.colorbar(im, ax=ax, label="Depth (m)")

    # 3. Edge detection (shows where we expect higher errors)
    ax = axes[2]
    edge_vis = np.zeros_like(original_depth)
    edge_vis[edge_valid_mask] = 2  # Edges (red)
    edge_vis[smooth_mask] = 1  # Smooth regions (green)
    edge_vis[~valid_mask] = 0  # Invalid (blue)
    im = ax.imshow(edge_vis, cmap="RdYlGn_r", vmin=0, vmax=2)
    ax.set_title("Depth Discontinuities")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Invalid", "Smooth", "Edge"])

    # 4. Error heatmap
    ax = axes[3]
    if num_valid > 0:
        max_error_display = np.nanmax(error_mm_masked)
    else:
        max_error_display = 5.0
    im = ax.imshow(error_mm_masked, cmap="hot", vmin=0, vmax=max_error_display)
    ax.set_title("Absolute Error (mm)")
    cbar = plt.colorbar(im, ax=ax, label="Error (mm)")
    ax.set_facecolor("lightgray")

    # Calculate statistics (separate smooth vs edges for analysis)
    if num_smooth > 0:
        smooth_errors = error_mm[smooth_mask]
        smooth_mean = np.mean(smooth_errors)
        smooth_p95 = np.percentile(smooth_errors, 95)
    else:
        smooth_mean = smooth_p95 = 0.0

    if num_edges > 0:
        edge_errors = error_mm[edge_valid_mask]
        edge_mean = np.mean(edge_errors)
        edge_p95 = np.percentile(edge_errors, 95)
    else:
        edge_mean = edge_p95 = 0.0

    if num_valid > 0:
        valid_errors = error_mm[valid_mask]
        mean_error = np.mean(valid_errors)
        p95_error = np.percentile(valid_errors, 95)
        max_error = np.max(valid_errors)

        stats_text = f"""Error Statistics:
Valid: {num_valid:,} ({num_valid / error.size * 100:.1f}%) = {num_smooth:,} smooth + {num_edges:,} edges
Invalid: {num_invalid:,} ({num_invalid / error.size * 100:.1f}%)

Smooth regions: mean={smooth_mean:.2f}mm, 95th={smooth_p95:.2f}mm
Edge regions: mean={edge_mean:.2f}mm, 95th={edge_p95:.2f}mm
Overall: mean={mean_error:.2f}mm, 95th={p95_error:.2f}mm, max={max_error:.2f}mm
"""
    else:
        stats_text = "No valid pixels found (all invalid)"

    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.15, 1, 0.96])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.debug(f"Saved error visualization to {save_path}")

    plt.close()


# =============================================================================
# Video encoding utilities
# =============================================================================


def save_depth_video(
    depth_frames: np.ndarray,
    video_path: str | Path,
    fps: float = 10,
    logger: logging.Logger | None = None,
) -> None:
    """Save depth frames as compressed video.

    This is the single source of truth for depth video compression settings.
    Encodes depth frames using 16-bit RG encoding and saves with configured codec.

    Args:
        depth_frames: (T, H, W) float32 array of depth values in meters
        video_path: Path to save the video file
        fps: Frames per second for the video
        logger: Optional logger for debugging

    Example:
        >>> depth_frames = np.random.rand(100, 480, 640).astype(np.float32) * 0.5 + 0.3
        >>> save_depth_video(depth_frames, "depth.mp4")
    """
    import imageio

    logger = logger or log

    # Validate input
    if depth_frames.ndim != 3:
        raise ValueError(f"Expected 3D array (T, H, W), got shape {depth_frames.shape}")

    if depth_frames.dtype != np.float32:
        logger.warning(f"Depth frames are {depth_frames.dtype}, expected float32. Converting...")
        depth_frames = depth_frames.astype(np.float32)

    # Check depth range and warn about potential issues
    depth_min = float(depth_frames.min())
    depth_max = float(depth_frames.max())

    # Warn if depth values seem wrong (likely wrong units)
    if depth_min > 1.0 or depth_max > 10.0:
        logger.warning(
            f"Depth range [{depth_min:.2f}m, {depth_max:.2f}m] seems large. "
            "Are you sure values are in meters (not mm or cm)? "
            f"Encoding range is {DEPTH_MIN}m-{DEPTH_MAX}m for D405 camera."
        )

    # Warn if excessive invalid pixels will occur
    invalid_pixels = np.sum((depth_frames < DEPTH_MIN) | (depth_frames > DEPTH_MAX))
    total_pixels = depth_frames.size
    invalid_fraction = invalid_pixels / total_pixels

    if invalid_fraction > 0.3:
        logger.warning(
            f"Warning: {invalid_fraction * 100:.1f}% of pixels will be set to 0 (invalid) "
            f"(outside {DEPTH_MIN}m-{DEPTH_MAX}m range). "
            f"Actual range: [{depth_min:.3f}m, {depth_max:.3f}m]. "
            "Consider adjusting DEPTH_MIN/DEPTH_MAX or checking your depth data."
        )

    # Log depth statistics for debugging
    logger.debug(
        f"Saving depth video: {depth_frames.shape} frames, "
        f"range [{depth_min:.3f}m, {depth_max:.3f}m]"
    )

    # Encode each depth frame to RGB
    encoded_frames = []
    for frame in depth_frames:
        encoded_frames.append(encode_depth_to_rgb(frame))
    encoded_frames = np.array(encoded_frames)

    logger.debug(f"Encoded {len(encoded_frames)} depth frames to RGB: {encoded_frames.shape}")

    # Prepare video path
    video_path = Path(video_path)
    os.makedirs(video_path.parent, exist_ok=True)
    if video_path.suffix != ".mp4":
        video_path = video_path.with_suffix(".mp4")

    # Configure codec - ALL depth compression settings in one place
    codec_kwargs = {
        "codec": DEPTH_VIDEO_CODEC,
        "pixelformat": DEPTH_VIDEO_PIXELFORMAT,
        "output_params": ["-crf", DEPTH_VIDEO_CRF],
    }

    # Save video
    try:
        writer = imageio.get_writer(str(video_path), format="ffmpeg", fps=fps, **codec_kwargs)
        for frame in encoded_frames:
            writer.append_data(frame)
        writer.close()
        logger.debug(f"Saved depth video to {video_path}")
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        logger.warning(f"FFmpeg writer failed ({type(e).__name__}: {e}), falling back to mimwrite")
        imageio.mimwrite(str(video_path), encoded_frames, format="mp4", fps=fps, **codec_kwargs)


def load_depth_video(
    video_path: str | Path,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Load depth video and decode frames back to metric depth.

    Companion function to save_depth_video(). Ensures proper codec settings
    for reading depth videos (RGB pixel format, no YUV conversion).

    Args:
        video_path: Path to the depth video file (.mp4)
        logger: Optional logger for debugging

    Returns:
        depth_frames: (T, H, W) float32 array of depth values in meters

    Example:
        >>> # Save and load round-trip
        >>> depth_original = np.random.rand(10, 480, 640).astype(np.float32) * 0.4 + 0.1
        >>> save_depth_video(depth_original, "test_depth.mp4")
        >>> depth_loaded = load_depth_video("test_depth.mp4")
        >>> depth_loaded.shape
        (10, 480, 640)
    """
    import imageio

    logger = logger or log

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.debug(f"Loading depth video from {video_path}")

    # Use imageio with RGB pixel format (critical - avoids YUV conversion artifacts)
    try:
        reader = imageio.get_reader(
            str(video_path),
            format="ffmpeg",
            pixelformat="rgb24",  # Force RGB to avoid chroma subsampling
        )
    except (ImportError, OSError, ValueError) as e:
        logger.warning(f"Failed to open with pixelformat='rgb24': {e}. Trying default...")
        reader = imageio.get_reader(str(video_path), format="ffmpeg")

    decoded_frames = []
    for frame_rgb in reader:
        # Validate frame format
        if frame_rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame with 3 channels, got shape {frame_rgb.shape}")

        decoded_depth = decode_depth_from_rgb(frame_rgb, validate=True)
        decoded_frames.append(decoded_depth)

    reader.close()

    depth_frames = np.array(decoded_frames, dtype=np.float32)

    logger.debug(
        f"Loaded {len(depth_frames)} depth frames: {depth_frames.shape}, "
        f"range [{depth_frames.min():.3f}m, {depth_frames.max():.3f}m]"
    )

    return depth_frames
