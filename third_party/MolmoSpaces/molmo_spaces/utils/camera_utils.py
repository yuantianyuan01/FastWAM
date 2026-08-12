import numpy as np
from scipy.ndimage import binary_erosion


def normalize_points(
    points: np.ndarray,
    img_width: int,
    img_height: int,
    distortion_map: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize image points to 0-1 range, optionally applying distortion correction.

    Args:
        points: Array of shape (N, 2) containing (x, y) pixel coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        distortion_map: Optional distortion map for warped cameras (e.g., GoPro)
                       Currently not implemented - will be added in future

    Returns:
        Normalized points in 0-1 range as array of shape (N, 2)
    """
    # Apply distortion correction if provided
    if distortion_map is not None:
        raise NotImplementedError("Distortion map correction not yet implemented")

    # Normalize to 0-1 range
    normalized_points = points.copy().astype(np.float32)
    normalized_points[:, 0] /= img_width  # x coordinate
    normalized_points[:, 1] /= img_height  # y coordinate

    return normalized_points


def erode_segmentation_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Apply binary erosion to a segmentation mask.

    Args:
        mask: Binary segmentation mask
        iterations: Number of erosion iterations

    Returns:
        Eroded binary mask
    """
    return binary_erosion(mask, iterations=iterations)
