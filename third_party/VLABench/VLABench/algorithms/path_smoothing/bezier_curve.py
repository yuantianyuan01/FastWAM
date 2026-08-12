import numpy as np
from scipy.interpolate import splprep, splev

def bezier_smoothing(path, smoothing_factor=0.1):
    """
    Smooth the path using a Bézier curve fitting technique.
    
    Args:
    - path: List of waypoints in the path [(x1, y1, z1), (x2, y2, z2), ...]
    - smoothing_factor: A parameter to control the smoothing.
    
    Returns:
    - A smoothed path.
    """
    if len(path) <= 3:
        return np.array(path)
    path = np.array(path)
    tck, u = splprep(path.T, s=smoothing_factor)  # Spline preparation
    u_fine = np.linspace(0, 1, num=len(path) * 10)  # Higher resolution for smoothness
    smooth_path = np.array(splev(u_fine, tck)).T  # Smoothed points

    return smooth_path