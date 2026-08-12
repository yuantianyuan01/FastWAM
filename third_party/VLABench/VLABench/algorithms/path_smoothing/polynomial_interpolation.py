import numpy as np
from scipy.interpolate import interp1d

def polynomial_smoothing(path, num_points=100):
    """
    Smooth the path using polynomial interpolation.
    
    Args:
    - path: List of waypoints in the path [(x1, y1, z1), (x2, y2, z2), ...]
    - num_points: Number of points to interpolate between the waypoints.
    
    Returns:
    - A smoothed path.
    """
    path = np.array(path)
    t = np.linspace(0, 1, len(path))
    interp_x = interp1d(t, path[:, 0], kind='cubic')
    interp_y = interp1d(t, path[:, 1], kind='cubic')
    interp_z = interp1d(t, path[:, 2], kind='cubic')
    
    t_new = np.linspace(0, 1, num_points)
    smooth_path = np.vstack((interp_x(t_new), interp_y(t_new), interp_z(t_new))).T
    
    return smooth_path