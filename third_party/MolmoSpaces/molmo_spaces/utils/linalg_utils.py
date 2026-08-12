from functools import wraps

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R


def inverse_homogeneous_matrix(matrix: np.ndarray):
    """
    Compute the inverse of a 4x4 homogeneous transformation matrix.

    Args:
    matrix (numpy.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
    numpy.ndarray: The inverse of the input matrix.
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix.")

    rotation_matrix = matrix[0:3, 0:3]
    translation_vector = matrix[0:3, 3]

    inverse_rotation = np.transpose(rotation_matrix)
    inverse_translation = -np.dot(inverse_rotation, translation_vector)

    inverse_matrix = np.identity(4)
    inverse_matrix[0:3, 0:3] = inverse_rotation
    inverse_matrix[0:3, 3] = inverse_translation
    return inverse_matrix


def swing_twist(quat: np.ndarray, axis: np.ndarray):
    """
    Decomposes quat into a rotation around axis and a rotation around an
    axis perpendicular to axis.

    Note: Assumes quaternions are [w,x,y,z]

    Returns quaternions (swing, twist) where quat = swing * twist, and
    twist is a rotation around axis
    """
    axis = axis.astype(np.float64) / np.linalg.norm(axis)
    rot_ax = quat[1:]
    p = np.dot(rot_ax, axis) * axis
    twist = np.hstack([quat[:1], p])
    twist /= np.linalg.norm(twist)
    quat_rot = R.from_quat(quat, scalar_first=True)
    twist_rot = R.from_quat(twist, scalar_first=True)
    swing = (quat_rot * twist_rot.inv()).as_quat(scalar_first=True)
    return swing, twist


def quat_to_euler_yaw(quat):
    """
    Convert quaternion (w, x, y, z) to euler yaw (radians)
    """
    return R.from_quat(quat, scalar_first=True).as_euler("xyz", degrees=False)[2]


def euler_yaw_to_quat(yaw):
    """
    Convert euler (0, 0, yaw) to quat (w, x, y, z)
    """
    return R.from_euler("xyz", [0, 0, yaw], degrees=False).as_quat(scalar_first=True)


def normalize_ang_error(ang):
    # Normalize to [-pi, pi] range
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    return ang


def global_to_relative_transform(x, base):
    return inverse_homogeneous_matrix(base) @ x


def relative_to_global_transform(x, base):
    return base @ x


def skew(v: np.ndarray):
    """
    Compute the skew-symmetric matrix of a 3D vector.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def transform_to_twist(T: np.ndarray):
    """
    Given a 4x4 transformation matrix, return the twist as (lin_vel, ang_vel).
    Mathematically, this is computing the logarithmic map of SE(3). Equivalent to pin.log6.

    See: https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf (Sec 9.4.2)
    """
    w = R.from_matrix(T[:3, :3]).as_rotvec()
    theta = np.linalg.norm(w)
    if np.abs(theta) < 1e-6:
        return T[:3, 3], w
    V = (
        np.eye(3)
        + (1 - np.cos(theta)) / theta**2 * skew(w)
        + (theta - np.sin(theta)) / theta**3 * np.dot(skew(w), skew(w))
    )
    t = np.linalg.solve(V, T[:3, 3])
    return t, w


def twist_to_transform(lin_vel: np.ndarray, ang_vel: np.ndarray):
    """
    Given a linear velocity and angular velocity, return the 4x4 transformation matrix.
    Mathematically, this is computing the exponential map of SE(3). Equivalent to pin.exp6.

    See: https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf (Sec 9.4.2)
    """
    theta = np.linalg.norm(ang_vel)
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(ang_vel).as_matrix()
    if np.abs(theta) < 1e-6:
        V = np.eye(3)
    else:
        V = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta**2 * skew(ang_vel)
            + (theta - np.sin(theta)) / theta**3 * np.dot(skew(ang_vel), skew(ang_vel))
        )
    T[:3, 3] = V @ lin_vel
    return T


def interp(
    x: ArrayLike,
    xp: ArrayLike,
    fp: ArrayLike,
    left: ArrayLike | None = None,
    right: ArrayLike | None = None,
):
    """
    Linear interpolation of vector-valued functions of scalars. Similar to np.interp but for multi-dimensional arrays.
    """
    x = np.asarray(x)
    is_batch = x.ndim > 0
    x = x.reshape(-1)
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    if len(fp.shape) == 1:
        fp = fp.reshape(-1, 1)
    assert len(xp.shape) == 1 and xp.shape[0] == fp.shape[0]

    # Handle out of bounds
    ret = np.zeros((x.shape[0], fp.shape[-1]), fp.dtype)
    lt_mask = x <= xp[0]
    gt_mask = x > xp[-1]
    if np.any(lt_mask):
        ret[lt_mask] = left if left is not None else fp[0]
    if np.any(gt_mask):
        ret[gt_mask] = right if right is not None else fp[-1]

    in_bounds_mask = ~lt_mask & ~gt_mask
    x_in_bounds = x[in_bounds_mask]
    i = np.searchsorted(xp, x_in_bounds)

    x0, x1 = xp[i - 1], xp[i]
    f0, f1 = fp[i - 1], fp[i]
    ret[in_bounds_mask] = f0 + (f1 - f0) / (x1 - x0)[:, None] * (x_in_bounds - x0)[:, None]
    return ret if is_batch else ret[0]


def single_or_batch(func):
    """
    Decorator to allow a function to accept a single input or a batch of inputs.
    The decorated function should always accept and return batches.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        idx = 1 if len(args) > 0 and hasattr(args[0], "__dict__") else 0
        x = np.asarray(args[idx])
        if not_batch := x.ndim == 1:
            x = x.reshape(1, -1)
        ret = func(*args[:idx], x, *args[idx + 1 :], **kwargs)
        return ret[0] if not_batch else ret

    return wrapper


@single_or_batch
def homogenize(x: np.ndarray):
    """
    Project a vector to homogenous coordinates. Accepts either a single vector or a batch.
    """
    assert x.ndim == 2
    return np.hstack([x, np.ones((x.shape[0], 1))])


def obb_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the oriented bounding box (OBB) of a set of 2D points.
    Parameters:
    points (np.ndarray): A 2D numpy array of shape (N, 2) representing the coordinates of the points.

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - pos (np.ndarray): The center position of the OBB.
        - minor_axis (np.ndarray): The minor axis of the OBB, i.e. half the shorter side.
        - major_axis (np.ndarray): The major axis of the OBB, i.e. half the longer side.
    """
    points = np.asarray(points)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    edges = np.diff(np.concatenate([hull_points, hull_points[:1]], axis=0), axis=0)
    x_axes = edges / np.linalg.norm(edges, axis=1)[:, None]
    y_axes = np.column_stack([-x_axes[:, 1], x_axes[:, 0]])
    rotmats = np.stack([x_axes, y_axes], axis=2)

    rot_points = np.expand_dims(points, axis=0) @ rotmats.transpose(0, 2, 1)
    rot_merged_mins = np.min(rot_points, axis=1)
    rot_merged_maxs = np.max(rot_points, axis=1)
    areas = np.prod(rot_merged_maxs - rot_merged_mins, axis=1)
    best_bbox_idx = np.argmin(areas)

    rotmat = rotmats[best_bbox_idx]
    rot_merged_min = rot_merged_mins[best_bbox_idx]
    rot_merged_max = rot_merged_maxs[best_bbox_idx]
    pos = rotmat.T @ (rot_merged_min + rot_merged_max) / 2
    half_size = (rot_merged_max - rot_merged_min) / 2
    minor_axis, major_axis = sorted(rotmat * half_size.reshape(-1, 1), key=np.linalg.norm)
    best_box = (pos, minor_axis, major_axis)
    return best_box
