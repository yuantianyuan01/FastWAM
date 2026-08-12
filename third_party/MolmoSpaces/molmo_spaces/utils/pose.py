import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_mat_to_7d(pose_matrix: np.ndarray) -> np.ndarray:
    """Convert 4x4 pose matrix to 7D vector (x, y, z ,qw, qx, qy, qz)."""
    assert pose_matrix.shape == (4, 4)
    pos = pose_matrix[:3, 3]
    rot_quat = R.from_matrix(pose_matrix[:3, :3]).as_quat(scalar_first=True)  # Returns [w, x, y, z]
    return np.concatenate([pos, rot_quat])


def pos_quat_to_pose_mat(
    pos: np.ndarray | list, quat: np.ndarray | list | None = None
) -> np.ndarray:
    if quat is None:
        assert len(pos) == 7
        quat = pos[3:7]
        pos = pos[0:3]

    assert len(pos) == 3
    assert len(quat) == 4
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    pose_matrix[:3, 3] = pos
    return pose_matrix


def pose_mat_to_pos_quat(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = pose[:3, 3]
    quat = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
    return pos, quat


def compute_lookat_forward_up(
    camera_pos: np.ndarray,
    lookat_target: np.ndarray,
    camera_up: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute forward and up unit vectors for a camera looking at a target.

    Args:
        camera_pos: Camera position in world frame.
        lookat_target: Point to look at in world frame.
        camera_up: Desired up direction. Defaults to world Z-up [0, 0, 1].

    Returns:
        (forward, up) unit vectors in world frame.
    """
    forward = np.asarray(lookat_target, dtype=float) - np.asarray(camera_pos, dtype=float)
    forward = forward / np.linalg.norm(forward)

    if camera_up is None:
        camera_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, camera_up)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-6:
        fallback_ref = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(forward, fallback_ref)) > 0.9:
            fallback_ref = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, fallback_ref)
        right = right / np.linalg.norm(right)
    else:
        right = right / right_norm

    up = np.cross(right, forward)
    return forward, up
