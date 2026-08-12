import numpy as np
from scipy.spatial.transform import Rotation, Rotation as R
import torch


def euler_angles_to_quat(
    euler_angles: torch.Tensor,
    degrees: bool = False,
    extrinsic: bool = False,
    device=None,
) -> torch.Tensor:
    """Vectorized version of converting euler angles to quaternion (scalar first)

    Args:
        euler_angles (typing.Union[np.ndarray, torch.Tensor]): euler angles with shape (N, 3)
        degrees (bool, optional): True if degrees, False if radians. Defaults to False.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        typing.Union[np.ndarray, torch.Tensor]: quaternions representation of the angles (N, 4) - scalar first.
    """
    if not isinstance(euler_angles, torch.Tensor):
        euler_angles = torch.tensor(euler_angles, dtype=torch.float32)
    if extrinsic:
        order = "xyz"
    else:
        order = "XYZ"
    rot = Rotation.from_euler(order, euler_angles.cpu().numpy(), degrees=degrees)
    result = rot.as_quat()
    if len(result.shape) == 1:
        result = result[[3, 0, 1, 2]]
    else:
        result = result[:, [3, 0, 1, 2]]
    result = torch.from_numpy(np.asarray(result, dtype=np.float32)).float().to(device)
    return result


def euler_to_quat(euler: list[float], order="xyz") -> list[float]:
    """Convert Euler angles (degrees) to quaternion in (w, x, y, z) format.

    Args:
        euler: Euler angles in (x, y, z) order (degrees)

    Returns:
        Quaternion in (w, x, y, z) format
    """
    rot = R.from_euler(order, euler, degrees=True)
    quat = rot.as_quat()  # Returns (x, y, z, w)
    return [quat[3], quat[0], quat[1], quat[2]]  # Convert to (w, x, y, z)
