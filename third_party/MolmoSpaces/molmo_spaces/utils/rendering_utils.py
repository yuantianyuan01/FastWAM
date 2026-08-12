import mujoco
import numpy as np
from mujoco import MjModel

from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms


def get_geom_seg_mask(model: MjModel, seg: np.ndarray, body_id: int) -> np.ndarray:
    """
    Get a mask of all geoms descended from a body in a segmentation mask.

    Args:
        model (MjModel): The model to use.
        seg (np.ndarray): The (H, W, 2) segmentation mask, as returned by the renderer.
        body_id (int): The id of the body to get the mask for.

    Returns:
        np.ndarray: A (H, W) mask of the geoms descended from the body.
    """
    geoms = descendant_geoms(model, body_id)
    is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
    is_geom_of_body = np.isin(seg[..., 0], geoms)
    return is_geom & is_geom_of_body
