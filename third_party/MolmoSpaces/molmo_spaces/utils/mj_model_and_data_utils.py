import itertools

import mujoco
import numpy as np
from mujoco import MjModel, mjtObj

from molmo_spaces.utils.pose import pos_quat_to_pose_mat


def extract_mj_names(model, name_adr: np.ndarray | None, num_obj: int, obj_type: mjtObj):
    """
    See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1127
    """
    # objects don't need to be named in the XML, so name might be None
    id2name = {i: None for i in range(num_obj)}
    name2id = {}
    for i in range(num_obj):
        name = mujoco.mj_id2name(model, obj_type, i)
        name2id[name] = i
        id2name[i] = name

    # sort names by increasing id to keep order deterministic
    return tuple(id2name[nid] for nid in sorted(name2id.values())), name2id, id2name


def descendant_bodies(model: MjModel, body_id: int):
    """
    Get all bodies descended from a body in a MuJoCo model.

    Args:
        model (MjModel): The MuJoCo model to use.
        body_id (int): The id of the body to get the descendants of.

    Returns:
        set[int]: A set of the ids of the bodies descended from the body, including the body itself.
    """
    if body_id == 0:
        return set(range(model.nbody))

    descendants = {body_id}
    for bid in np.where(model.body_parentid == body_id)[0]:
        descendants.update(descendant_bodies(model, bid))
    return descendants


def descendant_geoms(model: MjModel, body_id: int, visible_only: bool = True) -> list[int]:
    """
    Get all geoms attached to descendants of a body in a MuJoCo model.

    Args:
        model (MjModel): The MuJoCo model to use.
        body_id (int): The id of the body to get the geoms of.
        visible_only (bool): Whether to only include visible geoms (groups 0-2).

    Returns:
        list[int]: A sorted list of the ids of the geoms attached to descendants of the body, or the body itself.
    """
    bodies = np.array(list(descendant_bodies(model, body_id)))
    mask = np.any(model.geom_bodyid.reshape(1, -1) == bodies.reshape(-1, 1), axis=0)
    geoms = np.where(mask)[0]
    if visible_only:
        is_visible = model.geom_group[geoms] < 3
        geoms = geoms[is_visible]
    return geoms.tolist()


def body_aabb(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, visible_only: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the axis-aligned bounding box (AABB) for a body in a MuJoCo model.

    Args:
        model (mujoco.MjModel): The MuJoCo model containing the body.
        data (mujoco.MjData): The MuJoCo data containing the state of the model.
        body_id (int): The id of the body to compute the AABB for.
        visible_only (bool): Whether to only include visible geoms (groups 0-2).
            This can help make the AABB fit tighter.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The center of the AABB in world space.
            - numpy.ndarray: The x,y,z dimensions of the AABB.
    """
    geoms = descendant_geoms(model, body_id, visible_only=visible_only)
    if not geoms:
        # If body has no geoms, return body position as center with zero extent
        return data.xpos[body_id].copy(), np.zeros(3)
    return geom_aabb(model, data, geoms)


def mesh_aabb(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the tight AABB in world space for a mesh geom using its vertices.

    Args:
        model (mujoco.MjModel): The MuJoCo model containing the geom.
        data (mujoco.MjData): The MuJoCo data containing the state of the model.
        geom_id (int): The id of the mesh geom to compute the AABB for. Must be a mesh geom.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The center of the AABB in world space.
            - numpy.ndarray: The x,y,z dimensions of the AABB.
    """
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH.value
    mesh_id = model.geom_dataid[geom_id]
    vertadr = model.mesh_vertadr[mesh_id]
    n_vert = model.mesh_vertnum[mesh_id]

    geom_rel_pose = pos_quat_to_pose_mat(model.geom_pos[geom_id], model.geom_quat[geom_id])
    geom_body_id = model.geom_bodyid[geom_id]
    geom_pose = body_pose(data, geom_body_id) @ geom_rel_pose

    vertices_local = model.mesh_vert[vertadr : vertadr + n_vert]
    vertices = vertices_local @ geom_pose[:3, :3].T + geom_pose[:3, 3]

    aabb_min = np.min(vertices, axis=0)
    aabb_max = np.max(vertices, axis=0)
    return (aabb_min + aabb_max) / 2, aabb_max - aabb_min


def geom_aabb(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int], tight_mesh: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the axis-aligned bounding box (AABB) for a list of geometries in a MuJoCo model.

    Args:
        model (mujoco.MjModel): The MuJoCo model containing the geometries.
        data (mujoco.MjData): The MuJoCo data containing the state of the model.
        geom_ids (list[int]): A list of geometry IDs for which to compute the AABB.
        tight_mesh (bool): Whether to compute the tight AABB for mesh geoms.
            If False, the AABB will be computed using the geom_aabb field, and may not be tight in world space.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The center of the merged AABB in world space.
            - numpy.ndarray: The x,y,z dimensions of the merged AABB.
    """
    if not geom_ids:
        # If no geoms provided, return zero-sized AABB at origin
        return np.zeros(3), np.zeros(3)

    vertices = []
    corners = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
    for geom_id in geom_ids:
        if tight_mesh and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH.value:
            mesh_aabb_center, mesh_aabb_size = mesh_aabb(model, data, geom_id)
            vertices.append(mesh_aabb_center + corners * mesh_aabb_size / 2)
        else:
            geom_rotmat = data.geom_xmat[geom_id].reshape(3, 3)
            geom_pos = data.geom_xpos[geom_id]

            aabb = model.geom_aabb[geom_id]
            local_corners = aabb[:3] + corners * aabb[3:]
            world_corners = local_corners @ geom_rotmat.T + geom_pos
            vertices.append(world_corners)

    # merge aabbs
    vertices = np.concatenate(vertices, axis=0)
    merged_min = np.min(vertices, axis=0)
    merged_max = np.max(vertices, axis=0)
    return (merged_min + merged_max) / 2, merged_max - merged_min


def body_pose(data: mujoco.MjData, body_id: int) -> np.ndarray:
    trf = np.eye(4)
    trf[:3, 3] = data.xpos[body_id]
    trf[:3, :3] = data.xmat[body_id].reshape(3, 3)
    return trf


def site_pose(data: mujoco.MjData, site_id: int) -> np.ndarray:
    trf = np.eye(4)
    trf[:3, 3] = data.site_xpos[site_id]
    trf[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    return trf


def body_base_pos(data: mujoco.MjData, body_id: int, visible_only: bool = True) -> np.ndarray:
    """
    Returns the base position of a body in the world frame.
    In XY, this is the center of the AABB, and in Z, this is the bottom of the AABB.

    Args:
        data: MjData object
        body_id: ID of the body to get the base position of.
        visible_only (bool): Whether to only include visible geoms (groups 0-2).
            This can help make the AABB fit tighter.

    Returns:
        np.ndarray: The base position of the body in the world frame, of shape (3,).
    """
    body_aabb_center, body_aabb_size = body_aabb(
        data.model, data, body_id, visible_only=visible_only
    )
    return body_aabb_center - np.array([0, 0, body_aabb_size[2] / 2])
