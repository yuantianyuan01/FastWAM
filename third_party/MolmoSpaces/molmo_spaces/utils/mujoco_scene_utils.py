import logging

import mujoco
import numpy as np
from mujoco import MjData, MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.tasks.task_sampler_errors import ObjectPlacementError
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb, geom_aabb

log = logging.getLogger(__name__)


def add_visual_capsule(scene, point1, point2, radius, rgba) -> None:
    """Adds one capsule to an mjvScene.
    these geometries are automatically visual-only and don't participate in collision detection
    """
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_connector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom - 1], mujoco.mjtGeom.mjGEOM_CAPSULE, radius, point1, point2
    )


def randomize_door_joints(  # TODO: do these defaults make sense?
    spec: MjSpec,
    scene_metadata: dict,
    door_stiffness_range: tuple = (3, 7),
    door_damping_range: tuple = (8, 12),
    door_frictionloss_range: tuple = (8, 12),
    handle_stiffness_range: tuple = (200, 300),
    handle_damping_range: tuple = (80, 120),
    handle_frictionloss_range: tuple = (40, 60),
    add_handle_limits: bool = True,
) -> None:
    """
    Modify door and handle joint parameters in a house spec.

    This function identifies door joints and handle joints by their naming patterns and
    modifies their physical parameters (stiffness, damping, frictionloss) with randomized
    values within specified ranges.

    It also sets the ref and springref based on range heuristics.

    Args:
        spec: The model spec
        door_stiffness_range: (min, max) range for door joint stiffness (default: reduce from ~250 to 3-7)
        door_damping_range: (min, max) range for door joint damping (default: reduce from ~100 to 8-12)
        door_frictionloss_range: (min, max) range for door joint frictionloss (default: reduce from ~50 to 8-12)
        handle_stiffness_range: (min, max) range for handle joint stiffness (default: increase from ~0 to 200-300)
        handle_damping_range: (min, max) range for handle joint damping (default: increase from ~0.1 to 80-120)
        handle_frictionloss_range: (min, max) range for handle joint frictionloss (default: increase from ~0 to 40-60)
        add_handle_limits: Whether to add limited="true" and ref/springref attributes to handle joints
    """
    log.debug("Starting joint modifications")
    scene_objects = scene_metadata.get("objects", {})
    log.debug(f"Found {len(scene_objects)} scene objects to check")
    handle_joints = []
    door_joints = []
    for key, value in scene_objects.items():
        if "doorway" in key:
            name_map = value.get("name_map", {})
            joints = name_map.get("joints", {})
            if len(joints) > 0:
                for joint in joints:
                    if "handle" in joint:
                        handle_joints.append(joint)
                    else:
                        door_joints.append(joint)

    log.debug(f"Found {len(door_joints)} door joints: {door_joints}")
    log.debug(f"Found {len(handle_joints)} handle joints: {handle_joints}")

    modifications_count = {"doors": 0, "handles": 0}

    for joint in spec.joints:
        joint: mujoco.MjsJoint

        name = joint.name or ""
        is_door_joint = name in door_joints
        is_handle_joint = name in handle_joints

        if not (is_door_joint or is_handle_joint):
            continue

        if is_door_joint:
            log.debug(f"[DOOR JOINT MOD] Modifying door joint: {name}")

            # only adust the ref/springref if the joint is not already set to be a spring
            if joint.stiffness == 0.0:
                joint.springref = joint.range[0].item()

            old_stiffness = joint.stiffness
            old_damping = joint.damping
            old_frictionloss = joint.frictionloss
            joint.stiffness = np.random.uniform(*door_stiffness_range)
            joint.damping = np.random.uniform(*door_damping_range)
            joint.frictionloss = np.random.uniform(*door_frictionloss_range)
            joint.limited = 1
            if joint.armature == 0.0:
                joint.armature = 1.0

            modifications_count["doors"] += 1

            log.debug(f"[DOOR JOINT MOD] stiffness: {old_stiffness} -> {joint.stiffness}")
            log.debug(f"[DOOR JOINT MOD] damping: {old_damping} -> {joint.damping}")
            log.debug(f"[DOOR JOINT MOD] frictionloss: {old_frictionloss} -> {joint.frictionloss}")
        elif is_handle_joint:
            log.debug(f"Modifying handle joint: {name}")

            if add_handle_limits:
                joint.limited = 1
                if joint.stiffness == 0.0:
                    joint.springref = joint.range[0].item()

            old_stiffness = joint.stiffness
            old_damping = joint.damping
            old_frictionloss = joint.frictionloss
            joint.stiffness = np.random.uniform(*handle_stiffness_range)
            joint.damping = np.random.uniform(*handle_damping_range)
            joint.frictionloss = np.random.uniform(*handle_frictionloss_range)
            joint.armature = 1.0  # TODO(abhay): why set armature to 1.0?
            modifications_count["handles"] += 1

            log.debug(f"[HANDLE JOINT MOD] stiffness: {old_stiffness} -> {joint.stiffness}")
            log.debug(f"[HANDLE JOINT MOD] damping: {old_damping} -> {joint.damping}")
            log.debug(
                f"[HANDLE JOINT MOD] frictionloss: {old_frictionloss} -> {joint.frictionloss}"
            )

    log.debug(
        f"Completed joint modifications: {modifications_count['doors']} door joints, {modifications_count['handles']} handle joints"
    )


def place_object_near(
    data: MjData,
    object_id: int,
    placement_point: np.ndarray,
    min_dist: float,
    max_dist: float,
    max_tries: int = 100,
    reference_pos: np.ndarray | None = None,
    max_dist_to_reference: float = 1.0,
    supporting_geom_id: int | None = None,
    z_eps: float = 1e-3,
):
    """
    Place an object near a point such that the bottom of the object (i.e. the base) is at the specified z-value, with a random yaw.
    Optionally, ensure the placed object is within a certain distance of a reference position.

    Args:
        data: MjData object
        object_id: ID of the object to place
        placement_point: Point to place the object near
        min_dist: Minimum distance from the placement point
        max_dist: Maximum distance from the placement point
        max_tries: Maximum number of placement attempts
        reference_pos: Reference position to place the object near
        max_dist_to_reference: Maximum distance to the reference position
        supporting_geom_id: ID of the supporting geometry to optionally ensure the object is placed on top of
        z_eps: Epsilon to add to the z-offset to avoid collision

    Raises:
        ObjectPlacementError: If the object cannot be placed within the specified number of attempts
    """
    object_body = create_mlspaces_body(data, object_id)
    original_pose = object_body.pose

    body_aabb_center, body_aabb_size = body_aabb(data.model, data, object_id)
    z_offset = object_body.position[2] - (body_aabb_center[2] - body_aabb_size[2] / 2)

    if supporting_geom_id is not None:
        support_geom_aabb_center, support_geom_aabb_size = geom_aabb(
            data.model, data, [supporting_geom_id]
        )
        placement_pos_min = (
            support_geom_aabb_center[:2] - support_geom_aabb_size[:2] / 2 + body_aabb_size[:2] / 4
        )
        placement_pos_max = (
            support_geom_aabb_center[:2] + support_geom_aabb_size[:2] / 2 - body_aabb_size[:2] / 4
        )
        supporting_root_body_id = data.model.body_rootid[data.model.geom_bodyid[supporting_geom_id]]
        placement_point = placement_point.copy()
        placement_point[2] = support_geom_aabb_center[2] + support_geom_aabb_size[2] / 2
    else:
        placement_pos_min = np.full(2, -np.inf)
        placement_pos_max = np.full(2, np.inf)
        supporting_root_body_id = None

    # first generate the candidate placement positions that satisfy the distance constraints
    candidate_placement_pos_xy = np.zeros((max_tries, 2))
    n_candidates = 0
    i = 0
    while n_candidates < max_tries:
        N = 1024
        azimuth = np.random.uniform(-np.pi, np.pi, N)
        distance = np.random.uniform(min_dist, max_dist, N)
        xy_offset = distance.reshape(-1, 1) * np.stack([np.cos(azimuth), np.sin(azimuth)], axis=1)
        placement_pos_xy = placement_point[:2][None] + xy_offset

        eligible_mask = np.all(
            (placement_pos_min <= placement_pos_xy) & (placement_pos_xy <= placement_pos_max),
            axis=1,
        )
        if reference_pos is not None:
            dist_to_reference = np.linalg.norm(placement_pos_xy - reference_pos[:2][None], axis=1)
            eligible_mask &= dist_to_reference <= max_dist_to_reference

        new_n_candidates = min(max_tries, n_candidates + eligible_mask.sum())
        candidate_placement_pos_xy[n_candidates:new_n_candidates] = placement_pos_xy[eligible_mask][
            : new_n_candidates - n_candidates
        ]
        n_candidates = new_n_candidates

        i += 1
        if i >= max_tries:
            log.debug(
                f"Failed to sample {max_tries} candidate placement positions within {max_tries} attempts"
            )
            candidate_placement_pos_xy = candidate_placement_pos_xy[:n_candidates]
            break

    # for each candidate placement position, try to place the object and check for collisions
    for attempt, placement_pos_xy in enumerate(candidate_placement_pos_xy):
        yaw = np.random.uniform(-np.pi, np.pi)
        placement_pos = np.array(
            [placement_pos_xy[0], placement_pos_xy[1], placement_point[2] + z_offset + z_eps]
        )
        placement_pose = np.eye(4)
        placement_pose[:3, 3] = placement_pos
        placement_pose[:3, :3] = R.from_euler("z", yaw).as_matrix() @ original_pose[:3, :3]
        object_body.pose = placement_pose

        mujoco.mj_fwdPosition(data.model, data)

        in_collision = False
        # TODO(abhayd): do we need to place on the same surface? Why not just any surface?
        for c in data.contact:
            root_body1 = data.model.body_rootid[data.model.geom_bodyid[c.geom1]]
            root_body2 = data.model.body_rootid[data.model.geom_bodyid[c.geom2]]
            if (root_body1 == object_id) ^ (root_body2 == object_id):
                other_root_body = root_body1 if root_body1 != object_id else root_body2
                if other_root_body != supporting_root_body_id:
                    in_collision = True
                    break

        if not in_collision:
            log.debug(
                f"Successfully placed object with ID {object_id} after {attempt + 1} attempts"
            )
            break

    else:
        object_body.pose = original_pose
        mujoco.mj_forward(data.model, data)
        raise ObjectPlacementError(
            f"Failed to place object with ID {object_id} within {max_tries} attempts"
        )


def get_supporting_geom(
    data: MjData, object_id: int, angle_threshold: float = np.radians(80)
) -> int | None:
    """
    Finds the supporting geometry for an object, using a heuristic.
    Searches for a geom in contact with the object, such that the contact is in the bottom half of the object's AABB and the normal is pointing upwards.

    Args:
        data: MjData object
        object_id: Body ID of the root body to find the supporting geometry for
        angle_threshold: Threshold for the angle between the normal and the vertical axis to be considered parallel, in radians

    Returns:
        int: Geom ID of the supporting geometry, or None if no supporting geometry is found
    """
    model = data.model
    assert model.body_rootid[object_id] == object_id, "Object is not a root body"

    try:
        body_aabb_center, _ = body_aabb(model, data, object_id, visible_only=True)
    except ValueError:
        # fallback if body doesn't have any visible geoms (usually not the case)
        body_aabb_center, _ = body_aabb(model, data, object_id, visible_only=False)
    cos_threshold = np.cos(angle_threshold)

    for c in data.contact:
        root_body1, root_body2 = model.body_rootid[model.geom_bodyid[c.geom]]
        if (root_body1 == object_id) ^ (root_body2 == object_id):
            other_geom_id = c.geom[0] if root_body1 != object_id else c.geom[1]
            normal = c.frame[:3] / np.linalg.norm(c.frame[:3])
            if root_body1 == object_id:
                normal = -normal
            if c.pos[2] < body_aabb_center[2] and normal[2] >= cos_threshold:
                return other_geom_id
    return None


def is_object_supported_by_body(
    data: MjData,
    object_id: int,
    support_id: int,
    angle_threshold: float = np.radians(30),
    frac_weight_threshold: float = 0.5,
    eps: float = 1e-6,
) -> bool:
    """
    Checks if an object is supported by a given body, using heuristics.
    This is more precise than get_supporting_geom.

    Args:
        data: MjData object
        object_id: Body ID of the root body to check if it is supported by the supporting body
        support_id: Body ID of the supporting body to check if it is supporting the object
        angle_threshold: Threshold for the angle between the normal and the vertical axis to be considered parallel, in radians
        frac_weight_threshold: The upward component of the contact force must be at least this fraction of the object weight to be considered supported
        eps: Threshold for the net contact force to be considered non-zero

    Returns:
        bool: True if the object is supported by the given support, False otherwise
    """
    model = data.model
    assert model.body_rootid[object_id] == object_id, "Object is not a root body"
    body_rootid = model.body_rootid[support_id]

    net_force = np.zeros(3)

    for cid in range(data.ncon):
        c = data.contact[cid]
        root_body1, root_body2 = model.body_rootid[model.geom_bodyid[c.geom]]

        # only check contacts between the object and the body
        if {root_body1, root_body2} == {body_rootid, object_id}:
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, cid, contact_force)
            if root_body1 == object_id:
                contact_force = -contact_force

            contact_rotmat = c.frame.reshape(3, 3).T
            contact_force_world = contact_rotmat @ contact_force[:3]
            net_force += contact_force_world

    if np.linalg.norm(net_force) < eps:
        # no contact between objects
        return False

    cos_threshold = np.cos(angle_threshold)
    cos_to_z = net_force[2] / np.linalg.norm(net_force).item()
    contact_is_vertical = cos_to_z >= cos_threshold

    object_weight: float = (
        model.body_subtreemass[object_id] * np.linalg.norm(model.opt.gravity)
    ).item()
    is_supporting_weight = np.abs(net_force[2]).item() >= frac_weight_threshold * object_weight

    # TODO: this fails to capture transitive support, e.g. if the object is on another body which is on the support.
    # This is an unlikely edge case, so we'll leave it for now.
    # Future solutions could consider building a contact graph between the object and the support's support (e.g. the table)
    # and checking that a sufficient amount of weight is bottlenecked through the support node. This would handle
    # both transitive support and multiple supports.

    return contact_is_vertical and is_supporting_weight
