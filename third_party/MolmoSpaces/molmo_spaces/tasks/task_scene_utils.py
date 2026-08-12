import numpy as np
from mujoco import mjtGeom


def add_placement_target(spec, pos=None, randomize=False, name="place_target"):
    """
    Add a placement target (red cylinder) to the scene.

    Args:
        spec: MuJoCo MjSpec object
        pos: [x, y, z] position. If None, uses default or random
        randomize: Whether to randomize the position
        name: Name for the target body

    Returns:
        The created target body
    """

    if pos is None:
        if randomize:
            # Random position on table surface (approximate table bounds)
            pos = [
                np.random.uniform(-0.4, 0.4),  # x: table width
                np.random.uniform(0.2, 1.0),  # y: table depth
                0.71,  # z: table height
            ]
        else:
            pos = [-0.1, 0.4, 0.71]  # Default position from XML

    # Create target body
    target_body = spec.worldbody.add_body(name=name, pos=pos, mocap=True)

    # Add red cylinder geometry
    target_body.add_geom(
        name=f"{name}_geom",  # Add geometry name for identification
        type=mjtGeom.mjGEOM_CYLINDER,
        size=[0.05, 0.001, 0],  # For cylinders in Python API: [radius, radius, half-height]
        rgba=[1, 0, 0, 1],  # Red color
        group=2,  # Visual group
    )

    return spec, target_body


def add_pickup_target(spec, pos=None, randomize=False, name="obj_0", color=[0, 1, 0, 1]):
    """
    Add a pickup target (cube) to the scene.

    Args:
        spec: MuJoCo MjSpec object
        pos: [x, y, z] position. If None, uses default or random
        randomize: Whether to randomize the position
        name: Name for the object body
        color: RGBA color for the cube

    Returns:
        tuple: (spec, obj_body)
    """
    if pos is None:
        if randomize:
            pos = [np.random.uniform(-0.4, 0.4), np.random.uniform(0.2, 1.0), 0.735]
        else:
            pos = [6.0, 3.0, 0.76]

    obj_body = spec.worldbody.add_body(name=name, pos=pos)
    obj_body.add_freejoint()
    obj_body.add_geom(
        name=f"{name}_geom",
        type=mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0.025],
        size=[0.025, 0.025, 0.025],
        rgba=color,
    )

    return spec, obj_body
