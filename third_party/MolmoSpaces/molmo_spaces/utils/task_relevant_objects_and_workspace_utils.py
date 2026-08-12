"""Derive task-relevant object names and workspace center from task config fields.

Single source of truth for which objects cameras must see and what defines the
workspace center. Called from:
- Task samplers (resolve_visibility_object, get_workspace_center) during data generation
- create_json_benchmark.py to populate EpisodeSpec.task_relevant_objects
- Eval camera system for visibility checks and workspace center computation

Accepts either a pydantic config object or a plain dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get a value from a pydantic model or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_task_relevant_objects(task_config: Any) -> list[str]:
    """Return the list of object body names that are relevant for this task.

    These are the objects that cameras should be able to see (visibility
    constraints) and whose positions define the workspace center.

    Args:
        task_config: A task config object (e.g. PickTaskConfig) or a dict
            with the same keys (as stored in EpisodeSpec.task).

    Returns:
        Deduplicated list of object body names, in stable insertion order.
    """
    seen: set[str] = set()
    result: list[str] = []

    def _add(name: str | None) -> None:
        if name and name not in seen:
            seen.add(name)
            result.append(name)

    # Pickup / target object (pick, pnp, open/close, nav)
    _add(_get(task_config, "pickup_obj_name"))

    # Place receptacle (pnp, next-to, color)
    _add(_get(task_config, "place_receptacle_name"))

    # Distractor receptacles (pnp color)
    for name in _get(task_config, "other_receptacle_names") or []:
        _add(name)

    return result


def compute_workspace_center(positions: dict[str, np.ndarray]) -> np.ndarray:
    """Compute the workspace center as the centroid of named 3-D positions.

    This is the shared implementation used by both live task samplers (positions
    from the environment) and the eval camera system (positions from JSON episode
    data).

    Args:
        positions: Mapping of label -> 3-D position array.  Typical keys are
            object body names from :func:`get_task_relevant_objects` plus
            ``"gripper"`` for the end-effector.  Must contain at least one entry.

    Returns:
        3-D centroid (mean) of all positions.
    """
    pts = list(positions.values())
    if not pts:
        raise ValueError("positions dict must contain at least one entry")
    return np.mean(pts, axis=0)


def compute_workspace_center_from_object_poses(
    object_names: list[str],
    object_poses: dict[str, list[float]],
    gripper_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Compute workspace center from serialized object poses (e.g. JSON episode data).

    Convenience wrapper around :func:`compute_workspace_center` for the eval
    path, where positions come from ``EpisodeSpec.scene_modifications.object_poses``
    (each value is ``[x, y, z, qw, qx, qy, qz]``).

    Args:
        object_names: Body names whose positions should contribute (typically
            from :func:`get_task_relevant_objects` or ``EpisodeSpec.task_relevant_objects``).
        object_poses: Mapping of body name to 7-D pose ``[x, y, z, qw, qx, qy, qz]``.
        gripper_pos: Optional gripper position to include.

    Returns:
        3-D centroid.
    """
    positions: dict[str, np.ndarray] = {}
    for name in object_names:
        pose = object_poses.get(name)
        if pose is not None:
            positions[name] = np.asarray(pose[:3], dtype=float)
    if gripper_pos is not None:
        positions["gripper"] = np.asarray(gripper_pos, dtype=float)
    if not positions:
        raise ValueError(
            f"No positions found for any of {object_names} in object_poses "
            f"(available: {list(object_poses.keys())})"
        )
    return compute_workspace_center(positions)
