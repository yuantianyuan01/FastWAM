"""
Level → value scaling for camera and light randomization.

Level is in [0, 100]: 0 = no randomization, 100 = maximum.
Each parameter has an output range (min, max) and a mapping function that
maps level to a value in that range.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.utils.pose import compute_lookat_forward_up

if TYPE_CHECKING:
    from molmo_spaces.configs.camera_configs import FrankaEvalCameraSystem

log = logging.getLogger(__name__)


def _flatten_value(v: Any) -> list[float]:
    """Flatten a scalar or tuple of numbers to a list of floats."""
    if isinstance(v, int | float):
        return [float(v)]
    if isinstance(v, tuple | list):
        out: list[float] = []
        for x in v:
            out.extend(_flatten_value(x))
        return out
    return [float(v)]


def _get_anchors_from_config(
    config: FrankaEvalCameraSystem,
    camera_name: str,
    param_name: str,
) -> tuple[list[float], list[list[float]]]:
    """Return list of levels and list of lists of values from config. First level and value for calibrated state.
    Returns one list for each entry in the range, e.g. 6 lists for (x_min, y_min, z_min, x_max, y_max, z_max).
    """

    ref_levels = [cur[0] for cur in config.ref_level_ranges]
    ref_values = [cur[1] for cur in config.ref_level_ranges]

    values = np.atleast_2d(
        np.array(
            [
                _flatten_value(cur_ref_values[camera_name][param_name])
                for cur_ref_values in ref_values
            ]
        )
    ).transpose()  # transpose: ref_level_index x len(ref_levels)

    return ref_levels, values.tolist()


def _get_camera_param_at_level_from_config(
    config: FrankaEvalCameraSystem,
    camera_name: str,
    param_name: str,
    level: float,
) -> float | tuple[float, ...]:
    """Interpolated value at level from config. Uses camera instance for calibrated v0 when cam is provided."""
    ref_levels, ref_values = _get_anchors_from_config(config, camera_name, param_name)
    values = [piecewise_linear(level, ref_levels, cur_ref_values) for cur_ref_values in ref_values]
    return values[0] if len(values) == 1 else tuple(values)


def _reshape_to_original(flat_val: float | tuple[float, ...], original_val: Any) -> Any:
    """Reshape flat interpolated value to match the nested structure of original_val.

    Handles per-axis params like pos_noise_range=((-0.015,-0.005,-0.02),(0.015,0.005,0.02))
    where interpolation flattens to 6 values but pydantic expects nested tuples.
    """
    if isinstance(flat_val, int | float):
        return flat_val

    if not isinstance(flat_val, tuple) or not isinstance(original_val, tuple):
        return flat_val

    if (
        len(original_val) == 2
        and isinstance(original_val[0], tuple)
        and isinstance(original_val[1], tuple)
    ):
        n = len(original_val[0])
        return tuple(flat_val[:n]), tuple(flat_val[n:])

    return flat_val


def piecewise_linear(level: float, breakpoints: list[float], values: list[float]) -> float:
    """Piecewise linear interpolation."""
    if level <= breakpoints[0]:
        return values[0]
    if level >= breakpoints[-1]:
        return values[-1]

    for bp_low, bp_high, val_low, val_high in zip(breakpoints, breakpoints[1:], values, values[1:]):
        if bp_low <= level <= bp_high:
            span = bp_high - bp_low
            return val_low + (val_high - val_low) * ((level - bp_low) / span)

    return values[-1]  # fallback, should never reach


def apply_camera_randomization_level(
    camera_config: FrankaEvalCameraSystem, level: float
) -> FrankaEvalCameraSystem:
    """Return a copy of the camera system config with randomization params set via interpolation."""
    new_cameras: list[Any] = []
    for cam in camera_config.cameras:
        randomizable = camera_config.ref_level_ranges[0][1][cam.name]
        if not randomizable:
            new_cameras.append(cam)
            continue

        cam_dict = cam.model_dump()
        for param in randomizable:
            cam_dict[param] = _reshape_to_original(
                _get_camera_param_at_level_from_config(camera_config, cam.name, param, level),
                randomizable[param],
            )
        new_cameras.append(cam.__class__.model_validate(cam_dict))
    return camera_config.model_copy(update={"cameras": new_cameras})


def add_eval_camera_args(parser: argparse.ArgumentParser) -> None:
    """Add eval camera CLI flags to an argparse parser.

    These flags are shared across all JSON eval entry points (standalone and distributed).
    """
    group = parser.add_argument_group("Eval camera randomization")
    group.add_argument(
        "--use_eval_cameras",
        action="store_true",
        default=False,
        help="Use FrankaEvalCameraSystem with randomization instead of recorded cameras from JSON.",
    )
    group.add_argument(
        "--camera_rand_level",
        type=float,
        default=0.0,
        help="Camera randomization level (0-100). Only used with --use_eval_cameras.",
    )


def build_eval_camera_config_from_args(
    args: argparse.Namespace,
) -> FrankaEvalCameraSystem | None:
    """Build a FrankaEvalCameraSystem from parsed CLI args, or return None if not requested.

    Returns None if --use_eval_cameras was not passed. Otherwise, creates the eval camera
    system with the requested camera subset and randomization level applied.
    """
    if not args.use_eval_cameras:
        return None

    from molmo_spaces.configs.camera_configs import FrankaEvalCameraSystem

    print(f"Using camera randomization level {args.camera_rand_level}")
    return apply_camera_randomization_level(FrankaEvalCameraSystem(), args.camera_rand_level)


def derive_episode_camera_seed(episode_spec: Any) -> int:
    """Derive a deterministic seed for camera randomization from episode identity.

    The seed is a hash of fields that uniquely identify an episode so that the
    same (episode, level) pair always produces the same camera placement.
    """
    import hashlib

    parts = [
        str(getattr(episode_spec, "scene_dataset", "")),
        str(getattr(episode_spec, "data_split", "")),
        str(getattr(episode_spec, "house_index", 0)),
    ]
    source = getattr(episode_spec, "source", None)
    if source is not None:
        parts.append(str(getattr(source, "h5_file", "")))
        parts.append(str(getattr(source, "traj_key", "")))
    seed_val = getattr(episode_spec, "seed", None)
    if seed_val is not None:
        parts.append(str(seed_val))

    key = "|".join(parts).encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Spherical perturbation for eval exocentric cameras
# ---------------------------------------------------------------------------


def _decompose_to_spherical(
    camera_pos: np.ndarray,
    workspace_center: np.ndarray,
) -> tuple[float, float, float]:
    """Decompose a world-frame camera position into spherical coords relative to workspace center.

    Returns (azimuth, distance, height) where:
    - azimuth: angle in XY plane (radians)
    - distance: horizontal distance in XY plane (meters)
    - height: vertical offset from workspace center (meters)
    """
    delta = camera_pos - workspace_center
    distance = float(np.sqrt(delta[0] ** 2 + delta[1] ** 2))
    azimuth = float(np.arctan2(delta[1], delta[0]))
    height = float(delta[2])
    return azimuth, distance, height


def apply_camera_perturbation(
    cam: EvalExocentricCameraConfig,
    ref_forward: np.ndarray,
    ref_up: np.ndarray,
    workspace_center: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sample a camera pose by perturbing the reference pose in spherical coordinates.

    Orientation is computed via slerp between the calibrated rotation (from the
    shoulder-mount quaternion) and a lookat-at-workspace-center rotation, controlled
    by ``workspace_center_weight``.  At weight 0 the camera keeps its original
    orientation; at weight 1 it looks straight at the workspace center (plus
    optional noise).

    Args:
        cam: Resolved EvalExocentricCameraConfig (pos/forward/up already set).
        ref_forward: Forward vector from the resolved quaternion-based reference pose.
        ref_up: Up vector from the resolved quaternion-based reference pose.
        workspace_center: 3D point the camera should look at.
        rng: Seeded random state for deterministic sampling.

    Returns:
        (pos, forward, up, fov) in world frame.
    """
    ref_pos = np.array(cam.pos, dtype=np.float64)
    ref_azimuth, ref_distance, ref_height = _decompose_to_spherical(ref_pos, workspace_center)

    azimuth_shift = 0
    if cam.azimuth_range is not None:
        azimuth_shift = rng.uniform(cam.azimuth_range[0], cam.azimuth_range[1])
    azimuth = ref_azimuth + azimuth_shift

    distance_shift = 0
    if cam.distance_range is not None:
        distance_shift = rng.uniform(cam.distance_range[0], cam.distance_range[1])
    distance = max(ref_distance + distance_shift, 0.10)

    height_shift = 0
    if cam.height_range is not None:
        height_shift = rng.uniform(cam.height_range[0], cam.height_range[1])
    height = ref_height + height_shift

    # Reconstruct Cartesian position
    pos = workspace_center.copy().astype(np.float64)
    pos[0] += distance * np.cos(azimuth)
    pos[1] += distance * np.sin(azimuth)
    pos[2] += height

    # Orientation: slerp between calibrated (ref) and lookat-at-workspace-center
    lookat_weight = 0.0
    if cam.workspace_center_weight is not None:
        lookat_weight = cam.workspace_center_weight

    # Build calibrated rotation from ref_forward / ref_up
    ref_fwd = np.array(ref_forward, dtype=np.float64)
    ref_fwd /= np.linalg.norm(ref_fwd) + 1e-12
    ref_u = np.array(ref_up, dtype=np.float64)
    ref_u /= np.linalg.norm(ref_u) + 1e-12
    ref_right = np.cross(ref_fwd, ref_u)
    ref_right /= np.linalg.norm(ref_right) + 1e-12
    # Re-orthogonalise up
    ref_u = np.cross(ref_right, ref_fwd)
    rot_calibrated = R.from_matrix(np.column_stack([ref_right, ref_u, -ref_fwd]))

    # Build lookat rotation toward workspace center (+ optional noise)
    lookat_target = workspace_center.copy().astype(np.float64)
    if cam.lookat_noise_range is not None:
        lookat_target += rng.uniform(cam.lookat_noise_range[0], cam.lookat_noise_range[1], size=3)

    la_fwd, la_up = compute_lookat_forward_up(pos, lookat_target)
    la_right = np.cross(la_fwd, la_up)
    rot_lookat = R.from_matrix(np.column_stack([la_right, la_up, -la_fwd]))

    # Slerp between calibrated and lookat orientations
    slerp = Slerp([0.0, 1.0], R.concatenate([rot_calibrated, rot_lookat]))
    rot_interp = slerp(lookat_weight)
    mat = rot_interp.as_matrix()
    forward = (-mat[:, 2]).astype(np.float32)
    up = mat[:, 1].astype(np.float32)

    # FOV
    fov_shift = 0
    fov = cam.fov
    if cam.fov_range is not None:
        fov_shift = rng.uniform(cam.fov_range[0], cam.fov_range[1]) - fov
    fov += fov_shift

    log.info(
        "[EVAL CAMERA] Perturbation info:\n"
        f"  azimuth: {azimuth_shift} {cam.azimuth_range}\n"
        f"  distance: {distance_shift} {cam.distance_range}\n"
        f"  height: {height_shift} {cam.height_range}\n"
        f"  lookat_weight: {lookat_weight} noise_range={cam.lookat_noise_range}\n"
        f"  fov: {fov_shift} {cam.fov_range}"
    )

    return pos.astype(np.float32), forward, up, float(fov)


def _check_camera_visibility(
    env: Any,
    camera_name: str,
    target_bodies: list[str],
    threshold: float = 0.0001,
) -> bool:
    """Return True if camera *camera_name* sees all *target_bodies* above *threshold*."""
    if not target_bodies:
        return True
    try:
        results = env.check_visibility(camera_name, *target_bodies)
        if isinstance(results, dict):
            return all(results.get(b, 0.0) >= threshold for b in target_bodies)
        return results >= threshold
    except Exception:
        return False


def resolve_reference_pose(cam_config, env) -> EvalExocentricCameraConfig:
    """Compute world-frame pos/forward/up from the reference body and return an updated copy."""
    pos, forward, up = env.camera_manager.create_quaternion_camera_pose(
        env,
        reference_body_name=cam_config.reference_body_names[0],
        camera_offset=np.array(cam_config.camera_offset, dtype=np.float32),
        camera_quaternion=np.array(cam_config.camera_quaternion, dtype=np.float32),
    )
    return cam_config.model_copy(
        update={
            "pos": pos.tolist(),
            "forward": forward.tolist(),
            "up": up.tolist(),
        }
    )


def _apply_mjcf_camera_noise(
    env: CPUMujocoEnv,
    cam_config: MjcfCameraConfig,
    rng: np.random.RandomState,
) -> None:
    """Apply position, orientation, and FOV noise to an already-registered MJCF camera.

    Delegates to ``CameraManager.apply_mjcf_camera_noise`` (single source of
    truth) but passes the episode-seeded *rng* for deterministic repeatability.
    """
    from molmo_spaces.env.camera_manager import CameraManager, RobotMountedCamera

    camera_manager = env.camera_manager
    registered = camera_manager.registry.cameras.get(cam_config.name)
    if registered is None or not isinstance(registered, RobotMountedCamera):
        return

    pos, quat, fov = CameraManager.apply_mjcf_camera_noise(
        registered.camera_offset.copy(),
        registered.camera_quaternion.copy(),
        registered.fov,
        cam_config,
        rng=rng,
    )
    registered.camera_offset = pos.astype(np.float32)
    registered.camera_quaternion = quat.astype(np.float32)
    registered.fov = fov
    registered.update_pose(env)


def setup_eval_cameras(
    env: CPUMujocoEnv,
    eval_system: FrankaEvalCameraSystem,
    task_relevant_bodies: list[str],
    workspace_center: np.ndarray,
    rng_seed: int,
) -> None:
    """Set up eval cameras: wrist via MJCF, exo via spherical perturbation.

    For each camera in *eval_system*:

    - **Wrist** (``MjcfCameraConfig``): placed directly, no visibility check.
    - **Exo** (``EvalExocentricCameraConfig``): reference pose is resolved
      from the shoulder mount, then ``apply_camera_perturbation`` samples
      a pose in spherical coords around the workspace center.  Multiple
      attempts are made to satisfy visibility constraints; if all fail a
      ``CameraPlacementError`` is raised.

    Args:
        env: CPUMujocoEnv with the scene already set up.
        eval_system: FrankaEvalCameraSystem with randomization ranges
            already interpolated for the desired level.
        task_relevant_bodies: Body names to check visibility against.
        workspace_center: 3D centroid of task-relevant objects.
        rng_seed: Deterministic seed for repeatable placement.

    Raises:
        CameraPlacementError: If an exo camera cannot be placed with
            visibility constraints after ``max_placement_attempts``.
    """
    from molmo_spaces.configs.camera_configs import (
        EvalExocentricCameraConfig,
        MjcfCameraConfig,
    )
    from molmo_spaces.tasks.task_sampler_errors import CameraPlacementError

    camera_manager = env.camera_manager
    rng = np.random.RandomState(rng_seed)

    for cam in eval_system.cameras:
        # --- Wrist cameras: register clean, then apply noise with seeded rng ---
        if isinstance(cam, MjcfCameraConfig):
            clean_cam = cam.model_copy(
                update={
                    "pos_noise_range": None,
                    "orientation_noise_degrees": None,
                    "fov_noise_degrees": None,
                }
            )
            camera_manager._setup_mjcf_camera(env, clean_cam)
            _apply_mjcf_camera_noise(env, cam, rng)
            log.info(f"[EVAL CAMERA] '{cam.name}' placed (wrist)")
            continue

        # --- Exo cameras: spherical perturbation ---
        if not isinstance(cam, EvalExocentricCameraConfig):
            log.warning(f"[EVAL CAMERA] Unknown camera type for '{cam.name}': {type(cam).__name__}")
            continue

        resolved = resolve_reference_pose(cam, env)
        ref_forward = np.array(resolved.forward, dtype=np.float32)
        ref_up = np.array(resolved.up, dtype=np.float32)
        log.info(
            f"[EVAL CAMERA] '{cam.name}' reference pose: pos={np.round(resolved.pos, 3).tolist()}"
        )

        max_attempts = resolved.max_placement_attempts
        has_visibility = bool(task_relevant_bodies)

        for attempt in range(max_attempts):
            pos, forward, up, fov = apply_camera_perturbation(
                resolved,
                ref_forward,
                ref_up,
                workspace_center,
                rng,
            )
            camera_manager.add_camera(cam.name, pos, forward, up, fov)

            if not has_visibility or _check_camera_visibility(env, cam.name, task_relevant_bodies):
                log.info(
                    f"[EVAL CAMERA] '{cam.name}' placed (attempt {attempt + 1}/{max_attempts})"
                )
                break

            # Remove failed attempt before retrying
            if cam.name in camera_manager.registry.cameras:
                del camera_manager.registry.cameras[cam.name]
        else:
            raise CameraPlacementError(
                f"Failed to place eval camera '{cam.name}' with visibility of "
                f"{task_relevant_bodies} after {max_attempts} attempts"
            )


if __name__ == "__main__":

    def debug():
        import json

        from molmo_spaces.configs.camera_configs import FrankaEvalCameraSystem

        def print_randomizable_values_at_levels(levels=(0, 10, 25, 50, 75, 90, 100)) -> None:
            config = FrankaEvalCameraSystem()
            for cam in config.cameras:
                params = config.ref_level_ranges[0][1].get(cam.name, {})
                if not params:
                    continue
                for param in sorted(params.keys()):
                    print(f"{cam.name} {param}:")
                    for level in levels:
                        val = _get_camera_param_at_level_from_config(config, cam.name, param, level)
                        print(f"level {level:3.0f}: {val}")

        # print_randomizable_values_at_levels()

        # new_config = apply_camera_randomization_level(FrankaEvalCameraSystem(), level=33)
        # print(json.dumps(new_config.model_dump(), indent=2))

        def test_camera_from_args():
            from argparse import ArgumentParser

            parser = ArgumentParser()
            add_eval_camera_args(parser)
            args = parser.parse_args()
            args.use_eval_cameras = True
            args.camera_rand_level = 33
            args.no_gopro = True
            args.num_zeds = 1
            args.disable_shoulder = True

            num_cameras = 5
            if args.no_gopro:
                num_cameras -= 1
            if args.num_zeds < 2:
                num_cameras -= 2 - args.num_zeds
            if args.disable_shoulder:
                num_cameras -= 1

            new_config = build_eval_camera_config_from_args(args)
            assert len(new_config.cameras) == num_cameras

            print(json.dumps(new_config.model_dump(), indent=2))

        test_camera_from_args()

    debug()
    print("DONE")
