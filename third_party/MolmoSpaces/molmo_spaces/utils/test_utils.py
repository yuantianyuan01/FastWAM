"""
Shared utilities for data generation tests (Franka, RUM, etc.).
"""

import json
import logging
from pathlib import Path

import decord
import h5py
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

log = logging.getLogger(__name__)


def print_profiling_summary(profiler):
    """
    Print a formatted summary of profiling results.

    Args:
        profiler: Profiler instance with collected timing data

    Returns:
        str: Formatted summary string
    """
    if profiler is None:
        return "No profiler available"

    # Get all profiled operations
    operations = list(profiler._avg_time.keys())
    if not operations:
        return "No profiling data collected"

    # Sort by total time (descending)
    operations.sort(key=lambda k: profiler.get_avg_time(k) * profiler.get_n(k), reverse=True)

    # Build summary
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("PROFILING SUMMARY".center(80))
    lines.append("=" * 80)
    lines.append(f"{'Operation':<40} {'Calls':>8} {'Avg Time':>12} {'Total Time':>12}")
    lines.append("-" * 80)

    for op in operations:
        avg_time = profiler.get_avg_time(op)
        n_calls = profiler.get_n(op)
        total_time = avg_time * n_calls
        lines.append(f"{op:<40} {n_calls:>8} {avg_time:>11.4f}s {total_time:>11.4f}s")

    lines.append("=" * 80)

    return "\n".join(lines)


def run_policy_for_steps(task, policy, num_steps=10, profiler=None):
    """
    Run a policy on a task for a fixed number of steps, following pipeline.py API.

    Args:
        task: The task instance
        policy: The policy instance
        num_steps: Number of steps to run
        profiler: Optional profiler instance to track timing

    Returns:
        tuple: (initial_qpos, final_qpos) as numpy arrays
    """
    if profiler is not None:
        profiler.start("test_policy_execution")

    # Register policy with task (following pipeline.py line 322)
    task.register_policy(policy)

    # Reset task to get initial observation (following pipeline.py line 148)
    if profiler is not None:
        profiler.start("test_task_reset")
    observation, _info = task.reset()
    if profiler is not None:
        profiler.end("test_task_reset")

    # Get initial joint positions AFTER reset (this is the state from which the policy runs)
    robot = task.env.robots[0]
    robot_view = robot.robot_view

    # Get all move groups in consistent order and concatenate their qpos
    move_group_ids = robot_view.move_group_ids()
    initial_qpos_dict = robot_view.get_qpos_dict(move_group_ids)
    initial_qpos = np.concatenate([initial_qpos_dict[mg_id] for mg_id in move_group_ids])

    # Run policy for specified number of steps
    for _ in range(num_steps):
        # Get action from policy (following pipeline.py line 163)
        if profiler is not None:
            profiler.start("test_policy_get_action")
        action_cmd = policy.get_action(observation)
        if profiler is not None:
            profiler.end("test_policy_get_action")

        # Step the task
        if profiler is not None:
            profiler.start("test_task_step")
        observation, reward, terminal, truncated, infos = task.step(action_cmd)
        if profiler is not None:
            profiler.end("test_task_step")

        # Check if done (following pipeline.py line 159)
        if task.is_done():
            break

    # Get final joint positions using same move group order
    final_qpos_dict = robot_view.get_qpos_dict(move_group_ids)
    final_qpos = np.concatenate([final_qpos_dict[mg_id] for mg_id in move_group_ids])

    if profiler is not None:
        profiler.end("test_policy_execution")

    return initial_qpos, final_qpos


def run_task_for_steps_with_observations(task, policy, num_steps=10, profiler=None):
    """
    Run a policy on a task for a fixed number of steps and return both qpos and observations.

    This extends run_policy_for_steps by also capturing initial and final observations after running steps.
    Useful for testing that observations remain deterministic across runs and that they change appropriately.

    Args:
        task: The task instance
        policy: The policy instance
        num_steps: Number of steps to run
        profiler: Optional profiler instance to track timing

    Returns:
        tuple: (initial_qpos, final_qpos, initial_obs_dict, final_obs_dict) where:
            - initial_qpos: numpy array of initial joint positions
            - final_qpos: numpy array of final joint positions
            - initial_obs_dict: dictionary of initial observations from the single environment
            - final_obs_dict: dictionary of final observations from the single environment after running steps
    """
    if profiler is not None:
        profiler.start("test_policy_execution_with_obs")

    # Register policy with task
    task.register_policy(policy)

    # Reset task to get initial observation
    if profiler is not None:
        profiler.start("test_task_reset")
    observation, _info = task.reset()
    if profiler is not None:
        profiler.end("test_task_reset")

    # Get initial joint positions AFTER reset (this is the state from which the policy runs)
    robot = task.env.robots[0]
    robot_view = robot.robot_view

    # Get all move groups in consistent order and concatenate their qpos
    move_group_ids = robot_view.move_group_ids()
    initial_qpos_dict = robot_view.get_qpos_dict(move_group_ids)
    initial_qpos = np.concatenate([initial_qpos_dict[mg_id] for mg_id in move_group_ids])

    # Extract initial observations
    # observation is list[dict[str, Any]] from task.reset() - a list of obs dicts, one per env
    # Get the first (and only) environment's observations
    initial_obs_dict = observation[0]

    # Run policy for specified number of steps
    for _ in range(num_steps):
        # Get action from policy
        if profiler is not None:
            profiler.start("test_policy_get_action")
        action_cmd = policy.get_action(observation)
        if profiler is not None:
            profiler.end("test_policy_get_action")

        # Step the task
        if profiler is not None:
            profiler.start("test_task_step")
        observation, reward, terminal, truncated, infos = task.step(action_cmd)
        if profiler is not None:
            profiler.end("test_task_step")

        # Check if done
        if task.is_done():
            break

    # Get final joint positions using same move group order
    final_qpos_dict = robot_view.get_qpos_dict(move_group_ids)
    final_qpos = np.concatenate([final_qpos_dict[mg_id] for mg_id in move_group_ids])

    # Extract final observations
    # observation is list[dict[str, Any]] from task.step() - a list of obs dicts, one per env
    # Get the first (and only) environment's observations
    final_obs_dict = observation[0]

    if profiler is not None:
        profiler.end("test_policy_execution_with_obs")

    return initial_qpos, final_qpos, initial_obs_dict, final_obs_dict


def save_visual_observations(obs_dict, output_dir, prefix="obs"):
    """
    Save visual observations as viewable PNG images for debugging.

    Args:
        obs_dict: Dictionary of observations from a single environment
        output_dir: Path object or string for the debug output directory
        prefix: Prefix for the saved image filenames (e.g., "obs", "expected", "diff")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sensor_name, sensor_data in obs_dict.items():
        # Check if this is a visual observation (camera sensor)
        if "camera" in sensor_name and "sensor_param" not in sensor_name:
            # Ensure the data is in the right format (H, W, 3) with values in [0, 255]
            if (
                isinstance(sensor_data, np.ndarray)
                and len(sensor_data.shape) == 3
                and sensor_data.shape[2] == 3
            ):
                # Convert to uint8 if needed
                if sensor_data.dtype != np.uint8:
                    # Clip to valid range and convert
                    sensor_data = np.clip(sensor_data, 0, 255).astype(np.uint8)

                # Create PIL Image and save
                img = Image.fromarray(sensor_data)
                img_path = output_dir / f"{prefix}_{sensor_name}.png"
                img.save(img_path)


def assert_observations_match(
    actual_obs, expected_obs, sensor_name, atol=0, rtol=1e-7, ssim_threshold=0.9
):
    """
    Compare actual and expected observations using Structural Similarity Index (SSIM).

    Uses SSIM to compare images perceptually rather than pixel-by-pixel, which is more
    robust to minor rendering variations while still catching meaningful visual differences.

    Args:
        actual_obs: Actual observation array
        expected_obs: Expected observation array
        sensor_name: Name of the sensor (for error messages)
        atol: Unused, kept for API compatibility
        rtol: Unused, kept for API compatibility

    Raises:
        AssertionError: If observations have meaningful visual differences (low SSIM)
    """
    # Compute SSIM for each channel and average
    # SSIM returns a value between -1 and 1, where 1 is perfect similarity
    # Use channel_axis=2 to compute SSIM per color channel and average
    ssim_score = ssim(
        actual_obs,
        expected_obs,
        channel_axis=2,
        data_range=255,  # For uint8 images
    )

    # Also compute basic pixel difference stats for debugging
    diff = np.abs(actual_obs.astype(np.int32) - expected_obs.astype(np.int32))
    diff_max = np.max(diff)
    diff_mean = np.mean(diff)
    num_different_pixels = np.sum(diff > 0)
    total_pixels = diff.size
    percent_different = 100 * num_different_pixels / total_pixels

    # Define SSIM threshold
    # SSIM > 0.90 is considered very similar
    # SSIM > 0.95 is nearly identical (which these images should be since they are generated from the same policy)
    # high GPU variance is killing me
    MIN_SSIM_THRESHOLD = ssim_threshold

    if ssim_score < MIN_SSIM_THRESHOLD:
        # Print detailed statistics
        diff_sum = np.sum(diff)
        print(f"\n[DIFF] Sensor {sensor_name}:")
        print(f"  SSIM score: {ssim_score:.4f} (threshold: {MIN_SSIM_THRESHOLD})")
        print(f"  Total difference sum: {diff_sum}")
        print(f"  Max pixel difference: {diff_max}")
        print(f"  Mean pixel difference: {diff_mean:.4f}")
        print(
            f"  Different pixels: {num_different_pixels}/{total_pixels} ({percent_different:.2f}%)"
        )

        raise AssertionError(
            f"Sensor {sensor_name} observations have meaningful structural differences from saved test data. "
            f"SSIM score: {ssim_score:.4f} (threshold: {MIN_SSIM_THRESHOLD})"
        )


def verify_and_compare_camera_observations(
    obs,
    sensor_suite,
    test_data_dir,
    test_data_prefix,
    expected_cameras,
    debug_images_dir=None,
    debug_prefix="obs",
    expected_shape=(480, 480, 3),
    atol=1.0,
    rtol=0.0,
    ignore_cameras=None,
    skip_depth_exact_match=True,
    ssim_threshold=0.9,
):
    """
    Verify observation structure and compare camera observations against saved test data.

    This is a comprehensive helper for testing task observations that:
    1. Verifies the observation structure (vectorized format)
    2. Extracts camera sensors and checks their shapes
    3. Compares them against saved test data
    4. Optionally saves debug images for visual inspection
    5. Verifies all expected cameras are present

    Args:
        obs: Observation tuple from task.reset() or task.step()
        sensor_suite: The task's sensor suite
        test_data_dir: Path to directory containing test data files
        test_data_prefix: Prefix for test data files (e.g., "rum_pick_obs_")
        expected_cameras: List of expected camera sensor names
        debug_images_dir: Optional path to save debug images. If None, no images are saved.
        debug_prefix: Prefix for debug image filenames (default: "obs")
        expected_shape: Expected shape of camera observations (default: (480, 480, 3))
        atol: Absolute tolerance for pixel value comparison (default: 1.0).
              For uint8 images [0-255], 1.0 allows single-pixel differences due to
              floating point precision or slight numerical variations.
        rtol: Relative tolerance for comparison (default: 0.0)
        ignore_cameras: Optional list of camera sensor names to skip during comparison
        skip_depth_exact_match: Whether to skip pixel-exact depth comparison (default: True).
              When True, uses structural similarity (SSIM) on normalized depth for cross-platform
              robustness. When False, does pixel-exact comparison with edge masking (for local
              determinism tests). Depth rendering is NOT deterministic across platforms/GPUs.

    Returns:
        tuple: (obs_dict, camera_sensors_found) for further testing if needed
    """
    test_data_dir = Path(test_data_dir)
    ignore_cameras = ignore_cameras or []

    # Verify observations structure (vectorized format)
    assert obs is not None, "Observations should not be None"
    assert isinstance(obs, tuple) and len(obs) == 2, (
        "Observations should be a tuple of (obs_list, info_dict)"
    )
    obs_list, info_dict = obs
    assert isinstance(obs_list, list), "First element should be list of observation dictionaries"
    assert len(obs_list) == 1, f"Expected 1 environment observation, got {len(obs_list)}"

    # Extract the single environment's observations
    obs_dict = obs_list[0]
    assert isinstance(obs_dict, dict), "Observation should be a dictionary"
    assert len(obs_dict) == len(sensor_suite.sensors), (
        f"Expected {len(sensor_suite.sensors)} sensors, got {len(obs_dict)}"
    )

    # Track which camera sensors we find and whether any assertions failed
    camera_sensors_found = []
    assertion_failed = False

    try:
        for sensor_name in sensor_suite.sensors:
            assert sensor_name in obs_dict, f"Sensor {sensor_name} not found in observations"

            if "camera" in sensor_name and "sensor_param" not in sensor_name:
                camera_sensors_found.append(sensor_name)

                # Skip comparison for ignored cameras
                if sensor_name in ignore_cameras:
                    print(f"[SKIP] Ignoring camera {sensor_name} as requested")
                    continue

                sensor_obs = obs_dict[sensor_name]

                # Verify basic properties
                assert sensor_obs is not None, f"Observation for {sensor_name} is None"

                # Handle depth sensors separately (2D) vs RGB sensors (3D)
                if sensor_name.endswith("_depth"):
                    # Depth sensors are 2D (H, W)
                    assert sensor_obs.ndim == 2, (
                        f"Depth sensor {sensor_name} should be 2D (H, W), got {sensor_obs.ndim}D"
                    )
                    # Verify depth shape (expected_shape is (W, H, C), extract W, H for depth)
                    sensor_shape_swapped = (sensor_obs.shape[1], sensor_obs.shape[0])
                    expected_depth_shape = (expected_shape[0], expected_shape[1])
                    assert sensor_shape_swapped == expected_depth_shape, (
                        f"Expected depth shape {expected_depth_shape}, got {sensor_obs.shape} (h/w swapped)"
                    )
                else:
                    # RGB sensors are 3D (H, W, C)
                    assert sensor_obs.ndim == 3, (
                        f"RGB sensor {sensor_name} should be 3D (H, W, C), got {sensor_obs.ndim}D"
                    )
                    sensor_shape_swapped = (
                        sensor_obs.shape[1],
                        sensor_obs.shape[0],
                        sensor_obs.shape[2],
                    )
                    assert sensor_shape_swapped == expected_shape, (
                        f"Expected shape {expected_shape}, got {sensor_obs.shape} (h/w swapped)"
                    )

                # Load and compare against saved test data for regression testing
                test_data_path = test_data_dir / f"{test_data_prefix}{sensor_name}.npy"
                expected_obs = np.load(test_data_path)

                # Handle depth sensor comparison differently
                if sensor_name.endswith("_depth"):
                    # Convert float16 to float32 if needed (depth saved as float16 to reduce file size)
                    if expected_obs.dtype == np.float16:
                        expected_obs = expected_obs.astype(np.float32)

                    # Skip pixel-exact comparison if requested (for cross-platform CI)
                    # Depth rendering is NOT deterministic across platforms/GPUs/drivers
                    if skip_depth_exact_match:
                        print(
                            f"[DEPTH] Using structural similarity (SSIM) for {sensor_name} "
                            f"(skip_depth_exact_match=True). Checks structure, not exact pixels."
                        )

                        # Basic sanity checks first
                        assert not np.any(np.isnan(sensor_obs)), (
                            f"{sensor_name} contains NaN values"
                        )
                        assert not np.any(np.isinf(sensor_obs)), (
                            f"{sensor_name} contains inf values"
                        )
                        assert np.any(sensor_obs > 0), f"{sensor_name} is all zeros"

                        # Normalize depth to [0, 255] range for SSIM comparison
                        # Use a reasonable depth range (e.g., 0-2m for wrist camera)
                        depth_min_for_viz = 0.0
                        depth_max_for_viz = 2.0

                        def normalize_depth_for_comparison(
                            depth, depth_min=depth_min_for_viz, depth_max=depth_max_for_viz
                        ):
                            """Normalize depth to uint8 [0, 255] for SSIM."""
                            depth_clipped = np.clip(depth, depth_min, depth_max)
                            depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min)
                            return (depth_normalized * 255).astype(np.uint8)

                        sensor_obs_normalized = normalize_depth_for_comparison(sensor_obs)
                        expected_obs_normalized = normalize_depth_for_comparison(expected_obs)

                        # Use SSIM to compare depth structure (same as RGB comparison)
                        # SSIM is robust to small numerical differences while catching major changes
                        ssim_score = ssim(
                            sensor_obs_normalized,
                            expected_obs_normalized,
                            data_range=255,
                        )

                        # Depth should have high structural similarity, same as RGB
                        # SSIM is designed to be robust to small numerical differences
                        MIN_DEPTH_SSIM_THRESHOLD = ssim_threshold

                        if ssim_score < MIN_DEPTH_SSIM_THRESHOLD:
                            # Calculate basic pixel stats for debugging
                            diff = np.abs(sensor_obs - expected_obs)
                            diff_max = np.max(diff)
                            diff_mean = np.mean(diff)
                            num_different = np.sum(diff > 0.01)  # >10mm difference
                            total_pixels = diff.size
                            percent_different = 100 * num_different / total_pixels

                            raise AssertionError(
                                f"Depth sensor {sensor_name} has low structural similarity to saved test data. "
                                f"SSIM score: {ssim_score:.4f} (threshold: {MIN_DEPTH_SSIM_THRESHOLD}). "
                                f"This suggests major rendering differences across platforms. "
                                f"Pixel-level stats: max diff={diff_max * 1000:.1f}mm, "
                                f"mean diff={diff_mean * 1000:.1f}mm, "
                                f"{percent_different:.1f}% pixels differ >10mm"
                            )

                        print(
                            f"  ✓ Depth SSIM: {ssim_score:.4f} (threshold: {MIN_DEPTH_SSIM_THRESHOLD})"
                        )
                        continue

                    # If we get here, we're doing exact comparison (for local determinism tests)
                    from molmo_spaces.utils.depth_utils import detect_depth_edges

                    # Detect edges in expected depth (where large errors are expected from small motion)
                    # Use a moderate threshold (50mm gradient) to catch occlusion boundaries
                    edge_mask = detect_depth_edges(expected_obs, gradient_threshold_mm=50.0)

                    # Create mask for smooth (non-edge) regions
                    smooth_mask = ~edge_mask

                    # Calculate differences
                    diff = np.abs(sensor_obs - expected_obs)

                    # Compare smooth regions with tight tolerance (5mm)
                    # Edge regions may have large differences due to slight motion/alignment
                    if np.sum(smooth_mask) > 0:
                        smooth_diff = diff[smooth_mask]
                        max_smooth_diff = np.max(smooth_diff)
                        mean_smooth_diff = np.mean(smooth_diff)

                        # Check smooth regions only (edges can differ due to motion)
                        if max_smooth_diff > 0.005:
                            # Calculate stats for error reporting
                            max_diff_overall = np.max(diff)
                            mean_diff_overall = np.mean(diff)
                            edge_pixels = np.sum(edge_mask)
                            smooth_pixels = np.sum(smooth_mask)

                            raise AssertionError(
                                f"Depth sensor {sensor_name} differs from saved test data. "
                                f"Smooth regions (non-edges): Max diff: {max_smooth_diff * 1000:.3f}mm, "
                                f"Mean diff: {mean_smooth_diff * 1000:.3f}mm "
                                f"({smooth_pixels:,} pixels). "
                                f"Overall: Max diff: {max_diff_overall * 1000:.3f}mm, "
                                f"Mean diff: {mean_diff_overall * 1000:.3f}mm "
                                f"({edge_pixels:,} edge pixels masked out)"
                            )
                    else:
                        raise AssertionError(
                            f"Depth sensor {sensor_name}: All pixels are edges, cannot compare"
                        )
                else:
                    # Compare RGB observations using SSIM
                    assert_observations_match(
                        sensor_obs,
                        expected_obs,
                        sensor_name,
                        atol=atol,
                        rtol=rtol,
                        ssim_threshold=ssim_threshold,
                    )

        # Verify we found all expected camera sensors
        for expected_camera in expected_cameras:
            assert expected_camera in camera_sensors_found, (
                f"Expected camera {expected_camera} not found in observations"
            )

    except AssertionError:
        assertion_failed = True
        raise
    finally:
        # Always save debug images for visual inspection (if requested)
        if debug_images_dir is not None and camera_sensors_found:
            expected_dict = {}
            for sensor_name in camera_sensors_found:
                # Skip loading ignored cameras for debug images too
                if sensor_name in ignore_cameras:
                    continue
                test_data_path = test_data_dir / f"{test_data_prefix}{sensor_name}.npy"
                expected_dict[sensor_name] = np.load(test_data_path)

            # Add FAILED_ prefix if test failed
            final_prefix = f"FAILED_{debug_prefix}" if assertion_failed else debug_prefix
            save_observation_comparison(
                obs_dict, expected_dict, debug_images_dir, prefix=final_prefix
            )

    return obs_dict, camera_sensors_found


def verify_and_compare_camera_observations_after_steps(
    obs_dict,
    sensor_suite,
    test_data_dir,
    test_data_prefix,
    expected_cameras,
    initial_obs_dict=None,
    debug_images_dir=None,
    debug_prefix="obs_after_steps",
    expected_shape=(480, 480, 3),
    atol=1.0,
    rtol=0.0,
    ignore_cameras=None,
    skip_depth_exact_match=True,
    ssim_threshold=0.9,
):
    """
    Verify and compare camera observations after running policy steps against saved test data.

    Similar to verify_and_compare_camera_observations, but expects obs_dict directly
    rather than the tuple format from task.reset()/task.step().

    Args:
        obs_dict: Dictionary of observations from a single environment
        sensor_suite: The task's sensor suite
        test_data_dir: Path to directory containing test data files
        test_data_prefix: Prefix for test data files (e.g., "rum_pick_after_steps_")
        expected_cameras: List of expected camera sensor names
        initial_obs_dict: Optional dict of initial observations to verify that observations changed
        debug_images_dir: Optional path to save debug images. If None, no images are saved.
        debug_prefix: Prefix for debug image filenames (default: "obs_after_steps")
        expected_shape: Expected shape of camera observations (w,h,c) (default: (480, 480, 3))
        atol: Absolute tolerance for pixel value comparison (default: 1.0)
        rtol: Relative tolerance for comparison (default: 0.0)
        ignore_cameras: Optional list of camera sensor names to skip during comparison
        skip_depth_exact_match: Whether to skip pixel-exact depth comparison (default: True).
              When True, uses structural similarity (SSIM) on normalized depth for cross-platform
              robustness. When False, does pixel-exact comparison with edge masking (for local
              determinism tests). Depth rendering is NOT deterministic across platforms/GPUs.

    Returns:
        list: camera_sensors_found for further testing if needed
    """
    test_data_dir = Path(test_data_dir)
    ignore_cameras = ignore_cameras or []

    # Verify observations structure
    assert isinstance(obs_dict, dict), "Observation should be a dictionary"
    assert len(obs_dict) == len(sensor_suite.sensors), (
        f"Expected {len(sensor_suite.sensors)} sensors, got {len(obs_dict)}"
    )

    # Track which camera sensors we find and whether any assertions failed
    camera_sensors_found = []
    assertion_failed = False

    try:
        for sensor_name in sensor_suite.sensors:
            assert sensor_name in obs_dict, f"Sensor {sensor_name} not found in observations"

            if "camera" in sensor_name and "sensor_param" not in sensor_name:
                camera_sensors_found.append(sensor_name)

                # Skip comparison for ignored cameras
                if sensor_name in ignore_cameras:
                    print(f"[SKIP] Ignoring camera {sensor_name} as requested")
                    continue

                sensor_obs = obs_dict[sensor_name]

                # Handle depth sensors separately (2D) vs RGB sensors (3D)
                if sensor_name.endswith("_depth"):
                    # Depth sensors are 2D (H, W)
                    assert sensor_obs.ndim == 2, (
                        f"Depth sensor {sensor_name} should be 2D (H, W), got {sensor_obs.ndim}D"
                    )
                    # Verify depth shape (expected_shape is (W, H, C), extract W, H for depth)
                    sensor_shape_swapped = (sensor_obs.shape[1], sensor_obs.shape[0])
                    expected_depth_shape = (expected_shape[0], expected_shape[1])
                    assert sensor_shape_swapped == expected_depth_shape, (
                        f"Expected depth shape {expected_depth_shape}, got {sensor_obs.shape} (h/w swapped)"
                    )
                else:
                    # RGB sensors are 3D (H, W, C)
                    assert sensor_obs.ndim == 3, (
                        f"RGB sensor {sensor_name} should be 3D (H, W, C), got {sensor_obs.ndim}D"
                    )
                    sensor_shape_swapped = (
                        sensor_obs.shape[1],
                        sensor_obs.shape[0],
                        sensor_obs.shape[2],
                    )
                    assert sensor_shape_swapped == expected_shape, (
                        f"Expected shape {expected_shape}, got {sensor_obs.shape} (h/w swapped)"
                    )

                # Load and compare against saved test data for regression testing
                test_data_path = test_data_dir / f"{test_data_prefix}{sensor_name}.npy"
                expected_obs = np.load(test_data_path)

                # Handle depth sensor comparison differently
                if sensor_name.endswith("_depth"):
                    # Convert float16 to float32 if needed (depth saved as float16 to reduce file size)
                    if expected_obs.dtype == np.float16:
                        expected_obs = expected_obs.astype(np.float32)

                    # Skip pixel-exact comparison if requested (for cross-platform CI)
                    # Depth rendering is NOT deterministic across platforms/GPUs/drivers
                    if skip_depth_exact_match:
                        print(
                            f"[DEPTH] Using structural similarity (SSIM) for {sensor_name} "
                            f"(skip_depth_exact_match=True). Checks structure, not exact pixels."
                        )

                        # Basic sanity checks first
                        assert not np.any(np.isnan(sensor_obs)), (
                            f"{sensor_name} contains NaN values"
                        )
                        assert not np.any(np.isinf(sensor_obs)), (
                            f"{sensor_name} contains inf values"
                        )
                        assert np.any(sensor_obs > 0), f"{sensor_name} is all zeros"

                        # Normalize depth to [0, 255] range for SSIM comparison
                        # Use a reasonable depth range (e.g., 0-2m for wrist camera)
                        depth_min_for_viz = 0.0
                        depth_max_for_viz = 2.0

                        def normalize_depth_for_comparison(
                            depth, depth_min=depth_min_for_viz, depth_max=depth_max_for_viz
                        ):
                            """Normalize depth to uint8 [0, 255] for SSIM."""
                            depth_clipped = np.clip(depth, depth_min, depth_max)
                            depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min)
                            return (depth_normalized * 255).astype(np.uint8)

                        sensor_obs_normalized = normalize_depth_for_comparison(sensor_obs)
                        expected_obs_normalized = normalize_depth_for_comparison(expected_obs)

                        # Use SSIM to compare depth structure (same as RGB comparison)
                        # SSIM is robust to small numerical differences while catching major changes
                        ssim_score = ssim(
                            sensor_obs_normalized,
                            expected_obs_normalized,
                            data_range=255,
                        )

                        # Depth should have high structural similarity, same as RGB
                        # SSIM is designed to be robust to small numerical differences
                        MIN_DEPTH_SSIM_THRESHOLD = ssim_threshold

                        if ssim_score < MIN_DEPTH_SSIM_THRESHOLD:
                            # Calculate basic pixel stats for debugging
                            diff = np.abs(sensor_obs - expected_obs)
                            diff_max = np.max(diff)
                            diff_mean = np.mean(diff)
                            num_different = np.sum(diff > 0.01)  # >10mm difference
                            total_pixels = diff.size
                            percent_different = 100 * num_different / total_pixels

                            raise AssertionError(
                                f"Depth sensor {sensor_name} has low structural similarity to saved test data. "
                                f"SSIM score: {ssim_score:.4f} (threshold: {MIN_DEPTH_SSIM_THRESHOLD}). "
                                f"This suggests major rendering differences across platforms. "
                                f"Pixel-level stats: max diff={diff_max * 1000:.1f}mm, "
                                f"mean diff={diff_mean * 1000:.1f}mm, "
                                f"{percent_different:.1f}% pixels differ >10mm"
                            )

                        print(
                            f"  ✓ Depth SSIM: {ssim_score:.4f} (threshold: {MIN_DEPTH_SSIM_THRESHOLD})"
                        )
                        continue

                    # If we get here, we're doing exact comparison (for local determinism tests)
                    from molmo_spaces.utils.depth_utils import detect_depth_edges

                    # Detect edges in expected depth (where large errors are expected from small motion)
                    # Use a moderate threshold (50mm gradient) to catch occlusion boundaries
                    edge_mask = detect_depth_edges(expected_obs, gradient_threshold_mm=50.0)

                    # Create mask for smooth (non-edge) regions
                    smooth_mask = ~edge_mask

                    # Calculate differences
                    diff = np.abs(sensor_obs - expected_obs)

                    # Compare smooth regions with tight tolerance (5mm)
                    # Edge regions may have large differences due to slight motion/alignment
                    if np.sum(smooth_mask) > 0:
                        smooth_diff = diff[smooth_mask]
                        max_smooth_diff = np.max(smooth_diff)
                        mean_smooth_diff = np.mean(smooth_diff)

                        # Check smooth regions only (edges can differ due to motion)
                        if max_smooth_diff > 0.005:
                            # Calculate stats for error reporting
                            max_diff_overall = np.max(diff)
                            mean_diff_overall = np.mean(diff)
                            edge_pixels = np.sum(edge_mask)
                            smooth_pixels = np.sum(smooth_mask)

                            raise AssertionError(
                                f"Depth sensor {sensor_name} differs from saved test data. "
                                f"Smooth regions (non-edges): Max diff: {max_smooth_diff * 1000:.3f}mm, "
                                f"Mean diff: {mean_smooth_diff * 1000:.3f}mm "
                                f"({smooth_pixels:,} pixels). "
                                f"Overall: Max diff: {max_diff_overall * 1000:.3f}mm, "
                                f"Mean diff: {mean_diff_overall * 1000:.3f}mm "
                                f"({edge_pixels:,} edge pixels masked out)"
                            )
                    else:
                        raise AssertionError(
                            f"Depth sensor {sensor_name}: All pixels are edges, cannot compare"
                        )
                else:
                    # Compare RGB observations using SSIM
                    assert_observations_match(
                        sensor_obs,
                        expected_obs,
                        sensor_name,
                        atol=atol,
                        rtol=rtol,
                        ssim_threshold=ssim_threshold,
                    )

        # Verify we found all expected camera sensors
        for expected_camera in expected_cameras:
            assert expected_camera in camera_sensors_found, (
                f"Expected camera {expected_camera} not found in observations"
            )

        # Verify observations changed from initial (if initial observations provided)
        if initial_obs_dict is not None:
            for sensor_name in camera_sensors_found:
                # Skip ignored cameras from this verification too
                if sensor_name in ignore_cameras:
                    continue

                if sensor_name in initial_obs_dict:
                    initial_obs = initial_obs_dict[sensor_name]
                    final_obs = obs_dict[sensor_name]

                    # Calculate pixel difference
                    diff = np.abs(final_obs.astype(np.int32) - initial_obs.astype(np.int32))
                    num_different_pixels = np.sum(diff > 0)
                    total_pixels = diff.size
                    percent_changed = 100 * num_different_pixels / total_pixels

                    # Observations should have changed at least slightly (>0.1% of pixels)
                    assert percent_changed > 0.1, (
                        f"Sensor {sensor_name}: Observations didn't change after running steps. "
                        f"Only {num_different_pixels}/{total_pixels} pixels ({percent_changed:.4f}%) changed. "
                        f"This suggests the robot didn't move or the camera view didn't update."
                    )

                    print(f"[OBS CHANGE] {sensor_name}: {percent_changed:.2f}% of pixels changed")

    except AssertionError:
        assertion_failed = True
        raise
    finally:
        # Always save debug images for visual inspection (if requested)
        if debug_images_dir is not None and camera_sensors_found:
            expected_dict = {}
            for sensor_name in camera_sensors_found:
                # Skip loading ignored cameras for debug images too
                if sensor_name in ignore_cameras:
                    continue
                test_data_path = test_data_dir / f"{test_data_prefix}{sensor_name}.npy"
                expected_dict[sensor_name] = np.load(test_data_path)

            # Add FAILED_ prefix if test failed
            final_prefix = f"FAILED_{debug_prefix}" if assertion_failed else debug_prefix
            save_observation_comparison(
                obs_dict, expected_dict, debug_images_dir, prefix=final_prefix
            )

    return camera_sensors_found


def save_observation_comparison(obs_dict, expected_dict, output_dir, prefix="comparison"):
    """
    Save visual observation comparisons including actual, expected, and difference images.

    Args:
        obs_dict: Dictionary of actual observations from a single environment
        expected_dict: Dictionary of expected observations with same structure
        output_dir: Path object or string for the debug output directory
        prefix: Prefix for the saved image filenames
    """
    from molmo_spaces.utils.depth_utils import visualize_depth_error, visualize_depth_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sensor_name in obs_dict:
        if "camera" in sensor_name and "sensor_param" not in sensor_name:
            if sensor_name in expected_dict:
                actual_data = obs_dict[sensor_name]
                expected_data = expected_dict[sensor_name]

                if isinstance(actual_data, np.ndarray) and isinstance(expected_data, np.ndarray):
                    # Handle depth sensors differently
                    if sensor_name.endswith("_depth"):
                        # Convert float16 to float32 if needed
                        if expected_data.dtype == np.float16:
                            expected_data = expected_data.astype(np.float32)

                        # Visualize actual depth
                        visualize_depth_image(
                            actual_data,
                            f"{sensor_name} - Actual",
                            save_path=output_dir / f"{prefix}_{sensor_name}_actual.png",
                        )

                        # Visualize expected depth
                        visualize_depth_image(
                            expected_data,
                            f"{sensor_name} - Expected",
                            save_path=output_dir / f"{prefix}_{sensor_name}_expected.png",
                        )

                        # Visualize error/difference
                        error = np.abs(actual_data - expected_data)
                        visualize_depth_error(
                            expected_data,
                            actual_data,
                            error,
                            f"{sensor_name} - Comparison Error",
                            save_path=output_dir / f"{prefix}_{sensor_name}_error.png",
                        )

                        print(f"Saved depth visualizations for {sensor_name} at {output_dir}")
                    else:
                        # RGB sensors - save as images
                        if actual_data.dtype != np.uint8:
                            actual_data = np.clip(actual_data, 0, 255).astype(np.uint8)
                        img_actual = Image.fromarray(actual_data)
                        img_actual.save(output_dir / f"{prefix}_{sensor_name}_actual.png")

                        if expected_data.dtype != np.uint8:
                            expected_data = np.clip(expected_data, 0, 255).astype(np.uint8)
                        img_expected = Image.fromarray(expected_data)
                        img_expected.save(output_dir / f"{prefix}_{sensor_name}_expected.png")
                        print(
                            f"Saved {prefix}_{sensor_name}_actual.png and {prefix}_{sensor_name}_expected.png at path {output_dir}"
                        )


def compare_h5_groups(g1, g2, path="/", atol=1e-6, ignore_paths=None):
    """Recursively compare two HDF5 groups and check for differences.

    Args:
        g1: First HDF5 group
        g2: Second HDF5 group
        path: Current path in the HDF5 structure (for error messages)
        atol: Absolute tolerance for numerical comparisons
        ignore_paths: Optional set/list of path suffixes to skip (e.g., {"object_image_points"})
    """
    ignore_paths = ignore_paths or set()
    for name in g1:
        item_path = path + name

        # Skip paths that match any ignore pattern
        if any(item_path.endswith(ignore_suffix) for ignore_suffix in ignore_paths):
            continue

        assert name in g2, f"Missing in second file: {item_path}"

        obj1 = g1[name]
        obj2 = g2[name]

        assert type(obj1) is type(obj2), (
            f"Type mismatch at {item_path}: {type(obj1)} vs {type(obj2)}"
        )

        # Both are groups → recurse
        if isinstance(obj1, h5py.Group) and isinstance(obj2, h5py.Group):
            compare_h5_groups(
                obj1, obj2, path=item_path + "/", atol=atol, ignore_paths=ignore_paths
            )

        # Both are datasets → compare
        elif isinstance(obj1, h5py.Dataset) and isinstance(obj2, h5py.Dataset):
            d1 = obj1[()]
            d2 = obj2[()]

            assert type(d1) is type(d2), f"Type mismatch at {item_path}: {type(d1)} vs {type(d2)}"

            # only check values for numerical arrays and scalars, strings get arbitrarily complicated
            if isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
                assert d1.shape == d2.shape, (
                    f"Shape mismatch at {item_path}: {d1.shape} vs {d2.shape}"
                )
                assert d1.dtype == d2.dtype, (
                    f"Type mismatch at {item_path}: {d1.dtype} vs {d2.dtype}"
                )

                # don't check values for byte-encoded dicts, since that gets arbitrarily complicated
                if d1.dtype != np.uint8:
                    if not np.allclose(d1, d2, atol=atol, equal_nan=True):
                        if np.issubdtype(d1.dtype, np.bool_):
                            n_diff = np.sum(d1 != d2)
                            msg = (
                                f"Boolean mismatch at {item_path}, {n_diff}/{d1.size} elems differ"
                            )
                        else:
                            err = np.abs(d1 - d2).max()
                            msg = f"Data mismatch at {item_path}, w/ max err {err}"
                        log.warning(f"{d1}")
                        log.warning(f"{d2}")
                        log.warning(msg)
                        raise AssertionError(msg)

            elif isinstance(d1, float | int) and isinstance(d2, float | int):
                if not np.allclose(d1, d2, atol=atol):
                    err = np.abs(d1 - d2).max()
                    log.warning(f"Data mismatch at {item_path}, w/ max err {err}")
                    log.warning(f"{d1}")
                    log.warning(f"{d2}")
                    raise AssertionError

            # Compare attributes
            for attr in obj1.attrs:
                assert attr in obj2.attrs, (
                    f"Missing attribute '{attr}' in {item_path} (second file)"
                )
                a1 = obj1.attrs[attr]
                a2 = obj2.attrs[attr]
                if isinstance(a1, bytes):
                    a1 = a1.decode(errors="ignore")
                if isinstance(a2, bytes):
                    a2 = a2.decode(errors="ignore")
                if isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray):
                    assert np.allclose(a1, a2, equal_nan=True, atol=atol), (
                        f"Attribute mismatch at {item_path}/{attr}"
                    )
                else:
                    assert a1 == a2, f"Attribute mismatch at {item_path}/{attr}: {a1!r} vs {a2!r}"

    # Check for extra keys in g2
    for name in g2:
        assert name in g1, f"Missing in first file: {path + name}"


def assert_python_types_equal(pfx: str, v1, v2, atol=0.001):
    """
    General (recursive) function to assert that two python objects are equal, with tolerance applied for floats.
    Works for native python primitives only.
    """
    if type(v1) is not type(v2):
        raise AssertionError(f"{pfx} type mismatch: {type(v1)} vs {type(v2)}")

    if isinstance(v1, str | bytes | bool | int):
        assert v1 == v2, f"{pfx} value mismatch: {v1} vs {v2}"
    elif isinstance(v1, float):
        assert abs(v1 - v2) < atol, f"{pfx} value mismatch: {v1} vs {v2}"
    elif isinstance(v1, list | tuple):
        assert len(v1) == len(v2), f"{pfx} length mismatch: {len(v1)} vs {len(v2)}"
        for i in range(len(v1)):
            assert_python_types_equal(f"{pfx}[{i}]", v1[i], v2[i], atol=atol)
    elif isinstance(v1, dict):
        assert set(v1.keys()) == set(v2.keys()), (
            f"{pfx} keys mismatch: {set(v1.keys())} vs {set(v2.keys())}"
        )
        for k in v1:
            assert_python_types_equal(f"{pfx}[{k}]", v1[k], v2[k], atol=atol)
    else:
        raise AssertionError(f"{pfx} unknown type: {type(v1)}")


def assert_obs_scene_match(g1: h5py.Group, g2: h5py.Group, atol=0.001):
    """
    Assert that the obs_scenes of two trajectory groups are equal.

    Args:
        g1: h5py.Group of the first trajectory
        g2: h5py.Group of the second trajectory
    """
    obs_scene_1 = json.loads(g1["obs_scene"][()].decode("utf-8").rstrip("\x00"))
    obs_scene_2 = json.loads(g2["obs_scene"][()].decode("utf-8").rstrip("\x00"))
    if set(obs_scene_1.keys()) != set(obs_scene_2.keys()):
        obs_scene_1_keys = set(obs_scene_1.keys())
        obs_scene_2_keys = set(obs_scene_2.keys())
        f2_missing = obs_scene_1_keys - obs_scene_2_keys
        f1_missing = obs_scene_2_keys - obs_scene_1_keys
        raise AssertionError(
            f"obs_scene keys mismatch: {g1.file.filename}:{g1.name}/obs_scene missing {f1_missing} "
            f"and {g2.file.filename}:{g2.name}/obs_scene missing {f2_missing}"
        )

    try:
        for k, v1 in obs_scene_1.items():
            # frozen_config is a pickled object, so don't compare it
            if k == "frozen_config":
                continue

            v2 = obs_scene_2[k]
            assert_python_types_equal(f"obs_scene[{k}]", v1, v2, atol=atol)
    except AssertionError as e:
        raise AssertionError(
            f"obs_scene mismatch: {g1.file.filename}:{g1.name} vs {g2.file.filename}:{g2.name}: {e}"
        ) from e


def verify_video_fps(dir: Path, expected_fps: float):
    """Assert all videos in a directory have the expected FPS."""
    for vid_file in dir.glob("*.mp4"):
        vr = decord.VideoReader(str(vid_file))
        fps = vr.get_avg_fps()
        assert np.isclose(fps, expected_fps, atol=1e-2), (
            f"Expected {vid_file} to be {expected_fps} fps, got {fps}"
        )
