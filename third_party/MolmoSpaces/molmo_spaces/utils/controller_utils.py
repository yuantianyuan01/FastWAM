import numpy as np


def find_nearest_equivalent_angle(curr, target, steer_angle_range):
    k_min = int(np.floor((steer_angle_range[0] - target) / (2 * np.pi)))
    k_max = int(np.ceil((steer_angle_range[1] - target) / (2 * np.pi)))
    candidates = [target + k * 2 * np.pi for k in range(k_min, k_max + 1)]
    candidates = [c for c in candidates if steer_angle_range[0] <= c <= steer_angle_range[1]]
    if not candidates:
        return np.clip(target, steer_angle_range[0], steer_angle_range[1])
    candidates = np.array(candidates)
    idx = np.argmin(np.abs(candidates - curr))
    return candidates[idx]


def optimize_steer_and_drive(curr, target, speed, steer_angle_range):
    a = find_nearest_equivalent_angle(curr, target, steer_angle_range)
    cost_a = abs(a - curr)
    b = find_nearest_equivalent_angle(curr, target + np.pi, steer_angle_range)
    cost_b = abs(b - curr)
    if cost_a <= cost_b:
        return a, speed
    else:
        return b, -speed


def optimize_all_steer_and_drive(
    current_angles, target_angles, target_speeds, steer_angle_range, max_wheel_speed
):
    optimized = [
        optimize_steer_and_drive(c, t, s, steer_angle_range, max_wheel_speed)
        for c, t, s in zip(current_angles, target_angles, target_speeds)
    ]
    angles, speeds = zip(*optimized)
    speeds = np.array(speeds)
    if np.any(np.abs(speeds) > max_wheel_speed):
        factor = max_wheel_speed / np.max(np.abs(speeds))
        speeds = speeds * factor
    return np.array(angles), speeds
