import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from XPolicyLab.utils.data_loader import load
from XPolicyLab.utils.process_data import decode_image_bit


CAMERA_CANDIDATES = {
    "cam_high": [
        ("vision", "cam_head", "colors"),
        ("vision", "cam_head", "color"),
        ("cam_head", "color"),
        ("cam_head", "colors"),
        ("cam_high", "color"),
        ("cam_high", "colors"),
    ],
    "cam_left_wrist": [
        ("vision", "cam_left_wrist", "colors"),
        ("vision", "cam_left_wrist", "color"),
        ("cam_left_wrist", "color"),
        ("cam_left_wrist", "colors"),
    ],
    "cam_right_wrist": [
        ("vision", "cam_right_wrist", "colors"),
        ("vision", "cam_right_wrist", "color"),
        ("cam_right_wrist", "color"),
        ("cam_right_wrist", "colors"),
    ],
    "cam_wrist": [
        ("vision", "cam_wrist", "colors"),
        ("vision", "cam_wrist", "color"),
        ("cam_wrist", "color"),
        ("cam_wrist", "colors"),
    ],
}


def _get_nested(data, *keys, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_2d_float32(array, name):
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} should be 2D, got shape {arr.shape}")
    return arr


def _ensure_utf8_strings(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _ensure_utf8_strings(value.item())
        return [str(x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_ensure_utf8_strings(item))
        return result
    return [str(value)]


def _find_instructions(data):
    candidates = [
        _get_nested(data, "instructions"),
        _get_nested(data, "instruction"),
        _get_nested(data, "extra_episode_info", "instructions"),
        _get_nested(data, "extra_episode_info", "instruction"),
        _get_nested(data, "meta", "instructions"),
        _get_nested(data, "metadata", "instructions"),
    ]
    for candidate in candidates:
        strings = [s for s in _ensure_utf8_strings(candidate) if s]
        if strings:
            return strings
    return []


def _find_camera_array(data, camera_name):
    for keys in CAMERA_CANDIDATES[camera_name]:
        value = _get_nested(data, *keys)
        if value is not None:
            return _decode_images_if_needed(value)
    return None


def _decode_images_if_needed(images):
    frames = np.asarray(decode_image_bit(images))
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4:
        raise ValueError(f"Expected decoded frames with shape [T,H,W,3], got {frames.shape}")
    return frames.astype(np.uint8, copy=False)


def _concat_state_parts(parts, name):
    valid_parts = []
    horizon = None
    for part_name, value in parts:
        if value is None:
            continue
        arr = _ensure_2d_float32(value, f"{name}.{part_name}")
        if horizon is None:
            horizon = arr.shape[0]
        elif arr.shape[0] != horizon:
            raise ValueError(
                f"{name}.{part_name} horizon mismatch: expected {horizon}, got {arr.shape[0]}"
            )
        valid_parts.append(arr)

    if not valid_parts:
        return None
    return np.concatenate(valid_parts, axis=1)


def _extract_qpos(data):
    qpos = _get_nested(data, "qpos")
    if qpos is not None:
        return _ensure_2d_float32(qpos, "qpos")

    state_qpos = _concat_state_parts(
        [
            ("left_arm_joint_states", _get_nested(data, "state", "left_arm_joint_states")),
            ("left_ee_joint_states", _get_nested(data, "state", "left_ee_joint_states")),
            ("right_arm_joint_states", _get_nested(data, "state", "right_arm_joint_states")),
            ("right_ee_joint_states", _get_nested(data, "state", "right_ee_joint_states")),
        ],
        "state",
    )
    if state_qpos is not None:
        return state_qpos

    left_state = _concat_state_parts(
        [
            ("joint", _get_nested(data, "left_arm", "joint")),
            ("gripper", _get_nested(data, "left_arm", "gripper")),
        ],
        "left_arm",
    )
    right_state = _concat_state_parts(
        [
            ("joint", _get_nested(data, "right_arm", "joint")),
            ("gripper", _get_nested(data, "right_arm", "gripper")),
        ],
        "right_arm",
    )

    parts = [part for part in [left_state, right_state] if part is not None]
    if parts:
        horizon = parts[0].shape[0]
        if any(part.shape[0] != horizon for part in parts):
            raise ValueError("left/right arm qpos horizon mismatch")
        return np.concatenate(parts, axis=1)

    raise ValueError("Cannot find qpos data")


def _extract_action(data, qpos):
    action = _get_nested(data, "action")
    if action is not None:
        action = _ensure_2d_float32(action, "action")
        if action.shape[0] != qpos.shape[0]:
            raise ValueError(
                f"action horizon mismatch: action={action.shape[0]}, qpos={qpos.shape[0]}"
            )
        return action

    left_action = _concat_state_parts(
        [
            ("joint", _get_nested(data, "left_arm", "joint_action")),
            ("gripper", _get_nested(data, "left_arm", "gripper_action")),
        ],
        "left_arm_action",
    )
    right_action = _concat_state_parts(
        [
            ("joint", _get_nested(data, "right_arm", "joint_action")),
            ("gripper", _get_nested(data, "right_arm", "gripper_action")),
        ],
        "right_arm_action",
    )
    parts = [part for part in [left_action, right_action] if part is not None]
    if parts:
        horizon = parts[0].shape[0]
        if any(part.shape[0] != horizon for part in parts):
            raise ValueError("left/right arm action horizon mismatch")
        action = np.concatenate(parts, axis=1)
        if action.shape[0] != qpos.shape[0]:
            raise ValueError(
                f"action horizon mismatch: action={action.shape[0]}, qpos={qpos.shape[0]}"
            )
        return action.astype(np.float32)

    state_action = _concat_state_parts(
        [
            ("left_arm_joint_states", _get_nested(data, "state", "left_arm_joint_states")),
            ("left_ee_joint_states", _get_nested(data, "state", "left_ee_joint_states")),
            ("right_arm_joint_states", _get_nested(data, "state", "right_arm_joint_states")),
            ("right_ee_joint_states", _get_nested(data, "state", "right_ee_joint_states")),
        ],
        "state_action",
    )
    if state_action is not None:
        qpos = state_action.astype(np.float32)

    if len(qpos) == 1:
        return np.zeros_like(qpos, dtype=np.float32)

    next_qpos = qpos[1:]
    last_action = np.zeros((1, qpos.shape[1]), dtype=np.float32)
    return np.concatenate([next_qpos, last_action], axis=0).astype(np.float32)


def _infer_arm_dims(qpos):
    qpos_dim = qpos.shape[1]
    if qpos_dim % 2 == 0:
        return qpos_dim // 2, qpos_dim // 2
    return qpos_dim, 0


def convert_one(input_path, output_path, data_type, data_version):
    data = load(str(input_path), data_type=data_type, data_version=data_version)

    qpos = _extract_qpos(data)
    action = _extract_action(data, qpos)
    instructions = _find_instructions(data)
    left_arm_dim, right_arm_dim = _infer_arm_dims(qpos)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("action", data=action, dtype=np.float32)

        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos, dtype=np.float32)
        obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim, dtype=np.int32))
        obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim, dtype=np.int32))

        images_group = obs.create_group("images")
        for camera_name in CAMERA_CANDIDATES:
            images = _find_camera_array(data, camera_name)
            if images is not None:
                images_group.create_dataset(camera_name, data=images, dtype=np.uint8)

        if instructions:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("instructions", data=np.asarray(instructions, dtype=object), dtype=string_dtype)


def find_input_files(input_dir):
    input_dir = Path(input_dir)
    files = sorted(input_dir.rglob("*.hdf5"))
    files.extend(sorted(input_dir.rglob("*.h5")))
    unique_files = []
    seen = set()
    for file_path in files:
        resolved = str(file_path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(file_path)
    return unique_files


def main():
    parser = argparse.ArgumentParser(description="Transform dataset to aloha hdf5 format.")
    parser.add_argument("input_dir", type=str, help="Input dataset directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--data_type", type=str, default="xspark", help="Dataset type, e.g. xspark")
    parser.add_argument("--data_version", type=str, default="v1.0", help="Dataset version, e.g. v1.0")
    args = parser.parse_args()

    input_files = find_input_files(args.input_dir)
    if not input_files:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found under {args.input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for episode_idx, input_path in enumerate(tqdm(input_files, desc="Converting")):
        output_path = output_dir / f"episode_{episode_idx}.hdf5"
        try:
            convert_one(input_path, output_path, args.data_type, args.data_version)
        except Exception as exc:
            failures.append((str(input_path), str(exc)))

    print(f"Converted {len(input_files) - len(failures)}/{len(input_files)} files to {output_dir}")
    if failures:
        print("Failed files:")
        for file_path, reason in failures:
            print(f"  - {file_path}: {reason}")


if __name__ == "__main__":
    main()
