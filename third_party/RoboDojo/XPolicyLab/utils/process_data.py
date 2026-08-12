import numpy as np
import os
import cv2
from XPolicyLab.utils.load_file import load_json, load_yaml

def _validate_config(action_type: str, robot_action_dim_info: dict, source_type: str):
    """
    Validate configuration and return normalized values.

    Args:
        action_type: 'joint' or 'ee'.
        robot_action_dim_info: Dict with keys:
            - 'arm_dim': list[int]
            - 'ee_dim': list[int]
        source_type: 'obs' or 'dataset'.

    Returns:
        arm_dims, ee_dims, num_arms
    """
    if action_type not in {"joint", "ee"}:
        raise ValueError(
            f"Unsupported action_type: {action_type!r}. "
            "Supported values are 'joint' and 'ee'."
        )

    if source_type not in {"obs", "dataset"}:
        raise ValueError(
            f"Unsupported source_type: {source_type!r}. "
            "Supported values are 'obs' and 'dataset'."
        )

    if "arm_dim" not in robot_action_dim_info or "ee_dim" not in robot_action_dim_info:
        raise KeyError("robot_action_dim_info must contain both 'arm_dim' and 'ee_dim'.")

    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]

    if not isinstance(arm_dims, (list, tuple)) or not isinstance(ee_dims, (list, tuple)):
        raise TypeError("'arm_dim' and 'ee_dim' must be list or tuple.")

    if len(arm_dims) != len(ee_dims):
        raise ValueError(
            f"'arm_dim' and 'ee_dim' must have the same length, "
            f"got {len(arm_dims)} and {len(ee_dims)}."
        )

    if len(arm_dims) not in {1, 2}:
        raise ValueError(
            f"Only single-arm or dual-arm robots are supported, got {len(arm_dims)} arms."
        )

    if any(d <= 0 for d in arm_dims) or any(d <= 0 for d in ee_dims):
        raise ValueError("All dimensions in 'arm_dim' and 'ee_dim' must be positive.")

    return list(arm_dims), list(ee_dims), len(arm_dims)


def _get_state_keys(action_type: str, num_arms: int, source_type: str):
    """
    Return arm keys and ee keys for the current state schema.

    source_type='obs' uses singular keys:
        single-arm:
            action_type='joint' -> ['joint_state'], ['ee_joint_state']
            action_type='ee'    -> ['ee_pose'], ['ee_joint_state']
        dual-arm:
            action_type='joint' -> ['left_arm_joint_state', 'right_arm_joint_state'],
                                   ['left_ee_joint_state', 'right_ee_joint_state']
            action_type='ee'    -> ['left_ee_pose', 'right_ee_pose'],
                                   ['left_ee_joint_state', 'right_ee_joint_state']

    source_type='dataset' uses plural keys:
        single-arm:
            action_type='joint' -> ['joint_states'], ['ee_joint_states']
            action_type='ee'    -> ['ee_poses'], ['ee_joint_states']
        dual-arm:
            action_type='joint' -> ['left_arm_joint_states', 'right_arm_joint_states'],
                                   ['left_ee_joint_states', 'right_ee_joint_states']
            action_type='ee'    -> ['left_ee_poses', 'right_ee_poses'],
                                   ['left_ee_joint_states', 'right_ee_joint_states']
    """
    suffix = "" if source_type == "obs" else "s"

    if num_arms == 1:
        arm_keys = [f"joint_state{suffix}"] if action_type == "joint" else [f"ee_pose{suffix}"]
        ee_keys = [f"ee_joint_state{suffix}"]
    else:
        if action_type == "joint":
            arm_keys = [
                f"left_arm_joint_state{suffix}",
                f"right_arm_joint_state{suffix}",
            ]
        else:
            arm_keys = [
                f"left_ee_pose{suffix}",
                f"right_ee_pose{suffix}",
            ]

        ee_keys = [
            f"left_ee_joint_state{suffix}",
            f"right_ee_joint_state{suffix}",
        ]

    return arm_keys, ee_keys


def _ensure_valid_state_array(name: str, value, expected_last_dim: int) -> np.ndarray:
    """Convert value to np.ndarray and validate only the last dimension."""
    arr = np.asarray(value)

    if arr.shape[-1] != expected_last_dim:
        raise ValueError(
            f"State field '{name}' last dim mismatch: "
            f"expected {expected_last_dim}, got {arr.shape[-1]}."
        )

    return arr

def pack_robot_state(
    obs: dict,
    action_type: str,
    robot_action_dim_info: dict,
    source_type: str = "obs",
    state_type: str = "state",
) -> np.ndarray:
    """
    Pack robot state from obs['state'] into one vector along the last dimension.

    Packing order:
        single-arm:
            [arm_0, ee_0]
        dual-arm:
            [arm_0, ee_0, arm_1, ee_1]
    """
    if state_type not in obs:
        raise KeyError(f"Input obs must contain a '{state_type}' field.")

    state_dict = obs[state_type]

    arm_dims, ee_dims, num_arms = _validate_config(action_type, robot_action_dim_info, source_type)
    arm_keys, ee_keys = _get_state_keys(action_type, num_arms, source_type)

    parts = []
    expected_prefix_shape = None

    for i, (arm_key, ee_key, arm_dim, ee_dim) in enumerate(
        zip(arm_keys, ee_keys, arm_dims, ee_dims)
    ):
        if arm_key not in state_dict:
            raise KeyError(f"Missing key '{arm_key}' in obs['state'] for arm {i}.")
        if ee_key not in state_dict:
            raise KeyError(f"Missing key '{ee_key}' in obs['state'] for arm {i}.")

        arm_value = _ensure_valid_state_array(arm_key, state_dict[arm_key], arm_dim)
        ee_value = _ensure_valid_state_array(ee_key, state_dict[ee_key], ee_dim)

        if arm_value.shape[:-1] != ee_value.shape[:-1]:
            raise ValueError(
                f"'{arm_key}' and '{ee_key}' must share the same prefix shape, "
                f"got {arm_value.shape[:-1]} and {ee_value.shape[:-1]}."
            )

        if expected_prefix_shape is None:
            expected_prefix_shape = arm_value.shape[:-1]
        elif arm_value.shape[:-1] != expected_prefix_shape:
            raise ValueError(
                "All state fields must share the same prefix shape. "
                f"Expected {expected_prefix_shape}, got {arm_value.shape[:-1]} for '{arm_key}'."
            )

        parts.append(np.concatenate([arm_value, ee_value], axis=-1))

    return np.concatenate(parts, axis=-1)


def unpack_robot_state(
    packed_state,
    action_type: str,
    robot_action_dim_info: dict,
    source_type: str = "obs",
):
    """
    Unpack packed robot state.

    Rules:
        - source_type='obs':
            * ndim must be <= 2
            * if ndim == 1: return dict
            * if ndim == 2: return list[dict]
        - source_type='dataset':
            * return dict of arrays directly

    Unpacking order:
        single-arm:
            [arm_0, ee_0]
        dual-arm:
            [arm_0, ee_0, arm_1, ee_1]
    """
    arm_dims, ee_dims, num_arms = _validate_config(action_type, robot_action_dim_info, source_type)
    arm_keys, ee_keys = _get_state_keys(action_type, num_arms, source_type)

    packed = np.asarray(packed_state)
    expected_dim = sum(arm_dims) + sum(ee_dims)

    if packed.shape[-1] != expected_dim:
        raise ValueError(
            f"packed_state last dim mismatch: expected {expected_dim}, got {packed.shape[-1]}."
        )

    if source_type == "obs":
        assert packed.ndim <= 2, (
            f"When source_type='obs', packed_state.ndim must be <= 2, got {packed.ndim}."
        )

        def _unpack_single_action(single_action: np.ndarray) -> dict:
            result = {}
            offset = 0

            for arm_key, ee_key, arm_dim, ee_dim in zip(
                arm_keys, ee_keys, arm_dims, ee_dims
            ):
                result[arm_key] = single_action[offset : offset + arm_dim]
                offset += arm_dim

                result[ee_key] = single_action[offset : offset + ee_dim]
                offset += ee_dim

            return result

        if packed.ndim == 1:
            return _unpack_single_action(packed)

        return [_unpack_single_action(single_action) for single_action in packed]

    result = {}
    offset = 0

    for arm_key, ee_key, arm_dim, ee_dim in zip(arm_keys, ee_keys, arm_dims, ee_dims):
        result[arm_key] = packed[..., offset : offset + arm_dim]
        offset += arm_dim

        result[ee_key] = packed[..., offset : offset + ee_dim]
        offset += ee_dim

    return result

def get_robot_action_dim_info(env_cfg_type):
    env_cfg = load_yaml(os.path.join(os.path.dirname(__file__), "../../env_cfg", f"{env_cfg_type}.yml"))
    robot_name = env_cfg['config']['robot']
    robot_action_dim_info = load_json(os.path.join(os.path.dirname(__file__), "../../env_cfg/robot", "_robot_info.json"))[robot_name]

    return robot_action_dim_info

def get_batch_size(env_cfg_type):
    env_cfg = load_yaml(os.path.join(os.path.dirname(__file__), "../../env_cfg", f"{env_cfg_type}.yml"))
    sim_cfg = env_cfg['config']['sim']
    sim_info = load_yaml(os.path.join(os.path.dirname(__file__), "../../env_cfg/sim", f"{sim_cfg}.yml"))

    return sim_info['scene']['num_envs']

def get_action_dim(env_cfg_type):
    env_cfg = load_yaml(os.path.join(os.path.dirname(__file__), "../../env_cfg", f"{env_cfg_type}.yml"))
    robot_name = env_cfg['config']['robot']
    robot_action_dim_info = load_json(os.path.join(os.path.dirname(__file__), "../../env_cfg/robot", "_robot_info.json"))[robot_name]
    return sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"])

def _decode_single_image_bit(image_bit):
    """Decode one encoded image buffer into an HWC uint8 RGB array."""
    if isinstance(image_bit, np.ndarray) and image_bit.dtype.kind in {"S", "U"}:
        image_bit = image_bit.item() if image_bit.ndim == 0 else image_bit.tobytes()

    if isinstance(image_bit, str):
        image_bit = image_bit.encode("utf-8")
    elif isinstance(image_bit, memoryview):
        image_bit = image_bit.tobytes()

    if isinstance(image_bit, (bytes, bytearray)):
        # Fixed-width HDF5 byte columns pad the tail with NUL.
        image_bit = image_bit.rstrip(b"\0")
    elif isinstance(image_bit, np.ndarray):
        image_bit = np.ascontiguousarray(image_bit)

    # The returned array is RGB. This is the conclusion for this repo — do not
    # "correct" it with the usual "cv2 means BGR" rule. cv2.imencode/imdecode
    # only carry channels in the order they were handed in, so the round trip
    # is an identity on channel order, and XPolicyLab trajectory files and
    # runtime observations are encoded from RGB arrays. Adding a COLOR_BGR2RGB
    # swap here (or in any caller) is what actually breaks the channel order.
    image = cv2.imdecode(np.frombuffer(image_bit, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"Failed to decode image bits (type={type(image_bit).__name__}, "
            f"size={getattr(image_bit, 'size', len(image_bit) if hasattr(image_bit, '__len__') else '?')})."
        )

    return image


def _decode_image_bit_sequence(image_bits):
    frames = []

    for index, image_bit in enumerate(image_bits):
        try:
            frames.append(decode_image_bit(image_bit))
        except ValueError as exc:
            raise ValueError(f"Frame {index}: {exc}") from exc

    if not frames:
        return np.zeros((0,), dtype=np.uint8)

    return np.stack(frames, axis=0)


def decode_image_bit(image_bits):
    """
    Decode encoded image bit stream(s) into uint8 RGB image array(s).

    The output is RGB. Treat that as settled and do not apply the usual
    "OpenCV returns BGR" rule: XPolicyLab trajectory files and runtime
    observations store buffers encoded from RGB arrays, and encode/decode
    round trips preserve the channel order they were given. Never add a
    COLOR_BGR2RGB after this function to "correct" the output — there is
    nothing to correct, and the swap is what breaks the channel order.

    Deliberately converting the RGB result to BGR is a different thing and is
    allowed where a checkpoint was trained on BGR data; that must be an opt-in
    documented in the adapter (see Dexora_1B's `input_color_order`), never a
    silent fix applied at the decode site.

    Values that are already decoded are returned unchanged, so this function is
    safe to call on an observation or trajectory field without knowing whether
    the producer encoded it.

    Dispatch is on dtype first, then ndim:
        - bytes / bytearray / memoryview / str  -> one encoded buffer
        - ndarray of dtype kind 'S', 'U', 'O'   -> sequence of encoded buffers,
                                                   or one buffer when 0-d
        - uint8 ndarray, ndim == 1              -> one encoded buffer
        - uint8 ndarray, ndim == 2              -> (T, N) stack of encoded buffers
        - uint8 ndarray, ndim >= 3              -> already decoded, returned as is
        - ndarray of any other dtype            -> already decoded, returned as is
        - list / tuple                          -> element-wise, stacked on axis 0

    Grayscale (H, W) uint8 images are not supported: a 2-D uint8 array is always
    read as a stack of encoded buffers.

    Raises:
        ValueError: if any buffer fails to decode.
    """
    if isinstance(image_bits, (bytes, bytearray, memoryview, str)):
        return _decode_single_image_bit(image_bits)

    if isinstance(image_bits, np.ndarray):
        if image_bits.dtype.kind in {"S", "U", "O"}:
            if image_bits.ndim == 0:
                return _decode_single_image_bit(image_bits.item())
            return _decode_image_bit_sequence(image_bits)

        if image_bits.dtype == np.uint8:
            if image_bits.ndim == 1:
                return _decode_single_image_bit(image_bits)
            if image_bits.ndim == 2:
                return _decode_image_bit_sequence(image_bits)

        return image_bits

    if isinstance(image_bits, (list, tuple)):
        return _decode_image_bit_sequence(image_bits)

    return _decode_single_image_bit(image_bits)


# Color fields of one camera in an observation. 'depth' is deliberately absent:
# depth is stored as 16-bit or float data that cv2.IMREAD_COLOR would destroy.
OBS_IMAGE_KEYS = ("color", "colors", "rgb", "image")

_DECODABLE_TYPES = (bytes, bytearray, memoryview, np.ndarray, list, tuple)


def _decode_obs_image(value, camera_name, image_key):
    if value is None:
        return value

    try:
        return decode_image_bit(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Failed to decode obs['vision']['{camera_name}']['{image_key}']: {exc}"
        ) from exc


def decode_obs_images(obs):
    """
    Decode the encoded color streams of a runtime observation, in place.

    The policy server calls this before handing an observation to the model, so
    `update_obs` / `update_obs_batch` always receive plain RGB image arrays and
    no adapter has to decode anything. Values that are already decoded pass
    through untouched, and only the color fields listed in OBS_IMAGE_KEYS are
    visited — depth maps, intrinsics, extrinsics and shapes are left alone.

    What the model receives is RGB. That is the conclusion; `model.py` must not
    swap channels to "undo" an OpenCV BGR convention that does not apply here.

    Args:
        obs: One observation dict, or a list/tuple of them for batched eval.
             Anything else is returned unchanged.

    Returns:
        The same object, with encoded color fields replaced by decoded arrays.

    Raises:
        ValueError: if a color field fails to decode, naming the camera.
    """
    if isinstance(obs, (list, tuple)):
        for single_obs in obs:
            decode_obs_images(single_obs)
        return obs

    if not isinstance(obs, dict):
        return obs

    vision = obs.get("vision")
    if not isinstance(vision, dict):
        return obs

    for camera_name, camera in vision.items():
        if isinstance(camera, dict):
            for image_key in OBS_IMAGE_KEYS:
                if image_key in camera:
                    camera[image_key] = _decode_obs_image(
                        camera[image_key], camera_name, image_key
                    )
        elif isinstance(camera, _DECODABLE_TYPES):
            # Layout where vision/<camera> is the image itself.
            vision[camera_name] = _decode_obs_image(camera, camera_name, "color")

    return obs

def images_encoding(imgs):
    """
    JPEG-encode frames for storage in the XPolicyLab layout, padded to a
    common length so they fit a fixed-width HDF5 byte column.

    Feed this RGB, never BGR and never a channel-swapped array: cv2.imencode
    writes channels in the order it is given, so whatever goes in is what
    `decode_image_bit` hands back, and the rest of the pipeline assumes RGB.
    """
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len