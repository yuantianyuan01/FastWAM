"""
Convert MlSpaces h5 data to LeRobot format.

This script only works for MlSpaces data with Franka-DROID robots, not RB-Y1.
"""
import argparse
import json
from pathlib import Path

import decord
import h5py
import numpy as np
from decord import VideoReader
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

decord.bridge.set_bridge("native")

MLSPACES_GRIPPER_MAX_POS = 0.824033  # Franka DROID gripper max position
GRIPPER_ACTION_SCALE = 255.0
IMG_HW = (180, 320)
STATE_DIM = 17

CAMERAS = [
    ("wrist_left", "wrist_camera_zed_mini"),
    ("exterior_1_left", "randomized_zed2_analogue_1"),
    ("exterior_2_left", "randomized_zed2_analogue_2"),
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path, help="Postprocessed MlSpaces split dir (has valid_trajectory_index.json)")
    ap.add_argument("--repo-id", default="local/molmobot_droid_format")
    ap.add_argument("--root", type=Path, default=None, help="Output root (defaults to $HF_LEROBOT_HOME/<repo_id>)")
    ap.add_argument("--max-episodes", type=int, default=None, help="Limit episodes for quick tests")
    ap.add_argument("--image-writer-processes", type=int, default=2)
    ap.add_argument("--image-writer-threads", type=int, default=8)
    return ap.parse_args()


def decode_json_bytes(h5_row) -> dict:
    raw = h5_row.tobytes() if isinstance(h5_row, np.ndarray) else h5_row
    return json.loads(raw.decode("utf-8").rstrip("\x00"))


def pose_quat_wxyz_to_eef9d(pose7: np.ndarray) -> np.ndarray:
    """[x,y,z,qw,qx,qy,qz] (scalar-first) -> eef_9d = [x,y,z, R[:,0], R[:,1]]."""
    xyz = pose7[..., :3]
    rotmat = Rotation.from_quat(pose7[..., 3:], scalar_first=True).as_matrix()
    rot6d = np.concatenate([rotmat[..., :, 0], rotmat[..., :, 1]], axis=-1)
    return np.concatenate([xyz, rot6d], axis=-1).astype(np.float32)


def load_video_frames(h5_dir: Path, traj_group: h5py.Group, ml_cam: str, count: int) -> np.ndarray:
    raw = traj_group[f"obs/sensor_data/{ml_cam}"][()]
    filename = (raw.tobytes() if isinstance(raw, np.ndarray) else raw).decode("utf-8").rstrip("\x00")
    vr = VideoReader(str(h5_dir / filename))
    if count > len(vr):
        raise RuntimeError(f"Requested {count} frames but video {filename} has {len(vr)}")
    return vr.get_batch(list(range(count))).asnumpy()  # (count, H, W, 3) RGB uint8


def resize_rgb(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if frame.shape[:2] == (h, w):
        return frame
    return np.array(Image.fromarray(frame).resize((w, h), resample=Image.BICUBIC))


def build_features() -> dict:
    feats = {}
    for short, _ in CAMERAS:
        feats[f"observation.images.{short}"] = {
            "dtype": "video",
            "shape": (*IMG_HW, 3),
            "names": ["height", "width", "channel"],
        }
    feats["observation.state"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]}
    feats["action"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": ["action"]}
    for prefix in ("observation.state", "action"):
        feats[f"{prefix}.eef_9d"] = {"dtype": "float32", "shape": (9,), "names": ["eef_9d"]}
        feats[f"{prefix}.gripper_position"] = {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]}
        feats[f"{prefix}.joint_position"] = {"dtype": "float32", "shape": (7,), "names": ["joint_position"]}
    return feats


def iter_trajectories(data_dir: Path):
    with (data_dir / "valid_trajectory_index.json").open() as f:
        index: dict = json.load(f)
    for house_files in index.values():
        for h5_subpath, traj_lens in house_files.items():
            h5_path = data_dir / h5_subpath
            for traj_key, traj_len in traj_lens.items():
                yield h5_path, traj_key, int(traj_len)


def read_policy_fps(h5_path: Path, traj_key: str) -> int:
    """Derive policy fps from a trajectory's obs_scene metadata."""
    with h5py.File(h5_path, "r") as f:
        scene = decode_json_bytes(f[traj_key]["obs_scene"][()])
    return round(1000.0 / scene["policy_dt_ms"])


def convert_episode(dataset: LeRobotDataset, h5_path: Path, traj_key: str, traj_len: int) -> int:
    """Return number of frames written, or 0 if skipped."""
    # Drop dummy first action, done sentinel last action, and last 2 states
    # (per https://github.com/allenai/molmospaces/blob/main/docs/data_format.md)
    effective = traj_len - 2
    if effective < 1:
        return 0

    with h5py.File(h5_path, "r") as f:
        tg = f[traj_key]
        task: str = decode_json_bytes(tg["obs_scene"][()])["task_description"]

        qpos = [decode_json_bytes(tg["obs/agent/qpos"][i]) for i in range(effective)]
        obs_eef9d = pose_quat_wxyz_to_eef9d(tg["obs/extra/tcp_pose"][:effective])

        # action[i] pairs with state[i] after dropping the padded first action
        act_joint = [decode_json_bytes(tg["actions/joint_pos"][i + 1]) for i in range(effective)]
        act_ee_rows = np.stack(
            [np.asarray(decode_json_bytes(tg["actions/ee_pose"][i + 1])["arm"], dtype=np.float32)
             for i in range(effective)],
            axis=0,
        )
        act_eef9d = pose_quat_wxyz_to_eef9d(act_ee_rows)

        cam_frames = {
            short: load_video_frames(h5_path.parent, tg, ml_cam, effective)
            for short, ml_cam in CAMERAS
        }

    for j in range(effective):
        joint_pos = np.asarray(qpos[j]["arm"], dtype=np.float32)
        gripper_pos = np.asarray([qpos[j]["gripper"][0] / MLSPACES_GRIPPER_MAX_POS], dtype=np.float32)
        act_joint_pos = np.asarray(act_joint[j]["arm"], dtype=np.float32)
        act_gripper = np.asarray([act_joint[j]["gripper"][0] / GRIPPER_ACTION_SCALE], dtype=np.float32)

        frame = {
            "observation.state": np.concatenate([obs_eef9d[j], gripper_pos, joint_pos]),
            "observation.state.eef_9d": obs_eef9d[j],
            "observation.state.gripper_position": gripper_pos,
            "observation.state.joint_position": joint_pos,
            "action": np.concatenate([act_eef9d[j], act_gripper, act_joint_pos]),
            "action.eef_9d": act_eef9d[j],
            "action.gripper_position": act_gripper,
            "action.joint_position": act_joint_pos,
        }
        for short, frames in cam_frames.items():
            frame[f"observation.images.{short}"] = resize_rgb(frames[j], IMG_HW)
        dataset.add_frame(frame, task=task)

    dataset.save_episode()
    return effective


def write_modality_json(root: Path):
    split_17 = {
        "eef_9d": {"start": 0, "end": 9},
        "gripper_position": {"start": 9, "end": 10},
        "joint_position": {"start": 10, "end": 17},
    }
    modality = {
        "state": split_17,
        "action": split_17,
        "video": {short: {"original_key": f"observation.images.{short}"} for short, _ in CAMERAS},
        "annotation": {"language.language_instruction": {"original_key": "task_index"}},
    }
    with (root / "meta" / "modality.json").open("w") as f:
        json.dump(modality, f, indent=2)


def main():
    args = parse_args()

    trajs = list(iter_trajectories(args.data_dir))
    if args.max_episodes:
        trajs = trajs[: args.max_episodes]
    if not trajs:
        raise RuntimeError(f"No trajectories found in {args.data_dir}")

    fps = read_policy_fps(*trajs[0][:2])
    print(f"Converting {len(trajs)} episodes from {args.data_dir} (fps={fps})")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=build_features(),
        robot_type="panda",
        root=args.root,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    skipped = 0
    for h5_path, traj_key, traj_len in tqdm(trajs, desc="episodes"):
        try:
            n = convert_episode(dataset, h5_path, traj_key, traj_len)
            if n == 0:
                skipped += 1
            else:
                total_frames += n
        except Exception as e:
            print(f"\nFAILED {h5_path}::{traj_key}: {e}")
            skipped += 1

    write_modality_json(dataset.root)

    print(f"\nDone. {total_frames} frames across {len(trajs) - skipped} episodes ({skipped} skipped).")
    print(f"Dataset at: {dataset.root}")


if __name__ == "__main__":
    main()
