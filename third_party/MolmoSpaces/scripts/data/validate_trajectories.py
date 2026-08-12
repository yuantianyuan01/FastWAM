import argparse
from concurrent.futures import CancelledError, ProcessPoolExecutor, Future, as_completed
from threading import Semaphore, Lock
from pathlib import Path
import json
import glob
import os
import traceback

import numpy as np
import h5py
from tqdm import tqdm
import decord
from decord.ndarray import DECORDContext


VALID_TRAJECTORY_KEY = "valid_traj_mask"


def get_args():
    parser = argparse.ArgumentParser(
        description="Find and mark invalid trajectories in data files to be skipped in further processing and training."
    )
    parser.add_argument("data_root")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Do not overwrite existing valid trajectory mask in data files (skip instead)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find valid trajectories but do not write to data files or build the trajectory index",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only build the index of valid trajectories, do not write to data files",
    )

    parser.add_argument(
        "--check-visibility",
        nargs=2,
        metavar=("camera", "object"),
        action="append",
        help="Check that a given camera can see a given object in the first frame. Can be specified multiple times.",
    )
    parser.add_argument("--no-video", action="store_true", help="Do not validate video data")
    parser.add_argument(
        "--frames-to-check",
        type=int,
        default=3,
        help="Number of frames to check for each video, ignored if not checking videos",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=3,
        help="Minimum number of steps in a trajectory to be considered valid (including dummy and done steps)",
    )
    return parser.parse_args()


def is_traj_valid(
    args, data_file_path: Path, traj_idx: int, traj_group: h5py.Group, decord_ctx: DECORDContext
) -> bool:
    if "actions" not in traj_group:
        print(f"Trajectory {traj_idx} in {data_file_path} does not have an actions group")
        return False
    actions_group = traj_group["actions"]
    if len(actions_group.keys()) == 0:
        print(f"Trajectory {traj_idx} in {data_file_path} does not have any actions")
        return False
    n_actions = None
    for action_key in actions_group.keys():
        actions = actions_group[action_key]
        if n_actions is None:
            n_actions = actions.shape[0]
        else:
            if n_actions != actions.shape[0]:
                print(
                    f"Trajectory {traj_idx} in {data_file_path} has different numbers of actions for {action_key}"
                )
                return False
        if n_actions < args.min_steps:
            print(
                f"Trajectory {traj_idx} in {data_file_path} has {n_actions} steps, which is <{args.min_steps}"
            )
            return False
        for i in range(n_actions):
            try:
                json.loads(actions[i].tobytes().decode("utf-8").rstrip("\x00"))
            except json.JSONDecodeError:
                print(
                    f"Error decoding action {action_key} step {i} in {data_file_path} for trajectory {traj_idx}"
                )
                return False

    if "obs/agent" not in traj_group:
        print(f"Trajectory {traj_idx} in {data_file_path} does not have an agent group")
        return False
    agent_group = traj_group["obs/agent"]
    if "qpos" not in agent_group or "qvel" not in agent_group:
        print(f"Trajectory {traj_idx} in {data_file_path} does not have a qpos or qvel group")
        return False
    for obs_key in agent_group.keys():
        obs = agent_group[obs_key]
        if obs.shape[0] != n_actions:
            print(
                f"Trajectory {traj_idx} in {data_file_path} has {obs.shape[0]} steps for {obs_key}, but {n_actions} steps for actions"
            )
            return False
        for i in range(obs.shape[0]):
            try:
                json.loads(obs[i].tobytes().decode("utf-8").rstrip("\x00"))
            except json.JSONDecodeError:
                print(f"Error decoding {obs_key} {i} in {data_file_path} for trajectory {traj_idx}")
                return False

    if "obs/extra/object_image_points" in traj_group:
        if args.check_visibility:
            obj_points_group = traj_group["obs/extra/object_image_points"]

            # Handle new nested HDF5 group format vs old JSON byte-string format
            if isinstance(obj_points_group, h5py.Group):
                # New format: nested groups with points/num_points arrays
                for cam_name, obj_name in args.check_visibility:
                    if obj_name not in obj_points_group:
                        print(
                            f"Trajectory {traj_idx} in {data_file_path} does not record visibility for object {obj_name}"
                        )
                        return False
                    obj_group = obj_points_group[obj_name]
                    if cam_name not in obj_group:
                        print(
                            f"Trajectory {traj_idx} in {data_file_path} does not record visibility for object {obj_name} from camera {cam_name}"
                        )
                        return False
                    # Check num_points at first frame (index 0)
                    num_points = obj_group[cam_name]["num_points"][0, 0]
                    if num_points == 0:
                        # this failure case is so common that we don't need to print a warning
                        return False
            else:
                # Old format: JSON byte-string array
                image_points_str = (
                    obj_points_group[0][:]
                    .tobytes()
                    .decode("utf-8")
                    .rstrip("\x00")
                )
                try:
                    image_points: dict[str, dict[str, list[list[float]]]] = json.loads(image_points_str)
                except json.JSONDecodeError:
                    print(
                        f"Error decoding object_image_points in {data_file_path} for trajectory {traj_idx}"
                    )
                    return False
                for cam_name, obj_name in args.check_visibility:
                    if obj_name not in image_points:
                        print(
                            f"Trajectory {traj_idx} in {data_file_path} does not record visibility for object {obj_name}"
                        )
                        return False
                    if cam_name not in image_points[obj_name]:
                        print(
                            f"Trajectory {traj_idx} in {data_file_path} does not record visibility for object {obj_name} from camera {cam_name}"
                        )
                        return False
                    if len(image_points[obj_name][cam_name]) == 0:
                        # this failure case is so common that we don't need to print a warning
                        return False
    elif args.check_visibility:
        print(
            f"WARN: Trajectory {traj_idx} in {data_file_path} does not have a object_image_points group, skipping visibility check."
        )

    if not args.no_video:
        if "obs/sensor_data" not in traj_group:
            print(f"Trajectory {traj_idx} in {data_file_path} does not have a sensor data group")
            return False
        sensor_data_group = traj_group["obs/sensor_data"]
        if len(sensor_data_group.keys()) == 0:
            print(f"Trajectory {traj_idx} in {data_file_path} does not have any cameras")
            return False
        for camera_name in sensor_data_group.keys():
            video_filename: str = (
                sensor_data_group[camera_name][:].tobytes().decode("utf-8").rstrip("\x00")
            )
            video_path = data_file_path.parent / video_filename
            if not video_path.is_file():
                print(
                    f"Video file {video_filename} does not exist for trajectory {traj_idx} in {data_file_path}"
                )
                return False
            try:
                vr = decord.VideoReader(str(video_path), ctx=decord_ctx)
                n_frames = len(vr)
                traj_len = traj_group["obs/agent/qpos"].shape[0]
                if n_frames != traj_len:
                    print(
                        f"{video_filename} has {n_frames=} but trajectory has {traj_len=} for {traj_idx=} in {data_file_path}"
                    )
                    return False
                frame_idxs = np.round(np.linspace(0, n_frames - 1, args.frames_to_check)).astype(
                    int
                )
                vr.get_batch(frame_idxs)
            except (decord.DECORDError, RuntimeError) as e:
                print(
                    f"Error reading video file {video_filename} for trajectory {traj_idx} in {data_file_path}: {e}"
                )
                return False
    return True


def process_data_file(args, data_file_path: Path) -> tuple[int, int]:
    try:
        with h5py.File(data_file_path, "r" if args.dry_run else "r+") as f:
            if VALID_TRAJECTORY_KEY in f:
                if not args.overwrite:
                    valid_traj_mask = f[VALID_TRAJECTORY_KEY][:]
                    return np.sum(~valid_traj_mask).item(), len(valid_traj_mask)
                if not args.dry_run:
                    del f[VALID_TRAJECTORY_KEY]

            traj_idxs = sorted([int(k.split("_")[-1]) for k in f.keys() if k.startswith("traj_")])
            valid_traj_mask = np.zeros(len(traj_idxs), dtype=bool)
            # ensure trajectory indices are consecutive
            if traj_idxs != list(range(len(traj_idxs))):
                print(f"Trajectory indices are not consecutive in {data_file_path}")
            else:
                decord_ctx = decord.cpu() if not args.no_video else None
                for i, traj_idx in enumerate(traj_idxs):
                    valid_traj_mask[i] = is_traj_valid(
                        args, data_file_path, traj_idx, f[f"traj_{traj_idx}"], decord_ctx
                    )

            if not args.dry_run:
                try:
                    f.create_dataset(VALID_TRAJECTORY_KEY, data=valid_traj_mask)
                except:
                    if VALID_TRAJECTORY_KEY in f:
                        del f[VALID_TRAJECTORY_KEY]
                    raise
    except OSError:
        return 0, 0
    except Exception as e:
        raise RuntimeError(f"Error processing data file {data_file_path}: {e}") from e
    return np.sum(~valid_traj_mask).item(), len(valid_traj_mask)


def find_valid_trajectories(args, data_files: list[str]):
    total_n_invalid = 0
    total_n_trajs = 0
    if args.num_workers > 1:
        # use a semaphore to limit the number of queued jobs, helps with large quantity of datafiles
        submit_semaphore = Semaphore(args.num_workers * 4)
        lock = Lock()
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            with tqdm(total=len(data_files), desc="Processing files...") as pbar:

                def on_done(future: Future[tuple[int, int]]):
                    try:
                        n_invalid, n_total = future.result()
                    except CancelledError:
                        pass
                    except:
                        traceback.print_exc()
                        executor.shutdown(wait=False, cancel_futures=True)
                        return

                    with lock:
                        nonlocal total_n_invalid, total_n_trajs
                        total_n_invalid += n_invalid
                        total_n_trajs += n_total
                        pbar.set_postfix(
                            n_invalid=total_n_invalid,
                            n_total=total_n_trajs,
                            invalid_frac=total_n_invalid / total_n_trajs
                            if total_n_trajs > 0
                            else 0,
                        )
                        pbar.update(1)
                    submit_semaphore.release()

                futures: list[Future] = []
                for data_file in data_files:
                    submit_semaphore.acquire()
                    future = executor.submit(process_data_file, args, Path(data_file))
                    future.add_done_callback(on_done)
                    futures.append(future)

                for future in as_completed(futures):
                    future.result()
    else:
        for data_file in (pbar := tqdm(data_files)):
            n_invalid, n_total = process_data_file(args, data_file)
            total_n_invalid += n_invalid
            total_n_trajs += n_total
            pbar.set_postfix(
                n_invalid=total_n_invalid,
                n_total=total_n_trajs,
                invalid_frac=total_n_invalid / total_n_trajs,
            )
    print(f"Found {total_n_invalid} invalid trajectories out of {total_n_trajs} total trajectories")
    print(f"Invalid fraction: {total_n_invalid / total_n_trajs:.1%}")


def read_valid_trajectories(data_file: Path) -> tuple[Path, dict[str, int]]:
    try:
        with h5py.File(data_file, "r") as f:
            if VALID_TRAJECTORY_KEY in f:
                valid_trajs = f[VALID_TRAJECTORY_KEY][:].nonzero()[0].tolist()
                valid_traj_keys = [f"traj_{traj_idx}" for traj_idx in valid_trajs]
                traj_len_dict = {
                    traj_key: int(f[traj_key]["success"].shape[0]) for traj_key in valid_traj_keys
                }
            else:
                traj_len_dict = {}
    except OSError:
        traj_len_dict = {}
    return data_file, traj_len_dict


def build_trajectory_index(args, data_files: list[str]):
    data_root = Path(args.data_root)
    index_path = data_root / "valid_trajectory_index.json"
    if index_path.exists() and not args.overwrite:
        print("Trajectory index already exists, skipping")
        return

    submit_semaphore = Semaphore(args.num_workers * 4)
    lock = Lock()
    # {house_name: {datafile_path: {traj_key: traj_length}}}
    traj_index: dict[str, dict[str, dict[str, int]]] = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        with tqdm(total=len(data_files), desc="Building trajectory index...") as pbar:

            def on_done(future: Future):
                try:
                    future.result()
                except:
                    traceback.print_exc()
                    executor.shutdown(wait=False)
                    return

                with lock:
                    pbar.update(1)
                submit_semaphore.release()

            futures: list[Future] = []
            for data_file in data_files:
                submit_semaphore.acquire()
                future = executor.submit(read_valid_trajectories, Path(data_file))
                future.add_done_callback(on_done)
                futures.append(future)

            for future in as_completed(futures):
                data_file, valid_traj_dict = future.result()
                assert isinstance(data_file, Path)
                if not valid_traj_dict:
                    continue
                house_name = data_file.parent.name
                data_rel_path = data_file.relative_to(data_root)
                if house_name not in traj_index:
                    traj_index[house_name] = {}
                traj_index[house_name][str(data_rel_path)] = valid_traj_dict

    if not args.dry_run:
        with open(index_path, "w") as f:
            json.dump(traj_index, f, indent=2)


def main():
    args = get_args()

    print("Finding data files...")
    data_files = glob.glob(os.path.join(args.data_root, "**", "traj*.h5"), recursive=True)
    print(f"Found {len(data_files)} data files")

    if not args.index_only:
        find_valid_trajectories(args, data_files)

    build_trajectory_index(args, data_files)


if __name__ == "__main__":
    main()
