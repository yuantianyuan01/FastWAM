import argparse
from concurrent.futures import CancelledError, ProcessPoolExecutor, Future, as_completed
from threading import Semaphore, Lock
from pathlib import Path
import glob
import os
import traceback

import numpy as np
import h5py
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
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
        help="Find valid trajectories but do not write to data files",
    )
    return parser.parse_args()


def process_data_file(args, data_file_path: Path):
    n_updated = 0
    n_skipped = 0
    try:
        with h5py.File(data_file_path, "r" if args.dry_run else "r+") as f:
            for traj_name in f.keys():
                if not traj_name.startswith("traj_"):
                    continue
                traj_idx = int(traj_name.split("_")[-1])
                sensor_data_group = f[traj_name]["obs/sensor_data"]

                if len(sensor_data_group.keys()) > 0 and not args.overwrite:
                    n_skipped += 1
                    continue

                n_updated += 1
                batch_suffix = data_file_path.name.split("_", 1)[1][: -len(".h5")]

                for camera_name in f[traj_name]["obs/sensor_param"].keys():
                    # Process RGB video
                    if camera_name not in sensor_data_group:
                        video_filename = f"episode_{traj_idx:08d}_{camera_name}_{batch_suffix}.mp4"
                        video_path = str(data_file_path.parent / video_filename)
                        assert os.path.exists(video_path), f"Video path {video_path} does not exist"
                        video_filename_bytes = video_filename.encode("utf-8")
                        byte_array = np.zeros(100, dtype=np.uint8)
                        byte_array[: len(video_filename_bytes)] = list(video_filename_bytes)
                        if not args.dry_run:
                            sensor_data_group.create_dataset(
                                camera_name, data=byte_array, dtype=np.uint8
                            )

                    # Process depth video if it exists
                    depth_camera_name = f"{camera_name}_depth"
                    if depth_camera_name not in sensor_data_group:
                        depth_video_filename = (
                            f"episode_{traj_idx:08d}_{depth_camera_name}_{batch_suffix}.mp4"
                        )
                        depth_video_path = str(data_file_path.parent / depth_video_filename)
                        if os.path.exists(depth_video_path):
                            depth_video_filename_bytes = depth_video_filename.encode("utf-8")
                            depth_byte_array = np.zeros(100, dtype=np.uint8)
                            depth_byte_array[: len(depth_video_filename_bytes)] = list(
                                depth_video_filename_bytes
                            )
                            if not args.dry_run:
                                sensor_data_group.create_dataset(
                                    depth_camera_name, data=depth_byte_array, dtype=np.uint8
                                )
    except OSError:
        return n_updated, n_skipped, 1
    except Exception as e:
        raise RuntimeError(f"Error processing data file {data_file_path}: {e}") from e
    return n_updated, n_skipped, 0


def main():
    args = get_args()

    print("Finding data files...")
    data_files = glob.glob(os.path.join(args.data_root, "**", "traj*.h5"), recursive=True)
    print(f"Found {len(data_files)} data files")
    total_n_updated = 0
    total_n_skipped = 0
    total_n_corrupted_files = 0
    if args.num_workers > 1:
        # use a semaphore to limit the number of queued jobs, helps with large quantity of datafiles
        submit_semaphore = Semaphore(args.num_workers * 4)
        lock = Lock()
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            with tqdm(total=len(data_files), desc="Processing files...") as pbar:

                def on_done(future: Future[tuple[int, int, int]]):
                    try:
                        n_updated, n_skipped, n_corrupted_files = future.result()
                    except CancelledError:
                        pass
                    except:
                        traceback.print_exc()
                        executor.shutdown(wait=False, cancel_futures=True)

                    with lock:
                        nonlocal total_n_updated, total_n_skipped, total_n_corrupted_files
                        total_n_updated += n_updated
                        total_n_skipped += n_skipped
                        total_n_corrupted_files += n_corrupted_files
                        pbar.set_postfix(
                            n_traj_updated=total_n_updated,
                            n_traj_skipped=total_n_skipped,
                            n_corrupted_files=total_n_corrupted_files,
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
            n_updated, n_skipped, n_corrupted_files = process_data_file(args, Path(data_file))
            total_n_updated += n_updated
            total_n_skipped += n_skipped
            total_n_corrupted_files += n_corrupted_files
            pbar.set_postfix(
                n_traj_updated=total_n_updated,
                n_traj_skipped=total_n_skipped,
                n_corrupted_files=total_n_corrupted_files,
            )

    dry_run_str = "would be " if args.dry_run else ""
    print(
        f"Finished processing {len(data_files)} data files, "
        f"{total_n_updated} trajectories {dry_run_str}updated, "
        f"{total_n_skipped} trajectories {dry_run_str}skipped, "
        f"found {total_n_corrupted_files} corrupted files"
    )


if __name__ == "__main__":
    main()
