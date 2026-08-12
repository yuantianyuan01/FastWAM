import numpy as np
import os
import h5py
import multiprocessing
from tqdm import tqdm
import argparse
import subprocess

def extract_step_data(data):
    """Extract data from HDF5 group and convert it into a pickleable dictionary."""
    step_data = {
        'trajectory': np.array(data['trajectory']),
        'images': [np.array(data['observation']['rgb'][:, j]) for j in range(4)],  # Corrected to access images correctly
        'instruction': data['instruction'][0].decode('utf-8')
    }
    return step_data

def process_step(step_data):
    """Process a single step of extracted data and return the episode dictionary."""
    episode = []
    try:
        episode_length = step_data['trajectory'].shape[0]
        for i in range(episode_length):
            action = step_data['trajectory'][i][:-1] # only 7-dim action
            images = [step_data['images'][j][i] for j in range(4)]  # Ensures we access valid images only

            episode.append({
                'image_0': images[0],
                'image_1': images[1],
                'front': images[2],
                'wrist': images[3],
                'action': action,
                'language_instruction': step_data['instruction']
            })
    except Exception as e:
        print(f"Error processing step: {e}")
    return episode

def create_episode(file_path, save_dir, batch_size=10):
    # Open the HDF5 file and access its data
    with h5py.File(file_path, 'r') as hdf:
        data_group = hdf['data']
        time_stamps = list(data_group.keys())

        # Prepare the multiprocessing pool
        with multiprocessing.Pool() as pool:
            episodes = []
            for step, ts in enumerate(tqdm(time_stamps, desc="Processing steps")):
                # Extract data to make it pickleable
                try:
                    step_data = extract_step_data(data_group[ts])
                except Exception as e:
                    print(f"Error extracting step data: {e} at step {step}")
                    continue
                # Process each step in parallel
                episodes.append(pool.apply_async(process_step, (step_data,)))
                
                # Save in batches to reduce I/O
                if len(episodes) >= batch_size:
                    for idx, episode_result in enumerate(episodes):
                        episode_data = episode_result.get()  # Retrieve data from async result
                        if episode_data is None or len(episode_data) == 0:
                            print(f"skip {idx} episode")
                        save_path = os.path.join(save_dir, f"episode_{step - batch_size + idx + 1}.npy")
                        # if not os.path.exists(save_path):
                        np.save(save_path, episode_data)
                        print(f"Episode saved to {save_path}")
                    episodes = []  # Clear the batch
            
            # Save any remaining episodes
            for idx, episode_result in enumerate(episodes):
                episode_data = episode_result.get()
                save_path = os.path.join(save_dir, f"episode_{step - len(episodes) + idx + 1}.npy")
                np.save(save_path, episode_data)
                print(f"Episode saved to {save_path}")

def build_tfds_builder(src_path, dest_path, task_name):
    if not os.path.exists(src_path):
        print(f"Source file {src_path} does not exist.")
        return
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(src_path, 'r') as src_file:
        content = src_file.read()
        
    new_task_name = task_name.title().replace("_", "")
    content = content.replace("DemoBuilder", new_task_name)
    with open(dest_path, 'w') as dest_file:
        dest_file.write(content)
    
    print(f"Content copied from {src_path} to {dest_path}")

def copy_files(src_path, dest_path):
    if not os.path.exists(src_path):
        print(f"Source file {src_path} does not exist.")
        return
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(src_path, 'r') as src_file:
        content = src_file.read()
        
    with open(dest_path, 'w') as dest_file:
        dest_file.write(content)
    
    print(f"Content copied from {src_path} to {dest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='/remote-home1/sdzhang/datasets/train_dataset/Experiment-4')
    args = parser.parse_args()
    
    # Define the save directories
    builder_file = "VLABench/utils/multithread_rlds_builder.py"
    if args.task is None:
        tasks = os.listdir(args.save_dir)
        for task in tasks:
            target_builder_dir = os.path.join(args.save_dir, task)
            build_tfds_builder(builder_file, os.path.join(target_builder_dir, f"{task}.py"), task)
    else:
        target_builder_dir = os.path.join(args.save_dir, args.task)
        build_tfds_builder(builder_file, os.path.join(target_builder_dir, f"{args.task}.py"), args.task)
    
    # Create directories if not exist
    # os.makedirs(train_save_dir, exist_ok=True)
    # create_episode(h5py_file, train_save_dir, batch_size=10)
    
    # utils_to_cp = "vlabench_transfer/conversion_utils.py"
    # copy_files(utils_to_cp, os.path.join(target_builder_dir, "conversion_utils.py"))
    # print("successfully create dataset builder!")
    # print("start tfds build!")
    # result = subprocess.run(["tfds", "build"], cwd=target_builder_dir, capture_output=True, text=True)
    # print("convert done")