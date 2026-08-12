"""
data utils are designed for data generation and dataset processing
"""
from typing import List, Dict
import os
import h5py
import numpy as np
import json
from datetime import datetime
from VLABench.utils.utils import quaternion_to_euler

def process_observations(observations:List[Dict])->Dict:
    """
    Process the list of observations to a single dictionary
    """
    processed_observations = dict()
    for key in observations[0].keys():
        if key in ["point_cloud"]:
            processed_observations["point_cloud_points"] = list()
            processed_observations["point_cloud_colors"] = list()

        else:
            processed_observations[key] = list()
    processed_observations["trajectory"] = list()
    for observation in observations:
        for key in observation.keys():
            if key in ["point_cloud"]:
                processed_observations["point_cloud_points"].append(np.asarray(observation[key].points))
                processed_observations["point_cloud_colors"].append(np.asarray(observation[key].colors))
            else:
                processed_observations[key].append(observation[key])
    if processed_observations.get("point_cloud_points") is not None: # align the point cloud into array
        processed_observations["point_cloud_points"] = align_point_clouds(processed_observations["point_cloud_points"])
        processed_observations["point_cloud_colors"] = align_point_clouds(processed_observations["point_cloud_colors"])
    return processed_observations

def save_single_data(data:Dict, save_dir:str, filename:str, data_name:str=None):
    """
    Save data to h5py format
    """
    if data_name is None:
        data_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(os.path.join(save_dir))
    
    hdf5_file = h5py.File(os.path.join(save_dir, filename), "a")
    group = hdf5_file.get("data")
    if group is None:
        group = hdf5_file.create_group("data")
    data_group = group.create_group(data_name)
    obs_group = data_group.create_group("observation")
    info_group = data_group.create_group("meta_info")
    for key, buffer in data.items():
        if key in ["trajectory"]:
            buffer = np.array(buffer, dtype=np.float32)
            data_group.create_dataset(key, data=buffer, compression='gzip', compression_opts=9)
        elif isinstance(buffer, list) and isinstance(buffer[0], str):
            buffer = [x.encode('utf-8') for x in buffer]
            if key in ["instruction"]:
                data_group.create_dataset(key, data=np.array(buffer).astype("S"))
            else:
                info_group.create_dataset(key, data=np.array(buffer).astype("S"))
        elif isinstance(buffer, str):
            buffer = buffer.encode('utf-8')
            info_group.create_dataset(key, data=np.array(buffer).astype("S"))
        elif key in ["masked_point_cloud", "grasped_obj_name", "extrinsic", "instrinsic", "segmentation"]:
            continue
        else: # observation saving 
            try:
                buffer = np.array(buffer, dtype=np.float32) if key != "rgb" else np.array(buffer, dtype=np.uint8)
                obs_group.create_dataset(key, data=buffer, compression='gzip', compression_opts=9)
            except Exception as e:
                print(f"Error in saving {key}: {e}")
    hdf5_file.close()

def align_point_clouds(point_clouds:List[np.ndarray]):
    """
    Align the point clouds. e.g. [(10, 3), (11, 3)] -> [(11, 3), (11, 3)]
    """
    max_length = max([pc.shape[0] for pc in point_clouds])
    aligned_point_clouds = list()
    for pc in point_clouds:
        if len(pc) < max_length:
            aligned_point_clouds.append(np.vstack([pc, np.zeros((max_length - len(pc), 3))]))
        else:
            aligned_point_clouds.append(pc)
    return np.array(aligned_point_clouds)