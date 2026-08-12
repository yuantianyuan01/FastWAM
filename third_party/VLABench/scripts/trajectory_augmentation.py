"""
This script is to augment trajectory data with replaying and replacing. 
"""
import argparse
import h5py
import mediapy
from tqdm import tqdm
from VLABench.tasks import *
from VLABench.robots import *
from VLABench.envs import load_env
from VLABench.utils.data_utils import save_single_data, process_observations
from VLABench.utils.camera_utils import translate_camera_keep_target, orbital_camera_movement

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--argment-choice", nargs="+", default="camera_view", help="The argmentation dimensions")
    parser.add_argument("--origin-dataset", type=str, default="/remote-home1/sdzhang/datasets/OpenRT/vlabench_task/primitive/select_poker")
    parser.add_argument("--save-dir", type=str, default="/remote-home1/sdzhang/datasets/OpenRT/vlabench_task/camera_augment/primitive")
    parser.add_argument("--replay-mode", type=str, default="eef", choices=["eef", "joint"], help="Control mode of eef")
    parser.add_argument("--camera-aug-file", type=str, default="VLABench/configs/camera/front_camera_augmentation.json", help="Augmentation parameters for augmented cameras")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--record-video", action="store_true", default="whether to record the replay videos")
    parser.add_argument("--start-ratio", type=float, default=0, help="Start point in percentage")
    parser.add_argument("--ratio", type=float, default=0.1, help="Data ratio to replay.")
    parser.add_argument("--process-id", type=int, default=1, help="The index of the data replaying process")
    args = parser.parse_args()
    return args

def get_all_hdf5_files(directory):
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files

def augment_camera_view(env, **kwargs):
    "Currently only augment the front camera for the baseline experiments"
    cameras = env.task._arena.mjcf_model.find_all("camera")
    base_camera_indice = kwargs.get("base_camera_id", 0) # default is the front camera
    base_pos, base_xyaxes = cameras[base_camera_indice].pos, cameras[base_camera_indice].xyaxes
    camera_transform_params = kwargs.get("transform_params", None)
    assert isinstance(camera_transform_params, list) 
    for i, param in enumerate(camera_transform_params): 
        if i >= len(cameras):
            break
        camera_pos, camera_xyaxes = translate_camera_keep_target(base_pos, base_xyaxes, translation=param["translation"], target_distance=param["distance"])
        camera_pos, camera_xyaxes = orbital_camera_movement(camera_pos, camera_xyaxes, angle=param["orbital_angle"], axis=param["orbital_axis"], target_distance=param["distance"])
        for attr, value in zip(['pos', 'xyaxes', 'fovy'], [camera_pos, camera_xyaxes, param['fovy']]):
            setattr(cameras[i], attr, value)
        

def load_episode_data(episode_path):
    with h5py.File(episode_path, 'r') as f:
        for timestamp in f["data"].keys():
            ee_state = np.asarray(f["data"][timestamp]["observation"]["ee_state"]) # (n ,8)
            q_state = np.asarray(f["data"][timestamp]["observation"]["q_state"]).reshape(-1, 7) # (n, 7, 1)
            actions = np.asarray(f["data"][timestamp]["trajectory"]).reshape(-1, 8) # (n, 8, 1)
            
            instruction_bytes = np.asarray(f["data"][timestamp]["instruction"]).astype("S")
            instruction = instruction_bytes.item().decode('utf-8')
            # instruction = json.loads(instruction)
            
            gripper_state = ee_state[:, -1:].reshape(-1, 1)
            gripper_state = np.where(gripper_state > 0, 0.0, 0.04)
            gripper_state = np.hstack([gripper_state, gripper_state])
            joints = np.concatenate((q_state, gripper_state), axis=1)
            
            episode_config_bytes = np.asarray(f["data"][timestamp]["meta_info"]["episode_config"]).astype('S')
            episode_config = episode_config_bytes.item().decode('utf-8')
            episode_config = json.loads(episode_config)
            
            gripper_state = actions[:, -2:]
            
            # delta action 
            delta_action = actions[1:, :6] - actions[:-1, :6]
            first_action = actions[0, :6] - np.array([0, 0.2416, 0.46582, np.pi, 0.02, -1.6077])
            delta_actions = np.concatenate([first_action.reshape(1, -1), delta_action], axis=0)
            delta_actions = np.concatenate([delta_actions, gripper_state], axis=1)
            
            return joints, actions, delta_actions, ee_state, episode_config, instruction

def augment_trajectory(episode_path, 
                       replay_mode, 
                       save_dir, 
                       augment_choices,
                       camera_augment_config_file, 
                       record_video=False
                       ):
    
    joints, actions, delta_actions, ee_states, episode_config, instruction = load_episode_data(episode_path)    
    task = episode_path.split("/")[-2] # test here
    save_dir = os.path.join(save_dir, task)
    filename = episode_path.split("/")[-1]
    if os.path.exists(os.path.join(save_dir, filename)):
        return
    # load the tasks and environments in different ways
    if "camera_view" in augment_choices:
        env = load_env(task, reset_wait_step=0, episode_config=episode_config, random_init=False, xml_file="base/camera_augment_env.xml")
        with open(camera_augment_config_file, "r") as f:
            transform_params = json.load(f)
        augment_camera_view(env, transform_params=transform_params)
    env.reset()
    
    robot_position = env.robot.robot_config["position"]
    observations = []
    if replay_mode == "joint":
        for joint in joints:
            obs = env.get_observation()
            observations.append(obs)
            env.step(joint)
    
    elif replay_mode == "eef":
        for action in actions:
            obs = env.get_observation()
            observations.append(obs)
            point, euler, gripper_state = action[:3], action[3:6], action[-2:]
            point += robot_position # robot frame to world frame
            quat = euler_to_quaternion(*euler)
            success, qpos = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=point, quat=quat)
            joint = np.concatenate([qpos, gripper_state])
            env.step(joint)            
    else:
        raise ValueError(f"{replay_mode} is not a supported control mode!")
    
    camera_extrinsic = []
    camera_instrinsic = []
    for i in range(env.physics.model.ncam):
        instrinsic, extrinsic = env.get_camera_matrix(cam_id=i, width=480, height=480)
        camera_extrinsic.append(extrinsic)
        camera_instrinsic.append(instrinsic)
    data_to_save = process_observations(observations)
    data_to_save["trajectory"] = actions
    data_to_save["instruction"] = instruction
    data_to_save["episode_config"] = json.dumps(episode_config)
    data_to_save["camera_extrinsic"] = np.array(camera_extrinsic)
    data_to_save["camera_instrinsic"] = np.array(camera_instrinsic)
    save_single_data(data=data_to_save,
                     save_dir=save_dir, 
                     filename=filename,
                     )
    env.close()
    if record_video:
        frames = []
        for o in observations:
            frames.append(np.vstack([np.hstack(o["rgb"][:3]), np.hstack(o["rgb"][3:6]), np.hstack(o["rgb"][6:9])]))
        os.makedirs(save_dir, exist_ok=True)
        mediapy.write_video(os.path.join(save_dir, f"{filename.split('.')[0]}.mp4"), frames, fps=10)
        
    
if __name__ == "__main__":
    args = get_args()
    h5_files = get_all_hdf5_files(args.origin_dataset)
    h5_files.sort()
    start_index, end_index = int(len(h5_files) * args.start_ratio), int(len(h5_files) * (args.start_ratio + args.ratio))
    for h5_file in tqdm(h5_files[start_index:end_index], desc=f"Process {args.process_id} replaying data..."):
        try:
            augment_trajectory(h5_file, args.replay_mode, args.save_dir, args.argment_choice, args.camera_aug_file, args.record_video)
        except:
            pass
        if args.debug:
            print("Program exits in debug mode.")
            break