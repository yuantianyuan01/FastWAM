import json
import os
import numpy as np
import random
import mediapy
import traceback
from tqdm import tqdm
from VLABench.envs import load_env
from VLABench.configs import name2config
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler, find_key_by_value

class Evaluator:
    def __init__(self, 
                 tasks,
                 n_episodes,
                 episode_config=None,
                 max_substeps=1,
                 tolerance=1e-2,
                 metrics=["success_rate"],
                 save_dir=None,
                 visulization=False,
                 eval_unseen=False,
                 unnorm_key='primitive',
                 **kwargs
                 ):
        """
        Basic evaluator of policy
        params:
            tasks: list of task names to evaluate, e.g. ["task1", "task2"]
            n_episodes: number of episodes to evaluate in each task
            episode_config: dict or path of config file for episode generation
            max_substeps: maximum number of substeps for env.step
            metrics: list of metrics to evaluate
            save_dir: directory to save the evaluation results
            visulization: whether to visualize the evaluation progress as videos
            eval_unseen: whether to evaluate the unseen object categories
            unnorm_key: the dataset statistics name of the task suite
        """
        if isinstance(episode_config, str):
            with open(episode_config, "r") as f:
                self.episode_config = json.load(f)
        else:self.episode_config = episode_config
        if self.episode_config is None:
            print("Load the task episodes by seeds, instead of episodes")
        else:
            for task in tasks:
                assert len(self.episode_config[task]) >= n_episodes, "The number of episodes should be less than the number of configurations"
        self.eval_tasks = tasks
        self.n_episodes = n_episodes 
        
        self.max_substeps = max_substeps
        self.tolerance = tolerance
        self.target_metrics = metrics
        self.intention_score_threshold = kwargs.get("intention_score_threshold", 0.1)
        self.eval_unseen = eval_unseen
        self.unnorm_key = unnorm_key
        # log, store and visualization
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.visulization = visulization
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task_config.json"), "r") as f:
           self.task_configs = json.load(f)
        
    def evaluate(self, agent):
        """
        Evaluate the agent on all tasks defined in the evaluator.
        """   
        metrics = {}
        for task in self.eval_tasks:
            task_infos = []
            max_episode_length = 200
            if self.task_configs.get(find_key_by_value(name2config, task), None):
                if self.task_configs[find_key_by_value(name2config, task)].get("evaluation", {}).get("max_episode_length", None):
                    max_episode_length = self.task_configs[find_key_by_value(name2config, task)]["evaluation"]["max_episode_length"]
                
            for i in tqdm(range(self.n_episodes), desc=f"Evaluating {task} of {agent.name}"):
                agent.reset()
                kwargs = {
                    "unnorm_key": 'primitive',
                    "max_episode_length": max_episode_length
                }
                try:
                    # BUG some episodes are unstable and lead to crash
                    if self.episode_config is None: 
                        info = self.evaluate_single_episode(agent, task, i, None, seed=42+i, **kwargs)
                    else: 
                        info = self.evaluate_single_episode(agent, task, i, self.episode_config[task][i], **kwargs)
                    task_infos.append(info)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    
            metric_score = self.compute_metric(task_infos)       
            metrics[task] = metric_score
            
            if self.save_dir is not None:
                if os.path.exists(os.path.join(self.save_dir, "metrics.json")):
                    with open(os.path.join(self.save_dir, "metrics.json"), "r") as f:
                        previous_metrics = json.load(f)
                else:
                    previous_metrics = {}
                previous_metrics.update(metrics)
                with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
                    json.dump(previous_metrics, f, indent=4)
                os.makedirs(os.path.join(self.save_dir, task), exist_ok=True)
                with open(os.path.join(self.save_dir, task, f"detail_info.json"), "w") as f:
                    json.dump(task_infos, f, indent=4)
        return metrics
        
    def evaluate_single_episode(self, agent, task_name, episode_id, episode_config, seed=42, max_episode_length=200, **kwargs):
        """
        If episode_config is given, the task and scene will load deterministically.
        params:
            agent: policy to evaluate
            task_name: name of the task
            episode_id: id of the episode
            episode_config: configuration of the task
            seed: seed for the random number generator, if episode_config is None
            max_episode_length: maximum length of the episode
        """
        if episode_config is None: # use random seed to ditermine the task
            np.random.seed(seed)
            random.seed(seed)
        if episode_config is not None:
            env = load_env(task_name, episode_config=episode_config, random_init=False, eval=self.eval_unseen, run_mode="eval")
        else:
            env = load_env(task_name, random_init=True, eval=self.eval_unseen, run_mode="eval")
        env.reset()
        success = False
        info = {}
        frames_to_save = []
        last_action = None
        i = 0
        robot_frame = env.get_robot_frame_position()
        while i < max_episode_length:
            observation = env.get_observation(require_pcd=False)
            observation["instruction"] = env.task.get_instruction()
            ee_state = observation["ee_state"]
            observation['robot_frame'] = robot_frame
            if last_action is None:
                last_action = np.concatenate([ee_state[:3], quaternion_to_euler(ee_state[3:7])])
            observation["last_action"] = last_action
            if self.save_dir is not None and self.visulization:
                frames_to_save.append(observation["rgb"])
            if agent.control_mode == "ee":
                pos, euler, gripper_state = agent.predict(observation, **kwargs)
                last_action = np.concatenate([pos, euler])
                quat = euler_to_quaternion(*euler)
                _, action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=quat)
                action = np.concatenate([action, gripper_state])
            elif agent.control_mode == "joint":
                qpos, gripper_state = agent.predict(observation, **kwargs)
                action = np.concatenate([qpos, gripper_state])
            else:
                raise NotImplementedError(f"Control mode {agent.control_mode} is not implemented")    
            for _ in range(self.max_substeps):
                timestep = env.step(action)
                if timestep.last():
                    success=True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos-np.array(action)[:7]) < self.tolerance \
                    and np.min(current_qpos - np.array(action)[:7]) > -self.tolerance:
                    break
            if success:
                break
            i += 1
        intention_score =  env.get_intention_score(threshold=self.intention_score_threshold)
        progress_score = env.get_task_progress()
        info["task"] = task_name
        info["success"] = success
        info["consumed_step"] = i
        info["intention_score"] = intention_score
        info["progress_score"] = progress_score
        
        env.close()
        if self.save_dir is not None and self.visulization:
            os.makedirs(os.path.join(self.save_dir, task_name, "videos"), exist_ok=True)
            self.save_video(frames_to_save, os.path.join(self.save_dir, task_name, "videos", f"{episode_id}_success_{str(success)}_progress_{progress_score:.2f}.mp4"))
        return info
        
    def compute_metric(self, infos):
        """
        Compute the metric scores for the evaluation
        param:
            infos: list of episode information
        """
        metric = {}
        for key in self.target_metrics:
            if key == "success_rate": # compute the success rate
                success = [info["success"] for info in infos]
                sucess_rate = np.mean(success)
                metric["success_rate"] = sucess_rate
            elif key == "intention_score":
                intention_score = [info["intention_score"] for info in infos]
                avg_intention_score = np.mean(intention_score)
                metric["intention_score"] = avg_intention_score
            elif key == "progress_score":
                progress_score = [info["progress_score"] for info in infos]
                avg_progress_score = np.mean(progress_score)
                metric["progress_score"] = avg_progress_score
            else:
                raise NotImplementedError(f"Metric {key} is not implemented")
        return metric
    
    def save_video(self, frames, save_dir):
        frames_to_save = [] 
        for frame in frames:
            frames_to_save.append(np.vstack([np.hstack(frame[:2]), np.hstack(frame[2:4])]))
        mediapy.write_video(save_dir, 
                            frames_to_save, fps=10) 