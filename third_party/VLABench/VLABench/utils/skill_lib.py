"""
Skill Library for data generation.
"""
import numpy as np
import random
from VLABench.utils.utils import find_keypoint_and_prepare_grasp, distance, quaternion_to_euler, quaternion_from_axis_angle, quaternion_multiply
from VLABench.algorithms.motion_planning.rrt import rrt_motion_planning
from VLABench.algorithms.utils import interpolate_path, qauternion_slerp

PRIOR_EULERS = [[np.pi, 0, -np.pi/2], # face down, horizontal
                [np.pi, 0, 0], # face down, vertical
                [-np.pi/2, -np.pi/2, 0], # face forward, horizontal
                [-np.pi/2, 0, 0], # face forward, vertical
            ]

class SkillLib:
    @staticmethod
    def step_trajectory(env, 
                        points, 
                        quats, 
                        gripper_state, 
                        max_n_substep=1,
                        tolerance=0.01):
        """
        Universal step function for data generation.
        Input:
            env: LM4ManipEnv, for success detection
            points: np.array (n, 3), target positions
            quates: np.array(n, 4), target quaternions
            gripper_state: np.array(2), gripper state
            max_n_step: int, max number of substep for each step
            tolerance: float, tolerance for the qpos error between the target and the current qpos
        Return:
            observations: list of observations
            waypoints: list of waypoints
            stage_success: bool, whether the stage is successful
            task_success: bool, whether the task is successful
        """
        observations = []
        waypoints = []
        stage_success = False
        task_success = False
        for i, (point, quat) in enumerate(zip(points, quats)):
            success, action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=point, quat=quat)
            # if not success: # a wrong action beyond the embodied limit
            #     return None, None, False, False
            action = np.concatenate([action, gripper_state])
            waypoint = np.concatenate([point, quaternion_to_euler(quat), gripper_state])
            for _ in range(max_n_substep):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos - np.array(action[:7])) < tolerance \
                    and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                  break
            if task_success:
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(waypoint) 
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if distance(points[-1], env.robot.get_end_effector_pos(env.physics)) < tolerance:
            stage_success = True
        return observations, waypoints, stage_success, task_success
            
    @staticmethod
    def moveto(env, 
               target_pos,
               target_quat=None,
               target_velocity=0.05,
               gripper_state=None,
               **kwargs
               ):
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
        # env_pcd = observations[0]["masked_point_cloud"]
        # obstacle_pcd = np.asarray(env_pcd.points)

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)

        if target_quat is None:
            target_quat = start_quat
        motion_planning_path = rrt_motion_planning(tuple(start_pos), 
                                                tuple(target_pos), 
                                                obstacle_pcd)
        if motion_planning_path is None:
            motion_planning_path = [start_pos, target_pos]
        quats_in_path = []
        for t in np.linspace(0, 1, len(motion_planning_path), endpoint=False):
            quats_in_path.append(qauternion_slerp(start_quat, target_quat, t))
        interplate_path, interplate_quat = interpolate_path(np.array(motion_planning_path), 
                                                            np.array(quats_in_path), 
                                                            target_velocity)
        new_obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env, 
                                                                   interplate_path, 
                                                                   interplate_quat, 
                                                                   gripper_state,
                                                                   **kwargs)
        # if new_obs is None:
            # return None, None, False, False
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def pick(env, 
             target_entity_name, 
             target_pos=None, 
             target_quat=None, 
             prepare_distance=-0.1, 
             prepare_quat=None,
             prior_eulers=PRIOR_EULERS, 
             specific_keypoint=None,
             target_velocity=0.05,
             motion_planning_kwargs=dict(),
             **kwargs):
        """
        general pick function for data generation
        param:
            env: LM4manipEnv object
            target_entity_name: str, target entity name
            target_pos: np.array, target position. If None, will propose a target position automatically
            target_quat: np.array, target quaternion. If None, will propose a target quaternion automatically
            prepare_distance: float, distance to grasp point
            specific_keypoint: int, specific keypoint id, for multi keypoints such as drawer
        return: 
            observations: list of obs
            waypoints: list of waypoints
            key_frames: list of key action such as move to prepare point, grasp
        """
        target_entity = env.task.entities[target_entity_name]
        if target_pos is None or target_quat is None:
            key_pos, prepare_key_pos, key_quat = find_keypoint_and_prepare_grasp(env, target_entity, prior_eulers, specific_keypoint_id=specific_keypoint, move_vector=prepare_quat)
            if key_pos is None or prepare_key_pos is None:
                print("can not find valid keypoint and prepare point, reset the env")
                return None
        else:
            key_pos, key_quat = target_pos, target_quat
            if prepare_quat is None:
                gripper_pcd, move_quat = env.robot.gripper_pcd(key_pos, key_quat)
            else:
                move_quat = prepare_quat
            prepare_key_pos = key_pos + move_quat * prepare_distance
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        # env_pcd = env.get_observation()["masked_point_cloud"]
        # obstacle_pcd = np.asarray(env_pcd.points)

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)

        start_pos, start_quat, key_quat, prepare_pos, key_pos = np.array(start_pos), np.array(start_quat), np.array(key_quat), np.array(prepare_key_pos), np.array(key_pos)
        # motion planning -> path(start, prepare_point) & path(prepare_point, key_point)
        init2prepare_path = rrt_motion_planning(tuple(start_pos), 
                                                tuple(prepare_pos), 
                                                obstacle_pcd,
                                                **motion_planning_kwargs)
        if init2prepare_path is None:
            init2prepare_path = [start_pos, prepare_pos]
        quats_in_path = [start_quat for _ in range(len(init2prepare_path)-1)]
        quats_in_path.append(key_quat)
        # quats_in_path = []
        # for t in np.linspace(0, 1, len(init2prepare_path), endpoint=False):
        #     quats_in_path.append(qauternion_slerp(start_quat, key_quat, t))
        init2prepare_path.append(tuple(key_pos))
        path = np.array(init2prepare_path)
        quats_in_path.append(key_quat)
        
        interplate_path, interplate_quat = interpolate_path(path, quats_in_path, target_velocity)
        waypoints = []
        stage_success = False
        task_success = False
        observations = [env.get_observation()]
        # move along the interplated path
        gripper_state = np.ones(2) * 0.04
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env, 
                                                                   interplate_path, 
                                                                   interplate_quat, 
                                                                   gripper_state,
                                                                   **kwargs)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        if task_success:
            observations.pop(-1)
            return observations, waypoints, True, task_success
        # grasp
        new_obs, new_waypoints, _, task_success = SkillLib.close_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_entity_name].is_grasped(env.physics, env.robot):
            stage_success = True
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def place(env, 
              target_container_name, 
              target_pos=None, 
              target_quat=None,
              motion_planning_kwargs=dict()):
        """
        general place function for data generation
        param:
            env: LM4manipEnv object
            target_entity_name: str, target entity name
            target_pos: np.array, target position. If None, will propose a target position automatically
            target_quat: np.array, target quaternion. If None, will propose a target quaternion automatically
        return: 
            observations: list of obs
            waypoints: list of actions
            key_frame: list of key action such as move to prepare point, grasp
        """
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        if target_pos is None:
            place_points = target_container.get_place_point(env.physics)
            if not place_points:
                print("can not find valid place point, reset the env")
                return None
            if isinstance(place_points, list):
                place_point = random.choice(place_points)
            target_pos = place_point    
        if target_quat is None:
            # if no target_quat is provided, use the default quat
            target_quat = env.robot.get_end_effector_quat(env.physics)
        
        
        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
        # np.save("obstacle_pcd.npy", obstacle_pcd)
        
        start_pos, start_quat, target_pos, target_quat = np.array(start_pos), np.array(start_quat), np.array(target_pos), np.array(target_quat)
        #FIXME if can not find a path, consider change another algorithm
        #FIXME optimize the path with min margin to obstacles for safer moving
        init2target_path = rrt_motion_planning(tuple(start_pos), 
                                                tuple(target_pos), 
                                                obstacle_pcd,
                                                **motion_planning_kwargs)
        offset = env.robot.ee_offset(env.physics) # for avoid the collision
        if init2target_path is None:
            print("can not find a path to target position, use default lift")
            # default solution is lifting
            if start_pos[2] <= target_pos[2]:
                mid_point = np.array([start_pos[0], start_pos[1], target_pos[2]])
            else:
                mid_point = np.array([target_pos[0], target_pos[1], start_pos[2]])
        
            init2target_path = [start_pos, mid_point, target_pos]
        path = np.array(init2target_path)
        path += offset
        path_point_len = len(init2target_path)
        quats = [start_quat for _ in range(path_point_len)]
        quats[-1] = target_quat
        interplate_path, interplate_quat = interpolate_path(path, quats)   
        observations= [env.get_observation()]
        waypoints = []
        stage_success = True
        task_success = False
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env, 
                                                                interplate_path, 
                                                                interplate_quat, 
                                                                np.zeros(2))
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        if task_success:
            observations.pop(-1)
            assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
            return observations, waypoints, True, task_success
        # grasp
        new_obs, new_waypoints, _, task_success = SkillLib.open_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        
        observations.pop(-1)   
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        for entity in env.task.entities.values():
            if hasattr(entity, "is_grasped") and entity.is_grasped(env.physics, env.robot):
                stage_success = False
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def open_door(env, 
                  target_container_name):
        """
        Open the door of the target container
        Input:
            env: LM4manipEnv object
            target_container_name: str, target container name
        Return:
            observations: list of observations
            waypoints: list of waypoints
            trajectory: list of trajectory
            trajectory_quats: list of trajectory quaternions
        """
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        
        trajectory = target_container.get_open_trajectory(env.physics)
        trajectory_quats = []
        door_joint = target_container.door_joint
        rotation_axis = env.physics.bind(door_joint).xaxis
        # rotation_anchor = env.physics.bind(door_joint).xanchor
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        for i in range(len(trajectory)):
            rot_quat = quaternion_from_axis_angle(rotation_axis, -0.1*(i+1))
            new_quat = quaternion_multiply(start_quat, rot_quat)
            trajectory_quats.append(new_quat)
        # init_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                            interplate_path, 
                                                            interplate_quat, 
                                                            np.zeros(2))
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        pos, quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        for _ in range(10):
            action = np.concatenate([qpos, np.ones(2)*(0.04/10)*(i+1)])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([pos, quaternion_to_euler(quat), np.ones(2)*0.04]))
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_container_name].is_open(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def close_door(env, target_container_name, gripper_state=np.zeros(2)):
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        
        trajectory = target_container.get_close_trajectory(env.physics)
        trajectory_quats = [start_quat for _ in range(len(trajectory))]
        
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        if len(trajectory) == 0:
             return observations, waypoints, True, task_success
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                            interplate_path, 
                                                            interplate_quat, 
                                                            gripper_state)
        observations.extend(obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_container_name].is_closed(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def open_drawer(env, 
                    target_container_name, 
                    pick_prior_eulers=[[-np.pi/2, 0, 0]],
                    drawer_id=0):
        """
        common open drawer function
        Input:
            drawer_id: 0-2 means top to bottom drawer.
            pick_prioer_euelrs: list of list, prior eulers for pick
        """
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        target_container = env.task.entities[target_container_name]
        # grasp handle
        new_obs, new_waypoints, _, success_ = SkillLib.pick(env, target_container_name, prior_eulers=pick_prior_eulers, specific_keypoint=drawer_id)
        
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        task_success = task_success or success_
        # open drawer   
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        trajectory = target_container.get_drawer_open_trajectory(env.physics, drawer_id)
        trajectory_quats = [start_quat for _ in range(len(trajectory))]
        trajectory, trajectory_quats = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, success_ = SkillLib.step_trajectory(env, trajectory, trajectory_quats, np.zeros(2))
        
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        task_success = task_success or success_
        observations.pop(-1)
        
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        # TODO check the drawer state
        return observations, waypoints, True, task_success
    
    @staticmethod
    def press(env, target_pos, target_quat=None, move_vector=[0, 0, 0.1], max_n_substep=100): #TODO move vector to determine the press direction
        prepare_pos = target_pos + np.array(move_vector) if move_vector is not None else target_pos
        observations, waypoints, _, _ = SkillLib.moveto(env, 
                                                     prepare_pos, 
                                                     target_quat,
                                                     max_n_substep=max_n_substep)
        # close gripper
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        for i in range(10):
            gripper_state = np.ones(2) * (0.04 - i/10 * 0.04)
            action = np.concatenate([qpos, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        new_obs, new_waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos, target_quat, max_n_substep=max_n_substep)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def pull(env, target_pos=None, target_quat=None, gripper_state=None, pull_distance=0.3):
        """
        Common pull function.
        """
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        if target_pos is None: target_pos = np.array(start_pos) + np.array([0, -pull_distance, 0])
        if target_quat is None: target_quat = start_quat
        interplate_path, interplate_quat = interpolate_path([start_pos, target_pos], [np.array(start_quat), np.array(target_quat)])
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
        new_obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env, 
                                                           interplate_path, 
                                                           interplate_quat, 
                                                           gripper_state)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def push(env, target_pos=None, target_quat=None, gripper_state=None, push_distance=0.3):
        obs, waypoints, stage_success, task_success = SkillLib.pull(env, target_pos, target_quat, gripper_state, -push_distance)
        return obs, waypoints, stage_success, task_success
    
    @staticmethod
    def pour(env, target_delta_qpos=np.pi, target_q_velocity=np.pi/40, n_repeat_step=2, tolerance=0.01):
        """
        Common pour function.
        """
        waypoints = []
        stage_success = False
        task_success = False
        observations = [env.get_observation()]
        
        init_qpos = np.array(env.robot.get_qpos(env.physics))
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_closed: gripper_state = np.zeros(2)
        else: gripper_state = np.ones(2) * 0.04
        timesteps = int(target_delta_qpos / target_q_velocity)
        for i in range(timesteps):
            action = np.array(init_qpos).reshape(-1)
            action[-1] += target_q_velocity * i
            action = np.concatenate([action, gripper_state])
            for _ in range(n_repeat_step):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos - np.array(action[:7])) < tolerance \
                    and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                  break
            waypoint = np.concatenate([env.robot.get_end_effector_pos(env.physics), 
                                       quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)), 
                                       gripper_state])
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(waypoint)
            if task_success:
                break
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success
        
    @staticmethod
    def lift(env, target_pos=None, target_quat=None, gripper_state=None, lift_height=0.3):
        """
        Common lift function.
        """
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        if target_pos is None: 
            target_pos = np.array(start_pos) + np.array([0, 0, lift_height])
        if target_quat is None: 
            target_quat = start_quat
        interplate_path, interplate_quat = interpolate_path([start_pos, target_pos], [np.array(start_quat), np.array(target_quat)])
        observations = [env.get_observation()]
        waypoints = []
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
        obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env, 
                                                           interplate_path, 
                                                           interplate_quat, 
                                                           gripper_state)
        observations.extend(obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def reset(env, max_n_substep=200, tolerance=0.01):
        init_qpos = env.task.robot.default_qpos
        observations = [env.get_observation()]
        waypoints = []
        for _ in range(max_n_substep):
            action = np.array(init_qpos)
            env.step(action)
            obs = env.get_observation()
            pos, euler = np.array(env.robot.get_end_effector_pos(env.physics)), quaternion_to_euler(env.robot.get_end_effector_quat(env.physics))
            waypoint = np.concatenate([pos, euler, np.ones(2)*0.04])
            observations.append(obs)
            waypoints.append(waypoint)
            current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
            if np.max(current_qpos - np.array(action[:7])) < tolerance \
                and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                break
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, False
        
    
    @staticmethod
    def close_gripper(env, repeat=1):
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        observations = [env.get_observation()]
        waypoints = []
        success = False
        for i in range(10):
            gripper_state = np.ones(2) * (0.04 - i * 0.04/10)
            action = np.concatenate([qpos, gripper_state])
            for _ in range(repeat):
                timestep = env.step(action)
                if timestep.last():
                    success = True
                    obs = env.get_observation()
                    observations.append(obs)
                    waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
                    break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
            
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, success
    
    @staticmethod
    def open_gripper(env, repeat=1):
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        stage_success = False
        for i in range(10):
            gripper_state = np.ones(2) * (i+1)/10 * 0.04
            action = np.concatenate([qpos, gripper_state])
            for _ in range(repeat):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    obs = env.get_observation()
                    observations.append(obs)
                    waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
            if timestep.last():
                task_success = True
                break
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.robot.get_ee_open_state(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def flip(env, gripper_state=None, target_q_velocity=np.pi/40, max_n_substep=30, tolerance=0.01):
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        observations = [env.get_observation()]
        waypoints = []
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
        timestep = int(np.pi / target_q_velocity)
        success = False
        for i in range(timestep):
            action = np.array(qpos).copy()            
            action[-1] += target_q_velocity * i
            action = np.concatenate([action, gripper_state])
            for _ in range(max_n_substep):
                timestep = env.step(action)
                if timestep.last():
                    success = True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos - np.array(action[:7])) < tolerance \
                    and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                  break
            if success:
                break
            waypoint = np.concatenate([env.robot.get_end_effector_pos(env.physics), 
                                       quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)), 
                                       gripper_state])
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(waypoint)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, success
    
    @staticmethod
    def open_laptop(env, target_entity_name):
        laptop = env.task.entities[target_entity_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        
        trajectory = laptop.get_open_trajectory(env.physics)
        trajectory_quats = []
        screen_joint = laptop.screen_joint
        rotation_axis = env.physics.bind(screen_joint).xaxis
        # rotation_anchor = env.physics.bind(door_joint).xanchor
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        for i in range(len(trajectory)):
            rot_quat = quaternion_from_axis_angle(rotation_axis, 0.04*(i+1))
            new_quat = quaternion_multiply(start_quat, rot_quat)
            trajectory_quats.append(new_quat)
        # init_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                            interplate_path, 
                                                            interplate_quat, 
                                                            np.zeros(2))
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        if task_success:
            observations.pop(-1)
            assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
            return observations, waypoints, True, task_success
        
        new_obs, new_waypoints, _, task_success = SkillLib.open_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_entity_name].is_open(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def wait(env, wait_time=100, gripper_state=None):
        print(f"wait for {wait_time} steps")
        current_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        for _ in range(wait_time):
            action = np.concatenate([current_qpos, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success

    @staticmethod
    def move_offset(env, offset, target_quat=None, gripper_state=None):
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        target_pos = np.array(start_pos) + np.array(offset)
        if target_quat is None: target_quat = start_quat
        observations, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos, target_quat, gripper_state=gripper_state)
        return observations, waypoints, stage_success, task_success