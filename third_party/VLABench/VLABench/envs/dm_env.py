import numpy as np
import math
from dm_control import composer
from VLABench.utils.utils import euler_to_quaternion, expand_mask
from VLABench.utils.depth2cloud import rotMatList2NPRotMat, quat2Mat, posRotMat2Mat, PointCloudGenerator

class LM4ManipDMEnv(composer.Environment):
    def __init__(self, reset_wait_step=10, **kwargs):
        super().__init__(**kwargs)
        self.timestep = 0
        self.reset_wait_step = reset_wait_step
        self.n_distractor = kwargs.get("n_distractor", 0)
        self.render_options = dict(
            height=480,
            width=480,
        )
        self.register_pcd_generator()
        
    def reset(self):
        self.timestep = 0
        timestep = super().reset()
        self.cancel_gravity_and_improve_fluid()
        for i in range(self.reset_wait_step):
            # print(f"reset {i}")
            self.step()
        self.reset_gravity_and_fluid(self.physics)
        for _ in range(self.reset_wait_step):
            self.step()
        self.register_pcd_generator()
        return timestep
        
    def render(self, **kwargs):
        return self.physics.render(**kwargs)
    
    def step(self, action=None):
        if action is None: # robot stay in static
            action = self.robot.get_qpos(self.physics)
            action = np.concatenate([action, 0.04 * np.ones((2))], axis=-1)
            return super().step(action)
        self.timestep += 1
        return super().step(action)
    
    @property
    def mjmodel(self):
        return self.physics.model
    
    @property
    def mjdata(self):
        return self.physics.data

    @property
    def robot(self):
        return self.task.robot
    
    def get_element_by_name(self, name, type):
        if type == "camera":
            element = self.mjmodel.cam(name)
        else:
            element = self.task.get_element_by_name(name, type)
        return element
    
    def get_xpos_by_name(self, name, type="geom"):
        """
        get the position of the entity by name
        """
        element = self.get_element_by_name(name, type)
        data = self.physics.bind(element)
        return data.xpos
    
    def get_xquat_by_name(self, name, type="geom"):
        """
        get the orientation of the entity by name
        """
        element = self.get_element_by_name(name, type)
        data = self.physics.bind(element)
        return data.xquat
    
    def set_xpos_by_name(self, name, type, pos):
        element = self.get_element_by_name(name, type)
        assert len(pos) == 3 or pos.shape == (3, 0), "pos must be a 3D vector"
        element.pos = pos
        self.step()
        
    
    def set_xquat_by_name(self, name, type, quat=None, euler=None):
        obj = self.get_element_by_name(name, type)
        assert (quat is None) != (euler is None), "quat and euler must be provided exclusively"
        if euler is not None:
            quat = euler_to_quaternion(roll=euler[0], pitch=euler[1], yaw=euler[2])
        obj.quat = quat
        self.step()
    
    def attach_entity(self, entity):
        self.task.add_free_entity(entity)
        self.step()
    
    def get_ee_pos(self):
        return self.robot.get_end_effector_pos(self.physics)
    
    def get_ee_quat(self):
        return self.robot.get_end_effector_quat(self.physics)
    
    def get_camera_matrix(self, cam_id, width, height):
        if cam_id >= self.physics.model.ncam:
            raise ValueError(f"cam_id {cam_id} is out of range")
        fovy = math.radians(self.physics.model.cam_fovy[cam_id])
        f = height / (2 * math.tan(fovy / 2))
        intrinsic_mat = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        
        cam_pos = self.physics.data.cam_xpos[cam_id]
        c2b_r = rotMatList2NPRotMat(self.physics.model.cam_mat0[cam_id])
        b2w_r = quat2Mat([0, 1, 0, 0])
        cam_rot_mat = np.matmul(c2b_r, b2w_r)
        extrinsic_mat = posRotMat2Mat(cam_pos, cam_rot_mat)
        
        return intrinsic_mat, extrinsic_mat

    def get_observation(self, require_pcd=True):
        observation = dict()
        multi_view_rgb, multi_view_depth, multi_view_seg = [], [], []
        instrinsic_matrixs, extrinsic_matrixs = [], []
        for cam in range(self.physics.model.ncam):
            multi_view_rgb.append(self.render(camera_id=cam, **self.render_options))
            multi_view_depth.append(self.render(camera_id=cam, **self.render_options, depth=True))
            multi_view_seg.append(self.render(camera_id=cam, **self.render_options, segmentation=True))
            instrinsic, extrinsic = self.get_camera_matrix(cam, **self.render_options)
            instrinsic_matrixs.append(instrinsic)
            extrinsic_matrixs.append(extrinsic)
        observation["q_state"] = np.array(self.robot.get_qpos(self.physics))
        observation["q_velocity"] = np.array(self.robot.get_qvel(self.physics))
        observation["q_acceleration"] = np.array(self.robot.get_qacc(self.physics))
        observation["rgb"] = np.array(multi_view_rgb)
        observation["depth"] = np.array(multi_view_depth)
        observation["segmentation"] = np.array(multi_view_seg)
        observation["robot_mask"] = np.where((observation["segmentation"][..., 0] <= 72)&(observation["segmentation"][..., 0] > 0), 0, 1).astype(np.uint8)
        observation["instrinsic"] = np.array(instrinsic_matrixs)
        observation["extrinsic"] = np.array(extrinsic_matrixs)
        self.pcd_generator.physics = self.physics
        if require_pcd:
            observation["masked_point_cloud"] = self.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(self.physics.model.ncam - 1)), 
                                                                                            rgb=multi_view_rgb,
                                                                                            depth=multi_view_depth,
                                                                                            mask=expand_mask(observation["robot_mask"]))
            observation["point_cloud"] = self.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(self.physics.model.ncam - 1)),
                                                                                    rgb=multi_view_rgb,
                                                                                    depth=multi_view_depth)
        observation["ee_state"] = self.robot.get_ee_state(self.physics)
        observation["grasped_obj_name"] = self.get_grasped_entity()
        observation.update(self.task.task_observables)
        return observation

    
    def cancel_gravity_and_improve_fluid(self):
        """
        This function is used when initializing the scene.
        """
        self.task._arena.mjcf_model.option.flag.gravity = "disable"
        geoms = self.task._arena.mjcf_model.find_all("geom")
        for geom in geoms:
            geom.fluidshape = "ellipsoid"
            geom.fluidcoef = [1e4, 1e4, 1e4, 1e4, 1e4]
    
    def reset_gravity_and_fluid(self, physics):
        self.task._arena.mjcf_model.option.flag.gravity = "enable"
        geoms = self.task._arena.mjcf_model.find_all("geom")
        for geom in geoms:
            geom.fluidshape = "none"
            geom.fluidcoef = [0.5, 0.25, 1.5, 1.0, 1.0]
            physics.data.qvel = 0
    
    def register_pcd_generator(self):
        self.pcd_generator = PointCloudGenerator(self.physics, 
                                                 min_bound=[-1, -1, 0.7], 
                                                 max_bound=[1, 1, 2],
                                                 **self.render_options)
    
    def get_grasped_entity(self):
        name_list = []
        entity_list = []
        for name, entity in self.task.entities.items():
            if hasattr(entity, "is_grasped"):
                if entity.is_grasped(self.physics, self.robot):
                    name_list.append(name)
                    entity_list.append(entity)
        return name_list, entity_list
    
    def _reset_attempt(self):
        self.task.reset_distractors(n_distractor=self.n_distractor)
        self._hooks.refresh_entity_hooks()
        return super()._reset_attempt()

    def get_obstacle_pcd(self): 
        multi_view_rgb, multi_view_depth, multi_view_seg = [], [], []
        for cam in range(self.physics.model.ncam):
            multi_view_rgb.append(self.render(camera_id=cam, **self.render_options))
            multi_view_depth.append(self.render(camera_id=cam, **self.render_options, depth=True))
            multi_view_seg.append(self.render(camera_id=cam, **self.render_options, segmentation=True))
        segmentation = np.array(multi_view_seg)
        total_mask = np.ones_like(segmentation[..., 0])
        robot_mask = np.where((segmentation[..., 0] <= 72)&(segmentation[..., 0] > 0), 0, 1).astype(np.uint8)
        total_mask *= robot_mask
        grasped_obj_name_list, grasped_obj = self.get_grasped_entity()
        for name in grasped_obj_name_list:
            geom_ids = [self.physics.bind(geom).element_id for geom in self.task.entities[name].geoms]
            obj_mask = np.where((segmentation[..., 0] <= max(geom_ids))&(segmentation[..., 0] >= min(geom_ids)), 0, 1).astype(np.uint8)
            total_mask *= obj_mask
        obstacle_pcd= self.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(self.physics.model.ncam - 1)), 
                                                                                        rgb=multi_view_rgb,
                                                                                        depth=multi_view_depth,
                                                                                        mask=expand_mask(total_mask))
        return obstacle_pcd
    
    def get_intention_score(self, threshold=0.5, discrete=True):
        """
        Get the intention score of the task
        """
        return self.task.get_intention_score(self.physics, threshold, discrete)
    
    def get_task_progress(self):
        """
        Get the stage progress score of the task
        """
        return self.task.get_task_progress(self.physics)
    
    def get_expert_skill_sequence(self):
        """
        Get the expert demenstration of trajectory generation sequence
        """
        return self.task.get_expert_skill_sequence(self.physics)
    
    def save(self):
        """
        Save the task and env configuration
        """
        return self.task.save(self.physics)
    
    def get_robot_frame_position(self):
        return self.robot.get_base_position(self.physics)