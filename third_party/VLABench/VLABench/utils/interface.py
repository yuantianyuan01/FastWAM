import cv2
import numpy as np
import h5py
import json
import os
import time
import open3d as o3d
from functools import partial
from pynput.keyboard import Key, Listener
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler
from VLABench.utils.depth2cloud import *
from VLABench.utils.data_utils import save_single_data

def adaptive_stack(images, columns):
    """
    Stack the multiple images into a single image with adaptive rows and columns.

    params:
    images -- image list, np.ndarray.
    columns -- number of columns.

    return:
    stacked_image -- stacked image
    """
    rows = int(np.ceil(len(images) / columns))
    h, w, c = images[0].shape
    stacked_image = np.zeros((rows * h, columns * w, c), dtype=np.uint8)  
    
    for i, img in enumerate(images):
        row = i // columns
        col = i % columns
        stacked_image[row * h: (row + 1) * h, col * w: (col + 1) * w, :] = img
    
    return stacked_image
 
            
#TODO: use pyqt to add more interactive functions?
class Interface:
    """
    Interaface to collect demonstration trajectory.
    """
    def __init__(self, 
                 env, 
                 save_dir,
                 dir_name,
                 render_options,
                 move_step=0.01, 
                 rotate_step=0.1,
                 resize_resolution=(256, 256),
                 point_cloud_options=dict(
                     bound=[[-1, -1, 0],[1, 1, 1]],
                     
                 )
                 ):
        self.env = env
        self.move_step = move_step
        self.rotate_step = rotate_step
        self.grasp = False
        self.render_options = render_options
        self.save_dir = save_dir
        self.dir_name = dir_name
        self.camera_matrixs = list()
        self.transforms = [
            partial(cv2.resize, dsize=resize_resolution, interpolation=cv2.INTER_AREA),
        ]
        self.point_cloud_options = point_cloud_options
        self.reset()
        # self.device = device
        # env.display_world_coordinate()
    
    def transform(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image
    
    def set_move_step(self, move_step):
        self.move_step = move_step
    
    def set_rotate_step(self, rotate_step):
        self.rotate_step = rotate_step
    
    def reset(self):
        self.env.reset()
        self.env._reset_next_step = False
        self.grasp = False
        self.last_pos = None
        self.last_euler = None
        self.reset_step()
        self.timestep = 0
        self.buffer = dict(
            instruction=self.env.task.get_instruction(),
            images=[],
            trajectory=[],
            qpos=[],
            depth=[],
            point_cloud_points=[],
            point_cloud_colors=[],
        ) #TODO： depth
        self.quit = False
        self.log_menu()
    
    def reset_step(self):
        self.delta_pos = np.zeros(3)
        self.delta_rot = np.zeros(3)
        
    def render(self, camera_ids=[0,1,2,3,4,5], column=1, render_only=False):
        if len(camera_ids) == 0:
            raise ValueError("camera_ids should not be empty.")
        if not self.camera_matrixs:
            for camera_id in camera_ids:
                intrinsic_mat, extrinsic_mat = self.env.get_camera_matrix(camera_id, 
                                                                        **self.render_options)
                self.camera_matrixs.append([intrinsic_mat.tolist(), extrinsic_mat.tolist()])   
                    
        multi_view_images = []
        multi_view_depths = []
        for camera_id in camera_ids:
            self.render_options["camera_id"] = camera_id
            image = self.env.render(**self.render_options)
            depth = self.env.render(depth=True, **self.render_options)
            multi_view_images.append(image)
            multi_view_depths.append(np.uint8(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)))
        stacked_images = adaptive_stack(multi_view_images, column)
        stacked_images = cv2.cvtColor(stacked_images, cv2.COLOR_RGB2BGR)
        cv2.imshow("Renderer", stacked_images)
        if render_only:
            return
        key = cv2.waitKeyEx(1)
        
        valid = self.press(key)
        if self.delta_pos.sum() != 0 or self.delta_rot.sum() != 0:
            self.buffer["images"].append([self.transform(img) for img in multi_view_images])
            self.buffer["depth"].append([self.transform(depth) for depth in multi_view_depths])
            point_cloud = self.generate_point_cloud(multi_view_depths, multi_view_images)
            self.buffer["point_cloud_points"].append(np.asarray(point_cloud.points))
            self.buffer["point_cloud_colors"].append(np.asarray(point_cloud.colors))
            
        if valid:    
            self.step()
        
    def press(self, key):
        self.reset_step()
        # move
        if key == ord("w"): # along y-axis
            self.delta_pos[1] += self.move_step
        elif key == ord("s"):
            self.delta_pos[1] -= self.move_step
        elif key == ord("a"): # along x-axis
            self.delta_pos[0] -= self.move_step
        elif key == ord("d"):
            self.delta_pos[0] += self.move_step
        # along z-axis
        elif key == ord("i"):  # Up arrow key 
            self.delta_pos[2] += self.move_step
        elif key == ord("k"):  # Down arrow key
            self.delta_pos[2] -= self.move_step
        
        # rotate
        # raw
        elif key == 44: # <
            self.delta_rot[0] += self.rotate_step
        elif key == 46: # >
            self.delta_rot[0] -= self.rotate_step
        # pitch
        elif key == ord("n"):
            self.delta_rot[1] += self.rotate_step
        elif key == ord("m"):
            self.delta_rot[1] -= self.rotate_step
        # yaw
        elif key == ord("j"): # left arrow key
            self.delta_rot[2] += self.rotate_step
        elif key == ord("l"): # right arrow key
            self.delta_rot[2] -= self.rotate_step

        # other functions
        elif key == 27: # esc
            self.quit = True
        elif key == 32: # space
            self.grasp = not self.grasp
        elif key == 8: # backspace
            self.delete_last_step()
        elif key == 13:
            self.reset()
            
        else:
            pass
            # return False
        return True
            
    def step(self, max_n_substep=30, tolerance=1e-2):
        pos = np.array(self.env.robot.get_end_effector_pos(self.env.physics)) 
        quat = self.env.robot.get_end_effector_quat(self.env.physics)
        euler = np.array(quaternion_to_euler(quat)) 
        if self.last_pos is None and self.last_euler is None: # initial trajectory point
            self.last_pos = pos
            self.last_euler = euler    
            # self.buffer["trajectory"].append(np.concatenate([pos, euler, np.zeros(1)], axis=-1))
        
        next_pos = self.last_pos + self.delta_pos
        next_euler = self.last_euler + self.delta_rot
        quat = euler_to_quaternion(*next_euler)
        self.last_pos = next_pos
        self.last_euler = next_euler
        if self.delta_pos.sum() != 0 or self.delta_rot.sum() != 0:
            self.buffer["trajectory"].append(np.concatenate([next_pos, next_euler, np.array([int(self.grasp)])], axis=-1))
        success, action = self.env.robot.get_qpos_from_ee_pos(self.env.physics, next_pos, quat)[:7]
        # print("original", pos, euler)
        # print("target", next_pos, next_euler)
        if self.grasp:
            action = np.concatenate([action, np.zeros(2)])
        else:
            action = np.concatenate([action, np.ones(2) * 0.04])
        # action = np.zeros(9)
        # print("action", action)    
        for _ in range(max_n_substep):
            self.env.step(action)
            self.env._reset_next_step = False
            current_qpos = np.array(self.env.task.robot.get_qpos(self.env.physics)).reshape(-1)
            if np.max(current_qpos - np.array(action[:7])) < tolerance \
                and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                break
        self.timestep += 1
        
        
    def delete_last_step(self):
        """
        This only can be used when other objects are not moved.
        an example case is that the robot is rotated in unexpected direction(e.g. target +raw but operation is -raw)
        
        Return back the last action and delete the last frame of replay buffer.
        """
        if self.timestep == 0:
            print("No more steps to delete.")
            return
        self.buffer["trajectory"].pop(-1)
        self.buffer["images"].pop(-1)
        self.delta_pos = -self.delta_pos
        self.delta_rot = -self.delta_rot
        self.step()
        self.render(column=2, render_only=True)
        self.timestep -= 1
    
    def save_single_data(self, index, filename):
        save_single_data(self.buffer, self.save_dir, self.dir_name, filename)

    def generate_point_cloud(self, depth_images, rgb_images):
        conbimed_point_cloud = o3d.geometry.PointCloud()
        multi_view_pc = []
        for depth, rgb, mat in zip(depth_images, rgb_images, self.camera_matrixs):
            instrinsic_mat, extrinsic_mat = mat
            # uint8 convert to continuous array
            rgb = np.ascontiguousarray(rgb)
            width, height = depth.shape
            instrinsic_mat = cammat2o3d(np.array(instrinsic_mat), width, height)
            # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            #                 o3d.geometry.Image(rgb),
            #                 o3d.geometry.Image(depth),
            #                 convert_rgb_to_intensity=False
            #             )
            depth = o3d.geometry.Image(depth.astype(np.uint16))
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth, 
                                                                        instrinsic_mat,
                                                                        extrinsic_mat
                                                                        )
            # if self.point_cloud_options["bound"] is not None:
            #     bound = o3d.geometry.AxisAlignedBoundingBox(min_bound=self.point_cloud_options["bound"][0], 
            #                                                 max_bound=self.point_cloud_options["bound"][1])
            #     o3d_cloud = o3d_cloud.crop(bound)
            multi_view_pc.append(o3d_cloud)
        for pc in multi_view_pc:
            conbimed_point_cloud += pc
        pcd_downsampled = conbimed_point_cloud.voxel_down_sample(voxel_size=0.001)
        return pcd_downsampled
    
    def close(self):
        cv2.destroyAllWindows()
    
    def log_menu(self):
        print("="*30, "Operation Menu", "="*30)
        print("Press 'w', 's', 'a', 'd', 'i', 'k' to move the robot along y, x, z axis.")
        print("Press ',', '.' to rotate the robot along raw axis.")
        print("Press 'n', 'm' to rotate the robot along pitch axis.")
        print("Press 'j', 'l' to rotate the robot along yaw axis.")
        print("Press 'space' to grasp.")
        print("Press 'backspace' to delete the last step.")
        print("Press 'enter' to reset.")
        print("Press 'esc' to quit.")
        print("="*80)