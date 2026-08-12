#!/usr/bin/env python3

import math
import numpy as np

from PIL import Image as PIL_Image

import open3d as o3d

"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""
class PointCloudGenerator(object):
    """
    initialization function
    Parameters:
        -physics: dm_control physics object
        -min_bound: If not None, list len(3) containing smallest x, y, and z
            values that will not be cropped
        -max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self, physics, min_bound=None, max_bound=None, **kwargs):
        super(PointCloudGenerator, self).__init__()

        self.physics = physics

        self.img_width = kwargs.get("width", 480)
        self.img_height = kwargs.get("height", 480)
        self.voxel_size = kwargs.get("voxel_size", 0.01)

        self.ncams = self.physics.model.ncam

        self.target_bounds=None
        if min_bound != None and max_bound != None:
            self.target_bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

        # List of camera intrinsic matrices
        self.cam_mats = []
        for cam_id in range(self.ncams):
            fovy = math.radians(self.physics.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generate_pcd_from_rgbd(self, target_id=None, rgb=None, depth=None, mask=None):
        """
        Parameters:
            - target_id: list of camera indices to generate point clouds from
            - rgb: list of rgb images corresponding to each camera
            - depth: list of depth images corresponding to each camera
            - mask: list of masks corresponding to each camera, to gain pcd of specific objects
        """
        o3d_clouds = []
        cam_poses = []
        for cam_i in range(self.ncams):
            if target_id != None and cam_i not in target_id:
                continue
            # Render and optionally save image from camera corresponding to cam_i
            if depth is None: depth_img = self.capture_image(cam_i)
            else: depth_img = np.ascontiguousarray(depth[cam_i])
            if rgb is None: color_img = self.capture_image(cam_i, False)
            else: color_img = np.ascontiguousarray(rgb[cam_i])
            if mask is not None:
                # print("deptj img = " + str(depth_img.shape))
                depth_img = np.ascontiguousarray(depth_img * mask[cam_i])
                color_img = np.ascontiguousarray(color_img * np.repeat(mask[cam_i][:, :, np.newaxis], 3, axis=2))
        
            # convert camera matrix and depth image to Open3D format, then generate point cloud
            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            depth_img = o3d.geometry.Image(depth_img)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            o3d.geometry.Image(color_img),
                            o3d.geometry.Image(depth_img),
                            convert_rgb_to_intensity=False,
                            depth_scale=1.0,
                            depth_trunc=10,
                        )
            o3d_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, od_cammat)
           
           
            # Compute world to camera transformation matrix
            cam_pos = self.physics.data.cam_xpos[cam_i]
            cam_poses.append(cam_pos)
            c2b_r = rotMatList2NPRotMat(self.physics.model.cam_mat0[cam_i])
            # In MuJoCo, we assume that a camera is specified in XML as a body
            #    with pose p, and that that body has a camera sub-element
            #    with pos and euler 0.
            #    Therefore, camera frame with body euler 0 must be rotated about
            #    x-axis by 180 degrees to align it with the world frame.
            b2w_r = quat2Mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            transformed_cloud = o3d_cloud.transform(c2w)

            # If both minimum and maximum bounds are provided, crop cloud to fit
            #    inside them.
            if self.target_bounds != None:
                transformed_cloud = transformed_cloud.crop(self.target_bounds)

            # Estimate normals of cropped cloud, then flip them based on camera
            #    position.
            # transformed_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=250))
            # transformed_cloud.orient_normals_towards_camera_location(cam_pos)

            o3d_clouds.append(transformed_cloud)
            # o3d.visualization.draw_geometries([o3d_cloud])

        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        combined_cloud = combined_cloud.voxel_down_sample(voxel_size=self.voxel_size)
        return combined_cloud
    
    def capture_image(self, cam_ind, capture_depth=True):
        image = self.physics.render(width=self.img_width, height=self.img_height, camera_id=cam_ind, depth=capture_depth)
        return np.ascontiguousarray(image)
    
def build_pcd_from_rgbd(depths, rgbs, instrinsics, extrinsics, dawnsample_ratio=0.005):
    """
    depths: multi view depth image, (n_cam, height, width)
    rgbs: multi view rgb image, (n_cam, height, width, 3)
    instrinsics: multi view camera instrinsic matrix, (n_cam, 3, 3)
    extrinsics: multi view camera extrinsic matrix, (n_cam, 4, 4)
    """
    combined_point_cloud = o3d.geometry.PointCloud()
    multi_view_pc = []
    assert len(depths) == len(rgbs) == len(instrinsics) == len(extrinsics), "The number of cameras should be the same"
    for i, (depth, rgb, instrinsic, extrinsic) in enumerate(zip(depths, rgbs, instrinsics, extrinsics)):
        height, width = rgb.shape[:2]

        depth = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgb = o3d.geometry.Image(np.ascontiguousarray(rgb))
       
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            convert_rgb_to_intensity=False
        )
        instrinsic_mat = cammat2o3d(instrinsic, width, height)
    
        # point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, instrinsic_mat)
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth, instrinsic_mat)
        point_cloud.transform(extrinsic)
        
        multi_view_pc.append(point_cloud)
        
    for i, pc in enumerate(multi_view_pc[:6]):
        combined_point_cloud += pc   
    combined_point_cloud.estimate_normals()
    pcd_downsample = combined_point_cloud.voxel_down_sample(voxel_size=dawnsample_ratio)
    return pcd_downsample