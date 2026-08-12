import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import random
import copy
from scipy.spatial import cKDTree
import cv2
from sklearn.cluster import DBSCAN
import logging
import colorlog

def normalize(v):
    return v / np.linalg.norm(v)

def compute_rotation_quaternion(camera_pos, target_pos, forward_axis=[1, 0, 0]):
    """
        Compute the ratation quaternion from camera position to target position
    """
    target_direction = np.array(target_pos) - np.array(camera_pos)
    target_direction = normalize(target_direction)

    base_forward = normalize(np.array(forward_axis))

    if np.allclose(target_direction, base_forward):
        return R.from_quat([0, 0, 0, 1])
    elif np.allclose(target_direction, -base_forward):
        orthogonal_axis = np.array([base_forward[1], -base_forward[0], 0])
        orthogonal_axis = normalize(orthogonal_axis)
        return R.from_rotvec(np.pi * orthogonal_axis).as_quat()
    else:
        axis = np.cross(base_forward, target_direction)
        axis = normalize(axis)
        angle = np.arccos(np.clip(np.dot(base_forward, target_direction), -1.0, 1.0))
        return R.from_rotvec(angle * axis).as_quat()

def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return (qw, qx, qy, qz)

def quaternion_to_euler(quat, is_degree=False):
    # (w, x, y, z) -> (x, y, z, w)
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles

def matrix_to_quaternion(matrix):
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    r = R.from_matrix(matrix)
    quaternion = r.as_quat()
    # (x, y, z, w) -> (w, x, y, z)
    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    return quaternion

def quaternion_to_matrix(quat):
    # (w, x, y, z) -> (x, y, z, w)
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    matrix = r.as_matrix()
    return matrix

def move_long_quaternion(position, quaternion, distance):
    """
        Move along the quaternion direction
    """
    roation = R.from_quat(quaternion)
    direction = roation.as_rotvec()
    direction = direction / np.linalg.norm(direction)
    new_position = position + direction * distance
    return new_position

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def pcd_has_overlap(pcd1, pcd2, voxel_size=0.05):
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1, voxel_size)
    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd2, voxel_size)
    
    voxels1 = set(map(lambda v: (v.grid_index[0], v.grid_index[1], v.grid_index[2]), voxel_grid1.get_voxels()))
    voxels2 = set(map(lambda v: (v.grid_index[0], v.grid_index[1], v.grid_index[2]), voxel_grid2.get_voxels()))
    
    intersection = voxels1.intersection(voxels2)
    union = voxels1.union(voxels2)
    
    iou = len(intersection) / len(union) if len(union) > 0 else 0
    
    return iou

def find_keypoint_and_prepare_grasp(env, entity, prior_euler, std=0, max_retry=100, specific_keypoint_id=0, move_vector=None):
    """
    sample a keypoint and confirm the validation of it and its prepare point
    return the valid grasp keypoint, prepare point and the quaternion of the gripper
    """
    keypoints = entity.get_grasped_keypoints(env.physics)
    valid = False
    env_pcd = env.get_observation()["masked_point_cloud"]
    retry = 0
    while not valid:
        if specific_keypoint_id is not None:
            keypoint = keypoints[specific_keypoint_id]
        else:
            keypoint = random.choice(keypoints)
        key_euler = prior_euler[retry % len(prior_euler)]
        key_euler += np.random.normal(0, std, 3)
        key_quat = euler_to_quaternion(*key_euler)
        gripper_pcd, gripper_forward_vector = env.robot.gripper_pcd(keypoint, key_quat)
        if move_vector is None: move_vector = gripper_forward_vector
        distance = -0.1
        prepare_point = keypoint + move_vector * distance
        grasp_prepare_gripper_pcd = copy.deepcopy(gripper_pcd).translate(move_vector * distance)
        grasp_prepare_gripper_pcd.paint_uniform_color([0, 1, 0])
    
        collision1 = gripper_collision_check(gripper_pcd, env_pcd)
        collision2 = gripper_collision_check(grasp_prepare_gripper_pcd, env_pcd)    
        if not collision1 and not collision2:
            valid = True
        retry += 1
        if retry > max_retry:
            print("cant find a valid grasp point, take default one")
            return keypoint, keypoint+np.array([0, -0.1, 0]), euler_to_quaternion(*prior_euler[0])
        if retry % 10 == 0:
            # increase the search space by increasing the std
            std += np.pi/100
            
    return keypoint, prepare_point, key_quat    

def gripper_collision_check(gripper_pcd, env_pcd, threshold=0.1):
    iou = pcd_has_overlap(env_pcd, gripper_pcd)
    return iou > threshold

def pcd_has_overlap(pcd1, pcd2, distance_threshold=0.05):
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    distances1, _ = tree2.query(points1, k=1)
    matches1 = distances1 < distance_threshold

    distances2, _ = tree1.query(points2, k=1)
    matches2 = distances2 < distance_threshold

    intersection_count = np.sum(matches1) + np.sum(matches2)
    union_count = len(points1) + len(points2) - intersection_count
    
    iou = intersection_count / union_count if union_count > 0 else 0
    
    return iou

def distance(p1, p2):
    if not isinstance(p1, np.ndarray):
        p1 = np.array(p1)
    if not isinstance(p2, np.ndarray):
        p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def farthest_first_sampling(points, k):
    sampled_points = [points[np.random.randint(len(points))]]
    
    for _ in range(1, k):
        min_distances = [min(distance(p, sp) for sp in sampled_points) for p in points]
        
        # choose the point with max minimal distance
        farthest_point = points[np.argmax(min_distances)]
        sampled_points.append(farthest_point)
    
    return sampled_points

def grid_sample(workspace, grid_size, n_samples, farthest_sample=True):
    """
    workspace: [min_x, max_x, min_y, max_y, min_z, max_z]
    grid_size: [n_row, n_col]
    
    """
    min_x, max_x, min_y, max_y, _, _ = workspace
    n_row, n_col = grid_size
    x_step = (max_x - min_x) / n_col
    y_step = (max_y - min_y) / n_row
    
    grid_points = []
    for i in range(n_row):
        for j in range(n_col):
            center_x = min_x + (j + 0.5) * x_step
            center_y = min_y + (i + 0.5) * y_step
            grid_points.append((center_x, center_y))
    if farthest_sample:
        sampled_points = farthest_first_sampling(grid_points, n_samples)
    else:
        sampled_points = random.sample(grid_points, n_samples)
    
    return sampled_points

def point_to_line_distance(anchor, axis, point):
    """
    compute the distance from a point to a line
    
    param:
    - anchor: the anchor point of rotation axis (3D vector) [x, y, z]
    - axis: the direction vector of rotation axis [vx, vy, vz]
    - point: (3D vector) [x, y, z]
    
    return:
    - the distance
    """
    A = np.array(anchor)  
    V = np.array(axis)    
    Q = np.array(point)

    AQ = Q - A

    cross_product = np.cross(AQ, V)

    distance = np.linalg.norm(cross_product)

    return distance

def rotate_point_around_axis(point, anchor, axis, angle):
    """
    compute the point after rotation around the axis with Rodrigues' rotation formula

    params:
    - point: (3D vector) [x, y, z]
    - anchor:(3D vector) [x, y, z]
    - axis: (3D vector) [vx, vy, vz]
    - angle: rotation angle (radian)

    return:
    - the vector point after (3D vector)
    """
    P = np.array(point)
    A = np.array(anchor)
    V = np.array(axis) / np.linalg.norm(axis)  

    PA = P - A

    part1 = np.cos(angle) * PA
    part2 = np.sin(angle) * np.cross(V, PA)
    part3 = (1 - np.cos(angle)) * V * np.dot(V, PA)

    P_prime = A + part1 + part2 + part3

    return P_prime

def slide_point_along_axis(point, axis, distance):
    """
    compute the point after sliding along the axis 

    params:
    - point: (3D vector) [x, y, z]
    - axis: (3D vector) [vx, vy, vz]
    - angle: rotation angle (radian)

    return:
    - the vector point after (3D vector)
    """
    point = np.array(point)
    axis = np.array(axis)
    
    xaxis_normalized = axis / np.linalg.norm(axis)
    
    new_point = point + distance * xaxis_normalized
    
    return new_point

def quaternion_from_axis_angle(axis, angle):
    """
    param:
     - angle: radian
    """
    half_angle = angle / 2
    w = np.cos(half_angle)
    sin_half_angle = np.sin(half_angle)
    
    v = np.array(axis) / np.linalg.norm(axis)
    
    x = v[0] * sin_half_angle
    y = v[1] * sin_half_angle
    z = v[2] * sin_half_angle
    
    return np.array([w, x, y, z])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z])

def flatten_list(ls):
    new_list = []
    for item in ls:
        if isinstance(item, list):
            new_list.extend(item)
        elif isinstance(item, str):
            new_list.append(item)
    return new_list

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotate_point_by_quaternion(point, quat):
    p = np.array([0] + list(point))
    q_conj = quaternion_conjugate(quat)
    p_prime = quaternion_multiply(quaternion_multiply(quat, p), q_conj)
    
    return p_prime[1:]

def expand_mask(masks, kernel_size=3, iterations=1):
    """
    Expands a batch of binary masks (0 and 1 values) using morphological dilation.
    
    Parameters:
    - masks: np.ndarray, shape (n, h, w), batch of binary masks (0 and 1 values).
    - kernel_size: int, size of the kernel for dilation, default is 3x3.
    - iterations: int, number of times to apply dilation, default is 1.

    Returns:
    - expanded_masks: np.ndarray, shape (n, h, w), batch of masks with dilated edges.
    """
    if len(masks.shape) == 2: #  convert (h, w) to (1, h, w) for unified operation
        masks = masks.reshape(1, masks.shape[0], masks.shape[1])
    # Define the dilation kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Create an empty array to store the expanded masks
    expanded_masks = np.zeros_like(masks, dtype=np.uint8)
    # Loop through each mask in the batch
    for i in range(masks.shape[0]):
        # Invert the mask: 0 -> 1, 1 -> 0
        inverted_mask = 1 - masks[i]
        # Convert the inverted mask to uint8 (required for OpenCV functions)
        mask_uint8 = (inverted_mask * 255).astype(np.uint8)
        # Apply morphological dilation
        expanded_mask = cv2.dilate(mask_uint8, kernel, iterations=iterations)
        # Convert back to binary (0 and 1), then invert again: 1 -> 0, 0 -> 1
        expanded_masks[i] = 1 - (expanded_mask > 0).astype(np.uint8)
    return expanded_masks
    


def pcd_filtering(pcd, eps=0.05, min_samples=10):
    """
    DBSCAN clustering
    params:
        eps: The maximum distance two points can be considered neighbors.
        min_samples: The minimum number of points required to form a dense cluster.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(pcd)
    unique_labels, counts = np.unique(labels, return_counts=True)

    # ignore noise points
    if -1 in unique_labels:
        noise_index = np.where(unique_labels == -1)[0][0]
        unique_labels = np.delete(unique_labels, noise_index)
        counts = np.delete(counts, noise_index)

    max_cluster_label = unique_labels[np.argmax(counts)]
    most_dense_cluster = pcd[labels == max_cluster_label]
    return most_dense_cluster

def find_key_by_value(dictionary, target_value):
    """
    Given a dictionary and the corresponding value, find the key that contains the target value
    """
    for key, value in dictionary.items():
        if isinstance(value, list) and target_value in value:
            return key
        elif not isinstance(value, list) and value == target_value:
            return key
    return target_value

def get_logger(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(color_formatter)
    for handler in logger.handlers:
        logger.removeHandler(handler)
        
    logger.addHandler(console_handler)
    return logger

if __name__ == "__main__":
    # quat = compute_rotation_quaternion([0, 0, 1], [0, 0, 0])
    # print(quaternion_to_euler(quat, is_degree=True))
    workspace = [-1, 1, -1, 1, -1, 1]
    sampled_points = grid_sample(workspace, [5, 5], n_samples=10)
    def visulize_grid_point(points):
        import matplotlib.pyplot as plt
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(x_coords, y_coords, c='blue', marker='o')
        
        for i, point in enumerate(points):
            plt.text(point[0], point[1], f'({point[0]:.1f}, {point[1]:.1f})', fontsize=9, ha='right')
        
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Points Visualization')
        plt.grid(True)
        plt.axis('equal') 
        
        plt.show()
    visulize_grid_point(sampled_points)