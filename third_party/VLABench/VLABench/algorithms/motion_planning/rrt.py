import numpy as np
import open3d as o3d
from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot
from VLABench.algorithms.path_smoothing import bezier_smoothing, polynomial_smoothing
from VLABench.algorithms.utils import remove_pcd_near_point

SEARCH_DIMENSIONS = np.array([(-1, 1), (-1, 1), (0.8, 1.3)])

def rrt_motion_planning(start_pos, 
                        end_pos,  
                        obstacle_pcd, 
                        search_dimensions=SEARCH_DIMENSIONS,
                        q=0.05,
                        r=0.02,
                        max_samples=1024,
                        prc=0.1,
                        margin=2e-2,
                        z_threshold=0.8,
                        smooth_method=None,
                        retry_time=0,
            ):
    """
    3d rrt motion planning for robot arm.
    Input: 
        start_pos: tuple, (3,)
        end_pos: ttple, (3,)
        obstacle_pcd: np.ndarray, (n, 3)
        search_dimensions: np.ndarray, (3, 2), range of x-y-z space
        q: length of tree edges
        r: length of smallest edge to check for intersection with obstacles
        max_samples:  max number of samples to take before timing out
        prc: probability of checking for a connection to goal
        smooth_method: str, method to smooth the path, None or one of ["bezier", "polynomial"]
    """
    if isinstance(obstacle_pcd, o3d.geometry.PointCloud):
        obstacle_pcd = np.asarray(obstacle_pcd.points)
    obstacle_pcd = obstacle_pcd[obstacle_pcd[:, 2] >= z_threshold]
    obstacle_pcd = remove_pcd_near_point(obstacle_pcd, start_pos)
    if obstacle_pcd is not None and obstacle_pcd.shape[1] != 6:
        obstacle_pcd = np.concatenate([obstacle_pcd - margin * np.ones((obstacle_pcd.shape[0], 3)),
                                       obstacle_pcd + margin * np.ones((obstacle_pcd.shape[0], 3))], 
                                      axis=1)
    search_space = SearchSpace(search_dimensions, obstacle_pcd)
    rrt = RRT(search_space, q, start_pos, end_pos, max_samples, r, prc)
    path = rrt.rrt_search()
    retry = 0
    while retry < retry_time and path is None:
        retry += 1
        start_pos_list = list(start_pos)
        start_pos_list[-1] += 0.1
        start_pos = tuple(start_pos_list)
        rrt = RRT(search_space, q, start_pos, end_pos, max_samples, r, prc)
        path = rrt.rrt_search() 
    if smooth_method == "bezier":
        path_arr = bezier_smoothing(path)
        path = [tuple(p) for p in path_arr]
    elif smooth_method == "polynomial":
        path_arr = polynomial_smoothing(path)
        path = [tuple(p) for p in path_arr]
    return path

def rrt_star(start_pos, 
             end_pos, 
             obstacle_pcd, 
             search_dimensions=SEARCH_DIMENSIONS, 
             q=8, 
             r=1, 
             max_samples=1024, 
             rewire_count=32,
             prc=0.1):
    """
    3d rrt* motion planning for robot arm.
    Input: 
        start_pos: np.ndarray, (3,)
        end_pos: np.ndarray, (3,)
        obstacle_pcd: np.ndarray, (n, 3)
        search_dimensions: np.ndarray, (3, 2), range of x-y-z space
        q: length of tree edges
        r: length of smallest edge to check for intersection with obstacles
        max_samples:  max number of samples to take before timing out
        rewire_count: optional, number of nearby branches to rewire
        prc: probability of checking for a connection to goal
    """
    if isinstance(obstacle_pcd, o3d.geometry.PointCloud):
        obstacle_pcd = np.asarray(obstacle_pcd.points)
    if obstacle_pcd is not None and obstacle_pcd.shape[1] != 6:
        obstacle_pcd = np.concatenate([obstacle_pcd, obstacle_pcd + 
                                       0.01 * np.ones((obstacle_pcd.shape[0], 3))], axis=1)
    search_space = SearchSpace(search_dimensions, obstacle_pcd)
    rrt = RRTStar(search_space, q, start_pos, end_pos, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star()
    
    return path

def visulize_search_path(search_space, rrt, path, obstacles, start_pos, end_pos):
    """
    Visulize the search path of rrt algorithm.
    Input:
        search_space: SearchSpace, search space
        rrt: RRT, rrt object
    """
    plot = Plot("rrt_3d_with_random_obstacles")
    plot.plot_tree(search_space, rrt.trees)
    if path is not None:
        plot.plot_path(search_space, path)
    plot.plot_obstacles(search_space, obstacles)
    plot.plot_start(search_space, start_pos)
    plot.plot_goal(search_space, end_pos)
    plot.draw(auto_open=True)