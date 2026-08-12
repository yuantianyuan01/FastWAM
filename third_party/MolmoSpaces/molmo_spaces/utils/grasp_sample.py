"""
This module contains functionality for filtering and sampling grasps based on heuristics.
"""

import logging

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.utils.profiler_utils import Timer

log = logging.getLogger(__name__)


def get_grasp_collision_body_name(grasp_idx: int) -> str:
    return f"grasp_collision_{grasp_idx}"


def add_grasp_collision_bodies(
    spec: mujoco.MjSpec,
    num_grasps: int,
    grasp_width: float,
    grasp_length: float,
    grasp_height: float,
    grasp_base_pos: np.ndarray,
):
    """Add grasp collision bodies to the scene."""
    for i in range(num_grasps):
        # init grasp bodies in the sky (below the ground causes collision with the floor)
        grasp_body = spec.worldbody.add_body(
            name=get_grasp_collision_body_name(i),
            pos=[0, 0, 10],
            gravcomp=1.0,
        )
        grasp_body.add_freejoint()

        geom_kwargs = dict(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            rgba=[0, 0, 1, 1],
            group=3,
            contype=0,
            conaffinity=0b1111,
        )

        base_geom = grasp_body.add_geom(**geom_kwargs)
        base_geom.size[0] = grasp_height / 2
        base_geom.fromto[:3] = np.array([0, -grasp_width / 2, 0]) + grasp_base_pos
        base_geom.fromto[3:] = np.array([0, grasp_width / 2, 0]) + grasp_base_pos

        finger1_geom = grasp_body.add_geom(**geom_kwargs)
        finger1_geom.size[0] = grasp_height / 2
        finger1_geom.fromto[:3] = np.array([0, -grasp_width / 2, 0]) + grasp_base_pos
        finger1_geom.fromto[3:] = np.array([0, -grasp_width / 2, grasp_length]) + grasp_base_pos

        finger2_geom = grasp_body.add_geom(**geom_kwargs)
        finger2_geom.size[0] = grasp_height / 2
        finger2_geom.fromto[:3] = np.array([0, grasp_width / 2, 0]) + grasp_base_pos
        finger2_geom.fromto[3:] = np.array([0, grasp_width / 2, grasp_length]) + grasp_base_pos


def get_noncolliding_grasp_mask(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    grasp_poses_world: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    n_grasps = len(grasp_poses_world)
    grasp_bodies = [
        create_mlspaces_body(mj_data, get_grasp_collision_body_name(i)) for i in range(batch_size)
    ]
    start_poses = [body.pose.copy() for body in grasp_bodies]
    grasp_body_ids = set(body.body_id for body in grasp_bodies)

    try:
        colliding_grasp_mask = np.zeros(n_grasps, dtype=bool)
        for i in range(0, n_grasps, batch_size):
            grasp_bid_to_idx = {}
            n_grasps_in_batch = min(batch_size, n_grasps - i)
            for j in range(i, i + n_grasps_in_batch):
                grasp_body = grasp_bodies[j - i]
                grasp_body.pose = grasp_poses_world[j]
                grasp_bid_to_idx[grasp_body.body_id] = j
            for j in range(n_grasps_in_batch, len(grasp_bodies)):
                grasp_bodies[j].pose = start_poses[j]

            mujoco.mj_kinematics(mj_model, mj_data)
            mujoco.mj_collision(mj_model, mj_data)
            for contact in mj_data.contact:
                bid1 = mj_model.geom_bodyid[contact.geom1]
                bid2 = mj_model.geom_bodyid[contact.geom2]
                if bid1 in grasp_body_ids or bid2 in grasp_body_ids:
                    grasp_bid = bid1 if bid1 in grasp_body_ids else bid2
                    other_bid = bid2 if grasp_bid == bid1 else bid1
                    assert other_bid not in grasp_body_ids

                    grasp_idx = grasp_bid_to_idx[grasp_bid]
                    colliding_grasp_mask[grasp_idx] = True

        return ~colliding_grasp_mask
    finally:
        # move the grasp bodies back out of the way
        for body, pose in zip(grasp_bodies, start_poses):
            body.pose = pose
        mujoco.mj_fwdPosition(mj_model, mj_data)


def get_feasible_grasp_idx(
    mg_id: str,
    robot: Robot,
    grasp_poses_world: np.ndarray,
    n_ik_checks: int,
    ik_batch_size: int,
):
    n_checks_done = 0
    ret: int | None = None

    with Timer() as ik_check_time:
        for i in range(0, n_ik_checks, ik_batch_size):
            grasps = grasp_poses_world[i : i + ik_batch_size]
            n_checks_done += len(grasps)
            real_batch_size = len(grasps)

            if real_batch_size < ik_batch_size:
                # pad to batch size to avoid triggering recompilation
                grasps = np.concatenate(
                    [grasps, np.broadcast_to(grasps[-1:], (ik_batch_size - real_batch_size, 4, 4))]
                )

            ik_result = robot.parallel_kinematics.ik(
                mg_id,
                grasps,
                None,
                robot.robot_view.get_qpos_dict(),
                robot.robot_view.base.pose,
                rel_to_base=False,
            )
            for j, result in enumerate(ik_result[:real_batch_size]):
                if result is not None:
                    ret = i + j
                    break
            if ret is not None:
                break
    log.info(
        f"Feasibility-checked {n_checks_done} grasps in {ik_check_time.value:.3f}s, found feasible grasp: {ret is not None}"
    )

    return ret


def select_grasp_pose(
    env: CPUMujocoEnv,
    grasp_poses_world: np.ndarray,
    object_pose: np.ndarray,
    check_collision: bool,
    n_collision_checks: int,
    collision_batch_size: int,
    check_ik: bool,
    n_ik_checks: int,
    ik_batch_size: int,
    pos_cost_weight: float = 1.0,
    rot_cost_weight: float = 0.01,
    vertical_cost_weight: float = 2.0,
    horizontal_cost_weight: float = 0,
    com_dist_cost_weight: float = 8.0,
) -> np.ndarray:
    robot = env.current_robot
    gripper_mg_id = robot.robot_view.get_gripper_movegroup_ids()[0]
    tcp_pose = robot.robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world
    tcp_pose_inv = np.linalg.inv(tcp_pose)

    dist_tcp = tcp_pose_inv @ grasp_poses_world  # shape (N,4,4)
    dists_tcp_p = np.linalg.norm(dist_tcp[:, :3, 3], axis=1)
    dist_tcp_o = R.from_matrix(dist_tcp[:, :3, :3]).magnitude() * 180 / np.pi

    dists_up = grasp_poses_world[:, 2, 2]  # range = [-1, 1]

    dists_com = np.linalg.norm((np.linalg.inv(object_pose) @ grasp_poses_world)[:, :3, 3], axis=1)

    # Cost for horizontal orientation: 0 = perfectly horizontal (z-axis parallel to XY plane), 1 = vertical
    # Lower cost = more horizontal, so we want to minimize this
    # Use squared term to more strongly penalize vertical orientations
    dists_xy_parallel = np.abs(dists_up) ** 2

    dist_total = (
        pos_cost_weight * dists_tcp_p
        + rot_cost_weight * dist_tcp_o
        + vertical_cost_weight * dists_up
        + horizontal_cost_weight * dists_xy_parallel
        + com_dist_cost_weight * dists_com
    )
    close_grasp_ids = np.argsort(dist_total, kind="stable")  # weight positions and orientations
    close_grasp_ids = close_grasp_ids[:n_collision_checks]

    # filter for noncolliding grasps
    if check_collision:
        with Timer() as collision_check_time:
            noncolliding_grasp_mask = get_noncolliding_grasp_mask(
                env.current_model,
                env.current_data,
                grasp_poses_world[close_grasp_ids],
                collision_batch_size,
            )

        log.info(
            f"Collision-checked {len(close_grasp_ids)} grasps in {collision_check_time.value:.3f}s, found {np.sum(noncolliding_grasp_mask)} non-colliding grasps"
        )
    else:
        noncolliding_grasp_mask = np.ones(len(close_grasp_ids), dtype=bool)

    noncolliding_close_grasp_ids = close_grasp_ids[noncolliding_grasp_mask]
    # colliding_close_grasp_ids = close_grasp_ids[~noncolliding_grasp_mask]

    # filter for feasibility/reachability
    if check_ik:
        grasp_idx: int | None = None

        if noncolliding_close_grasp_ids.size > 0:
            noncolliding_grasp_idx = get_feasible_grasp_idx(
                gripper_mg_id,
                robot,
                grasp_poses_world[noncolliding_close_grasp_ids],
                n_ik_checks,
                ik_batch_size,
            )
            if noncolliding_grasp_idx is not None:
                grasp_idx = noncolliding_close_grasp_ids[noncolliding_grasp_idx]
    elif noncolliding_close_grasp_ids.size > 0:
        grasp_idx = int(noncolliding_close_grasp_ids[0])
    else:
        grasp_idx = None

    if grasp_idx is None:
        raise ValueError("No feasible grasp found")

    return grasp_poses_world[grasp_idx]
