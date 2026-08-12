# This file was modified from code copyrighted by the following:
# https://github.com/NVlabs/6dof-graspnet/blob/master/sample.py

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# -*- coding: utf-8 -*-


import argparse
import errno
import json
import multiprocessing as mp
import os
import sys
from functools import partial

import numpy as np
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from molmo_spaces.grasp_generation.robotiq_gripper import RobotiqGripper


class Object:
    def __init__(self, filename):
        self.mesh = trimesh.load(filename)
        self.scale = 1.0
        self.filename = filename
        if isinstance(self.mesh, list):
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)
        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object("object", self.mesh)

    def rescale(self, scale=1.0):
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def set_transform(self, position, rotation):
        if len(rotation) != 4:
            raise ValueError("Rotation must be a quaternion in xyzw format.")
        if not np.isclose(np.linalg.norm(rotation), 1.0):
            raise ValueError("Rotation must be a unit quaternion.")
        if len(position) != 3:
            raise ValueError("Position must be a 3D vector.")
        rotation = [rotation[3], rotation[0], rotation[1], rotation[2]]
        matrix = tra.quaternion_matrix(rotation)
        matrix[3, 3] = 1.0
        matrix[:3, 3] = position
        self.position = position
        self.rotation = rotation
        self.mesh.apply_transform(matrix)

    def resize(self, size=1.0):
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)


def create_gripper(configuration=None, root_folder=""):
    return RobotiqGripper(q=configuration, root_folder=root_folder)


def _check_collision_worker(object_mesh, gripper_mesh, transform_batch):
    manager = trimesh.collision.CollisionManager()
    manager.add_object("object", object_mesh)
    min_distances = []
    for tf in transform_batch:
        min_distances.append(manager.min_distance_single(gripper_mesh, transform=tf))
    return min_distances


def _quality_point_contacts_worker(batch_data):
    transform_batch, collision_batch, object_mesh = batch_data
    res = []
    gripper = create_gripper()
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)

    for p, colliding in zip(transform_batch, collision_batch):
        if colliding:
            res.append(-1)
        else:
            ray_origins, ray_directions = gripper.get_closing_rays(p)
            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False
            )
            if len(locations) == 0:
                res.append(0)
            else:
                valid_locations = (
                    np.linalg.norm(ray_origins[index_ray] - locations, axis=1) < 2.0 * gripper.q
                )
                if sum(valid_locations) == 0:
                    res.append(0)
                else:
                    contact_normals = object_mesh.face_normals[index_tri[valid_locations]]
                    motion_normals = ray_directions[index_ray[valid_locations]]
                    dot_prods = (motion_normals * contact_normals).sum(axis=1)
                    res.append(np.cos(dot_prods).sum() / len(ray_origins))
    return res


def _quality_antipodal_worker(batch_data):
    transform_batch, collision_batch, object_mesh = batch_data
    res = []
    contact_depths = []
    gripper = create_gripper()
    num_rays_per_finger = len(gripper.ray_origins) // 2

    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)

    for p, colliding in zip(transform_batch, collision_batch):
        if colliding:
            res.append(0)
            contact_depths.append(0.0)
            continue

        ray_origins, ray_directions = gripper.get_closing_rays(p)
        locations, index_ray, index_tri = intersector.intersects_location(
            ray_origins, ray_directions, multiple_hits=False
        )

        if locations.size == 0:
            res.append(0)
            contact_depths.append(0.0)
            continue

        index_ray_left = np.array(
            [
                i
                for i, num in enumerate(index_ray)
                if num % 2 == 0
                and np.linalg.norm(ray_origins[num] - locations[i]) < 2.0 * gripper.q
            ]
        )
        index_ray_right = np.array(
            [
                i
                for i, num in enumerate(index_ray)
                if num % 2 == 1
                and np.linalg.norm(ray_origins[num] - locations[i]) < 2.0 * gripper.q
            ]
        )

        if index_ray_left.size == 0 or index_ray_right.size == 0:
            res.append(0)
            contact_depths.append(0.0)
            continue

        left_contact_idx = np.linalg.norm(
            ray_origins[index_ray[index_ray_left]] - locations[index_ray_left], axis=1
        ).argmin()
        right_contact_idx = np.linalg.norm(
            ray_origins[index_ray[index_ray_right]] - locations[index_ray_right], axis=1
        ).argmin()
        left_contact_point = locations[index_ray_left[left_contact_idx]]
        right_contact_point = locations[index_ray_right[right_contact_idx]]

        left_contact_normal = object_mesh.face_normals[index_tri[index_ray_left[left_contact_idx]]]
        right_contact_normal = object_mesh.face_normals[
            index_tri[index_ray_right[right_contact_idx]]
        ]

        left_ray_num = index_ray[index_ray_left[left_contact_idx]] // 2
        right_ray_num = index_ray[index_ray_right[right_contact_idx]] // 2
        avg_contact_depth = ((left_ray_num + right_ray_num) / 2.0) / (num_rays_per_finger - 1)

        l_to_r = (right_contact_point - left_contact_point) / np.linalg.norm(
            right_contact_point - left_contact_point + 1e-8
        )
        r_to_l = (left_contact_point - right_contact_point) / np.linalg.norm(
            left_contact_point - right_contact_point + 1e-8
        )

        qual_left = np.dot(left_contact_normal, r_to_l)
        qual_right = np.dot(right_contact_normal, l_to_r)
        if qual_left < 0 or qual_right < 0:
            qual = 0
        else:
            qual = min(qual_left, qual_right)

        res.append(qual)
        contact_depths.append(avg_contact_depth)

    return res, contact_depths


def in_collision_with_gripper(object_mesh, gripper_transforms, silent=False, num_workers=None):
    if num_workers is None:
        num_workers = mp.cpu_count()

    if len(gripper_transforms) < 100 or num_workers <= 1:
        manager = trimesh.collision.CollisionManager()
        manager.add_object("object", object_mesh)
        gripper_meshes = [create_gripper().hand]
        min_distance = []
        for tf in tqdm(gripper_transforms, disable=silent):
            min_distance.append(
                np.min(
                    [
                        manager.min_distance_single(gripper_mesh, transform=tf)
                        for gripper_mesh in gripper_meshes
                    ]
                )
            )
        return [d == 0 for d in min_distance], min_distance

    gripper_mesh = create_gripper().hand
    num_transforms = len(gripper_transforms)
    batch_size = max(1, num_transforms // num_workers)
    batches = [gripper_transforms[i : i + batch_size] for i in range(0, num_transforms, batch_size)]
    worker_func = partial(_check_collision_worker, object_mesh, gripper_mesh)
    min_distances = []

    pbar = tqdm(
        total=num_transforms,
        disable=silent,
        desc=f"Checking collisions (using {num_workers} workers)",
    )
    with mp.Pool(processes=num_workers) as pool:
        for batch_result in pool.imap(worker_func, batches):
            min_distances.extend(batch_result)
            pbar.update(len(batch_result))
    pbar.close()

    return [d == 0 for d in min_distances], min_distances


def grasp_quality_point_contacts(
    transforms, collisions, object_mesh, silent=False, num_workers=None
):
    if num_workers is None:
        num_workers = mp.cpu_count()

    if len(transforms) < 100 or num_workers <= 1:
        res = []
        gripper = create_gripper()
        if trimesh.ray.has_embree:
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
                object_mesh, scale_to_box=True
            )
        else:
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)

        for p, colliding in tqdm(
            zip(transforms, collisions), total=len(transforms), disable=silent
        ):
            if colliding:
                res.append(-1)
            else:
                ray_origins, ray_directions = gripper.get_closing_rays(p)
                locations, index_ray, index_tri = intersector.intersects_location(
                    ray_origins, ray_directions, multiple_hits=False
                )
                if len(locations) == 0:
                    res.append(0)
                else:
                    valid_locations = (
                        np.linalg.norm(ray_origins[index_ray] - locations, axis=1) < 2.0 * gripper.q
                    )
                    if sum(valid_locations) == 0:
                        res.append(0)
                    else:
                        contact_normals = object_mesh.face_normals[index_tri[valid_locations]]
                        motion_normals = ray_directions[index_ray[valid_locations]]
                        dot_prods = (motion_normals * contact_normals).sum(axis=1)
                        res.append(np.cos(dot_prods).sum() / len(ray_origins))
        return res

    batch_size = max(1, len(transforms) // num_workers)
    transform_batches = [
        transforms[i : i + batch_size] for i in range(0, len(transforms), batch_size)
    ]
    collision_batches = [
        collisions[i : i + batch_size] for i in range(0, len(collisions), batch_size)
    ]
    batch_data = [
        (t_batch, c_batch, object_mesh)
        for t_batch, c_batch in zip(transform_batches, collision_batches)
    ]

    all_results = []
    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(
            total=len(transforms),
            disable=silent,
            desc=f"Computing point contact quality (using {num_workers} workers)",
        )
        for result in pool.imap(_quality_point_contacts_worker, batch_data):
            all_results.extend(result)
            pbar.update(len(result))
        pbar.close()

    return all_results


def grasp_quality_antipodal(transforms, collisions, object_mesh, silent=False, num_workers=None):
    if num_workers is None:
        num_workers = mp.cpu_count()

    if len(transforms) < 100 or num_workers <= 1:
        res = []
        contact_depths = []
        gripper = create_gripper()
        num_rays_per_finger = len(gripper.ray_origins) // 2

        if trimesh.ray.has_embree:
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
                object_mesh, scale_to_box=True
            )
        else:
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)

        for p, colliding in tqdm(
            zip(transforms, collisions), total=len(transforms), disable=silent
        ):
            if colliding:
                res.append(0)
                contact_depths.append(0.0)
                continue

            ray_origins, ray_directions = gripper.get_closing_rays(p)
            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False
            )

            if locations.size == 0:
                res.append(0)
                contact_depths.append(0.0)
                continue

            index_ray_left = np.array(
                [
                    i
                    for i, num in enumerate(index_ray)
                    if num % 2 == 0
                    and np.linalg.norm(ray_origins[num] - locations[i]) < 2.0 * gripper.q
                ]
            )
            index_ray_right = np.array(
                [
                    i
                    for i, num in enumerate(index_ray)
                    if num % 2 == 1
                    and np.linalg.norm(ray_origins[num] - locations[i]) < 2.0 * gripper.q
                ]
            )

            if index_ray_left.size == 0 or index_ray_right.size == 0:
                res.append(0)
                contact_depths.append(0.0)
                continue

            left_contact_idx = np.linalg.norm(
                ray_origins[index_ray[index_ray_left]] - locations[index_ray_left], axis=1
            ).argmin()
            right_contact_idx = np.linalg.norm(
                ray_origins[index_ray[index_ray_right]] - locations[index_ray_right], axis=1
            ).argmin()
            left_contact_point = locations[index_ray_left[left_contact_idx]]
            right_contact_point = locations[index_ray_right[right_contact_idx]]

            left_contact_normal = object_mesh.face_normals[
                index_tri[index_ray_left[left_contact_idx]]
            ]
            right_contact_normal = object_mesh.face_normals[
                index_tri[index_ray_right[right_contact_idx]]
            ]

            left_ray_num = index_ray[index_ray_left[left_contact_idx]] // 2
            right_ray_num = index_ray[index_ray_right[right_contact_idx]] // 2
            avg_contact_depth = ((left_ray_num + right_ray_num) / 2.0) / (num_rays_per_finger - 1)

            l_to_r = (right_contact_point - left_contact_point) / np.linalg.norm(
                right_contact_point - left_contact_point
            )
            r_to_l = (left_contact_point - right_contact_point) / np.linalg.norm(
                left_contact_point - right_contact_point
            )

            qual_left = np.dot(left_contact_normal, r_to_l)
            qual_right = np.dot(right_contact_normal, l_to_r)
            if qual_left < 0 or qual_right < 0:
                qual = 0
            else:
                qual = min(qual_left, qual_right)

            res.append(qual)
            contact_depths.append(avg_contact_depth)
        return res, contact_depths

    batch_size = max(1, len(transforms) // num_workers)
    transform_batches = [
        transforms[i : i + batch_size] for i in range(0, len(transforms), batch_size)
    ]
    collision_batches = [
        collisions[i : i + batch_size] for i in range(0, len(collisions), batch_size)
    ]
    batch_data = [
        (t_batch, c_batch, object_mesh)
        for t_batch, c_batch in zip(transform_batches, collision_batches)
    ]

    all_results = []
    all_contact_depths = []
    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm(
            total=len(transforms),
            disable=silent,
            desc=f"Computing antipodal quality (using {num_workers} workers)",
        )
        for result, depths in pool.imap(_quality_antipodal_worker, batch_data):
            all_results.extend(result)
            all_contact_depths.extend(depths)
            pbar.update(len(result))
        pbar.close()

    return all_results, all_contact_depths


def _process_points_batch(batch_data):
    points_batch, normals_batch, rotation_samples, standoff_samples, mesh = batch_data

    total_combinations = len(points_batch) * len(rotation_samples) * len(standoff_samples)
    batch_transforms = np.zeros((total_combinations, 4, 4))
    all_points = np.zeros((total_combinations, 3))
    all_normals = np.zeros((total_combinations, 3))
    all_roll_angles = np.zeros(total_combinations)
    all_standoffs = np.zeros(total_combinations)
    all_position_idx = np.zeros(total_combinations, dtype=int)

    idx = 0
    for i, (point, normal) in enumerate(zip(points_batch, normals_batch)):
        for roll in rotation_samples:
            orientation = tra.quaternion_matrix(tra.quaternion_about_axis(roll, [0, 0, 1]))
            for standoff in standoff_samples:
                origin = point + normal * standoff
                transform = np.dot(
                    np.dot(
                        tra.translation_matrix(origin),
                        trimesh.geometry.align_vectors([0, 0, -1], normal),
                    ),
                    orientation,
                )
                all_points[idx] = point
                all_normals[idx] = normal
                all_roll_angles[idx] = roll
                all_standoffs[idx] = standoff
                all_position_idx[idx] = i
                batch_transforms[idx] = transform
                idx += 1

    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    locations, index_rays, _ = intersector.intersects_location(
        batch_transforms[:, :3, 3], batch_transforms[:, :3, 2], multiple_hits=False
    )
    valid = np.array([False] * len(batch_transforms))
    valid[index_rays] = np.all(np.isclose(locations, all_points[index_rays]), axis=1)

    return (
        all_points[valid],
        all_normals[valid],
        batch_transforms[valid],
        all_roll_angles[valid],
        all_standoffs[valid],
        all_position_idx[valid],
    )


def _process_random_points(batch_data):
    points_batch, normals_batch, gripper, mesh = batch_data
    num_points = len(points_batch)
    batch_points = np.array(points_batch)
    batch_normals = np.array(normals_batch)
    batch_transforms = np.zeros((num_points, 4, 4))
    batch_roll_angles = np.zeros(num_points)
    batch_standoffs = np.zeros(num_points)

    angles = np.random.rand(num_points) * 2 * np.pi
    batch_roll_angles[:] = angles

    standoff_range = gripper.standoff_range
    standoffs = (standoff_range[1] - standoff_range[0]) * np.random.rand(
        num_points
    ) + standoff_range[0]
    batch_standoffs[:] = standoffs

    origins = batch_points + batch_normals * standoffs[:, np.newaxis]

    for i, (origin, normal, angle) in enumerate(zip(origins, batch_normals, angles)):
        orientation = tra.quaternion_matrix(tra.quaternion_about_axis(angle, [0, 0, 1]))
        batch_transforms[i] = np.dot(
            np.dot(
                tra.translation_matrix(origin), trimesh.geometry.align_vectors([0, 0, -1], normal)
            ),
            orientation,
        )

    return (batch_points, batch_normals, batch_transforms, batch_roll_angles, batch_standoffs)


def sample_multiple_grasps(
    number_of_candidates,
    mesh,
    systematic_sampling,
    surface_density=0.005 * 0.005,
    standoff_density=0.01,
    roll_density=15,
    type_of_quality="antipodal",
    min_quality=-1.0,
    silent=False,
    num_workers=None,
):
    if num_workers is None:
        num_workers = mp.cpu_count()

    transforms = []
    points = []
    normals = []
    roll_angles = []
    standoffs = []

    gripper = create_gripper()
    verboseprint = print if not silent else lambda *a, **k: None

    if systematic_sampling:
        surface_samples = int(np.ceil(mesh.area / surface_density))
        standoff_samples = np.linspace(
            gripper.standoff_range[0],
            gripper.standoff_range[1],
            max(1, int((gripper.standoff_range[1] - gripper.standoff_range[0]) / standoff_density)),
        )
        rotation_samples = np.arange(0, 1 * np.pi, np.deg2rad(roll_density))
        tmp_points, face_indices = mesh.sample(surface_samples, return_index=True)
        tmp_normals = mesh.face_normals[face_indices]
        estimated_candidates = len(tmp_points) * len(standoff_samples) * len(rotation_samples)

        if not silent:
            verboseprint(
                f"Estimated number of samples: {estimated_candidates:,} ({len(tmp_points):,} points x {len(standoff_samples)} standoffs x {len(rotation_samples)} rotations)"
            )

        batch_size = max(1, len(tmp_points) // num_workers)
        point_batches = [
            tmp_points[i : i + batch_size] for i in range(0, len(tmp_points), batch_size)
        ]
        normal_batches = [
            tmp_normals[i : i + batch_size] for i in range(0, len(tmp_normals), batch_size)
        ]

        batch_data = [
            (points_batch, normals_batch, rotation_samples, standoff_samples, mesh)
            for points_batch, normals_batch in zip(point_batches, normal_batches)
        ]

        all_points = []
        all_normals = []
        all_transforms = []
        all_roll_angles = []
        all_standoffs = []
        all_position_idx = []

        verboseprint("Sampling grasps in parallel...")
        with mp.Pool(processes=num_workers) as pool:
            batch_total = (
                sum(len(pb) for pb in point_batches) * len(rotation_samples) * len(standoff_samples)
            )
            pbar = tqdm(
                total=batch_total,
                disable=silent,
                desc=f"Sampling grasps (using {num_workers} workers)",
            )
            valid_count = 0
            processed_count = 0

            for result in pool.imap(_process_points_batch, batch_data):
                (
                    batch_points,
                    batch_normals,
                    batch_transforms,
                    batch_roll_angles,
                    batch_standoffs,
                    batch_position_idx,
                ) = result
                if len(batch_points) > 0:
                    all_points.extend(batch_points)
                    all_normals.extend(batch_normals)
                    all_transforms.extend(batch_transforms)
                    all_roll_angles.extend(batch_roll_angles)
                    all_standoffs.extend(batch_standoffs)
                    all_position_idx.extend(batch_position_idx)
                    valid_count += len(batch_points)

                processed_count += (
                    len(point_batches[0]) * len(rotation_samples) * len(standoff_samples)
                )
                pbar.update(len(point_batches[0]) * len(rotation_samples) * len(standoff_samples))
                pbar.set_postfix({"Valid": valid_count})
            pbar.close()

        points = np.array(all_points)
        normals = np.array(all_normals)
        transforms = np.array(all_transforms)
        roll_angles = np.array(all_roll_angles)
        standoffs = np.array(all_standoffs)

        verboseprint(f"Generated {len(transforms):,} valid grasps after sampling")

    else:
        points, face_indices = mesh.sample(number_of_candidates, return_index=True)
        normals = mesh.face_normals[face_indices]

        batch_size = max(1, len(points) // num_workers)
        point_batches = [points[i : i + batch_size] for i in range(0, len(points), batch_size)]
        normal_batches = [normals[i : i + batch_size] for i in range(0, len(normals), batch_size)]

        batch_data = [
            (points_batch, normals_batch, gripper, mesh)
            for points_batch, normals_batch in zip(point_batches, normal_batches)
        ]

        all_points = []
        all_normals = []
        all_transforms = []
        all_roll_angles = []
        all_standoffs = []

        verboseprint("Sampling grasps in parallel...")
        with mp.Pool(processes=num_workers) as pool:
            total_points = sum(len(pb) for pb in point_batches)
            pbar = tqdm(
                total=total_points,
                disable=silent,
                desc=f"Sampling grasps (using {num_workers} workers)",
            )

            for result in pool.imap(_process_random_points, batch_data):
                (
                    batch_points,
                    batch_normals,
                    batch_transforms,
                    batch_roll_angles,
                    batch_standoffs,
                ) = result
                all_points.extend(batch_points)
                all_normals.extend(batch_normals)
                all_transforms.extend(batch_transforms)
                all_roll_angles.extend(batch_roll_angles)
                all_standoffs.extend(batch_standoffs)
                pbar.update(len(batch_points))
            pbar.close()

        points = np.array(all_points)
        normals = np.array(all_normals)
        transforms = np.array(all_transforms)
        roll_angles = np.array(all_roll_angles)
        standoffs = np.array(all_standoffs)

        verboseprint(f"Generated {len(transforms):,} grasps with random sampling")

    verboseprint("Checking collisions...")
    collisions, _ = in_collision_with_gripper(
        mesh, transforms, silent=silent, num_workers=num_workers
    )

    verboseprint("Labelling grasps...")
    quality = {}
    contact_depths = []
    quality_key = "quality_" + type_of_quality
    if type_of_quality == "antipodal":
        quality[quality_key], contact_depths = grasp_quality_antipodal(
            transforms, collisions, object_mesh=mesh, silent=silent, num_workers=num_workers
        )
    elif type_of_quality == "number_of_contacts":
        quality[quality_key] = grasp_quality_point_contacts(
            transforms, collisions, object_mesh=mesh, silent=silent, num_workers=num_workers
        )
        contact_depths = [0.0] * len(transforms)
    else:
        raise Exception("Quality metric unknown: ", quality)

    quality_np = np.array(quality[quality_key])
    contact_depths_np = np.array(contact_depths)
    collisions = np.array(collisions)

    f_points = []
    f_normals = []
    f_transforms = []
    f_roll_angles = []
    f_standoffs = []
    f_collisions = []
    f_quality = []
    f_contact_depths = []

    for i, _ in enumerate(transforms):
        if quality_np[i] >= min_quality:
            f_points.append(points[i])
            f_normals.append(normals[i])
            f_transforms.append(transforms[i])
            f_roll_angles.append(roll_angles[i])
            f_standoffs.append(standoffs[i])
            f_collisions.append(int(collisions[i]))
            f_quality.append(quality_np[i])
            f_contact_depths.append(contact_depths_np[i])

    points = np.array(f_points)
    normals = np.array(f_normals)
    transforms = np.array(f_transforms)
    roll_angles = np.array(f_roll_angles)
    standoffs = np.array(f_standoffs)
    collisions = f_collisions
    quality[quality_key] = f_quality
    contact_depths = f_contact_depths

    verboseprint(f"Final result: {len(transforms):,} valid grasps with quality >= {min_quality}")

    return points, normals, transforms, roll_angles, standoffs, collisions, quality, contact_depths


def generate_per_joint_grasps(joint_meshes_json, base_prefix, args):
    with open(joint_meshes_json, "r") as f:
        joint_meshes = json.load(f)
    summary = []

    extra_collision_mesh = None
    if args.collision_object_file:
        extra_collision_obj = Object(args.collision_object_file)
        extra_collision_mesh = extra_collision_obj.mesh

    def get_collision_mesh(main_mesh, extra_mesh):
        if extra_mesh is not None:
            return trimesh.util.concatenate([main_mesh, extra_mesh])
        else:
            return main_mesh

    for entry in joint_meshes:
        joint_name = entry["joint"]
        mesh_file = entry["handle_mesh"]
        handle_geoms = entry.get("handle_geoms", [])
        mesh_path = os.path.join(os.path.dirname(joint_meshes_json), mesh_file)
        grasps_out = os.path.join(os.path.dirname(joint_meshes_json), f"{joint_name}_grasps.json")
        obj = Object(mesh_path)
        if args.resize:
            obj.resize(args.resize)
        else:
            obj.rescale(args.scale)
        obj.set_transform(position=args.position, rotation=args.rotation)
        (
            points,
            normals,
            transforms,
            roll_angles,
            standoffs,
            collisions,
            qualities,
            contact_depths,
        ) = sample_multiple_grasps(
            args.num_samples,
            obj.mesh,
            systematic_sampling=args.systematic_sampling,
            roll_density=args.systematic_roll_density,
            standoff_density=args.systematic_standoff_density,
            surface_density=args.systematic_surface_density,
            type_of_quality=args.quality,
            min_quality=args.min_quality,
            silent=args.silent,
            num_workers=args.num_workers,
        )

        if extra_collision_mesh is not None:
            combined_mesh = get_collision_mesh(obj.mesh, extra_collision_mesh)
            collisions, _ = in_collision_with_gripper(
                combined_mesh, transforms, silent=args.silent, num_workers=args.num_workers
            )
            valid_indices = [i for i, coll in enumerate(collisions) if not coll]
            points = points[valid_indices]
            normals = normals[valid_indices]
            transforms = transforms[valid_indices]
            roll_angles = roll_angles[valid_indices]
            standoffs = standoffs[valid_indices]
            collisions = [collisions[i] for i in valid_indices]
            qualities = {k: [v[i] for i in valid_indices] for k, v in qualities.items()}
            contact_depths = [contact_depths[i] for i in valid_indices]

        for i in range(len(transforms)):
            transforms[i][:3, 3] += transforms[i][:3, :3] @ RobotiqGripper.tcp_offset

        gripper = create_gripper()
        grasps = {
            "object": obj.filename,
            "object_scale": obj.scale,
            "object_position": obj.position,
            "object_rotation": obj.rotation,
            "object_class": args.classname,
            "object_dataset": args.dataset,
            "gripper_configuration": [gripper.q],
            "transforms": [t.tolist() for t in transforms],
            "roll_angles": roll_angles.tolist(),
            "standoffs": standoffs.tolist(),
            "contact_depths": contact_depths,
            "mesh_points": [p.tolist() for p in points],
            "mesh_normals": [n.tolist() for n in normals],
            "collisions": collisions,
        }
        with open(grasps_out, "w") as f:
            json.dump(grasps, f)
        new_entry = dict(entry)
        new_entry.update(
            {
                "grasps_file": os.path.basename(grasps_out),
                "handle_mesh": mesh_file,
                "handle_geoms": handle_geoms,
            }
        )
        summary.append(new_entry)

    with open(joint_meshes_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote per-joint grasps and summary to {joint_meshes_json}")


def make_parser():
    parser = argparse.ArgumentParser(
        description="Sample grasps for an object.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--object_file", type=str, default=None, help="Path to object mesh file.")
    parser.add_argument(
        "--collision_object_file",
        type=str,
        default=None,
        help="Optional: extra .obj file to use for collision checking (combined with main object).",
    )
    parser.add_argument(
        "--dataset", type=str, default="UNKNOWN", help="Metadata about the origin of the file."
    )
    parser.add_argument(
        "--classname", type=str, default="UNKNOWN", help="Metadata about the class of the object."
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Scale the object.")
    parser.add_argument(
        "--resize", type=float, help="Resize the object to a specific size (in meters)."
    )
    parser.add_argument("--use_stl", action="store_true", help="Use STL instead of obj.")
    parser.add_argument(
        "--quality",
        choices=["number_of_contacts", "antipodal"],
        default="number_of_contacts",
        help="Which type of quality metric to evaluate.",
    )
    parser.add_argument(
        "--position",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help="Position of the object in the world frame (x, y, z).",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        nargs=4,
        default=[0, 0, 0, 1],
        help="Rotation of the object in quaternion format (x, y, z, w).",
    )
    parser.add_argument(
        "--single_standoff", action="store_true", help="Use the closest possible standoff."
    )
    parser.add_argument(
        "--systematic_sampling", action="store_true", help="Systematically sample stuff."
    )
    parser.add_argument(
        "--systematic_surface_density",
        type=float,
        default=0.005 * 0.005,
        help="Surface density used for systematic sampling (in square meters).",
    )
    parser.add_argument(
        "--systematic_standoff_density",
        type=float,
        default=0.01,
        help="Standoff density used for systematic sampling (in meters).",
    )
    parser.add_argument(
        "--systematic_roll_density",
        type=float,
        default=15.0,
        help="Roll density used for systematic sampling (in degrees).",
    )
    parser.add_argument(
        "--filter_best_per_position",
        action="store_true",
        help="Only store one grasp (highest quality) if there are multiple per with the same position.",
    )
    parser.add_argument("--min_quality", type=float, default=0.005, help="min quality")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of samples.")
    parser.add_argument(
        "--output", type=str, default="grasps.json", help="File to store the results (json)."
    )
    parser.add_argument(
        "--add_quality_metric",
        nargs=2,
        type=str,
        default="",
        help="File (json) to calculate additional quality metric for.",
    )
    parser.add_argument("--silent", action="store_true", help="No commandline output.")
    parser.add_argument("--force", action="store_true", help="Do things my way.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers to use for collision checking. Default uses all available CPU cores.",
    )
    parser.add_argument(
        "--per_joint_grasps_from_meshes",
        type=str,
        default=None,
        help="Path to joint_meshes.json to generate per-joint grasps.",
    )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    verboseprint = print if not args.silent else lambda *a, **k: None

    if args.per_joint_grasps_from_meshes:
        base_prefix = os.path.splitext(os.path.basename(args.per_joint_grasps_from_meshes))[
            0
        ].replace("_joint_meshes", "")
        generate_per_joint_grasps(args.per_joint_grasps_from_meshes, base_prefix, args)
        exit(0)
    else:
        if args.object_file is None:
            print("Error: --object_file is required when not using --per_joint_grasps_from_meshes")
            exit(1)

        if os.path.dirname(args.output) != "":
            try:
                os.makedirs(os.path.dirname(args.output))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        obj = Object(args.object_file.replace(".obj", ".stl") if args.use_stl else args.object_file)

        if args.resize:
            obj.resize(args.resize)
        else:
            obj.rescale(args.scale)

        obj.set_transform(position=args.position, rotation=args.rotation)
        gripper = create_gripper()
        (
            points,
            normals,
            transforms,
            roll_angles,
            standoffs,
            collisions,
            qualities,
            contact_depths,
        ) = sample_multiple_grasps(
            args.num_samples,
            obj.mesh,
            systematic_sampling=args.systematic_sampling,
            roll_density=args.systematic_roll_density,
            standoff_density=args.systematic_standoff_density,
            surface_density=args.systematic_surface_density,
            type_of_quality=args.quality,
            min_quality=args.min_quality,
            silent=args.silent,
            num_workers=args.num_workers,
        )

        extra_collision_mesh = None
        if args.collision_object_file:
            extra_collision_obj = Object(args.collision_object_file)
            extra_collision_mesh = extra_collision_obj.mesh

        def get_collision_mesh(main_mesh, extra_mesh):
            if extra_mesh is not None:
                return trimesh.util.concatenate([main_mesh, extra_mesh])
            else:
                return main_mesh

        if extra_collision_mesh is not None:
            combined_mesh = get_collision_mesh(obj.mesh, extra_collision_mesh)
            collisions, _ = in_collision_with_gripper(
                combined_mesh, transforms, silent=args.silent, num_workers=args.num_workers
            )
            valid_indices = [i for i, coll in enumerate(collisions) if not coll]
            points = points[valid_indices]
            normals = normals[valid_indices]
            transforms = transforms[valid_indices]
            roll_angles = roll_angles[valid_indices]
            standoffs = standoffs[valid_indices]
            collisions = [collisions[i] for i in valid_indices]
            qualities = {k: [v[i] for i in valid_indices] for k, v in qualities.items()}
            contact_depths = [contact_depths[i] for i in valid_indices]

        for i in range(len(transforms)):
            transforms[i][:3, 3] += transforms[i][:3, :3] @ gripper.tcp_offset

        grasps = {
            "object": obj.filename,
            "object_scale": obj.scale,
            "object_position": obj.position,
            "object_rotation": obj.rotation,
            "object_class": args.classname,
            "object_dataset": args.dataset,
            "gripper_configuration": [gripper.q],
            "transforms": [t.tolist() for t in transforms],
            "roll_angles": roll_angles.tolist(),
            "standoffs": standoffs.tolist(),
            "contact_depths": contact_depths,
            "mesh_points": [p.tolist() for p in points],
            "mesh_normals": [n.tolist() for n in normals],
            "collisions": collisions,
        }

        with open(args.output, "w") as f:
            verboseprint("Writing results to:", args.output)
            json.dump(grasps, f)
