import argparse
import json
import multiprocessing as mp
import os
import re
import time
import traceback
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from molmo_spaces.molmo_spaces_constants import ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR, ASSETS_DIR


def rotation_matrix_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )


def rotate_vector(vector, rotation_matrix):
    return np.dot(rotation_matrix, vector)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])


def axis_angle_to_quat(axis, angle):
    axis = axis / np.linalg.norm(axis)
    half_angle = angle * 0.5
    s = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s])


def get_joint_position(model, data, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint '{joint_name}' not found in the model.")
    return data.joint(joint_id).qpos.copy()


def check_sufficient_joint_movement(joint_positions, max_range):
    max_joint_position = max(joint_positions)
    min_joint_position = min(joint_positions)
    return (max_joint_position - min_joint_position) / max_range >= 0.7


def is_grasping(model, data, handle_geoms):
    left_patterns = ["left_finger", "finger_l", "gripper_finger_left", "left"]
    right_patterns = ["right_finger", "finger_r", "gripper_finger_right", "right"]

    handle_geoms = [re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", geom) for geom in handle_geoms]
    for i in range(data.ncon):
        contact = data.contact[i]

        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        if not geom1 or not geom2:
            continue

        if any(
            [
                handle_geom.lower() in geom1.lower() or geom1.lower() in handle_geom.lower()
                for handle_geom in handle_geoms
            ]
        ) or any(
            [
                handle_geom.lower() in geom2.lower() or geom2.lower() in handle_geom.lower()
                for handle_geom in handle_geoms
            ]
        ):
            other = (
                geom2
                if np.any([handle_geom.lower() in geom1.lower() for handle_geom in handle_geoms])
                else geom1
            )
            if any(p in other.lower() for p in left_patterns) or any(
                p in other.lower() for p in right_patterns
            ):
                return True

    return False


def merge_xml_contents(base_xml_content, additional_xml_content):
    base_root = ET.fromstring(base_xml_content)
    additional_root = ET.fromstring(additional_xml_content)

    if base_root.tag != "mujoco" or additional_root.tag != "mujoco":
        raise ValueError("Both XML contents must have 'mujoco' as the root element")

    processed_sections = {}

    for additional_child in additional_root:
        tag_name = additional_child.tag

        base_section = base_root.find(tag_name)

        if base_section is not None:
            if tag_name in processed_sections:
                continue

            for element in additional_child:
                is_duplicate = False
                for existing in base_section:
                    if element.tag == existing.tag and all(
                        attr in existing.attrib and existing.attrib[attr] == val
                        for attr, val in element.attrib.items()
                        if attr != "name"
                    ):
                        if "name" in element.attrib and "name" in existing.attrib:
                            if element.attrib["name"] == existing.attrib["name"]:
                                is_duplicate = True
                                break
                        else:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    base_section.append(element)

            processed_sections[tag_name] = True
        else:
            base_root.append(additional_child)
            processed_sections[tag_name] = True

    return ET.tostring(base_root, encoding="unicode")


def is_object_grasped(model, data, object_name):
    left_finger_contact = False
    right_finger_contact = False

    left_patterns = ["left_finger", "finger_l", "gripper_finger_left", "left"]
    right_patterns = ["right_finger", "finger_r", "gripper_finger_right", "right"]

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        if not geom1 or not geom2:
            continue

        if object_name.lower() in geom1.lower() or object_name.lower() in geom2.lower():
            other = geom2 if object_name.lower() in geom1.lower() else geom1
            if any(p in other.lower() for p in left_patterns):
                left_finger_contact = True
            if any(p in other.lower() for p in right_patterns):
                right_finger_contact = True

    return left_finger_contact and right_finger_contact


def check_grasp(model, data, object_name, store_initial=False):
    global initial_relative_position, initial_grasp_verified

    object_pos = data.body(object_name).xpos
    gripper_pos = data.site("end_effector").xpos
    relative_position = object_pos - gripper_pos

    if store_initial:
        if is_object_grasped(model, data, object_name):
            initial_relative_position = relative_position.copy()
            initial_grasp_verified = True
            return True
        initial_grasp_verified = False
        return False

    if not initial_grasp_verified or initial_relative_position is None:
        return False

    position_change = np.linalg.norm(relative_position - initial_relative_position)
    return position_change < 0.03 and is_object_grasped(model, data, object_name)


def test_single_grasp(
    grasp_data,
    object_name,
    xml_content,
    args,
    handle_geoms,
    primary_joint=None,
    render=False,
):
    joint_info = primary_joint
    if len(handle_geoms) == 0:
        print(
            f"[ERROR] No handle geometry info provided for joint '{object_name}'. Skipping this grasp."
        )
        return grasp_data[0], None, None
    if not joint_info:
        print(
            f"[ERROR] No joint axis info provided for joint '{object_name}'. Skipping this grasp."
        )
        return grasp_data[0], None, None

    i, transform, quality, config = grasp_data

    pos = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

    pose_before = np.eye(4)
    pose_before[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    pose_before[:3, 3] = pos

    approach_distance = args.approach_distance
    approach_steps = args.approach_steps
    approach_vector = transform[:3, 2] * approach_distance
    approach_pos = pos - approach_vector

    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()
    gripper_base = root.find(".//body[@name='base']")
    if gripper_base is not None:
        gripper_base.set("pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}")
        gripper_base.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
    target_ee_pose = root.find(".//body[@name='target_ee_pose']")
    if target_ee_pose is not None:
        target_ee_pose.set("pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}")
        target_ee_pose.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)

    data.ctrl[0] = 0.0

    for _ in range(500):
        mujoco.mj_step(model, data)
        if render and viewer is not None:
            viewer.sync()

    joint_name = None
    if joint_info and "primary_joint" in joint_info:
        joint_name = joint_info["primary_joint"]["name"]
    elif joint_info and "name" in joint_info:
        joint_name = joint_info["name"]
    if not joint_name:
        if render and viewer is not None:
            viewer.close()
            del viewer
            time.sleep(0.1)
        return i, None, None
    joint_position_before = get_joint_position(model, data, joint_name)

    for step in range(approach_steps):
        new_pos = approach_pos + (step / approach_steps) * approach_vector
        data.mocap_pos[0] = new_pos
        data.mocap_quat[0] = quat
        for _ in range(100):
            mujoco.mj_step(model, data)
            if render and viewer is not None:
                viewer.sync()

    joint_position_after = get_joint_position(model, data, joint_name)

    if joint_position_before is None or joint_position_after is None:
        if render and viewer is not None:
            viewer.close()
            del viewer
            time.sleep(0.1)
        return i, None, None
    if np.abs(joint_position_after - joint_position_before) > 0.01:
        if render and viewer is not None:
            viewer.close()
            del viewer
            time.sleep(0.1)
        return i, None, None

    data.mocap_pos[0] = pos
    mujoco.mj_step(model, data, nstep=500)
    if render and viewer is not None:
        viewer.sync()

    data.ctrl[0] = 255.0

    for _ in range(500):
        mujoco.mj_step(model, data)
        if render and viewer is not None:
            viewer.sync()

    T_world_grasp = np.eye(4)
    T_world_grasp[:3, :3] = data.site("grasp_site").xmat.reshape(3, 3)
    T_world_grasp[:3, 3] = data.site("grasp_site").xpos

    T_world_joint = np.eye(4)
    joint_id = model.joint(joint_name).bodyid
    T_world_joint[:3, 3] = data.xpos[joint_id]
    T_world_joint[:3, :3] = data.xmat[joint_id].reshape(3, 3)

    T_joint_grasp = np.linalg.inv(T_world_joint) @ T_world_grasp

    transform = T_joint_grasp
    while 1:
        mujoco.mj_step(model, data)
        if render and viewer is not None:
            viewer.sync()
        break

    if not is_grasping(model, data, handle_geoms):
        if render and viewer is not None:
            viewer.close()
            del viewer
            time.sleep(0.1)
        return i, None, None

    num_waypoints = 400
    waypoints = []
    primary_joint_data = None
    if joint_info and "primary_joint" in joint_info:
        primary_joint_data = joint_info["primary_joint"]
    elif joint_info:
        primary_joint_data = joint_info
    if primary_joint_data:
        joint_type = primary_joint_data.get("type")
        joint_range_str = primary_joint_data.get("range", "0 0")
        joint_range = [float(x) for x in joint_range_str.split()]

        gripper_pos = data.site("grasp_site").xpos.copy()
        gripper_quat = quat

        if joint_type == "hinge" or joint_type == "unknown":
            rotation_axis = primary_joint_data.get("rotation_axis", {"x": 0, "y": 0, "z": 0})
            axis_world = np.array(
                [
                    rotation_axis.get("x", 0),
                    rotation_axis.get("y", 0),
                    rotation_axis.get("z", 0),
                ]
            )
            joint_position = primary_joint_data.get("position", {"x": 0, "y": 0, "z": 0})
            pivot_point = np.array(
                [
                    joint_position.get("x", 0),
                    joint_position.get("y", 0),
                    joint_position.get("z", 0),
                ]
            )

            max_angle = joint_range[1]
            if max_angle == 0:
                max_angle = joint_range[0]

            rel_pos = gripper_pos - pivot_point
            for i in range(num_waypoints + 1):
                angle = i * max_angle / num_waypoints
                rotation_matrix = rotation_matrix_from_axis_angle(axis_world, angle)
                new_rel_pos = rotate_vector(rel_pos, rotation_matrix)
                new_pos = pivot_point + new_rel_pos
                rotation_quat = axis_angle_to_quat(axis_world, angle)
                new_quat = quat_multiply(rotation_quat, gripper_quat)
                waypoints.append((new_pos, new_quat))

            max_range = np.abs(max_angle)
        elif joint_type == "slide":
            rotation_axis = primary_joint_data.get("rotation_axis", {"x": 0, "y": 0, "z": 0})
            slide_axis = np.array(
                [
                    rotation_axis.get("x", 0),
                    rotation_axis.get("y", 0),
                    rotation_axis.get("z", 0),
                ]
            )

            max_distance = joint_range[1]
            if max_distance == 0:
                max_distance = joint_range[0]
            for i in range(num_waypoints + 1):
                distance = i * max_distance / num_waypoints
                new_pos = gripper_pos + slide_axis * distance
                waypoints.append((new_pos, gripper_quat.copy()))

            max_range = np.abs(max_distance)

    joint_positions = []
    articulation_success = True
    mujoco.mj_step(model, data, nstep=2000)

    if waypoints:
        num_loops = args.articulation_loops
        for _ in range(num_loops):
            if not articulation_success:
                break

            for wp_idx, (wp_pos, wp_quat) in enumerate(waypoints):
                data.mocap_pos[0] = wp_pos
                data.mocap_quat[0] = wp_quat

                mujoco.mj_step(model, data, nstep=10)
                if render and viewer is not None:
                    viewer.sync()

                if wp_idx % 10 == 0 and wp_idx < len(waypoints) * 0.9:
                    is_currently_grasping = is_grasping(model, data, handle_geoms)
                else:
                    is_currently_grasping = True

                joint_position = get_joint_position(model, data, primary_joint["name"])
                joint_positions.append(joint_position)

                if not is_currently_grasping:
                    articulation_success = False
                    break

            if not articulation_success:
                break

            for wp_idx in range(len(waypoints) - 1, -1, -1):
                wp_pos, wp_quat = waypoints[wp_idx]
                data.mocap_pos[0] = wp_pos
                data.mocap_quat[0] = wp_quat

                mujoco.mj_step(model, data, nstep=10)
                if render and viewer is not None:
                    viewer.sync()

                if wp_idx % 10 == 0 and wp_idx < len(waypoints) * 0.9:
                    is_currently_grasping = is_grasping(model, data, handle_geoms)
                else:
                    is_currently_grasping = True

                joint_position = get_joint_position(model, data, primary_joint["name"])
                joint_positions.append(joint_position)

                if not is_currently_grasping:
                    articulation_success = False
                    break

    if joint_positions:
        sufficient_movement = check_sufficient_joint_movement(joint_positions, max_range)
        if not sufficient_movement:
            articulation_success = False

    if render and viewer is not None:
        viewer.close()
        del viewer
        time.sleep(0.1)
    if articulation_success:
        return i, transform, quality
    else:
        return i, None, None


def run_simulation_with_viewer(
    model,
    data,
    xml_content,
    object_name,
    use_viewer,
    args,
    primary_joint=None,
    handle_geoms=None,
):
    with open(args.grasps_path, "r") as f:
        grasp_data = json.load(f)
    transforms = np.array(grasp_data["transforms"])
    print(len(transforms), "grasps to evaluate")
    qualities = np.array(grasp_data.get("quality_antipodal", [1.0] * len(transforms)))
    width = np.array(grasp_data.get("grasp_widths", [0.1] * len(transforms)))
    contact_depths = np.array(grasp_data.get("contact_depths", [0.5] * len(transforms)))

    if args.min_contact_depth > 0.0 or args.max_contact_depth < 1.0:
        depth_mask = (contact_depths >= args.min_contact_depth) & (
            contact_depths <= args.max_contact_depth
        )
        transforms = transforms[depth_mask]
        qualities = qualities[depth_mask]
        width = width[depth_mask]
        contact_depths = contact_depths[depth_mask]
        print(
            f"Filtered by contact depth [{args.min_contact_depth}, {args.max_contact_depth}]: {len(transforms)} grasps remaining"
        )

    if args.diversity_mode:
        positions = transforms[:, :3, 3]
        num_clusters = min(args.num_clusters, len(transforms))

        try:
            rotations = np.array([R.from_matrix(t[:3, :3]).as_rotvec() for t in transforms])

            pos_normalized = positions / (np.std(positions, axis=0) + 1e-6)
            rot_normalized = rotations / (np.std(rotations, axis=0) + 1e-6)

            features = np.concatenate([pos_normalized, rot_normalized], axis=1)

            batch_size = min(1000, len(transforms))
            kmeans = MiniBatchKMeans(
                n_clusters=num_clusters,
                random_state=42,
                batch_size=batch_size,
                max_iter=100,
                n_init=3,
                reassignment_ratio=0.01,
                verbose=0,
            )
            cluster_labels = kmeans.fit_predict(features)
        except Exception:
            try:
                pos_normalized = positions / (np.std(positions, axis=0) + 1e-6)
                batch_size = min(1000, len(transforms))
                kmeans = MiniBatchKMeans(
                    n_clusters=num_clusters,
                    random_state=42,
                    batch_size=batch_size,
                    max_iter=100,
                    n_init=3,
                    reassignment_ratio=0.01,
                    verbose=0,
                )
                cluster_labels = kmeans.fit_predict(pos_normalized)
            except Exception:
                cluster_labels = np.arange(len(transforms)) % num_clusters

        cluster_orders = []
        for cluster_id in range(num_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            if args.center_contact_depth is not None:
                cluster_depths = contact_depths[cluster_indices]
                depth_distances = np.abs(cluster_depths - args.center_contact_depth)
                epsilon = 0.001
                inverse_distances = 1.0 / (depth_distances + epsilon)
                depth_scores = np.power(inverse_distances, args.contact_depth_bias)
                depth_weights = depth_scores / np.sum(depth_scores)
                sorted_local_indices = np.random.choice(
                    len(cluster_indices), size=len(cluster_indices), replace=False, p=depth_weights
                )
            else:
                sorted_local_indices = np.arange(len(cluster_indices))

            cluster_orders.append(cluster_indices[sorted_local_indices])

        priority_indices = []
        max_cluster_size = max(len(cluster) for cluster in cluster_orders)

        for i in range(max_cluster_size):
            for cluster in cluster_orders:
                if i < len(cluster):
                    priority_indices.append(cluster[i])

        priority_indices = np.array(priority_indices)
        transforms = transforms[priority_indices]
        qualities = qualities[priority_indices]
        width = width[priority_indices]
        contact_depths = contact_depths[priority_indices]
        print(f"Applied diversity clustering with {num_clusters} clusters")

    elif args.center_contact_depth is not None:
        depth_distances = np.abs(contact_depths - args.center_contact_depth)
        epsilon = 0.001
        inverse_distances = 1.0 / (depth_distances + epsilon)
        depth_weights = np.power(inverse_distances, args.contact_depth_bias)
        depth_weights = depth_weights / np.sum(depth_weights)

        num_grasps = len(transforms)
        priority_indices = np.random.choice(
            num_grasps, size=num_grasps, replace=False, p=depth_weights
        )

        transforms = transforms[priority_indices]
        qualities = qualities[priority_indices]
        width = width[priority_indices]
        contact_depths = contact_depths[priority_indices]
        print(f"Prioritized grasps by contact depth centered at {args.center_contact_depth}")

    joint_info_override = primary_joint

    if use_viewer:
        successful_transforms = []
        successful_qualities = []
        successful_widths = []

        pbar = tqdm(
            enumerate(zip(transforms, qualities)),
            total=len(transforms),
            desc="Testing grasps (0/0 successful)",
        )
        for i, (transform, quality) in pbar:
            pos = transform[:3, 3]
            quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)

            approach_distance = args.approach_distance
            approach_vector = transform[:3, 2] * approach_distance
            approach_pos = pos - approach_vector

            tree = ET.ElementTree(ET.fromstring(xml_content))
            root = tree.getroot()
            gripper_base = root.find(".//body[@name='base']")
            if gripper_base is not None:
                gripper_base.set("pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}")
                gripper_base.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
            target_ee_pose = root.find(".//body[@name='target_ee_pose']")
            if target_ee_pose is not None:
                target_ee_pose.set("pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}")
                target_ee_pose.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")

            for geom in root.findall(".//geom"):
                if geom.get("name", "").startswith("traj_sphere_"):
                    parent = geom.getparent() if hasattr(geom, "getparent") else None
                    if parent is not None:
                        parent.remove(geom)
                    else:
                        root.remove(geom)

            joint_info = joint_info_override
            if not joint_info or (
                isinstance(joint_info, dict)
                and "name" not in joint_info
                and "primary_joint" not in joint_info
            ):
                print(
                    f"[ERROR] No valid joint axis info for joint '{object_name}' or missing 'name'. Skipping this grasp."
                )
                continue

            i, transform_result, quality_result = test_single_grasp(
                (i, transform, quality, args),
                object_name,
                xml_content,
                args,
                primary_joint=joint_info_override,
                render=True,
                handle_geoms=handle_geoms,
            )

            if transform_result is not None:
                successful_transforms.append(transform_result)
                successful_qualities.append(quality_result)
                successful_widths.append(0.1)

            pbar.set_description(
                f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
            )

            if args.max_successful > 0 and len(successful_transforms) >= args.max_successful:
                print(
                    f"\nReached maximum successful grasps ({args.max_successful}). Stopping early."
                )
                break

        return successful_transforms, successful_qualities, successful_widths

    else:
        config = {
            "approach_distance": args.approach_distance,
            "approach_steps": args.approach_steps,
        }

        grasp_params = [
            (i, transform, quality, config)
            for i, (transform, quality) in enumerate(zip(transforms, qualities))
        ]

        num_groups = 4
        group_size = (len(grasp_params) + num_groups - 1) // num_groups
        grasp_groups = [
            grasp_params[i * group_size : (i + 1) * group_size] for i in range(num_groups)
        ]

        successful_transforms = []
        successful_qualities = []
        successful_widths = []

        total_to_process = len(grasp_params)
        pbar = tqdm(total=total_to_process, desc="Testing grasps (0/0 successful)")
        processed_count = 0
        success_count = 0

        for _, grasp_params_batch in enumerate(grasp_groups):
            if not grasp_params_batch:
                continue
            num_workers = min(args.num_workers, len(grasp_params_batch))
            with mp.Pool(processes=num_workers) as pool:
                results = [
                    pool.apply_async(
                        test_single_grasp,
                        args=(
                            param,
                            object_name,
                            xml_content,
                            args,
                            handle_geoms,
                            primary_joint,
                        ),
                    )
                    for param in grasp_params_batch
                ]

                for r in results:
                    try:
                        result = r.get()
                        i, transform_result, quality_result = result
                        processed_count += 1
                        if transform_result is not None:
                            success_count += 1
                            successful_transforms.append(transform_result)
                            successful_qualities.append(quality_result)
                            successful_widths.append(0.1)
                        pbar.set_description(
                            f"Testing grasps ({success_count}/{processed_count} successful)"
                        )
                        pbar.update(1)
                        if args.max_successful > 0 and success_count >= args.max_successful:
                            tqdm.write(
                                f"Reached maximum successful grasps ({args.max_successful}). Stopping early."
                            )
                            pool.terminate()
                            break
                    except Exception as e:
                        tqdm.write(f"Worker error: {str(e)}")
                pool.close()
                pool.join()
            if args.max_successful > 0 and success_count >= args.max_successful:
                break

        pbar.close()
        return (
            successful_transforms,
            successful_qualities,
            successful_widths,
        )


def main_single_file_filtering(
    grasps_path, object_name, xml_file, args, joint_axis_info=None, handle_geoms=None
):
    xml_path = os.path.join(
        os.path.dirname(__file__),
        f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/molmo_spaces/grasp_generation/main_scene.xml",
    )
    tree = ET.parse(xml_path)
    root = tree.getroot()

    try:
        with open(xml_file, "r") as f:
            obj_xml_content = f.read()
        obj_tree = ET.fromstring(obj_xml_content)
        free_joints = obj_tree.findall(".//joint[@type='free']")
        if free_joints:
            for joint in free_joints:
                for parent in obj_tree.findall(".//*"):
                    for child in parent.findall("joint"):
                        if child.get("name") == joint.get("name") and child.get("type") == "free":
                            parent.remove(child)
                            print(f"Removed free joint: {joint.get('name')}")
        with open(xml_file, "w") as f:
            f.write(ET.tostring(obj_tree, encoding="unicode"))
        include = ET.Element("include", {"file": xml_file})
    except Exception as e:
        print(f"Error modifying XML to remove free joints: {e}")
        include = ET.Element("include", {"file": xml_file})
    root.append(include)
    if joint_axis_info is not None:
        primary_joint = joint_axis_info
    else:
        print(
            f"[ERROR] No joint axis info provided for joint '{object_name}'. Skipping all grasps for this joint."
        )
        return 0, None

    xml_content = ET.tostring(root, encoding="unicode")
    robot_xml_path = str(ASSETS_DIR / "robots/floating_robotiq/model_articulate.xml")

    with open(robot_xml_path, "r") as f:
        robot_xml_content = f.read()
    xml_content = merge_xml_contents(xml_content, robot_xml_content)

    try:
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)

        successful_transforms, successful_qualities, successful_widths = run_simulation_with_viewer(
            model,
            data,
            xml_content,
            object_name,
            args.render,
            args,
            primary_joint=primary_joint,
            handle_geoms=handle_geoms,
        )
    except Exception as e:
        print(f"Error during filtering process: {e}")
        raise

    output_path_npz = grasps_path.replace(".json", "_filtered.npz")
    transforms_array = np.array(successful_transforms, dtype=np.float16)
    if len(successful_transforms) != 0:
        np.savez_compressed(output_path_npz, transforms=transforms_array)

    output_path_json = grasps_path.replace(".json", "_object_info.json")
    with open(grasps_path, "r") as original_f:
        original_data = json.load(original_f)
    with open(output_path_json, "w") as f:
        json.dump(
            {
                "object": original_data.get("object", "unknown_object"),
                "object_scale": original_data.get("object_scale", 1.0),
                "object_position": original_data.get("object_position", [0, 0, 0]),
                "object_rotation": original_data.get("object_rotation", [1, 0, 0, 0]),
                "approach_distance": args.approach_distance,
                "num_grasps": len(successful_transforms),
            },
            f,
            indent=2,
        )
    return len(successful_transforms), output_path_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_name", type=str)
    parser.add_argument("--grasps_path", type=str, help="Path to single grasps file (legacy mode)")
    parser.add_argument("--gripper", type=str, default="robotiq")
    parser.add_argument("--xml_file", type=str)
    parser.add_argument(
        "--per_joint_summary_json",
        type=str,
        default=None,
        help="Path to summary JSON mapping joints to handle meshes and grasp files",
    )
    parser.add_argument("--approach_distance", type=float, default=0.1)
    parser.add_argument("--approach_steps", type=int, default=1000)
    parser.add_argument("--articulation_loops", type=int, default=1)
    parser.add_argument("--waypoint_pause", type=float, default=0.025)
    parser.add_argument("--endpoint_pause", type=float, default=0.2)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    parser.add_argument("--max_successful", type=int, default=0)
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument(
        "--min_contact_depth",
        type=float,
        default=0.0,
        help="Minimum contact depth (0.0=base, 1.0=tip). Only test grasps with contact depth >= this value (default: 0.0)",
    )
    parser.add_argument(
        "--max_contact_depth",
        type=float,
        default=1.0,
        help="Maximum contact depth (0.0=base, 1.0=tip). Only test grasps with contact depth <= this value (default: 1.0)",
    )
    parser.add_argument(
        "--center_contact_depth",
        type=float,
        default=None,
        help="Center contact depth (0.0=base, 1.0=tip). Prioritize testing grasps closer to this depth value first (default: None = no prioritization)",
    )
    parser.add_argument(
        "--contact_depth_bias",
        type=float,
        default=2.0,
        help="Strength of bias towards center_contact_depth. Higher = stricter (1.0 = linear, 2.0 = squared, 0.5 = weak bias) (default: 2.0)",
    )
    parser.add_argument(
        "--diversity_mode",
        action="store_true",
        help="Use clustering-based diversity when testing grasps (default: False)",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=40,
        help="Number of position-rotation clusters for diversity (default: 40)",
    )
    args = parser.parse_args()

    # Convert xml_file to absolute path BEFORE chdir if provided
    if args.xml_file and not os.path.isabs(args.xml_file):
        args.xml_file = os.path.abspath(args.xml_file)

    if args.per_joint_summary_json:
        summary_path = os.path.abspath(args.per_joint_summary_json)
        os.chdir(os.path.dirname(summary_path))
        with open(summary_path, "r") as f:
            summary = json.load(f)
        updated_summary = []
        for entry in summary:
            joint = entry.get("joint")
            if args.filtered:
                grasps_file = entry.get("filtered_grasps_file")
            else:
                grasps_file = entry.get("grasps_file")
            xml_file = entry.get("xml_file") if entry.get("xml_file") else args.xml_file

            # Convert xml_file to absolute path BEFORE any processing
            # This ensures it works correctly after os.chdir
            if xml_file and not os.path.isabs(xml_file):
                xml_file = os.path.abspath(xml_file)

            joint_info = (
                entry.get("primary_joint") or entry.get("joint_axis") or entry.get("joint_info")
            )
            handle_geoms = entry.get("handle_geoms", [])
            if not (joint and grasps_file and os.path.exists(grasps_file)):
                print(f"Skipping joint {joint}: missing grasps file {grasps_file}")
                continue
            filter_args = args
            filter_args.grasps_path = grasps_file
            filter_args.object_name = joint if joint else args.object_name
            filter_args.xml_file = xml_file

            try:
                num_grasps, output_path = main_single_file_filtering(
                    grasps_file,
                    filter_args.object_name,
                    filter_args.xml_file,
                    filter_args,
                    joint_axis_info=joint_info,
                    handle_geoms=handle_geoms,
                )
                entry["filtered_grasps_file"] = output_path
                entry["filtered_object_info"] = grasps_file.replace(".json", "_object_info.json")
            except Exception as e:
                print(f"  Error filtering grasps for joint {joint}: {e}")
                traceback.print_exc()
                entry["filtered_grasps_file"] = None
                entry["filtered_object_info"] = None
            updated_summary.append(entry)

        any_success = any(
            entry.get("filtered_grasps_file") is not None for entry in updated_summary
        )

        if any_success:
            summary_out = summary_path.replace(".json", "_filtered.json")
            with open(summary_out, "w") as f:
                json.dump(updated_summary, f, indent=2)
            print(f"\nFiltered summary written to: {summary_out}")
        else:
            print("\nNo joints successfully filtered. Skipping filtered summary creation.")
        return


if __name__ == "__main__":
    main()
