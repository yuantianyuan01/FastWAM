import argparse
import json
import multiprocessing as mp
import sys
import time
import xml.etree.ElementTree as ET
from functools import partial

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from molmo_spaces.molmo_spaces_constants import ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR, ASSETS_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--object_name", type=str)
parser.add_argument("--grasps_path", type=str)
parser.add_argument("--xml_file", type=str)
parser.add_argument("--num_shakes", type=int, default=2)
parser.add_argument("--shake_magnitude", type=float, default=0.1)
parser.add_argument("--shake_steps", type=int, default=1000)
parser.add_argument(
    "--approach_distance",
    type=float,
    default=0.1,
    help="Distance in meters for the gripper to approach from before grasping",
)
parser.add_argument(
    "--approach_steps",
    type=int,
    default=1000,
    help="Number of simulation steps for the approach phase",
)
parser.add_argument("--render", action="store_true", help="Enable interactive viewer")
parser.add_argument("--rotate", action="store_true", help="Enable rotation shaking")
parser.add_argument(
    "--num_workers", type=int, default=mp.cpu_count(), help="Number of parallel processes to use"
)
parser.add_argument(
    "--max_successful",
    type=int,
    default=0,
    help="Stop after finding this many successful grasps (0 = process all grasps)",
)
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
    help="Number of position-rotation clusters for diversity (default: 20)",
)
args = parser.parse_args()

initial_relative_position = None
initial_grasp_verified = False

if args.render:
    args.num_workers = 1


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
        if (
            object_name.lower() in geom1.lower()
            or object_name.lower() in geom2.lower()
            or "collider" in geom1.lower()
            or "collider" in geom2.lower()
        ):
            other = geom2 if object_name.lower() in geom1.lower() else geom1
            if any(p in other.lower() for p in left_patterns):
                left_finger_contact = True
            if any(p in other.lower() for p in right_patterns):
                right_finger_contact = True
    grasped = left_finger_contact and right_finger_contact
    return grasped


def check_grasp(model, data, object_name, store_initial=False):
    global initial_relative_position, initial_grasp_verified
    object_pos = data.body(object_name).xpos
    gripper_pos = data.body("base").xpos
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
    grasping = is_object_grasped(model, data, object_name)
    return grasping


def test_single_grasp(grasp_data, object_name, model=None, data=None, viewer=None):
    i, transform, quality, config = grasp_data
    pos = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)
    approach_distance = config.get("approach_distance", 0.1)
    approach_vector = transform[:3, 2] * approach_distance
    approach_pos = pos - approach_vector
    if model is None or data is None:
        xml_path = (
            f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/molmo_spaces/grasp_generation/main_scene.xml"
        )
        tree = ET.parse(xml_path)
        root = tree.getroot()
        include = ET.Element("include", {"file": args.xml_file})
        root.append(include)
        xml_content = ET.tostring(root, encoding="unicode")
        gripper_xml_path = str(ASSETS_DIR / "robots/floating_robotiq/model_rigid.xml")
        with open(gripper_xml_path, "r") as f:
            additional_xml_content = f.read()
        xml_content = merge_xml_contents(xml_content, additional_xml_content)
        tree = ET.ElementTree(ET.fromstring(xml_content))
        root = tree.getroot()
        for body in root.findall(".//body"):
            if body.get("name") == "base":
                org_rot = R.from_matrix(transform[:3, :3])
                rot_new = org_rot
                new_quat = rot_new.as_quat(scalar_first=True)
                body.set("pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}")
                body.set("quat", f"{new_quat[0]} {new_quat[1]} {new_quat[2]} {new_quat[3]}")
                break
        for body in root.findall(".//body"):
            if body.get("name") == "target_ee_pose":
                org_rot = R.from_matrix(transform[:3, :3])
                rot_new = org_rot
                new_quat = rot_new.as_quat(scalar_first=True)
                body.set("pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}")
                body.set("quat", f"{new_quat[0]} {new_quat[1]} {new_quat[2]} {new_quat[3]}")
                break
        for body in root.findall(".//geom"):
            if body.get("name") == "test_sphere":
                body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
                body.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
                break
        xml_content = ET.tostring(root, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
    global initial_relative_position, initial_grasp_verified
    initial_relative_position = None
    initial_grasp_verified = False
    data.ctrl[model.actuator("fingers_actuator").id] = 0.0
    mujoco.mj_step(model, data, nstep=500)
    if viewer:
        viewer.sync()
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_ee_pose")
    approach_steps = config.get("approach_steps", 1000)
    for step in range(approach_steps):
        alpha = step / approach_steps
        current_target_pos = approach_pos + alpha * (pos - approach_pos)
        if mocap_id >= 0:
            data.mocap_pos[0] = current_target_pos
            data.mocap_quat[0] = quat
        mujoco.mj_step(model, data)
        if viewer and step % 50 == 0:
            viewer.sync()
    if mocap_id >= 0:
        data.mocap_pos[0] = pos
    mujoco.mj_step(model, data, nstep=300)
    if viewer:
        viewer.sync()
    data.ctrl[model.actuator("fingers_actuator").id] = 255.0
    mujoco.mj_step(model, data, nstep=300)
    # while 1:
    #     mujoco.mj_step(model, data)
    #     viewer.sync()
    if viewer:
        viewer.sync()
    tcp_pose = np.eye(4)
    tcp_pose[:3, 3] = data.site("grasp_site").xpos
    tcp_pose[:3, :3] = data.site("grasp_site").xmat.reshape(3, 3)
    tcp_pose = np.eye(4)
    tcp_pose[:3, 3] = data.site("grasp_site").xpos
    tcp_pose[:3, :3] = data.site("grasp_site").xmat.reshape(3, 3)
    object_pose = np.eye(4)
    object_pose[:3, :3] = data.body(object_name).xmat.reshape(3, 3)
    object_pose[:3, 3] = data.body(object_name).xpos
    transform = np.linalg.inv(object_pose) @ tcp_pose
    if not check_grasp(model, data, object_name, store_initial=True):
        return i, None, None
    directions = ["x", "y", "z"]
    shake_success = True
    if mocap_id >= 0:
        baseline_pos = data.mocap_pos[0].copy()
        baseline_quat = data.mocap_quat[0].copy()
    else:
        baseline_pos = pos.copy()
        baseline_quat = quat.copy()

    for direction_idx, _ in enumerate(directions):
        for _ in range(config["num_shakes"]):
            total_steps = config["shake_steps"] * 2
            for step in range(total_steps):
                angle = 2 * np.pi * step / total_steps
                shake_offset = config["shake_magnitude"] * np.sin(angle)
                shake_pos = baseline_pos.copy()
                shake_pos[direction_idx] += shake_offset
                if mocap_id >= 0:
                    data.mocap_pos[0] = shake_pos
                mujoco.mj_step(model, data)
                if viewer and step % 5 == 0:
                    viewer.sync()
                    if not viewer.is_running():
                        return i, None, None
                if step == total_steps // 4 or step == 3 * total_steps // 4:
                    grasp_maintained = check_grasp(model, data, object_name)
                    if not grasp_maintained:
                        shake_success = False
                        break
            if not shake_success:
                break
            if mocap_id >= 0:
                data.mocap_pos[0] = baseline_pos
            for step in range(50):
                mujoco.mj_step(model, data)
                if viewer and step % 20 == 0:
                    viewer.sync()
                    if not viewer.is_running():
                        return i, None, None
        if not shake_success:
            break

    if args.rotate and shake_success:
        if shake_success and mocap_id >= 0:
            baseline_rotation = R.from_quat(baseline_quat[[1, 2, 3, 0]])
            for axis_idx in range(2, 3):
                for _shake in range(1):
                    total_steps = config["shake_steps"] * 5
                    for step in range(total_steps):
                        angle = 2 * np.pi * step / total_steps
                        shake_angle = angle
                        rotation_vec = np.zeros(3)
                        rotation_vec[axis_idx] = shake_angle
                        shake_rotation = R.from_rotvec(rotation_vec)
                        new_rotation = baseline_rotation * shake_rotation
                        new_quat_scipy = new_rotation.as_quat()
                        new_quat_mujoco = np.array(
                            [
                                new_quat_scipy[3],
                                new_quat_scipy[0],
                                new_quat_scipy[1],
                                new_quat_scipy[2],
                            ]
                        )
                        data.mocap_quat[0] = new_quat_mujoco
                        mujoco.mj_step(model, data)
                        if viewer and step % 5 == 0:
                            viewer.sync()
                            if not viewer.is_running():
                                return i, None, None
                        if step == total_steps // 4 or step == 3 * total_steps // 4:
                            grasp_maintained = check_grasp(model, data, object_name)
                            if not grasp_maintained:
                                shake_success = False
                                break
                    if not shake_success:
                        break
                    data.mocap_quat[0] = baseline_quat
                    for step in range(50):
                        mujoco.mj_step(model, data)
                        if viewer and step % 20 == 0:
                            viewer.sync()
                            if not viewer.is_running():
                                return i, None, None
                if not shake_success:
                    break
    final_grasp_check = is_object_grasped(model, data, object_name)
    if shake_success and final_grasp_check:
        return i, transform.tolist(), quality
    else:
        return i, None, None


def run_simulation_with_viewer(xml_content, object_name, use_viewer):
    with open(args.grasps_path, "r") as f:
        grasp_data = json.load(f)
    transforms = np.array(grasp_data["transforms"])
    qualities = np.array(grasp_data.get("quality_antipodal", [1.0] * len(transforms)))
    widths = np.array(grasp_data.get("grasp_widths", [0.05] * len(transforms)))
    contact_depths = np.array(grasp_data.get("contact_depths", [0.5] * len(transforms)))

    if args.min_contact_depth > 0.0 or args.max_contact_depth < 1.0:
        depth_mask = (contact_depths >= args.min_contact_depth) & (
            contact_depths <= args.max_contact_depth
        )
        transforms = transforms[depth_mask]
        qualities = qualities[depth_mask]
        widths = widths[depth_mask]
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
        widths = widths[priority_indices]
        contact_depths = contact_depths[priority_indices]

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
        widths = widths[priority_indices]
        contact_depths = contact_depths[priority_indices]

    config = {
        "num_shakes": args.num_shakes,
        "shake_magnitude": args.shake_magnitude,
        "shake_steps": args.shake_steps,
        "approach_distance": args.approach_distance,
        "approach_steps": args.approach_steps,
    }
    if use_viewer:
        successful_transforms = []
        successful_qualities = []
        successful_widths = []
        pbar = tqdm(
            enumerate(zip(transforms, qualities)),
            total=len(transforms),
            desc="Testing grasps (0/0 successful)",
        )
        viewer_obj = None
        for i, (transform, quality) in pbar:
            grasp_params = (i, transform, quality, config)
            tree = ET.ElementTree(ET.fromstring(xml_content))
            root = tree.getroot()
            pos = transform[:3, 3]
            quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)
            approach_distance = args.approach_distance
            approach_vector = transform[:3, 2] * approach_distance
            approach_pos = pos - approach_vector
            for body in root.findall(".//body"):
                if body.get("name") == "base":
                    org_rot = R.from_matrix(transform[:3, :3])
                    rot_new = org_rot
                    new_quat = rot_new.as_quat(scalar_first=True)
                    body.set("pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}")
                    body.set("quat", f"{new_quat[0]} {new_quat[1]} {new_quat[2]} {new_quat[3]}")
                    break
            for body in root.findall(".//body"):
                if body.get("name") == "target_ee_pose":
                    org_rot = R.from_matrix(transform[:3, :3])
                    rot_new = org_rot
                    new_quat = rot_new.as_quat(scalar_first=True)
                    body.set("pos", f"{approach_pos[0]} {approach_pos[1]} {approach_pos[2]}")
                    body.set("quat", f"{new_quat[0]} {new_quat[1]} {new_quat[2]} {new_quat[3]}")
                    break
            for body in root.findall(".//geom"):
                if body.get("name") == "test_sphere":
                    body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
                    body.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
                    break
            xml_content_updated = ET.tostring(root, encoding="unicode")
            model = mujoco.MjModel.from_xml_string(xml_content_updated)
            data = mujoco.MjData(model)
            if viewer_obj is not None:
                viewer_obj.close()
                del viewer_obj
                time.sleep(0.1)
            with mujoco.viewer.launch_passive(
                model, data, show_left_ui=True, show_right_ui=True
            ) as viewer_obj:
                mujoco.mj_step(model, data)
                viewer_obj.sync()
                result_i, result_transform, result_quality = test_single_grasp(
                    grasp_params, object_name, model=model, data=data, viewer=viewer_obj
                )
                if result_transform is not None:
                    successful_transforms.append(result_transform)
                    successful_qualities.append(result_quality)
                    successful_widths.append(widths[i])
                    if (
                        args.max_successful > 0
                        and len(successful_transforms) >= args.max_successful
                    ):
                        tqdm.write(
                            f"Found {len(successful_transforms)} successful grasps (reached max_successful limit)"
                        )
                        viewer_obj.close()
                        return successful_transforms, successful_qualities, successful_widths
                pbar.set_description(
                    f"Testing grasps ({len(successful_transforms)}/{i + 1} successful)"
                )
        return successful_transforms, successful_widths, successful_qualities
    else:
        grasp_params = [
            (i, transform, quality, config)
            for i, (transform, quality) in enumerate(zip(transforms, qualities))
        ]
        num_workers = min(args.num_workers, len(grasp_params))
        successful_transforms = []
        successful_qualities = []
        successful_widths = []

        success_count = 0
        processed_count = 0

        pbar = tqdm(total=len(grasp_params), desc="Testing grasps (0/0 successful)")

        test_func = partial(
            test_single_grasp, object_name=object_name, model=None, data=None, viewer=None
        )

        with mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(test_func, grasp_params):
                i, transform_result, quality_result = result
                processed_count += 1

                if transform_result is not None:
                    success_count += 1
                    successful_transforms.append((i, transform_result, quality_result, widths[i]))
                    if args.max_successful > 0 and success_count >= args.max_successful:
                        tqdm.write(
                            f"Found {success_count} successful grasps (reached max_successful limit)"
                        )
                        pbar.update(1)
                        break

                pbar.set_description(
                    f"Testing grasps ({success_count}/{processed_count} successful)"
                )
                pbar.update(1)

        pbar.close()
        sys.stdout.flush()

        successful_transforms.sort()
        successful_transforms_only = [t for _, t, _, _ in successful_transforms]
        successful_qualities_only = [q for _, _, q, _ in successful_transforms]
        successful_widths_only = [w for _, _, _, w in successful_transforms]

        return successful_transforms_only, successful_qualities_only, successful_widths_only


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


if __name__ == "__main__":
    xml_path = (
        f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/molmo_spaces/grasp_generation/main_scene.xml"
    )
    tree = ET.parse(xml_path)
    root = tree.getroot()
    object_name = args.object_name
    include = ET.Element("include", {"file": args.xml_file})
    root.append(include)
    xml_content = ET.tostring(root, encoding="unicode")
    gripper_xml_path = str(ASSETS_DIR / "robots/floating_robotiq/model_rigid.xml")
    with open(gripper_xml_path, "r") as f:
        additional_xml_content = f.read()
    xml_content = merge_xml_contents(xml_content, additional_xml_content)
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    successful_transforms, successful_qualities, successful_widths = run_simulation_with_viewer(
        xml_content, object_name, args.render
    )

    # Save transforms as NPZ with float16 compression
    output_path_npz = args.grasps_path.replace(".json", "_filtered.npz")
    transforms_array = np.array(successful_transforms, dtype=np.float16)
    np.savez_compressed(output_path_npz, transforms=transforms_array)

    # Save metadata as object_info.json
    output_path_json = args.grasps_path.replace(".json", "_object_info.json")
    with open(args.grasps_path, "r") as original_f:
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

    tqdm.write(f"Saved {len(successful_transforms)} successful grasps to {output_path_npz}")
    tqdm.write(f"Saved object metadata to {output_path_json}")
    if not args.render and args.num_workers > 1:
        tqdm.write(f"Used {args.num_workers} parallel workers for grasp testing")
