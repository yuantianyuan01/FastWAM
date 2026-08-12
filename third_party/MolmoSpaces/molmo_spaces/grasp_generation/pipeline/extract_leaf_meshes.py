import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tra


def quaternion_to_matrix(quat):
    if len(quat) == 4:
        w, x, y, z = quat
    else:
        raise ValueError("Quaternion must have 4 elements")
    return tra.quaternion_matrix([w, x, y, z])


def parse_mujoco_xml(xml_path, handle_geoms_only=False, target_geoms=None):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_dir = Path(xml_path).parent
    meshes = {}
    assets = root.find("asset")
    if assets is not None:
        for mesh in assets.findall("mesh"):
            name = mesh.get("name")
            file_path = mesh.get("file")
            scale = mesh.get("scale", "1 1 1")
            scale_values = [float(x) for x in scale.split()]
            if len(scale_values) == 1:
                scale_matrix = np.diag([scale_values[0]] * 3 + [1])
            else:
                scale_matrix = np.diag(scale_values + [1])
            meshes[name] = {"file": file_path, "scale_matrix": scale_matrix}
    mesh_instances = []

    def parse_body(body_elem, parent_transform=None):
        if parent_transform is None:
            parent_transform = np.eye(4)
        pos = body_elem.get("pos", "0 0 0")
        quat = body_elem.get("quat", "1 0 0 0")
        pos_values = [float(x) for x in pos.split()]
        pos_matrix = tra.translation_matrix(pos_values)
        quat_values = [float(x) for x in quat.split()]
        quat_matrix = quaternion_to_matrix(quat_values)
        body_transform = np.dot(parent_transform, np.dot(pos_matrix, quat_matrix))
        for geom in body_elem.findall("geom"):
            if geom.get("type") == "mesh":
                geom_name = geom.get("name", f"geom_{len(mesh_instances)}")
                if handle_geoms_only and target_geoms is not None:
                    if geom_name not in target_geoms:
                        continue
                mesh_name = geom.get("mesh")
                if mesh_name in meshes:
                    geom_pos = geom.get("pos", "0 0 0")
                    geom_quat = geom.get("quat", "1 0 0 0")
                    geom_pos_values = [float(x) for x in geom_pos.split()]
                    geom_pos_matrix = tra.translation_matrix(geom_pos_values)
                    geom_quat_values = [float(x) for x in geom_quat.split()]
                    geom_quat_matrix = quaternion_to_matrix(geom_quat_values)
                    final_transform = np.dot(
                        body_transform, np.dot(geom_pos_matrix, geom_quat_matrix)
                    )
                    mesh_instances.append(
                        {
                            "mesh_name": mesh_name,
                            "file": meshes[mesh_name]["file"],
                            "scale_matrix": meshes[mesh_name]["scale_matrix"],
                            "transform": final_transform,
                            "geom_name": geom_name,
                        }
                    )
        for child_body in body_elem.findall("body"):
            parse_body(child_body, body_transform)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody.findall("body"):
            parse_body(body)
    return mesh_instances, xml_dir


def load_and_transform_mesh(mesh_info, xml_dir):
    file_path = xml_dir / mesh_info["file"]
    if not file_path.exists():
        print(f"Warning: Mesh file not found: {file_path}")
        return None
    try:
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        mesh.apply_transform(mesh_info["scale_matrix"])
        mesh.apply_transform(mesh_info["transform"])
        return mesh
    except Exception as e:
        print(f"Error loading mesh {file_path}: {e}")
        return None


def extract_joint_info_from_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        joints = []

        def parse_body_for_joints(body_elem, parent_transform=None, body_path=""):
            if parent_transform is None:
                parent_transform = np.eye(4)
            pos = body_elem.get("pos", "0 0 0")
            quat = body_elem.get("quat", "1 0 0 0")
            pos_values = [float(x) for x in pos.split()]
            pos_matrix = tra.translation_matrix(pos_values)
            quat_values = [float(x) for x in quat.split()]
            quat_matrix = quaternion_to_matrix(quat_values)
            body_transform = np.dot(parent_transform, np.dot(pos_matrix, quat_matrix))
            body_name = body_elem.get("name", "unnamed_body")
            current_path = f"{body_path}/{body_name}" if body_path else body_name
            for joint in body_elem.findall("joint"):
                joint_info = {
                    "name": joint.get("name", "unnamed"),
                    "type": joint.get("type", "unknown"),
                    "axis": joint.get("axis", "0 0 1"),
                    "range": joint.get("range", None),
                    "pos": joint.get("pos", "0 0 0"),
                    "limited": joint.get("limited", "false"),
                    "damping": joint.get("damping", "0"),
                    "frictionloss": joint.get("frictionloss", "0"),
                    "parent_body": body_name,
                    "body_hierarchy": current_path,
                }
                joint_pos = joint.get("pos", "0 0 0")
                joint_pos_values = [float(x) for x in joint_pos.split()]
                joint_pos_matrix = tra.translation_matrix(joint_pos_values)
                global_joint_transform = np.dot(body_transform, joint_pos_matrix)
                global_position = global_joint_transform[:3, 3]
                joint_info["position"] = {
                    "x": float(global_position[0]),
                    "y": float(global_position[1]),
                    "z": float(global_position[2]),
                }
                axis_str = joint.get("axis", "0 0 1")
                local_axis = np.array([float(x) for x in axis_str.split()])
                if np.linalg.norm(local_axis) > 0:
                    local_axis = local_axis / np.linalg.norm(local_axis)
                global_axis = body_transform[:3, :3].dot(local_axis)
                joint_info["rotation_axis"] = {
                    "x": float(global_axis[0]),
                    "y": float(global_axis[1]),
                    "z": float(global_axis[2]),
                }
                joint_info["local_position"] = {
                    "x": joint_pos_values[0] if len(joint_pos_values) > 0 else 0.0,
                    "y": joint_pos_values[1] if len(joint_pos_values) > 1 else 0.0,
                    "z": joint_pos_values[2] if len(joint_pos_values) > 2 else 0.0,
                }
                joint_info["local_axis"] = {
                    "x": float(local_axis[0]),
                    "y": float(local_axis[1]),
                    "z": float(local_axis[2]),
                }
                joint_info["parent_position"] = {
                    "x": float(body_transform[0, 3]),
                    "y": float(body_transform[1, 3]),
                    "z": float(body_transform[2, 3]),
                }
                body_quaternion = tra.quaternion_from_matrix(body_transform)
                joint_info["parent_rotation"] = {
                    "w": float(body_quaternion[0]),
                    "x": float(body_quaternion[1]),
                    "y": float(body_quaternion[2]),
                    "z": float(body_quaternion[3]),
                }
                joints.append(joint_info)
            for child_body in body_elem.findall("body"):
                parse_body_for_joints(child_body, body_transform, current_path)

        worldbody = root.find("worldbody")
        if worldbody is not None:
            for body in worldbody.findall("body"):
                parse_body_for_joints(body)
            for joint in worldbody.findall("joint"):
                joint_info = {
                    "name": joint.get("name", "unnamed"),
                    "type": joint.get("type", "unknown"),
                    "axis": joint.get("axis", "0 0 1"),
                    "range": joint.get("range", None),
                    "pos": joint.get("pos", "0 0 0"),
                    "limited": joint.get("limited", "false"),
                    "damping": joint.get("damping", "0"),
                    "frictionloss": joint.get("frictionloss", "0"),
                    "parent_body": "worldbody",
                    "body_hierarchy": "worldbody",
                }
                joint_pos = joint.get("pos", "0 0 0")
                joint_pos_values = [float(x) for x in joint_pos.split()]
                joint_info["position"] = {
                    "x": joint_pos_values[0] if len(joint_pos_values) > 0 else 0.0,
                    "y": joint_pos_values[1] if len(joint_pos_values) > 1 else 0.0,
                    "z": joint_pos_values[2] if len(joint_pos_values) > 2 else 0.0,
                }
                joint_info["local_position"] = joint_info["position"].copy()
                axis_str = joint.get("axis", "0 0 1")
                axis_values = [float(x) for x in axis_str.split()]
                joint_info["rotation_axis"] = {
                    "x": axis_values[0] if len(axis_values) > 0 else 0.0,
                    "y": axis_values[1] if len(axis_values) > 1 else 0.0,
                    "z": axis_values[2] if len(axis_values) > 2 else 1.0,
                }
                joint_info["local_axis"] = joint_info["rotation_axis"].copy()
                joint_info["parent_position"] = {"x": 0.0, "y": 0.0, "z": 0.0}
                joint_info["parent_rotation"] = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
                joints.append(joint_info)
        return joints
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return []
    except Exception as e:
        print(f"Error reading XML file: {e}")
        return []


def select_primary_joint(joints):
    if not joints:
        return None
    non_free = [j for j in joints if j.get("type", "").lower() != "free"]
    if not non_free:
        return None
    if len(non_free) == 1:
        return non_free[0]
    preferred = [
        j
        for j in non_free
        if any(
            x in j.get("name", "").lower() or x in j.get("type", "").lower()
            for x in ["hinge", "slide", "revolute"]
        )
    ]
    if preferred:
        return preferred[0]
    return non_free[0]


def combine_meshes_to_obj(
    xml_path, output_handles_path, output_full_path, include_visual_only=True
):
    print(f"Parsing MuJoCo XML: {xml_path}")
    print("Analyzing XML structure to identify handle components for each joint...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    handle_meshes_info = []

    if worldbody is not None:

        def collect_mesh_geoms_with_depth(body_elem, current_depth=0):
            geoms_with_depth = []
            for geom in body_elem.findall("geom"):
                if geom.get("type") == "mesh":
                    geom_name = geom.get("name")
                    mesh_name = geom.get("mesh")
                    if geom_name and mesh_name:
                        geoms_with_depth.append((geom_name, current_depth))
            for child_body in body_elem.findall("body"):
                geoms_with_depth.extend(
                    collect_mesh_geoms_with_depth(child_body, current_depth + 1)
                )
            return geoms_with_depth

        def process_body(body_elem):
            for joint in body_elem.findall("joint"):
                joint_type = joint.get("type", "hinge")
                if joint_type == "free":
                    continue

                joint_name = joint.get("name", "unnamed")
                geoms_with_depth = collect_mesh_geoms_with_depth(body_elem, 0)

                if geoms_with_depth:
                    max_depth = max([depth for _, depth in geoms_with_depth])
                else:
                    max_depth = None

                max_depth_geoms = [geom for geom, depth in geoms_with_depth if depth == max_depth]

                print(f"Joint '{joint_name}' handle geoms at max depth {max_depth}:")
                for geom in max_depth_geoms:
                    print(f"  - {geom}")

                mesh_instances, xml_dir = parse_mujoco_xml(
                    xml_path, handle_geoms_only=True, target_geoms=max_depth_geoms
                )

                transformed_meshes = []
                for mesh_info in mesh_instances:
                    if include_visual_only and "Collider" in mesh_info["geom_name"]:
                        continue
                    transformed_mesh = load_and_transform_mesh(mesh_info, xml_dir)
                    if transformed_mesh is not None:
                        transformed_meshes.append(transformed_mesh)

                if transformed_meshes:
                    combined_mesh = trimesh.util.concatenate(transformed_meshes)
                    safe_joint_name = joint_name.replace("/", "_").replace(" ", "_")
                    handles_path = output_handles_path.parent / f"{safe_joint_name}.obj"
                    combined_mesh.export(handles_path)
                    print(f"Exported handle mesh for joint {joint_name} to: {handles_path}")

                    handle_meshes_info.append(
                        {
                            "joint": joint_name,
                            "handle_mesh": str(handles_path.name),
                            "handle_geoms": max_depth_geoms,
                        }
                    )
                else:
                    print(f"No valid handle meshes found for joint {joint_name}")

            for child_body in body_elem.findall("body"):
                process_body(child_body)

        for body in worldbody.findall("body"):
            process_body(body)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    mesh_file_map = {}
    assets = root.find("asset")

    if assets is not None:
        for mesh in assets.findall("mesh"):
            mesh_name = mesh.get("name")
            mesh_file = mesh.get("file", "")
            mesh_file_map[mesh_name] = mesh_file

    all_obj_geoms = set()
    for body in root.findall(".//body"):
        for geom in body.findall("geom"):
            if geom.get("type") == "mesh":
                mesh_name = geom.get("mesh", "")
                mesh_file = mesh_file_map.get(mesh_name, "")
                if mesh_file and mesh_file.lower().endswith(".obj"):
                    geom_name = geom.get("name", mesh_name)
                    all_obj_geoms.add(geom_name)

    mesh_instances, xml_dir = parse_mujoco_xml(
        xml_path, handle_geoms_only=True, target_geoms=list(all_obj_geoms)
    )

    transformed_meshes = []
    for mesh_info in mesh_instances:
        if include_visual_only and "Collider" in mesh_info["geom_name"]:
            continue
        transformed_mesh = load_and_transform_mesh(mesh_info, xml_dir)
        if transformed_mesh is not None:
            transformed_meshes.append(transformed_mesh)

    if transformed_meshes:
        combined_mesh = trimesh.util.concatenate(transformed_meshes)
        main_mesh_path = output_full_path.parent / "main.obj"
        combined_mesh.export(main_mesh_path)
        print(f"Exported full mesh to: {main_mesh_path}")
    else:
        print("No valid full meshes found to combine")

    joints = extract_joint_info_from_xml(xml_path)
    handle_meshes_json_path = str(output_full_path.parent / "joint_meshes_info.json")

    for entry in handle_meshes_info:
        joint_name = entry["joint"]
        joint_info = next((j for j in joints if j.get("name") == joint_name), None)
        entry["joint_info"] = joint_info

    with open(handle_meshes_json_path, "w") as f:
        json.dump(handle_meshes_info, f, indent=2)
    print(f"Per-joint handle mesh mapping saved to: {handle_meshes_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine meshes from MuJoCo XML into handle-only and full OBJ files."
    )
    parser.add_argument("xml_file", help="Path to MuJoCo XML file")
    parser.add_argument(
        "output_prefix",
        help="Output OBJ file prefix (will create _handles.obj and _full.obj)",
    )
    parser.add_argument(
        "--include-collision",
        action="store_true",
        help="Include collision geometries (default: visual only)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    xml_path = Path(args.xml_file)
    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}")
        return 1
    output_prefix = str(args.output_prefix)
    if output_prefix.endswith(".obj"):
        output_prefix = output_prefix[:-4]
    if output_prefix.endswith("_handles"):
        output_prefix = output_prefix[:-8]
    if output_prefix.endswith("_full"):
        output_prefix = output_prefix[:-5]
    output_handles_path = Path(f"{output_prefix}_handles.obj")
    output_full_path = Path(f"{output_prefix}_full.obj")
    output_handles_path.parent.mkdir(parents=True, exist_ok=True)
    output_full_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        combine_meshes_to_obj(
            xml_path,
            output_handles_path,
            output_full_path,
            include_visual_only=not args.include_collision,
        )
        print("Success! Handle-only and full object meshes created.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
