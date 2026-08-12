import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation as R


def parse_mujoco_xml(xml_path):
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
        rot_3x3 = R.from_quat(
            [quat_values[1], quat_values[2], quat_values[3], quat_values[0]]
        ).as_matrix()
        quat_matrix = np.eye(4)
        quat_matrix[:3, :3] = rot_3x3
        body_transform = np.dot(parent_transform, np.dot(pos_matrix, quat_matrix))

        for geom in body_elem.findall("geom"):
            if geom.get("type") == "mesh":
                mesh_name = geom.get("mesh")
                if mesh_name in meshes:
                    geom_pos = geom.get("pos", "0 0 0")
                    geom_quat = geom.get("quat", "1 0 0 0")
                    geom_pos_values = [float(x) for x in geom_pos.split()]
                    geom_pos_matrix = tra.translation_matrix(geom_pos_values)
                    geom_quat_values = [float(x) for x in geom_quat.split()]
                    geom_rot_3x3 = R.from_quat(
                        [
                            geom_quat_values[1],
                            geom_quat_values[2],
                            geom_quat_values[3],
                            geom_quat_values[0],
                        ]
                    ).as_matrix()
                    geom_quat_matrix = np.eye(4)
                    geom_quat_matrix[:3, :3] = geom_rot_3x3
                    final_transform = np.dot(
                        body_transform, np.dot(geom_pos_matrix, geom_quat_matrix)
                    )
                    mesh_instances.append(
                        {
                            "mesh_name": mesh_name,
                            "file": meshes[mesh_name]["file"],
                            "scale_matrix": meshes[mesh_name]["scale_matrix"],
                            "transform": final_transform,
                            "geom_name": geom.get("name", f"geom_{len(mesh_instances)}"),
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


def combine_meshes_to_obj(xml_path, output_path, only_collision=True):
    mesh_instances, xml_dir = parse_mujoco_xml(xml_path)
    if not mesh_instances:
        print("No mesh instances found in XML file")
        return
    transformed_meshes = []
    for mesh_info in mesh_instances:
        if only_collision and "collider" not in mesh_info["geom_name"].lower():
            continue
        transformed_mesh = load_and_transform_mesh(mesh_info, xml_dir)
        if transformed_mesh is not None:
            transformed_meshes.append(transformed_mesh)
    if not transformed_meshes:
        print("No valid meshes found to combine")
        return
    combined_mesh = trimesh.util.concatenate(transformed_meshes)
    combined_mesh.export(output_path)


def main():
    parser = argparse.ArgumentParser(description="Combine meshes from MuJoCo XML into single OBJ")
    parser.add_argument("xml_file", help="Path to MuJoCo XML file")
    parser.add_argument("output_obj", help="Output OBJ file path")
    parser.add_argument(
        "--only_collision", action="store_true", help="Include only collision meshes"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    xml_path = Path(args.xml_file)
    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}")
        return 1
    output_path = Path(args.output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        combine_meshes_to_obj(xml_path, output_path, only_collision=args.only_collision)
        print("Success!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
