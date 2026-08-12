import argparse
import json
import os
import numpy as np
import trimesh
import trimesh.transformations as tra
import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molmo_spaces.grasp_generation.robotiq_gripper import RobotiqGripper

parser = argparse.ArgumentParser(description='Meshcat visualization of generated grasps')
parser.add_argument('--objects_list', type=str, required=True, help='Path to JSON file with object list')
parser.add_argument('--results_dir', type=str, default='results/rigid_objects', help='Results directory')
parser.add_argument('--max_grasps', type=int, default=30, help='Max grasps to display per object')
parser.add_argument('--articulable', action='store_true', help='Visualize articulable objects')
parser.add_argument('--max_grasps_per_joint', type=int, default=10, help='Max grasps per joint for articulable objects')
args = parser.parse_args()

with open(args.objects_list, 'r') as f:
    objects = json.load(f)

available_objects = []
for obj in objects:
    name = obj['name']
    xml_path = obj['xml']

    if args.articulable:
        json_path = os.path.join(args.results_dir, name, 'joint_meshes_info_filtered.json')
        main_mesh_path = os.path.join(args.results_dir, name, 'main.obj')
        if os.path.exists(json_path) and os.path.exists(main_mesh_path):
            available_objects.append({
                'name': name,
                'json': json_path,
                'xml': xml_path,
                'main_mesh': main_mesh_path
            })
    else:
        npz_path = os.path.join(args.results_dir, name, f'{name}_grasps_filtered.npz')
        if os.path.exists(npz_path):
            available_objects.append({'name': name, 'npz': npz_path, 'xml': xml_path})

if not available_objects:
    print("No objects with generated grasps found.")
    exit(1)

print(f"Found {len(available_objects)} objects with grasps")

vis = meshcat.Visualizer()
vis.open()
print(f"Meshcat URL: {vis.url()}")

gripper = RobotiqGripper(root_folder='')

def load_mesh_from_xml(xml_path):
    obj_dir = os.path.dirname(xml_path)
    obj_name = os.path.basename(xml_path).replace('.xml', '')

    candidates = [
        os.path.join(args.results_dir, obj_name, f'{obj_name}_combined.obj'),
        os.path.join(args.results_dir, obj_name, f'{obj_name}_simplified.obj'),
        os.path.join(args.results_dir, obj_name, f'{obj_name}_manifold.obj'),
        os.path.join(obj_dir, obj_name + '.obj'),
        os.path.join(obj_dir, obj_name + '.stl'),
    ]

    for mesh_path in candidates:
        if os.path.exists(mesh_path):
            print(f"  Loading mesh: {mesh_path}")
            return trimesh.load(mesh_path)

    print(f"  Loading mesh from XML: {xml_path}")
    return load_mesh_from_mujoco_xml(xml_path)

def load_mesh_from_mujoco_xml(xml_path):
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

    def parse_body(body_elem, parent_transform=np.eye(4)):
        pos = body_elem.get("pos", "0 0 0")
        quat = body_elem.get("quat", "1 0 0 0")
        pos_values = [float(x) for x in pos.split()]
        pos_matrix = tra.translation_matrix(pos_values)
        quat_values = [float(x) for x in quat.split()]
        rot_3x3 = R.from_quat([quat_values[1], quat_values[2], quat_values[3], quat_values[0]]).as_matrix()
        quat_matrix = np.eye(4)
        quat_matrix[:3, :3] = rot_3x3
        body_transform = np.dot(parent_transform, np.dot(pos_matrix, quat_matrix))

        for geom in body_elem.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name and mesh_name in meshes:
                geom_pos = geom.get("pos", "0 0 0")
                geom_quat = geom.get("quat", "1 0 0 0")
                geom_pos_values = [float(x) for x in geom_pos.split()]
                geom_pos_matrix = tra.translation_matrix(geom_pos_values)
                geom_quat_values = [float(x) for x in geom_quat.split()]
                geom_rot_3x3 = R.from_quat([geom_quat_values[1], geom_quat_values[2], geom_quat_values[3], geom_quat_values[0]]).as_matrix()
                geom_quat_matrix = np.eye(4)
                geom_quat_matrix[:3, :3] = geom_rot_3x3
                final_transform = np.dot(body_transform, np.dot(geom_pos_matrix, geom_quat_matrix))
                mesh_instances.append({
                    "file": meshes[mesh_name]["file"],
                    "scale_matrix": meshes[mesh_name]["scale_matrix"],
                    "transform": final_transform,
                })

        for child_body in body_elem.findall("body"):
            parse_body(child_body, body_transform)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody.findall("body"):
            parse_body(body)

    transformed_meshes = []
    for mesh_info in mesh_instances:
        file_path = xml_dir / mesh_info["file"]
        if file_path.exists():
            try:
                mesh = trimesh.load(str(file_path))
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate([m for m in mesh.geometry.values()])
                mesh.apply_transform(mesh_info["scale_matrix"])
                mesh.apply_transform(mesh_info["transform"])
                transformed_meshes.append(mesh)
            except Exception as e:
                print(f"    Error loading {file_path}: {e}")

    if transformed_meshes:
        return trimesh.util.concatenate(transformed_meshes)
    return None

def display_object(idx):
    vis.delete()

    obj = available_objects[idx]
    name = obj['name']
    xml_path = obj['xml']

    print(f"\n[{idx+1}/{len(available_objects)}] {name}")

    if args.articulable:
        main_mesh_path = obj['main_mesh']
        mesh = trimesh.load(main_mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
        mesh.apply_scale(1)
        vis['object'].set_object(
            g.TriangularMeshGeometry(mesh.vertices, mesh.faces),
            g.MeshLambertMaterial(color=0x888888, opacity=0.7)
        )

        json_path = obj['json']
        with open(json_path, 'r') as f:
            joint_grasps = json.load(f)

        colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff]

        total_grasps = 0
        grasp_idx = 0

        for joint_idx, entry in enumerate(joint_grasps):
            joint = entry['joint']
            filtered_grasps_file = entry.get('filtered_grasps_file')
            joint_info = entry.get('joint_info', {})

            if not filtered_grasps_file:
                continue

            grasps_path = os.path.join(os.path.dirname(json_path), filtered_grasps_file)

            if not os.path.exists(grasps_path):
                print(f"  Warning: {grasps_path} not found")
                continue

            data = np.load(grasps_path)
            transforms_joint_relative = data['transforms'].astype(np.float64)

            joint_position = joint_info.get('position', {'x': 0, 'y': 0, 'z': 0})
            joint_rotation = joint_info.get('parent_rotation', {'w': 1, 'x': 0, 'y': 0, 'z': 0})

            T_world_joint = np.eye(4)
            T_world_joint[:3, 3] = [joint_position['x'], joint_position['y'], joint_position['z']]

            quat = [joint_rotation['x'], joint_rotation['y'], joint_rotation['z'], joint_rotation['w']]
            T_world_joint[:3, :3] = R.from_quat(quat).as_matrix()

            num_grasps = min(len(transforms_joint_relative), args.max_grasps_per_joint)
            total_grasps += len(transforms_joint_relative)

            color = colors[joint_idx % len(colors)]

            print(f"  Joint {joint_idx+1} ({joint}): {num_grasps}/{len(transforms_joint_relative)} grasps")

            gripper_mesh = gripper.hand.copy()
            gripper_mesh.apply_transform(tra.euler_matrix(0, 0, np.pi/2))
            gripper_mesh.apply_translation(-gripper.tcp_offset)

            for i in range(num_grasps):
                T_joint_grasp = transforms_joint_relative[i]
                T_world_grasp = T_world_joint @ T_joint_grasp

                transformed_gripper = gripper_mesh.copy()
                transformed_gripper.apply_transform(T_world_grasp)

                vis[f'grasps/grasp_{grasp_idx}'].set_object(
                    g.TriangularMeshGeometry(transformed_gripper.vertices, transformed_gripper.faces),
                    g.MeshLambertMaterial(color=color, opacity=0.6)
                )
                grasp_idx += 1

        print(f"  Total: {total_grasps} grasps across {len(joint_grasps)} joints")

    else:
        npz_path = obj['npz']

        mesh = load_mesh_from_xml(xml_path)
        if mesh is not None:
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
            mesh.apply_scale(1)
            vis['object'].set_object(
                g.TriangularMeshGeometry(mesh.vertices, mesh.faces),
                g.MeshLambertMaterial(color=0x888888, opacity=0.7)
            )

        data = np.load(npz_path)
        transforms = data['transforms'].astype(np.float64)
        num_grasps = min(len(transforms), args.max_grasps)
        print(f"  Showing {num_grasps}/{len(transforms)} grasps")

        gripper_mesh = gripper.hand.copy()
        gripper_mesh.apply_transform(tra.euler_matrix(0, 0, np.pi/2))
        gripper_mesh.apply_translation(-gripper.tcp_offset)

        colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff]

        for i in range(num_grasps):
            T = transforms[i]
            color = colors[i % len(colors)]

            transformed_gripper = gripper_mesh.copy()
            transformed_gripper.apply_transform(T)

            vis[f'grasps/grasp_{i}'].set_object(
                g.TriangularMeshGeometry(transformed_gripper.vertices, transformed_gripper.faces),
                g.MeshLambertMaterial(color=color, opacity=0.6)
            )

current_idx = 0
display_object(current_idx)

print("\nControls:")
print("  n/right arrow: next object")
print("  p/left arrow:  previous object")
print("  q: quit")
print()

try:
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == 'q':
                break
            elif ch == 'n' or ch == '\x1b':
                if ch == '\x1b':
                    sys.stdin.read(1)
                    arrow = sys.stdin.read(1)
                    if arrow == 'C':
                        ch = 'n'
                    elif arrow == 'D':
                        ch = 'p'
                    else:
                        continue
                if ch == 'n':
                    current_idx = (current_idx + 1) % len(available_objects)
                    display_object(current_idx)
            elif ch == 'p':
                current_idx = (current_idx - 1) % len(available_objects)
                display_object(current_idx)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

except ImportError:
    print("Arrow key navigation not available. Enter object number (0 to quit):")
    while True:
        try:
            idx = int(input(f"Object (1-{len(available_objects)}, 0=quit): "))
            if idx == 0:
                break
            if 1 <= idx <= len(available_objects):
                current_idx = idx - 1
                display_object(current_idx)
        except ValueError:
            pass

print("\nDone.")
