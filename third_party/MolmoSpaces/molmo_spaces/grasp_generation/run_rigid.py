import argparse
import json
import os
import subprocess

import numpy as np

import wandb
from molmo_spaces.molmo_spaces_constants import ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR

# TODO: install floating_robotiq assets via resource manager


parser = argparse.ArgumentParser(description="Process rigid objects for grasp generation")
parser.add_argument(
    "--objects_list",
    type=str,
    default=f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/grasp_results/rigid_objects_list.json",
    help="Path to JSON file with object list",
)
parser.add_argument(
    "--max_successful_grasps", type=int, default=1000, help="Max successful grasps per object"
)
parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers (0 = all CPUs)")
parser.add_argument(
    "--approach_distance", type=float, default=0.3, help="Approach distance for grasp filtering"
)
parser.add_argument(
    "--approach_steps", type=int, default=3000, help="Approach steps for grasp filtering"
)
parser.add_argument(
    "--shake_magnitude", type=float, default=0.1, help="Shake magnitude for perturbation test"
)
parser.add_argument(
    "--shake_steps", type=int, default=1000, help="Shake steps for perturbation test"
)
parser.add_argument("--max_contact_depth", type=float, default=1.0, help="Max contact depth")
parser.add_argument("--min_contact_depth", type=float, default=0.0, help="Min contact depth")
parser.add_argument("--center_contact_depth", type=float, default=0.75, help="Center contact depth")
parser.add_argument("--contact_depth_bias", type=float, default=2.8, help="Contact depth bias")
args = parser.parse_args()

if args.num_workers == 0:
    args.num_workers = int(os.cpu_count() / 2)  # you can increase this

if not os.path.exists(args.objects_list):
    raise FileNotFoundError(f"Objects list file not found: {args.objects_list}")

with open(args.objects_list, "r") as f:
    data = json.load(f)

print(f"Total objects in dataset: {len(data)}")

if args.use_wandb:
    run_id = wandb.util.generate_id()

    wandb.init(
        project="grasp_generation",
        id=run_id,
        resume="allow",
        config={
            "total_objects": len(data),
            "args.max_successful_grasps": args.max_successful_grasps,
        },
    )
    wandb.define_metric("step")
    wandb.define_metric("completion_percentage", step_metric="step")
    wandb.define_metric("processed_objects", step_metric="step")
    wandb.define_metric("remaining_objects", step_metric="step")
    wandb.define_metric("grasp_count", step_metric="step")
    wandb.define_metric("filtered_count", step_metric="step")
    wandb.define_metric("filter_success_rate", step_metric="step")
    wandb.log({"total_objects": len(data), "step": 0})

processed_objects = 0
failed_objects = []

for obj in data:
    object_name = obj["name"]
    xml_file_path = obj["xml"]
    object_output_dir = os.path.join(
        f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/grasp_results/rigid_objects", object_name
    )
    os.makedirs(object_output_dir, exist_ok=True)

    mesh_path = os.path.join(object_output_dir, f"{object_name}_combined.obj")
    manifold_path = os.path.join(object_output_dir, f"{object_name}_manifold.obj")
    simplify_path = os.path.join(object_output_dir, f"{object_name}_simplified.obj")
    grasp_file_path = os.path.join(object_output_dir, f"{object_name}_grasps.json")
    filtered_npz_path = os.path.join(object_output_dir, f"{object_name}_grasps_filtered.npz")
    filtered_json_path = os.path.join(object_output_dir, f"{object_name}_grasps_object_info.json")

    if os.path.exists(filtered_npz_path):
        processed_objects += 1
        print(f"Object {object_name} already fully processed, skipping...")
        continue

    print(f"Starting processing for {object_name}")

    processing_failed = False
    failure_reason = ""

    try:
        if not os.path.exists(mesh_path):
            print("  Combining meshes...")
            try:
                subprocess.run(
                    [
                        "python",
                        "pipeline/combine_meshes.py",
                        xml_file_path,
                        mesh_path,
                        "--only_collision",
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError:
                print("  Failed: mesh generation")
                processing_failed = True
                failure_reason = "mesh_generation_failed"
                failed_objects.append(object_name)
                continue

        if not os.path.exists(manifold_path):
            print("  Creating manifold mesh...")
            try:
                subprocess.run(
                    [
                        "./manifold",
                        os.path.abspath(mesh_path),
                        os.path.abspath(manifold_path),
                        "-s",
                    ],
                    cwd=f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/external_src/Manifold/build",
                    check=True,
                )
            except subprocess.CalledProcessError:
                print("  Failed: manifold")
                processing_failed = True
                failure_reason = "manifold_failed"
                failed_objects.append(object_name)
                continue

            print("  Simplifying mesh...")
            try:
                subprocess.run(
                    [
                        "./simplify",
                        "-i",
                        os.path.abspath(manifold_path),
                        "-o",
                        os.path.abspath(simplify_path),
                        "-m",
                        "-r",
                        "0.5",
                    ],
                    cwd=f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/external_src/Manifold/build",
                    check=True,
                )
            except subprocess.CalledProcessError:
                print("  Failed: simplification")
                processing_failed = True
                failure_reason = "simplify_failed"
                failed_objects.append(object_name)
                continue

        if not os.path.exists(grasp_file_path):
            print("  Generating grasps...")
            try:
                subprocess.run(
                    [
                        "python",
                        "pipeline/generate_grasps.py",
                        "--object_file",
                        simplify_path,
                        "--systematic_sampling",
                        "--quality",
                        "antipodal",
                        "--output",
                        grasp_file_path,
                        "--num_workers",
                        str(args.num_workers),
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError:
                print("  Failed: grasp generation")
                processing_failed = True
                failure_reason = "grasp_generation_failed"
                failed_objects.append(object_name)
                continue

        xml_mesh_file_path = xml_file_path.replace(".xml", "_mesh.xml")
        if not os.path.exists(xml_mesh_file_path):
            xml_mesh_file_path = xml_file_path

        if not os.path.exists(filtered_npz_path):
            print("  Filtering grasps...")
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    cmd_args = [
                        "python",
                        "pipeline/perturbations_test.py",
                        "--object_name",
                        object_name,
                        "--grasps_path",
                        grasp_file_path,
                        "--xml_file",
                        xml_mesh_file_path,
                        "--approach_distance",
                        str(args.approach_distance),
                        "--approach_steps",
                        str(args.approach_steps),
                        "--shake_magnitude",
                        str(args.shake_magnitude),
                        "--shake_steps",
                        str(args.shake_steps),
                        "--max_contact_depth",
                        str(args.max_contact_depth),
                        "--min_contact_depth",
                        str(args.min_contact_depth),
                        "--center_contact_depth",
                        str(args.center_contact_depth),
                        "--contact_depth_bias",
                        str(args.contact_depth_bias),
                        "--num_workers",
                        str(args.num_workers),
                        "--rotate",
                        "--max_successful",
                        str(args.max_successful_grasps),
                    ]

                    if attempt == 1:
                        cmd_args.append("--diversity_mode")

                    subprocess.run(cmd_args, check=True)
                    break
                except subprocess.CalledProcessError:
                    if attempt == max_attempts:
                        print("  Failed: grasp filtering")
                        processing_failed = True
                        failure_reason = "filter_failed"
                        failed_objects.append(object_name)
                        continue

        filtered_count = 0
        if os.path.exists(filtered_npz_path):
            try:
                npz_data = np.load(filtered_npz_path)
                filtered_count = len(npz_data["transforms"])
            except Exception:
                pass
            for f in os.listdir(object_output_dir):
                file_path = os.path.join(object_output_dir, f)
                if file_path != filtered_npz_path and os.path.isfile(file_path):
                    os.remove(file_path)

        processed_objects += 1
        completion_percentage = (processed_objects / len(data)) * 100
        print(f"  Done: {filtered_count} grasps ({completion_percentage:.1f}% complete)")

    except Exception as e:
        print(f"  Error: {e}")

print(f"\nProcessed: {processed_objects}/{len(data)}")
if failed_objects:
    print(
        f"Failed: {', '.join(failed_objects[:10])}"
        + (f" (+{len(failed_objects) - 10} more)" if len(failed_objects) > 10 else "")
    )

if args.use_wandb:
    wandb.log(
        {
            "pipeline_complete": True,
            "total_processed": processed_objects,
            "total_failed": len(failed_objects),
        }
    )
