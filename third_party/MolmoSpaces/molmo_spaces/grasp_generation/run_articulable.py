import argparse
import json
import os
import subprocess

import numpy as np

import wandb
from molmo_spaces.molmo_spaces_constants import ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR

# TODO: install floating_robotiq assets via resource manager


parser = argparse.ArgumentParser(description="Process articulable objects for grasp generation")
parser.add_argument(
    "--objects_list",
    type=str,
    default=f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/grasp_results/articulable_objects_list.json",
    help="Path to JSON file with object list",
)
parser.add_argument(
    "--max_successful_grasps", type=int, default=1000, help="Max successful grasps per object"
)
parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers (0 = all CPUs)")
parser.add_argument(
    "--approach_distance", type=float, default=0.6, help="Approach distance for grasp filtering"
)
parser.add_argument(
    "--approach_steps", type=int, default=10, help="Approach steps for grasp filtering"
)
parser.add_argument("--max_contact_depth", type=float, default=1.0, help="Max contact depth")
parser.add_argument("--min_contact_depth", type=float, default=0.0, help="Min contact depth")
parser.add_argument("--center_contact_depth", type=float, default=0.75, help="Center contact depth")
parser.add_argument("--contact_depth_bias", type=float, default=2.8, help="Contact depth bias")
parser.add_argument(
    "--num_clusters", type=int, default=40, help="Number of clusters for diversity mode"
)
args = parser.parse_args()

if args.num_workers == 0:
    args.num_workers = int(os.cpu_count() / 2)

if not os.path.exists(args.objects_list):
    raise FileNotFoundError(f"Objects list file not found: {args.objects_list}")

with open(args.objects_list, "r") as f:
    data = json.load(f)

print(f"Total objects in dataset: {len(data)}")

if args.use_wandb:
    run_id = wandb.util.generate_id()

    wandb.init(
        project="mjcf2grasp",
        id=run_id,
        resume="allow",
        config={"total_objects": len(data), "max_successful_grasps": args.max_successful_grasps},
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
successful_objects = []


def run_grasp_filtering_stage(
    object_name, grasps_path, xml_file, output_dir, per_joint_grasps_json=None
):
    xml_mesh_file = xml_file.replace(".xml", "_mesh.xml")
    for prim_object in ["shelving", "table", "lightswitch", "laptop"]:
        if prim_object in object_name.lower():
            xml_mesh_file = xml_file.replace("_mesh.xml", "_prim.xml")

    if not os.path.exists(xml_mesh_file):
        xml_mesh_file = xml_file

    xml_file_for_filtering = xml_mesh_file

    if per_joint_grasps_json:
        try:
            print("   Filtering per-joint grasps...")
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    if attempt > 1:
                        print(
                            f"   Retry attempt {attempt}/{max_attempts} (without diversity mode)..."
                        )

                    cmd_args = [
                        "python",
                        "pipeline/articulation_test.py",
                        "--object_name",
                        object_name,
                        "--per_joint_summary_json",
                        per_joint_grasps_json,
                        "--xml_file",
                        xml_file_for_filtering,
                        "--num_workers",
                        str(args.num_workers),
                        "--approach_distance",
                        str(args.approach_distance),
                        "--approach_steps",
                        str(args.approach_steps),
                        "--max_successful",
                        str(args.max_successful_grasps),
                        "--max_contact_depth",
                        str(args.max_contact_depth),
                        "--min_contact_depth",
                        str(args.min_contact_depth),
                        "--center_contact_depth",
                        str(args.center_contact_depth),
                        "--contact_depth_bias",
                        str(args.contact_depth_bias),
                        "--num_clusters",
                        str(args.num_clusters),
                    ]

                    if attempt == 1:
                        cmd_args.append("--diversity_mode")

                    subprocess.run(cmd_args, check=True)
                    break
                except subprocess.CalledProcessError:
                    if attempt == max_attempts:
                        raise

            summary_path = per_joint_grasps_json.replace(".json", "_filtered.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                for entry in summary:
                    filtered_grasps_file = entry.get("filtered_grasps_file")
                    filtered_count = 0

                    if filtered_grasps_file and os.path.exists(
                        os.path.join(os.path.dirname(summary_path), filtered_grasps_file)
                    ):
                        filtered_path = os.path.join(
                            os.path.dirname(summary_path), filtered_grasps_file
                        )
                        try:
                            npz_data = np.load(filtered_path)
                            filtered_count = len(npz_data["transforms"])
                        except Exception as e:
                            print(f"      Warning: Could not read NPZ file {filtered_path}: {e}")
                            filtered_count = 0

                    grasps_file = entry.get("grasps_file", "")
                    original_count = 0
                    if grasps_file and os.path.exists(
                        os.path.join(os.path.dirname(summary_path), grasps_file)
                    ):
                        with open(
                            os.path.join(os.path.dirname(summary_path), grasps_file),
                            "r",
                        ) as gf:
                            gdata = json.load(gf)
                        original_count = len(gdata.get("transforms", []))

                    success_rate = (
                        (filtered_count / original_count * 100) if original_count > 0 else 0
                    )

                    status = "SUCCESS" if filtered_grasps_file else "FAILED"
                    print(
                        f"      Joint: {entry['joint']} | {status} | {filtered_count}/{original_count} ({success_rate:.1f}%)"
                    )
                return False, None
            return True, summary_path
        except subprocess.CalledProcessError as e:
            print(f"   Error in per-joint grasp filtering: {str(e)}")
            return False, None
        except Exception as e:
            print(f"   Unexpected error: {str(e)}")
            return False, None
    else:
        filtered_grasps_path = grasps_path.replace(".json", "_filtered.json")
        try:
            print("   Filtering grasps...")
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    if attempt > 1:
                        print(
                            f"   Retry attempt {attempt}/{max_attempts} (without diversity mode)..."
                        )

                    cmd_args = [
                        "python",
                        "pipeline/articulation_test.py",
                        "--object_name",
                        object_name,
                        "--grasps_path",
                        grasps_path,
                        "--xml_file",
                        xml_file_for_filtering,
                        "--num_workers",
                        str(args.num_workers),
                        "--approach_distance",
                        str(args.approach_distance),
                        "--approach_steps",
                        str(args.approach_steps),
                        "--max_successful",
                        str(args.max_successful_grasps),
                        "--max_contact_depth",
                        str(args.max_contact_depth),
                        "--min_contact_depth",
                        str(args.min_contact_depth),
                        "--center_contact_depth",
                        str(args.center_contact_depth),
                        "--contact_depth_bias",
                        str(args.contact_depth_bias),
                        "--num_clusters",
                        str(args.num_clusters),
                    ]

                    if attempt == 1:
                        cmd_args.append("--diversity_mode")

                    subprocess.run(cmd_args, check=True)
                    break
                except subprocess.CalledProcessError:
                    if attempt == max_attempts:
                        raise

            return True, filtered_grasps_path
        except subprocess.CalledProcessError as e:
            print(f"   Error in grasp filtering: {str(e)}")
            return False, None
        except Exception as e:
            print(f"   Unexpected error: {str(e)}")
            return False, None


def run_per_joint_grasp_generation(object_name, joint_meshes_json, output_dir, full_mesh):
    print(f"\nStage 2: Per-Joint Grasp Generation for {object_name}")
    if isinstance(joint_meshes_json, str):
        with open(joint_meshes_json, "r") as f:
            joint_meshes_data = json.load(f)
    else:
        joint_meshes_data = joint_meshes_json

    if all(
        joint.get("grasps_file")
        and os.path.exists(os.path.join(os.path.dirname(joint_meshes_json), joint["grasps_file"]))
        for joint in joint_meshes_data
    ):
        print("   All grasp files exist, skipping per-joint grasp generation.")
        return True, joint_meshes_json

    try:
        subprocess.run(
            [
                "python",
                "pipeline/generate_grasps.py",
                "--per_joint_grasps_from_meshes",
                joint_meshes_json,
                "--quality",
                "antipodal",
                "--min_quality",
                "0.005",
                "--classname",
                "articulated_handle",
                "--systematic_sampling",
                "--dataset",
                "thor_articulated",
                "--collision_object_file",
                full_mesh,
            ],
            check=True,
        )
        print(f"   Per-joint grasps generated and summary updated: {joint_meshes_json}")
        return True, joint_meshes_json
    except subprocess.CalledProcessError as e:
        print(f"   Error in per-joint grasp generation: {str(e)}")
        return False, None
    except Exception as e:
        print(f"   Unexpected error: {str(e)}")
        return False, None


def run_handle_detection_stage(obj, output_dir):
    object_name = obj["name"]
    xml_file_path = obj["xml"]

    print(f"\nStage 0: Handle Detection for {object_name}")
    full_mesh_path = os.path.join(output_dir, "main.obj")
    handle_mesh_path = None

    if os.path.exists(full_mesh_path):
        print("   Full mesh already exists, skipping...")
        return True, handle_mesh_path, full_mesh_path
    try:
        subprocess.run(
            [
                "python",
                "pipeline/extract_leaf_meshes.py",
                xml_file_path,
                os.path.join(output_dir, "main.obj"),
            ],
            check=True,
        )
        return True, handle_mesh_path, full_mesh_path
    except subprocess.CalledProcessError as e:
        print(f"   Error in handle detection: {str(e)}")
        return False, None, None
    except Exception as e:
        print(f"   Unexpected error: {str(e)}")
        return False, None, None


output_base = f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/grasp_results/articulable_objects"
os.makedirs(output_base, exist_ok=True)

for i, obj in enumerate(data):
    object_name = obj["name"]
    object_output_dir = os.path.join(output_base, object_name)
    os.makedirs(object_output_dir, exist_ok=True)

    if args.use_wandb:
        wandb.log(
            {
                "current_object": object_name,
                "progress": (i + 1) / len(data),
                "processed_count": i + 1,
                "step": i + 1,
            }
        )

    try:
        success, handle_mesh, full_mesh = run_handle_detection_stage(obj, object_output_dir)

        if not success:
            failed_objects.append(object_name)
            print(f"   Failed at Stage 0 (handle detection) for {object_name}")
            continue

        joint_meshes_json = os.path.join(object_output_dir, "joint_meshes_info.json")
        per_joint_grasps_success = False
        if os.path.exists(joint_meshes_json):
            per_joint_grasps_success, joint_meshes_json_out = run_per_joint_grasp_generation(
                object_name, joint_meshes_json, object_output_dir, full_mesh
            )
            if per_joint_grasps_success:
                print(f"   Per-joint grasps generated for {object_name}")
        else:
            print(f"   Error: joint_meshes.json not found: {joint_meshes_json}")

        joint_axis_success = os.path.exists(joint_meshes_json) if joint_meshes_json else False
        joint_axis_path = os.path.join(object_output_dir, f"{object_name}_joint_axis.json")

        grasps_success = per_joint_grasps_success
        grasps_path = joint_meshes_json if per_joint_grasps_success else None

        filtering_success = True
        print(f"\nStage 3: Grasp Filtering for {object_name}")
        if per_joint_grasps_success and os.path.exists(joint_meshes_json):
            filtered_grasps_file = "joint_meshes_info_filtered.json"
            if not os.path.exists(os.path.join(object_output_dir, filtered_grasps_file)):
                filtering_success, filtered_grasps_path = run_grasp_filtering_stage(
                    object_name,
                    None,
                    obj["xml"],
                    object_output_dir,
                    per_joint_grasps_json=joint_meshes_json,
                )
            else:
                filtering_success = True
                filtered_grasps_path = os.path.join(object_output_dir, filtered_grasps_file)
        elif grasps_success and grasps_path and os.path.exists(grasps_path):
            filtering_success, filtered_grasps_path = run_grasp_filtering_stage(
                object_name, grasps_path, obj["xml"], object_output_dir
            )
        else:
            print(f"   No valid grasps found for filtering for {object_name}")

        if success:
            processed_objects += 1
            successful_objects.append(
                {
                    "name": object_name,
                    "handle_mesh": handle_mesh,
                    "full_mesh": full_mesh,
                    "grasps": grasps_path if grasps_success else None,
                    "filtered_grasps": filtered_grasps_path if filtering_success else None,
                    "joint_axis": joint_axis_path if joint_axis_success else None,
                    "xml": obj["xml"],
                    "output_dir": object_output_dir,
                    "stages_completed": {
                        "handle_detection": True,
                        "joint_axis_analysis": joint_axis_success,
                        "grasp_generation": grasps_success,
                        "grasp_filtering": filtering_success,
                    },
                }
            )

            if filtering_success and joint_axis_success:
                status = "Fully processed (all stages)"
            elif grasps_success and joint_axis_success:
                status = "Partially processed (no filtering)"
            elif joint_axis_success:
                status = "Partially processed (no grasps)"
            else:
                status = "Partially processed (no joint analysis)"
            print(f"   {status}: {object_name}")

            if args.use_wandb:
                grasp_count = 0
                filtered_count = 0

                if (
                    filtering_success
                    and filtered_grasps_path
                    and os.path.exists(filtered_grasps_path)
                ):
                    try:
                        with open(filtered_grasps_path, "r") as f:
                            filtered_data = json.load(f)
                        for entry in filtered_data:
                            if "filtered_grasps_file" in entry:
                                filtered_file = os.path.join(
                                    object_output_dir, entry["filtered_grasps_file"]
                                )
                                if os.path.exists(filtered_file):
                                    try:
                                        npz_data = np.load(filtered_file)
                                        filtered_count += len(npz_data["transforms"])
                                    except Exception as e:
                                        print(
                                            f"      Warning: Could not read NPZ file {filtered_file}: {e}"
                                        )

                            if "grasps_file" in entry:
                                grasp_file = os.path.join(object_output_dir, entry["grasps_file"])
                                if os.path.exists(grasp_file):
                                    with open(grasp_file, "r") as gf:
                                        grasp_data = json.load(gf)
                                    grasp_count += len(grasp_data.get("transforms", []))
                    except Exception as e:
                        print(f"      Warning: Error reading grasp counts: {e}")

                filter_success_rate = (filtered_count / grasp_count * 100) if grasp_count > 0 else 0
                completion_percentage = (processed_objects / len(data)) * 100

                wandb.log(
                    {
                        "completion_percentage": completion_percentage,
                        "processed_objects": processed_objects,
                        "remaining_objects": len(data) - processed_objects,
                        "overall_progress": processed_objects / len(data),
                        "object_completed": object_name,
                        "grasp_count": grasp_count,
                        "filtered_count": filtered_count,
                        "filter_success_rate": filter_success_rate,
                        "stage_handle_detection": True,
                        "stage_joint_axis_analysis": joint_axis_success,
                        "stage_grasp_generation": grasps_success,
                        "stage_grasp_filtering": filtering_success,
                        "step": processed_objects,
                    }
                )
        else:
            failed_objects.append(object_name)
            print(f"   Failed to process {object_name}")

    except Exception as e:
        print(f"  Error: {e}")
        failed_objects.append(object_name)

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
