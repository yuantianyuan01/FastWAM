import argparse
import json
import os
import xml.etree.ElementTree as ET

from molmo_spaces.molmo_spaces_constants import ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR
from molmo_spaces.utils.constants.object_constants import (
    ALL_ARTICULATION_TYPES_THOR,
    ALL_PICKUP_TYPES_THOR,
)


def find_objs_with_matching_subfolder(
    base_dir, check_joints=False, all_pickup_types: List[str] = None
):
    result = []
    all_pickup_types = [item.lower() for item in all_pickup_types]

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".obj"):
                obj_name = os.path.splitext(file)[0]
                json_filename = obj_name + ".json"
                json_path = os.path.join(root, json_filename)

                if obj_name in dirs and os.path.isfile(json_path):
                    subfolder_path = os.path.join(root, obj_name)
                    found_object = False
                    for pickup_type in all_pickup_types:
                        if pickup_type.lower() in obj_name.lower():
                            found_object = True
                            break

                    if not found_object:
                        print(f"Skipping {obj_name} as it is not a recognized pickup type.")
                        continue

                    for subfile in os.listdir(subfolder_path):
                        if subfile.endswith("mesh.xml"):
                            abs_xml_path = os.path.abspath(os.path.join(subfolder_path, subfile))

                            if check_joints:
                                tree = ET.parse(abs_xml_path)
                                xml_root = tree.getroot()
                                has_valid_joint = False
                                for joint in xml_root.findall(".//joint"):
                                    if joint.get("type") != "free":
                                        has_valid_joint = True
                                        break

                                if not has_valid_joint:
                                    print(f"   Skipping {obj_name} as it has no valid joints.")
                                    continue

                            result.append({"name": obj_name, "xml": abs_xml_path})
                            break

    # shuffle
    import random

    random.shuffle(result)

    return result


def save_to_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find .obj files with matching subfolder, valid .xml, and same-named .json."
    )
    parser.add_argument("directory", help="Base directory to search")
    parser.add_argument(
        "--output",
        help="Output JSON file name",
    )
    parser.add_argument(
        "--check-joints",
        action="store_true",
        help="Only include objects with articulated joints (non-free joints)",
    )

    args = parser.parse_args()

    if not args.output:
        args.output = (
            f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/grasp_results/rigid_objects_list.json"
            if not args.check_joints
            else f"{ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR}/grasp_results/articulable_objects_list.json"
        )

    # Default is thor types that are pickupable or articulable
    all_pickup_types = ALL_PICKUP_TYPES_THOR
    if args.check_joints:
        all_pickup_types = ALL_ARTICULATION_TYPES_THOR

    matched_objs = find_objs_with_matching_subfolder(
        args.directory, args.check_joints, all_pickup_types=all_pickup_types
    )
    save_to_json(matched_objs, args.output)

    print(
        f"Found {len(matched_objs)} matching .obj files with valid .xml and .json. Results saved to {args.output}"
    )
