import os
import json
import argparse
import xml.etree.ElementTree as ET


def find_objs_with_matching_subfolder(base_dir, check_joints=False):
    result = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".xml"):
                #print(file)
                if "train" in file: # to ignore house xml files
                    continue

                json_filename = "thor_metadata.json"
                metadata = json.load(open(os.path.join(root, json_filename)))

                if metadata.get("assetMetadata", None) is not None:
                    primaryProperty = metadata.get("assetMetadata", {}).get("primaryProperty", None)
                    if primaryProperty is not None:
                        if primaryProperty in ["CanPickup"]:
                            obj_name = os.path.splitext(file)[0]
                            abs_xml_path = os.path.join(root, file)
                            result.append({"name": obj_name, "xml": abs_xml_path})
                            print(f"Processing object: {obj_name}")

    # shuffle
    import random

    random.shuffle(result)


    return result


def save_to_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find .obj files with matching subfolder, valid .xml, and same-named .json."
    )
    parser.add_argument("directory", help="Base directory to search")
    parser.add_argument(
        "--output",
        default="results/rigid_objects_list.json",
        help="Output JSON file name",
    )
    parser.add_argument(
        "--check-joints",
        action="store_true",
        help="Only include objects with articulated joints (non-free joints)",
    )

    args = parser.parse_args()

    matched_objs = find_objs_with_matching_subfolder(args.directory, args.check_joints)
    save_to_json(matched_objs, args.output)

    print(
        f"Found {len(matched_objs)} matching .obj files in {args.directory} with valid .xml and .json. Results saved to {args.output}"
    )
