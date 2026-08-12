import os.path

from mujoco import MjSpec


def xml_add_rby1_to_scene(config, scene_file_path, robot_file_path):
    # For RBY1: Add robot include to XML before loading into MjSpec
    with open(scene_file_path, "r") as f:
        xml_content = f.read()

    # Make robot path relative to scene file directory to preserve relative path context
    scene_dir = os.path.dirname(scene_file_path)

    # Add the include line after the opening <mujoco> tag
    include_line = f'  <include file="{robot_file_path}"/>\n'

    # Find the position after <mujoco ...> tag
    mujoco_tag_end = xml_content.find(">") + 1
    modified_xml = xml_content[:mujoco_tag_end] + "\n" + include_line + xml_content[mujoco_tag_end:]

    # Write to temporary file in same directory to preserve path context
    # TODO: this is a hack to preserve the path context, there's gotta be a better way
    # maybe just getting away from relative paths altogether. that would be nice
    temp_scene_path = os.path.join(scene_dir, f"temp_scene_{os.getpid()}.xml")
    try:
        with open(temp_scene_path, "w") as f:
            f.write(modified_xml)

        # Create spec from temporary file (preserves directory context)
        spec = MjSpec.from_file(temp_scene_path)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_scene_path):
            os.remove(temp_scene_path)

    return spec
