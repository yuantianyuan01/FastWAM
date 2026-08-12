import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

from molmospaces_resources import split_query_tokens
from pydantic import BaseModel, Field, TypeAdapter

from molmo_spaces.molmo_spaces_constants import (
    ASSETS_DIR,
    DATA_TYPE_TO_SOURCE_TO_VERSION,
    USER_ASSET_LIBRARIES,
    get_resource_manager,
    get_scenes_root,
)


class UserAssetLibraryIndexEntry(BaseModel):
    # UID of the object
    uid: str
    # Path to object MJCF
    object_path: Path
    # Path to a json file which contains the object metadata
    metadata_path: Path
    # Path to an npz file which contains array-valued metadata (e.g. clip features)
    metadata_npz_path: Path | None


UserAssetLibraryIndex = TypeAdapter(dict[str, UserAssetLibraryIndexEntry])


class UserGraspLibraryIndex(BaseModel):
    # {robot_name: {uid: {joint_name: grasp_path}}} to a NPZ file which contains "transforms" -> (N, 4, 4) array of grasp poses in robot frame
    articulated_grasp_paths: dict[str, dict[str, dict[str, Path]]] = Field(default_factory=dict)
    # {robot_name: {uid: grasp_path}} to a NPZ file which contains "transforms" -> (N, 4, 4) array of grasp poses in robot frame
    grasp_paths: dict[str, dict[str, Path]] = Field(default_factory=dict)


def install_scene_from_source_index(source, idx):
    archives = get_resource_manager().index_lookup("scenes", source, str(idx))
    if len(archives) == 0:
        raise ValueError(f"{source=} {idx=} returned {len(archives)} archives (expected 1)")
    assert len(archives) == 1, f"{source=} {idx=} returned {len(archives)} archives (expected 1)"
    source_to_paths = {source: archives}
    get_resource_manager().install_packages("scenes", source_to_paths)
    return source_to_paths


def install_scene_from_path(xml_path):
    scene_source = Path(xml_path).relative_to(get_scenes_root()).parts[0]

    rel_path = Path(xml_path).relative_to(get_scenes_root() / scene_source)
    archives = get_resource_manager().find_archives("scenes", scene_source, [rel_path])

    if not archives:
        raise RuntimeError(
            f"BUG: could not find archive for {xml_path} (relative {rel_path} for {scene_source})"
        )

    source_to_paths = {scene_source: archives}

    get_resource_manager().install_packages("scenes", source_to_paths)

    return source_to_paths


def find_object_paths(xml_path, exclude_thor=True):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    scene_dir = Path(xml_path).parent

    # Find all <asset> child elements
    for asset_type in ["mesh", "texture", "material", "hfield", "skin"]:
        for elem in root.findall(f".//asset/{asset_type}"):
            # Most assets store the file path in the 'file' attribute
            file_path = elem.attrib.get("file")
            if file_path and file_path.startswith("../"):
                if not exclude_thor or "/objects/thor/" not in file_path:
                    # Objects are globally linked from the cache
                    full_path = (scene_dir / file_path).resolve()
                    source = (
                        full_path.relative_to(get_resource_manager().cache_dir / "objects")
                    ).parts[0]
                    rel_asset = full_path.relative_to(
                        get_resource_manager().source_dir("objects", source)
                    )
                    yield source, rel_asset


def install_objects_for_scene(xml_path, exclude_thor=True):
    if "objaverse" not in DATA_TYPE_TO_SOURCE_TO_VERSION["objects"]:
        return {}

    source_to_archives = {}

    for source, rel_asset in find_object_paths(xml_path, exclude_thor=exclude_thor):
        archives = get_resource_manager().find_archives("objects", source, [rel_asset])
        if source not in source_to_archives:
            source_to_archives[source] = archives
        else:
            source_to_archives[source].extend(archives)

    source_to_archives = {
        source: list(set(archives)) for source, archives in source_to_archives.items()
    }

    get_resource_manager().install_packages("objects", source_to_archives)

    return source_to_archives


def install_grasps_for_scene(xml_path, grasp_source="droid_objaverse", exclude_thor=True):
    if grasp_source in ["droid", "rum"]:
        # These are thor-only
        if exclude_thor:
            return {}

    if grasp_source not in DATA_TYPE_TO_SOURCE_TO_VERSION["grasps"]:
        return {}

    source_to_archives = {grasp_source: set()}

    for _source, rel_asset in find_object_paths(xml_path, exclude_thor=exclude_thor):
        for substr in split_query_tokens(rel_asset.name):
            source_to_archives[grasp_source].update(
                get_resource_manager().index_lookup("grasps", grasp_source, substr)
            )

    get_resource_manager().install_packages("grasps", source_to_archives)

    return source_to_archives


def add_install_prefixes(data_type, source, relative_path):
    return ASSETS_DIR / data_type / source / relative_path


@cache
def get_user_library_index(user_library_path: Path):
    with open(user_library_path / "assets_index.json", "r") as f:
        return UserAssetLibraryIndex.validate_json(f.read())


@cache
def get_user_grasp_library_index(user_library_path: Path):
    with open(user_library_path / "grasps_index.json", "r") as f:
        return UserGraspLibraryIndex.model_validate_json(f.read())


@cache
def get_thor_uid_to_xmls() -> dict[str, Path]:
    base = (ASSETS_DIR / "objects" / "thor").resolve()
    return {xml.stem: xml for xml in base.rglob("*.xml")}


def locate_uid_package(
    uid: str,
    extension: str = "xml",
) -> tuple[str, str | None, Path] | tuple[None, None, None]:
    """
    Locate the package containing the given object UID.

    Args:
        uid: The UID of the object to locate.
        extension: The extension of the file to locate.

    Returns:
        A tuple containing the source, package, and XML path of the object.
            If the object is not found, returns (None, None, None).
    """

    # Since thor objects are always fully installed, we just search in the file system
    thor_uid_to_xmls = get_thor_uid_to_xmls()

    if uid in thor_uid_to_xmls:
        base = (ASSETS_DIR / "objects" / "thor").resolve()
        xml_path = thor_uid_to_xmls[uid]
        return "thor", None, add_install_prefixes("objects", "thor", xml_path.relative_to(base))

    file_name = f"{uid}.{extension}"

    # For other sources (aka Objaverse for now), we need to search through
    # the data tries from the resource manager
    for object_source in sorted(DATA_TYPE_TO_SOURCE_TO_VERSION["objects"].keys()):
        if object_source in ["thor"]:
            continue

        substrings = split_query_tokens(uid)
        for substring in substrings:
            possible_archives = get_resource_manager().index_lookup(
                "objects", object_source, substring
            )
            if not possible_archives:
                continue

            # TODO pass archives to avoid full trie initialization? it's kind of fast, but...
            tries = get_resource_manager().tries("objects", object_source)
            for possible_archive in possible_archives:
                for path in tries.get(possible_archive, {}).leaf_paths():
                    if path.endswith(file_name):
                        return (
                            object_source,
                            possible_archive,
                            add_install_prefixes("objects", object_source, path),
                        )

    for user_library_name, user_library_dir in USER_ASSET_LIBRARIES.items():
        user_library_index = get_user_library_index(user_library_dir)
        if uid in user_library_index:
            # Assume the user is getting the object
            return user_library_name, None, user_library_dir / user_library_index[uid].object_path

    return None, None, None


def install_uid(uid, grasp_source="droid_objaverse"):
    source, package, xml_path = locate_uid_package(uid)

    if source is None:
        raise ValueError(
            f"{uid} not found in object sources {sorted(DATA_TYPE_TO_SOURCE_TO_VERSION['objects'].keys())}"
        )

    if source in USER_ASSET_LIBRARIES:
        return xml_path

    if source != "thor":
        get_resource_manager().install_packages("objects", {source: [package]})

        # Install grasps (on-demand for objaverse)
        source_to_archives = {grasp_source: set()}
        for substr in split_query_tokens(Path(xml_path.name).stem):
            source_to_archives[grasp_source].update(
                get_resource_manager().index_lookup("grasps", grasp_source, substr)
            )
        get_resource_manager().install_packages("grasps", source_to_archives)

    return xml_path


def install_scene_with_objects_and_grasps_from_path(
    xml_path, grasp_sources=("droid_objaverse",), exclude_thor=True
):
    if isinstance(xml_path, dict):
        xml_path = xml_path["base"]
    type_to_source_to_archives = {
        "scenes": install_scene_from_path(xml_path),
    }

    if not get_resource_manager().cache_lock:
        # We just need to link the scene, no need to check for objects or grasps
        # (everything is pre-cached, and we use a global symlink for those data types)
        return type_to_source_to_archives

    type_to_source_to_archives["objects"] = install_objects_for_scene(
        xml_path, exclude_thor=exclude_thor
    )

    type_to_source_to_archives["grasps"] = {}
    for grasp_source in grasp_sources:
        type_to_source_to_archives["grasps"].update(
            install_grasps_for_scene(xml_path, grasp_source=grasp_source, exclude_thor=exclude_thor)
        )

    return type_to_source_to_archives


if __name__ == "__main__":

    def debug_lazy_search():
        uid = "0000c32fde7f45efb8d14e8ba737d50c"
        source, package, xml_path = locate_uid_package(uid)
        print(uid, source, package, xml_path)
        install_uid(uid)

        uid = "Bowl_1"
        source, package, xml_path = locate_uid_package(uid)
        print(uid, source, package, xml_path)
        install_uid(uid)

        print("DONE")

    debug_lazy_search()
