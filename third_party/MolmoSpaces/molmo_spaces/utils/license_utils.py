import re
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path
from typing import Any

from molmospaces_resources import SourceInfo

from molmo_spaces.molmo_spaces_constants import get_resource_manager
from molmo_spaces.utils.lazy_loading_utils import find_object_paths, install_scene_from_source_index
from molmo_spaces.utils.object_metadata import ObjectMeta

ROBOT_LICENSE = {}

DEFAULT_LICENSE = {
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "creator_name": "Allen Institute for AI (Ai2)",
    "source": "In-house",
}

ATTRIBUTION_TEMPLATE = (
    "{assets}" + f" by the {DEFAULT_LICENSE['creator_name']},"
    f" licensed under {DEFAULT_LICENSE['license'].replace('-', ' ')}."
)


def resolve_license(data_type, data_source, identifier):
    if data_type == "objects":
        return resolve_object_license(data_source, identifier)
    if data_type == "scenes":
        return resolve_scene_license(data_source, identifier)
    if data_type == "grasps":
        return resolve_grasps_license(data_source, identifier)
    if data_type == "robots":
        return resolve_robot_license(data_source, data_source)

    raise ValueError(f"Non-valid {data_type=}")


def validate_identifier(data_type, source, identifier):
    archives = get_resource_manager().tries(data_type, source).keys()
    for archive in archives:
        if identifier in archive:
            break
    else:
        raise ValueError(f"{identifier=} is not in {source=} ({data_type=})")
    return archive.split(".")[0].replace(f"{source}_", "")


def validate_thor_identifier(identifier):
    anno = ObjectMeta.annotation(identifier)
    if anno is not None:
        if anno["isObjaverse"]:
            raise ValueError(f"{identifier=} is not in `thor`")
        return identifier

    return validate_identifier("objects", "thor", identifier)


def validate_objaverse_identifier(identifier):
    anno = ObjectMeta.annotation(identifier)
    if anno is not None:
        if not anno["isObjaverse"]:
            raise ValueError(f"{identifier=} is not in `objaverse`")
        return identifier

    return validate_identifier("objects", "objaverse", identifier)


def resolve_object_license(data_source, identifier):
    if data_source == "objaverse":
        anno = ObjectMeta.annotation(validate_objaverse_identifier(identifier))
        lic = anno["license_info"]

        assert "sketchfab" in lic["creator_profile_url"], (
            f"Only sketchfab assets expected, got {lic['creator_profile_url']}"
        )

        cur_license = {
            "data_type": "objects",
            "data_source": data_source,
            "asset_id": anno["assetId"],
            "creator_username": lic["creator_username"],
            "creator_display_name": lic["creator_display_name"],
            "creator_profile_url": lic["creator_profile_url"],
            "source": "Sketchfab",
            "uri": lic["uri"],
            "downloaded": "2021-2022",
            "license_determination": "License inferred from Sketchfab designation at time"
            " of download (circa 2021/2022).",
            "modifications": "The model has been significantly modified to reduce memory and processing requirements,"
            " including mesh decimation, convex collider extraction, and baking of visual effects via Blender scripts."
            " The provided quality may not reflect the original model.",
            "dataset_license": "This subset of Objaverse is licensed under ODC-BY 1.0.",
        }

        if lic["license"] == "by":
            cur_license["license"] = "CC-BY-4.0"
            cur_license["license_url"] = "https://creativecommons.org/licenses/by/4.0/"
            cur_license["derivative_notice"] = "This work is a derivative of the original model."
            cur_license["attribution"] = (
                f"Model by {lic['creator_display_name']} ({lic['creator_username']}), licensed under CC BY 4.0."
            )
        elif lic["license"] == "by-sa":
            cur_license["license"] = "CC-BY-SA-4.0"
            cur_license["license_url"] = "https://creativecommons.org/licenses/by-sa/4.0/"
            cur_license["derivative_license"] = (
                "This derivative work is licensed under CC BY-SA 4.0."
            )
            cur_license["attribution"] = (
                f"Model by {lic['creator_display_name']} ({lic['creator_username']}), licensed under CC BY-SA 4.0."
            )
        elif lic["license"] == "cc0":
            cur_license["license"] = "CC0-1.0"
            cur_license["license_url"] = "https://creativecommons.org/publicdomain/zero/1.0/"
            cur_license["derivative_notice"] = (
                "This work is a derivative of the original asset, which was released under CC0."
            )
            cur_license["attribution"] = (
                f"Model by {lic['creator_display_name']} ({lic['creator_username']}), licensed under CC0-1.0."
            )
        elif lic["license"] == "by-nc":
            cur_license["license"] = "CC-BY-NC-4.0"
            cur_license["license_url"] = "https://creativecommons.org/licenses/by-nc/4.0/"
            cur_license["commercial_use"] = False
            cur_license["derivative_notice"] = (
                "This work is a derivative of the original asset and may not be used for commercial purposes."
            )
            cur_license["attribution"] = (
                f"Model by {lic['creator_display_name']} ({lic['creator_username']}), licensed under CC BY-NC 4.0."
                f" Non-commercial use only."
            )
        elif lic["license"] == "by-nc-sa":
            cur_license["license"] = "CC-BY-NC-SA-4.0"
            cur_license["license_url"] = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
            cur_license["commercial_use"] = False
            cur_license["derivative_license"] = (
                "This derivative work is licensed under CC BY-NC-SA 4.0."
            )
            cur_license["derivative_notice"] = (
                "This work is a derivative of the original asset and may not be used for commercial purposes."
            )
            cur_license["attribution"] = (
                f"Model by {lic['creator_display_name']} ({lic['creator_username']}), licensed under CC BY-NC-SA"
                f" 4.0. Non-commercial use only."
            )
        else:
            raise NotImplementedError(f"Got unsupported license {lic['license']}")

    elif data_source == "thor":
        cur_license = {
            "data_type": "objects",
            "data_source": data_source,
            "asset_id_or_archive_name": validate_thor_identifier(identifier),
            **DEFAULT_LICENSE,
            "attribution": ATTRIBUTION_TEMPLATE.format(assets="Model(s)"),
        }

    elif data_source == "objathor_metadata":
        assets = (
            "Object annotation (bounding boxes, masses, synsets, CLIP features, etc.) extracted"
        )

        return {
            "data_type": "objects",
            "data_source": "objathor_metadata",
            "asset_id": identifier,
            **DEFAULT_LICENSE,
            "attribution": ATTRIBUTION_TEMPLATE.format(assets=assets),
            "scope": "Annotation data derived from objects, including bounding boxes, physical properties,"
            " descriptions, CLIP embedding–based representations, and semantic labels.",
            "relationship_to_objects": "annotation",
            "object_licenses": "Underlying objects are independently licensed; see each object's license for details"
            " (under `objects` - `thor` and `objects` - `objaverse`).",
            "license_note": "This license applies only to the annotation data. Underlying objects are independently"
            " licensed and are not covered by this license.",
        }

    else:
        raise ValueError(
            f"Can't determine license for `objects` with {data_source=} and {identifier=}"
        )

    return cur_license


def scene_path_resolve(
    source: str, idx: int, source_to_archives: dict[str, Collection[str]]
) -> Path:
    if "procthor" in source or "holodeck" in source:
        fn = procthor_resolver
    elif "ithor" in source:
        fn = ithor_resolver
    else:
        raise NotImplementedError(f"Missing implementation for {source}")

    archive = list(source_to_archives[source])[0]
    scene_info = get_resource_manager().source_info("scenes", source, recursive=False)
    modalities = scene_info["archive_to_relative_paths"][archive]
    return fn(source, idx, scene_info, modalities)


def ithor_resolver(
    source: str, idx: int, scene_info: SourceInfo, modalities: list[Path], variant: str = ""
) -> Path:
    return scene_info["root_dir"] / modalities[0]


def procthor_resolver(
    source: str, idx: int, scene_info: SourceInfo, modalities: list[Path], variant: str = "_ceiling"
) -> Path:
    split = source.split("-")[-1]
    target = f"{split}_{idx}{variant}.xml"
    if Path(target) not in modalities:
        target = f"{split}_{idx}.xml"
    assert Path(target) in modalities, f"Missing {target=} with available modalities {modalities}"
    return scene_info["root_dir"] / target


def scene_includes(scene_path):
    def identifier_and_asset_part(rel_asset):
        archives = get_resource_manager().find_archives("objects", source, [rel_asset])

        assert len(archives) == 1, (
            f"Expected exactly one archive for {rel_asset}, got {len(archives)}"
        )

        return validate_identifier("objects", source, archives[0]), rel_asset.name

    includes = defaultdict(set)
    identifier_to_objs = defaultdict(set)
    for source, rel_asset in find_object_paths(scene_path, exclude_thor=False):
        identifier, obj = identifier_and_asset_part(rel_asset)
        identifier_to_objs[identifier].add(obj)
        includes[source].add(identifier)

    ret: dict[str, list[dict[str, Any]]] = {
        source: [
            {
                "identifier": identifier,
                # "includes": sorted(identifier_to_objs[identifier]),
                "attribution": resolve_object_license(source, identifier)["attribution"],
            }
            for identifier in includes[source]
        ]
        for source in sorted(includes.keys())
    }

    return ret


def resolve_scene_license(data_source, identifier):
    original_identifier = identifier

    if isinstance(identifier, str):
        match = re.search(r"\d+", identifier)
        if match:
            identifier = int(match.group())

    try:
        identifier = int(identifier)
        is_idx = True
    except ValueError:
        is_idx = False

    if is_idx:
        source_to_archives = install_scene_from_source_index(data_source, identifier)
        scene_path = scene_path_resolve(data_source, identifier, source_to_archives)
        if isinstance(original_identifier, str) and original_identifier not in str(scene_path):
            raise ValueError(f"Non-valid identifier {original_identifier}")
        includes = scene_includes(scene_path)

    else:
        archives = get_resource_manager().tries("scenes", data_source).keys()
        archive = [archive for archive in archives if identifier in archive]

        if len(archive) == 0:
            raise ValueError(f"No archives for `scenes` {data_source} {identifier}")

        assert len(archive) == 1, (
            f"Error: multiple archives for `scenes` {data_source} {identifier}"
        )

        get_resource_manager().install_packages_bulk("scenes", {data_source: archive})

        includes = []

    scene_license = {
        "data_type": "scenes",
        "data_source": data_source,
        "asset_id": str(identifier),
        **DEFAULT_LICENSE,
        "attribution": ATTRIBUTION_TEMPLATE.format(assets="Scene"),
        "scope": "Scene composition, layout, non-object-specific textures, and metadata.",
        "relationship_to_assets": "collection",
        "asset_licenses": "Assets are independently licensed; see assets info below for details.",
        "license_determination": "Scenes are collections referencing independently licensed assets;"
        f" {DEFAULT_LICENSE['license']} applies only to scene composition, layout, and metadata.",
    }
    if includes:
        scene_license["assets"] = includes

    return scene_license


def grasp_targets(data_source, identifier):
    info = get_resource_manager().source_info("grasps", data_source, recursive=True)
    archives = info["archive_to_relative_paths"].keys()
    archive = [archive for archive in archives if identifier in archive]

    if len(archive) == 0:
        raise ValueError(f"No archives for `grasps` {data_source} {identifier}")

    assert len(archive) == 1, f"Error: multiple archives for `grasps` {data_source} {identifier}"

    get_resource_manager().install_packages_bulk("grasps", {data_source: archive})

    targets = [
        str(path).split("/")[-1].split("_grasps_")[0]
        for path in info["archive_to_relative_paths"][archive[0]]
        if (
            str(path).endswith("_grasps_filtered.npz")
            or str(path).endswith("_grasps_filtered.json")
        )
    ]
    targets = [
        (f"{identifier}_" + target.replace(identifier, "").split("_")[:2][-1]).strip("_")
        for target in targets
    ]
    objaverse_targets = [
        target
        for target in targets
        if (ObjectMeta.annotation(target) or {}).get("isObjaverse", False)
    ]
    thor_targets = [
        target
        for target in targets
        if not (ObjectMeta.annotation(target) or {}).get("isObjaverse", True)
    ]
    ithor_targets = [
        target
        for target in targets
        if target not in objaverse_targets and target not in thor_targets
    ]
    ret = {
        "objaverse": [
            {
                "identifier": target,
                "attribution": resolve_object_license("objaverse", target)["attribution"],
            }
            for target in sorted(objaverse_targets)
        ],
        "thor": [
            {
                "identifier": target,
                "attribution": ATTRIBUTION_TEMPLATE.format(assets="Model(s)"),
            }
            for target in sorted(thor_targets)
        ],
        "ithor builtin": [
            {
                "identifier": target + " (ithor scene builtin)",
                "attribution": ATTRIBUTION_TEMPLATE.format(assets="Asset(s)"),
            }
            for target in sorted(ithor_targets)
        ],
    }

    return {key: value for key, value in ret.items() if value}


def resolve_grasps_license(data_source, identifier):
    identifier = validate_identifier("grasps", data_source, identifier)

    license = {
        "data_type": "grasps",
        "data_source": data_source,
        "asset_id": identifier,
        **DEFAULT_LICENSE,
        "attribution": ATTRIBUTION_TEMPLATE.format(assets="Grasps generated"),
        "scope": "Grasp poses, collision data, and metadata (not the underlying 3D model or object).",
        "relationship_to_assets": "annotation",
        "asset_licenses": "Underlying objects are independently licensed; see asset license info for details.",
        "license_determination": (
            "Grasp data and metadata are derived or created independently of the assets to which they apply."
            f" {DEFAULT_LICENSE['license']} applies only to grasp data and metadata, not object meshes, textures, or other"
            " underlying assets."
        ),
        "external_target_assets": grasp_targets(data_source, identifier),
    }

    return license


def resolve_robot_license(data_source, identifier):
    common_license = {
        "data_type": "robots",
        "data_source": data_source,
        "asset_id": identifier,
    }

    if "franka" in identifier:
        cur_license = {
            **common_license,
            "creator_username": "Franka Robotics",
            "attribution": "Developed by Franka Robotics",
            "source": "mujoco_menagerie/franka_fr3",
            "license": "Apache 2.0",
            "uri": "https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_fr3/LICENSE",
            "downloaded": "2025",
            "modifications": "Changed position controller to force controller for hand gripper",
        }
    elif "rby" in identifier:
        cur_license = {
            **common_license,
            "creator_username": "Rainbow Robotics",
            "attribution": "Copyright 2024-2025 Rainbow Robotics",
            "source": "RainbowRobotics/rby1-sdk",
            "license": "Apache 2.0",
            "uri": "https://github.com/RainbowRobotics/rby1-sdk/blob/main/LICENSE",
            "downloaded": "2025",
            "modifications": "Added holonomic base and removed wheel controller and slider controllers",
        }
    elif "robotiq" in identifier:
        cur_license = {
            **common_license,
            "creator_username": "ROS-Industrial",
            "attribution": "Copyright (c) 2013, ROS-Industrial",
            "source": "mujoco_menagerie/robotiq_2f85_v4",
            "license": "BSD-2-Clause License",
            "uri": "https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotiq_2f85_v4/LICENSE",
            "downloaded": "2025",
        }
    elif "rum" in identifier:
        cur_license = {
            **common_license,
            "creator_username": "NYU Generalizable Robotics and AI Lab (GRAIL)",
            "attribution": "Copyright (c) 2026 NYU Generalizable Robotics and AI Lab (GRAIL)",
            "source": "jeffacce/cap-policy",
            "license": "MIT",
            "uri": "https://github.com/jeffacce/cap-policy/blob/main/LICENSE",
            "downloaded": "2025",
        }
    elif "yam" in identifier:
        cur_license = {
            **common_license,
            "creator_username": "I2RT Robotics, LLC",
            "attribution": "Copyright (c) I2RT Robotics",
            "source": "https://github.com/i2rt-robotics/i2rt/blob/d36027fc50e12d9261f091f9d91c4715bb5e398f/i2rt/robots/get_robot.py#L129",
            "license": "MIT",
            "uri": "https://github.com/i2rt-robotics/i2rt/blob/main/LICENSE",
            "downloaded": "2026",
        }
    elif "g1" in identifier:
        cur_license = {
            **common_license,
            "creator_username": 'HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")',
            "attribution": 'Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")',
            "source": "mujoco_menagerie/unitree_g1",
            "license": "BSD-3-Clause License",
            "uri": "https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_g1/LICENSE",
            "downloaded": "2026",
        }
    else:
        raise NotImplementedError(f"Got unknown robot {identifier}")

    return cur_license
