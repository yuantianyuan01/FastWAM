import functools
import json
import logging
from collections import defaultdict
from pathlib import Path

import mujoco
from mujoco import MjModel

from molmo_spaces.molmo_spaces_constants import get_resource_manager, get_scenes

log = logging.getLogger(__name__)


def get_scene_metadata(mj_base_scene_path: str | Path) -> dict | None:
    """Get scene metadata from the scene path."""

    # Just in case we received a Path instead of a str
    mj_base_scene_path = str(mj_base_scene_path)

    assert mj_base_scene_path.endswith(".xml"), (
        f"Scene is supposed to be xml ({mj_base_scene_path} given)"
    )

    if "ceiling" in mj_base_scene_path:
        metadata_file = mj_base_scene_path.replace("_ceiling.xml", "_metadata.json")
    else:
        metadata_file = mj_base_scene_path.replace(".xml", "_metadata.json")

    if not Path(metadata_file).exists():
        # Fallback: metadata search by iteratively removing underscore-connected suffixes
        dir_path = Path(mj_base_scene_path).parent

        # First attempt to replace .xml by _metadata.json,
        # then continue removing suffixes separated by "_" until something found
        scene_name = str(Path(mj_base_scene_path).name).replace(".xml", "")
        parts = scene_name.split("_")

        while parts:
            cur_name = "_".join(parts + ["metadata.json"])
            if (dir_path / cur_name).exists():
                with open(dir_path / cur_name, "r") as f:
                    return json.load(f)

            # Not found, remove last part and try again
            parts.pop()

        log.warning(f"Scene metadata file not found for {mj_base_scene_path}")

        return None

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return metadata


def is_object_articulable_from_metadata(
    model: MjModel, scene_metadata: dict, object_name: str
) -> bool:
    """Return True if the object has at least one hinge or slide joint per scene metadata.

    Uses the scene's name_map for joints and checks each corresponding MuJoCo joint type.
    """
    joint_maps: dict | None = (
        scene_metadata.get("objects", {})
        .get(object_name, {})
        .get("name_map", {})
        .get("joints", None)
        if scene_metadata
        else None
    )
    if not joint_maps:
        return False
    for joint_name, _ in joint_maps.items():
        joint_type = model.joint(joint_name).type
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE or joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            return True
    return False


_ALL_SCENES_INSTALLED = False


def ensure_all_scenes_installed():
    global _ALL_SCENES_INSTALLED
    if not _ALL_SCENES_INSTALLED:
        print("Installing all scenes")

        get_resource_manager().install_all_for_data_type("scenes", skip_linking=True)

        _ALL_SCENES_INSTALLED = True


class SceneMeta:
    @staticmethod
    def get_scene_metadata(mj_base_scene_path: str | Path) -> dict:
        return get_scene_metadata(mj_base_scene_path)

    @staticmethod
    def scene_datasets() -> list[str]:
        return sorted(set(get_resource_manager().versions["scenes"].keys()) - {"refs"})

    @staticmethod
    def extraction_dir(data_source: str) -> Path:
        ensure_all_scenes_installed()
        cache_dir = get_resource_manager().cache_dir
        version = get_resource_manager().versions["scenes"][data_source]
        scene_dir = cache_dir / "scenes" / data_source / version
        assert scene_dir.is_dir()
        return scene_dir

    @classmethod
    @functools.lru_cache
    def for_dataset_split(cls, data_source, split):
        extract_dir = cls.extraction_dir(data_source)
        install_dir = get_resource_manager().symlink_dir / "scenes" / data_source
        all_scenes = get_scenes(data_source, split)[split]
        meta = {}
        for scene_idx, scene_info in all_scenes.items():
            if scene_info is None:
                continue

            if isinstance(scene_info, Path):
                scene_path = scene_info

            else:
                if all(v is None for v in scene_info.values()):
                    continue

                scene_path = Path(next(v for v in scene_info.values() if v is not None))

            scene_path = extract_dir / scene_path.relative_to(install_dir)
            meta[scene_idx] = get_scene_metadata(scene_path)
        return meta

    @classmethod
    def for_split(cls, split):
        metas = {}
        for dataset in cls.scene_datasets():
            metas[dataset] = cls.for_dataset_split(dataset, split)
        return metas


@functools.lru_cache
def synsets_to_scenes_and_assets():
    synset_to_scenes = defaultdict(set)
    synset_to_assets = defaultdict(set)

    for dataset, index_to_meta in SceneMeta.for_split("train").items():
        for index, meta in index_to_meta.items():
            for entry in meta["objects"].values():
                asset_id = entry["asset_id"]
                from molmo_spaces.utils.object_metadata import ObjectMeta

                ometa = ObjectMeta.annotation(asset_id)
                if ometa:
                    synset = ometa["synset"]
                    synset_to_scenes[synset].add((dataset, index))
                    synset_to_assets[synset].add(asset_id)

    return synset_to_scenes, synset_to_assets


if __name__ == "__main__":
    import time

    for _ in range(2):
        for dataset in ["ithor", "procthor-objaverse-debug"]:
            for split in ["train", "val", "test"]:
                ctime = time.time()
                meta = SceneMeta.for_dataset_split(dataset, split)
                ctime = time.time() - ctime
                print(
                    dataset,
                    split,
                    len(meta),
                    min(meta.keys() or [-1]),
                    max(meta.keys() or [-1]),
                    f"{ctime:.6f}",
                )

    for _ in range(2):
        ctime = time.time()
        synset_to_scenes, synset_to_assets = synsets_to_scenes_and_assets()
        ctime = time.time() - ctime
        print(len(synset_to_scenes), len(synset_to_assets), f"{ctime:.6f}")

    print("DONE")
