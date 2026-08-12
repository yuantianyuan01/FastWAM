import argparse
import json
import logging
import multiprocessing
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco as mj
import prior
from p_tqdm import p_uimap
from prior.utils.types import Dataset
from tqdm import tqdm

from molmo_spaces import MOLMO_SPACES_BASE_DIR
from molmo_spaces.housegen.builder import MlSpacesSceneBuilder, SceneType
from molmo_spaces.housegen.constants import (
    DEFAULT_SETTLE_TIME,
    DYNAMIC_OBJ_GEOMS_MARGIN,
    FREE_JOINT_DAMPING,
    FREE_JOINT_FRICTIONLOSS,
)
from molmo_spaces.housegen.utils import load_holodeck_houses, load_objaverse_houses

log = logging.getLogger(__name__)

ID_TO_CATEGORY_FILE = MOLMO_SPACES_BASE_DIR / "resources" / "asset_id_to_object_type.json"
MATERIALS_TO_TEXTURES_FILE = MOLMO_SPACES_BASE_DIR / "resources" / "material_to_textures.json"

BUILD_SETTINGS_FILENAME = "housegen_build_settings.json"


def json_serializer(obj):
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@dataclass
class ExportSettings:
    flavor: str
    split: str
    assets_dir: Path
    target_dir: Path
    save_dir: Path
    objaverse_mjcf_dir: Path
    objaverse_data_dir: Path
    identifier: str
    param_geom_margin: float
    param_freejoint_damping: float
    param_freejoint_frictionloss: float
    use_sleep_island: bool
    settle_time: float
    copy_original: bool
    copy_non_settled: bool

    use_presaved_houses: bool = False
    presaved_houses_json: str = ""

    export_lights: bool = False


DATASET: Dataset | None = None
ASSET_ID_TO_OBJECT_TYPE: dict[str, str] | None = None
MATERIALS_TO_TEXTURES: dict[str, Any] | None = None
SETTINGS: ExportSettings | None = None


def run_procthor_generation(house_index: int) -> tuple[bool, str]:
    global DATASET, ASSET_ID_TO_OBJECT_TYPE, MATERIALS_TO_TEXTURES, SETTINGS

    if ASSET_ID_TO_OBJECT_TYPE is None or MATERIALS_TO_TEXTURES is None or SETTINGS is None:
        return False, f"House index {house_index}, some required globals are not defined yet"

    house_name = f"{SETTINGS.split}_{house_index}"
    xml_save_path = SETTINGS.target_dir / f"{house_name}.xml"
    json_save_path = SETTINGS.target_dir / f"{house_name}.json"

    success = True
    error_msg = ""
    try:
        # if True:
        house_json: dict[str, Any] | None = None
        if not SETTINGS.use_presaved_houses:
            if DATASET is None:
                return False, f"House {house_name} not found in the prior-Dataset"
            if house_index < 0 or house_index >= len(DATASET):
                return False, f"House {house_name} got index '{house_index}' out of range"
            house_json = DATASET[house_index]
        else:
            houses_json_folder = Path(SETTINGS.presaved_houses_json)
            house_json_filepath = houses_json_folder / f"{house_name}.json"
            if not house_json_filepath.is_file():
                return (
                    False,
                    f"House {house_json_filepath.as_posix()} not found in pre-saved houses",
                )
            with open(house_json_filepath, "r") as fhandle:
                house_json = json.load(fhandle)

        if house_json is None:
            return False, f"Got an unexpected error when trying to use house '{house_name}'"

        with open(json_save_path, "w") as fhandle:
            json.dump(house_json, fhandle, indent=4, default=json_serializer)

        scene_builder = MlSpacesSceneBuilder(
            scene_type=SceneType(SETTINGS.flavor),
            asset_dir=SETTINGS.assets_dir,
            asset_id_to_object_type=ASSET_ID_TO_OBJECT_TYPE,
            materials_to_textures=MATERIALS_TO_TEXTURES,
            objaverse_mjcf_dir=SETTINGS.objaverse_mjcf_dir,
            objaverse_data_dir=SETTINGS.objaverse_data_dir,
            add_ceiling=True,
            use_sleep_island=SETTINGS.use_sleep_island,
            settle_time=SETTINGS.settle_time,
            copy_original=SETTINGS.copy_original,
            copy_non_settled=SETTINGS.copy_non_settled,
            export_lights=SETTINGS.export_lights,
        )

        scene_builder.load_from_json(
            thor_house=house_json,
            target_dir=SETTINGS.target_dir,
            house_id=house_name,
            stability_params={
                "param_geom_margin": SETTINGS.param_geom_margin,
                "param_freejoint_damping": SETTINGS.param_freejoint_damping,
                "param_freejoint_frictionloss": SETTINGS.param_freejoint_frictionloss,
            },
        )

        # Copy the generated assets from the default place to another desired location
        if SETTINGS.save_dir.absolute() != SETTINGS.target_dir.absolute():
            src_path = xml_save_path
            dst_path = SETTINGS.save_dir / f"{house_name}.xml"
            shutil.copy(src_path, dst_path)

            src_path = SETTINGS.target_dir / f"{house_name}.json"
            dst_path = SETTINGS.save_dir / f"{house_name}.json"
            shutil.copy(src_path, dst_path)

            src_path = SETTINGS.target_dir / f"{house_name}_metadata.json"
            dst_path = SETTINGS.save_dir / f"{house_name}_metadata.json"
            if src_path.is_file():
                shutil.copy(src_path, dst_path)

            src_path = SETTINGS.target_dir / f"{house_name}_ceiling.xml"
            dst_path = SETTINGS.save_dir / f"{house_name}_ceiling.xml"
            if src_path.is_file():
                shutil.copy(src_path, dst_path)

            if SETTINGS.copy_original:
                src_path = SETTINGS.target_dir / f"{house_name}_orig.xml"
                dst_path = SETTINGS.save_dir / f"{house_name}_orig.xml"
                if src_path.is_file():
                    shutil.copy(src_path, dst_path)

            if SETTINGS.copy_non_settled:
                src_path = SETTINGS.target_dir / f"{house_name}_non_settled.xml"
                dst_path = SETTINGS.save_dir / f"{house_name}_non_settled.xml"
                if src_path.is_file():
                    shutil.copy(src_path, dst_path)

            src_path = SETTINGS.target_dir / f"{house_name}_assets"
            dst_path = SETTINGS.save_dir / f"{house_name}_assets"
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    except Exception as e:
        success = False
        error_msg = f"Error saving house: {house_index}\n"
        error_msg += f"Error message: {e} \n"

    return success, error_msg


def run_ithor_generation(house_index: int) -> tuple[bool, str]:
    global ASSET_ID_TO_OBJECT_TYPE, MATERIALS_TO_TEXTURES, SETTINGS

    if ASSET_ID_TO_OBJECT_TYPE is None or MATERIALS_TO_TEXTURES is None or SETTINGS is None:
        return False, f"House index {house_index}, some required globals are not defined yet"

    house_name = f"FloorPlan{house_index}_physics"
    robo_path = SETTINGS.target_dir / house_name / f"{house_name}.json"
    xml_save_path = SETTINGS.target_dir / f"{house_name}.xml"

    success = True
    error_msg = ""
    # try:
    if True:
        scene_builder = MlSpacesSceneBuilder(
            scene_type=SceneType.ITHOR,
            asset_dir=SETTINGS.assets_dir,
            asset_id_to_object_type=ASSET_ID_TO_OBJECT_TYPE,
            materials_to_textures=MATERIALS_TO_TEXTURES,
            objaverse_mjcf_dir=None,
            objaverse_data_dir=None,
            use_sleep_island=SETTINGS.use_sleep_island,
            settle_time=SETTINGS.settle_time,
            copy_original=SETTINGS.copy_original,
            copy_non_settled=SETTINGS.copy_non_settled,
            export_lights=SETTINGS.export_lights,
        )

        scene_builder.load_from_json_path(
            thor_house_path=robo_path,
            target_dir=SETTINGS.target_dir,
            house_id=house_name,
            stability_params={
                "param_geom_margin": SETTINGS.param_geom_margin,
                "param_freejoint_damping": SETTINGS.param_freejoint_damping,
                "param_freejoint_frictionloss": SETTINGS.param_freejoint_frictionloss,
            },
        )

        # Copy the generated assets from the default place to another desired location -----
        if SETTINGS.save_dir.absolute() != SETTINGS.target_dir.absolute():
            src_path = SETTINGS.target_dir / f"{house_name}.xml"
            dst_path = SETTINGS.save_dir / f"{house_name}.xml"
            shutil.copy(src_path, dst_path)

            src_path = SETTINGS.target_dir / f"{house_name}_metadata.json"
            dst_path = SETTINGS.save_dir / f"{house_name}_metadata.json"
            if src_path.is_file():
                shutil.copy(src_path, dst_path)

            src_path = os.path.join(
                os.path.dirname(xml_save_path), f"FloorPlan{house_index}_physics"
            )
            src_path = SETTINGS.target_dir / house_name
            dst_path = SETTINGS.save_dir / house_name
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        # ----------------------------------------------------------------------------------

    # except Exception as e:
    #     success = False
    #     error_msg = f"Error saving house: FloorPlan{house_index}\n"
    #     error_msg += f"Error message: {e} \n"

    return success, error_msg


def save_build_settings(settings: ExportSettings) -> None:
    build_settings = asdict(settings)
    build_settings["mujoco-version"] = mj.__version__
    build_settings["datetime"] = datetime.now().strftime("%m-%d-%y %H:%M:%S")
    build_settings_savepath = settings.target_dir / BUILD_SETTINGS_FILENAME
    if os.path.islink(build_settings_savepath):
        os.unlink(build_settings_savepath)
    with open(build_settings_savepath, "w") as fhandle:
        json.dump(build_settings, fhandle, indent=4, default=json_serializer)

    if settings.save_dir.absolute() != settings.target_dir.absolute():
        src_path = build_settings_savepath
        dst_path = settings.save_dir / build_settings_savepath.name
        if src_path.is_file() and src_path.absolute() != dst_path.absolute():
            shutil.copy(src_path, dst_path)


def save_errors(settings: ExportSettings, messages: list[str]) -> None:
    cant_generate_filepath = (
        f"{settings.flavor}-{settings.split}-cant-generate-{settings.identifier}.txt"
    )
    with open(cant_generate_filepath, "a") as fhandle:
        for msg in messages:
            fhandle.write(msg + "\n")


def main() -> int:
    global DATASET, ASSET_ID_TO_OBJECT_TYPE, MATERIALS_TO_TEXTURES, SETTINGS

    parser = argparse.ArgumentParser(description="Process THOR houses")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ithor",
        choices=["ithor", "procthor-10k", "procthor-objaverse", "holodeck-objaverse"],
        help="The type of dataset to use to generate the houses",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="The split to use when grabbing the houses from the dataset",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting house index",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="Ending house index (exclusive)",
    )
    parser.add_argument(
        "--target-folder",
        type=str,
        default="",
        help="Path to folder where to generate the scenes",
    )
    parser.add_argument(
        "--copy-folder",
        type=str,
        default="",
        help="Path to folder where to copy the generated scenes (if required)",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default="",
        help="The path to the assets directory containing molmo-spaces assets",
    )
    parser.add_argument(
        "--house-dataset-path",
        type=str,
        default="",
        help="Path to house dataset",
    )
    parser.add_argument(
        "--objaverse-mjcf-path",
        type=str,
        default="",
        help="The path to the folder with mjcf files of the objaverse assets",
    )
    parser.add_argument(
        "--objaverse-data-path",
        type=str,
        default="",
        help="The path to the original data (msgpack vertex data) for the objaverse assets",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers to use for house generation",
    )
    parser.add_argument(
        "--save-procthor-json",
        action="store_true",
        help="Whether or not to save the .json file for the exported procthor houses",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        default=f"{datetime.now().strftime('%m%d%y')}",
        help="A suffix to use when saving the results json files",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Whether or not to keep the existing scene, otherwise overwrite it",
    )
    parser.add_argument(
        "--param-geom-margin",
        type=float,
        default=DYNAMIC_OBJ_GEOMS_MARGIN,
        help="The value applied to the margin attribute for collider geoms",
    )
    parser.add_argument(
        "--param-freejoint-damping",
        type=float,
        default=FREE_JOINT_DAMPING,
        help="The value applied to the damping attribute for free joints",
    )
    parser.add_argument(
        "--param-freejoint-frictionloss",
        type=float,
        default=FREE_JOINT_FRICTIONLOSS,
        help="The value applied to the frictionloss attribute for free joints",
    )
    parser.add_argument(
        "--use-sleep-island",
        action="store_true",
        help="If enabled, will hint the builder to export with sleeping island enabled (requires mujoco>=3.3.8)",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=DEFAULT_SETTLE_TIME,
        help="The amount of time used to settle the objects when exporting the houses",
    )
    parser.add_argument(
        "--copy-original",
        action="store_true",
        help="Whether or not to save a copy of the original (to compare later after filtering)",
    )
    parser.add_argument(
        "--copy-non-settled",
        action="store_true",
        help="Whether or not to save a copy of the scene before settling it (as defined in the house json)",
    )
    parser.add_argument(
        "--export-lights",
        action="store_true",
        help="Whether or not to export the lights that come from the house json",
    )
    parser.add_argument(
        "--presaved-houses-json",
        type=str,
        default="",
        help="The path to the folder that contains pre-saved jsons for the houses, to avoid memory exploding from prior-Dataset",
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------------------------------
    if args.assets_dir == "":
        log.error("Must give a path to the assets folder via --assets-dir")
        return 1

    assets_dir = Path(args.assets_dir)
    if not assets_dir.is_dir():
        log.error(f"Given assets-dir path '{assets_dir.as_posix()}' is not a valid directory")
        return 1
    # ----------------------------------------------------------------------------------------------

    if not ID_TO_CATEGORY_FILE.is_file():
        log.error(
            f"Couldn't find valid id-to-category mapping @ '{ID_TO_CATEGORY_FILE.as_posix()}'"
        )
        return 1

    with open(ID_TO_CATEGORY_FILE, "r") as fhandle:
        ASSET_ID_TO_OBJECT_TYPE = json.load(fhandle)

    # ----------------------------------------------------------------------------------------------

    if not MATERIALS_TO_TEXTURES_FILE.is_file():
        log.error(f"Couldn't retrieve mat-to-tex file @ '{MATERIALS_TO_TEXTURES_FILE.as_posix()}'")
        return 1

    with open(MATERIALS_TO_TEXTURES_FILE, "r") as fhandle:
        MATERIALS_TO_TEXTURES = json.load(fhandle)

    # ----------------------------------------------------------------------------------------------

    if args.target_folder == "":
        log.error(
            "Must provide path to the folder where to generate the scenes, via --target-folder"
        )
        return 1
    target_dir = Path(args.target_folder).absolute()
    if not target_dir.is_dir():
        log.error(f"Given path to target-folder '{target_dir.as_posix()}' is not a valid directory")
        return 1

    save_dir = target_dir if args.copy_folder == "" else Path(args.copy_folder).absolute()
    save_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------------------------

    objaverse_mjcf_dir = Path(args.objaverse_mjcf_path)
    objaverse_data_dir = Path(args.objaverse_data_path)

    use_presaved_houses = False
    presaved_houses_json = ""
    if args.presaved_houses_json != "" and Path(args.presaved_houses_json).is_dir():
        use_presaved_houses = True
        presaved_houses_json = args.presaved_houses_json

    SETTINGS = ExportSettings(
        flavor=args.dataset,
        split=args.split,
        assets_dir=assets_dir,
        target_dir=target_dir,
        save_dir=save_dir,
        objaverse_mjcf_dir=objaverse_mjcf_dir,
        objaverse_data_dir=objaverse_data_dir,
        identifier=args.identifier,
        param_geom_margin=args.param_geom_margin,
        param_freejoint_damping=args.param_freejoint_damping,
        param_freejoint_frictionloss=args.param_freejoint_frictionloss,
        use_sleep_island=args.use_sleep_island,
        settle_time=args.settle_time,
        copy_original=args.copy_original,
        copy_non_settled=args.copy_non_settled,
        export_lights=args.export_lights,
        use_presaved_houses=use_presaved_houses,
        presaved_houses_json=presaved_houses_json,
    )

    save_build_settings(SETTINGS)

    # ----------------------------------------------------------------------------------------------

    if args.dataset in {"procthor-10k", "procthor-objaverse", "holodeck-objaverse"}:
        ntotal: int = 0
        if not use_presaved_houses:
            match args.dataset:
                case "procthor-10k":
                    dataset_dict = prior.load_dataset("procthor-10k")
                    DATASET = dataset_dict[args.split]
                case "procthor-objaverse":
                    if args.house_dataset_path == "":
                        log.error(
                            "Must provide the path to the folder containing the .jsonl.gz files that contain the houses json data"
                        )
                        return 1
                    house_dataset_path = Path(args.house_dataset_path)
                    if not house_dataset_path.is_dir():
                        log.error(
                            f"Given folder path '{house_dataset_path}' is not a valid directory"
                        )
                        return 1
                    DATASET = load_objaverse_houses(house_dataset_path.absolute(), args.split)
                case "holodeck-objaverse":
                    if args.house_dataset_path == "":
                        log.error(
                            "Must provide the path to the folder containing the .jsonl.gz files that contain the houses json data"
                        )
                        return 1
                    house_dataset_path = Path(args.house_dataset_path)
                    if not house_dataset_path.is_dir():
                        log.error(
                            f"Given folder path '{house_dataset_path}' is not a valid directory"
                        )
                        return 1
                    DATASET = load_holodeck_houses(house_dataset_path.absolute(), args.split)

        if DATASET is not None:
            ntotal = len(DATASET)
        else:
            match args.dataset:
                case "procthor-10k":
                    if args.split == "train":
                        ntotal = 10000
                    elif args.split == "test" or args.split == "val":
                        ntotal = 1000
                    else:
                        raise RuntimeError("Invalid split for 'procthor-10k' dataset")
                case "procthor-objaverse" | "holodeck-objaverse":
                    if args.split == "train":
                        ntotal = 100000
                    elif args.split == "val":
                        ntotal = 10000
                    else:
                        raise RuntimeError(f"Invalid split for '{args.dataset}' dataset")
                case _:
                    raise RuntimeError(f"Invalid dataset '{args.dataset}'")

        end_idx = args.end if args.end is not None else ntotal
        end_idx = min(end_idx, ntotal)

        houses_indices = list(range(args.start, end_idx))

        def exists_already(settings: ExportSettings, idx: int) -> bool:
            house_path = save_dir / f"{args.split}_{idx}.xml"
            house_path_original = save_dir / f"{args.split}_{idx}_orig.xml"
            house_path_non_settled = save_dir / f"{args.split}_{idx}_non_settled.xml"
            if not house_path.is_file():
                return False
            elif settings.copy_original and not house_path_original.is_file():
                return False
            elif settings.copy_non_settled and not house_path_non_settled.is_file():
                return False

            return True

        if args.keep_existing:
            houses_indices = [i for i in houses_indices if not exists_already(SETTINGS, i)]

        orig_cwd = Path.cwd()

        errors_messages = []
        if args.max_workers > 1:
            results = p_uimap(run_procthor_generation, houses_indices, num_cpus=args.max_workers)
            for success, error_msg in results:
                if not success:
                    errors_messages.append(error_msg)
        else:
            for house_index in tqdm(houses_indices):
                success, error_msg = run_procthor_generation(house_index)
                if not success:
                    errors_messages.append(error_msg)

        os.chdir(orig_cwd)

        if len(errors_messages) > 0:
            save_errors(SETTINGS, errors_messages)

    elif args.dataset == "ithor":
        houses_indices = []
        for i in range(1, 431, 1):
            if (args.end is not None) and (i < args.start or i >= args.end):
                continue
            robo_path = target_dir / f"FloorPlan{i}_physics" / f"FloorPlan{i}_physics.json"
            if not robo_path.is_file():
                continue
            houses_indices.append(i)

        if args.keep_existing:
            houses_indices = [
                i for i in houses_indices if not (save_dir / f"FloorPlan{i}_physics.xml").is_file()
            ]

        orig_cwd = Path.cwd()

        errors_messages = []

        if args.max_workers > 1:
            results = p_uimap(run_ithor_generation, houses_indices, num_cpus=args.max_workers)
            for success, error_msg in results:
                if not success:
                    errors_messages.append(error_msg)
        else:
            for house_index in tqdm(houses_indices):
                success, error_msg = run_ithor_generation(house_index)
                if not success:
                    errors_messages.append(error_msg)

        os.chdir(orig_cwd)

        if len(errors_messages) > 0:
            save_errors(SETTINGS, errors_messages)

    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")

    raise SystemExit(main())
